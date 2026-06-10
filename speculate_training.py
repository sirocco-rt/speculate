# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
#
# Speculate Training Tool
# =======================
# Interactive notebook for training PCA + Gaussian Process emulators on
# Sirocco spectral grids.  The pipeline is:
#
#   1. Select a downloaded grid and the parameters to include.
#   2. Optionally run a quick PCA variance test to choose n_components.
#   3. Process the raw .spec files into a compact grid NPZ if needed.
#   4. Train the emulator (PCA decomposition → GP weight fitting via
#      Nelder-Mead) with live loss plotting and a training log.
#   5. Inspect GP weight diagnostics and the final loss curve.
#
# The notebook writes the trained emulator to Grid-Emulator_Files/ as a
# .npz file consumable by the Inference Tool and Benchmark Suite.

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Training Tool")


@app.cell
def _(mo):
    # ── Reactive state variables ──
    # These act as the cross-cell communication bus for the training pipeline:
    #   loss_history     – NLL per Nelder-Mead iteration (drives live chart)
    #   training_trigger – bumped after training to re-evaluate the "retrain?"
    #                      button label
    #   console_logs     – aggregated stdout captured during training
    #   training_status  – success/failure callout widget
    #   pca_result       – summary string from the quick PCA variance test
    #   trained_emu      – the Emulator instance (for GP diagnostics)
    get_loss_history, set_loss_history = mo.state([])
    get_training_trigger, set_training_trigger = mo.state(0)
    get_console_logs, set_console_logs = mo.state("")
    get_training_status, set_training_status = mo.state(None)
    get_pca_result, set_pca_result = mo.state("Click test to see variance")
    get_trained_emu, set_trained_emu = mo.state(None)
    get_loaded_emu_config, set_loaded_emu_config = mo.state(None)
    get_loaded_emu_display, set_loaded_emu_display = mo.state(None)
    get_diagnostics_emu_display, set_diagnostics_emu_display = mo.state(None)
    return (
        get_console_logs,
        get_diagnostics_emu_display,
        get_loaded_emu_config,
        get_loaded_emu_display,
        get_loss_history,
        get_pca_result,
        get_trained_emu,
        get_training_status,
        get_training_trigger,
        set_console_logs,
        set_diagnostics_emu_display,
        set_loaded_emu_config,
        set_loaded_emu_display,
        set_loss_history,
        set_pca_result,
        set_trained_emu,
        set_training_status,
        set_training_trigger,
    )


@app.cell
def _(
    get_diagnostics_emu_display,
    get_loaded_emu_config,
    get_loaded_emu_display,
    get_trained_emu,
    set_diagnostics_emu_display,
    set_loaded_emu_config,
    set_loaded_emu_display,
    set_trained_emu,
):
    # ── Helpers for releasing GPU/CPU caches when the user changes the
    # configuration or selects a different emulator from disk.  These run
    # outside the cells that own the heavy state so any cell can call them
    # without violating marimo's "value access in creating cell" rule.
    def clear_trained_emu_cache(force=False):
        """Drop GP/PCA caches on the currently-loaded emulator and free GPU memory.

        Sets every known GPU-resident attribute on the trained emulator to
        ``None`` (V11, iPhiPhi, Cholesky factors, kernel hyper-parameters,
        memory-efficient block caches, …), then clears the kernel distance
        cache and asks the CUDA caching allocator to release reserved blocks.

        Parameters
        ----------
        force : bool, optional
            When ``True`` the kernel cache and CUDA empty_cache call run even
            if no emulator was loaded.  Used after a failed ``Emulator.load``
            where partially-allocated tensors may still be live in the
            allocator pool.
        """
        _emu = get_trained_emu()
        _had_emu = _emu is not None
        _had_diagnostics = get_diagnostics_emu_display() is not None

        if _emu is not None:
            # Hard-coded list of GPU/CPU caches the Emulator may attach.
            # hasattr-guarded so missing attributes are ignored.
            for _attr in (
                "_v11_gpu",
                "_iPhiPhi_gpu",
                "_dots_inv_diag_gpu",
                "_grid_points_gpu",
                "_w_hat_gpu",
                "_variances_gpu",
                "_lengthscales_gpu",
                "_L_gpu",
                "_alpha_gpu",
                "_L_gpu_source_id",
                "_mem_eff_L_blocks",
                "_mem_eff_alpha_blocks",
                "_v11",
                "_iPhiPhi",
            ):
                if hasattr(_emu, _attr):
                    try:
                        setattr(_emu, _attr, None)
                    except Exception:
                        pass

        if _had_emu:
            set_trained_emu(None)
        if _had_diagnostics:
            set_diagnostics_emu_display(None)

        if force or _had_emu or _had_diagnostics:
            try:
                from Starfish.emulator.kernels import clear_kernel_cache as _clear_kernel_cache
                _clear_kernel_cache()
            except Exception:
                pass
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

    def invalidate_loaded_emu_config(_value=None):
        """on_change handler used by every config widget.

        Whenever the user edits a parameter that defines the emulator (grid,
        params, wavelength range, scale, smoothing, n_components, …) we drop
        the cached emulator so its GPU footprint is freed and the GP
        diagnostics view stops showing stale data.  Called via marimo
        ``on_change`` so it ignores its argument.
        """
        clear_trained_emu_cache()
        if get_loaded_emu_display() is not None:
            set_loaded_emu_display(None)

    def update_loaded_emu_config(key, transform=lambda value: value, remove_keys=()):
        """Create an on_change callback that preserves user edits after load reset.

        The load dropdown is cleared whenever a setting changes.  Without first
        updating the stored config, marimo can rebuild widgets from the stale
        loaded-emulator defaults and undo the user's current edit.
        """
        def _handler(value=None):
            config = get_loaded_emu_config()
            if config is not None:
                updated_config = dict(config)
                if key is not None:
                    try:
                        updated_config[key] = transform(value)
                    except Exception:
                        updated_config[key] = value
                for remove_key in remove_keys:
                    updated_config.pop(remove_key, None)
                set_loaded_emu_config(updated_config)
            invalidate_loaded_emu_config(value)

        return _handler

    return clear_trained_emu_cache, invalidate_loaded_emu_config, update_loaded_emu_config


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    logo_path = "assets/logos/Speculate_logo2.png"

    # Left column: Title and Description
    title_col = mo.vstack([mo.md(
        """
        # Emulator Training Tool
        """), mo.md(
        """
        Train custom emulator models on spectral grids.
        """
    )])

    # Right column: Logo with link
    # Using flex-end to align it to the right
    logo_col = mo.vstack([
        mo.image(src=logo_path, width=400, height=95),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    # Combine in horizontal stack
    mo.hstack([title_col, logo_col], justify="space-between", align="center")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    usage_toggle = mo.ui.switch(value=False, label=f"{mo.icon('lucide:activity')} System Resources")
    usage_refresh = mo.ui.refresh(default_interval="10s", label="")
    return usage_refresh, usage_toggle


@app.cell(hide_code=True)
def _(mo, usage_refresh, usage_toggle):
    from Speculate_addons.speculate_usage_bars import get_usage_html
    if usage_toggle.value:
        usage_refresh
        _html = get_usage_html()
        usage_bars = mo.vstack([usage_toggle, mo.Html(_html)])
    else:
        usage_bars = usage_toggle
    return (usage_bars,)


@app.cell
def _(mo, os, usage_bars):
    _is_hf = os.environ.get("SPACE_ID") is not None

    _items = [mo.md(f"#Speculate {mo.icon('lucide:telescope')}")]
    _items.extend([mo.md(" "), mo.md("---"), mo.md(" ")])

    if _is_hf:
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
        }, orientation="vertical"))
        _items.extend([
            mo.md(" "),
            mo.md("---"),
            mo.md(f"### {mo.icon('lucide:lock')} Locked Tools:"),
            mo.md("Install Speculate Locally"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:download')} Model Downloader"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:brain')} Training Tool"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:sparkles')} Inference Tool"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:test-tubes')} Benchmark Suite")
        ])
    else:
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/downloader": f"###{mo.icon('lucide:download')} Model Downloader",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/training": f"###{mo.icon('lucide:brain')} Training Tool",
            "/inference": f"###{mo.icon('lucide:sparkles')} Inference Tool",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
            "/benchmark": f"###{mo.icon('lucide:test-tubes')} Benchmark Suite",
        }, orientation="vertical"))

    _items.extend([
        mo.md(" "), mo.md("---"),
        mo.nav_menu({
            "https://github.com/sirocco-rt/speculate": f"###{mo.icon('lucide:github')} Speculate Github",
            "https://github.com/sirocco-rt/speculate/wiki": f"###{mo.icon('lucide:book-open')} Speculate Docs",
        }, orientation="vertical"),
        mo.md(" "), mo.md("---"),
        mo.nav_menu({
            "https://github.com/sirocco-rt/sirocco": f"###{mo.icon('lucide:wind')} Sirocco Github",
            "https://sirocco-rt.readthedocs.io/en/latest/": f"###{mo.icon('lucide:wind')} Sirocco Docs",
        }, orientation="vertical"),
        mo.md("---"),
    ])
    _items.extend([mo.md("---"), usage_bars])
    mo.sidebar(mo.vstack(_items))
    return


@app.cell
def _(mo):
    import torch

    # GPU Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_text = f"**{mo.icon('lucide:cpu')} GPU Active:** {gpu_name} ({vram_gb:.1f} GB VRAM)"
        except:
            gpu_text = f"**{mo.icon('lucide:cpu')} GPU Active:** {gpu_name}"

        status_widget = mo.callout(mo.md(gpu_text), kind="success")

    else:
        status_widget = mo.callout(
            mo.md(f"## {mo.icon('lucide:cpu')} No NVIDIA GPU Detected - Large grids will train slower."), 
            kind="warn"
        )

    status_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ---
    ## {mo.icon('lucide:monitor')} Emulator Specification:
    """)
    return


@app.cell
def _(mo):
    # ── Imports and grid interface setup ──
    # Starfish.grid_tools.HDF5Creator is the upstream grid-processing base
    # class; MarimoHDF5Creator overrides its loop to inject a marimo progress
    # bar.  The Speculate grid interfaces (Speculate_cv_bl_grid_v87f, etc.)
    # know how to parse raw Sirocco .spec files for each grid variant.
    import os
    import sys
    import numpy as np
    import itertools
    from pathlib import Path
    from tqdm import tqdm
    import logging
    import pandas as pd
    import altair as alt
    import importlib
    from Speculate_addons import Spec_functions as spec_functions
    if not hasattr(spec_functions, "enable_speculate_altair_theme"):
        spec_functions = importlib.reload(spec_functions)
    spec_functions.enable_speculate_altair_theme(alt)
    alt.data_transformers.enable("vegafusion")
    import time

    # Import Starfish and Speculate modules
    from Starfish.grid_tools import HDF5Creator
    from Starfish.emulator import Emulator
    from Starfish.emulator.plotting import plot_eigenspectra
    from Speculate_addons.grid_registry import get_grid_configs

    class MarimoHDF5Creator(HDF5Creator):
        """Override the grid-processing loop to surface a marimo progress bar
        instead of a plain tqdm counter.  The resulting NPZ contains:
          wl            - common wavelength grid
          grid_points   - (N, n_params) array of parameter coordinates
          flux_data     - dict mapping key_name → {flux, header}
          param_names   - list of "paramN" strings
        """
        def process_grid(self):
            """
            Run :meth:`process_flux` for all of the spectra within the `ranges`
            and store the processed spectra in the HDF5 file.
            """
            # Enumerate the Cartesian product of every allowed parameter value in
            # the active grid interface; each row becomes one candidate spectrum.
            param_list = []

            # itertools.product expands the per-parameter grids into the full set
            # of model coordinates the creator will try to load from disk.
            for i in itertools.product(*self.points):
                param_list.append(np.array(i))

            all_params = np.array(param_list)
            invalid_params = []

            self.log.debug("Total of {} files to process.".format(len(param_list)))

            # Process each valid parameter tuple once, storing both the transformed
            # flux array and the cleaned FITS-style header metadata in the NPZ.
            for i, param in enumerate(mo.status.progress_bar(all_params, title="Processing Grid Points")):
                try:
                    flux, header = self.grid_interface.load_flux(param, header=True)
                except ValueError:
                    self.log.warning(
                        "Deleting {} from all params, does not exist.".format(param)
                    )
                    invalid_params.append(i)
                    continue

                _, fl_final = self.transform(flux)

                # The key_name format mirrors the parameter ordering so later tools
                # can reconstruct which processed spectrum belongs to which point.
                flux_key = self.key_name.format(*param)
                # Drop blank/comment-only FITS header entries because numpy's NPZ
                # serialization handles plain dict metadata better than raw cards.
                clean_header = {k: v for k, v in header.items() 
                              if k != "" and k != "COMMENT" and v != ""}

                self.flux_data[flux_key] = {
                    "flux": fl_final,
                    "header": clean_header
                }

            # Remove any grid points whose backing spectrum file was missing so the
            # saved grid_points array matches the serialized flux_data exactly.
            all_params = np.delete(all_params, invalid_params, axis=0)

            # Save all data to NPZ file
            np.savez_compressed(
                self.filename,
                wl=self.wl_final_data,
                wl_header=self.wl_header_data,
                grid_points=all_params,
                param_names=self.grid_interface.param_names,
                wave_units=self.grid_interface.wave_units,
                flux_units=self.flux_units,
                flux_key_name=self.key_name,
                flux_data=self.flux_data,
                grid_name=self.grid_name
            )

    # Discover which downloaded grid folders are both present on disk and known
    # to this tool's interface registry.
    sirocco_grids_path = Path("sirocco_grids")
    available_grids = {}

    # grid_configs is the bridge from a folder name to the concrete grid interface,
    # the FITS columns to read, and the parameter indices supported by that grid.
    # "max_params" lists every axis the interface can expose; the user selects a
    # subset in the multiselect widget.
    grid_configs = get_grid_configs()

    if sirocco_grids_path.exists():
        # Scan for grid folders
        grid_folders = [d for d in sirocco_grids_path.iterdir() if d.is_dir()]

        for folder in grid_folders:
            # Check if grid has files
            spec_files = list(folder.glob("*.spec")) + list(folder.glob("*.spec.xz"))
            if spec_files and folder.name in grid_configs:
                # Use label as key, folder name as value for dropdown
                label = f"{grid_configs[folder.name]['name']} ({len(spec_files)} files)"
                available_grids[label] = folder.name
    return (
        Emulator,
        MarimoHDF5Creator,
        alt,
        available_grids,
        grid_configs,
        logging,
        np,
        os,
        pd,
        sirocco_grids_path,
        sys,
        time,
    )


@app.cell
def _():
    # Thin notebook wrappers around grid_registry keep the marimo dependency graph
    # stable while moving all grid-specific decisions into one source of truth.
    from Speculate_addons.grid_registry import (
        default_fixed_inclination as _default_fixed_inclination,
        format_param_tag,
        has_trainable_inclination as _has_trainable_inclination,
        inclination_to_usecols as _inclination_to_usecols,
        inclination_values as _inclination_values,
    )

    def has_trainable_inclination(param_values, grid_name=None):
        """Return whether the selected model parameters include inclination."""
        return _has_trainable_inclination(grid_name, param_values)

    def fixed_inclination_values_for_grid(grid_name=None):
        """Return valid fixed observer angles for the selected grid."""
        return _inclination_values(grid_name)

    def fixed_inclination_to_usecols(base_usecols, inclination, grid_name=None):
        """Map a fixed inclination angle to the grid-interface flux column."""
        return _inclination_to_usecols(grid_name, base_usecols, inclination)

    def default_fixed_inclination(grid_name=None):
        """Return the default fixed observer angle for a grid."""
        return _default_fixed_inclination(grid_name)

    return default_fixed_inclination, fixed_inclination_to_usecols, fixed_inclination_values_for_grid, format_param_tag, has_trainable_inclination


@app.cell
def _(
    clear_trained_emu_cache,
    get_diagnostics_emu_display,
    get_loaded_emu_display,
    get_training_trigger,
    set_loaded_emu_config,
    set_loaded_emu_display,
    mo,
    np,
):
    import re as _re
    from pathlib import Path as _Path
    from Speculate_addons.grid_registry import parse_param_tag

    # Rescan after training completes so newly created emulators appear
    _ = get_training_trigger()

    def _parse_emu_filename(name):
        """Parse emulator NPZ filename into a configuration dict.

        Right-to-left parsing of the naming convention:
          {grid}_emu_{params}_{scale}[_smooth][_{inc}inc]_{wlmin}-{wlmax}AA_{n}PCA

                Parameter tags use the original concatenated increasing-ID form,
                e.g. 12391011 for params 1, 2, 3, 9, 10, and 11.
        """
        name = name.replace('.npz', '')
        result = {}

        # 1. PCA components
        m = _re.search(r'_(\d+)PCA$', name)
        if not m:
            return None
        result['n_components'] = int(m.group(1))
        name = name[:m.start()]

        # 2. Wavelength range
        m = _re.search(r'_(\d+)-(\d+)AA$', name)
        if not m:
            return None
        result['wl_min'] = int(m.group(1))
        result['wl_max'] = int(m.group(2))
        name = name[:m.start()]

        # 3. Optional inclination tag (present when no inclination param)
        m = _re.search(r'_(\d+)inc$', name)
        if m:
            result['fixed_inclination'] = int(m.group(1))
            name = name[:m.start()]

        # 4. Smooth tag
        if name.endswith('_smooth'):
            result['smoothing'] = True
            name = name[:-len('_smooth')]
        else:
            result['smoothing'] = False

        # 5. Grid name, params string, scale
        m = _re.match(r'^(.+?)_emu_([^_]+)_(.+)$', name)
        if not m:
            return None
        result['grid_name'] = m.group(1)
        result['scale'] = m.group(3)
        result['params'] = parse_param_tag(m.group(2))

        return result

    def _read_kernel_from_npz(stem):
        """Try to read the kernel type from a saved emulator NPZ."""
        try:
            with np.load(f'Grid-Emulator_Files/{stem}.npz', allow_pickle=True) as data:
                if 'kernel' in data:
                    return str(data['kernel'])
        except Exception:
            pass
        return None

    # Scan for emulator files
    _emu_dir = _Path("Grid-Emulator_Files")
    _emu_options = {"-- No Emulator Selected --": ""}
    if _emu_dir.exists():
        for _f in sorted(_emu_dir.glob("*_emu_*.npz")):
            _emu_options[_f.stem] = _f.stem

    # Sync dropdown visual state with loaded_emu_display
    _display = get_loaded_emu_display()
    _initial = "-- No Emulator Selected --"
    if _display and _display in _emu_options:
        _initial = _display

    def _on_load_select(value):
        clear_trained_emu_cache()
        if value:
            config = _parse_emu_filename(value)
            if config:
                # Also attach kernel from NPZ
                kernel = _read_kernel_from_npz(value)
                if kernel:
                    config['kernel'] = kernel
                set_loaded_emu_config(config)
                set_loaded_emu_display(value)
            else:
                set_loaded_emu_config(None)
                set_loaded_emu_display(None)
        else:
            # If a config widget invalidated the currently selected emulator,
            # get_loaded_emu_display() is already None and this callback may be
            # invoked by the dropdown's visual reset. Preserve the parsed config
            # so freshly-edited widgets don't snap back to notebook defaults.
            if get_loaded_emu_display() is not None:
                set_loaded_emu_config(None)
                set_loaded_emu_display(None)

    load_emu_dropdown = mo.ui.dropdown(
        options=_emu_options,
        value=_initial,
        label=f"{mo.icon('lucide:folder-open')} Load Existing Emulator:",
        on_change=_on_load_select
    )

    load_emu_button = mo.ui.run_button(
        label=f"{mo.icon('lucide:download')} Load",
        kind="neutral",
    )

    _status = ""
    if _display:
        _parsed = _parse_emu_filename(_display)
        if _parsed:
            _kernel = _read_kernel_from_npz(_display)
            _kernel_str = f" | Kernel: `{_kernel}`" if _kernel else ""
            _cache_state = "Diagnostics loaded" if get_diagnostics_emu_display() == _display else "Selected"
            _load_hint = "" if get_diagnostics_emu_display() == _display else " Click **Load** to enable GP diagnostics."
            _status = (f"{mo.icon('lucide:check-circle')} **{_cache_state}:** params={_parsed['params']}, "
                       f"scale=`{_parsed['scale']}`, "
                       f"{'smoothed, ' if _parsed['smoothing'] else ''}"
                       f"{_parsed['wl_min']}-{_parsed['wl_max']}Å, "
                       f"{_parsed['n_components']} PCA{_kernel_str}.{_load_hint}")

    mo.vstack([
        mo.hstack([load_emu_dropdown, load_emu_button], justify="start", align="end", gap=1),
        mo.md(_status) if _status else mo.md(""),
    ])
    return (load_emu_button,)


@app.cell
def _(
    Emulator,
    clear_trained_emu_cache,
    get_loaded_emu_display,
    load_emu_button,
    set_diagnostics_emu_display,
    set_loss_history,
    set_trained_emu,
    mo,
):
    # ── Explicit "Load" action ──
    # The dropdown only updates configuration state; the heavy
    # ``Emulator.load`` (which materialises V11 on the GPU and can OOM) runs
    # only when the user clicks the run button.  We read ``load_emu_button``
    # in this downstream cell rather than in the cell that defines it,
    # because marimo forbids reading a UI element's value in its creating
    # cell.  Any failure path force-clears caches so a partially-loaded
    # emulator does not leak GPU memory.
    _load_status = mo.md("")

    if load_emu_button.value:
        _display = get_loaded_emu_display()
        if not _display:
            _load_status = mo.callout(
                mo.md("Select an emulator first."),
                kind="warn",
            )
        else:
            clear_trained_emu_cache()
            try:
                _loaded_emu = Emulator.load(f"Grid-Emulator_Files/{_display}.npz")
                set_trained_emu(_loaded_emu)
                set_diagnostics_emu_display(_display)
                if hasattr(_loaded_emu, "loss_history") and _loaded_emu.loss_history:
                    set_loss_history(list(_loaded_emu.loss_history))
                _load_status = mo.callout(
                    mo.md(f"{mo.icon('lucide:check-circle')} Diagnostics loaded for `{_display}.npz`."),
                    kind="success",
                )
            except Exception as e:
                clear_trained_emu_cache(force=True)
                _load_status = mo.callout(
                    mo.md(f"{mo.icon('lucide:x-circle')} Failed to load `{_display}.npz`: {e}"),
                    kind="alert",
                )

    _load_status
    return


@app.cell
def _(available_grids, get_loaded_emu_config, mo, update_loaded_emu_config):
    mo.md("### 1. Grid Selection")

    _cfg = get_loaded_emu_config()
    grid_selection_ui = mo.md("")

    if available_grids:
        # Find the label for the loaded grid, or fall back to the no-BL grid
        # when available because it is the lighter default training path.
        _preferred_grid = "speculate_cv_no-bl_grid_v87f"
        _default_label = list(available_grids.keys())[0]
        for _label, _name in available_grids.items():
            if _name == _preferred_grid:
                _default_label = _label
                break
        if _cfg and _cfg.get('grid_name'):
            for _label, _name in available_grids.items():
                if _name == _cfg['grid_name']:
                    _default_label = _label
                    break

        grid_selector = mo.ui.dropdown(
            options=available_grids,
            value=_default_label,
            label="Select Grid:",
            on_change=update_loaded_emu_config(
                "grid_name",
                remove_keys=("params", "physical_params", "inclination_choice", "fixed_inclination"),
            )
        )
        _grid_info = mo.accordion({
            f"{mo.icon('lucide:info')} Grid options": mo.md(
                "Grid availability and parameter coverage are documented in the "
                "**Speculate wiki** page **Grid Inspector - Available Grids**. "
                "Use that page to check the current supported grids."
            ),
        })
        grid_selection_ui = mo.vstack([
            mo.md(f"{mo.icon('lucide:check-circle')} Found **{len(available_grids)}** grid(s) in `sirocco_grids/`"),
            mo.hstack([grid_selector, _grid_info], justify="space-between", align="center", gap=0.5),
        ])
    else:
        grid_selection_ui = mo.callout(
            mo.md(f"""
            {mo.icon('lucide:triangle-alert')} **No grids found in `sirocco_grids/` folder**

            Please use the **Model Downloader** tool first:
            """),
            kind="warn"
        )
        grid_selector = None
        
    grid_selection_ui
    return (grid_selector,)


@app.cell
def _(
    default_fixed_inclination,
    fixed_inclination_values_for_grid,
    grid_configs,
    grid_selector,
    get_loaded_emu_config,
    mo,
    sirocco_grids_path,
    update_loaded_emu_config,
):
    # Introspect the selected grid interface so the Stage 1 multiselect shows the
    # non-inclination parameters that grid exposes.  Inclination is selected in a
    # separate dropdown below because it is mutually exclusive by construction.
    param_names = {}
    _cfg = get_loaded_emu_config()

    if grid_selector is not None and grid_selector.value:
        selected_grid = grid_selector.value

        if selected_grid in grid_configs:
            config = grid_configs[selected_grid]
            max_params = config["max_params"]

            try:
                # Build a temporary interface instance solely to query the available
                # parameter descriptions without committing to a training run yet.
                # The trailing slash matches the path format expected by these grid
                # interfaces when they build internal file paths.
                temp_path = str(sirocco_grids_path / selected_grid) + "/"

                # Request the maximum supported parameter set so descriptions for
                # every selectable axis are available to the UI.
                temp_interface = config["class"](
                    path=temp_path,
                    usecols=config["usecols"],
                    model_parameters=tuple(max_params)
                )

                # Get dictionary {'param1': 'Desc', ...}
                desc_map = temp_interface.parameters_description()

                # Convert to {1: 'Desc', ...}
                for p_key, p_desc in desc_map.items():
                    p_idx = int(p_key.replace("param", ""))
                    param_names[p_idx] = p_desc

            except Exception as e:
                # Fallback if interface fails (e.g. files missing)
                param_names = {i: f"Parameter {i}" for i in max_params}

            # The UI intentionally separates two ideas that used to be tangled in
            # one multiselect: physical/file-defining axes and the observer
            # inclination choice.  This prevents users from selecting sparse,
            # mid, and full inclination axes at the same time while keeping the
            # downstream training code's old ``params.value`` contract intact.
            _inclination_param_ids = set(config.get("inclination_param_ids", ()))
            _physical_param_ids = [
                param_id for param_id in max_params
                if param_id not in _inclination_param_ids
            ]
            _label_to_param = {
                param_names.get(param_index, f"Parameter {param_index}"): param_index
                for param_index in _physical_param_ids
            }
            _param_to_label = {
                param_index: param_names.get(param_index, f"Parameter {param_index}")
                for param_index in _physical_param_ids
            }

            def _coerce_physical_params(selected_values):
                """Normalize multiselect labels/values into ordered physical IDs."""
                selected_params = []
                for selected_value in selected_values or []:
                    try:
                        selected_params.append(int(selected_value))
                    except (TypeError, ValueError):
                        param_index = _label_to_param.get(str(selected_value))
                        if param_index is not None:
                            selected_params.append(param_index)
                _selected_param_ids = set(selected_params)
                return [
                    param_id for param_id in _physical_param_ids
                    if param_id in _selected_param_ids
                ]

            def _param_labels(selected_params):
                """Convert saved physical parameter IDs back to multiselect labels."""
                return [_param_to_label[p] for p in selected_params if p in _param_to_label]

            if _cfg and selected_grid == _cfg.get('grid_name') and _cfg.get('physical_params'):
                default_physical_params = _coerce_physical_params(_cfg.get('physical_params'))
            elif _cfg and selected_grid == _cfg.get('grid_name') and _cfg.get('params'):
                default_physical_params = _coerce_physical_params(_cfg.get('params'))
            else:
                default_physical_params = _coerce_physical_params(
                    config.get("default_params", max_params)
                )

            # Dropdown option values are compact tokens consumed by the merge cell:
            # ``fixed:55`` means train at one flux column, while ``param:9`` means
            # expose that inclination axis as part of the emulator grid.  Marimo's
            # dict dropdown API uses display labels for initial selection, so the
            # reverse map below converts stored tokens back to labels.
            _fixed_inclination_values = fixed_inclination_values_for_grid(selected_grid)
            _inclination_options = {
                f"Fixed {value}°": f"fixed:{value}"
                for value in _fixed_inclination_values
            }
            for _param_id in max_params:
                if _param_id in _inclination_param_ids:
                    _inclination_options[
                        param_names.get(_param_id, f"Parameter {_param_id}")
                    ] = f"param:{_param_id}"
            _choice_to_label = {
                _choice: _label for _label, _choice in _inclination_options.items()
            }

            _default_choice = None
            if _cfg and selected_grid == _cfg.get('grid_name'):
                # New configs store the explicit fixed:/param: token.  Older
                # emulator filenames only stored params plus an optional fixed
                # inclination suffix, so the fallback reconstructs the token.
                _stored_choice = _cfg.get('inclination_choice')
                if _stored_choice in _choice_to_label:
                    _default_choice = _stored_choice
                else:
                    _loaded_params = []
                    for _param_id in _cfg.get('params') or []:
                        try:
                            _loaded_params.append(int(_param_id))
                        except (TypeError, ValueError):
                            pass
                    _loaded_inclinations = [
                        _param_id for _param_id in _loaded_params
                        if _param_id in _inclination_param_ids
                    ]
                    if _loaded_inclinations:
                        _default_choice = f"param:{_loaded_inclinations[0]}"
                    else:
                        _fixed_inc = int(_cfg.get(
                            'fixed_inclination',
                            default_fixed_inclination(selected_grid),
                        ))
                        _default_choice = f"fixed:{_fixed_inc}"

            if _default_choice not in _choice_to_label:
                _default_trainable_inclinations = [
                    _param_id for _param_id in config.get("default_params", max_params)
                    if _param_id in _inclination_param_ids
                ]
                if _default_trainable_inclinations:
                    _default_choice = f"param:{_default_trainable_inclinations[0]}"
                else:
                    _default_choice = f"fixed:{default_fixed_inclination(selected_grid)}"
            if _default_choice not in _choice_to_label:
                _default_choice = next(iter(_choice_to_label))

            physical_params_selector = mo.ui.multiselect(
                options={param_names.get(i, f"Parameter {i}"): str(i) for i in _physical_param_ids},
                value=_param_labels(default_physical_params),
                label="Select parameters to include:",
                on_change=update_loaded_emu_config("physical_params", _coerce_physical_params),
            )
            inclination_selector = mo.ui.dropdown(
                options=_inclination_options,
                value=_choice_to_label[_default_choice],
                label="Inclination parameter:",
                on_change=update_loaded_emu_config("inclination_choice", str),
            )
        else:
            physical_params_selector = mo.ui.multiselect(
                options={},
                value=[],
                label="Select parameters to include:"
            )
            inclination_selector = None
    else:
        physical_params_selector = None
        inclination_selector = None
    return inclination_selector, physical_params_selector


@app.cell
def _(
    inclination_selector,
    mo,
    physical_params_selector,
):
    if physical_params_selector is None:
        _controls = mo.md("*Select a grid first*")
    elif inclination_selector is None:
        _controls = physical_params_selector
    else:
        _controls = mo.hstack(
            [physical_params_selector, inclination_selector],
            justify="start",
            align="end",
            gap=1,
        )

    _controls
    return


@app.cell
def _(default_fixed_inclination, grid_selector, inclination_selector, physical_params_selector):
    from types import SimpleNamespace as _SimpleNamespace

    # Keep the rest of the Training Tool unchanged by rebuilding the legacy
    # objects it expects: ``params.value`` contains the selected physical axes
    # plus at most one trainable inclination axis, and fixed_inclination_selector
    # exists only when inclination is baked into a single flux column.
    params = None
    fixed_inclination_selector = None

    if physical_params_selector is not None:
        _selected_params = [int(param_id) for param_id in (physical_params_selector.value or [])]
        _choice = str(inclination_selector.value) if inclination_selector is not None else ""
        if _choice.startswith("param:"):
            _selected_params.append(int(_choice.split(":", 1)[1]))
        elif _choice.startswith("fixed:"):
            fixed_inclination_selector = _SimpleNamespace(value=int(_choice.split(":", 1)[1]))
        else:
            _grid_name = grid_selector.value if grid_selector is not None else None
            fixed_inclination_selector = _SimpleNamespace(value=default_fixed_inclination(_grid_name))

        _merged_params = sorted(set(_selected_params))
        params = _SimpleNamespace(value=[str(param_id) for param_id in _merged_params])

    return fixed_inclination_selector, params


@app.cell
def _(grid_configs, grid_selector, mo, n_components, params):
    # Estimate the combinatorial grid size and the dominant GPU memory footprint
    # before training begins, using the selected parameterization and PCA size.
    high_vram = mo.md("")

    if params is not None and params.value:
        try:
            # The UI stores parameter ids as strings; convert them back to ints for
            # the size heuristics used below.
            _selected = [int(p) for p in params.value]
            _grid_name = grid_selector.value if grid_selector is not None else None
            _grid_config = grid_configs.get(_grid_name) if _grid_name else None
            _point_axes = _grid_config.get("points", {}) if _grid_config else {}

            # Use the selected grid's registered axis sizes.  CV and AGN do not
            # share the same physical-axis cardinalities, and their inclination
            # axes are registry-defined rather than inferred from the parameter ID.
            total_points = 1
            for p in _selected:
                if p in _point_axes:
                    total_points *= len(_point_axes[p])
                else:
                    # Legacy fallback for unknown grids that predate the registry.
                    if p == 11:
                        total_points *= 12
                    elif p == 10:
                        total_points *= 6
                    else:
                        total_points *= 3

            # V11 is the dominant dense tensor in emulator training. In standard
            # mode it scales as (n_components, M, M), so memory grows roughly with
            # the square of the number of grid points.
            n_comp_estimate = n_components.value
            elem_bytes = 8  # float64
            v11_bytes = n_comp_estimate * total_points * total_points * elem_bytes
            # Add rough working-room for factorization and solve buffers on top of
            # the raw V11 tensor itself.
            estimated_total_bytes = v11_bytes * 3
            estimated_gb = estimated_total_bytes * 1.1 / (1024**3) # * 1.1 more closely aligns with observed GPU usage

            # Detect actual GPU VRAM — both total and currently free.
            # Threshold decisions use free memory so other processes' allocations
            # are accounted for (mirrors the fix in emulator.py __init__).
            gpu_vram_total_gb = None
            gpu_vram_free_gb = None
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    gpu_vram_total_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    free_bytes, _ = _torch.cuda.mem_get_info(0)
                    gpu_vram_free_gb = free_bytes / (1024**3)
            except Exception:
                pass

            # Format sizes
            v11_mb = v11_bytes / (1024**2)
            v11_display = f"{v11_mb:.0f} MB" if v11_mb < 1024 else f"{v11_mb/1024:.1f} GB"
            total_display = f"{estimated_gb:.1f} GB"
            single_component_bytes = 2 * total_points * total_points * elem_bytes
            single_component_gb = single_component_bytes / (1024**3)
            memory_efficient_training_display = f"{single_component_gb:.1f} GB"
            fast_cache_gb = v11_bytes / (1024**3)
            fast_cache_display = f"{fast_cache_gb:.1f} GB"
            # Inference cache build peak: retained per-component Cholesky cache
            # plus a one-component workspace while the cache is being built.
            # Mirrors the (cache + workspace) * 1.1 < free * 0.9 test inside
            # Emulator._call_gpu_memory_efficient so this callout agrees with
            # the runtime decision to build a reusable cache or stream blocks.
            fast_cache_build_gb = (v11_bytes + single_component_bytes) * 1.1 / (1024**3)
            fast_cache_build_display = f"{fast_cache_build_gb:.1f} GB"

            if gpu_vram_total_gb is not None and gpu_vram_free_gb is not None:
                # All percentage comparisons are against free VRAM so the
                # callout reflects what is actually available right now.
                usage_pct = (estimated_gb / gpu_vram_free_gb) * 100
                usage_pct_inference = (fast_cache_build_gb / gpu_vram_free_gb) * 100
                vram_info = (
                    f"GPU VRAM: {gpu_vram_total_gb:.1f} GB total "
                    f"&nbsp;|&nbsp; **{gpu_vram_free_gb:.1f} GB** free"
                )
                can_fast_cache_inference = fast_cache_build_gb < gpu_vram_free_gb * 0.9
                inference_note = (
                    f"Inference cache estimate: **~{fast_cache_build_display} required** ({usage_pct_inference:.0f}% of free VRAM)"
                    if can_fast_cache_inference else
                    f"Inference cache estimate: **~{fast_cache_build_display} required** ({usage_pct_inference:.0f}% of free VRAM). Inference caching will be disabled, so MLE/MCMC will stream per component and be much slower. This setup is not recommended."
                )

                if estimated_gb > gpu_vram_free_gb:
                    # Memory-efficient mode never materializes the full V11 tensor;
                    # it only needs a couple of MxM working blocks at a time.
                    efficient_pct = (single_component_gb / gpu_vram_free_gb) * 100

                    if single_component_gb > gpu_vram_free_gb:
                        # Even a single block won't fit — truly too large
                        high_vram = mo.callout(
                            mo.md(f"{mo.icon('lucide:ban')} STOP: Training will be out of memory\n\n"
                                  f"Grid Size: {total_points:,} grid points\n\n"
                                  f"{vram_info}\n\n"
                                  f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Est. Training Usage: **~{memory_efficient_training_display}** ({efficient_pct:.0f}% of free)\n\n"
                                  f"Memory-efficient mode cannot fit a single M×M workspace. Standard mode would require ~{total_display}. Reduce parameters or PCA components.\n\n"
                                  f"{inference_note}"),
                            kind="alert"
                        )
                    elif usage_pct_inference > 100:
                        # Training can technically run in memory-efficient mode, but inference caching is also expected to OOM — warn about the full implications of this setup.
                        high_vram = mo.callout(
                            mo.md(f"{mo.icon('lucide:ban')} STOP: Inference cache will be out of memory\n\n"
                                  f"Grid Size: {total_points:,} grid points\n\n"
                                  f"{vram_info}\n\n"
                                  f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Est. Training Usage: **~{memory_efficient_training_display}** ({efficient_pct:.0f}% of free)\n\n"
                                  f"Memory-efficient mode will be used because standard mode would require ~{total_display}. It builds V11 blocks on-the-fly instead of "
                                  f"storing the full tensor. Mathematically identical results, training may be slightly slower.\n\n"
                                  f"{inference_note}"),
                            kind="alert"
                        )
                    else:
                        # Memory-efficient mode can handle it — report its peak as
                        # the effective usage, not the unachievable standard-mode value.
                        high_vram = mo.callout(
                            mo.md(f"{mo.icon('lucide:triangle-alert')} Memory-efficient training mode enabled\n\n"
                                  f"Grid Size: {total_points:,} grid points\n\n"
                                  f"{vram_info}\n\n"
                                  f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Est. Training Usage: **~{memory_efficient_training_display}** ({efficient_pct:.0f}% of free)\n\n"
                                  f"Memory-efficient mode will be used because standard mode would require ~{total_display}. It builds V11 blocks on-the-fly instead of "
                                  f"storing the full tensor. Mathematically identical results, training may be slightly slower.\n\n"
                                  f"{inference_note}"),
                            kind="warn"
                        )
                elif usage_pct >= 90 or usage_pct_inference >= 90:
                    _high_vram_title = (
                        f"{mo.icon('lucide:ban')} STOP: Inference cache will be out of memory"
                        if usage_pct_inference > 100 else
                        f"{mo.icon('lucide:triangle-alert')} Warning: nearing VRAM limits"
                    )
                    high_vram = mo.callout(
                        mo.md(f"{_high_vram_title}\n\n"
                              f"Grid Size: {total_points:,} grid points\n\n"
                              f"{vram_info}\n\n"
                              f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Est. Training Usage: **~{total_display}** ({usage_pct:.0f}% of free)\n\n"
                              f"Training should fit but memory will be tight.\n\n"
                              f"{inference_note}"),
                        kind="alert" if usage_pct_inference > 100 else "warn"
                    )
                else:
                    # GOOD TRAINING SCENARIO — both training and inference cache fit
                    # comfortably (the >=90% elif above already routed tight cases away).
                    high_vram = mo.callout(
                        mo.md(f"{mo.icon('lucide:check-circle')} Good to go!\n\n"
                              f"Grid Size: {total_points:,} grid points\n\n"
                              f"{vram_info}\n\n"
                              f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Est. Training Usage: **~{total_display}** ({usage_pct:.0f}% of free)\n\n"
                              f"{inference_note}"),
                        kind="success"
                    )
            else:
                # No GPU detected — show estimate without comparison
                high_vram = mo.callout(
                    mo.md(f"{mo.icon('lucide:info')} No NVIDIA GPU detected\n\n"
                          f"Grid Size: {total_points:,} grid points\n\n"
                          f"V11 matrix: ~{v11_display} &nbsp;|&nbsp; Total estimated: **~{total_display}**\n\n"
                          f"No GPU detected. CPU training will be used. Likely very slow for moderate grids."),
                    kind="info"
                )
        except Exception:
            pass

    high_vram
    return


@app.cell
def _(get_loaded_emu_config, mo, update_loaded_emu_config):
    # Smoothing sigma is defined once in Spec_gridinterfaces so the checkbox
    # label always matches the kernel actually applied during grid processing.
    from Speculate_addons.Spec_gridinterfaces import GAUSSIAN_SMOOTHING_SIGMA as _smooth_sigma

    _cfg = get_loaded_emu_config()
    _scales = ["linear", "log", "continuum-normalised"]

    wl_min = mo.ui.number(start=800, stop=8000, value=_cfg['wl_min'] if _cfg and 'wl_min' in _cfg else 850, step=1, label="Min Wavelength (Å):", on_change=update_loaded_emu_config("wl_min", lambda value: int(value)))
    wl_max = mo.ui.number(start=800, stop=8000, value=_cfg['wl_max'] if _cfg and 'wl_max' in _cfg else 1850, step=1, label="Max Wavelength (Å):", on_change=update_loaded_emu_config("wl_max", lambda value: int(value)))

    scale_selector = mo.ui.dropdown(
        options=_scales,
        value=_cfg['scale'] if _cfg and _cfg.get('scale') in _scales else "linear",
        label="Flux Scale:",
        on_change=update_loaded_emu_config("scale")
    )

    use_smoothing = mo.ui.checkbox(value=_cfg['smoothing'] if _cfg and 'smoothing' in _cfg else False, label=f"Smooth Spectra (Gaussian σ={_smooth_sigma:g})", on_change=update_loaded_emu_config("smoothing", bool))

    n_components = mo.ui.slider(start=2, stop=30, value=max(2, min(30, _cfg['n_components'])) if _cfg and 'n_components' in _cfg else 10, step=1, label="PCA Components:",show_value=True, on_change=update_loaded_emu_config("n_components", lambda value: int(value)))

    test_pca_btn = mo.ui.run_button(label="Test PCA Reconstruction", kind="neutral")
    return (
        n_components,
        scale_selector,
        test_pca_btn,
        use_smoothing,
        wl_max,
        wl_min,
    )


@app.cell
def _(
    get_pca_result,
    mo,
    n_components,
    scale_selector,
    test_pca_btn,
    use_smoothing,
    wl_max,
    wl_min,
):
    mo.md("### 3. Wavelength Range, Scale & PCA Components")

    # Pull the shared smoothing sigma so the info text tracks the real kernel.
    from Speculate_addons.Spec_gridinterfaces import GAUSSIAN_SMOOTHING_SIGMA as _smooth_sigma_info

    # Display result using state
    pca_result_display = mo.md(f"_{get_pca_result()}_")

    mo.vstack([
        mo.hstack([
            mo.hstack([wl_min, wl_max], justify="start", align="center"),
            mo.accordion({
                f"{mo.icon('lucide:info')} Wavelength range": mo.md(
                    "Specify the wavelength range to train the emulator on. \n\n The full Sirocco CV grids span 800-8000Å, but choosing a narrower range to focus on "
                    "specific features can speed up training, improve accuracy and lessen the number of PCA components required."
                ),
            }),
        ], align="center", gap=0.5),
        mo.hstack([
            mo.hstack([scale_selector, use_smoothing], justify="start", align="center"),
            mo.accordion({
                f"{mo.icon('lucide:info')} Scale & Smoothing": mo.md(
                    "**Flux Scale**: Possible transformations to the input spectra before PCA may help capture features more efficiently."
                    "Linear, log and continuum-normalised options are standard choices.\n\n"
                    "**Smoothing**: Applying a Gaussian smoothing filter (σ=" + f"{_smooth_sigma_info:g}" + ") can suppress high-frequency noise in the spectra before PCA. This may improve reconstruction performance, but it will also slightly broaden sharp spectral features, so enable it deliberately."
                ),
            }),
        ], align="center", gap=0.5),
        mo.hstack([
            mo.hstack([n_components, test_pca_btn, pca_result_display], justify="start", align="center"),
            mo.accordion({
                f"{mo.icon('lucide:info')} PCA Components": mo.md(
                    "The number of PCA components determines the maximum accuracy achievable and the dimensionality of the emulator's latent space.\n\n The more components, the more accurate the emulator can be, but it also increases training time and has diminishing returns beyond a certain point.\n\n"
                    r"Test the PCA reconstruction quality before training the emulator to see the accuracy.\n\n It is recommended to choose at minimum a 90-95% reconstruction quality, but the optimal number of components depends ones needs. More spectral lines = More components required."
                ),
            }),
        ], align="center", gap=0.5),
    ])          
    return


@app.cell
def _(get_loaded_emu_config, mo, update_loaded_emu_config):
    mo.md("### 4. Training Options")

    _cfg = get_loaded_emu_config()
    _method_options = ["Nelder-Mead", "L-BFGS-B", "CMA-ES"]
    _kernel_labels = {"rbf": "RBF (Squared Exponential)", "matern52": "Matérn-5/2", "matern32": "Matérn-3/2"}
    _default_kernel = _kernel_labels.get(_cfg.get('kernel', ''), "RBF (Squared Exponential)") if _cfg else "RBF (Squared Exponential)"

    method = mo.ui.dropdown(
        options=_method_options,
        value=_cfg.get('method', "Nelder-Mead") if _cfg and _cfg.get('method') in _method_options else "Nelder-Mead",
        label="Optimisation Method:",
        on_change=update_loaded_emu_config("method"),
    )

    max_iter = mo.ui.number(start=100, stop=100000, value=_cfg.get('max_iter', 10000) if _cfg else 10000, step=100, label="Max Iterations:", on_change=update_loaded_emu_config("max_iter", lambda value: int(value)))

    strict_weight_fit = mo.ui.checkbox(
        value=_cfg.get('strict_weight_fit', False) if _cfg else False,
        label="Strict Weight Fit (bypass λ_ξ truncation penalty)",
        on_change=update_loaded_emu_config("strict_weight_fit", bool),
    )

    per_component = mo.ui.checkbox(
        value=_cfg.get('per_component', True) if _cfg else True,
        label="Per-Component Training (optimise each PCA component independently)",
        on_change=update_loaded_emu_config("per_component", bool),
    )

    refine_lambda_xi = mo.ui.checkbox(
        value=_cfg.get('refine_lambda_xi', True) if _cfg else True,
        label="Refine λ_ξ after training (1-D bounded optimisation of truncation noise)",
        on_change=update_loaded_emu_config("refine_lambda_xi", bool),
    )

    kernel_selector = mo.ui.dropdown(
        options={"RBF (Squared Exponential)": "rbf", "Matérn-5/2": "matern52", "Matérn-3/2": "matern32"},
        value=_default_kernel,
        label="GP Kernel:",
        on_change=update_loaded_emu_config("kernel")
    )

    mo.vstack([
        mo.hstack([
            mo.hstack([method, max_iter], justify="start", align="center"),
            mo.accordion({
            f"{mo.icon('lucide:info')} Optimisation Settings": mo.md(
                "**Nelder-Mead** is a robust simplex method that doesn't require gradients but can be slower and struggle with high-dimensional problems.\n\n "
                "**L-BFGS-B** is a quasi-Newton method that uses gradients for **faster** convergence but may struggle finding good minima in complex landscapes.\n\n"
                "**CMA-ES** is a population-based evolutionary strategy that can escape local minima but is computationally intensive but fair better in high-dimensional problems.\n\n"
                "**Max Iterations** sets a hard cap on optimisation steps to prevent runaway training times; increase for larger grids / high dimensional problems."
            ),
        })], align="center", gap=0.5),
        mo.hstack([
            kernel_selector,
            mo.accordion({
                f"{mo.icon('lucide:info')} Kernel info": mo.md(
                    "**RBF** (C∞ smooth) is the classic choice. **Matérn-5/2** (C² smooth) "
                    "and **Matérn-3/2** (C¹ smooth) allow sharper transitions — literature has " 
                    "shown matern kernel tend to outperform other kernels, particularly if the "
                    "weight space have non-smooth features that the RBF can't capture."
                ),
            }),
        ], align="center", gap=0.5),
        mo.hstack([
            strict_weight_fit,
            mo.accordion({
                f"{mo.icon('lucide:info')} Strict weight fit": mo.md(
                    "When enabled, the GP is trained to fit the PCA weights directly "
                    "without the Czekala et al. (2015) truncation noise matrix (λ_ξ). "
                    "This can produce tighter fits at the cost of losing the formal "
                    "flux-space equivalence."
                ),
            }),
        ], align="center", gap=0.5),
        mo.hstack([
            per_component,
            mo.accordion({
                f"{mo.icon('lucide:info')} Per-component training": mo.md(
                    "When enabled, runs a separate low-dimensional optimisation for each PCA component "
                    "instead of one large joint optimisation. Default is enabled as per "
                    "component is a lower dimensional problem, making it easier for the optimiser "
                    "to find minima and leads to faster training overall."
                ),
            }),
        ], align="center", gap=0.5),
        mo.hstack([
            refine_lambda_xi,
            mo.accordion({
                f"{mo.icon('lucide:info')} λ_ξ refinement": mo.md(
                    "After per-component training, run a 1-D bounded optimisation to "
                    "refine the shared λ_ξ (truncation noise) parameter. Higher values lead to more confident predictions. Only applies "
                    "when strict weight fit is disabled (unticked)."
                ),
            }),
        ], align="center", gap=0.5),
    ])
    return kernel_selector, max_iter, method, per_component, refine_lambda_xi, strict_weight_fit


@app.cell
def _(
    Emulator,
    MarimoHDF5Creator,
    fixed_inclination_selector,
    fixed_inclination_to_usecols,
    format_param_tag,
    grid_configs,
    grid_selector,
    has_trainable_inclination,
    logging,
    n_components,
    np,
    os,
    params,
    scale_selector,
    set_pca_result,
    sirocco_grids_path,
    test_pca_btn,
    use_smoothing,
    wl_max,
    wl_min,
):
    if test_pca_btn.value:
        if grid_selector is not None and grid_selector.value and params is not None and params.value:
            # 1. Determine parameters.  format_param_tag() preserves the
            # original concatenated filename convention for sorted param IDs.
            _model_params = tuple(sorted([int(p) for p in params.value]))
            _model_params_str = format_param_tag(_model_params)
            _grid_name = grid_selector.value
            _base_name = _grid_name + "_"
            _smoothing = use_smoothing.value
            _scale = scale_selector.value
            _smooth_suffix = "_smooth" if _smoothing else ""
            _wl_lo = wl_min.value
            _wl_hi = wl_max.value
            _inclination_fixed = not has_trainable_inclination(_model_params, _grid_name)
            _fixed_inc = int(fixed_inclination_selector.value) if fixed_inclination_selector is not None else 55
            _fixed_inc_suffix = f"_{_fixed_inc}inc" if _inclination_fixed else ""
            _grid_file_name_pca = f"{_base_name}grid_{_model_params_str}_{_scale}{_smooth_suffix}{_fixed_inc_suffix}_{_wl_lo}-{_wl_hi}AA"
            _grid_file_path_pca = f'Grid-Emulator_Files/{_grid_file_name_pca}.npz'
            _grid_ready = True

            # If grid file doesn't exist, create it first
            if not os.path.exists(_grid_file_path_pca):
                set_pca_result("Building grid file...")
                try:
                    _scale = scale_selector.value
                    _wl_range = (wl_min.value, wl_max.value)
                    _grid_path = sirocco_grids_path / _grid_name

                    if _grid_name in grid_configs:
                        _grid_config = grid_configs[_grid_name]
                        _usecols = (
                            fixed_inclination_to_usecols(_grid_config["usecols"], _fixed_inc, _grid_name)
                            if _inclination_fixed else _grid_config["usecols"]
                        )
                        _grid = _grid_config["class"](
                            path=str(_grid_path) + "/",
                            usecols=_usecols,
                            wl_range=_wl_range,
                            model_parameters=_model_params,
                            scale=_scale,
                            smoothing=_smoothing,
                        )

                        # Set up logging
                        os.makedirs('logs', exist_ok=True)
                        logging.basicConfig(
                            level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('logs/grid_processing.log', mode='w')],
                            force=True,
                        )

                        _keyname = ["param{}{{}}".format(i) for i in _model_params]
                        _keyname = ''.join(_keyname)

                        os.makedirs('Grid-Emulator_Files', exist_ok=True)
                        _creator = MarimoHDF5Creator(
                            _grid,
                            _grid_file_path_pca,
                            key_name=_keyname,
                            wl_range=_wl_range,
                        )
                        _creator.process_grid()

                        _data = np.load(_grid_file_path_pca, allow_pickle=True)
                        set_pca_result(f"Grid built ({_data['grid_points'].shape[0]} spectra). Running PCA...")
                    else:
                        set_pca_result(f"Error: Unknown grid config for {_grid_name}")
                        _grid_ready = False
                except Exception as e:
                    set_pca_result(f"Grid build error: {e}")
                    _grid_ready = False

            # Now run the PCA test
            if _grid_ready:
                try:
                    set_pca_result("Running PCA...")
                    _exp_var, _n_comps = Emulator.test_pca(_grid_file_path_pca, n_components=n_components.value)
                    set_pca_result(f"Variance: {_exp_var:.5f} ({_exp_var*100:.3f}%) [{_n_comps} comps]")
                except Exception as e:
                    set_pca_result(f"Error: {e}")
        else:
             set_pca_result("Select grid/params first")
    return


@app.cell
def _(
    clear_trained_emu_cache,
    fixed_inclination_selector,
    format_param_tag,
    get_training_trigger,
    grid_selector,
    has_trainable_inclination,
    mo,
    n_components,
    np,
    os,
    params,
    scale_selector,
    set_loss_history,
    use_smoothing,
    wl_max,
    wl_min,
):
    mo.md("### 5. Start Training")

    # Register dependency on training completion
    _ = get_training_trigger()

    # Recompute the expected emulator filename reactively so the UI can warn when
    # the current configuration would overwrite an existing trained model.
    _emu_btn_label = f"{mo.icon('lucide:rocket')} Train Emulator"
    _emu_btn_kind = "success"
    _emu_info_text = "Click to begin training. This may take several minutes to hours depending on grid size and hardware."

    if grid_selector is not None and grid_selector.value and params is not None and params.value:
        # Duplicate the naming logic from the training cell so this preview stays
        # consistent with the file that would actually be written on train.  The
        # registry formatter keeps the original concatenated increasing-ID tag.
        _model_params = tuple(sorted([int(p) for p in params.value]))
        _model_params_str = format_param_tag(_model_params)
        _wl_range = (wl_min.value, wl_max.value)
        _scale = scale_selector.value
        _grid_name = grid_selector.value
        _smooth_tag = '_smooth' if use_smoothing.value else ''

        # Standardize base name
        _base_name = _grid_name + "_"

        # Inclination-aware models follow a shorter filename convention because
        # the inclination axis is already part of the parameter tuple; otherwise
        # the selected fixed inclination is encoded into the name explicitly.
        _inclination_fixed = not has_trainable_inclination(_model_params, _grid_name)
        _fixed_inc = int(fixed_inclination_selector.value) if fixed_inclination_selector is not None else 55
        if not _inclination_fixed:
            _chk_name = f'{_base_name}emu_{_model_params_str}_{_scale}{_smooth_tag}_{_wl_range[0]}-{_wl_range[1]}AA_{n_components.value}PCA'
        else:
            _chk_name = f'{_base_name}emu_{_model_params_str}_{_scale}{_smooth_tag}_{_fixed_inc}inc_{_wl_range[0]}-{_wl_range[1]}AA_{n_components.value}PCA'

        if os.path.exists(f'Grid-Emulator_Files/{_chk_name}.npz'):
            _emu_btn_label = f"{mo.icon('lucide:refresh-cw')} Re-train (An Emulator Already Exists)"
            _emu_btn_kind = "warn"
            _emu_info_text = f"**Emulator found:** `{_chk_name}.npz`\n\nClicking **re-train** overwrites existing. Clicking **continue training** will resume from the existing model's state. Use the **Load Existing Emulator** button only when you want to build the GP diagnostics cache."

            # If a prior training run saved loss_history, hydrate it now so the
            # diagnostics panel shows something useful before retraining starts.
            try:
                # Load minimal data from npz to check for loss_history
                # We do checking inside try/except to avoid crashes on old files
                with np.load(f'Grid-Emulator_Files/{_chk_name}.npz', allow_pickle=True) as _data:
                    if 'loss_history' in _data:
                         set_loss_history(_data['loss_history'].tolist())
                    else:
                         set_loss_history([])
            except Exception:
                set_loss_history([])

        else:
             # If emulator doesn't exist (new config), clear the graph and diagnostics
             set_loss_history([])
             clear_trained_emu_cache()

    train_button = mo.ui.run_button(label=_emu_btn_label, kind=_emu_btn_kind)
    continue_train_button = mo.ui.run_button(
        label=f"{mo.icon('lucide:fast-forward')} Continue Training",
        kind="neutral"
    )

    _callout = mo.callout(
        mo.md(_emu_info_text),
        kind="info" if _emu_btn_kind == "success" else "warn"
    )

    if _emu_btn_kind == "warn":
        # Emulator exists: show both retrain and continue training buttons
        _buttons = mo.hstack([train_button, continue_train_button], justify="start", gap=1)
    else:
        _buttons = train_button

    mo.vstack([_callout, _buttons])
    return continue_train_button, train_button


@app.cell
def _(
    MarimoHDF5Creator,
    continue_train_button,
    fixed_inclination_selector,
    fixed_inclination_to_usecols,
    format_param_tag,
    grid_configs,
    grid_selector,
    has_trainable_inclination,
    kernel_selector,
    logging,
    max_iter,
    method,
    mo,
    n_components,
    np,
    os,
    params,
    per_component,
    refine_lambda_xi,
    scale_selector,
    sirocco_grids_path,
    strict_weight_fit,
    train_button,
    use_smoothing,
    wl_max,
    wl_min,
):
    # This cell resolves the on-disk filenames, reports the planned training
    # configuration, and creates the processed grid file on demand.
    active_training_config = None
    emu_exists = False
    emu_file_name = ""
    grid_file_name = ""
    training_setup_display = mo.md("*Configure all settings and click 'Train Emulator' to start*")

    if (train_button.value or continue_train_button.value) and grid_selector is not None and params is not None:
        # Normalize the selected UI controls into the exact identifiers used by the
        # downstream grid-processing and emulator-training code.  The formatted
        # parameter tag becomes part of both processed-grid and emulator filenames.
        model_parameters = tuple(sorted([int(p) for p in params.value]))
        model_parameters_str = format_param_tag(model_parameters)
        wl_range = (wl_min.value, wl_max.value)
        scale = scale_selector.value
        smoothing = use_smoothing.value
        smooth_tag = '_smooth' if smoothing else ''
        grid_name = grid_selector.value
        grid_path = sirocco_grids_path / grid_name

        # Standardize base name from grid folder name (e.g. no-bl -> no_bl)
        base_name = grid_name + "_"

        # Include the flux scale, smoothing tag, and wavelength range in the grid
        # filename because the processed spectra themselves differ when any of these
        # options change.  Without the wavelength range a cached grid from a previous
        # run with a different range would be silently reused.
        inclination_fixed = not has_trainable_inclination(model_parameters, grid_name)
        fixed_inc = int(fixed_inclination_selector.value) if fixed_inclination_selector is not None else 55
        fixed_inc_tag = f"_{fixed_inc}inc" if inclination_fixed else ""
        grid_file_name = f"{base_name}grid_{model_parameters_str}_{scale}{smooth_tag}{fixed_inc_tag}_{wl_range[0]}-{wl_range[1]}AA"

        _grid_config = grid_configs.get(grid_name, {})
        _inclination_param_ids = set(_grid_config.get("inclination_param_ids", ()))
        _physical_params = [
            param_id for param_id in model_parameters
            if param_id not in _inclination_param_ids
        ]
        _trainable_inclinations = [
            param_id for param_id in model_parameters
            if param_id in _inclination_param_ids
        ]
        _inclination_choice = (
            f"param:{_trainable_inclinations[0]}"
            if _trainable_inclinations
            else f"fixed:{fixed_inc}"
        )
        active_training_config = {
            "grid_name": grid_name,
            "params": list(model_parameters),
            "physical_params": _physical_params,
            "inclination_choice": _inclination_choice,
            "fixed_inclination": fixed_inc,
            "wl_min": int(wl_range[0]),
            "wl_max": int(wl_range[1]),
            "scale": scale,
            "smoothing": bool(smoothing),
            "n_components": int(n_components.value),
            "method": method.value,
            "max_iter": int(max_iter.value),
            "strict_weight_fit": bool(strict_weight_fit.value),
            "per_component": bool(per_component.value),
            "refine_lambda_xi": bool(refine_lambda_xi.value),
            "kernel": kernel_selector.value,
        }

        # Determine emulator file name based on Speculate_dev.py conventions
        if not inclination_fixed:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}{smooth_tag}_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'
        else:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}{smooth_tag}_{fixed_inc}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'

        # Auto-process the grid only when the processed NPZ for this exact config
        # does not already exist.
        grid_file_path_check = f'Grid-Emulator_Files/{grid_file_name}.npz'
        process_grid_auto = not os.path.isfile(grid_file_path_check)

        # Shared smoothing sigma keeps this summary in sync with the kernel.
        from Speculate_addons.Spec_gridinterfaces import GAUSSIAN_SMOOTHING_SIGMA as _smooth_sigma_cfg

        _config_lines = [
            f"- **Grid:** `{grid_name}`",
            f"- **Grid Path:** `{grid_path}`",
            f"- **Parameters:** {model_parameters}",
            f"- **Inclination parameter:** {f'Fixed {fixed_inc}°' if inclination_fixed else 'Trainable'}",
            f"- **Wavelength Range:** {wl_range[0]}-{wl_range[1]} Å",
            f"- **Flux Scale:** {scale}",
            f"- **Smoothing:** {f'Yes (Gaussian σ={_smooth_sigma_cfg:g})' if smoothing else 'No'}",
            f"- **PCA Components:** {n_components.value}",
            f"- **Method:** {method.value}",
            f"- **Max Iterations:** {max_iter.value}",
            f"- **Strict Weight Fit:** {'Yes (λ_ξ penalty bypassed)' if strict_weight_fit.value else 'No (standard Czekala+2015)'}",
            f"- **Per-Component Training:** {'Yes' if per_component.value else 'No'}",
            f"- **Refine λ_ξ:** {'Yes' if refine_lambda_xi.value else 'No'}",
            f"- **GP Kernel:** {kernel_selector.value}",
            f"- **Process Grid:** {'Auto (file not found; creating new)' if process_grid_auto else 'Auto (file found; loading existing)'}",
            f"- **Grid File:** `{grid_file_name}.npz`",
            f"- **Emulator File:** `{emu_file_name}.npz`",
        ]
        config_md = mo.accordion({
            f"{mo.icon('lucide:chart-bar')} Training Configuration": mo.md("\n".join(_config_lines))
        })

        # Build the concrete grid interface that knows how to read this grid's raw
        # spectra and convert them into the processed training representation.
        grid = None
        grid_error_md = None
        if grid_name in grid_configs:
            grid_config = grid_configs[grid_name]
            # If inclination is not a trained axis, replace the default flux
            # column with the grid-specific fixed observer column.  AGN uses a
            # different angle set from CV, so this must go through the registry.
            grid_usecols = (
                fixed_inclination_to_usecols(grid_config["usecols"], fixed_inc, grid_name)
                if inclination_fixed else grid_config["usecols"]
            )
            grid = grid_config["class"](
                path=str(grid_path) + "/",
                usecols=grid_usecols,
                wl_range=wl_range,
                model_parameters=model_parameters,
                scale=scale,
                smoothing=smoothing
            )
        else:
            grid_error_md = mo.md(f"{mo.icon('lucide:triangle-alert')} **Error:** Unknown grid configuration for `{grid_name}`")

        # Reuse the selected grid interface's own metadata so processed-grid
        # summaries display physical parameter names instead of bare param IDs.
        try:
            param_description_map = grid.parameters_description() if grid is not None else {}
        except Exception:
            param_description_map = {}

        def _format_param_label(param_name):
            """Render a friendly parameter label while retaining the raw key."""
            raw_name = param_name.decode() if isinstance(param_name, bytes) else str(param_name)
            friendly_name = param_description_map.get(raw_name)
            if friendly_name and friendly_name != raw_name:
                return f"{friendly_name} (`{raw_name}`)"
            return f"`{raw_name}`"

        # Grid processing is the expensive preprocessing step that converts the raw
        # Sirocco files into the compact NPZ consumed by emulator training.
        if process_grid_auto and grid is not None:
            # Log grid processing to disk because this can take a long time and the
            # notebook UI alone is not sufficient for post-mortem debugging.
            os.makedirs('logs', exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/grid_processing.log', mode='w')
                ],
                force=True
            )

            status_md = mo.md(f"{mo.icon('lucide:wrench')} **Processing grid...** (logging to `logs/grid_processing.log`)")

            # key_name reproduces the parameter ordering inside each flux-data entry
            # of the processed NPZ so later code can decode the stored spectra.
            keyname = ["param{}{{}}".format(i) for i in model_parameters]
            keyname = ''.join(keyname)

            os.makedirs('Grid-Emulator_Files', exist_ok=True)
            creator = MarimoHDF5Creator(
                grid,
                f'Grid-Emulator_Files/{grid_file_name}.npz',
                key_name=keyname,
                wl_range=wl_range
            )
            creator.process_grid()

            # Reload the processed grid immediately to surface its shape and unique
            # parameter values in the notebook before training begins.
            data = np.load(f'Grid-Emulator_Files/{grid_file_name}.npz', allow_pickle=True)
            grid_points = data['grid_points']

            _grid_lines = [
                f"- **Grid shape:** {grid_points.shape}",
                "- **Unique values per parameter:**",
            ]
            for i, param_name in enumerate(data['param_names']):
                unique_vals = np.unique(grid_points[:, i])
                _grid_lines.append(f"  - {_format_param_label(param_name)}: `{unique_vals}`")

            results_md = mo.vstack([
                mo.callout(
                    mo.md(f"{mo.icon('lucide:check-circle')} **Grid processed successfully.**"),
                    kind="success",
                ),
                mo.accordion({
                    f"{mo.icon('lucide:table-properties')} Processed Grid Details": mo.md("\n".join(_grid_lines))
                }),
            ])
        elif grid_error_md is not None:
            results_md = grid_error_md
        else:
            # If the grid was already processed, summarize the cached NPZ instead
            # of rebuilding it.
            if os.path.isfile(f'Grid-Emulator_Files/{grid_file_name}.npz'):
                data = np.load(f'Grid-Emulator_Files/{grid_file_name}.npz', allow_pickle=True)
                grid_points = data['grid_points']

                results_md = mo.accordion({
                    f"{mo.icon('lucide:check-circle')} Existing Grid File": mo.md(
                        f"- **Grid shape:** {grid_points.shape}"
                    )
                })
            else:
                results_md = mo.md(f"{mo.icon('lucide:triangle-alert')} **Error:** Grid file `{grid_file_name}.npz` not found. Enable 'Process grid' to create it.")

        # Check if emulator exists
        if os.path.isfile(f'Grid-Emulator_Files/{emu_file_name}.npz'):
            emu_status = mo.md(f"{mo.icon('lucide:info')} **Emulator `{emu_file_name}.npz` already exists.** Uncheck 'Process grid' to use it or continue to retrain.")
            emu_exists = True
        else:
            emu_status = mo.md(f"{mo.icon('lucide:file-text')} **Emulator `{emu_file_name}.npz` does not exist.** Ready to train.")
            emu_exists = False

        training_setup_display = mo.vstack([config_md, results_md, emu_status])

    training_setup_display
    return active_training_config, emu_file_name, grid_file_name


@app.cell
def _(mo):
    mo.md("---")
    mo.md("## Training Results")
    mo.md("*Results will appear here after training completes*")
    return


@app.cell
def _(
    Emulator,
    active_training_config,
    alt,
    clear_trained_emu_cache,
    continue_train_button,
    emu_file_name,
    get_console_logs,
    get_training_status,
    grid_file_name,
    kernel_selector,
    max_iter,
    method,
    mo,
    n_components,
    np,
    os,
    pd,
    per_component,
    refine_lambda_xi,
    set_console_logs,
    set_diagnostics_emu_display,
    set_loaded_emu_config,
    set_loss_history,
    set_trained_emu,
    set_training_status,
    set_training_trigger,
    strict_weight_fit,
    sys,
    time,
    train_button,
):
    import matplotlib.pyplot as plt

    training_complete = False
    emu = None
    is_continue = continue_train_button.value and not train_button.value

    if (train_button.value or continue_train_button.value) and grid_file_name:
        import contextlib

        clear_trained_emu_cache()

        # Clear previous history only for fresh training (not continue)
        if not is_continue:
            set_loss_history([])
        set_console_logs("")
        set_training_status(None)

        # Choose status text based on mode
        if is_continue:
            status_box = mo.md(f"{mo.icon('lucide:fast-forward')} **Continuing emulator training...** Check the console below for progress.")
        else:
            status_box = mo.md(f"{mo.icon('lucide:rocket')} **Training emulator...** Check the console below for progress.")

        # Live Logger setup
        log_buffer = []
        current_chart = [None] # Mutable container for the latest chart
        ui_state = {'last_update': 0}

        def update_ui(active_status, chart=None, force=False):
            if chart is not None:
                current_chart[0] = chart

            current_time = time.time()
            # Throttle UI updates to avoid flashing (max 4 times per second)
            if not force and (current_time - ui_state['last_update'] < 0.25):
                return

            ui_state['last_update'] = current_time

            text = "".join(log_buffer)

            # Build accordion content
            accordion_content = []
            if current_chart[0]:
                 # Add chart to accordion
                accordion_content.append(mo.md(f"### {mo.icon('lucide:trending-up')} Live Training Loss"))
                # Use chart directly instead of ui wrapper to reduce flashing
                accordion_content.append(current_chart[0])
                accordion_content.append(mo.md("---"))

            # Add logs to accordion
            accordion_content.append(mo.md(f"```text\n{text}\n```"))

            elements = [
                active_status,
                mo.accordion({"Training Progress & Logs": mo.vstack(accordion_content)})
            ]

            mo.output.replace(mo.vstack(elements))

        class LiveLogger:
            def write(self, text):
                log_buffer.append(text)
                # Update UI with current status (preserves chart)
                update_ui(status_box)
            def flush(self):
                # Force UI refresh so status prints appear immediately
                update_ui(status_box, force=True)

        logger = LiveLogger()
        update_ui(status_box, force=True)

        try:
            with contextlib.redirect_stdout(logger):
                print("--- Starting Emulator Training ---")

                grid_path_npz = f'Grid-Emulator_Files/{grid_file_name}.npz'
                print(f"Loading grid data: {grid_path_npz}")

                # Diagnostic: Check file integrity
                try:
                    file_size = os.path.getsize(grid_path_npz)
                    print(f"File size: {file_size / (1024*1024):.2f} MB")

                    print("Verifying .npz file structure...")
                    with np.load(grid_path_npz, allow_pickle=True) as verification_data:
                        print(f"Keys found: {list(verification_data.keys())}")
                        if 'flux' in verification_data:
                            print(f"Flux shape: {verification_data['flux'].shape}")
                            # rapid check for NaNs
                            if np.isnan(verification_data['flux']).any():
                                print("⚠️ WARNING: NaNs detected in flux data!")
                        if 'grid_points' in verification_data:
                            print(f"Grid points shape: {verification_data['grid_points'].shape}")
                    print("File verification passed.")
                except Exception as e:
                    print(f"❌ Error inspecting grid file: {e}")
                    print("The file might be corrupted. Try selecting optional parameters again to trigger regeneration, or delete the file manually.")
                    raise e

                sys.stdout.flush()

                # Create or load emulator depending on mode
                if is_continue:
                    # Continue training: load existing emulator
                    emu_path = f'Grid-Emulator_Files/{emu_file_name}.npz'
                    print(f"Loading existing emulator for continued training: {emu_path}")
                    sys.stdout.flush()

                    emu = Emulator.load(emu_path)

                    saved_per_component = bool(getattr(emu, 'per_component', False))
                    saved_strict_weight_fit = bool(getattr(emu, 'strict_weight_fit', False))
                    emu.per_component = bool(per_component.value)
                    emu.strict_weight_fit = bool(strict_weight_fit.value)

                    # Guard: warn if the UI kernel selector differs from the
                    # kernel baked into the saved emulator.  Training always
                    # uses the kernel stored in the .npz, so a mismatch would
                    # silently ignore the user's dropdown selection.
                    if emu.kernel != kernel_selector.value:
                        print(f"⚠️  Kernel mismatch — loaded emulator uses '{emu.kernel}', "
                              f"but UI has '{kernel_selector.value}' selected. "
                              f"Continuing with the emulator's original kernel ('{emu.kernel}').")

                    if saved_per_component != emu.per_component or saved_strict_weight_fit != emu.strict_weight_fit:
                        print("Continue settings override saved training flags:")
                        print(f"  per_component: {saved_per_component} → {emu.per_component}")
                        print(f"  strict_weight_fit: {saved_strict_weight_fit} → {emu.strict_weight_fit}")

                    # Preserve existing loss history so the plot is cumulative
                    prev_history = list(emu.loss_history) if hasattr(emu, 'loss_history') and emu.loss_history else []
                    set_loss_history(prev_history)

                    print(f"Emulator loaded. {emu.ncomps} PCA components, kernel='{emu.kernel}'.")
                    print(f"Previous training iterations: {len(prev_history)}")
                    print(f"Continuing optimisation using {method.value} for {max_iter.value} more iterations...")
                    sys.stdout.flush()
                else:
                    # Fresh training: create emulator from grid
                    # Matching arguments from Speculate_dev.py
                    # block_diagonal=True and svd_solver="full" are used there
                    print("Initializing Emulator (running PCA/SVD)...")
                    sys.stdout.flush()

                    emu = Emulator.from_grid(
                        grid_path_npz,
                        n_components=n_components.value,
                        svd_solver="full",
                        block_diagonal=True,
                        strict_weight_fit=strict_weight_fit.value,
                        per_component=per_component.value,
                        kernel=kernel_selector.value
                    )

                    print(f"Grid loaded. Initialized {emu.ncomps} PCA components.")
                    print(f"Starting optimisation using {method.value} (max_iter={max_iter.value})...")
                    sys.stdout.flush()

                # --- Live Plotting Setup ---

                # Callback for live plotting
                callback_state = {'counter': 0}

                def training_callback(xk):
                    callback_state['counter'] += 1

                    # Update only if 10 iterations passed
                    if callback_state['counter'] % 10 == 0:

                        # Update Training Loss
                        if hasattr(emu, 'loss_history') and len(emu.loss_history) > 0:
                            # Update global state - Marimo will react to this eventually
                            # We copy the list to ensure change detection
                            current_history = list(emu.loss_history)
                            set_loss_history(current_history)

                            # Generate live chart frame
                            df = pd.DataFrame({
                                'Iteration': range(len(current_history)),
                                'Loss': current_history
                            })
                            current_iter = len(current_history)

                            base = alt.Chart(df).encode(
                                x=alt.X('Iteration:Q', axis=alt.Axis(title='Iteration')),
                                y=alt.Y('Loss:Q', axis=alt.Axis(title='Negative Log-Likelihood'), scale=alt.Scale(zero=False)),
                            )

                            line = base.mark_line(color='#3b82f6').properties(
                                title=f'Training Progress (Iter {current_iter})',
                                width=600,
                                height=300
                            )

                            if len(current_history) > 0:
                                min_val = min(current_history)
                                min_rule = alt.Chart(pd.DataFrame({'y': [min_val]})).mark_rule(
                                    color='green', strokeDash=[5,5], size=2
                                ).encode(y='y')
                                chart = (line + min_rule)
                            else:
                                chart = line

                            # Update UI with new chart
                            update_ui(status_box, chart=chart)

                # Train emulator with callback
                # Note: disp=False preferred to avoid cluttered output interfering with graph
                emu.train(
                    optimizer=method.value.lower(),
                    refine_lambda_xi=refine_lambda_xi.value,
                    options=dict(maxiter=max_iter.value, disp=False),
                    callback=training_callback,
                )

                print("--- Training Finished ---")

                # Save emulator
                os.makedirs('Grid-Emulator_Files', exist_ok=True)
                print(emu)
                emu.save(f'Grid-Emulator_Files/{emu_file_name}.npz')
                print(f"Emulator saved to Grid-Emulator_Files/{emu_file_name}.npz")
                training_complete = True
                if active_training_config is not None:
                    set_loaded_emu_config(dict(active_training_config))
                set_trained_emu(emu)
                set_diagnostics_emu_display(emu_file_name)

                # Save persistent state
                set_console_logs("".join(log_buffer))
                # Also save loss history
                if hasattr(emu, 'loss_history'):
                    set_loss_history(list(emu.loss_history))

                # Trigger button update after state needed by rebuilt controls
                # is already available.
                set_training_trigger(lambda v: v + 1)

            # Create result display
            train_result = mo.md(f"""
            {mo.icon('lucide:check-circle')} **Training complete!**
            """)

            set_training_status(train_result)
            # Update final UI
            update_ui(train_result)

        except Exception as e:
            error_result = mo.md(f"""
            {mo.icon('lucide:x-circle')} **Training failed!**

            **Error:** {str(e)}

            Check the logs for more details.
            """)
            set_console_logs("".join(log_buffer))
            set_training_status(error_result)
            clear_trained_emu_cache(force=True)
            update_ui(error_result)

    else:
        # Check for persisted results (from previous run)
        last_status = get_training_status()
        last_logs = get_console_logs()

        if last_status is not None:
             mo.output.replace(mo.vstack([
                last_status,
                mo.accordion({"Training Console Output": mo.md(f"```text\n{last_logs}\n```")})
            ]))
        else:
            mo.md("*Train an emulator to see results*")
    return


@app.cell
def _(get_trained_emu, mo, np):
    # ── GP Weight Diagnostics: UI setup ──
    # After training, the GP diagnostics panel lets the user inspect how well
    # the Gaussian Processes map the parameter space to PCA weight values.
    # The X-axis selector chooses the parameter to vary; all other parameters
    # are held at a single grid value controlled by the "Fixed Parameter Index"
    # slider.  The component range selects which PCA components to plot.
    _emu = get_trained_emu()

    if _emu is not None:
        _n_params = _emu.grid_points.shape[1]

        from Speculate_addons.grid_registry import infer_grid_name as _infer_grid_name, parameter_label_map as _parameter_label_map

        # Parameter descriptions lookup
        _grid_name = _infer_grid_name(getattr(_emu, 'name', None))
        _desc_map = dict(_parameter_label_map(_grid_name))

        # Build parameter options: map display name -> column index
        _param_indices = [int(pn.replace("param", "")) for pn in _emu.param_names]
        _param_options = {}
        for _col_idx, _p_idx in enumerate(_param_indices):
            _desc = _desc_map.get(_p_idx, f"param{_p_idx}")
            _param_options[f"{_desc} (param{_p_idx})"] = str(_col_idx)

        gp_xaxis_selector = mo.ui.dropdown(
            options=_param_options,
            value=list(_param_options.keys())[0],
            label="Varying Parameter (X-axis):"
        )

        _min_unique = min(len(np.unique(_emu.grid_points[:, _i])) for _i in range(_n_params))
        gp_fixed_slider = mo.ui.slider(
            start=0, stop=_min_unique - 1, value=0, step=1,
            label="Fixed Grid Position (low → high):", show_value=True
        )

        gp_comp_start = mo.ui.slider(
            start=0, stop=_emu.ncomps - 1, value=0, step=1,
            label="From Component:", show_value=True
        )
        gp_comp_end = mo.ui.slider(
            start=0, stop=_emu.ncomps - 1, value=min(5, _emu.ncomps - 1), step=1,
            label="To Component:", show_value=True
        )
    else:
        gp_xaxis_selector = None
        gp_fixed_slider = None
        gp_comp_start = None
        gp_comp_end = None
    return gp_comp_end, gp_comp_start, gp_fixed_slider, gp_xaxis_selector


@app.cell
def _(
    alt,
    get_trained_emu,
    gp_comp_end,
    gp_comp_start,
    gp_fixed_slider,
    gp_xaxis_selector,
    mo,
    np,
    pd,
):
    # GP diagnostics - placeholder when no emulator is loaded
    gp_figure = mo.md("")
    _emu = get_trained_emu()

    if _emu is not None and gp_xaxis_selector is not None and gp_xaxis_selector.value is not None:
        # Read UI values
        _not_fixed_col = int(gp_xaxis_selector.value)
        _fixed_idx = gp_fixed_slider.value
        _comp_start = gp_comp_start.value
        _comp_end = gp_comp_end.value
        if _comp_end < _comp_start:
            _comp_end = _comp_start

        _n_params = _emu.grid_points.shape[1]

        # Get unique values per parameter dimension
        _unique_per_dim = [np.unique(_emu.grid_points[:, _i]) for _i in range(_n_params)]

        # Build parameter combinations: vary _not_fixed_col, fix others at _fixed_idx.
        # Map slider proportionally so 0/1/2 → low/mid/high even for dimensions
        # with more unique values (e.g. inclination).
        _slider_max = min(len(v) for v in _unique_per_dim) - 1
        _fixed_values = []
        for _i in range(_n_params):
            _vals = _unique_per_dim[_i]
            if _i == _not_fixed_col:
                _fixed_values.append(_vals)  # all values for varying param
            else:
                _idx = round(_fixed_idx * (len(_vals) - 1) / _slider_max) if _slider_max > 0 else 0
                _fixed_values.append(_vals[_idx])  # proportionally mapped value

        # Create grid point combinations for actual weight lookup
        _params_list = []
        for _j in range(len(_fixed_values[_not_fixed_col])):
            _point = []
            for _i in range(_n_params):
                if _i == _not_fixed_col:
                    _point.append(_fixed_values[_i][_j])
                else:
                    _point.append(float(_fixed_values[_i]))
            _params_list.append(tuple(_point))
        _params_arr = np.array(_params_list)

        # Get weight indices and actual weights at grid points
        _idxs = np.array([_emu.get_index(_p) for _p in _params_arr])
        _weights = _emu.weights[_idxs.astype("int")].T  # (ncomps, n_varying_points)

        # X-axis: unique values of the varying parameter
        _param_x = _unique_per_dim[_not_fixed_col]

        # GP predictions along a fine test grid
        _param_x_test = np.linspace(_param_x.min(), _param_x.max(), 100)
        _Xtest = []
        for _j in range(len(_param_x_test)):
            _point = []
            for _i in range(_n_params):
                if _i == _not_fixed_col:
                    _point.append(_param_x_test[_j])
                else:
                    _point.append(float(_fixed_values[_i]))
            _Xtest.append(tuple(_point))
        _Xtest = np.array(_Xtest)

        # --- Universal Noise Variance Calculation ---
        # λ_ξ (lambda_xi) from Czekala et al. (2015) assigns a truncation noise
        # variance to each PCA weight.  The noise bar on each grid-point scatter
        # marker shows ±σ_k = sqrt( diag((ΦᵀΦ)⁻¹)_k / λ_ξ ),  where Φ is
        # the eigenspectra matrix.  This visualises how much freedom the GP has
        # to deviate from the exact PCA weights at each training point.
        _dots = _emu.eigenspectra @ _emu.eigenspectra.T
        _dots_inv_diag = np.diag(np.linalg.inv(_dots))

        def _predict_selected_components_streaming(_components):
            """Evaluate only the plotted PCA components without caching all V11 factors."""
            if not getattr(_emu, "block_diagonal", False):
                _all_mus, _all_covs = _emu(_Xtest, full_cov=False, reinterpret_batch=True)
                return (
                    {_comp: _all_mus[:, _comp] for _comp in _components},
                    {_comp: np.sqrt(np.maximum(_all_covs[:, _comp], 0.0)) for _comp in _components},
                    None,
                )

            try:
                import torch as _torch
                from Starfish.emulator.emulator import TORCH_FLOAT64 as _TORCH_DTYPE
                from Starfish.emulator.kernels import _apply_kernel_gpu_inplace as _apply_kernel_gpu_inplace_diag
            except Exception as _import_error:
                return {}, {}, f"GPU diagnostics dependencies are unavailable: {_import_error}"

            if not _torch.cuda.is_available():
                return {}, {}, "GPU diagnostics require CUDA for this loaded emulator."

            _device = _torch.device("cuda")
            _dtype = _TORCH_DTYPE or _torch.float64
            _means = {}
            _sigmas = {}

            try:
                with _torch.no_grad():
                    _grid_gpu = _torch.from_numpy(_emu._grid_points_norm).to(_device, _dtype)
                    _query_norm = (_Xtest - _emu._param_min) / _emu._param_range
                    _query_gpu = _torch.from_numpy(_query_norm).to(_device, _dtype)
                    _w_hat_stacked = _emu.w_hat.reshape(_emu.ncomps, -1)

                    for _comp in _components:
                        _variance = float(_emu.variances[_comp])
                        _variance_gpu = _torch.tensor(_variance, device=_device, dtype=_dtype)
                        _lengthscale_gpu = _torch.from_numpy(_emu.lengthscales[_comp]).to(_device, _dtype)

                        _grid_scaled = _grid_gpu / _lengthscale_gpu
                        _query_scaled = _query_gpu / _lengthscale_gpu

                        _v11 = _torch.cdist(_grid_scaled, _grid_scaled, p=2.0)
                        _apply_kernel_gpu_inplace_diag(_v11, _variance_gpu, _emu.kernel)

                        _jitter = max(1e-5, 1e-6 * _variance)
                        if _emu.strict_weight_fit:
                            _v11.diagonal().add_(_jitter)
                        else:
                            _v11.diagonal().add_(float(_dots_inv_diag[_comp]) / _emu.lambda_xi + _jitter)

                        _L = _torch.linalg.cholesky(_v11)
                        del _v11

                        _w_hat_gpu = _torch.from_numpy(_w_hat_stacked[_comp]).to(_device, _dtype).unsqueeze(-1)
                        _alpha = _torch.cholesky_solve(_w_hat_gpu, _L)

                        _v12 = _torch.cdist(_grid_scaled, _query_scaled, p=2.0)
                        _apply_kernel_gpu_inplace_diag(_v12, _variance_gpu, _emu.kernel)
                        _v22 = _torch.cdist(_query_scaled, _query_scaled, p=2.0)
                        _apply_kernel_gpu_inplace_diag(_v22, _variance_gpu, _emu.kernel)

                        _mu = _torch.matmul(_v12.T, _alpha).squeeze(-1)
                        _v11_inv_v12 = _torch.cholesky_solve(_v12, _L)
                        _cov_diag = (_v22 - _torch.matmul(_v12.T, _v11_inv_v12)).diagonal()

                        _means[_comp] = _mu.cpu().numpy()
                        _sigmas[_comp] = _torch.sqrt(_torch.clamp(_cov_diag, min=0.0)).cpu().numpy()

                        del (
                            _variance_gpu,
                            _lengthscale_gpu,
                            _grid_scaled,
                            _query_scaled,
                            _L,
                            _w_hat_gpu,
                            _alpha,
                            _v12,
                            _v22,
                            _mu,
                            _v11_inv_v12,
                            _cov_diag,
                        )
                        _torch.cuda.empty_cache()

                    del _grid_gpu, _query_gpu
                    _torch.cuda.empty_cache()
            except Exception as _diag_error:
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                return {}, {}, str(_diag_error)

            return _means, _sigmas, None

        _selected_components = list(range(_comp_start, _comp_end + 1))
        _component_means, _component_sigmas, _diagnostics_error = _predict_selected_components_streaming(_selected_components)
        if _diagnostics_error is not None:
            try:
                import gc as _gc
                import torch as _torch
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

        from Speculate_addons.grid_registry import infer_grid_name as _infer_grid_name, log_param_ids as _log_param_ids, parameter_label_map as _parameter_label_map

        # Parameter description for x-axis label
        _grid_name = _infer_grid_name(getattr(_emu, 'name', None))
        _desc_map = dict(_parameter_label_map(_grid_name))
        _param_name = _emu.param_names[_not_fixed_col]
        _p_idx = int(_param_name.replace("param", ""))
        _xlabel = _desc_map.get(_p_idx, _param_name)
        _logged_params = _log_param_ids(_grid_name)
        _xlabel_axis = f'log\u2081\u2080({_xlabel})' if _p_idx in _logged_params else _xlabel

        # Build Altair charts for selected components
        _charts = []
        for _comp in _selected_components:
            if _diagnostics_error is not None:
                break

            # Extract the variance attributed to the PCA truncation error
            _var = _dots_inv_diag[_comp] / _emu.lambda_xi
            _err_bar = float(np.sqrt(_var))

            # Scatter: actual weights at grid points (with error bar bounds)
            _scatter_df = pd.DataFrame({
                'x': _param_x,
                'weight': _weights[_comp],
                'err_upper': _weights[_comp] + _err_bar,
                'err_lower': _weights[_comp] - _err_bar
            })

            # GP prediction line + uncertainty band
            _gp_df = pd.DataFrame({
                'x': _param_x_test,
                'mean': _component_means[_comp],
                'upper': _component_means[_comp] + 2 * _component_sigmas[_comp],
                'lower': _component_means[_comp] - 2 * _component_sigmas[_comp],
            })

            _scatter = alt.Chart(_scatter_df).mark_circle(size=60, color='#3b82f6').encode(
                x=alt.X('x:Q', title=_xlabel_axis, scale=alt.Scale(zero=False)),
                y=alt.Y('weight:Q', title=f'Weight {_comp}', scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip('x:Q', title=_xlabel, format='.4f'),
                         alt.Tooltip('weight:Q', title='Weight', format='.4e')]
            )

            # Error bars (vertical lines)
            _error_bars = alt.Chart(_scatter_df).mark_rule(color='#3b82f6', strokeWidth=1.5).encode(
                x='x:Q',
                y='err_lower:Q',
                y2='err_upper:Q'
            )

            _gp_line = alt.Chart(_gp_df).mark_line(color='#f97316').encode(
                x='x:Q',
                y='mean:Q'
            )

            _gp_band = alt.Chart(_gp_df).mark_area(opacity=0.3, color='#f97316').encode(
                x='x:Q',
                y='lower:Q',
                y2='upper:Q'
            )

            # Layer components: Band -> Line -> Error Bars -> Points
            _chart = (_gp_band + _gp_line + _error_bars + _scatter).properties(
                title=f'PCA Component {_comp}',
                width=650,
                height=160
            )
            _charts.append(_chart)

        if _diagnostics_error is not None:
            _combined = mo.callout(
                mo.md(f"{mo.icon('lucide:triangle-alert')} **GP diagnostics could not be computed:** {_diagnostics_error}"),
                kind="warn",
            )
        elif _charts:
            _combined = alt.vconcat(*_charts).resolve_scale(x='shared')
        else:
            _combined = alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_point()

        # Fixed parameters info string
        _fixed_info_parts = []
        for _i in range(_n_params):
            if _i != _not_fixed_col:
                _vals = _unique_per_dim[_i]
                _idx = round(_fixed_idx * (len(_vals) - 1) / _slider_max) if _slider_max > 0 else 0
                _pn = _emu.param_names[_i]
                _pi = int(_pn.replace("param", ""))
                _pdesc = _desc_map.get(_pi, _pn)
                _fixed_info_parts.append(f"{_pdesc}={_vals[_idx]:.4g}")
        _fixed_info = ", ".join(_fixed_info_parts)

        gp_figure = mo.accordion({
            "\U0001f52c GP Weight Diagnostics": mo.vstack([
                mo.md("Explore the Gaussian Process fit to the PCA weight latent space. "
                       "Blue dots are the actual PCA weights at grid points. Vertical bars represent the noise ($\\sigma_k$) assigned to the weights by the $\\lambda_\\xi$ truncation penalty. "
                       "Orange line and shaded region show the GP mean \u00b1 2\u03c3."),
                mo.hstack([gp_xaxis_selector, gp_fixed_slider], justify="start", gap=1),
                mo.hstack([gp_comp_start, gp_comp_end], justify="start", gap=1),
                mo.md(f"**Fixed parameters:** {_fixed_info}"),
                _combined
            ])
        })

    return (gp_figure,)


@app.cell
def _(alt, get_loss_history, get_trained_emu, gp_figure, mo, pd):
    # Reactive plot cell
    loss_data = get_loss_history()
    _emu = get_trained_emu()

    # Create DataFrame (handles empty case)
    if len(loss_data) > 0:
        df = pd.DataFrame({
            'Iteration': range(len(loss_data)),
            'Loss': loss_data
        })
        current_iter = len(loss_data)
    else:
        df = pd.DataFrame({'Iteration': [], 'Loss': []})
        current_iter = 0

    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X('Iteration:Q', axis=alt.Axis(title='Iteration')),
        y=alt.Y('Loss:Q', axis=alt.Axis(title='Negative Log-Likelihood'), scale=alt.Scale(zero=False)),
        tooltip=['Iteration', 'Loss']
    )

    # Line chart
    line = base.mark_line(color='#3b82f6').properties(
        title=f'Emulator Training Progress (Iter {current_iter})',
        width=600,
        height=300
    )

    # Minimum loss rule (only if data exists)
    if len(loss_data) > 0:
        min_val = min(loss_data)
        min_rule = alt.Chart(pd.DataFrame({'y': [min_val]})).mark_rule(
            color='green', strokeDash=[5,5], size=2
        ).encode(y='y')
        chart = (line + min_rule)
    else:
        chart = line

    # ── Emulator summary (mirrors the training-log __repr__ output) ──
    # Prefer the materialized-V11 total objective when available.  For very
    # large models without V11, fall back to the recorded optimizer values.
    _emu_summary = mo.md("")
    if _emu is not None:
        _s = "Emulator\n" + "-" * 40 + "\n"
        if _emu.name:
            _s += f"Name:           {_emu.name}\n"
        _s += f"Trained:        {_emu._trained}\n"
        _s += f"lambda_xi:      {_emu.lambda_xi:.4f}\n"
        _s += f"Kernel:         {getattr(_emu, 'kernel', 'rbf')}\n"
        _s += f"PCA Components: {_emu.ncomps}\n"
        _s += f"Grid Points:    {_emu.grid_points.shape[0]}\n"
        from Speculate_addons.grid_registry import infer_grid_name as _infer_grid_name, parameter_label_map as _parameter_label_map

        _grid_name = _infer_grid_name(getattr(_emu, 'name', None))
        _desc_map = dict(_parameter_label_map(_grid_name))
        _raw_pnames = list(_emu.param_names) if hasattr(_emu, 'param_names') else []
        _pnames = []
        for _pn in _raw_pnames:
            _pidx = int(_pn.replace("param", "")) if _pn.startswith("param") else None
            _pnames.append(_desc_map.get(_pidx, _pn) if _pidx is not None else _pn)
        if _pnames:
            _s += f"Parameters:     {', '.join(_pnames)}\n"
        _s += "\nVariances:\n"
        for _i, _v in enumerate(_emu.variances):
            _s += f"  Comp {_i:2d}:  {_v:.4f}\n"
        _s += "\nLengthscales:\n"
        if _pnames:
            _hdr = "          " + "  ".join(f"{p:>8s}" for p in _pnames)
            _s += _hdr + "\n"
        for _i, _ls in enumerate(_emu.lengthscales):
            _s += f"  Comp {_i:2d}: [ " + "  ".join(f"{l:.4f}" for l in _ls) + " ]\n"
        _final_log_likelihood = None
        if getattr(_emu, '_v11', None) is not None or getattr(_emu, '_v11_gpu', None) is not None:
            try:
                # Per-component loss_history stores component-local optimizer
                # evaluations.  A materialized V11 is the authoritative total.
                _final_log_likelihood = _emu.log_likelihood()
            except Exception:
                _final_log_likelihood = None
        if _final_log_likelihood is not None:
            _final_nll = -_final_log_likelihood
            _s += f"\nFinal Total NLL: {_final_nll:.2f}\n"
            _s += f"Final Log Likelihood: {_final_log_likelihood:.2f}\n"
        elif len(loss_data) > 0:
            _best_nll = min(loss_data)
            _s += f"\nBest Recorded Loss: {_best_nll:.2f}\n"
            _s += f"Recorded Loss Sign-Flipped: {-_best_nll:.2f}\n"
        if len(loss_data) > 0:
            _s += f"Iterations:     {len(loss_data)}\n"
        _emu_summary = mo.accordion(
            {f"{mo.icon('lucide:cpu')} Emulator Summary": mo.md(f"```\n{_s}```")}
        )

    # Always display the chart container with GP diagnostics below
    mo.vstack([
        mo.md(f"### {mo.icon('lucide:trending-up')} Trained Emulator Loss Curve"),
        mo.ui.altair_chart(chart),
        _emu_summary,
        gp_figure
    ])
    return


if __name__ == "__main__":
    app.run()
