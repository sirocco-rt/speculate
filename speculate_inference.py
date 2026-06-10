# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
#
# Speculate Inference Tool
# ========================
# Interactive notebook for fitting spectral models to observations or
# validation test-grid spectra.  The workflow proceeds through four stages:
#
#   Stage 1 — Setup & Data Loading  (grid, emulator, observation/test run)
#   Stage 2 — Parameter Configuration  (priors, fixed/free toggles)
#   Stage 3 — MLE  (Nelder-Mead simplex optimisation over negative log-likelihood)
#   Stage 4 — MCMC  (emcee ensemble sampler for posterior exploration)
#   Export  — Write .pf templates and posterior CSV files
#
# Depends on the Starfish spectral emulator library and the Speculate
# benchmark add-on for export helpers.

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Inference Tool")


@app.cell
def _():
    # ── Imports & global configuration ──
    # Starfish provides the core emulation primitives:
    #   Emulator  – PCA + GP spectral emulator trained on a Sirocco grid
    #   Spectrum  – lightweight wavelength/flux/sigma container
    #   SpectrumModel – wraps an Emulator + Spectrum for likelihood evaluation
    # scipy.stats supplies the frozen prior distributions used in Stages 2-4.
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    import os
    import re
    import numpy as np
    import pandas as pd
    import altair as alt
    import importlib
    from Starfish.emulator import Emulator
    from Starfish.spectrum import Spectrum
    import Starfish.models.spectrum_model as _spectrum_model_module
    _spectrum_model_module = importlib.reload(_spectrum_model_module)
    SpectrumModel = _spectrum_model_module.SpectrumModel
    import pathlib
    import scipy.stats as stats
    from Speculate_addons import Spec_functions as spec_functions
    from Speculate_addons.gp_covariance import (
        GP_LOG_AMP_PRIOR_SIGMA,
        bounds_for_frozen_prior,
        estimate_log_amp_centre_from_sigma,
        get_frozen_dist_loc_scale,
        global_covariance_diagnostics,
    )

    # Marimo can keep an older helper module cached across notebook reloads.
    # If a newly added helper is missing, reload the module before binding names.
    if (
        not hasattr(spec_functions, "build_default_observation_sigma")
        or not hasattr(spec_functions, "build_bestfit_spectrum_altair")
        or not hasattr(spec_functions, "enable_speculate_altair_theme")
        or "model_label" not in spec_functions.build_bestfit_spectrum_altair.__code__.co_varnames
    ):
        spec_functions = importlib.reload(spec_functions)

    spec_functions.enable_speculate_altair_theme(alt)
    build_bestfit_spectrum_altair = spec_functions.build_bestfit_spectrum_altair
    build_default_observation_sigma = spec_functions.build_default_observation_sigma
    build_synthetic_sirocco_sigma = spec_functions.build_synthetic_sirocco_sigma
    fit_power_law_continuum = spec_functions.fit_power_law_continuum
    logo_path = "assets/logos/Speculate_logo2.png"


    # Left column: Title and Description
    title_col = mo.vstack([mo.md(
        """
        # Inference Tool
        """), mo.md(
        """
        Compare observations with emulated spectral models.
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
    return (
        Emulator,
        GP_LOG_AMP_PRIOR_SIGMA,
        Spectrum,
        SpectrumModel,
        alt,
        bounds_for_frozen_prior,
        build_bestfit_spectrum_altair,
        build_default_observation_sigma,
        build_synthetic_sirocco_sigma,
        estimate_log_amp_centre_from_sigma,
        fit_power_law_continuum,
        get_frozen_dist_loc_scale,
        global_covariance_diagnostics,
        mo,
        np,
        os,
        pd,
        re,
        stats,
    )


@app.cell
def _(mo):
    # The selected emulator filename is cheap UI state; the loaded Emulator
    # object can materialise V11/Cholesky caches on the GPU, so it lives behind
    # an explicit Load button instead of being rebuilt on every dropdown change.
    get_loaded_emu, set_loaded_emu = mo.state(None)
    get_loaded_emu_filename, set_loaded_emu_filename = mo.state(None)
    get_selected_grid, set_selected_grid = mo.state(None)
    get_selected_emu_filename, set_selected_emu_filename = mo.state(None)
    return (
        get_loaded_emu,
        get_loaded_emu_filename,
        get_selected_emu_filename,
        get_selected_grid,
        set_loaded_emu,
        set_loaded_emu_filename,
        set_selected_emu_filename,
        set_selected_grid,
    )


@app.cell
def _(
    get_loaded_emu,
    get_loaded_emu_filename,
    set_loaded_emu,
    set_loaded_emu_filename,
    set_selected_emu_filename,
    set_selected_grid,
):
    def clear_loaded_emu_cache(_value=None, *, force=False):
        """Drop the loaded Emulator and ask PyTorch to release cached GPU memory.

        The Emulator may attach large GPU tensors lazily during load and first
        inference, including V11, iPhiPhi, Cholesky factors, and block-wise
        memory-efficient caches.  This helper is used as a widget on_change
        callback and before explicit loads, so it accepts and ignores the UI
        callback value.
        """
        _emu = get_loaded_emu()
        _had_emu = _emu is not None or get_loaded_emu_filename() is not None

        if _emu is not None:
            # Ignore attributes that are not present on this Emulator version.
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

        if _emu is not None:
            set_loaded_emu(None)
        if get_loaded_emu_filename() is not None:
            set_loaded_emu_filename(None)

        if force or _had_emu:
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

    def on_grid_select(value):
        """Persist the selected grid while clearing any stale loaded emulator."""
        set_selected_grid(value)
        set_selected_emu_filename(None)
        clear_loaded_emu_cache()

    def on_emulator_select(value):
        """Persist the selected emulator filename while clearing stale caches."""
        set_selected_emu_filename(value)
        clear_loaded_emu_cache()

    return clear_loaded_emu_cache, on_emulator_select, on_grid_select


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
            mo.md(f"###{mo.icon('lucide:test-tubes')} Benchmark Suite"),
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
            mo.md(f"## {mo.icon('lucide:cpu')} No NVIDIA GPU Detected\n*Performance will be slower on CPU.*"), 
            kind="warn"
        )

    # status_widget
    return (status_widget,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 1: Setup & Data Loading
    """)
    return


@app.cell
def _():
    from Speculate_addons.grid_registry import param_map_db as _registry_param_map_db

    def inf_param_map_db_for_grid(grid_name=None):
        """Return Inference Tool parameter labels/ranges from the registry."""
        return _registry_param_map_db(grid_name)

    param_map_db = inf_param_map_db_for_grid()
    return inf_param_map_db_for_grid, param_map_db


@app.cell
def _(get_loaded_emu_filename, get_selected_grid, mo, on_grid_select, os):
    # --- Grid Selection ---
    # Discover grid families from local raw grids, processed grid files, and GP
    # emulator files.  A grid can be present before a trained emulator exists;
    # in that case the downstream emulator dropdown will simply show no matching
    # GP emulators for that grid.
    from Speculate_addons.grid_registry import GRID_REGISTRY as _GRID_REGISTRY, infer_grid_name as _infer_grid_name

    _emu_dir = "Grid-Emulator_Files"
    _sirocco_grids_dir = "sirocco_grids"
    _unique_grids = set()

    if os.path.exists(_sirocco_grids_dir):
        _local_grid_dirs = {
            _name
            for _name in os.listdir(_sirocco_grids_dir)
            if os.path.isdir(os.path.join(_sirocco_grids_dir, _name))
        }
        for _grid_name, _config in _GRID_REGISTRY.items():
            if _grid_name in _local_grid_dirs or _config.get("test_grid_name") in _local_grid_dirs:
                _unique_grids.add(_grid_name)

    if os.path.exists(_emu_dir):
        _files = [f for f in os.listdir(_emu_dir) if f.endswith(".npz")]
        for _f in _files:
            _grid_name = _infer_grid_name(_f)
            if _grid_name is not None and ("_emu_" in _f or "_grid_" in _f):
                _unique_grids.add(_grid_name)
                continue

            # Legacy fallback for emulator filenames from unregistered grids.
            _parts = _f.split("_emu_")
            if len(_parts) > 1:
                _unique_grids.add(_parts[0])

    _sorted_grids = sorted(list(_unique_grids))

    if _sorted_grids:
        _preferred_grid = "speculate_cv_no-bl_grid_v87f"
        _initial_grid = _preferred_grid if _preferred_grid in _sorted_grids else _sorted_grids[0]
        _selected_grid = get_selected_grid()
        _loaded_filename = get_loaded_emu_filename()
        if _selected_grid in _sorted_grids:
            _initial_grid = _selected_grid
        elif _loaded_filename and "_emu_" in _loaded_filename:
            _loaded_grid = _loaded_filename.split("_emu_")[0]
            if _loaded_grid in _sorted_grids:
                _initial_grid = _loaded_grid

        grid_selector = mo.ui.dropdown(
            options=_sorted_grids,
            value=_initial_grid,
            label="Select Grid Dataset:",
            full_width=True,
            on_change=on_grid_select,
        )
    else:
        grid_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Grids Found",
            full_width=True,
            on_change=on_grid_select,
        )
    return (grid_selector,)


@app.cell
def _(get_loaded_emu_filename, get_selected_emu_filename, grid_selector, inf_param_map_db_for_grid, mo, on_emulator_select, os, re):
    # --- Emulator Selection (Dependent on Grid) ---
    # Filter the .npz files to those matching the selected grid stem and
    # discover which parameter indices this grid supports.  The registry gives
    # the complete supported set for the selected grid, while filenames provide
    # the specific parameter tag for saved emulators/grids.

    _emu_dir = "Grid-Emulator_Files"
    _filtered_emus = []
    from Speculate_addons.grid_registry import get_grid_config as _get_grid_config, parse_param_tag as _parse_param_tag

    # Grid Parameter Detection.  Prefer registry metadata so AGN and CV grids do
    # not need separate filename heuristics for boundary-layer or inclination axes.
    grid_indices = set()
    _param_map_db = inf_param_map_db_for_grid(grid_selector.value)
    _grid_config = _get_grid_config(grid_selector.value)
    if _grid_config is not None:
        grid_indices.update(_grid_config.get("max_params", []))

    if grid_selector.value and os.path.exists(_emu_dir):
        _files = [f for f in os.listdir(_emu_dir) if f.endswith(".npz")]

        for _f in _files:
            if _f.startswith(grid_selector.value):
                # Check if it is an emulator
                if "_emu_" in _f:
                    _filtered_emus.append(_f)

                # Check for parameters based on the tag after _emu_ or _grid_.
                # parse_param_tag() uses the increasing-ID convention to
                # disambiguate concatenated IDs such as 12391011.
                _m = re.search(r"(_emu_|_grid_)([^_]+)", _f)
                if _m:
                    try:
                        grid_indices.update(_parse_param_tag(_m.group(2)))
                    except ValueError:
                        pass

    _filtered_emus = sorted(_filtered_emus)

    if _filtered_emus:
        _initial_emu = _filtered_emus[0]
        _selected_emu = get_selected_emu_filename()
        _loaded_emu = get_loaded_emu_filename()
        if _selected_emu in _filtered_emus:
            _initial_emu = _selected_emu
        elif _loaded_emu in _filtered_emus:
            _initial_emu = _loaded_emu

        emulator_selector = mo.ui.dropdown(
            options=_filtered_emus,
            value=_initial_emu,
            label="Select Trained Emulator:",
            full_width=True,
            on_change=on_emulator_select,
        )
    else:
        emulator_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Emulators for this Grid",
            full_width=True,
            on_change=on_emulator_select,
        )

    emulator_load_button = mo.ui.run_button(
        label=f"{mo.icon('lucide:download')} Load Emulator",
        kind="neutral",
    )

    # Valid Parameters Accordion
    _param_lookup_info = []
    if grid_indices:
        _sorted_ind = sorted(list(grid_indices))
        for _idx in _sorted_ind:
            if _idx in _param_map_db:
                _p_name = _param_map_db[_idx][0]
                _param_lookup_info.append(f"**{_idx}** -> `{_p_name}`")
            else:
                _param_lookup_info.append(f"**{_idx}** -> `Unknown`")
    else:
        _param_lookup_info.append("No parameters detected from filenames.")

    param_info_accordion = mo.accordion({
        f"{mo.icon('lucide:key-round')} Parameter Key": mo.md("\n".join([f"- {s}" for s in _param_lookup_info]))
    })
    return emulator_load_button, emulator_selector, grid_indices, param_info_accordion


@app.cell
def _(mo):
    # UI widgets for the data-source half of Stage 1.  The user switches
    # between two modes:
    #   "Observation File"   — user-uploaded CSV with Wavelength/Flux columns
    #   "Test Grid"          — extracted Sirocco .spec files from a paired
    #                          validation grid, for ground-truth comparison.
    get_obs_refresh, set_obs_refresh = mo.state(0)

    # Source Type Selector: Observation vs Test Grid
    data_source_selector = mo.ui.dropdown(
        options=["Observation File", "Test Grid (Validation)"],
        value="Observation File",
        label="Data Source:",
        full_width=True
    )

    # File upload for observational spectra
    obs_file_uploader = mo.ui.file(
        kind="area",
        label="**Drag and drop a new observational spectra (CSV)**",
        multiple=False
    )

    file_upload_ui = mo.vstack([
         obs_file_uploader,
         mo.md("<center><small>Format: CSV with headers `WAVELENGTH`, `FLUX`, and optionally `ERROR`</small></center>")
    ])
    return (
        data_source_selector,
        file_upload_ui,
        get_obs_refresh,
        obs_file_uploader,
        set_obs_refresh,
    )


@app.cell
def _(mo, obs_file_uploader, os, set_obs_refresh):
    # Persist an uploaded observation into the shared observation directory and
    # bump the refresh state so downstream file-picker cells rerun.
    if obs_file_uploader.value:
        # Create directory if it doesn't exist
        _obs_dir = "observation_files"
        os.makedirs(_obs_dir, exist_ok=True)

        # Save file (taking the first one since multiple=False)
        uploaded_file = obs_file_uploader.value[0]
        file_path = os.path.join(_obs_dir, uploaded_file.name)

        try:
            with open(file_path, "wb") as _f:
                _f.write(uploaded_file.contents)

            # Force refresh of the file list
            set_obs_refresh(lambda v: v + 1)

            mo.status.toast(f"{mo.icon('lucide:check-circle')} Uploaded {uploaded_file.name}")
        except Exception as e:
             mo.status.toast(f"{mo.icon('lucide:x-circle')} Upload failed: {str(e)}")
    return


@app.cell(hide_code=True)
def _(get_obs_refresh, grid_selector, obs_file_uploader, os):
    # Build the selectable observation-file list and, when possible, the paired
    # test-grid metadata derived from the currently selected emulator grid.
    from Speculate_addons.grid_registry import get_grid_config as _get_grid_config

    # Trigger refresh on upload or delete
    _ = obs_file_uploader.value
    _ = get_obs_refresh()

    _obs_dir = "observation_files"  # Underscore to make local to this cell
    obs_files = []

    if os.path.exists(_obs_dir):
        # Filter for likely data files
        obs_files = sorted([f for f in os.listdir(_obs_dir) if f.endswith(('.csv', '.txt', '.dat'))])

    if not obs_files:
        obs_files = [] # Ensure it's empty list if no files found

    # Test Grid Files (based on selected grid)
    test_grid_files = []
    test_grid_path = None
    test_grid_params_df = None

    if grid_selector.value:
        # The registry records the paired validation grid where available.  The
        # string replacement is kept only as a backward-compatible fallback for
        # older emulator names that predate the registry.
        _grid_name = grid_selector.value
        _grid_config = _get_grid_config(_grid_name)
        _test_grid_name = _grid_config.get("test_grid_name") if _grid_config else _grid_name.replace("_grid_", "_testgrid_")
        test_grid_path = f"sirocco_grids/{_test_grid_name}"

        if os.path.exists(test_grid_path):
            # Discover available validation spectra in run-number order so the UI
            # presents the same ordering as the lookup table.
            test_grid_files = sorted(
                [f for f in os.listdir(test_grid_path) if f.endswith('.spec')],
                key=lambda x: int(x.replace('run', '').replace('.spec', '')) if x.replace('run', '').replace('.spec', '').isdigit() else 0
            )
            # Load the optional lookup table that maps each run back to its known
            # physical parameters for later validation against the inference result.
            _lookup_path = os.path.join(test_grid_path, "grid_run_lookup_table.parquet")
            if os.path.exists(_lookup_path):
                import pandas as _pd
                test_grid_params_df = _pd.read_parquet(_lookup_path)
    return obs_files, test_grid_files, test_grid_params_df, test_grid_path


@app.cell
def _(grid_selector, mo, obs_files, os, set_obs_refresh, test_grid_files):
    from Speculate_addons.grid_registry import default_fixed_inclination as _default_fixed_inclination, inclination_values as _get_inclination_values

    # Dropdown for selecting observation file
    if obs_files:
         obs_file_selector = mo.ui.dropdown(
            options=obs_files,
            value=obs_files[0],
            label="Select Observational Spectrum:",
            full_width=True
        )
    else:
         obs_file_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Observations Found",
            full_width=True
        )

    # Dropdown for selecting test grid run
    if test_grid_files:
        test_run_selector = mo.ui.dropdown(
            options=test_grid_files,
            value=test_grid_files[0],
            label="Select Test Simulation:",
            full_width=True
        )
    else:
        test_run_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Test Grid Available",
            full_width=True
        )

    # The delete action only applies to uploaded observation files; test-grid
    # spectra remain immutable because they are part of the grid dataset.
    def delete_selected_file():
        if obs_file_selector.value:
            try:
                _obs_dir = "observation_files"
                file_path = os.path.join(_obs_dir, obs_file_selector.value)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    set_obs_refresh(lambda v: v + 1)
                    mo.status.toast(f"{mo.icon('lucide:trash-2')} Deleted {obs_file_selector.value}")
            except Exception as e:
                mo.status.toast(f"Error deleting file: {e}")

    delete_obs_btn = mo.ui.button(
        label=f"{mo.icon('lucide:trash-2')}",
        kind="danger",
        tooltip="Delete selected file",
        on_click=lambda _: delete_selected_file()
    )

    # Inclination selector for test grids.  CV and AGN .spec files expose
    # different angle sets, so both the options and default come from the registry.
    _inclination_values = _get_inclination_values(grid_selector.value if grid_selector is not None else None)
    _default_inclination = _default_fixed_inclination(grid_selector.value if grid_selector is not None else None)
    test_inclination_selector = mo.ui.dropdown(
        options=[str(value) for value in _inclination_values],
        value=str(_default_inclination),
        label="Inclination (°):",
        full_width=True
    )
    return (
        delete_obs_btn,
        obs_file_selector,
        test_inclination_selector,
        test_run_selector,
    )


@app.cell
def _(
    Emulator,
    clear_loaded_emu_cache,
    emulator_load_button,
    emulator_selector,
    get_loaded_emu,
    get_loaded_emu_filename,
    mo,
    set_loaded_emu,
    set_loaded_emu_filename,
    set_selected_emu_filename,
    set_selected_grid,
):
    # The dropdown only selects a file.  The expensive Emulator.load call is
    # deliberately gated behind this run button because load can allocate V11
    # and Cholesky caches large enough to exhaust GPU/CPU memory.  Note that
    # the heaviest GPU caches (Cholesky blocks, V11) are populated lazily on
    # the first inference call, not at load time, so a successful Load is
    # itself a relatively light memory event.
    emu = get_loaded_emu()
    _loaded_name = get_loaded_emu_filename()
    _selected_name = emulator_selector.value
    emulator_load_status = mo.md("")

    def _format_cache_size(_gb):
        return f"{_gb * 1024:.0f} MB" if _gb < 1.0 else f"{_gb:.1f} GB"

    def _selected_emulator_cache_note(_filename):
        if not _filename:
            return "", "neutral"

        try:
            import numpy as _np
            import os as _os

            _path = _os.path.join("Grid-Emulator_Files", _filename)
            if not _os.path.isfile(_path):
                return "", "neutral"

            with _np.load(_path, allow_pickle=True) as _npz:
                _n_grid = int(_npz["grid_points"].shape[0])
                if "eigenspectra" in _npz:
                    _n_components = int(_npz["eigenspectra"].shape[0])
                elif "weights" in _npz:
                    _n_components = int(_npz["weights"].shape[1])
                else:
                    return "", "neutral"

            _block_bytes = _n_grid * _n_grid * 8
            _cache_bytes = _n_components * _block_bytes
            _workspace_bytes = 2 * _block_bytes
            _required_gb = (_cache_bytes + _workspace_bytes) * 1.1 / (1024**3)
            _display = _format_cache_size(_required_gb)

            _note = (
                f"Inference cache estimate: **~{_display} required** to build the "
                f"reusable MLE/MCMC cache ({_n_components} PCA components × "
                f"{_n_grid:,} grid points)."
            )
            _kind = "neutral"

            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _free_bytes, _ = _torch.cuda.mem_get_info(0)
                    _free_gb = _free_bytes / (1024**3)
                    _usage_pct = (_required_gb / _free_gb) * 100 if _free_gb > 0 else float("inf")
                    _can_cache = _required_gb < _free_gb * 0.9
                    if _can_cache:
                        _note += f" Current free VRAM: **{_free_gb:.1f} GB** ({_usage_pct:.0f}% required)."
                    else:
                        _note += (
                            f" Current free VRAM: **{_free_gb:.1f} GB** ({_usage_pct:.0f}% required). "
                            "If this cache cannot fit, Speculate will stream per component and MLE/MCMC will be much slower."
                        )
                        _kind = "warn"
                else:
                    _note += (
                        " No NVIDIA GPU detected. The reusable GPU inference cache "
                        "cannot be built, so inference will use CPU/fallback paths and may be slower."
                    )
                    _kind = "warn"
            except Exception:
                _note += " GPU status could not be checked, so cache fit is unknown."
                _kind = "warn"

            return _note, _kind
        except Exception as _exc:
            return f"Could not estimate inference cache memory for this emulator: {_exc}", "warn"

    if emu is not None and _loaded_name != _selected_name:
        clear_loaded_emu_cache()
        emu = None
        _loaded_name = None

    if emulator_load_button.value:
        if not _selected_name:
            emulator_load_status = mo.callout(
                mo.md("Select an emulator before loading."),
                kind="warn",
            )
        else:
            clear_loaded_emu_cache(force=True)
            try:
                with mo.status.spinner(title=f"Loading emulator {_selected_name}..."):
                    emu_path = f"Grid-Emulator_Files/{_selected_name}"
                    emu = Emulator.load(emu_path)
                set_loaded_emu(emu)
                set_loaded_emu_filename(_selected_name)
                set_selected_emu_filename(_selected_name)
                if "_emu_" in _selected_name:
                    set_selected_grid(_selected_name.split("_emu_")[0])
                emulator_load_status = mo.callout(
                    mo.md(f"{mo.icon('lucide:check-circle')} Loaded `{_selected_name}`."),
                    kind="success",
                )
            except Exception as e:
                emu = None
                clear_loaded_emu_cache(force=True)
                emulator_load_status = mo.callout(
                    mo.md(f"{mo.icon('lucide:x-circle')} Error loading `{_selected_name}`: {e}"),
                    kind="danger",
                )
    elif emu is not None and _loaded_name == _selected_name:
        emulator_load_status = mo.callout(
            mo.md(f"{mo.icon('lucide:check-circle')} Loaded `{_selected_name}`."),
            kind="success",
        )
    elif _selected_name and emu is None:
        _cache_note, _cache_kind = _selected_emulator_cache_note(_selected_name)
        _selected_message = f"Selected `{_selected_name}`. Click **Load Emulator** to enable inference."
        if _cache_note:
            _selected_message += f"\n\n{_cache_note}"
        emulator_load_status = mo.callout(
            mo.md(_selected_message),
            kind=_cache_kind,
        )

    return emu, emulator_load_status


@app.cell
def _(
    build_synthetic_sirocco_sigma,
    data_source_selector,
    emu,
    emulator_selector,
    grid_selector,
    mo,
    np,
    obs_file_selector,
    os,
    pd,
    re,
    test_grid_params_df,
    test_grid_path,
    test_inclination_selector,
    test_run_selector,
):
    # Load observation data (from file OR test grid)
    from Speculate_addons.grid_registry import inclination_column as _inclination_column, lookup_row_to_emulator_values as _lookup_row_to_emulator_values

    obs_data = None
    ground_truth_params = None  # For test grid validation
    _obs_dir = "observation_files"
    data_source_info = ""

    is_test_grid = "Test Grid" in data_source_selector.value

    if is_test_grid and test_run_selector.value and test_grid_path:
        # Validation mode reads a raw Sirocco .spec file and converts the chosen
        # inclination column into the same dataframe shape used for observations.
        try:
            spec_path = os.path.join(test_grid_path, test_run_selector.value)

            # Sirocco .spec files contain a variable-length text header, so scan
            # for the first data row before passing the file to numpy.
            skiprows = 0
            with open(spec_path, 'r') as _f:
                for _i, line in enumerate(_f):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('Freq.'):
                        continue
                    skiprows = _i
                    break

            # After frequency and wavelength, the remaining columns are fluxes at
            # fixed viewing angles.  Use the registry rather than CV-only column
            # arithmetic because AGN has a different angle grid.
            inc_val = int(test_inclination_selector.value)
            inc_col_idx = _inclination_column(grid_selector.value if grid_selector is not None else None, inc_val)

            # Flip both arrays so wavelength increases left-to-right in the UI.
            data_raw = np.loadtxt(spec_path, skiprows=skiprows, unpack=True)
            wavelengths = np.flip(data_raw[1])  # Lambda column, flip to ascending
            fluxes = np.flip(data_raw[inc_col_idx])  # Selected inclination flux

            # Treat synthetic validation data like observations by attaching a
            # continuum-anchored fallback uncertainty column in native units.
            sigma, continuum = build_synthetic_sirocco_sigma(wavelengths, fluxes)
            obs_data = pd.DataFrame({
                'wavelength': wavelengths,
                'flux': fluxes,
                'error': sigma,
                'continuum': continuum,
                'sigma_source': 'synthetic_sirocco_continuum',
            })

            # Translate lookup-table values into the active emulator
            # parameterisation.  The registry handles the CV ratio/log transforms
            # and the AGN Eddington-fraction/R_g-scaled transforms.
            if test_grid_params_df is not None:
                run_num = int(test_run_selector.value.replace('run', '').replace('.spec', ''))
                gt_row = test_grid_params_df[test_grid_params_df['Run Number'] == run_num]
                if len(gt_row) > 0:
                    ground_truth_params = _lookup_row_to_emulator_values(
                        grid_selector.value if grid_selector is not None else None,
                        gt_row.iloc[0],
                        inc_val,
                    )

            data_source_info = f"Test Grid: {test_run_selector.value} @ {inc_val}°"

        except Exception as e:
            mo.output.replace(mo.callout(mo.md(f"{mo.icon('lucide:x-circle')} Error loading test grid: {e}"), kind="danger"))

    elif obs_file_selector.value:
        # Observation mode accepts user-uploaded tabular data as long as it has
        # wavelength and flux columns after column-name normalization.
        try:
            path = os.path.join(_obs_dir, obs_file_selector.value)
            # Try flexible loading
            df = pd.read_csv(path)

            # Normalize column names
            df.columns = df.columns.astype(str).str.lower()

            # Check for required columns
            if 'wavelength' in df.columns and 'flux' in df.columns:
                # Sort by wavelength ascending (handles decreasing-wavelength files)
                df = df.sort_values('wavelength', ascending=True).reset_index(drop=True)
                obs_data = df
                data_source_info = f"Observation: {obs_file_selector.value}"
            else:
                 mo.output.replace(mo.callout(mo.md(f"{mo.icon('lucide:x-circle')} Invalid file format. Metadata columns 'Wavelength' and 'Flux' not found."), kind="danger"))

        except Exception as e:
             mo.output.replace(mo.callout(mo.md(f"{mo.icon('lucide:x-circle')} Error reading file: {e}"), kind="danger"))

    # Infer the most likely flux transform from the emulator filename so the UI
    # starts on the training scale, while still letting the user override it.
    _detected_scale = "linear"  # default
    if emulator_selector.value:
        _emu_name = emulator_selector.value.lower()
        if '_log_' in _emu_name:
            _detected_scale = "log"
        elif '_continuum-normalised_' in _emu_name:
            _detected_scale = "continuum-normalised"

    obs_flux_scale = mo.ui.dropdown(
        options=["linear", "log", "continuum-normalised"],
        value=_detected_scale,
        label="Observation Flux Transform:",
        full_width=True,
    )

    # --- Wavelength range slider bounded by emulator ---
    if emu is not None:
        _emu_min = float(emu.wl.min())
        _emu_max = float(emu.wl.max())
        # Inset by 10 Å for edge effects
        _slider_min = _emu_min + 10
        _slider_max = _emu_max - 10
    elif obs_data is not None:
        _slider_min = float(obs_data['wavelength'].min())
        _slider_max = float(obs_data['wavelength'].max())
    else:
        _slider_min = 800.0
        _slider_max = 8000.0

    wl_range_slider = mo.ui.range_slider(
        start=_slider_min,
        stop=_slider_max,
        value=[_slider_min, _slider_max],
        step=1.0,
        label="Wavelength Range (Å):",
        show_value=True,
        full_width=True,
    )
    return (
        data_source_info,
        ground_truth_params,
        obs_data,
        obs_flux_scale,
        wl_range_slider,
    )


@app.cell
def _(
    alt,
    build_default_observation_sigma,
    build_synthetic_sirocco_sigma,
    data_source_info,
    fit_power_law_continuum,
    mo,
    np,
    obs_data,
    obs_flux_scale,
    pd,
    wl_range_slider,
):
    # Rebuild the preview chart whenever the wavelength window or requested flux
    # transform changes, without mutating the underlying loaded dataframe.
    obs_chart = None
    if obs_data is not None:
         # Restrict the preview to the current wavelength window before applying
         # any display-only flux transformation.
         _current_min = wl_range_slider.value[0]
         _current_max = wl_range_slider.value[1]

         _mask = (obs_data['wavelength'] >= _current_min) & (obs_data['wavelength'] <= _current_max)
         _plot_df = obs_data[_mask].copy()
         _sigma_source = (
             str(_plot_df['sigma_source'].iat[0]).strip().lower()
             if 'sigma_source' in _plot_df.columns and len(_plot_df) > 0
             else None
         )
         if _sigma_source == 'synthetic_sirocco_continuum':
             _fallback_sigma_native, _ = build_synthetic_sirocco_sigma(
                 np.array(_plot_df['wavelength']),
                 np.array(_plot_df['flux']),
             )
         else:
             _fallback_sigma_native, _ = build_default_observation_sigma(
                 np.array(_plot_df['wavelength']),
                 np.array(_plot_df['flux']),
             )
         if 'error' in _plot_df.columns:
             _native_sigma = np.array(_plot_df['error'], dtype=float)
             _native_sigma = np.where(
                 np.isfinite(_native_sigma) & (_native_sigma > 0),
                 _native_sigma,
                 _fallback_sigma_native,
             )
         else:
             _native_sigma = _fallback_sigma_native

         # Mirror the emulator training transform so the plotted data matches the
         # scale used during inference.
         _flux_vals = np.array(_plot_df['flux'])
         _sigma_vals = None
         _show_sigma_band = 'error' in _plot_df.columns
         _scale_label = obs_flux_scale.value
         if _scale_label == 'log':
             _flux_vals = np.where(_flux_vals > 0, np.log10(_flux_vals), np.log10(np.abs(_flux_vals) + 1e-30))
             if _show_sigma_band:
                 _orig_flux = np.array(_plot_df['flux'])
                 _sigma_vals = _native_sigma / (
                     np.abs(_orig_flux) * np.log(10) + 1e-30
                 )
         elif _scale_label == 'continuum-normalised':
             _continuum, _ = fit_power_law_continuum(
                 np.array(_plot_df['wavelength']),
                 _flux_vals,
             )
             _cont_safe = np.where(_continuum > 0, _continuum, 1.0)
             _flux_vals = _flux_vals / _cont_safe
             if _show_sigma_band:
                 _sigma_vals = _native_sigma / _cont_safe
         elif _show_sigma_band:
             _sigma_vals = _native_sigma

         if _sigma_vals is not None:
             _sigma_vals = np.where(
                 np.isfinite(_sigma_vals) & (_sigma_vals > 0),
                 _sigma_vals,
                 1e-30,
             )
         _plot_df = pd.DataFrame({
             'Wavelength': _plot_df['wavelength'],
             'Flux': _flux_vals,
             'Type': 'Observation'
         })
         _plot_band_df = None
         if _sigma_vals is not None:
             _plot_band_df = pd.DataFrame({
                 'Wavelength': _plot_df['Wavelength'],
                 'Lower': _plot_df['Flux'] - _sigma_vals,
                 'Upper': _plot_df['Flux'] + _sigma_vals,
             })

         # Downsample only for plotting so large spectra remain responsive in Altair.
         if len(_plot_df) > 5000:
             _plot_df = _plot_df.iloc[::int(len(_plot_df)/5000)]
         if _plot_band_df is not None and len(_plot_band_df) > 5000:
             _plot_band_df = _plot_band_df.iloc[::max(1, len(_plot_band_df) // 5000)]

         _y_title = f'Flux ({_scale_label})'
         _y_format = '.1e' if _scale_label == 'linear' else '.2f'

         # Log-scale plots need an explicit domain because Altair otherwise tries
         # to include zero, which visually collapses the spectrum.
         if _scale_label == 'log':
             _y_min = float(np.nanmin(_flux_vals))
             _y_max = float(np.nanmax(_flux_vals))
             _y_pad = (_y_max - _y_min) * 0.05
             _y_scale = alt.Scale(domain=[_y_min - _y_pad, _y_max + _y_pad])
         else:
             _y_scale = alt.Undefined

         _obs_line = alt.Chart(_plot_df).mark_line(color='cyan').encode(
             x=alt.X('Wavelength', title='Wavelength (Å)'),
             y=alt.Y('Flux', title=_y_title, scale=_y_scale, axis=alt.Axis(format=_y_format)),
             tooltip=['Wavelength', 'Flux']
         )

         _obs_band = alt.LayerChart()
         if _plot_band_df is not None:
             _obs_band = alt.Chart(_plot_band_df).mark_area(
                 opacity=0.10, color='cyan'
             ).encode(
                 x=alt.X('Wavelength:Q'),
                 y=alt.Y('Lower:Q', scale=alt.Scale(zero=False)),
                 y2=alt.Y2('Upper:Q'),
             )

         obs_chart = (_obs_band + _obs_line).properties(
             width="container",
             height=400,
             title=f"{data_source_info} [{_scale_label}]" if data_source_info else "Spectrum"
         ).interactive(bind_y=False)

    obs_plot_accordion = mo.accordion({
        f"{mo.icon('lucide:trending-up')} View Selected Spectrum": obs_chart if obs_chart else mo.md("No data loaded.")
    })
    return (obs_plot_accordion,)


@app.cell
def _(
    data_source_info,
    data_source_selector,
    delete_obs_btn,
    emu,
    emulator_load_button,
    emulator_load_status,
    emulator_selector,
    file_upload_ui,
    grid_selector,
    ground_truth_params,
    mo,
    obs_data,
    obs_file_selector,
    obs_flux_scale,
    obs_plot_accordion,
    param_info_accordion,
    status_widget,
    set_inf_playground_target,
    test_inclination_selector,
    test_run_selector,
    wl_range_slider,
):
    # Assemble the top-level Stage 1 layout from reusable UI blocks so the same
    # cell can switch cleanly between observation mode and test-grid mode.
    loading_alert = None
    _is_test_grid = "Test Grid" in data_source_selector.value

    # Emulator Status
    if emu is not None:
        emu_status = mo.callout(
            mo.md(f"""
            **Emulator Loaded**
            - **Parameters:** {len(emu.param_names)}
            - **Wavelength Range:** {emu.wl.min():.1f} - {emu.wl.max():.1f} Å
            - **PCA Components:** {emu.ncomps}
            """),
            kind="success"
        )
    else:
        emu_status = mo.md("") # Empty placeholder

    # Data Status
    if obs_data is not None:
         # Update points based on range slider
        min_w = wl_range_slider.value[0]
        max_w = wl_range_slider.value[1]

        visible_points = len(obs_data[(obs_data['wavelength'] >= min_w) & (obs_data['wavelength'] <= max_w)])

        obs_status = mo.callout(
            mo.md(f"""
            **Data Loaded** ({data_source_info})
            - **Points:** {visible_points} (Target) / {len(obs_data)} (Total)
            - **Range:** {min_w:.1f} - {max_w:.1f} Å
            """), 
            kind="success"
        )
    else:
        obs_status = mo.md("")

    # Ground truth only exists in validation mode, where the selected run is tied
    # back to the lookup table loaded from the paired test grid.
    gt_display = None
    if ground_truth_params and _is_test_grid:
        from Speculate_addons.grid_registry import benchmark_param_map as _benchmark_param_map

        gt_lines = ["**Ground Truth Parameters:**"]
        for k, v in ground_truth_params.items():
            gt_lines.append(f"- **{k}**: {v:.4f}")

        _truth_label_map = {
            int(_idx): _label
            for _idx, (_label, _sirocco_key) in _benchmark_param_map(
                grid_selector.value if grid_selector is not None else None
            ).items()
        }

        def _truth_key_for_playground(_param_name):
            _name = str(_param_name)
            if "inclination" in _name.lower():
                return "Inclination"
            _digits = "".join(_char for _char in _name if _char.isdigit())
            if _digits:
                return _truth_label_map.get(int(_digits), _name)
            return _name

        _truth_phys = None
        _missing_truth_keys = []
        if emu is not None:
            _truth_phys = []
            for _param_name in emu.param_names:
                _truth_key = _truth_key_for_playground(_param_name)
                if _truth_key in ground_truth_params:
                    _truth_phys.append(float(ground_truth_params[_truth_key]))
                else:
                    _missing_truth_keys.append(_truth_key)
            if _missing_truth_keys:
                _truth_phys = None

        _gt_body = mo.callout(mo.md("\n".join(gt_lines)), kind="info")
        if _truth_phys is not None:
            def _send_ground_truth_to_playground(_):
                set_inf_playground_target({
                    "source": "test_grid_ground_truth",
                    "grid_name": grid_selector.value if grid_selector is not None else None,
                    "phys": [float(_value) for _value in _truth_phys],
                    "Av": 0.0,
                    "Distance (pc)": 100.0,
                    "cheb_1": 0.0,
                })
                mo.status.toast("Loaded ground truth into the Parameter Playground")

            _gt_export_btn = mo.ui.button(
                label=f"{mo.icon('lucide:sliders-horizontal')} Export ground truth to parameter playground",
                on_click=_send_ground_truth_to_playground,
                kind="success",
            )
            gt_display = mo.vstack([_gt_body, _gt_export_btn])
        elif emu is None:
            gt_display = mo.vstack([
                _gt_body,
                mo.callout(
                    mo.md("Load an emulator to export ground truth into the Parameter Playground."),
                    kind="neutral",
                ),
            ])
        else:
            gt_display = mo.vstack([
                _gt_body,
                mo.callout(
                    mo.md("Ground truth could not be matched to every loaded emulator parameter."),
                    kind="warn",
                ),
            ])

    # Layout Construction

    # 1. Emulator Row
    # Includes Grid Selector, Emulator Selector, and Parameter Key
    emu_row = mo.vstack([
        grid_selector,
        mo.hstack([emulator_selector, emulator_load_button], align="end", widths=[6, 1]),
        emulator_load_status,
        param_info_accordion
    ])

    # 2. Data Source Row - conditional UI based on selection
    # Swap the data controls to match the source type: test grids need run +
    # inclination selectors, while uploaded observations need file management.
    if _is_test_grid:
        # Test grid selection: run selector + inclination
        data_control_row = mo.vstack([
            data_source_selector,
            mo.hstack([test_run_selector, test_inclination_selector], widths=[3, 1]),
            wl_range_slider,
            obs_flux_scale,
        ])
    else:
        # Observation file selection
        if obs_data is not None:
            data_control_row = mo.vstack([
                data_source_selector,
                mo.hstack([obs_file_selector, delete_obs_btn], align="end", widths=[6, 1]),
                wl_range_slider,
                obs_flux_scale,
                file_upload_ui
            ])
        else:
            data_control_row = mo.vstack([
                data_source_selector,
                mo.hstack([obs_file_selector, delete_obs_btn], align="end"),
                file_upload_ui
            ])

    # 3. Status Row
    status_row = mo.hstack([status_widget, emu_status, obs_status], justify="start", gap="1rem", align="center")

    # Final Stack
    display_elements = [
         emu_row,
         mo.md("---"),
         data_control_row,
         obs_plot_accordion,
    ]
    if gt_display:
        display_elements.append(gt_display)
    display_elements.extend([mo.md(" "), status_row])

    mo.vstack(display_elements)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 2: Parameter Configuration
    """)
    return


@app.cell
def _(mo):
    get_inf_playground_target, set_inf_playground_target = mo.state(None)
    return get_inf_playground_target, set_inf_playground_target


@app.cell
def _(
    build_default_observation_sigma,
    build_synthetic_sirocco_sigma,
    emu,
    estimate_log_amp_centre_from_sigma,
    fit_power_law_continuum,
    get_inf_playground_target,
    grid_indices,
    grid_selector,
    inf_param_map_db_for_grid,
    mo,
    np,
    obs_data,
    obs_flux_scale,
    re,
    wl_range_slider,
):
    # ── Parameter Playground ──
    # Interactive sliders for the nuisance/inference parameters and the
    # physical grid axes so the user can visually explore how each shifts
    # the emulated spectrum before committing to prior bounds.
    #
    # Slider ranges for distance and log_amp are auto-computed from the ratio
    # of the emulator's midpoint flux to the (transformed) observation flux, so
    # they stay sensible regardless of emmulator training transform.

    _playground_ui = mo.md("*Load an emulator and observation data to use the playground.*")
    _pg_target = get_inf_playground_target() or {}
    if not isinstance(_pg_target, dict):
        _pg_target = {}

    def _target_value(_key, _default, _lo, _hi):
        try:
            _value = float(_pg_target.get(_key, _default))
        except Exception:
            _value = float(_default)
        if not np.isfinite(_value):
            _value = float(_default)
        return float(np.clip(_value, _lo, _hi))

    if emu is not None:
        _selected_grid = grid_selector.value if grid_selector is not None else None
        if _pg_target.get("grid_name") and _pg_target.get("grid_name") != _selected_grid:
            _pg_target = {}
        _param_map_db = inf_param_map_db_for_grid(_selected_grid)
        # Midpoint physical parameters for preview
        if hasattr(emu, 'min_params') and hasattr(emu, 'max_params'):
            pg_mid_params = (np.array(emu.min_params) + np.array(emu.max_params)) / 2.0
        else:
            pg_mid_params = np.zeros(len(emu.param_names))

        _target_phys = _pg_target.get("phys")
        if _target_phys is not None:
            try:
                _target_phys = np.asarray(_target_phys, dtype=float)
                if _target_phys.shape[0] != len(emu.param_names):
                    _pg_target = {}
                    _target_phys = None
            except Exception:
                _pg_target = {}
                _target_phys = None

        # --- Auto-compute log_amp centre from observational uncertainty ---
        # GP amplitude is a residual/noise scale. It should be tied to the data
        # uncertainty, not to mismatch against an arbitrary midpoint model.
        _distance_centre = 100.0
        _la_centre = -60.0 if obs_flux_scale.value == "linear" else -5.0
        if obs_data is not None:
            try:
                _wl_lo, _wl_hi = wl_range_slider.value
                _om = (obs_data['wavelength'] >= _wl_lo) & (obs_data['wavelength'] <= _wl_hi)
                _obs_sub = obs_data[_om]
                _obs_wl = np.array(_obs_sub['wavelength'])
                _obs_flux_native = np.array(_obs_sub['flux'])
                _sigma_source = (
                    str(_obs_sub['sigma_source'].iat[0]).strip().lower()
                    if 'sigma_source' in _obs_sub.columns and len(_obs_sub) > 0
                    else None
                )
                if _sigma_source == 'synthetic_sirocco_continuum':
                    _fallback_sigma, _ = build_synthetic_sirocco_sigma(_obs_wl, _obs_flux_native)
                else:
                    _fallback_sigma, _ = build_default_observation_sigma(_obs_wl, _obs_flux_native)

                if 'error' in _obs_sub.columns:
                    _sigma = np.array(_obs_sub['error'], dtype=float)
                    _sigma = np.where(np.isfinite(_sigma) & (_sigma > 0), _sigma, _fallback_sigma)
                else:
                    _sigma = _fallback_sigma

                _scale_label = obs_flux_scale.value
                if _scale_label == 'log':
                    _sigma = _sigma / (np.abs(_obs_flux_native) * np.log(10.0) + 1e-30)
                elif _scale_label == 'continuum-normalised':
                    _ct, _ = fit_power_law_continuum(_obs_wl, _obs_flux_native)
                    _ct_safe = np.where(_ct > 0, _ct, 1.0)
                    _sigma = _sigma / _ct_safe

                _la_centre = estimate_log_amp_centre_from_sigma(
                    _sigma,
                    fallback=_la_centre,
                    decimals=2,
                )
            except Exception:
                pass  # keep fallback values

        # Round for cleaner slider display
        _la_centre = round(_la_centre, 2)
        try:
            _logamp_default = float(_pg_target.get("log_amp", _la_centre))
        except Exception:
            _logamp_default = _la_centre
        if not np.isfinite(_logamp_default):
            _logamp_default = _la_centre
        _logamp_start = min(_la_centre - 5.0, _logamp_default - 1.0)
        _logamp_stop = max(_la_centre + 5.0, _logamp_default + 1.0)

        # Nuisance slider ranges centred on auto-detected values
        pg_av_slider = mo.ui.slider(
            start=0.0, stop=5.0, value=_target_value("Av", 0.0, 0.0, 5.0), step=0.01,
            label="Av (Extinction)", show_value=True, full_width=True,
        )
        pg_logscale_slider = mo.ui.slider(
            start=10.0, stop=1000.0,
            value=_target_value("Distance (pc)", _distance_centre, 10.0, 1000.0), step=1.0,
            label="Distance (pc)", show_value=True, full_width=True,
        )
        pg_cheb1_slider = mo.ui.slider(
            start=-2.0, stop=2.0, value=_target_value("cheb_1", 0.0, -2.0, 2.0), step=0.01,
            label="cheb_1 (continuum tilt)", show_value=True, full_width=True,
        )
        pg_logamp_slider = mo.ui.slider(
            start=_logamp_start, stop=_logamp_stop,
            value=_logamp_default, step=0.01,
            label="log_amp (ln GP amplitude)", show_value=True, full_width=True,
        )
        pg_logls_slider = mo.ui.slider(
            start=0.0, stop=15.0, value=_target_value("log_ls", 4.5, 0.0, 15.0), step=0.01,
            label="log_ls (ln GP length scale)", show_value=True, full_width=True,
        )

        # Midpoint physical parameters for preview
        if hasattr(emu, 'min_params') and hasattr(emu, 'max_params'):
            pg_mid_params = (np.array(emu.min_params) + np.array(emu.max_params)) / 2.0
        else:
            pg_mid_params = np.zeros(len(emu.param_names))

        # ── Physical parameter sliders ──
        # One slider per emulator axis so the user can reshape the emulated
        # spectrum.  Ranges span the emulator's training grid; default values
        # sit at the midpoint (matching pg_mid_params used for auto-centring).
        # Labels are resolved through param_map_db so users see physical names
        # (e.g. "disk.mdot") instead of generic "paramN" tags, and a log10()
        # wrapper is added for parameters stored on a log10 axis.
        _phys_sliders = []
        _sorted_phys_idx = sorted(list(grid_indices)) if grid_indices else []

        def _clean_slider_number(_value, _decimals=4):
            _value = float(_value)
            if not np.isfinite(_value):
                return 0.0
            if _value == 0.0:
                return 0.0
            if 1e-4 <= abs(_value) < 1e6:
                return float(f"{_value:.{_decimals}f}")
            return float(f"{_value:.{_decimals}g}")

        def _nice_slider_step(_lo, _hi, _target_steps=300):
            _span = abs(float(_hi) - float(_lo))
            if not np.isfinite(_span) or _span <= 0.0:
                return 1e-6
            _raw = _span / float(_target_steps)
            _exp = np.floor(np.log10(_raw))
            _frac = _raw / (10.0 ** _exp)
            if _frac <= 1.0:
                _nice = 1.0
            elif _frac <= 2.0:
                _nice = 2.0
            elif _frac <= 5.0:
                _nice = 5.0
            else:
                _nice = 10.0
            return max(_clean_slider_number(_nice * (10.0 ** _exp)), 1e-12)

        for _pi, _pname in enumerate(emu.param_names):
            _lo = float(emu.min_params[_pi]) if hasattr(emu, 'min_params') else 0.0
            _hi = float(emu.max_params[_pi]) if hasattr(emu, 'max_params') else 1.0
            _mid = (_lo + _hi) / 2.0
            if _target_phys is not None:
                _mid = float(np.clip(_target_phys[_pi], _lo, _hi))
            _lo_ui = _clean_slider_number(_lo)
            _hi_ui = _clean_slider_number(_hi)
            _mid_ui = _clean_slider_number(_mid)
            _step = _nice_slider_step(_lo_ui, _hi_ui)

            # Prefer the grid_indices mapping when it lines up; otherwise
            # fall back to parsing "paramN" directly from the axis name.
            _db_idx = None
            if len(_sorted_phys_idx) == len(emu.param_names):
                _db_idx = _sorted_phys_idx[_pi]
            else:
                _m = re.search(r"param(\d+)", str(_pname))
                if _m:
                    _db_idx = int(_m.group(1))

            if _db_idx is not None and _db_idx in _param_map_db:
                _base_name = _param_map_db[_db_idx][0]
                _is_log = (len(_param_map_db[_db_idx]) > 3
                           and bool(_param_map_db[_db_idx][3]))
                _label = f"log10({_base_name})" if _is_log else _base_name
            else:
                _label = str(_pname)

            _phys_sliders.append(mo.ui.slider(
                start=_lo_ui, stop=_hi_ui, value=_mid_ui, step=_step,
                label=_label, show_value=True, full_width=True,
            ))
        pg_phys_sliders = mo.ui.array(_phys_sliders)

        _playground_ui = mo.md("")
    else:
        pg_av_slider = None
        pg_logscale_slider = None
        pg_cheb1_slider = None
        pg_logamp_slider = None
        pg_logls_slider = None
        pg_mid_params = None
        pg_phys_sliders = None

    _playground_ui
    return pg_av_slider, pg_cheb1_slider, pg_logamp_slider, pg_logls_slider, pg_logscale_slider, pg_mid_params, pg_phys_sliders


@app.cell
def _(
    alt,
    build_default_observation_sigma,
    build_synthetic_sirocco_sigma,
    emu,
    fit_power_law_continuum,
    mo,
    np,
    obs_data,
    obs_flux_scale,
    pd,
    pg_av_slider,
    pg_cheb1_slider,
    pg_logamp_slider,
    pg_logls_slider,
    pg_logscale_slider,
    pg_mid_params,
    pg_nll_result,
    pg_phys_sliders,
    pg_test_nll_btn,
    wl_range_slider,
):
    # ── Playground: Reactive Emulated vs Observed Spectrum Chart ──
    # Generates the emulated spectrum at the user-selected physical parameters
    # (from the Physical tab sliders) combined with the current nuisance
    # slider values, then overlays it with the observation inside an
    # interactive Altair chart.

    _playground_chart = None

    if (emu is not None and obs_data is not None
            and pg_av_slider is not None and pg_mid_params is not None):
        try:
            from Speculate_addons.distance_scale import distance_to_log_scale as _distance_to_log_scale

            from Starfish.transforms import extinct, resample

            _wl_min = wl_range_slider.value[0]
            _wl_max = wl_range_slider.value[1]

            # Physical parameter vector — from sliders when available, otherwise
            # fall back to the emulator training-grid midpoint.
            if pg_phys_sliders is not None:
                _phys_params = np.array(pg_phys_sliders.value, dtype=float)
            else:
                _phys_params = pg_mid_params

            # --- Emulated spectrum reconstruction ---
            weights, cov = emu(_phys_params)
            X = emu.eigenspectra * emu.flux_std
            emu_flux = (weights @ X) + emu.flux_mean
            emu_wl = np.array(emu.wl)

            # Crop to wavelength range
            _emu_mask = (emu_wl >= _wl_min) & (emu_wl <= _wl_max)
            emu_wl_crop = emu_wl[_emu_mask]
            emu_flux_crop = emu_flux[_emu_mask]

            _is_log = (obs_flux_scale.value == "log")
            _ln10 = np.log(10.0)
            try:
                _norm_factor = float(np.asarray(emu.norm_factor(_phys_params)).squeeze())
            except Exception:
                _norm_factor = 0.0 if _is_log else 1.0
            if _is_log:
                emu_flux_crop = emu_flux_crop + _norm_factor
            else:
                emu_flux_crop = emu_flux_crop * _norm_factor

            if _is_log:
                # ── Log₁₀-space nuisance transforms (additive) ──────
                _av = pg_av_slider.value
                if _av > 0:
                    import extinction as _ext_mod
                    _a_lambda = _ext_mod.fitzpatrick99(
                        emu_wl_crop.astype(np.float64), _av, 3.1)
                    emu_flux_crop = emu_flux_crop + (-0.4 * _a_lambda)

                _cheb1 = pg_cheb1_slider.value
                if _cheb1 != 0.0:
                    from numpy.polynomial.chebyshev import chebval
                    _scale_wl = emu_wl_crop / emu_wl_crop.max()
                    _cheb_poly = chebval(_scale_wl, [1.0, _cheb1])
                    _cheb_poly = np.clip(_cheb_poly, 1e-30, None)
                    emu_flux_crop = emu_flux_crop + np.log10(_cheb_poly)

                _log_scale = _distance_to_log_scale(pg_logscale_slider.value)
                emu_flux_crop = emu_flux_crop + _log_scale / _ln10
            else:
                # ── Linear-space nuisance transforms (original) ──────
                _av = pg_av_slider.value
                if _av > 0:
                    emu_flux_crop = extinct(emu_wl_crop, emu_flux_crop, Av=_av)

                _cheb1 = pg_cheb1_slider.value
                if _cheb1 != 0.0:
                    from numpy.polynomial.chebyshev import chebval
                    _scale_wl = emu_wl_crop / emu_wl_crop.max()
                    _cheb_poly = chebval(_scale_wl, [1.0, _cheb1])
                    emu_flux_crop = emu_flux_crop * _cheb_poly

                _log_scale = _distance_to_log_scale(pg_logscale_slider.value)
                emu_flux_crop = emu_flux_crop * np.exp(_log_scale)

            # GP uncertainty envelope (±1σ from GP amplitude)
            _log_amp = pg_logamp_slider.value
            _gp_sigma = np.sqrt(np.exp(_log_amp))
            emu_upper = emu_flux_crop + _gp_sigma
            emu_lower = emu_flux_crop - _gp_sigma

            # --- Observation data ---
            _obs_mask = (
                (obs_data['wavelength'] >= _wl_min)
                & (obs_data['wavelength'] <= _wl_max)
            )
            _obs_sub = obs_data[_obs_mask].copy()
            _sigma_source = (
                str(_obs_sub['sigma_source'].iat[0]).strip().lower()
                if 'sigma_source' in _obs_sub.columns and len(_obs_sub) > 0
                else None
            )
            if _sigma_source == 'synthetic_sirocco_continuum':
                _fallback_sigma_native, _ = build_synthetic_sirocco_sigma(
                    np.array(_obs_sub['wavelength']),
                    np.array(_obs_sub['flux']),
                )
            else:
                _fallback_sigma_native, _ = build_default_observation_sigma(
                    np.array(_obs_sub['wavelength']),
                    np.array(_obs_sub['flux']),
                )
            if 'error' in _obs_sub.columns:
                _native_sigma = np.array(_obs_sub['error'], dtype=float)
                _native_sigma = np.where(
                    np.isfinite(_native_sigma) & (_native_sigma > 0),
                    _native_sigma,
                    _fallback_sigma_native,
                )
            else:
                _native_sigma = _fallback_sigma_native

            # Transform observation flux to match the emulator's output space.
            # The emulator already outputs in the trained space (e.g. continuum-
            # normalised), so we only transform the observation — NOT the emulator.
            _scale_label = obs_flux_scale.value
            _obs_flux = np.array(_obs_sub['flux'])
            _obs_sigma = None
            _show_obs_sigma_band = 'error' in _obs_sub.columns
            if _scale_label == 'log':
                _obs_flux = np.where(
                    _obs_flux > 0,
                    np.log10(_obs_flux),
                    np.log10(np.abs(_obs_flux) + 1e-30),
                )
                if _show_obs_sigma_band:
                    _orig_flux = np.array(_obs_sub['flux'])
                    _obs_sigma = _native_sigma / (
                        np.abs(_orig_flux) * np.log(10) + 1e-30
                    )
            elif _scale_label == 'continuum-normalised':
                _obs_cont, _ = fit_power_law_continuum(
                    np.array(_obs_sub['wavelength']), _obs_flux
                )
                _obs_cont_safe = np.where(_obs_cont > 0, _obs_cont, 1.0)
                _obs_flux = _obs_flux / _obs_cont_safe
                if _show_obs_sigma_band:
                    _obs_sigma = _native_sigma / _obs_cont_safe
            elif _show_obs_sigma_band:
                _obs_sigma = _native_sigma

            if _obs_sigma is not None:
                _obs_sigma = np.where(
                    np.isfinite(_obs_sigma) & (_obs_sigma > 0),
                    _obs_sigma,
                    1e-30,
                )

            # --- Build Altair chart ---
            _obs_df = pd.DataFrame({
                'Wavelength': np.array(_obs_sub['wavelength']),
                'Flux': _obs_flux,
                'Type': 'Observation',
            })
            _emu_df = pd.DataFrame({
                'Wavelength': emu_wl_crop,
                'Flux': emu_flux_crop,
                'Type': 'Emulated (midpoint)',
            })
            _band_df = pd.DataFrame({
                'Wavelength': emu_wl_crop,
                'Upper': emu_upper,
                'Lower': emu_lower,
            })
            _obs_band_df = None
            if _show_obs_sigma_band and _obs_sigma is not None:
                _obs_band_df = pd.DataFrame({
                    'Wavelength': np.array(_obs_sub['wavelength']),
                    'Lower': _obs_flux - _obs_sigma,
                    'Upper': _obs_flux + _obs_sigma,
                })

            # Downsample for responsiveness
            if len(_obs_df) > 5000:
                _obs_df = _obs_df.iloc[::max(1, len(_obs_df) // 5000)]
            if len(_emu_df) > 5000:
                _emu_df = _emu_df.iloc[::max(1, len(_emu_df) // 5000)]
                _band_df = _band_df.iloc[::max(1, len(_band_df) // 5000)]
            if _obs_band_df is not None and len(_obs_band_df) > 5000:
                _obs_band_df = _obs_band_df.iloc[::max(1, len(_obs_band_df) // 5000)]

            _combined = pd.concat([_obs_df, _emu_df], ignore_index=True)

            _y_title = f'Flux ({_scale_label})'
            _y_format = '.1e' if _scale_label == 'linear' else '.2f'

            _color_scale = alt.Scale(
                domain=['Observation', 'Emulated (midpoint)'],
                range=['cyan', 'orange'],
            )

            _lines = alt.Chart(_combined).mark_line(opacity=0.9).encode(
                x=alt.X('Wavelength:Q', title='Wavelength (Å)'),
                y=alt.Y('Flux:Q', title=_y_title, axis=alt.Axis(format=_y_format), scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', scale=_color_scale, legend=alt.Legend(title="Spectrum")),
                tooltip=['Wavelength:Q', 'Flux:Q', 'Type:N'],
            )

            _band = alt.Chart(_band_df).mark_area(
                opacity=0.15, color='orange',
            ).encode(
                x=alt.X('Wavelength:Q'),
                y=alt.Y('Lower:Q', scale=alt.Scale(zero=False)),
                y2=alt.Y2('Upper:Q'),
            )

            _obs_band = alt.LayerChart()
            if _obs_band_df is not None:
                _obs_band = alt.Chart(_obs_band_df).mark_area(
                    opacity=0.10, color='cyan',
                ).encode(
                    x=alt.X('Wavelength:Q'),
                    y=alt.Y('Lower:Q', scale=alt.Scale(zero=False)),
                    y2=alt.Y2('Upper:Q'),
                )

            _playground_chart = (_band + _obs_band + _lines).properties(
                width="container",
                height=400,
                title=f"Playground: Emulated (midpoint params) vs Observation [{_scale_label}]",
            ).interactive(bind_y=False)

        except Exception as _e:
            _playground_chart = mo.callout(
                mo.md(f"Playground error: {_e}"), kind="danger"
            )

    # Assemble everything inside the accordion: callout, sliders, then chart
    if (pg_av_slider is not None and _playground_chart is not None):
        _nuisance_tab = mo.vstack([
            pg_av_slider,
            pg_logscale_slider,
            pg_cheb1_slider,
            pg_logamp_slider,
            pg_logls_slider,
        ])
        if pg_phys_sliders is not None and len(pg_phys_sliders) > 0:
            _physical_tab = mo.vstack(list(pg_phys_sliders))
        else:
            _physical_tab = mo.md("*No physical parameters available.*")
        _slider_tabs = mo.ui.tabs({
            "Nuisance Parameters": _nuisance_tab,
            "Physical Parameters": _physical_tab,
        })
        _pg_content = mo.vstack([
            mo.callout(mo.md(
                "**Parameter Playground - Find sensible prior settings**\n\n"
                "Use the **Nuisance Parameters** tab to bring the emulated spectrum "
                "to a similar scale and uncertainty as the observation, and the "
                "**Physical Parameters** tab to reshape the emulated spectrum by "
                "varying the grid axes.\n\n"
                "- **Av** — Interstellar dust extinction (magnitudes). Reddens and dims the spectrum.\n"
                "- **Distance** — Source distance in parsecs. Internally converted to backend log_scale.\n"
                "- **cheb_1** — Chebyshev c₁ coefficient. Applies a linear tilt to the continuum gradient.\n"
                "- **log_amp** — GP covariance amplitude (affects uncertainty envelope, not the mean spectrum).\n"
                "- **log_ls** — GP covariance length scale (affects uncertainty smoothness, not the mean spectrum)."
            ), kind="neutral"),
            _slider_tabs,
            _playground_chart,
            pg_test_nll_btn,
            pg_nll_result,
        ])
    elif _playground_chart is not None:
        _pg_content = mo.vstack([
            _playground_chart,
            pg_test_nll_btn,
            pg_nll_result,
        ])
    else:
        _pg_content = mo.md(
            "*Load an emulator and observation to see the playground chart.*"
        )

    mo.accordion({
        "Parameter Playground - Aid to find optimal prior settings": _pg_content
    })
    return


@app.cell
def _(mo):
    # Run button for on-demand NLL evaluation from the playground sliders.
    # Kept in its own cell so the evaluator cell below picks up click events
    # reactively (a run_button's `.value` only flips in downstream cells).
    pg_test_nll_btn = mo.ui.run_button(
        label=f"{mo.icon('lucide:calculator')} Test NLL"
    )
    return (pg_test_nll_btn,)


@app.cell
def _(
    build_default_observation_sigma,
    distance_prior_ack,
    Spectrum,
    SpectrumModel,
    build_synthetic_sirocco_sigma,
    emu,
    fit_power_law_continuum,
    mo,
    np,
    obs_data,
    obs_flux_scale,
    pg_av_slider,
    pg_cheb1_slider,
    pg_logamp_slider,
    pg_logls_slider,
    pg_logscale_slider,
    pg_phys_sliders,
    pg_test_nll_btn,
    stats,
    wl_range_slider,
):
    # ── Test NLL ──
    # On click, build a SpectrumModel at the current playground slider values
    # and evaluate the negative log-likelihood — the same objective the MLE
    # optimiser minimises — so the user can sanity-check whether their
    # chosen prior region is close to a reasonable fit.  The result widget
    # is rendered inside the playground accordion by the cell above.
    pg_nll_result = mo.md("")

    if pg_test_nll_btn.value:
        if emu is None or obs_data is None or pg_phys_sliders is None:
            pg_nll_result = mo.callout(
                mo.md("Load an emulator and observation first."),
                kind="warn",
            )
        else:
            try:
                from Speculate_addons.distance_scale import distance_to_log_scale as _distance_to_log_scale

                _wl_lo, _wl_hi = wl_range_slider.value
                _mask = ((obs_data['wavelength'] >= _wl_lo)
                         & (obs_data['wavelength'] <= _wl_hi))
                _sub = obs_data[_mask].copy()

                _scale = obs_flux_scale.value
                _raw_fl = np.array(_sub['flux'])
                _cont_safe = None
                if _scale == 'log':
                    _raw_fl = np.where(
                        _raw_fl > 0,
                        np.log10(_raw_fl),
                        np.log10(np.abs(_raw_fl) + 1e-30),
                    )
                elif _scale == 'continuum-normalised':
                    _cont, _ = fit_power_law_continuum(
                        np.array(_sub['wavelength']), _raw_fl
                    )
                    _cont_safe = np.where(_cont > 0, _cont, 1.0)
                    _raw_fl = _raw_fl / _cont_safe
                _sub['flux'] = _raw_fl

                _sigma_source = (
                    str(_sub['sigma_source'].iat[0]).strip().lower()
                    if 'sigma_source' in _sub.columns and len(_sub) > 0
                    else None
                )
                if _sigma_source == 'synthetic_sirocco_continuum':
                    _fallback_sigma_native, _ = build_synthetic_sirocco_sigma(
                        np.array(_sub['wavelength']),
                        np.array(obs_data[_mask]['flux']),
                    )
                else:
                    _fallback_sigma_native, _ = build_default_observation_sigma(
                        np.array(_sub['wavelength']),
                        np.array(obs_data[_mask]['flux']),
                    )

                if 'error' in _sub.columns:
                    _sigma = np.array(_sub['error'], dtype=float)
                    _sigma = np.where(
                        np.isfinite(_sigma) & (_sigma > 0),
                        _sigma,
                        _fallback_sigma_native,
                    )
                    if _scale == 'log':
                        _orig = np.array(obs_data[_mask]['flux'])
                        _sigma = _sigma / (np.abs(_orig) * np.log(10) + 1e-30)
                    elif _scale == 'continuum-normalised':
                        _sigma = _sigma / _cont_safe
                else:
                    _sigma = _fallback_sigma_native
                    if _scale == 'log':
                        _orig = np.array(obs_data[_mask]['flux'])
                        _sigma = _sigma / (np.abs(_orig) * np.log(10) + 1e-30)
                    elif _scale == 'continuum-normalised':
                        _sigma = _sigma / _cont_safe
                _sigma = np.where(
                    np.isfinite(_sigma) & (_sigma > 0),
                    _sigma,
                    1e-30,
                )

                _spec = Spectrum(
                    np.array(_sub['wavelength']),
                    np.array(_sub['flux']),
                    sigmas=_sigma,
                )

                _phys = np.array(pg_phys_sliders.value, dtype=float)
                _global = {
                    'Av': float(pg_av_slider.value),
                    'log_scale': float(_distance_to_log_scale(pg_logscale_slider.value)),
                    'cheb': [float(pg_cheb1_slider.value)],
                    'global_cov': {
                        'log_amp': float(pg_logamp_slider.value),
                        'log_ls': float(pg_logls_slider.value),
                    },
                }
                _model = SpectrumModel(
                    emulator=emu,
                    data=_spec,
                    grid_params=list(_phys),
                    flux_scale=_scale,
                    norm=True,
                    **_global,
                )

                # Flat (uniform) priors spanning the slider ranges so the
                # prior contribution is constant and NLL reflects only the
                # data-likelihood term at the chosen slider point.
                _priors = {}
                for _pi, _pname in enumerate(emu.param_names):
                    _lo = float(emu.min_params[_pi])
                    _hi = float(emu.max_params[_pi])
                    if _hi > _lo:
                        _priors[_pname] = stats.uniform(loc=_lo, scale=_hi - _lo)
                _priors['Av'] = stats.uniform(loc=0.0, scale=5.0)
                _ls_a = _distance_to_log_scale(pg_logscale_slider.start)
                _ls_b = _distance_to_log_scale(pg_logscale_slider.stop)
                _priors['log_scale'] = stats.uniform(
                    loc=min(_ls_a, _ls_b),
                    scale=abs(_ls_b - _ls_a),
                )
                _priors['cheb:1'] = stats.uniform(loc=-2.0, scale=4.0)
                _priors['global_cov:log_amp'] = stats.uniform(
                    loc=pg_logamp_slider.start,
                    scale=pg_logamp_slider.stop - pg_logamp_slider.start,
                )
                _priors['global_cov:log_ls'] = stats.uniform(loc=0.0, scale=15.0)

                _ll = float(_model.log_likelihood(_priors)) 
                _ll_without_prior = float(_model.log_likelihood())
                _nll = -_ll
                _nll_without_prior = -_ll_without_prior
                _diff = -1*(_nll - _nll_without_prior)
                pg_nll_result = mo.callout(
                    mo.md(
                        f"**NLL = {_nll:.4f}** "
                        f"(log-likelihood = {_ll:.4f})\n\n" 
                        f"(Without priors: NLL = {_nll_without_prior:.4f}\n\n"
                        f"log-likelihood = {_ll_without_prior:.4f})\n\n"
                        f"Difference due to priors: {_diff:.4f}\n\n"
                        f"Evaluated at current playground slider values."
                    ),
                    kind="success" if np.isfinite(_nll) else "danger",
                )
            except Exception as _e:
                pg_nll_result = mo.callout(
                    mo.md(f"NLL evaluation failed: `{_e}`"),
                    kind="danger",
                )

    return (pg_nll_result,)


@app.cell
def _(GP_LOG_AMP_PRIOR_SIGMA, build_default_observation_sigma, build_synthetic_sirocco_sigma, emu, estimate_log_amp_centre_from_sigma, fit_power_law_continuum, grid_indices, grid_selector, inf_param_map_db_for_grid, mo, np, obs_data, obs_flux_scale, re, wl_range_slider):
    # ── Stage 2: Prior & Parameter Setup ──
    # Build per-parameter UI widgets (fixed/free toggle, value, min, max) that
    # drive the prior construction in Stage 3.  Parameters fall into two groups:
    #
    #   Grid parameters — the physical model axes stored in the emulator (e.g.
    #       disk.mdot, KWD.d, Inclination).  Bounds are seeded from the
    #       emulator’s training-grid limits.
    #
    #   Inference / nuisance parameters — added on top of the grid axes:
    #       Av         – interstellar extinction in magnitudes
    #       log_scale  – backend ln(flux scale), shown to users as Distance (pc)
    #       log_amp    – ln(GP global covariance amplitude)
    #       log_ls     – ln(GP global covariance length scale)
    #
    # All bounds and defaults can be adjusted interactively before MLE/MCMC.

    param_names = []
    defaults = {}
    bounds = {}
    log10_params = set()  # names of params whose values are log10-scaled

    # Parameter Mapping Dictionary
    if emu is not None:
        _param_map_db = inf_param_map_db_for_grid(grid_selector.value if grid_selector is not None else None)
        current_names = []
        sorted_indices = sorted(list(grid_indices)) if grid_indices else []

        # Recover human-readable parameter names for the active emulator axes.
        # When the filename already encodes the grid indices, prefer that mapping;
        # otherwise fall back to parsing the internal paramX labels directly.
        if len(sorted_indices) == len(emu.param_names):
             for idx in sorted_indices:
                 if idx in _param_map_db:
                     _p_name = _param_map_db[idx][0]
                     current_names.append(_p_name)
                     if _param_map_db[idx][3]:
                         log10_params.add(_p_name)
                 else:
                     current_names.append(f"Param {idx} (Unknown)")
        else:
            # Fallback: Try to parse index from the param name itself if it is "paramX"
            for _p in emu.param_names:
                match = re.search(r'param(\d+)', _p)
                if match:
                    idx = int(match.group(1))
                    if idx in _param_map_db:
                        current_names.append(_param_map_db[idx][0])
                        if _param_map_db[idx][3]:
                            log10_params.add(_param_map_db[idx][0])
                    else:
                        current_names.append(f"Param {idx}")
                else:
                    current_names.append(_p)

        param_names.extend(current_names)

        # Seed the UI bounds from the emulator limits. These values are already in
        # emulator space, so the UI stays consistent with the model inputs.
        if hasattr(emu, 'min_params') and hasattr(emu, 'max_params'):
            for _i, _name in enumerate(current_names):
                mn = float(emu.min_params[_i])
                mx = float(emu.max_params[_i])
                bounds[_name] = [mn, mx]
                defaults[_name] = (mn + mx) / 2.0
        else:
             for _name in current_names:
                bounds[_name] = [0.0, 1.0]
                defaults[_name] = 0.5

    # Add Inference Parameters (Global / Nuisance)
    # These are not part of the emulator grid but are optimised alongside
    # the grid parameters during MLE/MCMC.  They account for:
    #   Av        – dust reddening / extinction
    #   log_scale – backend distance/solid-angle scale, shown as Distance (pc)
    #   log_amp   – GP global covariance amplitude (model flexibility)
    #   log_ls    – GP global covariance length scale (smoothness)
    inf_params = [
        'Av',        # Extinction (magnitudes)
        'log_scale', # Distance in pc in the UI; backend stores natural-log flux scale
        'cheb_1',    # Chebyshev c1 coefficient (continuum tilt)
        'log_amp',   # GP Global Covariance Amplitude (natural log)
        'log_ls'     # GP Global Covariance Length Scale (natural log)
    ] 

    # Only add if emu is loaded, otherwise empty
    if emu is not None:
        param_names.extend(inf_params)

        # Defaults for Global Params
        bounds['Av'] = [0.0, 2.0]
        defaults['Av'] = 0.0

        # log_scale is exposed as a distance prior. With SpectrumModel(norm=True),
        # the emulator's stored normalisation is the 100 pc reference and
        # backend log_scale=0 means Distance=100 pc.
        _distance_default = 100.0
        _la_default = -60.0 if obs_flux_scale.value == "linear" else -5.0
        if obs_data is not None:
            try:
                _wl_lo, _wl_hi = wl_range_slider.value
                _om_tmp = (obs_data['wavelength'] >= _wl_lo) & (obs_data['wavelength'] <= _wl_hi)
                _obs_sub = obs_data[_om_tmp]
                _owl = np.array(_obs_sub['wavelength'])
                _ofl_native = np.array(_obs_sub['flux'])
                _sigma_source = (
                    str(_obs_sub['sigma_source'].iat[0]).strip().lower()
                    if 'sigma_source' in _obs_sub.columns and len(_obs_sub) > 0
                    else None
                )
                if _sigma_source == 'synthetic_sirocco_continuum':
                    _fallback_sigma, _ = build_synthetic_sirocco_sigma(_owl, _ofl_native)
                else:
                    _fallback_sigma, _ = build_default_observation_sigma(_owl, _ofl_native)
                if 'error' in _obs_sub.columns:
                    _sigma = np.array(_obs_sub['error'], dtype=float)
                    _sigma = np.where(np.isfinite(_sigma) & (_sigma > 0), _sigma, _fallback_sigma)
                else:
                    _sigma = _fallback_sigma

                _sc = obs_flux_scale.value
                if _sc == 'log':
                    _sigma = _sigma / (np.abs(_ofl_native) * np.log(10.0) + 1e-30)
                elif _sc == 'continuum-normalised':
                    _ct2, _ = fit_power_law_continuum(_owl, _ofl_native)
                    _ct2s = np.where(_ct2 > 0, _ct2, 1.0)
                    _sigma = _sigma / _ct2s

                _la_default = estimate_log_amp_centre_from_sigma(
                    _sigma,
                    fallback=_la_default,
                )
            except Exception:
                pass
        _la_default = round(_la_default, 1)

        bounds['log_scale'] = [90.0, 110.0]
        defaults['log_scale'] = _distance_default

        # cheb_1: Chebyshev c1 coefficient – linear continuum tilt
        # Multiplies the model by a Chebyshev polynomial; c0 is fixed to 1.
        # Small values nudge the continuum gradient without changing lines.
        bounds['cheb_1'] = [-0.5, 0.5]
        defaults['cheb_1'] = 0.0

        # log_amp: GP global covariance amplitude (natural log)
        # Normal prior: center is tied to observational uncertainty, while the
        # displayed range gives the optimizer room to discover extra model
        # discrepancy without starting from arbitrary line mismatch.
        _log_amp_half_width = 2.0 * GP_LOG_AMP_PRIOR_SIGMA
        bounds['log_amp'] = [_la_default - _log_amp_half_width, _la_default + _log_amp_half_width]
        defaults['log_amp'] = _la_default

        # log_ls: GP global covariance length scale (natural log)
        bounds['log_ls'] = [1.0, 8.0]
        defaults['log_ls'] = 4.5

    # Build one widget bundle per parameter: a fixed/free toggle, a point value,
    # and lower/upper controls for the prior bounds shown in Stage 2.
    _fix_dict = {}
    _val_dict = {}
    w_min = {}
    w_max = {}
    distance_prior_ack = mo.ui.checkbox(
        value=False,
        label="I have entered the target distance and uncertainty",
    )

    _fix_distance_for_shape_only = obs_flux_scale.value == "continuum-normalised"
    for _name in param_names:
        _fix_dict[_name] = mo.ui.checkbox(
            value=(_name == 'log_scale' and _fix_distance_for_shape_only)
        )

        mn, mx = bounds.get(_name, [0.0, 1.0])
        default = defaults.get(_name, 0.5)

        _val_dict[_name] = mo.ui.number(value=default, step=0.01, label="", full_width=True)
        w_min[_name] = mo.ui.number(value=mn, step=0.01, label="Min", full_width=True)
        w_max[_name] = mo.ui.number(value=mx, step=0.01, label="Max", full_width=True)

    # Wrap in mo.ui.dictionary for Marimo reactivity
    w_fix = mo.ui.dictionary(_fix_dict) if _fix_dict else mo.ui.dictionary({})
    w_val = mo.ui.dictionary(_val_dict) if _val_dict else mo.ui.dictionary({})
    return bounds, distance_prior_ack, log10_params, param_names, w_fix, w_max, w_min, w_val


@app.cell(hide_code=True)
def _(bounds, distance_prior_ack, log10_params, mo, param_names, w_fix, w_max, w_min, w_val):
    # Render Parameter UI
    if not param_names:
         param_settings = mo.md("*Load an emulator to see parameters.*")
    else:
        # Parameters that use a normal prior instead of uniform
        _normal_prior_params = {'log_amp', 'log_scale'}

        # Friendly display names
        _label_map = {
            'log_scale': 'Distance (pc)',
            'cheb_1': 'Continuum Tilt (c₁)',
            'log_amp': 'GP Log Amp (ln)',
            'log_ls': 'GP Log Length (ln)',
        }

        # Read the reactive dictionary values
        _fix_values = w_fix.value   # dict: {name: bool}
        _val_elements = w_val.elements  # dict: {name: widget}

        # Keep physical grid parameters separate from nuisance / GP parameters so
        # the user can see which controls map to the emulator grid itself.
        _grid_rows = []
        _global_rows = []

        for _name in param_names:
            is_fixed = _fix_values.get(_name, False)
            display_name = _label_map.get(_name) or str(_name)
            # Wrap log10-scaled parameters so the user knows the value is a logarithm
            if _name in log10_params and not display_name.startswith('log'):
                display_name = f"log10({display_name})"
            _mn, _mx = bounds.get(_name, [0.0, 1.0])
            is_global = _name in ('Av', 'log_scale', 'cheb_1', 'log_amp', 'log_ls')

            # Fix toggle — access the individual checkbox widget from w_fix
            _fix_widget = w_fix.elements[_name]

            if is_fixed:
                # FIXED: show value input + locked indicator
                _prior_col = mo.hstack([
                    _val_elements[_name],
                    mo.md(f"<span style='color: orange;'>{mo.icon('lucide:lock')} Fixed</span>"),
                ], widths=[3, 2], align="center")
            else:
                # FREE: show prior distribution controls
                if _name in _normal_prior_params:
                    # Normal prior: Value = center, Min/Max define ±2σ range
                    _prior_col = mo.hstack([
                        mo.vstack([mo.md("<small>Center</small>"), _val_elements[_name]], gap=0),
                        mo.vstack([mo.md("<small>-2σ</small>"), w_min[_name]], gap=0),
                        mo.vstack([mo.md("<small>+2σ</small>"), w_max[_name]], gap=0),
                        mo.md("<small style='color: cyan;'>𝒩 Normal</small>"),
                    ], widths=[2, 2, 2, 1], align="end")
                else:
                    # Uniform prior: Min/Max define range
                    _prior_col = mo.hstack([
                        mo.vstack([mo.md("<small>Lower</small>"), w_min[_name]], gap=0),
                        mo.vstack([mo.md("<small>Upper</small>"), w_max[_name]], gap=0),
                    ], widths=[1, 1], align="end")

            _row = mo.hstack([
                mo.md(f"**{display_name}**"),
                _fix_widget,
                _prior_col,
            ], widths=[2, 1, 5], align="center")

            if is_global:
                _global_rows.append(_row)
            else:
                _grid_rows.append(_row)

        # Assemble sections
        _sections = []
        if _grid_rows:
            _sections.append(mo.md("#### Grid Parameters"))
            _sections.extend(_grid_rows)
        if _global_rows:
            _sections.append(mo.md("---"))
            _sections.append(mo.md("#### Inference / Nuisance Parameters"))
            _sections.extend(_global_rows)
            if 'log_scale' in param_names and not _fix_values.get('log_scale', False):
                _sections.append(distance_prior_ack)

        param_settings = mo.vstack(_sections)

    mo.vstack([
        mo.md("### Prior Configuration"),
        mo.callout(mo.md(
            "Set prior distributions for each parameter. "
            "**Fix** a parameter to lock it at a specific value (tight δ-prior). "
            "Free parameters use **Uniform** priors (lower–upper) except "
            "Distance and GP Log Amp, which use **Normal** priors (center ± 2σ)."
        ), kind="neutral"),
        param_settings
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 3: Inference (Maximum Likelihood Estimation)
    """)
    return


@app.cell
def _(mo):
    # Reactive state holders for the MLE → MCMC → Export pipeline.
    # Each state variable is set by the producing cell and consumed by the
    # next stage, allowing the notebook to remain correct under partial reruns.
    get_mle_model, set_mle_model = mo.state(None)       # fitted SpectrumModel
    get_mle_priors, set_mle_priors = mo.state(None)     # frozen scipy priors dict
    get_mle_result, set_mle_result = mo.state(None)     # rendered Stage 3 summary/plot payload
    # MCMC State - stores results for export
    get_mcmc_samples, set_mcmc_samples = mo.state(None)     # (n_samples, n_dim) array
    get_mcmc_labels, set_mcmc_labels = mo.state(None)       # friendly parameter names
    get_mcmc_summary_df, set_mcmc_summary_df = mo.state(None) # ArviZ summary table
    get_mcmc_corner_meta, set_mcmc_corner_meta = mo.state(None) # corner-plot export metadata
    return (
        get_mle_model, get_mle_priors, get_mle_result,
        set_mle_model, set_mle_priors, set_mle_result,
        get_mcmc_samples, set_mcmc_samples,
        get_mcmc_labels, set_mcmc_labels,
        get_mcmc_corner_meta, set_mcmc_corner_meta,
        get_mcmc_summary_df, set_mcmc_summary_df,
    )


@app.cell
def _(mo):
    # Inference Control
    mle_method = mo.ui.dropdown(
        options=["CMA-ES", "Nelder-Mead", "L-BFGS-B"],
        value="CMA-ES",
        label="Optimization Method:",
    )
    mle_max_iter = mo.ui.number(
        start=100, stop=100000, value=10000, step=100,
        label="Max Iterations:",
    )
    mle_restarts = mo.ui.number(
        start=1, stop=10, value=3, step=1,
        label="Restarts:",
    )
    run_mle_btn = mo.ui.run_button(label=f"{mo.icon('lucide:rocket')} Run MLE", kind="success")

    _method_info = mo.accordion({
        f"{mo.icon('lucide:info')} Optimizer Details": mo.md("""
        | Method | Type | Best For |
        |--------|------|----------|
        | **CMA-ES** | Evolutionary strategy | **Recommended default** — robust in multi-modal, high-dimensional landscapes |
        | **Nelder-Mead** | Derivative-free simplex | Fast for low-dimensional problems; can stall in >6 dimensions |
        | **L-BFGS-B** | Quasi-Newton (gradient) | Fast convergence when the likelihood surface is smooth |

        **CMA-ES** is population-based and adapts its search covariance — best for 7+ dimensional spectral fitting.
        **Nelder-Mead** uses an adaptive simplex spanning the prior volume.
        **L-BFGS-B** uses box constraints derived from your priors and finite-difference gradients.

        **Restarts** runs the optimiser multiple times from different random starting
        points and keeps the best result, greatly reducing the chance of local-minimum trapping.
        """)
    })

    mo.vstack([
        mo.md("Run Maximum Likelihood Estimation (MLE) to find the best fit parameters."),
        mo.hstack([mle_method, mle_max_iter, mle_restarts], justify="start", gap=1),
        _method_info,
        run_mle_btn,
    ])
    return (mle_max_iter, mle_method, mle_restarts, run_mle_btn,)


@app.cell
def _(mo):
    mle_show_ground_truth_spectrum = mo.ui.checkbox(
        value=False,
        label="Show ground-truth emulator spectrum",
    )
    return (mle_show_ground_truth_spectrum,)


@app.cell
def _(
    build_default_observation_sigma,
    distance_prior_ack,
    Spectrum,
    SpectrumModel,
    bounds_for_frozen_prior,
    build_synthetic_sirocco_sigma,
    emu,
    fit_power_law_continuum,
    get_frozen_dist_loc_scale,
    global_covariance_diagnostics,
    ground_truth_params,
    grid_selector,
    log10_params,
    mle_max_iter,
    mle_method,
    mle_restarts,
    mo,
    np,
    obs_data,
    param_names,
    run_mle_btn,
    set_mle_model,
    set_mle_priors,
    set_mle_result,
    stats,
    w_fix,
    w_max,
    w_min,
    w_val,
    obs_flux_scale,
    wl_range_slider,
):
    # ── Stage 3: Maximum Likelihood Estimation (MLE) ──
    # Supports Nelder-Mead, L-BFGS-B, and CMA-ES optimizers.
    fit_status = None
    fit_results = None

    if run_mle_btn.value:
        set_mle_model(None)
        set_mle_priors(None)
        set_mle_result(None)
        if emu is None or obs_data is None:
            mo.stop(True, mo.md("Please load both an emulator and data."))
        if ('log_scale' in param_names
            and not w_fix.value.get('log_scale', False)
            and not distance_prior_ack.value):
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "Confirm the Distance (pc) prior in Stage 2 before running MLE. "
                        "The center is the target distance and the -2σ/+2σ fields define its uncertainty."
                    ),
                    kind="warn",
                ),
            )

        # Use a spinner rather than a static markdown block so the notebook keeps
        # surfacing progress while the optimizer and plotting code run.
        with mo.status.spinner("Preparing MLE...") as _spinner:
            try:
                from scipy.optimize import minimize as scipy_minimize
                import time as _time
                from Speculate_addons.distance_scale import (
                    distance_prior_to_log_scale_prior as _distance_prior_to_log_scale_prior,
                    distance_to_log_scale as _distance_to_log_scale,
                    log_scale_to_distance_pc as _log_scale_to_distance_pc,
                )

                # Restrict the observation to the selected wavelength window before
                # transforming fluxes and uncertainties onto the emulator scale.
                _min_w = wl_range_slider.value[0]
                _max_w = wl_range_slider.value[1]

                _mask = (obs_data['wavelength'] >= _min_w) & (obs_data['wavelength'] <= _max_w)
                _data_subset = obs_data[_mask].copy()

                # Apply exactly the same flux transform used during emulator training.
                _flux_scale = obs_flux_scale.value
                _native_flux = np.array(_data_subset['flux'], dtype=float)
                _sigma_source = (
                    str(_data_subset['sigma_source'].iat[0]).strip().lower()
                    if 'sigma_source' in _data_subset.columns and len(_data_subset) > 0
                    else None
                )
                if _sigma_source == 'synthetic_sirocco_continuum':
                    _fallback_sigma_native, _ = build_synthetic_sirocco_sigma(
                        np.array(_data_subset['wavelength']),
                        _native_flux,
                    )
                else:
                    _fallback_sigma_native, _ = build_default_observation_sigma(
                        np.array(_data_subset['wavelength']),
                        _native_flux,
                    )

                if 'error' in _data_subset.columns:
                    _native_sigma = np.array(_data_subset['error'], dtype=float)
                    _native_sigma = np.where(
                        np.isfinite(_native_sigma) & (_native_sigma > 0),
                        _native_sigma,
                        _fallback_sigma_native,
                    )
                else:
                    _native_sigma = _fallback_sigma_native

                _raw_flux = _native_flux.copy()
                if _flux_scale == 'log':
                    # Guard against non-positive values
                    _raw_flux = np.where(_raw_flux > 0, np.log10(_raw_flux), np.log10(np.abs(_raw_flux) + 1e-30))
                    _sigma = _native_sigma / (np.abs(_native_flux) * np.log(10) + 1e-30)
                elif _flux_scale == 'continuum-normalised':
                    _cont, _ = fit_power_law_continuum(
                        np.array(_data_subset['wavelength']), _native_flux
                    )
                    _cont_safe = np.where(_cont > 0, _cont, 1.0)
                    _raw_flux = _native_flux / _cont_safe
                    _sigma = _native_sigma / _cont_safe
                else:
                    _sigma = _native_sigma
                # else: linear (no transform)
                _data_subset['flux'] = _raw_flux

                # Ensure no zero/negative sigmas after propagation.
                _sigma = np.where(
                    np.isfinite(_sigma) & (_sigma > 0),
                    _sigma,
                    1e-30,
                )

                spec_data = Spectrum(
                    np.array(_data_subset['wavelength']),
                    np.array(_data_subset['flux']),
                    sigmas=_sigma
                )

                # Split UI controls into grid parameters and global nuisance terms,
                # because SpectrumModel expects them in different constructor slots.
                num_grid_params = len(emu.param_names)
                phys_names = param_names[:num_grid_params]

                # Initialize Grid Params from UI values
                grid_params_init = [w_val.value[n] for n in phys_names]

                # Global Params construction
                global_params = {}

                # Handle 'Av'
                if 'Av' in param_names:
                    global_params['Av'] = w_val.value['Av']

                # Handle GP (log_amp, log_ls)
                if 'log_amp' in param_names and 'log_ls' in param_names:
                    global_params['global_cov'] = {
                        'log_amp': w_val.value['log_amp'],
                        'log_ls': w_val.value['log_ls']
                    }

                # Handle Chebyshev continuum tilt (cheb_1)
                # SpectrumModel expects cheb=[c1, c2, ...] (c0=1 is enforced internally)
                if 'cheb_1' in param_names:
                    global_params['cheb'] = [w_val.value['cheb_1']]

                # Handle Distance (pc) UI wrapper. Backend remains log_scale.
                if 'log_scale' in param_names:
                    global_params['log_scale'] = _distance_to_log_scale(
                        w_val.value['log_scale']
                    )

                # 3. Initialize Model
                _model = SpectrumModel(
                    emulator=emu,
                    data=spec_data,
                    grid_params=grid_params_init,
                    flux_scale=_flux_scale,
                    norm=True,
                    **global_params,
                )

                # Translate the Stage 2 UI state into scipy prior objects and freeze
                # any parameters the user explicitly locked.
                _priors = {}

                # A. Grid Parameter Priors
                for _internal, _ui in zip(emu.param_names, phys_names):
                    if w_fix.value[_ui]:
                        # Set the fixed value on the model, then freeze
                        _fixed_val = w_val.value[_ui]
                        _model.params[_internal] = _fixed_val
                        _model.freeze(_internal)
                    else:
                        _mn = w_min[_ui].value
                        _mx = w_max[_ui].value
                        if _mx > _mn:
                            _priors[_internal] = stats.uniform(loc=_mn, scale=_mx - _mn)

                # B. Global Parameter Priors
                if 'Av' in param_names:
                    if w_fix.value['Av']:
                        _model.params['Av'] = w_val.value['Av']
                        _model.freeze('Av')
                    else:
                        _mn = w_min['Av'].value
                        _mx = w_max['Av'].value
                        _priors['Av'] = stats.uniform(loc=_mn, scale=_mx - _mn)

                # C. Distance prior (backend key: log_scale)
                if 'log_scale' in param_names:
                    _distance_pc = float(w_val.value['log_scale'])
                    _distance_lo = float(w_min['log_scale'].value)
                    _distance_hi = float(w_max['log_scale'].value)
                    _distance_sigma = max((_distance_hi - _distance_lo) / 4.0, 1e-12)
                    _ls_loc, _ls_sigma = _distance_prior_to_log_scale_prior(
                        _distance_pc, _distance_sigma
                    )
                    if w_fix.value['log_scale']:
                        _model.params['log_scale'] = _ls_loc
                        _model.freeze('log_scale')
                    else:
                        _model.params['log_scale'] = _ls_loc
                        _priors['log_scale'] = stats.norm(loc=_ls_loc, scale=_ls_sigma)

                # D. GP Priors (log_amp and log_ls handled independently)
                if 'log_amp' in param_names:
                    if w_fix.value['log_amp']:
                        _model.params['global_cov:log_amp'] = w_val.value['log_amp']
                        _model.freeze('global_cov:log_amp')
                    else:
                        # Normal prior for log_amp (matches Speculate_dev.py)
                        _center = w_val.value['log_amp']
                        _sigma = (w_max['log_amp'].value - w_min['log_amp'].value) / 4.0  # ±2σ spans range
                        _priors['global_cov:log_amp'] = stats.norm(loc=_center, scale=_sigma)

                if 'log_ls' in param_names:
                    if w_fix.value['log_ls']:
                        _model.params['global_cov:log_ls'] = w_val.value['log_ls']
                        _model.freeze('global_cov:log_ls')
                    else:
                        _mn = w_min['log_ls'].value
                        _mx = w_max['log_ls'].value
                        _priors['global_cov:log_ls'] = stats.uniform(loc=_mn, scale=_mx - _mn)

                # If BOTH GP params are fixed, freeze the entire global_cov group
                if ('log_amp' in param_names and 'log_ls' in param_names 
                        and w_fix.value['log_amp'] and w_fix.value['log_ls']):
                    _model.freeze('global_cov')

                # E. Chebyshev continuum tilt (cheb_1 -> internal key cheb:1)
                if 'cheb_1' in param_names:
                    if w_fix.value['cheb_1']:
                        _model['cheb:1'] = w_val.value['cheb_1']
                        _model.freeze('cheb')
                    else:
                        _mn = w_min['cheb_1'].value
                        _mx = w_max['cheb_1'].value
                        _priors['cheb:1'] = stats.uniform(loc=_mn, scale=_mx - _mn)

                # 4. Build Initial Simplex / Bounds
                _spinner.update("Preparing optimizer...")
                _opt_method = mle_method.value

                _active_labels = list(_model.labels)
                _N = len(_active_labels)

                # Derive per-parameter bounds from the priors (used by all methods).
                _lo_bounds = []
                _hi_bounds = []
                for _label in _active_labels:
                    if _label in _priors:
                        _dist = _priors[_label]
                        _prior_bounds = bounds_for_frozen_prior(_label, _dist)
                        if _prior_bounds is not None:
                            _lo, _hi = _prior_bounds
                            _lo_bounds.append(_lo)
                            _hi_bounds.append(_hi)
                        else:
                            _cv = _model.get_param_vector()[_active_labels.index(_label)]
                            _lo_bounds.append(_cv - abs(_cv) * 0.5)
                            _hi_bounds.append(_cv + abs(_cv) * 0.5)
                    else:
                        _cv = _model.get_param_vector()[_active_labels.index(_label)]
                        _lo_bounds.append(_cv - abs(_cv) * 0.5 - 1e-6)
                        _hi_bounds.append(_cv + abs(_cv) * 0.5 + 1e-6)

                _simplex = None
                if _opt_method == "Nelder-Mead":
                    # Build an initial simplex spanning the prior volume.
                    def _simplex_column_uniform(loc, scale, N):
                        mn = loc
                        mx = loc + scale
                        rng = mx - mn
                        margin = rng / 20
                        t_mn, t_mx = mn + margin, mx - margin
                        interval = (t_mx - t_mn) / N
                        return [t_mn + interval * k for k in range(N + 1)]

                    def _simplex_column_norm(mean, std, N):
                        mn = mean - 2 * std
                        mx = mean + 2 * std
                        interval = (mx - mn) / N
                        return [mn + interval * k for k in range(N + 1)]

                    _simplex = np.zeros((_N + 1, _N))
                    _simplex_info = []

                    for _col_idx, _label in enumerate(_active_labels):
                        if _label in _priors:
                            _dist = _priors[_label]
                            _loc, _sc = get_frozen_dist_loc_scale(_dist)
                            if _dist.dist.name == 'uniform':
                                _col = _simplex_column_uniform(_loc, _sc, _N)
                            elif _dist.dist.name == 'norm':
                                _col = _simplex_column_norm(_loc, _sc, _N)
                            else:
                                _cv = _model.get_param_vector()[_col_idx]
                                _col = [_cv + (_cv * 0.01 * k) for k in range(_N + 1)]
                        else:
                            _cv = _model.get_param_vector()[_col_idx]
                            _col = [_cv] * (_N + 1)

                        _simplex[:, _col_idx] = _col
                        _simplex_info.append(
                            f"  {_label}: [{min(_col):.4f} .. {max(_col):.4f}]")
                        _simplex[:, _col_idx] = np.roll(
                            _simplex[:, _col_idx], _col_idx)

                    _centroid = _simplex.mean(axis=0)
                    _model.set_param_vector(_centroid)

                # 5. Run MLE Optimization
                # Warm up emulator caches (Cholesky factorisation etc.)
                # before starting the optimizer timer, so the first eval
                # isn't artificially slow.
                _spinner.update("Building emulator cache (one-time)...")
                try:
                    _ = _model()
                except Exception:
                    pass

                _spinner.update(f"Running {_opt_method} optimisation...")

                _nll_history = []
                _iter_count = [0]
                _start_time = _time.time()
                _max_iter = int(mle_max_iter.value)
                _n_restarts = max(1, int(mle_restarts.value))

                _global_best_f = [float("inf")]  # mutable for callback
                _cur_restart = [0]

                def _nll_with_callback(P):
                    _model.set_param_vector(P)
                    # Fast bounds check — skip expensive GPU eval for out-of-range proposals
                    _gp = np.array(_model.grid_params)
                    if (np.any(_gp < _model.emulator.min_params) or
                            np.any(_gp > _model.emulator.max_params)):
                        nll = 1e30
                    else:
                        try:
                            nll = -_model.log_likelihood(_priors)
                        except (ValueError, np.linalg.LinAlgError):
                            nll = 1e30
                    _nll_history.append(nll)
                    _iter_count[0] += 1
                    if _iter_count[0] % 50 == 0:
                        _elapsed = _time.time() - _start_time
                        _spinner.update(
                            f"{_opt_method} Restart {_cur_restart[0]}/{_n_restarts} | "
                            f"Eval {_iter_count[0]} | "
                            f"Best NLL: {_global_best_f[0]:.10f} | "
                            f"Time: {_elapsed:.1f}s"
                        )
                    return nll

                _p0 = _model.get_param_vector()
                _lo_arr = np.array(_lo_bounds)
                _hi_arr = np.array(_hi_bounds)

                # Generate starting points: first = bootstrapped x0, rest = random
                _start_points = [_p0.copy()]
                if _n_restarts > 1:
                    np.random.seed(None)
                    for _ in range(_n_restarts - 1):
                        _rnd = _lo_arr + np.random.rand(_N) * (_hi_arr - _lo_arr)
                        _start_points.append(_rnd)

                _global_best_x = _p0.copy()
                _global_best_nit = 0
                _restart_summaries = []

                def _restart_param_label(_name):
                    return f"log10({_name})" if _name in log10_params else _name

                def _capture_restart_result(_restart_number, _solution):
                    _model.set_param_vector(_solution.x)
                    _values = {}
                    for _i, _p in enumerate(phys_names):
                        _values[_restart_param_label(_p)] = float(_model.grid_params[_i])
                    _params = _model.params
                    if 'Av' in _params:
                        _values['Av'] = float(_params['Av'])
                    if 'log_scale' in _params:
                        _values['Distance (pc)'] = float(_log_scale_to_distance_pc(_params['log_scale']))
                    elif hasattr(_model, '_log_scale') and _model._log_scale is not None:
                        _values['Distance (pc) (auto)'] = float(_log_scale_to_distance_pc(_model._log_scale))
                    if 'cheb:1' in _params:
                        _values['cheb₁'] = float(_params['cheb:1'])
                    if 'global_cov:log_amp' in _params:
                        _values['ln(GP amp)'] = float(_params['global_cov:log_amp'])
                    if 'global_cov:log_ls' in _params:
                        _values['ln(GP length)'] = float(_params['global_cov:log_ls'])

                    _restart_summaries.append({
                        "restart": int(_restart_number),
                        "nll": float(_solution.fun),
                        "values": _values,
                    })

                for _restart_idx, _x0 in enumerate(_start_points):
                    _cur_restart[0] = _restart_idx + 1
                    _iter_count[0] = 0  # reset eval counter per restart
                    _spinner.update(
                        f"{_opt_method} | Restart {_cur_restart[0]}/{_n_restarts} | "
                        f"Starting... | {_time.time() - _start_time:.1f}s"
                    )
                    _model.set_param_vector(_x0)

                    if _opt_method == "Nelder-Mead":
                        # For restart 0 use the pre-built simplex; for later
                        # restarts use adaptive simplex from the random x0.
                        _nm_opts = dict(maxiter=_max_iter, disp=False, adaptive=True)
                        if _restart_idx == 0 and _simplex is not None:
                            _nm_opts["initial_simplex"] = _simplex
                        _run_soln = scipy_minimize(
                            _nll_with_callback,
                            _x0,
                            method="Nelder-Mead",
                            options=_nm_opts,
                        )

                    elif _opt_method == "L-BFGS-B":
                        _bounds_list = list(zip(_lo_bounds, _hi_bounds))
                        _run_soln = scipy_minimize(
                            _nll_with_callback,
                            _x0,
                            method="L-BFGS-B",
                            bounds=_bounds_list,
                            options=dict(
                                maxiter=_max_iter,
                                ftol=1e-15,
                                gtol=1e-12,
                                eps=1e-5,
                            ),
                        )

                    elif _opt_method == "CMA-ES":
                        try:
                            import cma
                        except ImportError:
                            raise ImportError(
                                "CMA-ES requires the 'cma' package. "
                                "Install it with: pip install cma"
                            )
                        _cma_bounds = [_lo_bounds, _hi_bounds]
                        _p0_cma = np.clip(
                            _x0,
                            _lo_arr + 1e-8,
                            _hi_arr - 1e-8,
                        )
                        _cma_stds = [0.2 * (hi - lo) for lo, hi in zip(_lo_bounds, _hi_bounds)]
                        _popsize = 2 * (4 + int(3 * np.log(_N)))
                        _es = cma.CMAEvolutionStrategy(
                            _p0_cma.tolist(), 1.0,
                            {
                                "bounds": _cma_bounds,
                                "CMA_stds": _cma_stds,
                                "popsize": _popsize,
                                "maxfevals": _max_iter,
                                "verbose": -9,
                                "tolfun": 1e-10,
                            },
                        )
                        _run_best_x, _run_best_f = _x0.copy(), float("inf")
                        while not _es.stop():
                            _solutions = _es.ask()
                            _fits = [_nll_with_callback(np.array(s)) for s in _solutions]
                            _es.tell(_solutions, _fits)
                            _gen_best = min(_fits)
                            if _gen_best < _run_best_f:
                                _run_best_f = _gen_best
                                _run_best_x = np.array(_solutions[_fits.index(_gen_best)])
                            if _run_best_f < _global_best_f[0]:
                                _global_best_f[0] = _run_best_f
                        from types import SimpleNamespace
                        _run_soln = SimpleNamespace(
                            x=_run_best_x, fun=_run_best_f, success=True,
                            message="CMA-ES terminated",
                            nit=_es.result.iterations,
                        )

                    _capture_restart_result(_cur_restart[0], _run_soln)

                    # Keep global best across restarts
                    if _run_soln.fun <= _global_best_f[0]:
                        _global_best_f[0] = _run_soln.fun
                        _global_best_x = _run_soln.x.copy()
                        _global_best_nit = getattr(_run_soln, 'nit', 0)

                from types import SimpleNamespace
                _soln = SimpleNamespace(
                    x=_global_best_x, fun=_global_best_f[0], success=True,
                    message=f"Best of {_n_restarts} restart(s)",
                    nit=_global_best_nit,
                )

                if _soln.success:
                    _model.set_param_vector(_soln.x)

                # Store for MCMC
                set_mle_model(_model)
                set_mle_priors(_priors)

                _elapsed_total = _time.time() - _start_time
                _restart_msg = f" ({_n_restarts} restart{'s' if _n_restarts > 1 else ''})" if _n_restarts > 1 else ""
                fit_status = mo.callout(
                    mo.md(f"{mo.icon('lucide:check-circle')} MLE Complete via {_opt_method}{_restart_msg}! ({_iter_count[0]} iterations, {_elapsed_total:.1f}s)"),
                    kind="success"
                )

                # Resolve lookup-table truth labels once, then reuse them for both
                # the validation table and the optional ground-truth spectrum.
                from Speculate_addons.grid_registry import benchmark_param_map as _benchmark_param_map
                _truth_label_map = {
                    int(_idx): _label
                    for _idx, (_label, _sirocco_key) in _benchmark_param_map(
                        grid_selector.value if grid_selector is not None else None
                    ).items()
                }

                def _truth_key_for(_display_name, _internal_label):
                    if "Inclination" in _display_name:
                        return "Inclination"
                    _internal_text = str(_internal_label)
                    if _internal_text.startswith("param") and _internal_text[5:].isdigit():
                        return _truth_label_map.get(int(_internal_text[5:]), _display_name)
                    return _display_name

                def _format_nll(_value):
                    try:
                        _value = float(_value)
                    except Exception:
                        return "N/A"
                    if not np.isfinite(_value):
                        return "inf"
                    return f"{_value:.2f}"

                # 6. Plotting payload
                import matplotlib.pyplot as _plt
                _plot_flux, _plot_cov = _model()
                if hasattr(_plot_flux, 'detach'):
                    _plot_flux = _plot_flux.detach().cpu().numpy()
                if hasattr(_plot_cov, 'detach'):
                    _plot_cov = _plot_cov.detach().cpu().numpy()
                _plot_cov = np.asarray(_plot_cov, dtype=float)
                _plot_cov_diag = np.diag(_plot_cov) if _plot_cov.ndim == 2 else _plot_cov.reshape(-1)
                _plot_data_flux = np.asarray(_model.data.flux, dtype=float)
                _gp_covariance_diagnostics = global_covariance_diagnostics(
                    _model,
                    _priors,
                    np.asarray(_plot_flux, dtype=float) - _plot_data_flux,
                )
                _mle_nll = float(_soln.fun)
                _mle_plot_payload = {
                    "wavelength": np.asarray(_model.data.wave, dtype=float).copy(),
                    "data_flux": _plot_data_flux.copy(),
                    "model_flux": np.asarray(_plot_flux, dtype=float).copy(),
                    "model_cov_diag": _plot_cov_diag.copy(),
                    "title": f"Best-Fit Model — {_model.data_name}",
                    "zoom_name": "inference_mle_bestfit_zoom",
                    "model_label": f"MLE Best Fit (NLL={_format_nll(_mle_nll)})",
                    "model_nll": _mle_nll,
                    "ground_truth": None,
                    "ground_truth_error": None,
                }

                if ground_truth_params:
                    _truth_grid_values = []
                    _missing_truth_keys = []
                    for _i, _p in enumerate(phys_names):
                        _gt_key = _truth_key_for(_p, emu.param_names[_i])
                        if _gt_key in ground_truth_params:
                            _truth_grid_values.append(float(ground_truth_params[_gt_key]))
                        else:
                            _missing_truth_keys.append(_gt_key)

                    if not _missing_truth_keys:
                        try:
                            _truth_global_params = {}
                            if 'Av' in _model.params:
                                _truth_global_params['Av'] = float(_model.params['Av'])
                            if 'log_scale' in _model.params:
                                _truth_global_params['log_scale'] = float(_model.params['log_scale'])
                            if 'cheb' in _model.params or 'cheb:1' in _model.params:
                                _truth_global_params['cheb'] = [float(v) for v in _model.cheb]
                            if ('global_cov:log_amp' in _model.params
                                    and 'global_cov:log_ls' in _model.params):
                                _truth_global_params['global_cov'] = {
                                    'log_amp': float(_model.params['global_cov:log_amp']),
                                    'log_ls': float(_model.params['global_cov:log_ls']),
                                }

                            _truth_model = SpectrumModel(
                                emulator=emu,
                                data=spec_data,
                                grid_params=_truth_grid_values,
                                flux_scale=_flux_scale,
                                norm=True,
                                **_truth_global_params,
                            )
                            _truth_nll = -float(_truth_model.log_likelihood(_priors))
                            _truth_flux, _truth_cov = _truth_model()
                            if hasattr(_truth_flux, 'detach'):
                                _truth_flux = _truth_flux.detach().cpu().numpy()
                            if hasattr(_truth_cov, 'detach'):
                                _truth_cov = _truth_cov.detach().cpu().numpy()
                            _mle_plot_payload["ground_truth"] = {
                                "wavelength": np.asarray(_truth_model.data.wave, dtype=float).copy(),
                                "flux": np.asarray(_truth_flux, dtype=float).copy(),
                                "label": f"Ground Truth Emulator (NLL={_format_nll(_truth_nll)})",
                                "color": "#009E73",
                                "dash": [6, 3],
                                "nll": _truth_nll,
                            }
                        except Exception as _gt_exc:
                            _mle_plot_payload["ground_truth_error"] = str(_gt_exc)
                    else:
                        _mle_plot_payload["ground_truth_error"] = (
                            "Missing truth values for: " + ", ".join(_missing_truth_keys)
                        )

                # 6b. Loss curve plot
                if len(_nll_history) > 10:
                    _fig_loss, _ax_loss = _plt.subplots(figsize=(8, 3))
                    _ax_loss.plot(_nll_history, 'g-', alpha=0.7, linewidth=0.5)
                    # Rolling minimum
                    _rolling_min = np.minimum.accumulate(_nll_history)
                    _ax_loss.plot(_rolling_min, 'r-', linewidth=2, label='Best NLL')

                    # Smart Scaling: Ignore initial drop (first 15%) and scale to Best NLL range
                    _burn_in = max(5, int(len(_rolling_min) * 0.15))
                    if _burn_in < len(_rolling_min):
                         _view_data = _rolling_min[_burn_in:]
                         _ymin, _ymax = _view_data.min(), _view_data.max()
                         _yrange = _ymax - _ymin
                         if _yrange == 0: _yrange = abs(_ymin) * 0.1 or 1.0
                         _ax_loss.set_ylim(_ymin - 0.05*_yrange, _ymax + 0.05*_yrange)

                    _ax_loss.set_xlabel('Iteration')
                    _ax_loss.set_ylabel('Negative Log Likelihood')
                    _ax_loss.set_title('MLE Convergence')
                    _ax_loss.legend()
                    _ax_loss.grid(True, alpha=0.3)
                    _fig_loss.tight_layout()
                else:
                    _fig_loss = None

                # Present the fitted grid parameters in the same friendly order used
                # in Stage 2, with optional comparison against validation truth.
                res_grid = _model.grid_params
                res_global = _model.params

                _results_md = "### Best Fit Parameters\n\n"
                _results_md += "| Parameter | Fitted | "
                if ground_truth_params:
                    _results_md += "Ground Truth | Δ |\n"
                    _results_md += "|-----------|--------|--------------|---|\n"
                else:
                    _results_md += "\n|-----------|--------|\n"

                for _i, _p in enumerate(phys_names):
                    fitted_val = res_grid[_i]
                    _display_p = f"log10({_p})" if _p in log10_params else _p
                    _results_md += f"| **{_display_p}** | {fitted_val:.4f} | "

                    _gt_key = _truth_key_for(_p, emu.param_names[_i])

                    if ground_truth_params and _gt_key in ground_truth_params:
                        _gt_val = ground_truth_params[_gt_key]
                        delta = fitted_val - _gt_val
                        _results_md += f"{_gt_val:.4f} | {delta:+.4f} |\n"
                    elif ground_truth_params:
                        _results_md += "- | - |\n"
                    else:
                        _results_md += "\n"

                # Global parameters in a separate table for side-by-side layout
                _global_md = "### Global Parameters\n\n"
                _global_md += "| Parameter | Fitted |\n"
                _global_md += "|-----------|--------|\n"
                if 'Av' in res_global:
                    _global_md += f"| **Av** | {res_global['Av']:.4f} |\n"
                if 'log_scale' in res_global:
                    from Speculate_addons.distance_scale import log_scale_to_distance_pc as _log_scale_to_distance_pc
                    _distance_pc = _log_scale_to_distance_pc(res_global['log_scale'])
                    _global_md += f"| **Distance (pc)** | {_distance_pc:.4f} |\n"
                elif hasattr(_model, '_log_scale') and _model._log_scale is not None:
                    from Speculate_addons.distance_scale import log_scale_to_distance_pc as _log_scale_to_distance_pc
                    _distance_pc = _log_scale_to_distance_pc(_model._log_scale)
                    _global_md += f"| **Distance (pc) (auto)** | {_distance_pc:.4f} |\n"
                if 'cheb:1' in res_global:
                    _global_md += f"| **cheb₁** | {res_global['cheb:1']:.4f} |\n"
                if 'global_cov:log_amp' in res_global:
                    _global_md += f"| **ln(GP amp)** | {res_global['global_cov:log_amp']:.4f} |\n"
                if 'global_cov:log_ls' in res_global:
                    _global_md += f"| **ln(GP length)** | {res_global['global_cov:log_ls']:.4f} |\n"

                _playground_distance_pc = 100.0
                if 'log_scale' in res_global:
                    _playground_distance_pc = float(_log_scale_to_distance_pc(res_global['log_scale']))
                elif hasattr(_model, '_log_scale') and _model._log_scale is not None:
                    _playground_distance_pc = float(_log_scale_to_distance_pc(_model._log_scale))

                _playground_payload = {
                    "source": "inference_mle",
                    "grid_name": grid_selector.value if grid_selector is not None else None,
                    "phys": [float(_v) for _v in res_grid],
                    "Av": float(res_global.get('Av', 0.0)),
                    "Distance (pc)": _playground_distance_pc,
                    "cheb_1": float(res_global.get('cheb:1', 0.0)),
                    "log_amp": float(res_global.get('global_cov:log_amp', -60.0)),
                    "log_ls": float(res_global.get('global_cov:log_ls', 4.5)),
                }

                _restart_table_rows = None
                if _restart_summaries:
                    def _format_restart_value(_value):
                        try:
                            _value = float(_value)
                        except Exception:
                            return "-"
                        if not np.isfinite(_value):
                            return "inf"
                        return f"{_value:.4f}"

                    _restart_columns = [
                        f"Restart {_summary['restart']}"
                        for _summary in _restart_summaries
                    ]
                    _restart_param_order = []
                    for _summary in _restart_summaries:
                        for _param_name in _summary["values"]:
                            if _param_name not in _restart_param_order:
                                _restart_param_order.append(_param_name)

                    _nll_row = {"Parameter": "NLL"}
                    for _column, _summary in zip(_restart_columns, _restart_summaries):
                        _nll_row[_column] = _format_nll(_summary["nll"])
                    _restart_table_rows = [_nll_row]

                    for _param_name in _restart_param_order:
                        _row = {"Parameter": _param_name}
                        for _column, _summary in zip(_restart_columns, _restart_summaries):
                            _row[_column] = _format_restart_value(
                                _summary["values"].get(_param_name)
                            )
                        _restart_table_rows.append(_row)

                set_mle_result({
                    "fit_status": fit_status,
                    "results_md": _results_md,
                    "global_md": _global_md,
                    "playground_payload": _playground_payload,
                    "gp_covariance_diagnostics": _gp_covariance_diagnostics,
                    "restart_table_rows": _restart_table_rows,
                    "loss_fig": _fig_loss,
                    "plot": _mle_plot_payload,
                })

            except Exception as _e:
                import traceback as _traceback
                _tb = _traceback.format_exc()
                fit_status = mo.callout(mo.md(f"{mo.icon('lucide:x-circle')} MLE Failed: {str(_e)}\n\n```\n{_tb}\n```"), kind="danger")
                fit_results = fit_status
                set_mle_model(None)
                set_mle_priors(None)
                set_mle_result({"error": True, "fit_status": fit_status})

    return


@app.cell
def _(
    alt,
    build_bestfit_spectrum_altair,
    get_mle_result,
    mle_show_ground_truth_spectrum,
    mo,
    set_inf_playground_target,
):
    # Render the latest MLE result from stored arrays.  This keeps the
    # validation overlay checkbox reactive without rerunning the optimizer.
    _result = get_mle_result()
    _mle_result_display = None
    if _result is None:
        _mle_result_display = mo.md("*Click 'Run MLE' to start inference*")
    elif _result.get("error"):
        _mle_result_display = _result["fit_status"]
    else:
        _plot = _result["plot"]
        _gt_payload = _plot.get("ground_truth")
        _show_gt = bool(_gt_payload) and bool(mle_show_ground_truth_spectrum.value)
        _extra_series = [_gt_payload] if _show_gt else None

        _fig = build_bestfit_spectrum_altair(
            alt,
            wavelength=_plot["wavelength"],
            data_flux=_plot["data_flux"],
            model_flux=_plot["model_flux"],
            model_cov_diag=_plot["model_cov_diag"],
            title=_plot["title"],
            zoom_name=_plot["zoom_name"],
            model_label=_plot["model_label"],
            extra_flux_series=_extra_series,
        )

        def _send_mle_to_playground(_):
            set_inf_playground_target(_result["playground_payload"])
            mo.status.toast("Loaded MLE best fit into the Parameter Playground")

        _playground_button = mo.ui.button(
            label=f"{mo.icon('lucide:sliders-horizontal')} Export to Parameter Playground",
            on_click=_send_mle_to_playground,
            kind="success",
        )

        _result_elements = [
            _result["fit_status"],
            _playground_button,
            mo.hstack([
                mo.md(_result["results_md"]),
                mo.md(_result["global_md"]),
            ], align="start", gap="2rem"),
        ]
        _gp_diag = _result.get("gp_covariance_diagnostics") or {}
        if _gp_diag.get("warning"):
            _gp_msg = (
                "The fitted GP covariance may be absorbing model-data mismatch. "
                f"log_amp={_gp_diag.get('log_amp', float('nan')):.3g}, "
                f"log_amp - ln(residual variance)="
                f"{_gp_diag.get('log_amp_minus_residual_log_variance', float('nan')):.3g}."
            )
            if _gp_diag.get("log_ls_at_upper_bound"):
                _gp_msg += " log_ls is at the upper prior bound."
            _result_elements.append(mo.callout(mo.md(_gp_msg), kind="warn"))
        if _result.get("restart_table_rows"):
            _restart_rows = _result["restart_table_rows"]
            _restart_columns = list(_restart_rows[0].keys()) if _restart_rows else []
            _restart_align = {
                _column: "left" if _column == "Parameter" else "right"
                for _column in _restart_columns
            }
            _result_elements.append(
                mo.accordion({
                    f"{mo.icon('lucide:rotate-cw')} MLE Restart Best Fits": mo.ui.table(
                        _restart_rows,
                        selection=None,
                        pagination=False,
                        show_column_summaries=False,
                        show_data_types=False,
                        show_download=False,
                        freeze_columns_left=["Parameter"],
                        text_justify_columns=_restart_align,
                        max_columns=None,
                        page_size=len(_restart_rows),
                    )
                })
            )
        if _result.get("loss_fig") is not None:
            _result_elements.extend([
                mo.md("### Convergence"),
                _result["loss_fig"],
            ])
        _result_elements.extend([
            mo.md("### Model Fit & Residuals"),
            _fig,
        ])
        if _gt_payload:
            _result_elements.append(mle_show_ground_truth_spectrum)
        elif _plot.get("ground_truth_error"):
            _result_elements.append(
                mo.callout(
                    mo.md(f"Ground-truth emulator overlay unavailable: {_plot['ground_truth_error']}"),
                    kind="warn",
                )
            )

        _mle_result_display = mo.vstack(_result_elements)

    _mle_result_display
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 4: MCMC Sampling (Posterior Exploration)

    After MLE finds the best-fit point, MCMC explores the posterior distribution to quantify parameter uncertainties.
    """)
    return


@app.cell
def _(emu, get_mle_model, mo, param_names, re):
    # ── Stage 4 configuration: MCMC sampling controls ──
    # The user sets walker count, step count, and manual burn-in, then
    # optionally freezes select parameters at their MLE values so the
    # sampler explores a reduced subspace.  Defaults were chosen for a
    # reasonable balance of speed vs. posterior quality:
    #   64 walkers  – twice the typical active-parameter count
    #   2500 steps  – gives ~30 000 post-burn samples at thin=1
    #   500 burn-in – conservative; the auto-burn heuristic may override
    _model = get_mle_model()

    mcmc_nwalkers = mo.ui.number(value=64, label="Walkers", step=4, start=8)
    mcmc_nsteps = mo.ui.number(value=2500, label="Steps", step=100, start=100)
    mcmc_burnin = mo.ui.number(value=500, label="Min Burn-in (floor)", step=50, start=0)

    run_mcmc_btn = mo.ui.run_button(
        label=f"{mo.icon('lucide:shuffle')} Run MCMC", 
        kind="warn",
    )

    # Build internal→friendly label map
    # Model uses "param1", "param2", etc. internally; we map to "disk.mdot", etc.
    mcmc_label_map = {}
    if _model is not None and emu is not None:
        _num_grid = len(emu.param_names)
        _phys = param_names[:_num_grid]  # Friendly names from Stage 2
        for _internal, _friendly in zip(emu.param_names, _phys):
            mcmc_label_map[_internal] = _friendly
        # Map global/nuisance params to friendly names
        mcmc_label_map['Av'] = 'Av'
        mcmc_label_map['log_scale'] = 'Distance (pc)'
        mcmc_label_map['cheb:1'] = 'cheb_1'
        mcmc_label_map['global_cov:log_amp'] = 'GP log_amp'
        mcmc_label_map['global_cov:log_ls'] = 'GP log_ls'

    # Build freeze checkboxes for each active (thawed) parameter
    _freeze_dict = {}
    if _model is not None:
        for _label in _model.labels:
            _display = mcmc_label_map.get(_label, _label)
            _freeze_dict[_label] = mo.ui.checkbox(value=False, label=f"{_display}")

    mcmc_freeze = mo.ui.dictionary(_freeze_dict) if _freeze_dict else mo.ui.dictionary({})

    if _model is None:
        mcmc_config_ui = mo.callout(mo.md(f"{mo.icon('lucide:triangle-alert')} Run MLE first before MCMC"), kind="warn")
    else:
        # Show friendly names in the active params list
        _friendly_labels = [mcmc_label_map.get(l, l) for l in _model.labels]

        mcmc_config_ui = mo.vstack([
            mo.md(f"**Active Parameters:** {', '.join(_friendly_labels)}"),
            mo.hstack([mcmc_nwalkers, mcmc_nsteps, mcmc_burnin], justify="start"),
            mo.md(f"### {mo.icon('lucide:lock')} Freeze Parameters for MCMC"),
            mo.callout(mo.md(
                "Tick parameters to **freeze** them at their MLE best-fit values. "
                "Frozen parameters will not be sampled by MCMC. "
                "This is useful for nuisance/hyperparameters (e.g. GP amplitude, length scale)."
            ), kind="neutral"),
            mcmc_freeze,
            run_mcmc_btn
        ])

    mcmc_config_ui
    return mcmc_burnin, mcmc_freeze, mcmc_label_map, mcmc_nsteps, mcmc_nwalkers, run_mcmc_btn


@app.cell
def _(
    alt,
    build_bestfit_spectrum_altair,
    get_mle_model,
    get_mle_priors,
    ground_truth_params,
    grid_selector,
    mcmc_burnin,
    mcmc_freeze,
    mcmc_label_map,
    mcmc_nsteps,
    mcmc_nwalkers,
    mo,
    np,
    run_mcmc_btn,
    set_mcmc_labels,
    set_mcmc_corner_meta,
    set_mcmc_samples,
    set_mcmc_summary_df,
    set_inf_playground_target,
    stats,
):
    # ── Stage 4: MCMC Posterior Exploration ──
    # Steps:
    #  1. Optionally freeze parameters the user flagged (GP hypers, etc.).
    #  2. Build a Gaussian proposal ball around the MLE best-fit, scaled
    #     per-parameter to match typical posterior widths.
    #  3. Run emcee’s EnsembleSampler with 1-step increments inside a
    #     marimo progress bar.
    #  4. Estimate autocorrelation times; use max(τ) as the automatic
    #     burn-in and 0.3×min(τ) as thinning (ArviZ/emcee heuristic).
    #  5. Convert chains to ArviZ InferenceData for trace plots, posterior
    #     distributions, and summary statistics.
    #  6. Generate a corner plot (truths drawn from the lookup table when
    #     running in test-grid validation mode).
    #  7. Set the model to the posterior mean and plot the best-fit.
    #  8. Persist the cleaned samples and labels in reactive state for the
    #     export cells.
    mcmc_results = None

    if run_mcmc_btn.value:
        set_mcmc_samples(None)
        set_mcmc_labels(None)
        set_mcmc_summary_df(None)
        set_mcmc_corner_meta(None)

        _model = get_mle_model()
        _priors = get_mle_priors()

        if _model is None or _priors is None:
            mo.stop(True, mo.md("Please run MLE first."))

        # Apply the optional MCMC-only freezes to the already-fit model so the
        # sampler explores only the subset of parameters the user left thawed.
        # First, thaw every param that the MCMC checkboxes cover.  This
        # prevents freezes from a previous MCMC run accumulating when the
        # user re-runs with a different freeze selection.
        _freeze_values = mcmc_freeze.value  # {internal_name: bool}
        for _key in _freeze_values:
            _model.thaw(_key)
        _frozen_list = []
        for _key, _is_frozen in _freeze_values.items():
            if _is_frozen:
                _model.freeze(_key)
                _frozen_list.append(mcmc_label_map.get(_key, _key))

        # Helper to get friendly name from internal label
        def _friendly(internal_label):
            return mcmc_label_map.get(internal_label, internal_label)

        with mo.status.spinner("Running MCMC..."):
            try:
                import emcee as _emcee
                import arviz as _az
                import corner as _corner
                import matplotlib.pyplot as _plt
                import matplotlib
                import warnings as _warnings
                from Speculate_addons.distance_scale import log_scale_to_distance_pc as _log_scale_to_distance_pc
                matplotlib.rcParams['figure.max_open_warning'] = 50
                _warnings.filterwarnings("ignore", module="arviz", message="More chains")

                _nwalkers = mcmc_nwalkers.value
                _nsteps = mcmc_nsteps.value
                _burnin_manual = mcmc_burnin.value
                _ndim = len(_model.labels)

                # Define log probability function
                def _log_prob(P, model, priors):
                    model.set_param_vector(P)
                    # Reject proposals outside the emulator training grid.
                    # Without this, walkers that wander beyond the grid get
                    # garbage likelihood values instead of being rejected,
                    # causing them to diffuse across the full prior volume.
                    gp = np.array(model.grid_params)
                    if np.any(gp < model.emulator.min_params) or np.any(gp > model.emulator.max_params):
                        return -np.inf
                    try:
                        return model.log_likelihood(priors)
                    except (ValueError, np.linalg.LinAlgError):
                        return -np.inf

                # Initialise walkers in a truncated-normal ball around the MLE.
                # For Normal priors: σ = the distribution's own std (1σ).
                # For Uniform / other priors: σ = _INIT_FRAC × prior width.
                # Truncated normal avoids edge pile-up at the prior edges
                # when the MLE sits near a bound.
                _INIT_FRAC = 0.15  # σ as a fraction of prior width
                _ball = np.empty((_nwalkers, _ndim))
                for _i, _key in enumerate(_model.labels):
                    _pr = _priors.get(_key)
                    _mle_val = _model[_key]
                    if _pr is not None and hasattr(_pr, 'interval'):
                        _lo, _hi = _pr.interval(1.0)
                        # Use the distribution's native σ only for Normal-family
                        # priors; for Uniform (and anything else) use a controlled
                        # fraction of the prior width.
                        _is_normal = getattr(getattr(_pr, 'dist', None), 'name', '') in ('norm', 'truncnorm')
                        if _is_normal:
                            _sigma = _pr.std()          # Normal prior
                        else:
                            _sigma = _INIT_FRAC * (_hi - _lo)  # Uniform prior
                        _a = (_lo - _mle_val) / _sigma  # lower bound in std units
                        _b = (_hi - _mle_val) / _sigma  # upper bound in std units
                        _ball[:, _i] = stats.truncnorm.rvs(
                            _a, _b, loc=_mle_val, scale=_sigma, size=_nwalkers
                        )
                    else:
                        # No prior registered (shouldn't happen for thawed
                        # params) — fall back to small absolute jitter.
                        _ball[:, _i] = _mle_val + 0.1 * np.random.randn(_nwalkers)

                # Create sampler — use DEMove + DESnookerMove for better
                # performance in ≥5D (Ter Braak 2006; Nelson et al. 2014).
                # The default StretchMove becomes increasingly inefficient
                # above ~5 dimensions.
                _moves = [
                    (_emcee.moves.DEMove(), 0.8),
                    (_emcee.moves.DESnookerMove(), 0.2),
                ]
                _sampler = _emcee.EnsembleSampler(
                    _nwalkers, _ndim, _log_prob, args=(_model, _priors),
                    moves=_moves,
                )

                # Run with progress bar
                for _ in mo.status.progress_bar(range(_nsteps), title="MCMC Sampling"):
                    _sampler.run_mcmc(_ball if _sampler.iteration == 0 else None, 1, progress=False)

                # ============================================================
                # Stage 15: Raw MCMC Chains (Full, pre burn-in)
                # ============================================================
                # _sampler.get_chain() shape: (nsteps, nwalkers, ndim)
                _full_chain = _sampler.get_chain()

                def _display_chain_values(_internal_label, _values):
                    if _internal_label == 'log_scale':
                        return _log_scale_to_distance_pc(_values)
                    return _values

                def _display_samples(_samples):
                    _display = _samples.copy()
                    for _idx, _label in enumerate(_model.labels):
                        if _label == 'log_scale':
                            _display[:, _idx] = _log_scale_to_distance_pc(_display[:, _idx])
                    return _display

                # Convert the raw emcee chain into ArviZ's (chains, draws) layout
                # using friendly parameter names for all user-facing plots.
                _friendly_labels = [_friendly(l) for l in _model.labels]
                _full_dd = {}
                for _i, _internal_label in enumerate(_model.labels):
                    _full_dd[_friendly(_internal_label)] = _display_chain_values(
                        _internal_label, _full_chain[:, :, _i].T
                    )  # (walkers, steps)
                _full_data = _az.from_dict(posterior=_full_dd)

                # Plot full chains (trace plot)
                _fig_full_trace = _az.plot_trace(_full_data)
                _fig_full_chain = _fig_full_trace.ravel()[0].figure
                _fig_full_chain.suptitle("Raw MCMC Chains (Full)", y=1.02)
                _fig_full_chain.tight_layout()

                # ============================================================
                # Stage 16: Autocorrelation & Burn-in
                # ============================================================
                # Autocorrelation estimates drive the automatic burn-in and thinning
                # heuristics, but the code falls back gracefully when chains are too
                # short or unstable for ArviZ/emcee to estimate tau reliably.
                try:
                    _tau = _sampler.get_autocorr_time(tol=0)
                    _tau_valid = not (np.isnan(_tau).any() or (_tau == 0).any())
                except Exception:
                    _tau = np.full(_ndim, np.nan)
                    _tau_valid = False

                if _tau_valid:
                    _auto_burnin = int(2 * _tau.max())  # 2×τ (Foreman-Mackey 2013)
                    _auto_thin = max(1, int(0.3 * np.min(_tau)))
                    _burnin_used = max(_burnin_manual, _auto_burnin)
                else:
                    _auto_burnin = 0
                    _auto_thin = 1
                    _burnin_used = _burnin_manual

                _thin_used = _auto_thin

                # Ensure we don't discard everything
                if _burnin_used >= _nsteps:
                    _burnin_used = max(0, _nsteps // 2)

                _burn_chain = _sampler.get_chain(discard=_burnin_used, thin=_thin_used)
                _burn_samples = _burn_chain.reshape((-1, _ndim))
                _burn_samples_display = _display_samples(_burn_samples)

                _burnin_info = (
                    f"- **Manual burn-in requested:** {_burnin_manual}\n"
                    f"- **Autocorrelation burn-in:** {_auto_burnin if _tau_valid else 'N/A (unconverged)'}\n"
                    f"- **Burn-in used:** {_burnin_used}\n"
                    f"- **Thinning:** {_thin_used}\n"
                    f"- **Effective samples:** {_burn_samples.shape[0]} "
                    f"({_burn_chain.shape[0]} steps × {_burn_chain.shape[1]} walkers)"
                )
                if _tau_valid:
                    _tau_strs = [f"  - {_friendly(_model.labels[_j])}: {_tau[_j]:.1f}" for _j in range(_ndim)]
                    _burnin_info += "\n- **Autocorrelation times:**\n" + "\n".join(_tau_strs)

                # ============================================================
                # Stage 17: Burnt chain trace, summary & posteriors
                # ============================================================
                _burn_dd = {}
                for _i, _internal_label in enumerate(_model.labels):
                    _burn_dd[_friendly(_internal_label)] = _display_chain_values(
                        _internal_label, _burn_chain[:, :, _i].T
                    )  # (walkers, steps_after_burn)
                _burn_data = _az.from_dict(posterior=_burn_dd)

                # Trace plot (post burn-in)
                _fig_burn_trace = _az.plot_trace(_burn_data)
                _fig_burn_chain = _fig_burn_trace.ravel()[0].figure
                _fig_burn_chain.suptitle("MCMC Chains (Post Burn-in)", y=1.02)
                _fig_burn_chain.tight_layout()

                # Summary table
                _summary_df = _az.summary(_burn_data, round_to=5)
                _summary_md = _summary_df.to_markdown()

                # Persist the post-burn samples and summary table so the export cells
                # can write files without rerunning the sampler.
                set_mcmc_samples(_burn_samples_display.copy())
                set_mcmc_labels(list(_friendly_labels))
                set_mcmc_summary_df(_summary_df)

                # Posterior distributions
                _fig_posterior = _az.plot_posterior(
                    _burn_data,
                    var_names=_friendly_labels
                )
                _fig_post = _fig_posterior.ravel()[0].figure
                _fig_post.suptitle("Posterior Distributions", y=1.02)
                _fig_post.tight_layout()

                # ============================================================
                # Stage 18: Corner plot
                # ============================================================
                _corner_quantiles = [0.16, 0.5, 0.84]
                _corner_levels = [0.6827, 0.9545, 0.9973]

                # Corner-plot truths are matched through the same friendly-name map
                # used elsewhere, with all inclination variants collapsed onto the
                # single lookup-table "Inclination" key.
                from Speculate_addons.grid_registry import benchmark_param_map as _benchmark_param_map
                _truth_label_map = {
                    int(_idx): _label
                    for _idx, (_label, _sirocco_key) in _benchmark_param_map(
                        grid_selector.value if grid_selector is not None else None
                    ).items()
                }

                def _truth_key_for(_display_name, _internal_label):
                    if "Inclination" in _display_name:
                        return "Inclination"
                    _internal_text = str(_internal_label)
                    if _internal_text.startswith("param") and _internal_text[5:].isdigit():
                        return _truth_label_map.get(int(_internal_text[5:]), _display_name)
                    return _display_name

                _truths = None
                if ground_truth_params:
                    _truths = []
                    _has_any = False
                    for _label in _model.labels:
                        _gt_key = _truth_key_for(_friendly(_label), _label)
                        if _gt_key in ground_truth_params:
                            _truths.append(ground_truth_params[_gt_key])
                            _has_any = True
                        else:
                            _truths.append(None)
                    if not _has_any:
                        _truths = None

                _fig_corner = _corner.corner(
                    _burn_samples_display,
                    labels=_friendly_labels,
                    show_titles=True,
                    quantiles=_corner_quantiles,
                    levels=_corner_levels,
                    title_fmt=".4f",
                    truths=_truths,
                    truth_color="#ff4444",
                    truth_kwargs={"linewidth": 2},
                )
                for _ax in _fig_corner.axes:
                    _ax.grid(False)

                # Prior-range corner plot: axes span the full prior support so
                # the user can judge posterior breadth relative to the prior.
                _prior_ranges = []
                for _label in _model.labels:
                    _pr = _priors.get(_label)
                    if _pr is not None and hasattr(_pr, 'interval'):
                        if getattr(getattr(_pr, 'dist', None), 'name', None) == 'norm':
                            _lo = float(_pr.mean() - 4.0 * _pr.std())
                            _hi = float(_pr.mean() + 4.0 * _pr.std())
                        else:
                            _lo, _hi = _pr.interval(1.0)
                        if _label == 'log_scale':
                            _d0 = _log_scale_to_distance_pc(_lo)
                            _d1 = _log_scale_to_distance_pc(_hi)
                            _prior_ranges.append((min(_d0, _d1), max(_d0, _d1)))
                        else:
                            _prior_ranges.append((_lo, _hi))
                    else:
                        _prior_ranges.append(None)
                # Only build the second plot if we resolved at least one range
                if any(_r is not None for _r in _prior_ranges):
                    # Replace None entries with auto-range sentinel (corner uses 0.999 quantile)
                    _safe_ranges = [_r if _r is not None else (1.0,) for _r in _prior_ranges]
                    _fig_corner_prior = _corner.corner(
                        _burn_samples_display,
                        labels=_friendly_labels,
                        show_titles=True,
                        quantiles=_corner_quantiles,
                        levels=_corner_levels,
                        title_fmt=".4f",
                        truths=_truths,
                        truth_color="#ff4444",
                        truth_kwargs={"linewidth": 2},
                        range=_safe_ranges,
                    )
                    for _ax in _fig_corner_prior.axes:
                        _ax.grid(False)
                    _corner_display = mo.ui.tabs({
                        "Posterior (Auto Range)": _fig_corner,
                        "Full Prior Range": _fig_corner_prior,
                    })
                else:
                    _corner_display = _fig_corner

                set_mcmc_corner_meta({
                    "source": "inference",
                    "data_name": getattr(_model, "data_name", None),
                    "internal_labels": list(_model.labels),
                    "truths": list(_truths) if _truths is not None else None,
                    "ground_truth": dict(ground_truth_params or {}),
                    "prior_ranges": [
                        list(_r) if _r is not None else None
                        for _r in _prior_ranges
                    ],
                    "plot_variants": [
                        "posterior_auto_range",
                        "full_prior_range",
                    ] if any(_r is not None for _r in _prior_ranges) else [
                        "posterior_auto_range"
                    ],
                    "corner_settings": {
                        "show_titles": True,
                        "quantiles": list(_corner_quantiles),
                        "levels": list(_corner_levels),
                        "title_fmt": ".4f",
                    },
                    "mcmc": {
                        "nsteps": int(_nsteps),
                        "nwalkers": int(_nwalkers),
                        "manual_burnin": int(_burnin_manual),
                        "auto_burnin": int(_auto_burnin),
                        "burnin_used": int(_burnin_used),
                        "thin_used": int(_thin_used),
                        "effective_samples": int(_burn_samples.shape[0]),
                    },
                })

                # ============================================================
                # Stage 19: Best-fit MCMC model plot
                # ============================================================
                # Set model to posterior means (using internal labels)
                _mcmc_means = {}
                for _i, _label in enumerate(_model.labels):
                    _mcmc_means[_label] = float(np.mean(_burn_samples[:, _i]))
                _model.set_param_dict(_mcmc_means)

                _mcmc_global = _model.params
                _mcmc_distance_pc = 100.0
                if 'log_scale' in _mcmc_global:
                    _mcmc_distance_pc = float(_log_scale_to_distance_pc(_mcmc_global['log_scale']))
                elif hasattr(_model, '_log_scale') and _model._log_scale is not None:
                    _mcmc_distance_pc = float(_log_scale_to_distance_pc(_model._log_scale))
                _mcmc_playground_payload = {
                    "source": "inference_mcmc",
                    "grid_name": grid_selector.value if grid_selector is not None else None,
                    "phys": [float(_v) for _v in _model.grid_params],
                    "Av": float(_mcmc_global.get('Av', 0.0)),
                    "Distance (pc)": _mcmc_distance_pc,
                    "cheb_1": float(_mcmc_global.get('cheb:1', 0.0)),
                    "log_amp": float(_mcmc_global.get('global_cov:log_amp', -60.0)),
                    "log_ls": float(_mcmc_global.get('global_cov:log_ls', 4.5)),
                }

                _plot_flux, _plot_cov = _model()
                if hasattr(_plot_flux, 'detach'):
                    _plot_flux = _plot_flux.detach().cpu().numpy()
                if hasattr(_plot_cov, 'detach'):
                    _plot_cov = _plot_cov.detach().cpu().numpy()
                _fig_bestfit = build_bestfit_spectrum_altair(
                    alt,
                    wavelength=_model.data.wave,
                    data_flux=_model.data.flux,
                    model_flux=_plot_flux,
                    model_cov_diag=_plot_cov,
                    title=f"Best-Fit Model (MCMC Posterior Mean) — {_model.data_name}",
                    zoom_name="inference_mcmc_bestfit_zoom",
                )

                # ============================================================
                # Results table with ground truth comparison
                # ============================================================
                _results_md = "### MCMC Parameter Estimates\n\n"
                if _frozen_list:
                    _results_md += f"**Frozen (not sampled):** {', '.join(_frozen_list)}\n\n"
                _results_md += "| Parameter | Mean | Std | Median | "
                if ground_truth_params:
                    _results_md += "Truth | Δσ |\n"
                    _results_md += "|-----------|------|-----|--------|-------|----|\n"
                else:
                    _results_md += "\n|-----------|------|-----|--------|\n"

                for _i, _label in enumerate(_model.labels):
                    _display_name = _friendly(_label)
                    _display_vals = _display_chain_values(
                        _label, _burn_samples[:, _i]
                    )
                    _mean = np.mean(_display_vals)
                    _std = np.std(_display_vals)
                    _median = np.median(_display_vals)

                    _results_md += f"| **{_display_name}** | {_mean:.4f} | {_std:.4f} | {_median:.4f} | "

                    if ground_truth_params:
                        _gt_key = _truth_key_for(_display_name, _label)
                        if _gt_key in ground_truth_params:
                            _gt_val = ground_truth_params[_gt_key]
                            _delta_sigma = (_mean - _gt_val) / _std if _std > 0 else 0
                            _results_md += f"{_gt_val:.4f} | {_delta_sigma:+.2f}σ |\n"
                        else:
                            _results_md += "- | - |\n"
                    else:
                        _results_md += "\n"

                # ============================================================
                # Assemble output with accordions
                # ============================================================
                _frozen_info = f", frozen: {', '.join(_frozen_list)}" if _frozen_list else ""
                _status_callout = mo.callout(
                    mo.md(
                        f"{mo.icon('lucide:check-circle')} MCMC Complete! ({_nsteps} steps, "
                        f"{_nwalkers} walkers, burn-in {_burnin_used}, thin {_thin_used}"
                        f"{_frozen_info})"
                    ),
                    kind="success"
                )

                def _send_mcmc_to_playground(_):
                    set_inf_playground_target(_mcmc_playground_payload)
                    mo.status.toast("Loaded MCMC posterior mean into the Parameter Playground")

                _playground_button = mo.ui.button(
                    label=f"{mo.icon('lucide:sliders-horizontal')} Export to Parameter Playground",
                    on_click=_send_mcmc_to_playground,
                    kind="success",
                )

                _chain_accordion = mo.accordion({
                    f"{mo.icon('lucide:link')} Raw MCMC Chains (Full)": mo.vstack([
                        mo.md("Walker traces for all parameters before burn-in removal."),
                        _fig_full_chain
                    ]),
                    f"{mo.icon('lucide:flame')} Burnt MCMC Chains (Post Burn-in)": mo.vstack([
                        mo.md(_burnin_info),
                        _fig_burn_chain
                    ]),
                    f"{mo.icon('lucide:chart-bar')} Posterior Distributions": mo.vstack([
                        _fig_post
                    ]),
                    f"{mo.icon('lucide:clipboard-list')} Arviz Summary Statistics": mo.vstack([
                        mo.md(_summary_md)
                    ]),
                })

                mcmc_results = mo.vstack([
                    _status_callout,
                    _playground_button,
                    mo.md(_results_md),
                    _chain_accordion,
                    mo.md("### Corner Plot"),
                    _corner_display,
                    mo.md("### Best-Fit Model (MCMC Posterior Mean)"),
                    _fig_bestfit,
                ])

            except Exception as _e:
                import traceback as _traceback
                _tb = _traceback.format_exc()
                mcmc_results = mo.callout(mo.md(f"{mo.icon('lucide:x-circle')} MCMC Failed: {str(_e)}\n\n```\n{_tb}\n```"), kind="danger")

    mcmc_results if mcmc_results else mo.md("*Configure and run MCMC after MLE*")
    return


# =====================================================================
# Export .pf / Posterior CSV
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Export Results

    Export the MCMC posterior to a **Sirocco `.pf` template** and/or a **CSV file**
    of posterior samples for downstream analysis.
    """)
    return


@app.cell(hide_code=True)
def _(
    emu,
    get_mcmc_labels,
    get_mcmc_corner_meta,
    get_mcmc_samples,
    get_mcmc_summary_df,
    get_mle_model,
    mo,
    np,
    os,
):
    _samples = get_mcmc_samples()
    _labels = get_mcmc_labels()
    _summary_df = get_mcmc_summary_df()
    _model = get_mle_model()

    # Widgets must be defined unconditionally so the downstream export cell never
    # receives an undefined reference — mo.stop() would prevent their creation if
    # they were placed after it.
    export_pf_btn = mo.ui.run_button(label=f"{mo.icon('lucide:file-text')} Export .pf Template", kind="success")
    export_csv_btn = mo.ui.run_button(label=f"{mo.icon('lucide:chart-bar')} Export Posterior CSV", kind="success")
    export_corner_btn = mo.ui.run_button(label=f"{mo.icon('lucide:download')} Export Cornerplot Data", kind="success")
    export_dir_input = mo.ui.text(value="exports", label="Output directory")

    # Gate the display (not the widget creation) behind a completed MCMC run.
    mo.stop(
        _samples is None or _model is None,
        mo.callout(mo.md("Run MCMC first to enable export."), kind="neutral"),
    )

    mo.vstack([
        mo.hstack([export_dir_input], gap=1),
        mo.hstack([export_pf_btn, export_csv_btn, export_corner_btn], gap=1),
    ])
    return export_corner_btn, export_csv_btn, export_dir_input, export_pf_btn


@app.cell
def _(
    emu,
    export_corner_btn,
    export_csv_btn,
    export_dir_input,
    export_pf_btn,
    get_mcmc_corner_meta,
    get_mcmc_labels,
    get_mcmc_samples,
    get_mcmc_summary_df,
    get_mle_model,
    grid_selector,
    mo,
    np,
    os,
):
    import time as _time
    _samples = get_mcmc_samples()
    _labels = get_mcmc_labels()
    _summary_df = get_mcmc_summary_df()
    _corner_meta = get_mcmc_corner_meta()
    _model = get_mle_model()
    _out_dir = export_dir_input.value or "exports"
    _ts = _time.strftime("%Y%m%d_%H%M%S")

    _msg = ""

    def _summary_to_dict(_df):
        if _df is None:
            return None
        _summary = {}
        for _idx_name in _df.index:
            _row = _df.loc[_idx_name]
            _summary[str(_idx_name)] = {
                k: float(v) for k, v in _row.items() if np.isfinite(v)
            }
        return _summary

    _summary_dict = _summary_to_dict(_summary_df)

    if export_pf_btn.value and _model is not None and _samples is not None:
        try:
            from Speculate_addons.speculate_benchmark import export_pf_template, emulator_to_physical

            os.makedirs(_out_dir, exist_ok=True)

            # The .pf export uses posterior-mean grid parameters as the template
            # centre and attaches percentile-based uncertainties per grid axis.
            _n_grid = len(emu.param_names)
            _grid_means = np.mean(_samples[:, :_n_grid], axis=0)

            # Export uncertainties as 16th/84th percentile bounds in the same
            # friendly label space used in the results tables.
            _uncertainties = {}
            for _i, _label in enumerate(_labels[:_n_grid]):
                _lo = np.percentile(_samples[:, _i], 16)
                _hi = np.percentile(_samples[:, _i], 84)
                _uncertainties[_label] = (_lo, _hi)

            # Preserve posterior-mean nuisance parameters alongside the grid values
            # so the exported template records the full fitted configuration.
            _global = {}
            for _i in range(_n_grid, _samples.shape[1]):
                _global[_labels[_i]] = float(np.mean(_samples[:, _i]))

            _pf_path = os.path.join(_out_dir, f"speculate_export_{_ts}.pf")
            export_pf_template(
                emu, _grid_means, _pf_path,
                uncertainties=_uncertainties,
                global_params=_global,
                grid_name=grid_selector.value,
            )
            _msg += f"{mo.icon('lucide:check-circle')} .pf template exported to `{_pf_path}`\n\n"
        except Exception as _e:
            _msg += f"{mo.icon('lucide:x-circle')} .pf export failed: {_e}\n\n"

    if export_csv_btn.value and _samples is not None:
        try:
            from Speculate_addons.speculate_benchmark import export_posterior_csv

            os.makedirs(_out_dir, exist_ok=True)

            _csv_path = os.path.join(_out_dir, f"speculate_posterior_{_ts}.csv")
            export_posterior_csv(_samples, _labels, _csv_path, summary=_summary_dict)
            _msg += f"{mo.icon('lucide:check-circle')} Posterior CSV exported to `{_csv_path}`\n\n"
        except Exception as _e:
            _msg += f"{mo.icon('lucide:x-circle')} CSV export failed: {_e}\n\n"

    if export_corner_btn.value and _samples is not None:
        try:
            from Speculate_addons.speculate_benchmark import export_cornerplot_data

            os.makedirs(_out_dir, exist_ok=True)
            _corner_labels = _labels or [
                f"param_{_i}" for _i in range(_samples.shape[1])
            ]
            _record = dict(_corner_meta or {})
            _record.update({
                "source": "inference",
                "record_id": f"inference_{_ts}",
                "samples": _samples,
                "labels": _corner_labels,
                "summary": _summary_dict,
            })
            _export = export_cornerplot_data(
                _record,
                _out_dir,
                bundle_name=f"speculate_cornerplot_{_ts}",
                manifest_metadata={"source": "inference"},
            )
            _msg += (
                f"{mo.icon('lucide:check-circle')} Cornerplot data exported to "
                f"`{_export['bundle_dir']}`\n\n"
            )
        except Exception as _e:
            _msg += f"{mo.icon('lucide:x-circle')} Cornerplot export failed: {_e}\n\n"

    if _msg:
        mo.output.replace(mo.callout(mo.md(_msg), kind="success"))
    return


if __name__ == "__main__":
    app.run()
