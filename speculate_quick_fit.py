# /// script
# [tool.marimo.display]
# theme = "dark"
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
# ///
#
# Speculate Quick Fit (Neural Network Ensemble + Grid Interpolation) Tool
# =========================
#
# Pipeline:
#   Stage 1: Select grid data, parameters, wavelength range, flux scale,
#            PCA components and model type.
#   Stage 2: Train the Quick Fit emulator (PCA decomposition + NN/Grid Interpolation).
#   Stage 3: Inference — load trained QF model, observation data, configure
#            parameters, run chi-squared MLE.
#   Stage 4: Results & Export.

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Quick Fit")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import os
    import re
    import json
    import time as time_mod
    import pathlib
    import torch
    import torch.nn as nn
    from scipy.optimize import minimize as scipy_minimize
    from scipy.interpolate import RegularGridInterpolator
    from numpy.polynomial.chebyshev import chebval
    from sklearn.decomposition import PCA

    alt.data_transformers.enable("vegafusion")

    # Logo
    logo_path = pathlib.Path("assets/logos/Speculate_logo2.png")

    title_col = mo.vstack([
        mo.md(f"# {mo.icon('lucide:zap')} Quick Fit"),
        mo.md("*Lightweight emulators and chi-squared MLE fittings*")
    ])

    logo_col = mo.vstack([
        mo.image(src=str(logo_path), width=400, height=95),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    mo.hstack([title_col, logo_col], justify="space-between", align="center")
    return (
        PCA,
        RegularGridInterpolator,
        alt,
        chebval,
        json,
        mo,
        nn,
        np,
        os,
        pathlib,
        pd,
        re,
        scipy_minimize,
        time_mod,
        torch,
    )


@app.cell
def _(np):
    def qf_estimate_continuum(wl, flux, max_degree=5):
        _wl = np.asarray(wl, dtype=np.float64)
        _flux = np.asarray(flux, dtype=np.float64)

        if _flux.size == 0:
            return _flux
        if _flux.size == 1 or _wl.size <= 1:
            _level = float(_flux[0]) if _flux.size else 1.0
            return np.full_like(_flux, _level if abs(_level) > 1e-30 else 1.0)

        _degree = min(max_degree, _wl.size - 1)
        if _degree < 1:
            _level = float(np.mean(_flux))
            return np.full_like(_flux, _level if abs(_level) > 1e-30 else 1.0)

        try:
            _coeffs = np.polyfit(_wl, _flux, _degree)
            _continuum = np.polyval(_coeffs, _wl)
        except Exception:
            _level = float(np.mean(_flux))
            _continuum = np.full_like(_flux, _level)

        return np.where(
            np.isfinite(_continuum) & (np.abs(_continuum) > 1e-30),
            _continuum,
            1.0,
        )

    def qf_transform_observation_data(wl, flux, err, scale):
        _wl = np.asarray(wl, dtype=np.float64)
        _flux = np.asarray(flux, dtype=np.float64)

        if err is None:
            _err = np.abs(_flux) * 0.05
        else:
            _err = np.asarray(err, dtype=np.float64)

        _default_err = np.abs(_flux) * 0.05
        _default_err = np.where(_default_err > 0, _default_err, 1e-30)
        _err = np.where(np.isfinite(_err) & (_err > 0), _err, _default_err)

        if scale == "log":
            _safe_abs_flux = np.clip(np.abs(_flux), 1e-30, None)
            _flux = np.where(_flux > 0, np.log10(_flux), np.log10(_safe_abs_flux))
            _err = _err / (_safe_abs_flux * np.log(10))
        elif scale == "continuum-normalised":
            _continuum = qf_estimate_continuum(_wl, _flux)
            _continuum = np.where(np.abs(_continuum) > 1e-30, _continuum, 1.0)
            _flux = _flux / _continuum
            _err = _err / np.abs(_continuum)
        _err = np.where(np.isfinite(_err) & (_err > 0), _err, 1e-30)
        return _flux.astype(np.float64), _err.astype(np.float64)
    return (qf_transform_observation_data,)


@app.cell
def _(mo):
    _is_hf = __import__('os').environ.get("SPACE_ID") is not None
    if _is_hf:
        hf_mode_switch = None
        is_hf_space_nav = True
    else:
        hf_mode_switch = mo.ui.switch(value=False, label="🤗 HuggingFace Space Mode")
        is_hf_space_nav = False
    return hf_mode_switch, is_hf_space_nav


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
def _(hf_mode_switch, is_hf_space_nav, mo, usage_bars):
    if is_hf_space_nav:
        _mode = "HuggingFace Space"
    elif hf_mode_switch is not None and hf_mode_switch.value:
        _mode = "HuggingFace Space"
    else:
        _mode = "Local Machine"

    _items = [mo.md(f"#Speculate {mo.icon('lucide:telescope')}")]
    _items.extend([mo.md(" "), mo.md("---"), mo.md(" ")])

    if _mode == "HuggingFace Space":
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
        }, orientation="vertical"))
        _items.extend([
            mo.md(" "),
            mo.md("---"),
            mo.md(f"### {mo.icon('lucide:lock')} Locked Tools:"),
            mo.md("Install Speculate Locally"), mo.md(" "),
            mo.md(f"###{mo.icon('lucide:download')} Grid Downloader"), mo.md(" "),
            mo.md(f"###{mo.icon('lucide:brain')} Training Tool"), mo.md(" "),
            mo.md(f"###{mo.icon('lucide:sparkles')} Inference Tool"), mo.md(" "),
            mo.md(f"###{mo.icon('lucide:test-tubes')} Benchmark Suite"),
        ])
    else:
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/downloader": f"###{mo.icon('lucide:download')} Grid Downloader",
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
    if hf_mode_switch is not None:
        _items.extend([hf_mode_switch])

    _items.extend([mo.md("---"), usage_bars])
    mo.sidebar(mo.vstack(_items))
    return


@app.cell
def _():
    # Parameter database: maps Sirocco parameter index → (human-readable name, min, max, is_log).
    # These are the physical parameters that define each grid spectrum.
    # Params 1-6 are common to both no-BL and BL grids.
    # Params 7-8 exist only in BL grids (boundary layer luminosity/temperature).
    # Param 11 = viewing inclination angle (30°–85°), shared by both grids.
    # Note: params 9, 10 were sparse/mid inclination options, now removed.
    # is_log=True means the emulator-space value is log10 of the physical quantity.
    param_map_db = {
        1: ("disk.mdot", 0.0, 1.0, True),
        2: ("wind.mdot", 0.0, 1.0, False),
        3: ("KWD.d", 0.0, 1.0, True),
        4: ("KWD.mdot_r_exponent", 0.0, 1.0, False),
        5: ("KWD.acceleration_length", 0.0, 1.0, True),
        6: ("KWD.acceleration_exponent", 0.0, 1.0, False),
        7: ("Boundary_layer.luminosity", 0.0, 1.0, True),
        8: ("Boundary_layer.temp", 0.0, 1.0, False),
        11: ("Inclination", 30.0, 85.0, False),
    }
    return (param_map_db,)


@app.cell
def _(hf_mode_switch, is_hf_space_nav):
    # Determine whether we are running in HuggingFace Space mode.
    # In HF mode, training is disabled — only pre-trained model inference is available.
    # This flag gates visibility of grid selectors, training controls, etc.
    if is_hf_space_nav:
        qf_is_hf_mode = True
    elif hf_mode_switch is not None and hf_mode_switch.value:
        qf_is_hf_mode = True
    else:
        qf_is_hf_mode = False
    return (qf_is_hf_mode,)


@app.cell
def _(pathlib):
    # Grid configuration registry: defines the two available Sirocco spectral grids.
    # Each entry maps a grid folder name to its interface class, column range for
    # reading .spec files, and the maximum parameter set (used for the param selector).
    # - no-BL grid: 6 wind/disk params + inclination = 7 parameters
    # - BL grid: 8 wind/disk/BL params + inclination = 9 parameters
    # The grid interfaces (from Speculate_addons) handle reading .spec files and
    # extracting spectra at each inclination angle.
    from Speculate_addons.Spec_gridinterfaces import Speculate_cv_bl_grid_v87f
    from Speculate_addons.Spec_gridinterfaces import Speculate_cv_no_bl_grid_v87f

    qf_grid_configs = {
        "speculate_cv_no-bl_grid_v87f": {
            "class": Speculate_cv_no_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_no-bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 11],
        },
        "speculate_cv_bl_grid_v87f": {
            "class": Speculate_cv_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 7, 8, 11],
        },
    }

    # Scan sirocco_grids/ for locally available grid folders.
    # Only folders that match a known grid config AND contain .spec files are listed.
    qf_sirocco_grids_path = pathlib.Path("sirocco_grids")
    qf_available_grids = {}  # {display_label: folder_name}

    if qf_sirocco_grids_path.exists():
        for _folder in sorted(qf_sirocco_grids_path.iterdir()):
            if _folder.is_dir() and _folder.name in qf_grid_configs:
                _spec_files = list(_folder.glob("*.spec")) + list(_folder.glob("*.spec.xz"))
                if _spec_files:
                    _label = f"{qf_grid_configs[_folder.name]['name']} ({len(_spec_files)} files)"
                    qf_available_grids[_label] = _folder.name
    return qf_available_grids, qf_grid_configs, qf_sirocco_grids_path


@app.cell
def _(mo, np):
    import itertools as _itertools
    from Starfish.grid_tools import HDF5Creator as _HDF5Creator

    class QFMarimoHDF5Creator(_HDF5Creator):
        """Subclass of Starfish HDF5Creator that adds a marimo progress bar.

        Reads raw .spec files from sirocco_grids/ via the grid interface,
        applies wavelength trimming and flux transformations, then saves
        a compressed .npz file to Grid-Emulator_Files/ containing:
          - grid_points: (N_spectra, N_params) parameter combinations
          - flux_data: dict of {key: {flux, header}} per spectrum
          - wl: wavelength array
          - param_names, wave_units, flux_units metadata
        """
        def process_grid(self):
            param_list = [np.array(i) for i in _itertools.product(*self.points)]
            all_params = np.array(param_list)
            invalid_params = []

            for i, param in enumerate(mo.status.progress_bar(all_params, title="Processing Grid Points")):
                try:
                    flux, header = self.grid_interface.load_flux(param, header=True)
                except ValueError:
                    invalid_params.append(i)
                    continue

                _, fl_final = self.transform(flux)
                flux_key = self.key_name.format(*param)
                clean_header = {k: v for k, v in header.items()
                                if k != "" and k != "COMMENT" and v != ""}
                self.flux_data[flux_key] = {"flux": fl_final, "header": clean_header}

            all_params = np.delete(all_params, invalid_params, axis=0)

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
                grid_name=self.grid_name,
            )
    return (QFMarimoHDF5Creator,)


@app.cell
def _(mo, np, os):
    # Scan Grid-Emulator_Files/ for previously exported Quick Fit models.
    # These are .npz files with "_qfnn_" (NN) or "_qfgi_" (grid interpolation) in the name.
    # GP emulator files ("_emu_") are excluded — they belong to the main Training Tool.
    # Populates a dropdown so the user can skip training and go straight to inference.
    _emu_dir = "Grid-Emulator_Files"
    _qf_files = {}  # {display_label: filename}

    if os.path.exists(_emu_dir):
        for _f in sorted(os.listdir(_emu_dir)):
            if not _f.endswith(".npz"):
                continue
            # Only detect Quick Fit models, skip GP emulators
            if "_qfnn-ensemble_" in _f or "_qfnn_" in _f or "_qfgi_" in _f:
                _tag = "NN-Ensemble" if "_qfnn-ensemble_" in _f else ("NN" if "_qfnn_" in _f else "GridInterp")
                try:
                    _path = os.path.join(_emu_dir, _f)
                    _npz = np.load(_path, allow_pickle=True)
                    _n_par = len(list(_npz["param_names"]))
                    _label = f"[{_tag}] {_f} ({_n_par} params)"
                    _qf_files[_label] = _f
                    _npz.close()
                except Exception:
                    _qf_files[f"[{_tag}] {_f}"] = _f

    qf_pretrained_options = {"— None —": ""}
    qf_pretrained_options.update(_qf_files)

    qf_pretrained_selector = mo.ui.dropdown(
        options=qf_pretrained_options,
        value="— None —",
        label=f"{mo.icon('lucide:box')} Load Pre-trained Quick Fit Model:",
        full_width=True,
    )
    return (qf_pretrained_selector,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 1: Grid Selection & Configuration
    """)
    return


@app.cell
def _(mo, qf_available_grids, qf_is_hf_mode, qf_pretrained_selector):
    # Create the raw grid selector dropdown for local training mode.
    # In HF mode, this is None (no local grids available).
    # When a pre-trained model is selected, the grid selector is replaced
    # with a disabled placeholder so training config disappears.
    _pretrained_active = bool(qf_pretrained_selector.value and qf_pretrained_selector.value != "")

    if qf_is_hf_mode:
        # HF mode: no raw grid training path
        qf_grid_selector = None
    elif _pretrained_active:
        qf_grid_selector = mo.ui.dropdown(
            options={"— None (pre-trained model selected) —": ""},
            value="— None (pre-trained model selected) —",
            label=f"{mo.icon('lucide:database')} Train from Raw Grid:",
            full_width=True,
        )
    elif qf_available_grids:
        qf_grid_selector = mo.ui.dropdown(
            options=qf_available_grids,
            value=list(qf_available_grids.keys())[0],
            label=f"{mo.icon('lucide:database')} Train from Raw Grid:",
            full_width=True,
        )
    else:
        qf_grid_selector = mo.ui.dropdown(
            options={"— No grids found in sirocco_grids/ —": ""},
            value="— No grids found in sirocco_grids/ —",
            label=f"{mo.icon('lucide:database')} Train from Raw Grid:",
            full_width=True,
        )
    return (qf_grid_selector,)


@app.cell
def _(
    mo,
    param_map_db,
    qf_grid_configs,
    qf_grid_selector,
    qf_is_hf_mode,
    qf_sirocco_grids_path,
):
    # Build the parameter multiselect widget based on the selected grid.
    # Instantiates the grid interface to get human-readable parameter descriptions
    # (e.g. "disk.mdot" instead of "param1"). Falls back to param_map_db names on error.
    # All parameters are selected by default.
    qf_params_selector = None

    if qf_is_hf_mode or qf_grid_selector is None:
        pass
    elif qf_grid_selector.value and qf_grid_selector.value != "":
        _selected = qf_grid_selector.value  # folder name
        if _selected in qf_grid_configs:
            _config = qf_grid_configs[_selected]
            _max_params = _config["max_params"]
            _param_names = {}

            try:
                _temp_path = str(qf_sirocco_grids_path / _selected) + "/"
                _temp_iface = _config["class"](
                    path=_temp_path,
                    usecols=_config["usecols"],
                    model_parameters=tuple(_max_params),
                )
                _desc_map = _temp_iface.parameters_description()
                for _pk, _pd in _desc_map.items():
                    _pidx = int(_pk.replace("param", ""))
                    _param_names[_pidx] = _pd
            except Exception:
                _param_names = {i: param_map_db[i][0] if i in param_map_db else f"Parameter {i}" for i in _max_params}

            _options = {
                _param_names.get(i, f"Parameter {i}"): str(i)
                for i in _max_params
            }
            _default = [k for k, v in _options.items()]

            qf_params_selector = mo.ui.multiselect(
                options=_options,
                value=_default,
                label="Select parameters to include:",
            )
    return (qf_params_selector,)


@app.cell
def _(mo):
    # Stage 1 configuration widgets for training.
    # These are all defined in one cell so marimo creates them in a single pass.
    #
    # Wavelength range: restricts which part of the spectrum is used for PCA/training.
    # Flux scale: how spectra are represented (linear, log, continuum-normalised, etc.).
    # Smoothing: optional boxcar-5 smoothing to reduce noise in raw spectra.
    # PCA components: how many principal components to retain (2-20).
    # Model type: "Neural Network" or "Grid Interpolation (Linear)".
    #   - NN: learns a mapping from parameters → PCA weights via a feedforward network.
    #   - Grid Interpolation: uses scipy RegularGridInterpolator for exact multilinear
    #     interpolation on the Cartesian product grid (no training, O(d) per query).
    # Ensemble checkbox: if checked, runs Optuna hyperparameter search (30 trials)
    #   then trains 5 ensemble members for uncertainty estimation.
    qf_wl_min = mo.ui.number(
        start=800, stop=8000,
        value=850, step=1,
        label="Min Wavelength (Å):",
    )
    qf_wl_max = mo.ui.number(
        start=800, stop=8000,
        value=1850, step=1,
        label="Max Wavelength (Å):",
    )

    qf_scale_selector = mo.ui.dropdown(
        options=["linear", "log", "continuum-normalised"],
        value="linear",
        label="Flux Scale:",
    )

    qf_use_smoothing = mo.ui.checkbox(
        value=False,
        label="Smooth Spectra (Boxcar=5)",
    )

    qf_n_components = mo.ui.slider(
        start=2, stop=30, value=10, step=1,
        label="PCA Components:",
        show_value=True,
    )

    qf_test_pca_btn = mo.ui.run_button(label="Test PCA Reconstruction", kind="neutral")

    qf_model_type = mo.ui.dropdown(
        options=["Neural Network", "Grid Interpolation (Linear)"],
        value="Neural Network",
        label="Model Type:",
    )

    qf_ensemble_checkbox = mo.ui.checkbox(
        value=False,
        label="Ensemble (multiple members)",
    )
    qf_optuna_checkbox = mo.ui.checkbox(
        value=False,
        label="Optuna Search (hyperparameter tuning)",
    )
    return (
        qf_ensemble_checkbox,
        qf_model_type,
        qf_optuna_checkbox,
        qf_n_components,
        qf_scale_selector,
        qf_test_pca_btn,
        qf_use_smoothing,
        qf_wl_max,
        qf_wl_min,
    )


@app.cell
def _(np, os, qf_grid_selector, qf_is_hf_mode, qf_params_selector, qf_scale_selector, qf_use_smoothing, qf_wl_max, qf_wl_min):
    # Grid data loader: finds and loads the processed .npz grid file that
    # matches the user's current configuration (grid, params, scale,
    # smoothing AND wavelength range).
    #
    # Filename convention:
    #   {grid_name}_grid_{param_digits}_{scale}[_smooth]_{wl_lo}-{wl_hi}AA.npz
    #
    # The NPZ contains: grid_points (N×D), flux_data (dict of spectra keyed by
    # parameter combination), wl (wavelength array), and param_names.
    qf_grid_data = None

    if qf_is_hf_mode or qf_grid_selector is None:
        pass
    elif qf_grid_selector.value and qf_grid_selector.value != "":
        _grid_name = qf_grid_selector.value  # e.g. "speculate_cv_no-bl_grid_v87f"
        _emu_dir = "Grid-Emulator_Files"

        # Derive selected param indices to find matching processed grid file
        _selected_indices = []
        if qf_params_selector is not None and qf_params_selector.value:
            _selected_indices = sorted([int(v) for v in qf_params_selector.value])

        # Build the EXACT expected filename so only a grid with the same params,
        # scale, smoothing AND wavelength range can match.
        _param_tag = "".join(str(i) for i in _selected_indices) if _selected_indices else ""
        _scale = qf_scale_selector.value
        _smooth_suffix = "_smooth" if qf_use_smoothing.value else ""
        _wl_lo_tag = int(round(qf_wl_min.value))
        _wl_hi_tag = int(round(qf_wl_max.value))
        _expected_file = f"{_grid_name}_grid_{_param_tag}_{_scale}{_smooth_suffix}_{_wl_lo_tag}-{_wl_hi_tag}AA.npz"

        # Load only if the exact file exists
        _path = os.path.join(_emu_dir, _expected_file) if _param_tag else ""
        if _param_tag and os.path.isfile(_path):
            try:
                _npz = np.load(_path, allow_pickle=True)
                qf_grid_data = {
                    "grid_points": np.array(_npz["grid_points"]),
                    "flux_data": _npz["flux_data"].item() if "flux_data" in _npz else None,
                    "flux_key_name": str(_npz["flux_key_name"]) if "flux_key_name" in _npz else None,
                    "wl": np.array(_npz["wl"]) if "wl" in _npz else None,
                    "param_names": list(_npz["param_names"]) if "param_names" in _npz else None,
                    "grid_name": str(_npz["grid_name"]) if "grid_name" in _npz else "",
                    "source_file": _expected_file,
                }
                _npz.close()
            except Exception:
                pass
    return (qf_grid_data,)


@app.cell
def _(
    get_qf_pca_result,
    mo,
    qf_ensemble_checkbox,
    qf_grid_selector,
    qf_is_hf_mode,
    qf_model_type,
    qf_n_components,
    qf_optuna_checkbox,
    qf_params_selector,
    qf_pretrained_selector,
    qf_scale_selector,
    qf_test_pca_btn,
    qf_use_smoothing,
    qf_wl_max,
    qf_wl_min,
):
    # Stage 1 display layout cell: assembles the sidebar left/right panel structure.
    # Left panel: pre-trained model selector vs raw grid selector (with "OR" separator).
    # Right panel: parameter selector, training config widgets (model type, wl range,
    #   flux scale, smoothing, PCA), and the PCA test button.
    # In HF mode, only the pre-trained selector is shown.
    _elements = []

    # ── Pre-trained vs Raw grid selector (side by side) ──────────────────
    if not qf_is_hf_mode and qf_grid_selector is not None:
        _left = mo.vstack([
            qf_pretrained_selector,
        ])
        _right = mo.vstack([
            qf_grid_selector,
        ])
        _elements.append(
            mo.hstack(
                [_left, mo.md("&nbsp; **OR** &nbsp;"), _right],
                justify="start", align="center", gap=1,
            )
        )
    else:
        _elements.append(qf_pretrained_selector)

    if qf_pretrained_selector.value and qf_pretrained_selector.value != "":
        _elements.append(
            mo.callout(
                mo.md(f"**Pre-trained model selected:** `{qf_pretrained_selector.value}`\n\n"
                      "Skip to **Stage 3: Inference** below."),
                kind="success",
            )
        )

    # ── Raw grid training config (local mode only) ───────────────────────
    # Hidden when a pre-trained model is selected so the UI is unambiguous.
    _pretrained_active = bool(qf_pretrained_selector.value and qf_pretrained_selector.value != "")
    if not qf_is_hf_mode and not _pretrained_active:
        _elements.append(mo.md("---"))
        _elements.append(
            qf_params_selector if qf_params_selector is not None
            else mo.md("*Select a grid first*")
        )

        _elements.append(mo.md("---"))
        _model_type_row = [qf_model_type]
        if qf_model_type.value == "Neural Network":
            _model_type_row.append(qf_ensemble_checkbox)
            _model_type_row.append(qf_optuna_checkbox)
        _elements.append(
            mo.hstack(_model_type_row, justify="start", align="center")
        )
        _elements.append(mo.hstack([qf_wl_min, qf_wl_max], justify="start"))
        _elements.append(
            mo.hstack([qf_scale_selector, qf_use_smoothing], justify="start", align="center")
        )
        _pca_result_text = mo.md(f"_{get_qf_pca_result()}_")
        _elements.append(
            mo.hstack(
                [qf_n_components, qf_test_pca_btn, _pca_result_text],
                justify="start", align="center",
            )
        )
    elif qf_is_hf_mode and (not qf_pretrained_selector.value or qf_pretrained_selector.value == ""):
        _elements.append(
            mo.callout(
                mo.md(f"{mo.icon('lucide:info')} Select a pre-trained Quick Fit model above to begin inference."),
                kind="info",
            )
        )

    mo.vstack(_elements)
    return


@app.cell
def _(mo):
    get_qf_pca_result, set_qf_pca_result = mo.state("Click test to see variance")
    return get_qf_pca_result, set_qf_pca_result


@app.cell
def _(mo, qf_is_hf_mode, qf_pretrained_selector, torch):
    # GPU status callout and toggle: displays whether CUDA GPU is available
    # and provides a switch to enable/disable GPU training.
    # Live plot toggle: when off, uses a lightweight spinner during training
    # and only renders the loss chart once training completes.
    qf_use_gpu = mo.ui.switch(value=False, label="Use GPU")
    qf_live_plot = mo.ui.switch(value=True, label="Live Loss Plot")

    if qf_is_hf_mode or (qf_pretrained_selector.value and qf_pretrained_selector.value != ""):
        mo.stop(True, mo.md(""))

    if torch.cuda.is_available():
        _gpu_name = torch.cuda.get_device_name(0)
        try:
            _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _gpu_text = f"**{mo.icon('lucide:cpu')} GPU Available:** {_gpu_name} ({_vram_gb:.1f} GB VRAM)"
        except Exception:
            _gpu_text = f"**{mo.icon('lucide:cpu')} GPU Available:** {_gpu_name}"
        qf_use_gpu = mo.ui.switch(value=False, label="Use GPU")
        mo.output.replace(mo.callout(
            mo.hstack([mo.md(_gpu_text), qf_use_gpu, qf_live_plot], justify="start", align="center"),
            kind="success",
        ))
    else:
        mo.output.replace(
            mo.callout(
                mo.hstack([
                    mo.md(f"{mo.icon('lucide:cpu')} **CPU-only mode** \u2014 No CUDA GPU detected."),
                    qf_live_plot,
                ], justify="start", align="center"),
                kind="info",
            )
        )
    return qf_live_plot, qf_use_gpu


@app.cell
def _(
    QFMarimoHDF5Creator,
    mo,
    np,
    os,
    qf_grid_configs,
    qf_grid_selector,
    qf_is_hf_mode,
    qf_n_components,
    qf_params_selector,
    qf_scale_selector,
    qf_sirocco_grids_path,
    qf_test_pca_btn,
    qf_use_smoothing,
    qf_wl_max,
    qf_wl_min,
    set_qf_pca_result,
):
    if not qf_test_pca_btn.value:
        pass
    elif qf_is_hf_mode:
        set_qf_pca_result("Not available in HF mode")
    elif qf_grid_selector is None or not qf_grid_selector.value:
        set_qf_pca_result("Select a grid first")
    elif qf_params_selector is None or not qf_params_selector.value:
        set_qf_pca_result("Select parameters first")
    else:
        # Build the processed grid file if it doesn't already exist.
        # This uses QFMarimoHDF5Creator to read raw .spec files from sirocco_grids/
        # and save a compressed .npz for PCA testing and later training.
        _grid_name = qf_grid_selector.value
        _param_indices = sorted([int(v) for v in qf_params_selector.value])
        _param_tag = "".join(str(i) for i in _param_indices)
        _scale = qf_scale_selector.value
        _smooth_suffix = "_smooth" if qf_use_smoothing.value else ""
        _wl_lo_tag = int(round(qf_wl_min.value))
        _wl_hi_tag = int(round(qf_wl_max.value))
        _grid_file = f"{_grid_name}_grid_{_param_tag}_{_scale}{_smooth_suffix}_{_wl_lo_tag}-{_wl_hi_tag}AA.npz"
        _grid_path = os.path.join("Grid-Emulator_Files", _grid_file)

        _grid_ready = True

        if not os.path.exists(_grid_path):
            set_qf_pca_result("Building grid file...")
            try:
                _config = qf_grid_configs[_grid_name]
                _wl_range = (qf_wl_min.value, qf_wl_max.value)
                _iface = _config["class"](
                    path=str(qf_sirocco_grids_path / _grid_name) + "/",
                    usecols=_config["usecols"],
                    wl_range=_wl_range,
                    model_parameters=tuple(_param_indices),
                    scale=_scale,
                    smoothing=qf_use_smoothing.value,
                )
                _keyname = "".join(f"param{i}{{}}" for i in _param_indices)
                os.makedirs("Grid-Emulator_Files", exist_ok=True)
                _creator = QFMarimoHDF5Creator(
                    _iface, _grid_path, key_name=_keyname, wl_range=_wl_range,
                )
                _creator.process_grid()

                _data = np.load(_grid_path, allow_pickle=True)
                set_qf_pca_result(f"Grid built ({_data['grid_points'].shape[0]} spectra). Running PCA...")
            except Exception as e:
                set_qf_pca_result(f"Grid build error: {e}")
                _grid_ready = False

        if _grid_ready and os.path.exists(_grid_path):
            try:
                with mo.status.spinner(title="Running PCA variance test..."):
                    from Starfish.emulator import Emulator
                    _exp_var, _n_comps = Emulator.test_pca(_grid_path, n_components=int(qf_n_components.value))
                set_qf_pca_result(f"Variance: {_exp_var:.5f} ({_exp_var*100:.3f}%) [{_n_comps} comps]")
            except Exception as e:
                set_qf_pca_result(f"PCA error: {e}")
    return


@app.cell
def _(nn):
    # QuickFitNN: lightweight PyTorch feedforward network for mapping
    # physical parameters → PCA component weights.
    #
    # Architecture is fully configurable via hyperparameters:
    #   - hidden_sizes: list of layer widths (e.g. [256, 256, 256])
    #   - activation: ReLU, GELU, SiLU, Mish, etc.
    #   - dropout: rate + strategy (uniform/increasing/last_only)
    #   - batch_norm: optional BatchNorm1d after each hidden layer
    #   - skip_connections: residual connection from input to output
    #     (with learned projection if input/output dims differ)
    #
    # build_hidden_sizes(): generates layer width lists for Optuna trials
    # using four patterns:
    #   - constant: all layers same width, e.g. [256, 256, 256]
    #   - funnel: decreasing, e.g. [512, 256, 128]
    #   - expanding: increasing, e.g. [64, 128, 256]
    #   - bottleneck: wide → narrow → wide, e.g. [256, 64, 256]
    class QuickFitNN(nn.Module):
        def __init__(self, n_in, n_out, hidden_sizes, activation_name="GELU",
                     dropout=0.0, dropout_strategy="uniform",
                     batch_norm=False, skip_connections=False):
            super().__init__()
            self.skip_connections = skip_connections

            act_map = {
                "ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU,
                "Mish": nn.Mish, "LeakyReLU": nn.LeakyReLU,
                "Tanh": nn.Tanh, "ELU": nn.ELU,
            }
            act_cls = act_map.get(activation_name, nn.GELU)

            layers = []
            prev_size = n_in
            n_layers = len(hidden_sizes)

            for i, h in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, h))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(act_cls())
                if dropout > 0:
                    if dropout_strategy == "uniform":
                        layers.append(nn.Dropout(dropout))
                    elif dropout_strategy == "increasing":
                        layers.append(nn.Dropout(dropout * (i + 1) / n_layers))
                    elif dropout_strategy == "last_only" and i == n_layers - 1:
                        layers.append(nn.Dropout(dropout))
                prev_size = h

            layers.append(nn.Linear(prev_size, n_out))
            self.net = nn.Sequential(*layers)

            self._skip_proj = None
            if skip_connections and n_in != n_out:
                self._skip_proj = nn.Linear(n_in, n_out)

        def forward(self, x):
            out = self.net(x)
            if self.skip_connections:
                if self._skip_proj is not None:
                    out = out + self._skip_proj(x)
                elif x.shape[-1] == out.shape[-1]:
                    out = out + x
            return out

    def build_hidden_sizes(trial, n_layers, pattern):
        if pattern == "constant":
            w = trial.suggest_categorical("hidden_width", [32, 64, 128, 256])
            return [w] * n_layers
        elif pattern == "funnel":
            w_start = trial.suggest_categorical("funnel_start", [128, 256, 512])
            return [max(32, w_start // (2 ** i)) for i in range(n_layers)]
        elif pattern == "expanding":
            w_start = trial.suggest_categorical("expand_start", [32, 64])
            return [min(512, w_start * (2 ** i)) for i in range(n_layers)]
        return [128] * n_layers
    return QuickFitNN, build_hidden_sizes


@app.cell
def _(mo, qf_is_hf_mode, qf_pretrained_selector):
    # Stage 2 header: conditionally shown only in local training mode.
    if qf_is_hf_mode or (qf_pretrained_selector.value and qf_pretrained_selector.value != ""):
        mo.output.replace(mo.md(""))
    else:
        mo.output.replace(mo.md("## Stage 2: Model Training"))
    return


@app.cell
def _(mo, qf_is_hf_mode, qf_model_type, qf_pretrained_selector):
    # Stage 2 training controls.
    # For NN: shows max-epochs slider + train button.
    # For Grid Interpolation: only shows train button (no epochs needed — construction is instant).
    # In HF mode or with a pretrained model selected, controls are hidden.
    qf_max_train_iter = mo.ui.number(
        value=2000, start=100, stop=10000, step=100,
        label="Max Epochs:",
    )
    qf_train_btn = mo.ui.run_button(label=f"{mo.icon('lucide:play')} Train Emulator")

    # Hide controls when in HF mode or pre-trained model selected
    if qf_is_hf_mode or (qf_pretrained_selector.value and qf_pretrained_selector.value != ""):
        _output = mo.md("")
    elif "Neural" in qf_model_type.value:
        _output = mo.hstack([qf_max_train_iter, qf_train_btn], justify="start")
    else:
        # Grid Interpolation: no epochs needed
        _output = mo.hstack([qf_train_btn], justify="start")
    _output
    return qf_max_train_iter, qf_train_btn


@app.cell
def _(
    PCA,
    QFMarimoHDF5Creator,
    QuickFitNN,
    RegularGridInterpolator,
    alt,
    build_hidden_sizes,
    json,
    mo,
    nn,
    np,
    os,
    pd,
    qf_ensemble_checkbox,
    qf_grid_configs,
    qf_grid_data,
    qf_grid_selector,
    qf_live_plot,
    qf_max_train_iter,
    qf_model_type,
    qf_n_components,
    qf_optuna_checkbox,
    qf_params_selector,
    qf_scale_selector,
    qf_sirocco_grids_path,
    qf_train_btn,
    qf_use_gpu,
    qf_use_smoothing,
    qf_wl_max,
    qf_wl_min,
    re,
    time_mod,
    torch,
):
    # Initialise output variables to None. Marimo reactive cells must always
    # define their return variables, even when execution is stopped early.
    qf_trained_models = None
    qf_train_info = None
    qf_input_scaler = None
    qf_output_scaler = None
    qf_best_hparams = None
    qf_pca_data = None

    # Block until the user clicks the Train button.
    mo.stop(not qf_train_btn.value)

    if qf_params_selector is None or not qf_params_selector.value:
        mo.stop(True, mo.callout(mo.md("Select at least one parameter."), kind="warn"))

    # ── Auto-build grid .npz if it doesn't exist yet ────────────────────
    # The grid data loader cell only reads files already on disk.  If the
    # user clicks Train without first clicking "Test PCA Reconstruction",
    # we build the .npz here so training can proceed.
    _qf_grid_data_local = qf_grid_data
    if _qf_grid_data_local is None and qf_grid_selector is not None and qf_grid_selector.value:
        _grid_name = qf_grid_selector.value
        _param_indices = sorted([int(v) for v in qf_params_selector.value])
        _param_tag = "".join(str(i) for i in _param_indices)
        _scale = qf_scale_selector.value
        _smooth_suffix = "_smooth" if qf_use_smoothing.value else ""
        _wl_lo_tag = int(round(qf_wl_min.value))
        _wl_hi_tag = int(round(qf_wl_max.value))
        _grid_file = f"{_grid_name}_grid_{_param_tag}_{_scale}{_smooth_suffix}_{_wl_lo_tag}-{_wl_hi_tag}AA.npz"
        _grid_path = os.path.join("Grid-Emulator_Files", _grid_file)

        if not os.path.exists(_grid_path):
            with mo.status.spinner(title="Building grid file from raw spectra..."):
                try:
                    _config = qf_grid_configs[_grid_name]
                    _wl_range = (qf_wl_min.value, qf_wl_max.value)
                    _iface = _config["class"](
                        path=str(qf_sirocco_grids_path / _grid_name) + "/",
                        usecols=_config["usecols"],
                        wl_range=_wl_range,
                        model_parameters=tuple(_param_indices),
                        scale=_scale,
                        smoothing=qf_use_smoothing.value,
                    )
                    _keyname = "".join(f"param{i}{{}}" for i in _param_indices)
                    os.makedirs("Grid-Emulator_Files", exist_ok=True)
                    _creator = QFMarimoHDF5Creator(
                        _iface, _grid_path, key_name=_keyname, wl_range=_wl_range,
                    )
                    _creator.process_grid()
                except Exception as _e:
                    mo.stop(True, mo.callout(mo.md(f"Grid build failed: {_e}"), kind="danger"))

        # Load the freshly-built (or pre-existing) .npz
        if os.path.exists(_grid_path):
            try:
                _npz = np.load(_grid_path, allow_pickle=True)
                _qf_grid_data_local = {
                    "grid_points": np.array(_npz["grid_points"]),
                    "flux_data": _npz["flux_data"].item() if "flux_data" in _npz else None,
                    "flux_key_name": str(_npz["flux_key_name"]) if "flux_key_name" in _npz else None,
                    "wl": np.array(_npz["wl"]) if "wl" in _npz else None,
                    "param_names": list(_npz["param_names"]) if "param_names" in _npz else None,
                    "grid_name": str(_npz["grid_name"]) if "grid_name" in _npz else "",
                    "source_file": _grid_file,
                }
                _npz.close()
            except Exception as _e:
                mo.stop(True, mo.callout(mo.md(f"Failed to load grid: {_e}"), kind="danger"))

    if _qf_grid_data_local is None:
        mo.stop(True, mo.callout(mo.md("Select a grid dataset first."), kind="warn"))
    if _qf_grid_data_local.get("flux_data") is None:
        mo.stop(True, mo.callout(mo.md("Grid file has no flux_data."), kind="danger"))

    # ── Step 1: Extract flux matrix from grid data ───────────────────────
    with mo.status.spinner(title="Processing grid data...") as _spinner:
        _wl = _qf_grid_data_local["wl"]
        _grid_points = _qf_grid_data_local["grid_points"]
        _flux_data = _qf_grid_data_local["flux_data"]
        _flux_key_name = _qf_grid_data_local.get("flux_key_name")
        _param_names = _qf_grid_data_local["param_names"]

        _selected_param_indices = [int(v) for v in qf_params_selector.value]

        # Map param column names (e.g. "param1", "param3") to their integer indices.
        # Then filter X_grid to only the user-selected parameter columns.
        _all_indices = []
        for _pn in _param_names:
            _m = re.search(r"(\d+)", _pn)
            _all_indices.append(int(_m.group(1)) if _m else 0)

        _col_mask = [i for i, idx in enumerate(_all_indices) if idx in _selected_param_indices]

        if not _col_mask:
            mo.stop(True, mo.callout(mo.md("No matching parameter columns found."), kind="danger"))

        # X_grid: subset of grid_points with only the selected parameter columns.
        X_grid = _grid_points[:, _col_mask].astype(np.float32)

        # Trim wavelength array to the user-specified range.
        _wl_lo = float(qf_wl_min.value)
        _wl_hi = float(qf_wl_max.value)
        _wl_mask = (_wl >= _wl_lo) & (_wl <= _wl_hi)
        _wl_trimmed = _wl[_wl_mask]

        if _wl_trimmed.size == 0:
            mo.stop(True, mo.callout(mo.md("Selected wavelength range does not overlap the grid."), kind="danger"))

        # Helper to extract flux rows from the grid data dict.
        # Each entry is {flux: array, header: ...}. We skip malformed entries
        # and trim each spectrum to the selected wavelength range.
        def _collect_flux_rows(_row_and_entries):
            _rows = []
            _fluxes = []
            for _row_index, _entry in _row_and_entries:
                if not isinstance(_entry, dict) or "flux" not in _entry:
                    continue
                _fl = np.asarray(_entry["flux"], dtype=np.float32)
                if _fl.size < _wl.size:
                    continue
                _rows.append(_row_index)
                _fluxes.append(_fl[_wl_mask])
            return _rows, _fluxes

        # Try keyed lookup first (faster for large grids), fall back to sequential.
        _valid_rows = []
        _flux_matrix = []
        if _flux_key_name:
            _valid_rows, _flux_matrix = _collect_flux_rows(
                (
                    _row_index,
                    _flux_data.get(_flux_key_name.format(*_params)),
                )
                for _row_index, _params in enumerate(_grid_points)
            )

        if not _flux_matrix:
            _valid_rows, _flux_matrix = _collect_flux_rows(enumerate(_flux_data.values()))

        if not _flux_matrix:
            mo.stop(True, mo.callout(mo.md("Could not extract flux data from grid."), kind="danger"))

        _flux_matrix = np.array(_flux_matrix, dtype=np.float32)
        if _valid_rows:
            X_grid = X_grid[_valid_rows]

        # NOTE: Scale (log, continuum-normalised, etc.) and smoothing were
        # already applied by the grid interface when the NPZ was created.
        # The flux in the grid file is ready for PCA — no re-application.
        _scale = qf_scale_selector.value

    # ── Step 2: PCA decomposition ────────────────────────────────────────
    # Reduce the flux matrix from (N_spectra, N_wavelengths) to (N_spectra, N_pca_components).
    # The emulator learns to predict these low-dimensional PCA weights rather than
    # the full spectral flux, dramatically reducing the output dimension.
    # PCA data is saved so we can reconstruct spectra during inference:
    #   flux_rn = (weights @ eigenspectra) * std + mean   (row-normalised flux)
    with mo.status.spinner(title="PCA decomposition..."):
        _n_comp = int(qf_n_components.value)

        # Row-mean normalisation: divide each spectrum by its mean so all
        # spectra have approximately unit average flux.  This prevents PCA
        # from wasting components on trivial brightness differences (raw
        # fluxes can span orders of magnitude, e.g. 10^-12).  This matches
        # the Starfish Emulator.test_pca normalisation.
        _norm_factors = _flux_matrix.mean(axis=1)
        _norm_factors = np.where(_norm_factors != 0, _norm_factors, 1.0)
        _flux_matrix = _flux_matrix / _norm_factors[:, np.newaxis]

        # Per-wavelength standardisation (zero mean, unit variance).
        _flux_mean = _flux_matrix.mean(axis=0)
        _flux_std = _flux_matrix.std(axis=0)
        _flux_std = np.where(_flux_std > 0, _flux_std, 1.0)  # avoid division by zero
        _flux_norm = (_flux_matrix - _flux_mean) / _flux_std

        # Fit PCA and project flux onto the principal components.
        # _weights: (N_spectra, N_components) — the PCA weight targets for emulation.
        # _eigenspectra: (N_components, N_wavelengths) — used to reconstruct flux.
        _pca = PCA(n_components=_n_comp)
        _weights = _pca.fit_transform(_flux_norm).astype(np.float32)
        _eigenspectra = _pca.components_.astype(np.float32)
        _var_explained = _pca.explained_variance_ratio_.sum() * 100

        # Compute per-wavelength PCA truncation RMSE — the irreducible error
        # from discarding components beyond N_comp.  This is the "PCA floor"
        # that no emulator (GP, NN, grid interp) can beat.
        _pca_recon = _weights @ _eigenspectra          # (N, n_wl) in normalised space
        _pca_resid = _flux_norm - _pca_recon           # truncation residual
        _pca_per_wl_rmse = np.sqrt(np.mean((_pca_resid * _flux_std) ** 2, axis=0))  # (n_wl,) in flux units

        # Store all PCA metadata needed for inference reconstruction later.
        qf_pca_data = {
            "grid_points": X_grid,
            "weights": _weights,
            "eigenspectra": _eigenspectra,
            "flux_mean": _flux_mean,
            "flux_std": _flux_std,
            "wl": _wl_trimmed,
            "param_names": [_param_names[c] for c in _col_mask] if _param_names else [f"param_{c}" for c in _col_mask],
            "min_params": X_grid.min(axis=0),
            "max_params": X_grid.max(axis=0),
            "n_components": _n_comp,
            "variance_explained": _var_explained,
            "pca_per_wl_rmse": _pca_per_wl_rmse,
            "scale": _scale,
            "smoothing": qf_use_smoothing.value,
            "source_grid_file": qf_grid_selector.value,
            "selected_param_indices": _selected_param_indices,
        }

    # ── Shared setup ─────────────────────────────────────────────────────
    # Common setup for both model paths.
    # Y_weights: PCA component targets, shape (N_spectra, N_components).
    # X_norm: input parameters normalised to [0,1] via min-max scaling.
    #   Grid Interpolation uses raw X_grid (unscaled) — the scaler is saved
    #   for the NN path and for consistent export format.
    Y_weights = _weights
    n_in = X_grid.shape[1]   # number of physical parameters
    n_out = Y_weights.shape[1]  # number of PCA components

    # Min-max scaler for input parameters.
    x_min = X_grid.min(axis=0)
    x_max = X_grid.max(axis=0)
    x_range = np.where(x_max - x_min > 0, x_max - x_min, 1.0)
    qf_input_scaler = {"method": "minmax", "min": x_min, "max": x_max, "range": x_range}
    X_norm = (X_grid - x_min) / x_range  # normalised params for NN training

    # Output (PCA weight) statistics — used if NN output normalisation is enabled.
    y_mean = Y_weights.mean(axis=0)
    y_std = Y_weights.std(axis=0)
    y_std = np.where(y_std > 0, y_std, 1.0)

    # ── Step 3: Model Training ───────────────────────────────────────────
    # Two paths: Grid Interpolation (instant construction) or NN (gradient descent).

    if "Grid" in qf_model_type.value:
        # ── Grid Interpolation (Linear) ──────────────────────────────────
        with mo.status.spinner(title="Constructing RegularGridInterpolator..."):
            # Extract sorted unique values along each parameter axis.
            _axes = [np.unique(X_grid[:, i]) for i in range(n_in)]
            _grid_shape = [len(a) for a in _axes]
            _expected = int(np.prod(_grid_shape))

            # Verify grid is a complete Cartesian product.
            if _expected != X_grid.shape[0]:
                mo.stop(True, mo.callout(mo.md(
                    f"Grid is not regular: expected {_expected} points "
                    f"({'×'.join(str(s) for s in _grid_shape)}), got {X_grid.shape[0]}. "
                    "Grid Interpolation requires a Cartesian product grid."
                ), kind="danger"))

            # Sort grid points in Cartesian order (lexsort by last axis first)
            # so we can reshape the 1D weight array into the N-D grid shape.
            _sorted_idx = np.lexsort(X_grid.T[::-1])
            _Y_sorted = Y_weights[_sorted_idx]

            # Build one RegularGridInterpolator per PCA component.
            qf_trained_models = []
            for _c in range(n_out):
                # Reshape this component's weights from flat → N-D grid shape.
                _vals_grid = _Y_sorted[:, _c].reshape(_grid_shape)
                # bounds_error=False, fill_value=None enables nearest-neighbor
                # extrapolation outside the grid domain.
                _interp = RegularGridInterpolator(
                    _axes, _vals_grid, method="linear",
                    bounds_error=False, fill_value=None,
                )
                qf_trained_models.append(_interp)

            qf_train_info = {
                "source": "grid_interp",
                "n_models": n_out,
                "grid_shape": _grid_shape,
                "axes": _axes,
            }

    else:
        # ── Neural Network Training ──────────────────────────────────────
        # Trains one or more QuickFitNN models to learn the mapping:
        #   normalised_params → PCA_weights
        # Two modes:
        #   - Single NN: uses optimal default hyperparameters, trains once.
        #   - Ensemble: trains N members with different seeds for uncertainty.
        # Optuna search (optional): explores hyperparameter space before training.
        _max_epochs = int(qf_max_train_iter.value)
        _use_ensemble = qf_ensemble_checkbox.value
        _use_optuna = qf_optuna_checkbox.value
        _n_ensemble = 3

        # 90/10 random train/validation split (fixed seed for reproducibility).
        _n_samples = len(X_norm)
        _rng = np.random.RandomState(42)
        _perm_idx = _rng.permutation(_n_samples)
        _split = max(1, int(_n_samples * 0.1))
        _val_idx = _perm_idx[:_split]
        _train_idx = _perm_idx[_split:]

        # Resolve device once: GPU if user toggled on and CUDA available, else CPU.
        _device = torch.device("cuda" if qf_use_gpu.value and torch.cuda.is_available() else "cpu")

        def _train_single_nn(n_in, n_out, hidden, act, dp, dp_strat, bn, skip,
                             opt_name, lr, wd, bs, sched_name, out_norm,
                             max_epochs, train_idx, val_idx, X_norm, Y_weights,
                             y_mean, y_std, seed=42, trial=None,
                             epoch_callback=None):
            """Train a single QuickFitNN and return (model, best_val_loss, train_losses, val_losses).

            Implements mini-batch gradient descent with early stopping (patience=50).
            Optionally integrates with Optuna for pruning unpromising trials.
            Supports three LR schedulers: cosine annealing, reduce-on-plateau, one-cycle.
            If *epoch_callback* is provided, it is called every 50 epochs as
            ``epoch_callback(epoch, max_epochs, best_val_loss, train_losses, val_losses)``.
            """
            torch.manual_seed(seed)
            np.random.seed(seed)

            _Xt = torch.tensor(X_norm, dtype=torch.float32).to(_device)
            if out_norm == "standard":
                _Yt = torch.tensor((Y_weights - y_mean) / y_std, dtype=torch.float32).to(_device)
            else:
                _Yt = torch.tensor(Y_weights, dtype=torch.float32).to(_device)

            _model = QuickFitNN(n_in, n_out, hidden, act, dp, dp_strat, bn, skip).to(_device)

            if opt_name == "adam":
                _optim = torch.optim.Adam(_model.parameters(), lr=lr, weight_decay=wd)
            elif opt_name == "adamw":
                _optim = torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=wd)
            else:
                _optim = torch.optim.SGD(_model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

            _actual_bs = len(train_idx) if bs == 9999 else bs

            _scheduler = None
            if sched_name == "cosine_annealing":
                _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optim, T_max=max_epochs // 4)
            elif sched_name == "reduce_on_plateau":
                _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optim, patience=20, factor=0.5)
            elif sched_name == "one_cycle":
                _batches_per_epoch = max(1, (len(train_idx) + _actual_bs - 1) // _actual_bs)
                _steps = _batches_per_epoch * max_epochs
                _scheduler = torch.optim.lr_scheduler.OneCycleLR(_optim, max_lr=lr, total_steps=_steps)
                _onecycle_step_count = 0

            _Xtr = _Xt[train_idx]
            _Ytr = _Yt[train_idx]
            _Xval = _Xt[val_idx]
            _Yval = _Yt[val_idx]

            _best_val = float("inf")
            _best_state = None
            _patience_counter = 0
            _train_losses = []
            _val_losses = []

            for _ep in range(max_epochs):
                _model.train()
                _epoch_train_loss = 0.0
                _n_batches = 0
                _perm = torch.randperm(len(_Xtr))
                for _start in range(0, len(_Xtr), _actual_bs):
                    _idx = _perm[_start:_start + _actual_bs]
                    _pred = _model(_Xtr[_idx])
                    _loss = nn.functional.mse_loss(_pred, _Ytr[_idx])
                    _optim.zero_grad()
                    _loss.backward()
                    _optim.step()
                    _epoch_train_loss += _loss.item()
                    _n_batches += 1
                    if sched_name == "one_cycle" and _scheduler is not None:
                        _onecycle_step_count += 1
                        if _onecycle_step_count < _steps:
                            _scheduler.step()

                _train_losses.append(_epoch_train_loss / max(_n_batches, 1))

                if sched_name == "cosine_annealing" and _scheduler is not None:
                    _scheduler.step()

                _model.eval()
                with torch.no_grad():
                    _val_loss = nn.functional.mse_loss(_model(_Xval), _Yval).item()
                _val_losses.append(_val_loss)

                if sched_name == "reduce_on_plateau" and _scheduler is not None:
                    _scheduler.step(_val_loss)

                if _val_loss < _best_val - 1e-8:
                    _best_val = _val_loss
                    _best_state = {k: v.clone() for k, v in _model.state_dict().items()}
                    _patience_counter = 0
                else:
                    _patience_counter += 1
                if _patience_counter >= 50:
                    break

                if epoch_callback is not None and _ep % 50 == 0:
                    epoch_callback(_ep + 1, max_epochs, _best_val, _train_losses, _val_losses)

                if trial is not None:
                    import optuna
                    trial.report(_val_loss, _ep)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if _best_state is not None:
                _model.load_state_dict(_best_state)
            # Always return model on CPU for inference/export compatibility.
            _model.cpu().eval()
            return _model, _best_val, _train_losses, _val_losses

        if _use_optuna:
            # ── Optuna Hyperparameter Search ─────────────────────────────
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            _n_trials = 30

            # Timer and throttle state for the Optuna search phase.
            _optuna_t0 = time_mod.time()
            _optuna_ui = {"last_update": 0, "trial": 0}

            def _optuna_fmt_elapsed():
                _s = int(time_mod.time() - _optuna_t0)
                return f"{_s // 60}m {_s % 60:02d}s"

            def _build_optuna_chart(train_losses, val_losses, ep, mx, best_loss, elapsed_str, trial_num, n_trials):
                """Build an Altair chart for the current Optuna trial."""
                if len(val_losses) < 2:
                    return None
                _rows = []
                for _i, (_tl, _vl) in enumerate(zip(train_losses, val_losses)):
                    _rows.append({"Epoch": _i + 1, "Loss": _tl, "Series": "Train MSE"})
                    _rows.append({"Epoch": _i + 1, "Loss": _vl, "Series": "Val MSE"})
                _df = pd.DataFrame(_rows)
                _title = f"Optuna Trial {trial_num}/{n_trials} — Epoch {ep}/{mx} [{elapsed_str}]"
                _chart = alt.Chart(_df).mark_line().encode(
                    x=alt.X("Epoch:Q"),
                    y=alt.Y("Loss:Q", scale=alt.Scale(zero=False), title="MSE Loss"),
                    color=alt.Color("Series:N", legend=alt.Legend(title=""),
                                    scale=alt.Scale(domain=["Train MSE", "Val MSE"],
                                                    range=["#3b82f6", "#ef4444"])),
                ).properties(title=_title, width=650, height=300)
                _rule = alt.Chart(pd.DataFrame([{"y": best_loss}])).mark_rule(
                    strokeDash=[6, 3], color="#22c55e"
                ).encode(y="y:Q")
                return _chart + _rule

            def _optuna_epoch_cb(ep, mx, best_loss, train_losses, val_losses):
                """Epoch callback during Optuna search — shows trial + epoch progress."""
                _now = time_mod.time()
                if _now - _optuna_ui["last_update"] < 0.5:
                    return
                _optuna_ui["last_update"] = _now
                _el = _optuna_fmt_elapsed()
                _t_num = _optuna_ui["trial"]
                _bar_pct = int((_t_num - 1) / _n_trials * 100)
                _status = mo.md(
                    f"{mo.icon('lucide:brain')} **Optuna Hyperparameter Search** — "
                    f"Trial {_t_num}/{_n_trials} | "
                    f"Epoch {ep}/{mx} | Best Val MSE: {best_loss:.2e} | {_el}"
                )
                if qf_live_plot.value:
                    _chart = _build_optuna_chart(train_losses, val_losses, ep, mx, best_loss, _el, _t_num, _n_trials)
                    if _chart is not None:
                        mo.output.replace(mo.vstack([_status, _chart]))
                    else:
                        mo.output.replace(_status)
                else:
                    mo.output.replace(_status)

            _MAX_PARAMS = 250_000  # Hard cap on model parameter count

            def _objective(trial):
                # Optuna objective: sample hyperparameters, train one NN, return val MSE.
                # Each trial explores a different architecture and training configuration.
                _optuna_ui["trial"] = trial.number + 1
                _n_layers = trial.suggest_int("n_layers", 2, 5)
                _pattern = trial.suggest_categorical("width_pattern", ["constant", "expanding", "funnel"])
                _hidden = build_hidden_sizes(trial, _n_layers, _pattern)
                _act = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"])
                _dp = trial.suggest_categorical("dropout", [0.0, 0.01, 0.05, 0.1])
                _dp_strat = trial.suggest_categorical("dropout_strategy", ["uniform", "increasing", "last_only"])
                _bn = True
                _skip = False
                _lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                _wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
                _bs = trial.suggest_categorical("batch_size", [64, 128, 256])
                _opt_name = "adamw"
                _sched = "reduce_on_plateau"
                _out_norm = "standard"

                # Reject architectures that exceed the parameter budget.
                _sizes = [n_in] + _hidden + [n_out]
                _est_params = sum(_sizes[i] * _sizes[i+1] + _sizes[i+1]
                                  for i in range(len(_sizes) - 1))
                if _est_params > _MAX_PARAMS:
                    raise optuna.TrialPruned()

                _, _val_loss, _, _ = _train_single_nn(
                    n_in, n_out, _hidden, _act, _dp, _dp_strat, _bn, _skip,
                    _opt_name, _lr, _wd, _bs, _sched, _out_norm,
                    _max_epochs, _train_idx, _val_idx, X_norm, Y_weights,
                    y_mean, y_std, seed=trial.number * 137 + 42, trial=trial,
                    epoch_callback=_optuna_epoch_cb,
                )
                return float(_val_loss)

            # Run Optuna study: one trial at a time with progress updates.
            _study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

            mo.output.replace(mo.md(f"{mo.icon('lucide:brain')} **Starting Optuna Hyperparameter Search** ({_n_trials} trials)..."))
            for _t_i in range(_n_trials):
                _study.optimize(_objective, n_trials=1, show_progress_bar=False)

            _best = _study.best_trial
            qf_best_hparams = {**_best.params,
                "batch_norm": True, "skip_connections": False,
                "optimizer": "adamw", "output_norm": "standard",
                "lr_scheduler": "reduce_on_plateau"}
            _optuna_elapsed = _optuna_fmt_elapsed()

            # Locked params that were not part of the Optuna suggest calls.
            _locked_params = {"batch_norm": True, "skip_connections": False,
                              "optimizer": "adamw", "output_norm": "standard",
                              "lr_scheduler": "reduce_on_plateau"}

            # Save full Optuna trial report as JSON for future reference.
            _trial_report = []
            for _tr in _study.trials:
                _tr_info = {
                    "number": _tr.number + 1,
                    "value": _tr.value,
                    "state": str(_tr.state),
                    "params": {**_tr.params, **_locked_params},
                    "duration_s": (_tr.datetime_complete - _tr.datetime_start).total_seconds()
                                 if _tr.datetime_complete and _tr.datetime_start else None,
                }
                # Estimate parameter count from hidden sizes.
                try:
                    _tr_hidden = build_hidden_sizes(_tr, _tr.params.get("n_layers", 2),
                                                     _tr.params.get("width_pattern", "constant"))
                    _tr_sizes = [n_in] + _tr_hidden + [n_out]
                    _tr_n_params = sum(_tr_sizes[i] * _tr_sizes[i+1] + _tr_sizes[i+1]
                                       for i in range(len(_tr_sizes) - 1))
                    _tr_info["hidden_sizes"] = _tr_hidden
                    _tr_info["n_params"] = _tr_n_params
                except Exception:
                    pass
                _trial_report.append(_tr_info)

            _report_path = os.path.join("Grid-Emulator_Files",
                f"{qf_grid_selector.value}_optuna_report_{time_mod.strftime('%Y%m%d_%H%M%S', time_mod.localtime())}.json")
            with open(_report_path, "w") as _f:
                json.dump({
                    "n_trials": _n_trials,
                    "n_ensemble": _n_ensemble,
                    "best_trial": _best.number + 1,
                    "best_val_mse": _best.value,
                    "best_params": qf_best_hparams,
                    "search_elapsed": _optuna_elapsed,
                    "trials": _trial_report,
                }, _f, indent=2, default=str)

            mo.output.replace(mo.md(
                f"{mo.icon('lucide:check-circle')} **Optuna search complete!** ({_optuna_elapsed}) — "
                f"Best trial #{_best.number + 1} | Val MSE: {_best.value:.2e}. "
                f"Report saved to `{_report_path}`."
            ))
            _hp = qf_best_hparams
            # Reconstruct the hidden layer sizes from the best trial's params.
            _hidden = build_hidden_sizes(_best, _hp.get("n_layers", 4), _hp.get("width_pattern", "constant"))

        else:
            # ── Optimal Defaults ─────────────────────────────────────────
            # Architecture determined by 4 rounds of Optuna search (120 trials).
            _hp = {
                "n_layers": 4, "hidden_width": 256, "width_pattern": "constant",
                "activation": "ReLU",
                "dropout": 0.01, "dropout_strategy": "increasing",
                "batch_norm": True, "skip_connections": False,
                "optimizer": "adamw", "lr": 1.5e-3, "weight_decay": 5e-3,
                "batch_size": 128, "lr_scheduler": "reduce_on_plateau",
                "output_norm": "standard",
            }
            _hidden = [_hp["hidden_width"]] * _hp["n_layers"]
            qf_best_hparams = _hp

        # Common: output normalisation scaler.
        _out_norm = _hp.get("output_norm", "standard")
        if _out_norm == "standard":
            qf_output_scaler = {"mean": y_mean, "std": y_std}

        if _use_ensemble:
            # ── Ensemble Training ────────────────────────────────────────
            # Train N members with the same architecture but different random
            # seeds. At inference, predictions are averaged for robustness.
            if _use_optuna:
                mo.output.replace(mo.md(
                    f"{mo.icon('lucide:brain')} **Training {_n_ensemble}-member ensemble "
                    f"with Optuna-optimised hyperparameters...**"
                ))
            else:
                mo.output.replace(mo.md(
                    f"{mo.icon('lucide:brain')} **Training {_n_ensemble}-member ensemble "
                    f"with optimal defaults...**"
                ))
            qf_trained_models = []
            _full_idx = np.arange(_n_samples)  # Train ensemble on ALL data
            _ensemble_val_histories = []
            _ensemble_train_histories = []
            _ens_val_losses = []

            # Colours for ensemble member loss curves.
            _member_colours = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7"]

            # UI throttle state — shared across all ensemble members.
            _ens_ui = {"last_update": 0}
            _ens_t0 = time_mod.time()

            def _fmt_elapsed(t0):
                """Format elapsed seconds since t0 as 'Xm Ys'."""
                _s = int(time_mod.time() - t0)
                return f"{_s // 60}m {_s % 60:02d}s"

            def _build_ensemble_chart(_val_hists, _train_hists, _current_member, _current_epoch, _max_ep, _elapsed_str=""):
                """Build an overlaid Altair chart of ensemble member loss curves."""
                _rows = []
                for _mi, (_th, _vh) in enumerate(zip(_train_hists, _val_hists)):
                    for _ei, (_tl, _vl) in enumerate(zip(_th, _vh)):
                        _rows.append({"Epoch": _ei + 1, "Loss": _vl, "Series": f"Member {_mi+1} (val)", "type": "val"})
                        _rows.append({"Epoch": _ei + 1, "Loss": _tl, "Series": f"Member {_mi+1} (train)", "type": "train"})
                if not _rows:
                    return None
                _df = pd.DataFrame(_rows)
                _title = (f"Ensemble Training — Member {_current_member}/{len(_val_hists)} "
                          f"(Epoch {_current_epoch}/{_max_ep})")
                if _elapsed_str:
                    _title += f" [{_elapsed_str}]"
                _chart = alt.Chart(_df).mark_line().encode(
                    x=alt.X("Epoch:Q"),
                    y=alt.Y("Loss:Q", scale=alt.Scale(zero=False), title="MSE Loss"),
                    color=alt.Color("Series:N", legend=alt.Legend(title="")),
                    strokeDash=alt.StrokeDash("type:N", legend=None,
                                              scale=alt.Scale(domain=["train", "val"], range=[[4, 4], [0]])),
                ).properties(
                    title=_title,
                    width=650, height=300,
                )
                return _chart

            def _ens_epoch_cb(_ep, _mx, _best, _tl, _vl):
                """Epoch callback for ensemble members — updates live chart or spinner."""
                _now = time_mod.time()
                if _now - _ens_ui["last_update"] < 0.5:
                    return
                _ens_ui["last_update"] = _now
                _el = _fmt_elapsed(_ens_t0)
                if qf_live_plot.value:
                    # Build chart of all completed members + current member in-progress.
                    _all_vh = _ensemble_val_histories + [_vl]
                    _all_th = _ensemble_train_histories + [_tl]
                    _chart = _build_ensemble_chart(_all_vh, _all_th,
                                                   len(_ensemble_val_histories) + 1, _ep, _mx, _el)
                    if _chart is not None:
                        _status = mo.md(f"{mo.icon('lucide:brain')} **Training ensemble member "
                                        f"{len(_ensemble_val_histories)+1}/{_n_ensemble}** — "
                                        f"Epoch {_ep}/{_mx} | Best Val MSE: {_best:.2e} | {_el}")
                        mo.output.replace(mo.vstack([_status, _chart]))
                else:
                    _status = mo.md(f"{mo.icon('lucide:brain')} **Training ensemble member "
                                    f"{len(_ensemble_val_histories)+1}/{_n_ensemble}** — "
                                    f"Epoch {_ep}/{_mx} | Best Val MSE: {_best:.2e} | {_el}")
                    mo.output.replace(_status)

            for _seed in range(_n_ensemble):
                _model, _loss, _tl, _vl = _train_single_nn(
                    n_in, n_out, _hidden,
                    _hp.get("activation", "ReLU"), _hp.get("dropout", 0.01),
                    _hp.get("dropout_strategy", "increasing"), _hp.get("batch_norm", True),
                    _hp.get("skip_connections", False), _hp.get("optimizer", "adamw"),
                    _hp.get("lr", 1.5e-3), _hp.get("weight_decay", 5e-3),
                    _hp.get("batch_size", 128), _hp.get("lr_scheduler", "reduce_on_plateau"),
                    _out_norm, _max_epochs, _full_idx, _val_idx,
                    X_norm, Y_weights, y_mean, y_std, seed=_seed * 137 + 42,
                    epoch_callback=_ens_epoch_cb,
                )
                qf_trained_models.append(_model)
                _ens_val_losses.append(_loss)
                _ensemble_val_histories.append(_vl)
                _ensemble_train_histories.append(_tl)

            # Show final ensemble chart.
            _ens_elapsed = _fmt_elapsed(_ens_t0)
            _final_ens_chart = _build_ensemble_chart(_ensemble_val_histories, _ensemble_train_histories,
                                                     _n_ensemble, len(_ensemble_val_histories[-1]), _max_epochs, _ens_elapsed)
            if _final_ens_chart is not None:
                mo.output.replace(mo.vstack([
                    mo.md(f"{mo.icon('lucide:check-circle')} **Ensemble training complete!** ({_ens_elapsed})"),
                    _final_ens_chart,
                ]))

            qf_train_info = {
                "source": "nn",
                "n_models": _n_ensemble,
                "best_val_mse": min(_ens_val_losses),
                "hparams": qf_best_hparams,
                "hidden_sizes": _hidden,
                "ensemble": True,
                "ensemble_val_histories": _ensemble_val_histories,
                "ensemble_train_histories": _ensemble_train_histories,
            }

        else:
            # ── Single NN Training ───────────────────────────────────────
            # Trains one NN using the selected hyperparameters (from Optuna
            # or optimal defaults).  No uncertainty estimation.

            # Live chart UI throttle state and timer.
            _nn_ui = {"last_update": 0}
            _nn_t0 = time_mod.time()

            def _nn_fmt_elapsed():
                _s = int(time_mod.time() - _nn_t0)
                return f"{_s // 60}m {_s % 60:02d}s"

            def _build_nn_chart(_train_losses, _val_losses, _ep, _mx, _best, _elapsed_str=""):
                """Build a live Altair chart of the NN training loss."""
                _n = len(_train_losses)
                _epochs = list(range(1, _n + 1))
                _df = pd.DataFrame({
                    "Epoch": _epochs * 2,
                    "Loss": list(_train_losses) + list(_val_losses),
                    "Series": ["Train MSE"] * _n + ["Val MSE"] * _n,
                })
                _title = f"NN Training — Epoch {_ep}/{_mx} | Best Val MSE: {_best:.2e}"
                if _elapsed_str:
                    _title += f" [{_elapsed_str}]"
                _base = alt.Chart(_df).mark_line().encode(
                    x=alt.X("Epoch:Q"),
                    y=alt.Y("Loss:Q", scale=alt.Scale(zero=False), title="MSE Loss"),
                    color=alt.Color("Series:N",
                                    scale=alt.Scale(domain=["Train MSE", "Val MSE"],
                                                    range=["#3b82f6", "#ef4444"]),
                                    legend=alt.Legend(title="")),
                ).properties(
                    title=_title,
                    width=650, height=300,
                )
                _rule = alt.Chart(pd.DataFrame({"y": [_best]})).mark_rule(
                    color="green", strokeDash=[5, 5], size=2,
                ).encode(y="y")
                return _base + _rule

            def _nn_epoch_cb(ep, mx, best_loss, train_losses, val_losses):
                _now = time_mod.time()
                if _now - _nn_ui["last_update"] < 0.5:
                    return
                _nn_ui["last_update"] = _now
                _el = _nn_fmt_elapsed()
                if qf_live_plot.value:
                    _chart = _build_nn_chart(train_losses, val_losses, ep, mx, best_loss, _el)
                    _status = mo.md(f"{mo.icon('lucide:brain')} **Training single Neural Network...** "
                                    f"Epoch {ep}/{mx} | Best Val MSE: {best_loss:.2e} | {_el}")
                    mo.output.replace(mo.vstack([_status, _chart]))
                else:
                    _status = mo.md(f"{mo.icon('lucide:brain')} **Training single Neural Network...** "
                                    f"Epoch {ep}/{mx} | Best Val MSE: {best_loss:.2e} | {_el}")
                    mo.output.replace(_status)

            mo.output.replace(mo.md(f"{mo.icon('lucide:brain')} **Training single Neural Network...**"))

            _model, _val_loss, _nn_train_losses, _nn_val_losses = _train_single_nn(
                n_in, n_out, _hidden,
                _hp["activation"], _hp["dropout"],
                _hp["dropout_strategy"], _hp["batch_norm"],
                _hp["skip_connections"], _hp["optimizer"],
                _hp["lr"], _hp["weight_decay"],
                _hp["batch_size"], _hp["lr_scheduler"],
                _hp["output_norm"], _max_epochs,
                _train_idx, _val_idx, X_norm, Y_weights, y_mean, y_std,
                epoch_callback=_nn_epoch_cb,
            )

            # Show final loss curve.
            _nn_elapsed = _nn_fmt_elapsed()
            if _nn_train_losses:
                _final_chart = _build_nn_chart(_nn_train_losses, _nn_val_losses,
                                               len(_nn_train_losses), _max_epochs, _val_loss, _nn_elapsed)
                mo.output.replace(mo.vstack([
                    mo.md(f"{mo.icon('lucide:check-circle')} **Training complete!** "
                          f"Best Val MSE: {_val_loss:.2e} after {len(_nn_train_losses)} epochs ({_nn_elapsed})"),
                    _final_chart,
                ]))

            qf_trained_models = [_model]

            qf_train_info = {
                "source": "nn",
                "n_models": 1,
                "best_val_mse": _val_loss,
                "hparams": qf_best_hparams,
                "hidden_sizes": _hidden,
                "ensemble": False,
                "train_losses": _nn_train_losses,
                "val_losses": _nn_val_losses,
            }
    return (
        qf_best_hparams,
        qf_input_scaler,
        qf_output_scaler,
        qf_pca_data,
        qf_train_info,
        qf_trained_models,
    )


@app.cell
def _(
    alt,
    mo,
    np,
    param_map_db,
    pd,
    qf_input_scaler,
    qf_output_scaler,
    qf_pca_data,
    qf_train_info,
    qf_trained_models,
    re,
    torch,
):
    # ── Training Results: Summary + R² ───────────────────────────────────
    # Computes R² (LOO-CV for grid interp, standard for NN) and displays
    # a summary callout + per-component R² bar chart.
    # Also stores shared data that the reactive diagnostics cells below need.
    qf_diag_data = None

    if qf_trained_models is None or qf_train_info is None:
        mo.stop(True, mo.md(""))

    _source = qf_train_info["source"]
    _X_grid = qf_pca_data["grid_points"].astype(np.float32)
    _Y_true = qf_pca_data["weights"].astype(np.float32)
    _n_comp = _Y_true.shape[1]
    _param_names = qf_pca_data["param_names"]
    _n_params = _X_grid.shape[1]
    _N = _X_grid.shape[0]

    # Resolve human-readable parameter labels from param_names
    def _get_param_label(p_idx):
        _pn = _param_names[p_idx] if p_idx < len(_param_names) else f"param_{p_idx}"
        _m = re.search(r"(\d+)", str(_pn))
        _pidx = int(_m.group(1)) if _m else 0
        return param_map_db[_pidx][0] if _pidx in param_map_db else str(_pn)

    _unique_per_dim = [np.sort(np.unique(_X_grid[:, d])) for d in range(_n_params)]

    _type_label = ("Grid Interpolation (Linear)" if _source == "grid_interp"
                   else f"Neural Network ({'Ensemble' if qf_train_info.get('ensemble') else 'Single'})")

    # ── Compute R² ───────────────────────────────────────────────────────
    if _source == "grid_interp":
        # LOO-CV R²: for each grid point, estimate its weight using the mean
        # of its immediate neighbours (±1 along each axis).
        _n_loo = min(_N, 500)
        _loo_indices = (np.random.default_rng(42).choice(_N, _n_loo, replace=False)
                        if _N > _n_loo else np.arange(_N))
        _loo_pred = np.zeros((_n_loo, _n_comp))
        _loo_true = _Y_true[_loo_indices]

        _axes = [np.unique(_X_grid[:, d]) for d in range(_n_params)]
        _sorted_idx = np.lexsort(_X_grid.T[::-1])
        _grid_shape = [len(a) for a in _axes]

        for _li, _leave_idx in enumerate(_loo_indices):
            _pt = _X_grid[_leave_idx]
            _grid_pos = []
            for _d in range(_n_params):
                _grid_pos.append(int(np.searchsorted(_axes[_d], _pt[_d])))
            for _c in range(_n_comp):
                _vals_sorted = _Y_true[_sorted_idx, _c].reshape(_grid_shape).copy()
                _vals_sorted[tuple(_grid_pos)] = np.nan
                _neighbors = []
                for _d in range(_n_params):
                    for _delta in [-1, 1]:
                        _npos = list(_grid_pos)
                        _npos[_d] = _npos[_d] + _delta
                        if 0 <= _npos[_d] < _grid_shape[_d]:
                            _nval = _vals_sorted[tuple(_npos)]
                            if not np.isnan(_nval):
                                _neighbors.append(_nval)
                _loo_pred[_li, _c] = np.nanmean(_neighbors) if _neighbors else _Y_true[_leave_idx, _c]

        _ss_res = np.sum((_loo_true - _loo_pred) ** 2, axis=0)
        _ss_tot = np.sum((_loo_true - _loo_true.mean(axis=0)) ** 2, axis=0)
        _r2 = 1 - _ss_res / np.where(_ss_tot > 0, _ss_tot, 1.0)
        _r2_title = f"Per-Component LOO-CV R² ({_n_loo} samples)"
    else:
        # NN: standard R² at all training points.
        _sc = qf_input_scaler
        _X_norm = (_X_grid - _sc["min"]) / _sc["range"]
        _Xt = torch.tensor(_X_norm, dtype=torch.float32)
        _preds = []
        for _m in qf_trained_models:
            _m.eval()
            with torch.no_grad():
                _p = _m(_Xt).numpy()
            if qf_output_scaler is not None:
                _p = _p * qf_output_scaler["std"] + qf_output_scaler["mean"]
            _preds.append(_p)
        _mean_pred = np.mean(_preds, axis=0)
        _ss_res = np.sum((_Y_true - _mean_pred) ** 2, axis=0)
        _ss_tot = np.sum((_Y_true - _Y_true.mean(axis=0)) ** 2, axis=0)
        _r2 = 1 - _ss_res / np.where(_ss_tot > 0, _ss_tot, 1.0)
        _r2_title = "Per-Component R² (Predicted vs Grid)"

    _summary = mo.callout(
        mo.md(f"""
        **Training Complete** — {_type_label}
        - **Models:** {qf_train_info['n_models']}
        - **PCA:** {qf_pca_data['n_components']} components, {qf_pca_data['variance_explained']:.2f}% variance ({'log' if qf_pca_data['scale'] == 'log' else qf_pca_data['scale']} scale{', smoothed' if qf_pca_data['smoothing'] else ''})
        - **Mean R²:** {float(_r2.mean()):.6f} | **Min R²:** {float(_r2.min()):.6f}
        {'- _R² computed via leave-one-out cross-validation (neighbor interpolation)_' if _source == 'grid_interp' else ''}
        """),
        kind="success"
    )

    # ── Per-Wavelength RMSE Envelope ─────────────────────────────────────
    # Two lines, matching the Tier 1 benchmark diagnostic:
    #   Blue  — "PCA truncation only": irreducible floor from discarding higher components.
    #   Red   — "LOO (PCA + model)":   total error = PCA truncation + interpolation/NN error.
    _eigenspectra = qf_pca_data["eigenspectra"]   # (n_comp, n_wl)
    _flux_std = qf_pca_data["flux_std"]           # (n_wl,)
    _wl = qf_pca_data["wl"]                       # (n_wl,)
    _pca_rmse = qf_pca_data["pca_per_wl_rmse"]    # (n_wl,) computed during training

    if _source == "grid_interp":
        _w_resid = _loo_true - _loo_pred           # (n_loo, n_comp)
    else:
        _w_resid = _Y_true - _mean_pred            # (N, n_comp)

    # Project weight residuals to flux space: Δflux = Δw @ (eigenspectra * flux_std)
    _flux_resid = _w_resid @ (_eigenspectra * _flux_std)   # (n_samples, n_wl)
    _model_per_wl_rmse = np.sqrt(np.mean(_flux_resid ** 2, axis=0))  # (n_wl,)

    # Total RMSE = sqrt(PCA_truncation² + model_prediction²) — the two sources add in quadrature
    _total_per_wl_rmse = np.sqrt(_pca_rmse ** 2 + _model_per_wl_rmse ** 2)

    _model_label = f"LOO (PCA + {'Interp' if _source == 'grid_interp' else 'NN'})"
    _pca_df = pd.DataFrame({"Wavelength": _wl, "RMSE": _pca_rmse, "Source": "PCA truncation only"})
    _total_df = pd.DataFrame({"Wavelength": _wl, "RMSE": _total_per_wl_rmse, "Source": _model_label})
    _rmse_df = pd.concat([_pca_df, _total_df], ignore_index=True)

    _rmse_chart = alt.Chart(_rmse_df).mark_line(
        strokeWidth=1.5,
    ).encode(
        x=alt.X("Wavelength:Q", title="Wavelength (Å)",
                scale=alt.Scale(domain=[float(_wl.min()), float(_wl.max())])),
        y=alt.Y("RMSE:Q", title="RMSE (normalised flux)", axis=alt.Axis(format=".1e")),
        color=alt.Color("Source:N", title="",
                        scale=alt.Scale(domain=["PCA truncation only", _model_label],
                                        range=["#3498db", "#e74c3c"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[
            alt.Tooltip("Source:N"),
            alt.Tooltip("Wavelength:Q", title="Wavelength (Å)", format=".1f"),
            alt.Tooltip("RMSE:Q", format=".4e"),
        ],
    ).properties(
        width="container", height=200,
        title="Per-Wavelength Reconstruction Error"
    )

    # ── Assemble summary output ──────────────────────────────────────────
    _elements = [_summary, _rmse_chart]

    if _source == "nn" and "hparams" in qf_train_info and qf_train_info["hparams"]:
        _lines = [f"- **{k}**: `{v}`" for k, v in qf_train_info["hparams"].items()]
        _hparams_acc = mo.accordion({
            f"{mo.icon('lucide:settings')} Hyperparameters": mo.md("\n".join(_lines))
        })
        _elements.append(_hparams_acc)

    # Store shared diagnostics data for the reactive chart cells below.
    _param_labels = [_get_param_label(i) for i in range(_n_params)]
    qf_diag_data = {
        "source": _source,
        "X_grid": _X_grid,
        "Y_true": _Y_true,
        "n_comp": _n_comp,
        "n_params": _n_params,
        "N": _N,
        "param_labels": _param_labels,
        "unique_per_dim": _unique_per_dim,
    }

    _result = mo.vstack(_elements)
    _result
    return (qf_diag_data,)


@app.cell
def _(mo, qf_diag_data, qf_train_info, qf_trained_models):
    # ── Weight Diagnostics: UI controls ──────────────────────────────────
    # Reactive controls matching the GP diagnostics pattern in speculate_training.py:
    #   - Dropdown: which parameter to vary (X-axis)
    #   - Slider: which fixed grid-point index to use for the non-varying params
    #   - Component range sliders: from/to PCA component
    # Changing any control reactively re-renders the chart cell below.
    qf_diag_xaxis = None
    qf_diag_fixed_slider = None
    qf_diag_comp_start = None
    qf_diag_comp_end = None

    if qf_trained_models is None or qf_train_info is None or qf_diag_data is None:
        mo.stop(True, mo.md(""))

    _n_params = qf_diag_data["n_params"]
    _n_comp = qf_diag_data["n_comp"]
    _param_labels = qf_diag_data["param_labels"]
    _unique_per_dim = qf_diag_data["unique_per_dim"]

    # Build parameter options: display label → column index string
    _param_options = {}
    for _col_idx in range(_n_params):
        _label = _param_labels[_col_idx]
        _param_options[_label] = str(_col_idx)

    qf_diag_xaxis = mo.ui.dropdown(
        options=_param_options,
        value=list(_param_options.keys())[0],
        label="Varying Parameter (X-axis):",
    )

    # Fixed position slider: 0=low, 1=mid, 2=high.  Parameters with more
    # unique values (e.g. inclination) are mapped proportionally so that
    # 0 → lowest value, 1 → middle value, 2 → highest value.
    _min_unique = min(len(u) for u in _unique_per_dim)
    qf_diag_fixed_slider = mo.ui.slider(
        start=0, stop=_min_unique - 1, value=_min_unique // 2, step=1,
        label="Fixed Grid Position (low → high):", show_value=True,
    )

    qf_diag_comp_start = mo.ui.slider(
        start=0, stop=_n_comp - 1, value=0, step=1,
        label="From Component:", show_value=True,
    )
    qf_diag_comp_end = mo.ui.slider(
        start=0, stop=_n_comp - 1, value=min(5, _n_comp - 1), step=1,
        label="To Component:", show_value=True,
    )

    mo.output.replace(mo.md(""))
    return (
        qf_diag_comp_end,
        qf_diag_comp_start,
        qf_diag_fixed_slider,
        qf_diag_xaxis,
    )


@app.cell
def _(
    alt,
    mo,
    np,
    pd,
    qf_diag_comp_end,
    qf_diag_comp_start,
    qf_diag_data,
    qf_diag_fixed_slider,
    qf_diag_xaxis,
    qf_input_scaler,
    qf_output_scaler,
    qf_train_info,
    qf_trained_models,
    torch,
):
    # ── Weight Diagnostics: Reactive Chart ───────────────────────────────
    # Renders 1D slice plots through parameter space, matching the GP weight
    # diagnostics pattern in speculate_training.py.
    # Varies one parameter (selected by dropdown) along the X-axis,
    # fixes all other parameters at a grid value (controlled by slider).
    # Blue dots = actual PCA weights at grid nodes.
    # Orange line = model prediction (linear interpolation or NN).
    qf_diag_figure = mo.md("")

    if (qf_trained_models is None or qf_train_info is None or qf_diag_data is None
            or qf_diag_xaxis is None or qf_diag_xaxis.value is None):
        mo.stop(True, qf_diag_figure)

    _source = qf_diag_data["source"]
    _X_grid = qf_diag_data["X_grid"]
    _Y_true = qf_diag_data["Y_true"]
    _n_comp = qf_diag_data["n_comp"]
    _n_params = qf_diag_data["n_params"]
    _N = qf_diag_data["N"]
    _param_labels = qf_diag_data["param_labels"]
    _unique_per_dim = qf_diag_data["unique_per_dim"]

    # Read UI control values
    _not_fixed_col = int(qf_diag_xaxis.value)
    _fixed_idx = qf_diag_fixed_slider.value
    _comp_start = qf_diag_comp_start.value
    _comp_end = qf_diag_comp_end.value
    if _comp_end < _comp_start:
        _comp_end = _comp_start

    _xlabel = _param_labels[_not_fixed_col]

    # Build fixed parameter values: for each dimension, map the slider position
    # proportionally into that dimension's unique values.  This ensures that
    # slider 0/1/2 → low/mid/high even for dimensions like inclination that
    # have more unique values than the slider range.
    _slider_max = min(len(u) for u in _unique_per_dim) - 1
    _fixed_values = []
    for _i in range(_n_params):
        _vals = _unique_per_dim[_i]
        if _i == _not_fixed_col:
            _fixed_values.append(_vals)  # all values for varying param
        else:
            _idx = round(_fixed_idx * (len(_vals) - 1) / _slider_max) if _slider_max > 0 else 0
            _fixed_values.append(_vals[_idx])  # proportionally mapped value

    # Find grid rows matching: varying param = any value, all others = fixed value
    _mask = np.ones(_N, dtype=bool)
    for _d in range(_n_params):
        if _d != _not_fixed_col:
            _mask &= np.isclose(_X_grid[:, _d], float(_fixed_values[_d]))
    _slice_idx = np.where(_mask)[0]

    if len(_slice_idx) == 0:
        qf_diag_figure = mo.callout(
            mo.md("No grid points found for this fixed-parameter combination."),
            kind="warn",
        )
        mo.stop(True, qf_diag_figure)

    # Sort the slice by the varying parameter for consistent plotting
    _slice_x = _X_grid[_slice_idx, _not_fixed_col]
    _slice_w = _Y_true[_slice_idx]
    _sort_order = np.argsort(_slice_x)
    _slice_x = _slice_x[_sort_order]
    _slice_w = _slice_w[_sort_order]

    # Dense prediction line along the varying axis (100 points)
    _x_vals = _unique_per_dim[_not_fixed_col]
    _test_axis = np.linspace(_x_vals.min(), _x_vals.max(), 100)
    _test_pts = np.zeros((100, _n_params), dtype=np.float32)
    for _i in range(_n_params):
        if _i == _not_fixed_col:
            _test_pts[:, _i] = _test_axis
        else:
            _test_pts[:, _i] = float(_fixed_values[_i])

    if _source == "grid_interp":
        _pred_line = np.zeros((100, _n_comp))
        for _c, _interp in enumerate(qf_trained_models):
            _pred_line[:, _c] = _interp(_test_pts)
    else:
        _sc2 = qf_input_scaler
        _test_norm = (_test_pts - _sc2["min"]) / _sc2["range"]
        _Xtt = torch.tensor(_test_norm, dtype=torch.float32)
        _pred_runs = []
        for _m in qf_trained_models:
            _m.eval()
            with torch.no_grad():
                _pr = _m(_Xtt).numpy()
            if qf_output_scaler is not None:
                _pr = _pr * qf_output_scaler["std"] + qf_output_scaler["mean"]
            _pred_runs.append(_pr)
        _pred_line = np.mean(_pred_runs, axis=0)

    # Build Altair charts for selected component range
    _charts = []
    for _comp in range(_comp_start, _comp_end + 1):
        _scatter_df = pd.DataFrame({
            'x': _slice_x,
            'weight': _slice_w[:, _comp],
        })
        _line_df = pd.DataFrame({
            'x': _test_axis,
            'pred': _pred_line[:, _comp],
        })

        # Scatter: actual PCA weights at grid nodes
        _scatter = alt.Chart(_scatter_df).mark_circle(
            size=60, color='#3b82f6', opacity=0.8,
        ).encode(
            x=alt.X('x:Q', title=_xlabel, scale=alt.Scale(zero=False)),
            y=alt.Y('weight:Q', title=f'Weight {_comp}', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('x:Q', title=_xlabel, format='.4f'),
                     alt.Tooltip('weight:Q', title='Weight', format='.4e')],
        )

        # Prediction line: model interpolation / NN prediction
        _line = alt.Chart(_line_df).mark_line(
            color='#f97316', strokeWidth=2,
        ).encode(
            x='x:Q',
            y='pred:Q',
        )

        _chart = (_scatter + _line).properties(
            title=f'PCA Component {_comp}',
            width=650, height=160,
        )

        _charts.append(_chart)

    if _charts:
        _combined = alt.vconcat(*_charts).resolve_scale(x='shared')
    else:
        _combined = alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_point()

    # Fixed parameters info string (like GP diagnostics)
    _fixed_info_parts = []
    for _i in range(_n_params):
        if _i != _not_fixed_col:
            _fixed_info_parts.append(f"{_param_labels[_i]}={float(_fixed_values[_i]):.4g}")
    _fixed_info = ", ".join(_fixed_info_parts)

    _method = 'linear interpolation' if _source == 'grid_interp' else 'NN prediction'
    qf_diag_figure = mo.accordion({
        f"{mo.icon('lucide:microscope')} Weight Space Diagnostics": mo.vstack([
            mo.md(f"Explore the {_method} fit to the PCA weight latent space. "
                  "Blue dots are the actual PCA weights at grid points. "
                  f"Orange line shows the {_method}."),
            mo.hstack([qf_diag_xaxis, qf_diag_fixed_slider], justify="start", gap=1),
            mo.hstack([qf_diag_comp_start, qf_diag_comp_end], justify="start", gap=1),
            mo.md(f"**Fixed parameters:** {_fixed_info}"),
            _combined,
        ])
    })

    qf_diag_figure
    return


@app.cell
def _(
    json,
    mo,
    np,
    os,
    qf_best_hparams,
    qf_grid_selector,
    qf_input_scaler,
    qf_is_hf_mode,
    qf_output_scaler,
    qf_pca_data,
    qf_train_info,
    qf_trained_models,
    set_qf_inf_refresh,
):
    # ── Export Cell: saves trained Quick Fit model to .npz file ───────────
    # Auto-saves on training completion (local mode only; skipped in HF mode
    # to avoid littering the public space).  Manual re-export button provided.
    # Creates a compressed .npz in Grid-Emulator_Files/ containing:
    #   - PCA reconstruction data: eigenspectra, flux_mean, flux_std, wl
    #   - Grid metadata: grid_points, weights, param_names, scale, etc.
    #   - Input scaler: min/max/range for parameter normalisation
    #   - For NN: state dicts + architecture config for each ensemble member
    #   - For Grid Interp: grid_points + weights (enough to rebuild interpolators)
    # Filename convention:
    #   NN (single):   {grid_name}_qfnn_{param_tag}_{scale}[_smooth]_{wlmin}-{wlmax}AA_{n}PCA.npz
    #   NN (ensemble): {grid_name}_qfnn-ensemble_{param_tag}_{scale}[_smooth]_{wlmin}-{wlmax}AA_{n}PCA.npz
    #   Grid Interp:   {grid_name}_qfgi_{param_tag}_{scale}[_smooth]_{wlmin}-{wlmax}AA_{n}PCA.npz
    if qf_trained_models is None or qf_train_info is None or qf_pca_data is None:
        mo.stop(True, mo.md(""))

    _source = qf_train_info["source"]

    def _build_save_dict():
        _save_dict = {
            "model_type": _source,
            "best_hparams": json.dumps(qf_best_hparams or {}),
            "source_grid_file": qf_pca_data["source_grid_file"],
            "eigenspectra": qf_pca_data["eigenspectra"],
            "flux_mean": qf_pca_data["flux_mean"],
            "flux_std": qf_pca_data["flux_std"],
            "wl": qf_pca_data["wl"],
            "grid_points": qf_pca_data["grid_points"],
            "weights": qf_pca_data["weights"],
            "param_names": qf_pca_data["param_names"],
            "n_components": qf_pca_data["n_components"],
            "scale": qf_pca_data["scale"],
            "smoothing": qf_pca_data["smoothing"],
            "selected_param_indices": qf_pca_data["selected_param_indices"],
            "input_scaler_min": qf_input_scaler["min"],
            "input_scaler_max": qf_input_scaler["max"],
            "input_scaler_range": qf_input_scaler["range"],
        }
        if qf_output_scaler is not None:
            _save_dict["output_scaler_mean"] = qf_output_scaler["mean"]
            _save_dict["output_scaler_std"] = qf_output_scaler["std"]

        if _source == "nn":
            _save_dict["n_models"] = len(qf_trained_models)
            for _i, _m in enumerate(qf_trained_models):
                _save_dict[f"model_{_i}_state"] = _m.state_dict()
                _cfg = {
                    "n_in": qf_input_scaler["min"].shape[0],
                    "n_out": _m.net[-1].out_features,
                    "hidden_sizes": qf_train_info.get("hidden_sizes", [128]),
                    "activation": (qf_best_hparams or {}).get("activation", "GELU"),
                    "dropout": (qf_best_hparams or {}).get("dropout", 0.0),
                    "dropout_strategy": (qf_best_hparams or {}).get("dropout_strategy", "uniform"),
                    "batch_norm": (qf_best_hparams or {}).get("batch_norm", False),
                    "skip_connections": (qf_best_hparams or {}).get("skip_connections", False),
                }
                _save_dict[f"model_{_i}_config"] = json.dumps(_cfg)
        return _save_dict

    _emu_dir = "Grid-Emulator_Files"
    os.makedirs(_emu_dir, exist_ok=True)
    if _source == "grid_interp":
        _tag = "qfgi"
    elif _source == "nn" and len(qf_trained_models) > 1:
        _tag = "qfnn-ensemble"
    else:
        _tag = "qfnn"
    _grid_base = qf_grid_selector.value  # e.g. "speculate_cv_no-bl_grid_v87f"
    _param_digits = "".join(str(i) for i in sorted(qf_pca_data["selected_param_indices"]))
    _scale = qf_pca_data["scale"]
    _smooth_suffix = "_smooth" if qf_pca_data["smoothing"] else ""
    _wl_arr = qf_pca_data["wl"]
    _wl_lo, _wl_hi = int(round(_wl_arr.min())), int(round(_wl_arr.max()))
    _n_pca = int(qf_pca_data["n_components"])
    _out_name = f"{_grid_base}_{_tag}_{_param_digits}_{_scale}{_smooth_suffix}_{_wl_lo}-{_wl_hi}AA_{_n_pca}PCA.npz"
    _out_path = os.path.join(_emu_dir, _out_name)

    # Auto-save when running locally (not on HuggingFace)
    _auto_saved = False
    if not qf_is_hf_mode:
        np.savez_compressed(_out_path, **_build_save_dict())
        _auto_saved = True
        # Trigger a refresh of the Stage 3 model selector dropdown
        # so the newly trained model appears immediately.
        set_qf_inf_refresh(lambda n: n + 1)

    def _export_quickfit(_):
        np.savez_compressed(_out_path, **_build_save_dict())
        mo.status.toast(f"Exported to {_out_path}", kind="success")

    _type_label = {"nn": "NN", "grid_interp": "Grid Interp"}.get(_source, _source)

    _elements = []
    if _auto_saved:
        _elements.append(mo.callout(
            mo.md(f"Model auto-saved to `{_out_path}`"),
            kind="success",
        ))
    _export_btn = mo.ui.button(
        label=f"{mo.icon('lucide:save')} {'Re-export' if _auto_saved else 'Export'} Quick Fit Model ({_type_label})",
        kind="success",
        on_click=_export_quickfit,
    )
    _elements.append(_export_btn)
    mo.vstack(_elements)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 3: Inference
    *Load a trained Quick Fit model and observational data, then run chi-squared MLE.*
    """)
    return


@app.cell
def _(mo):
    get_qf_inf_refresh, set_qf_inf_refresh = mo.state(0)
    return get_qf_inf_refresh, set_qf_inf_refresh


@app.cell
def _(get_qf_inf_refresh, mo, os, qf_pretrained_selector):
    _ = get_qf_inf_refresh()
    _emu_dir = "Grid-Emulator_Files"
    _qf_files = {}

    if os.path.exists(_emu_dir):
        for _f in sorted(os.listdir(_emu_dir)):
            if _f.endswith(".npz") and ("_qfnn-ensemble_" in _f or "_qfnn_" in _f or "_qfgi_" in _f):
                _tag = "NN-Ensemble" if "_qfnn-ensemble_" in _f else ("NN" if "_qfnn_" in _f else "GridInterp")
                _qf_files[f"[{_tag}] {_f}"] = _f

    _qf_options = {"— None (select a trained model) —": ""}
    _qf_options.update(_qf_files)

    # If a pre-trained model was selected in Stage 1, auto-select it here.
    _default = "— None (select a trained model) —"
    _pre = qf_pretrained_selector.value if qf_pretrained_selector is not None else ""
    if _pre and _pre != "":
        for _label, _val in _qf_files.items():
            if _val == _pre:
                _default = _label
                break

    qf_inf_model_selector = mo.ui.dropdown(
        options=_qf_options,
        value=_default,
        label="Select Trained Quick Fit Model:",
        full_width=True,
    )
    return (qf_inf_model_selector,)


@app.cell
def _(
    QuickFitNN,
    RegularGridInterpolator,
    json,
    mo,
    np,
    os,
    qf_inf_model_selector,
):
    # ── Stage 3 Model Loader: reconstructs a trained Quick Fit model from .npz ──
    # Loads the exported model file and rebuilds the prediction pipeline.
    # For NN models: instantiates QuickFitNN with saved architecture config,
    #   loads state_dict for each ensemble member.
    # For Grid Interpolation: rebuilds RegularGridInterpolator from saved
    #   grid_points and PCA weights (same construction as in training cell).
    # Also extracts: PCA reconstruction data (eigenspectra, flux_mean/std),
    #   input/output scalers, wavelength array, and parameter metadata.
    # All outputs feed into the inference pipeline downstream.

    # Initialise all outputs to None — marimo cells must always define returns.
    qf_inf_emu_data = None
    qf_inf_models = None
    qf_inf_input_scaler = None
    qf_inf_output_scaler = None
    qf_inf_is_grid_interp = False

    if qf_inf_model_selector.value and qf_inf_model_selector.value != "":
        _emu_dir = "Grid-Emulator_Files"
        _path = os.path.join(_emu_dir, qf_inf_model_selector.value)
        try:
            _npz = np.load(_path, allow_pickle=True)
            _model_type = str(_npz.get("model_type", "nn"))

            # Extract PCA reconstruction data and grid metadata.
            # These are shared by both NN and Grid Interp models.
            qf_inf_emu_data = {
                "grid_points": np.array(_npz["grid_points"]),
                "weights": np.array(_npz["weights"]),
                "eigenspectra": np.array(_npz["eigenspectra"]),
                "flux_mean": np.array(_npz["flux_mean"]),
                "flux_std": np.array(_npz["flux_std"]),
                "wl": np.array(_npz["wl"]),
                "param_names": list(_npz["param_names"]),
                "min_params": np.array(_npz["grid_points"]).min(axis=0),
                "max_params": np.array(_npz["grid_points"]).max(axis=0),
                "source_file": qf_inf_model_selector.value,
                "scale": str(_npz.get("scale", "linear")),
            }

            # Restore input parameter scaler (min-max normalisation).
            if "input_scaler_min" in _npz:
                qf_inf_input_scaler = {
                    "method": "minmax",
                    "min": np.array(_npz["input_scaler_min"]),
                    "max": np.array(_npz["input_scaler_max"]),
                    "range": np.array(_npz["input_scaler_range"]),
                }

            # Restore output scaler (only present if NN used standard output normalisation).
            if "output_scaler_mean" in _npz:
                qf_inf_output_scaler = {
                    "mean": np.array(_npz["output_scaler_mean"]),
                    "std": np.array(_npz["output_scaler_std"]),
                }

            if _model_type == "nn":
                # ── Reconstruct Neural Network ensemble ──────────────────
                # For each saved model: read its architecture config (JSON),
                # instantiate a blank QuickFitNN, load the saved state_dict.
                qf_inf_is_grid_interp = False
                qf_inf_models = []
                _n_models = int(_npz["n_models"])
                for _i in range(_n_models):
                    _state = _npz[f"model_{_i}_state"]
                    _cfg = json.loads(str(_npz[f"model_{_i}_config"]))
                    # Reconstruct with the same dropout config used during
                    # training so that layer indices match the saved state_dict.
                    # .eval() disables dropout and switches BatchNorm to inference mode.
                    # Fall back to best_hparams for files saved before dropout was
                    # included in per-model config.
                    _hp = json.loads(str(_npz.get("best_hparams", "{}"))) if "best_hparams" in _npz else {}
                    _model = QuickFitNN(
                        n_in=_cfg["n_in"], n_out=_cfg["n_out"],
                        hidden_sizes=_cfg["hidden_sizes"],
                        activation_name=_cfg.get("activation", "GELU"),
                        dropout=_cfg.get("dropout", _hp.get("dropout", 0.0)),
                        dropout_strategy=_cfg.get("dropout_strategy", _hp.get("dropout_strategy", "uniform")),
                        batch_norm=_cfg.get("batch_norm", False),
                        skip_connections=_cfg.get("skip_connections", False),
                    )
                    _model.load_state_dict(_state.item())
                    _model.eval()
                    qf_inf_models.append(_model)

            elif _model_type == "grid_interp":
                # ── Reconstruct RegularGridInterpolator from saved grid ───
                # Uses the same construction logic as the training cell:
                # 1. Extract unique values along each parameter axis
                # 2. Lexsort points into Cartesian order
                # 3. Reshape weights into N-D grid and build interpolator
                # One interpolator per PCA component.
                qf_inf_is_grid_interp = True
                _gp = np.array(_npz["grid_points"]).astype(np.float32)
                _wt = np.array(_npz["weights"]).astype(np.float32)
                _n_in = _gp.shape[1]   # number of physical parameters
                _n_out = _wt.shape[1]  # number of PCA components
                _axes = [np.unique(_gp[:, i]) for i in range(_n_in)]
                _sorted_idx = np.lexsort(_gp.T[::-1])
                _wt_sorted = _wt[_sorted_idx]
                _grid_shape = [len(a) for a in _axes]

                qf_inf_models = []
                for _c in range(_n_out):
                    _vals = _wt_sorted[:, _c].reshape(_grid_shape)
                    qf_inf_models.append(RegularGridInterpolator(
                        _axes, _vals, method="linear",
                        bounds_error=False, fill_value=None,
                    ))

            _npz.close()
        except Exception as e:
            mo.output.replace(
                mo.callout(mo.md(f"Error loading model: {e}"), kind="danger")
            )
    return (
        qf_inf_emu_data,
        qf_inf_input_scaler,
        qf_inf_is_grid_interp,
        qf_inf_models,
        qf_inf_output_scaler,
    )


@app.cell
def _(mo):
    get_qf_last_uploaded_obs, set_qf_last_uploaded_obs = mo.state(None)
    get_qf_obs_refresh, set_qf_obs_refresh = mo.state(0)
    return (
        get_qf_last_uploaded_obs,
        get_qf_obs_refresh,
        set_qf_last_uploaded_obs,
        set_qf_obs_refresh,
    )


@app.cell
def _(get_qf_last_uploaded_obs, get_qf_obs_refresh, mo, os):
    _ = get_qf_obs_refresh()
    _obs_dir = "observation_files"
    qf_obs_files = []
    if os.path.exists(_obs_dir):
        qf_obs_files = sorted([f for f in os.listdir(_obs_dir) if f.endswith(('.csv', '.txt', '.dat'))])

    if qf_obs_files:
        _last_uploaded = get_qf_last_uploaded_obs()
        _default_file = _last_uploaded if _last_uploaded in qf_obs_files else qf_obs_files[0]
        qf_obs_selector = mo.ui.dropdown(
            options=qf_obs_files,
            value=_default_file,
            label="Select Observation File:",
            full_width=True,
        )
    else:
        qf_obs_selector = mo.ui.dropdown(
            options={"— No observation files found —": ""},
            value="— No observation files found —",
            label="Select Observation File:",
            full_width=True,
        )

    qf_obs_uploader = mo.ui.file(
        kind="area",
        label="**Drag and drop an observational spectrum (CSV)**",
        multiple=False,
    )
    return qf_obs_selector, qf_obs_uploader


@app.cell
def _(mo, os, qf_obs_uploader, set_qf_last_uploaded_obs, set_qf_obs_refresh):
    if qf_obs_uploader.value:
        _obs_dir = "observation_files"
        os.makedirs(_obs_dir, exist_ok=True)

        uploaded_file = qf_obs_uploader.value[0]
        file_path = os.path.join(_obs_dir, uploaded_file.name)

        try:
            with open(file_path, "wb") as _f:
                _f.write(uploaded_file.contents)

            set_qf_last_uploaded_obs(uploaded_file.name)
            set_qf_obs_refresh(lambda v: v + 1)
            mo.status.toast(f"{mo.icon('lucide:check-circle')} Uploaded {uploaded_file.name}")
        except Exception as e:
            mo.status.toast(f"{mo.icon('lucide:x-circle')} Upload failed: {str(e)}")
    return


@app.cell
def _(mo, np, os, pd, qf_obs_selector):
    qf_obs_data = None
    _obs_dir = "observation_files"

    if qf_obs_selector.value and qf_obs_selector.value != "":
        try:
            _path = os.path.join(_obs_dir, qf_obs_selector.value)
            _df = pd.read_csv(_path)
            _df.columns = _df.columns.astype(str).str.lower()

            if 'wavelength' in _df.columns and 'flux' in _df.columns:
                _df = _df.sort_values('wavelength', ascending=True).reset_index(drop=True)
                if 'error' not in _df.columns:
                    _df['error'] = np.abs(_df['flux']) * 0.05
                qf_obs_data = _df
            else:
                mo.output.replace(
                    mo.callout(mo.md("CSV must have 'Wavelength' and 'Flux' columns."), kind="danger")
                )
        except Exception as e:
            mo.output.replace(
                mo.callout(mo.md(f"Error reading file: {e}"), kind="danger")
            )
    return (qf_obs_data,)


@app.cell
def _(mo, qf_inf_emu_data, qf_obs_data):
    if qf_inf_emu_data is not None and "wl" in qf_inf_emu_data:
        _wl_min = float(qf_inf_emu_data["wl"].min()) + 10
        _wl_max = float(qf_inf_emu_data["wl"].max()) - 10
    elif qf_obs_data is not None:
        _wl_min = float(qf_obs_data['wavelength'].min())
        _wl_max = float(qf_obs_data['wavelength'].max())
    else:
        _wl_min, _wl_max = 800.0, 8000.0

    qf_inf_wl_slider = mo.ui.range_slider(
        start=_wl_min, stop=_wl_max,
        value=[_wl_min, _wl_max],
        step=1.0,
        label="Wavelength Range (Å):",
        show_value=True, full_width=True,
    )
    return (qf_inf_wl_slider,)


@app.cell
def _(mo, qf_inf_emu_data, qf_inf_model_selector):
    # Auto-detect the emulator's training scale from its metadata so the
    # observation transform dropdown starts on a sensible default.  Users can
    # override this when their observation is already pre-processed (e.g. a
    # pre-computed continuum-normalised spectrum would just use "linear").
    _detected_scale = "linear"
    if qf_inf_emu_data is not None:
        _detected_scale = qf_inf_emu_data.get("scale", "linear")
    elif qf_inf_model_selector is not None and qf_inf_model_selector.value:
        _name = qf_inf_model_selector.value.lower()
        if "_log_" in _name:
            _detected_scale = "log"
        elif "_continuum-normalised_" in _name:
            _detected_scale = "continuum-normalised"

    # Build dropdown options: the emulator's training scale (which auto-
    # transforms the raw observation to match) and "linear" (no transform,
    # for observations that are already pre-processed into the right space).
    # Offering ALL four scales would let users pick an incompatible one
    # (e.g. "log" observation vs "continuum-normalised" emulator), which
    # produces garbage that no nuisance parameter can fix.
    if _detected_scale == "linear":
        _options = ["linear"]
    else:
        _options = [_detected_scale, "linear"]

    qf_obs_scale_selector = mo.ui.dropdown(
        options=_options,
        value=_detected_scale,
        label="Observation Flux Transform:",
        full_width=True,
    )
    return (qf_obs_scale_selector,)


@app.cell
def _(
    json,
    mo,
    np,
    os,
    qf_inf_emu_data,
    qf_inf_is_grid_interp,
    qf_inf_model_selector,
    qf_inf_models,
    qf_inf_wl_slider,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_obs_selector,
    qf_obs_uploader,
):
    _model_status = mo.md("")
    if qf_inf_models is not None and qf_inf_emu_data is not None:
        _n_models = len(qf_inf_models)
        _n_params = qf_inf_emu_data["grid_points"].shape[1]
        _wl = qf_inf_emu_data.get("wl", None)
        _wl_info = f"{_wl.min():.0f} – {_wl.max():.0f} Å" if _wl is not None else "N/A"
        _scale = qf_inf_emu_data.get("scale", "linear")

        _nn_info = ""
        if not qf_inf_is_grid_interp:
            try:
                _npz = np.load(os.path.join("Grid-Emulator_Files", qf_inf_model_selector.value), allow_pickle=True)
                _cfg = json.loads(str(_npz["model_0_config"]))
                _npz.close()
                _hs = _cfg.get("hidden_sizes", [])
                _act = _cfg.get("activation", "?")
                _do = _cfg.get("dropout", 0.0)
                _bn = _cfg.get("batch_norm", False)
                _skip = _cfg.get("skip_connections", False)
                _total_params = sum(p.numel() for p in qf_inf_models[0].parameters())
                _nn_info = (
                    f"\n            - **Architecture:** {_hs} | **Activation:** {_act}"
                    f"\n            - **Ensemble:** {_n_models} members | **Params/model:** {_total_params:,}"
                    f"\n            - **Dropout:** {_do} | **BatchNorm:** {_bn} | **Skip:** {_skip}"
                )
            except Exception:
                pass

        _model_status = mo.callout(
            mo.md(f"""
            **Model Loaded:** `{qf_inf_model_selector.value}`
            - **{'Ensemble NN' if not qf_inf_is_grid_interp else 'Grid Interpolation'}** | **Parameters:** {_n_params} | **Scale:** {_scale}
            - **Wavelength:** {_wl_info}{_nn_info}
            """),
            kind="success"
        )

    _obs_status = mo.md("")
    if qf_obs_data is not None:
        _obs_status = mo.callout(
            mo.md(f"""
            **Observation Loaded:** `{qf_obs_selector.value}`
            - **Points:** {len(qf_obs_data)}
            """),
            kind="success"
        )

    mo.vstack([
        qf_inf_model_selector,
        _model_status,
        mo.md("---"),
        qf_obs_selector,
        qf_obs_uploader,
        mo.hstack([qf_inf_wl_slider, qf_obs_scale_selector], widths=[3, 1], align="end"),
        mo.md("---"),
        _obs_status,
    ])
    return


@app.cell
def _(alt, mo, np, pd, qf_inf_emu_data, qf_inf_wl_slider, qf_obs_data, qf_obs_scale_selector, qf_transform_observation_data):
    # ── View Selected Spectrum: preview the loaded observation in the current
    # wavelength window, transformed to the user-selected flux scale.
    _obs_chart = None
    if qf_obs_data is not None and qf_inf_emu_data is not None:
        _wl_lo, _wl_hi = qf_inf_wl_slider.value
        _mask = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
        _plot_df = qf_obs_data[_mask].copy()

        _scale = qf_obs_scale_selector.value
        _obs_wl_v = np.array(_plot_df['wavelength'])
        _obs_fl_v = np.array(_plot_df['flux'])
        _obs_err_v = np.array(_plot_df['error']) if 'error' in _plot_df.columns else None
        _flux_vals, _ = qf_transform_observation_data(_obs_wl_v, _obs_fl_v, _obs_err_v, _scale)

        _plot_df = pd.DataFrame({
            'Wavelength': _obs_wl_v,
            'Flux': _flux_vals,
            'Type': 'Observation',
        })

        if len(_plot_df) > 5000:
            _plot_df = _plot_df.iloc[::max(1, len(_plot_df) // 5000)]

        _y_title = f'Flux ({_scale})'
        _y_format = '.1e' if _scale == 'linear' else '.2f'

        _obs_chart = alt.Chart(_plot_df).mark_line(color='cyan').encode(
            x=alt.X('Wavelength:Q', title='Wavelength (Å)'),
            y=alt.Y('Flux:Q', title=_y_title, axis=alt.Axis(format=_y_format)),
            tooltip=['Wavelength', 'Flux'],
        ).properties(width="container", height=400, title="Observation Spectrum").interactive()

    mo.accordion({
        f"{mo.icon('lucide:trending-up')} View Selected Spectrum":
            _obs_chart if _obs_chart else mo.md("*Load a model and observation to preview.*")
    })
    return


@app.cell
def _(np, torch):
    def qf_predict_flux(models, params, emu_data, input_scaler, output_scaler=None, is_grid_interp=False):
        X_raw = np.array(params, dtype=np.float32).reshape(1, -1)

        eigenspectra = emu_data["eigenspectra"]
        flux_mean = emu_data["flux_mean"]
        flux_std = emu_data["flux_std"]

        if is_grid_interp:
            # RegularGridInterpolator uses raw parameter values
            w = np.array([m(X_raw)[0] for m in models])
            flux = (w @ (eigenspectra * flux_std)) + flux_mean
            return flux, [flux]
        else:
            X_norm = (X_raw - input_scaler["min"]) / input_scaler["range"]
            Xt = torch.tensor(X_norm, dtype=torch.float32)
            fluxes = []
            for m in models:
                m.eval()
                with torch.no_grad():
                    w = m(Xt).numpy()[0]
                if output_scaler is not None:
                    w = w * output_scaler["std"] + output_scaler["mean"]
                flux = (w @ (eigenspectra * flux_std)) + flux_mean
                fluxes.append(flux)
            mean_flux = np.mean(fluxes, axis=0)
            return mean_flux, fluxes
    return (qf_predict_flux,)


@app.cell
def _(
    mo,
    np,
    param_map_db,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_predict_flux,
    qf_transform_observation_data,
    re,
):
    qf_param_config = None

    if qf_inf_emu_data is None:
        mo.stop(True, mo.md("*Select a trained model above.*"))

    _min_p = qf_inf_emu_data["min_params"]
    _max_p = qf_inf_emu_data["max_params"]
    _param_names = qf_inf_emu_data["param_names"]
    _n_params = len(_param_names)

    _fixed_toggles = []
    _value_inputs = []
    _min_inputs = []
    _max_inputs = []
    _labels = []

    for _i in range(_n_params):
        _name = _param_names[_i]
        _m = re.search(r"(\d+)", _name)
        _idx = int(_m.group(1)) if _m else 0
        _base = param_map_db[_idx][0] if _idx in param_map_db else _name
        _is_log = param_map_db[_idx][3] if _idx in param_map_db and len(param_map_db[_idx]) > 3 else False
        _display = f"log10({_base})" if _is_log else _base

        _lo = float(_min_p[_i])
        _hi = float(_max_p[_i])
        _mid = (_lo + _hi) / 2.0
        _labels.append(_display)

        _fixed_toggles.append(mo.ui.checkbox(label="Fix", value=False))
        _value_inputs.append(mo.ui.number(value=round(_mid, 4), step=0.001, label="Value"))
        _min_inputs.append(mo.ui.number(value=round(_lo, 4), step=0.001, label="Min"))
        _max_inputs.append(mo.ui.number(value=round(_hi, 4), step=0.001, label="Max"))

    # ── Auto-configure nuisance parameter defaults ──
    # Estimate log_scale from the ratio of observation to emulator midpoint flux
    # so the MLE starts with both spectra at roughly the same scale.
    _ls_default = 0.0
    if qf_obs_data is not None and qf_inf_models is not None:
        try:
            _wl_lo, _wl_hi = qf_inf_wl_slider.value
            _mid_params = (_min_p + _max_p) / 2.0
            _emu_fl, _ = qf_predict_flux(
                qf_inf_models, _mid_params, qf_inf_emu_data,
                qf_inf_input_scaler, qf_inf_output_scaler,
                is_grid_interp=qf_inf_is_grid_interp,
            )
            _emu_wl = qf_inf_emu_data["wl"]
            _em = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)
            _emu_mean = np.mean(np.abs(_emu_fl[_em])) if _em.any() else 1.0

            _om = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
            _obs_wl = np.array(qf_obs_data[_om]['wavelength'])
            _raw_fl = np.array(qf_obs_data[_om]['flux'])
            _raw_err = np.array(qf_obs_data[_om]['error']) if 'error' in qf_obs_data.columns else None
            _scale = qf_obs_scale_selector.value
            _obs_fl, _ = qf_transform_observation_data(_obs_wl, _raw_fl, _raw_err, _scale)
            # Auto-centre log_scale.  When the emulator outputs log₁₀ flux,
            # the transformed means are negative and their ratio is meaningless.
            # Convert both back to linear space to get a sensible scaling offset.
            if _scale == "log":
                _emu_mean_lin = np.mean(np.abs(10.0 ** _emu_fl[_em])) if _em.any() else 1.0
                _obs_mean_lin = np.mean(np.abs(_raw_fl)) if len(_raw_fl) > 0 else 1.0
                if _emu_mean_lin > 0 and _obs_mean_lin > 0:
                    _ls_default = round(float(np.log(_obs_mean_lin / _emu_mean_lin)), 1)
            else:
                _obs_mean = np.mean(np.abs(_obs_fl)) if len(_obs_fl) > 0 else 1.0
                if _emu_mean > 0 and _obs_mean > 0:
                    _ls_default = round(float(np.log(_obs_mean / _emu_mean)), 1)
        except Exception:
            pass

    _nuisance = [
        ("Av",        0.0,         0.0,               2.0),
        ("log_scale", _ls_default, _ls_default - 5.0, _ls_default + 5.0),
        ("cheb_1",    0.0,         -0.5,              0.5),
    ]
    for _name, _val, _lo, _hi in _nuisance:
        _labels.append(_name)
        _fixed_toggles.append(mo.ui.checkbox(label="Fix", value=(_name == "cheb_1")))
        _value_inputs.append(mo.ui.number(value=round(_val, 4), step=0.01, label="Value"))
        _min_inputs.append(mo.ui.number(value=round(_lo, 4), step=0.01, label="Min"))
        _max_inputs.append(mo.ui.number(value=round(_hi, 4), step=0.01, label="Max"))

    qf_param_config = {
        "labels": _labels,
        "fixed": _fixed_toggles,
        "values": _value_inputs,
        "mins": _min_inputs,
        "maxs": _max_inputs,
        "n_physical": _n_params,
    }

    _config_elements = []
    for _i, _name in enumerate(_labels):
        _row = mo.hstack([
            mo.md(f"**{_name}**"),
            _fixed_toggles[_i],
            _value_inputs[_i],
            _min_inputs[_i],
            _max_inputs[_i],
        ], justify="start", gap="0.5rem")
        _config_elements.append(_row)

    _config_accordion = mo.accordion({
        f"{mo.icon('lucide:sliders-horizontal')} Parameter Configuration": mo.vstack(_config_elements)
    }, lazy=True)

    _config_accordion
    return (qf_param_config,)


@app.cell
def _(
    mo,
    np,
    param_map_db,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_predict_flux,
    qf_transform_observation_data,
    re,
):
    # ── Parameter Playground ──
    # Interactive sliders to preview how Av, log_scale, cheb_1, and the
    # physical grid axes shift the emulated spectrum relative to the
    # observation. Helps find sensible prior bounds before running MLE.

    # Default slider widgets — created unconditionally so downstream cells
    # always receive valid references even before a model is loaded.
    qf_pg_av = mo.ui.slider(start=0.0, stop=5.0, value=0.0, step=0.01,
                             label="Av (Extinction)", show_value=True, full_width=True)
    qf_pg_logscale = mo.ui.slider(start=-10.0, stop=10.0, value=0.0, step=0.1,
                                   label="log_scale (ln flux scaling)", show_value=True, full_width=True)
    qf_pg_cheb1 = mo.ui.slider(start=-2.0, stop=2.0, value=0.0, step=0.01,
                                label="cheb_1 (continuum tilt)", show_value=True, full_width=True)

    if qf_inf_emu_data is None or qf_inf_models is None or qf_obs_data is None:
        mo.stop(True, mo.md(""))

    _min_p = qf_inf_emu_data["min_params"]
    _max_p = qf_inf_emu_data["max_params"]
    _mid_params = (_min_p + _max_p) / 2.0
    _wl_lo, _wl_hi = qf_inf_wl_slider.value

    # Auto-detect sensible log_scale centre from flux ratio.
    _ls_centre = 0.0
    try:
        _emu_fl, _ = qf_predict_flux(
            qf_inf_models, _mid_params, qf_inf_emu_data,
            qf_inf_input_scaler, qf_inf_output_scaler,
            is_grid_interp=qf_inf_is_grid_interp,
        )
        _emu_wl = qf_inf_emu_data["wl"]
        _em = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)
        _emu_mean = np.mean(np.abs(_emu_fl[_em])) if _em.any() else 1.0

        _om = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
        _obs_wl_pg = np.array(qf_obs_data[_om]['wavelength'])
        _raw_fl_pg = np.array(qf_obs_data[_om]['flux'])
        _raw_err_pg = np.array(qf_obs_data[_om]['error']) if 'error' in qf_obs_data.columns else None
        _scale_pg = qf_obs_scale_selector.value
        _obs_fl_pg, _ = qf_transform_observation_data(_obs_wl_pg, _raw_fl_pg, _raw_err_pg, _scale_pg)
        if _scale_pg == "log":
            _emu_mean_lin = np.mean(np.abs(10.0 ** _emu_fl[_em])) if _em.any() else 1.0
            _obs_mean_lin = np.mean(np.abs(_raw_fl_pg)) if len(_raw_fl_pg) > 0 else 1.0
            if _emu_mean_lin > 0 and _obs_mean_lin > 0:
                _ls_centre = round(float(np.log(_obs_mean_lin / _emu_mean_lin)), 1)
        else:
            _obs_mean = np.mean(np.abs(_obs_fl_pg)) if len(_obs_fl_pg) > 0 else 1.0
            if _emu_mean > 0 and _obs_mean > 0:
                _ls_centre = round(float(np.log(_obs_mean / _emu_mean)), 1)
    except Exception:
        pass

    qf_pg_av = mo.ui.slider(start=0.0, stop=5.0, value=0.0, step=0.01,
                             label="Av (Extinction)", show_value=True, full_width=True)
    qf_pg_logscale = mo.ui.slider(start=_ls_centre - 10.0, stop=_ls_centre + 10.0,
                                   value=_ls_centre, step=0.1,
                                   label="log_scale (ln flux scaling)", show_value=True, full_width=True)
    qf_pg_cheb1 = mo.ui.slider(start=-2.0, stop=2.0, value=0.0, step=0.01,
                                label="cheb_1 (continuum tilt)", show_value=True, full_width=True)

    # ── Physical parameter sliders ──
    # One slider per emulator axis so the user can reshape the emulated
    # spectrum. Ranges span the emulator's training grid; defaults sit at
    # the midpoint (matching the reference point used for auto-centring).
    # Labels are resolved through param_map_db so users see physical names
    # (e.g. "disk.mdot") instead of generic "paramN" tags, and a log10()
    # wrapper is added for parameters stored on a log10 axis.
    _qf_phys_sliders = []
    _qf_param_names = qf_inf_emu_data["param_names"]
    for _pi, _pname in enumerate(_qf_param_names):
        _lo = float(_min_p[_pi])
        _hi = float(_max_p[_pi])
        _mid = (_lo + _hi) / 2.0
        _step = max((_hi - _lo) / 200.0, 1e-6)

        _m = re.search(r"(\d+)", str(_pname))
        _db_idx = int(_m.group(1)) if _m else None
        if _db_idx is not None and _db_idx in param_map_db:
            _base_name = param_map_db[_db_idx][0]
            _is_log = (len(param_map_db[_db_idx]) > 3
                       and bool(param_map_db[_db_idx][3]))
            _label = f"log10({_base_name})" if _is_log else _base_name
        else:
            _label = str(_pname)

        _qf_phys_sliders.append(mo.ui.slider(
            start=_lo, stop=_hi, value=_mid, step=_step,
            label=_label, show_value=True, full_width=True,
        ))
    qf_pg_phys_sliders = mo.ui.array(_qf_phys_sliders)
    return qf_pg_av, qf_pg_cheb1, qf_pg_logscale, qf_pg_phys_sliders


@app.cell
def _(
    alt,
    chebval,
    mo,
    np,
    pd,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_pg_av,
    qf_pg_cheb1,
    qf_pg_logscale,
     qf_pg_nll_result,
    qf_pg_phys_sliders,
    qf_pg_test_nll_btn,
    qf_predict_flux,
    qf_transform_observation_data,
):
    # ── Playground chart: reactive Altair overlay of emulated spectrum
    # (at user-selected physical parameters with current nuisance slider
    # values) vs observation.
    _pg_chart = None

    if qf_inf_emu_data is not None and qf_inf_models is not None and qf_obs_data is not None:
        try:
            _min_p = qf_inf_emu_data["min_params"]
            _max_p = qf_inf_emu_data["max_params"]
            # Physical parameter vector — from sliders when available, otherwise
            # fall back to the training-grid midpoint.
            if qf_pg_phys_sliders is not None:
                _mid_params = np.array(qf_pg_phys_sliders.value, dtype=float)
            else:
                _mid_params = (_min_p + _max_p) / 2.0
            _wl_lo, _wl_hi = qf_inf_wl_slider.value

            _emu_fl, _ = qf_predict_flux(
                qf_inf_models, _mid_params, qf_inf_emu_data,
                qf_inf_input_scaler, qf_inf_output_scaler,
                is_grid_interp=qf_inf_is_grid_interp,
            )
            _emu_wl = qf_inf_emu_data["wl"]
            _em = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)
            _emu_wl_c = _emu_wl[_em]
            _emu_fl_c = _emu_fl[_em].copy()

            # Apply nuisance transforms
            # When the emulator outputs log₁₀(flux), convert to linear
            # before applying nuisance transforms, then convert back.
            _is_log_scale = (qf_obs_scale_selector.value == "log")
            if _is_log_scale:
                _emu_fl_c = 10.0 ** _emu_fl_c

            _av = qf_pg_av.value
            if _av > 0:
                try:
                    from extinction import fitzpatrick99
                    _a_lambda = fitzpatrick99(_emu_wl_c.astype(np.float64), _av, 3.1)
                    _emu_fl_c = _emu_fl_c * 10 ** (-0.4 * _a_lambda)
                except ImportError:
                    pass

            _cheb1 = qf_pg_cheb1.value

            if _cheb1 != 0.0:
                _scale_wl = _emu_wl_c / _emu_wl_c.max()
                _emu_fl_c = _emu_fl_c * chebval(_scale_wl, [1.0, _cheb1])

            _emu_fl_c = _emu_fl_c * np.exp(qf_pg_logscale.value)

            if _is_log_scale:
                _emu_fl_c = np.log10(np.clip(_emu_fl_c, 1e-30, None))

            # Observation in emulator scale
            _om = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
            _obs_wl_c = np.array(qf_obs_data[_om]['wavelength'])
            _raw_fl_c = np.array(qf_obs_data[_om]['flux'])
            _raw_err_c = np.array(qf_obs_data[_om]['error']) if 'error' in qf_obs_data.columns else None
            _scale_c = qf_obs_scale_selector.value
            _obs_fl_c, _ = qf_transform_observation_data(_obs_wl_c, _raw_fl_c, _raw_err_c, _scale_c)

            # Build chart
            _obs_df = pd.DataFrame({'Wavelength': _obs_wl_c, 'Flux': _obs_fl_c, 'Type': 'Observation'})
            _emu_df = pd.DataFrame({'Wavelength': _emu_wl_c, 'Flux': _emu_fl_c, 'Type': 'Emulated (midpoint)'})

            if len(_obs_df) > 5000:
                _obs_df = _obs_df.iloc[::max(1, len(_obs_df) // 5000)]
            if len(_emu_df) > 5000:
                _emu_df = _emu_df.iloc[::max(1, len(_emu_df) // 5000)]

            _combined = pd.concat([_obs_df, _emu_df], ignore_index=True)
            _color_scale = alt.Scale(domain=['Observation', 'Emulated (midpoint)'], range=['cyan', 'orange'])
            _y_format = '.1e' if _scale_c == 'linear' else '.2f'

            _pg_chart = alt.Chart(_combined).mark_line(opacity=0.9).encode(
                x=alt.X('Wavelength:Q', title='Wavelength (Å)'),
                y=alt.Y('Flux:Q', title=f'Flux ({_scale_c})', axis=alt.Axis(format=_y_format)),
                color=alt.Color('Type:N', scale=_color_scale),
                tooltip=['Wavelength:Q', 'Flux:Q', 'Type:N'],
            ).properties(width="container", height=400,
                         title=f"Playground: Emulated (midpoint) vs Observation [{_scale_c}]"
            ).interactive()
        except Exception as _e:
            _pg_chart = mo.callout(mo.md(f"Playground error: {_e}"), kind="danger")

    if _pg_chart is not None:
        _nuisance_tab = mo.vstack([
            qf_pg_av,
            qf_pg_logscale,
            qf_pg_cheb1,
        ])
        if qf_pg_phys_sliders is not None and len(qf_pg_phys_sliders) > 0:
            _physical_tab = mo.vstack(list(qf_pg_phys_sliders))
        else:
            _physical_tab = mo.md("*No physical parameters available.*")
        _slider_tabs = mo.ui.tabs({
            "Nuisance Parameters": _nuisance_tab,
            "Physical Parameters": _physical_tab,
        })
        _pg_content = mo.vstack([
            mo.callout(mo.md(
                "**Parameter Playground — Find sensible prior settings**\n\n"
                "Use the **Nuisance Parameters** tab to bring the emulated spectrum "
                "to a similar scale as the observation, and the **Physical Parameters** "
                "tab to reshape the emulated spectrum by varying the grid axes.\n\n"
                "- **Av** — Interstellar dust extinction (magnitudes)\n"
                "- **log_scale** — Natural log of flux scaling factor\n"
                "- **cheb_1** — Chebyshev c₁ coefficient (continuum tilt)"
            ), kind="neutral"),
            _slider_tabs,
            _pg_chart,
            qf_pg_test_nll_btn,
            qf_pg_nll_result,
        ])
    else:
        _pg_content = mo.md("*Load a model and observation to use the playground.*")

    mo.accordion({
        "Parameter Playground — Aid to find optimal prior settings": _pg_content
    })
    return


@app.cell
def _(mo):
    # Run button for on-demand NLL evaluation from the playground sliders.
    # Kept in its own cell so the evaluator below picks up click events
    # reactively (a run_button's `.value` only flips in downstream cells).
    qf_pg_test_nll_btn = mo.ui.run_button(
        label=f"{mo.icon('lucide:calculator')} Test NLL"
    )
    return (qf_pg_test_nll_btn,)


@app.cell
def _(
    chebval,
    mo,
    np,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_pg_av,
    qf_pg_cheb1,
    qf_pg_logscale,
    qf_pg_phys_sliders,
    qf_pg_test_nll_btn,
    qf_predict_flux,
    qf_transform_observation_data,
):
    # ── Test NLL ──
    # On click, replicate the Quick Fit MLE objective (χ² = Σ((model −
    # obs)/σ)²) at the current playground slider values, so the user can
    # sanity-check whether their chosen prior region is near a reasonable
    # fit before committing to a full MLE run.
    qf_pg_nll_result = mo.md("")

    if qf_pg_test_nll_btn.value:
        if (qf_inf_emu_data is None or qf_inf_models is None
                or qf_obs_data is None or qf_pg_phys_sliders is None):
            qf_pg_nll_result = mo.callout(
                mo.md("Load a model and observation first."),
                kind="warn",
            )
        else:
            try:
                _wl_lo, _wl_hi = qf_inf_wl_slider.value
                _obs_mask = ((qf_obs_data['wavelength'] >= _wl_lo)
                             & (qf_obs_data['wavelength'] <= _wl_hi))
                _obs_wl = np.array(qf_obs_data[_obs_mask]['wavelength'])
                _raw_fl = np.array(qf_obs_data[_obs_mask]['flux'])
                _raw_err = (np.array(qf_obs_data[_obs_mask]['error'])
                            if 'error' in qf_obs_data.columns else None)

                _scale = qf_obs_scale_selector.value
                _obs_fl, _obs_err = qf_transform_observation_data(
                    _obs_wl, _raw_fl, _raw_err, _scale
                )

                _phys = np.array(qf_pg_phys_sliders.value, dtype=float)
                _model_fl, _ = qf_predict_flux(
                    qf_inf_models, _phys, qf_inf_emu_data,
                    qf_inf_input_scaler, qf_inf_output_scaler,
                    is_grid_interp=qf_inf_is_grid_interp,
                )
                _emu_wl = qf_inf_emu_data["wl"]
                _em = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)
                _fl = _model_fl[_em].copy()
                _wl = _emu_wl[_em]

                if _scale == "log":
                    _fl = 10.0 ** _fl

                _av = float(qf_pg_av.value)
                if _av > 0:
                    try:
                        from extinction import fitzpatrick99
                        _a_lambda = fitzpatrick99(
                            _wl.astype(np.float64), _av, 3.1
                        )
                        _fl = _fl * 10 ** (-0.4 * _a_lambda)
                    except ImportError:
                        pass

                _cheb1 = float(qf_pg_cheb1.value)
                if _cheb1 != 0.0:
                    _scale_wl = _wl / _wl.max()
                    _fl = _fl * chebval(_scale_wl, [1.0, _cheb1])

                _fl = _fl * np.exp(float(qf_pg_logscale.value))

                if _scale == "log":
                    _fl = np.log10(np.clip(_fl, 1e-30, None))

                _model_interp = np.interp(_obs_wl, _wl, _fl)
                _resid = (_model_interp - _obs_fl) / _obs_err
                _chi2_val = float(np.sum(_resid ** 2))
                # Gaussian NLL (up to an additive constant independent of
                # parameters): NLL = 0.5·χ² + 0.5·Σ ln(2π σ²).
                _n = len(_obs_err)
                _const = 0.5 * float(
                    np.sum(np.log(2.0 * np.pi * _obs_err ** 2))
                )
                _nll_val = 0.5 * _chi2_val + _const

                qf_pg_nll_result = mo.callout(
                    mo.md(
                        f"**NLL = {_nll_val:.4f}** "
                        f"(χ² = {_chi2_val:.4f}, "
                        f"reduced χ² = {_chi2_val / max(_n, 1):.4f})\n\n"
                        f"Evaluated at current playground slider values "
                        f"over {_n} observation points."
                    ),
                    kind="success" if np.isfinite(_nll_val) else "danger",
                )
            except Exception as _e:
                qf_pg_nll_result = mo.callout(
                    mo.md(f"NLL evaluation failed: `{_e}`"),
                    kind="danger",
                )

    return (qf_pg_nll_result,)


@app.cell
def _(mo):
    qf_opt_method = mo.ui.dropdown(
        options=["CMA-ES", "Nelder-Mead", "L-BFGS-B"],
        value="CMA-ES",
        label="Optimizer:",
    )
    qf_max_iter = mo.ui.number(
        value=20000, start=100, stop=50000, step=100,
        label="Max Iterations:",
    )
    qf_restarts = mo.ui.number(
        value=3, start=1, stop=10, step=1,
        label="Restarts:",
    )
    qf_run_btn = mo.ui.run_button(label=f"{mo.icon('lucide:play')} Run Quick Fit")

    mo.vstack([
        mo.hstack([qf_opt_method, qf_max_iter, qf_restarts, qf_run_btn], justify="start", align="end"),
        mo.callout(mo.md(
            "**Restarts** runs the optimiser multiple times from different random "
            "starting points and keeps the best result. This greatly reduces the "
            "chance of getting stuck in a local minimum. The first run uses the "
            "configured initial values; subsequent runs start from random points "
            "within the parameter bounds."
        ), kind="neutral"),
    ])
    return qf_max_iter, qf_opt_method, qf_restarts, qf_run_btn


@app.cell
def _(
    chebval,
    mo,
    np,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_max_iter,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_opt_method,
    qf_param_config,
    qf_predict_flux,
    qf_restarts,
    qf_run_btn,
    qf_transform_observation_data,
    scipy_minimize,
    time_mod,
):
    qf_mle_result = None

    mo.stop(not qf_run_btn.value)

    if qf_inf_models is None:
        mo.stop(True, mo.callout(mo.md("Select a trained model first."), kind="warn"))
    if qf_obs_data is None:
        mo.stop(True, mo.callout(mo.md("Load observation data first."), kind="warn"))
    if qf_param_config is None:
        mo.stop(True, mo.md(""))

    _labels = qf_param_config["labels"]
    _fixed = [f.value for f in qf_param_config["fixed"]]
    _vals = [v.value for v in qf_param_config["values"]]
    _mins = [m.value for m in qf_param_config["mins"]]
    _maxs = [m.value for m in qf_param_config["maxs"]]
    _n_phys = qf_param_config["n_physical"]

    _active_idx = [i for i in range(len(_labels)) if not _fixed[i]]
    _active_labels = [_labels[i] for i in _active_idx]

    _p0 = np.array([_vals[i] for i in _active_idx], dtype=np.float64)
    _lo = np.array([_mins[i] for i in _active_idx], dtype=np.float64)
    _hi = np.array([_maxs[i] for i in _active_idx], dtype=np.float64)
    # Clamp initial guess to lie within bounds.
    _p0 = np.clip(_p0, _lo, _hi)

    _wl_lo, _wl_hi = qf_inf_wl_slider.value
    _obs_mask = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
    _obs_wl = np.array(qf_obs_data[_obs_mask]['wavelength'])
    _raw_obs_fl = np.array(qf_obs_data[_obs_mask]['flux'])
    _raw_obs_err = np.array(qf_obs_data[_obs_mask]['error']) if 'error' in qf_obs_data.columns else None

    if _obs_wl.size == 0:
        mo.stop(True, mo.callout(mo.md("No observation points fall within the selected wavelength range."), kind="warn"))

    _scale = qf_obs_scale_selector.value
    _obs_fl, _obs_err = qf_transform_observation_data(_obs_wl, _raw_obs_fl, _raw_obs_err, _scale)

    _emu_wl = qf_inf_emu_data["wl"]
    _emu_mask = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)

    if not np.any(_emu_mask):
        mo.stop(True, mo.callout(mo.md("No emulator wavelengths fall within the selected wavelength range."), kind="warn"))

    def _chi2(active_params):
        # ── Enforce prior bounds ─────────────────────────────────────────
        # If any free parameter is outside its [min, max] range, return a
        # large penalty so the optimiser rejects this region of parameter
        # space.  This is essential for Nelder-Mead which does not support
        # box constraints natively, and acts as a safety net for L-BFGS-B.
        for _ai_b in range(len(active_params)):
            if active_params[_ai_b] < _lo[_ai_b] or active_params[_ai_b] > _hi[_ai_b]:
                # Penalty proportional to distance outside bounds
                _dist = 0.0
                for _j in range(len(active_params)):
                    if active_params[_j] < _lo[_j]:
                        _dist += (_lo[_j] - active_params[_j]) ** 2
                    elif active_params[_j] > _hi[_j]:
                        _dist += (active_params[_j] - _hi[_j]) ** 2
                return 1e20 + 1e10 * _dist

        full_params = np.array(_vals, dtype=np.float64)
        for _ai, _gi in enumerate(_active_idx):
            full_params[_gi] = active_params[_ai]

        phys = full_params[:_n_phys]
        _av = full_params[_n_phys] if len(full_params) > _n_phys else 0.0
        _log_scale = full_params[_n_phys + 1] if len(full_params) > _n_phys + 1 else 0.0
        _cheb1 = full_params[_n_phys + 2] if len(full_params) > _n_phys + 2 else 0.0

        model_flux, _ = qf_predict_flux(
            qf_inf_models, phys, qf_inf_emu_data,
            qf_inf_input_scaler, qf_inf_output_scaler, is_grid_interp=qf_inf_is_grid_interp,
        )

        _fl = model_flux[_emu_mask]
        _wl = _emu_wl[_emu_mask]

        # Convert log₁₀ emulator output to linear before nuisance transforms.
        if _scale == "log":
            _fl = 10.0 ** _fl

        if _av > 0:
            try:
                from extinction import fitzpatrick99
                _a_lambda = fitzpatrick99(_wl.astype(np.float64), _av, 3.1)
                _fl = _fl * 10 ** (-0.4 * _a_lambda)
            except ImportError:
                pass

        if _cheb1 != 0.0:
            _scale_wl = _wl / _wl.max()
            _cheb_poly = chebval(_scale_wl, [1.0, _cheb1])
            _fl = _fl * _cheb_poly

        _fl = _fl * np.exp(_log_scale)

        # Convert back to log₁₀ after nuisance transforms.
        if _scale == "log":
            _fl = np.log10(np.clip(_fl, 1e-30, None))

        _model_interp = np.interp(_obs_wl, _wl, _fl)

        _resid = (_model_interp - _obs_fl) / _obs_err
        return float(np.sum(_resid ** 2))

    _n_restarts = max(1, int(qf_restarts.value))
    with mo.status.spinner(title=f"Running {qf_opt_method.value} MLE...") as _spinner:
        _nll_history = []
        _iter_count = [0]
        _t0 = time_mod.time()

        _global_best_f = [float("inf")]  # mutable for callback access
        _cur_restart = [0]

        def _chi2_cb(P):
            try:
                val = _chi2(P)
            except (ValueError, np.linalg.LinAlgError):
                val = 1e30
            _nll_history.append(val)
            _iter_count[0] += 1
            if _iter_count[0] % 50 == 0:
                _spinner.update(
                    f"{qf_opt_method.value} | Restart {_cur_restart[0]}/{_n_restarts} | "
                    f"Eval {_iter_count[0]} | "
                    f"Best χ² = {_global_best_f[0]:.4f} | "
                    f"{time_mod.time() - _t0:.1f}s"
                )
            return val

        # Generate starting points: first = user guess, rest = random
        _start_points = [_p0.copy()]
        if _n_restarts > 1:
            np.random.seed(None)
            for _ in range(_n_restarts - 1):
                _start_points.append(_lo + np.random.rand(len(_lo)) * (_hi - _lo))

        _global_best_x = _p0.copy()
        _global_best_nit = 0

        for _restart_idx, _x0 in enumerate(_start_points):
            _cur_restart[0] = _restart_idx + 1
            _iter_count[0] = 0  # reset eval counter per restart
            _spinner.update(
                f"{qf_opt_method.value} | Restart {_cur_restart[0]}/{_n_restarts} | "
                f"Starting... | {time_mod.time() - _t0:.1f}s"
            )

            if qf_opt_method.value == "Nelder-Mead":
                _run_soln = scipy_minimize(
                    _chi2_cb, _x0,
                    method="Nelder-Mead",
                    options=dict(maxiter=int(qf_max_iter.value), adaptive=True),
                )
            elif qf_opt_method.value == "CMA-ES":
                try:
                    import cma
                except ImportError:
                    raise ImportError(
                        "CMA-ES requires the 'cma' package. "
                        "Install it with: pip install cma"
                    )
                _cma_bounds = [_lo.tolist(), _hi.tolist()]
                _p0_cma = np.clip(_x0, _lo + 1e-8, _hi - 1e-8)
                _cma_stds = [0.2 * float(hi - lo) for lo, hi in zip(_lo, _hi)]
                _N_cma = len(_p0_cma)
                _popsize = 2 * (4 + int(3 * np.log(_N_cma)))
                _es = cma.CMAEvolutionStrategy(
                    _p0_cma.tolist(), 1.0,
                    {
                        "bounds": _cma_bounds,
                        "CMA_stds": _cma_stds,
                        "popsize": _popsize,
                        "maxfevals": int(qf_max_iter.value),
                        "verbose": -9,
                        "tolfun": 1e-10,
                    },
                )
                _run_best_x, _run_best_f = _x0.copy(), float("inf")
                while not _es.stop():
                    _solutions = _es.ask()
                    _fits = [_chi2_cb(np.array(s)) for s in _solutions]
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
            else:
                _bounds = list(zip(_lo, _hi))
                _run_soln = scipy_minimize(
                    _chi2_cb, np.clip(_x0, _lo, _hi),
                    method="L-BFGS-B",
                    bounds=_bounds,
                    options=dict(maxiter=int(qf_max_iter.value), ftol=1e-15, gtol=1e-12),
                )

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

        _elapsed = time_mod.time() - _t0

    _ensemble_spread = np.full(len(_active_idx), np.nan)
    if not qf_inf_is_grid_interp and len(qf_inf_models) > 1:
        _ensemble_best_fits = []
        for _m_idx, _single_model in enumerate(qf_inf_models):
            def _chi2_single(P, _mdl=_single_model):
                # Enforce prior bounds (same penalty as main _chi2).
                for _ai_b in range(len(P)):
                    if P[_ai_b] < _lo[_ai_b] or P[_ai_b] > _hi[_ai_b]:
                        _dist = 0.0
                        for _j in range(len(P)):
                            if P[_j] < _lo[_j]:
                                _dist += (_lo[_j] - P[_j]) ** 2
                            elif P[_j] > _hi[_j]:
                                _dist += (P[_j] - _hi[_j]) ** 2
                        return 1e20 + 1e10 * _dist

                full_params = np.array(_vals, dtype=np.float64)
                for _ai, _gi in enumerate(_active_idx):
                    full_params[_gi] = P[_ai]
                phys = full_params[:_n_phys]
                _av_s = full_params[_n_phys] if len(full_params) > _n_phys else 0.0
                _ls_s = full_params[_n_phys + 1] if len(full_params) > _n_phys + 1 else 0.0
                _ch_s = full_params[_n_phys + 2] if len(full_params) > _n_phys + 2 else 0.0

                mf, _ = qf_predict_flux(
                    [_mdl], phys, qf_inf_emu_data,
                    qf_inf_input_scaler, qf_inf_output_scaler, is_grid_interp=False,
                )
                fl = mf[_emu_mask]
                wl = _emu_wl[_emu_mask]

                if _scale == "log":
                    fl = 10.0 ** fl

                if _av_s > 0:
                    try:
                        from extinction import fitzpatrick99
                        a_l = fitzpatrick99(wl.astype(np.float64), _av_s, 3.1)
                        fl = fl * 10 ** (-0.4 * a_l)
                    except ImportError:
                        pass
                if _ch_s != 0.0:
                    fl = fl * chebval(wl / wl.max(), [1.0, _ch_s])
                fl = fl * np.exp(_ls_s)

                if _scale == "log":
                    fl = np.log10(np.clip(fl, 1e-30, None))

                mi = np.interp(_obs_wl, wl, fl)
                return float(np.sum(((_obs_fl - mi) / _obs_err) ** 2))

            try:
                if qf_opt_method.value == "Nelder-Mead":
                    _s = scipy_minimize(_chi2_single, _soln.x, method="Nelder-Mead",
                                        options=dict(maxiter=2000, adaptive=True))
                else:
                    _s = scipy_minimize(_chi2_single, _soln.x, method="L-BFGS-B",
                                        bounds=list(zip(_lo, _hi)),
                                        options=dict(maxiter=2000))
                _ensemble_best_fits.append(_s.x)
            except Exception:
                pass

        if len(_ensemble_best_fits) > 1:
            _ensemble_spread = np.std(_ensemble_best_fits, axis=0)

    _best_full = np.array(_vals, dtype=np.float64)
    for _ai, _gi in enumerate(_active_idx):
        _best_full[_gi] = _soln.x[_ai]

    qf_mle_result = {
        "best_params": _best_full,
        "active_idx": _active_idx,
        "active_labels": _active_labels,
        "best_active": _soln.x,
        "ensemble_errors": _ensemble_spread,
        "chi2": _soln.fun,
        "success": _soln.success,
        "nit": getattr(_soln, 'nit', _iter_count[0]),
        "elapsed": _elapsed,
        "chi2_history": _nll_history,
        "labels": _labels,
        "n_physical": _n_phys,
        "n_restarts": _n_restarts,
    }
    return (qf_mle_result,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 4: Results & Export
    """)
    return


@app.cell
def _(
    alt,
    chebval,
    mo,
    np,
    pd,
    qf_inf_emu_data,
    qf_inf_input_scaler,
    qf_inf_is_grid_interp,
    qf_inf_models,
    qf_inf_output_scaler,
    qf_inf_wl_slider,
    qf_mle_result,
    qf_obs_data,
    qf_obs_scale_selector,
    qf_predict_flux,
    qf_transform_observation_data,
):
    if qf_mle_result is None:
        mo.stop(True, mo.md("*Run Quick Fit to see results.*"))

    _best_full = qf_mle_result["best_params"]
    _n_phys = qf_mle_result["n_physical"]
    _phys = _best_full[:_n_phys]
    _av = _best_full[_n_phys] if len(_best_full) > _n_phys else 0.0
    _log_scale = _best_full[_n_phys + 1] if len(_best_full) > _n_phys + 1 else 0.0
    _cheb1 = _best_full[_n_phys + 2] if len(_best_full) > _n_phys + 2 else 0.0

    _mean_fl, _ens_fl = qf_predict_flux(
        qf_inf_models, _phys, qf_inf_emu_data,
        qf_inf_input_scaler, qf_inf_output_scaler, is_grid_interp=qf_inf_is_grid_interp,
    )

    _emu_wl = qf_inf_emu_data["wl"]
    _wl_lo, _wl_hi = qf_inf_wl_slider.value
    _mask = (_emu_wl >= _wl_lo) & (_emu_wl <= _wl_hi)
    _wl = _emu_wl[_mask]

    _sc = qf_obs_scale_selector.value

    def _apply_nuisance(fl, wl, av, log_scale, cheb1):
        fl = fl.copy()
        if _sc == "log":
            fl = 10.0 ** fl
        if av > 0:
            try:
                from extinction import fitzpatrick99
                a_l = fitzpatrick99(wl.astype(np.float64), av, 3.1)
                fl = fl * 10 ** (-0.4 * a_l)
            except ImportError:
                pass
        if cheb1 != 0.0:
            fl = fl * chebval(wl / wl.max(), [1.0, cheb1])
        fl = fl * np.exp(log_scale)
        if _sc == "log":
            fl = np.log10(np.clip(fl, 1e-30, None))
        return fl

    _model_fl = _apply_nuisance(_mean_fl[_mask], _wl, _av, _log_scale, _cheb1)

    if len(_ens_fl) > 1:
        _ens_processed = [_apply_nuisance(f[_mask], _wl, _av, _log_scale, _cheb1) for f in _ens_fl]
        _ens_lo = np.min(_ens_processed, axis=0)
        _ens_hi = np.max(_ens_processed, axis=0)
    else:
        _ens_lo = _model_fl
        _ens_hi = _model_fl

    _obs_mask_wl = (qf_obs_data['wavelength'] >= _wl_lo) & (qf_obs_data['wavelength'] <= _wl_hi)
    _obs_wl = np.array(qf_obs_data[_obs_mask_wl]['wavelength'])
    _raw_obs_fl = np.array(qf_obs_data[_obs_mask_wl]['flux'])
    _raw_obs_err = np.array(qf_obs_data[_obs_mask_wl]['error']) if 'error' in qf_obs_data.columns else None

    _sc = qf_obs_scale_selector.value
    _obs_fl, _obs_err = qf_transform_observation_data(_obs_wl, _raw_obs_fl, _raw_obs_err, _sc)

    _model_interp = np.interp(_obs_wl, _wl, _model_fl)
    _residuals = (_obs_fl - _model_interp) / _obs_err

    _max_pts = 5000
    if len(_obs_wl) > _max_pts:
        _step = len(_obs_wl) // _max_pts
        _obs_wl_p = _obs_wl[::_step]
        _obs_fl_p = _obs_fl[::_step]
        _resid_p = _residuals[::_step]
    else:
        _obs_wl_p = _obs_wl
        _obs_fl_p = _obs_fl
        _resid_p = _residuals

    if len(_wl) > _max_pts:
        _step_m = len(_wl) // _max_pts
        _wl_p = _wl[::_step_m]
        _model_p = _model_fl[::_step_m]
        _elo_p = _ens_lo[::_step_m]
        _ehi_p = _ens_hi[::_step_m]
    else:
        _wl_p = _wl
        _model_p = _model_fl
        _elo_p = _ens_lo
        _ehi_p = _ens_hi

    _obs_df = pd.DataFrame({'Wavelength': _obs_wl_p, 'Flux': _obs_fl_p, 'Type': 'Observation'})
    _mod_df = pd.DataFrame({'Wavelength': _wl_p, 'Flux': _model_p, 'Type': 'Best Fit'})
    _band_df = pd.DataFrame({'Wavelength': _wl_p, 'Lower': _elo_p, 'Upper': _ehi_p})

    _combined = pd.concat([_obs_df, _mod_df], ignore_index=True)
    _color_scale = alt.Scale(domain=['Observation', 'Best Fit'], range=['cyan', 'orange'])

    _line = alt.Chart(_combined).mark_line(strokeWidth=1.5).encode(
        x=alt.X('Wavelength:Q', title='Wavelength (Å)'),
        y=alt.Y('Flux:Q', title=f'Flux ({_sc})', axis=alt.Axis(format='.1e')),
        color=alt.Color('Type:N', scale=_color_scale),
        tooltip=['Wavelength', 'Flux', 'Type']
    )

    _band = alt.Chart(_band_df).mark_area(opacity=0.2, color='orange').encode(
        x='Wavelength:Q',
        y='Lower:Q',
        y2='Upper:Q',
    )

    _overlay_chart = (_band + _line).properties(
        width="container", height=350,
        title="Best-Fit Spectrum Overlay"
    ).interactive()

    _resid_df = pd.DataFrame({'Wavelength': _obs_wl_p, 'Residual (σ)': _resid_p})
    _resid_chart = alt.Chart(_resid_df).mark_point(size=3, color='cyan').encode(
        x=alt.X('Wavelength:Q', title='Wavelength (Å)', scale=alt.Scale(zero=False)),
        y=alt.Y('Residual (σ):Q', title='(Obs − Model) / σ'),
        tooltip=['Wavelength', 'Residual (σ)']
    ).properties(
        width="container", height=200,
        title="Normalised Residuals"
    ).interactive()

    _zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='grey', strokeDash=[4, 4]).encode(y='y:Q')

    _status = mo.callout(
        mo.md(f"""
        **Quick Fit Complete**
        - **χ²** = {qf_mle_result['chi2']:.2f} | **Reduced χ²** = {qf_mle_result['chi2'] / max(1, len(_obs_wl) - len(qf_mle_result['active_idx'])):.4f}
        - **Iterations:** {qf_mle_result['nit']} | **Restarts:** {qf_mle_result.get('n_restarts', 1)} | **Time:** {qf_mle_result['elapsed']:.1f}s
        - **Converged:** {'Yes' if qf_mle_result['success'] else 'No'}
        """),
        kind="success" if qf_mle_result['success'] else "warn"
    )

    mo.vstack([_status, _overlay_chart, _resid_chart + _zero_line])
    return


@app.cell
def _(mo, np, pd, qf_mle_result):
    if qf_mle_result is None:
        mo.stop(True, mo.md(""))

    _labels = qf_mle_result["labels"]
    _best = qf_mle_result["best_params"]
    _active_idx = qf_mle_result["active_idx"]
    _e_err = qf_mle_result["ensemble_errors"]

    _rows = []
    _ai = 0
    for _i, _name in enumerate(_labels):
        _val = _best[_i]
        if _i in _active_idx:
            _ee = _e_err[_ai] if _ai < len(_e_err) else np.nan
            _ai += 1
        else:
            _ee = np.nan

        _rows.append({
            "Parameter": _name,
            "Best Fit": f"{_val:.6f}",
            "±1σ (Ensemble)": f"{_ee:.6f}" if np.isfinite(_ee) else "—",
        })

    _df = pd.DataFrame(_rows)
    mo.ui.table(_df, label="Parameter Summary", selection=None)
    return


@app.cell
def _(json, mo, np, os, pd, qf_inf_emu_data, qf_mle_result):
    if qf_mle_result is None:
        mo.stop(True, mo.md(""))

    _src_file = qf_inf_emu_data.get("source_file", "unknown") if qf_inf_emu_data else "unknown"

    def _export_csv(_):
        _dir = "exports"
        os.makedirs(_dir, exist_ok=True)
        _labels = qf_mle_result["labels"]
        _best = qf_mle_result["best_params"]
        _active_idx = qf_mle_result["active_idx"]
        _e_err = qf_mle_result["ensemble_errors"]

        _rows = []
        _ai = 0
        for _i, _name in enumerate(_labels):
            _ee = float(_e_err[_ai]) if _i in _active_idx and _ai < len(_e_err) else None
            if _i in _active_idx:
                _ai += 1
            _rows.append({"parameter": _name, "value": float(_best[_i]),
                          "ensemble_1sigma": _ee})

        _df = pd.DataFrame(_rows)
        _path = os.path.join(_dir, f"quickfit_{_src_file}_params.csv")
        _df.to_csv(_path, index=False)
        mo.status.toast(f"Saved {_path}", kind="success")

    def _export_json(_):
        _dir = "exports"
        os.makedirs(_dir, exist_ok=True)
        _out = {
            "source_model": _src_file,
            "chi2": float(qf_mle_result["chi2"]),
            "converged": bool(qf_mle_result["success"]),
            "parameters": {},
        }
        _labels = qf_mle_result["labels"]
        _best = qf_mle_result["best_params"]
        _active_idx = qf_mle_result["active_idx"]
        _e_err = qf_mle_result["ensemble_errors"]
        _ai = 0
        for _i, _name in enumerate(_labels):
            _entry = {"value": float(_best[_i]), "fixed": _i not in _active_idx}
            if _i in _active_idx:
                _entry["ensemble_1sigma"] = float(_e_err[_ai]) if np.isfinite(_e_err[_ai]) else None
                _ai += 1
            _out["parameters"][_name] = _entry

        _path = os.path.join(_dir, f"quickfit_{_src_file}_summary.json")
        with open(_path, 'w') as f:
            json.dump(_out, f, indent=2)
        mo.status.toast(f"Saved {_path}", kind="success")

    def _export_pf(_):
        _dir = "exports"
        os.makedirs(_dir, exist_ok=True)
        try:
            from Speculate_addons.speculate_benchmark import emulator_to_physical
            from exports.templates.speculate_pf_exporter import write_pf
            import time as _time

            _labels = qf_mle_result["labels"]
            _best = qf_mle_result["best_params"]
            _n_phys = qf_mle_result["n_physical"]
            _param_names = qf_inf_emu_data["param_names"]

            physical = emulator_to_physical(_param_names, _best[:_n_phys])

            # Determine grid name from source file
            _src = qf_inf_emu_data.get("source_file", "unknown")
            # Strip _qfnn_/_qfnn-ensemble_/_qfgi_ suffix to recover grid name
            import re as _re
            _gn_match = _re.match(r"(.+?)_(qfnn-ensemble|qfnn|qfgi)_", _src)
            _grid_name = _gn_match.group(1) if _gn_match else _src.split("_qf")[0]

            header = [
                "### Sirocco .pf Template",
                f"### Generated by Speculate Quick Fit",
                f"### Date: {_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"### Source model: {_src}",
                "###",
            ]

            _nuisance = {}
            for _i in range(_n_phys, len(_labels)):
                _nuisance[_labels[_i]] = float(_best[_i])
            if _nuisance:
                header.append("### Nuisance Parameters:")
                for k, v in _nuisance.items():
                    header.append(f"###   {k}: {v:.6f}")
                header.append("###")

            _ts = _time.strftime("%Y%m%d_%H%M%S")
            _pf_path = os.path.join(_dir, f"quickfit_export_{_ts}.pf")
            write_pf(
                grid_name=_grid_name,
                physical_params=physical,
                output_path=_pf_path,
                header_lines=header,
            )
            mo.status.toast(f"Exported {_pf_path}", kind="success")
        except Exception as _e:
            mo.status.toast(f".pf export failed: {_e}", kind="danger")

    _csv_btn = mo.ui.button(label=f"{mo.icon('lucide:file-text')} Export CSV", on_click=_export_csv)
    _json_btn = mo.ui.button(label=f"{mo.icon('lucide:file-json')} Export JSON", on_click=_export_json)
    _pf_btn = mo.ui.button(label=f"{mo.icon('lucide:file-text')} Export .pf", on_click=_export_pf, kind="success")

    mo.hstack([_csv_btn, _json_btn, _pf_btn], justify="start")
    return


if __name__ == "__main__":
    app.run()
