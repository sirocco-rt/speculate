# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Training Tool")


@app.cell
def _(mo):
    get_loss_history, set_loss_history = mo.state([])
    get_training_trigger, set_training_trigger = mo.state(0)
    get_console_logs, set_console_logs = mo.state("")
    get_training_status, set_training_status = mo.state(None)
    get_pca_result, set_pca_result = mo.state("Click test to see variance")
    return (
        get_console_logs,
        get_loss_history,
        get_pca_result,
        get_training_status,
        get_training_trigger,
        set_console_logs,
        set_loss_history,
        set_pca_result,
        set_training_status,
        set_training_trigger,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo
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
        mo.image(src=logo_path, width=400),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    # Combine in horizontal stack
    mo.hstack([title_col, logo_col], justify="space-between", align="center")
    return (mo,)


@app.cell
def _(mo):
    # Static sidebar - always shows all options
    mo.sidebar(
        mo.vstack([
                    mo.md("# 🔭 Speculate"),
                    mo.md(" "),
                    mo.md(" "),
                    mo.md("---"),
                    mo.md("---"),
                    mo.md(" "),
                    mo.md(" "),
                    mo.nav_menu({
                        "/": f"###{mo.icon('lucide:home')} Home",
                    }, orientation="vertical"),
                    mo.md(" "),
                    mo.md("---"),
                    mo.md("---"),
                    mo.nav_menu({
                    "https://github.com/sirocco-rt/speculate": f"###{mo.icon('lucide:github')} Speculate Github",
                    "https://github.com/sirocco-rt/speculate/wiki": f"###{mo.icon('lucide:book-open')} Speculate Docs",
                    }, orientation="vertical"),
                    mo.md(" "),
                    mo.md("---"),
                    mo.nav_menu({
                    "https://github.com/sirocco-rt/sirocco": f"###{mo.icon('lucide:wind')} Sirocco Github",
                    "https://sirocco-rt.readthedocs.io/en/latest/": f"###{mo.icon('lucide:wind')} Sirocco Docs"
                    }, orientation="vertical")
                ])
    )
    return


@app.cell
def _(mo):
    import torch

    # GPU Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_text = f"## 🟢 GPU Active: **{gpu_name}** ({vram_gb:.1f} GB VRAM)"
        except:
            gpu_text = f"## 🟢 GPU Active: **{gpu_name}**"

        status_widget = mo.callout(mo.md(gpu_text), kind="success")

    else:
        status_widget = mo.callout(
            mo.md("## 🟠 No NVIDIA GPU Detected - Large grids will train slower."), 
            kind="warn"
        )

    status_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ⚠️ Emulator Training Configuration Requirements

    - Run a Local Installation of Speculate
    - GPU recommended for large grids (2500+ grid points or 6+ parameters)
    - RAM/VRAM requirements scales by an order of magnitude with each added parameters.

    ## 💻 Emulator Specification:
    """)
    return


@app.cell
def _(mo):
    import os
    import sys
    import numpy as np
    import itertools
    from pathlib import Path
    from tqdm import tqdm
    import logging
    import pandas as pd
    import altair as alt
    import time

    # Import Starfish and Speculate modules
    from Starfish.grid_tools import HDF5Creator
    from Starfish.emulator import Emulator
    from Starfish.emulator.plotting import plot_eigenspectra
    from Speculate_addons.Spec_gridinterfaces import Speculate_cv_bl_grid_v87f
    from Speculate_addons.Spec_gridinterfaces import Speculate_cv_no_bl_grid_v87f

    class MarimoHDF5Creator(HDF5Creator):
        def process_grid(self):
            """
            Run :meth:`process_flux` for all of the spectra within the `ranges`
            and store the processed spectra in the HDF5 file.
            """
            # param_list will be a list of numpy arrays, specifying the parameters
            param_list = []

            # use itertools.product to create permutations of all possible values
            for i in itertools.product(*self.points):
                param_list.append(np.array(i))

            all_params = np.array(param_list)
            invalid_params = []

            self.log.debug("Total of {} files to process.".format(len(param_list)))

            # Use marimo progress bar
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

                # Store flux data in dictionary
                flux_key = self.key_name.format(*param)
                # Filter out empty FITS keywords  
                clean_header = {k: v for k, v in header.items() 
                              if k != "" and k != "COMMENT" and v != ""}

                self.flux_data[flux_key] = {
                    "flux": fl_final,
                    "header": clean_header
                }

            # Remove parameters that do no exist
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

    # Check available grids in sirocco_grids folder
    sirocco_grids_path = Path("sirocco_grids")
    available_grids = {}

    # Map grid folder names to their configuration
    grid_configs = {
        "speculate_cv_no-bl_grid_v87f": {
            "class": Speculate_cv_no_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_no-bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 9, 10, 11]
        },
        "speculate_cv_bl_grid_v87f": {
            "class": Speculate_cv_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }
    }

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
def _(available_grids, mo):
    mo.md("### 1. Grid Selection")

    if available_grids:
        grid_selector = mo.ui.dropdown(
            options=available_grids,
            value=list(available_grids.keys())[0] if available_grids else None,
            label="Select Grid:"
        )
        mo.vstack([
            grid_selector,
            mo.md(f"✓ Found **{len(available_grids)}** grid(s) in `sirocco_grids/`")
        ])
    else:
        mo.callout(
            mo.md("""
            ⚠️ **No grids found in `sirocco_grids/` folder**

            Please use the **Grid Downloader** tool first:
            """),
            kind="warn"
        )
        grid_selector = None
    grid_selector
    return (grid_selector,)


@app.cell
def _(grid_configs, grid_selector, mo, sirocco_grids_path):
    # Dynamically fetch parameter options from the grid interface
    param_names = {}

    if grid_selector is not None and grid_selector.value:
        selected_grid = grid_selector.value

        if selected_grid in grid_configs:
            config = grid_configs[selected_grid]
            max_params = config["max_params"]

            try:
                # Instantiate temporary interface to get descriptions
                # We use the raw path + "/" as required by star fish interfaces usually
                temp_path = str(sirocco_grids_path / selected_grid) + "/"

                # Initialize with all possible parameters to get all descriptions
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

            # Set default parameters logic
            # Heuristic: exclude inclination alternatives (typically > 9 if 9 is sparse inc)
            # Default to selecting "Standard" parameters + sparse inclination
            default_params = [p for p in max_params if p <= 9]

            # Create options
            params = mo.ui.multiselect(
                options={param_names.get(i, f"Parameter {i}"): str(i) for i in max_params},
                value=[param_names.get(p, f"Parameter {p}") for p in default_params if p in max_params],
                label="Select parameters to include:"
            )
        else:
            params = mo.ui.multiselect(
                options={},
                value=[],
                label="Select parameters to include:"
            )
    else:
        params = None
        mo.md("*Select a grid first*")

    params
    return (params,)


@app.cell
def _(mo, params):
    # Calculate estimated grid size to warn about VRAM
    high_vram = mo.md("")

    if params is not None and params.value:
        try:
            # Parse selected parameter indices
            selected = [int(p) for p in params.value]

            # Calculate total permutations based on parameter weights
            # Standard params (1-9) have 3 values
            # Param 10 (Mid Inc) has 6 values (weight = 2x standard)
            # Param 11 (Full Inc) has 12 values (weight = 4x standard)
            total_points = 1
            for p in selected:
                if p == 11:
                    total_points *= 12
                elif p == 10:
                    total_points *= 6
                else:
                    total_points *= 3

            # Thresholds:
            # 3^8 = 6,561 (High VRAM start)
            # 3^9 = 19,683 (Very High VRAM start)

            if total_points >= 19683:
                high_vram = mo.callout(
                    mo.md(f"⚠️ **Warning: Very High VRAM Usage (~{total_points:,} grid points)**\n\nGrid size likely exceeds 100GB in GPU VRAM."),
                    kind="alert"
                )
            elif total_points >= 6561:
                high_vram = mo.callout(
                    mo.md(f"⚠️ **Warning: High VRAM Usage (~{total_points:,} grid points)**\n\nLarge grid size may exceed current GPU limits."),
                    kind="warn"
                )
            else: 
                high_vram = mo.callout(
                    mo.md(f"✅ **Estimated Grid Size: ~{total_points:,} grid points**\n\nGrid size should be manageable on most GPUs."),
                    kind="success"
                )
        except Exception:
            pass

    high_vram
    return


@app.cell
def _(mo):
    wl_min = mo.ui.number(start=800, stop=8000, value=850, step=10, label="Min Wavelength (Å):")
    wl_max = mo.ui.number(start=800, stop=8000, value=1850, step=10, label="Max Wavelength (Å):")

    scale_selector = mo.ui.dropdown(
        options=["linear", "log", "scaled"],
        value="linear",
        label="Flux Scale:"
    )

    use_smoothing = mo.ui.checkbox(value=False, label="Smooth Spectra (Boxcar=5)")

    n_components = mo.ui.slider(start=2, stop=30, value=10, step=1, label="PCA Components:",show_value=True,)

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

    # Display result using state
    pca_result_display = mo.md(f"_{get_pca_result()}_")

    mo.vstack([wl_min, wl_max, mo.hstack([scale_selector, use_smoothing], justify="start", align="center"), mo.hstack([n_components, test_pca_btn, pca_result_display], justify="start", align="start")])
    return


@app.cell
def _(mo):
    mo.md("### 4. Training Options")

    method = mo.ui.dropdown(
        options=["Nelder-Mead"],
        value="Nelder-Mead",
        label="Optimization Method:"
    )

    max_iter = mo.ui.number(start=100, stop=100000, value=10000, step=100, label="Max Iterations:")

    mo.vstack([method, max_iter])
    return max_iter, method


@app.cell
def _(
    Emulator,
    MarimoHDF5Creator,
    grid_configs,
    grid_selector,
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
            # 1. Determine parameters
            _model_params = tuple(sorted([int(p) for p in params.value]))
            _model_params_str = ''.join(str(i) for i in _model_params)
            _grid_name = grid_selector.value
            _base_name = _grid_name + "_"
            _smoothing = use_smoothing.value
            _scale = scale_selector.value
            _smooth_suffix = "_smooth" if _smoothing else ""
            _grid_file_name_pca = f"{_base_name}grid_{_model_params_str}_{_scale}{_smooth_suffix}"
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
                        _grid = _grid_config["class"](
                            path=str(_grid_path) + "/",
                            usecols=_grid_config["usecols"],
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
    get_training_trigger,
    grid_selector,
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

    # Determine filename to check for existence default
    _emu_btn_label = "🚀 Train Emulator"
    _emu_btn_kind = "success"
    _emu_info_text = "Click to begin training. This may take several minutes to hours depending on grid size and hardware."

    if grid_selector is not None and grid_selector.value and params is not None and params.value:
         # Configuration (duplicate logic to check filename)
        _model_params = tuple(sorted([int(p) for p in params.value]))
        _model_params_str = ''.join(str(i) for i in _model_params)
        _wl_range = (wl_min.value, wl_max.value)
        _scale = scale_selector.value
        _grid_name = grid_selector.value
        _smooth_tag = '_smooth' if use_smoothing.value else ''

        # Standardize base name
        _base_name = _grid_name + "_"

        # Determine emulator file name
        _fixed_inc = 55
        if any(x in _model_params for x in [9, 10, 11]):
             _chk_name = f'{_base_name}emu_{_model_params_str}_{_scale}{_smooth_tag}_{_wl_range[0]}-{_wl_range[1]}AA_{n_components.value}PCA'
        else:
             _chk_name = f'{_base_name}emu_{_model_params_str}_{_scale}{_smooth_tag}_{_fixed_inc}inc_{_wl_range[0]}-{_wl_range[1]}AA_{n_components.value}PCA'

        if os.path.exists(f'Grid-Emulator_Files/{_chk_name}.npz'):
            _emu_btn_label = "♻️ Re-train (An Emulator Already Exists)"
            _emu_btn_kind = "warn"
            _emu_info_text = f"**Emulator found:** `{_chk_name}.npz`\n\nClick to **re-train** (overwrites existing)."

            # Load existing loss history if available (Side Effect: Update Graph)
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
             # If emulator doesn't exist (new config), clear the graph
             set_loss_history([])

    train_button = mo.ui.run_button(label=_emu_btn_label, kind=_emu_btn_kind)

    mo.callout(
        mo.md(_emu_info_text),
        kind="info" if _emu_btn_kind == "success" else "warn"
    )

    train_button
    return (train_button,)


@app.cell
def _(
    MarimoHDF5Creator,
    grid_configs,
    grid_selector,
    logging,
    max_iter,
    method,
    mo,
    n_components,
    np,
    os,
    params,
    scale_selector,
    sirocco_grids_path,
    train_button,
    use_smoothing,
    wl_max,
    wl_min,
):
    # Initialize return variables
    emu_exists = False
    emu_file_name = ""
    grid_file_name = ""

    if train_button.value and grid_selector is not None and params is not None:
        # Configuration
        model_parameters = tuple(sorted([int(p) for p in params.value]))
        model_parameters_str = ''.join(str(i) for i in model_parameters)
        wl_range = (wl_min.value, wl_max.value)
        scale = scale_selector.value
        smoothing = use_smoothing.value
        smooth_tag = '_smooth' if smoothing else ''
        grid_name = grid_selector.value
        grid_path = sirocco_grids_path / grid_name

        # Standardize base name from grid folder name (e.g. no-bl -> no_bl)
        base_name = grid_name + "_"

        # Generate file names (include scale since flux is transformed during grid processing)
        grid_file_name = f"{base_name}grid_{model_parameters_str}_{scale}{smooth_tag}"

        # Determine emulator file name based on Speculate_dev.py conventions
        # Default fixed inclination is 55 degrees per grid_configs setup
        fixed_inc = 55

        if 9 in model_parameters or 10 in model_parameters or 11 in model_parameters:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}{smooth_tag}_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'
        else:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}{smooth_tag}_{fixed_inc}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'

        # Check for existing grid file
        grid_file_path_check = f'Grid-Emulator_Files/{grid_file_name}.npz'
        process_grid_auto = not os.path.isfile(grid_file_path_check)

        # Display configuration
        config_md = mo.md(f"""
        ## 📊 Training Configuration

        - **Grid:** `{grid_name}`
        - **Grid Path:** `{grid_path}`
        - **Parameters:** {model_parameters}
        - **Wavelength Range:** {wl_range[0]}-{wl_range[1]} Å
        - **Flux Scale:** {scale}
        - **Smoothing:** {'Yes (Gaussian σ=50)' if smoothing else 'No'}
        - **PCA Components:** {n_components.value}
        - **Method:** {method.value}
        - **Max Iterations:** {max_iter.value}
        - **Process Grid:** {'Auto (File not found, creating new)' if process_grid_auto else 'Auto (File found, loading existing)'}
        - **Grid File:** `{grid_file_name}.npz`
        - **Emulator File:** `{emu_file_name}.npz`

        ---
        """)

        # Initialize grid interface
        grid = None
        if grid_name in grid_configs:
            grid_config = grid_configs[grid_name]
            grid = grid_config["class"](
                path=str(grid_path) + "/",
                usecols=grid_config["usecols"],
                wl_range=wl_range,
                model_parameters=model_parameters,
                scale=scale,
                smoothing=smoothing
            )
        else:
            error_md = mo.md(f"⚠️ **Error:** Unknown grid configuration for `{grid_name}`")
            mo.vstack([config_md, error_md])

        # Process grid if requested
        if process_grid_auto and grid is not None:
            # Set up logging
            os.makedirs('logs', exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/grid_processing.log', mode='w')
                ],
                force=True
            )

            status_md = mo.md("🔧 **Processing grid...** (logging to `logs/grid_processing.log`)")

            # Create grid file
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

            # Load and verify grid
            data = np.load(f'Grid-Emulator_Files/{grid_file_name}.npz', allow_pickle=True)
            grid_points = data['grid_points']

            grid_info = f"""
            ✅ **Grid processed successfully!**

            - **Grid shape:** {grid_points.shape}
            - **Unique values per parameter:**
            """
            for i, param_name in enumerate(data['param_names']):
                unique_vals = np.unique(grid_points[:, i])
                grid_info += f"\n  - {param_name}: {unique_vals}"

            results_md = mo.md(grid_info)
        else:
            # Check if grid file exists
            if os.path.isfile(f'Grid-Emulator_Files/{grid_file_name}.npz'):
                data = np.load(f'Grid-Emulator_Files/{grid_file_name}.npz', allow_pickle=True)
                grid_points = data['grid_points']

                grid_info = f"""
                ✅ **Using existing grid file**

                - **Grid shape:** {grid_points.shape}
                """
                results_md = mo.md(grid_info)
            else:
                results_md = mo.md(f"⚠️ **Error:** Grid file `{grid_file_name}.npz` not found. Enable 'Process grid' to create it.")

        # Check if emulator exists
        if os.path.isfile(f'Grid-Emulator_Files/{emu_file_name}.npz'):
            emu_status = mo.md(f"ℹ️ **Emulator `{emu_file_name}.npz` already exists.** Uncheck 'Process grid' to use it or continue to retrain.")
            emu_exists = True
        else:
            emu_status = mo.md(f"📝 **Emulator `{emu_file_name}.npz` does not exist.** Ready to train.")
            emu_exists = False

        mo.vstack([config_md, results_md, emu_status])
    else:
        mo.md("*Configure all settings and click 'Train Emulator' to start*")
    return emu_file_name, grid_file_name


@app.cell
def _(mo):
    mo.md("---")
    mo.md("## Training Results")
    mo.md("*Results will appear here after training completes*")
    return


@app.cell
def _(
    Emulator,
    alt,
    emu_file_name,
    get_console_logs,
    get_training_status,
    grid_file_name,
    max_iter,
    method,
    mo,
    n_components,
    np,
    os,
    pd,
    set_console_logs,
    set_loss_history,
    set_training_status,
    set_training_trigger,
    sys,
    time,
    train_button,
):
    import matplotlib.pyplot as plt

    training_complete = False
    emu = None

    if train_button.value and grid_file_name:
        import contextlib

        # Clear previous history
        set_loss_history([])
        set_console_logs("")
        set_training_status(None)

        # Train new emulator (Always re-train if button is clicked)
        status_box = mo.md("🚀 **Training emulator...** Check the console below for progress.")

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
                accordion_content.append(mo.md("### 📈 Live Training Loss"))
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
            def flush(self): pass

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

                # Create emulator
                # Matching arguments from Speculate_dev.py
                # block_diagonal=True and svd_solver="full" are used there
                print("Initializing Emulator (running PCA/SVD)...")
                sys.stdout.flush()

                emu = Emulator.from_grid(
                    grid_path_npz,
                    n_components=n_components.value,
                    svd_solver="full",
                    block_diagonal=True
                )

                print(f"Grid loaded. Initialized {emu.ncomps} PCA components.")
                print(f"Starting optimization using {method.value} (max_iter={max_iter.value})...")
                print("Optimization output:")
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
                if method.value == "Nelder-Mead":
                    # Use ftol for relative tolerance as discussed previously
                    emu.train(method="Nelder-Mead", options=dict(maxiter=max_iter.value, disp=False), callback=training_callback)
                else:
                    emu.train_bfgs(options=dict(maxiter=max_iter.value, disp=False), callback=training_callback)

                print("--- Training Finished ---")

                # Save emulator
                os.makedirs('Grid-Emulator_Files', exist_ok=True)
                print(emu)
                emu.save(f'Grid-Emulator_Files/{emu_file_name}.npz')
                print(f"Emulator saved to Grid-Emulator_Files/{emu_file_name}.npz")
                training_complete = True

                # Trigger button update
                set_training_trigger(lambda v: v + 1)

                # Save persistent state
                set_console_logs("".join(log_buffer))
                # Also save loss history
                if hasattr(emu, 'loss_history'):
                    set_loss_history(list(emu.loss_history))

            # Create result display
            train_result = mo.md(f"""
            ✅ **Training complete!**
            """)

            set_training_status(train_result)
            # Update final UI
            update_ui(train_result)

        except Exception as e:
            error_result = mo.md(f"""
            ❌ **Training failed!**

            **Error:** {str(e)}

            Check the logs for more details.
            """)
            set_console_logs("".join(log_buffer))
            set_training_status(error_result)
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

             # Attempt to restore emulator for visualizations if file matches
             if os.path.exists(f'Grid-Emulator_Files/{emu_file_name}.npz'):
                 try:
                    emu = Emulator.load(f'Grid-Emulator_Files/{emu_file_name}.npz')
                    # Only mark complete if successful load
                    training_complete = True
                 except:
                    pass
        else:
            mo.md("*Train an emulator to see results*")
    return


@app.cell
def _(alt, get_loss_history, mo, pd):
    # Reactive plot cell
    loss_data = get_loss_history()

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

    # Always display the chart container
    mo.vstack([
        mo.md("### 📈 Trained Emulator Loss Curve"),
        mo.ui.altair_chart(chart)
    ])
    return


if __name__ == "__main__":
    app.run()
