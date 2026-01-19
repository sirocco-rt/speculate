# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full", app_title="Speculate Training Tool")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md(
        """
        # Speculate Training Tool

        Train custom emulator models on spectral grids.
        """
    )
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
    # Add logo
    import pathlib

    logo_path = pathlib.Path("assets/logos/Speculate_logo2.png")

    if logo_path.exists():
        logo = mo.image(src=str(logo_path), width=300)
        link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
        mo.vstack([logo, link], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Training Configuration

    Configure your emulator training parameters below.

    ### ⚠️ Requirements
    - Local installation with full spectral grids
    - GPU recommended for large grids (6561+ points)
    - ~4-16 GB RAM minimum (depends on grid size)

    #### Current Interface Bugs:
    - Changing grids doesn't change available parameters in the dropdown correctly.
    - Streamline text outputs and logging.
    - Better control over iteration logging, maybe stream a training loss curve instead?
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
        "speculate_cv_bl_grid_v87f": {
            "class": Speculate_cv_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        },
        "speculate_cv_no-bl_grid_v87f": {
            "class": Speculate_cv_no_bl_grid_v87f,
            "usecols": (1, 7),
            "name": "speculate_cv_no-bl_grid_v87f",
            "max_params": [1, 2, 3, 4, 5, 6, 9, 10, 11]
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
        available_grids,
        grid_configs,
        logging,
        np,
        os,
        plot_eigenspectra,
        sirocco_grids_path,
        sys,
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
def _(mo):
    mo.md("### 2. Model Parameters")

    # Parameter selection info in an accordion
    param_accordion = mo.accordion({
        "Grid Parameter Details": mo.md("""
        **Available Parameters by Grid:**

        - **CV BL Grid (v87f)**: 
          1. Disk.mdot, 2. wind.mdot, 3. KWD.d, 4. mdot_r_exponent, 5. acceleration_length, 
          6. acceleration_exponent, 7. BL.luminosity, 8. BL.temp, 9/10/11. Inclination (sparse/mid/full)

        - **CV NO-BL Grid (v87f)**: 
          1. Disk.mdot, 2. wind.mdot, 3. KWD.d, 4. mdot_r_exponent, 5. acceleration_length, 
          6. acceleration_exponent, 9/10/11. Inclination (sparse/mid/full)
        """)
    })

    param_accordion
    return


@app.cell
def _(grid_configs, grid_selector, mo):
    # Parameter names mapping (number -> name)
    param_names = {
        1: "Disk.mdot",
        2: "wind.mdot",
        3: "KWD.d",
        4: "mdot_r_exponent",
        5: "acceleration_length",
        6: "acceleration_exponent",
        7: "BL.luminosity",
        8: "BL.temp",
        9: "Inclination (sparse)",
        10: "Inclination (mid)",
        11: "Inclination (full)"
    }

    # Dynamically show parameter options based on grid
    if grid_selector is not None and grid_selector.value:
        selected_grid = grid_selector.value
        if selected_grid in grid_configs:
            max_params = grid_configs[selected_grid]["max_params"]
            # Set default parameters based on grid type
            if "bl" in selected_grid and "no-bl" not in selected_grid:
                # BL grid: includes boundary layer parameters
                default_params = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                # NO-BL grid: excludes boundary layer parameters
                default_params = [1, 2, 3, 4, 5, 6, 9]

            # Create options with name as key, number as value - only for available params
            params = mo.ui.multiselect(
                options={param_names.get(i, f"Parameter {i}"): str(i) for i in max_params},
                value=[param_names.get(p, f"Parameter {p}") for p in default_params if p in max_params],
                label="Select parameters to include:"
            )
        else:
            params = mo.ui.multiselect(
                options={param_names.get(i, f"Parameter {i}"): str(i) for i in range(1, 10)},
                value=[param_names.get(i, f"Parameter {i}") for i in [1, 2, 3, 4, 5, 6]],
                label="Select parameters to include:"
            )
        params
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
        except Exception:
            pass

    high_vram
    return


@app.cell
def _(mo):
    mo.md("### 3. Wavelength Range, Scale & PCA Components")

    wl_min = mo.ui.number(start=800, stop=8000, value=850, step=10, label="Min Wavelength (Å):")
    wl_max = mo.ui.number(start=800, stop=8000, value=1850, step=10, label="Max Wavelength (Å):")

    scale_selector = mo.ui.dropdown(
        options=["linear", "log", "scaled"],
        value="linear",
        label="Flux Scale:"
    )

    n_components = mo.ui.slider(start=2, stop=20, value=10, step=1, label="PCA Components:",show_value=True,)

    mo.vstack([
        mo.hstack([wl_min, wl_max], justify="start"),
        mo.hstack([scale_selector, n_components], justify="start")
    ])
    return n_components, scale_selector, wl_max, wl_min


@app.cell
def _(mo):
    mo.md("### 4. Training Options")

    method = mo.ui.dropdown(
        options=["Nelder-Mead"],
        value="Nelder-Mead",
        label="Optimization Method:"
    )

    max_iter = mo.ui.number(start=100, stop=10000, value=1000, step=100, label="Max Iterations:")

    mo.vstack([method, max_iter])
    return max_iter, method


@app.cell
def _(mo):
    mo.md("### 5. Start Training")

    train_button = mo.ui.run_button(label="🚀 Train Emulator")

    mo.callout(
        mo.md("Click to begin training. This may take several minutes to hours depending on grid size and hardware."),
        kind="info"
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
        grid_name = grid_selector.value
        grid_path = sirocco_grids_path / grid_name

        # Standardize base name from grid folder name (e.g. no-bl -> no_bl)
        base_name = grid_name.replace("-", "_") + "_"

        # Generate file names
        grid_file_name = f"{base_name}grid_{model_parameters_str}"

        # Determine emulator file name based on Speculate_dev.py conventions
        # Default fixed inclination is 55 degrees per grid_configs setup
        fixed_inc = 55

        if 9 in model_parameters or 10 in model_parameters or 11 in model_parameters:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'
        else:
            emu_file_name = f'{base_name}emu_{model_parameters_str}_{scale}_{fixed_inc}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components.value}PCA'

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
                scale=scale
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
    return emu_exists, emu_file_name, grid_file_name


@app.cell
def _(mo):
    mo.md("---")
    mo.md("## Training Results")
    mo.md("*Results will appear here after training completes*")
    return


@app.cell
def _(
    Emulator,
    emu_exists,
    emu_file_name,
    grid_file_name,
    max_iter,
    method,
    mo,
    n_components,
    np,
    os,
    sys,
    train_button,
):
    import matplotlib.pyplot as plt

    training_complete = False
    emu = None

    if train_button.value and not emu_exists:
        # Train new emulator
        # Use simple status text that will be replaced by the result
        status_box = mo.md("🚀 **Training emulator...** Check the console below for progress.")

        # Container for results
        # result_container = mo.empty() # Removed: not a valid attribute

        # show initial status
        mo.output.replace(status_box)

        try:
            with mo.status.spinner(title="Training Emulator", subtitle=" optimizing parameters... (logs printed below)") as _spinner:
                with mo.redirect_stdout():
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

                    # Train emulator
                    if method.value == "Nelder-Mead":
                        # Use ftol for relative tolerance as discussed previously
                        emu.train(method="Nelder-Mead", options=dict(maxiter=max_iter.value, disp=True, ftol=1e-3))
                    else:
                        emu.train_bfgs(options=dict(maxiter=max_iter.value, disp=True))

                    print("--- Training Finished ---")

                    # Save emulator
                    os.makedirs('Grid-Emulator_Files', exist_ok=True)
                    emu.save(f'Grid-Emulator_Files/{emu_file_name}.npz')
                    print(f"Emulator saved to Grid-Emulator_Files/{emu_file_name}.npz")
                    training_complete = True

            # Create result display
            train_result = mo.md(f"""
            ✅ **Training complete!**

            - **Emulator saved:** `Grid-Emulator_Files/{emu_file_name}.npz`
            - **PCA Components:** {emu.ncomps}
            - **Grid Points:** {len(emu.grid_points)}
            - **Wavelength Range:** {emu.wl[0]:.1f}-{emu.wl[-1]:.1f} Å

            **Emulator Info:**
            ```
            {str(emu)}
            ```
            """)

            # Return the result stack
            mo.vstack([train_result])

        except Exception as e:
            error_result = mo.md(f"""
            ❌ **Training failed!**

            **Error:** {str(e)}

            Check the logs for more details.
            """)
            mo.vstack([error_result])

    elif train_button.value and emu_exists:
        # Load existing emulator
        emu = Emulator.load(f'Grid-Emulator_Files/{emu_file_name}.npz')
        training_complete = True

        existing_result = mo.md(f"""
        ℹ️ **Loaded existing emulator**

        - **File:** `Grid-Emulator_Files/{emu_file_name}.npz`
        - **PCA Components:** {emu.ncomps}
        - **Grid Points:** {len(emu.grid_points)}
        - **Wavelength Range:** {emu.wl[0]:.1f}-{emu.wl[-1]:.1f} Å

        **Emulator Info:**
        ```
        {str(emu)}
        ```
        """)
        existing_result
    else:
        mo.md("*Train an emulator to see results*")
    return emu, training_complete


@app.cell
def _(emu, mo, plot_eigenspectra, training_complete):
    if training_complete and emu is not None:
        mo.md("""
        ---
        ## 📊 Emulator Visualization

        View the eigenspectra components that make up your emulator.
        """)

        # Plot eigenspectra
        try:
            fig = plot_eigenspectra(emu)
            mo.mpl.interactive(fig)
        except Exception as e:
            mo.md(f"⚠️ Could not plot eigenspectra: {str(e)}")
    else:
        mo.md("*Eigenspectra visualization will appear after training*")
    return


if __name__ == "__main__":
    app.run()
