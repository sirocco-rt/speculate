# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Inference Tool")


@app.cell
def _():
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    import os
    import re
    import numpy as np
    import pandas as pd
    import altair as alt
    from Starfish.emulator import Emulator
    from Starfish.spectrum import Spectrum
    from Starfish.models import SpectrumModel
    import pathlib
    import scipy.stats as stats
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
        mo.image(src=logo_path, width=400),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    # Combine in horizontal stack
    mo.hstack([title_col, logo_col], justify="space-between", align="center")
    return Emulator, Spectrum, SpectrumModel, alt, mo, np, os, pd, re, stats


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
            gpu_text = f"**🟢 GPU Active:** {gpu_name} ({vram_gb:.1f} GB VRAM)"
        except:
            gpu_text = f"**🟢 GPU Active:** {gpu_name}"

        status_widget = mo.callout(mo.md(gpu_text), kind="success")

    else:
        status_widget = mo.callout(
            mo.md("## 🟠 No NVIDIA GPU Detected\n*Performance will be slower on CPU.*"), 
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
    # Parameter Mapping Dictionary
    # Maps parameter index (1-based) to (Name, Default Min, Default Max)
    # Based on Speculate_dev.py
    param_map_db = {
        1: ("disk.mdot", 0.0, 1.0),
        2: ("wind.mdot", 0.0, 1.0), 
        3: ("KWD.d", 0.0, 1.0),
        4: ("KWD.mdot_r_exponent", 0.0, 1.0),
        5: ("KWD.acceleration_length", 0.0, 1.0),
        6: ("KWD.acceleration_exponent", 0.0, 1.0),
        7: ("Boundary_layer.luminosity", 0.0, 1.0),
        8: ("Boundary_layer.temp", 0.0, 1.0),
        9: ("Inclination (Sparse)", 30.0, 80.0),  # 30, 55, 80
        10: ("Inclination (Mid)", 30.0, 80.0),    # More inc points
        11: ("Inclination (Full)", 30.0, 80.0),   # All inc points
    }
    return (param_map_db,)


@app.cell
def _(mo, os):
    # --- Grid Selection ---

    _emu_dir = "Grid-Emulator_Files"
    _unique_grids = set()

    if os.path.exists(_emu_dir):
        _files = [f for f in os.listdir(_emu_dir) if f.endswith(".npz")]
        for _f in _files:
            # Parse grid name
            # Pattern: {GRID}_emu_{INDICES}... OR {GRID}_grid_{INDICES}...
            # We assume the grid name is everything before "_emu_" or "_grid_"

            # Try splitting by _emu_ first (most common for emulators)
            _parts = _f.split("_emu_")
            if len(_parts) > 1:
                _unique_grids.add(_parts[0])
                continue

            # Try splitting by _grid_
            _parts = _f.split("_grid_")
            if len(_parts) > 1:
                pass

    _sorted_grids = sorted(list(_unique_grids))

    if _sorted_grids:
        grid_selector = mo.ui.dropdown(
            options=_sorted_grids,
            value=_sorted_grids[0],
            label="Select Grid Dataset:",
            full_width=True
        )
    else:
        grid_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Grids Found",
            full_width=True
        )
    return (grid_selector,)


@app.cell
def _(grid_selector, mo, os, param_map_db, re):
    # --- Emulator Selection (Dependent on Grid) ---

    _emu_dir = "Grid-Emulator_Files"
    _filtered_emus = []

    # Grid Parameter Detection
    grid_indices = set()

    if grid_selector.value and os.path.exists(_emu_dir):
        _files = [f for f in os.listdir(_emu_dir) if f.endswith(".npz")]

        for _f in _files:
            if _f.startswith(grid_selector.value):
                # Check if it is an emulator
                if "_emu_" in _f:
                    _filtered_emus.append(_f)

                # Check for parameters based on digits after _emu_ or _grid_
                # Match _emu_(\d+) or _grid_(\d+)
                _m = re.search(r"(_emu_|_grid_)(\d+)", _f)
                if _m:
                    _digits = _m.group(2)
                    for _d in _digits:
                        grid_indices.add(int(_d))

    # Force inclusion of specific parameters based on Grid Name conventions (cv)
    if grid_selector.value and "cv" in grid_selector.value:
         # Inclination variants (9, 10, 11) are present in the grid definition
         grid_indices.update([9, 10, 11])

         # 7, 8 are Boundary Layer, present unless specifically "no_bl"
         if "_no_bl_" not in grid_selector.value and "_no-bl_" not in grid_selector.value:
              grid_indices.update([7, 8])

    _filtered_emus = sorted(_filtered_emus)

    if _filtered_emus:
        emulator_selector = mo.ui.dropdown(
            options=_filtered_emus,
            value=_filtered_emus[0],
            label="Select Trained Emulator:",
            full_width=True
        )
    else:
        emulator_selector = mo.ui.dropdown(
            options=[],
            #disabled=True,
            label="No Emulators for this Grid",
            full_width=True
        )

    # Valid Parameters Accordion
    _param_lookup_info = []
    if grid_indices:
        _sorted_ind = sorted(list(grid_indices))
        for _idx in _sorted_ind:
            if _idx in param_map_db:
                 _p_name = param_map_db[_idx][0]
                 _param_lookup_info.append(f"**{_idx}** -> `{_p_name}`")
            else:
                 _param_lookup_info.append(f"**{_idx}** -> `Unknown`")
    else:
        _param_lookup_info.append("No parameters detected from filenames.")

    param_info_accordion = mo.accordion({
        "🔑 Parameter Key": mo.md("\n".join([f"- {s}" for s in _param_lookup_info]))
    })
    return emulator_selector, grid_indices, param_info_accordion


@app.cell
def _(mo):
    # Update trigger for file list
    get_obs_refresh, set_obs_refresh = mo.state(0)

    # Source Type Selector: Observation vs Test Grid
    data_source_selector = mo.ui.dropdown(
        options=["📁 Observation File", "🧪 Test Grid (Validation)"],
        value="📁 Observation File",
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
         mo.md("<center><small>Format: CSV with headers `Wavelength` and `Flux`</small></center>")
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
    # Handle file upload logic (Recycled from Grid Inspector)
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

            mo.status.toast(f"✅ Uploaded {uploaded_file.name}")
        except Exception as e:
             mo.status.toast(f"❌ Upload failed: {str(e)}")
    return


@app.cell(hide_code=True)
def _(get_obs_refresh, grid_selector, obs_file_uploader, os):
    # Select observational spectrum - File List Calculation

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
        # Derive test grid name from main grid name
        # e.g., speculate_cv_no-bl_grid_v87f -> speculate_cv_no-bl_testgrid_v87f
        _grid_name = grid_selector.value
        _test_grid_name = _grid_name.replace("_grid_", "_testgrid_")
        test_grid_path = f"sirocco_grids/{_test_grid_name}"

        if os.path.exists(test_grid_path):
            # Get .spec files
            test_grid_files = sorted(
                [f for f in os.listdir(test_grid_path) if f.endswith('.spec')],
                key=lambda x: int(x.replace('run', '').replace('.spec', '')) if x.replace('run', '').replace('.spec', '').isdigit() else 0
            )
            # Try to load lookup table for ground truth params
            _lookup_path = os.path.join(test_grid_path, "grid_run_lookup_table.parquet")
            if os.path.exists(_lookup_path):
                import pandas as _pd
                test_grid_params_df = _pd.read_parquet(_lookup_path)
    return obs_files, test_grid_files, test_grid_params_df, test_grid_path


@app.cell
def _(mo, obs_files, os, set_obs_refresh, test_grid_files):

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

    # Delete button (only for observation files)
    def delete_selected_file():
        if obs_file_selector.value:
            try:
                _obs_dir = "observation_files"
                file_path = os.path.join(_obs_dir, obs_file_selector.value)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    set_obs_refresh(lambda v: v + 1)
                    mo.status.toast(f"🗑️ Deleted {obs_file_selector.value}")
            except Exception as e:
                mo.status.toast(f"Error deleting file: {e}")

    delete_obs_btn = mo.ui.button(
        label="🗑️",
        kind="danger",
        tooltip="Delete selected file",
        on_click=lambda _: delete_selected_file()
    )

    # Inclination selector for test grid (spec files have multiple angles)
    test_inclination_selector = mo.ui.dropdown(
        options=["30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85"],
        value="55",
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
def _(Emulator, emulator_selector, mo):
    # Load the emulator when selected

    emu = None
    stop_computation = False

    if emulator_selector.value:
        try:
             with mo.status.spinner(title=f"Loading emulator {emulator_selector.value}..."):
                emu_path = f"Grid-Emulator_Files/{emulator_selector.value}"
                emu = Emulator.load(emu_path)
        except Exception as e:
            mo.output.replace(mo.callout(mo.md(f"❌ Error loading emulator: {e}"), kind="danger"))
            stop_computation = True

    if stop_computation:
        mo.stop()
    return (emu,)


@app.cell
def _(
    data_source_selector,
    emu,
    emulator_selector,
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
    obs_data = None
    ground_truth_params = None  # For test grid validation
    _obs_dir = "observation_files"
    data_source_info = ""

    is_test_grid = "Test Grid" in data_source_selector.value

    if is_test_grid and test_run_selector.value and test_grid_path:
        # Load from Test Grid (.spec file)
        try:
            spec_path = os.path.join(test_grid_path, test_run_selector.value)

            # Parse .spec file (Sirocco format)
            # First, find the header length
            skiprows = 0
            with open(spec_path, 'r') as _f:
                for _i, line in enumerate(_f):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('Freq.'):
                        continue
                    skiprows = _i
                    break

            # Determine inclination column
            # Columns: Freq, Lambda, then 12 inclination columns (A30P0.50, A35P0.50, etc.)
            inc_val = int(test_inclination_selector.value)
            inc_options = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
            inc_col_idx = inc_options.index(inc_val) + 2  # +2 for Freq, Lambda columns

            # Load data
            data_raw = np.loadtxt(spec_path, skiprows=skiprows, unpack=True)
            wavelengths = np.flip(data_raw[1])  # Lambda column, flip to ascending
            fluxes = np.flip(data_raw[inc_col_idx])  # Selected inclination flux

            # Create DataFrame
            obs_data = pd.DataFrame({
                'wavelength': wavelengths,
                'flux': fluxes,
                'error': fluxes * 0.05  # Assume 5% error for test data
            })

            # Extract ground truth params from lookup table
            if test_grid_params_df is not None:
                run_num = int(test_run_selector.value.replace('run', '').replace('.spec', ''))
                gt_row = test_grid_params_df[test_grid_params_df['Run Number'] == run_num]
                if len(gt_row) > 0:
                    ground_truth_params = {
                        'disk.mdot': np.log10(gt_row['Disk.mdot(msol/yr)'].values[0]),
                        'wind.mdot': gt_row['Wind.mdot(msol/yr)'].values[0] / gt_row['Disk.mdot(msol/yr)'].values[0],  # Ratio
                        'KWD.d': gt_row['KWD.d(in_units_of_rstar)'].values[0],
                        'KWD.mdot_r_exponent': gt_row['KWD.mdot_r_exponent'].values[0],
                        'KWD.acceleration_length': np.log10(gt_row['KWD.acceleration_length(cm)'].values[0]),
                        'KWD.acceleration_exponent': gt_row['KWD.acceleration_exponent'].values[0],
                        'Inclination': inc_val
                    }

            data_source_info = f"Test Grid: {test_run_selector.value} @ {inc_val}°"

        except Exception as e:
            mo.output.replace(mo.callout(mo.md(f"❌ Error loading test grid: {e}"), kind="danger"))

    elif obs_file_selector.value:
        # Load from observation file (CSV)
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
                 mo.output.replace(mo.callout(mo.md("❌ Invalid file format. Metadata columns 'Wavelength' and 'Flux' not found."), kind="danger"))

        except Exception as e:
             mo.output.replace(mo.callout(mo.md(f"❌ Error reading file: {e}"), kind="danger"))

    # --- Auto-detect flux scale from emulator filename ---
    # Emulator filenames encode the scale: e.g. ..._linear_850-1850AA_... or ..._log_850-1850AA_...
    _detected_scale = "linear"  # default
    if emulator_selector.value:
        _emu_name = emulator_selector.value.lower()
        if '_log_' in _emu_name:
            _detected_scale = "log"
        elif '_scaled_' in _emu_name:
            _detected_scale = "scaled"

    obs_flux_scale = mo.ui.dropdown(
        options=["linear", "log", "scaled"],
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
def _(alt, data_source_info, mo, np, obs_data, obs_flux_scale, pd, wl_range_slider):
    # Obs Plot (Reactive to range slider and flux scale)
    obs_chart = None
    if obs_data is not None:
         # Filter based on range slider
         _current_min = wl_range_slider.value[0]
         _current_max = wl_range_slider.value[1]

         _mask = (obs_data['wavelength'] >= _current_min) & (obs_data['wavelength'] <= _current_max)
         _plot_df = obs_data[_mask].copy()

         # Apply flux transformation to match emulator scale
         _flux_vals = np.array(_plot_df['flux'])
         _scale_label = obs_flux_scale.value
         if _scale_label == 'log':
             _flux_vals = np.where(_flux_vals > 0, np.log10(_flux_vals), np.log10(np.abs(_flux_vals) + 1e-30))
         elif _scale_label == 'scaled':
             _flux_mean = np.mean(_flux_vals)
             if _flux_mean != 0:
                 _flux_vals = _flux_vals / _flux_mean

         _plot_df = pd.DataFrame({
             'Wavelength': _plot_df['wavelength'],
             'Flux': _flux_vals,
             'Type': 'Observation'
         })

         # Downsample for visualization if too large (>5000 points)
         if len(_plot_df) > 5000:
             _plot_df = _plot_df.iloc[::int(len(_plot_df)/5000)]

         _y_title = f'Flux ({_scale_label})'
         _y_format = '.1e' if _scale_label == 'linear' else '.2f'

         # For log scale, set explicit y-limits based on data range to avoid
         # Altair defaulting to include 0 (which compresses the plot)
         if _scale_label == 'log':
             _y_min = float(np.nanmin(_flux_vals))
             _y_max = float(np.nanmax(_flux_vals))
             _y_pad = (_y_max - _y_min) * 0.05
             _y_scale = alt.Scale(domain=[_y_min - _y_pad, _y_max + _y_pad])
         else:
             _y_scale = alt.Undefined

         obs_chart = alt.Chart(_plot_df).mark_line(color='cyan').encode(
             x=alt.X('Wavelength', title='Wavelength (Å)'),
             y=alt.Y('Flux', title=_y_title, scale=_y_scale, axis=alt.Axis(format=_y_format)),
             tooltip=['Wavelength', 'Flux']
         ).properties(
             width="container",
             height=400,
             title=f"{data_source_info} [{_scale_label}]" if data_source_info else "Spectrum"
         ).interactive()

    obs_plot_accordion = mo.accordion({
        "📈 View Selected Spectrum": obs_chart if obs_chart else mo.md("No data loaded.")
    })
    return (obs_plot_accordion,)


@app.cell
def _(
    data_source_info,
    data_source_selector,
    delete_obs_btn,
    emu,
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
    test_inclination_selector,
    test_run_selector,
    wl_range_slider,
):
    # Display Status Stack
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

    # Ground Truth Display (for test grid validation)
    gt_display = None
    if ground_truth_params and _is_test_grid:
        gt_lines = ["**Ground Truth Parameters:**"]
        for k, v in ground_truth_params.items():
            gt_lines.append(f"- **{k}**: {v:.4f}")
        gt_display = mo.callout(mo.md("\n".join(gt_lines)), kind="info")

    # Layout Construction

    # 1. Emulator Row
    # Includes Grid Selector, Emulator Selector, and Parameter Key
    emu_row = mo.vstack([
        grid_selector,
        emulator_selector,
        param_info_accordion
    ])

    # 2. Data Source Row - conditional UI based on selection
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
def _(emu, grid_indices, mo, param_map_db, re):

    param_names = []
    defaults = {}
    bounds = {}

    # Parameter Mapping Dictionary
    if emu is not None:
        current_names = []
        sorted_indices = sorted(list(grid_indices)) if grid_indices else []

        # Attempt to map Emulator params (usually "param1", "param2") to Names using grid_indices
        # We assume grid_indices found in filename correspond 1:1 with emulator dimensions when sorted.
        if len(sorted_indices) == len(emu.param_names):
             for idx in sorted_indices:
                 if idx in param_map_db:
                     _p_name = param_map_db[idx][0]
                     current_names.append(_p_name)
                 else:
                     current_names.append(f"Param {idx} (Unknown)")
        else:
            # Fallback: Try to parse index from the param name itself if it is "paramX"
            for _p in emu.param_names:
                match = re.search(r'param(\d+)', _p)
                if match:
                    idx = int(match.group(1))
                    if idx in param_map_db:
                        current_names.append(param_map_db[idx][0])
                    else:
                        current_names.append(f"Param {idx}")
                else:
                    current_names.append(_p)

        param_names.extend(current_names)

        # Setup bounds (Physical Parameters)
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
    # These deal with Extinction, Scaling, and GP Noise properties
    inf_params = [
        'Av',        # Extinction (magnitudes)
        'log_scale', # Log Flux Scaling Factor (natural log) - combines distance/solid angle
        'log_amp',   # GP Global Covariance Amplitude (natural log)
        'log_ls'     # GP Global Covariance Length Scale (natural log)
    ] 

    # Only add if emu is loaded, otherwise empty
    if emu is not None:
        param_names.extend(inf_params)

        # Defaults for Global Params
        bounds['Av'] = [0.0, 5.0]
        defaults['Av'] = 0.0

        # log_scale: natural log of flux scaling factor
        # Typically auto-calculated if not set, but can be optimized
        # Range allows for large scaling adjustments
        bounds['log_scale'] = [-80.0, 10.0]
        defaults['log_scale'] = -35.0  # Typical for linear-scale spectra

        # log_amp: GP global covariance amplitude (natural log)
        # For linear-scale spectra, typically around -52 (see Speculate_dev.py)
        bounds['log_amp'] = [-80.0, 0.0]
        defaults['log_amp'] = -52.0

        # log_ls: GP global covariance length scale (natural log)
        # Typical range ~0.1 to 11 Angstroms (see Speculate_dev.py)
        bounds['log_ls'] = [0.1, 11.0]
        defaults['log_ls'] = 5.0

    # Create Widgets
    # w_fix uses mo.ui.dictionary so checkbox changes are reactive
    _fix_dict = {}
    _val_dict = {}
    w_min = {}
    w_max = {}

    for _name in param_names:
        _fix_dict[_name] = mo.ui.checkbox(value=False)

        mn, mx = bounds.get(_name, [0.0, 1.0])
        default = defaults.get(_name, 0.5)

        _val_dict[_name] = mo.ui.number(value=default, step=0.01, label="", full_width=True)
        w_min[_name] = mo.ui.number(value=mn, step=0.01, label="Min", full_width=True)
        w_max[_name] = mo.ui.number(value=mx, step=0.01, label="Max", full_width=True)

    # Wrap in mo.ui.dictionary for Marimo reactivity
    w_fix = mo.ui.dictionary(_fix_dict) if _fix_dict else mo.ui.dictionary({})
    w_val = mo.ui.dictionary(_val_dict) if _val_dict else mo.ui.dictionary({})
    return bounds, param_names, w_fix, w_max, w_min, w_val


@app.cell(hide_code=True)
def _(bounds, mo, param_names, w_fix, w_max, w_min, w_val):
    # Render Parameter UI
    if not param_names:
         param_settings = mo.md("*Load an emulator to see parameters.*")
    else:
        # Parameters that use a normal prior instead of uniform
        _normal_prior_params = {'log_amp'}

        # Friendly display names
        _label_map = {
            'log_scale': 'Log Scale (ln)',
            'log_amp': 'GP Log Amp (ln)',
            'log_ls': 'GP Log Length (ln)',
        }

        # Read the reactive dictionary values
        _fix_values = w_fix.value   # dict: {name: bool}
        _val_elements = w_val.elements  # dict: {name: widget}

        # --- Build Grid Parameters Section ---
        _grid_rows = []
        _global_rows = []

        for _name in param_names:
            is_fixed = _fix_values.get(_name, False)
            display_name = _label_map.get(_name, _name)
            _mn, _mx = bounds.get(_name, [0.0, 1.0])
            is_global = _name in ('Av', 'log_scale', 'log_amp', 'log_ls')

            # Fix toggle — access the individual checkbox widget from w_fix
            _fix_widget = w_fix.elements[_name]

            if is_fixed:
                # FIXED: show value input + locked indicator
                _prior_col = mo.hstack([
                    _val_elements[_name],
                    mo.md(f"<span style='color: orange;'>🔒 Fixed</span>"),
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

        param_settings = mo.vstack(_sections)

    mo.vstack([
        mo.md("### Prior Configuration"),
        mo.callout(mo.md(
            "Set prior distributions for each parameter. "
            "**Fix** a parameter to lock it at a specific value (tight δ-prior). "
            "Free parameters use **Uniform** priors (lower–upper) except "
            "GP Log Amp which uses a **Normal** prior (center ± 2σ)."
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
    # MLE State - stores results for MCMC stage
    get_mle_model, set_mle_model = mo.state(None)
    get_mle_priors, set_mle_priors = mo.state(None)
    # MCMC State - stores results for export
    get_mcmc_samples, set_mcmc_samples = mo.state(None)
    get_mcmc_labels, set_mcmc_labels = mo.state(None)
    get_mcmc_summary_df, set_mcmc_summary_df = mo.state(None)
    return (
        get_mle_model, get_mle_priors, set_mle_model, set_mle_priors,
        get_mcmc_samples, set_mcmc_samples,
        get_mcmc_labels, set_mcmc_labels,
        get_mcmc_summary_df, set_mcmc_summary_df,
    )


@app.cell
def _(mo):
    # Inference Control
    run_mle_btn = mo.ui.run_button(label="🚀 Run MLE", kind="success")

    mo.vstack([
        mo.md("Run Maximum Likelihood Estimation (MLE) to find the best fit parameters."),
        run_mle_btn
    ])
    return (run_mle_btn,)


@app.cell
def _(
    Spectrum,
    SpectrumModel,
    emu,
    ground_truth_params,
    mo,
    np,
    obs_data,
    param_names,
    run_mle_btn,
    set_mle_model,
    set_mle_priors,
    stats,
    w_fix,
    w_max,
    w_min,
    w_val,
    obs_flux_scale,
    wl_range_slider,
):
    # Perform MLE Inference
    fit_status = None
    fit_results = None

    if run_mle_btn.value:
        if emu is None or obs_data is None:
            mo.stop(True, mo.md("Please load both an emulator and data."))

        # Use output context for live updates
        with mo.status.spinner("Preparing MLE...") as _spinner:
            try:
                from scipy.optimize import minimize as scipy_minimize
                import time as _time

                # 1. Prepare Data
                _min_w = wl_range_slider.value[0]
                _max_w = wl_range_slider.value[1]

                _mask = (obs_data['wavelength'] >= _min_w) & (obs_data['wavelength'] <= _max_w)
                _data_subset = obs_data[_mask].copy()

                # Apply flux transformation to match emulator training scale
                _flux_scale = obs_flux_scale.value
                _raw_flux = np.array(_data_subset['flux'])
                if _flux_scale == 'log':
                    # Guard against non-positive values
                    _raw_flux = np.where(_raw_flux > 0, np.log10(_raw_flux), np.log10(np.abs(_raw_flux) + 1e-30))
                elif _flux_scale == 'scaled':
                    _flux_mean = np.mean(_raw_flux)
                    if _flux_mean != 0:
                        _raw_flux = _raw_flux / _flux_mean
                # else: linear (no transform)
                _data_subset['flux'] = _raw_flux

                # Use error column if present, otherwise assume 5%
                if 'error' in _data_subset.columns:
                    _sigma = np.array(_data_subset['error'])
                    # Transform errors to match scale
                    if _flux_scale == 'log':
                        # Propagate error through log10: sigma_log = sigma / (val * ln(10))
                        _orig_flux = np.array(obs_data[_mask]['flux'])
                        _sigma = _sigma / (np.abs(_orig_flux) * np.log(10) + 1e-30)
                    elif _flux_scale == 'scaled':
                        _sigma = _sigma / _flux_mean if _flux_mean != 0 else _sigma
                else:
                    _sigma = np.abs(_raw_flux) * 0.05

                # Ensure no zero/negative sigmas
                _sigma = np.maximum(_sigma, np.abs(_data_subset['flux'].values) * 0.01)

                spec_data = Spectrum(
                    np.array(_data_subset['wavelength']),
                    np.array(_data_subset['flux']),
                    sigmas=_sigma
                )

                # 2. Prepare Parameters
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

                # Handle log_scale (flux scaling)
                # If user has fixed log_scale, use their value
                # If free, let Starfish auto-calculate initially, then optimize
                if 'log_scale' in param_names and w_fix.value['log_scale']:
                    global_params['log_scale'] = w_val.value['log_scale']

                # 3. Initialize Model
                _model = SpectrumModel(
                       emulator=emu, 
                       data=spec_data, 
                       grid_params=grid_params_init,
                       **global_params
                )

                # Setup Priors
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

                # C. log_scale Prior (if not fixed and user wants to optimize it)
                if 'log_scale' in param_names:
                    if w_fix.value['log_scale']:
                        _model.params['log_scale'] = w_val.value['log_scale']
                        _model.freeze('log_scale')
                    else:
                        # Bootstrap log_scale from auto-calculation
                        try:
                            _ = _model()  # Trigger one evaluation
                            if _model._log_scale is not None and np.isfinite(_model._log_scale):
                                _model.params['log_scale'] = _model._log_scale
                        except:
                            _model.params['log_scale'] = 0.0

                        _mn = w_min['log_scale'].value
                        _mx = w_max['log_scale'].value
                        _priors['log_scale'] = stats.uniform(loc=_mn, scale=_mx - _mn)

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

                # 4. Build Initial Simplex
                # Spans the prior space for ALL active parameters (grid + hyperparams)
                # so Nelder-Mead starts from a sensible region instead of a single point.
                _spinner.update("Building initial simplex...")

                _active_labels = list(_model.labels)
                _N = len(_active_labels)

                def _get_loc_scale(dist):
                    """Extract loc and scale from a frozen scipy distribution,
                    regardless of whether it was built with positional or keyword args."""
                    args = dist.args
                    kwds = dist.kwds
                    if len(args) >= 2:
                        return args[0], args[1]
                    elif len(args) == 1:
                        return args[0], kwds.get('scale', 1.0)
                    else:
                        return kwds.get('loc', 0.0), kwds.get('scale', 1.0)

                def _simplex_column_uniform(loc, scale, N):
                    """Evenly spaced points across a truncated uniform range."""
                    mn = loc
                    mx = loc + scale
                    rng = mx - mn
                    margin = rng / 20  # 5% inset on each end
                    t_mn, t_mx = mn + margin, mx - margin
                    interval = (t_mx - t_mn) / N
                    return [t_mn + interval * k for k in range(N + 1)]

                def _simplex_column_norm(mean, std, N):
                    """Evenly spaced points across ±2σ of a normal prior."""
                    mn = mean - 2 * std
                    mx = mean + 2 * std
                    interval = (mx - mn) / N
                    return [mn + interval * k for k in range(N + 1)]

                _simplex = np.zeros((_N + 1, _N))
                _simplex_info = []

                for _col_idx, _label in enumerate(_active_labels):
                    if _label in _priors:
                        _dist = _priors[_label]
                        _loc, _sc = _get_loc_scale(_dist)
                        if _dist.dist.name == 'uniform':
                            _col = _simplex_column_uniform(_loc, _sc, _N)
                        elif _dist.dist.name == 'norm':
                            _col = _simplex_column_norm(_loc, _sc, _N)
                        else:
                            # Fallback: small perturbation around current value
                            _cv = _model.get_param_vector()[_col_idx]
                            _col = [_cv + (_cv * 0.01 * k) for k in range(_N + 1)]
                    else:
                        # Parameter is active but has no explicit prior — use current value
                        _cv = _model.get_param_vector()[_col_idx]
                        _col = [_cv] * (_N + 1)

                    _simplex[:, _col_idx] = _col
                    _simplex_info.append(
                        f"  {_label}: [{min(_col):.4f} .. {max(_col):.4f}]")

                    # Roll each column by its index so rows are offset from each other
                    _simplex[:, _col_idx] = np.roll(
                        _simplex[:, _col_idx], _col_idx)

                # --- Bootstrap log_scale into the simplex ---
                # Evaluate the model at each simplex vertex to auto-calculate
                # the appropriate log_scale, giving the optimizer a head start.
                if 'log_scale' in _active_labels:
                    _ls_idx = _active_labels.index('log_scale')
                    for _row in range(_N + 1):
                        try:
                            _model.set_param_vector(_simplex[_row])
                            _ = _model()  # triggers auto log_scale calc
                            if (_model._log_scale is not None
                                    and np.isfinite(_model._log_scale)):
                                _simplex[_row, _ls_idx] = _model._log_scale
                        except Exception:
                            pass  # keep the prior-based value

                # Restore model to the centroid of the simplex
                _centroid = _simplex.mean(axis=0)
                _model.set_param_vector(_centroid)

                # 5. Run Training (MLE) with Progress Tracking
                _spinner.update("Running MLE Optimization...")

                # Track optimization progress
                _nll_history = []
                _iter_count = [0]
                _start_time = _time.time()
                _max_iter = 10000

                def _nll_with_callback(P):
                    _model.set_param_vector(P)
                    nll = -_model.log_likelihood(_priors)
                    _nll_history.append(nll)
                    _iter_count[0] += 1

                    # Update spinner every 50 iterations
                    if _iter_count[0] % 50 == 0:
                        _elapsed = _time.time() - _start_time
                        _best_nll = min(_nll_history) if _nll_history else nll
                        _spinner.update(
                            f"MLE Iteration {_iter_count[0]}/{_max_iter} | "
                            f"Best NLL: {_best_nll:.2f} | "
                            f"Time: {_elapsed:.1f}s"
                        )
                    return nll

                _p0 = _model.get_param_vector()
                _soln = scipy_minimize(
                    _nll_with_callback, 
                    _p0, 
                    method="Nelder-Mead",
                    options=dict(
                        maxiter=_max_iter,
                        disp=False,
                        adaptive=True,
                        initial_simplex=_simplex,
                    )
                )

                if _soln.success:
                    _model.set_param_vector(_soln.x)

                # Store for MCMC
                set_mle_model(_model)
                set_mle_priors(_priors)

                _elapsed_total = _time.time() - _start_time
                fit_status = mo.callout(
                    mo.md(f"✅ MLE Complete! ({_iter_count[0]} iterations, {_elapsed_total:.1f}s)"),
                    kind="success"
                )

                # 6. Plotting
                import matplotlib.pyplot as _plt
                _model.plot(yscale="linear")
                _fig = _plt.gcf()
                _fig.tight_layout()

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

                # 7. Result extraction with ground truth comparison
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
                    _results_md += f"| **{_p}** | {fitted_val:.4f} | "

                    if ground_truth_params and _p in ground_truth_params:
                        _gt_val = ground_truth_params[_p]
                        delta = fitted_val - _gt_val
                        _results_md += f"{_gt_val:.4f} | {delta:+.4f} |\n"
                    elif ground_truth_params:
                        _results_md += "- | - |\n"
                    else:
                        _results_md += "\n"

                # Add Global Params
                _results_md += "\n**Global Parameters:**\n"
                if 'Av' in res_global: 
                    _results_md += f"- **Av**: {res_global['Av']:.4f}\n"
                if 'log_scale' in res_global: 
                    _results_md += f"- **log_scale**: {res_global['log_scale']:.4f}\n"
                elif hasattr(_model, '_log_scale') and _model._log_scale is not None:
                    _results_md += f"- **log_scale (auto)**: {_model._log_scale:.4f}\n"
                if 'global_cov:log_amp' in res_global:
                    _results_md += f"- **log_amp**: {res_global['global_cov:log_amp']:.4f}\n"
                if 'global_cov:log_ls' in res_global:
                    _results_md += f"- **log_ls**: {res_global['global_cov:log_ls']:.4f}\n"

                # Build results display
                _result_elements = [
                    fit_status,
                    mo.md(_results_md),
                ]
                if _fig_loss is not None:
                    _result_elements.extend([
                        mo.md("### Convergence"),
                        _fig_loss
                    ])
                _result_elements.extend([
                    mo.md("### Model Fit & Residuals"),
                    _fig
                ])

                fit_results = mo.vstack(_result_elements)

            except Exception as _e:
                import traceback as _traceback
                _tb = _traceback.format_exc()
                fit_status = mo.callout(mo.md(f"❌ MLE Failed: {str(_e)}\n\n```\n{_tb}\n```"), kind="danger")
                fit_results = fit_status
                set_mle_model(None)
                set_mle_priors(None)

    fit_results if fit_results else mo.md("*Click 'Run MLE' to start inference*")
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
    # MCMC Configuration
    _model = get_mle_model()

    mcmc_nwalkers = mo.ui.number(value=32, label="Walkers", step=4, start=8)
    mcmc_nsteps = mo.ui.number(value=1000, label="Steps", step=100, start=100)
    mcmc_burnin = mo.ui.number(value=200, label="Burn-in", step=50, start=0)

    run_mcmc_btn = mo.ui.run_button(
        label="🎲 Run MCMC", 
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
        mcmc_label_map['log_scale'] = 'log_scale'
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
        mcmc_config_ui = mo.callout(mo.md("⚠️ Run MLE first before MCMC"), kind="warn")
    else:
        # Show friendly names in the active params list
        _friendly_labels = [mcmc_label_map.get(l, l) for l in _model.labels]
        mcmc_config_ui = mo.vstack([
            mo.md(f"**Active Parameters:** {', '.join(_friendly_labels)}"),
            mo.hstack([mcmc_nwalkers, mcmc_nsteps, mcmc_burnin], justify="start"),
            mo.md("### 🔒 Freeze Parameters for MCMC"),
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
    get_mle_model,
    get_mle_priors,
    ground_truth_params,
    mcmc_burnin,
    mcmc_freeze,
    mcmc_label_map,
    mcmc_nsteps,
    mcmc_nwalkers,
    mo,
    np,
    run_mcmc_btn,
    set_mcmc_labels,
    set_mcmc_samples,
    set_mcmc_summary_df,
):
    # Run MCMC
    mcmc_results = None

    if run_mcmc_btn.value:
        _model = get_mle_model()
        _priors = get_mle_priors()

        if _model is None or _priors is None:
            mo.stop(True, mo.md("Please run MLE first."))

        # Apply freezes from UI checkboxes
        _freeze_values = mcmc_freeze.value  # {internal_name: bool}
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
                matplotlib.rcParams['figure.max_open_warning'] = 50
                _warnings.filterwarnings("ignore", module="arviz", message="More chains")

                _nwalkers = mcmc_nwalkers.value
                _nsteps = mcmc_nsteps.value
                _burnin_manual = mcmc_burnin.value
                _ndim = len(_model.labels)

                # Define log probability function
                def _log_prob(P, model, priors):
                    model.set_param_vector(P)
                    return model.log_likelihood(priors)

                # Initialize walkers in a ball around MLE solution
                # Scales use both internal and friendly names for robust lookup
                _default_scales = {
                    "disk.mdot": 0.05, "wind.mdot": 0.02, "KWD.d": 0.1,
                    "KWD.mdot_r_exponent": 0.05, "KWD.acceleration_length": 0.1,
                    "KWD.acceleration_exponent": 0.1,
                    "Boundary_layer.luminosity": 0.05, "Boundary_layer.temp": 0.05,
                    "Inclination (Sparse)": 2.0, "Inclination (Mid)": 2.0,
                    "Inclination (Full)": 1.0,
                    "param1": 0.05, "param2": 0.02, "param3": 0.1,
                    "param4": 0.05, "param5": 0.1, "param6": 0.1,
                    "param7": 0.05, "param8": 0.05, "param9": 2.0,
                    "param10": 2.0, "param11": 1.0,
                    "global_cov:log_amp": 1.0, "global_cov:log_ls": 0.5,
                    "GP log_amp": 1.0, "GP log_ls": 0.5,
                    "Av": 0.1, "log_scale": 0.5
                }

                _ball = np.random.randn(_nwalkers, _ndim)
                for _i, _key in enumerate(_model.labels):
                    # Try internal name first, then friendly name
                    _scale = _default_scales.get(_key, _default_scales.get(_friendly(_key), 0.1))
                    _ball[:, _i] *= _scale
                    _ball[:, _i] += _model[_key]

                # Create sampler
                _sampler = _emcee.EnsembleSampler(
                    _nwalkers, _ndim, _log_prob, args=(_model, _priors)
                )

                # Run with progress bar
                for _ in mo.status.progress_bar(range(_nsteps), title="MCMC Sampling"):
                    _sampler.run_mcmc(_ball if _sampler.iteration == 0 else None, 1, progress=False)

                # ============================================================
                # Stage 15: Raw MCMC Chains (Full, pre burn-in)
                # ============================================================
                # _sampler.get_chain() shape: (nsteps, nwalkers, ndim)
                _full_chain = _sampler.get_chain()

                # Build arviz InferenceData for full chain
                # arviz from_dict expects {name: (chains, draws)}
                # Use friendly names for display
                _friendly_labels = [_friendly(l) for l in _model.labels]
                _full_dd = {}
                for _i, _label in enumerate(_friendly_labels):
                    _full_dd[_label] = _full_chain[:, :, _i].T  # (walkers, steps)
                _full_data = _az.from_dict(posterior=_full_dd)

                # Plot full chains (trace plot)
                _fig_full_trace = _az.plot_trace(_full_data)
                _fig_full_chain = _fig_full_trace.ravel()[0].figure
                _fig_full_chain.suptitle("Raw MCMC Chains (Full)", y=1.02)
                _fig_full_chain.tight_layout()

                # ============================================================
                # Stage 16: Autocorrelation & Burn-in
                # ============================================================
                # Compute autocorrelation time
                try:
                    _tau = _sampler.get_autocorr_time(tol=0)
                    _tau_valid = not (np.isnan(_tau).any() or (_tau == 0).any())
                except Exception:
                    _tau = np.full(_ndim, np.nan)
                    _tau_valid = False

                if _tau_valid:
                    _auto_burnin = int(_tau.max())
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
                for _i, _label in enumerate(_friendly_labels):
                    _burn_dd[_label] = _burn_chain[:, :, _i].T  # (walkers, steps_after_burn)
                _burn_data = _az.from_dict(posterior=_burn_dd)

                # Trace plot (post burn-in)
                _fig_burn_trace = _az.plot_trace(_burn_data)
                _fig_burn_chain = _fig_burn_trace.ravel()[0].figure
                _fig_burn_chain.suptitle("MCMC Chains (Post Burn-in)", y=1.02)
                _fig_burn_chain.tight_layout()

                # Summary table
                _summary_df = _az.summary(_burn_data, round_to=5)
                _summary_md = _summary_df.to_markdown()

                # Store MCMC results in state for export cell
                set_mcmc_samples(_burn_samples.copy())
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
                # Build truths for corner plot from ground truth
                # Use the friendly label map to look up ground truth keys
                _truths = None
                if ground_truth_params:
                    _truths = []
                    _has_any = False
                    for _label in _model.labels:
                        _gt_key = _friendly(_label)
                        # Handle inclination variants → common key
                        if 'Inclination' in _gt_key:
                            _gt_key = 'Inclination'
                        if _gt_key in ground_truth_params:
                            _truths.append(ground_truth_params[_gt_key])
                            _has_any = True
                        else:
                            _truths.append(None)
                    if not _has_any:
                        _truths = None

                _fig_corner = _corner.corner(
                    _burn_samples,
                    labels=_friendly_labels,
                    show_titles=True,
                    quantiles=[0.16, 0.5, 0.84],
                    title_fmt=".4f",
                    truths=_truths
                )

                # ============================================================
                # Stage 19: Best-fit MCMC model plot
                # ============================================================
                # Set model to posterior means (using internal labels)
                _mcmc_means = {}
                for _i, _label in enumerate(_model.labels):
                    _mcmc_means[_label] = float(np.mean(_burn_samples[:, _i]))
                _model.set_param_dict(_mcmc_means)

                _model.plot(yscale="linear")
                _fig_bestfit = _plt.gcf()
                _fig_bestfit.suptitle("Best-Fit Model (MCMC Posterior Mean)", y=1.02)
                _fig_bestfit.tight_layout()

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
                    _mean = np.mean(_burn_samples[:, _i])
                    _std = np.std(_burn_samples[:, _i])
                    _median = np.median(_burn_samples[:, _i])

                    _results_md += f"| **{_display_name}** | {_mean:.4f} | {_std:.4f} | {_median:.4f} | "

                    if ground_truth_params:
                        _gt_key = _display_name
                        if 'Inclination' in _gt_key:
                            _gt_key = 'Inclination'
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
                        f"✅ MCMC Complete! ({_nsteps} steps, "
                        f"{_nwalkers} walkers, burn-in {_burnin_used}, thin {_thin_used}"
                        f"{_frozen_info})"
                    ),
                    kind="success"
                )

                _chain_accordion = mo.accordion({
                    "🔗 Raw MCMC Chains (Full)": mo.vstack([
                        mo.md("Walker traces for all parameters before burn-in removal."),
                        _fig_full_chain
                    ]),
                    "🔥 Burnt MCMC Chains (Post Burn-in)": mo.vstack([
                        mo.md(_burnin_info),
                        _fig_burn_chain
                    ]),
                    "📊 Posterior Distributions": mo.vstack([
                        _fig_post
                    ]),
                    "📋 Arviz Summary Statistics": mo.vstack([
                        mo.md(_summary_md)
                    ]),
                })

                mcmc_results = mo.vstack([
                    _status_callout,
                    mo.md(_results_md),
                    _chain_accordion,
                    mo.md("### Corner Plot"),
                    _fig_corner,
                    mo.md("### Best-Fit Model (MCMC Posterior Mean)"),
                    _fig_bestfit,
                ])

            except Exception as _e:
                import traceback as _traceback
                _tb = _traceback.format_exc()
                mcmc_results = mo.callout(mo.md(f"❌ MCMC Failed: {str(_e)}\n\n```\n{_tb}\n```"), kind="danger")

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
    get_mcmc_samples,
    get_mcmc_summary_df,
    get_mle_model,
    mo,
    np,
    os,
    time,
):
    _samples = get_mcmc_samples()
    _labels = get_mcmc_labels()
    _summary_df = get_mcmc_summary_df()
    _model = get_mle_model()

    if _samples is None or _model is None:
        mo.output.replace(mo.callout(mo.md("Run MCMC first to enable export."), kind="neutral"))
        mo.stop(True)

    export_pf_btn = mo.ui.run_button(label="📄 Export .pf Template", kind="success")
    export_csv_btn = mo.ui.run_button(label="📊 Export Posterior CSV", kind="success")
    export_dir_input = mo.ui.text(value="benchmark_results/exports", label="Output directory")

    mo.vstack([
        mo.hstack([export_dir_input], gap=1),
        mo.hstack([export_pf_btn, export_csv_btn], gap=1),
    ])
    return export_csv_btn, export_dir_input, export_pf_btn


@app.cell
def _(
    emu,
    export_csv_btn,
    export_dir_input,
    export_pf_btn,
    get_mcmc_labels,
    get_mcmc_samples,
    get_mcmc_summary_df,
    get_mle_model,
    mo,
    np,
    os,
    time,
):
    _samples = get_mcmc_samples()
    _labels = get_mcmc_labels()
    _summary_df = get_mcmc_summary_df()
    _model = get_mle_model()
    _out_dir = export_dir_input.value or "benchmark_results/exports"
    _ts = time.strftime("%Y%m%d_%H%M%S")

    _msg = ""

    if export_pf_btn.value and _model is not None and _samples is not None:
        try:
            from Speculate_addons.speculate_benchmark import export_pf_template, emulator_to_physical

            os.makedirs(_out_dir, exist_ok=True)

            # Posterior means for grid params
            _n_grid = len(emu.param_names)
            _grid_means = np.mean(_samples[:, :_n_grid], axis=0)

            # Uncertainties
            _uncertainties = {}
            for _i, _label in enumerate(_labels[:_n_grid]):
                _lo = np.percentile(_samples[:, _i], 16)
                _hi = np.percentile(_samples[:, _i], 84)
                _uncertainties[_label] = (_lo, _hi)

            # Global params
            _global = {}
            for _i in range(_n_grid, _samples.shape[1]):
                _global[_labels[_i]] = float(np.mean(_samples[:, _i]))

            _pf_path = os.path.join(_out_dir, f"speculate_export_{_ts}.pf")
            export_pf_template(
                emu, _grid_means, _pf_path,
                uncertainties=_uncertainties,
                global_params=_global,
            )
            _msg += f"✅ .pf template exported to `{_pf_path}`\n\n"
        except Exception as _e:
            _msg += f"❌ .pf export failed: {_e}\n\n"

    if export_csv_btn.value and _samples is not None:
        try:
            from Speculate_addons.speculate_benchmark import export_posterior_csv

            os.makedirs(_out_dir, exist_ok=True)

            # Build summary dict from DataFrame if available
            _summary_dict = None
            if _summary_df is not None:
                _summary_dict = {}
                for _idx_name in _summary_df.index:
                    _row = _summary_df.loc[_idx_name]
                    _summary_dict[str(_idx_name)] = {
                        k: float(v) for k, v in _row.items() if np.isfinite(v)
                    }

            _csv_path = os.path.join(_out_dir, f"speculate_posterior_{_ts}.csv")
            export_posterior_csv(_samples, _labels, _csv_path, summary=_summary_dict)
            _msg += f"✅ Posterior CSV exported to `{_csv_path}`\n\n"
        except Exception as _e:
            _msg += f"❌ CSV export failed: {_e}\n\n"

    if _msg:
        mo.output.replace(mo.callout(mo.md(_msg), kind="success"))
    return


if __name__ == "__main__":
    app.run()
