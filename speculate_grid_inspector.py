# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="full",
    app_title="Speculate Grid Inspector",
    layout_file="layouts/speculate_grid_inspector.grid.json",
)


@app.cell
async def _(mo):
    import asyncio
    with mo.status.spinner(title="Loading Grid Inspector...") as _spinner:
        await asyncio.sleep(0.5)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Grid Inspector

    Visualize and analyse the sirocco spectral grid data interactively.
    """)
    return


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
    # Add logo using marimo's image display
    import pathlib

    logo_path = pathlib.Path("assets/logos/Speculate_logo2.png")

    logo = mo.image(src=str(logo_path), width=400)
    link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    mo.vstack([mo.md("---"), logo, link], align="center")
    return


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import altair as alt

    # Detect environment
    IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None
    return IS_HUGGINGFACE_SPACE, Path, alt, np, pd


@app.cell
def _(mo):
    mo.md("""
    ## Grid Selection:
    """)
    return


@app.cell
def _(IS_HUGGINGFACE_SPACE, mo):
    # Environment mode display
    if IS_HUGGINGFACE_SPACE:
        env_badge = mo.md("🌐 **Running on HuggingFace Space** - Streaming data from cloud")
    else:
        env_badge = mo.md("💻 **Running Locally** - Using pre-downloaded grid files")

    env_badge
    return


@app.cell
def _(IS_HUGGINGFACE_SPACE, Path, mo):
    # Grid selection based on environment
    if IS_HUGGINGFACE_SPACE:
        # HuggingFace: select from available datasets
        available_grids = [
            "speculate_cv_bl_grid_v87f",
            "speculate_cv_no-bl_grid_v87f",
        ]
        grid_selector = mo.ui.dropdown(
            options=available_grids,
            value="speculate_cv_bl_grid_v87f",
            label="Select Grid Dataset:"
        )
    else:
        # Local: scan raw_grids directory
        raw_grids_path = Path("raw_grids")
        if raw_grids_path.exists():
            available_grids = [
                d.name for d in raw_grids_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ]
        else:
            available_grids = []

        if available_grids:
            grid_selector = mo.ui.dropdown(
                options=available_grids,
                value=available_grids[0] if available_grids else None,
                label="Select Local Grid:"
            )
        else:
            grid_selector = None
            mo.md("⚠️ **No grids found in `raw_grids/` directory**\n\nPlease download grids using the Grid Downloader tool first.")

    grid_selector
    return (grid_selector,)


@app.cell
def _(grid_selector, mo):
    # Only proceed if grid is selected
    if grid_selector is None or grid_selector.value is None:
        mo.md("Please download a grid first using the Grid Downloader tool.")
        mo.stop()
    return


@app.cell
def _(IS_HUGGINGFACE_SPACE, Path, grid_selector, mo, np, pd):
    # Load grid metadata
    import lzma

    with mo.status.spinner(title="Loading grid metadata..."):
        selected_grid = grid_selector.value

        if IS_HUGGINGFACE_SPACE:
            # HuggingFace mode: load lookup table
            from huggingface_hub import hf_hub_url

            ORG_ID = "sirocco-rt"
            repo_id = f"{ORG_ID}/{selected_grid}"

            try:
                # Load lookup table
                lookup_url = hf_hub_url(
                    repo_id=repo_id,
                    filename="grid_run_lookup_table.parquet",
                    repo_type="dataset"
                )
                lookup_df = pd.read_parquet(lookup_url)

                # Extract parameter columns (exclude Run Number)
                param_cols = [col for col in lookup_df.columns if col != 'Run Number']

                # Each row is a unique parameter combination (one run file)
                num_runs = len(lookup_df)

                # Load wavelengths and inclination angles from first spectrum
                from huggingface_hub import hf_hub_download
                first_file = f"run{lookup_df['Run Number'].iloc[0]}.spec.xz"
                spec_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=first_file,
                    repo_type="dataset"
                )
                with lzma.open(spec_path, 'rt') as f:
                    # Find where data starts by looking for "Freq." header
                    lines = f.readlines()
                    f.seek(0)
                    skiprows = 0
                    header_line = None
                    for i, line in enumerate(lines):
                        if 'Freq.' in line:
                            skiprows = i + 1
                            header_line = line
                            break

                    # Extract inclination angles from header (e.g., A30P0.50 -> 30)
                    inclination_angles = []
                    if header_line:
                        parts = header_line.split()
                        for part in parts:
                            if part.startswith('A') and 'P' in part:
                                angle = int(part[1:part.index('P')])
                                inclination_angles.append(angle)

                    data = np.loadtxt(f, skiprows=skiprows, unpack=True)
                    wavelengths = np.flip(data[1])  # Column 1 is Lambda (wavelength)

                grid_mode = "huggingface"

            except Exception as e:
                mo.md(f"❌ Error loading grid from HuggingFace: {e}")
                mo.stop()

        else:
            # Local mode: scan directory for spec files
            grid_path = Path("raw_grids") / selected_grid

            if not grid_path.exists():
                mo.md(f"❌ Grid directory not found: {grid_path}")
                mo.stop()

            # Find all .spec files
            spec_files = sorted(grid_path.glob("run*.spec"))

            if not spec_files:
                mo.md(f"❌ No .spec files found in {grid_path}")
                mo.stop()

            # Try to load lookup table - download from HuggingFace if not present
            lookup_path = grid_path / "grid_run_lookup_table.parquet"

            if not lookup_path.exists():
                # Try to download from HuggingFace
                try:
                    from huggingface_hub import hf_hub_download as local_hf_download
                    ORG_ID = "sirocco-rt"
                    local_repo_id = f"{ORG_ID}/{selected_grid}"

                    mo.md("📥 Lookup table not found locally. Downloading from HuggingFace...")

                    downloaded_path = local_hf_download(
                        repo_id=local_repo_id,
                        filename="grid_run_lookup_table.parquet",
                        repo_type="dataset",
                        local_dir=grid_path,
                        local_dir_use_symlinks=False
                    )
                    lookup_path = Path(downloaded_path)
                except Exception as e:
                    mo.md(f"⚠️ No lookup table found locally and failed to download from HuggingFace: {e}")
                    mo.stop()

            # Load lookup table
            lookup_df = pd.read_parquet(lookup_path)
            param_cols = [col for col in lookup_df.columns if col != 'Run Number']

            # Each row is a unique parameter combination (one run file)
            num_runs = len(lookup_df)

            # Load wavelengths and inclination angles from first spectrum
            first_spec = spec_files[0]
            with open(first_spec, 'r') as f:
                # Find where data starts by looking for "Freq." header
                lines = f.readlines()
                f.seek(0)
                skiprows = 0
                header_line = None
                for i, line in enumerate(lines):
                    if 'Freq.' in line:
                        skiprows = i + 1
                        header_line = line
                        break

                # Extract inclination angles from header (e.g., A30P0.50 -> 30)
                inclination_angles = []
                if header_line:
                    parts = header_line.split()
                    for part in parts:
                        if part.startswith('A') and 'P' in part:
                            angle = int(part[1:part.index('P')])
                            inclination_angles.append(angle)

                data = np.loadtxt(f, skiprows=skiprows, unpack=True)
                wavelengths = np.flip(data[1])  # Column 1 is Lambda (wavelength)

            grid_mode = "local"

        num_runs = len(lookup_df)
        num_wavelengths = len(wavelengths)
        num_inclinations = len(inclination_angles)

    mo.md(f"""
    ✅ **Grid loaded successfully**
    - **Grid**: `{selected_grid}`
    - **Mode**: `{grid_mode}`
    - **Run Files**: {num_runs:,}
    - **Inclinations per Run**: {num_inclinations} ({', '.join(map(str, inclination_angles))}°)
    - **Wavelength Points**: {num_wavelengths:,}
    - **Wavelength Range**: {wavelengths.min():.1f} - {wavelengths.max():.1f} Å
    - **Parameters**: {len(param_cols)} ({', '.join(param_cols)})
    """)
    return (
        grid_mode,
        grid_path,
        hf_hub_download,
        inclination_angles,
        lookup_df,
        lzma,
        num_runs,
        param_cols,
        repo_id,
        selected_grid,
        wavelengths,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Controls:
    """)
    return


@app.cell
def _(inclination_angles, mo, num_runs):
    # Run file selector slider with built-in editable input
    run_slider = mo.ui.slider(
        start=0,
        stop=num_runs - 1,
        value=0,
        step=1,
        label="Run File Index:",
        show_value=True,
        include_input=True
    )

    # Inclination angle selector
    inclination_selector = mo.ui.dropdown(
        options={f"{angle}°": angle for angle in inclination_angles},
        value=f"{inclination_angles[0]}°" if inclination_angles else None,
        label="Inclination Angle:"
    )

    mo.vstack([
        run_slider,
        inclination_selector
    ])
    return inclination_selector, run_slider


@app.cell
def _(run_slider):
    # Use slider value directly - the built-in input is already integrated
    current_run_index = run_slider.value
    return (current_run_index,)


@app.cell
def _(mo, wavelengths):
    # Wavelength range selector - exclude first and last 10 Å to avoid Sirocco artifacts
    wl_min_limit = float(wavelengths.min() + 10)
    wl_max_limit = float(wavelengths.max() - 10)

    wavelength_range = mo.ui.range_slider(
        start=wl_min_limit,
        stop=wl_max_limit,
        value=[wl_min_limit, wl_max_limit],
        step=1.0,
        label="Wavelength Range (Å):",
        show_value=True
    )

    wavelength_range
    return (wavelength_range,)


@app.cell
def _(mo):
    # Visibility and display options
    show_current = mo.ui.checkbox(value=True, label="Show Current Spectrum")
    show_fixed = mo.ui.checkbox(value=True, label="Show Fixed Spectra")
    use_dimensionless = mo.ui.checkbox(value=False, label="Dimensionless Data")

    mo.hstack([show_current, show_fixed, use_dimensionless], justify="start", gap=2)
    return show_current, show_fixed, use_dimensionless


@app.cell
def _(mo):
    # Buttons for managing fixed spectra
    add_spectrum_btn = mo.ui.button(label="➕ Add Spectrum", kind="success")
    clear_spectra_btn = mo.ui.button(label="🗑️ Clear All", kind="danger")

    mo.hstack([add_spectrum_btn, clear_spectra_btn], justify="start", gap=1)
    return add_spectrum_btn, clear_spectra_btn


@app.cell
def _(mo):
    # Initialize state for fixed spectra storage
    fixed_spectra_state = mo.state([])
    return (fixed_spectra_state,)


@app.cell
def _(
    add_spectrum_btn,
    clear_spectra_btn,
    current_run_index,
    fixed_spectra_state,
    inclination_selector,
    mo,
):
    # Manage fixed spectra using marimo state
    get_fixed, set_fixed = fixed_spectra_state

    # Store both run index and inclination angle
    current_combo = (current_run_index, inclination_selector.value)

    # Add current spectrum when button clicked
    if add_spectrum_btn.value:
        current_list = get_fixed()
        if current_combo not in current_list:  # Avoid duplicates
            set_fixed(current_list + [current_combo])

    # Clear all fixed spectra when button clicked
    if clear_spectra_btn.value:
        set_fixed([])

    # Get current list of fixed (run, inclination) tuples
    fixed_spectra_list = get_fixed()
    num_fixed = len(fixed_spectra_list)

    if num_fixed > 0:
        fixed_labels = [f"run{run}@{inc}°" for run, inc in fixed_spectra_list]
        mo.md(f"📌 **Fixed Spectra**: {num_fixed} pinned ({', '.join(fixed_labels)})")
    return (fixed_spectra_list,)


@app.cell
def _(
    Path,
    current_run_index,
    fixed_spectra_list,
    grid_mode,
    grid_path,
    hf_hub_download,
    inclination_angles,
    inclination_selector,
    lookup_df,
    lzma,
    np,
    param_cols,
    repo_id,
    use_dimensionless,
):
    # Function to load a spectrum by run index and inclination
    def load_spectrum_with_skiprows(file_handle):
        """Helper to find correct skiprows and load data"""
        # Read lines to find where "Freq." header is
        lines = file_handle.readlines()
        file_handle.seek(0)  # Reset file position

        # Find the line with "Freq." header
        skiprows = 0
        for i, line in enumerate(lines):
            if 'Freq.' in line:
                skiprows = i + 1  # Skip header line + 1 to get to data
                break

        # Load data skipping the determined number of rows
        data = np.loadtxt(file_handle, skiprows=skiprows, unpack=True)
        return data

    def load_spectrum(run_idx, inclination):
        """Load spectrum flux by run index and inclination angle"""
        # Find which column corresponds to the inclination
        col_idx = inclination_angles.index(inclination) + 2  # +2 because cols 0=Freq, 1=Lambda

        if grid_mode == "huggingface":
            # HuggingFace: download and decompress
            run_number = lookup_df['Run Number'].iloc[run_idx]
            spec_file = f"run{run_number}.spec.xz"

            spec_path_temp = hf_hub_download(
                repo_id=repo_id,
                filename=spec_file,
                repo_type="dataset"
            )

            with lzma.open(spec_path_temp, 'rt') as f:
                data = load_spectrum_with_skiprows(f)
                flux = np.flip(data[col_idx])

        else:
            # Local: read from file using run number from lookup table
            run_number = lookup_df['Run Number'].iloc[run_idx]
            spec_file_path = Path(grid_path) / f"run{run_number}.spec"

            with open(spec_file_path, 'r') as f:
                data = load_spectrum_with_skiprows(f)
                flux = np.flip(data[col_idx])

        # Get parameters for this run from lookup table
        params = {col: lookup_df[col].iloc[run_idx] for col in param_cols}

        return flux, params

    # Load current spectrum
    # with mo.status.spinner(title="Loading spectrum..."):
    current_flux, current_params = load_spectrum(current_run_index, inclination_selector.value)

    # Load fixed spectra
    fixed_spectra_data = []
    for run_idx, inclination in fixed_spectra_list:
        try:
            fixed_flux, fixed_params = load_spectrum(run_idx, inclination)
            fixed_spectra_data.append({
                'run_idx': run_idx,
                'inclination': inclination,
                'flux': fixed_flux,
                'params': fixed_params
            })
        except Exception as e:
            pass  # Skip failed loads

    # Dimensionless transformation if requested
    if use_dimensionless.value:
        # Normalize current spectrum
        current_flux_plot = current_flux.copy()
        norm_factor = current_flux_plot.mean()
        current_flux_plot /= norm_factor
        current_flux_plot -= current_flux_plot.mean()

        # Normalize fixed spectra
        for fixed_data in fixed_spectra_data:
            fixed_flux_copy = fixed_data['flux'].copy()
            norm_factor = fixed_flux_copy.mean()
            fixed_flux_copy /= norm_factor
            fixed_flux_copy -= fixed_flux_copy.mean()
            fixed_data['flux_plot'] = fixed_flux_copy
    else:
        current_flux_plot = current_flux
        for fixed_data in fixed_spectra_data:
            fixed_data['flux_plot'] = fixed_data['flux']
    return current_flux_plot, current_params, fixed_spectra_data


@app.cell
def _(
    alt,
    current_flux_plot,
    current_params,
    current_run_index,
    fixed_spectra_data,
    inclination_selector,
    pd,
    selected_grid,
    show_current,
    show_fixed,
    use_dimensionless,
    wavelength_range,
    wavelengths,
):
    # Get wavelength range filter
    wl_min, wl_max = wavelength_range.value
    wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    filtered_wavelengths = wavelengths[wl_mask]

    # Prepare data for Altair
    plot_data_list = []

    # Add current spectrum data
    if show_current.value:
        current_param_label = ', '.join([f'{col}={val:.3e}' for col, val in current_params.items()])
        current_df = pd.DataFrame({
            'Wavelength': filtered_wavelengths,
            'Flux': current_flux_plot[wl_mask],
            'Spectrum': f'run{current_run_index}@{inclination_selector.value}°',
            'Type': 'Current'
        })
        plot_data_list.append(current_df)

    # Add fixed spectra data
    if show_fixed.value and len(fixed_spectra_data) > 0:
        for idx, plot_fixed_data in enumerate(fixed_spectra_data):
            fixed_df = pd.DataFrame({
                'Wavelength': filtered_wavelengths,
                'Flux': plot_fixed_data['flux_plot'][wl_mask],
                'Spectrum': f'run{plot_fixed_data["run_idx"]}@{plot_fixed_data["inclination"]}°',
                'Type': 'Fixed'
            })
            plot_data_list.append(fixed_df)

    # Combine all data
    if plot_data_list:
        plot_df = pd.concat(plot_data_list, ignore_index=True)

        # Create Altair chart
        spectrum_chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('Wavelength:Q', title='Wavelength (Å)', scale=alt.Scale(zero=False)),
            y=alt.Y('Flux:Q', 
                    title='Flux' if not use_dimensionless.value else 'Normalized Flux',
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format='~e')),
            color=alt.Color('Spectrum:N', 
                           scale=alt.Scale(scheme='viridis'),
                           legend=alt.Legend(title='Spectra')),
            strokeWidth=alt.condition(
                alt.datum.Type == 'Current',
                alt.value(3),
                alt.value(1.5)
            ),
            opacity=alt.condition(
                alt.datum.Type == 'Current',
                alt.value(0.9),
                alt.value(0.7)
            )
        ).properties(
            width=1000,
            height=600,
            title=f'Spectral Data Exploration - {selected_grid}'
        ).interactive()
    else:
        spectrum_chart = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_point().encode(
            x='x:Q',
            y='y:Q'
        ).properties(
            width=900,
            height=400,
            title='No spectra to display'
        )

    spectrum_chart
    return


@app.cell
def _(current_params, current_run_index, inclination_selector, lookup_df, mo):
    # Display current parameters
    run_number = lookup_df['Run Number'].iloc[current_run_index]
    params_table_text = "### Current Spectrum Parameters\n\n"
    params_table_text += f"**Run Index**: {current_run_index} | **Run Number**: {run_number} | **Inclination**: {inclination_selector.value}°\n\n"
    params_table_text += "| Parameter | Value |\n"
    params_table_text += "|-----------|-------|\n"

    for param_name, param_value in current_params.items():
        params_table_text += f"| {param_name} | {param_value:.6e} |\n"

    mo.md(params_table_text)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Usage Guide

    ### Navigation
    - **Slider**: Drag to browse through spectra sequentially
    - **Index Input**: Jump directly to a specific spectrum by index
    - **➕ Add Spectrum**: Pin the current spectrum for comparison
    - **🗑️ Clear All**: Remove all pinned spectra

    ### Display Options
    - **Show Current Spectrum**: Toggle visibility of the active spectrum
    - **Show Fixed Spectra**: Toggle visibility of all pinned spectra
    - **Dimensionless Data**: Normalize spectra (mean-centered) for comparison

    ### Performance Notes
    - Spectra are loaded on-demand (no caching)
    - Each spectrum is ~2.5MB, loads in <0.1s locally
    - HuggingFace Space mode streams from cloud (slower)
    - Pinned spectra are reloaded on each view

    ### Tips
    - Pin interesting spectra before changing parameters
    - Use dimensionless mode to compare spectral shapes
    - The interactive plot supports zoom, pan, and save
    """)
    return


if __name__ == "__main__":
    app.run()
