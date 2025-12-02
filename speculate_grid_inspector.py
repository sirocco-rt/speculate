# /// script
# [tool.marimo.display]
# theme = "dark"
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
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

    # Enable VegaFusion for large datasets
    alt.data_transformers.enable("vegafusion")

    # Detect environment
    IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None
    return IS_HUGGINGFACE_SPACE, Path, alt, np, os, pd


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
def _(IS_HUGGINGFACE_SPACE, cache_tracker_state, grid_selector, mo):
    # Clear cache when grid changes (HuggingFace mode only)
    if IS_HUGGINGFACE_SPACE:
        get_cache_check, set_cache_check = cache_tracker_state

        # Track previous grid to detect changes
        import os as os_cache

        prev_grid_key = "_prev_selected_grid"
        prev_grid = os_cache.environ.get(prev_grid_key, "")
        current_grid = grid_selector.value if grid_selector else ""

        if prev_grid and current_grid and prev_grid != current_grid:
            # Grid changed - clear cache
            cached_files = get_cache_check()
            cleared = 0
            for file_path in cached_files:
                if os_cache.path.exists(file_path):
                    try:
                        os_cache.remove(file_path)
                        cleared += 1
                    except Exception:
                        pass
            set_cache_check(set())

            if cleared > 0:
                mo.md(f"🧹 Cleared {cleared} cached spectra from previous grid")

        # Update tracked grid
        if current_grid:
            os_cache.environ[prev_grid_key] = current_grid

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
    # Initialize state for pinned spectra storage
    get_pinned_spectra, set_pinned_spectra = mo.state([])

    # Initialize cache tracker for HuggingFace downloads
    cache_tracker_state = mo.state(set())
    return cache_tracker_state, get_pinned_spectra, set_pinned_spectra


@app.cell
def _(
    current_run_index,
    get_pinned_spectra,
    inclination_selector,
    mo,
    set_pinned_spectra,
):
    # Define button callbacks
    def add_current_spectrum():
        current_combo = (current_run_index, inclination_selector.value)
        current_list = list(get_pinned_spectra())
        if current_combo not in current_list:
            current_list.append(current_combo)
            set_pinned_spectra(current_list)

    def clear_all_spectra():
        set_pinned_spectra([])

    # Create buttons with on_click callbacks
    add_spectrum_btn = mo.ui.button(
        label="➕ Add Spectrum", 
        kind="success",
        on_click=lambda _: add_current_spectrum()
    )
    clear_spectra_btn = mo.ui.button(
        label="🗑️ Clear All", 
        kind="danger",
        on_click=lambda _: clear_all_spectra()
    )

    mo.hstack([add_spectrum_btn, clear_spectra_btn], justify="start", gap=1)
    return


@app.cell
def _(get_pinned_spectra, mo):
    # Display pinned spectra status
    pinned_spectra_indices = list(get_pinned_spectra())
    num_pinned = len(pinned_spectra_indices)

    # Display status message
    if num_pinned > 0:
        pinned_labels = [f"run{run}@{inc}°" for run, inc in pinned_spectra_indices]
        status_msg = mo.md(f"📌 **Pinned Spectra**: {num_pinned} ({', '.join(pinned_labels)})")
    else:
        status_msg = mo.md("_No pinned spectra_")

    status_msg
    return (pinned_spectra_indices,)


@app.cell
def _(
    Path,
    cache_tracker_state,
    current_run_index,
    grid_mode,
    grid_path,
    hf_hub_download,
    inclination_angles,
    inclination_selector,
    lookup_df,
    lzma,
    np,
    os,
    param_cols,
    repo_id,
):
    # Cache management functions
    get_cache, set_cache = cache_tracker_state

    def clear_spectrum_cache():
        """Clear all cached spectrum files from HuggingFace downloads"""
        cached_files = get_cache()
        cleared_count = 0

        for file_path in cached_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleared_count += 1
                except Exception:
                    pass  # Ignore errors, file might be in use

        # Reset cache tracker
        set_cache(set())
        return cleared_count

    def track_cached_file(file_path):
        """Add file to cache tracker"""
        cached_files = get_cache()
        cached_files.add(file_path)
        set_cache(cached_files)

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

            # Track this file in cache
            track_cached_file(spec_path_temp)

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
    current_flux, current_params = load_spectrum(current_run_index, inclination_selector.value)
    return current_flux, current_params, load_spectrum


@app.cell
def _(
    alt,
    current_flux,
    current_run_index,
    inclination_selector,
    load_spectrum,
    pd,
    pinned_spectra_indices,
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

    # Process and add current spectrum data
    if show_current.value:
        current_flux_plot = current_flux.copy()

        # Apply dimensionless transformation if requested
        if use_dimensionless.value:
            norm_factor = current_flux_plot.mean()
            current_flux_plot /= norm_factor
            current_flux_plot -= current_flux_plot.mean()

        current_df = pd.DataFrame({
            'Wavelength': filtered_wavelengths,
            'Flux': current_flux_plot[wl_mask],
            'Spectrum': f'run{current_run_index}@{inclination_selector.value}°',
            'Type': 'Current'
        })
        plot_data_list.append(current_df)

    # Load and add pinned spectra data
    # Force reactivity by explicitly checking the list
    num_pinned_to_plot = len(pinned_spectra_indices)

    if show_fixed.value and num_pinned_to_plot > 0:
        for run_idx, inclination in pinned_spectra_indices:
            try:
                pinned_flux, pinned_params = load_spectrum(run_idx, inclination)

                # Apply dimensionless transformation if requested
                if use_dimensionless.value:
                    pinned_flux_plot = pinned_flux.copy()
                    norm_factor = pinned_flux_plot.mean()
                    pinned_flux_plot /= norm_factor
                    pinned_flux_plot -= pinned_flux_plot.mean()
                else:
                    pinned_flux_plot = pinned_flux

                pinned_df = pd.DataFrame({
                    'Wavelength': filtered_wavelengths,
                    'Flux': pinned_flux_plot[wl_mask],
                    'Spectrum': f'run{run_idx}@{inclination}°',
                    'Type': 'Pinned'
                })
                plot_data_list.append(pinned_df)
            except Exception as e:
                pass  # Skip failed loads

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
    - Spectra are cached within a grid session for fast browsing
    - Each spectrum is ~2.5MB, first load from cloud, then cached
    - Cache automatically cleared when switching grids
    - HuggingFace Space: ~1-2s first load, <0.5s cached loads

    ### Tips
    - Pin interesting spectra before changing parameters
    - Use dimensionless mode to compare spectral shapes
    - The interactive plot supports zoom, pan, and save
    """)
    return


if __name__ == "__main__":
    app.run()
