# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="full",
    app_title="Speculate Grid Downloader",
    layout_file="layouts/speculate_grid_downloader.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md(
        """
        # Grid Downloader

        Download and decompress Sirocco spectral grid files from HuggingFace datasets.

        This tool fetches spectral grids from the **Sirocco-rt** organization on HuggingFace.
        """
    )
    return (mo,)


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
    # Imports and configuration
    import os
    import lzma
    import shutil
    from typing import List, Tuple

    # Disable progress bars from huggingface_hub
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_VERBOSITY'] = 'error'

    from huggingface_hub import list_datasets, list_repo_files, hf_hub_download
    import logging

    # Suppress huggingface_hub logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    ORG_ID = "Sirocco-rt"
    REPO_TYPE = "dataset"
    return (
        ORG_ID,
        REPO_TYPE,
        hf_hub_download,
        list_datasets,
        list_repo_files,
        lzma,
        os,
        shutil,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fetching Available Datasets:
    """)
    return


@app.cell
def _(ORG_ID, list_datasets, mo):
    # Fetch available datasets from Sirocco-rt organization
    grid_datasets = []
    try:
        all_datasets = list(list_datasets(author=ORG_ID))
        dataset_ids = [ds.id.split("/")[-1] for ds in all_datasets if ds.id.startswith(ORG_ID)]

        # Filter for speculate grid datasets
        grid_datasets = [d for d in dataset_ids if "grid" in d.lower()]

        if grid_datasets:
            dataset_status = mo.md(f"✓ Found **{len(grid_datasets)}** grid datasets")
        else:
            dataset_status = mo.md("⚠️ No grid datasets found")

    except Exception as e:
        grid_datasets = []
        dataset_status = mo.md(f"❌ Error fetching datasets: {e}")

    dataset_status
    return (grid_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1: Select Dataset
    """)
    return


@app.cell
def _(grid_datasets, mo):
    # Dataset selection dropdown
    if grid_datasets:
        dataset_dropdown = mo.ui.dropdown(
            options=grid_datasets,
            value=grid_datasets[0] if grid_datasets else None,
            label="Select a dataset:"
        )
    else:
        dataset_dropdown = None

    dataset_dropdown
    return (dataset_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Files Identified
    """)
    return


@app.cell
def _(ORG_ID, REPO_TYPE, dataset_dropdown, list_repo_files, mo):
    # Fetch files from selected dataset
    spec_files = []
    repo_id = None

    if dataset_dropdown.value:
        try:
            repo_id = f"{ORG_ID}/{dataset_dropdown.value}"
            all_files = list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE)
            spec_files = [f for f in all_files if f.startswith("run") and f.endswith(".spec.xz")]

            # Sort numerically by extracting the number from the filename
            spec_files.sort(key=lambda x: int(x.replace("run", "").replace(".spec.xz", "")))

            file_info = mo.md(f"""
            **Dataset:** `{dataset_dropdown.value}`

            **Files found:** {len(spec_files)} spectral files (.spec.xz)
            """)

        except Exception as e:
            spec_files = []
            file_info = mo.md(f"❌ Error fetching files: {e}")
    else:
        file_info = mo.md("⚠️ Please select a dataset first")

    file_info
    return repo_id, spec_files


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2: Select Files to Download
    """)
    return


@app.cell
def _(mo, spec_files):
    # Option to download all or specific file
    download_mode = mo.ui.radio(
        options=["All files", "Specific file"],
        value="All files",
        label="What would you like to download?"
    )

    # Specific file dropdown (only shown if needed)
    if spec_files:
        specific_file_dropdown = mo.ui.dropdown(
            options=spec_files,
            value=spec_files[0] if spec_files else None,
            label="Select file:"
        )
    else:
        specific_file_dropdown = None

    # Display UI elements
    download_mode
    return download_mode, specific_file_dropdown


@app.cell
def _(download_mode, mo, specific_file_dropdown):
    # Show file dropdown only when "Specific file" is selected
    _display = None
    if download_mode.value == "Specific file":
        if specific_file_dropdown is not None:
            _display = specific_file_dropdown
        else:
            _display = mo.md("⚠️ No files available")

    _display
    return


@app.cell
def _(ORG_ID, REPO_TYPE, dataset_dropdown, mo):
    # Load the parameter lookup table for the selected dataset
    import pandas as pd
    from huggingface_hub import hf_hub_url

    lookup_df = None
    lookup_status = None

    if dataset_dropdown is not None and dataset_dropdown.value:
        try:
            repo_id_lookup = f"{ORG_ID}/{dataset_dropdown.value}"
            lookup_file = "grid_run_lookup_table.parquet"

            # Get direct URL to the parquet file (no download needed)
            file_url = hf_hub_url(
                repo_id=repo_id_lookup,
                filename=lookup_file,
                repo_type=REPO_TYPE
            )

            # Read the parquet file directly from URL
            lookup_df = pd.read_parquet(file_url)
            lookup_status = mo.md(f"✓ Loaded parameter lookup table with **{len(lookup_df)}** runs")

        except Exception as e:
            lookup_status = mo.md(f"⚠️ Could not load parameter lookup table: {e}")

    lookup_status
    return (lookup_df,)


@app.cell
def _(download_mode, lookup_df, mo, specific_file_dropdown):
    # Display parameters for the selected run (only in specific file mode)
    _param_display = None

    if download_mode.value == "Specific file" and specific_file_dropdown is not None and specific_file_dropdown.value:
        if lookup_df is not None:
            try:
                # Extract run number from filename (e.g., "run145.spec.xz" -> 145)
                selected_file = specific_file_dropdown.value
                run_number = int(selected_file.replace("run", "").replace(".spec.xz", ""))

                # Find the row for this run
                run_row = lookup_df[lookup_df['Run Number'] == run_number]

                if not run_row.empty:
                    # Get all parameter columns (exclude Run_Number)
                    param_cols = [col for col in lookup_df.columns if col != 'Run Number']

                    # Build parameter display
                    params_text = f"### Parameters for `{selected_file}`\n\n"
                    params_text += "| Parameter | Value |\n"
                    params_text += "|-----------|-------|\n"

                    for col in param_cols:
                        value = run_row[col].iloc[0]
                        params_text += f"| {col} | {value} |\n"

                    _param_display = mo.md(params_text)
                else:
                    _param_display = mo.md(f"⚠️ No parameters found for run {run_number}")

            except Exception as e:
                _param_display = mo.md(f"⚠️ Error displaying parameters: {e}")

    _param_display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Download Location
    """)
    return


@app.cell
def _(
    dataset_dropdown,
    download_mode,
    mo,
    os,
    spec_files,
    specific_file_dropdown,
):
    # Determine which files to download
    files_to_download = []

    if spec_files:
        if download_mode.value == "All files":
            files_to_download = spec_files
            download_summary = mo.md(f"📦 Ready to download **{len(files_to_download)}** files")
        else:
            # Specific file mode
            if specific_file_dropdown is not None and specific_file_dropdown.value:
                files_to_download = [specific_file_dropdown.value]
                download_summary = mo.md(f"📦 Ready to download: `{specific_file_dropdown.value}`")
            else:
                files_to_download = []
                download_summary = mo.md("⚠️ Please select a file")
    else:
        download_summary = mo.md("⚠️ No files available")

    # Show save location if files are ready to download
    _summary_display = download_summary
    if files_to_download and dataset_dropdown is not None and dataset_dropdown.value:
        save_location = os.path.abspath(f"sirocco_grids/{dataset_dropdown.value}/")
        location_info = mo.md(f"📁 Files will be saved to: `{save_location}`")
        _summary_display = mo.vstack([download_summary, location_info])

    _summary_display
    return (files_to_download,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3: Download and Decompress
    """)
    return


@app.cell
def _(mo):
    # Action buttons
    download_and_decompress_button = mo.ui.run_button(label="📥 Download & Decompress")
    download_only_button = mo.ui.run_button(label="Download Only")
    decompress_only_button = mo.ui.run_button(label="Decompress Downloaded Files")


    mo.hstack([
        download_and_decompress_button,
        download_only_button,
        decompress_only_button
    ], justify="start", gap=1)
    return (
        decompress_only_button,
        download_and_decompress_button,
        download_only_button,
    )


@app.cell
def _(
    REPO_TYPE,
    dataset_dropdown,
    decompress_only_button,
    download_and_decompress_button,
    download_only_button,
    files_to_download,
    hf_hub_download,
    lzma,
    mo,
    os,
    repo_id,
    shutil,
):
    # Download logic
    download_results = []

    if (download_only_button.value or download_and_decompress_button.value) and files_to_download:
        extraction_dir = f"sirocco_grids/{dataset_dropdown.value}/"

        with mo.status.progress_bar(total=len(files_to_download), title="Downloading...") as bar:
            for i, filename in enumerate(files_to_download, 1):
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=REPO_TYPE
                    )
                    download_results.append((filename, local_path, None))
                except Exception as e:
                    download_results.append((filename, None, str(e)))
                    mo.output.append(mo.md(f"❌ Failed to download `{filename}`: {e}"))
                bar.update()

        # Decompress if requested
        if download_and_decompress_button.value:
            os.makedirs(extraction_dir, exist_ok=True)

            with mo.status.progress_bar(total=len(download_results), title="Decompressing...") as bar:
                for filename, local_path, error in download_results:
                    if local_path:
                        try:
                            output_filename = filename[:-3]  # Remove .xz
                            output_file_path = os.path.join(extraction_dir, output_filename)

                            with lzma.open(local_path, 'rb') as f_in:
                                with open(output_file_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                        except Exception as e:
                            mo.output.append(mo.md(f"❌ Failed to decompress `{filename}`: {e}"))
                    bar.update()

            mo.md(f"### ✅ Complete!\n\nFiles saved to: `{os.path.abspath(extraction_dir)}`")
        else:
            mo.md("### ✅ Download complete!\n\nFiles cached by HuggingFace (use 'Decompress' to extract)")

    elif decompress_only_button.value and files_to_download:
        # Decompress previously downloaded files
        extraction_dir = f"sirocco_grids/{dataset_dropdown.value}/"
        os.makedirs(extraction_dir, exist_ok=True)

        with mo.status.progress_bar(total=len(files_to_download), title="Decompressing cached files...") as bar:
            for filename in files_to_download:
                try:
                    # Files should already be cached
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=REPO_TYPE
                    )

                    output_filename = filename[:-3]
                    output_file_path = os.path.join(extraction_dir, output_filename)

                    with lzma.open(local_path, 'rb') as f_in:
                        with open(output_file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                except Exception as e:
                    mo.output.append(mo.md(f"❌ Failed to decompress `{filename}`: {e}"))
                bar.update()

        mo.md(f"### ✅ Complete!\n\nFiles saved to: `{os.path.abspath(extraction_dir)}`")
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


if __name__ == "__main__":
    app.run()
