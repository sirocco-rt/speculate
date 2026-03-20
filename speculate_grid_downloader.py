# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
#
# Speculate Grid Downloader
# =========================
# Download and decompress Sirocco spectral grid files from HuggingFace
# datasets hosted under the **Sirocco-rt** organisation.  The tool:
#
#   1. Discovers available grid datasets via the HuggingFace Hub API.
#   2. Lists the compressed spectrum files (.spec.xz) and auxiliary
#      metadata (lookup table, README, etc.) in the selected dataset.
#   3. Downloads selected files into the HuggingFace cache, then
#      decompresses them with LZMA into ``sirocco_grids/<dataset>/``.
#
# Once extracted, the files are consumable by the Training Tool, Grid
# Inspector, Benchmark Suite, and Inference Tool.

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

    logo = mo.image(src=str(logo_path), width=400, height=95)
    link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    mo.vstack([mo.md("---"), logo, link], align="center")
    return


@app.cell
def _():
    # ── Imports and HuggingFace configuration ──
    # Progress bars and verbose logging from huggingface_hub are suppressed
    # because the notebook provides its own progress widgets.
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

    # All Speculate grid datasets are published under the Sirocco-rt org
    # on HuggingFace as "dataset" repos (not model repos).
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
    # ── Dataset discovery ──
    # Query the Sirocco-rt org for any dataset whose name contains "grid".
    # Each matching dataset ID becomes a selectable option in Step 1.
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
    # Inspect the selected HuggingFace dataset and split its contents into the
    # compressed spectral payloads versus auxiliary metadata files.
    spec_files = []
    aux_files = []
    repo_id = None

    if dataset_dropdown.value:
        try:
            repo_id = f"{ORG_ID}/{dataset_dropdown.value}"
            all_files = list(list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE))
            spec_files = [f for f in all_files if f.startswith("run") and f.endswith(".spec.xz")]

            # Keep the spectrum files in run-number order so the UI and lookup
            # table refer to the same sequence of simulations.
            spec_files.sort(key=lambda x: int(x.replace("run", "").replace(".spec.xz", "")))

            # Everything else is treated as auxiliary material to be staged next
            # to the spectra after download, except repository bookkeeping files.
            _skip = {".gitattributes", "README.md"}
            aux_files = [
                f for f in all_files
                if f not in _skip and not (f.startswith("run") and f.endswith(".spec.xz"))
            ]

            _aux_note = f" + {len(aux_files)} auxiliary file(s)" if aux_files else ""
            file_info = mo.md(f"""
            **Dataset:** `{dataset_dropdown.value}`

            **Files found:** {len(spec_files)} spectral files (.spec.xz){_aux_note}
            """)

        except Exception as e:
            spec_files = []
            aux_files = []
            file_info = mo.md(f"❌ Error fetching files: {e}")
    else:
        file_info = mo.md("⚠️ Please select a dataset first")

    file_info
    return aux_files, repo_id, spec_files


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
    # ── Lookup-table preview ──
    # Fetch the run-lookup parquet directly over HTTP so the notebook can show
    # parameter metadata without downloading the entire dataset first.  This
    # table maps each run number back to its physical parameter values (e.g.
    # Disk.mdot, KWD.d, Inclination) and is bundled with every published grid.
    import pandas as pd
    from huggingface_hub import hf_hub_url

    lookup_df = None
    lookup_status = None

    if dataset_dropdown is not None and dataset_dropdown.value:
        try:
            repo_id_lookup = f"{ORG_ID}/{dataset_dropdown.value}"
            lookup_file = "grid_run_lookup_table.parquet"

            # Build the signed HuggingFace URL explicitly; pandas can read the
            # parquet stream directly from that endpoint.
            file_url = hf_hub_url(
                repo_id=repo_id_lookup,
                filename=lookup_file,
                repo_type=REPO_TYPE
            )

            # This table powers the per-run parameter preview in "Specific file"
            # mode and mirrors the metadata bundled with the dataset.
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
    # Normalize the UI selection into the exact list of compressed spectrum files
    # the download/decompress cell should act on.
    files_to_download = []

    if spec_files:
        if download_mode.value == "All files":
            files_to_download = spec_files
            download_summary = mo.md(f"📦 Ready to download **{len(files_to_download)}** files")
        else:
            # "Specific file" mode keeps the workflow identical but collapses the
            # target list to a single selected run.
            if specific_file_dropdown is not None and specific_file_dropdown.value:
                files_to_download = [specific_file_dropdown.value]
                download_summary = mo.md(f"📦 Ready to download: `{specific_file_dropdown.value}`")
            else:
                files_to_download = []
                download_summary = mo.md("⚠️ Please select a file")
    else:
        download_summary = mo.md("⚠️ No files available")

    # The extracted files always land under a dataset-named local directory so
    # other tools can find them with the same naming convention.
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
    # Step 3 action buttons.
    # "Download & Decompress" is the typical workflow (cache + extract).
    # The split buttons let the user decouple downloading from extraction,
    # useful when re-extracting from a populated HuggingFace cache.
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
    aux_files,
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
    # Execute the chosen workflow: download only, download+extract, or extract
    # already-cached files from the local HuggingFace cache.
    download_results = []

    if (download_only_button.value or download_and_decompress_button.value) and files_to_download:
        extraction_dir = f"sirocco_grids/{dataset_dropdown.value}/"

        # Download the selected compressed spectra into the HuggingFace cache and
        # keep track of which ones succeeded for the extraction pass.
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

        # Always stage auxiliary files into the extraction directory so the grid
        # inspection and validation tools see the same on-disk layout regardless
        # of whether the user downloaded one spectrum or the whole dataset.
        os.makedirs(extraction_dir, exist_ok=True)
        for _aux in aux_files:
            try:
                _aux_local = hf_hub_download(
                    repo_id=repo_id,
                    filename=_aux,
                    repo_type=REPO_TYPE
                )
                _dest = os.path.join(extraction_dir, _aux)
                shutil.copy2(_aux_local, _dest)
            except Exception as e:
                mo.output.append(mo.md(f"⚠️ Could not fetch auxiliary file `{_aux}`: {e}"))

        # In the combined mode, immediately expand each .xz archive into its raw
        # .spec file under the dataset directory.
        if download_and_decompress_button.value:
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
        # Rehydrate raw .spec files from archives that should already exist in the
        # HuggingFace cache, without forcing a full redownload.
        extraction_dir = f"sirocco_grids/{dataset_dropdown.value}/"
        os.makedirs(extraction_dir, exist_ok=True)

        # Make the extracted directory self-contained by backfilling any missing
        # auxiliary metadata before expanding the spectra.
        for _aux in aux_files:
            _dest = os.path.join(extraction_dir, _aux)
            if not os.path.exists(_dest):
                try:
                    _aux_local = hf_hub_download(
                        repo_id=repo_id,
                        filename=_aux,
                        repo_type=REPO_TYPE
                    )
                    shutil.copy2(_aux_local, _dest)
                except Exception:
                    pass

        with mo.status.progress_bar(total=len(files_to_download), title="Decompressing cached files...") as bar:
            for filename in files_to_download:
                try:
                    # hf_hub_download resolves the existing cached archive path and
                    # only contacts the hub if the cache is missing.
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
