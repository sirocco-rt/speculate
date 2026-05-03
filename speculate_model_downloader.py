# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
#
# Speculate Model Downloader
# =========================
# Download Sirocco spectral grids and pre-trained emulator models from
# HuggingFace repositories hosted under the **Sirocco-rt** organisation. The tool:
#
#   1. Discovers available grid datasets via the HuggingFace Hub API.
#   2. Lists the compressed spectrum files (.spec.xz) and auxiliary
#      metadata (lookup table, README, etc.) in the selected dataset.
#   3. Downloads selected files into the HuggingFace cache, then
#      decompresses them with LZMA into ``sirocco_grids/<dataset>/``.
#   4. Downloads selected pre-trained GP and Quick Fit emulator models into
#      ``Grid-Emulator_Files/``.
#
# Once extracted, the files are consumable by the Training Tool, Grid
# Inspector, Benchmark Suite, and Inference Tool.

import marimo

__generated_with = "0.23.1"
app = marimo.App(
    width="full",
    app_title="Speculate Model Downloader",
    layout_file="layouts/speculate_model_downloader.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md(
        """
        # Model Downloader

        Download Sirocco spectral grids and pre-trained emulator models from HuggingFace.

        This tool fetches spectral grids and curated Speculate emulator models from the **Sirocco-rt** organization on HuggingFace.
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
    from Speculate_addons.hf_model_registry import (
        HF_MODEL_REPO_ID,
        download_model_to_local,
        is_gp_model,
        is_quickfit_model,
        list_hf_model_files,
        model_type_label,
    )
    import logging

    # Suppress huggingface_hub logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # All Speculate grid datasets are published under the Sirocco-rt org
    # on HuggingFace as "dataset" repos (not model repos).
    ORG_ID = "Sirocco-rt"
    REPO_TYPE = "dataset"
    return (
        HF_MODEL_REPO_ID,
        ORG_ID,
        REPO_TYPE,
        download_model_to_local,
        hf_hub_download,
        is_gp_model,
        is_quickfit_model,
        list_datasets,
        list_hf_model_files,
        list_repo_files,
        lzma,
        model_type_label,
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
            dataset_status = mo.md(f"{mo.icon('lucide:check-circle')} Found **{len(grid_datasets)}** grid datasets")
        else:
            dataset_status = mo.md(f"{mo.icon('lucide:triangle-alert')} No grid datasets found")

    except Exception as e:
        grid_datasets = []
        dataset_status = mo.md(f"{mo.icon('lucide:x-circle')} Error fetching datasets: {e}")

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

    if dataset_dropdown is not None and dataset_dropdown.value:
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
            file_info = mo.md(f"{mo.icon('lucide:x-circle')} Error fetching files: {e}")
    else:
        file_info = mo.md(f"{mo.icon('lucide:triangle-alert')} Please select a dataset first")

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
            _display = mo.md(f"{mo.icon('lucide:triangle-alert')} No files available")

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
            lookup_status = mo.md(f"{mo.icon('lucide:check-circle')} Loaded parameter lookup table with **{len(lookup_df)}** runs")

        except Exception as e:
            lookup_status = mo.md(f"{mo.icon('lucide:triangle-alert')} Could not load parameter lookup table: {e}")

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
                    _param_display = mo.md(f"{mo.icon('lucide:triangle-alert')} No parameters found for run {run_number}")

            except Exception as e:
                _param_display = mo.md(f"{mo.icon('lucide:triangle-alert')} Error displaying parameters: {e}")

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
            download_summary = mo.md(f"{mo.icon('lucide:package')} Ready to download **{len(files_to_download)}** files")
        else:
            # "Specific file" mode keeps the workflow identical but collapses the
            # target list to a single selected run.
            if specific_file_dropdown is not None and specific_file_dropdown.value:
                files_to_download = [specific_file_dropdown.value]
                download_summary = mo.md(f"{mo.icon('lucide:package')} Ready to download: `{specific_file_dropdown.value}`")
            else:
                files_to_download = []
                download_summary = mo.md(f"{mo.icon('lucide:triangle-alert')} Please select a file")
    else:
        download_summary = mo.md(f"{mo.icon('lucide:triangle-alert')} No files available")

    # The extracted files always land under a dataset-named local directory so
    # other tools can find them with the same naming convention.
    _summary_display = download_summary
    if files_to_download and dataset_dropdown is not None and dataset_dropdown.value:
        save_location = os.path.abspath(f"sirocco_grids/{dataset_dropdown.value}/")
        location_info = mo.md(f"{mo.icon('lucide:folder')} Files will be saved to: `{save_location}`")
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
    download_and_decompress_button = mo.ui.run_button(label=f"{mo.icon('lucide:download')} Download & Decompress")
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
                    mo.output.append(mo.md(f"{mo.icon('lucide:x-circle')} Failed to download `{filename}`: {e}"))
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
                mo.output.append(mo.md(f"{mo.icon('lucide:triangle-alert')} Could not fetch auxiliary file `{_aux}`: {e}"))

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
                            mo.output.append(mo.md(f"{mo.icon('lucide:x-circle')} Failed to decompress `{filename}`: {e}"))
                    bar.update()

            mo.md(f"### {mo.icon('lucide:check-circle')} Complete!\n\nFiles saved to: `{os.path.abspath(extraction_dir)}`")
        else:
            mo.md(f"### {mo.icon('lucide:check-circle')} Download complete!\n\nFiles cached by HuggingFace (use 'Decompress' to extract)")

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
                    mo.output.append(mo.md(f"{mo.icon('lucide:x-circle')} Failed to decompress `{filename}`: {e}"))
                bar.update()

        mo.md(f"### {mo.icon('lucide:check-circle')} Complete!\n\nFiles saved to: `{os.path.abspath(extraction_dir)}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Pre-trained Emulator Models

    Download curated GP and Quick Fit emulator models from HuggingFace Models into `Grid-Emulator_Files/`.
    """)
    return


@app.cell
def _(
    HF_MODEL_REPO_ID,
    is_gp_model,
    is_quickfit_model,
    list_hf_model_files,
    mo,
):
    hf_model_files = []
    try:
        hf_model_files = list_hf_model_files()
        _n_gp = sum(1 for _f in hf_model_files if is_gp_model(_f))
        _n_qf = sum(1 for _f in hf_model_files if is_quickfit_model(_f))
        if hf_model_files:
            model_repo_status = mo.md(
                f"{mo.icon('lucide:check-circle')} Found **{len(hf_model_files)}** models "
                f"in `{HF_MODEL_REPO_ID}` (**{_n_gp}** GP, **{_n_qf}** Quick Fit)"
            )
        else:
            model_repo_status = mo.md(f"{mo.icon('lucide:triangle-alert')} No supported `.npz` models found in `{HF_MODEL_REPO_ID}`")
    except Exception as e:
        hf_model_files = []
        model_repo_status = mo.md(f"{mo.icon('lucide:x-circle')} Error fetching model list: {e}")

    model_repo_status
    return (hf_model_files,)


@app.cell
def _(hf_model_files, mo, model_type_label):
    model_download_mode = mo.ui.radio(
        options=["All models", "All GP emulators", "All Quick Fit models", "Specific model"],
        value="All models",
        label="Which pre-trained models would you like to download?",
    )

    if hf_model_files:
        _options = {f"[{model_type_label(_f)}] {_f}": _f for _f in hf_model_files}
        specific_model_dropdown = mo.ui.dropdown(
            options=_options,
            value=next(iter(_options.keys())),
            label="Select model:",
            full_width=True,
        )
    else:
        specific_model_dropdown = None

    model_download_mode
    return model_download_mode, specific_model_dropdown


@app.cell
def _(mo, model_download_mode, specific_model_dropdown):
    _display = None
    if model_download_mode.value == "Specific model":
        if specific_model_dropdown is not None:
            _display = specific_model_dropdown
        else:
            _display = mo.md(f"{mo.icon('lucide:triangle-alert')} No models available")

    _display
    return


@app.cell
def _(
    hf_model_files,
    is_gp_model,
    is_quickfit_model,
    mo,
    model_download_mode,
    os,
    specific_model_dropdown,
):
    models_to_download = []

    if hf_model_files:
        if model_download_mode.value == "All models":
            models_to_download = hf_model_files
        elif model_download_mode.value == "All GP emulators":
            models_to_download = [_f for _f in hf_model_files if is_gp_model(_f)]
        elif model_download_mode.value == "All Quick Fit models":
            models_to_download = [_f for _f in hf_model_files if is_quickfit_model(_f)]
        elif specific_model_dropdown is not None and specific_model_dropdown.value:
            models_to_download = [specific_model_dropdown.value]

    if models_to_download:
        _destination = os.path.abspath("Grid-Emulator_Files")
        model_download_summary = mo.vstack([
            mo.md(f"{mo.icon('lucide:package')} Ready to download **{len(models_to_download)}** model file(s)"),
            mo.md(f"{mo.icon('lucide:folder')} Models will be saved to: `{_destination}`"),
        ])
    else:
        model_download_summary = mo.md(f"{mo.icon('lucide:triangle-alert')} No model files selected")

    model_download_summary
    return (models_to_download,)


@app.cell
def _(mo):
    model_download_button = mo.ui.run_button(label=f"{mo.icon('lucide:download')} Download Selected Models")
    model_download_button
    return (model_download_button,)


@app.cell
def _(
    download_model_to_local,
    mo,
    model_download_button,
    models_to_download,
    os,
):
    if model_download_button.value and models_to_download:
        _downloaded = []
        _skipped = []
        _failed = []

        with mo.status.progress_bar(total=len(models_to_download), title="Downloading models...") as _bar:
            for _filename in models_to_download:
                try:
                    _result = download_model_to_local(_filename)
                    if _result["status"] == "downloaded":
                        _downloaded.append(_filename)
                    else:
                        _skipped.append(_filename)
                except Exception as e:
                    _failed.append((_filename, str(e)))
                    mo.output.append(mo.md(f"{mo.icon('lucide:x-circle')} Failed to download `{_filename}`: {e}"))
                _bar.update()

        _lines = [
            f"### {mo.icon('lucide:check-circle')} Model Download Complete",
            "",
            f"- Downloaded: **{len(_downloaded)}**",
            f"- Already present: **{len(_skipped)}**",
            f"- Failed: **{len(_failed)}**",
            f"- Location: `{os.path.abspath('Grid-Emulator_Files')}`",
        ]
        mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
