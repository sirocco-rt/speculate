"""Shared CSV/TXT observation parsing, file management and upload controls."""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Speculate_addons.observation_priors import OBSERVATION_PRIORS


OBSERVATION_EXTENSIONS = (".csv", ".txt")
_FORMAT_HELP = (
    "Use WAVELENGTH and FLUX headers, with optional ERROR, or exactly two "
    "or three numeric columns in that order. Separate columns with commas, "
    "tabs, or spaces."
)
OBSERVATION_UPLOAD_HELP = """
- **File types:** CSV or TXT.
- **Columns:** `WAVELENGTH`, `FLUX`, and optional `ERROR`.
- **Separators:** Commas, tabs, or spaces.
- **Headers:** Column names are case-insensitive.
- **Without headers:** Use exactly two or three columns in the order above.
- **Comments:** Blank lines and lines starting with `#` are allowed.
"""

# Keep format examples inline so the help and downloads need no bundled files.
_UPLOAD_EXAMPLES = (
    (
        "CSV",
        "practice_spectrum.csv",
        "Comma-separated columns with a header and optional uncertainty values.",
        "WAVELENGTH,FLUX,ERROR\n"
        "1000,2e-13,1e-14\n"
        "2000,3e-13,1e-14\n"
        "3000,2.5e-13,1e-14\n",
    ),
    (
        "TXT (spaces)",
        "practice_spaces.txt",
        "Space-separated columns. Blank lines and comments beginning with `#` are allowed.",
        "# Synthetic observation\n\n"
        "Wavelength Flux Error\n"
        "1000 2e-13 1e-14\n"
        "2000 3e-13 1e-14\n"
        "3000 2.5e-13 1e-14\n",
    ),
    (
        "TXT (tabs)",
        "practice_tabs.txt",
        "Tab-separated columns. Header names can use any letter case.",
        "Wavelength\tflux\tERROR\n"
        "1000\t2e-13\t1e-14\n"
        "2000\t3e-13\t1e-14\n"
        "3000\t2.5e-13\t1e-14\n",
    ),
    (
        "No headers",
        "practice_headerless.txt",
        "Two columns in order: wavelength, then flux. You can add uncertainty as a third column.",
        "1000 2e-13\n"
        "2000 3e-13\n"
        "3000 2.5e-13\n",
    ),
)


def is_demo_observation(filename: str | None) -> bool:
    """Recognise every shipped CV/AGN demo using the shared catalogue."""
    return bool(filename) and Path(filename).name.lower() in OBSERVATION_PRIORS


def list_observation_files(directory: str | Path = "observation_files") -> list[str]:
    """List CSV/TXT files without parsing, so malformed uploads stay deletable."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in OBSERVATION_EXTENSIONS
    )


def _split_rows(lines: list[str]) -> list[list[str]]:
    """Use one delimiter convention for the whole table, retaining CSV quoting."""
    if "," in lines[0]:
        try:
            return list(csv.reader(lines, skipinitialspace=True, strict=True))
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV: {exc}. {_FORMAT_HELP}") from exc
    return [re.split(r"\s+", line) for line in lines]


def read_observation_file(source: str | Path | bytes) -> pd.DataFrame:
    """Read a file or uploaded bytes into a sorted, lowercase observation table.

    Blank lines and ``#`` comments are allowed. Named tables retain additional
    columns such as ``sigma_source``; headerless tables must have exactly
    wavelength, flux, and optionally error. No units or uncertainty conventions
    are inferred or changed here.
    """
    contents = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Upload a UTF-8 CSV or TXT observation table.") from exc

    lines = []
    commented_header = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Honour a named comment header, including reordered columns,
            # instead of silently treating its data as a headerless table.
            candidate = line.lstrip("#").strip()
            if not lines and candidate:
                try:
                    columns = [
                        value.strip().strip('"').lower()
                        for value in _split_rows([candidate])[0]
                    ]
                except ValueError:
                    continue  # Ordinary comments need not be valid CSV.
                if {"wavelength", "flux"}.issubset(columns):
                    commented_header = candidate
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"No observation data found. {_FORMAT_HELP}")

    rows = _split_rows(lines)
    columns = [value.strip().strip('"').lower() for value in rows[0]]
    if {"wavelength", "flux"}.issubset(columns):
        rows = rows[1:]
    elif commented_header is not None:
        columns = [
            value.strip().strip('"').lower()
            for value in _split_rows([commented_header])[0]
        ]
    else:
        try:
            for value in rows[0]:
                float(value)
        except ValueError as exc:
            raise ValueError(_FORMAT_HELP) from exc
        if len(columns) not in (2, 3):
            raise ValueError(_FORMAT_HELP)
        columns = ["wavelength", "flux", "error"][:len(columns)]

    if len(set(columns)) != len(columns):
        raise ValueError("Observation column headers must be unique.")
    if not rows:
        raise ValueError("The observation table has headers but no data rows.")
    for number, row in enumerate(rows, start=1):
        if len(row) != len(columns):
            raise ValueError(
                f"Data row {number} has {len(row)} columns; expected {len(columns)}. "
                + _FORMAT_HELP
            )

    data = pd.DataFrame(rows, columns=columns)
    for column in ("wavelength", "flux", "error", "continuum"):
        if column not in data.columns:
            continue
        try:
            # Missing uncertainties retain the existing downstream fallback;
            # wavelength/flux still need a finite value for every row.
            values = data[column].str.strip()
            values = values.mask(
                values.isin(["", "nan", "NaN", "NA", "N/A", "null", "None"])
            )
            data[column] = pd.to_numeric(values, errors="raise")
        except ValueError as exc:
            raise ValueError(f"{column.upper()} must contain numeric values.") from exc
        if column in ("wavelength", "flux") and not np.isfinite(data[column]).all():
            raise ValueError(f"{column.upper()} must contain finite numeric values.")
    return data.sort_values("wavelength", kind="stable").reset_index(drop=True)


def _upload_path(filename: str, directory: str | Path) -> Path:
    """Restrict upload mutations to ordinary files and preserve bundled demos."""
    if not filename or Path(filename).name != filename or "\\" in filename:
        raise ValueError("Choose a file name without directory components.")
    if Path(filename).suffix.lower() not in OBSERVATION_EXTENSIONS:
        raise ValueError("Only CSV and TXT observation files are supported.")
    if is_demo_observation(filename):
        raise ValueError("Bundled demo spectra cannot be overwritten or deleted.")
    path = Path(directory) / filename
    if path.is_symlink():
        raise ValueError("Uploaded observation files must not be symbolic links.")
    return path


def save_uploaded_observation(
    filename: str,
    contents: bytes,
    directory: str | Path = "observation_files",
) -> None:
    """Validate before writing, leaving existing files intact on invalid uploads."""
    path = _upload_path(filename, directory)
    read_observation_file(contents)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def delete_uploaded_observation(
    filename: str,
    directory: str | Path = "observation_files",
) -> None:
    """Delete the selected upload even when its contents cannot be parsed."""
    _upload_path(filename, directory).unlink()


def build_observation_uploader(mo, refresh):
    """Save uploads and let the widget's Clear link remove its saved file.

    Track only this widget's current successful upload, independently of the
    observation dropdown. A rejected replacement must never make Clear delete
    an older saved file. Using an event callback also prevents list refreshes
    from repeating file writes or deletions.
    """
    saved_filename = None

    def on_change(files):
        nonlocal saved_filename
        if files:
            # The widget now displays this attempt, even if validation fails.
            saved_filename = None
            uploaded_file = files[0]
            try:
                save_uploaded_observation(uploaded_file.name, uploaded_file.contents)
            except (OSError, ValueError) as exc:
                mo.status.toast(f"Upload failed: {exc}")
                return
            saved_filename = uploaded_file.name
            mo.status.toast(f"Uploaded {saved_filename}")
        elif saved_filename is not None:
            try:
                delete_uploaded_observation(saved_filename)
                mo.status.toast(f"Deleted {saved_filename}")
            except FileNotFoundError:
                # Delete Selected, or another session, may have removed it.
                mo.status.toast(f"{saved_filename} was already removed; refreshing the list.")
            except (OSError, ValueError) as exc:
                mo.status.toast(f"Could not delete {saved_filename}: {exc}")
                return
            saved_filename = None
        else:
            return
        refresh(lambda value: value + 1)

    return mo.ui.file(
        filetypes=OBSERVATION_EXTENSIONS,
        kind="area",
        label="**Drag and drop an observational spectrum (CSV or TXT)**",
        multiple=False,
        on_change=on_change,
    )


def build_observation_upload_ui(mo, uploader):
    """Show the native upload widget with shared guidance and format examples."""
    return mo.vstack([
        uploader,
        build_observation_upload_help(mo),
    ])


def build_observation_upload_help(mo):
    """Build the format accordion with self-contained examples and downloads.

    Display and download the same small example; viewing help never saves an
    upload or reads external practice files.
    """
    tabs = {"Format guide": mo.md(OBSERVATION_UPLOAD_HELP)}
    for title, filename, description, contents in _UPLOAD_EXAMPLES:
        tabs[title] = mo.vstack([
            mo.md(description),
            mo.md("A small synthetic example. Download this file to try an upload."),
            mo.md(f"```text\n{contents}```"),
            mo.download(
                data=contents.encode("utf-8"),
                filename=filename,
                mimetype="text/csv" if filename.endswith(".csv") else "text/plain",
                label="Download example",
            ),
        ])
    return mo.accordion({
        f"{mo.icon('lucide:info')} Observation file format": mo.ui.tabs(tabs),
    })
