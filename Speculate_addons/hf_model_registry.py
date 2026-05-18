"""Discovery and download helpers for pre-trained Speculate emulator models.

The Model Downloader and Quick Fit notebook both use this module to keep
HuggingFace model-repository access in one place. Model files are classified by
the filename markers used by Speculate exports, then copied from the Hub cache
into ``Grid-Emulator_Files`` so the existing local loaders can read them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

HF_MODEL_REPO_IDS = (
    "Sirocco-rt/speculate_cv_emulators",
    "Sirocco-rt/speculate_agn_emulators",
)
HF_MODEL_REPO_TYPE = "model"
LOCAL_MODEL_DIR = "Grid-Emulator_Files"

GP_MODEL_MARKERS = ("_emu_",)
QUICKFIT_MODEL_MARKERS = ("_qfnn-ensemble_", "_qfnn_", "_qfgi_")
SUPPORTED_SUFFIX = ".npz"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def is_gp_model(filename: str) -> bool:
    """Return True when a filename follows the saved GP emulator pattern."""
    name = Path(filename).name
    return name.endswith(SUPPORTED_SUFFIX) and any(marker in name for marker in GP_MODEL_MARKERS)


def is_quickfit_model(filename: str) -> bool:
    """Return True when a filename follows a Quick Fit export pattern."""
    name = Path(filename).name
    return name.endswith(SUPPORTED_SUFFIX) and any(marker in name for marker in QUICKFIT_MODEL_MARKERS)


def is_supported_model(filename: str) -> bool:
    """Return True for any model file that Speculate knows how to stage."""
    return is_gp_model(filename) or is_quickfit_model(filename)


def model_type_label(filename: str) -> str:
    """Return a short display label for a supported emulator filename."""
    name = Path(filename).name
    if "_qfnn-ensemble_" in name:
        return "NN-Ensemble"
    if "_qfnn_" in name:
        return "NN"
    if "_qfgi_" in name:
        return "GridInterp"
    if "_emu_" in name:
        return "GP Emulator"
    return "Model"


def model_group(filename: str) -> str:
    """Return the broad model family used for filtering download choices."""
    if is_quickfit_model(filename):
        return "quickfit"
    if is_gp_model(filename):
        return "gp"
    return "unknown"


def list_hf_model_files() -> list[str]:
    """List supported Speculate model files in configured HF model repos."""
    from huggingface_hub import list_repo_files

    model_files: set[str] = set()
    errors: list[str] = []
    listed_any_repo = False
    for repo_id in HF_MODEL_REPO_IDS:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type=HF_MODEL_REPO_TYPE)
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
            continue
        listed_any_repo = True
        model_files.update(filename for filename in files if is_supported_model(filename))

    if not listed_any_repo and errors:
        raise RuntimeError("; ".join(errors))
    return sorted(model_files)


def list_local_model_files(destination_dir: str | Path = LOCAL_MODEL_DIR) -> list[str]:
    """List supported model files already staged in the local model folder."""
    model_dir = Path(destination_dir)
    if not model_dir.exists():
        return []
    return sorted(path.name for path in model_dir.iterdir() if path.is_file() and is_supported_model(path.name))


def list_local_quickfit_model_files(destination_dir: str | Path = LOCAL_MODEL_DIR) -> list[str]:
    """List locally staged Quick Fit models, excluding GP emulator files."""
    return [filename for filename in list_local_model_files(destination_dir) if is_quickfit_model(filename)]


def local_model_path(filename: str, destination_dir: str | Path = LOCAL_MODEL_DIR) -> Path:
    """Return the local staging path for a Hub filename."""
    return Path(destination_dir) / Path(filename).name


def download_model_to_local(
    filename: str,
    destination_dir: str | Path = LOCAL_MODEL_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download one model from HuggingFace and stage it for local loading.

    Files already present in ``destination_dir`` are skipped unless
    ``overwrite`` is true. The returned status is used directly by marimo
    progress/status cells.
    """
    from huggingface_hub import hf_hub_download

    destination = local_model_path(filename, destination_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return {
            "filename": filename,
            "path": str(destination),
            "status": "skipped",
            "error": None,
        }

    errors: list[str] = []
    cached_path = None
    for repo_id in HF_MODEL_REPO_IDS:
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=HF_MODEL_REPO_TYPE,
            )
            break
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
    if cached_path is None:
        raise RuntimeError("; ".join(errors))

    shutil.copy2(cached_path, destination)
    return {
        "filename": filename,
        "path": str(destination),
        "status": "downloaded",
        "error": None,
    }


def ensure_quickfit_models_cached(
    destination_dir: str | Path = LOCAL_MODEL_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Ensure configured Quick Fit models are available in the local cache.

    HF Space mode calls this before building Quick Fit model selectors, so the
    existing file-based ``np.load`` path can stay unchanged for local and hosted
    inference.
    """
    result: dict[str, Any] = {
        "downloaded": [],
        "skipped": [],
        "failed": [],
        "available": [],
        "error": None,
    }

    try:
        quickfit_files = [filename for filename in list_hf_model_files() if is_quickfit_model(filename)]
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["available"] = quickfit_files
    for filename in quickfit_files:
        try:
            item = download_model_to_local(filename, destination_dir=destination_dir, overwrite=overwrite)
            result[item["status"]].append(item["filename"])
        except Exception as exc:
            result["failed"].append((filename, str(exc)))

    return result
