"""Prior metadata and selection helpers for bundled observations."""

from pathlib import Path
from typing import Iterable

from Speculate_addons.grid_registry import grid_type

# The CV entries reproduce the published Table A1 values in speculate release paper.  The continuum-
# normalised AGN composites use the 100 pc emulator reference distance and
# style-specific inclination ranges supplied with the bundled demonstrations.
# Keys are shared by Benchmark, Inference, and Quick Fit.
OBSERVATION_PRIORS = {
    "cv:ixvel_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 90.36, "sigma": 0.15},
        "inclination_deg": {"kind": "normal", "mean": 57.0, "sigma": 2.0},
    },
    "cv:rwsex_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 223.12, "sigma": 1.23},
        "inclination_deg": {"kind": "uniform", "min": 28.0, "max": 40.0},
    },
    "cv:rwtri_dered.csv": {
        "distance_pc": {"kind": "normal", "mean": 306.08, "sigma": 2.09},
        "inclination_deg": {"kind": "normal", "mean": 72.5, "sigma": 2.5},
    },
    "cv:uxuma_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 291.95, "sigma": 1.34},
        "inclination_deg": {"kind": "normal", "mean": 71.0, "sigma": 0.6},
    },
    "cv:v3885sgr_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 128.97, "sigma": 0.57},
        "inclination_deg": {"kind": "uniform", "min": 45.0, "max": 70.0},
    },
    "agn:felobal_qso_composite.csv": {
        "distance_pc": {"kind": "normal", "mean": 100.0, "sigma": 0.01},
        "inclination_deg": {"kind": "uniform", "min": 70.0, "max": 85.0, "start": 77.0},
    },
    "agn:lobal_qso_composite.csv": {
        "distance_pc": {"kind": "normal", "mean": 100.0, "sigma": 0.01},
        "inclination_deg": {"kind": "uniform", "min": 70.0, "max": 85.0, "start": 77.0},
    },
    "agn:nonbal_qso_composite.csv": {
        "distance_pc": {"kind": "normal", "mean": 100.0, "sigma": 0.01},
        "inclination_deg": {"kind": "uniform", "min": 10.0, "max": 25.0, "start": 17.0},
    },
    "agn:hibal_qso_composite.csv": {
        "distance_pc": {"kind": "normal", "mean": 100.0, "sigma": 0.01},
        "inclination_deg": {"kind": "uniform", "min": 10.0, "max": 85.0, "start": 55.0},
    },
    "agn:hilobal_qso_composite.csv": {
        "distance_pc": {"kind": "normal", "mean": 100.0, "sigma": 0.01},
        "inclination_deg": {"kind": "uniform", "min": 10.0, "max": 85.0, "start": 55.0},
    },
}


def filter_observation_files_for_grid(
    files: Iterable[str],
    grid_name: str | None,
) -> list[str]:
    """Hide bundled demo spectra tagged for the opposite object family.

    ``AGN:`` and ``CV:`` prefixes identify the demonstrations shipped with
    Speculate.  Untagged files are user observations and therefore remain
    selectable for every grid family.
    """
    family = grid_type(grid_name)
    allowed_prefix = "agn:" if family == "agn" else "cv:" if family and family.startswith("cv") else None
    if allowed_prefix is None:
        return list(files)

    filtered = []
    for file in files:
        filename = Path(file).name.lower()
        tagged_demo = filename.startswith(("agn:", "cv:"))
        if not tagged_demo or filename.startswith(allowed_prefix):
            filtered.append(file)
    return filtered
