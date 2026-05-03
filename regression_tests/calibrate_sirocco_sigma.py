"""One-time offline calibration study for synthetic Sirocco fallback sigma.

This script measures robust fractional continuum residuals in low-structure
windows of representative Sirocco spectra and reports candidate values for the
fixed continuum-relative fallback epsilon used by Speculate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Speculate_addons.Spec_functions import (
    SYNTHETIC_SIROCCO_SIGMA_EPSILON,
    fit_power_law_continuum,
)


@dataclass(frozen=True)
class SpectrumSelection:
    run: int
    inclination: int


def parse_selection_list(text: str, default_inclinations: list[int]) -> list[SpectrumSelection]:
    selections: list[SpectrumSelection] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if "@" in token:
            run_str, inc_str = token.split("@", 1)
            selections.append(SpectrumSelection(run=int(run_str), inclination=int(inc_str)))
        else:
            run = int(token)
            for inclination in default_inclinations:
                selections.append(SpectrumSelection(run=run, inclination=inclination))
    return selections


def parse_windows(text: str | None) -> list[tuple[float, float]] | None:
    if text is None:
        return None
    windows: list[tuple[float, float]] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        lo_str, hi_str = token.split("-", 1)
        windows.append((float(lo_str), float(hi_str)))
    return windows


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad


def load_sirocco_spectrum(spec_path: Path, inclination: int) -> tuple[np.ndarray, np.ndarray]:
    skiprows = 0
    with spec_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Freq."):
                continue
            skiprows = index
            break

    data = np.loadtxt(spec_path, skiprows=skiprows)
    column = int(2 + (inclination - 30) / 5)
    wavelength = np.flip(data[:, 1].astype(np.float64))
    flux = np.flip(data[:, column].astype(np.float64))
    return wavelength, flux


def select_low_structure_windows(
    wavelength: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    window_width: float,
    n_windows: int,
    edge_buffer: float,
) -> list[tuple[float, float]]:
    fractional = (flux - continuum) / np.where(continuum > 0, continuum, 1.0)
    start = float(wavelength.min()) + edge_buffer
    stop = float(wavelength.max()) - edge_buffer
    step = window_width / 2.0

    candidates: list[tuple[float, tuple[float, float]]] = []
    current = start
    while current + window_width <= stop:
        lo = current
        hi = current + window_width
        mask = (wavelength >= lo) & (wavelength <= hi)
        if np.count_nonzero(mask) >= 30:
            score = robust_sigma(fractional[mask])
            if np.isfinite(score):
                candidates.append((score, (lo, hi)))
        current += step

    candidates.sort(key=lambda item: item[0])
    chosen: list[tuple[float, float]] = []
    for _, window in candidates:
        if all(window[1] <= old[0] or window[0] >= old[1] for old in chosen):
            chosen.append(window)
        if len(chosen) >= n_windows:
            break
    return sorted(chosen)


def analyse_windows(
    wavelength: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    windows: list[tuple[float, float]],
) -> list[dict]:
    fractional = (flux - continuum) / np.where(continuum > 0, continuum, 1.0)
    results: list[dict] = []
    for lo, hi in windows:
        mask = (wavelength >= lo) & (wavelength <= hi)
        values = fractional[mask]
        sigma_frac = robust_sigma(values)
        results.append(
            {
                "window": [float(lo), float(hi)],
                "n_points": int(np.count_nonzero(mask)),
                "median_fractional_residual": float(np.nanmedian(values)),
                "sigma_fractional_residual": float(sigma_frac),
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir",
        default="sirocco_grids/speculate_cv_no-bl_testgrid_v87f",
        help="Directory containing Sirocco .spec files.",
    )
    parser.add_argument(
        "--selections",
        default="0",
        help=(
            "Comma-separated run selections. Use '0' to expand to default inclinations, "
            "or explicit '0@30,0@55,0@80'."
        ),
    )
    parser.add_argument(
        "--default-inclinations",
        default="30,55,80",
        help="Inclinations used when a run is given without '@inclination'.",
    )
    parser.add_argument(
        "--windows",
        default=None,
        help="Optional manual wavelength windows, e.g. '2000-2200,3000-3200'.",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=160.0,
        help="Automatic low-structure window width in Angstrom.",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Number of automatic windows to retain when --windows is omitted.",
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=150.0,
        help="Ignore the spectrum edges by this many Angstrom when auto-selecting windows.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile of window sigma estimates used as the suggested epsilon.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_inclinations = [int(token) for token in args.default_inclinations.split(",") if token.strip()]
    selections = parse_selection_list(args.selections, default_inclinations)
    if not selections:
        raise SystemExit("No spectra selected for analysis.")

    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        raise SystemExit(f"Grid directory not found: {grid_dir}")

    anchor = selections[0]
    anchor_path = grid_dir / f"run{anchor.run}.spec"
    anchor_wl, anchor_flux = load_sirocco_spectrum(anchor_path, anchor.inclination)
    anchor_continuum, _ = fit_power_law_continuum(anchor_wl, anchor_flux)

    windows = parse_windows(args.windows)
    if windows is None:
        windows = select_low_structure_windows(
            anchor_wl,
            anchor_flux,
            anchor_continuum,
            window_width=args.window_width,
            n_windows=args.n_windows,
            edge_buffer=args.edge_buffer,
        )

    if not windows:
        raise SystemExit("No valid wavelength windows found for analysis.")

    spectra_reports: list[dict] = []
    sigma_samples: list[float] = []
    for selection in selections:
        spec_path = grid_dir / f"run{selection.run}.spec"
        wavelength, flux = load_sirocco_spectrum(spec_path, selection.inclination)
        continuum, continuum_meta = fit_power_law_continuum(wavelength, flux)
        window_report = analyse_windows(wavelength, flux, continuum, windows)
        sigma_samples.extend(
            item["sigma_fractional_residual"]
            for item in window_report
            if np.isfinite(item["sigma_fractional_residual"])
        )
        spectra_reports.append(
            {
                "run": selection.run,
                "inclination": selection.inclination,
                "spec_path": str(spec_path),
                "continuum": {
                    "poly_deg": int(continuum_meta["poly_deg"]),
                    "n_masked": int(continuum_meta["n_masked"]),
                },
                "windows": window_report,
            }
        )

    sigma_array = np.asarray(sigma_samples, dtype=np.float64)
    sigma_array = sigma_array[np.isfinite(sigma_array)]
    if sigma_array.size == 0:
        raise SystemExit("No finite sigma estimates were produced.")

    suggested_epsilon = float(np.quantile(sigma_array, args.quantile))
    report = {
        "grid_dir": str(grid_dir),
        "windows": [[float(lo), float(hi)] for lo, hi in windows],
        "current_default_epsilon": float(SYNTHETIC_SIROCCO_SIGMA_EPSILON),
        "requested_quantile": float(args.quantile),
        "summary": {
            "n_window_estimates": int(sigma_array.size),
            "median_sigma_fraction": float(np.median(sigma_array)),
            "p90_sigma_fraction": float(np.quantile(sigma_array, 0.90)),
            "p95_sigma_fraction": float(np.quantile(sigma_array, 0.95)),
            "max_sigma_fraction": float(np.max(sigma_array)),
            "suggested_epsilon": suggested_epsilon,
            "default_is_conservative": bool(
                SYNTHETIC_SIROCCO_SIGMA_EPSILON >= suggested_epsilon
            ),
        },
        "spectra": spectra_reports,
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())