"""
Speculate Benchmark Module
===========================
Core functions for evaluating emulator performance across three tiers:

    Tier 1 — Grid Reconstruction Fidelity (PCA + Leave-One-Out CV)
    Tier 2 — Test Grid Parameter Recovery  (MLE + MCMC + calibration)
    Tier 3 — Observational Spectra         (goodness-of-fit + PPC)

Import these functions directly or use the marimo Benchmark Viewer
(``speculate_benchmark_viewer.py``) for interactive analysis.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from _version import __version__

import numpy as np
import pandas as pd
import cma
from scipy import stats
from scipy.optimize import minimize as scipy_minimize

from Speculate_addons.grid_registry import (
    benchmark_param_map,
    defaulted_physical_param_ids,
    emulator_values_to_physical,
    inclination_column,
    infer_grid_name,
    lookup_row_to_emulator_values,
)

log = logging.getLogger(__name__)

_SIROCCO_CYCLE_RE = re.compile(
    r"\b(?P<action>Starting|Finished)\s+"
    r"(?P<current>\d+)\s+of\s+(?P<total>\d+)\s+"
    r"(?P<kind>ionization|spectrum)\s+cycles\b",
    re.IGNORECASE,
)
_SIROCCO_ION_CONVERGED_RE = re.compile(
    r"\bIonization converged early at cycle\s+"
    r"(?P<current>\d+)\s+of\s+(?P<total>\d+)\b",
    re.IGNORECASE,
)


# ======================================================================
# Parameter Mapping
# ======================================================================
# The registry supplies the parameter-index maps used by Tier 2 summaries and
# Tier 3 exports.  All benchmark paths pass grid_name through to the
# registry-aware helpers below.  For CV grids, param2 is the wind/disk mass-loss
# ratio in emulator space and params 1, 3, and 5 are log10 axes; see
# grid_registry.py for the AGN Eddington-fraction and R_g-scaled equivalents.


def internal_to_friendly(param_names: Sequence[str], grid_name: Optional[str] = None) -> List[str]:
    """Map internal ``paramN`` names to grid-specific human-readable names."""
    param_map = benchmark_param_map(grid_name)
    out = []
    for pn in param_names:
        idx = int(pn.replace("param", ""))
        friendly, _ = param_map.get(idx, (pn, pn))
        out.append(friendly)
    return out


TIER2_NUISANCE_LABELS = (
    "Av",
    "log_scale",
    "cheb:1",
    "global_cov:log_amp",
    "global_cov:log_ls",
)

TIER2_FRIENDLY_LABELS = {
    "Av": "Av",
    "log_scale": "log_scale",
    "cheb:1": "cheb_1",
    "global_cov:log_amp": "GP log_amp",
    "global_cov:log_ls": "GP log_ls",
}


def build_tier2_label_map(param_names: Sequence[str], grid_name: Optional[str] = None) -> Dict[str, str]:
    """Return friendly labels for Tier 2 grid and nuisance parameters."""
    label_map = {
        internal: friendly
        for internal, friendly in zip(param_names, internal_to_friendly(param_names, grid_name))
    }
    label_map.update(TIER2_FRIENDLY_LABELS)
    return label_map


def build_tier2_freeze_defaults(param_names: Sequence[str], grid_name: Optional[str] = None) -> dict:
    """Return the default Tier 2 Stage 2/4 freeze dictionaries.

    Grid parameter labels depend on the active registry entry, while nuisance
    labels stay shared across grids.
    """
    label_map = build_tier2_label_map(param_names, grid_name)
    mle = {label: False for label in label_map}
    for mle_label in ("Av", "cheb:1"):
        if mle_label in mle:
            mle[mle_label] = True

    mcmc = {label: False for label in label_map}
    for label in ("Av", "log_scale", "cheb:1", "global_cov:log_amp", "global_cov:log_ls"):
        if label in mcmc:
            mcmc[label] = True

    return {
        "labels": label_map,
        "mle": mle,
        "mcmc": mcmc,
    }


def _serialise_freeze_settings(freeze_params: Optional[dict]) -> Dict[str, bool]:
    """Coerce a freeze settings mapping to JSON-safe bools."""
    return {
        str(label): bool(is_frozen)
        for label, is_frozen in (freeze_params or {}).items()
    }


def _snapshot_model_params(model) -> Dict[str, float]:
    """Capture the current SpectrumModel parameter state as plain floats."""
    return {str(label): float(model.params[label]) for label in model.params.keys()}


def _apply_freeze_settings(model, freeze_params: Optional[dict], thaw_first: bool = False) -> List[str]:
    """Apply a freeze dictionary to the current SpectrumModel."""
    applied = []
    for label, is_frozen in (freeze_params or {}).items():
        if label not in model.params:
            continue
        if thaw_first:
            try:
                model.thaw(label)
            except Exception:
                pass
        if is_frozen:
            model.freeze(label)
            applied.append(label)
    return applied


def emulator_to_physical(param_names: Sequence[str], values: np.ndarray, grid_name: Optional[str] = None) -> dict:
    """
    Convert emulator-space parameter values to physical Sirocco units.

    Returns a dict mapping Sirocco parameter keywords to physical values.  The
    registry fills omitted physical axes from grid defaults before converting.
    """
    return emulator_values_to_physical(grid_name, param_names, values)


# ======================================================================
# Tier 1 — Grid Reconstruction
# ======================================================================


def run_tier1(emu, grid_path: Optional[str] = None) -> dict:
    """
    Tier 1 benchmark: grid reconstruction fidelity.

    Parameters
    ----------
    emu : Emulator
        A trained emulator instance.
    grid_path : str or None
        Path to the grid NPZ file. If provided, flux-space metrics are
        computed in addition to weight-space Leave-One-Out metrics.

    Returns
    -------
    dict
        Benchmark results.  JSON-safe scalars/lists plus an ``'_arrays'``
        sub-dict containing large numpy arrays (``original_flux``,
        ``pca_recon_flux``, ``loo_recon_flux``, ``wavelength``) that are
        kept in-memory only and excluded from JSON serialisation.
    """
    t0 = time.time()
    results = emu.loo_cv(grid=grid_path)

    # True PCA explained variance: fraction of flux variance captured by the
    # PCA components.  This is a fixed property of the grid + n_components
    # and does NOT depend on the GP training.
    results["pca_explained_variance"] = (
        float(emu.pca_explained_variance)
        if emu.pca_explained_variance is not None
        else None
    )

    results["n_components"] = emu.ncomps
    results["n_grid_points"] = emu.grid_points.shape[0]
    results["n_params"] = emu.grid_points.shape[1]
    results["tier1_time_s"] = time.time() - t0

    # Aggregate Q² across all components (LOO R²).
    # Uses the same safeguard as per-component Q² in loo_cv().
    _q2 = results.get("q2_per_comp")
    if _q2 is not None:
        _total_mse = np.sum(results["loo_mse_per_comp"])
        _total_var = np.sum(np.var(emu.weights, axis=0))
        results["q2_aggregate"] = float(
            1.0 - _total_mse / max(_total_var, 1e-30)
        )

    # LOO NLPD: proper scoring rule (Bastos & O’Hagan 2009).
    _nlpd = results.get("nlpd_per_comp")
    if _nlpd is not None:
        results["nlpd_mean"] = float(np.mean(_nlpd))

    # Emulator Accuracy Score (EAS): single 0–100% summary metric.
    # Combines true PCA explained variance (how well n_components capture the
    # grid variance — fixed for a given setup) with Leave-One-Out flux
    # reconstruction accuracy (how well the GP interpolates unseen training
    # points in flux space — depends on training).  100% = perfect.
    _pca_ev = results["pca_explained_variance"]
    _loo_rmse_med = results.get("loo_flux_rmse_median")
    if _loo_rmse_med is not None:
        results["emulator_accuracy_score"] = float(
            100.0
            * max(0.0, min(1.0, _pca_ev))
            * max(0.0, 1.0 - _loo_rmse_med)
        )
    else:
        results["emulator_accuracy_score"] = None

    # Separate large flux arrays (kept in-memory, never serialised)
    _ARRAY_KEYS = {
        "original_flux", "pca_recon_flux", "loo_recon_flux", "wavelength",
        "display_original_flux", "display_pca_recon_flux", "display_loo_recon_flux",
        "pca_per_wl_rmse", "loo_per_wl_rmse",
        "loo_flux_rmse", "pca_recon_rmse",
        "loo_recon_var", "display_loo_recon_var", "loo_mu", "loo_var",
    }
    arrays = {}
    for k in _ARRAY_KEYS:
        if k in results:
            arrays[k] = results.pop(k)
    # Also stash grid_points + param_names for the interactive viewer
    arrays["grid_points"] = emu.grid_points
    arrays["param_names"] = list(emu.param_names)
    if getattr(emu, "flux_scale", "linear") == "log":
        arrays["flux_axis_title"] = "log10 Flux"
    elif getattr(emu, "flux_scale", "linear") == "continuum-normalised":
        arrays["flux_axis_title"] = "Continuum-normalised Flux"
    else:
        arrays["flux_axis_title"] = "Flux"

    # Convert remaining numpy arrays to lists for JSON serialisation
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        else:
            serialisable[k] = v

    # Attach the arrays sub-dict (viewer will pop this before saving JSON)
    serialisable["_arrays"] = arrays
    return serialisable


# ======================================================================
# Tier 2 — Test Grid Parameter Recovery
# ======================================================================


def _load_test_grid_spectrum(
    spec_file: str, inclination: float, wl_range: Tuple[float, float], grid_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Sirocco ``.spec`` file and extract wavelength, flux, and error arrays.

    Sirocco .spec files have a variable-length text header (comments starting
    with '#' and a column-label row beginning with 'Freq.'), followed by numeric
    data columns.  Column 0 is frequency (Hz), column 1 is wavelength (Å), and
    columns 2+ hold fluxes at the grid's registered viewing inclinations.  CV
    grids use 30-85 degrees in 5-degree steps; AGN uses 10, 25, 40, 55, 70, and
    85 degrees.  Wavelengths in the file are descending; this function flips
    them to ascending order.
    """
    # Determine header lines (skip comments, blank lines, and the 'Freq.' label row)
    skiprows = 0
    with open(spec_file, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Freq."):
                skiprows = i + 1
            else:
                break

    data = np.loadtxt(spec_file, skiprows=skiprows)
    # Column 0 = Freq (Hz), column 1 = Lambda (Å) — always use Lambda.
    # Flip to ascending wavelength order for consistency with the emulator.
    wl = np.flip(data[:, 1])
    # Map the requested inclination angle to the correct data column.
    col = inclination_column(grid_name, inclination)
    flux = np.flip(data[:, col])

    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    wl = wl[mask]
    flux = flux[mask]

    if len(wl) < 2:
        raise ValueError(
            f"Only {len(wl)} wavelength point(s) remain after filtering to "
            f"wl_range={wl_range} in {spec_file}. "
            "Check that wl_range matches the spectrum's wavelength array (Å) "
            "and that the correct inclination column is being read."
        )

    from Speculate_addons.Spec_functions import build_synthetic_sirocco_sigma

    sigma, _ = build_synthetic_sirocco_sigma(wl, flux)
    return wl, flux, sigma


def _extract_ground_truth(
    parquet_path: str, run_idx: int, emu_param_names: Sequence[str], grid_name: Optional[str] = None,
    inclination: Optional[float] = None,
) -> dict:
    """Extract ground-truth parameters from the test grid's lookup table.

    The lookup table (a parquet file co-located with the .spec files) records the
    Sirocco simulation inputs for every run.  This function reads the raw physical
    values, then delegates to the registry so CV and AGN grids apply the same
    forward transforms used during emulator training.  The returned dict lives
    in emulator space and is directly comparable to inference outputs.
    """
    df = pd.read_parquet(parquet_path)

    # Try common column names for run number
    run_col = None
    for candidate in ["Run Number", "run_number", "Run", "run"]:
        if candidate in df.columns:
            run_col = candidate
            break
    if run_col is None:
        # Assume row index matches run index (sorted)
        row = df.iloc[run_idx]
    else:
        row = df[df[run_col] == run_idx].iloc[0]

    all_values = lookup_row_to_emulator_values(grid_name, row, inclination)
    friendly_names = internal_to_friendly(emu_param_names, grid_name)
    return {name: float(all_values[name]) for name in friendly_names if name in all_values}


def _get_loc_scale(dist):
    """Extract loc and scale from a frozen scipy distribution."""
    args = dist.args
    kwds = dist.kwds
    if len(args) >= 2:
        return args[0], args[1]
    elif len(args) == 1:
        return args[0], kwds.get("scale", 1.0)
    else:
        return kwds.get("loc", 0.0), kwds.get("scale", 1.0)


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


def _emulator_wavelength_bounds(emu) -> Tuple[float, float]:
    """Return finite wavelength bounds for an emulator in Angstrom."""
    wl = np.asarray(getattr(emu, "wl", []), dtype=np.float64)
    wl = wl[np.isfinite(wl)]
    if wl.size < 2:
        raise ValueError("Emulator does not expose a usable wavelength grid.")
    return float(np.min(wl)), float(np.max(wl))


def _assert_wavelengths_within_emulator(
    emu,
    wl: np.ndarray,
    context: str = "spectrum",
) -> Tuple[float, float]:
    """Raise before fitting if data would require emulator extrapolation."""
    emu_min, emu_max = _emulator_wavelength_bounds(emu)
    wl = np.asarray(wl, dtype=np.float64)
    finite_wl = wl[np.isfinite(wl)]
    if finite_wl.size == 0:
        raise ValueError(f"No finite wavelengths were supplied for {context}.")
    data_min = float(np.min(finite_wl))
    data_max = float(np.max(finite_wl))
    tol = max(1e-6, 1e-9 * max(abs(emu_min), abs(emu_max)))
    if data_min < emu_min - tol or data_max > emu_max + tol:
        raise ValueError(
            f"{context} wavelength range ({data_min:.3f}, {data_max:.3f}) Å extends outside "
            f"the emulator coverage ({emu_min:.3f}, {emu_max:.3f}) Å. "
            "Use a matching emulator or restrict wl_range before fitting."
        )
    return emu_min, emu_max


def _resolve_tier3_wl_range(
    emu,
    wl_range: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Resolve Tier 3's fitting window without leaving emulator support."""
    emu_min, emu_max = _emulator_wavelength_bounds(emu)
    if wl_range is None:
        return emu_min, emu_max

    wl_min, wl_max = float(wl_range[0]), float(wl_range[1])
    if wl_min >= wl_max:
        raise ValueError(f"wl_range must be increasing, got {wl_range}.")
    tol = max(1e-6, 1e-9 * max(abs(emu_min), abs(emu_max)))
    if wl_min < emu_min - tol or wl_max > emu_max + tol:
        raise ValueError(
            f"Requested wl_range=({wl_min:.3f}, {wl_max:.3f}) Å extends outside "
            f"the emulator coverage ({emu_min:.3f}, {emu_max:.3f}) Å. "
            "Use a matching emulator or restrict the Tier 3 wavelength range."
        )
    return max(wl_min, emu_min), min(wl_max, emu_max)


def run_mle_single(
    emu,
    wl: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    priors: Optional[dict] = None,
    flux_scale: str = "linear",
    max_iter: int = 10_000,
    freeze_av: bool = True,
    freeze_params: Optional[Dict[str, bool]] = None,
    iteration_callback=None,
    n_restarts: int = 1,
) -> dict:
    """
    Run MLE inference on a single spectrum.

    Parameters
    ----------
    emu : Emulator
    wl, flux, sigma : ndarray
        Observation wavelength, flux, and uncertainty.
    priors : dict or None
        Priors dict (internal param names → scipy frozen distributions).
    flux_scale : str
        'linear', 'log', or 'continuum-normalised'.
    max_iter : int
        Maximum CMA-ES function evaluations per restart.
    freeze_av : bool
        If True (default), fix Av=0.  Use True for synthetic test-grid
        spectra (Tier 2) which have no dust; False for observational
        spectra (Tier 3) where Av should be optimised.
    freeze_params : dict or None
        Optional Stage 2 freeze settings keyed by internal parameter name.
        Checked parameters are held at the benchmark's current initial value:
        grid midpoints, ``Av=0``, bootstrapped ``log_scale``, ``cheb:1=0``,
        and the benchmark's default GP hyperparameter initialisation.
    iteration_callback : callable or None
        If provided, called as ``iteration_callback(iter_num, max_iter, best_nll, elapsed_s)``
        every 50 iterations during optimisation.
    n_restarts : int
        Number of CMA-ES restarts.  The first restart starts from the
        bootstrapped midpoint; subsequent restarts use random starting
        points drawn uniformly from the prior bounds (with log_scale
        re-bootstrapped at each).  The best result across all restarts
        is returned.

    Returns
    -------
    dict with keys: 'grid_params', 'all_params', 'nll', 'success',
                    'n_iter', 'labels'.
    """
    from Starfish.spectrum import Spectrum
    from Starfish.models import SpectrumModel

    # Match the observation onto the emulator's training scale before building a
    # SpectrumModel, including consistent propagation of the uncertainties.
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sigma is defined upstream in native linear-flux units; only keep a tiny
    # absolute floor here so transform propagation does not reintroduce a
    # flux-proportional collapse in deep features.
    sigma = np.maximum(sigma, 1e-30)

    if flux_scale == "log":
        sigma = sigma / (np.abs(flux) * np.log(10) + 1e-30)
        flux = np.where(flux > 0, np.log10(flux), np.log10(np.abs(flux) + 1e-30))
    elif flux_scale == "continuum-normalised":
        from Speculate_addons.Spec_functions import fit_power_law_continuum
        continuum, _ = fit_power_law_continuum(wl, flux)
        cont_safe = np.where(continuum > 0, continuum, 1.0)
        sigma = sigma / cont_safe
        flux = flux / cont_safe

    _assert_wavelengths_within_emulator(emu, wl, context="MLE input spectrum")
    sigma = np.maximum(sigma, 1e-30)
    spec = Spectrum(wl, flux, sigmas=sigma)

    # Default grid params: midpoints
    grid_init = [
        float(np.mean([emu.min_params[i], emu.max_params[i]]))
        for i in range(len(emu.param_names))
    ]

    model = SpectrumModel(
        emulator=emu,
        data=spec,
        grid_params=grid_init,
        global_cov={"log_amp": -55.0, "log_ls": 4.5},
        cheb=[0.0],
        flux_scale=flux_scale,
    )

    _requested_freeze = _serialise_freeze_settings(freeze_params)
    if "Av" not in _requested_freeze:
        _requested_freeze["Av"] = bool(freeze_av)

    # Optionally fix Av = 0 (synthetic test-grid spectra have no dust
    # extinction, so allowing Av to float wastes degrees of freedom and
    # can push the optimizer into poor local minima).
    model.params["Av"] = 0.0
    if _requested_freeze.get("Av", False):
        model.freeze("Av")

    # Bootstrap log_scale from auto-calculation *before* building priors
    # so we can centre the log_scale prior on the data-informed value.
    _bootstrapped_ls = -25.0  # sensible fallback
    try:
        _ = model()
        if model._log_scale is not None and np.isfinite(model._log_scale):
            _bootstrapped_ls = float(model._log_scale)
            model.params["log_scale"] = _bootstrapped_ls
    except Exception:
        model.params["log_scale"] = _bootstrapped_ls

    # Build default priors — tightly bounded to match the inference tool's
    # optimised settings. Tight priors matter twice here: they regularize the
    # likelihood itself, and they define the simplex footprint used to seed
    # Nelder-Mead below.
    if priors is None:
        priors = {}
        for i, pn in enumerate(emu.param_names):
            lo, hi = float(emu.min_params[i]), float(emu.max_params[i])
            priors[pn] = stats.uniform(loc=lo, scale=hi - lo)
        if not _requested_freeze.get("Av", False):
            priors["Av"] = stats.uniform(loc=0, scale=2.0)
        # Keep log_scale centered on the bootstrap estimate so the optimizer does
        # not waste its early iterations finding the gross normalization.
        priors["log_scale"] = stats.uniform(
            loc=_bootstrapped_ls - 5.0, scale=10.0
        )
        # Chebyshev c1 continuum tilt — small correction for gradient mismatch
        priors["cheb:1"] = stats.uniform(loc=-0.5, scale=1.0)
        # GP log_amp: auto-detect centre from residual variance between the
        # bootstrapped model and the data, so it adapts to any flux scale.
        _la_centre = -55.0  # fallback for linear-scale raw flux
        try:
            model.params["log_scale"] = _bootstrapped_ls
            _model_result = model()
            _model_flux = _model_result[0] if isinstance(_model_result, tuple) else _model_result
            if hasattr(_model_flux, 'detach'):
                _model_flux = _model_flux.detach().cpu().numpy()
            _data_flux = np.array(spec.fluxes)
            _n = min(len(_data_flux), len(_model_flux))
            _resid = _data_flux[:_n] - _model_flux[:_n]
            _resid_var = float(np.mean(_resid ** 2))
            if _resid_var > 0:
                _la_centre = float(np.log(max(_resid_var, 1e-30)))
        except Exception:
            pass
        _la_centre = round(_la_centre, 1)
        priors["global_cov:log_amp"] = stats.norm(loc=_la_centre, scale=2.5)
        # GP log_ls: [1, 8] — matches inference tool optimised range
        priors["global_cov:log_ls"] = stats.uniform(loc=1.0, scale=7.0)

    _applied_freezes = _apply_freeze_settings(model, _requested_freeze)

    labels_before = list(model.labels)

    # Derive per-parameter bounds from the priors for CMA-ES box constraints.
    active_labels = list(model.labels)
    N = len(active_labels)

    if N == 0:
        try:
            _nll = float(-model.log_likelihood(priors))
        except Exception:
            _nll = 1e10
        return {
            "grid_params": model.grid_params.tolist(),
            "all_params": _snapshot_model_params(model),
            "nll": _nll,
            "success": True,
            "n_iter": 0,
            "labels": labels_before,
            "freeze_params": dict(_requested_freeze),
            "frozen_params": list(_applied_freezes),
            "model": model,
            "priors": priors,
        }

    lo_bounds = []
    hi_bounds = []
    for label in active_labels:
        if label in priors:
            dist = priors[label]
            loc, sc = _get_loc_scale(dist)
            if dist.dist.name == "uniform":
                lo_bounds.append(loc)
                hi_bounds.append(loc + sc)
            elif dist.dist.name == "norm":
                lo_bounds.append(loc - 4 * sc)
                hi_bounds.append(loc + 4 * sc)
            else:
                cv = model.get_param_vector()[active_labels.index(label)]
                lo_bounds.append(cv - abs(cv) * 0.5)
                hi_bounds.append(cv + abs(cv) * 0.5)
        else:
            cv = model.get_param_vector()[active_labels.index(label)]
            lo_bounds.append(cv - abs(cv) * 0.5 - 1e-6)
            hi_bounds.append(cv + abs(cv) * 0.5 + 1e-6)

    # Bootstrap log_scale at the starting point so the optimizer begins
    # with a data-informed normalisation.
    if "log_scale" in active_labels:
        try:
            _ = model()  # triggers auto log_scale calc
            if model._log_scale is not None and np.isfinite(model._log_scale):
                model.params["log_scale"] = model._log_scale
        except Exception:
            pass

    _nll_history = []
    _iter_count = [0]
    _mle_t0 = time.time()
    _global_best_f = [float("inf")]  # mutable for callback visibility
    _cur_restart = [0]  # 1-based index of current restart

    def nll(P):
        model.set_param_vector(P)
        try:
            val = -model.log_likelihood(priors)
        except Exception:
            val = 1e10
        _nll_history.append(val)
        _iter_count[0] += 1
        if iteration_callback is not None and _iter_count[0] % 50 == 0:
            _best = min(_global_best_f[0], min(_nll_history) if _nll_history else val)
            iteration_callback(
                _iter_count[0], max_iter, _best, time.time() - _mle_t0,
                _cur_restart[0], n_restarts,
            )
        return val

    # CMA-ES: start from the bootstrapped model state (includes the
    # data-informed log_scale) rather than the raw bounds midpoint.
    p0_cma = np.clip(
        model.get_param_vector(),
        np.array(lo_bounds) + 1e-8,
        np.array(hi_bounds) - 1e-8,
    )

    # Generate starting points: first = bootstrapped x0, rest = random
    lo_arr = np.array(lo_bounds)
    hi_arr = np.array(hi_bounds)
    start_points = [p0_cma.copy()]
    if n_restarts > 1:
        for _ri in range(n_restarts - 1):
            rnd = lo_arr + np.random.rand(N) * (hi_arr - lo_arr)
            # Bootstrap log_scale for random starts too
            if "log_scale" in active_labels:
                ls_idx = active_labels.index("log_scale")
                try:
                    model.set_param_vector(rnd)
                    model()
                    if model._log_scale is not None and np.isfinite(model._log_scale):
                        rnd[ls_idx] = model._log_scale
                except Exception:
                    pass
            start_points.append(rnd)

    global_best_x = p0_cma.copy()
    global_best_nit = 0

    for restart_idx, x0 in enumerate(start_points):
        _cur_restart[0] = restart_idx + 1
        _iter_count[0] = 0  # reset eval counter per restart
        # Per-coordinate initial σ = 20% of prior range.
        cma_stds = [0.2 * (hi - lo) for lo, hi in zip(lo_bounds, hi_bounds)]
        # Double the default popsize for better landscape sampling.
        popsize = 2 * (4 + int(3 * np.log(N)))
        x0_clipped = np.clip(x0, lo_arr + 1e-8, hi_arr - 1e-8)
        es = cma.CMAEvolutionStrategy(
            x0_clipped.tolist(), 1.0,
            {
                "bounds": [lo_bounds, hi_bounds],
                "CMA_stds": cma_stds,
                "popsize": popsize,
                "maxfevals": max_iter,
                "verbose": -9,
                "tolfun": 1e-10,
            },
        )
        run_best_x, run_best_f = x0_clipped.copy(), float("inf")
        while not es.stop():
            solutions = es.ask()
            fits = [nll(np.array(s)) for s in solutions]
            es.tell(solutions, fits)
            gen_best = min(fits)
            if gen_best < run_best_f:
                run_best_f = gen_best
                run_best_x = np.array(solutions[fits.index(gen_best)])
            if run_best_f < _global_best_f[0]:
                _global_best_f[0] = run_best_f

        # Keep global best across restarts
        if run_best_f <= _global_best_f[0]:
            _global_best_f[0] = run_best_f
            global_best_x = run_best_x.copy()
            global_best_nit = es.result.iterations

    from types import SimpleNamespace
    soln = SimpleNamespace(
        x=global_best_x, fun=_global_best_f[0], success=True,
        message=f"Best of {n_restarts} CMA-ES restart(s)",
        nit=global_best_nit,
    )

    if soln.success:
        model.set_param_vector(soln.x)

    return {
        "grid_params": model.grid_params.tolist(),
        "all_params": _snapshot_model_params(model),
        "nll": float(soln.fun),
        "success": bool(soln.success),
        "n_iter": int(soln.nit),
        "labels": labels_before,
        "freeze_params": dict(_requested_freeze),
        "frozen_params": list(_applied_freezes),
        "model": model,
        "priors": priors,
    }


def run_mcmc_single(
    model,
    priors: dict,
    nwalkers: int = 64,
    nsteps: int = 2500,
    burnin: int = 500,
    iteration_callback=None,
    freeze_nuisance: bool = False,
    freeze_params: Optional[Dict[str, bool]] = None,
    grid_name: Optional[str] = None,
) -> dict:
    """
    Run MCMC on a SpectrumModel already set to MLE best-fit.

    Parameters
    ----------
    iteration_callback : callable or None
        If provided, called as ``iteration_callback(step, nsteps, elapsed_s)``
        every 50 steps.
    freeze_nuisance : bool
        If True, freeze the nuisance parameters (log_scale, cheb:1,
        global_cov:log_amp, global_cov:log_ls) at their MLE-optimised
        values so the MCMC only samples over the physical grid parameters.
        This follows the Starfish paper recommendation (Czekala+2015,
        Section 2.5) and significantly speeds up convergence.
    freeze_params : dict or None
        Optional Stage 4 freeze settings keyed by internal parameter name.
        When provided, this overrides ``freeze_nuisance`` and is applied with
        a thaw-then-refreeze pass so each run starts from a clean MCMC state.

    Returns
    -------
    dict with keys:
        'samples'     : (N, ndim) burnt+thinned flat samples
        'summary'     : dict of {label: {mean, std, median, hdi_3, hdi_97}}
        'r_hat'       : dict of {label: r_hat}
        'ess_bulk'    : dict of {label: ess}
        'converged'   : bool — all r_hat < 1.1 and ess > 100
        'n_effective'  : int
    """
    import emcee

    # Optionally freeze parameters at the MLE state before sampling.  The
    # Starfish paper directly motivates fixing the kernel hyperparameters after
    # optimisation; broader Stage 4 freeze choices are treated as an explicit
    # benchmark policy rather than a paper-mandated rule.
    _requested_freeze = _serialise_freeze_settings(freeze_params)
    if not _requested_freeze and freeze_nuisance:
        _requested_freeze = {
            label: True for label in TIER2_NUISANCE_LABELS if label != "Av"
        }

    _managed_labels = [label for label in _requested_freeze if label in model.params]
    _pre_mcmc_values = {label: float(model.params[label]) for label in _managed_labels}
    _pre_mcmc_frozen = {label: (label not in model.labels) for label in _managed_labels}
    _applied_freezes = _apply_freeze_settings(model, _requested_freeze, thaw_first=True)
    _frozen_param_values = {
        label: float(model.params[label]) for label in _applied_freezes
    }

    if "Av" in model.params and "Av" in model.labels and "Av" not in priors:
        priors = dict(priors)
        priors["Av"] = stats.uniform(loc=0, scale=2.0)

    _sampled_labels = list(model.labels)
    ndim = len(_sampled_labels)
    if ndim == 0:
        raise ValueError("MCMC has no thawed parameters to sample after applying the freeze settings.")

    # Initialise walkers in a truncated-normal ball around MLE.
    # For Normal priors: σ = the distribution's own std (1σ).
    # For Uniform / other priors: σ = _INIT_FRAC × prior width.
    # Truncated normal avoids edge pile-up that np.clip would cause
    # near prior bounds.
    _INIT_FRAC = 0.15
    ball = np.empty((nwalkers, ndim))
    for i, key in enumerate(model.labels):
        pr = priors.get(key)
        mle_val = model[key]
        if pr is not None and hasattr(pr, 'interval'):
            lo, hi = pr.interval(1.0)
            # Use the distribution's native σ only for Normal-family
            # priors; for Uniform (and anything else) use a controlled
            # fraction of the prior width.
            _is_normal = getattr(getattr(pr, 'dist', None), 'name', '') in ('norm', 'truncnorm')
            if _is_normal:
                sigma = pr.std()
            else:
                sigma = _INIT_FRAC * (hi - lo)
            a = (lo - mle_val) / sigma
            b = (hi - mle_val) / sigma
            ball[:, i] = stats.truncnorm.rvs(
                a, b, loc=mle_val, scale=sigma, size=nwalkers
            )
        else:
            ball[:, i] = mle_val + 0.1 * np.random.randn(nwalkers)

    def log_prob(P):
        model.set_param_vector(P)
        # Returning -inf is the emcee-compatible way to reject proposals outside
        # the emulator grid without aborting the whole sampling run.
        gp = np.array(model.grid_params)
        if np.any(gp < model.emulator.min_params) or np.any(gp > model.emulator.max_params):
            return -np.inf
        try:
            return model.log_likelihood(priors)
        except (ValueError, np.linalg.LinAlgError):
            return -np.inf

    # Use DEMove + DESnookerMove for better performance in ≥5D
    # (Ter Braak 2006; Nelson et al. 2014).
    _moves = [
        (emcee.moves.DEMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=_moves)

    # Run MCMC, calling iteration_callback every 50 steps if provided
    if iteration_callback is not None:
        _mcmc_t0 = time.time()
        _state = None
        for _step, _state in enumerate(
            sampler.sample(ball, iterations=nsteps), start=1
        ):
            if _step % 50 == 0:
                iteration_callback(_step, nsteps, time.time() - _mcmc_t0)
    else:
        sampler.run_mcmc(ball, nsteps, progress=False)

    # Use the estimated autocorrelation time to choose a conservative burn-in and
    # thinning rule, but degrade gracefully when the chain is too short for tau.
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_valid = not (np.isnan(tau).any() or (tau == 0).any())
    except Exception:
        tau = np.full(ndim, np.nan)
        tau_valid = False

    if tau_valid:
        auto_burnin = int(2 * tau.max())  # 2×τ (Foreman-Mackey 2013)
        thin = max(1, int(0.3 * np.min(tau)))
        burnin_used = max(burnin, auto_burnin)
    else:
        thin = 1
        burnin_used = burnin

    if burnin_used >= nsteps:
        burnin_used = max(0, nsteps // 2)

    chain = sampler.get_chain(discard=burnin_used, thin=thin)
    flat = chain.reshape((-1, ndim))

    # Summary
    friendly = internal_to_friendly(
        [l for l in model.labels if l.startswith("param")],
        grid_name,
    )
    all_labels = []
    fi = 0
    for l in model.labels:
        if l.startswith("param"):
            all_labels.append(friendly[fi])
            fi += 1
        else:
            all_labels.append(l)

    summary = {}
    r_hat_dict = {}
    ess_dict = {}

    # Compute simple Gelman-Rubin style diagnostics from the walker-wise chains.
    per_chain = chain.transpose(1, 0, 2)  # (walkers, steps_after, ndim)

    for i, label in enumerate(all_labels):
        vals = flat[:, i]
        summary[label] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "hdi_3": float(np.percentile(vals, 3)),
            "hdi_97": float(np.percentile(vals, 97)),
        }

        # Compare within-walker variance to between-walker variance; r_hat values
        # close to 1 indicate the walkers are exploring the same stationary region.
        chain_means = np.array([np.mean(per_chain[w, :, i]) for w in range(nwalkers)])
        chain_vars = np.array([np.var(per_chain[w, :, i]) for w in range(nwalkers)])
        W = np.mean(chain_vars)
        B = np.var(chain_means) * chain.shape[0]
        var_est = (1 - 1.0 / chain.shape[0]) * W + B / chain.shape[0]
        r_hat = np.sqrt(var_est / W) if W > 0 else np.nan
        r_hat_dict[label] = float(r_hat)

        # The code currently uses the post-thinning flat sample count as a simple,
        # conservative ESS proxy rather than an autocorrelation-based estimator.
        ess_dict[label] = int(flat.shape[0])  # conservative: total thinned samples

    # Treat convergence as a pragmatic quality gate for the viewer: all finite
    # r_hat values must be below 1.1 and there must be enough retained samples
    # to support posterior summaries Vivekananda 2019 (Vehtari et al. 2021, arXiv:1903.08008 criticizes R_hat choice).
    converged = all(rh < 1.1 for rh in r_hat_dict.values() if np.isfinite(rh))
    converged = converged and flat.shape[0] >= 100

    # Best-fit spectrum at posterior means — set model params, evaluate, and
    # store the arrays so the viewer can reconstruct the Starfish-style plot
    # without needing the live model object.
    bestfit_spec = {}
    try:
        _mean_params = {}
        for i, label in enumerate(_sampled_labels):
            _mean_params[label] = float(np.mean(flat[:, i]))
        model.set_param_dict(_mean_params)
        _bf_flux, _bf_cov = model()
        if hasattr(_bf_flux, 'detach'):
            _bf_flux = _bf_flux.detach().cpu().numpy()
        if hasattr(_bf_cov, 'detach'):
            _bf_cov = _bf_cov.detach().cpu().numpy()
        bestfit_spec = {
            "wavelength": model.data.wave.tolist(),
            "data_flux": model.data.flux.tolist(),
            "model_flux": np.asarray(_bf_flux).tolist(),
            "model_cov_diag": np.diag(np.asarray(_bf_cov)).tolist(),
        }
    except Exception:
        pass  # non-critical — viewer will skip the plot

    # Restore any nuisance params that were frozen for this MCMC run so the
    # model object is left in a usable state for downstream callers.
    for _lbl in _managed_labels:
        if _lbl not in model.params:
            continue
        if _pre_mcmc_frozen.get(_lbl, False):
            model.freeze(_lbl)
        else:
            try:
                model.thaw(_lbl)
            except Exception:
                pass
        model[_lbl] = _pre_mcmc_values[_lbl]

    # Full chain (nsteps, nwalkers, ndim) before burn-in — used for
    # chain trace plots in the benchmark viewer.
    full_chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)

    return {
        "samples": flat,
        "full_chain": full_chain,
        "burnin_used": burnin_used,
        "thin": thin,
        "summary": summary,
        "r_hat": r_hat_dict,
        "ess_bulk": ess_dict,
        "converged": converged,
        "n_effective": flat.shape[0],
        "labels": all_labels,
        "internal_labels": _sampled_labels,
        "bestfit_spec": bestfit_spec,
        "freeze_params": dict(_requested_freeze),
        "frozen_params": list(_applied_freezes),
        "frozen_param_values": dict(_frozen_param_values),
    }


def compute_crps(samples: np.ndarray, truth: float) -> float:
    """
    Empirical Continuous Ranked Probability Score.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are iid draws from the posterior and y is the observation.
    """
    n = len(samples)
    if n == 0:
        return np.nan
    term1 = np.mean(np.abs(samples - truth))
    # Efficient pairwise computation
    sorted_s = np.sort(samples)
    # E|X - X'| = 2 * sum_i (2*i - n - 1) * x_{(i)} / n^2
    idx = np.arange(1, n + 1)
    term2 = 2.0 * np.sum((2 * idx - n - 1) * sorted_s) / (n * n)
    return float(term1 - 0.5 * term2)


def compute_coverage(
    samples_list: List[np.ndarray],
    truths: List[float],
    alphas: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coverage probability curve (PP-plot) from multiple posterior samples.

    Parameters
    ----------
    samples_list : list of (N,) arrays
        Posterior samples for one parameter across multiple test spectra.
    truths : list of float
        Ground truth values for each test spectrum.
    alphas : ndarray or None
        Nominal credible levels to evaluate. Default: np.linspace(0.01, 0.99, 50).

    Returns
    -------
    alphas, coverage : tuple of ndarrays
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.999, 50)

    coverage = np.zeros_like(alphas)
    n = len(samples_list)
    if n == 0:
        return alphas, coverage

    for i, alpha in enumerate(alphas):
        count = 0
        for samples, truth in zip(samples_list, truths):
            lo = np.percentile(samples, (1 - alpha) / 2 * 100)
            hi = np.percentile(samples, (1 + alpha) / 2 * 100)
            if lo <= truth <= hi:
                count += 1
        coverage[i] = count / n

    return alphas, coverage


# --- Public aliases for helpers used by the viewer-driven loop ---------
load_test_grid_spectrum = _load_test_grid_spectrum
extract_ground_truth = _extract_ground_truth


def ensure_lookup_table(test_grid_path: Union[str, Path]) -> Path:
    """Return the path to the parquet lookup table, downloading it if missing."""
    p = Path(test_grid_path) / "grid_run_lookup_table.parquet"
    if not p.exists():
        dataset_name = Path(test_grid_path).name
        try:
            import shutil
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=f"Sirocco-rt/{dataset_name}",
                filename="grid_run_lookup_table.parquet",
                repo_type="dataset",
            )
            shutil.copy2(local, str(p))
            log.info(f"Downloaded lookup table to {p}")
        except Exception as e:
            log.warning(f"Could not download lookup table: {e}")
    return p


def aggregate_tier2_results(
    per_spectrum: List[dict],
    all_samples: Dict[str, list],
    all_truths: Dict[str, list],
    friendly_names: List[str],
    n_params: int,
    emu_min_params: np.ndarray,
    emu_max_params: np.ndarray,
    spec_files_count: int,
    failures: int,
    failure_log: List[dict],
    mcmc_walkers: int,
    mcmc_steps: int,
    mcmc_burnin: int,
    elapsed: float,
    n_not_converged: int = 0,
    mle_freeze_params: Optional[Dict[str, bool]] = None,
    mcmc_freeze_params: Optional[Dict[str, bool]] = None,
) -> dict:
    """Aggregate per-spectrum Tier 2 results into final metrics dict.

    This is the post-loop aggregation extracted from ``run_tier2`` so that
    callers can drive the per-spectrum loop themselves (e.g. to show
    nested progress widgets) while still reusing the same scoring logic.
    """
    aggregate = {}
    for fname in friendly_names:
        if len(all_truths[fname]) == 0:
            continue

        truths_arr = np.array(all_truths[fname])
        means_arr = np.array(
            [ps.get(f"{fname}_mean", np.nan) for ps in per_spectrum]
        )
        means_arr = means_arr[~np.isnan(means_arr)]

        if len(means_arr) == len(truths_arr):
            rmse = float(np.sqrt(np.mean((means_arr - truths_arr) ** 2)))
            bias = float(np.mean(means_arr - truths_arr))
        else:
            rmse = np.nan
            bias = np.nan

        crps_vals = []
        for samples, truth in zip(all_samples[fname], all_truths[fname]):
            crps_vals.append(compute_crps(samples, truth))
        crps_mean = float(np.mean(crps_vals)) if crps_vals else np.nan

        prior_range = float(
            emu_max_params[friendly_names.index(fname)]
            - emu_min_params[friendly_names.index(fname)]
        ) if fname in friendly_names[:n_params] else np.nan
        post_stds = [ps.get(f"{fname}_std", np.nan) for ps in per_spectrum]
        mean_post_std = float(np.nanmean(post_stds))
        shrinkage = (
            1.0 - mean_post_std / (prior_range / np.sqrt(12))
            if np.isfinite(prior_range) and prior_range > 0
            else np.nan
        )

        alphas, cov = compute_coverage(all_samples[fname], all_truths[fname])
        cov_68 = float(np.interp(0.68, alphas, cov))
        cov_95 = float(np.interp(0.95, alphas, cov))
        cov_997 = float(np.interp(0.997, alphas, cov))

        aggregate[fname] = {
            "rmse": rmse,
            "bias": bias,
            "crps": crps_mean,
            "shrinkage": shrinkage,
            "coverage_68": cov_68,
            "coverage_95": cov_95,
            "coverage_997": cov_997,
            "coverage_alphas": alphas.tolist(),
            "coverage_values": cov.tolist(),
        }

    return {
        "per_spectrum": per_spectrum,
        "aggregate": aggregate,
        "n_spectra": spec_files_count,
        "n_processed": len(per_spectrum),
        "n_failures": failures,
        "n_not_converged": n_not_converged,
        "failure_log": failure_log,
        "mle_config": {
            "freeze_params": _serialise_freeze_settings(mle_freeze_params),
        },
        "mcmc_config": {
            "walkers": mcmc_walkers,
            "steps": mcmc_steps,
            "burnin": mcmc_burnin,
            "freeze_params": _serialise_freeze_settings(mcmc_freeze_params),
        },
        "tier2_time_s": elapsed,
    }


def run_tier2(
    emu,
    test_grid_path: str,
    flux_scale: str = "linear",
    wl_range: Tuple[float, float] = (850, 1850),
    inclination: float = 55.0,
    mcmc_walkers: int = 64,
    mcmc_steps: int = 2500,
    mcmc_burnin: int = 500,
    max_mle_iter: int = 5000,
    mle_restarts: int = 1,
    max_spectra: Optional[int] = None,
    mle_freeze_params: Optional[Dict[str, bool]] = None,
    mcmc_freeze_params: Optional[Dict[str, bool]] = None,
    progress_callback=None,
) -> dict:
    """
    Tier 2 benchmark: test grid parameter recovery.

    Parameters
    ----------
    emu : Emulator
    test_grid_path : str
        Path to the test grid folder (e.g. ``sirocco_grids/speculate_cv_no-bl_testgrid_v87f``).
    flux_scale : str
    wl_range : tuple
    inclination : float
    mcmc_walkers, mcmc_steps, mcmc_burnin : int
    max_mle_iter : int
    mle_restarts : int
        Number of CMA-ES restarts per spectrum.  More restarts reduce the
        chance of settling in a local minimum at the cost of runtime.
    max_spectra : int or None
        If set, limit the number of test spectra processed.
    progress_callback : callable or None
        If provided, called as ``progress_callback(current, total, spec_name, stage)``
        after each spectrum completes (or fails). Useful for live UI updates.

    Returns
    -------
    dict with calibration metrics, per-spectrum results, and aggregate scores.
    """
    t0 = time.time()
    test_path = Path(test_grid_path)
    # Prefer the test-grid path for grid inference because Tier 2 may run against
    # an emulator object loaded from a generic file-like source with no name.
    _tier_grid_name = infer_grid_name(str(test_path).replace("_testgrid_", "_grid_")) or infer_grid_name(getattr(emu, "name", None))
    if _tier_grid_name is None:
        log.warning(
            "Could not infer grid family from Tier 2 test path '%s' or emulator name '%s'; "
            "falling back to legacy CV no-BL metadata.",
            test_path,
            getattr(emu, "name", None),
        )

    # Tier 2 is driven directly from the decompressed test-grid spectra, with the
    # optional max_spectra cap used to keep exploratory runs tractable in the UI.
    spec_files = sorted(test_path.glob("run*.spec"))
    if max_spectra is not None:
        spec_files = spec_files[:max_spectra]

    if not spec_files:
        raise FileNotFoundError(f"No .spec files found in {test_grid_path}")

    # The lookup parquet links each run file back to its known simulation inputs.
    parquet_file = ensure_lookup_table(test_path)

    # Friendly labels, freeze defaults, inclination columns, and ground-truth
    # conversion all depend on the inferred grid family.
    friendly_names = internal_to_friendly(emu.param_names, _tier_grid_name)
    n_params = len(emu.param_names)
    _tier2_defaults = build_tier2_freeze_defaults(emu.param_names, _tier_grid_name)
    _mle_defaults = _tier2_defaults["mle"]
    _mcmc_defaults = _tier2_defaults["mcmc"]
    mle_freeze_params = {
        label: bool((mle_freeze_params or {}).get(label, _mle_defaults[label]))
        for label in _mle_defaults
    }
    mcmc_freeze_params = {
        label: bool((mcmc_freeze_params or {}).get(label, _mcmc_defaults[label]))
        for label in _mcmc_defaults
    }

    per_spectrum = []
    all_samples = {name: [] for name in friendly_names}
    all_truths = {name: [] for name in friendly_names}
    failures = 0
    n_not_converged = 0
    failure_log: List[dict] = []  # surfaced to viewer

    _n_total = len(spec_files)
    for _spec_idx, sf in enumerate(spec_files):
        run_idx = int(sf.stem.replace("run", ""))
        log.info(f"Processing {sf.name}...")

        if progress_callback is not None:
            progress_callback(_spec_idx, _n_total, sf.name, "loading")

        try:
            wl, flux, sigma = _load_test_grid_spectrum(str(sf), inclination, wl_range, _tier_grid_name)
        except Exception as e:
            import traceback as _tb
            _msg = str(e)
            log.warning(f"Failed to load {sf.name}: {_msg}")
            failure_log.append({"run": sf.name, "stage": "load", "error": _msg,
                                 "traceback": _tb.format_exc()})
            failures += 1
            if progress_callback is not None:
                progress_callback(_spec_idx + 1, _n_total, sf.name, "failed:load")
            continue

        # Ground truth extraction is best-effort: a missing or malformed lookup
        # row should be logged, but it should not prevent inference on the flux.
        gt = {}
        if parquet_file.exists():
            try:
                gt = _extract_ground_truth(str(parquet_file), run_idx, emu.param_names, _tier_grid_name, inclination)
            except Exception as e:
                _msg = str(e)
                log.warning(f"Failed to extract GT for {sf.name}: {_msg}")
                failure_log.append({"run": sf.name, "stage": "ground_truth", "error": _msg,
                                     "traceback": ""})

        # Always record the true inclination so it can be overlaid on corner
        # plots when inclination is an MCMC-sampled parameter (param9-11).
        gt.setdefault("Inclination", inclination)

        # MLE
        if progress_callback is not None:
            progress_callback(_spec_idx, _n_total, sf.name, "MLE")

        try:
            mle_result = run_mle_single(
                emu, wl, flux, sigma, flux_scale=flux_scale,
                max_iter=max_mle_iter, n_restarts=mle_restarts,
                freeze_params=mle_freeze_params,
            )
        except Exception as e:
            import traceback as _tb
            _msg = str(e)
            log.warning(f"MLE failed for {sf.name}: {_msg}")
            failure_log.append({"run": sf.name, "stage": "MLE", "error": _msg,
                                 "traceback": _tb.format_exc()})
            failures += 1
            if progress_callback is not None:
                progress_callback(_spec_idx + 1, _n_total, sf.name, "failed:MLE")
            continue

        # MCMC
        if progress_callback is not None:
            progress_callback(_spec_idx, _n_total, sf.name, "MCMC")

        try:
            mcmc_result = run_mcmc_single(
                mle_result["model"],
                mle_result["priors"],
                nwalkers=mcmc_walkers,
                nsteps=mcmc_steps,
                burnin=mcmc_burnin,
                freeze_params=mcmc_freeze_params,
                grid_name=_tier_grid_name,
            )
        except Exception as e:
            import traceback as _tb
            _msg = str(e)
            log.warning(f"MCMC failed for {sf.name}: {_msg}")
            failure_log.append({"run": sf.name, "stage": "MCMC", "error": _msg,
                                 "traceback": _tb.format_exc()})
            failures += 1
            if progress_callback is not None:
                progress_callback(_spec_idx + 1, _n_total, sf.name, "failed:MCMC")
            continue

        if not mcmc_result["converged"]:
            n_not_converged += 1

        # Keep both a compact per-spectrum summary and the raw posterior draws
        # needed for aggregate calibration metrics across the whole test grid.
        spec_result = {
            "run": run_idx,
            "filename": sf.name,
            "inclination": inclination,
            "mle_success": mle_result["success"],
            "mcmc_converged": mcmc_result["converged"],
            "n_effective": mcmc_result["n_effective"],
            "mle_grid_params": mle_result["grid_params"],
            "mle_all_params": mle_result.get("all_params", {}),
            "mle_freeze_settings": mle_result.get("freeze_params", {}),
            "mle_frozen_params": mle_result.get("frozen_params", []),
            "mcmc_freeze_settings": mcmc_result.get("freeze_params", {}),
            "mcmc_frozen_params": mcmc_result.get("frozen_params", []),
            "mcmc_frozen_param_values": mcmc_result.get("frozen_param_values", {}),
        }

        # Map friendly names to their column index in the flat samples array.
        # The columns follow model.labels order (nuisance params first, then
        # grid params), so we cannot assume grid param i maps to column i.
        _mcmc_labels = mcmc_result.get("labels", [])
        for i, fname in enumerate(friendly_names):
            if fname in mcmc_result["summary"]:
                spec_result[f"{fname}_mean"] = mcmc_result["summary"][fname]["mean"]
                spec_result[f"{fname}_std"] = mcmc_result["summary"][fname]["std"]

                # Find the correct column for this parameter in the samples array
                if fname in _mcmc_labels:
                    _col = _mcmc_labels.index(fname)
                else:
                    _col = i  # fallback (legacy behaviour)
                samples_i = mcmc_result["samples"][:, _col]
                all_samples[fname].append(samples_i)

                if fname in gt:
                    all_truths[fname].append(gt[fname])
                    spec_result[f"{fname}_truth"] = gt[fname]
                    spec_result[f"{fname}_delta_sigma"] = (
                        mcmc_result["summary"][fname]["mean"] - gt[fname]
                    ) / max(mcmc_result["summary"][fname]["std"], 1e-10)

        per_spectrum.append(spec_result)

        if progress_callback is not None:
            _status = "done" if mcmc_result["converged"] else "done:not_converged"
            progress_callback(_spec_idx + 1, _n_total, sf.name, _status)

    # ----- Aggregate metrics across all test spectra -----
    # For each physical parameter we compute:
    #   RMSE   — root-mean-square error of posterior means vs ground truth
    #   Bias   — signed mean offset (positive = overestimate)
    #   CRPS   — Continuous Ranked Probability Score (proper scoring rule that
    #            penalises both miscalibration and low sharpness)
    #   Shrinkage — how much the posterior narrows relative to the prior
    #              (1 = perfectly informative, 0 = no information gain)
    #   Coverage — empirical coverage at 68% and 95% credible levels (if
    #             well-calibrated, ~68% and ~95% of truths fall inside the
    #             posterior intervals at those levels)
    aggregate = {}
    for fname in friendly_names:
        if len(all_truths[fname]) == 0:
            continue

        truths_arr = np.array(all_truths[fname])
        means_arr = np.array(
            [ps.get(f"{fname}_mean", np.nan) for ps in per_spectrum]
        )
        means_arr = means_arr[~np.isnan(means_arr)]

        # RMSE and bias of posterior means vs ground truth
        if len(means_arr) == len(truths_arr):
            rmse = float(np.sqrt(np.mean((means_arr - truths_arr) ** 2)))
            bias = float(np.mean(means_arr - truths_arr))
        else:
            rmse = np.nan
            bias = np.nan

        # CRPS: averaged over all test spectra for this parameter
        crps_vals = []
        for samples, truth in zip(all_samples[fname], all_truths[fname]):
            crps_vals.append(compute_crps(samples, truth))
        crps_mean = float(np.mean(crps_vals)) if crps_vals else np.nan

        # Posterior shrinkage: 1 − (mean posterior σ) / (prior σ)
        # Uses the standard deviation of the uniform prior: range / √12
        prior_range = float(emu.max_params[friendly_names.index(fname)] - emu.min_params[friendly_names.index(fname)]) if fname in friendly_names[:n_params] else np.nan
        post_stds = [ps.get(f"{fname}_std", np.nan) for ps in per_spectrum]
        mean_post_std = float(np.nanmean(post_stds))
        shrinkage = 1.0 - mean_post_std / (prior_range / np.sqrt(12)) if np.isfinite(prior_range) and prior_range > 0 else np.nan

        # PP-plot coverage: the fraction of test cases whose ground truth falls
        # inside the α-level credible interval, evaluated at many α values.
        alphas, cov = compute_coverage(all_samples[fname], all_truths[fname])
        # Interpolate to standard reporting levels
        cov_68 = float(np.interp(0.68, alphas, cov))
        cov_95 = float(np.interp(0.95, alphas, cov))
        cov_997 = float(np.interp(0.997, alphas, cov))

        aggregate[fname] = {
            "rmse": rmse,
            "bias": bias,
            "crps": crps_mean,
            "shrinkage": shrinkage,
            "coverage_68": cov_68,
            "coverage_95": cov_95,
            "coverage_997": cov_997,
            "coverage_alphas": alphas.tolist(),
            "coverage_values": cov.tolist(),
        }

    elapsed = time.time() - t0
    return {
        "per_spectrum": per_spectrum,
        "aggregate": aggregate,
        "n_spectra": len(spec_files),
        "n_processed": len(per_spectrum),
        "n_failures": failures,
        "n_not_converged": n_not_converged,
        "failure_log": failure_log,
        "mle_config": {
            "freeze_params": _serialise_freeze_settings(mle_freeze_params),
        },
        "mcmc_config": {
            "walkers": mcmc_walkers,
            "steps": mcmc_steps,
            "burnin": mcmc_burnin,
            "freeze_params": _serialise_freeze_settings(mcmc_freeze_params),
        },
        "tier2_time_s": elapsed,
    }


# ======================================================================
# Tier 3 — Observational Spectra
# ======================================================================


def _prepend_env_path(name: str, path: Union[str, Path]) -> bool:
    """Prepend a directory to a path-like environment variable if needed."""
    path = str(Path(path).resolve())
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path in parts:
        return False
    os.environ[name] = path if not current else path + os.pathsep + current
    return True


def _candidate_sirocco_roots() -> List[Path]:
    """Return plausible Sirocco install roots for reconnect-safe discovery."""
    candidates = []
    env_root = os.environ.get("SIROCCO")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    try:
        repo_parent = Path(__file__).resolve().parents[2]
        candidates.append(repo_parent / "sirocco")
    except Exception:
        pass

    candidates.append(Path.home() / "sirocco")

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _configure_sirocco_environment() -> dict:
    """Patch this Python process environment from a discoverable Sirocco root."""
    for root in _candidate_sirocco_roots():
        root = root.expanduser()
        bin_dir = root / "bin"
        py_dir = root / "py_progs"
        if not ((bin_dir / "sirocco").is_file() and (bin_dir / "Setup_Sirocco_Dir").is_file()):
            continue

        os.environ["SIROCCO"] = str(root.resolve())
        added = []
        if _prepend_env_path("PATH", bin_dir):
            added.append(str(bin_dir.resolve()))
        if py_dir.is_dir():
            if _prepend_env_path("PATH", py_dir):
                added.append(str(py_dir.resolve()))
            _prepend_env_path("PYTHONPATH", py_dir)

        return {
            "configured": True,
            "sirocco_root": str(root.resolve()),
            "added_paths": added,
        }

    return {
        "configured": False,
        "sirocco_root": None,
        "added_paths": [],
    }


def check_sirocco_runtime(cpus: int = 1) -> dict:
    """Return executable availability for the Tier 3 Sirocco subprocess path.

    Parameters
    ----------
    cpus : int
        Requested Sirocco process count. Values greater than one require
        ``mpirun`` in addition to the ``sirocco`` executable.

    ``Setup_Sirocco_Dir`` is always required because Tier 3 prepares each
    per-observation export directory before launching Sirocco.
    """
    cpus = max(1, int(cpus or 1))
    env_setup = {
        "configured": False,
        "sirocco_root": os.environ.get("SIROCCO"),
        "added_paths": [],
    }
    if shutil.which("sirocco") is None or shutil.which("Setup_Sirocco_Dir") is None:
        env_setup = _configure_sirocco_environment()

    result = {
        "cpus": cpus,
        "sirocco": shutil.which("sirocco"),
        "mpirun": shutil.which("mpirun") if cpus > 1 else None,
        "setup_sirocco_dir": shutil.which("Setup_Sirocco_Dir"),
        "missing": [],
        "environment": env_setup,
    }
    if result["sirocco"] is None:
        result["missing"].append("sirocco")
    if cpus > 1 and result["mpirun"] is None:
        result["missing"].append("mpirun")
    if result["setup_sirocco_dir"] is None:
        result["missing"].append("Setup_Sirocco_Dir")
    result["ok"] = not result["missing"]
    return result


def _repo_relative(path: Union[str, Path]) -> str:
    """Return a stable path string relative to the current working directory."""
    try:
        return os.path.relpath(path, os.getcwd())
    except Exception:
        return str(path)


def _emit_progress(callback, event: dict) -> None:
    """Invoke an optional progress callback without letting UI errors abort work."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:
        log.debug("Progress callback failed: %s", exc)


def _parse_sirocco_signal_line(line: str) -> Optional[dict]:
    """Parse a Sirocco `.sig` line into a user-facing cycle progress event."""
    match = _SIROCCO_CYCLE_RE.search(line)
    if match:
        kind = match.group("kind").lower()
        action = match.group("action").lower()
        current = int(match.group("current"))
        total = int(match.group("total"))
        label = "Ionization" if kind == "ionization" else "Spectrum"
        state = "complete" if action == "finished" else "started"
        return {
            "phase": "sirocco",
            "cycle_type": kind,
            "state": state,
            "current": current,
            "total": total,
            "line": line.rstrip(),
            "message": f"{label} cycle {current}/{total} {state}",
        }

    match = _SIROCCO_ION_CONVERGED_RE.search(line)
    if match:
        current = int(match.group("current"))
        total = int(match.group("total"))
        return {
            "phase": "sirocco",
            "cycle_type": "ionization",
            "state": "converged",
            "current": current,
            "total": total,
            "line": line.rstrip(),
            "message": f"Ionization converged early at cycle {current}/{total}",
        }

    return None


def _poll_sirocco_signal(
    signal_path: Union[str, Path],
    offset: int,
    last_event_key: Optional[Tuple],
    progress_callback=None,
) -> Tuple[int, Optional[Tuple]]:
    """Read new `.sig` lines and emit the latest unseen cycle progress event."""
    signal_path = Path(signal_path)
    if not signal_path.exists():
        return offset, last_event_key

    size = signal_path.stat().st_size
    if size < offset:
        offset = 0

    latest_event = None
    with open(signal_path, "r", errors="replace") as signal_file:
        signal_file.seek(offset)
        for line in signal_file:
            event = _parse_sirocco_signal_line(line)
            if event is not None:
                latest_event = event
        offset = signal_file.tell()

    if latest_event is not None:
        event_key = (
            latest_event.get("cycle_type"),
            latest_event.get("state"),
            latest_event.get("current"),
            latest_event.get("total"),
        )
        if event_key != last_event_key:
            _emit_progress(progress_callback, latest_event)
            last_event_key = event_key

    return offset, last_event_key


def _find_sirocco_spec_files(pf_path: Union[str, Path]) -> List[Path]:
    """Return native Sirocco `.spec` outputs for an exported `.pf` run."""
    pf_path = Path(pf_path)
    work_dir = pf_path.parent
    root_spec = work_dir / f"{pf_path.stem}.spec"
    if root_spec.is_file():
        return [root_spec]
    return sorted(work_dir.glob("*.spec"))


def _safe_artifact_stem(path: Union[str, Path]) -> str:
    """Return a filesystem-safe stem for one observation's artifact names."""
    stem = Path(path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return safe.strip("_") or "observation"


def _run_command_logged(
    command: List[str],
    cwd: Union[str, Path],
    log_path: Union[str, Path],
    progress_callback=None,
    signal_path: Optional[Union[str, Path]] = None,
    append: bool = False,
    raise_on_error: bool = True,
) -> int:
    """Run a subprocess, capture combined output, and return its exit code."""
    command_display = " ".join(command)
    log_mode = "a" if append else "w"

    def _write_command_header(log_file) -> None:
        if append and log_file.tell() > 0:
            log_file.write("\n\n")
        log_file.write("$ " + command_display + "\n\n")

    if progress_callback is None or signal_path is None:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        with open(log_path, log_mode) as log_file:
            _write_command_header(log_file)
            log_file.write(completed.stdout or "")
        return_code = completed.returncode
    else:
        offset = 0
        last_event_key = None
        with open(log_path, log_mode) as log_file:
            _write_command_header(log_file)
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                offset, last_event_key = _poll_sirocco_signal(
                    signal_path, offset, last_event_key, progress_callback
                )
                time.sleep(1.0)
            return_code = process.returncode
            _poll_sirocco_signal(signal_path, offset, last_event_key, progress_callback)

    if return_code != 0 and raise_on_error:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: "
            f"{command_display}. See {_repo_relative(log_path)}"
        )
    return return_code


def run_sirocco_pf(
    pf_path: Union[str, Path],
    cpus: int = 1,
    progress_callback=None,
) -> dict:
    """Run Sirocco for one exported .pf file and return log/output metadata.

    ``Setup_Sirocco_Dir`` and the Sirocco command are both executed inside the
    `.pf` file's directory so fixed-name outputs, such as ``run0.spec``, remain
    isolated per observation. If provided, ``progress_callback`` receives
    best-effort cycle updates parsed from Sirocco's live ``root.sig`` file.
    Multi-CPU MPI runs first try the standard ``mpirun -np`` form, then retry
    with ``--use-hwthread-cpus`` if the standard command fails.
    """
    pf_path = Path(pf_path)
    work_dir = pf_path.parent
    cpus = max(1, int(cpus or 1))
    runtime = check_sirocco_runtime(cpus)
    if not runtime["ok"]:
        missing = ", ".join(runtime["missing"])
        raise RuntimeError(f"Tier 3 requires Sirocco runtime command(s): {missing}")

    setup_log = work_dir / "setup_sirocco_dir.log"
    _emit_progress(progress_callback, {
        "phase": "sirocco",
        "state": "setup",
        "message": "Preparing Sirocco run directory",
    })
    _run_command_logged([runtime["setup_sirocco_dir"]], work_dir, setup_log)

    run_log = work_dir / "sirocco_run.log"
    signal_path = work_dir / f"{pf_path.stem}.sig"
    if cpus > 1:
        command = [runtime["mpirun"], "-np", str(cpus), runtime["sirocco"], pf_path.name]
    else:
        command = [runtime["sirocco"], pf_path.name]
    successful_command = command
    _emit_progress(progress_callback, {
        "phase": "sirocco",
        "state": "started",
        "message": "Simulation started; waiting for cycle log",
    })
    if cpus > 1:
        return_code = _run_command_logged(
            command,
            work_dir,
            run_log,
            progress_callback=progress_callback,
            signal_path=signal_path,
            raise_on_error=False,
        )
        if return_code != 0:
            fallback_command = [
                runtime["mpirun"],
                "--use-hwthread-cpus",
                "-np",
                str(cpus),
                runtime["sirocco"],
                pf_path.name,
            ]
            _emit_progress(progress_callback, {
                "phase": "sirocco",
                "state": "retrying",
                "message": "MPI run failed; retrying with hardware-thread CPUs",
            })
            _run_command_logged(
                fallback_command,
                work_dir,
                run_log,
                progress_callback=progress_callback,
                signal_path=signal_path,
                append=True,
            )
            successful_command = fallback_command
    else:
        _run_command_logged(
            command,
            work_dir,
            run_log,
            progress_callback=progress_callback,
            signal_path=signal_path,
        )

    spec_files = _find_sirocco_spec_files(pf_path)
    if not spec_files:
        raise FileNotFoundError(
            f"Sirocco completed but no .spec file was found in {_repo_relative(work_dir)}"
        )

    return {
        "command": " ".join(successful_command),
        "setup_log_path": str(setup_log),
        "run_log_path": str(run_log),
        "signal_log_path": str(signal_path),
        "spec_files": [str(path) for path in spec_files],
    }


def _load_single_observer_sirocco_spectrum(
    spec_file: Union[str, Path],
    wl_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load wavelength and flux from a reduced single-observer Sirocco `.spec`.

    Sirocco writes frequency and wavelength in columns 0 and 1, followed by one
    flux column per observer. Tier 3 exports exactly one observer, so column 2
    is the comparison spectrum.
    """
    skiprows = 0
    with open(spec_file, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Freq."):
                skiprows = i + 1
            else:
                break

    data = np.loadtxt(spec_file, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in Sirocco spectrum {spec_file}")

    order = np.argsort(data[:, 1])
    wl = data[order, 1]
    flux = data[order, 2]
    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    wl = wl[mask]
    flux = flux[mask]
    if len(wl) < 2:
        raise ValueError(
            f"Only {len(wl)} Sirocco wavelength point(s) remain after filtering "
            f"to wl_range={wl_range} in {spec_file}"
        )
    return wl, flux


def _transform_flux_for_scale(
    wl: np.ndarray,
    flux: np.ndarray,
    flux_scale: str,
) -> np.ndarray:
    """Transform a native linear-flux spectrum onto the emulator fit scale."""
    wl = np.asarray(wl, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if flux_scale == "log":
        flux = np.where(flux > 0, np.log10(flux), np.log10(np.abs(flux) + 1e-30))
        finite = np.isfinite(flux)
        offset = float(np.mean(flux[finite])) if np.any(finite) else 0.0
        return flux - offset
    if flux_scale == "continuum-normalised":
        from Speculate_addons.Spec_functions import fit_power_law_continuum

        continuum, _ = fit_power_law_continuum(wl, flux)
        flux = flux / np.where(continuum > 0, continuum, 1.0)

    finite = np.isfinite(flux)
    factor = float(np.mean(flux[finite])) if np.any(finite) else 1.0
    if not np.isfinite(factor) or abs(factor) == 0.0:
        factor = 1.0
    flux = flux / factor
    return flux


def _extract_spectrum_nuisance_transforms(model) -> dict:
    """Capture fitted continuum/scale nuisance values from a SpectrumModel."""
    transforms = {}
    for label in ("Av", "Rv", "log_scale"):
        if label in model.params:
            transforms[label] = float(model.params[label])
    if "Av" in transforms and "Rv" not in transforms:
        transforms["Rv"] = 3.1

    cheb_terms = []
    idx = 1
    while f"cheb:{idx}" in model.params:
        cheb_terms.append(float(model.params[f"cheb:{idx}"]))
        idx += 1
    if cheb_terms:
        transforms["cheb"] = cheb_terms
    return transforms


def _apply_spectrum_nuisance_transforms(
    wl: np.ndarray,
    flux: np.ndarray,
    flux_scale: str,
    transforms: dict,
) -> np.ndarray:
    """Apply fitted Starfish nuisance transforms to an external spectrum.

    The order mirrors ``SpectrumModel`` evaluation: extinction, Chebyshev
    continuum correction, then global flux scale.  Log-scale spectra use the
    additive forms of those transforms; linear and continuum-normalised spectra
    use the multiplicative forms.
    """
    wl = np.asarray(wl, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64).copy()

    if "Av" in transforms:
        from Starfish.transforms import extinct

        flux = extinct(
            wl,
            flux,
            float(transforms["Av"]),
            Rv=float(transforms.get("Rv", 3.1)),
            flux_scale="log" if flux_scale == "log" else "linear",
        )

    cheb_terms = transforms.get("cheb") or []
    if cheb_terms:
        coeffs = np.asarray([1.0, *[float(value) for value in cheb_terms]], dtype=np.float64)
        if flux_scale == "log":
            from numpy.polynomial.chebyshev import chebval

            cheb_poly = chebval(wl / np.max(wl), coeffs)
            flux = flux + np.log10(np.clip(cheb_poly, 1e-30, None))
        else:
            from Starfish.transforms import chebyshev_correct

            flux = chebyshev_correct(wl, flux, coeffs)

    if "log_scale" in transforms:
        log_scale = float(transforms["log_scale"])
        if flux_scale == "log":
            flux = flux + log_scale / np.log(10.0)
        else:
            flux = flux * np.exp(log_scale)
    return flux


def _format_sirocco_transform_label(transforms: dict) -> str:
    """Return a compact plot legend label for the transformed Sirocco spectrum."""
    parts = []
    if "Av" in transforms:
        parts.append(f"Av={float(transforms['Av']):.3g}")
    if "log_scale" in transforms:
        parts.append(f"log_scale={float(transforms['log_scale']):.3g}")
    for idx, value in enumerate(transforms.get("cheb") or [], start=1):
        parts.append(f"cheb{idx}={float(value):.3g}")
    return "Sirocco Model" if not parts else "Sirocco Model (" + ", ".join(parts) + ")"


def _extract_posterior_mean_inclination(
    emu,
    samples: np.ndarray,
    friendly_labels: Sequence[str],
    grid_name: Optional[str] = None,
) -> float:
    """Return the posterior-mean inclination used for the Sirocco observer."""
    friendly_grid = internal_to_friendly(emu.param_names, grid_name)
    if "Inclination" not in friendly_grid:
        raise ValueError(
            "Tier 3 Sirocco comparison requires an emulator with an Inclination "
            "parameter (param9, param10, or param11)."
        )
    if "Inclination" not in friendly_labels:
        raise ValueError("MCMC samples do not contain an Inclination column.")
    col = list(friendly_labels).index("Inclination")
    inclination = float(np.mean(samples[:, col]))
    return float(np.clip(inclination, 0.0, 90.0))


def _save_tier3_artifacts(
    output_dir: Union[str, Path],
    obs_stem: str,
    mcmc: dict,
    bestfit_spec: dict,
    ppc_envelope: dict,
    sirocco_plot: dict,
    summary: dict,
) -> dict:
    """Persist bulky Tier 3 posterior and plot arrays outside the JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior_path = output_dir / f"tier3_{obs_stem}_posterior.npz"
    np.savez_compressed(
        posterior_path,
        samples=mcmc.get("samples", np.empty((0, 0))),
        full_chain=mcmc.get("full_chain", np.empty((0, 0, 0))),
        labels=np.asarray(mcmc.get("labels", []), dtype=str),
        internal_labels=np.asarray(mcmc.get("internal_labels", []), dtype=str),
        burnin_used=np.asarray([mcmc.get("burnin_used", 0)], dtype=np.int64),
    )

    plot_path = output_dir / f"tier3_{obs_stem}_plot_data.npz"
    np.savez_compressed(
        plot_path,
        wavelength=np.asarray(bestfit_spec.get("wavelength", []), dtype=np.float64),
        data_flux=np.asarray(bestfit_spec.get("data_flux", []), dtype=np.float64),
        model_flux=np.asarray(bestfit_spec.get("model_flux", []), dtype=np.float64),
        model_cov_diag=np.asarray(bestfit_spec.get("model_cov_diag", []), dtype=np.float64),
        ppc_wavelength=np.asarray(ppc_envelope.get("wavelength", []), dtype=np.float64),
        ppc_low=np.asarray(ppc_envelope.get("low", []), dtype=np.float64),
        ppc_high=np.asarray(ppc_envelope.get("high", []), dtype=np.float64),
        sirocco_wavelength=np.asarray(sirocco_plot.get("wavelength", []), dtype=np.float64),
        sirocco_flux=np.asarray(sirocco_plot.get("flux", []), dtype=np.float64),
        sirocco_label=np.asarray([sirocco_plot.get("label", "Sirocco Model")], dtype=str),
    )

    summary_path = output_dir / f"tier3_{obs_stem}_summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2, default=str)

    return {
        "posterior_npz": str(posterior_path),
        "plot_data_npz": str(plot_path),
        "summary_json": str(summary_path),
    }


def run_tier3_single(
    emu,
    obs_csv: str,
    flux_scale: str = "linear",
    wl_range: Optional[Tuple[float, float]] = None,
    max_mle_iter: int = 5000,
    mle_restarts: int = 5,
    n_ppc_draws: int = 100,
    mcmc_walkers: int = 64,
    mcmc_steps: int = 2500,
    mcmc_burnin: int = 500,
    grid_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    sirocco_cpus: int = 1,
    require_sirocco: bool = True,
    run_sirocco: bool = True,
    mle_iteration_callback=None,
    mcmc_iteration_callback=None,
    sirocco_progress_callback=None,
) -> dict:
    """
    Tier 3 benchmark: goodness-of-fit for a single observational spectrum.

    Parameters
    ----------
    grid_name : str
        Grid identifier (e.g. ``speculate_cv_bl_grid_v87f``) used to select
        the correct Sirocco template for .pf export. Strict Tier 3 runs require
        this value before inference starts.
    output_dir : str or None
        Per-observation artifact directory.  Defaults to ``exports/``.
    wl_range : tuple or None
        Observation fitting window in Angstrom.  When None, Tier 3 fits over
        the selected emulator's wavelength coverage.  Explicit ranges must sit
        inside that coverage to avoid spline extrapolation.
    sirocco_cpus : int
        Number of CPUs to use when launching Sirocco.  Values above 1 use
        ``mpirun -np N sirocco <pf>``.
    require_sirocco : bool
        If True, missing runtime commands or failed Sirocco runs raise.
    run_sirocco : bool
        If True, run Sirocco after exporting the posterior-mean .pf file.
    mle_iteration_callback, mcmc_iteration_callback, sirocco_progress_callback : callable or None
        Optional progress callbacks forwarded to the MLE, MCMC, and Sirocco
        stages. The benchmark viewer uses these to show restart, evaluation,
        sampling-step, and radiative-transfer cycle progress during long Tier 3 fits.

    Returns
    -------
    dict with compact JSON-safe metrics and artifact path references.
    """
    t0 = time.time()
    obs_stem = _safe_artifact_stem(obs_csv)
    artifact_dir = Path(output_dir or "exports")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wl_range = _resolve_tier3_wl_range(emu, wl_range)

    if grid_name is None and (require_sirocco or run_sirocco):
        raise ValueError("Tier 3 Sirocco workflow requires a grid_name for .pf export.")
    if require_sirocco or run_sirocco:
        runtime = check_sirocco_runtime(sirocco_cpus)
        if not runtime["ok"]:
            missing = ", ".join(runtime["missing"])
            raise RuntimeError(f"Tier 3 requires Sirocco runtime command(s): {missing}")

    # Load observation
    df = pd.read_csv(obs_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    wl = np.array(df["wavelength"])
    flux = np.array(df["flux"])
    if "error" in df.columns:
        sigma = np.array(df["error"])
    else:
        from Speculate_addons.Spec_functions import build_default_observation_sigma

        sigma, _ = build_default_observation_sigma(wl, flux)

    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    wl, flux, sigma = wl[mask], flux[mask], sigma[mask]
    if len(wl) < 2:
        raise ValueError(
            f"Only {len(wl)} wavelength point(s) remain after filtering "
            f"{obs_csv} to wl_range={wl_range}."
        )

    # MLE — run_mle_single handles flux_scale transforms internally;
    # applying them here as well would double-scale the data.
    if mle_iteration_callback is not None:
        mle_iteration_callback(0, max_mle_iter, np.nan, 0.0, 1, mle_restarts)
    mle = run_mle_single(
        emu, wl, flux, sigma,
        flux_scale=flux_scale,
        max_iter=max_mle_iter,
        freeze_av=False,
        n_restarts=mle_restarts,
        iteration_callback=mle_iteration_callback,
    )
    model = mle["model"]
    priors = mle["priors"]

    # MCMC
    if mcmc_iteration_callback is not None:
        mcmc_iteration_callback(0, mcmc_steps, 0.0)
    mcmc = run_mcmc_single(
        model, priors,
        nwalkers=mcmc_walkers, nsteps=mcmc_steps, burnin=mcmc_burnin,
        freeze_nuisance=True,
        iteration_callback=mcmc_iteration_callback,
        grid_name=grid_name,
    )
    friendly_labels = mcmc.get("labels", [])
    internal_labels = mcmc.get("internal_labels", friendly_labels)

    # Posterior Predictive Check (PPC)
    # Draw random posterior samples, evaluate the model flux at each one, and
    # check what fraction of the observed data falls within the 2.5–97.5%
    # envelope of the predicted flux.  A well-calibrated model should cover
    # ~95% of the data points.
    ppc_in = np.nan
    n_ppc = min(n_ppc_draws, mcmc["samples"].shape[0])
    envelopes = np.zeros((n_ppc, len(flux)))

    if n_ppc > 0:
        indices = np.random.choice(mcmc["samples"].shape[0], size=n_ppc, replace=False)
        for j, idx in enumerate(indices):
            _sample_dict = dict(zip(internal_labels, mcmc["samples"][idx]))
            model.set_param_dict(_sample_dict)
            try:
                pred, _ = model()
                if hasattr(pred, "detach"):
                    pred = pred.detach().cpu().numpy()
                envelopes[j] = pred
            except Exception:
                envelopes[j] = np.nan

    valid = ~np.any(np.isnan(envelopes), axis=1)
    if valid.sum() > 1:
        env_lo = np.percentile(envelopes[valid], 2.5, axis=0)
        env_hi = np.percentile(envelopes[valid], 97.5, axis=0)
        data_for_ppc = np.asarray(model.data.flux, dtype=np.float64)
        ppc_in = float(np.mean((data_for_ppc >= env_lo) & (data_for_ppc <= env_hi)))
    else:
        env_lo = np.full(len(model.data.wave), np.nan)
        env_hi = np.full(len(model.data.wave), np.nan)
        ppc_in = np.nan

    # Set model back to posterior mean for export
    mcmc_means = {}
    for i, label in enumerate(internal_labels):
        mcmc_means[label] = float(np.mean(mcmc["samples"][:, i]))
    model.set_param_dict(mcmc_means)

    try:
        model_flux, model_cov = model()
        if hasattr(model_flux, "detach"):
            model_flux = model_flux.detach().cpu().numpy()
        if hasattr(model_cov, "detach"):
            model_cov = model_cov.detach().cpu().numpy()
        model_cov = np.asarray(model_cov)
        bestfit_spec = {
            "wavelength": np.asarray(model.data.wave).tolist(),
            "data_flux": np.asarray(model.data.flux).tolist(),
            "model_flux": np.asarray(model_flux).tolist(),
            "model_cov_diag": np.diag(model_cov).tolist() if model_cov.ndim == 2 else model_cov.tolist(),
        }
    except Exception:
        bestfit_spec = mcmc.get("bestfit_spec", {})

    try:
        _bf_data = np.asarray(bestfit_spec.get("data_flux", []), dtype=np.float64)
        _bf_model = np.asarray(bestfit_spec.get("model_flux", []), dtype=np.float64)
        _model_sigma = np.asarray(model.data.sigma, dtype=np.float64)
        _n = min(len(_bf_data), len(_bf_model), len(_model_sigma))
        residuals = _bf_data[:_n] - _bf_model[:_n]
        n_dof = _n - len(internal_labels)
        chi2 = float(np.sum((residuals / np.maximum(_model_sigma[:_n], 1e-30)) ** 2))
        reduced_chi2 = chi2 / max(n_dof, 1)
    except Exception:
        reduced_chi2 = np.nan

    exact_inclination = _extract_posterior_mean_inclination(
        emu, mcmc["samples"], friendly_labels, grid_name
    )

    result = {
        "obs_file": os.path.basename(obs_csv),
        "reduced_chi2": reduced_chi2,
        "ppc_coverage": ppc_in,
        "mle_params": mle["grid_params"],
        "mle_all_params": mle.get("all_params", {}),
        "mcmc_summary": mcmc["summary"],
        "mcmc_converged": mcmc["converged"],
        "n_effective": mcmc.get("n_effective"),
        "labels": friendly_labels,
        "exact_inclination": exact_inclination,
        "export_dir": str(artifact_dir),
        "wl_range": [float(wl_range[0]), float(wl_range[1])],
        "emulator_wl_range": list(_emulator_wavelength_bounds(emu)),
        "tier3_time_s": None,
    }

    # Export a Sirocco .pf file from the posterior-mean parameters
    if grid_name is not None:
        _pf_path = artifact_dir / f"tier3_{obs_stem}.pf"

        _grid_means = []
        _uncertainties = {}
        _friendly_grid = internal_to_friendly(emu.param_names, grid_name)
        for _pn, _friendly in zip(emu.param_names, _friendly_grid):
            if _pn not in internal_labels:
                raise ValueError(f"MCMC samples do not contain required grid parameter {_pn}")
            _col = list(internal_labels).index(_pn)
            _grid_means.append(float(np.mean(mcmc["samples"][:, _col])))
            _lo = float(np.percentile(mcmc["samples"][:, _col], 16))
            _hi = float(np.percentile(mcmc["samples"][:, _col], 84))
            _uncertainties[_friendly] = (_lo, _hi)

        _global = {}
        for _i, _label in enumerate(internal_labels):
            if not str(_label).startswith("param"):
                _global[str(_label)] = float(np.mean(mcmc["samples"][:, _i]))

        export_pf_template(
            emu, np.asarray(_grid_means), str(_pf_path),
            uncertainties=_uncertainties,
            global_params=_global,
            grid_name=grid_name,
            observer_angles=[exact_inclination],
        )
        result["pf_path"] = str(_pf_path)
        log.info(f"Tier 3 exported .pf to {_pf_path}")

    sirocco_plot = {"wavelength": [], "flux": []}
    sirocco_reduced_chi2 = np.nan
    emulator_sirocco_frac_rmse = np.nan
    sirocco_transforms = _extract_spectrum_nuisance_transforms(model)
    sirocco_transform_label = _format_sirocco_transform_label(sirocco_transforms)

    if run_sirocco:
        sirocco_meta = run_sirocco_pf(
            result["pf_path"],
            cpus=sirocco_cpus,
            progress_callback=sirocco_progress_callback,
        )
        native_spec_path = Path(sirocco_meta["spec_files"][0])
        reduced_dir = artifact_dir / "reduced_spec"
        from Speculate_addons.lighten_spec_files import reduce_spec_files

        reduced_paths = reduce_spec_files(
            str(artifact_dir),
            output_dir=str(reduced_dir),
            show_progress=False,
            strict=True,
        )
        if not reduced_paths:
            raise FileNotFoundError(f"No reduced .spec files were written in {_repo_relative(reduced_dir)}")
        reduced_spec_path = next(
            (Path(path) for path in reduced_paths if Path(path).name == native_spec_path.name),
            Path(reduced_paths[0]),
        )

        sirocco_wl, sirocco_flux = _load_single_observer_sirocco_spectrum(
            reduced_spec_path, wl_range
        )
        sirocco_flux_plot = _transform_flux_for_scale(
            sirocco_wl,
            sirocco_flux,
            flux_scale,
        )
        sirocco_flux_plot = _apply_spectrum_nuisance_transforms(
            sirocco_wl,
            sirocco_flux_plot,
            flux_scale,
            sirocco_transforms,
        )
        sirocco_plot = {
            "wavelength": sirocco_wl,
            "flux": sirocco_flux_plot,
            "label": sirocco_transform_label,
        }

        _bf_wl = np.asarray(bestfit_spec.get("wavelength", []), dtype=np.float64)
        _bf_data = np.asarray(bestfit_spec.get("data_flux", []), dtype=np.float64)
        _bf_model = np.asarray(bestfit_spec.get("model_flux", []), dtype=np.float64)
        _model_sigma = np.asarray(model.data.sigma, dtype=np.float64)
        if len(_bf_wl) and len(_bf_data) and len(_model_sigma):
            sirocco_interp_plot = np.interp(_bf_wl, sirocco_wl, sirocco_flux_plot)
            _n_sirocco = min(len(_bf_data), len(_model_sigma), len(sirocco_interp_plot))
            n_dof_sirocco = _n_sirocco - len(internal_labels)
            sirocco_reduced_chi2 = float(
                np.sum(
                    ((_bf_data[:_n_sirocco] - sirocco_interp_plot[:_n_sirocco])
                     / np.maximum(_model_sigma[:_n_sirocco], 1e-30)) ** 2
                ) / max(n_dof_sirocco, 1)
            )
        if len(_bf_wl) and len(_bf_data) and len(_bf_model):
            sirocco_interp_plot = np.interp(_bf_wl, sirocco_wl, sirocco_flux_plot)
            denom = np.maximum(np.abs(_bf_data), 1e-30)
            emulator_sirocco_frac_rmse = float(
                np.sqrt(np.nanmean(((_bf_model - sirocco_interp_plot) / denom) ** 2))
            )

        result.update({
            "sirocco_command": sirocco_meta.get("command"),
            "sirocco_log_path": sirocco_meta.get("run_log_path"),
            "sirocco_setup_log_path": sirocco_meta.get("setup_log_path"),
            "sirocco_signal_log_path": sirocco_meta.get("signal_log_path"),
            "sirocco_spec_path": str(native_spec_path),
            "sirocco_reduced_spec_path": str(reduced_spec_path),
            "sirocco_transform_params": sirocco_transforms,
            "sirocco_transform_label": sirocco_transform_label,
        })
    elif require_sirocco:
        raise RuntimeError("Tier 3 was configured to require Sirocco but run_sirocco=False.")

    result["sirocco_reduced_chi2"] = sirocco_reduced_chi2
    result["emulator_sirocco_frac_rmse"] = emulator_sirocco_frac_rmse
    result["sirocco_transform_params"] = sirocco_transforms
    result["sirocco_transform_label"] = sirocco_transform_label

    ppc_envelope = {
        "wavelength": np.asarray(model.data.wave, dtype=np.float64),
        "low": env_lo,
        "high": env_hi,
    }
    artifact_summary = {
        "obs_file": result["obs_file"],
        "labels": friendly_labels,
        "mcmc_summary": mcmc["summary"],
        "mle_params": result["mle_params"],
        "mle_all_params": result.get("mle_all_params", {}),
        "mcmc_converged": result["mcmc_converged"],
        "exact_inclination": exact_inclination,
        "wl_range": result["wl_range"],
        "emulator_wl_range": result["emulator_wl_range"],
        "sirocco_transform_params": sirocco_transforms,
        "sirocco_transform_label": sirocco_transform_label,
        "metrics": {
            "reduced_chi2": reduced_chi2,
            "ppc_coverage": ppc_in,
            "sirocco_reduced_chi2": sirocco_reduced_chi2,
            "emulator_sirocco_frac_rmse": emulator_sirocco_frac_rmse,
        },
    }
    artifacts = _save_tier3_artifacts(
        artifact_dir,
        obs_stem,
        mcmc,
        bestfit_spec,
        ppc_envelope,
        sirocco_plot,
        artifact_summary,
    )
    result["artifacts"] = artifacts
    result["tier3_time_s"] = time.time() - t0

    return result


# ======================================================================
# Export to Sirocco .pf
# ======================================================================


def export_pf_template(
    emu,
    param_values: np.ndarray,
    output_path: str,
    uncertainties: Optional[dict] = None,
    global_params: Optional[dict] = None,
    grid_name: Optional[str] = None,
    observer_angles: Optional[Sequence[float]] = None,
):
    """
    Export a Sirocco .pf file by updating a full template with emulator results.

    Parameters
    ----------
    emu : Emulator
    param_values : ndarray
        Grid parameter values in emulator space (length = n_grid_params).
    output_path : str
        Path to write the .pf file.
    uncertainties : dict or None
        {friendly_name: (lo_1sigma, hi_1sigma)} for annotation.
    global_params : dict or None
        Global params (Av, log_scale, etc.) for annotation.
    grid_name : str or None
        Grid identifier (e.g. ``speculate_cv_bl_grid_v87f``).  Used to select
        the correct Sirocco template.  If None, falls back to a minimal export.
    observer_angles : sequence of float or None
        Optional replacement observer angles for ``Spectrum.angle(0=pole)``.
    """
    from exports.templates.speculate_pf_exporter import write_pf

    if grid_name is None:
        raise ValueError("grid_name is required for Sirocco .pf template export")

    # Convert from emulator coordinates into the exact physical keys expected by
    # the selected Sirocco template.  For AGN this also inverts Eddington-fraction
    # and R_g-scaled axes before the exporter generates QSOSED files.
    physical = emulator_to_physical(emu.param_names, param_values, grid_name)
    defaulted_ids = defaulted_physical_param_ids(grid_name, emu.param_names)
    param_map = benchmark_param_map(grid_name)

    # Build the metadata header (### comments, ignored by Sirocco)
    header = [
        "### Sirocco .pf Template",
        f"### Generated by Speculate",
        f"### Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "###",
    ]

    if uncertainties:
        header.append("### Posterior Uncertainties (1σ):")
        for name, (lo, hi) in uncertainties.items():
            header.append(f"###   {name}: [{lo:.6f}, {hi:.6f}]")
        header.append("###")

    if global_params:
        header.append("### Global/Nuisance Parameters:")
        for k, v in global_params.items():
            header.append(f"###   {k}: {v:.6f}")
        header.append("###")

    if defaulted_ids:
        header.append("### Registry-defaulted physical parameters:")
        for param_id in defaulted_ids:
            _, sirocco_key = param_map.get(param_id, (f"param{param_id}", f"param{param_id}"))
            if sirocco_key in physical:
                header.append(f"###   {sirocco_key}: {physical[sirocco_key]:.6g}")
        header.append("###")

    write_pf(
        grid_name=grid_name,
        physical_params=physical,
        output_path=output_path,
        header_lines=header,
        observer_angles=list(observer_angles) if observer_angles is not None else None,
    )

    log.info(f"Exported .pf template to {output_path}")


def export_posterior_csv(
    samples: np.ndarray,
    labels: List[str],
    output_path: str,
    summary: Optional[dict] = None,
):
    """
    Export MCMC posterior samples and summary statistics to CSV.

    Parameters
    ----------
    samples : ndarray (N, ndim)
    labels : list of str
    output_path : str
    summary : dict or None — if provided, appended as comment header.
    """
    header_lines = ["# Speculate Posterior Samples"]
    if summary:
        header_lines.append("# Summary Statistics:")
        for label, stats_dict in summary.items():
            parts = ", ".join(f"{k}={v:.6f}" for k, v in stats_dict.items())
            header_lines.append(f"#   {label}: {parts}")

    df = pd.DataFrame(samples, columns=labels)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, index=False)

    log.info(f"Exported posterior samples to {output_path}")


def _cornerplot_json_safe(value):
    """Convert numpy/path values into JSON-serialisable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _cornerplot_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cornerplot_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _safe_cornerplot_stem(value, fallback: str) -> str:
    """Return a compact filesystem-safe identifier for one export record."""
    raw = str(value or fallback)
    stem = Path(raw).stem if raw else fallback
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return safe.strip("_") or fallback


def export_cornerplot_data(
    records: Union[dict, Sequence[dict]],
    output_dir: Union[str, Path],
    bundle_name: Optional[str] = None,
    manifest_metadata: Optional[dict] = None,
) -> dict:
    """
    Export raw MCMC samples and metadata needed to recreate corner plots.

    The bundle layout is intentionally simple for downstream plotting tools:
    each record gets a ``samples.csv`` with stable ``param_000`` columns and a
    ``metadata.json`` that maps those columns back to display labels, truth
    values, prior ranges, summaries, and plot settings.  The top-level
    ``manifest.json`` lists every exported record.
    """
    if isinstance(records, dict):
        record_list = [records]
    else:
        record_list = list(records or [])

    if not record_list:
        raise ValueError("No cornerplot records were provided for export.")

    bundle = bundle_name or f"cornerplot_data_{time.strftime('%Y%m%d_%H%M%S')}"
    root = Path(output_dir or ".") / _safe_cornerplot_stem(bundle, "cornerplot_data")
    if root.exists():
        base = root
        suffix = 1
        while root.exists():
            root = Path(f"{base}_{suffix:02d}")
            suffix += 1
    root.mkdir(parents=True, exist_ok=False)

    manifest_records = []
    for idx, record in enumerate(record_list, start=1):
        samples = np.asarray(record.get("samples"))
        if samples.ndim != 2:
            raise ValueError(f"Cornerplot record {idx} samples must be a 2D array.")

        labels = [str(label) for label in record.get("labels", [])]
        if len(labels) != samples.shape[1]:
            raise ValueError(
                f"Cornerplot record {idx} has {samples.shape[1]} sample columns "
                f"but {len(labels)} labels."
            )

        record_id = str(
            record.get("record_id")
            or record.get("id")
            or record.get("filename")
            or record.get("obs_file")
            or f"record_{idx:03d}"
        )
        record_stem = _safe_cornerplot_stem(record_id, f"record_{idx:03d}")
        record_dir = root / f"{idx:03d}_{record_stem}"
        record_dir.mkdir(parents=False, exist_ok=False)

        sample_columns = [f"param_{col:03d}" for col in range(samples.shape[1])]
        samples_path = record_dir / "samples.csv"
        pd.DataFrame(samples, columns=sample_columns).to_csv(
            samples_path,
            index=False,
            float_format="%.17g",
        )

        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"samples", "full_chain"}
        }
        metadata.update({
            "format": "speculate.cornerplot_record.v1",
            "record_index": idx,
            "record_id": record_id,
            "sample_shape": [int(samples.shape[0]), int(samples.shape[1])],
            "sample_columns": [
                {"index": col, "column": column, "label": labels[col]}
                for col, column in enumerate(sample_columns)
            ],
            "labels": labels,
            "samples_file": "samples.csv",
        })

        metadata_path = record_dir / "metadata.json"
        with open(metadata_path, "w") as metadata_file:
            json.dump(_cornerplot_json_safe(metadata), metadata_file, indent=2)

        manifest_records.append({
            "record_index": idx,
            "record_id": record_id,
            "source": record.get("source"),
            "samples_csv": str(samples_path.relative_to(root)),
            "metadata_json": str(metadata_path.relative_to(root)),
            "n_samples": int(samples.shape[0]),
            "n_parameters": int(samples.shape[1]),
            "labels": labels,
        })

    manifest = {
        "format": "speculate.cornerplot_bundle.v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundle_name": root.name,
        "record_count": len(manifest_records),
        "metadata": _cornerplot_json_safe(manifest_metadata or {}),
        "records": manifest_records,
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(_cornerplot_json_safe(manifest), manifest_file, indent=2)

    log.info(f"Exported {len(manifest_records)} cornerplot dataset(s) to {root}")
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "record_count": len(manifest_records),
        "records": manifest_records,
    }


# ======================================================================
# Report Card
# ======================================================================


def build_report_card(
    tier1: Optional[dict] = None,
    tier2: Optional[dict] = None,
    tier3: Optional[list] = None,
    config: Optional[dict] = None,
    tier2_posteriors: Optional[list] = None,
) -> dict:
    """
    Assemble a JSON-serialisable benchmark report card.

    Parameters
    ----------
    tier2_posteriors : list or None
        Per-spectrum posterior dicts (samples, full_chain, labels, summary,
        truths, etc.) to embed in the report so corner plots and chain
        trace plots can be reconstructed when reloading.
    """
    report = {
        "speculate_benchmark_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if config:
        report["config"] = config

    if tier1 is not None:
        # Remove large arrays for the report card, keep scalars
        t1_summary = {}
        for k, v in tier1.items():
            if k == "_arrays":
                # Pass through in-memory arrays (stripped by save_report)
                t1_summary[k] = v
            elif isinstance(v, (int, float, str, bool)):
                t1_summary[k] = v
            elif isinstance(v, list) and len(v) < 100:
                t1_summary[k] = v

        # Serialise per-wavelength RMSE arrays so loaded reports can
        # reconstruct the per-wavelength envelope plot without re-running.
        _t1_arrays = tier1.get("_arrays") or {}
        for _ak in ("pca_per_wl_rmse", "loo_per_wl_rmse", "wavelength"):
            _av = _t1_arrays.get(_ak)
            if _av is not None:
                t1_summary[_ak] = (
                    _av.tolist() if hasattr(_av, "tolist") else list(_av)
                )

        report["tier1"] = t1_summary

    if tier2 is not None:
        t2_summary = {
            "n_spectra": tier2["n_spectra"],
            "n_processed": tier2["n_processed"],
            "n_failures": tier2["n_failures"],
            "n_not_converged": tier2.get("n_not_converged", 0),
            "failure_log": tier2.get("failure_log", []),
            "per_spectrum": tier2.get("per_spectrum", []),
            "tier2_time_s": tier2["tier2_time_s"],
            "mle_config": tier2.get("mle_config", {}),
            "mcmc_config": tier2["mcmc_config"],
            "aggregate": tier2["aggregate"],
        }
        # Embed per-spectrum posteriors so saved reports can reconstruct
        # corner plots and chain trace plots without re-running the benchmark.
        if tier2_posteriors:
            _serialised_posteriors = []
            for _p in tier2_posteriors:
                _entry = {
                    "run": _p["run"],
                    "filename": _p.get("filename", f"run{_p['run']}.spec"),
                    "inclination": _p.get("inclination", 55.0),
                    "labels": _p.get("labels", []),
                    "summary": _p.get("summary", {}),
                    "converged": _p.get("converged", False),
                    "truths": _p.get("truths", {}),
                }
                # Convert numpy arrays to lists for JSON serialisation
                if "samples" in _p:
                    _s = _p["samples"]
                    _entry["samples"] = _s.tolist() if hasattr(_s, "tolist") else _s
                if "full_chain" in _p:
                    _c = _p["full_chain"]
                    _entry["full_chain"] = _c.tolist() if hasattr(_c, "tolist") else _c
                if "burnin_used" in _p:
                    _entry["burnin_used"] = _p["burnin_used"]
                if "bestfit_spec" in _p and _p["bestfit_spec"]:
                    _entry["bestfit_spec"] = _p["bestfit_spec"]
                if "prior_ranges" in _p and _p["prior_ranges"]:
                    _entry["prior_ranges"] = _p["prior_ranges"]
                if "mle_all_params" in _p and _p["mle_all_params"]:
                    _entry["mle_all_params"] = _p["mle_all_params"]
                if "mle_freeze_settings" in _p and _p["mle_freeze_settings"]:
                    _entry["mle_freeze_settings"] = _p["mle_freeze_settings"]
                if "mle_frozen_params" in _p and _p["mle_frozen_params"]:
                    _entry["mle_frozen_params"] = _p["mle_frozen_params"]
                if "mcmc_freeze_settings" in _p and _p["mcmc_freeze_settings"]:
                    _entry["mcmc_freeze_settings"] = _p["mcmc_freeze_settings"]
                if "mcmc_frozen_params" in _p and _p["mcmc_frozen_params"]:
                    _entry["mcmc_frozen_params"] = _p["mcmc_frozen_params"]
                if "mcmc_frozen_param_values" in _p and _p["mcmc_frozen_param_values"]:
                    _entry["mcmc_frozen_param_values"] = _p["mcmc_frozen_param_values"]
                _serialised_posteriors.append(_entry)
            t2_summary["posteriors"] = _serialised_posteriors
        report["tier2"] = t2_summary

    if tier3 is not None:
        t3_summaries = []
        for r in tier3:
            entry = {
                "obs_file": r.get("obs_file", ""),
                "reduced_chi2": r.get("reduced_chi2", None),
                "ppc_coverage": r.get("ppc_coverage", None),
                "mcmc_converged": r.get("mcmc_converged", None),
            }
            for key in (
                "n_effective",
                "exact_inclination",
                "mle_params",
                "mle_all_params",
                "mcmc_summary",
                "labels",
                "wl_range",
                "emulator_wl_range",
                "export_dir",
                "pf_path",
                "sirocco_command",
                "sirocco_log_path",
                "sirocco_setup_log_path",
                "sirocco_signal_log_path",
                "sirocco_spec_path",
                "sirocco_reduced_spec_path",
                "sirocco_transform_params",
                "sirocco_transform_label",
                "sirocco_reduced_chi2",
                "emulator_sirocco_frac_rmse",
                "artifacts",
                "tier3_time_s",
            ):
                if key in r:
                    entry[key] = r[key]
            t3_summaries.append(entry)
        report["tier3"] = t3_summaries

    return report


def save_report(report: dict, output_path: str):
    """Save a report card to JSON, stripping in-memory-only arrays."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Deep-copy top-level and tier dicts so we don't mutate the caller's data
    clean = {}
    for k, v in report.items():
        if isinstance(v, dict) and "_arrays" in v:
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "_arrays"}
        else:
            clean[k] = v
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    log.info(f"Report saved to {output_path}")


def load_report(path: str) -> dict:
    """Load a report card from JSON."""
    with open(path, "r") as f:
        return json.load(f)
