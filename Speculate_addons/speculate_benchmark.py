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
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import cma
from scipy import stats
from scipy.optimize import minimize as scipy_minimize

log = logging.getLogger(__name__)


# ======================================================================
# Parameter Mapping
# ======================================================================
# Maps the 1-based parameter index used in emulator filenames (e.g.
# "param1", "param3") to a (friendly_name, sirocco_keyword) pair.
#
# Indices 1-6 are the Knigge-Wood-Drew (KWD) wind model parameters that
# define the accretion disk and biconical wind geometry in Sirocco.
# Indices 7-8 add a boundary-layer emission component (only present in
# grids labelled "bl").  Indices 9-11 encode the observer inclination
# at progressively finer angular resolution (sparse / mid / full).
#
# In emulator space, parameters 1, 3, and 5 are stored in log10; parameter 2
# is the wind-to-disk mass-loss *ratio* rather than the absolute value.

PARAM_MAP = {
    1: ("disk.mdot", "Disk.mdot(msol/yr)"),          # log10(Msol/yr)
    2: ("wind.mdot", "Wind.mdot(msol/yr)"),          # ratio: wind/disk
    3: ("KWD.d", "KWD.d(in_units_of_rstar)"),        # log10(wind launch radius)
    4: ("KWD.mdot_r_exponent", "KWD.mdot_r_exponent"),  # radial mdot power law
    5: ("KWD.acceleration_length", "KWD.acceleration_length(cm)"),  # log10(cm)
    6: ("KWD.acceleration_exponent", "KWD.acceleration_exponent"),  # velocity law exponent
    7: ("Boundary_layer.luminosity", "Boundary_layer.luminosity(ergs/s)"),
    8: ("Boundary_layer.temp", "Boundary_layer.temp(K)"),
    9: ("Inclination", "Inclination"),   # sparse: 3 angles (30, 55, 80)
    10: ("Inclination", "Inclination"),  # mid:    6 angles
    11: ("Inclination", "Inclination"),  # full:  12 angles
}


def internal_to_friendly(param_names: Sequence[str]) -> List[str]:
    """Map internal ``paramN`` names to human-readable names."""
    out = []
    for pn in param_names:
        idx = int(pn.replace("param", ""))
        friendly, _ = PARAM_MAP.get(idx, (pn, pn))
        out.append(friendly)
    return out


def emulator_to_physical(param_names: Sequence[str], values: np.ndarray) -> dict:
    """
    Convert emulator-space parameter values to physical Sirocco units.

    Returns a dict mapping Sirocco parameter keywords to physical values.
    """
    physical = {}
    vals = np.atleast_1d(values)
    for pn, v in zip(param_names, vals):
        idx = int(pn.replace("param", ""))
        friendly, sirocco_key = PARAM_MAP.get(idx, (pn, pn))
        # The emulator does not store every parameter in native Sirocco units:
        # some axes are log-transformed and param2 is represented as a ratio.
        if idx == 1:
            physical[sirocco_key] = 10 ** v  # log10(Msol/yr) -> Msol/yr
        elif idx == 2:
            # Wind mass loss is inferred as a fraction of disk.mdot and becomes
            # an absolute quantity only after disk.mdot has been converted.
            physical[sirocco_key] = v
        elif idx == 3:
            physical[sirocco_key] = 10 ** v  # log10(R_star) -> R_star
        elif idx == 5:
            physical[sirocco_key] = 10 ** v  # log10(cm) -> cm
        else:
            physical[sirocco_key] = v
    # Compute absolute wind.mdot if both present
    if "Disk.mdot(msol/yr)" in physical and "Wind.mdot(msol/yr)" in physical:
        # Once both pieces exist in physical space, collapse the stored ratio into
        # the absolute wind mass-loss rate expected by downstream Sirocco inputs.
        physical["Wind.mdot(msol/yr)"] = (
            physical["Wind.mdot(msol/yr)"] * physical["Disk.mdot(msol/yr)"]
        )
    return physical


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
        "pca_per_wl_rmse", "loo_per_wl_rmse",
        "loo_flux_rmse", "pca_recon_rmse",
        "loo_recon_var", "loo_mu", "loo_var",
    }
    arrays = {}
    for k in _ARRAY_KEYS:
        if k in results:
            arrays[k] = results.pop(k)
    # Also stash grid_points + param_names for the interactive viewer
    arrays["grid_points"] = emu.grid_points
    arrays["param_names"] = list(emu.param_names)

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
    spec_file: str, inclination: float, wl_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Sirocco ``.spec`` file and extract wavelength, flux, and error arrays.

    Sirocco .spec files have a variable-length text header (comments starting
    with '#' and a column-label row beginning with 'Freq.'), followed by numeric
    data columns.  Column 0 is frequency (Hz), column 1 is wavelength (Å), and
    columns 2+ hold the flux at successive viewing inclinations in 5-degree steps
    starting from 30°.  Wavelengths in the file are descending; this function
    flips them to ascending order.
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
    # Map the requested inclination angle to the correct data column:
    # 30° → col 2, 35° → col 3, 40° → col 4, …
    col = int(2 + (inclination - 30) / 5)
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

    sigma = np.abs(flux) * 0.05  # 5% assumed error
    sigma = np.maximum(sigma, np.abs(flux) * 0.01)
    return wl, flux, sigma


def _extract_ground_truth(
    parquet_path: str, run_idx: int, emu_param_names: Sequence[str]
) -> dict:
    """Extract ground-truth parameters from the test grid's lookup table.

    The lookup table (a parquet file co-located with the .spec files) records the
    Sirocco simulation inputs for every run.  This function reads the raw physical
    values, then applies the same forward transforms used during emulator training
    (log10 for params 1, 3, and 5; wind/disk ratio for param 2) so the returned dict
    lives in emulator space and is directly comparable to inference outputs.
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

    gt = {}
    for pn in emu_param_names:
        idx = int(pn.replace("param", ""))
        friendly, sirocco_key = PARAM_MAP.get(idx, (pn, pn))

        # Try to find the value in the parquet columns
        value = None
        for col in df.columns:
            col_lower = col.lower().replace(" ", "").replace("(", "").replace(")", "")
            key_lower = sirocco_key.lower().replace(" ", "").replace("(", "").replace(")", "")
            if key_lower in col_lower or col_lower in key_lower:
                raw = row[col]

                # Apply forward transforms to match emulator space
                if idx == 1:
                    value = np.log10(raw) if raw > 0 else raw
                elif idx == 2:
                    # Wind.mdot in parquet is absolute (msol/yr)
                    # Need to convert to fraction: wind/disk
                    disk_mdot = None
                    for dc in df.columns:
                        if "disk" in dc.lower() and "mdot" in dc.lower():
                            disk_mdot = row[dc]
                            break
                    value = raw / disk_mdot if disk_mdot and disk_mdot > 0 else raw
                elif idx == 3:
                    value = np.log10(raw) if raw > 0 else raw
                elif idx == 5:
                    value = np.log10(raw) if raw > 0 else raw
                elif idx in (9, 10, 11):
                    # Inclination is in degrees directly — might be labelled differently
                    value = float(raw)
                else:
                    value = float(raw)
                break

        if value is not None:
            gt[friendly] = float(value)

    return gt


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


def run_mle_single(
    emu,
    wl: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    priors: Optional[dict] = None,
    flux_scale: str = "linear",
    max_iter: int = 10_000,
    freeze_av: bool = True,
    iteration_callback=None,
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
        Maximum Nelder-Mead iterations.
    freeze_av : bool
        If True (default), fix Av=0.  Use True for synthetic test-grid
        spectra (Tier 2) which have no dust; False for observational
        spectra (Tier 3) where Av should be optimised.
    iteration_callback : callable or None
        If provided, called as ``iteration_callback(iter_num, max_iter, best_nll, elapsed_s)``
        every 50 iterations during optimisation.

    Returns
    -------
    dict with keys: 'grid_params', 'all_params', 'nll', 'success',
                    'n_iter', 'labels'.
    """
    from Starfish.spectrum import Spectrum
    from Starfish.models import SpectrumModel

    # Match the observation onto the emulator's training scale before building a
    # SpectrumModel, including consistent propagation of the uncertainties.
    if flux_scale == "log":
        sigma = sigma / (np.abs(flux) * np.log(10) + 1e-30)
        flux = np.where(flux > 0, np.log10(flux), np.log10(np.abs(flux) + 1e-30))
    elif flux_scale == "continuum-normalised":
        from Speculate_addons.Spec_functions import fit_power_law_continuum
        continuum, _ = fit_power_law_continuum(wl, flux)
        cont_safe = np.where(continuum > 0, continuum, 1.0)
        sigma = sigma / cont_safe
        flux = flux / cont_safe

    sigma = np.maximum(sigma, np.abs(flux) * 0.01)
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

    # Optionally fix Av = 0 (synthetic test-grid spectra have no dust
    # extinction, so allowing Av to float wastes degrees of freedom and
    # can push the optimizer into poor local minima).
    if freeze_av:
        model.params["Av"] = 0.0
        model.freeze("Av")
    else:
        model.params["Av"] = 0.0

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
        if not freeze_av:
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

    labels_before = list(model.labels)

    # Derive per-parameter bounds from the priors for CMA-ES box constraints.
    active_labels = list(model.labels)
    N = len(active_labels)

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

    def nll(P):
        model.set_param_vector(P)
        try:
            val = -model.log_likelihood(priors)
        except Exception:
            val = 1e10
        _nll_history.append(val)
        _iter_count[0] += 1
        if iteration_callback is not None and _iter_count[0] % 50 == 0:
            _best = min(_nll_history) if _nll_history else val
            iteration_callback(_iter_count[0], max_iter, _best, time.time() - _mle_t0)
        return val

    # CMA-ES: start from the bootstrapped model state (includes the
    # data-informed log_scale) rather than the raw bounds midpoint.
    p0_cma = np.clip(
        model.get_param_vector(),
        np.array(lo_bounds) + 1e-8,
        np.array(hi_bounds) - 1e-8,
    )
    # Per-coordinate initial σ = 20% of prior range.
    cma_stds = [0.2 * (hi - lo) for lo, hi in zip(lo_bounds, hi_bounds)]
    # Double the default popsize for better landscape sampling.
    popsize = 2 * (4 + int(3 * np.log(N)))
    es = cma.CMAEvolutionStrategy(
        p0_cma.tolist(), 1.0,
        {
            "bounds": [lo_bounds, hi_bounds],
            "CMA_stds": cma_stds,
            "popsize": popsize,
            "maxfevals": max_iter,
            "verbose": -9,
            "tolfun": 1e-10,
        },
    )
    best_x, best_f = model.get_param_vector().copy(), float("inf")
    while not es.stop():
        solutions = es.ask()
        fits = [nll(np.array(s)) for s in solutions]
        es.tell(solutions, fits)
        gen_best = min(fits)
        if gen_best < best_f:
            best_f = gen_best
            best_x = np.array(solutions[fits.index(gen_best)])

    from types import SimpleNamespace
    soln = SimpleNamespace(
        x=best_x, fun=best_f, success=True,
        message="CMA-ES terminated",
        nit=es.result.iterations,
    )

    if soln.success:
        model.set_param_vector(soln.x)

    return {
        "grid_params": model.grid_params.tolist(),
        "all_params": {k: float(model.params[k]) for k in model.labels},
        "nll": float(soln.fun),
        "success": bool(soln.success),
        "n_iter": int(soln.nit),
        "labels": labels_before,
        "model": model,
        "priors": priors,
    }


def run_mcmc_single(
    model,
    priors: dict,
    nwalkers: int = 32,
    nsteps: int = 1000,
    burnin: int = 200,
    iteration_callback=None,
    freeze_nuisance: bool = False,
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

    Returns
    -------
    dict with keys:
        'samples'     : (N, ndim) burnt+thinned flat samples
        'summary'     : dict of {label: {mean, std, median, hdi_3, hdi_97}}
        'r_hat'       : dict of {label: r_hat}
        'ess_bulk'    : dict of {label: ess}
        'converged'   : bool — all r_hat < 1.05 and ess > 100
        'n_effective'  : int
    """
    import emcee

    # Optionally freeze nuisance params at MLE best-fit (Czekala+2015 §2.5):
    # "one can first optimize the kernel parameters and then proceed with
    # them fixed, since the stellar parameter posteriors are relatively
    # insensitive to the precise value of the kernel parameters."
    _nuisance_labels = {"log_scale", "cheb:1", "global_cov:log_amp", "global_cov:log_ls"}
    _frozen_nuisance = {}
    if freeze_nuisance:
        for _lbl in list(_nuisance_labels):
            if _lbl in model.labels:
                _frozen_nuisance[_lbl] = float(model.params[_lbl])
                model.freeze(_lbl)

    ndim = len(model.labels)

    # Initialise walkers in a truncated-normal ball around MLE.
    # σ = 2% of prior width (or 1σ for Normal priors).  Truncated normal
    # avoids edge pile-up that np.clip would cause near prior bounds.
    _INIT_FRAC = 0.02
    ball = np.empty((nwalkers, ndim))
    for i, key in enumerate(model.labels):
        pr = priors.get(key)
        mle_val = model[key]
        if pr is not None and hasattr(pr, 'interval'):
            lo, hi = pr.interval(1.0)
            if hasattr(pr, 'std'):
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
        [l for l in model.labels if l.startswith("param")]
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
    # r_hat values must be below 1.05 and there must be enough retained samples
    # to support posterior summaries.
    converged = all(rh < 1.05 for rh in r_hat_dict.values() if np.isfinite(rh))
    converged = converged and flat.shape[0] >= 100

    # Restore any nuisance params that were frozen for this MCMC run so the
    # model object is left in a usable state for downstream callers.
    for _lbl, _val in _frozen_nuisance.items():
        model.thaw(_lbl)
        model[_lbl] = _val

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
        alphas = np.linspace(0.01, 0.99, 50)

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

        aggregate[fname] = {
            "rmse": rmse,
            "bias": bias,
            "crps": crps_mean,
            "shrinkage": shrinkage,
            "coverage_68": cov_68,
            "coverage_95": cov_95,
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
        "mcmc_config": {
            "walkers": mcmc_walkers,
            "steps": mcmc_steps,
            "burnin": mcmc_burnin,
        },
        "tier2_time_s": elapsed,
    }


def run_tier2(
    emu,
    test_grid_path: str,
    flux_scale: str = "linear",
    wl_range: Tuple[float, float] = (850, 1850),
    inclination: float = 55.0,
    mcmc_walkers: int = 32,
    mcmc_steps: int = 1000,
    mcmc_burnin: int = 200,
    max_mle_iter: int = 5000,
    max_spectra: Optional[int] = None,
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

    # Tier 2 is driven directly from the decompressed test-grid spectra, with the
    # optional max_spectra cap used to keep exploratory runs tractable in the UI.
    spec_files = sorted(test_path.glob("run*.spec"))
    if max_spectra is not None:
        spec_files = spec_files[:max_spectra]

    if not spec_files:
        raise FileNotFoundError(f"No .spec files found in {test_grid_path}")

    # The lookup parquet links each run file back to its known simulation inputs.
    parquet_file = ensure_lookup_table(test_path)

    friendly_names = internal_to_friendly(emu.param_names)
    n_params = len(emu.param_names)

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
            wl, flux, sigma = _load_test_grid_spectrum(str(sf), inclination, wl_range)
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
                gt = _extract_ground_truth(str(parquet_file), run_idx, emu.param_names)
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
                emu, wl, flux, sigma, flux_scale=flux_scale, max_iter=max_mle_iter
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
                freeze_nuisance=True,
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

        aggregate[fname] = {
            "rmse": rmse,
            "bias": bias,
            "crps": crps_mean,
            "shrinkage": shrinkage,
            "coverage_68": cov_68,
            "coverage_95": cov_95,
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
        "mcmc_config": {
            "walkers": mcmc_walkers,
            "steps": mcmc_steps,
            "burnin": mcmc_burnin,
        },
        "tier2_time_s": elapsed,
    }


# ======================================================================
# Tier 3 — Observational Spectra
# ======================================================================


def run_tier3_single(
    emu,
    obs_csv: str,
    flux_scale: str = "linear",
    wl_range: Tuple[float, float] = (850, 1850),
    max_mle_iter: int = 5000,
    n_ppc_draws: int = 100,
    mcmc_walkers: int = 32,
    mcmc_steps: int = 1000,
    mcmc_burnin: int = 200,
    grid_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Tier 3 benchmark: goodness-of-fit for a single observational spectrum.

    Parameters
    ----------
    grid_name : str or None
        Grid identifier (e.g. ``speculate_cv_bl_grid_v87f``) used to select
        the correct Sirocco template for .pf export.  If None, .pf export
        is skipped.
    output_dir : str or None
        Directory for exported .pf files.  Defaults to ``exports/``.

    Returns
    -------
    dict with keys: 'reduced_chi2', 'ppc_coverage', 'mle_params',
                    'mcmc_summary', 'obs_file', and optionally 'pf_path'.
    """
    from Starfish.spectrum import Spectrum
    from Starfish.models import SpectrumModel

    # Load observation
    df = pd.read_csv(obs_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    wl = np.array(df["wavelength"])
    flux = np.array(df["flux"])
    sigma = np.array(df["error"]) if "error" in df.columns else np.abs(flux) * 0.05

    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    wl, flux, sigma = wl[mask], flux[mask], sigma[mask]

    # MLE — run_mle_single handles flux_scale transforms internally;
    # applying them here as well would double-scale the data.
    mle = run_mle_single(emu, wl, flux, sigma, flux_scale=flux_scale,
                          max_iter=max_mle_iter, freeze_av=False)
    model = mle["model"]
    priors = mle["priors"]

    # Reduced chi^2
    try:
        model_flux, model_cov = model()
        residuals = flux - model_flux
        n_dof = len(flux) - len(model.labels)
        chi2 = float(np.sum((residuals / sigma) ** 2))
        reduced_chi2 = chi2 / max(n_dof, 1)
    except Exception:
        reduced_chi2 = np.nan

    # MCMC
    try:
        mcmc = run_mcmc_single(
            model, priors,
            nwalkers=mcmc_walkers, nsteps=mcmc_steps, burnin=mcmc_burnin,
            freeze_nuisance=True,
        )
    except Exception as e:
        log.warning(f"MCMC failed for {obs_csv}: {e}")
        return {
            "obs_file": os.path.basename(obs_csv),
            "reduced_chi2": reduced_chi2,
            "ppc_coverage": np.nan,
            "mle_params": mle["grid_params"],
        }

    # Posterior Predictive Check (PPC)
    # Draw random posterior samples, evaluate the model flux at each one, and
    # check what fraction of the observed data falls within the 2.5–97.5%
    # envelope of the predicted flux.  A well-calibrated model should cover
    # ~95% of the data points.
    ppc_in = 0
    n_ppc = min(n_ppc_draws, mcmc["samples"].shape[0])
    indices = np.random.choice(mcmc["samples"].shape[0], size=n_ppc, replace=False)
    envelopes = np.zeros((n_ppc, len(flux)))

    _mcmc_labels = mcmc["labels"]
    for j, idx in enumerate(indices):
        _sample_dict = dict(zip(_mcmc_labels, mcmc["samples"][idx]))
        model.set_param_dict(_sample_dict)
        try:
            pred, _ = model()
            envelopes[j] = pred
        except Exception:
            envelopes[j] = np.nan

    valid = ~np.any(np.isnan(envelopes), axis=1)
    if valid.sum() > 1:
        env_lo = np.percentile(envelopes[valid], 2.5, axis=0)
        env_hi = np.percentile(envelopes[valid], 97.5, axis=0)
        ppc_in = float(np.mean((flux >= env_lo) & (flux <= env_hi)))
    else:
        ppc_in = np.nan

    # Set model back to posterior mean for export
    mcmc_means = {}
    for i, label in enumerate(mcmc["labels"]):
        mcmc_means[label] = float(np.mean(mcmc["samples"][:, i]))
    model.set_param_dict(mcmc_means)

    result = {
        "obs_file": os.path.basename(obs_csv),
        "reduced_chi2": reduced_chi2,
        "ppc_coverage": ppc_in,
        "mle_params": mle["grid_params"],
        "mcmc_summary": mcmc["summary"],
        "mcmc_converged": mcmc["converged"],
        "model": model,
        "samples": mcmc["samples"],
        "labels": mcmc["labels"],
    }

    # Export a Sirocco .pf file from the posterior-mean parameters
    if grid_name is not None:
        try:
            _out = output_dir or "exports"
            _obs_stem = os.path.splitext(os.path.basename(obs_csv))[0]
            _pf_path = os.path.join(_out, f"tier3_{_obs_stem}.pf")

            _n_grid = len(emu.param_names)
            _grid_means = np.mean(mcmc["samples"][:, :_n_grid], axis=0)

            _uncertainties = {}
            _friendly = internal_to_friendly(emu.param_names)
            for _i, _label in enumerate(_friendly):
                _lo = np.percentile(mcmc["samples"][:, _i], 16)
                _hi = np.percentile(mcmc["samples"][:, _i], 84)
                _uncertainties[_label] = (_lo, _hi)

            _global = {}
            for _i in range(_n_grid, mcmc["samples"].shape[1]):
                _global[mcmc["labels"][_i]] = float(
                    np.mean(mcmc["samples"][:, _i])
                )

            export_pf_template(
                emu, _grid_means, _pf_path,
                uncertainties=_uncertainties,
                global_params=_global,
                grid_name=grid_name,
            )
            result["pf_path"] = _pf_path
            log.info(f"Tier 3 exported .pf to {_pf_path}")
        except Exception as e:
            log.warning(f"Tier 3 .pf export failed for {obs_csv}: {e}")

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
    """
    from exports.templates.speculate_pf_exporter import write_pf

    physical = emulator_to_physical(emu.param_names, param_values)

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

    write_pf(
        grid_name=grid_name,
        physical_params=physical,
        output_path=output_path,
        header_lines=header,
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
        "speculate_benchmark_version": "1.0.0",
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
            if "pf_path" in r:
                entry["pf_path"] = r["pf_path"]
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
