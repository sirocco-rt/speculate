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
# In emulator space, parameters 1 and 5 are stored in log10; parameter 2
# is the wind-to-disk mass-loss *ratio* rather than the absolute value.

PARAM_MAP = {
    1: ("disk.mdot", "Disk.mdot(msol/yr)"),          # log10(Msol/yr)
    2: ("wind.mdot", "Wind.mdot(msol/yr)"),          # ratio: wind/disk
    3: ("KWD.d", "KWD.d(in_units_of_rstar)"),        # wind launch radius
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
    results["pca_explained_variance"] = float(
        1.0 - np.sum(results["loo_mse_per_comp"])
        / np.sum(np.var(emu.weights, axis=0))
    )
    results["n_components"] = emu.ncomps
    results["n_grid_points"] = emu.grid_points.shape[0]
    results["n_params"] = emu.grid_points.shape[1]
    results["tier1_time_s"] = time.time() - t0

    # Emulator Accuracy Score (EAS): single 0–100% summary metric.
    # Combines PCA explained variance (how well n_components capture the grid
    # variance) with Leave-One-Out flux reconstruction accuracy (how well the GP
    # interpolates unseen training points in flux space).  100% = perfect.
    # Both components are clamped to [0, 1]: a negative pca_explained_variance
    # (Leave-One-Out weight-MSE > weight variance — GP worse than mean) scores 0, not
    # a negative EAS which would be misleading.
    # Only computable when grid_path is provided (flux-space metrics present).
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
    _ARRAY_KEYS = {"original_flux", "pca_recon_flux", "loo_recon_flux", "wavelength"}
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
    (log10 for params 1 and 5; wind/disk ratio for param 2) so the returned dict
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
        'linear', 'log', or 'scaled'.
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
        flux = np.where(flux > 0, np.log10(flux), np.log10(np.abs(flux) + 1e-30))
        sigma = sigma / (np.abs(flux) * np.log(10) + 1e-30)
    elif flux_scale == "scaled":
        fmean = np.mean(flux)
        if fmean != 0:
            sigma = sigma / fmean
            flux = flux / fmean

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
    )

    # Optionally fix Av = 0 (synthetic test-grid spectra have no dust
    # extinction, so allowing Av to float wastes degrees of freedom and
    # can push the optimizer into poor local minima).
    if freeze_av:
        model.params["Av"] = 0.0
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
        if not freeze_av:
            priors["Av"] = stats.uniform(loc=0, scale=2.0)
        # Keep log_scale centered on the bootstrap estimate so the optimizer does
        # not waste its early iterations finding the gross normalization.
        priors["log_scale"] = stats.uniform(
            loc=_bootstrapped_ls - 5.0, scale=10.0
        )
        # GP log_amp: normal prior centred on -55 with σ=2.5 (±2σ = [-60, -50])
        priors["global_cov:log_amp"] = stats.norm(loc=-55.0, scale=2.5)
        # GP log_ls: [1, 8] — matches inference tool optimised range
        priors["global_cov:log_ls"] = stats.uniform(loc=1.0, scale=7.0)

    labels_before = list(model.labels)

    # Build a simplex whose vertices span the prior support of every active
    # parameter. This is much more stable than starting Nelder-Mead from a single
    # midpoint in this highly coupled likelihood surface.
    active_labels = list(model.labels)
    N = len(active_labels)
    simplex = np.zeros((N + 1, N))

    for col_idx, label in enumerate(active_labels):
        if label in priors:
            dist = priors[label]
            loc, sc = _get_loc_scale(dist)
            if dist.dist.name == "uniform":
                # Spread the simplex across the truncated support so each row sees
                # a different region of the allowed parameter interval.
                col = _simplex_column_uniform(loc, sc, N)
            elif dist.dist.name == "norm":
                # For normal priors, approximate the interesting support as ±2σ.
                col = _simplex_column_norm(loc, sc, N)
            else:
                # Unknown prior families fall back to a small perturbation around
                # the current parameter vector instead of failing outright.
                cv = model.get_param_vector()[col_idx]
                col = [cv + (cv * 0.01 * k) for k in range(N + 1)]
        else:
            cv = model.get_param_vector()[col_idx]
            col = [cv] * (N + 1)

        simplex[:, col_idx] = col
        # Rotate each column so simplex vertices are not all aligned on the same
        # coordinate-wise corner of the prior hyper-rectangle.
        simplex[:, col_idx] = np.roll(simplex[:, col_idx], col_idx)

    # Bootstrap log_scale at every simplex vertex so the optimizer starts
    # with a data-informed scale at each point in parameter space.
    if "log_scale" in active_labels:
        ls_idx = active_labels.index("log_scale")
        for row in range(N + 1):
            try:
                model.set_param_vector(simplex[row])
                _ = model()  # triggers auto log_scale calc
                if model._log_scale is not None and np.isfinite(model._log_scale):
                    simplex[row, ls_idx] = model._log_scale
            except Exception:
                pass  # keep the prior-based value

    # Restore model to the centroid of the simplex
    centroid = simplex.mean(axis=0)
    model.set_param_vector(centroid)

    _nll_history = []
    _iter_count = [0]
    _mle_t0 = time.time()

    def nll(P):
        model.set_param_vector(P)
        # Penalise out-of-bounds grid params so Nelder-Mead steers back in range
        # rather than propagating an exception and aborting the whole run.
        gp = np.array(model.grid_params)
        violation = (
            np.sum(np.maximum(0.0, emu.min_params - gp) ** 2)
            + np.sum(np.maximum(0.0, gp - emu.max_params) ** 2)
        )
        if violation > 0.0:
            return 1e10 * (1.0 + violation)
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

    p0 = model.get_param_vector()
    soln = scipy_minimize(
        nll,
        p0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "adaptive": True,
            "initial_simplex": simplex,
        },
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
) -> dict:
    """
    Run MCMC on a SpectrumModel already set to MLE best-fit.

    Parameters
    ----------
    iteration_callback : callable or None
        If provided, called as ``iteration_callback(step, nsteps, elapsed_s)``
        every 50 steps.

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

    ndim = len(model.labels)

    # Initialise walkers around MLE
    ball = np.random.randn(nwalkers, ndim)
    for i, key in enumerate(model.labels):
        ball[:, i] *= 0.1
        ball[:, i] += model[key]

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

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

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
        auto_burnin = int(tau.max())
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

    return {
        "samples": flat,
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
            "mle_success": mle_result["success"],
            "mcmc_converged": mcmc_result["converged"],
            "n_effective": mcmc_result["n_effective"],
            "mle_grid_params": mle_result["grid_params"],
        }

        for i, fname in enumerate(friendly_names):
            if fname in mcmc_result["summary"]:
                spec_result[f"{fname}_mean"] = mcmc_result["summary"][fname]["mean"]
                spec_result[f"{fname}_std"] = mcmc_result["summary"][fname]["std"]

                samples_i = mcmc_result["samples"][:, i]
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
) -> dict:
    """
    Tier 3 benchmark: goodness-of-fit for a single observational spectrum.

    Returns
    -------
    dict with keys: 'reduced_chi2', 'ppc_coverage', 'mle_params',
                    'mcmc_summary', 'obs_file'.
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

    if flux_scale == "log":
        sigma = sigma / (np.abs(flux) * np.log(10) + 1e-30)
        flux = np.where(flux > 0, np.log10(flux), np.log10(np.abs(flux) + 1e-30))
    elif flux_scale == "scaled":
        fmean = np.mean(flux)
        if fmean != 0:
            sigma = sigma / fmean
            flux = flux / fmean

    sigma = np.maximum(sigma, np.abs(flux) * 0.01)

    # MLE
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

    for j, idx in enumerate(indices):
        model.set_param_vector(mcmc["samples"][idx])
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
    for i, label in enumerate(model.labels):
        mcmc_means[label] = float(np.mean(mcmc["samples"][:, i]))
    model.set_param_dict(mcmc_means)

    return {
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


# ======================================================================
# Export to Sirocco .pf
# ======================================================================


def export_pf_template(
    emu,
    param_values: np.ndarray,
    output_path: str,
    uncertainties: Optional[dict] = None,
    global_params: Optional[dict] = None,
):
    """
    Export a Sirocco .pf file template from emulator parameters.

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
    """
    physical = emulator_to_physical(emu.param_names, param_values)

    lines = [
        "### Sirocco .pf Template",
        f"### Generated by Speculate Benchmark Suite",
        f"### Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "###",
    ]

    if uncertainties:
        lines.append("### Posterior Uncertainties (1σ):")
        for name, (lo, hi) in uncertainties.items():
            lines.append(f"###   {name}: [{lo:.6f}, {hi:.6f}]")
        lines.append("###")

    if global_params:
        lines.append("### Global/Nuisance Parameters:")
        for k, v in global_params.items():
            lines.append(f"###   {k}: {v:.6f}")
        lines.append("###")

    lines.append("")
    lines.append("### Wind and Disk Parameters")

    for key, value in physical.items():
        # Format scientifically for very large/small values
        if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
            lines.append(f"{key:<45s} {value:.6e}")
        else:
            lines.append(f"{key:<45s} {value:.6f}")

    lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

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
) -> dict:
    """
    Assemble a JSON-serialisable benchmark report card.
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
        report["tier2"] = t2_summary

    if tier3 is not None:
        t3_summaries = []
        for r in tier3:
            t3_summaries.append({
                "obs_file": r.get("obs_file", ""),
                "reduced_chi2": r.get("reduced_chi2", None),
                "ppc_coverage": r.get("ppc_coverage", None),
                "mcmc_converged": r.get("mcmc_converged", None),
            })
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
