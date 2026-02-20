"""
Speculate Benchmark Module
===========================
Core functions for evaluating emulator performance across three tiers:

    Tier 1 — Grid Reconstruction Fidelity (PCA + LOO-CV)
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

PARAM_MAP = {
    1: ("disk.mdot", "Disk.mdot(msol/yr)"),
    2: ("wind.mdot", "Wind.mdot(msol/yr)"),
    3: ("KWD.d", "KWD.d(in_units_of_rstar)"),
    4: ("KWD.mdot_r_exponent", "KWD.mdot_r_exponent"),
    5: ("KWD.acceleration_length", "KWD.acceleration_length(cm)"),
    6: ("KWD.acceleration_exponent", "KWD.acceleration_exponent"),
    7: ("Boundary_layer.luminosity", "Boundary_layer.luminosity(ergs/s)"),
    8: ("Boundary_layer.temp", "Boundary_layer.temp(K)"),
    9: ("Inclination", "Inclination"),
    10: ("Inclination", "Inclination"),
    11: ("Inclination", "Inclination"),
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
        if idx == 1:
            physical[sirocco_key] = 10 ** v  # log10(Msol/yr) -> Msol/yr
        elif idx == 2:
            physical[sirocco_key] = v  # fraction; absolute needs disk.mdot
        elif idx == 5:
            physical[sirocco_key] = 10 ** v  # log10(cm) -> cm
        else:
            physical[sirocco_key] = v
    # Compute absolute wind.mdot if both present
    if "Disk.mdot(msol/yr)" in physical and "Wind.mdot(msol/yr)" in physical:
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
        computed in addition to weight-space LOO metrics.

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
    """Load a ``.spec`` file and extract wavelength, flux, and error arrays."""
    # Determine header lines
    skiprows = 0
    with open(spec_file, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Freq."):
                skiprows = i + 1
            else:
                break

    data = np.loadtxt(spec_file, skiprows=skiprows)
    wl = np.flip(data[:, 0])  # Freq → Wavelength
    # Inclination → column: 30°=col2, 35°=col3, ...
    col = int(2 + (inclination - 30) / 5)
    flux = np.flip(data[:, col])

    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    wl = wl[mask]
    flux = flux[mask]
    sigma = np.abs(flux) * 0.05  # 5% assumed error
    sigma = np.maximum(sigma, np.abs(flux) * 0.01)
    return wl, flux, sigma


def _extract_ground_truth(
    parquet_path: str, run_idx: int, emu_param_names: Sequence[str]
) -> dict:
    """Extract ground truth parameters from the test grid lookup table."""
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


def run_mle_single(
    emu,
    wl: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    priors: Optional[dict] = None,
    flux_scale: str = "linear",
    max_iter: int = 5000,
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

    Returns
    -------
    dict with keys: 'grid_params', 'all_params', 'nll', 'success',
                    'n_iter', 'labels'.
    """
    from Starfish.spectrum import Spectrum
    from Starfish.models import SpectrumModel

    # Transform flux if needed
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
        global_cov={"log_amp": -36.0, "log_ls": 4.0},
    )

    # Build default priors from emulator bounds
    if priors is None:
        priors = {}
        for i, pn in enumerate(emu.param_names):
            lo, hi = float(emu.min_params[i]), float(emu.max_params[i])
            margin = (hi - lo) * 0.05
            priors[pn] = stats.uniform(loc=lo - margin, scale=(hi - lo) + 2 * margin)
        priors["Av"] = stats.uniform(loc=0, scale=2.0)
        priors["log_scale"] = stats.uniform(loc=-80, scale=120)
        priors["global_cov:log_amp"] = stats.norm(loc=-36.0, scale=5.0)
        priors["global_cov:log_ls"] = stats.uniform(loc=0, scale=12)

    # Bootstrap log_scale
    try:
        _ = model()
        if model._log_scale is not None and np.isfinite(model._log_scale):
            model.params["log_scale"] = model._log_scale
    except Exception:
        model.params["log_scale"] = 0.0

    labels_before = list(model.labels)

    def nll(P):
        model.set_param_vector(P)
        return -model.log_likelihood(priors)

    p0 = model.get_param_vector()
    soln = scipy_minimize(
        nll,
        p0,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "adaptive": True},
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
) -> dict:
    """
    Run MCMC on a SpectrumModel already set to MLE best-fit.

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
        return model.log_likelihood(priors)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(ball, nsteps, progress=False)

    # Autocorrelation & burn-in
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

    # Per-chain means for r_hat
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

        # Simple r_hat (between-chain vs within-chain variance)
        chain_means = np.array([np.mean(per_chain[w, :, i]) for w in range(nwalkers)])
        chain_vars = np.array([np.var(per_chain[w, :, i]) for w in range(nwalkers)])
        W = np.mean(chain_vars)
        B = np.var(chain_means) * chain.shape[0]
        var_est = (1 - 1.0 / chain.shape[0]) * W + B / chain.shape[0]
        r_hat = np.sqrt(var_est / W) if W > 0 else np.nan
        r_hat_dict[label] = float(r_hat)

        # Effective sample size (simple estimate)
        ess_dict[label] = int(flat.shape[0])  # conservative: total thinned samples

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

    Returns
    -------
    dict with calibration metrics, per-spectrum results, and aggregate scores.
    """
    t0 = time.time()
    test_path = Path(test_grid_path)

    # Find spec files
    spec_files = sorted(test_path.glob("run*.spec"))
    if max_spectra is not None:
        spec_files = spec_files[:max_spectra]

    if not spec_files:
        raise FileNotFoundError(f"No .spec files found in {test_grid_path}")

    # Ground truth
    parquet_file = test_path / "grid_run_lookup_table.parquet"
    has_gt = parquet_file.exists()

    friendly_names = internal_to_friendly(emu.param_names)
    n_params = len(emu.param_names)

    per_spectrum = []
    all_samples = {name: [] for name in friendly_names}
    all_truths = {name: [] for name in friendly_names}
    failures = 0

    for sf in spec_files:
        run_idx = int(sf.stem.replace("run", ""))
        log.info(f"Processing {sf.name}...")

        try:
            wl, flux, sigma = _load_test_grid_spectrum(str(sf), inclination, wl_range)
        except Exception as e:
            log.warning(f"Failed to load {sf.name}: {e}")
            failures += 1
            continue

        # Ground truth
        gt = {}
        if has_gt:
            try:
                gt = _extract_ground_truth(str(parquet_file), run_idx, emu.param_names)
            except Exception as e:
                log.warning(f"Failed to extract GT for {sf.name}: {e}")

        # MLE
        try:
            mle_result = run_mle_single(
                emu, wl, flux, sigma, flux_scale=flux_scale, max_iter=max_mle_iter
            )
        except Exception as e:
            log.warning(f"MLE failed for {sf.name}: {e}")
            failures += 1
            continue

        # MCMC
        try:
            mcmc_result = run_mcmc_single(
                mle_result["model"],
                mle_result["priors"],
                nwalkers=mcmc_walkers,
                nsteps=mcmc_steps,
                burnin=mcmc_burnin,
            )
        except Exception as e:
            log.warning(f"MCMC failed for {sf.name}: {e}")
            failures += 1
            continue

        if not mcmc_result["converged"]:
            failures += 1

        # Store per-parameter samples and truths
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

    # Aggregate metrics
    aggregate = {}
    for fname in friendly_names:
        if len(all_truths[fname]) == 0:
            continue

        truths_arr = np.array(all_truths[fname])
        means_arr = np.array(
            [ps.get(f"{fname}_mean", np.nan) for ps in per_spectrum]
        )
        means_arr = means_arr[~np.isnan(means_arr)]

        # RMSE of posterior mean
        if len(means_arr) == len(truths_arr):
            rmse = float(np.sqrt(np.mean((means_arr - truths_arr) ** 2)))
            bias = float(np.mean(means_arr - truths_arr))
        else:
            rmse = np.nan
            bias = np.nan

        # CRPS
        crps_vals = []
        for samples, truth in zip(all_samples[fname], all_truths[fname]):
            crps_vals.append(compute_crps(samples, truth))
        crps_mean = float(np.mean(crps_vals)) if crps_vals else np.nan

        # Posterior shrinkage
        prior_range = float(emu.max_params[friendly_names.index(fname)] - emu.min_params[friendly_names.index(fname)]) if fname in friendly_names[:n_params] else np.nan
        post_stds = [ps.get(f"{fname}_std", np.nan) for ps in per_spectrum]
        mean_post_std = float(np.nanmean(post_stds))
        shrinkage = 1.0 - mean_post_std / (prior_range / np.sqrt(12)) if np.isfinite(prior_range) and prior_range > 0 else np.nan

        # Coverage
        alphas, cov = compute_coverage(all_samples[fname], all_truths[fname])
        # Extract coverage at 68% and 95%
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
    mle = run_mle_single(emu, wl, flux, sigma, flux_scale=flux_scale, max_iter=max_mle_iter)
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

    # Posterior Predictive Check
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
