"""Shared policy helpers for the spectral GP covariance nuisance terms.

The global covariance amplitude is a variance scale in the same transformed
flux units as the data.  These helpers keep benchmark and interactive
inference workflows aligned so both initialise and constrain ``log_amp`` from
the propagated observational uncertainty rather than from model mismatch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


GP_LOG_AMP_PRIOR_SIGMA = 2.5
# Hard optimizer bounds are expressed in units of the prior sigma so that
# user-adjusted prior widths rescale the box constraints consistently.
# At the default sigma=2.5 these multipliers reproduce the original fixed
# offsets of -8 (below) and +4 (above) around the noise-calibrated centre.
GP_LOG_AMP_BOUND_SIGMAS_BELOW = 3.2
GP_LOG_AMP_BOUND_SIGMAS_ABOVE = 1.6
# Absolute tolerance (ln-variance units) for flagging a fitted log_amp as
# pinned against its upper optimizer bound. Bounded optimizers asymptote
# against the box constraint rather than landing exactly on it.
GP_LOG_AMP_BOUND_TOL = 0.05
GP_LOG_AMP_RESIDUAL_WARNING_MARGIN = float(np.log(10.0))
GP_LOG_LS_BOUND_TOL = 1e-6


def get_frozen_dist_loc_scale(dist):
    """Extract ``loc`` and ``scale`` from a scipy frozen distribution."""
    args = dist.args
    kwds = dist.kwds
    if len(args) >= 2:
        return args[0], args[1]
    if len(args) == 1:
        return args[0], kwds.get("scale", 1.0)
    return kwds.get("loc", 0.0), kwds.get("scale", 1.0)


def estimate_log_amp_centre_from_sigma(
    sigma: np.ndarray,
    fallback: float = -55.0,
    decimals: int = 1,
) -> float:
    """Estimate the GP variance centre from propagated data uncertainty.

    ``sigma`` must already be transformed onto the emulator flux scale.  Using
    the observation variance keeps the global covariance prior tied to measured
    uncertainty rather than to mismatch against an arbitrary starting model.
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    sigma_arr = sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0)]
    if sigma_arr.size == 0:
        return round(float(fallback), decimals)

    noise_var = float(np.median(sigma_arr ** 2))
    if not np.isfinite(noise_var) or noise_var <= 0:
        return round(float(fallback), decimals)
    return round(float(np.log(max(noise_var, 1e-30))), decimals)


def log_amp_optimizer_bounds(
    center: float,
    sigma: float = GP_LOG_AMP_PRIOR_SIGMA,
) -> tuple[float, float]:
    """Return conservative MLE bounds around a noise-calibrated ``log_amp``.

    The asymmetric box keeps the fitted GP variance within
    ``exp(+1.6 sigma)`` of the centre (~55x the noise variance at the default
    width) while leaving generous room below. Bounds scale with the prior
    sigma so widening or narrowing the prior in the UI consistently widens or
    narrows the hard constraint as well.
    """
    center = float(center)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = GP_LOG_AMP_PRIOR_SIGMA
    return (
        center - GP_LOG_AMP_BOUND_SIGMAS_BELOW * sigma,
        center + GP_LOG_AMP_BOUND_SIGMAS_ABOVE * sigma,
    )


def bounds_for_frozen_prior(label: str, dist) -> Optional[tuple[float, float]]:
    """Convert a scipy frozen prior to optimizer bounds when possible."""
    loc, scale = get_frozen_dist_loc_scale(dist)
    if dist.dist.name == "uniform":
        return float(loc), float(loc + scale)
    if dist.dist.name == "norm":
        if label == "global_cov:log_amp":
            # Pass the prior's own sigma so user-widened/narrowed priors
            # move the hard box constraints with them.
            return log_amp_optimizer_bounds(float(loc), float(scale))
        return float(loc - 4 * scale), float(loc + 4 * scale)
    return None


def global_covariance_diagnostics(model, priors: Optional[dict], residual: np.ndarray) -> dict:
    """Return JSON-safe warning diagnostics for the global GP covariance."""
    diagnostics = {}
    if "global_cov:log_amp" not in model.params:
        return diagnostics

    log_amp = float(model.params["global_cov:log_amp"])
    residual_var = float(np.mean(np.asarray(residual, dtype=np.float64) ** 2))
    residual_log_var = float(np.log(max(residual_var, 1e-300)))
    log_amp_minus_residual_log_var = float(log_amp - residual_log_var)
    log_amp_above_residual = bool(
        log_amp_minus_residual_log_var > GP_LOG_AMP_RESIDUAL_WARNING_MARGIN
    )

    sigma_arr = np.asarray(model.data.sigma, dtype=np.float64)
    sigma_arr = sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0)]
    noise_var = float(np.median(sigma_arr ** 2)) if sigma_arr.size else None
    noise_log_var = (
        float(np.log(max(noise_var, 1e-300)))
        if noise_var is not None and np.isfinite(noise_var)
        else None
    )

    prior_center = None
    prior_sigma = None
    log_amp_bounds = None
    if priors and "global_cov:log_amp" in priors:
        prior = priors["global_cov:log_amp"]
        if getattr(prior, "dist", None) is not None and prior.dist.name == "norm":
            prior_center, prior_sigma = get_frozen_dist_loc_scale(prior)
            log_amp_bounds = list(
                log_amp_optimizer_bounds(float(prior_center), float(prior_sigma))
            )

    # Pinned-at-bound detection: hitting the upper box constraint is the
    # designed failure mode when the GP tries to absorb model-data mismatch,
    # so surface it explicitly rather than leaving it implicit in the value.
    log_amp_at_upper_bound = bool(
        log_amp_bounds is not None
        and log_amp >= log_amp_bounds[1] - GP_LOG_AMP_BOUND_TOL
    )

    log_ls_at_upper_bound = False
    log_ls_upper_bound = None
    if "global_cov:log_ls" in model.params:
        log_ls = float(model.params["global_cov:log_ls"])
        if priors and "global_cov:log_ls" in priors:
            prior = priors["global_cov:log_ls"]
            if getattr(prior, "dist", None) is not None and prior.dist.name == "uniform":
                loc, scale = get_frozen_dist_loc_scale(prior)
                log_ls_upper_bound = float(loc + scale)
                log_ls_at_upper_bound = bool(
                    log_ls >= log_ls_upper_bound - GP_LOG_LS_BOUND_TOL
                )
        diagnostics["log_ls"] = log_ls

    warning = bool(log_amp_above_residual or log_amp_at_upper_bound or log_ls_at_upper_bound)
    diagnostics.update({
        "warning": warning,
        "log_amp": log_amp,
        "gp_variance": float(np.exp(np.clip(log_amp, -745.0, 709.0))),
        "residual_variance": residual_var,
        "residual_log_variance": residual_log_var,
        "noise_variance_median": noise_var,
        "noise_log_variance_median": noise_log_var,
        "log_amp_minus_residual_log_variance": log_amp_minus_residual_log_var,
        "log_amp_above_residual_variance_10x": log_amp_above_residual,
        "log_amp_at_upper_bound": log_amp_at_upper_bound,
        "log_ls_at_upper_bound": log_ls_at_upper_bound,
        "log_ls_upper_bound": log_ls_upper_bound,
        "log_amp_prior_center": float(prior_center) if prior_center is not None else None,
        "log_amp_prior_sigma": float(prior_sigma) if prior_sigma is not None else None,
        "log_amp_optimizer_bounds": log_amp_bounds,
    })
    return diagnostics