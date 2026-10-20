"""Distance/log-scale conversion helpers for Speculate UI wrappers."""

from __future__ import annotations

import numpy as np
from scipy.stats import rv_continuous, truncnorm


REFERENCE_DISTANCE_PC = 100.0


def validate_distance_pc(distance_pc, name: str = "distance_pc"):
    """Return distance as an array/scalar and require finite positive parsecs."""
    distance = np.asarray(distance_pc, dtype=np.float64)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError(f"{name} must be finite and positive")
    if np.ndim(distance_pc) == 0:
        return float(distance)
    return distance


def distance_to_log_scale(
    distance_pc,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
):
    """Convert distance in parsecs to backend natural-log flux scale."""
    distance = validate_distance_pc(distance_pc)
    reference = validate_distance_pc(reference_distance_pc, "reference_distance_pc")
    value = 2.0 * np.log(reference / distance)
    if np.ndim(distance_pc) == 0:
        return float(value)
    return value


def log_scale_to_distance_pc(
    log_scale,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
):
    """Convert backend natural-log flux scale to distance in parsecs."""
    reference = validate_distance_pc(reference_distance_pc, "reference_distance_pc")
    scale = np.asarray(log_scale, dtype=np.float64)
    distance = reference * np.exp(-0.5 * scale)
    if np.ndim(log_scale) == 0:
        return float(distance)
    return distance


class _DistanceLogScalePrior(rv_continuous):
    """A positive-distance Normal density expressed in backend log_scale.

    For s = 2 ln(reference / d), p(s) = p(d(s)) * d(s) / 2.
    The Jacobian preserves probability in parsecs while sampling in s. Using
    a scipy distribution also supplies interval/rvs through the exact quantiles.
    """

    def _logpdf(self, s, mean_pc, sigma_pc, reference_distance_pc):
        """Return log p(s), including the log(d / 2) change-of-variable term.

        The distance density is Normal with centre mean_pc and width sigma_pc,
        restricted to d > 0. The Jacobian makes it valid for sampling in s.
        """
        log_distance = np.log(reference_distance_pc) - 0.5 * s
        # Extreme proposals have vanishing density; avoid overflow warnings.
        with np.errstate(over="ignore"):
            distance = np.exp(log_distance)
            return (
                truncnorm.logpdf(distance, -mean_pc / sigma_pc, np.inf,
                                 loc=mean_pc, scale=sigma_pc)
                + log_distance - np.log(2.0)
            )

    def _pdf(self, s, mean_pc, sigma_pc, reference_distance_pc):
        """Return the density in log_scale by exponentiating its log density."""
        return np.exp(self._logpdf(s, mean_pc, sigma_pc, reference_distance_pc))

    def _cdf(self, s, mean_pc, sigma_pc, reference_distance_pc):
        """Return P(S <= s), equal to P(D >= d(s)) because distance decreases."""
        with np.errstate(over="ignore"):
            distance = reference_distance_pc * np.exp(-0.5 * s)
        # Distance decreases as s increases, reversing the cumulative tails.
        return truncnorm.sf(distance, -mean_pc / sigma_pc, np.inf,
                            loc=mean_pc, scale=sigma_pc)

    def _ppf(self, q, mean_pc, sigma_pc, reference_distance_pc):
        """Return log_scale quantiles for bounds, initialisation and draws.

        The decreasing transform maps the upper-tail distance quantile to the
        lower-tail log_scale quantile; isf computes that distance directly.
        """
        distance = truncnorm.isf(q, -mean_pc / sigma_pc, np.inf,
                                loc=mean_pc, scale=sigma_pc)
        return 2.0 * (np.log(reference_distance_pc) - np.log(distance))


# Our custom SciPy name, exposed on frozen priors as prior.dist.name.
# Bounds, walker initialisation and plots use it to recognise this prior.
_distance_log_scale = _DistanceLogScalePrior(name="distance_log_scale")


def distance_prior_to_log_scale_prior(
    distance_pc: float,
    distance_sigma_pc: float,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
):
    """Return the exact frozen log_scale prior for a Normal distance, d > 0.

    The inputs are the Normal centre and sigma in parsecs before truncating
    at zero. Unlike the old first-order (loc, scale) approximation, the result
    has a logpdf with the Jacobian and is generally not Normal in log_scale.
    """
    return _distance_log_scale(
        mean_pc=float(validate_distance_pc(distance_pc)),
        sigma_pc=float(validate_distance_pc(distance_sigma_pc, "distance_sigma_pc")),
        reference_distance_pc=float(validate_distance_pc(reference_distance_pc, "reference_distance_pc")),
    )


def distance_to_flux_scale(
    distance_pc,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
):
    """Convert distance in parsecs to multiplicative flux scaling."""
    return np.exp(distance_to_log_scale(distance_pc, reference_distance_pc))
