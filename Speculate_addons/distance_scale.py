"""Distance/log-scale conversion helpers for Speculate UI wrappers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


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


def distance_sigma_to_log_scale_sigma(
    distance_pc: float,
    distance_sigma_pc: float,
) -> float:
    """Approximate a small Gaussian distance uncertainty in log-scale space."""
    distance = validate_distance_pc(distance_pc)
    sigma = validate_distance_pc(distance_sigma_pc, "distance_sigma_pc")
    return float(2.0 * sigma / distance)


def distance_prior_to_log_scale_prior(
    distance_pc: float,
    distance_sigma_pc: float,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
) -> Tuple[float, float]:
    """Return normal-prior ``(loc, scale)`` for backend ``log_scale``."""
    loc = distance_to_log_scale(distance_pc, reference_distance_pc)
    scale = distance_sigma_to_log_scale_sigma(distance_pc, distance_sigma_pc)
    return float(loc), float(scale)


def distance_to_flux_scale(
    distance_pc,
    reference_distance_pc: float = REFERENCE_DISTANCE_PC,
):
    """Convert distance in parsecs to multiplicative flux scaling."""
    return np.exp(distance_to_log_scale(distance_pc, reference_distance_pc))