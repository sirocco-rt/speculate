import logging

import numpy as np
from scipy.optimize import minimize
import scipy.stats as st

import Starfish.constants as C
from .kernels import global_covariance_matrix


def _covariance_diagonal(covariance, expected_size):
    """Return a validated NumPy diagonal from NumPy or Torch covariance data."""
    if hasattr(covariance, "detach"):
        covariance = covariance.detach().cpu().numpy()
    covariance = np.asarray(covariance, dtype=float)
    diagonal = (
        np.diag(covariance)
        if covariance.ndim == 2
        else covariance.reshape(-1)
    )
    if diagonal.size != int(expected_size):
        raise ValueError(
            "Covariance diagonal length does not match the residual spectrum"
        )
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("Covariance diagonal contains non-finite values")
    if np.any(diagonal < 0):
        raise ValueError("Covariance diagonal contains negative variances")
    return diagonal


def find_residual_peaks(
    model,
    num_residuals=100,
    threshold=4.0,
    buffer=2,
    wl_range=(0, np.inf),
    grow_threshold=None,
    return_regions=False,
    covariance=None,
):
    """
    Find coherent regions in the recent residuals for local-kernel placement.

    A region is seeded by at least one pixel above ``threshold`` and then grown
    through adjacent, same-sign pixels above ``grow_threshold``. This prevents
    several above-threshold pixels from the same spectral feature becoming
    independent candidates that later optimise onto the same wavelength.

    Parameters
    ----------
    model : Model
        The model to determine peaks from. Need only have a residuals array.
    num_residuals : int, optional
        The number of residuals to average together for determining peaks. By default
        100.
    threshold : float, optional
        The sigma clipping threshold, by default 4.0
    buffer : float, optional
        Merge same-sign residual regions separated by no more than this many
        Angstrom, by default 2.0.
    wl_range : 2-tuple
        The (min, max) wavelengths to consider. Default is (0, np.inf)
    grow_threshold : float, optional
        Lower sigma threshold used to grow a region around a seed. Defaults to
        half of ``threshold``.
    return_regions : bool, optional
        If True, return dictionaries containing the centre, wavelength bounds,
        residual sign, estimated velocity width, and peak score. The default
        False preserves the historical list-of-centres return value.
    covariance : array-like or None, optional
        Complete pre-local covariance matrix (or its diagonal). When supplied,
        residual significance uses ``sqrt(diag(covariance))`` so observational,
        emulator, and global-GP uncertainty are all included consistently.
        The historical robust residual-scale calculation is retained when this
        argument is omitted.

    Returns
    -------
    list
        Region dictionaries when ``return_regions`` is True; otherwise their
        centre wavelengths in the same units as ``model.data.wave``.
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)
    wave = np.asarray(model.data.wave, dtype=float)

    residual = np.asarray(residual, dtype=float)
    centered_residual = residual - np.median(residual)
    if covariance is not None:
        # A wavelength-local outlier score needs one uncertainty per pixel.
        # Use the diagonal of the *complete* covariance before local kernels;
        # off-diagonal correlations remain in the likelihood itself.
        base_variance = _covariance_diagonal(covariance, len(wave))
        sigma = np.sqrt(
            np.maximum(base_variance, np.finfo(float).tiny)
        )
    else:
        # Backwards-compatible fallback for callers without a model covariance.
        # A median/MAD baseline is not inflated by the very outlier lines that
        # the local kernels are intended to identify.
        sigma = 1.4826 * np.median(np.abs(centered_residual))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = centered_residual.std()
        if not np.isfinite(sigma) or sigma <= 0:
            return []

        if "global_cov" in model.params:
            ag = np.exp(model.params["global_cov:log_amp"])
            lg = np.exp(model.params["global_cov:log_ls"])
            # ``sigma`` is a flux standard deviation, while a covariance
            # diagonal is a variance. Combine independent variances before
            # returning to standard-deviation units.
            global_variance = global_covariance_matrix(
                wave, ag, lg
            ).diagonal()
            sigma = np.sqrt(sigma**2 + global_variance)

    # ``sigma`` may be a scalar or a wavelength-dependent array after adding
    # the global covariance diagonal. In either case this produces a
    # dimensionless significance score for region detection.
    score = np.abs(centered_residual) / np.maximum(sigma, np.finfo(float).tiny)
    grow_threshold = (
        0.5 * float(threshold)
        if grow_threshold is None else float(grow_threshold)
    )
    valid_wavelength = (wave > wl_range[0]) & (wave < wl_range[1])
    seed_mask = (score > threshold) & valid_wavelength
    grow_mask = (score > grow_threshold) & valid_wavelength
    # Match the historical helper's exclusion of the two spectrum endpoints,
    # where incomplete line coverage can otherwise create edge candidates.
    seed_mask[[0, -1]] = False
    grow_mask[[0, -1]] = False

    positive_steps = np.diff(wave)
    positive_steps = positive_steps[positive_steps > 0]
    pixel_step = (
        float(np.median(positive_steps)) if positive_steps.size else 0.0
    )
    merge_gap = max(float(buffer), 2.0 * pixel_step)
    regions = []

    # Treat positive and negative residual lobes separately. The covariance
    # amplitude is sign-independent, but the auxiliary Gaussian fit must be
    # aligned with the observed residual sign instead of fitting only positive
    # model-minus-data discrepancies.
    for sign in (-1.0, 1.0):
        signed = sign * centered_residual
        signed_seed = seed_mask & (signed > 0)
        signed_grow = grow_mask & (signed > 0)
        indices = np.flatnonzero(signed_grow)
        if indices.size == 0:
            continue

        # First form ordinary connected runs, then merge small wavelength gaps
        # so a shallow dip within one broad line does not create two kernels.
        split_at = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = [part for part in np.split(indices, split_at) if part.size]
        merged_runs = []
        for run in runs:
            gap_start = merged_runs[-1][-1] + 1 if merged_runs else 0
            gap_stop = run[0]
            gap_keeps_sign = (
                not merged_runs
                or np.all(signed[gap_start:gap_stop] > 0)
            )
            if (
                merged_runs
                and wave[run[0]] - wave[merged_runs[-1][-1]] <= merge_gap
                and gap_keeps_sign
            ):
                merged_runs[-1] = np.arange(
                    merged_runs[-1][0], run[-1] + 1, dtype=int
                )
            else:
                merged_runs.append(run)

        for run in merged_runs:
            seed_indices = run[signed_seed[run]]
            if seed_indices.size == 0:
                continue

            peak_index = seed_indices[np.argmax(score[seed_indices])]
            mu = float(wave[peak_index])

            # Expand the bounds by half a pixel so a one-pixel region still
            # gives the bounded centre optimiser a non-zero interval.
            left_step = (
                wave[run[0]] - wave[run[0] - 1]
                if run[0] > 0 else pixel_step
            )
            right_step = (
                wave[run[-1] + 1] - wave[run[-1]]
                if run[-1] < len(wave) - 1 else pixel_step
            )
            lower = float(wave[run[0]] - 0.5 * max(left_step, 0.0))
            upper = float(wave[run[-1]] + 0.5 * max(right_step, 0.0))

            # Estimate the kernel width from the detected residual region in
            # velocity space. Enforce a one-pixel floor because a sub-pixel
            # covariance kernel is not resolved by the supplied spectrum.
            region_weights = np.maximum(
                score[run] - grow_threshold, np.finfo(float).eps
            )
            region_velocity = C.c_kms / mu * (wave[run] - mu)
            sigma_v = np.sqrt(
                np.average(region_velocity**2, weights=region_weights)
            )
            pixel_velocity = (
                C.c_kms / mu * pixel_step if pixel_step > 0 else 0.0
            )
            sigma_v = float(max(sigma_v, pixel_velocity, np.finfo(float).eps))

            regions.append({
                "mu": mu,
                "lower": lower,
                "upper": upper,
                "sign": int(sign),
                "sigma_v": sigma_v,
                "peak_score": float(score[peak_index]),
            })

    # Preserve the historical strongest-first candidate ordering.
    regions.sort(key=lambda region: region["peak_score"], reverse=True)
    if return_regions:
        return regions
    return [region["mu"] for region in regions]


def optimize_residual_peaks(
    model,
    mus,
    threshold=0.1,
    sigma0=50,
    num_residuals=100,
    covariance=None,
):
    """
    Optimize the local covariance parameters based on fitting the residual input means
    as Gaussians around the residuals

    Parameters
    ----------
    model : Model
        The model to determine peaks from. Need only have a residuals array.
    mus : array-like
        Centre wavelengths to instantiate and optimise, or region dictionaries
        returned by ``find_residual_peaks(..., return_regions=True)``.
    threshold : float, optional
        This is the threshold for restricting kernels; i.e. if a fit amplitude is less
        than threshold standard deviations then it will be thrown away. Default is 0.1
    sigma0 : float, optional
        Fallback initial velocity standard deviation in km/s for callers that
        provide centre wavelengths only. Region dictionaries provide their own
        data-driven velocity widths. Default is 50 km/s.
    num_residuals : int, optional
        The number of residuals to average together for determining peaks. By default
        100.
    covariance : array-like or None, optional
        Complete pre-local covariance matrix (or its diagonal). When supplied,
        its diagonal provides the per-pixel uncertainty used by the auxiliary
        Gaussian fit. This keeps optimisation consistent with covariance-aware
        residual-region detection.

    Returns
    -------
    list of dict
        Optimized parameter dictionaries ready to be assigned to
        ``model.params["local_cov"]``.

    Warning
    -------
    I have had inconsistent results with this optimization, be mindful of your outputs
    and consider hand-tuning after optimizing.
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)
    residual = np.asarray(residual, dtype=float)
    residual = residual - np.median(residual)
    residual_scale = 1.4826 * np.median(np.abs(residual))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        residual_scale = residual.std()
    amp_cutoff = threshold * residual_scale
    covariance_sigma = None
    global_cov = None
    if covariance is not None:
        covariance_sigma = np.sqrt(
            np.maximum(
                _covariance_diagonal(covariance, len(model.data.wave)),
                np.finfo(float).tiny,
            )
        )
    elif "global_cov" in model.params:
        ag = np.exp(model.params["global_cov:log_amp"])
        lg = np.exp(model.params["global_cov:log_ls"])
        global_cov = global_covariance_matrix(model.data.wave, ag, lg)

    def chi2(P, wave, resid, sigma, width_min, width_max):
        log_amp, mu, log_sigma = P
        _amp = np.exp(log_amp)
        _sigma = np.exp(log_sigma)
        if _amp == 0 or _sigma == 0:
            return np.inf
        # Keep the fitted width resolved by the data and local to the detected
        # region. This remains a simple uniform prior; the region itself supplies
        # the data-driven scale rather than a fixed 50 km/s initial width.
        prior = st.uniform.logpdf(
            _sigma, width_min, width_max - width_min
        )
        # Put prior on widths and heights such that the integrated area should be less
        # than the trapezoidal area of the residual
        area_max = 2 * _sigma * np.abs(resid).max()
        # Area under a gaussian
        # https://en.wikipedia.org/wiki/Gaussian_function#Integral_of_a_Gaussian_function
        area = np.sqrt(2 * np.pi) * _sigma * _amp
        prior += st.uniform.logpdf(area, 0, area_max)
        prior += st.uniform.logpdf(_amp, 0, np.abs(resid).max())
        rr = C.c_kms / mu * np.abs(wave - mu)
        gauss = _amp * np.exp(-0.5 * (rr / _sigma) ** 2)
        R = gauss - resid
        return np.sum((R / sigma) ** 2) - prior

    params = []

    for candidate in mus:
        is_region = isinstance(candidate, dict)
        if is_region:
            mu = float(candidate["mu"])
            sign = float(candidate.get("sign", 1.0))
            lower = float(candidate["lower"])
            upper = float(candidate["upper"])
            local_sigma0 = float(candidate.get("sigma_v", sigma0))

            positive_steps = np.diff(model.data.wave)
            positive_steps = positive_steps[positive_steps > 0]
            pixel_step = (
                float(np.median(positive_steps))
                if positive_steps.size else 0.0
            )
            pixel_velocity = (
                C.c_kms / mu * pixel_step if pixel_step > 0 else 0.0
            )
            width_min = max(pixel_velocity, np.finfo(float).eps)
            local_sigma0 = max(local_sigma0, width_min)
            width_max = max(2.0 * local_sigma0, 1.01 * width_min)

            # Fit only the detected structure plus enough padding to constrain
            # the Gaussian tails. This avoids the old ±50 Angstrom window in
            # which several candidates could all see and select the same line.
            sigma_lambda = mu * local_sigma0 / C.c_kms
            padding = max(2.0 * sigma_lambda, pixel_step)
            mask = (
                (model.data.wave > lower - padding)
                & (model.data.wave < upper + padding)
            )
            mu_bounds = (lower, upper)
        else:
            # Backwards-compatible path for callers that provide only centres.
            mu = float(candidate)
            sign = 1.0
            local_sigma0 = float(sigma0)
            width_min = np.finfo(float).eps
            width_max = 2.0 * local_sigma0
            mask = (
                (model.data.wave > mu - sigma0)
                & (model.data.wave < mu + sigma0)
            )
            mu_bounds = (mu - sigma0, mu + sigma0)

        wave = model.data.wave[mask]
        # Flip negative residual regions before fitting the positive Gaussian
        # height. Opposite-sign pixels in the padded window are set to zero;
        # they constrain the Gaussian tail without pulling its amplitude in the
        # wrong direction.
        resid = np.maximum(sign * residual[mask], 0.0)
        sigma = (
            covariance_sigma[mask]
            if covariance_sigma is not None
            else model.data.sigma[mask]
        )
        if wave.size == 0 or not np.any(resid > 0):
            continue
        if covariance_sigma is None and global_cov is not None:
            # Observational ``sigma`` is a standard deviation; the kernel
            # diagonal is a variance and therefore combines in quadrature.
            sigma = np.sqrt(sigma**2 + global_cov.diagonal()[mask])

        # Start inside both uniform amplitude constraints. At the residual
        # maximum, the Gaussian area A*sigma*sqrt(2*pi) exceeds the existing
        # 2*sigma*max(residual) area limit and gives Nelder-Mead an infinite
        # initial objective.
        initial_amp = 0.5 * np.abs(resid).max()
        P0 = np.array([np.log(initial_amp), mu, np.log(local_sigma0)])

        soln = minimize(
            chi2,
            P0,
            args=(wave, resid, sigma, width_min, width_max),
            method="Nelder-Mead",
            bounds=[
                (None, None),
                mu_bounds,
                (np.log(width_min), np.log(width_max)),
            ],
            options=dict(maxiter=1000),
        )
        if np.isfinite(soln.fun) and soln.x[0] > np.log(amp_cutoff):
            params.append(
                {"log_amp": soln.x[0], "mu": soln.x[1], "log_sigma": soln.x[2]}
            )

    params = sorted(params, key=lambda s: s["mu"])
    if len(params) < 2:
        return params

    # Final safety net: centres separated by less than one wavelength pixel
    # cannot represent independently resolved residual structures. Keep the
    # stronger fitted kernel if numerical optimisation still produces such a
    # pair.
    positive_steps = np.diff(model.data.wave)
    positive_steps = positive_steps[positive_steps > 0]
    min_separation = (
        float(np.median(positive_steps)) if positive_steps.size else 0.0
    )
    deduplicated = []
    for kernel in params:
        if (
            deduplicated
            and kernel["mu"] - deduplicated[-1]["mu"] < min_separation
        ):
            if kernel["log_amp"] > deduplicated[-1]["log_amp"]:
                deduplicated[-1] = kernel
        else:
            deduplicated.append(kernel)
    return deduplicated


log = logging.getLogger(__name__)


def covariance_debugger(cov: np.ndarray):
    """
    Special debugging information for the covariance matrix decomposition.
    """
    log.info(f"{'Covariance Debugger':-^60}".format())
    log.info("See https://github.com/iancze/Starfish/issues/26")
    log.info("Covariance matrix at a glance:")
    if cov.diagonal().min() < 0.0:
        log.warning("- Negative entries on the diagonal:")
        log.info("\t- Check uncertainty estimates: should all be positive")
    elif np.any(np.isnan(cov.diagonal())):
        log.warning("- Covariance matrix has a NaN value on the diagonal")
    else:
        if not np.allclose(cov, cov.T):
            log.warning("- The covariance matrix is highly asymmetric")

        # Still might have an asymmetric matrix below `allclose` threshold
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        n_neg = (eigenvalues < 0).sum()
        n_tot = len(eigenvalues)
        log.info(f"- There are {n_neg} negative eigenvalues out of {n_tot}.")

        def mark(val):
            return ">" if val < 0 else "."

        log.info("Covariance matrix eigenvalues:")
        for i in range(10):
            log.info(
                "{: >6} {:{fill}>20.3e}".format(
                    i, eigenvalues[i], fill=mark(eigenvalues[i])
                )
            )
        log.info("{: >15}".format("..."))
        for i in range(10):
            log.info(
                "{: >6} {:{fill}>20.3e}".format(
                    n_tot - 10 + i,
                    eigenvalues[-10 + i],
                    fill=mark(eigenvalues[-10 + i]),
                )
            )

    log.info(f"{'-':-^60}")
