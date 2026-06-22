"""
Fully likelihood-based asymmetric global impacts.

For each selected nuisance i, shift the auxiliary (global) observable theta0[i]
by +/- 1 prefit sigma and re-run the full fit. The resulting POI shifts are the
asymmetric global impacts. This is option (c) of the three definitions listed
in global_impacts.py and complements the Gaussian variants there:

  - global_impacts_parms / gaussian_global_impacts_parms: analytic, single-sided.
  - global_asym_impacts_parms (this module): fully likelihood, two-sided, exact.

In the Gaussian limit this reproduces gaussian_global_impacts_parms; deviations
measure the non-Gaussianity of the joint profile around the postfit minimum.

Differs from nonprofiled_impacts in two ways:
  - nuisance i is *profiled* (not frozen), so it can equilibrate at the new
    constraint center together with all correlated nuisances;
  - beta (BBB) parameters are profiled too, regardless of --noPostfitProfileBB.
"""

import time

import numpy as np
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


def _envelope(values):
    """Quadrature envelope of asymmetric impacts within a group, separately for
    the down and up sides.

    Args:
        values: numpy array of shape (n_in_group, 2, n_total_params), where
            axis 1 is [down, up].

    Returns:
        Array of shape (2, n_total_params).
    """
    zeros = np.zeros((values.shape[0], values.shape[-1]), dtype=values.dtype)
    vmin = np.min(values, axis=1)
    vmax = np.max(values, axis=1)
    lower = -np.sqrt(np.sum(np.minimum(zeros, vmin) ** 2, axis=0))
    upper = np.sqrt(np.sum(np.maximum(zeros, vmax) ** 2, axis=0))
    return np.stack([lower, upper])


def global_asym_impacts_parms(
    fitter,
    selected_x_idxs,
    selected_names,
    group_members=None,
    sigma=1.0,
    signs=(-1, 1),
    linear_warmstart=False,
):
    """Run a per-source x0-shift + re-fit and assemble the asymmetric global
    impact tensor.

    A "source" is any constraint center: a constrained nuisance or a priored
    ParamModel parameter. For each one its center x0[idx] is shifted by +/-
    sigma and the full fit is re-run; the resulting POI/NOI shifts are the
    asymmetric global impacts. This mirrors the sources of
    gaussian_global_impacts_parms (which exposes priored params as
    <param>_prior columns), so the two agree in the Gaussian limit.

    Args:
        fitter: the Fitter instance (used for x, x0 and minimize).
        selected_x_idxs: full-x indices of the sources to scan (a nuisance i
            is at fitter.param_model.nparams + i; a priored param is at its
            own position in the leading param block).
        selected_names: labels for those sources (bytes), used as impact-axis
            labels (priored params are labelled <param>_prior by the caller).
        group_members: optional dict {group_name(bytes): [full-x idxs]} used
            to build the grouped quadrature envelopes. Covers both syst groups
            and ParamModel impact groups.
        sigma: shift magnitude in units of the prefit constraint width
            (constraints are unit-sigma in rabbit, so 1.0 = 1 prefit sigma).
        signs: sequence (down, up). Bin 0 of axis_downUpVar -> first sign.
        linear_warmstart: experimental. If True, warm-start each refit at
            x_nom + dxds[:, source] * shift, the Gaussian-approximation new
            minimum for the shifted center. Should drastically reduce the
            number of optimizer iterations on near-Gaussian sources.
            Requires fitter.cov to exist (same prerequisite as
            --gaussianGlobalImpacts).

    Returns:
        parms: np.ndarray of bytes, the impact-axis labels.
        impacts: np.ndarray of shape (n_scanned, 2, n_total_params).
            Axis 1 is [down, up] matching axis_downUpVar.
        group_names: np.ndarray of bytes for groups containing scanned
            sources (plus a trailing "Total").
        impacts_grouped: np.ndarray of shape (n_groups, 2, n_total_params).
    """
    if group_members is None:
        group_members = {}
    selected_x_idxs = [int(idx) for idx in selected_x_idxs]
    n_scanned = len(selected_x_idxs)
    n_total = len(fitter.parms)
    impacts = np.zeros((n_scanned, 2, n_total))

    # Prefit width of each constraint center: sqrt(var_x0) = 1/sqrt(cw). For
    # nuisances cw = 1 so this is 1 (the historical "1.0 = 1 sigma"); for
    # priored params it is the prior sigma, so a unit-sigma shift moves x0 by
    # sigma, not by 1. Scaling by it keeps the asym impact in the same per-1-
    # prefit-sigma units as gaussian_global_impacts_parms.
    src_sigma_np = np.sqrt(fitter.var_x0.numpy())

    # Snapshot postfit nominal state to restore between iterations.
    x_nom = tf.identity(fitter.x.value())
    x0_nom = tf.identity(fitter.x0.value())
    x0_nom_np = x0_nom.numpy()
    x_nom_np = x_nom.numpy()

    logger.info(
        f"global_asym_impacts: shifting constraint centers by +/- {sigma} sigma "
        f"and re-fitting for {n_scanned} sources"
        + (" (linear warm-start enabled)" if linear_warmstart else "")
    )

    # Optional Gaussian-approximation warm-start. _dxdvars returns the postfit
    # response to a unit shift of each constraint center as a single collection
    # of columns aligned with fitter.x0_source_idxs; we map each scanned
    # source's full-x index to its column. Computing it once is the same cost
    # as one --gaussianGlobalImpacts call.
    warmstart_col = None
    if linear_warmstart:
        if fitter.cov is None:
            raise RuntimeError(
                "global_asym_impacts: linear_warmstart requires fitter.cov "
                "(incompatible with --noHessian)."
            )
        t_lws = time.perf_counter()
        dxdx0_tf, _, _ = fitter._dxdvars()
        dxdx0_np = dxdx0_tf.numpy()
        src_col = {int(s): k for k, s in enumerate(fitter.x0_source_idxs.numpy())}

        def _warmstart_col(idx):
            return dxdx0_np[:, src_col[idx]]

        warmstart_col = _warmstart_col

        logger.info(
            f"global_asym_impacts: dxds prepared in "
            f"{time.perf_counter() - t_lws:.2f}s"
        )

    t_per = np.zeros(n_scanned, dtype=np.float64)
    t_total0 = time.perf_counter()

    for k, idx in enumerate(selected_x_idxs):
        name = selected_names[k]
        name_str = name.decode() if isinstance(name, bytes) else name
        logger.info(f"  [{k + 1}/{n_scanned}] x0-shift refit for {name_str}")

        # shift the center by `sigma` prefit sigmas of this source (1 for
        # unit-constrained nuisances, the prior sigma for priored params).
        width = src_sigma_np[idx]

        t0 = time.perf_counter()
        for j, sign in enumerate(signs):
            shift = float(sign) * float(sigma) * float(width)

            # Always shift the constraint center for source idx by `shift`.
            x0_shifted = x0_nom_np.copy()
            x0_shifted[idx] += shift

            # Warm-start x. With linear_warmstart, use the Gaussian-approx new
            # minimum x_nom + dxds[:, source] * shift -- on near-Gaussian
            # sources this lands at the new minimum to within roundoff.
            # Without it, just shift x[idx] by `shift` so the parameter itself
            # starts at the new constraint center.
            if warmstart_col is not None:
                x_shifted = x_nom_np + warmstart_col(idx) * shift
            else:
                x_shifted = x_nom_np.copy()
                x_shifted[idx] += shift

            fitter.x0.assign(x0_shifted)
            fitter.x.assign(x_shifted)

            try:
                fitter.minimize()
                if fitter.bbstat.enabled:
                    fitter._profile_beta()
                impacts[k, j] = (fitter.x.value() - x_nom).numpy()
            except Exception as e:
                logger.warning(
                    f"    refit for {name_str} sign={sign:+d} failed: {e}; "
                    "leaving impact at zero"
                )

        t_per[k] = time.perf_counter() - t0
        logger.info(f"    took {t_per[k]:.2f}s")

    # Restore the fit state so downstream postfit computations see the nominal.
    fitter.x0.assign(x0_nom)
    fitter.x.assign(x_nom)
    if fitter.bbstat.enabled:
        fitter._profile_beta()

    if n_scanned > 0:
        t_total = time.perf_counter() - t_total0
        logger.info(
            f"global_asym_impacts: total {t_total:.1f}s "
            f"(mean {t_per.mean():.2f}s, min {t_per.min():.2f}s, "
            f"max {t_per.max():.2f}s per source)"
        )

    # Grouped impacts via quadrature envelope, separately for down/up.
    selected_set = set(selected_x_idxs)
    pos_in_scanned = {idx: k for k, idx in enumerate(selected_x_idxs)}

    group_names = []
    group_impacts = []
    for gname, gidxs in group_members.items():
        in_scanned = [pos_in_scanned[i] for i in gidxs if int(i) in selected_set]
        if not in_scanned:
            continue
        group_names.append(gname)
        group_impacts.append(_envelope(impacts[in_scanned]))

    if n_scanned > 0:
        group_names.append(b"Total")
        group_impacts.append(_envelope(impacts))

    if group_impacts:
        impacts_grouped = np.stack(group_impacts)
    else:
        impacts_grouped = np.zeros((0, 2, n_total))

    return (
        np.asarray(selected_names),
        impacts,
        np.asarray(group_names),
        impacts_grouped,
    )
