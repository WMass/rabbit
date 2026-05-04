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
    selected_idxs,
    selected_names,
    sigma=1.0,
    signs=(-1, 1),
    linear_warmstart=False,
):
    """Run a per-nuisance theta0-shift + re-fit and assemble the asymmetric
    global impact tensor.

    Args:
        fitter: the Fitter instance (used for x, theta0 and minimize).
        selected_idxs: indices into the syst axis (0..nsyst-1) of nuisances to
            scan.
        selected_names: names of those nuisances (bytes), used as impact-axis
            labels.
        sigma: shift magnitude in units of the prefit constraint width
            (constraints are unit-sigma in rabbit, so 1.0 = 1 prefit sigma).
        signs: sequence (down, up). Bin 0 of axis_downUpVar -> first sign.
        linear_warmstart: experimental. If True, warm-start each refit at
            x_nom + dxdtheta0[:, i] * shift, the Gaussian-approximation new
            minimum for the shifted theta0. Should drastically reduce the
            number of optimizer iterations on near-Gaussian nuisances.
            Requires fitter.cov to exist (same prerequisite as
            --gaussianGlobalImpacts).

    Returns:
        parms: np.ndarray of bytes, the impact-axis labels.
        impacts: np.ndarray of shape (n_scanned, 2, n_total_params).
            Axis 1 is [down, up] matching axis_downUpVar.
        group_names: np.ndarray of bytes for groups containing scanned
            nuisances (plus a trailing "Total").
        impacts_grouped: np.ndarray of shape (n_groups, 2, n_total_params).
    """
    n_scanned = len(selected_idxs)
    n_total = len(fitter.parms)
    impacts = np.zeros((n_scanned, 2, n_total))

    nparams = fitter.param_model.nparams

    # Snapshot postfit nominal state to restore between iterations.
    x_nom = tf.identity(fitter.x.value())
    theta0_nom = tf.identity(fitter.theta0.value())
    theta0_nom_np = theta0_nom.numpy()
    x_nom_np = x_nom.numpy()

    logger.info(
        f"global_asym_impacts: shifting theta0 by +/- {sigma} sigma and "
        f"re-fitting for {n_scanned} nuisances"
        + (" (linear warm-start enabled)" if linear_warmstart else "")
    )

    # Optional Gaussian-approximation warm-start.
    # dxdtheta0 has shape [npar, nsyst]; column i gives the linearised
    # response of the postfit minimum to a unit shift of theta0[i]. Computing
    # it once before the loop is the same cost as one --gaussianGlobalImpacts
    # call.
    dxdtheta0_np = None
    if linear_warmstart:
        if fitter.cov is None:
            raise RuntimeError(
                "global_asym_impacts: linear_warmstart requires fitter.cov "
                "(incompatible with --noHessian)."
            )
        t_lws = time.perf_counter()
        dxdtheta0_tf, _, _ = fitter._dxdvars()
        dxdtheta0_np = dxdtheta0_tf.numpy()
        logger.info(
            f"global_asym_impacts: dxdtheta0 prepared in "
            f"{time.perf_counter() - t_lws:.2f}s"
        )

    t_per = np.zeros(n_scanned, dtype=np.float64)
    t_total0 = time.perf_counter()

    for k, i in enumerate(selected_idxs):
        i = int(i)
        name = selected_names[k]
        name_str = name.decode() if isinstance(name, bytes) else name
        logger.info(f"  [{k + 1}/{n_scanned}] theta0-shift refit for {name_str}")

        t0 = time.perf_counter()
        for j, sign in enumerate(signs):
            shift = float(sign) * float(sigma)

            # Always shift the constraint center for nuisance i by `shift`.
            theta0_shifted = theta0_nom_np.copy()
            theta0_shifted[i] += shift

            # Warm-start x. With linear_warmstart, use the Gaussian-approx new
            # minimum x_nom + dxdtheta0[:, i] * shift -- on near-Gaussian
            # nuisances this lands at the new minimum to within roundoff.
            # Without it, just shift x[nparams+i] by `shift` so the nuisance
            # itself starts at the new constraint center.
            if linear_warmstart:
                x_shifted = x_nom_np + dxdtheta0_np[:, i] * shift
            else:
                x_shifted = x_nom_np.copy()
                x_shifted[nparams + i] += shift

            fitter.theta0.assign(theta0_shifted)
            fitter.x.assign(x_shifted)

            try:
                fitter.minimize()
                if fitter.binByBinStat:
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
    fitter.theta0.assign(theta0_nom)
    fitter.x.assign(x_nom)
    if fitter.binByBinStat:
        fitter._profile_beta()

    if n_scanned > 0:
        t_total = time.perf_counter() - t_total0
        logger.info(
            f"global_asym_impacts: total {t_total:.1f}s "
            f"(mean {t_per.mean():.2f}s, min {t_per.min():.2f}s, "
            f"max {t_per.max():.2f}s per nuisance)"
        )

    # Grouped impacts via quadrature envelope, separately for down/up.
    selected_set = set(int(idx) for idx in selected_idxs)
    pos_in_scanned = {int(idx): k for k, idx in enumerate(selected_idxs)}

    group_names = []
    group_impacts = []
    for gname, gidxs in zip(fitter.indata.systgroups, fitter.indata.systgroupidxs):
        gidxs = np.asarray(gidxs).astype(int)
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
