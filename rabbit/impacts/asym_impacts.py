"""
Traditional asymmetric impacts.

For each selected nuisance, find the asymmetric +/- 1 sigma points on the
Delta(2NLL)=q likelihood contour via constrained minimization (contour_scan).
The shifts of every fitter parameter at those points are the asymmetric
impacts. Group impacts follow the traditional grouped-impact convention,
generalized to the asymmetric case: per side, sqrt(delta^T rho_GG^-1 delta)
over the group's scanned nuisances, with rho_GG the postfit correlation
submatrix (see _group_impact).

All constrained nuisances are scanned by default (nonlinear effects can
produce asymmetric impacts even for structurally symmetric templates);
use include/exclude to restrict the selection. Unconstrained nuisances
(constraintweight = 0) are skipped since they have no finite Delta(2NLL)
contour. An optional structural-symmetry skip (logkhalfdiff identically
zero) is available programmatically via skip_symmetric.
"""

import time

import numpy as np
from wums import logging

logger = logging.child_logger(__name__)


def asymmetric_nuisance_mask(indata, atol=0.0):
    """Boolean mask of length nsyst, True for nuisances with nonzero asymmetric
    (logkhalfdiff) tensor content. False if the entire tensor is symmetric or
    if a particular nuisance has zero halfdiff."""
    if indata.symmetric_tensor:
        return np.zeros(indata.nsyst, dtype=bool)

    nsyst = indata.nsyst
    if indata.sparse:
        idx = indata.logk.indices.numpy()
        vals = indata.logk.values.numpy()
        syst_col = idx[:, -1]
        halfdiff_entries = (syst_col >= nsyst) & (np.abs(vals) > atol)
        nz_systs = np.unique(syst_col[halfdiff_entries]) - nsyst
        mask = np.zeros(nsyst, dtype=bool)
        mask[nz_systs] = True
        return mask

    # Dense layout: logk has shape [nbinsfull, nproc, 2, nsyst] when asymmetric;
    # axis -2 is [logkavg, logkhalfdiff]. Reduce over (bin, proc) for halfdiff.
    halfdiff = indata.logk[..., 1, :].numpy()
    axes = tuple(range(halfdiff.ndim - 1))
    return np.any(np.abs(halfdiff) > atol, axis=axes)


def _group_impact(values, rho_inv):
    """Correlation-aware group combination of asymmetric impacts.

    Asymmetric analogue of the traditional grouped-impact convention
    (sqrt(v^T C_GG^-1 v) in traditional_impacts._compute_impact_group):
    per side s, the group impact on each parameter is

        sqrt( delta_s^T rho_GG^-1 delta_s )

    where delta_s is the vector of per-nuisance impacts on that parameter at
    the side-s contour points and rho_GG is the postfit correlation submatrix
    of the group's nuisances. In the symmetric Gaussian limit
    (delta_j = cov[i, j] / sqrt(cov[j, j])) this reproduces the traditional
    grouped impact exactly; for uncorrelated nuisances it reduces to a
    quadrature sum.

    Args:
        values: numpy array of shape (n_in_group, 2, n_total_params), where
            axis 1 is [down, up].
        rho_inv: numpy array of shape (n_in_group, n_in_group), the inverse
            postfit correlation matrix of the group's nuisances.

    Returns:
        Array of shape (2, n_total_params) with the group's lower (negative)
        and upper (positive) sides.
    """
    down = values[:, 0, :]
    up = values[:, 1, :]
    # rho_inv is positive-definite so the quadratic form is >= 0; the clip
    # only guards against floating-point round-off.
    q_down = np.einsum("jt,jk,kt->t", down, rho_inv, down)
    q_up = np.einsum("jt,jk,kt->t", up, rho_inv, up)
    lower = -np.sqrt(np.clip(q_down, 0.0, None))
    upper = np.sqrt(np.clip(q_up, 0.0, None))
    return np.stack([lower, upper])


def _rho_inv_for(cov, x_idxs):
    """Inverse postfit correlation submatrix for the given full-x indices."""
    sub = cov[np.ix_(x_idxs, x_idxs)]
    sigma = np.sqrt(np.diag(sub))
    rho = sub / np.outer(sigma, sigma)
    # pinv for robustness against near-degenerate (highly correlated) groups.
    return np.linalg.pinv(rho)


def asym_impacts_parms(
    fitter,
    nll_min,
    selected_idxs,
    selected_names,
    q=1,
    contour_xtol=1e-4,
    contour_gtol=1e-4,
    contour_maxiter=5000,
    hess_mode="exact",
):
    """Run a per-nuisance contour scan and assemble the asymmetric-impact tensor.

    Args:
        fitter: the Fitter instance (used for contour_scan and indata).
        nll_min: postfit reduced NLL.
        selected_idxs: indices into the syst axis (0..nsyst-1) of nuisances to scan.
        selected_names: names of those nuisances (bytes), used as impact-axis labels.
        q: contour level (q=1 -> 1 sigma, q=4 -> 2 sigma).

    Returns:
        parms: np.ndarray of bytes, the impact-axis labels.
        impacts: np.ndarray of shape (n_scanned, 2, n_total_params).
            Axis 1 is [down, up] matching axis_downUpVar.
        group_names: np.ndarray of bytes for groups containing scanned nuisances
            (plus a trailing "Total").
        impacts_grouped: np.ndarray of shape (n_groups, 2, n_total_params).
    """
    n_scanned = len(selected_idxs)
    n_total = len(fitter.parms)
    impacts = np.zeros((n_scanned, 2, n_total))

    logger.info(f"asym_impacts: scanning {n_scanned} nuisances")

    t_per = np.zeros(n_scanned, dtype=np.float64)
    t_total0 = time.perf_counter()

    for i, name in enumerate(selected_names):
        name_str = name.decode() if isinstance(name, bytes) else name
        logger.info(f"  [{i + 1}/{n_scanned}] contour scan for {name_str}")
        t0 = time.perf_counter()
        _, params_values = fitter.contour_scan(
            name_str,
            nll_min,
            q=q,
            signs=[-1, 1],
            xtol=contour_xtol,
            gtol=contour_gtol,
            maxiter=contour_maxiter,
            hess_mode=hess_mode,
        )
        t_per[i] = time.perf_counter() - t0
        logger.info(f"    took {t_per[i]:.2f}s")
        # signs=[-1, +1] -> bin 0 = down, bin 1 = up (matches axis_downUpVar).
        # params_values rows are NaN where convergence failed; leave as zero impact.
        if not np.any(np.isnan(params_values)):
            impacts[i] = params_values
        else:
            valid = ~np.any(np.isnan(params_values), axis=1)
            impacts[i, valid] = params_values[valid]

    if n_scanned > 0:
        t_total = time.perf_counter() - t_total0
        logger.info(
            f"asym_impacts: total {t_total:.1f}s "
            f"(mean {t_per.mean():.2f}s, min {t_per.min():.2f}s, "
            f"max {t_per.max():.2f}s per nuisance)"
        )

    # Build grouped impacts with the correlation-aware combination,
    # separately for down/up (see _group_impact). Nuisance j (syst index)
    # sits at full-x index nparams + j.
    selected_set = set(selected_idxs.tolist())
    pos_in_scanned = {int(idx): k for k, idx in enumerate(selected_idxs)}
    cov = fitter.cov.numpy()
    nparams = fitter.param_model.nparams

    group_names = []
    group_impacts = []
    for gname, gidxs in zip(fitter.indata.systgroups, fitter.indata.systgroupidxs):
        gidxs = np.asarray(gidxs).astype(int)
        in_group = [int(i) for i in gidxs if int(i) in selected_set]
        if not in_group:
            continue
        in_scanned = [pos_in_scanned[i] for i in in_group]
        rho_inv = _rho_inv_for(cov, [nparams + i for i in in_group])
        group_names.append(gname)
        group_impacts.append(_group_impact(impacts[in_scanned], rho_inv))

    if n_scanned > 0:
        rho_inv = _rho_inv_for(cov, [nparams + int(i) for i in selected_idxs])
        group_names.append(b"Total")
        group_impacts.append(_group_impact(impacts, rho_inv))

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
