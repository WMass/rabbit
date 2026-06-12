"""
Traditional asymmetric impacts.

For each selected nuisance, find the asymmetric +/- 1 sigma points on the
Delta(2NLL)=q likelihood contour via constrained minimization (contour_scan).
The shifts of every fitter parameter at those points are the asymmetric
impacts.

Group impacts use the freeze-group convention (as in combine): for each
POI/NOI target, scan its own Delta(2NLL)=q contour with nothing frozen
(total interval), then once more per group with the group's nuisances frozen
at their postfit values; per side the group impact is the quadrature
difference sqrt(sigma_tot^2 - sigma_frozen^2). Both intervals come from full
profile likelihoods, so all correlations within the group and with the rest
of the fit are accounted for. The "stat" column is the interval with every
constrained nuisance frozen (reported directly, not subtracted), and "Total"
is the unfrozen interval itself.

All nuisances are scanned by default, including unconstrained ones — like
the symmetric traditional impacts, the data still gives them a finite
postfit uncertainty and hence a finite Delta(2NLL)=q contour (nonlinear
effects can also produce asymmetric impacts even for structurally
symmetric templates). Use include/exclude to restrict the selection. An
optional structural-symmetry skip (logkhalfdiff identically zero) is
available programmatically via skip_symmetric.
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


def _scan_interval(fitter, target, nll_min, q, scan_kwargs):
    """Signed [down, up] Delta(2NLL)=q interval of a single parameter from its
    contour scan. Entries are NaN where the scan did not converge."""
    intervals, _ = fitter.contour_scan(
        target, nll_min, q=q, signs=[-1, 1], **scan_kwargs
    )
    return np.asarray(intervals, dtype=np.float64)


def _quad_subtract(tot, frozen):
    """Per-side quadrature difference of two signed [down, up] intervals.

    Returns signed [down, up] with down <= 0 <= up. The argument of the sqrt
    is clipped at zero: freezing parameters can only shrink the profile
    interval in the Gaussian limit, so a (small) negative difference is
    numerical noise from the scan tolerances; a large one indicates an
    unconverged scan and is logged.
    """
    out = np.full(2, np.nan)
    for s in range(2):
        if not (np.isfinite(tot[s]) and np.isfinite(frozen[s])):
            continue
        diff = tot[s] ** 2 - frozen[s] ** 2
        if diff < -0.05 * tot[s] ** 2:
            logger.warning(
                f"freeze-group quadrature difference is significantly "
                f"negative (tot={tot[s]:.4g}, frozen={frozen[s]:.4g}); "
                f"likely an unconverged contour scan. Clipping to 0."
            )
        mag = np.sqrt(max(diff, 0.0))
        out[s] = -mag if s == 0 else mag
    return out


def grouped_asym_impacts(
    fitter,
    nll_min,
    targets,
    q=1,
    scan_kwargs=None,
):
    """Freeze-group grouped asymmetric impacts (combine convention).

    For each target parameter (POIs and NOIs), the asymmetric Delta(2NLL)=q
    interval is scanned with nothing frozen (total) and once per nuisance
    group with the group's members frozen at their postfit values; the group
    impact per side is sqrt(sigma_tot^2 - sigma_frozen^2). The "stat" column
    is the interval with all constrained nuisances frozen (reported directly),
    "Total" is the unfrozen interval. Note the bin-by-bin stat beta
    parameters remain profiled throughout, so "stat" includes the
    bin-by-bin MC-stat contribution (stat (+) binByBinStat in the
    traditional grouped-impact decomposition).

    Args:
        fitter: the Fitter instance.
        nll_min: postfit reduced NLL.
        targets: list of parameter names (str) to compute group impacts for.
        q: contour level (q=1 -> 1 sigma).
        scan_kwargs: forwarded to contour_scan (xtol/gtol/maxiter/hess_mode).

    Returns:
        group_names: np.ndarray of bytes, [systgroups..., stat, Total].
        impacts_grouped: np.ndarray of shape (n_groups, 2, n_total_params),
            NaN outside the target columns.
    """
    scan_kwargs = scan_kwargs or {}
    n_total = len(fitter.parms)
    parms_str = fitter.parms.astype(str)
    target_cols = {t: int(np.where(parms_str == t)[0][0]) for t in targets}
    syst_names_str = np.array(fitter.indata.systs).astype(str)
    # only freeze what this function froze; never defreeze user-frozen params
    already_frozen = {
        p.decode() if isinstance(p, bytes) else str(p) for p in fitter.frozen_params
    }

    n_groups = len(fitter.indata.systgroups)
    n_scans = len(targets) * (n_groups + 2) * 2
    logger.info(
        f"grouped asym impacts (freeze-group): {len(targets)} targets x "
        f"({n_groups} groups + stat + Total) -> {n_scans} contour scans"
    )
    t0 = time.perf_counter()

    # total (unfrozen) interval per target
    totals = {t: _scan_interval(fitter, t, nll_min, q, scan_kwargs) for t in targets}

    def scan_with_frozen(members):
        """Scan all targets with the given nuisances (str names) frozen at
        their postfit values."""
        to_freeze = [m for m in members if m not in already_frozen]
        arr = np.full((2, n_total), np.nan)
        if to_freeze:
            fitter.freeze_params(to_freeze)
        try:
            frozen_set = set(members) | already_frozen
            for t, c in target_cols.items():
                if t in frozen_set:
                    continue  # cannot scan a frozen target
                arr[:, c] = _scan_interval(fitter, t, nll_min, q, scan_kwargs)
        finally:
            if to_freeze:
                fitter.defreeze_params(to_freeze)
        return arr

    group_names = []
    group_impacts = []
    for gname, gidxs in zip(fitter.indata.systgroups, fitter.indata.systgroupidxs):
        members = [str(syst_names_str[int(i)]) for i in np.asarray(gidxs).astype(int)]
        frozen_arr = scan_with_frozen(members)
        arr = np.full((2, n_total), np.nan)
        for t, c in target_cols.items():
            arr[:, c] = _quad_subtract(totals[t], frozen_arr[:, c])
        group_names.append(gname)
        group_impacts.append(arr)

    # stat: freeze every constrained nuisance; the remaining interval is the
    # statistical (+ unconstrained-NOI) uncertainty, reported directly.
    cw = fitter.indata.constraintweights.numpy()
    constrained = [str(syst_names_str[i]) for i in np.where(cw > 0)[0]]
    group_names.append(b"stat")
    group_impacts.append(scan_with_frozen(constrained))

    # Total: the unfrozen interval itself.
    arr = np.full((2, n_total), np.nan)
    for t, c in target_cols.items():
        arr[:, c] = totals[t]
    group_names.append(b"Total")
    group_impacts.append(arr)

    logger.info(f"grouped asym impacts: total {time.perf_counter() - t0:.1f}s")

    return np.asarray(group_names), np.stack(group_impacts)


def asym_impacts_parms(
    fitter,
    nll_min,
    selected_idxs,
    selected_names,
    targets=None,
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
        targets: parameter names (str, typically POIs + NOIs) for which the
            freeze-group grouped impacts are computed. Empty/None skips the
            grouped computation.
        q: contour level (q=1 -> 1 sigma, q=4 -> 2 sigma).

    Returns:
        parms: np.ndarray of bytes, the impact-axis labels.
        impacts: np.ndarray of shape (n_scanned, 2, n_total_params).
            Axis 1 is [down, up] matching axis_downUpVar.
        group_names: np.ndarray of bytes, [systgroups..., stat, Total].
        impacts_grouped: np.ndarray of shape (n_groups, 2, n_total_params),
            NaN outside the target columns (see grouped_asym_impacts).
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

    # Grouped impacts via the freeze-group convention (independent of the
    # per-nuisance scan selection above).
    if targets:
        scan_kwargs = dict(
            xtol=contour_xtol,
            gtol=contour_gtol,
            maxiter=contour_maxiter,
            hess_mode=hess_mode,
        )
        group_names, impacts_grouped = grouped_asym_impacts(
            fitter, nll_min, targets, q=q, scan_kwargs=scan_kwargs
        )
    else:
        group_names = np.asarray([], dtype=bytes)
        impacts_grouped = np.zeros((0, 2, n_total))

    return (
        np.asarray(selected_names),
        impacts,
        group_names,
        impacts_grouped,
    )
