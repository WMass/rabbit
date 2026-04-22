"""
Global impacts from shifting of auxiliary (global) observables
  i.e. shifting the nuisance parameter by its pre-fit uncertainty, the number of observed events in data by its uncertainty, etc.
This is the "shifted global observable" method.
Grouped impacts are obtained from adding the impacts of individual sources in quadrature.

There are three different definitions
a) the fully gaussian global impacts, which are extracted from a gaussian approximation of the likelihood and the global observables.
b) the likelihood based global impacts in the gaussian approximation, which are extracted from the likelihood terms of the global observables.
In this definition the individual impacts add up in quadrature to the total.
c) (TODO to be implemented) the fully likelihood based impacts, which are extracted by shifting the global observables and repeating the fit.

Ref. https://arxiv.org/abs/2307.04007
"""

import tensorflow as tf


def _gather_poi_noi_vector(v, noiidxs, nsignal_params=0, nmodel_params=None):
    if nmodel_params is None:
        nmodel_params = nsignal_params
    v_poi = v[:nsignal_params]
    v_noi = tf.gather(v[nmodel_params:], noiidxs)
    return tf.concat([v_poi, v_noi], axis=0)


def _compute_global_impact_group(d_squared, idxs):
    gathered = tf.gather(d_squared, idxs, axis=-1)
    d_squared_summed = tf.reduce_sum(gathered, axis=-1)
    return tf.sqrt(d_squared_summed)


def _compute_global_impacts_beta0_jvp(
    x,
    ubeta,
    beta_shape,
    compute_yields_with_beta_fn,
    compute_lbeta_fn,
    cov_dexpdx,
    profile=True,
):
    """
    Computes global impacts from beta parameters via JVP in forward accumulator mode.
    This is fast in case of more beta parameters than explicit parameters (x) and 'cov_dexpdx' has only a few columns.
    It should always be more memory efficient.
    """
    with tf.GradientTape() as t2:
        t2.watch(ubeta)
        with tf.GradientTape() as t1:
            t1.watch(ubeta)
            *_, beta = compute_yields_with_beta_fn(
                profile=profile, compute_norm=False, full=False
            )
            lbeta = compute_lbeta_fn(beta)
        pdlbetadbeta = t1.gradient(lbeta, ubeta)

    # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
    pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, ubeta)

    # this is the cholesky decomposition of pd2lbetadbeta2
    sbeta = tf.linalg.LinearOperatorDiag(
        tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
    )

    impacts_beta_shape = (*beta_shape, cov_dexpdx.shape[-1])
    impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

    if profile:
        # dbeta/dx is None if not profiled (no relation)
        def _tangents(tangent_vector):
            with tf.autodiff.ForwardAccumulator(x, tangent_vector) as acc:
                *_, beta = compute_yields_with_beta_fn(
                    profile=True, compute_norm=False, full=False
                )
            return acc.jvp(beta)

        tangents = tf.transpose(cov_dexpdx)
        dbetadx_cov_dexpdx = tf.vectorized_map(_tangents, tangents)

        # flatten all but first axes
        dbetadx_cov_dexpdx = tf.reshape(
            dbetadx_cov_dexpdx, [tf.shape(dbetadx_cov_dexpdx)[0], -1]
        )
        dbetadx_cov_dexpdx = tf.transpose(dbetadx_cov_dexpdx)

        impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

    return impacts_beta0, sbeta


def _compute_global_impacts_beta0(
    x,
    ubeta,
    beta_shape,
    compute_yields_with_beta_fn,
    compute_lbeta_fn,
    cov_dexpdx,
    profile=True,
):
    """
    Computes global impacts from beta parameters in the traditional mode.
    This is fast in case of less beta parameters than explicit parameters (x) or 'cov_dexpdx' has many columns.
    """
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([x, ubeta])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([x, ubeta])
            *_, beta = compute_yields_with_beta_fn(
                profile=profile, compute_norm=False, full=False
            )
            lbeta = compute_lbeta_fn(beta)
        pdlbetadbeta = t1.gradient(lbeta, ubeta)
        dbetadx = t1.jacobian(beta, x)
    # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
    pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, ubeta)

    # this is the cholesky decomposition of pd2lbetadbeta2
    sbeta = tf.linalg.LinearOperatorDiag(
        tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
    )

    impacts_beta_shape = (*beta_shape, cov_dexpdx.shape[-1])
    impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

    if profile:
        dbetadx_cov_dexpdx = dbetadx @ cov_dexpdx
        dbetadx_cov_dexpdx = tf.reshape(
            dbetadx_cov_dexpdx, [-1, tf.shape(dbetadx_cov_dexpdx)[-1]]
        )

        impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

    return impacts_beta0, sbeta


def _compute_beta0_impacts(
    x,
    ubeta,
    beta_shape,
    compute_yields_with_beta_fn,
    compute_lbeta_fn,
    global_impacts_from_jvp,
    bin_by_bin_stat_mode,
    cov_dexpdx,
    profile,
    pdexpdbeta,
    pd2ldbeta2_pdexpdbeta,
):
    """Compute beta0 impacts and variance, shared between parms and obs variants."""
    if global_impacts_from_jvp:
        impacts_beta0, sbeta = _compute_global_impacts_beta0_jvp(
            x,
            ubeta,
            beta_shape,
            compute_yields_with_beta_fn,
            compute_lbeta_fn,
            cov_dexpdx,
            profile,
        )
    else:
        impacts_beta0, sbeta = _compute_global_impacts_beta0(
            x,
            ubeta,
            beta_shape,
            compute_yields_with_beta_fn,
            compute_lbeta_fn,
            cov_dexpdx,
            profile,
        )

    if pdexpdbeta is not None:
        impacts_beta0 += tf.reshape(sbeta @ pd2ldbeta2_pdexpdbeta, impacts_beta0.shape)

    var_beta0 = tf.reduce_sum(tf.square(impacts_beta0), axis=0)
    impacts_beta0_process = None
    if bin_by_bin_stat_mode == "full":
        impacts_beta0_process = tf.sqrt(var_beta0)
        var_beta0 = tf.reduce_sum(var_beta0, axis=0)

    return tf.sqrt(var_beta0), impacts_beta0_process, var_beta0


def _compute_global_impacts_x0(x, compute_lc_fn, cov_dexpdx):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            lc = compute_lc_fn()
        dlcdx = t1.gradient(lc, x)
    # d2lcdx2 is diagonal so we can use gradient instead of jacobian
    d2lcdx2_diag = t2.gradient(dlcdx, x)

    # sc is the cholesky decomposition of d2lcdx2
    sc = tf.linalg.LinearOperatorDiag(tf.sqrt(d2lcdx2_diag), is_self_adjoint=True)
    return sc @ cov_dexpdx


def _compute_grouped_impacts(
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    systgroupidxs,
    impacts_theta0_sq,
    impacts_nobs,
    impacts_beta0_total,
    impacts_beta0_process,
):
    """Assemble the grouped impacts tensor from all contributions."""
    if bin_by_bin_stat:
        impacts_grouped = tf.stack([impacts_nobs, impacts_beta0_total], axis=-1)
        if bin_by_bin_stat_mode == "full":
            impacts_grouped = tf.concat(
                [impacts_grouped, tf.transpose(impacts_beta0_process)], axis=-1
            )
    else:
        impacts_grouped = impacts_nobs[..., None]

    if len(systgroupidxs):
        impacts_grouped_syst = tf.map_fn(
            lambda idxs: _compute_global_impact_group(impacts_theta0_sq, idxs),
            tf.ragged.constant(systgroupidxs, dtype=tf.int64),
            fn_output_signature=tf.TensorSpec(
                shape=(impacts_theta0_sq.shape[0],), dtype=impacts_theta0_sq.dtype
            ),
        )
        impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
        impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=-1)

    return impacts_grouped


def global_impacts_parms(
    x,
    ubeta,
    beta_shape,
    compute_yields_with_beta_fn,
    compute_lbeta_fn,
    compute_lc_fn,
    nsignal_params,
    nmodel_params,
    noiidxs,
    systgroupidxs,
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    global_impacts_from_jvp,
    cov,
):
    idxs_poi = tf.range(nsignal_params, dtype=tf.int64)
    idxs_noi = tf.constant(nmodel_params + noiidxs, dtype=tf.int64)
    idxsout = tf.concat([idxs_poi, idxs_noi], axis=0)

    dexpdx = tf.one_hot(idxsout, depth=cov.shape[0], dtype=cov.dtype)
    cov_dexpdx = tf.matmul(cov, dexpdx, transpose_b=True)

    var_total = tf.gather(tf.linalg.diag_part(cov), idxsout)

    impacts_beta0_total, impacts_beta0_process, var_beta0 = None, None, None
    if bin_by_bin_stat:
        impacts_beta0_total, impacts_beta0_process, var_beta0 = _compute_beta0_impacts(
            x,
            ubeta,
            beta_shape,
            compute_yields_with_beta_fn,
            compute_lbeta_fn,
            global_impacts_from_jvp,
            bin_by_bin_stat_mode,
            cov_dexpdx,
            profile=True,
            pdexpdbeta=None,
            pd2ldbeta2_pdexpdbeta=None,
        )

    impacts_x0 = _compute_global_impacts_x0(x, compute_lc_fn, cov_dexpdx)
    impacts_theta0 = tf.transpose(impacts_x0[nmodel_params:])

    impacts_theta0_sq = tf.square(impacts_theta0)
    var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)
    var_nobs = var_total - var_theta0
    if bin_by_bin_stat:
        var_nobs -= var_beta0

    impacts_grouped = _compute_grouped_impacts(
        bin_by_bin_stat,
        bin_by_bin_stat_mode,
        systgroupidxs,
        impacts_theta0_sq,
        tf.sqrt(var_nobs),
        impacts_beta0_total,
        impacts_beta0_process,
    )

    return impacts_theta0, impacts_grouped


def global_impacts_obs(
    x,
    ubeta,
    beta_shape,
    compute_yields_with_beta_fn,
    compute_lbeta_fn,
    compute_lc_fn,
    nsignal_params,
    nmodel_params,
    systgroupidxs,
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    global_impacts_from_jvp,
    cov_dexpdx,
    expvar_flat,
    expvar_shape,
    profile,
    pdexpdbeta=None,
    pd2ldbeta2_pdexpdbeta=None,
    prefit_unconstrained_nuisance_uncertainty=0.0,
):
    """
    Global impacts on observable bins, used inside _expected_with_variance.

    Unlike global_impacts_parms, this variant uses the already-computed cov_dexpdx
    and expvar_flat from the observable Jacobian, and includes the mapping contribution
    via pdexpdbeta / pd2ldbeta2_pdexpdbeta.

    The fully general contribution to the covariance matrix for a factorized likelihood
    L = sum_i L_i can be written as:
        cov_i = dexpdx @ cov_x @ d2L_i/dx2 @ cov_x @ dexpdx.T
    This is totally general and always adds up to the total covariance matrix.

    This can be factorized into impacts only if the individual contributions are rank 1.
    This is not the case in general for the data stat uncertainties, in particular where
    postfit nexpected != nobserved and nexpected is not a linear function of the poi's and
    nuisance parameters x.

    For the systematic and MC stat uncertainties this is equivalent to the more conventional
    global impact calculation (and without needing to insert the uncertainty on the global
    observables "by hand", which can be non-trivial beyond the Gaussian case).
    """
    # protect against inconsistency
    # FIXME this should be handled more generally e.g. through modification of
    # the constraintweights for prefit vs postfit, though special handling of the zero
    # uncertainty case would still be needed
    if (not profile) and prefit_unconstrained_nuisance_uncertainty != 0.0:
        raise NotImplementedError(
            "Global impacts calculation not implemented for prefit case where prefitUnconstrainedNuisanceUncertainty != 0."
        )

    impacts_beta0_total, impacts_beta0_process, var_beta0 = None, None, None
    if bin_by_bin_stat:
        impacts_beta0_total, impacts_beta0_process, var_beta0 = _compute_beta0_impacts(
            x,
            ubeta,
            beta_shape,
            compute_yields_with_beta_fn,
            compute_lbeta_fn,
            global_impacts_from_jvp,
            bin_by_bin_stat_mode,
            cov_dexpdx,
            profile,
            pdexpdbeta,
            pd2ldbeta2_pdexpdbeta,
        )

    impacts_x0 = _compute_global_impacts_x0(x, compute_lc_fn, cov_dexpdx)
    impacts_theta0 = tf.transpose(impacts_x0[nmodel_params:])

    impacts_theta0_sq = tf.square(impacts_theta0)
    var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)
    var_nobs = expvar_flat - var_theta0
    if bin_by_bin_stat:
        var_nobs -= var_beta0

    impacts_grouped = _compute_grouped_impacts(
        bin_by_bin_stat,
        bin_by_bin_stat_mode,
        systgroupidxs,
        impacts_theta0_sq,
        tf.sqrt(var_nobs),
        impacts_beta0_total,
        impacts_beta0_process,
    )

    impacts = tf.reshape(impacts_theta0, [*expvar_shape, impacts_theta0.shape[-1]])
    impacts_grouped = tf.reshape(
        impacts_grouped, [*expvar_shape, impacts_grouped.shape[-1]]
    )

    return impacts, impacts_grouped


def _gaussian_global_impacts(
    dxdtheta0,
    dxdnobs,
    dxdbeta0,
    vartheta0,
    varnobs,
    varbeta0,
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    beta_shape,
    systgroupidxs,
    data_cov_inv=None,
):
    if data_cov_inv is not None:
        data_cov = tf.linalg.inv(data_cov_inv)
        # equivalent to tf.linalg.diag_part(dxdnobs @ data_cov @ tf.transpose(dxdnobs)) but avoiding computing full matrix
        data_stat = tf.einsum("ij,jk,ik->i", dxdnobs, data_cov, dxdnobs)
    else:
        data_stat = tf.reduce_sum(tf.square(dxdnobs) * varnobs, axis=-1)

    impacts_data_stat = tf.sqrt(data_stat)

    if bin_by_bin_stat:
        var_beta0 = tf.reduce_sum(
            tf.reshape(tf.square(dxdbeta0), (-1, *beta_shape)) * varbeta0, axis=1
        )

        impacts_beta0_process = None
        if bin_by_bin_stat_mode == "full":
            impacts_beta0_process = tf.sqrt(var_beta0)
            var_beta0 = tf.reduce_sum(var_beta0, axis=-1)

        impacts_beta0_total = tf.sqrt(var_beta0)

        impacts_grouped = tf.stack([impacts_data_stat, impacts_beta0_total], axis=-1)
        if bin_by_bin_stat_mode == "full":
            impacts_grouped = tf.concat(
                [impacts_grouped, impacts_beta0_process], axis=-1
            )
    else:
        impacts_grouped = impacts_data_stat

    if len(systgroupidxs):
        dxdtheta0_squared = tf.square(dxdtheta0) * vartheta0

        impacts_grouped_syst = tf.map_fn(
            lambda idxs: _compute_global_impact_group(dxdtheta0_squared, idxs),
            tf.ragged.constant(systgroupidxs, dtype=tf.int64),
            fn_output_signature=tf.TensorSpec(
                shape=(dxdtheta0_squared.shape[0],), dtype=tf.float64
            ),
        )
        impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
        impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

    return dxdtheta0, impacts_grouped


def gaussian_global_impacts_parms(
    dxdtheta0,
    dxdnobs,
    dxdbeta0,
    vartheta0,
    varnobs,
    varbeta0,
    nsignal_params,
    nmodel_params,
    noiidxs,
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    beta_shape,
    systgroupidxs,
    data_cov_inv=None,
):
    # compute impacts for pois and nois
    dxdtheta0 = _gather_poi_noi_vector(
        dxdtheta0, noiidxs, nsignal_params, nmodel_params
    )
    dxdnobs = _gather_poi_noi_vector(dxdnobs, noiidxs, nsignal_params, nmodel_params)
    dxdbeta0 = _gather_poi_noi_vector(dxdbeta0, noiidxs, nsignal_params, nmodel_params)

    return _gaussian_global_impacts(
        dxdtheta0,
        dxdnobs,
        dxdbeta0,
        vartheta0,
        varnobs,
        varbeta0,
        bin_by_bin_stat,
        bin_by_bin_stat_mode,
        beta_shape,
        systgroupidxs,
        data_cov_inv,
    )


def gaussian_global_impacts_obs(
    dndtheta0,
    dndnobs,
    dndbeta0,
    vartheta0,
    varnobs,
    varbeta0,
    bin_by_bin_stat,
    bin_by_bin_stat_mode,
    beta_shape,
    systgroupidxs,
    data_cov_inv=None,
):
    return _gaussian_global_impacts(
        dndtheta0,
        dndnobs,
        dndbeta0,
        vartheta0,
        varnobs,
        varbeta0,
        bin_by_bin_stat,
        bin_by_bin_stat_mode,
        beta_shape,
        systgroupidxs,
        data_cov_inv,
    )
