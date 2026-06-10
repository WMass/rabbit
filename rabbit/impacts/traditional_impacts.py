"""
Traditional impacts
impacts from individual nuisance parameters are extracted using the "nuisance parameter fix-and-shift" method
impacts from groups of nuisance parameters are extracted using the "conditional uncertainty" method
"""

import tensorflow as tf


def _compute_impact_group(cov, v, idxs, nsignal_params=0):
    cov_reduced = tf.gather(cov[nsignal_params:, nsignal_params:], idxs, axis=0)
    cov_reduced = tf.gather(cov_reduced, idxs, axis=1)
    v_reduced = tf.gather(v, idxs, axis=1)
    invC_v = tf.linalg.solve(cov_reduced, tf.transpose(v_reduced))
    v_invC_v = tf.einsum("ij,ji->i", v_reduced, invC_v)
    return tf.sqrt(v_invC_v)


def _gather_poi_noi_vector(v, noiidxs, nsignal_params=0, nmodel_params=None):
    # nmodel_params = npoi + npou
    if nmodel_params is None:
        nmodel_params = nsignal_params
    v_poi = v[:nsignal_params]
    # protection for constained NOIs, set them to 0
    mask = (noiidxs >= 0) & (noiidxs < tf.shape(v[nmodel_params:])[0])
    safe_idxs = tf.where(mask, noiidxs, 0)
    mask = tf.cast(mask, v.dtype)
    mask = tf.reshape(
        mask,
        tf.concat([tf.shape(mask), tf.ones(tf.rank(v) - 1, dtype=tf.int32)], axis=0),
    )
    v_noi = tf.gather(v[nmodel_params:], safe_idxs) * mask
    v_gathered = tf.concat([v_poi, v_noi], axis=0)
    return v_gathered


def impacts_parms(
    cov,
    cov_stat,
    cov_stat_no_bbb,
    nsignal_params=0,
    noiidxs=[],
    systgroupidxs=[],
    nmodel_params=None,
    param_groupidxs=None,
):
    """
    Gaussian approximation
    """
    if nmodel_params is None:
        nmodel_params = nsignal_params

    # impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
    v = _gather_poi_noi_vector(cov, noiidxs, nsignal_params, nmodel_params)
    # Frozen / fixed parameters have zero variance (cov diag == 0); their impact
    # is zero, not 0/0 = nan. Guard the denominator so those columns read 0.
    sigma = tf.sqrt(tf.linalg.diag_part(cov))
    safe_sigma = tf.where(sigma > 0, sigma, tf.ones_like(sigma))
    impacts = v / tf.reshape(safe_sigma, [1, -1])

    if cov_stat_no_bbb is not None:
        # impact bin-by-bin stat
        impacts_data_stat = tf.sqrt(tf.linalg.diag_part(cov_stat_no_bbb))
        impacts_data_stat = _gather_poi_noi_vector(
            impacts_data_stat, noiidxs, nsignal_params, nmodel_params
        )
        impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))

        impacts_bbb_sq = tf.linalg.diag_part(cov_stat - cov_stat_no_bbb)
        impacts_bbb_sq = _gather_poi_noi_vector(
            impacts_bbb_sq, noiidxs, nsignal_params, nmodel_params
        )
        impacts_bbb = tf.sqrt(tf.nn.relu(impacts_bbb_sq))  # max(0,x)
        impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
        impacts_grouped = tf.concat([impacts_data_stat, impacts_bbb], axis=1)
    else:
        impacts_data_stat = tf.sqrt(tf.linalg.diag_part(cov_stat))
        impacts_data_stat = _gather_poi_noi_vector(
            impacts_data_stat, noiidxs, nsignal_params, nmodel_params
        )
        impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))
        impacts_grouped = impacts_data_stat

    if len(systgroupidxs):
        # systgroupidxs are syst-relative -> shift into full-x by nmodel_params
        # and gather from the full covariance (nsignal_params=0 path).
        impacts_grouped_syst = tf.map_fn(
            lambda idxs: _compute_impact_group(cov, v, nmodel_params + idxs, 0),
            tf.ragged.constant(systgroupidxs, dtype=tf.int32),
            fn_output_signature=tf.TensorSpec(
                shape=(impacts.shape[0],), dtype=tf.float64
            ),
        )
        impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
        impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

    if param_groupidxs:
        # ParamModel parameter groups, already in full-x space (floating-filtered).
        # Appended at the END so the grouped-impact axis is
        # [syst groups, stat, (binByBinStat), param groups].
        param_cols = [
            tf.reshape(
                _compute_impact_group(cov, v, tf.constant(g, dtype=tf.int32), 0),
                (-1, 1),
            )
            for g in param_groupidxs
        ]
        impacts_grouped = tf.concat([impacts_grouped, *param_cols], axis=1)

    return impacts, impacts_grouped
