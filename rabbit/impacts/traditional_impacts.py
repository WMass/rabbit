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


def _gather_poi_noi_vector(v, noiidxs, nsignal_params=0):
    v_poi = v[:nsignal_params]
    # protection for constained NOIs, set them to 0
    mask = (noiidxs >= 0) & (noiidxs < tf.shape(v[nsignal_params:])[0])
    safe_idxs = tf.where(mask, noiidxs, 0)
    mask = tf.cast(mask, v.dtype)
    mask = tf.reshape(
        mask,
        tf.concat([tf.shape(mask), tf.ones(tf.rank(v) - 1, dtype=tf.int32)], axis=0),
    )
    v_noi = tf.gather(v[nsignal_params:], safe_idxs) * mask
    v_gathered = tf.concat([v_poi, v_noi], axis=0)
    return v_gathered


def impacts_parms(
    cov, cov_stat, cov_stat_no_bbb, nsignal_params=0, noiidxs=[], systgroupidxs=[]
):
    """
    Gaussian approximation
    """

    # impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
    v = _gather_poi_noi_vector(cov, noiidxs, nsignal_params)
    impacts = v / tf.reshape(tf.sqrt(tf.linalg.diag_part(cov)), [1, -1])

    if cov_stat_no_bbb is not None:
        # impact bin-by-bin stat
        impacts_data_stat = tf.sqrt(tf.linalg.diag_part(cov_stat_no_bbb))
        impacts_data_stat = _gather_poi_noi_vector(
            impacts_data_stat, noiidxs, nsignal_params
        )
        impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))

        impacts_bbb_sq = tf.linalg.diag_part(cov_stat - cov_stat_no_bbb)
        impacts_bbb_sq = _gather_poi_noi_vector(impacts_bbb_sq, noiidxs, nsignal_params)
        impacts_bbb = tf.sqrt(tf.nn.relu(impacts_bbb_sq))  # max(0,x)
        impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
        impacts_grouped = tf.concat([impacts_data_stat, impacts_bbb], axis=1)
    else:
        impacts_data_stat = tf.sqrt(tf.linalg.diag_part(cov_stat))
        impacts_data_stat = _gather_poi_noi_vector(
            impacts_data_stat, noiidxs, nsignal_params
        )
        impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))
        impacts_grouped = impacts_data_stat

    if len(systgroupidxs):
        impacts_grouped_syst = tf.map_fn(
            lambda idxs: _compute_impact_group(
                cov, v[:, nsignal_params:], idxs, nsignal_params
            ),
            tf.ragged.constant(systgroupidxs, dtype=tf.int32),
            fn_output_signature=tf.TensorSpec(
                shape=(impacts.shape[0],), dtype=tf.float64
            ),
        )
        impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
        impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

    return impacts, impacts_grouped
