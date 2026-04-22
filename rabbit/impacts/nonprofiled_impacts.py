"""
Nonprofiled impacts of varying frozen parameters and repeating the fit.
"""

import tensorflow as tf
from wums import logging

from rabbit import tfhelpers as tfh

logger = logging.child_logger(__name__)


@tf.function
def _envelope(values):
    zeros = tf.zeros((tf.shape(values)[0], tf.shape(values)[-1]), dtype=values.dtype)
    vmin = tf.reduce_min(values, axis=1)
    vmax = tf.reduce_max(values, axis=1)
    lower = -tf.sqrt(tf.reduce_sum(tf.minimum(zeros, vmin) ** 2, axis=0))
    upper = tf.sqrt(tf.reduce_sum(tf.maximum(zeros, vmax) ** 2, axis=0))
    return tf.stack([lower, upper])


def nonprofiled_impacts_parms(
    x,
    theta0,
    frozen_indices,
    frozen_params,
    constraintweights,
    systgroups,
    systgroupidxs,
    nparams,
    minimize_fn,
    diagnostics=False,
    loss_val_grad_hess_fn=None,
    unconstrained_err=1.0,
):
    """
    Args:
        x: TF Variable holding all fit parameters (POIs + nuisances).
        theta0: TF Variable list of nuisance parameter central values.
        frozen_indices: indices (into x) of the frozen parameters.
        frozen_params: names of the frozen parameters.
        constraintweights: constraint weights for each nuisance parameter.
        systgroups: systematic group names.
        systgroupidxs: per-group lists of nuisance parameter indices.
        nparams: total number of model parameters (nparams + npou); offset from x index to theta0 index.
        minimize_fn: callable that runs the fit (no arguments).
        diagnostics: if True, log EDM after each minimization (requires loss_val_grad_hess_fn).
        loss_val_grad_hess_fn: callable returning (val, grad, hess); used only when diagnostics=True.
        unconstrained_err: sigma to use for unconstrained nuisance parameters.
    """
    x_tmp = tf.identity(x.value())
    x_tmp_tiled = tf.tile(tf.reshape(x_tmp, [1, 1, -1]), [len(frozen_indices), 2, 1])
    nonprofiled_impacts = tf.Variable(x_tmp_tiled)

    theta0_tmp = tf.identity(theta0.value())

    err_theta = tf.where(
        constraintweights == 0.0,
        unconstrained_err,
        tf.math.reciprocal(constraintweights),
    )

    for i, idx in enumerate(frozen_indices):
        logger.info(f"Now at parameter {frozen_params[i]}")

        for j, sign in enumerate((1, -1)):
            variation = sign * err_theta[idx - nparams] + theta0_tmp[idx - nparams]
            # vary the non-profiled parameter
            theta0[idx - nparams].assign(variation)
            x[idx].assign(
                variation
            )  # this should not be needed but should accelerate the minimization
            # minimize
            minimize_fn()
            if diagnostics:
                _, grad, hess = loss_val_grad_hess_fn()
                edmval, _ = tfh.edmval_cov(grad, hess)
                logger.info(f"edmval: {edmval}")
            # difference w.r.t. nominal fit
            diff = x_tmp - x.value()
            nonprofiled_impacts[i, j].assign(diff)
            x.assign(x_tmp)

        # back to original value
        theta0[idx - nparams].assign(theta0_tmp[idx - nparams])

    impact_group_names = []
    impact_groups = []

    for group, idxs in zip(systgroups, systgroupidxs):
        frozen_mask = tf.reduce_any(
            tf.equal(
                tf.reshape(tf.cast(frozen_indices, tf.int32), [-1, 1]),
                tf.reshape(idxs, [1, -1]),
            ),
            axis=1,
        )
        frozen_idxs = tf.where(frozen_mask)
        if tf.size(frozen_idxs) > 0:
            selected_impacts = tf.gather(nonprofiled_impacts, frozen_idxs[:, 0])
            impact_groups.append(_envelope(selected_impacts))
            impact_group_names.append(group)

    # Add total envelope
    impact_groups.append(_envelope(nonprofiled_impacts))
    impact_group_names.append(b"Total")

    impact_groups = tf.stack(impact_groups)

    return (
        frozen_params,
        nonprofiled_impacts.numpy(),
        impact_group_names,
        impact_groups.numpy(),
    )
