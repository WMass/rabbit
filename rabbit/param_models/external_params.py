"""Param model declaring fit parameters from an auxiliary bundle.

An :mod:`external likelihood term <rabbit.external_likelihood>` consumes fit
parameters *by name*, exactly like an :class:`~rabbit.unbinned.UnbinnedTerm`.
Those names have to exist in the fit parameter vector, which is built as
``[ParamModel params | systs]`` -- so something has to declare them. For an
unbinned term that is
:class:`~rabbit.param_models.unbinned_params.UnbinnedParams`, which reads the
declarations out of the term itself. An external term carries no declarations
(it is only ``g``, ``H`` and a parameter list), so this model reads them from
an ``auxiliary`` bundle written next to the term.

Bundle schema (all but ``params`` optional, see
:meth:`rabbit.tensorwriter.TensorWriter.add_auxiliary`)::

    params        list[str]           parameter names, in fit order
    defaults      float64 (n,)        starting values          (default 0)
    prior_sigmas  float64 (n,)        Gaussian prior widths, NaN/0 = free
    prior_means   float64 (n,)        prior centers            (default: defaults)
    is_poi        int (n,)            1 = report as a POI      (default 0)

Like ``UnbinnedParams`` it does not scale any process yield: :meth:`compute`
returns ones. Combine it with other models when the same fit also has binned
channels or unbinned terms::

    --paramModel UnbinnedParams --paramModel ExternalParams

The parameter names of the combined models must be *disjoint* (the fitter
requires unique names), so a card that declares some parameters through an
unbinned term must leave those out of this bundle.

CLI::

    --paramModel ExternalParams                      # bundle "external_params"
    --paramModel ExternalParams bundle:global_params
    --paramModel ExternalParams poi:bfield_mode0     # override the POI flags
"""

import numpy as np
import tensorflow as tf
from wums import logging

from rabbit.param_models.param_model import ParamModel

logger = logging.child_logger(__name__)

DEFAULT_BUNDLE = "external_params"


class ExternalParams(ParamModel):
    """Declare the parameters listed in an auxiliary bundle.

    ``npoi`` counts the parameters flagged as POIs in the bundle (reported as
    POIs, targets of ``--doImpacts`` and the contour scans); everything else
    becomes a model nuisance (``npou``). ``allowNegativeParam`` is forced to
    True: these are physical quantities that can be negative (a field-mode
    coefficient, a material scale), not signal strengths, so the fitter's
    ``sqrt`` storage transform must not be applied.
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        bundle = DEFAULT_BUNDLE
        poi_names = None
        for a in args:
            if a.startswith("bundle:"):
                bundle = a[len("bundle:") :]
            elif a.startswith("poi:"):
                poi_names = [s for s in a[len("poi:") :].split(",") if s]
            else:
                raise ValueError(
                    f"ExternalParams: unknown argument '{a}'; expected "
                    "'bundle:<name>' or 'poi:<names>'"
                )
        return cls(indata, bundle=bundle, poi_names=poi_names, **kwargs)

    def __init__(self, indata, bundle=DEFAULT_BUNDLE, poi_names=None, **kwargs):
        self.indata = indata

        auxiliary = getattr(indata, "auxiliary", {}) or {}
        if bundle not in auxiliary:
            raise ValueError(
                f"ExternalParams: no auxiliary bundle '{bundle}' in the input; "
                f"have {sorted(auxiliary)}. Write it with "
                "TensorWriter.add_auxiliary()."
            )
        data = auxiliary[bundle]
        if "params" not in data:
            raise ValueError(
                f"ExternalParams: auxiliary bundle '{bundle}' has no 'params' "
                f"dataset, found {sorted(data)}"
            )

        names = [str(p) for p in data["params"]]
        n = len(names)
        if n == 0:
            raise ValueError(f"ExternalParams: bundle '{bundle}' declares no params")
        if len(set(names)) != n:
            raise ValueError(
                f"ExternalParams: bundle '{bundle}' has duplicate parameter names"
            )

        def _get(key, default, dtype=np.float64):
            if key not in data:
                return np.full(n, default, dtype=dtype)
            a = np.asarray(data[key], dtype=dtype).ravel()
            if a.shape != (n,):
                raise ValueError(
                    f"ExternalParams: bundle '{bundle}' dataset '{key}' has "
                    f"shape {a.shape}, expected {(n,)}"
                )
            return a

        defaults = _get("defaults", 0.0)
        sigmas = _get("prior_sigmas", np.nan)
        means = _get("prior_means", np.nan)
        means = np.where(np.isfinite(means), means, defaults)
        is_poi = _get("is_poi", 0, dtype=np.int64).astype(bool)

        if poi_names is not None:
            unknown = set(poi_names) - set(names)
            if unknown:
                raise ValueError(
                    f"ExternalParams: poi:{sorted(unknown)} not among the "
                    f"parameters of bundle '{bundle}'"
                )
            is_poi = np.array([nm in poi_names for nm in names])

        order = np.concatenate([np.where(is_poi)[0], np.where(~is_poi)[0]])
        ordered = [names[i] for i in order]

        self.npoi = int(is_poi.sum())
        self.npou = n - self.npoi
        self.params = np.array([nm.encode() for nm in ordered])
        self.allowNegativeParam = True
        # The model itself is linear (it scales nothing), but the external
        # term it declares parameters for is quadratic, which the Fitter's
        # own external-term handling accounts for. Declaring True keeps the
        # composite honest for a purely quadratic card.
        self.is_linear = True

        self.xparamdefault = tf.constant(defaults[order], dtype=indata.dtype)
        self.prior_sigmas = sigmas[order]
        self.prior_means = means[order]

        n_prior = int(np.sum(np.isfinite(self.prior_sigmas) & (self.prior_sigmas > 0)))
        logger.info(
            f"ExternalParams['{bundle}']: {self.npoi} POI(s), {self.npou} "
            f"nuisance(s), {n_prior} with a Gaussian prior"
        )

    def compute(self, param, full=False):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        return tf.reshape(rnorm, [1, -1])
