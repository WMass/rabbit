"""Param model declaring the parameters of the datacard's unbinned terms.

An :class:`~rabbit.unbinned.UnbinnedTerm` consumes fit parameters *by name*
(exactly like an external likelihood term). Those names still have to exist in
the fit parameter vector, which is built as ``[ParamModel params | systs]`` --
so something has to declare them. This model does, reading the declarations
(name, starting value, Gaussian prior, POI flag) straight out of the
``unbinned_terms`` group of the input file, so the parameter list can never
drift from the term that uses it.

It does not scale any process yield: :meth:`compute` returns ones, like
``Ones``. Combine it with another model (e.g. ``--paramModel Mu --paramModel
UnbinnedParams``) when the same fit also has binned channels with free
normalisations.

CLI::

    --paramModel UnbinnedParams

Optional tokens::

    --paramModel UnbinnedParams poi:alpha,mZ    # override which are POIs
    --paramModel UnbinnedParams terms:jpsi,ups  # only these terms' parameters
"""

import numpy as np
import tensorflow as tf
from wums import logging

from rabbit.param_models.param_model import ParamModel

logger = logging.child_logger(__name__)


class UnbinnedParams(ParamModel):
    """Declare the union of the parameters used by the unbinned terms.

    Parameters shared between terms (e.g. one common momentum scale for
    several resonance channels) are declared once; their declarations must
    agree between terms, which is checked here rather than silently taking
    the first.

    ``npoi`` counts the parameters flagged as POIs in the datacard (reported
    as POIs, targets of ``--doImpacts`` and the contour scans); everything
    else becomes a model nuisance (``npou``). ``allowNegativeParam`` is
    forced to True: the parameters are physical quantities that can be
    negative (a momentum-scale offset, a mass shift), not signal strengths,
    so the fitter's ``sqrt`` storage transform must not be applied.
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        poi_names = None
        terms = None
        for a in args:
            if a.startswith("poi:"):
                poi_names = [s for s in a[len("poi:") :].split(",") if s]
            elif a.startswith("terms:"):
                terms = [s for s in a[len("terms:") :].split(",") if s]
            else:
                raise ValueError(
                    f"UnbinnedParams: unknown argument '{a}'; expected "
                    "'poi:<names>' or 'terms:<names>'"
                )
        return cls(indata, poi_names=poi_names, terms=terms, **kwargs)

    def __init__(self, indata, poi_names=None, terms=None, **kwargs):
        self.indata = indata

        unbinned_terms = getattr(indata, "unbinned_terms", [])
        if terms is not None:
            missing = set(terms) - {t.name for t in unbinned_terms}
            if missing:
                raise ValueError(
                    f"UnbinnedParams: no unbinned term(s) {sorted(missing)} in "
                    f"the input; have {[t.name for t in unbinned_terms]}"
                )
            unbinned_terms = [t for t in unbinned_terms if t.name in terms]
        if not unbinned_terms:
            raise ValueError(
                "UnbinnedParams: the input file declares no unbinned terms. "
                "Write them with TensorWriter.add_unbinned_term()."
            )

        decl = {}
        order = []
        for term in unbinned_terms:
            for i, name in enumerate(term.param_names):
                entry = (
                    float(term.param_defaults[i]),
                    float(term.param_prior_sigmas[i]),
                    float(term.param_prior_means[i]),
                    int(term.param_is_poi[i]),
                )
                if name in decl:
                    if not _same(decl[name], entry):
                        raise ValueError(
                            f"UnbinnedParams: parameter '{name}' is declared "
                            f"differently by two terms: {decl[name]} vs {entry}"
                        )
                else:
                    decl[name] = entry
                    order.append(name)

        if poi_names is not None:
            unknown = set(poi_names) - set(order)
            if unknown:
                raise ValueError(
                    f"UnbinnedParams: poi:{sorted(unknown)} not among the "
                    f"unbinned parameters {order}"
                )
            decl = {k: (v[0], v[1], v[2], int(k in poi_names)) for k, v in decl.items()}

        pois = [n for n in order if decl[n][3]]
        pous = [n for n in order if not decl[n][3]]
        names = pois + pous

        self.npoi = len(pois)
        self.npou = len(pous)
        self.params = np.array([n.encode() for n in names])
        self.allowNegativeParam = True
        # The unbinned NLL is not quadratic in the parameters; declaring this
        # keeps CompositeParamModel from advertising a linear likelihood. The
        # Fitter additionally disables its Cholesky shortcut whenever unbinned
        # terms are present, see Fitter.is_linear.
        self.is_linear = False

        self.xparamdefault = tf.constant(
            np.array([decl[n][0] for n in names]), dtype=indata.dtype
        )
        sigmas = np.array([decl[n][1] for n in names])
        self.prior_sigmas = sigmas
        self.prior_means = np.array([decl[n][2] for n in names])

        n_prior = int(np.sum(np.isfinite(sigmas) & (sigmas > 0)))
        logger.info(
            f"UnbinnedParams: {self.npoi} POI(s) {pois}, {self.npou} nuisance(s) "
            f"{pous}, {n_prior} with a Gaussian prior"
        )

    def compute(self, param, full=False):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        return tf.reshape(rnorm, [1, -1])


def _same(a, b):
    for x, y in zip(a, b):
        if isinstance(x, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True
