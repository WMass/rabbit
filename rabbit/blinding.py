"""Blinding for the rabbit fitter.

Owns everything about the offsets: the deterministic per-parameter draw, the
offset TF Variables, arming and disarming, the hot-path application, and the
two warnings about configurations where the offsets do not hide what they are
meant to hide.

The invariant is stated here once instead of in comments spread over four
Fitter methods: ``Fitter.x`` is the INTERNAL (blinded) coordinate, and
``Fitter.get_poi()`` / ``get_theta()`` are the PHYSICAL values the param model
and the likelihood see. The offset lives in this object and in the difference
between those two frames only -- it is never written into ``x``, so nothing
that reads the fit's own state can subtract a public quantity and recover it
(see :meth:`Blinding._arm`).

Constructed once per parameter layout by ``Fitter.init_fit_parms``, on the
:class:`rabbit.bbstat.bbstat.BinByBinStat` pattern: :attr:`Blinding.enabled`
mirrors the Fitter's ``do_blinding`` and callers branch on it, and a disabled
instance owns no Variables, draws nothing and applies nothing.
"""

import hashlib

import numpy as np
import tensorflow as tf
from wums import logging

from rabbit.common import match_regexp_params

logger = logging.child_logger(__name__)


# Standard deviation of the deterministic blinding draw. The multiplicative
# form is scale free, so this is only meaningful for the additive form, where
# the smearing distribution is BLINDING_DRAW_STD * blind_additive_scale wide in
# the parameter's own fit units.
BLINDING_DRAW_STD = 5.0


class _OffsetApplication:
    """How the armed offsets enter the physical parameters.

    Shared by :class:`Blinding`, whose offsets are the Variables it owns, and
    :class:`BlindingView`, whose offsets are the per-call tensors
    ``rabbit.sharding`` pins on each shard evaluator. Written once so the
    shards cannot apply blinding differently from the single-device Fitter:
    what differs between them is which tensors are read, never the formula.

    Both are identity when blinding is off, so the callers in ``Fitter`` need
    no branch of their own.
    """

    def apply_poi(self, xpoi):
        """Offset the POI coordinate (additive, in the POI's own fit units)."""
        if not self.enabled:
            return xpoi
        return xpoi + self.offsets_poi_add

    def apply_theta(self, theta):
        """Offset the nuisance coordinate; only NOIs have a non-zero entry."""
        if not self.enabled:
            return theta
        return theta + self.offsets_theta


class BlindingView(_OffsetApplication):
    """Offset holder for the duck-typed shard evaluators.

    ``rabbit.sharding`` cannot hand the shards the Blinding object itself:
    XLA-compiled shard functions cannot read a tf.Variable resident on
    another device, so the offset VALUES are threaded in as tensors and
    pinned here per call, the same way ``x`` is. Carries no draw and no
    Variables -- the values come from the parent Fitter's :class:`Blinding`.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.offsets_poi_add = None
        self.offsets_theta = None


class Blinding(_OffsetApplication):
    """Deterministic blinding offsets for the POIs and the blinded NOIs.

    Constructed once per parameter layout by ``Fitter.init_fit_parms``, and
    created DISARMED: the offsets are drawn here but only assigned into the
    Variables by :meth:`arm`. A layout change re-creates the object, which is
    why the saturated-test path in ``bin/rabbit_fit.py`` re-arms after it
    re-runs ``init_fit_parms``.

    Parameters
    ----------
    indata : FitInputData
        Loaded input tensor; ``nsyst``, ``systs``, ``noiidxs``, ``data_obs``
        and ``dtype`` are read.
    param_model : ParamModel
        The model whose POIs are blinded. Read for ``npoi`` / ``params`` and
        ``allowNegativeParam``, plus the optional declarations
        ``blind_additive_scale`` and ``blind_exempt`` /
        ``blind_exempt_params``.
    enabled : bool
        Mirrors the Fitter's ``do_blinding``.
    unblind : list of str
        ``--unblind`` expressions: parameters that keep their true frame.
    blinding_group : list of str
        ``--blindingGroup`` expressions: parameters sharing one offset.
    """

    def __init__(
        self,
        indata,
        param_model,
        *,
        enabled=False,
        unblind=[],
        blinding_group=[],
    ):
        self.indata = indata
        self.param_model = param_model
        self.enabled = enabled

        # Armed offsets (Variables, read on the hot path) and the values
        # arming assigns into them (numpy, drawn once). Left None when
        # disabled: nothing may read them without checking enabled.
        self.offsets_poi_add = None
        self.offsets_theta = None
        self.values_poi_add = None
        self.values_theta = None

        if not self.enabled:
            return

        self.offsets_theta = tf.Variable(
            tf.zeros([self.indata.nsyst], dtype=self.indata.dtype),
            trainable=False,
            name="offset_theta",
        )
        # POI offsets. Additive is the only form: a translation has unit
        # Jacobian, so the covariance, the uncertainties and every impact
        # come out EXACTLY unblinded, and applied before the positivity
        # transform it composes with squared storage too. The
        # multiplicative form divided all of those by the random factor --
        # leaving only relative uncertainties usable, and making the
        # reported sigma itself a channel for the offset, since
        # sigma_true / sigma_reported WAS the offset.
        self.offsets_poi_add = tf.Variable(
            tf.zeros([self.param_model.npoi], dtype=self.indata.dtype),
            trainable=False,
            name="offset_poi_add",
        )
        self._init_values(unblind, blinding_group)

    def _arm(self, blind=True):
        """Arm or disarm the offsets. Does NOT touch the fit coordinate.

        Private: arm through ``Fitter.set_blinding_offsets``, which follows
        arming with the check that the blinded start can be evaluated.

        Blinding is a change of variables: ``Fitter.x`` is the internal
        (blinded) coordinate and ``Fitter.get_x()`` is the physical value the
        model and the likelihood see. Compensating ``x`` when the offsets
        change would hold the physical point fixed, which is superficially
        attractive -- the fit would then always open at the start value the
        model declared.

        It is not done, because the compensation MATERIALISES THE SECRET.
        ``x`` would be set to ``x0default`` mapped through the offsets, and
        ``x0default`` is public: it is the default the model declares. Anything
        that then observes ``x`` before the minimiser runs -- a saved prefit
        parameter vector, a debug dump, a debugger, or any output added later --
        recovers the offset by one subtraction, and the offset together with the
        postfit value is the unblinded result. Blinding has to survive someone
        looking at the fit's own state, so the offset is never written into a
        coordinate that a known quantity can be subtracted from.

        The cost is that the fit opens at the blinded frame's default rather
        than the declared one, i.e. at a different starting point than an
        unblinded run. That is accepted: the minimiser converges to the same
        minimum, so the physical result is unchanged -- only the path to it
        differs. See ``tests/test_blinding_start_point.py``, which fits the same
        data armed and disarmed and requires the physical minimum to agree.

        Callers that need the declared physical start must run disarmed.

        Whether the model can be EVALUATED at the armed start is a question
        about the likelihood rather than about the offsets, so the Fitter asks
        it: see ``Fitter.set_blinding_offsets``, which is what callers use.
        """
        if not self.enabled:
            return
        if blind:
            self.offsets_poi_add.assign(self.values_poi_add)
            self.offsets_theta.assign(self.values_theta)
        else:
            # zeros_like the drawn values, which were built at exactly the
            # Variables' shape and dtype
            self.offsets_poi_add.assign(np.zeros_like(self.values_poi_add))
            self.offsets_theta.assign(np.zeros_like(self.values_theta))

    def warn_if_weak(self, variances):
        """Say so when the smearing was too narrow to hide the POI it blinded.

        The yardstick is the MEASURED uncertainty, which is the only thing that
        decides whether a value is actually hidden: an offset of half a sigma
        leaves the truth recoverable whatever units it is in. Judged against
        ``cov``, so it costs nothing -- the driver has already computed it --
        and it works for any number of POIs, where a prefit Asimov sigma would
        need one linear solve per POI before the fit had even started.

        This replaces a prefit check against ``prior_sigmas`` that could not
        work: no baseline model declares that attribute, and a prior width only
        exists for a CONSTRAINED POI, whereas the physical free parameter this
        feature exists for has none. It also compared two numbers the same model
        author had written, so it could only ever report that their own
        declarations disagreed.

        Reports no numbers, for the same reason the rest of this machinery does
        not: the offset and the smearing width each give the other away. The
        verdict is a boolean about sigma, and sigma is not the secret.

        Blinded-ness is read off the offsets rather than remembered from
        the draw: a POI left out by --unblind, or exempted by
        the model, has an offset of exactly zero.

        Reads the variance vector the driver already has, the POI offsets and
        the model's declared scales -- all [nparams]-sized, with no bin axis
        anywhere, so a sharded fit reaches it unchanged.

        Takes the per-parameter VARIANCE vector rather than a covariance matrix,
        so it works wherever the driver has one. With a Hessian that is the full
        diagonal; under --noHessian it is the POI and NOI entries alone, solved
        for by edmval_cov_rows_hessfree, with the rest left NaN -- which is all
        this needs, and non-finite entries are skipped. Only --noEDM computes
        neither, and there the driver says the check was skipped.
        """
        if not self.enabled or not self.param_model.npoi:
            return
        npoi = self.param_model.npoi

        sigma = np.sqrt(np.asarray(variances)[:npoi])
        offsets = np.asarray(self.offsets_poi_add)[:npoi]
        scales = np.broadcast_to(
            np.asarray(
                getattr(self.param_model, "blind_additive_scale", 1.0),
                dtype=np.float64,
            ),
            (npoi,),
        )

        weak = []
        for i in range(npoi):
            if offsets[i] == 0.0:  # not blinded: --unblind, or model-exempt
                continue
            if not np.isfinite(sigma[i]) or sigma[i] <= 0.0:
                continue
            if BLINDING_DRAW_STD * scales[i] < 5.0 * sigma[i]:
                param = self.param_model.params[i]
                weak.append(param.decode() if isinstance(param, bytes) else str(param))

        if weak:
            logger.warning(
                "Blinding was too narrow to hide "
                f"{len(weak)} of {npoi} POIs: the smearing is less than 5 sigma "
                "of the uncertainty this fit measured, so the true value stays "
                f"recoverable to within a few sigma of the blinded one. "
                f"Affected: {', '.join(weak)}. Raise "
                "param_model.blind_additive_scale for these and refit; the "
                "result already written is not safely blinded."
            )

    def _init_values(
        self, unblind_parameter_expressions=[], blinding_group_expressions=[]
    ):
        logger.debug(f"Unblind parameters with {unblind_parameter_expressions}")
        all_param_names = [
            *self.param_model.params[: self.param_model.npoi],
            *[self.indata.systs[i] for i in self.indata.noiidxs],
        ]
        unblind_parameters = match_regexp_params(
            unblind_parameter_expressions, all_param_names
        )

        # Parameters the MODEL declares exempt, on top of whatever --unblind
        # asked for. These are auxiliary quantities that are not results (the
        # saturated test's per-bin scales), so blinding them buys nothing and
        # costs them their declared start point.
        exempt = getattr(self.param_model, "blind_exempt_params", None)
        if exempt is None and getattr(self.param_model, "blind_exempt", False):
            exempt = self.param_model.params[: self.param_model.npoi]
        if exempt is not None and len(exempt):
            unblind_parameters = list(unblind_parameters) + [
                q for q in exempt if q not in unblind_parameters
            ]
        # unblinding is sensitive: always report exactly which parameters the
        # expressions resolved to, so an over-broad pattern is visible
        if unblind_parameters:
            unblind_names = [
                p.decode() if isinstance(p, bytes) else str(p)
                for p in unblind_parameters
            ]
            logger.info(f"Unblinding {len(unblind_names)} parameters: {unblind_names}")

        # Real data is integer in EVERY bin; an Asimov or MC card is not, but can
        # still have exactly-0.0 (empty) bins. So require all bins integral: with
        # "any" such a card would draw the real-data offsets, and a closure fit
        # against known truth would then read them off as blinded - truth.
        # This looks at the card's data_obs, not the dataset being fitted: a
        # --pseudoData fit on a real-data card shares the data fit's frame.
        is_dataobs_int = np.all(
            np.equal(self.indata.data_obs, np.floor(self.indata.data_obs))
        )

        def deterministic_random_from_string(s, mean=0.0, std=BLINDING_DRAW_STD):
            # random value with seed taken based on string of parameter name
            if isinstance(s, str):
                s = s.encode("utf-8")

            if is_dataobs_int:
                s += b"_data"

            # Hash the string
            hash = hashlib.sha256(s).hexdigest()

            seed_seq = np.random.SeedSequence(int(hash, 16))
            rng = np.random.default_rng(seed_seq)

            value = rng.normal(loc=mean, scale=std)
            return value

        # Build a map: param name -> seed string. Default is the param's own name; for
        # parameters matching a blinding group the seed is the group regex string, so all
        # members of the group share an identical deterministic offset (preserving relative
        # differences while blinding absolute values).
        if isinstance(blinding_group_expressions, str):
            blinding_group_expressions = [blinding_group_expressions]
        param_to_seed = {}
        param_to_group = {}
        for expr in blinding_group_expressions:
            matched = match_regexp_params(expr, all_param_names)
            for p in matched:
                if p in param_to_seed:
                    continue  # first matching group wins
                param_to_seed[p] = expr
                param_to_group[p] = expr

        # Refuse to run if --unblind and --blindingGroup match the same parameter:
        # the user's intent is ambiguous (unblind it, or share a group offset?), and
        # silently picking one risks accidentally unblinding a sensitive parameter.
        overlap = [p for p in unblind_parameters if p in param_to_group]
        if overlap:
            details = ", ".join(
                f"{p.decode() if isinstance(p, bytes) else p}"
                f" (group '{param_to_group[p]}')"
                for p in overlap
            )
            raise RuntimeError(
                "The following parameters match both --unblind and --blindingGroup; "
                "refusing to proceed to avoid an ambiguous (un)blinding. "
                f"Tighten the regexes to make the intent explicit: {details}"
            )

        # multiply offset to nois
        self.values_theta = np.zeros(self.indata.nsyst, dtype=np.float64)
        for i in self.indata.noiidxs:
            param = self.indata.systs[i]
            if param in unblind_parameters:
                continue
            seed = param_to_seed.get(param, param)
            logger.debug(f"Blind parameter {param} (seed='{seed}')")
            value = deterministic_random_from_string(seed)
            self.values_theta[i] = value

        # Offset the POIs, additively, in each parameter's own fit units. This
        # is the only form: see the offset Variables in __init__ for why
        # the multiplicative one was removed.
        self.values_poi_add = np.zeros(self.param_model.npoi, dtype=np.float64)
        # The additive draw is NOT scale free the way exp(N(0, 5)) is: it is an
        # absolute shift, so how well it hides depends on the POI's units. The
        # model is the only thing that knows them, so it declares the scale.
        # Default 1.0 == the historical draw.
        #
        # Scalar (one scale for all of a model's POIs) or per-POI vector, which
        # is what CompositeParamModel produces: the scale is in each
        # parameter's own units, so a composite cannot reduce its submodels'
        # declarations to a single number.
        additive_scale = np.broadcast_to(
            np.asarray(
                getattr(self.param_model, "blind_additive_scale", 1.0),
                dtype=np.float64,
            ),
            (self.param_model.npoi,),
        )
        for i in range(self.param_model.npoi):
            param = self.param_model.params[i]
            if param in unblind_parameters:
                continue
            seed = param_to_seed.get(param, param)
            logger.debug(f"Blind parameter {param} (seed='{seed}')")
            value = deterministic_random_from_string(seed)
            self.values_poi_add[i] = additive_scale[i] * value

        self._warn_if_squared_storage_leaks()

    def _warn_if_squared_storage_leaks(self):
        """Say so when the POI parameterisation leaks the value it is hiding.

        With ``allowNegativeParam=False`` the stored coordinate is
        ``sqrt(poi)``, so at the minimum ``d(poi)/dx = 2*sqrt(poi)`` and the
        reported uncertainty is ``sigma_poi / (2*sqrt(poi))`` -- a function of
        the TRUE value. The offset does not appear in it (blinding preserves
        sigma_x exactly), but anyone holding an expected sigma_poi, which an
        Asimov study gives for free, inverts it: ``sqrt(poi) = sigma_poi /
        (2*sigma_x)``, and subtracting the reported coordinate leaves the
        offset.

        A linearly stored POI (``allowNegativeParam=True``) has unit Jacobian,
        so its reported sigma is ``sigma_poi`` itself and carries nothing about
        where the minimum sits. That is the configuration blinding actually
        works in.

        A warning rather than a refusal: the squaring is the default everywhere
        in rabbit, blinding is on by default for a data fit, and refusing would
        make every existing analysis unrunnable rather than merely leaky. The
        leak also predates this code -- it is a property of reporting sqrt(poi)
        and its uncertainty, not of any offset -- so the honest thing is to
        name it and point at the fix.
        """
        if self.param_model.allowNegativeParam:
            return
        if not self.param_model.npoi:
            return
        # Nothing is leaked if nothing was blinded. --unblind and model-declared
        # exemptions leave the drawn value at exactly zero, and the standard
        # "now unblind my result" run has every POI in that state, so without
        # this the default squared Mu setup warns about an offset that does not
        # exist and pushes the reader to change parametrisation for no reason.
        if not np.any(self.values_poi_add[: self.param_model.npoi]):
            return
        logger.warning(
            "Blinding a POI stored as sqrt(poi) (allowNegativeParam=False): the "
            "reported uncertainty is sigma_poi / (2*sqrt(poi)), which depends on "
            "the true value, so an expected sigma recovers the blinded value and "
            "hence the offset. The offset itself is not in the covariance -- "
            "sigma is preserved exactly -- but the parameterisation is. Pass "
            "--allowNegativeParam for a linearly stored POI, whose reported "
            "uncertainty carries no such dependence."
        )
