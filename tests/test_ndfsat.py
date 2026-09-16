"""The saturated ndf must charge FREE parameters, wherever they are declared.

``ndfsat`` was ``nbins - param_model.nparams - indata.nsystnoconstraint``: every
model parameter charged, but only the UNCONSTRAINED card systematics. That
distinguishes parameters by where they are declared, which is not a statistical
property. A ``pdfEig23`` model parameter with a sigma = 1 Gaussian prior is the
same object as a ``pdf23...`` card nuisance with a sigma = 1 Gaussian prior, and
must be charged the same way.

WHY ZERO IS THE RIGHT CHARGE for a constrained parameter: it adds one parameter
AND one pseudo-measurement, so it costs net zero degrees of freedom. That is
already why rabbit charges constrained card nuisances nothing.

The rule is therefore ``ndf = nbins - (number of free parameters)``, counted off
``Fitter.cw`` -- the one vector that records which parameters are constrained,
over ``[ParamModel params | systs]``, and the same one ``_compute_lc``
penalises, so the count cannot drift from what the likelihood actually does.

The last test is the backward-compatibility claim: for a model that declares no
priors the new expression is EXACTLY the old one, so nothing moves for any
analysis that does not put priors on model parameters.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.param_model import ParamModel

NBINS = 20
NPOU = 6  # model nuisances, some of which carry priors below


class PriorModel(ParamModel):
    """One free POI plus NPOU model nuisances, ``n_priored`` of them constrained.

    Shaped after a real theory param model: a single free parameter of interest
    and a block of nuisances the model itself constrains at sigma = 1.
    """

    def __init__(self, indata, n_priored):
        super().__init__(indata)
        self.npoi = 1
        self.npou = NPOU
        self.params = np.array([b"alphaS"] + [f"lam{i}".encode() for i in range(NPOU)])
        self.allowNegativeParam = True
        self.is_linear = False
        self.xparamdefault = tf.constant(
            np.concatenate([[0.118], np.zeros(NPOU)]), dtype=indata.dtype
        )
        if n_priored:
            # NaN = free, finite and > 0 = constrained at that width. The POI is
            # left free, as a physics POI must be.
            sigmas = np.full(self.nparams, np.nan)
            sigmas[1 : 1 + n_priored] = 1.0
            self.prior_sigmas = sigmas

    def compute(self, param, full=False):
        col = tf.reshape(
            1.0 + 0.1 * (param[0] - 0.118) + 0.01 * tf.reduce_sum(param[1:]), [1, 1]
        )
        return tf.concat(
            [col, tf.ones([1, self.indata.nproc - 1], dtype=col.dtype)], axis=1
        )


FREE_PARAM_NAME = "alphaS"  # the POI: free in every model here


class NoPriorModel(PriorModel):
    """The backward-compatibility case: a model that declares no priors at all."""

    def __init__(self, indata):
        super().__init__(indata, 0)
        assert not hasattr(self, "prior_sigmas")


def make_tensor(path, n_free_systs):
    """``n_free_systs`` unconstrained card systematics plus two constrained ones.

    Both kinds have to be present for the test to say anything: the bug was
    charging the two blocks by different rules.
    """
    np.random.seed(1234)
    ax = hist.axis.Regular(NBINS, -5, 5, name="x")
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_data.fill(
        np.concatenate([np.random.normal(0, 1, 8000), np.random.uniform(-5, 5, 4000)])
    )
    h_sig.fill(np.random.normal(0, 1, 8000))
    h_bkg.fill(np.random.uniform(-5, 5, 4000))

    w = tensorwriter.TensorWriter()
    w.add_channel(h_data.axes, "ch0")
    w.add_data(h_data, "ch0")
    w.add_process(h_sig, "sig", "ch0", signal=True)
    w.add_process(h_bkg, "bkg", "ch0", signal=False)
    for i in range(n_free_systs):
        w.add_norm_systematic(f"freeSyst{i}", ["bkg"], "ch0", 1.05, constrained=False)
    w.add_norm_systematic("bkgNorm", ["bkg"], "ch0", 1.05)
    w.add_norm_systematic("sigNorm", ["sig"], "ch0", 1.03)
    w.write(outfolder=os.path.dirname(path), outfilename=os.path.basename(path))


def make_options(**kwargs):
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=True,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
        blindingGroup=[],
        maxRestarts=-1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def build(path, model, **opts):
    ind = inputdata.FitInputData(path)
    m = model(ind) if callable(model) else model
    f = fitter.Fitter(ind, m, make_options(**opts), do_blinding=False)
    f.defaultassign()
    f.set_nobs(f.expected_yield())
    return f


@pytest.fixture(scope="module")
def paths():
    """One tensor with free card systematics, one with none."""
    with tempfile.TemporaryDirectory() as d:
        p2 = os.path.join(d, "ndf_free2.hdf5")
        p0 = os.path.join(d, "ndf_free0.hdf5")
        make_tensor(p2, 2)
        make_tensor(p0, 0)
        yield {"two_free_systs": p2, "no_free_systs": p0}


def _ndfsat(f):
    """The driver's expression, as bin/rabbit_fit.py computes it."""
    return int(tf.size(f.nobs).numpy()) - f.nfreeparms


def _ndfsat_old(f):
    """The expression this change replaces."""
    return (
        int(tf.size(f.nobs).numpy())
        - f.param_model.nparams
        - f.indata.nsystnoconstraint
    )


# --- the rule ----------------------------------------------------------------


@pytest.mark.parametrize("n_priored", [0, 1, NPOU])
def test_constrained_model_params_cost_no_dof(paths, n_priored):
    """Each prior a model declares buys back exactly one degree of freedom.

    Fails before the change for every n_priored > 0: the old expression charged
    all NPOU + 1 model parameters regardless.
    """
    f = build(paths["two_free_systs"], lambda ind: PriorModel(ind, n_priored))
    nfree_model = f.param_model.nparams - n_priored
    assert f.nfreeparms == nfree_model + f.indata.nsystnoconstraint
    assert _ndfsat(f) == NBINS - nfree_model - f.indata.nsystnoconstraint


def test_a_constrained_model_param_and_a_constrained_card_syst_cost_the_same(paths):
    """The heart of it: same statistical object, same charge.

    Moving a sigma = 1 constraint from the card into the model must not change
    the ndf. Compared here as a delta against the fully-free model, so the check
    does not depend on the rest of the card.
    """
    free = build(paths["two_free_systs"], NoPriorModel)
    one_prior = build(paths["two_free_systs"], lambda ind: PriorModel(ind, 1))

    # constraining one model parameter buys back one dof ...
    assert _ndfsat(one_prior) - _ndfsat(free) == 1
    # ... which is exactly what a constrained CARD systematic is worth: the card
    # has 2 constrained systs and they are charged nothing, so the free model's
    # ndf is nbins minus its own parameters and the free systs only
    assert _ndfsat(free) == NBINS - free.param_model.nparams - 2


def test_breakdown_matches_the_count(paths):
    """The logged decomposition must be the same number the ndf uses."""
    f = build(paths["two_free_systs"], lambda ind: PriorModel(ind, 3))
    nfree_params, nfree_systs = f.nfreeparms_breakdown
    assert nfree_params + nfree_systs == f.nfreeparms
    assert nfree_params == f.param_model.nparams - 3
    # the systematics half is nsystnoconstraint by construction
    assert nfree_systs == f.indata.nsystnoconstraint == 2


def test_count_agrees_with_what_the_likelihood_penalises(paths):
    """nfreeparms is read off cw, so it must match the constraint term itself.

    Move every constrained parameter to its constraint centre and the penalty
    must vanish; the parameters that can then be moved without cost are exactly
    the free ones. This is the check that the count cannot drift from
    _compute_lc.
    """
    f = build(paths["two_free_systs"], lambda ind: PriorModel(ind, NPOU))
    f.x.assign(f.x0)
    assert np.isclose(float(f._compute_lc().numpy()), 0.0, rtol=0, atol=1e-12)

    cw = f.cw.numpy()
    n_penalising = 0
    for i in range(len(cw)):
        x = f.x0.numpy().copy()
        x[i] += 1.0
        f.x.assign(x)
        if float(f._compute_lc().numpy()) > 1e-12:
            n_penalising += 1
    assert len(cw) - n_penalising == f.nfreeparms


# --- backward compatibility --------------------------------------------------


@pytest.mark.parametrize("key", ["two_free_systs", "no_free_systs"])
def test_exact_noop_for_a_model_without_priors(paths, key):
    """THE compatibility claim, and why this is safe for existing analyses.

    With no priors declared, every ParamModel entry has cw = 0, so nfreeparms is
    exactly ``param_model.nparams + indata.nsystnoconstraint`` and the new
    expression reduces to the old one identically -- not approximately.
    """
    f = build(paths[key], NoPriorModel)
    assert f.nfreeparms == f.param_model.nparams + f.indata.nsystnoconstraint
    assert _ndfsat(f) == _ndfsat_old(f)


def test_not_a_noop_once_priors_are_declared(paths):
    """Guard against the previous test passing vacuously.

    If the two expressions agreed for a priored model as well, this change would
    be doing nothing at all.
    """
    f = build(paths["two_free_systs"], lambda ind: PriorModel(ind, NPOU))
    assert _ndfsat(f) - _ndfsat_old(f) == NPOU


# --- frozen parameters -------------------------------------------------------


def test_freezing_a_free_param_buys_back_a_dof(paths):
    """A frozen parameter is fixed, so it costs no degree of freedom.

    ``cw`` records CONSTRAINTS; frozen-ness lives in ``frozen_params_mask``. An
    unconstrained frozen parameter therefore has ``cw == 0`` and was counted as
    free, making ndfsat too small and the saturated p-value too pessimistic by
    exactly that many parameters.
    """
    free = build(paths["two_free_systs"], NoPriorModel)
    frozen = build(
        paths["two_free_systs"], NoPriorModel, freezeParameters=[FREE_PARAM_NAME]
    )

    assert frozen.nfreeparms == free.nfreeparms - 1
    assert _ndfsat(frozen) == _ndfsat(free) + 1


def test_freezing_an_already_constrained_param_changes_nothing(paths):
    """It was already costing nothing, so freezing it cannot buy anything back.

    Guards the obvious way to get this wrong -- subtracting the frozen count
    from nfreeparms instead of intersecting it with the unconstrained set,
    which would double-count and make ndfsat too large.
    """
    priored = build(paths["two_free_systs"], lambda ind: PriorModel(ind, NPOU))
    name = priored.parms[priored.cw.numpy() != 0.0][0]
    name = name.decode() if isinstance(name, bytes) else str(name)

    frozen = build(
        paths["two_free_systs"],
        lambda ind: PriorModel(ind, NPOU),
        freezeParameters=[name],
    )
    assert frozen.nfreeparms == priored.nfreeparms
    assert _ndfsat(frozen) == _ndfsat(priored)


def test_breakdown_also_excludes_frozen(paths):
    """The logged decomposition must stay the same number the ndf uses."""
    frozen = build(
        paths["two_free_systs"], NoPriorModel, freezeParameters=[FREE_PARAM_NAME]
    )
    nfree_params, nfree_systs = frozen.nfreeparms_breakdown
    assert nfree_params + nfree_systs == frozen.nfreeparms


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
