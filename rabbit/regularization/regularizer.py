import numpy as np


class Regularizer:

    #: Whether ``compute_nll_penalty`` reads its ``observables`` argument.
    #:
    #: A penalty on the *parameters* alone (a bound on polynomial
    #: coefficients, a smoothness prior) does not need the predicted yields;
    #: one that shapes the prediction itself does. The distinction is not
    #: cosmetic: computing observables for a penalty that ignores them forces
    #: the fitter to evaluate yields with ``full=True`` on every likelihood
    #: call, and it is what made regularizers unavailable in bins-sharded
    #: multi-device fits at all, since no single device holds the full yield
    #: vector.
    #:
    #: The default is True because it is the safe answer: a subclass that
    #: silently got False would be handed ``None`` and fail. Subclasses that
    #: genuinely ignore ``observables`` should set it to False and gain
    #: multi-device support.
    needs_observables = True

    def __init__(self, mapping, dtype):
        """
        Initialize the regularization depending on the mapping
        """

    def set_expectations(self, initial_params, initial_observables, parms=None):
        """
        Set the expectations to use in the regularization. Called once per
        parameter layout: the fitter can swap its ParamModel mid-session (the
        saturated goodness-of-fit path wraps it in a CompositeParamModel), which
        reorders and resizes the parameter vector. Do not cache positions across
        calls; re-resolve them by name from ``parms``, the names of every entry
        of ``initial_params``.
        """

    def compute_nll_penalty(self, params, observables):
        """
        Compute the penalty term that gets added to -ln(L), this function should be called in each step of the minimization

        ``observables`` is None when ``needs_observables`` is False, so that a
        regularizer which declares it does not need them cannot quietly start
        using them (it would raise rather than read a truncated vector).
        """
        return 0

    @staticmethod
    def resolve_indices(parms, names, who="Regularizer"):
        """Map parameter names to positions in ``parms``, raising if any is absent."""
        if parms is None:
            raise ValueError(
                f"{who}: parameter names are required to resolve positions"
            )
        parms = np.asarray(parms).astype(str)
        index = {name: i for i, name in enumerate(parms)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValueError(f"{who}: {missing} not in the fit's parameter vector")
        return {n: index[n] for n in names}
