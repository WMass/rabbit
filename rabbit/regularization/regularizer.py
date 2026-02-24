class Regularizer:

    def __init__(self, mapping, dtype):
        """
        Initialize the regularization depending on the mapping
        """

    def set_expectations(self, initial_params, initial_observables):
        """
        Set the expectations to use in the regularization, this step should be called once per fit configuration
        """

    def compute_nll_penalty(self, params, observables):
        """
        Compute the penalty term that gets added to -ln(L), this function should be called in each step of the minimization
        """
        return 0
