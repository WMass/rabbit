import argparse

from rabbit.bbstat.bbstat import VALID_BIN_BY_BIN_STAT_TYPES


class OptionalListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            setattr(namespace, self.dest, [".*"])
        else:
            setattr(namespace, self.dest, values)


def _add_base_args(parser):
    """Add verbosity and logging arguments shared by all rabbit scripts."""
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )


def _add_output_args(parser):
    """Add output path and postfix arguments shared by fitting and plotting scripts."""
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./",
        help="Base path for output",
    )
    parser.add_argument(
        "-p",
        "--postfix",
        default=None,
        type=str,
        help="Postfix to append on output file name",
    )


def add_impact_args(parser):
    """Add common impact arguments shared by impact print and plot scripts."""
    parser.add_argument(
        "--impactType",
        type=str,
        default="traditional",
        choices=[
            None,
            "none",
            "traditional",
            "global",
            "gaussian_global",
            "nonprofiled",
        ],
        help="Impact definition",
    )
    parser.add_argument(
        "--asymImpacts",
        action="store_true",
        help="Use asymmetric impacts from likelihood, otherwise symmetric from hessian",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        default=None,
        type=str,
        nargs="+",
        help="Impacts on observables, use '-m <mapping> channel axes' for mapping results.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use relative uncertainties",
    )


def add_style_args(parser):
    """Add common style arguments for histogram plot scripts."""
    choices_padding = ["auto", "lower left", "lower right", "upper left", "upper right"]
    parser.add_argument(
        "--noEnergy",
        action="store_true",
        help="Don't include the energy in the upper right corner of the plot",
    )
    parser.add_argument(
        "--legPos", type=str, default="upper right", help="Set legend position"
    )
    parser.add_argument(
        "--legSize",
        type=str,
        default="small",
        help="Legend text size (small: axis ticks size, large: axis label size, number)",
    )
    parser.add_argument(
        "--legCols", type=int, default=2, help="Number of columns in legend"
    )
    parser.add_argument(
        "--legPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in legend",
    )
    parser.add_argument(
        "--lowerLegPos",
        type=str,
        default="upper left",
        help="Set lower legend position",
    )
    parser.add_argument(
        "--lowerLegCols", type=int, default=2, help="Number of columns in lower legend"
    )
    parser.add_argument(
        "--lowerLegPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in lower legend",
    )
    parser.add_argument(
        "--noSciy",
        action="store_true",
        help="Don't allow scientific notation for y axis",
    )
    parser.add_argument(
        "--yscale",
        type=float,
        help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="Min and max values for y axis (if not specified, range set automatically)",
    )
    parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[0.9, 1.1],
        help="y range for ratio plot",
    )
    parser.add_argument(
        "--logy", action="store_true", help="Make the yscale logarithmic"
    )
    parser.add_argument(
        "--customFigureWidth",
        type=float,
        default=None,
        help="Use a custom figure width, otherwise chosen automatic",
    )
    parser.add_argument(
        "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
    )


def common_parser():
    """Return a parser with common arguments for fitting scripts (rabbit_fit, rabbit_limit)."""
    parser = argparse.ArgumentParser()
    _add_base_args(parser)
    _add_output_args(parser)
    parser.add_argument("filename", help="filename of the main hdf5 input")
    parser.add_argument(
        "--eager",
        action="store_true",
        default=False,
        help="Run tensorflow in eager mode (for debugging)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Calculate and print additional info for diagnostics (condition number, edm value)",
    )
    parser.add_argument(
        "--earlyStopping",
        default=20,
        type=int,
        help="Number of iterations with no improvement after which the minimizer is "
        "considered stalled. Paired with --maxRestarts this means 'restart here', not "
        "'give up here'. On by default because scipy will not stop on its own: its "
        "trust-region loop has no termination test for a collapsed trust radius, and "
        "its maxiter defaults to 200*nparams, so a stalled fit spins for hours making "
        "no progress rather than exiting. Specify -1 to disable.",
    )
    parser.add_argument(
        "--maxRestarts",
        default=-1,
        type=int,
        help="When --earlyStopping triggers, restart the minimizer from the stalled "
        "point instead of giving up. scipy's trust-region methods shrink the trust "
        "radius 4x per rejected step with no lower bound, and the radius is reset by a "
        "fresh minimize() call -- so a stall is often just a collapsed step size, and "
        "restarting resumes the descent (the same effect as chaining --externalPostfit "
        "by hand). -1 (default) restarts as often as the loss keeps improving and stops "
        "only once a restart no longer reduces it; 0 disables restarting; N > 0 caps "
        "the number of restarts.",
    )
    parser.add_argument(
        "--minimizerMethod",
        default="trust-krylov",
        type=str,
        choices=[
            "trust-krylov",
            "trust-exact",
            "tf-trust-exact",
            "tf-trust-ncg",
            "tf-trust-krylov",
            "BFGS",
            "L-BFGS-B",
            "CG",
            "trust-ncg",
            "dogleg",
        ],
        help="Minimizer method used for the nominal fit minimization. The "
        "'tf-' prefixed methods are native TensorFlow implementations, the rest "
        "are dispatched to scipy.optimize.minimize. 'tf-trust-exact' ports "
        "trust-exact keeping the Hessian and the subproblem's Cholesky "
        "factorizations on the TensorFlow device instead of round-tripping "
        "through LAPACK; 'tf-trust-ncg' is the matrix-free Steihaug-CG "
        "counterpart (the same subproblem as scipy's trust-ncg, and the "
        "practical stand-in for trust-krylov) with the whole CG inner loop "
        "compiled as one TF graph call, i.e. no python round trip per "
        "Hessian-vector product; 'tf-trust-krylov' is a native GLTR (the "
        "trust-krylov algorithm): Lanczos on device with the subproblem "
        "solved to optimality within the Krylov subspace via host-side "
        "tridiagonal solves, reusing the radius-independent Krylov data "
        "across re-solves after rejected steps",
    )
    parser.add_argument(
        "--nDevices",
        default=1,
        type=int,
        help="Shard the likelihood over this many devices (bins-sharded "
        "data parallelism, GPUs preferred). The main motivation is memory: "
        "every large tensor in the fit is bins-proportional and gets split "
        "across the devices, so models that exceed a single GPU's memory "
        "become fittable; compute also scales when each shard is large "
        "enough to saturate its device. Only the dense-tensor Poisson/chi2 "
        "likelihoods are supported (no sparse mode, no --covarianceFit).",
    )
    parser.add_argument(
        "--devices",
        default=None,
        type=int,
        nargs="+",
        help="Explicit physical GPU indices to use (overrides the automatic "
        "least-occupied selection). The number given should match --nDevices.",
    )
    parser.add_argument(
        "--precondition",
        action="store_true",
        help="Reparameterise a block of parameters so the reference Hessian is the "
        "identity there (theta = theta_ref + L^-T y). This is preconditioning of the "
        "trust-region subproblem obtained as a change of variables, so the minimizer "
        "itself is untouched. Helps where many unconstrained, strongly correlated "
        "parameters make the Krylov inner solve struggle and outer steps get rejected. "
        "Off by default; a pure reparameterisation, so results are unchanged.",
    )
    parser.add_argument(
        "--preconditionParams",
        default=None,
        type=str,
        nargs="+",
        help="Parameters to precondition: exact names, regexes matched against the full "
        "parameter name, or systematic group names. Default (flag given without this "
        "option) is every unconstrained parameter, which is where it pays off; "
        "constrained nuisances are already normalised by their prior. Frozen "
        "parameters are always excluded.",
    )
    parser.add_argument(
        "--preconditionBlocks",
        nargs="?",
        const="auto",
        default="auto",
        type=str,
        choices=["auto", "expressions", "none"],
        help="How to group the selected parameters into blocks; the transform is "
        "block diagonal, one Cholesky per block. 'auto' (the default, and what a "
        "bare --preconditionBlocks selects) reads the clusters off the reference "
        "matrix by thresholding its correlations and taking connected components: "
        "no parameter naming needed, much cheaper since Cholesky is O(m^3), and a "
        "singular cluster only costs its own block rather than all of them. "
        "'expressions' makes one block per --preconditionParams entry. 'none' does "
        "no grouping at all and factorises the whole selected scope as a single "
        "block, which keeps every cross-correlation but is the most expensive and "
        "fails entirely if any part of the scope is singular.",
    )
    parser.add_argument(
        "--preconditionBlockThreshold",
        default=0.1,
        type=float,
        help="Correlation threshold for --preconditionBlocks auto. Correlations "
        "below it are left unpreconditioned. Too low and every parameter "
        "percolates into a single block; too high and genuinely coupled "
        "parameters are split apart.",
    )
    parser.add_argument(
        "--preconditionFrom",
        default="hessian",
        type=str,
        choices=["hessian", "gaussnewton"],
        help="Source of the reference matrix. 'hessian' (default) takes the exact "
        "Hessian at the starting point, one extra Hessian evaluation, roughly one "
        "trust-exact iteration. 'gaussnewton' takes the Fisher information instead: "
        "positive semi-definite by construction, so it factorises at the default ridge, "
        "but for that same reason it cannot represent negative curvature. Where the "
        "exact Hessian is indefinite at the starting point, whitening with it leaves "
        "those directions negative and the fit stalls, so 'hessian' is usually the "
        "better choice; prefer 'gaussnewton' only where the Hessian is positive "
        "definite anyway, e.g. near a minimum.",
    )
    parser.add_argument(
        "--preconditionRidge",
        default=1e-8,
        type=float,
        help="Ridge added to the preconditioning block diagonal, relative to its largest "
        "diagonal entry, to keep near-degenerate blocks factorisable. Escalated "
        "automatically if the Cholesky still fails; a block that cannot be factorised "
        "falls back to no preconditioning.",
    )
    parser.add_argument(
        "--hvpBatch",
        default=256,
        type=int,
        help="Number of Hessian-vector products evaluated together when the "
        "dense Hessian is assembled from HVPs (preconditioning and, on the "
        "multi-device path, the postfit Hessian). Memory scales with this and "
        "the number of graph calls scales inversely: on a 4-way shard of a "
        "92144-bin model 256 costs a few GB and turns 6538 sequential HVPs into "
        "26 batched ones. The batch is halved automatically if the device "
        "cannot hold it, so this is an upper bound rather than a value that "
        "has to be right. Set 1 for the sequential loop.",
    )
    parser.add_argument(
        "--hvpMethod",
        default="revrev",
        type=str,
        choices=["fwdrev", "revrev"],
        help="Autodiff mode for the Hessian-vector product. 'revrev' (reverse-over-reverse) "
        "is the default and works well in combination with --jitCompile. 'fwdrev' "
        "(forward-over-reverse, via tf.autodiff.ForwardAccumulator) is an alternative.",
    )
    parser.add_argument(
        "--jitCompile",
        default="auto",
        type=str,
        choices=["auto", "on", "off"],
        help="Control XLA jit_compile=True on the loss/gradient/HVP tf.functions. "
        "'auto' (default) enables jit_compile in dense mode and disables it in "
        "sparse mode (where the CSR matmul kernels have no XLA implementation). "
        "'on' forces jit_compile on (falling back to off with a warning in sparse "
        "mode). 'off' disables jit_compile unconditionally.",
    )
    parser.add_argument(
        "--chisqFit",
        default=False,
        action="store_true",
        help="Perform diagonal chi-square fit instead of poisson likelihood fit",
    )
    parser.add_argument(
        "--covarianceFit",
        default=False,
        action="store_true",
        help="Perform chi-square fit using covariance matrix for the observations",
    )
    parser.add_argument(
        "--noHessian",
        default=False,
        action="store_true",
        help="Don't compute the hessian of parameters",
    )
    parser.add_argument(
        "--noEDM",
        default=False,
        action="store_true",
        help="Skip the Hessian-free EDM/CG postfit step (only meaningful with "
        "--noHessian). Skips both the edmval estimate and the POI+NOI "
        "uncertainty rows; their entries in the output are NaN.",
    )
    parser.add_argument(
        "--minimizerMaxiter",
        type=int,
        default=None,
        help="Cap the number of scipy.optimize.minimize iterations. "
        "Passed as options={'maxiter': N} to the minimizer. None (default) "
        "uses scipy's method-specific default (typically 1000+).",
    )
    parser.add_argument(
        "--minimizerGtol",
        type=float,
        default=None,
        help="Gradient-norm tolerance for the minimizer. Passed as "
        "options={'gtol': X} (recognized by trust-krylov, L-BFGS-B, BFGS, CG). "
        "None (default) uses scipy's per-method default.",
    )
    parser.add_argument(
        "--minimizerFtol",
        type=float,
        default=None,
        help="Function-value tolerance for the minimizer (L-BFGS-B only). "
        "Passed as options={'ftol': X}. None (default) uses scipy's default "
        "(2.22e-9 for L-BFGS-B). Ignored by methods that don't honor 'ftol'.",
    )
    parser.add_argument(
        "--forceLinear",
        default=False,
        action="store_true",
        help="Force the fitter to treat the likelihood as purely quadratic and "
        "solve via a single Gaussian (Cholesky) step, even when the param "
        "model, asymmetry, or systematic type would otherwise mark it as "
        "nonlinear. Useful for iterative linearized unfolding where the outer "
        "loop re-anchors the linearization point.",
    )
    parser.add_argument(
        "--prefitUnconstrainedNuisanceUncertainty",
        default=0.0,
        type=float,
        help="Assumed prefit uncertainty for unconstrained nuisances",
    )
    parser.add_argument(
        "--unblind",
        type=str,
        default=[],
        nargs="*",
        action=OptionalListAction,
        help="""
        Specify list of expressions to unblind matching parameters of interest.
        Each entry is an exact parameter name or a regex matched against the full parameter
        name (use e.g. 'alphaS.*' to match a family of parameters).
        E.g. use '--unblind signal' to unblind a parameter named signal or '--unblind' to unblind all.
        """,
    )
    parser.add_argument(
        "--blindingGroup",
        type=str,
        default=[],
        nargs="*",
        help="""
        Specify list of regex (matched against the full parameter name) defining groups of
        parameters that share a single deterministic
        blinding offset (seeded from the regex string itself). Useful to keep relative pulls /
        differences between matched parameters meaningful while still blinding their absolute
        values. E.g. '--blindingGroup ^alphaS_y\\d+$' applies the same offset to all alphaS
        rapidity-bin parameters. Parameters not matching any group keep per-name blinding.
        Overlap with --unblind is treated as a configuration error and aborts the fit.
        """,
    )
    parser.add_argument(
        "--setConstraintMinimum",
        default=[],
        nargs=2,
        action="append",
        help="Set the constraint minima of specified parameter to specified value",
    )
    parser.add_argument(
        "--freezeParameters",
        type=str,
        default=[],
        nargs="+",
        help="""
        Specify list of expressions to freeze matching parameters of interest
        (exact names or regex matched against the full parameter name).
        """,
    )
    parser.add_argument(
        "--pseudoData",
        default=None,
        type=str,
        nargs="*",
        help="run fit on pseudo data with the given name",
    )
    parser.add_argument(
        "-t",
        "--toys",
        default=[-1],
        type=int,
        nargs="+",
        help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)",
    )
    parser.add_argument(
        "--toysSystRandomize",
        default="frequentist",
        choices=["frequentist", "bayesian", "none"],
        help="""
        Type of randomization for systematic uncertainties (including binByBinStat if present).
        Options are 'frequentist' which randomizes the contraint minima a.k.a global observables
        and 'bayesian' which randomizes the actual nuisance parameters used in the pseudodata generation
        """,
    )
    parser.add_argument(
        "--toysDataRandomize",
        default="poisson",
        choices=["poisson", "normal", "none"],
        help="Type of randomization for pseudodata.  Options are 'poisson',  'normal', and 'none'",
    )
    parser.add_argument(
        "--toysDataMode",
        default="expected",
        choices=["expected", "observed"],
        help="central value for pseudodata used in the toys",
    )
    parser.add_argument(
        "--toysRandomizeParameters",
        default=False,
        action="store_true",
        help="randomize the parameter starting values for toys",
    )
    parser.add_argument(
        "--seed", default=123456789, type=int, help="random seed for toys"
    )
    parser.add_argument(
        "--expectSignal",
        default=None,
        nargs=2,
        action="append",
        help="Specify tuple with signal name and rate multiplier for signal expectation (used for fit starting values and for toys). E.g. '--expectSignal BSM 0.0 --expectSignal SM 1.0'",
    )
    parser.add_argument(
        "--allowNegativeParam",
        default=False,
        action="store_true",
        help="allow signal strengths to be negative (otherwise constrained to be non-negative)",
    )

    parser.add_argument(
        "--noBinByBinStat",
        default=False,
        action="store_true",
        help="Don't add bin-by-bin statistical uncertainties on templates (by default adding sumW2 on variance)",
    )
    parser.add_argument(
        "--binByBinStatType",
        default="automatic",
        choices=["automatic", *VALID_BIN_BY_BIN_STAT_TYPES],
        help="probability density for bin-by-bin statistical uncertainties, ('automatic' is 'gamma' except for data covariance where it is 'normal')",
    )
    parser.add_argument(
        "--binByBinStatMode",
        default="lite",
        choices=["lite", "full"],
        help="Barlow-Beeston mode bin-by-bin statistical uncertainties",
    )
    parser.add_argument(
        "--minBBKstat",
        default=0.0,
        type=float,
        help="Mask (bin, process) entries with effective MC stats kstat = sumw**2/sumw2 "
        "below this threshold so their bin-by-bin nuisances are fixed at beta0. "
        "Default 0 keeps the original behaviour. Useful for full-mode BBB to avoid "
        "ill-conditioned profiles for processes with very low effective stats per bin "
        "(e.g. mixed-sign-weight cancellations).",
    )
    parser.add_argument(
        "--paramModel",
        default=None,
        nargs="+",
        action="append",
        help="Specify param model to be used to introduce non standard parameterization. "
        "Can be specified multiple times to combine models via CompositeParamModel, "
        "e.g. '--paramModel Mu --paramModel ABCD nonprompt ch_A ch_B ch_C ch_D'.",
    )

    parser.add_argument(
        "-m",
        "--mapping",
        nargs="+",
        action="append",
        default=[],
        help="""
        perform mappings on observables or parameters for the prefit and postfit histograms,
        specifying the mapping defined in rabbit/mappings/ followed by arguments passed in the mapping __init__,
        e.g. '-m Project ch0 eta pt' to get a 2D projection to eta-pt or '-m Project ch0' to get the total yield.
        This argument can be called multiple times.
        Custom mappings can be specified with the full path to the custom mapping e.g. '-m custom_mappings.MyCustomMapping'.
        """,
    )
    parser.add_argument(
        "--compositeMapping",
        action="store_true",
        help="Make a composite mapping and compute the covariance matrix across all mappings.",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        nargs="+",
        action="append",
        default=[],
        help="""
        apply regularization on the output "nout" of a mapping by including a penalty term P(nout) in the -log(L) of the minimization.
        As argument, specify the regulaization defined in rabbit/regularization/, followed by a mapping using the same syntax as discussed above. 
        e.g. '-r SVD Select ch0_masked' to apply SVD regularization on the channel 'ch0_masked' or '-r SVD Project ch0 pt' for the 1D projection to pt.
        Custom regularization can be specified with the full path e.g. '-r custom_regularization.MyCustomRegularization Project ch0 pt'.
        """,
    )

    return parser


def plot_parser():
    """Return a parser with common arguments for plotting scripts.

    Scripts extend this parser by calling plot_parser() and adding their
    own arguments, mirroring how fitting scripts use common_parser().
    """
    parser = argparse.ArgumentParser()
    _add_base_args(parser)
    _add_output_args(parser)
    parser.add_argument(
        "infile",
        type=str,
        help="hdf5 file from rabbit",
    )
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Override use of xrdcp and use the mount instead",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--title",
        default="Rabbit",
        type=str,
        help="Title to be printed in upper left",
    )
    parser.add_argument(
        "--subtitle",
        default="",
        type=str,
        help="Subtitle to be printed after title",
    )
    parser.add_argument("--titlePos", type=int, default=2, help="title position")
    parser.add_argument(
        "--scaleTextSize",
        type=float,
        default=1.0,
        help="Scale all text sizes by this number",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=None,
        help="Luminosity for plot labeling (in fb-1)",
    )
    return parser


def print_parser():
    """Return a parser with common arguments for print scripts.

    Scripts extend this parser by calling print_parser() and adding their
    own arguments.
    """
    parser = argparse.ArgumentParser()
    _add_base_args(parser)
    parser.add_argument(
        "infile",
        type=str,
        help="fitresults output",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    return parser
