import argparse

from rabbit import fitter


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
        default=-1,
        type=int,
        help="Number of iterations with no improvement after which training will be stopped. Specify -1 to disable.",
    )
    parser.add_argument(
        "--minimizerMethod",
        default="trust-krylov",
        type=str,
        choices=[
            "trust-krylov",
            "trust-exact",
            "BFGS",
            "L-BFGS-B",
            "CG",
            "trust-ncg",
            "dogleg",
        ],
        help="Mnimizer method used in scipy.optimize.minimize for the nominal fit minimization",
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
        Specify list of regex to unblind matching parameters of interest.
        E.g. use '--unblind ^signal$' to unblind a parameter named signal or '--unblind' to unblind all.
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
        Specify list of regex to freeze matching parameters of interest.
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
        choices=["automatic", *fitter.Fitter.valid_bin_by_bin_stat_types],
        help="probability density for bin-by-bin statistical uncertainties, ('automatic' is 'gamma' except for data covariance where it is 'normal')",
    )
    parser.add_argument(
        "--binByBinStatMode",
        default="lite",
        choices=["lite", "full"],
        help="Barlow-Beeston mode bin-by-bin statistical uncertainties",
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
