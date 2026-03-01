import argparse

from rabbit import fitter


class OptionalListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            setattr(namespace, self.dest, [".*"])
        else:
            setattr(namespace, self.dest, values)


def common_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("filename", help="filename of the main hdf5 input")
    parser.add_argument("-o", "--output", default="./", help="output directory")
    parser.add_argument(
        "--postfix",
        default=None,
        type=str,
        help="Postfix to append on output file name",
    )
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
        "--allowNegativePOI",
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
        "--poiModel",
        default=["Mu"],
        nargs="+",
        help="Specify POI model to be used to introduce non standard parameterization",
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

    return parser
