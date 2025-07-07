#!/usr/bin/env python3

import tensorflow as tf

tf.config.experimental.enable_op_determinism()

import argparse
import time

import h5py
import numpy as np
import scipy

from rabbit import fitter, inputdata, io_tools, workspace
from rabbit.physicsmodels import helpers as ph
from rabbit.physicsmodels import physicsmodel as pm
from rabbit.tfhelpers import edmval_cov

from wums import output_tools, logging  # isort: skip

logger = None


class OptionalListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 0:
            setattr(namespace, self.dest, [".*"])
        else:
            setattr(namespace, self.dest, values)


def make_parser():
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
        "--fullNll",
        action="store_true",
        help="Calculate and store full value of -log(L)",
    )
    parser.add_argument("filename", help="filename of the main hdf5 input")
    parser.add_argument("-o", "--output", default="./", help="output directory")
    parser.add_argument("--outname", default="fitresults.hdf5", help="output file name")
    parser.add_argument(
        "--postfix",
        default=None,
        type=str,
        help="Postfix to append on output file name",
    )
    parser.add_argument(
        "--minimizerMethod",
        default="trust-krylov",
        type=str,
        choices=["trust-krylov", "trust-exact"],
        help="Mnimizer method used in scipy.optimize.minimize for the nominal fit minimization",
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
        default=1.0,
        type=float,
        help="rate multiplier for signal expectation (used for fit starting values and for toys)",
    )
    parser.add_argument("--POIMode", default="mu", help="mode for POI's")
    parser.add_argument(
        "--allowNegativePOI",
        default=False,
        action="store_true",
        help="allow signal strengths to be negative (otherwise constrained to be non-negative)",
    )
    parser.add_argument("--POIDefault", default=1.0, type=float, help="mode for POI's")
    parser.add_argument(
        "--contourScan",
        default=None,
        type=str,
        nargs="*",
        help="run likelihood contour scan on the specified variables, specify w/o argument for all parameters",
    )
    parser.add_argument(
        "--contourLevels",
        default=[
            1.0,
        ],
        type=float,
        nargs="+",
        help="Confidence level in standard deviations for contour scans (1 = 1 sigma = 68%)",
    )
    parser.add_argument(
        "--contourScan2D",
        default=None,
        type=str,
        nargs="+",
        action="append",
        help="run likelihood contour scan on the specified variable pairs",
    )
    parser.add_argument(
        "--scan",
        default=None,
        type=str,
        nargs="*",
        help="run likelihood scan on the specified variables, specify w/o argument for all parameters",
    )
    parser.add_argument(
        "--scan2D",
        default=None,
        type=str,
        nargs="+",
        action="append",
        help="run 2D likelihood scan on the specified variable pairs",
    )
    parser.add_argument(
        "--scanPoints",
        default=15,
        type=int,
        help="default number of points for likelihood scan",
    )
    parser.add_argument(
        "--scanRange",
        default=3.0,
        type=float,
        help="default scan range in terms of hessian uncertainty",
    )
    parser.add_argument(
        "--scanRangeUsePrefit",
        default=False,
        action="store_true",
        help="use prefit uncertainty to define scan range",
    )
    parser.add_argument(
        "--prefitOnly",
        default=False,
        action="store_true",
        help="Only compute prefit outputs",
    )
    parser.add_argument(
        "--noHessian",
        default=False,
        action="store_true",
        help="Don't compute the hessian of parameters",
    )
    parser.add_argument(
        "--saveHists",
        default=False,
        action="store_true",
        help="save prefit and postfit histograms",
    )
    parser.add_argument(
        "--saveHistsPerProcess",
        default=False,
        action="store_true",
        help="save prefit and postfit histograms for each process",
    )
    parser.add_argument(
        "--computeHistErrors",
        default=False,
        action="store_true",
        help="propagate uncertainties to inclusive prefit and postfit histograms",
    )
    parser.add_argument(
        "--computeHistErrorsPerProcess",
        default=False,
        action="store_true",
        help="propagate uncertainties to prefit and postfit histograms for each process",
    )
    parser.add_argument(
        "--computeHistCov",
        default=False,
        action="store_true",
        help="propagate covariance of histogram bins (inclusive in processes)",
    )
    parser.add_argument(
        "--computeHistImpacts",
        default=False,
        action="store_true",
        help="propagate global impacts on histogram bins (inclusive in processes)",
    )
    parser.add_argument(
        "--computeVariations",
        default=False,
        action="store_true",
        help="save postfit histograms with each noi varied up to down",
    )
    parser.add_argument(
        "--noChi2",
        default=False,
        action="store_true",
        help="Do not compute chi2 on prefit/postfit histograms",
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
        choices=["automatic", "gamma", "normal"],
        help="probability density for bin-by-bin statistical uncertainties, ('automatic' is 'gamma' except for data covariance where it is 'normal')",
    )
    parser.add_argument(
        "--binByBinStatMode",
        default="light",
        choices=["light", "full"],
        help="Barlow-Beeston mode bin-by-bin statistical uncertainties",
    )
    parser.add_argument(
        "--externalPostfit",
        default=None,
        type=str,
        help="load posfit nuisance parameters and covariance from result of an external fit.",
    )
    parser.add_argument(
        "--externalPostfitResult",
        default=None,
        type=str,
        help="Specify result from external postfit file",
    )
    parser.add_argument(
        "--pseudoData",
        default=None,
        type=str,
        nargs="*",
        help="run fit on pseudo data with the given name",
    )
    parser.add_argument(
        "-m",
        "--physicsModel",
        nargs="+",
        action="append",
        default=[],
        help="""
        add physics model to perform transformations on observables for the prefit and postfit histograms, 
        specifying the model defined in rabbit/physicsmodels/ followed by arguments passed in the model __init__, 
        e.g. '-m Project ch0 eta pt' to get a 2D projection to eta-pt or '-m Project ch0' to get the total yield.  
        This argument can be called multiple times.
        Custom models can be specified with the full path to the custom model e.g. '-m custom_modesl.MyCustomModel'.
        """,
    )
    parser.add_argument(
        "--compositeModel",
        action="store_true",
        help="Make a composite model and compute the covariance matrix across all physics models.",
    )
    parser.add_argument(
        "--doImpacts",
        default=False,
        action="store_true",
        help="Compute impacts on POIs per nuisance parameter and per-nuisance parameter group",
    )
    parser.add_argument(
        "--globalImpacts",
        default=False,
        action="store_true",
        help="compute impacts in terms of variations of global observables (as opposed to nuisance parameters directly)",
    )
    parser.add_argument(
        "--chisqFit",
        default=False,
        action="store_true",
        help="Perform chi-square fit instead of likelihood fit",
    )
    parser.add_argument(
        "--externalCovariance",
        default=False,
        action="store_true",
        help="Using an external covariance matrix for the observations in the chi-square fit",
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

    return parser.parse_args()


def save_observed_hists(args, models, fitter, ws):
    for model in models:
        if not model.has_data:
            continue

        print(f"Save data histogram for {model.key}")
        ws.add_observed_hists(
            model,
            fitter.indata.data_obs,
            fitter.nobs.value(),
            fitter.indata.data_cov_inv,
            fitter.data_cov_inv,
        )


def save_hists(args, models, fitter, ws, prefit=True, profile=False):

    for model in models:
        logger.info(f"Save inclusive histogram for {model.key}")

        if prefit and model.skip_prefit:
            continue

        if not model.skip_incusive:
            exp, aux = fitter.expected_events(
                model,
                inclusive=True,
                compute_variance=args.computeHistErrors,
                compute_cov=args.computeHistCov,
                compute_chi2=not args.noChi2 and model.has_data,
                compute_global_impacts=args.computeHistImpacts and not prefit,
                profile=profile,
            )

            ws.add_expected_hists(
                model,
                exp,
                var=aux[0],
                cov=aux[1],
                impacts=aux[2],
                impacts_grouped=aux[3],
                prefit=prefit,
            )

            if aux[4] is not None:
                ws.add_chi2(aux[4], aux[5], prefit, model)

        if args.saveHistsPerProcess and not model.skip_per_process:
            logger.info(f"Save processes histogram for {model.key}")

            exp, aux = fitter.expected_events(
                model,
                inclusive=False,
                compute_variance=args.computeHistErrorsPerProcess,
                profile=profile,
            )

            ws.add_expected_hists(
                model,
                exp,
                var=aux[0],
                process_axis=True,
                prefit=prefit,
            )

        if args.computeVariations:
            if prefit:
                cov_prefit = fitter.cov.numpy()
                fitter.cov.assign(fitter.prefit_covariance(unconstrained_err=1.0))

            exp, aux = fitter.expected_events(
                model,
                inclusive=True,
                compute_variance=False,
                compute_variations=True,
                profile=profile,
            )

            ws.add_expected_hists(
                model,
                exp,
                var=aux[0],
                variations=True,
                prefit=prefit,
            )

            if prefit:
                fitter.cov.assign(tf.constant(cov_prefit))


def fit(args, fitter, ws, dofit=True):

    edmval = None

    if args.externalPostfit is not None:
        # load results from external fit and set postfit value and covariance elements for common parameters
        with h5py.File(args.externalPostfit, "r") as fext:
            if "x" in fext.keys():
                # fitresult from combinetf
                x_ext = fext["x"][...]
                parms_ext = fext["parms"][...].astype(str)
                cov_ext = fext["cov"][...]
            else:
                # fitresult from rabbit
                h5results_ext = io_tools.get_fitresult(fext, args.externalPostfitResult)
                h_parms_ext = h5results_ext["parms"].get()

                x_ext = h_parms_ext.values()
                parms_ext = np.array(h_parms_ext.axes["parms"])
                cov_ext = h5results_ext["cov"].get().values()

        xvals = fitter.x.numpy()
        covval = fitter.cov.numpy()
        parms = fitter.parms.astype(str)

        # Find common elements with their matching indices
        common_elements, idxs, idxs_ext = np.intersect1d(
            parms, parms_ext, assume_unique=True, return_indices=True
        )
        xvals[idxs] = x_ext[idxs_ext]
        covval[np.ix_(idxs, idxs)] = cov_ext[np.ix_(idxs_ext, idxs_ext)]

        fitter.x.assign(xvals)
        fitter.cov.assign(tf.constant(covval))
    else:
        if dofit:
            fitter.minimize()

        # compute the covariance matrix and estimated distance to minimum

        if not args.noHessian:

            val, grad, hess = fitter.loss_val_grad_hess()

            edmval, cov = edmval_cov(grad, hess)

            logger.info(f"edmval: {edmval}")

            fitter.cov.assign(cov)

            del cov

            if fitter.binByBinStat and fitter.diagnostics:
                # This is the estimated distance to minimum with respect to variations of
                # the implicit binByBinStat nuisances beta at fixed parameter values.
                # It should be near-zero by construction as long as the analytic profiling is
                # correct
                _, gradbeta, hessbeta = fitter.loss_val_grad_hess_beta()
                edmvalbeta, covbeta = edmval_cov(gradbeta, hessbeta)
                logger.info(f"edmvalbeta: {edmvalbeta}")

            if args.doImpacts:
                ws.add_impacts_hists(*fitter.impacts_parms(hess))

            del hess

            if args.globalImpacts:
                ws.add_global_impacts_hists(*fitter.global_impacts_parms())

    nllvalreduced = fitter.reduced_nll().numpy()

    ndfsat = (
        tf.size(fitter.nobs) - fitter.npoi - fitter.indata.nsystnoconstraint
    ).numpy()

    chi2 = 2.0 * nllvalreduced
    p_val = scipy.stats.chi2.sf(chi2, ndfsat)

    logger.info("Saturated chi2:")
    logger.info(f"    ndof: {ndfsat}")
    logger.info(f"    2*deltaNLL: {round(chi2, 2)}")
    logger.info(rf"    p-value: {round(p_val * 100, 2)}%")

    ws.results.update(
        {
            "nllvalreduced": nllvalreduced,
            "ndfsat": ndfsat,
            "edmval": edmval,
            "postfit_profile": args.externalPostfit is None,
        }
    )

    if args.fullNll:
        nllvalfull = fitter.full_nll().numpy()
        logger.info(f"2*deltaNLL(full): {nllvalfull}")
        ws.results["nllvalfull"] = nllvalfull

    ws.add_parms_hist(
        values=fitter.x,
        variances=tf.linalg.diag_part(fitter.cov) if not args.noHessian else None,
        hist_name="parms",
    )

    if not args.noHessian:
        ws.add_cov_hist(fitter.cov)

    if args.scan is not None:
        parms = np.array(fitter.parms).astype(str) if len(args.scan) == 0 else args.scan

        for param in parms:
            x_scan, dnll_values = fitter.nll_scan(
                param, args.scanRange, args.scanPoints, args.scanRangeUsePrefit
            )
            ws.add_nll_scan_hist(
                param,
                x_scan,
                dnll_values,
            )

    if args.scan2D is not None:
        for param_tuple in args.scan2D:
            x_scan, yscan, dnll_values = fitter.nll_scan2D(
                param_tuple, args.scanRange, args.scanPoints, args.scanRangeUsePrefit
            )
            ws.add_nll_scan2D_hist(param_tuple, x_scan, yscan, dnll_values)

    if args.contourScan is not None:
        # do likelihood contour scans
        nllvalreduced = fitter.reduced_nll().numpy()

        parms = (
            np.array(fitter.parms).astype(str)
            if len(args.contourScan) == 0
            else args.contourScan
        )

        contours = np.zeros((len(parms), len(args.contourLevels), 2, len(fitter.parms)))
        for i, param in enumerate(parms):
            for j, cl in enumerate(args.contourLevels):

                # find confidence interval
                contour = fitter.contour_scan(param, nllvalreduced, cl)
                contours[i, j, ...] = contour

        ws.add_contour_scan_hist(parms, contours, args.contourLevels)

    if args.contourScan2D is not None:
        raise NotImplementedError(
            "Likelihood contour scans in 2D are not yet implemented"
        )

        # do likelihood contour scans in 2D
        nllvalreduced = fitter.reduced_nll().numpy()

        contours = np.zeros(
            (len(args.contourScan2D), len(args.contourLevels), 2, args.scanPoints)
        )
        for i, param_tuple in enumerate(args.contourScan2D):
            for j, cl in enumerate(args.contourLevels):

                # find confidence interval
                contour = fitter.contour_scan2D(
                    param_tuple, nllvalreduced, cl, n_points=args.scanPoints
                )
                contours[i, j, ...] = contour

        ws.contour_scan2D_hist(args.contourScan2D, contours, args.contourLevels)


def main():
    start_time = time.time()
    args = make_parser()

    if args.eager:
        tf.config.run_functions_eagerly(True)

    if args.noHessian and args.doImpacts:
        raise Exception('option "--noHessian" only works without "--doImpacts"')

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    # make list of fits with -1: asimov; 0: fit to data; >=1: toy
    fits = np.concatenate(
        [np.array([x]) if x <= 0 else 1 + np.arange(x, dtype=int) for x in args.toys]
    )
    blinded_fits = [f == 0 or (f > 0 and args.toysDataMode == "observed") for f in fits]

    indata = inputdata.FitInputData(args.filename, args.pseudoData)
    ifitter = fitter.Fitter(indata, args, do_blinding=any(blinded_fits))

    # physics models for observables and transformations
    if len(args.physicsModel) == 0 and args.saveHists:
        # if no model is explicitly added and --saveHists is specified, fall back to Basemodel
        args.physicsModel = [["Basemodel"]]
    models = []
    for margs in args.physicsModel:
        model = ph.instance_from_class(margs[0], indata, *margs[1:])
        models.append(model)

    if args.compositeModel:
        models = [
            pm.CompositeModel(models),
        ]

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # pass meta data into output file
    meta = {
        "meta_info": output_tools.make_meta_info_dict(args=args),
        "meta_info_input": ifitter.indata.metadata,
        "signals": ifitter.indata.signals,
        "procs": ifitter.indata.procs,
        "nois": ifitter.parms[ifitter.npoi :][indata.noigroupidxs],
    }

    with workspace.Workspace(
        args.output,
        args.outname,
        postfix=args.postfix,
        fitter=ifitter,
    ) as ws:

        ws.write_meta(meta=meta)

        init_time = time.time()
        prefit_time = []
        postfit_time = []
        fit_time = []
        for i, ifit in enumerate(fits):
            group = "results"

            if args.pseudoData is None:
                datasets = [
                    ifitter.indata.data_obs,
                ]
            else:
                # shape nobs x npseudodata
                datasets = tf.transpose(indata.pseudodata_obs)

            # loop over (pseudo)data sets
            for j, data_values in enumerate(datasets):

                ifitter.defaultassign()
                if ifit == -1:
                    group += "_asimov"
                    ifitter.set_nobs(ifitter.expected_yield())
                else:
                    if ifit == 0:
                        ifitter.set_nobs(data_values)
                    elif ifit >= 1:
                        group += f"_toy{ifit}"
                        ifitter.toyassign(
                            data_values,
                            syst_randomize=args.toysSystRandomize,
                            data_randomize=args.toysDataRandomize,
                            data_mode=args.toysDataMode,
                            randomize_parameters=args.toysRandomizeParameters,
                        )

                if np.shape(datasets)[0] > 1:
                    # in case there are more than 1 pseudodata set, label each one
                    group += f"_{indata.pseudodatanames[j]}"

                ws.add_parms_hist(
                    values=ifitter.x,
                    variances=tf.linalg.diag_part(ifitter.cov),
                    hist_name="parms_prefit",
                )

                if args.saveHists:
                    save_observed_hists(args, models, ifitter, ws)
                    save_hists(args, models, ifitter, ws, prefit=True)
                prefit_time.append(time.time())

                if not args.prefitOnly:
                    ifitter.set_blinding_offsets(blind=blinded_fits[i])
                    fit(args, ifitter, ws, dofit=ifit >= 0)
                    fit_time.append(time.time())

                    if args.saveHists:
                        save_hists(
                            args,
                            models,
                            ifitter,
                            ws,
                            prefit=False,
                            profile=args.externalPostfit is None,
                        )
                else:
                    fit_time.append(time.time())

                ws.dump_and_flush(group)
                postfit_time.append(time.time())

    end_time = time.time()
    logger.info(f"{end_time - start_time:.2f} seconds total time")
    logger.debug(f"{init_time - start_time:.2f} seconds initialization time")
    for i, ifit in enumerate(fits):
        logger.debug(f"For fit {ifit}:")
        dt = init_time if i == 0 else fit_time[i - 1]
        t0 = prefit_time[i] - dt
        t1 = fit_time[i] - prefit_time[i]
        t2 = postfit_time[i] - fit_time[i]
        logger.debug(f"{t0:.2f} seconds for prefit")
        logger.debug(f"{t1:.2f} seconds for fit")
        logger.debug(f"{t2:.2f} seconds for postfit")


if __name__ == "__main__":
    main()
