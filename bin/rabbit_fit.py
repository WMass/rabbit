#!/usr/bin/env python3

import tensorflow as tf

tf.config.experimental.enable_op_determinism()

import time

import numpy as np
from scipy.stats import chi2

from rabbit import fitter, inputdata, parsing, workspace
from rabbit.mappings import helpers as mh
from rabbit.mappings import mapping as mp
from rabbit.poi_models import helpers as ph
from rabbit.tfhelpers import edmval_cov

from wums import output_tools, logging  # isort: skip

logger = None


def make_parser():
    parser = parsing.common_parser()
    parser.add_argument("--outname", default="fitresults.hdf5", help="output file name")
    parser.add_argument(
        "--fullNll",
        action="store_true",
        help="Calculate and store full value of -log(L)",
    )
    parser.add_argument(
        "--contourScan",
        default=None,
        type=str,
        nargs="*",
        help="run likelihood contour scan on the specified variables, specify w/o argument for all parameters",
    )
    parser.add_argument(
        "--contourLevels",
        default=[1.0, 2.0],
        type=float,
        nargs="+",
        help="Confidence level in standard deviations for contour scans (1 = 1 sigma = 68%%)",
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
        "--noPostfitProfileBB",
        default=False,
        action="store_true",
        help="Do not profile the bin-by-bin parameters in the postfit (e.g. if parameters are loaded from another fit using --externalPostfit and/or no data is available to be profiled).",
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
        "--globalImpactsDisableJVP",
        default=False,
        action="store_true",
        help="""
        Compute global impacts on parameters and observables in the traditional 'backward mode' 
        and not using the more memory efficient 'forward mode' implementation using jacobian vector products (JVP)
        """,
    )
    parser.add_argument(
        "--nonProfiledImpacts",
        default=False,
        action="store_true",
        help="compute impacts of frozen (non-profiled) systematics",
    )

    return parser.parse_args()


def save_observed_hists(args, mappings, fitter, ws):
    for mapping in mappings:
        if not mapping.has_data:
            continue

        print(f"Save data histogram for {mapping.key}")
        ws.add_observed_hists(
            mapping,
            fitter.indata.data_obs,
            fitter.nobs.value(),
            fitter.indata.data_cov_inv,
            fitter.data_cov_inv,
        )


def save_hists(args, mappings, fitter, ws, prefit=True, profile=False):

    for mapping in mappings:
        logger.info(f"Save inclusive histogram for {mapping.key}")

        if prefit and mapping.skip_prefit:
            continue

        if not mapping.skip_incusive:
            exp, aux = fitter.expected_events(
                mapping,
                inclusive=True,
                compute_variance=args.computeHistErrors,
                compute_cov=args.computeHistCov,
                compute_chi2=not args.noChi2 and mapping.has_data,
                compute_global_impacts=args.computeHistImpacts,
                profile=profile,
            )

            ws.add_expected_hists(
                mapping,
                exp,
                var=aux[0],
                cov=aux[1],
                impacts=aux[2],
                impacts_grouped=aux[3],
                prefit=prefit,
            )

            if aux[4] is not None:
                ws.add_chi2(aux[4], aux[5], prefit, mapping)

        if args.saveHistsPerProcess and not mapping.skip_per_process:
            logger.info(f"Save processes histogram for {mapping.key}")

            exp, aux = fitter.expected_events(
                mapping,
                inclusive=False,
                compute_variance=args.computeHistErrorsPerProcess,
                profile=profile,
            )

            ws.add_expected_hists(
                mapping,
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
                mapping,
                inclusive=True,
                compute_variance=False,
                compute_variations=True,
                profile=profile,
            )

            ws.add_expected_hists(
                mapping,
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
        fitter.load_fitresult(args.externalPostfit, args.externalPostfitResult)

    if dofit:
        cb = fitter.minimize()

        # force profiling of beta with final parameter values
        # TODO avoid the extra calculation and jitting if possible since the relevant calculation
        # usually would have been done during the minimization
        if fitter.binByBinStat and not args.noPostfitProfileBB:
            fitter._profile_beta()

        if cb is not None:
            ws.add_loss_time_hist(cb.loss_history, cb.time_history)

    if not args.noHessian:
        # compute the covariance matrix and estimated distance to minimum

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
        tf.size(fitter.nobs) - fitter.poi_model.npoi - fitter.indata.nsystnoconstraint
    ).numpy()

    chi2_val = 2.0 * nllvalreduced
    p_val = chi2.sf(chi2_val, ndfsat)

    logger.info("Saturated chi2:")
    logger.info(f"    ndof: {ndfsat}")
    logger.info(f"    2*deltaNLL: {round(chi2_val, 2)}")
    logger.info(rf"    p-value: {round(p_val * 100, 2)}%")

    ws.results.update(
        {
            "nllvalreduced": nllvalreduced,
            "ndfsat": ndfsat,
            "edmval": edmval,
            "postfit_profile": not args.noPostfitProfileBB,
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

    if args.nonProfiledImpacts:
        # TODO: based on covariance
        ws.add_nonprofiled_impacts_hist(*fitter.nonprofiled_impacts_parms())

    # Likelihood scans
    if args.scan is not None:
        parms = np.array(fitter.parms).astype(str) if len(args.scan) == 0 else args.scan

        for param in parms:
            logger.info(f"-delta log(L) scan for {param}")
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

    # Likelihood contour scans
    if args.contourScan is not None:
        nllvalreduced = fitter.reduced_nll().numpy()

        parms = (
            np.array(fitter.parms).astype(str)
            if len(args.contourScan) == 0
            else args.contourScan
        )

        contours = np.zeros((len(parms), len(args.contourLevels), 2, len(fitter.parms)))
        for i, param in enumerate(parms):
            logger.info(f"Contour scan for {param}")
            for j, cl in enumerate(args.contourLevels):
                logger.info(f"    Now at CL {cl}")

                # find confidence interval
                _, params = fitter.contour_scan(param, nllvalreduced, cl**2)
                contours[i, j, ...] = params

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

    margs = args.poiModel
    poi_model = ph.load_model(margs[0], indata, *margs[1:], **vars(args))

    ifitter = fitter.Fitter(
        indata,
        poi_model,
        args,
        do_blinding=any(blinded_fits),
        globalImpactsFromJVP=not args.globalImpactsDisableJVP,
    )

    # mappings for observables and parameters
    if len(args.mapping) == 0 and args.saveHists:
        # if no mapping is explicitly added and --saveHists is specified, fall back to BaseMapping
        args.mapping = [["BaseMapping"]]
    mappings = []
    for margs in args.mapping:
        mapping = mh.load_mapping(margs[0], indata, *margs[1:])
        mappings.append(mapping)

    if args.compositeMapping:
        mappings = [
            mp.CompositeMapping(mappings),
        ]

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # pass meta data into output file
    meta = {
        "meta_info": output_tools.make_meta_info_dict(args=args),
        "meta_info_input": ifitter.indata.metadata,
        "procs": ifitter.indata.procs,
        "pois": ifitter.poi_model.pois,
        "nois": ifitter.parms[ifitter.poi_model.npoi :][indata.noiidxs],
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
            group = ["results"]

            if args.pseudoData is None:
                datasets = zip([ifitter.indata.data_obs], [ifitter.indata.data_var])
            else:
                # shape nobs x npseudodata
                datasets = zip(
                    tf.transpose(indata.pseudodata_obs),
                    (
                        tf.transpose(indata.pseudodata_var)
                        if indata.pseudodata_var is not None
                        else [None] * indata.pseudodata_obs.shape[-1]
                    ),
                )

            # loop over (pseudo)data sets
            for j, (data_values, data_variances) in enumerate(datasets):

                ifitter.defaultassign()
                if ifit == -1:
                    group.append("asimov")
                    ifitter.set_nobs(ifitter.expected_yield())
                else:
                    if ifit == 0:
                        ifitter.set_nobs(data_values)
                    elif ifit >= 1:
                        group.append(f"toy{ifit}")
                        ifitter.toyassign(
                            data_values,
                            data_variances,
                            syst_randomize=args.toysSystRandomize,
                            data_randomize=args.toysDataRandomize,
                            data_mode=args.toysDataMode,
                            randomize_parameters=args.toysRandomizeParameters,
                        )

                if args.pseudoData is not None:
                    # label each pseudodata set
                    if j == 0:
                        group.append(indata.pseudodatanames[j])
                    else:
                        group[-1] = indata.pseudodatanames[j]

                ws.add_parms_hist(
                    values=ifitter.x,
                    variances=tf.linalg.diag_part(ifitter.cov),
                    hist_name="parms_prefit",
                )

                if args.saveHists:
                    save_observed_hists(args, mappings, ifitter, ws)
                    save_hists(args, mappings, ifitter, ws, prefit=True)
                prefit_time.append(time.time())

                if not args.prefitOnly:
                    ifitter.set_blinding_offsets(blind=blinded_fits[i])
                    fit(args, ifitter, ws, dofit=ifit >= 0)
                    fit_time.append(time.time())

                    if args.saveHists:
                        save_hists(
                            args,
                            mappings,
                            ifitter,
                            ws,
                            prefit=False,
                            profile=not args.noPostfitProfileBB,
                        )
                else:
                    fit_time.append(time.time())

                ws.dump_and_flush("_".join(group))
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
