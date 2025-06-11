#!/usr/bin/env python3

import argparse
import inspect
import itertools
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import colormaps

import combinetf2.io_tools

from wums import boostHistHelpers as hh  # isort: skip
from wums import logging, output_tools, plot_tools  # isort: skip


hep.style.use(hep.style.ROOT)

logger = None


def parseArgs():

    # choices for legend padding
    choices_padding = ["auto", "lower left", "lower right", "upper left", "upper right"]

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
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("./test"),
        help="Base path for output",
    )
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Override use of xrdcp and use the mount instead",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help="Pre-defined grouping in config to select nuisance groups",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=16.8,
        help="Luminosity used in the fit, needed to get the absolute cross section",
    )
    parser.add_argument(
        "--title",
        default="CombineTF2",
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
        "--scaleTextSize",
        type=float,
        default=1.0,
        help="Scale all text sizes by this number",
    )
    parser.add_argument(
        "infile",
        type=str,
        help="hdf5 file from combinetf2 or root file from combinetf1",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--logy", action="store_true", help="Make the yscale logarithmic"
    )
    parser.add_argument(
        "--noUncertainty", action="store_true", help="Don't plot total uncertainty band"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Give the uncertainties in absolute numbers (relative by default)",
    )
    parser.add_argument(
        "-m",
        "--physicsModel",
        nargs="+",
        action="append",
        default=[],
        help="""
        Make plot of physics model prefit and postfit histograms. Loop over all by deault. 
        Can also specify the model name, followed by the arguments, e.g. "-m Project ch0 eta pt". 
        This argument can be called multiple times.
        """,
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="List of channels to be plotted, default is all",
    )
    parser.add_argument(
        "--flterUncertainties",
        type=str,
        nargs="*",
        default=None,
        help="Only plot the filtered processes",
    )
    parser.add_argument(
        "--suppressProcsLabel",
        type=str,
        nargs="*",
        default=[],
        help="Don't show given processes in the legends",
    )
    parser.add_argument(
        "--selectionAxes",
        type=str,
        nargs="*",
        default=["charge", "passIso", "passMT", "cosThetaStarll", "qGen"],
        help="List of axes where for each bin a separate plot is created",
    )
    parser.add_argument(
        "--axlim",
        type=float,
        default=None,
        nargs="*",
        help="min and max for axes (2 values per axis)",
    )
    parser.add_argument(
        "--invertAxes",
        action="store_true",
        help="Invert the order of the axes when plotting",
    )
    parser.add_argument(
        "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
    )
    parser.add_argument(
        "--processGrouping", type=str, default=None, help="key for grouping processes"
    )
    parser.add_argument(
        "--extraTextLoc",
        type=float,
        nargs="*",
        default=None,
        help="Location in (x,y) for additional text, aligned to upper left",
    )
    parser.add_argument(
        "--varNames", type=str, nargs="*", default=None, help="Name of variation hist"
    )
    parser.add_argument(
        "--varLabels",
        type=str,
        nargs="*",
        default=None,
        help="Label(s) of variation hist for plotting",
    )
    parser.add_argument(
        "--varColors",
        type=str,
        nargs="*",
        default=None,
        help="Color(s) of variation hist for plotting",
    )
    parser.add_argument(
        "--varOneSided",
        type=int,
        nargs="*",
        default=[],
        help="Only plot one sided variation (1) or two default two-sided (0)",
    )
    parser.add_argument(
        "--scaleVariation",
        nargs="*",
        type=float,
        default=[],
        help="Scale a variation by this factor",
    )
    parser.add_argument(
        "--subplotSizes",
        nargs=2,
        type=int,
        default=[4, 2],
        help="Relative sizes for upper and lower panels",
    )
    parser.add_argument(
        "--correlatedVariations", action="store_true", help="Use correlated variations"
    )
    args = parser.parse_args()

    return args


def make_plot(
    outdir,
    h_impacts,
    h_total,
    axes,
    uncertainties,
    colors=None,
    labels=None,
    args=None,
    suffix="",
    meta=None,
    lumi=None,
    selection=None,
    config={},
):
    axes_names = [a.name for a in axes]

    if args.ylabel is not None:
        ylabel = args.ylabel
    else:
        ylabel = (
            r"Relative uncertainty in %"
            if not args.absolute
            else "Absolute uncertainty"
        )

    if len(axes) > 1:
        if args.invertAxes:
            axes_names = axes_names[::-1]
            axes = axes[::-1]

    xlabel = plot_tools.get_axis_label(config, axes_names, args.xlabel)

    fig, ax1 = plot_tools.figure(
        h_total, xlabel, ylabel, args.ylim, automatic_scale=False, width_scale=1.2
    )

    translate_label = getattr(config, "systematics_labels", {})
    grouping = getattr(config, "nuisance_grouping", {}).get(args.grouping, None)
    for (
        u,
        c,
        l,
    ) in zip(uncertainties, colors, labels):

        if grouping is not None and l not in grouping:
            continue

        h_impact = h_impacts[{"impacts": u}]
        if len(axes) > 1:
            # unrolled ND histograms
            h_impact = hh.unrolledHist(h_impact, obs=[a.name for a in axes])

        if len(h_impacts.axes) == 1:
            h_impact = hist.Hist(
                hist.axis.Integer(0, 1, name="yield", overflow=False, underflow=False),
                data=h_impact.value,
            )

        hep.histplot(
            h_impact,
            xerr=False,
            yerr=False,
            histtype="step",
            color=c,
            label=translate_label.get(l, l),
            density=False,
            ax=ax1,
            zorder=1,
            flow="none",
        )

    if grouping is not None and "Total" in grouping:
        if len(axes) > 1:
            # unrolled ND histograms
            h_total = hh.unrolledHist(h_total, obs=[a.name for a in axes])
        hep.histplot(
            h_total,
            xerr=False,
            yerr=False,
            histtype="step",
            color="black",
            linestyle="dashed",
            label=translate_label.get("Total", "Total"),
            density=False,
            ax=ax1,
            zorder=1,
            flow="none",
        )

    text_pieces = []
    if selection is not None:
        text_pieces.extend(selection)

    plot_tools.addLegend(
        ax1,
        ncols=args.legCols,
        loc=args.legPos,
        text_size=args.legSize,
        extra_text=text_pieces,
        extra_text_loc=None if args.extraTextLoc is None else args.extraTextLoc[:2],
        padding_loc=args.legPadding,
    )

    plot_tools.add_decor(
        ax1,
        args.title,
        args.subtitle,
        lumi=None,
        loc=args.titlePos,
        text_size=args.legSize,
    )

    plot_tools.fix_axes(ax1, None, fig, yscale=args.yscale, noSci=True)

    to_join = ["uncertainties", args.postfix, *axes_names, suffix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if args.absolute:
        outfile += "_absolute"
    if args.subtitle == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)

    analysis_meta_info = None
    if meta is not None:
        if "meta_info_input" in meta:
            analysis_meta_info = {
                "Combinetf2Output": meta["meta_info"],
                "AnalysisOutput": meta["meta_info_input"]["meta_info"],
            }
        else:
            analysis_meta_info = {"AnalysisOutput": meta["meta_info"]}

    output_tools.write_logfile(
        outdir,
        outfile,
        args=args,
        meta_info={**analysis_meta_info},
    )


def make_plots(
    outdir,
    result,
    config,
    args=None,
    channel="",
    lumi=1,
    *opts,
    **kwopts,
):
    if "hist_postfit_inclusive_global_impacts_grouped" in result.keys():
        hist_impacts = result["hist_postfit_inclusive_global_impacts_grouped"].get()
    else:
        hist_impacts = None

    hist_total = result["hist_postfit_inclusive"].get()
    axes = [a for a in hist_total.axes]

    if hist_impacts is not None:
        uncertainties = np.array(hist_impacts.axes["impacts"], dtype=str)
    else:
        uncertainties = np.array([])

    if not args.absolute:
        # give impacts as relative uncertainty
        if hist_impacts is not None:
            hist_impacts = hh.divideHists(hist_impacts, hist_total, rel_unc=True)
            hist_impacts = hh.scaleHist(hist_impacts, 100)  # impacts in %

        hist_total = hh.divideHists(hist_total, hist_total, rel_unc=True)

    hist_total.values()[...] = hist_total.variances()[...] ** 0.5

    if not args.absolute:
        hist_total = hh.scaleHist(hist_total, 100)  # impacts in %

    if args.flterUncertainties is not None:
        uncertainties = [p for p in uncertainties if p in args.flterUncertainties]
    cmap = plt.get_cmap("tab10" if len(uncertainties) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(uncertainties))]
    labels = uncertainties[:]

    # make plots in slices (e.g. for charge plus an minus separately)
    selection_axes = [a for a in axes if a.name in args.selectionAxes]
    if len(selection_axes) > 0:
        selection_bins = [
            np.arange(a.size) for a in axes if a.name in args.selectionAxes
        ]
        other_axes = [a for a in axes if a not in selection_axes]

        for bins in itertools.product(*selection_bins):
            idxs = {a.name: i for a, i in zip(selection_axes, bins)}
            idxs_centers = {
                a.name: (
                    a.centers[i]
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    else a.edges[i]
                )
                for a, i in zip(selection_axes, bins)
            }
            idxs_edges = {
                a.name: (
                    (a.edges[i], a.edges[i + 1])
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    else a.edges[i]
                )
                for a, i in zip(selection_axes, bins)
            }

            h_impacts = hist_impacts[idxs] if hist_impacts else None
            h_total = hist_total[idxs]

            ts = getattr(config, "translate_selection", {})

            selection = []
            for a in selection_axes:
                n = a.name
                sel = ts.get(n, lambda x: f"{n}={x}")

                nparams = len(inspect.signature(sel).parameters)

                if nparams == 2:
                    selection.append(sel(*idxs_edges[n]))
                elif nparams == 1:
                    selection.append(sel(idxs_centers[n]))

            suffix = f"{channel}_" + "_".join(
                [
                    f"{a}_{str(i).replace('.','p').replace('-','m')}"
                    for a, i in idxs.items()
                ]
            )
            logger.info(
                f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}"
            )
            make_plot(
                outdir,
                h_impacts,
                h_total,
                other_axes,
                uncertainties,
                labels=labels,
                colors=colors,
                args=args,
                suffix=suffix,
                selection=selection,
                lumi=lumi,
                config=config,
                *opts,
                **kwopts,
            )
    else:
        make_plot(
            outdir,
            hist_impacts,
            hist_total,
            axes,
            uncertainties,
            labels=labels,
            colors=colors,
            args=args,
            suffix=channel,
            lumi=lumi,
            config=config,
            *opts,
            **kwopts,
        )


def main():
    """
    Plot the uncertainty breakdown of the histogram bins based on global impacts
    """
    args = parseArgs()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)
    varNames = args.varNames
    if varNames is not None:
        varLabels = args.varLabels
        varColors = args.varColors
        if len(varLabels) != len(varNames):
            raise ValueError(
                "Must specify the same number of args for --varNames, and --varLabels"
                f" found varNames={len(varNames)} and varLabels={len(varLabels)}"
            )
        if varColors is None:
            varColors = [
                colormaps["tab10" if len(varNames) < 10 else "tab20"](i)
                for i in range(len(varNames))
            ]

        varOneSided = [
            args.varOneSided[i] if i < len(args.varOneSided) else 0
            for i in range(len(varNames))
        ]
        scaleVariation = [
            args.scaleVariation[i] if i < len(args.scaleVariation) else 1
            for i in range(len(varNames))
        ]

    # load .hdf5 file first, must exist in combinetf and combinetf2
    fitresult, meta = combinetf2.io_tools.get_fitresult(
        args.infile, args.result, meta=True
    )

    plt.rcParams["font.size"] = plt.rcParams["font.size"] * args.scaleTextSize

    channel_info = meta["meta_info_input"]["channel_info"]

    opts = dict(
        args=args,
        meta=meta,
    )

    results = fitresult["physics_models"]
    for margs in args.physicsModel:
        if margs == []:
            instance_keys = results.keys()
        else:
            model_key = " ".join(margs)
            instance_keys = [k for k in results.keys() if k.startswith(model_key)]
            if len(instance_keys) == 0:
                raise ValueError(f"No model found under {model_key}")

        for instance_key in instance_keys:
            instance = results[instance_key]

            for channel, result in instance["channels"].items():
                logger.info(f"Make plot for {instance_key} in channel {channel}")
                if args.channels is not None and channel not in args.channels:
                    continue

                info = channel_info.get(channel, {})

                suffix = f"{channel}_{instance_key}"
                if args.correlatedVariations:
                    suffix += "_correlated"
                for sign, rpl in [
                    (" ", "_"),
                    (".", "p"),
                    ("-", "m"),
                    (":", ""),
                    (",", ""),
                    ("slice(None)", ""),
                    ("(", ""),
                    (")", ""),
                    (":", ""),
                ]:
                    suffix = suffix.replace(sign, rpl)

                make_plots(
                    outdir,
                    result,
                    config,
                    channel=suffix,
                    lumi=info.get("lumi", None),
                    **opts,
                )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
