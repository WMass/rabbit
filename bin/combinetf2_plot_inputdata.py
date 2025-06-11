#!/usr/bin/env python3

import argparse
import itertools
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools

from combinetf2 import debugdata, inputdata


def parseArgs():

    # choices for legend padding
    choices_padding = ["auto", "lower left", "lower right", "upper left", "upper right"]

    parser = argparse.ArgumentParser()

    parser.add_argument("infile", help="Output h5py file of the setupCombine.py")
    parser.add_argument(
        "--logx", action="store_true", help="Enable log scale for x axis"
    )
    parser.add_argument(
        "--logy", action="store_true", help="Enable log scale for y axis"
    )
    parser.add_argument(
        "--invertAxes",
        action="store_true",
        help="Invert the order of the axes when plotting",
    )
    parser.add_argument("--noData", action="store_true", help="Don't plot data")
    parser.add_argument("--noRatio", action="store_true", help="Don't plot the ratio")
    parser.add_argument(
        "--noStack", action="store_true", help="Don't plot the individual processes"
    )
    parser.add_argument(
        "--processes", type=str, nargs="*", default=[], help="Select processes"
    )
    parser.add_argument(
        "--channels", type=str, nargs="*", default=None, help="Select channels"
    )
    parser.add_argument(
        "--splitByProcess",
        action="store_true",
        help="Make a separate plot for each of the selected processes",
    )
    parser.add_argument(
        "--selectionAxes",
        type=str,
        nargs="*",
        default=["charge", "passIso", "passMT"],
        help="List of axes where for each bin a seperate plot is created",
    )
    parser.add_argument(
        "--select",
        type=int,
        nargs="*",
        default=[],
        help="Select specific bins of the selectionAxis e.g. '0 1' to select the first bin of the first axis and second bin of the second axis",
    )
    parser.add_argument(
        "--hists",
        type=str,
        nargs="*",
        default=None,
        help="List of hists to plot; dash separated for unrolled hists",
    )
    parser.add_argument(
        "--normToData", action="store_true", help="Normalize MC to data", default=False
    )
    parser.add_argument(
        "--pseudodata",
        type=str,
        default=None,
        help="Pseuododata name to plot instead of data.",
    )
    parser.add_argument(
        "--dataName", type=str, default="Data", help="Data name for plot labeling"
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
    # variations
    parser.add_argument(
        "--varName", type=str, nargs="*", default=[], help="Name of variation hist"
    )
    parser.add_argument(
        "--varLabel",
        type=str,
        nargs="*",
        default=[],
        help="Label(s) of variation hist for plotting",
    )
    parser.add_argument(
        "--varColor", type=str, nargs="*", default=[], help="Variation colors"
    )
    parser.add_argument(
        "--oneSidedVariations",
        action="store_true",
        help="Plot variations one sided",
    )
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
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
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
        "--scaleTextSize",
        type=float,
        default=1.0,
        help="Scale all text sizes by this number",
    )
    parser.add_argument(
        "--customFigureWidth",
        type=float,
        default=None,
        help="Use a custom figure width, otherwise chosen automatic",
    )

    args = parser.parse_args()

    return args


def make_plots(
    args,
    config,
    outdir,
    indata,
    hists_proc,
    hist_data,
    *opts,
    labels=None,
    colors=None,
    procs=None,
    **info,
):
    # make full unrollsed plot and lower dimensional projections

    all_axes_names = [n for n in hists_proc[0].axes.name]
    if args.invertAxes:
        logger.info("invert axes order")
        all_axes_names = all_axes_names[::-1]

    axes_combinations = all_axes_names[:]
    # make lower dimensional combinations of axes
    for n in range(2, len(all_axes_names) + 1):
        axes_combinations += [k for k in itertools.combinations(all_axes_names, n)]

    for axes_names in axes_combinations:
        if isinstance(axes_names, str):
            axes_names = [axes_names]

        if args.hists:
            if not any(set(axes_names) == set(h.split("-")) for h in args.hists):
                continue

        logger.info(f"Make plot(s) with axes {axes_names}")

        if args.splitByProcess:

            for i, hp in enumerate(hists_proc):
                if (hp.values() != 0).sum() == 0:
                    # skip processes with all 0
                    continue

                make_plot(
                    args,
                    config,
                    outdir,
                    indata,
                    [hp],
                    hist_data,
                    axes_names=axes_names,
                    *opts,
                    labels=labels[i],
                    colors=colors[i],
                    procs=procs[i],
                    **info,
                )
        else:
            make_plot(
                args,
                config,
                outdir,
                indata,
                hists_proc,
                hist_data,
                axes_names=axes_names,
                *opts,
                labels=labels,
                colors=colors,
                procs=procs,
                **info,
            )


def make_plot(
    args,
    config,
    outdir,
    indata,
    hists_proc,
    hist_data,
    hists_syst_up,
    hists_syst_dn,
    axes_names,
    selections=None,
    selection_edges=None,
    channel="",
    colors=[],
    labels=[],
    procs=[],
    rlabel="1/Pred.",
    density=False,
    systematics=None,
    colors_syst=None,
    labels_syst=None,
):
    if args.processGrouping is not None:
        hists_proc, labels, colors, procs = config.process_grouping(
            args.processGrouping, hists_proc, procs
        )

    if any(x in axes_names for x in ["ptll", "mll", "ptVgen", "ptVGen"]):
        # in case of variable bin width normalize to unit
        binwnorm = 1.0
        ylabel = "Events/unit"
    else:
        binwnorm = None
        ylabel = "Events/bin"

    if args.ylabel is not None:
        ylabel = args.ylabel

    if len(axes_names) == 1:
        fu = lambda h: h.project(*axes_names)
    else:
        fu = lambda h: hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names)

    # make 1D histograms
    h_stack = [fu(h) for h in hists_proc]

    if hist_data is not None:
        h_data = fu(hist_data)

    xlabel = plot_tools.get_axis_label(config, axes_names, args.xlabel)

    if args.splitByProcess:
        hists_pred = h_stack
    else:
        hists_pred = [hh.sumHists(h_stack)]

        if args.normToData:
            scale = h_data.values().sum() / hists_pred[0].values().sum()
            h_stack = [hh.scaleHist(h, scale) for h in h_stack]
            hists_pred[0] = hh.scaleHist(hists_pred[0], scale)
            hists_syst_up = [hh.scaleHist(h, scale) for h in hists_syst_up]
            hists_syst_dn = [hh.scaleHist(h, scale) for h in hists_syst_dn]

    # loop over all processes if plots for each process is requested, or inclusive otherwise
    for i, h_pred in enumerate(hists_pred):

        infos_figure = dict(
            xlabel=xlabel,
            ylabel=ylabel,
            logy=args.logy,
            logx=args.logx,
            xlim=args.xlim,
            ylim=args.ylim,
        )
        if not args.noRatio:
            fig, ax1, ratio_axes = plot_tools.figureWithRatio(
                h_pred, rlabel=rlabel, rrange=args.rrange, **infos_figure
            )
            ax2 = ratio_axes[-1]
        else:
            fig, ax1 = plot_tools.figure(h_pred, **infos_figure)

        if args.noStack or args.splitByProcess:
            hep.histplot(
                h_pred,
                xerr=False,
                yerr=False,
                histtype="step",
                color="black",
                label=labels[i] if args.splitByProcess else "Prediction",
                binwnorm=binwnorm,
                ax=ax1,
                zorder=1,
                flow="none",
            )
        else:
            hep.histplot(
                h_stack,
                xerr=False,
                yerr=False,
                histtype="fill",
                color=colors,
                label=labels,
                stack=True,
                density=False,
                binwnorm=binwnorm,
                ax=ax1,
                zorder=1,
                flow="none",
            )

        if hist_data is not None:
            hep.histplot(
                h_data,
                yerr=True,
                histtype="errorbar",
                color="black",
                label=args.dataName,
                binwnorm=binwnorm,
                ax=ax1,
                alpha=1.0,
                zorder=2,
                flow="none",
            )

        for hup, hdn, color, label in zip(
            hists_syst_up, hists_syst_dn, colors_syst, labels_syst
        ):
            if args.splitByProcess:
                hup = hup[{"processes": procs[i]}]
                hdn = hdn[{"processes": procs[i]}]
            else:
                hup = hup[{"processes": hist.sum}]
                hdn = hdn[{"processes": hist.sum}]

            hup = fu(hup)
            hdn = fu(hdn)

            hep.histplot(
                hup,
                xerr=False,
                yerr=False,
                histtype="step",
                color=color,
                label=label,
                binwnorm=binwnorm,
                ax=ax1,
                zorder=1,
                flow="none",
            )
            if not args.oneSidedVariations:
                hep.histplot(
                    hdn,
                    xerr=False,
                    yerr=False,
                    histtype="step",
                    color=color,
                    linestyle="--",
                    binwnorm=binwnorm,
                    ax=ax1,
                    zorder=1,
                    flow="none",
                )

            if not args.noRatio:
                h_ratios = (
                    [hh.divideHists(hup, h_pred)]
                    if args.oneSidedVariations
                    else [hh.divideHists(hup, h_pred), hh.divideHists(hdn, h_pred)]
                )
                hep.histplot(
                    h_ratios,
                    xerr=False,
                    yerr=False,
                    histtype="step",
                    color=color,
                    linestyle=["-", "--"][: len(h_ratios)],
                    ax=ax2,
                    linewidth=2,
                    flow="none",
                )

        if not args.noRatio:
            hep.histplot(
                hh.divideHists(
                    h_pred,
                    h_pred,
                    cutoff=1e-8,
                    rel_unc=True,
                    flow=False,
                    by_ax_name=False,
                ),
                histtype="step",
                color="black",
                alpha=0.5,
                yerr=False,
                ax=ax2,
                linewidth=1,
                flow="none",
            )

            if hist_data is not None:
                hep.histplot(
                    hh.divideHists(h_data, h_pred, cutoff=0.01, rel_unc=True),
                    histtype="errorbar",
                    color="black",
                    yerr=True,
                    linewidth=2,
                    ax=ax2,
                )

        scale = max(1, np.divide(*ax1.get_figure().get_size_inches()) * 0.3)

        if selections is not None:
            text = []
            for i, (key, idx) in enumerate(selections.items()):
                lo, hi = selection_edges[i]
                if key == "charge":
                    label = f"charge = {'-1' if hi==0 else '+1'}"
                else:
                    label = plot_tools.get_axis_label(config, key, with_unit=False)
                    if lo != None:
                        label = f"{lo} < {label}"
                    if hi != None:
                        label = f"{label} < {hi}"
                text.append(label)

            plot_tools.wrap_text(
                text,
                ax1,
                # 0.05,
                *args.extraTextLoc,
                # 0.96 - i * 0.08,
                ha="left",
                text_size="small",
            )

        if not args.noRatio:
            plot_tools.fix_axes(ax1, ax2, fig, yscale=args.yscale, noSci=args.noSciy)
        else:
            plot_tools.fix_axes(ax1, yscale=args.yscale, logy=args.logy)

        plot_tools.add_decor(
            ax1,
            args.title,
            args.subtitle,
            data=hist_data is not None or "Nonprompt" in labels,
            lumi=args.lumi,  # if args.dataName == "Data" and not args.noData else None,
            loc=args.titlePos,
            text_size=args.legSize,
        )

        plot_tools.addLegend(
            ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
        )

        outfile = "hist_"
        if not args.noStack:
            outfile += "stack_"
        if args.splitByProcess:
            outfile += f"{procs[i]}_"
        outfile += "_".join(axes_names)
        outfile += f"_{channel}"
        if selections is not None:
            outfile += "_" + "_".join([f"{a}{i}" for a, i in selections.items()])
        if args.postfix:
            outfile += f"_{args.postfix}"
        plot_tools.save_pdf_and_png(outdir, outfile)

        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={"setupCombine": indata.metadata["meta_info"]},
            args=args,
        )


def main():
    args = parseArgs()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    indata = inputdata.FitInputData(args.infile, pseudodata=args.pseudodata)

    debug = debugdata.FitDebugData(indata)

    config = plot_tools.load_config(args.config)

    syst_labels = getattr(config, "systematics_labels", {})
    systematics = []
    colors_syst = []
    labels_syst = []
    for syst, color, label in itertools.zip_longest(
        args.varName, args.varColor, args.varLabel
    ):
        if syst not in indata.systs.astype(str):
            logger.error(f"Syst {syst} not available, skip!")
            continue
        systematics.append(syst)
        colors_syst.append(color if color is not None else "black")
        labels_syst.append(label if label is not None else syst_labels.get(syst, syst))

    if args.oneSidedVariations:
        labels_syst = [l.replace(r"\pm", "+") for l in labels_syst]

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    for channel in indata.channel_info.keys():
        if args.channels is not None and channel not in args.channels:
            continue
        logger.info(f"Make plots for channel: {channel}")

        hist_proc = debug.nominal_hists[channel]
        procs = [p for p in hist_proc.axes["processes"]]

        if len(args.processes):
            procs_tmp = procs[:]
            procs = []
            for p in args.processes:
                if p not in procs_tmp:
                    logger.warning(f"Process {p} requested but not found, skip")
                    continue
                procs.append(p)

        if hasattr(config, "get_labels_colors_procs_sorted"):
            labels, colors, procs = config.get_labels_colors_procs_sorted(procs)
        else:
            labels = procs[:]
            cmap = plt.get_cmap("tab10")
            proc_colors = getattr(config, "process_colors", {})
            colors = [proc_colors.get(p, cmap(i % cmap.N)) for i, p in enumerate(procs)]

        hists_proc = [hist_proc[{"processes": p}] for p in procs]

        if len(systematics):
            hist_syst = debug.syst_hists[channel]
            hists_syst_dn = [
                hist_syst[{"DownUp": "Down", "systs": s}] for s in systematics
            ]
            hists_syst_up = [
                hist_syst[{"DownUp": "Up", "systs": s}] for s in systematics
            ]
        else:
            hists_syst_dn = []
            hists_syst_up = []

        # setup data histogram
        if (
            args.noData
            or args.splitByProcess
            or channel not in debug.data_obs_hists.keys()
        ):
            hist_data = None
        else:
            hist_data_tmp = debug.data_obs_hists[channel]

            # poisson errors on data hist for correct errors in ratio plot
            hist_data = hist.Hist(*hist_data_tmp.axes, storage=hist.storage.Weight())
            hist_data.values(flow=True)[...] = hist_data_tmp.values(flow=True)
            hist_data.variances(flow=True)[...] = hist_data_tmp.values(flow=True)

        info = dict(
            channel=channel,
            systematics=systematics,
            colors_syst=colors_syst,
            labels_syst=labels_syst,
        )

        # make plots in slices (e.g. for charge plus an minus separately)
        selection_axes = [
            hists_proc[0].axes[n]
            for n in args.selectionAxes
            if n in hists_proc[0].axes.name
        ]
        if len(selection_axes) > 0:
            other_axes = [a for a in hists_proc[0].axes if a not in selection_axes]
            if len(args.select):
                bin_combinations = [args.select]
            else:
                selection_bins = [
                    np.arange(a.size)
                    for a in hists_proc[0].axes
                    if a.name in args.selectionAxes
                ]
                bin_combinations = itertools.product(*selection_bins)

            for bins in bin_combinations:
                idxs = {a.name: i for a, i in zip(selection_axes, bins)}
                selection_edges = [
                    (a.edges[i], a.edges[i + 1] if len(a.edges - 1) > i else None)
                    for a, i in zip(selection_axes, bins)
                ]

                hs_proc = [h[idxs] for h in hists_proc]

                if hist_data is not None:
                    h_data = hist_data[idxs]
                else:
                    h_data = None

                hs_syst_dn = [h[idxs] for h in hists_syst_dn]
                hs_syst_up = [h[idxs] for h in hists_syst_up]

                make_plots(
                    args,
                    config,
                    outdir,
                    indata,
                    hs_proc,
                    h_data,
                    hs_syst_dn,
                    hs_syst_up,
                    selections=idxs,
                    selection_edges=selection_edges,
                    labels=labels,
                    colors=colors,
                    procs=procs,
                    **info,
                )
        else:
            make_plots(
                args,
                config,
                outdir,
                indata,
                hists_proc,
                hist_data,
                hists_syst_dn,
                hists_syst_up,
                labels=labels,
                colors=colors,
                procs=procs,
                **info,
            )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
