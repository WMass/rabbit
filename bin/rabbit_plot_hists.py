#!/usr/bin/env python3

import inspect
import itertools

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import rabbit.io_tools
from rabbit import parsing

from wums import boostHistHelpers as hh  # isort: skip
from wums import logging, output_tools, plot_tools  # isort: skip

hep.style.use(hep.style.ROOT)

logger = None


def make_parser():
    parser = parsing.plot_parser()
    parsing.add_style_args(parser)
    parser.add_argument(
        "--noLowerPanel",
        action="store_true",
        help="Don't plot the lower panel in the plot",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Plot difference in lower panel instead of ratio",
    )
    parser.add_argument(
        "--dataHist",
        type=str,
        default="nobs",
        choices=["data_obs", "nobs"],
        help="Which data to plot ('data_obs': data histogram provided in input data; 'nobs': Plot (pseudo) data used in the fit)",
    )
    parser.add_argument("--noData", action="store_true", help="Don't plot the data")
    parser.add_argument(
        "--noUncertainty", action="store_true", help="Don't plot total uncertainty band"
    )
    parser.add_argument(
        "--normToData", action="store_true", help="Normalize MC to data"
    )
    parser.add_argument("--noStack", action="store_true", help="Don't stack processes")
    parser.add_argument("--density", action="store_true", help="Density")
    parser.add_argument(
        "--prefit", action="store_true", help="Make prefit plot, else postfit"
    )
    parser.add_argument(
        "-m",
        "--mapping",
        nargs="+",
        action="append",
        default=[],
        help="""
        Make plot of mapping prefit and postfit histograms. Loop over all by deault. 
        Can also specify the mapping name, followed by the arguments, e.g. "-m Project ch0 eta pt". 
        This argument can be called multiple times.
        """,
    )
    parser.add_argument(
        "--filterProcs",
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
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="List of channels to be plotted, default is all",
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
        "--chisq",
        type=str,
        default="automatic",
        choices=["automatic", "saturated", "linear", " ", "none", None],
        help="Type of chi2 to print on plot (saturated from fit likelihood. linear from observables, or none) 'automatic' means pick saturated for basemapping and otherwise linear",
    )
    parser.add_argument(
        "--dataName", type=str, default="Data", help="Data name for plot labeling"
    )
    parser.add_argument(
        "--predName",
        type=str,
        default="Pred.",
        help="Name for nominal prediction in plot labeling",
    )
    parser.add_argument(
        "--ratioToData",
        action="store_true",
        help="Make the ratio or diff w.r.t. prediction, (default is data)",
    )
    parser.add_argument(
        "--processGrouping", type=str, default=None, help="key for grouping processes"
    )
    parser.add_argument(
        "--binSeparationLines",
        type=float,
        default=None,
        nargs="*",
        help="Plot vertical lines for makro bin edges in unrolled plots, specify bin boundaries to plot lines, if empty plot for all",
    )
    parser.add_argument(
        "--noExtraText", action="store_true", help="Suppress extra text"
    )
    parser.add_argument(
        "--extraTextLoc",
        type=float,
        nargs="*",
        default=None,
        help="Location in (x,y) for additional text, aligned to upper left",
    )
    parser.add_argument(
        "--varFiles",
        type=str,
        nargs="*",
        default=[],
        help="Fitresult files with the variation hist",
    )
    parser.add_argument(
        "--varFilesFitTypes",
        type=str,
        nargs="*",
        default=["prefit"],
        choices=["prefit", "postfit"],
        help="Fit types for the fitresult files with the variation hist",
    )
    parser.add_argument(
        "--varMarkers",
        type=str,
        nargs="*",
        default=None,
        help="Use markers for variations, if provided plot variations using errorbar instead of step plots",
    )
    parser.add_argument(
        "--varNames",
        type=str,
        nargs="*",
        default=None,
        help="""
        Name of variation hist; for each varFile one varName has to be specified. 
        Additional varNames can be specified to add variations from the nominal input.
        """,
    )
    parser.add_argument(
        "--varGroups",
        type=str,
        nargs="*",
        default=[],
        help="""
        Names of grouped postfit histogram impacts to overlay as nominal +/- impact.
        These are read from hist_postfit_inclusive_global_impacts_grouped.
        """,
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
        "--showVariations",
        type=str,
        default="lower",
        choices=["upper", "lower", "both"],
        help="Plot the variations in the upper, lower panels, or both",
    )
    parser.add_argument(
        "--fillVariationsAlphas",
        type=float,
        nargs="*",
        default=[],
        help="Alpha values for filled two-sided variation envelopes; 0 disables filling",
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
    parser.add_argument(
        "--unfoldedXsec", action="store_true", help="Plot unfolded cross sections"
    )
    parser.add_argument(
        "--noPrefit",
        action="store_true",
        help="Don't plot prefit distribution",
    )
    parser.add_argument(
        "--noBinWidthNorm",
        action="store_true",
        help="Do not normalize bin yields by bin width",
    )
    parser.add_argument(
        "--upperPanelUncertaintyBand",
        action="store_true",
        help="Plot an uncertainty band in the upper panel around the prediction",
    )
    parser.add_argument(
        "--uncertaintyLabel",
        type=str,
        default=None,
        help="Label for uncertainty shown in the (ratio) plot",
    )
    parser.add_argument(
        "--dataCovariance",
        action="store_true",
        help="Use covariance information to plot the data uncertainty",
    )
    return parser


def make_plot(
    h_data,
    h_inclusive,
    h_stack,
    axes,
    outdir,
    config,
    colors=None,
    labels=None,
    args=None,
    hup=None,
    hdown=None,
    h_data_stat=None,
    variation="",
    suffix="",
    chi2=None,
    meta=None,
    saturated_chi2=False,
    lumi=None,
    selection=None,
    fittype="postfit",
    varNames=None,
    varLabels=None,
    varColors=None,
    varMarkers=None,
    is_normalized=False,
    binwnorm=1.0,
    counts=True,
    dataCovariance=False,
):
    ratio = not args.noLowerPanel and h_data is not None
    diff = not args.noLowerPanel and args.diff and h_data is not None
    data = not args.noData and h_data is not None

    axes_names = [a.name for a in axes]
    if len(axes_names) == 0:
        axes_names = ["yield"]
        return

    if args.density:
        ylabel = "Density"
    elif any(x.startswith("pt") or x.startswith("mll") for x in axes_names):
        # in case of variable bin width normalize to unit
        ylabel = (
            r"$Events\,/\,GeV$" if not args.unfoldedXsec else r"$d\sigma (pb\,/\,GeV)$"
        )
    else:
        ylabel = r"$Normalized\ units$" if is_normalized else r"$Events\,/\,unit$"
        if args.unfoldedXsec:
            ylabel = r"$d\sigma (pb)$"

    if args.ylabel is not None:
        ylabel = args.ylabel

    # compute event yield table before dividing by bin width
    yield_tables = {
        "stacked": pd.DataFrame(
            [
                (
                    k,
                    np.sum(h.project(*axes_names).values()),
                    np.sum(h.project(*axes_names).variances()) ** 0.5,
                )
                for k, h in zip(labels, h_stack)
            ],
            columns=["Process", "Yield", "Uncertainty"],
        ),
        "unstacked": pd.DataFrame(
            [
                (
                    k,
                    np.sum(h.project(*axes_names).values()),
                    np.sum(h.project(*axes_names).variances()) ** 0.5,
                )
                for k, h in zip(
                    [args.dataName, "Inclusive"],
                    [h_data, h_inclusive],
                )
                if h is not None
            ],
            columns=["Process", "Yield", "Uncertainty"],
        ),
    }

    histtype_data = "errorbar"
    if args.unfoldedXsec:
        histtype_mc = "errorbar"
    elif args.noStack:
        histtype_mc = "step"
    else:
        histtype_mc = "fill"

    if len(h_inclusive.axes) > 1:
        if args.invertAxes:
            logger.info("invert eta order")
            axes_names = axes_names[::-1]
            axes = axes[::-1]

        # make unrolled 1D histograms
        if (
            h_data is not None
            and binwnorm is not None
            and h_data.storage_type != hist.storage.Weight
        ):
            # need hist with variances to handle bin width normaliztion
            h_data_tmp = hist.Hist(
                *[a for a in h_data.axes], storage=hist.storage.Weight()
            )
            h_data_tmp.values()[...] = h_data.values()
            h_data_tmp.variances()[...] = h_data.values()
            h_data = h_data_tmp

        if h_data is not None:
            h_data = hh.unrolledHist(h_data, binwnorm=binwnorm, obs=axes_names)

        h_inclusive = hh.unrolledHist(h_inclusive, binwnorm=binwnorm, obs=axes_names)
        h_stack = [
            hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names) for h in h_stack
        ]
        if hup is not None:
            hup = [hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names) for h in hup]
        if hdown is not None:
            hdown = [
                hh.unrolledHist(h, binwnorm=binwnorm, obs=axes_names) for h in hdown
            ]
        if h_data_stat is not None:
            h_data_stat = hh.unrolledHist(
                h_data_stat, binwnorm=binwnorm, obs=axes_names
            )

    if args.normToData and h_data is not None:
        scale = h_data.values().sum() / h_inclusive.values().sum()
        h_stack = [hh.scaleHist(h, scale) for h in h_stack]
        h_inclusive = hh.scaleHist(h_inclusive, scale)

    xlabel = plot_tools.get_axis_label(config, axes_names, args.xlabel)

    # plot prediction first, the data on top
    zorder_pred = 0
    zorder_data = 1

    hatchstyle_pred = None
    facecolor_pred = "silver"
    facecolor_alpha_pred = 0.5
    pred_label = "Prefit model" if args.unfoldedXsec else args.predName

    # for uncertaity bands
    edges = h_inclusive.axes[0].edges

    # need to divide by bin width
    binwidth = edges[1:] - edges[:-1] if binwnorm else 1.0
    if h_inclusive.storage_type != hist.storage.Weight:
        raise ValueError(
            f"Did not find uncertainties in {fittype} hist. Make sure you run rabbit_fit with --computeHistErrors!"
        )

    if ratio or diff:
        if args.ratioToData:
            rlabel = r"Pred\ "
            if diff:
                rlabel += r"\,-\,"
            else:
                rlabel += r"\,/\,"

            rlabel = f"${rlabel} {args.dataName}$"
        elif args.noData:
            rlabel = ("Diff." if diff else "Ratio") + " to nominal"
        else:
            rlabel = args.dataName.replace(" ", r"\ ")
            if diff:
                rlabel += r"\,-\,"
            else:
                rlabel += r"\,/\,"

            rlabel = f"${rlabel} {args.predName}$"

        fig, ax1, ratio_axes = plot_tools.figureWithRatio(
            h_inclusive,
            xlabel,
            ylabel,
            args.ylim,
            rlabel,
            args.rrange,
            xlim=args.axlim,
            width_scale=(
                args.customFigureWidth
                if args.customFigureWidth is not None
                else 1.25 if len(axes_names) == 1 else 1
            ),
            automatic_scale=args.customFigureWidth is None,
            subplotsizes=args.subplotSizes,
            logy=args.logy,
        )
        ax2 = ratio_axes[-1]
    else:
        fig, ax1 = plot_tools.figure(
            h_inclusive, xlabel, ylabel, args.ylim, logy=args.logy
        )

    for (
        h,
        c,
        l,
    ) in zip(h_stack, colors, labels):
        # only for labels
        hep.histplot(
            h,
            xerr=False,
            yerr=False,
            histtype=histtype_mc,
            color=c,
            label=getattr(config, "process_labels", {}).get(l, l),
            density=args.density,
            binwnorm=binwnorm if not args.density else None,
            ax=ax1,
            zorder=1,
            flow="none",
        )

    if len(h_stack):
        hep.histplot(
            h_stack,
            xerr=False,
            yerr=False,
            histtype=histtype_mc,
            color=colors,
            stack=not args.noStack,
            density=args.density,
            binwnorm=binwnorm if not args.density else None,
            ax=ax1,
            zorder=1,
            flow="none",
        )

    extra_handles_upper = []
    extra_labels_upper = []

    if args.showVariations in ["upper", "both"]:
        linewidth = 2
        step_offset = 0
        for i, l in enumerate(varLabels):

            if (
                varMarkers is not None
                and len(varMarkers) > i
                and varMarkers[i] not in ["none", None]
            ):
                # plot errorbar if varMarkers are provided

                x = hup[i].axes.centers[0]
                x_diff = np.diff(hup[i].axes.edges[0])

                y_up = hup[i].values()
                y_dn = hdown[i].values()
                if binwnorm:
                    y_up = y_up / x_diff
                    y_dn = y_dn / x_diff

                y = (y_dn + y_up) / 2.0

                # plot markers slightly shifted
                step_size = 2 / (len(varMarkers) + 2)
                step = -1 + (i + 1.5) * step_size
                # make sure top make space to plot the data in the center
                if step == 0:
                    step_offset = step_size
                step += step_offset
                x = x + step * x_diff / 2

                ax1.errorbar(
                    x,
                    y,
                    color=varColors[i],
                    marker=varMarkers[i],
                    linestyle="none",
                    yerr=[y - y_dn, y_up - y],
                    label=l,
                )
                continue

            two_sided = (
                hdown is not None
                and hdown[i] is not None
                and len(args.varOneSided) == 0
            )
            fill_alpha = (
                args.fillVariationsAlphas[i]
                if i < len(args.fillVariationsAlphas)
                else 0.0
            )

            if fill_alpha > 0 and hup is not None and two_sided:
                edges = hup[i].axes[0].edges
                y_up = hup[i].values()
                y_dn = hdown[i].values()
                if binwnorm:
                    widths = edges[1:] - edges[:-1]
                    y_up = y_up / widths
                    y_dn = y_dn / widths

                ax1.fill_between(
                    edges,
                    np.append(y_up, y_up[-1]),
                    np.append(y_dn, y_dn[-1]),
                    step="post",
                    facecolor=varColors[i],
                    alpha=fill_alpha,
                    edgecolor=varColors[i],
                    linewidth=linewidth,
                    label=None,
                    zorder=1.5,
                )
                extra_handles_upper.append(
                    Polygon(
                        [[0, 0], [1, 0], [1, 1], [0, 1]],
                        closed=True,
                        facecolor=varColors[i],
                        alpha=fill_alpha,
                        edgecolor=varColors[i],
                        linewidth=linewidth,
                        linestyle="-",
                    )
                )
                extra_labels_upper.append(l)
                hep.histplot(
                    [hup[i], hdown[i]],
                    histtype="step",
                    color=varColors[i],
                    linestyle=["-", "--"],
                    yerr=False,
                    linewidth=linewidth,
                    binwnorm=binwnorm,
                    ax=ax1,
                    flow="none",
                )
                continue

            if hup is not None:
                hep.histplot(
                    hup[i],
                    histtype="step",
                    color=varColors[i],
                    linestyle="-",
                    yerr=False,
                    linewidth=linewidth,
                    label=l,
                    binwnorm=binwnorm,
                    ax=ax1,
                    flow="none",
                )
            if two_sided:
                hep.histplot(
                    hdown[i],
                    histtype="step",
                    color=varColors[i],
                    linestyle="--",
                    yerr=False,
                    linewidth=linewidth,
                    binwnorm=binwnorm,
                    ax=ax1,
                    flow="none",
                )

    if data:
        if dataCovariance:
            nom_data = h_data.values() / binwidth
            std_data = np.sqrt(h_data.variances()) / binwidth
            hatchstyle_data = None
            linecolor_data = "red"  # "black"
            facecolor_data = "red"  # silver"
            facecolor_alpha_data = 0.5

            ax1.fill_between(
                edges,
                np.append((nom_data + std_data), ((nom_data + std_data))[-1]),
                np.append((nom_data - std_data), ((nom_data - std_data))[-1]),
                step="post",
                facecolor=facecolor_data,
                alpha=facecolor_alpha_data,
                hatch=hatchstyle_data,
                edgecolor="k",
                zorder=zorder_data,
                linewidth=0.0,
            )

            hep.histplot(
                h_data,
                histtype="step",
                color=linecolor_data,
                binwnorm=binwnorm,
                yerr=False,
                ax=ax1,
                linestyle="--",
                zorder=zorder_data,
                flow="none",
            )

            extra_handles_upper.append(
                plot_tools.LineBandPolygon(
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    closed=True,
                    facecolor=facecolor_data,
                    edgecolor=linecolor_data,
                    linestyle="--",
                    alpha=facecolor_alpha_data,
                )
            )
            extra_labels_upper.append(args.dataName)
        else:
            hep.histplot(
                h_data,
                yerr=True if counts else h_data.variances() ** 0.5,
                histtype=histtype_data,
                color="black",
                label=args.dataName,
                binwnorm=binwnorm,
                ax=ax1,
                alpha=1.0,
                zorder=zorder_data,
                flow="none",
            )

            if h_data_stat is not None:
                var_stat = h_data_stat.values() ** 2
                h_data_stat = h_data.copy()
                h_data_stat.variances()[...] = var_stat

                hep.histplot(
                    h_data_stat,
                    yerr=True if counts else h_data_stat.variances() ** 0.5,
                    histtype=histtype_data,
                    color="black",
                    binwnorm=binwnorm,
                    capsize=2,
                    ax=ax1,
                    alpha=1.0,
                    zorder=zorder_data,
                    flow="none",
                )

    if (args.unfoldedXsec or len(h_stack) == 0) and not args.noPrefit:
        hep.histplot(
            h_inclusive,
            yerr=False,
            histtype="step",
            color="black",
            binwnorm=binwnorm,
            ax=ax1,
            alpha=1.0,
            zorder=zorder_pred,
            label=pred_label if not args.upperPanelUncertaintyBand else None,
            flow="none",
        )

        if args.upperPanelUncertaintyBand:
            nom = h_inclusive.values() / binwidth
            std = np.sqrt(h_inclusive.variances()) / binwidth

            ax1.fill_between(
                edges,
                np.append((nom + std), ((nom + std))[-1]),
                np.append((nom - std), ((nom - std))[-1]),
                step="post",
                facecolor=facecolor_pred,
                alpha=facecolor_alpha_pred,
                zorder=zorder_pred,
                hatch=hatchstyle_pred,
                edgecolor="k",
                linewidth=0.0,
            )

            extra_handles_upper.append(
                plot_tools.LineBandPolygon(
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    closed=True,
                    facecolor=facecolor_pred,
                    edgecolor="black",
                    alpha=facecolor_alpha_pred,
                )
            )
            extra_labels_upper.append(pred_label)

    if args.ylim is None and binwnorm is None:
        max_y = np.max(h_inclusive.values() + h_inclusive.variances() ** 0.5)
        min_y = np.min(h_inclusive.values() - h_inclusive.variances() ** 0.5)

        if h_data is not None:
            max_y = max(max_y, np.max(h_data.values() + h_data.variances() ** 0.5))
            min_y = min(min_y, np.min(h_data.values() - h_data.variances() ** 0.5))

        range_y = max_y - min_y

        ax1.set_ylim(min_y - range_y * 0.1, max_y + range_y * 0.1)

    if args.ylim is not None and args.ylim[0] > args.ylim[1]:
        # configuration to set minimum to args.ylim[0] (e.g. 0 when we have event yields) and maximum automatically
        max_y = np.max(h_inclusive.values() + h_inclusive.variances() ** 0.5)
        if h_data is not None:
            max_y = max(max_y, np.max(h_data.values() + h_data.variances() ** 0.5))
        min_y = args.ylim[0]
        range_y = max_y - min_y
        ax1.set_ylim(min_y, max_y + range_y * 0.1)

    if len(axes_names) > 1 and args.binSeparationLines is not None:
        # plot dashed vertical lines to sepate makro bins

        s_range = lambda x, n=1: (
            int(x) if round(x, n) == float(int(round(x, n))) else round(x, n)
        )
        max_y = np.max(h_inclusive.values()[...])
        min_y = ax1.get_ylim()[0]

        range_y = max_y - min_y

        for i in range(1, axes[0].size + 1):
            if len(args.binSeparationLines) > 0 and not any(
                np.isclose(x, axes[0].edges[i]) for x in args.binSeparationLines
            ):
                continue

            x = axes[-1].size * i
            x_lo = axes[-1].size * (i - 1)

            if i < axes[0].size + 1:
                # don't plot last line since it's the axis line already
                ax1.plot([x, x], [min_y, max_y], linestyle="--", color="black")

            if len(args.binSeparationLines) == 0 or any(
                np.isclose(x, axes[0].edges[i - 1]) for x in args.binSeparationLines
            ):
                y = min_y + range_y * (
                    0.15 if np.min(h_inclusive.values()[x_lo:x]) > max_y * 0.3 else 0.8
                )
                lo = s_range(axes[0].edges[i - 1])
                hi = s_range(axes[0].edges[i])
                plot_tools.wrap_text(
                    [axes_names[0], f"${lo}-{hi}$"],
                    ax1,
                    x_lo,
                    y,
                    x,
                    text_size="small",
                    transform=ax1.transData,
                )

    if ratio or diff:
        extra_handles = []
        extra_labels = []
        if is_normalized:
            cutoff = 0.5 * np.stack((h_data.values(), h_inclusive.values())).min()
        else:
            cutoff = 0.01

        if diff:
            r_data = lambda x, y: hh.addHists(x, y, scale2=-1)
            r_pred = lambda x, y: hh.addHists(x, y, scale2=-1)
            r_stat = lambda x, y: hh.addHists(x, y, scale2=-1)
        else:
            r_data = lambda x, y: hh.divideHists(
                x,
                y,
                cutoff=1e-8,
                rel_unc=True,
                flow=False,
                by_ax_name=False,
            )
            r_pred = lambda x, y: hh.divideHists(x, y, cutoff=cutoff, rel_unc=True)
            r_stat = lambda x, y: hh.divideHists(x, y, cutoff=cutoff, rel_unc=True)

        h_den = h_data if args.ratioToData else h_inclusive

        if not dataCovariance:
            ax2.axhline(0 if diff else 1, linestyle="--", color="black")

        # prediction
        hep.histplot(
            r_pred(h_inclusive, h_den),
            histtype="step",
            color="black",
            yerr=False,
            ax=ax2,
            zorder=zorder_pred,
            flow="none",
        )

        if data:
            h2 = r_data(h_data, h_den)
            if dataCovariance:
                den_data = 1 if diff else h_den.values() / binwidth

                hep.histplot(
                    r_pred(h_data, h_den),
                    histtype="step",
                    color=linecolor_data,
                    yerr=False,
                    ax=ax2,
                    linestyle="--",
                    zorder=zorder_data,
                    flow="none",
                )

                ax2.fill_between(
                    edges,
                    np.append(
                        (nom_data + std_data) / den_data,
                        ((nom_data + std_data) / den_data)[-1],
                    ),
                    np.append(
                        (nom_data - std_data) / den_data,
                        ((nom_data - std_data) / den_data)[-1],
                    ),
                    step="post",
                    facecolor=facecolor_data,
                    alpha=facecolor_alpha_data,
                    zorder=zorder_data,
                    hatch=hatchstyle_data,
                    edgecolor="k",
                    linewidth=0.0,
                )

            else:
                hep.histplot(
                    h2,
                    histtype="errorbar",
                    color="black",
                    yerr=True if counts else h2.variances() ** 0.5,
                    linewidth=2,
                    ax=ax2,
                    zorder=zorder_data,
                    flow="none",
                )
                if h_data_stat is not None:
                    h2_stat = r_stat(h_data_stat, h_den)
                    hep.histplot(
                        h2_stat,
                        histtype="errorbar",
                        color="black",
                        yerr=True if counts else h2_stat.variances() ** 0.5,
                        linewidth=2,
                        capsize=2,
                        ax=ax2,
                        zorder=zorder_data,
                        flow="none",
                    )

        if not args.noUncertainty:
            nom = h_inclusive.values() / binwidth
            std = np.sqrt(h_inclusive.variances()) / binwidth
            den = 1 if diff else h_den.values() / binwidth

            default_unc_label = (
                "Normalized model unc." if is_normalized else f"{args.predName} unc."
            )
            label_unc = default_unc_label if not args.unfoldedXsec else "Prefit unc."
            if args.uncertaintyLabel:
                label_unc = args.uncertaintyLabel

            ax2.fill_between(
                edges,
                np.append((nom + std) / den, ((nom + std) / den)[-1]),
                np.append((nom - std) / den, ((nom - std) / den)[-1]),
                step="post",
                facecolor=facecolor_pred,
                alpha=facecolor_alpha_pred,
                zorder=zorder_pred,
                hatch=hatchstyle_pred,
                edgecolor="k",
                linewidth=0.0,
                label=label_unc if not args.upperPanelUncertaintyBand else None,
            )

        if (
            args.showVariations in ["lower", "both"]
            and hup is not None
            and any(h is not None for h in hup)
        ):
            linewidth = 2
            scaleVariation = [
                args.scaleVariation[i] if i < len(args.scaleVariation) else 1
                for i in range(len(varLabels))
            ]
            varOneSided = [
                args.varOneSided[i] if i < len(args.varOneSided) else 0
                for i in range(len(varLabels))
            ]

            step_offset = 0
            for i, (hu, hd) in enumerate(zip(hup, hdown)):

                if scaleVariation[i] != 1:
                    hdiff = hh.addHists(hu, h_den, scale2=-1)
                    hdiff = hh.scaleHist(hdiff, scaleVariation[i])
                    hu = hh.addHists(hdiff, h_den)

                    if not varOneSided[i]:
                        hdiff = hh.addHists(hd, h_den, scale2=-1)
                        hdiff = hh.scaleHist(hdiff, scaleVariation[i])
                        hd = hh.addHists(hdiff, h_den)

                if diff:
                    op = lambda h, hI=h_den: hh.addHists(h, hI, scale2=-1)
                else:
                    op = lambda h, hI=h_den: hh.divideHists(
                        h, hI, cutoff=cutoff, rel_unc=True
                    )

                if varOneSided[i]:
                    hvars = op(hu)
                    linestyles = "-"
                else:
                    hvars = [
                        op(hu),
                        op(hd),
                    ]
                    linestyles = ["-", "--"]

                if (
                    varMarkers is not None
                    and len(varMarkers) > i
                    and varMarkers[i] not in ["none", None]
                ):
                    # plot errorbar if varMarkers are provided
                    hvar = hvars[0].copy()
                    hvar.values()[...] = (hvars[0].values() + hvars[1].values()) / 2.0

                    # plot markers slightly shifted
                    step_size = 2 / (len(varMarkers) + 2)
                    step = -1 + (i + 1.5) * step_size
                    # make sure top make space to plot the data in the center
                    if step == 0:
                        step_offset = step_size
                    step += step_offset

                    x = hvar.axes.centers[0]
                    x_diff = np.diff(hvar.axes.edges[0])
                    x = x + step * x_diff / 2

                    ax2.errorbar(
                        x,
                        hvar.values(),
                        color=varColors[i],
                        marker=varMarkers[i],
                        linestyle="none",
                        yerr=[
                            abs(hvars[1].values() - hvar.values()),
                            abs(hvar.values() - hvars[0].values()),
                        ],
                        label=varLabels[i] if args.showVariations != "both" else None,
                        zorder=0,
                    )
                    continue

                fill_alpha = (
                    args.fillVariationsAlphas[i]
                    if i < len(args.fillVariationsAlphas)
                    else 0.0
                )

                if fill_alpha > 0 and not varOneSided[i]:
                    edges = hvars[0].axes[0].edges
                    y_up = hvars[0].values()
                    y_dn = hvars[1].values()
                    ax2.fill_between(
                        edges,
                        np.append(y_up, y_up[-1]),
                        np.append(y_dn, y_dn[-1]),
                        step="post",
                        facecolor=varColors[i],
                        alpha=fill_alpha,
                        edgecolor=varColors[i],
                        linewidth=linewidth,
                        label=None,
                        zorder=0,
                    )
                    hep.histplot(
                        hvars,
                        histtype="step",
                        color=varColors[i],
                        linestyle=linestyles,
                        yerr=False,
                        linewidth=linewidth,
                        ax=ax2,
                        flow="none",
                    )
                    extra_handles.append(
                        Polygon(
                            [[0, 0], [1, 0], [1, 1], [0, 1]],
                            closed=True,
                            facecolor=varColors[i],
                            alpha=fill_alpha,
                            edgecolor=varColors[i],
                            linewidth=linewidth,
                            linestyle="-",
                        )
                    )
                    extra_labels.append(varLabels[i])
                    continue

                hep.histplot(
                    hvars,
                    histtype="step",
                    color=varColors[i],
                    linestyle=linestyles,
                    yerr=False,
                    linewidth=linewidth,
                    label=(
                        varLabels[i]
                        if varOneSided[i] and args.showVariations != "both"
                        else None
                    ),
                    ax=ax2,
                    flow="none",
                )
                if not varOneSided[i] and args.showVariations != "both":
                    extra_handles.append(
                        Line2D([0], [0], color=varColors[i], linewidth=linewidth)
                    )
                    extra_labels.append(varLabels[i])

    scale = max(1, np.divide(*ax1.get_figure().get_size_inches()) * 0.3)

    text_pieces = []
    if not args.unfoldedXsec:
        if is_normalized:
            text_pieces.append(fittype.capitalize() + " (normalized)")
        else:
            text_pieces.append(fittype.capitalize())

    if selection is not None:
        text_pieces.extend(selection)

    if chi2[0] is not None and data and not args.noExtraText:
        p_val = int(np.round(scipy.stats.chi2.sf(chi2[0], chi2[1]) * 100))
        if saturated_chi2:
            chi2_name = r"$\mathit{\chi}_{\mathrm{sat.}}^2/\mathit{ndf}$"
        else:
            chi2_name = r"$\mathit{\chi}^2/\mathit{ndf}$"

        chi2_text = [
            rf"{chi2_name} = ${np.round(chi2[0],1)}/{chi2[1]}$",
            rf"$(\mathit{{p}}={p_val}\%)$",
        ]

        if args.extraTextLoc is None or len(args.extraTextLoc) <= 2:
            text_pieces.extend(chi2_text)
        else:
            plot_tools.wrap_text(
                chi2_text,
                ax1,
                *args.extraTextLoc[2:],
                text_size=args.legSize,
                ha="left",
                va="top",
            )

    if ratio or diff:
        plot_tools.fix_axes(
            ax1, ax2, fig, yscale=args.yscale, noSci=args.noSciy, logy=args.logy
        )
    else:
        plot_tools.fix_axes(ax1, yscale=args.yscale, logy=args.logy)

    plot_tools.add_decor(
        ax1,
        args.title,
        args.subtitle,
        data=data or "Nonprompt" in labels,
        lumi=lumi,  # if args.dataName == "Data" and not args.noData else None,
        loc=args.titlePos,
        text_size=args.legSize,
        no_energy=args.noEnergy,
    )

    if len(h_stack) < 10:
        plot_tools.addLegend(
            ax1,
            ncols=args.legCols,
            loc=args.legPos,
            text_size=args.legSize,
            extra_handles=extra_handles_upper,
            extra_labels=extra_labels_upper,
            custom_handlers=(
                ["bandfilled", "lineband"]
                if any(alpha > 0 for alpha in args.fillVariationsAlphas)
                else ["lineband"]
            ),
            extra_text=text_pieces if not args.noExtraText else None,
            extra_text_loc=None if args.extraTextLoc is None else args.extraTextLoc[:2],
            padding_loc=args.legPadding,
        )

    if ratio or diff:
        plot_tools.addLegend(
            ax2,
            ncols=args.lowerLegCols,
            loc=args.lowerLegPos,
            text_size=args.legSize,
            extra_handles=extra_handles,
            extra_labels=extra_labels,
            custom_handlers=(
                ["stacked", "bandfilled"]
                if any(alpha > 0 for alpha in args.fillVariationsAlphas)
                else ["stacked"]
            ),
            padding_loc=args.lowerLegPadding,
        )

    to_join = [fittype, args.postfix, *axes_names, suffix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if is_normalized:
        outfile += "_normalized"
    if args.subtitle == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)

    analysis_meta_info = None
    if meta is not None:
        if "meta_info_input" in meta:
            analysis_meta_info = {
                "RabbitOutput": meta["meta_info"],
                "AnalysisOutput": meta["meta_info_input"]["meta_info"],
            }
        else:
            analysis_meta_info = {"AnalysisOutput": meta["meta_info"]}

    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={
            "Stacked processes": yield_tables["stacked"],
            "Unstacked processes": yield_tables["unstacked"],
            **analysis_meta_info,
        },
        args=args,
    )


def make_plots(
    result,
    outdir,
    config,
    procs=None,
    labels=None,
    colors=None,
    args=None,
    channel="",
    lumi=1,
    fittype="postit",
    varResults=None,
    varFilesFitTypes=None,
    varMarkers=None,
    varNames=None,
    varLabels=None,
    varColors=None,
    binwnorm=None,
    *opts,
    **kwopts,
):

    hist_data_stat = None

    if args.unfoldedXsec:
        hist_data = result[f"hist_{fittype}_inclusive"].get()
        name_impacts = f"hist_global_impacts_grouped_{fittype}_inclusive"
        if name_impacts in result.keys():
            hist_data_stat = result[name_impacts].get()[{"impacts": "stat"}]
        hist_inclusive = result[f"hist_prefit_inclusive"].get()
        hist_stack = []
    else:
        if f"hist_{args.dataHist}" in result.keys():
            hist_data = result[f"hist_{args.dataHist}"].get()
        else:
            hist_data = None

        hist_inclusive = result[f"hist_{fittype}_inclusive"].get()
        if f"hist_{fittype}" in result.keys():
            hist_stack = result[f"hist_{fittype}"].get()
            hist_stack = [hist_stack[{"processes": p}] for p in procs]
        else:
            hist_stack = []

    if args.dataCovariance:
        hist_data_cov = result[f"cov_data_obs"].get()

        # from sklearn.decomposition import FactorAnalysis
        # fa = FactorAnalysis(n_components=1)
        # fa.fit(hist_data_cov)
        # D_elements = fa.noise_variance_
        # D = np.diag(D_elements)

    axes = [a for a in hist_inclusive.axes]

    if args.processGrouping is not None and len(hist_stack):
        hist_stack, labels, colors, procs = config.process_grouping(
            args.processGrouping, hist_stack, procs
        )

    labels = [
        l if p not in args.suppressProcsLabel else None for l, p in zip(labels, procs)
    ]

    if varNames is not None or len(args.varGroups):
        # take the first variations from the varFiles, empty if no varFiles are provided
        if varNames is not None and len(varFilesFitTypes) == 1:
            varFilesFitTypes = varFilesFitTypes * len(varResults)

        hists_down = []
        hists_up = []
        if varNames is not None:
            for r, t in zip(varResults, varFilesFitTypes):
                h = r[f"hist_{t}_inclusive"].get()

                hist_up = h.copy()
                hist_up.values()[...] = (
                    hist_up.values()[...] + hist_up.variances()[...] ** 0.5
                )
                hist_down = h.copy()
                hist_down.values()[...] = (
                    hist_down.values()[...] - hist_down.variances()[...] ** 0.5
                )

                hists_down.append(hist_down)
                hists_up.append(hist_up)

        # take the next variations from the nominal input file
        if varNames is not None and len(varNames) > len(varResults):
            # variations from the nominal input file
            hist_var = result[
                f"hist_{fittype}_inclusive_variations{'_correlated' if args.correlatedVariations else ''}"
            ].get()

            hists_down.extend(
                [
                    hist_var[{"downUpVar": 0, "vars": n}].project(
                        *[a.name for a in axes]
                    )
                    for n in varNames[len(varResults) :]
                ]
            )
            hists_up.extend(
                [
                    hist_var[{"downUpVar": 1, "vars": n}].project(
                        *[a.name for a in axes]
                    )
                    for n in varNames[len(varResults) :]
                ]
            )

        if len(args.varGroups):
            print(result.keys())
            key = f"hist_{fittype}_inclusive_global_impacts_grouped"
            if key not in result.keys():
                raise ValueError(
                    f"Grouped histogram impacts '{key}' not found. Make sure the fit was produced with --computeHistImpacts."
                )
            hist_grouped = result[key].get()
            available_groups = np.array(hist_grouped.axes["impacts"], dtype=str)
            missing_groups = [g for g in args.varGroups if g not in available_groups]
            if missing_groups:
                raise ValueError(
                    f"Requested grouped variations {missing_groups} not found. Available groups include {available_groups.tolist()}"
                )

            for group in args.varGroups:
                impact = hist_grouped[{"impacts": group}].project(
                    *[a.name for a in axes]
                )
                hist_up = hist_inclusive.copy()
                hist_down = hist_inclusive.copy()
                hist_up.values()[...] = hist_inclusive.values() + impact.values()
                hist_down.values()[...] = hist_inclusive.values() - impact.values()
                hists_up.append(hist_up)
                hists_down.append(hist_down)
    else:
        hists_down = None
        hists_up = None

    # make plots in slices (e.g. for charge plus an minus separately)
    selection_axes = [a for a in axes if a.name in args.selectionAxes]
    if len(selection_axes) > 0:
        selection_bins = [
            np.arange(a.size) for a in axes if a.name in args.selectionAxes
        ]
        other_axes = [a for a in axes if a not in selection_axes]

        ts = getattr(config, "translate_selection", {})

        for bins in itertools.product(*selection_bins):
            idxs = {a.name: i for a, i in zip(selection_axes, bins)}
            # next two dictionaries are built to print the bin values in the plot
            # if the axis name is not in the configuration dictionary, just use the bin index
            # (might print the two bin edges, but usually the bin index is enough)
            idxs_centers = {
                a.name: (
                    a.centers[i]
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    and a.name in ts
                    else a.edges[i] if a.name in ts else i
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

            h_inclusive = hist_inclusive[idxs]
            h_stack = [h[idxs] for h in hist_stack]

            if hist_data is not None:
                h_data = hist_data[idxs]
            else:
                h_data = None

            if hist_data_stat is not None:
                h_data_stat = hist_data_stat[idxs]
            else:
                h_data_stat = None

            if hists_up is not None:
                hup = [h[idxs] if h is not None else None for h in hists_up]
            else:
                hup = None

            if hists_down is not None:
                hdown = [h[idxs] if h is not None else None for h in hists_down]
            else:
                hdown = None

            selection = []
            for a in selection_axes:
                n = a.name
                if n in ts:
                    sel = ts.get(n, lambda x: f"{n}={x}")
                else:
                    sel = lambda x: f"{n} bin = {int(x)}"

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
                h_data,
                h_inclusive,
                h_stack,
                other_axes,
                outdir,
                config,
                labels=labels,
                colors=colors,
                args=args,
                suffix=suffix,
                hup=hup,
                hdown=hdown,
                h_data_stat=h_data_stat,
                selection=selection,
                lumi=lumi,
                fittype=fittype,
                varNames=varNames,
                varLabels=varLabels,
                varColors=varColors,
                varMarkers=varMarkers,
                binwnorm=binwnorm,
                *opts,
                **kwopts,
            )
    else:
        make_plot(
            hist_data,
            hist_inclusive,
            hist_stack,
            axes,
            outdir,
            config,
            labels=labels,
            colors=colors,
            args=args,
            suffix=channel,
            hup=hists_up,
            hdown=hists_down,
            h_data_stat=hist_data_stat,
            lumi=lumi,
            fittype=fittype,
            varNames=varNames,
            varLabels=varLabels,
            varColors=varColors,
            varMarkers=varMarkers,
            binwnorm=binwnorm,
            *opts,
            **kwopts,
        )


def get_chi2(result, no_chi2=True, fittype="postfit", chi2type="automatic"):
    if no_chi2:
        return None, None, False

    if fittype == "prefit":
        if chi2type in [
            "saturated",
        ]:
            raise RuntimeError("No saturated test statistic in prefit")

        chi2_key = "chi2_prefit"
        ndf_key = "ndf_prefit"
        saturated = False
    else:
        chi2_key = "chi2"
        ndf_key = "ndf"

        if chi2type in ["automatic", "saturated"]:
            if (
                result.get("postfit_profile", False)
                and "nllvalreduced" in result.keys()
            ):
                # use saturated likelihood test from actual fit
                chi2 = 2.0 * result["nllvalreduced"]
                ndf = result["ndfsat"]
                return chi2, ndf, True
            elif f"{chi2_key}_saturated" in result.keys():
                # use saturated likelihood test from mapping
                chi2_key += "_saturated"
                ndf_key += "_saturated"
                saturated = True
            elif chi2type == "automatic":
                saturated = False
            else:
                raise ValueError("No saturated test statistic in postfit results found")
        else:
            saturated = False

    return result.get(chi2_key, None), result.get(ndf_key, None), saturated


def main():
    args = make_parser().parse_args()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    varFiles = args.varFiles
    varNames = args.varNames
    varGroups = args.varGroups
    variation_names = []
    if varNames is not None:
        variation_names.extend(varNames)
    variation_names.extend(varGroups)

    varLabels = args.varLabels
    varColors = args.varColors
    if variation_names:
        if varLabels is None:
            syst_labels = getattr(config, "systematics_labels", {})
            varLabels = [syst_labels.get(x, x) for x in variation_names]
        elif len(varLabels) != len(variation_names):
            raise ValueError(
                f"Must specify the same number of args for variation names and --varLabels. Found variations={len(variation_names)} and varLabels={len(varLabels)}"
            )
        if varColors is None:
            varColors = [
                colormaps["tab10" if len(variation_names) < 10 else "tab20"](i)
                for i in range(len(variation_names))
            ]

    if len(args.fillVariationsAlphas) == 1 and len(variation_names) > 1:
        args.fillVariationsAlphas = args.fillVariationsAlphas * len(variation_names)
    elif len(args.fillVariationsAlphas) not in [0, len(variation_names)]:
        raise ValueError(
            f"Must specify either zero, one, or exactly one alpha per variation with --fillVariationsAlphas. Found {len(args.fillVariationsAlphas)} alphas for {len(variation_names)} variations."
        )

    fittype = "prefit" if args.prefit else "postfit"

    # load .hdf5 file
    fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, args.result, meta=True)

    varFitresults = [
        rabbit.io_tools.get_fitresult(f, args.result, meta=False) for f in varFiles
    ]

    plt.rcParams["font.size"] = plt.rcParams["font.size"] * args.scaleTextSize

    channel_info = meta["meta_info_input"]["channel_info"]

    procs = meta["procs"].astype(str)[::-1]
    if args.filterProcs is not None:
        procs = [p for p in procs if p in args.filterProcs]

    if hasattr(config, "get_labels_colors_procs_sorted"):
        labels, colors, procs = config.get_labels_colors_procs_sorted(procs)
    else:
        labels = procs[:]
        cmap = plt.get_cmap("tab10")
        proc_colors = getattr(config, "process_colors", {})
        colors = [proc_colors.get(p, cmap(i % cmap.N)) for i, p in enumerate(procs)]

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    opts = dict(
        args=args,
        procs=procs,
        labels=labels,
        colors=colors,
        meta=meta,
        fittype=fittype,
        varNames=varNames,
        varLabels=varLabels,
        varColors=varColors,
        varMarkers=args.varMarkers,
        dataCovariance=args.dataCovariance,
    )

    results = fitresult.get("mappings", fitresult.get("physics_models"))
    for margs in args.mapping:
        if margs == []:
            instance_keys = results.keys()
        else:
            mapping_key = " ".join(margs)
            instance_keys = [k for k in results.keys() if k.startswith(mapping_key)]
            if len(instance_keys) == 0:
                raise ValueError(
                    f"No mapping found under {mapping_key}. Available mappings: {results.keys()}."
                )

        for instance_key in instance_keys:

            is_normalized = any(
                instance_key.startswith(x) for x in ["Normalize", "Normratio"]
            )

            instance = results[instance_key]

            chi2, ndf, saturated_chi2 = get_chi2(
                (
                    fitresult
                    if fittype == "postfit"
                    and (
                        instance_key
                        in [
                            "BaseMapping",
                        ]
                        and args.chisq != "linear"
                    )
                    else instance
                ),
                args.chisq in [" ", "none", None],
                fittype,
                args.chisq,
            )

            for channel, result in instance["channels"].items():
                if args.channels is not None and channel not in args.channels:
                    continue
                logger.info(f"Make plot for {instance_key} in channel {channel}")

                if instance_key == "CompositeMapping":
                    info = channel_info.get(" ".join(channel.split(" ")[-1:]), {})
                else:
                    info = channel_info.get(channel, {})

                suffix = f"{channel}_{instance_key}"
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

                counts = not args.unfoldedXsec  # if histograms represent counts or not
                binwnorm = (
                    1.0
                    if any(
                        instance_key.startswith(x)
                        for x in [
                            "BaseMapping",
                            "Project",
                            "Select",
                            "Norm",
                            "CompositeMapping",
                        ]
                    )
                    and not args.noBinWidthNorm
                    else None
                )

                opts["counts"] = counts

                varResults = [
                    r.get("mappings", r.get("physics_models"))[
                        instance_key.replace("_masked", "")
                    ]["channels"][channel.replace("_masked", "")]
                    for r in varFitresults
                ]

                if args.lumi is None:
                    # try to automatically find lumi
                    lumi = set(
                        [c["lumi"] for c in channel_info.values() if "lumi" in c.keys()]
                    )
                    lumi = [l for l in lumi]
                    if len(lumi) == 1:
                        lumi = lumi[0]
                    elif len(lumi) == 0:
                        lumi = None
                    else:
                        lumi = list(lumi)
                else:
                    lumi = args.lumi

                make_plots(
                    result,
                    outdir,
                    config,
                    channel=suffix,
                    chi2=[chi2, ndf],
                    saturated_chi2=saturated_chi2,
                    lumi=lumi,
                    is_normalized=is_normalized,
                    binwnorm=binwnorm,
                    varResults=varResults,
                    varFilesFitTypes=args.varFilesFitTypes,
                    **opts,
                )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
