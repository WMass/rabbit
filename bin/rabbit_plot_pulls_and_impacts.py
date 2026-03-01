#!/usr/bin/env python3

import argparse
import itertools
import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rabbit import io_tools

from wums import logging, output_tools, plot_tools  # isort: skip

logger = None


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        if ext == ".html":
            fig.write_html(output, include_mathjax=False)
        else:
            fig.write_image(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    output_tools.write_logfile(
        *output,
        args=args,
        meta_info=meta_info,
    )


def get_marker(filled=True, color="#377eb8", opacity=1.0):
    if filled:
        marker = {
            "marker": {
                "color": color,  # Fill color for the filled bars
                "opacity": opacity,  # Opacity for the filled bars (adjust as needed)
            }
        }
    else:
        marker = {
            "marker": {
                "color": "rgba(0, 0, 0, 0)",  # Transparent fill color
                "opacity": opacity,
                "line": {"color": color, "width": 2},  # Border color  # Border width
            }
        }
    return marker


def plotImpacts(
    df,
    impact_title="Impacts",
    pulls=False,
    oneSidedImpacts=False,
    pullrange=None,
    title=None,
    subtitle=None,
    impacts=True,
    asym=False,
    asym_pulls=False,
    include_ref=False,
    ref_name="ref.",
    name="",
    show_numbers=False,
    show_legend=True,
    legend_pos="bottom",
    group=None,
    diff_pulls=True,
):
    impacts = impacts and bool(np.count_nonzero(df["absimpact"]))
    ncols = pulls + impacts
    fig = make_subplots(rows=1, cols=ncols, horizontal_spacing=0.1, shared_yaxes=True)

    loffset = 40
    if title is not None:
        if subtitle is not None:
            loffset += max(len(subtitle), len(title)) * 7
        else:
            loffset += len(title) * 6

    if legend_pos == "bottom":
        legend = dict(
            orientation="h",
            xanchor="left",
            yanchor="top",
            x=0.0,
            y=0.0,
        )
    elif legend_pos == "right":
        legend = dict(
            orientation="v",
            xanchor="left",
            yanchor="top",
            x=1.0,
            y=1.0,
        )
    else:
        raise NotImplementedError("Supported legend positions are ['bottom', 'left']")

    ndisplay = len(df)
    fig.update_layout(
        paper_bgcolor="rgba(100%,100%,100%,100%)",
        plot_bgcolor="rgba(100%,100%,100%,100%)",
        xaxis_title=impact_title if impacts else "Pull",
        margin=dict(l=loffset, r=20, t=50, b=20),
        yaxis=dict(range=[-1, ndisplay]),
        showlegend=show_legend,
        legend=legend,
        legend_itemsizing="constant",
        height=100 * (ndisplay < 100)
        + ndisplay * 20.5
        + show_legend
        * (legend_pos != "right")
        * (impacts + pulls + asym_pulls)
        * (1 + include_ref)
        * 25,
        width=800 if show_legend and legend_pos == "right" else 640,
        font=dict(
            color="black",
        ),
    )

    gridargs = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="Gray",
    )
    tickargs = dict(
        tick0=0.0,
        tickmode="linear",
        tickangle=0,
        side="top",
    )

    text_on_bars = False
    labels = df["label"]
    if impacts and show_numbers:
        if include_ref:
            # append numerical values of impacts on nuisance name; fill up empty room with spaces to align numbers
            frmt = (
                "{:0"
                + str(
                    int(
                        max(0, np.log10(max(df["absimpact"])))
                        if max(df[f"absimpact_ref"]) > 0
                        else 0
                    )
                    + 2
                )
                + ".2f}"
            )
            nval = df["absimpact"].apply(
                lambda x, frmt=frmt: frmt.format(x)
            )  # .astype(str)
            nspace = nval.apply(
                lambda x, n=nval.apply(len).max(): " " * (n - len(x) + 1)
            )
            if include_ref:
                frmt_ref = (
                    "{:0"
                    + str(
                        int(
                            max(0, np.log10(max(df[f"absimpact_ref"])))
                            if max(df[f"absimpact_ref"]) > 0
                            else 0
                        )
                        + 2
                    )
                    + ".2f}"
                )
                nval_ref = df[f"absimpact_ref"].apply(
                    lambda x, frmt=frmt_ref: " (" + frmt.format(x) + ")"
                )
                nspace_ref = nval_ref.apply(
                    lambda x, n=nval_ref.apply(len).max(): " " * (n - len(x))
                )
                nval = nval + nspace_ref + nval_ref
            labels = labels + nspace + nval
        else:
            text_on_bars = True

    name_suffix = f" ({name})" if name else ""
    ref_name_suffix = f" ({ref_name})" if ref_name else ""

    if impacts:

        def make_bar(
            key="impact",
            color="#377eb8",
            name="+1σ impact",
            text_on_bars=False,
            filled=True,
            opacity=1,
        ):
            x = np.where(df[key] < 0, np.nan, df[key]) if oneSidedImpacts else df[key]

            if text_on_bars:
                text = np.where(np.isnan(x), None, [f"{value:.2f}" for value in x])
            else:
                text = None

            return go.Bar(
                orientation="h",
                x=x,
                y=labels,
                text=text,
                textposition="outside",
                **get_marker(filled=filled, color=color, opacity=opacity),
                name=name,
            )

        sign = "+/-" if (oneSidedImpacts and not any(df["impact_down"] > 0)) else "+"
        label = f"{sign}1σ impact{name_suffix}"
        fig.add_trace(
            make_bar(
                key="impact_up",
                name=label,
                text_on_bars=text_on_bars,
                opacity=0.5 if include_ref else 1,
            ),
            row=1,
            col=1,
        )
        if include_ref:
            fig.add_trace(
                make_bar(
                    key="impact_up_ref",
                    name=f"{sign}1σ impact{ref_name_suffix}",
                    filled=False,
                ),
                row=1,
                col=1,
            )

        if (oneSidedImpacts and any(df["impact_down"] > 0)) or not oneSidedImpacts:
            fig.add_trace(
                make_bar(
                    key="impact_down",
                    name=f"-1σ impact{name_suffix}",
                    color="#e41a1c",
                    text_on_bars=text_on_bars,
                    opacity=0.5 if include_ref else 1,
                ),
                row=1,
                col=1,
            )
            if include_ref:
                fig.add_trace(
                    make_bar(
                        key="impact_down_ref",
                        name=f"-1σ impact{ref_name_suffix}",
                        color="#e41a1c",
                        filled=False,
                    ),
                    row=1,
                    col=1,
                )

        impact_range = df["absimpact"].max()
        if include_ref:
            impact_range = max(impact_range, df[f"absimpact_ref"].max())

        tick_spacing = math.pow(10, math.floor(math.log(impact_range, 10)))
        if tick_spacing > impact_range / 2:
            tick_spacing /= 2
        elif tick_spacing * 2 < impact_range:
            tick_spacing *= int(impact_range / (2 * tick_spacing))

        fig.update_layout(barmode="overlay")
        fig.update_layout(
            xaxis=dict(
                range=[
                    -impact_range * 1.2 if not oneSidedImpacts else -impact_range / 20,
                    impact_range * 1.2,
                ],
                dtick=tick_spacing,
                **gridargs,
                **tickargs,
            ),
        )

    if pulls:
        error_x = dict(
            color="black",
            thickness=1.5,
            width=5,
        )
        if asym:
            error_x["array"] = df["constraint_up"]
            error_x["arrayminus"] = df["constraint_down"]
        else:
            error_x["array"] = df["constraint"]

        fig.add_trace(
            go.Scatter(
                x=df["pull"],
                y=labels,
                mode="markers",
                marker=dict(
                    color="black",
                    size=8,
                ),
                error_x=error_x,
                name=f"Pulls ± Constraints{name_suffix}",
                showlegend=include_ref,
            ),
            row=1,
            col=ncols,
        )
        if include_ref:
            if asym:
                base = df["pull_ref"] - df["constraint_down"]
                x = df["constraint_up"] + df["constraint_down"]
            else:
                base = df["pull_ref"] - df["constraint_ref"]
                x = 2 * df["constraint_ref"]

            fig.add_trace(
                go.Bar(
                    base=base,
                    x=x,
                    y=labels,
                    orientation="h",
                    **get_marker(filled=False, color="black"),
                    name=f"Pulls ± Constraints{ref_name_suffix}",
                    showlegend=True,
                ),
                row=1,
                col=ncols,
            )

        if asym_pulls:
            fig.add_trace(
                go.Scatter(
                    x=df["newpull"],
                    y=labels,
                    mode="markers",
                    marker=dict(
                        color="green",
                        symbol="x",
                        size=8,
                        # line=dict(width=1),  # Adjust the thickness of the marker lines
                    ),
                    name="Asym. pulls",
                    showlegend=include_ref,
                ),
                row=1,
                col=ncols,
            )

            if include_ref:
                fig.add_trace(
                    go.Scatter(
                        x=df["newpull_ref"],
                        y=labels,
                        mode="markers",
                        marker=dict(
                            color="green",
                            symbol="circle-open",
                            size=8,
                            line=dict(
                                width=1
                            ),  # Adjust the thickness of the marker lines
                        ),
                        name=f"Asym. pulls{ref_name_suffix}",
                        showlegend=include_ref,
                    ),
                    row=1,
                    col=ncols,
                )
        max_pull = np.max(df["abspull"])
        if pullrange is None:
            # Round up to nearest 0.5, add 1.1 for display
            pullrange = 0.5 * np.ceil(max_pull) + 1.1
        # Keep it a factor of 0.5, but no bigger than 1
        spacing = min(1, np.ceil(pullrange) / 2.0)
        if spacing > 0.5 * pullrange:  # make sure to have at least two ticks
            spacing /= 2.0
        xaxis_title = "Nuisance parameter"
        if diff_pulls:
            xaxis_title += "<br> θ-θ₀"
        else:
            xaxis_title += "<br> θ"
        info = dict(
            xaxis=dict(
                range=[-pullrange, pullrange], dtick=spacing, **gridargs, **tickargs
            ),
            xaxis_title=xaxis_title,
            yaxis=dict(range=[-1, ndisplay]),
            yaxis_visible=not impacts,
        )
        if impacts:
            new_info = {}
            for k in info.keys():
                new_info[k.replace("axis", "axis2")] = info[k]
            info = new_info
        fig.update_layout(barmode="overlay", **info)

    if title is not None:
        fig.add_annotation(
            x=0,
            y=1,
            xshift=-loffset,
            yshift=50,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=title,
            font=dict(size=24, color="black", family="Arial", weight="bold"),
        )
        if subtitle is not None:
            fig.add_annotation(
                x=0,
                y=1,
                xshift=-loffset,
                yshift=25,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"<i>{subtitle}</i>",
                font=dict(
                    size=20,
                    color="black",
                    family="Arial",
                ),
            )

    return fig


def readFitInfoFromFile(
    fitresult,
    poi,
    group=False,
    impact_type=False,
    grouping=None,
    asym=False,
    filters=[],
    stat=0.0,
    normalize=False,
    scale=1,
    diff_pulls=True,
):
    if poi is not None:
        out = io_tools.read_impacts_poi(
            fitresult,
            poi,
            group,
            pulls=not group,
            asym=asym,
            impact_type=impact_type,
            add_total=group and not impact_type == "nonprofiled",
        )
        if group:
            impacts, labels = out
            if normalize:
                idx = np.argwhere(labels == "Total")
                impacts /= impacts[idx].flatten()
        else:
            pulls, pulls_prefit, constraints, constraints_prefit, impacts, labels = out
            if normalize:
                imp, lab = io_tools.read_impacts_poi(
                    fitresult,
                    poi,
                    True,
                    pulls=False,
                    asym=asym,
                    impact_type=impact_type,
                    add_total=True,
                )

                idx = np.argwhere(lab == "Total")
                impacts /= imp[idx].flatten()

        if stat > 0 and "stat" in labels:
            idx = np.argwhere(labels == "stat")
            impacts[idx] = stat
    else:
        labels = io_tools.get_syst_labels(fitresult)
        _, pulls, constraints = io_tools.get_pulls_and_constraints(fitresult, asym=asym)
        _, pulls_prefit, constraints_prefit = io_tools.get_pulls_and_constraints(
            fitresult, prefit=True
        )

    apply_mask = (group and grouping is not None) or filters is not None

    if apply_mask:
        mask = np.ones(len(labels), dtype=bool)

        if group and grouping:
            mask &= np.isin(labels, grouping)  # Check if labels are in the grouping

        if filters:
            mask &= np.array(
                [any(re.search(f, label) for f in filters) for label in labels]
            )  # Apply regex filter

        labels = labels[mask]

    df = pd.DataFrame(np.array(labels, dtype=str), columns=["label"])

    if poi is not None:
        if apply_mask:
            impacts = impacts[mask]

        if scale and not normalize:
            impacts = impacts * scale

        if asym:
            df["impact_down"] = impacts[..., 1]
            df["impact_up"] = impacts[..., 0]
            df["absimpact"] = np.abs(impacts).max(axis=-1)
        else:
            df["impact_down"] = -impacts
            df["impact_up"] = impacts
            df["absimpact"] = np.abs(impacts)

    if not group:
        if apply_mask:
            pulls = pulls[mask]
            constraints = constraints[mask]
            pulls_prefit = pulls_prefit[mask]
            constraints_prefit = constraints_prefit[mask]

        df["pull_prefit"] = pulls_prefit

        if diff_pulls:
            df["pull"] = pulls - pulls_prefit
        else:
            df["pull"] = pulls
        df["abspull"] = np.abs(df["pull"])

        if asym:
            df["constraint_down"] = -constraints[..., 1]
            df["constraint_up"] = constraints[..., 0]
        else:
            df["constraint"] = constraints
            valid = (1 - constraints**2) > 0
            df["newpull"] = 999.0
            df.loc[valid, "newpull"] = df.loc[valid]["pull"] / np.sqrt(
                1 - df.loc[valid]["constraint"] ** 2
            )

    if poi:
        df = df.drop(df.loc[df["label"] == poi].index)

    return df


def readHistImpacts(
    fitresult,
    hist_impacts,
    hist_total,
    group=False,
    grouping=None,
    asym=False,
    filters=[],
    stat=0.0,
    normalize=False,
    scale=1,
):
    labels = np.array(hist_impacts.axes["impacts"])
    impacts = hist_impacts.values()

    if not group:
        labels_pulls, pulls, constraints = io_tools.get_pulls_and_constraints(
            fitresult, asym=asym
        )
        labels_pulls_prefit, pulls_prefit, constraints_prefit = (
            io_tools.get_pulls_and_constraints(fitresult, asym=asym, prefit=True)
        )

        labels, idxs, idxs_pulls = np.intersect1d(
            labels, labels_pulls, assume_unique=True, return_indices=True
        )

        pulls = pulls[idxs_pulls]
        constraints = constraints[idxs_pulls]
        pulls_prefit = pulls_prefit[idxs_pulls]
        constraints_prefit = constraints_prefit[idxs_pulls]

        impacts = impacts[idxs]

    total = np.sqrt(hist_total.variance)
    if group:
        impacts = np.append(impacts, total)
        labels = np.append(labels, "Total")

    # if args.relative:
    #     unit = "rel. unc. in %"
    #     impacts /= hist_total.value
    #     scale=100
    # else:
    #     unit = "bin unc."
    #     scale=1

    if normalize:
        impacts /= total

    if stat > 0 and "stat" in labels:
        idx = np.argwhere(labels == "stat")
        impacts[idx] = stat

    apply_mask = (group and grouping is not None) or filters is not None

    if apply_mask:
        mask = np.ones(len(labels), dtype=bool)

        if group and grouping:
            mask &= np.isin(labels, grouping)  # Check if labels are in the grouping

        if filters:
            mask &= np.array(
                [any(re.search(f, label) for f in filters) for label in labels]
            )  # Apply regex filter

        labels = labels[mask]

    df = pd.DataFrame(np.array(labels, dtype=str), columns=["label"])

    if apply_mask:
        impacts = impacts[mask]

    if scale and not normalize:
        impacts = impacts * scale

    if asym:
        df["impact_down"] = impacts[..., 1]
        df["impact_up"] = impacts[..., 0]
        df["absimpact"] = np.abs(impacts).max(axis=-1)
    else:
        df["impact_down"] = -impacts
        df["impact_up"] = impacts
        df["absimpact"] = np.abs(impacts)

    if not group:
        if apply_mask:
            pulls = pulls[mask]
            constraints = constraints[mask]
            pulls_prefit = pulls_prefit[mask]
            constraints_prefit = constraints_prefit[mask]

        df["pull"] = pulls
        df["pull_prefit"] = pulls_prefit
        df["pull"] = pulls - pulls_prefit
        df["abspull"] = np.abs(df["pull"])

        if asym:
            df["constraint_down"] = -constraints[..., 1]
            df["constraint_up"] = constraints[..., 0]
        else:
            df["constraint"] = constraints
            valid = (1 - constraints**2) > 0
            df["newpull"] = 999.0
            df.loc[valid, "newpull"] = df.loc[valid]["pull"] / np.sqrt(
                1 - df.loc[valid]["constraint"] ** 2
            )

    return df


def parseArgs():
    sort_choices = ["label", "pull", "abspull", "constraint", "absimpact"]
    sort_choices += [
        *[
            f"{c}_diff" for c in sort_choices
        ],  # possibility to sort based on largest difference between input and referencefile
        *[
            f"{c}_ref" for c in sort_choices
        ],  # possibility to sort based on reference file
        *[f"{c}_both" for c in sort_choices],
    ]  # possibility to sort based on the largest/smallest of both input and reference file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output hdf5 file from fit",
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
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        default=None,
        type=str,
        nargs="+",
        help="Plot impacts on observables use '-m <mapping> channel axes' for mapping results.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        type=str,
        help="Plot impacts for given channel",
    )
    parser.add_argument(
        "--refMapping",
        default=None,
        type=str,
        nargs="+",
        help="Plot impacts on observables from mapping results for reference.",
    )
    parser.add_argument(
        "--refChannel",
        default=None,
        type=str,
        help="Plot impacts for given channel",
    )
    parser.add_argument(
        "-r",
        "--referenceFile",
        type=str,
        help="fitresults output hdf5 file from fit for reference",
    )
    parser.add_argument(
        "--refResult",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--refName",
        type=str,
        default="ref.",
        help="Name of reference input for legend",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of input for legend",
    )
    parser.add_argument(
        "-s",
        "--sort",
        default="absimpact",
        type=str,
        help="Sort mode for nuisances",
        choices=sort_choices,
    )
    parser.add_argument(
        "--stat",
        default=0.0,
        type=float,
        help="Overwrite stat. uncertainty with this value",
    )
    parser.add_argument(
        "-d",
        "--sortDescending",
        dest="ascending",
        action="store_false",
        help="Sort mode for nuisances",
    )
    parser.add_argument(
        "--mode",
        choices=["group", "ungrouped", "both"],
        default="both",
        help="Impact mode",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize impacts on poi, leading to relative uncertainties.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Relative uncertainty w.r.t. central value of parameter of interest (only for hisograms)",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--diffPullAsym",
        action="store_true",
        help="Also add the pulls after the diffPullAsym definition",
    )
    parser.add_argument(
        "--oneSidedImpacts", action="store_true", help="Make impacts one-sided"
    )
    parser.add_argument(
        "--filters",
        nargs="*",
        type=str,
        help="Filter regexes to select nuisances by name",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help="Pre-defined grouping in config to select nuisance groups",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
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
    parser.add_argument(
        "--impactTitle",
        default="Impacts",
        type=str,
        help="Title for impacts",
    )
    parser.add_argument("--noImpacts", action="store_true", help="Don't show impacts")
    parser.add_argument(
        "--globalImpacts", action="store_true", help="Print global impacts"
    )
    parser.add_argument(
        "--nonprofiledImpacts", action="store_true", help="Print non-profiled impacts"
    )
    parser.add_argument(
        "--asymImpacts",
        action="store_true",
        help="Print asymmetric impacts from likelihood, otherwise symmetric from hessian",
    )
    parser.add_argument(
        "--showNumbers", action="store_true", help="Show values of impacts"
    )
    parser.add_argument(
        "--poi",
        type=str,
        default=None,
        help="Specify POI to make impacts for, otherwise use all",
    )
    parser.add_argument(
        "--poiType", type=str, default=None, help="POI type to make impacts for"
    )
    parser.add_argument(
        "--pullrange", type=float, default=None, help="POI type to make impacts for"
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./test",
        help="Folder path for output",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Override use of xrdcp and use the mount instead",
    )
    parser.add_argument(
        "--otherExtensions",
        default=[],
        type=str,
        nargs="*",
        help="Additional output file types to write",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=50, help="Number of nuisances to plot"
    )
    parser.add_argument(
        "--noPulls",
        action="store_true",
        help="Don't show pulls (not defined for groups)",
    )
    parser.add_argument(
        "--scaleImpacts",
        type=float,
        default=1.0,
        help="Scale impacts by this number",
    )
    parser.add_argument(
        "--scaleImpactsRef",
        type=float,
        default=-1.0,
        help="Scale impacts of reference by this number (default uses same as nominal)",
    )
    parser.add_argument(
        "--pullsNoDiff",
        action="store_true",
        help="Plot actual nuisance parameter value, by default nuisance parameter difference w.r.t. prefit value",
    )
    return parser.parse_args()


def make_plots(
    df,
    args,
    outdir,
    outfile,
    group=False,
    asym=False,
    pullrange=None,
    meta=None,
    postfix=None,
    impact_title=None,
):

    df = df.fillna(0)

    if args.sort:
        if args.sort.endswith("diff"):
            key = args.sort.replace("_diff", "")
            df[f"{key}_diff"] = abs(df[key] - df[f"{key}_ref"])
        elif args.sort.endswith("both"):
            key = args.sort.replace("_both", "")
            if args.ascending:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].max(axis=1)
            else:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].min(axis=1)

        if args.sort in df.keys():
            if f"{args.sort}_ref" in df.keys():
                df = df.sort_values(
                    by=[args.sort, f"{args.sort}_ref"], ascending=args.ascending
                )
            else:
                df = df.sort_values(by=args.sort, ascending=args.ascending)
        else:
            print(
                f"Trying to sort {args.sort} but not found in dataframe, continue without sorting"
            )

    outfile = os.path.join(outdir, outfile)
    extensions = [outfile.split(".")[-1], *args.otherExtensions]

    include_ref = any(
        x in df.keys()
        for x in ["impact_ref", "absimpact_ref", "pull_ref", "constraint_ref"]
    )

    kwargs = dict(
        pulls=not args.noPulls and not group,
        impact_title=impact_title,
        oneSidedImpacts=args.oneSidedImpacts,
        pullrange=pullrange,
        title=args.title,
        subtitle=args.subtitle,
        impacts=not args.noImpacts,
        asym=asym,
        asym_pulls=args.diffPullAsym,
        include_ref=include_ref,
        ref_name=args.refName,
        name=args.name,
        show_numbers=args.showNumbers,
        show_legend=(not group and not args.noImpacts)
        or include_ref
        or args.name is not None,
        group=group,
        diff_pulls=not args.pullsNoDiff,
    )

    if args.num and args.num < int(df.shape[0]):
        # in case multiple extensions are given including html, don't do the skimming on html but all other formats
        if "html" in extensions and len(extensions) > 1:
            fig = plotImpacts(df, legend_pos="right", **kwargs)
            outfile_html = ".".join([*outfile.split(".")[:-1], "html"])
            writeOutput(fig, outfile_html, [".html"], postfix=postfix)
            extensions = [e for e in extensions if e != "html"]
            outfile = ".".join([*outfile.split(".")[:-1], extensions[0]])

        df = df[-args.num :]

    fig = plotImpacts(df, **kwargs)

    writeOutput(fig, outfile, extensions, postfix=postfix, args=args, meta_info=meta)


def load_dataframe_parms(
    args,
    fitresult,
    poi=None,
    group=False,
    asym=False,
    normalize=False,
    fitresult_ref=None,
    grouping=None,
    translate_label={},
):
    if args.globalImpacts:
        impact_type = "global"
    elif args.nonprofiledImpacts:
        impact_type = "nonprofiled"
    else:
        impact_type = "traditional"

    if not group:
        df = readFitInfoFromFile(
            fitresult,
            poi,
            False,
            asym=asym,
            impact_type=impact_type,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            scale=args.scaleImpacts,
            diff_pulls=not args.pullsNoDiff,
        )
    elif group:
        df = readFitInfoFromFile(
            fitresult,
            poi,
            True,
            asym=asym,
            impact_type=impact_type,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            grouping=grouping,
            scale=args.scaleImpacts,
        )

    if fitresult_ref:
        df_ref = readFitInfoFromFile(
            fitresult_ref,
            poi,
            group,
            asym=asym,
            impact_type=impact_type,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            grouping=grouping,
            scale=(
                args.scaleImpactsRef if args.scaleImpactsRef > 0 else args.scaleImpacts
            ),
            diff_pulls=not args.pullsNoDiff,
        )
        df = df.merge(df_ref, how="outer", on="label", suffixes=("", "_ref"))

    df["label"] = df["label"].apply(lambda l: translate_label.get(l, l))

    if df.empty:
        print("WARNING: Empty dataframe")
        if group and grouping:
            print(
                f"WARNING: This can happen if no group is found that belongs to {grouping}"
            )
            print(
                "WARNING: Try a different mode for --grouping or use '--mode ungrouped' to skip making impacts for groups"
            )
        print("WARNING: Skipping this part")
        return None
    else:
        return df


def load_dataframe_hists(
    fitresult,
    args,
    hist_impacts,
    hist_total,
    ibin=None,
    group=False,
    asym=False,
    normalize=False,
    fitresult_ref=None,
    hist_impacts_ref=None,
    hist_total_ref=None,
    grouping=None,
    translate_label={},
):

    if args.relative:
        scale = args.scaleImpacts * 1.0 / hist_total.value
        if fitresult_ref:
            if args.scaleImpactsRef > 0:
                scale_ref = args.scaleImpactsRef * 1.0 / hist_total_ref.value
            else:
                scale_ref = args.scaleImpacts * 1.0 / hist_total_ref.value
    else:
        scale = args.scaleImpacts
        if fitresult_ref:
            scale_ref = (
                args.scaleImpactsRef if args.scaleImpactsRef > 0 else args.scaleImpacts
            )

    df = readHistImpacts(
        fitresult,
        hist_impacts,
        hist_total,
        group,
        filters=args.filters,
        stat=args.stat / 100.0,
        normalize=normalize,
        grouping=grouping,
        scale=scale,
    )

    if fitresult_ref:
        df_ref = readHistImpacts(
            fitresult_ref,
            hist_impacts_ref,
            hist_total_ref,
            group,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            grouping=grouping,
            scale=scale_ref,
        )
        df = df.merge(df_ref, how="outer", on="label", suffixes=("", "_ref"))

    df["label"] = df["label"].apply(lambda l: translate_label.get(l, l))

    if df.empty:
        print("WARNING: Empty dataframe")
        if group and grouping:
            print(
                f"WARNING: This can happen if no group is found that belongs to {grouping}"
            )
            print(
                "WARNING: Try a different mode for --grouping or use '--mode ungrouped' to skip making impacts for groups"
            )
        print("WARNING: Skipping this part")
        return None
    else:
        return df


def produce_plots_parms(
    args,
    fitresult,
    outdir,
    outfile,
    poi=None,
    group=False,
    asym=False,
    normalize=False,
    fitresult_ref=None,
    pullrange=None,
    meta=None,
    postfix=None,
    impact_title=None,
    grouping=None,
    translate_label={},
):

    df = load_dataframe_parms(
        args,
        fitresult,
        poi=poi,
        group=group,
        asym=asym,
        normalize=normalize,
        fitresult_ref=fitresult_ref,
        grouping=grouping,
        translate_label=translate_label,
    )

    if df is None:
        return

    make_plots(
        df,
        args,
        outdir,
        outfile,
        group=group,
        asym=asym,
        pullrange=pullrange,
        meta=meta,
        postfix=postfix,
        impact_title=impact_title,
    )


def produce_plots_hist(
    args,
    fitresult,
    outdir,
    outfile,
    hist_impacts,
    hist_total,
    ibin=None,
    group=False,
    asym=False,
    normalize=False,
    fitresult_ref=None,
    hist_impacts_ref=None,
    hist_total_ref=None,
    pullrange=None,
    meta=None,
    postfix=None,
    impact_title=None,
    grouping=None,
    translate_label={},
):

    df = load_dataframe_hists(
        fitresult,
        args,
        hist_impacts,
        hist_total,
        ibin=ibin,
        group=group,
        asym=asym,
        normalize=normalize,
        fitresult_ref=fitresult_ref,
        hist_impacts_ref=hist_impacts_ref,
        hist_total_ref=hist_total_ref,
        grouping=grouping,
        translate_label=translate_label,
    )
    if df is None:
        return

    make_plots(
        df,
        args,
        outdir,
        outfile,
        group=group,
        asym=asym,
        pullrange=pullrange,
        meta=meta,
        postfix=postfix,
        impact_title=impact_title,
    )


def main():
    args = parseArgs()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    translate_label = getattr(config, "impact_labels", {})

    fitresult, meta = io_tools.get_fitresult(args.inputFile, args.result, meta=True)
    if any(x is not None for x in [args.referenceFile, args.refResult]):
        ref_file = (
            args.referenceFile if args.referenceFile is not None else args.inputFile
        )
        ref_result = args.refResult if args.refResult is not None else args.result
        fitresult_ref = io_tools.get_fitresult(ref_file, ref_result)
    elif args.refMapping is not None:
        fitresult_ref = fitresult
    else:
        fitresult_ref = None

    meta_out = {
        "rabbit": meta["meta_info"],
    }

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    kwargs = dict(
        pullrange=args.pullrange,
        asym=args.asymImpacts,
        fitresult_ref=fitresult_ref,
        meta=meta_out,
        postfix=args.postfix,
        translate_label=translate_label,
    )

    if args.noImpacts:
        # do one pulls plot, ungrouped
        produce_plots_parms(args, fitresult, outdir, outfile="pulls.html", **kwargs)
    else:
        kwargs.update(dict(normalize=args.normalize, impact_title=args.impactTitle))

        impacts_name = "impacts"
        if args.globalImpacts:
            impacts_name = f"global_{impacts_name}"
        elif args.nonprofiledImpacts:
            impacts_name = f"nonprofiled_{impacts_name}"

        grouping = None
        if args.grouping is not None:
            grouping = getattr(config, "nuisance_grouping", {}).get(args.grouping, None)

        if args.mapping is not None:
            if args.asymImpacts:
                raise NotImplementedError(
                    "Asymetric impacts on observables is not yet implemented"
                )
            if not args.globalImpacts:
                raise NotImplementedError(
                    "Only global impacts on observables is implemented (use --globalImpacts)"
                )

            def get_mapping_key(result, key):

                res = result.get("mappings", fitresult.get("physics_models"))
                if key in res.keys():
                    channels = res[key]["channels"]
                    return channels, key
                else:
                    keys = [k for k in res.keys() if k.startswith(key)]
                    if len(keys) == 0:
                        raise ValueError(
                            f"Mapping {key} not found, available mappings are: {res.keys()}"
                        )

                    channels = res[keys[0]]["channels"]
                    logger.info(
                        f"Found mapping {keys[0]} with channels {[k for k in channels.keys()]}"
                    )

                    return channels, keys[0]

            mapping_key = " ".join(args.mapping)

            channels, mapping_key = get_mapping_key(fitresult, mapping_key)

            if fitresult_ref:
                mapping_key_ref = (
                    " ".join(args.refMapping)
                    if args.refMapping is not None
                    else mapping_key
                )

                channels_ref, mapping_key_ref = get_mapping_key(
                    fitresult_ref, mapping_key_ref
                )

            for channel, hists in channels.items():
                if args.channel and channel not in args.channel:
                    continue

                modes = ["ungrouped", "group"] if args.mode == "both" else [args.mode]
                for mode in modes:
                    group = mode == "group"

                    key = "hist_postfit_inclusive_global_impacts"
                    if group:
                        key += "_grouped"

                    hist_total = hists["hist_postfit_inclusive"].get()

                    hist = hists[key].get()

                    if fitresult_ref:
                        if args.refChannel:
                            channel_ref = args.refChannel
                        elif channel in channels_ref.keys():
                            channel_ref = channel
                        elif len(channels_ref.keys()) == 1:
                            channel_ref = [v for v in channels_ref.keys()][0]
                        else:
                            raise NotImplementedError(
                                f"Could not decide which is the right channel from reference file with channels: {channels_ref.keys()}"
                            )

                        res_ref = fitresult_ref.get(
                            "mappings", fitresult.get("physics_models")
                        )
                        hists_ref = res_ref[mapping_key_ref]["channels"][channel_ref]

                        hist_ref = hists_ref[key].get()
                        hist_total_ref = hists_ref["hist_postfit_inclusive"].get()

                    for idxs in itertools.product(
                        *[np.arange(a.size) for a in hist_total.axes]
                    ):
                        ibin = {a: i for a, i in zip(hist_total.axes.name, idxs)}
                        print(f"Now at {ibin}")

                        ibin_str = "_".join([f"{a}{i}" for a, i in ibin.items()])
                        if group:
                            outfile = f"{impacts_name}_grouped_{ibin_str}.html"
                        else:
                            outfile = f"{impacts_name}_{ibin_str}.html"
                            if not args.noPulls:
                                outfile = f"pulls_and_{outfile}"

                        produce_plots_hist(
                            args,
                            fitresult,
                            outdir,
                            outfile,
                            hist[ibin],
                            hist_total[ibin],
                            ibin,
                            group=group,
                            grouping=grouping,
                            hist_impacts_ref=hist_ref[ibin] if fitresult_ref else None,
                            hist_total_ref=(
                                hist_total_ref[ibin] if fitresult_ref else None
                            ),
                            **kwargs,
                        )
        else:
            pois = [args.poi] if args.poi else io_tools.get_poi_names(meta)
            for poi in pois:
                print(f"Now at {poi}")
                if args.mode in ["both", "ungrouped"]:
                    name = f"{impacts_name}_{poi}.html"
                    if not args.noPulls:
                        name = f"pulls_and_{name}"
                    produce_plots_parms(
                        args, fitresult, outdir, outfile=name, poi=poi, **kwargs
                    )
                if args.mode in ["both", "group"]:
                    produce_plots_parms(
                        args,
                        fitresult,
                        outdir,
                        outfile=f"{impacts_name}_grouped_{poi}.html",
                        poi=poi,
                        group=True,
                        grouping=grouping,
                        **kwargs,
                    )

    output_tools.write_indexfile(outdir)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
