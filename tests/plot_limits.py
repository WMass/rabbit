#!/usr/bin/env python3

import argparse
import os

import mplhep as hep
import numpy as np
from wums import boostHistHelpers as hh
from wums import logging

from rabbit import io_tools

from wums import logging, output_tools, plot_tools  # isort: skip

hep.style.use(hep.style.ROOT)


logger = None


def parseArgs():
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
        "infiles",
        type=str,
        nargs="*",
        help="hdf5 files from rabbit or root file from combinetf",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("./test"),
        help="Base path for output",
    )
    parser.add_argument(
        "--noObs", action="store_true", help="Don't plot observed limit"
    )
    parser.add_argument(
        "--xvals", type=float, nargs="+", default=None, help="x-axis ticks for plotting"
    )
    parser.add_argument(
        "--biases",
        type=float,
        nargs="+",
        default=None,
        help="Add additional line to highlight the bias in each point",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale limits by this value"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "likelihood"],
        help="Different modes of approximations to be used",
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
        "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
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
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Min and max values for y axis (if not specified, range set automatically)",
    )
    args = parser.parse_args()

    return args


def main():
    """
    Plot the covariance matrix of the histogram bins
    """

    args = parseArgs()

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    key_expected = f"{args.mode}_asymptoticLimits_expected"
    key_observed = f"{args.mode}_asymptoticLimits_observed"

    cls_list = set()
    clb_list = set()
    limits = []
    limits_asimov = []
    lumi = None
    xvals = []
    for infile in args.infiles:
        fitresult_asimov, meta = io_tools.get_fitresult(
            infile, result="asimov", meta=True
        )

        bsm_sample, mixing = meta["meta_info_input"]["meta_info"]["args"][
            "addBSMMixing"
        ]
        bsm_name, mass = bsm_sample.split("_")
        xvals.append(int(mass))

        scale = float(mixing) * args.scale

        if key_expected in fitresult_asimov.keys():
            h_limits_asimov = fitresult_asimov[key_expected].get()
            h_limits_asimov = hh.scaleHist(h_limits_asimov, scale)
        else:
            raise RuntimeError(
                f"Key {key_expected} not found, available keyes are {[key for key in fitresult_asimov.keys()]}"
            )

        cls_list = cls_list.union(set([cls for cls in h_limits_asimov.axes["cls"]]))
        clb_list = clb_list.union(set([clb for clb in h_limits_asimov.axes["clb"]]))

        limits_asimov.append(h_limits_asimov)

        fitresult = io_tools.get_fitresult(infile)
        if fitresult != fitresult_asimov and not args.noObs:
            if (
                key_observed in fitresult.keys()
                and "clb" not in fitresult[key_observed].get().axes.name
            ):
                h_limits = fitresult[key_observed].get()
                h_limits = hh.scaleHist(h_limits, scale)
                cls_list = cls_list.union(set([cls for cls in h_limits.axes["cls"]]))
                limits.append(h_limits)
            else:
                limits.append(h_limits_asimov[{"clb": "0.5"}])

        channel_info = meta["meta_info_input"]["channel_info"]
        lumis = set([c.get("lumi", None) for c in channel_info.values()])
        if len(lumis) > 1 or (lumi is not None and tuple(lumis)[0] != lumi):
            raise ValueError("Different channels with different luminosity values")
        lumi = tuple(lumis)[0]

    if args.xvals is not None:
        x = np.array(args.xvals)
    else:
        x = np.array(xvals)

    clb_list = list(clb_list)

    for cls in cls_list:

        yexp = np.array(
            [l[{"cls": cls, "clb": "0.5"}] for l in limits_asimov]
        ).flatten()

        yexp_m2 = np.array(
            [l[{"cls": cls, "clb": "0.025"}] for l in limits_asimov]
        ).flatten()
        yexp_m1 = np.array(
            [l[{"cls": cls, "clb": "0.16"}] for l in limits_asimov]
        ).flatten()
        yexp_p1 = np.array(
            [l[{"cls": cls, "clb": "0.84"}] for l in limits_asimov]
        ).flatten()
        yexp_p2 = np.array(
            [l[{"cls": cls, "clb": "0.975"}] for l in limits_asimov]
        ).flatten()

        ylist = [yexp, yexp_m1, yexp_m2, yexp_p1, yexp_p2]

        if len(limits) > 0:
            yobs = np.array([l[{"cls": cls}] for l in limits]).flatten()
            ylist.append(yobs)

        ylist = np.array(ylist)
        ymin, ymax = np.min(ylist), np.max(ylist)
        yrange = ymax - ymin

        fig, ax = plot_tools.figure(
            None,
            args.xlabel,
            args.ylabel,
            automatic_scale=False,
            xlim=(min(x) - 0.5, max(x) + 0.5),
            ylim=(
                (ymin - yrange * 0.1, ymax + yrange * 0.6)
                if args.ylim is None
                else args.ylim
            ),
        )

        ax.fill_between(x, yexp_m2, yexp_p2, color="#F5BB54", label=r"95% expected")

        ax.fill_between(x, yexp_m1, yexp_p1, color="#607641", label=r"68% expected")

        # Expected (median)
        ax.plot(
            x,
            yexp,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Median expected",
        )

        if args.biases is not None:
            color_bias = "tab:red"
            # add right y-axis
            ax2 = ax.twinx()
            ax2.plot(
                x,
                args.biases,
                linestyle="--",
                color=color_bias,
                linewidth=1.5,
                label="Expected bias",
            )
            ax2.set_ylabel("1 - relative expected bias", color=color_bias)
            ax2.tick_params(axis="y", labelcolor=color_bias)

        if len(limits) > 0:
            # Observed
            ax.plot(x, yobs, color="black", linewidth=1.5, label="Observed")

        plot_tools.add_decor(
            ax,
            args.title,
            args.subtitle,
            data=len(limits) > 0,
            lumi=lumi,  # if args.dataName == "Data" and not args.noData else None,
            loc=args.titlePos,
            text_size=args.legSize,
        )
        plot_tools.addLegend(
            ax,
            text_size=args.legSize,
            ncols=args.legCols,
            loc=args.legPos,
            title=f"{round((1-float(cls)) * 100)}% CL upper limits",
        )

        to_join = ["limits", args.postfix, f"CLs{str(cls).replace('.','p')}"]
        outfile = "_".join(filter(lambda x: x, to_join))
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
                # "Stacked processes": yield_tables["stacked"],
                # "Unstacked processes": yield_tables["unstacked"],
                **analysis_meta_info,
            },
            args=args,
        )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
