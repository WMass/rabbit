#!/usr/bin/env python3

import argparse
import itertools
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import seaborn as sns

import combinetf2.io_tools

from wums import boostHistHelpers as hh  # isort: skip
from wums import logging, output_tools, plot_tools  # isort: skip


hep.style.use(hep.style.ROOT)


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
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
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
        "--correlation",
        action="store_true",
        help="Plot correlation instad of covariance",
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
        "--prefit", action="store_true", help="Make prefit plot, else postfit"
    )
    parser.add_argument(
        "--selectionAxes",
        type=str,
        default=["charge", "passIso", "passMT", "cosThetaStarll", "qGen"],
        help="List of axes where for each bin a separate plot is created",
    )
    parser.add_argument(
        "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
    )
    args = parser.parse_args()

    return args


def plot_matrix(
    outdir,
    hist_obj,
    args,
    channel=None,
    axes=None,
    cmap="coolwarm",
    annot=False,
    config={},
    meta=None,
    suffix=None,
):

    matrix = hist_obj.values()

    if len(matrix.shape) > 2:
        flat = np.prod(matrix.shape[: len(matrix.shape) // 2])
        matrix = matrix.reshape((flat, flat))

    if args.correlation:
        std_dev = np.sqrt(np.diag(matrix))
        matrix = matrix / np.outer(std_dev, std_dev)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=annot,
        fmt=".2g",
        square=True,
        cbar=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1 if args.correlation else None,
        vmax=1 if args.correlation else None,
    )

    xlabel = plot_tools.get_axis_label(config, axes, args.xlabel, is_bin=True)
    ylabel = plot_tools.get_axis_label(config, axes, args.ylabel, is_bin=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plot_tools.add_decor(
        ax,
        args.title,
        args.subtitle,
        data=False,
        lumi=None,
        loc=args.titlePos,
    )

    to_join = [f"hist_{'corr' if args.correlation else 'cov'}"]
    to_join.append("prefit" if args.prefit else "postfit")
    if channel is not None:
        to_join.append(channel)
    if axes is not None:
        to_join.append("_".join(axes))
    if suffix is not None:
        to_join.append(suffix)
    to_join = [*to_join, args.postfix]

    outfile = "_".join(filter(lambda x: x, to_join))
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
        meta_info=analysis_meta_info,
    )


def main():
    """
    Plot the covariance matrix of the histogram bins
    """

    args = parseArgs()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    # load .hdf5 file first, must exist in combinetf and combinetf2
    fitresult, meta = combinetf2.io_tools.get_fitresult(
        args.infile, args.result, meta=True
    )

    plt.rcParams["font.size"] = plt.rcParams["font.size"] * args.scaleTextSize

    channel_info = meta["meta_info_input"]["channel_info"]

    hist_cov_key = f"hist_{'prefit' if args.prefit else 'postfit'}_inclusive_cov"

    results = fitresult["physics_models"]
    for margs in args.physicsModel:
        if margs == []:
            instance_keys = results.keys()
        else:
            model_key = " ".join(margs)
            instance_keys = [k for k in results.keys() if k.startswith(model_key)]
            if len(instance_keys) == 0:
                raise ValueError(
                    f"No model found under {model_key}, available models are {results.keys()}"
                )

        for instance_key in instance_keys:
            instance = results[instance_key]

            h_cov = instance[hist_cov_key].get()

            suffix = (
                instance_key.replace(" ", "_")
                .replace(".", "p")
                .replace(",", "k")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
            )

            plot_matrix(
                outdir,
                h_cov,
                args,
                config=config,
                meta=meta,
                suffix=suffix,
            )

            start = 0
            for channel, info in instance["channels"].items():
                channel_hist = info[f"hist_postfit_inclusive"].get()
                axes = [a for a in channel_hist.axes]
                if len(instance.get("channels", {}).keys()) > 1:
                    # plot covariance matrix in each channel
                    nbins = np.prod(channel_hist.shape)
                    stop = int(start + nbins)

                    h_cov_channel = h_cov[start:stop, start:stop]

                    start = stop

                    plot_matrix(
                        outdir,
                        h_cov_channel,
                        args,
                        channel=channel,
                        config=config,
                        meta=meta,
                        suffix=suffix,
                    )
                else:
                    h_cov_channel = h_cov

                selection_axes = [a for a in axes if a.name in args.selectionAxes]

                if len(selection_axes) > 0:
                    # plot covariance matrix in each bin
                    h1d = hist.Hist(*axes)
                    h2d = hh.expand_hist_by_duplicate_axes(
                        h1d,
                        [a.name for a in axes],
                        [f"y_{a.name}" for a in axes],
                        put_trailing=True,
                    )

                    vals = np.reshape(
                        h_cov_channel.values(),
                        (h_cov_channel.shape[0], *h2d.shape[: len(h2d.shape) // 2]),
                    )
                    h2d.values()[...] = np.reshape(vals, h2d.shape)

                    selection_bins = [
                        np.arange(a.size) for a in axes if a.name in args.selectionAxes
                    ]
                    other_axes = [a.name for a in axes if a not in selection_axes]

                    for bins in itertools.product(*selection_bins):
                        idxs = {a.name: i for a, i in zip(selection_axes, bins)}
                        idxs.update(
                            {f"y_{a.name}": i for a, i in zip(selection_axes, bins)}
                        )
                        idxs_centers = {
                            a.name: (
                                a.centers[i]
                                if isinstance(
                                    a, (hist.axis.Regular, hist.axis.Variable)
                                )
                                else a.edges[i]
                            )
                            for a, i in zip(selection_axes, bins)
                        }
                        h_cov_i = h2d[idxs]
                        suffix = f"{channel}_" + "_".join(
                            [
                                f"{a}_{str(i).replace('.','p').replace('-','m')}"
                                for a, i in idxs_centers.items()
                            ]
                        )
                        plot_matrix(
                            outdir,
                            h_cov_i,
                            args,
                            channel=channel,
                            axes=other_axes,
                            config=config,
                            meta=meta,
                            suffix=suffix,
                        )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
