#!/usr/bin/env python3

import itertools

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import seaborn as sns

import rabbit.io_tools
from rabbit import parsing

from wums import boostHistHelpers as hh  # isort: skip
from wums import logging, output_tools, plot_tools  # isort: skip

hep.style.use(hep.style.ROOT)


def make_parser():
    parser = parsing.plot_parser()
    parsing.add_style_args(parser)
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Plot correlation instad of covariance",
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
        "--params",
        type=str,
        nargs="*",
        default=None,
        help="Plot parameter matrix, specify list of parameter names to include in the matrix, leave free to plot all parameters",
    )
    parser.add_argument(
        "--prefit", action="store_true", help="Make prefit plot, else postfit"
    )
    parser.add_argument(
        "--showNumbers", action="store_true", help="Show numbers in the plot"
    )
    parser.add_argument(
        "--selectionAxes",
        type=str,
        nargs="*",
        default=["charge", "passIso", "passMT", "cosThetaStarll", "qGen"],
        help="List of axes where for each bin a separate plot is created",
    )
    parser.add_argument(
        "--dataCovariance",
        action="store_true",
        help="Use covariance information to plot the data uncertainty",
    )
    return parser


def plot_matrix(
    outdir,
    matrix,
    args,
    channel=None,
    axes_names=None,
    cmap="coolwarm",
    config={},
    meta=None,
    suffix=None,
    ticklabels=None,
):
    opts = dict()

    if not isinstance(matrix, np.ndarray):
        matrix = matrix.values()

    if len(matrix.shape) > 2:
        flat = np.prod(matrix.shape[: len(matrix.shape) // 2])
        matrix = matrix.reshape((flat, flat))

    if args.correlation:
        std_dev = np.sqrt(np.diag(matrix))
        matrix = matrix / np.outer(std_dev, std_dev)
        opts.update(dict(vmin=-1, vmax=1))

    if ticklabels is not None:
        opts.update(
            dict(
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            )
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=args.showNumbers,
        fmt=".2g",
        square=True,
        cbar=True,
        linewidths=0.5,
        ax=ax,
        **opts,
    )
    if ticklabels is None:
        xlabel = plot_tools.get_axis_label(config, axes_names, args.xlabel, is_bin=True)
        ylabel = plot_tools.get_axis_label(config, axes_names, args.ylabel, is_bin=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plot_tools.add_decor(
        ax,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,
        loc=args.titlePos,
        no_energy=args.noEnergy,
    )

    to_join = [f"hist_{'corr' if args.correlation else 'cov'}"]
    if args.dataCovariance:
        to_join.append("data")
    else:
        to_join.append("prefit" if args.prefit else "postfit")
    if channel is not None:
        to_join.append(channel)
    if axes_names is not None:
        to_join.append("_".join(axes_names))
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
                "RabbitOutput": meta["meta_info"],
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

    args = make_parser().parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    # load .hdf5 file
    fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, args.result, meta=True)

    plt.rcParams["font.size"] = plt.rcParams["font.size"] * args.scaleTextSize

    if args.params is not None:
        h_cov = fitresult["cov"].get()
        axes_names = h_cov.axes.name

        if len(args.params) > 0:
            h_param = fitresult["parms"].get()
            params = np.array([n for n in h_param.axes["parms"]])

            isin = np.isin(args.params, params)
            if not np.all(isin):
                missing = args.params[~isin]
                raise RuntimeError(f"Parameters {missing} not found")

            indices = np.array([np.where(params == val)[0][0] for val in args.params])

            h_cov = h_cov.values()[np.ix_(indices, indices)]

            ticklabels = [
                getattr(config, "systematics_labels", {}).get(p, p)
                for p in params[indices]
            ]
        else:
            ticklabels = None

        plot_matrix(
            outdir,
            h_cov,
            args,
            axes_names=axes_names,
            config=config,
            meta=meta,
            suffix="params",
            ticklabels=ticklabels,
        )

    if args.dataCovariance:
        hist_cov_key = "cov_data_obs"
    else:
        hist_cov_key = f"hist_{'prefit' if args.prefit else 'postfit'}_inclusive_cov"

    results = fitresult.get("mappings", fitresult.get("physics_models"))
    for margs in args.mapping:

        if margs == []:
            instance_keys = results.keys()
        else:
            mapping_key = " ".join(margs)
            instance_keys = [k for k in results.keys() if k.startswith(mapping_key)]
            if len(instance_keys) == 0:
                raise ValueError(
                    f"No mapping found under {mapping_key}; available mappings are {results.keys()}"
                )

        for instance_key in instance_keys:
            instance = results[instance_key]

            suffix = (
                instance_key.replace(" ", "_")
                .replace(".", "p")
                .replace(",", "k")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
            )

            if hist_cov_key in instance.keys():
                h_cov = instance[hist_cov_key].get()
                plot_matrix(
                    outdir,
                    h_cov,
                    args,
                    config=config,
                    meta=meta,
                    suffix=suffix,
                )
            else:
                h_cov = None

            start = 0
            for channel, info in instance["channels"].items():
                channel_hist = info["hist_postfit_inclusive"].get()
                axes = [a for a in channel_hist.axes]
                axes_names = channel_hist.axes.name

                if h_cov is None:
                    if hist_cov_key not in info.keys():
                        raise ValueError(
                            f"No key {hist_cov_key}; available; keys are {info.keys()}"
                        )
                    h_cov = info[hist_cov_key].get()

                    plot_matrix(
                        outdir,
                        h_cov,
                        args,
                        channel=channel,
                        config=config,
                        meta=meta,
                        suffix=suffix,
                        axes_names=axes_names,
                    )
                    h_cov_channel = h_cov
                elif len(instance.get("channels", {}).keys()) > 1:
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
                        axes_names=axes_names,
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

                    cov_channel_vals = (
                        h_cov_channel.values()
                        if hasattr(h_cov_channel, "values")
                        else h_cov_channel
                    )
                    if len(cov_channel_vals.shape) > 2:
                        flat = np.prod(
                            cov_channel_vals.shape[: len(cov_channel_vals.shape) // 2]
                        )
                        cov_channel_vals = cov_channel_vals.reshape((flat, flat))
                    vals = np.reshape(
                        cov_channel_vals,
                        (cov_channel_vals.shape[0], *h2d.shape[: len(h2d.shape) // 2]),
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
                            axes_names=other_axes,
                            config=config,
                            meta=meta,
                            suffix=suffix,
                        )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
