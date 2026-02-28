#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import chi2

from rabbit import io_tools

from wums import output_tools, plot_tools  # isort: skip

hep.style.use(hep.style.ROOT)


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        plt.savefig(output)

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


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
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
        "--params",
        type=str,
        nargs=2,
        action="append",
        help="Parameters to plot the likelihood scan",
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
        "--xlabel",
        type=str,
        default=None,
        help="x axis label",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="y axis label",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
    )
    parser.add_argument(
        "--noHessian",
        action="store_true",
        help="Don't show hessian contour",
    )
    parser.add_argument(
        "--noScan",
        action="store_true",
        help="Don't show map from likelihood scan",
    )
    parser.add_argument(
        "--noScanContour",
        action="store_true",
        help="Don't show contour line from likelihood scan",
    )
    parser.add_argument(
        "--noContour",
        action="store_true",
        help="Don't show contour from contour scan",
    )
    parser.add_argument(
        "--spiralScan",
        action="store_true",
        help="Plot spiral scan illustration",
    )
    return parser.parse_args()


def ellipse(cov, mu0, mu1, cl, cartesian_angle=False):

    a = cov[0, 0]
    b = cov[1, 0]
    c = cov[1, 1]

    l1 = (a + c) / 2 + np.sqrt((a - c) ** 2 / 4 + b**2)
    l2 = (a + c) / 2 - np.sqrt((a - c) ** 2 / 4 + b**2)

    theta = np.arctan2(l1 - a, b)

    sl1 = np.sqrt(cl * l1)
    sl2 = np.sqrt(cl * l2)

    def func(t):
        x = mu0 + sl1 * np.cos(theta) * np.cos(t) - sl2 * np.sin(theta) * np.sin(t)
        y = mu1 + sl1 * np.sin(theta) * np.cos(t) + sl2 * np.cos(theta) * np.sin(t)
        return x, y

    return func


def plot_scan(
    args,
    px,
    py,
    cov,
    h_scan,
    h_contour,
    scan_contour=True,
    xlabel="x",
    ylabel="y",
    confidence_levels=[
        0.68,
        0.95,
    ],
    n_points=100,
    title=None,
    subtitle=None,
    titlePos=0,
    spiral_scan=False,
):

    # Parameterize ellipse
    t = np.linspace(0, 2 * np.pi, n_points)

    # Compute confidence levels dynamically
    dof = 2  # Degrees of freedom for a 2D likelihood scan
    levels = chi2.ppf(confidence_levels, df=dof)

    if h_scan is not None:
        right = 0.97
    else:
        right = 0.99

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.subplots_adjust(left=0.14, bottom=0.14, right=right, top=0.93)

    # Plot mean point
    ax.scatter(px, py, color="red", marker="x", label="Best fit")

    for i, cl in enumerate(levels):
        if cov is not None:
            xy = ellipse(cov, px, py, cl)(t)

            label_contour = None
            label_hess = None
            if i == 0:
                label_contour = "Contour scan"
                label_hess = "Hessian"
                linestyle = "-"
            if i == 1:
                linestyle = "--"

            ax.plot(
                xy[0], xy[1], color="red", linestyle=linestyle, label=label_hess
            )  # f"{cl}σ")

        if h_contour is not None and str(cl) in h_contour.axes["confidence_level"]:
            x_contour = h_contour[{"confidence_level": str(cl), "params": 0}].values()
            y_contour = h_contour[{"confidence_level": str(cl), "params": 1}].values()
            ax.plot(
                x_contour,
                y_contour,
                marker="o",
                markerfacecolor="none",
                color="black",
                linestyle=linestyle,
                label=label_contour,
            )

    if h_scan is not None:
        x_scan = np.array(h_scan.axes["scan_x"]).astype(float)
        y_scan = np.array(h_scan.axes["scan_y"]).astype(float)
        nll_values = 2 * h_scan.values()
        plt.pcolormesh(
            x_scan, y_scan, nll_values, shading="auto", cmap="Blues", zorder=0
        )
        plt.colorbar(label=r"$-2\,\Delta \log L$")

        if spiral_scan:
            # plot path of spiral scan for illustration
            Xc, Yc = np.meshgrid(x_scan, y_scan)

            spiral_path = []
            visited = np.zeros_like(h_scan, dtype=bool)
            ny = len(y_scan)
            nx = len(x_scan)
            i, j = ny // 2, nx // 2  # start from the center
            spiral_path.append((Xc[i, j], Yc[i, j]))
            visited[i, j] = True

            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            step = 1

            while not visited.all():
                for d in range(4):
                    di, dj = dirs[d]
                    for _ in range(step):
                        i += di
                        j += dj
                        if step == 7 and Xc[i, j] < np.float32(px):
                            break

                        if 0 <= i < ny and 0 <= j < nx and not visited[i, j]:
                            spiral_path.append((Xc[i, j], Yc[i, j]))
                            visited[i, j] = True
                    if step == 7:
                        break
                    if d % 2 == 1:
                        step += (
                            1  # increase step size after completing a horizontal pass
                        )
                if step == 7:
                    break

            spiral_path = np.array(spiral_path)
            ax.plot(
                spiral_path[:-5, 0],
                spiral_path[:-5, 1],
                color="red",
                lw=2,
                label="Spiral Path",
            )

            end = spiral_path[-5]
            prev = spiral_path[-6]

            dx = end[0] - prev[0]
            dy = end[1] - prev[1]

            scale = 1.2  # scale up the arrow length
            arrow_end = end + 0.005 + scale * np.array([dx, dy])

            ax.annotate(
                "",
                xy=arrow_end,
                xytext=end + 0.005,
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
            )

        if scan_contour:
            # Overlay contours for 68% and 95% confidence levels
            linestyles = ["-", "--", "."]

            contour = plt.contour(
                x_scan,
                y_scan,
                nll_values,
                linestyles=linestyles[: len(levels)],
                levels=levels,
                colors="black",
            )
            plt.clabel(
                contour,
                fmt={l: rf"{c*100}%" for l, c in zip(levels, confidence_levels)},
                colors="black",
            )

            ax.plot(
                [], [], label="Likelihood", marker="none", color="black", linestyle="-"
            )

    plot_tools.add_decor(
        ax,
        title,
        subtitle,
        data=False,
        lumi=False,
        loc=titlePos,
        no_energy=True,
        # text_size=args.legSize,
    )

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(py, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(px, color="gray", linestyle="--", alpha=0.5)

    ax.legend(loc=args.legPos)

    return fig


def main():
    args = parseArgs()
    fitresult, meta = io_tools.get_fitresult(args.inputFile, args.result, meta=True)
    config = plot_tools.load_config(args.config)

    meta = {
        "tensorfit": meta["meta_info"],
    }

    h_params = fitresult["parms"].get()

    h_cov = None
    if "cov" in fitresult.keys():
        h_cov = fitresult["cov"].get()

    h_contour = None
    if "contour_scans2D" in fitresult.keys():
        h_contour = fitresult["contour_scans2D"].get()

    for px, py in args.params:
        px_value = h_params[{"parms": px}].value
        py_value = h_params[{"parms": py}].value

        cov = None
        if h_cov is not None and not args.noHessian:
            cov = h_cov[{"parms_x": [px, py], "parms_y": [px, py]}].values()

        h_contour_params = None
        if (
            h_contour is not None
            and f"{px}-{py}" in h_contour.axes["param_tuple"]
            and not args.noContour
        ):
            h_contour_params = h_contour[{"param_tuple": f"{px}-{py}"}]

        h_scan = None
        if f"nll_scan2D_{px}_{py}" in fitresult.keys() and not args.noScan:
            h_scan = fitresult[f"nll_scan2D_{px}_{py}"].get()

        fig = plot_scan(
            args,
            px_value,
            py_value,
            cov,
            h_scan,
            h_contour_params,
            scan_contour=not args.noScanContour,
            xlabel=plot_tools.get_axis_label(config, [px], args.xlabel),
            ylabel=plot_tools.get_axis_label(config, [py], args.ylabel),
            title=args.title,
            subtitle=args.subtitle,
            titlePos=args.titlePos,
            spiral_scan=args.spiralScan,
        )
        os.makedirs(args.outpath, exist_ok=True)
        outfile = os.path.join(args.outpath, f"nll_scan2D_{px}_{py}")
        writeOutput(
            fig,
            outfile=outfile,
            extensions=["png", "pdf"],
            meta_info=meta,
            args=args,
            postfix=args.postfix,
        )


if __name__ == "__main__":
    main()
