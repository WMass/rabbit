import argparse

import numpy as np

import rabbit.io_tools

from wums import output_tools, plot_tools, logging  # isort: skip

logger = logging.setup_logger(__file__, 4, False)

parser = argparse.ArgumentParser()
parser.add_argument(
    "infile",
    type=str,
    nargs="+",
    help="hdf5 file from rabbit or root file",
)
parser.add_argument(
    "--labels",
    type=str,
    nargs="+",
    help="Label for each input file",
)
parser.add_argument(
    "--result",
    default=None,
    type=str,
    help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
)
parser.add_argument("--logy", action="store_true", help="Make the yscale logarithmic")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
parser.add_argument("-o", "--outpath", default="./", help="output directory")
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
    "--legCols", type=int, default=2, help="Number of columns in legend"
)
parser.add_argument("--startEpoch", type=int, default=0, help="Epoch to start plotting")
parser.add_argument(
    "--types",
    nargs="+",
    default=["loss"],
    choices=["loss", "lcurve"],
    help="Make 1D plot as function of epoch/step/...",
)
parser.add_argument(
    "--ylim",
    type=float,
    nargs=2,
    help="Min and max values for y axis (if not specified, range set automatically)",
)
args = parser.parse_args()

outdir = output_tools.make_plot_dir(args.outpath)

epochs = []
times = []
losses = []
dlosses = []
tau_steps = []
lcurves = []
best_tau = []
best_curvature = []
for infile in args.infile:
    fitresult, meta = rabbit.io_tools.get_fitresult(infile, args.result, meta=True)

    if "loss" in args.types:
        h_loss = fitresult["epoch_loss"].get()
        loss = 2 * h_loss.values()
        losses.append(loss)
        dlosses.append(-np.diff(loss))  # reduction of loss after each epoch
        epochs.append(np.arange(1, len(loss) + 1))

        h_time = fitresult["epoch_time"].get()
        times.append(h_time.values())

    if "lcurve" in args.types:
        tau_steps.append(fitresult["step_tau"].get().values())
        lcurves.append(fitresult["step_lcurve"].get().values())

        if "best_tau" in fitresult.keys():
            best_tau.append(fitresult["best_tau"].get().values())
            best_curvature.append(fitresult["best_lcurve"].get().values())

linestyles = [
    "-",
    "--",
    ":",
    "-.",
    "-",
    "--",
    ":",
    "-.",
]
linestyles = [linestyles[i % len(linestyles)] for i in range(len(args.infile))]

start = args.startEpoch
stop = None

if args.labels:
    labels = args.labels
else:
    labels = [None] * len(args.infile)


def plot(x, y, xlabel, ylabel, stop, suffix, points=[]):
    if any(x in suffix for x in ["loss"]):
        # Normalize to 0
        ymin = min([min(iy) for iy in y])
        y = [iy - ymin for iy in y]

    x = [ix[start:stop] for ix in x]

    if args.ylim:
        ylim = args.ylim
    else:
        if args.logy:
            ylim = [
                min([min(iy[iy > 0]) for iy in y]) * 0.5,
                max([max(iy) for iy in y]) * 2,
            ]
        else:
            ymin = min([min(iy) for iy in y])
            ylim = [ymin * 1.1 if ymin < 0 else 0, max([max(iy) for iy in y]) * 1.1]

    fig, ax1 = plot_tools.figure(
        None,
        xlabel,
        ylabel,
        width_scale=1,
        xlim=[min([min(ix) for ix in x]), max([max(ix) for ix in x])],
        ylim=ylim,
        automatic_scale=False,
        logy=args.logy,
    )

    ax1.plot([1.85, 1.85], ylim, color="grey", linestyle="--")

    for ix, iy, l, s in zip(x, y, labels, linestyles):
        ax1.plot(ix, iy, label=l, linestyle=s)

    for point_x, point_y in points:
        ax1.plot(
            point_x,
            point_y,
            marker="*",
            markersize=15,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="None",
        )

    plot_tools.add_decor(
        ax1,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,
        loc=args.titlePos,
        no_energy=True,
    )

    plot_tools.addLegend(
        ax1,
        ncols=args.legCols,
        loc=args.legPos,
    )

    name = f"epoch_{suffix}"
    to_join = [name, args.postfix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if args.subtitle == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)
    analysis_meta_info = None
    output_tools.write_index_and_log(
        outdir,
        outfile,
        args=args,
    )


combinations = []
if "loss" in args.types:
    plot(epochs, losses, "epoch", r"$-2\Delta \ln(L)$", None, "loss")
    plot(
        epochs,
        dlosses,
        "time [s]",
        r"$-2(\ln(L_{t}) - \ln(L_{t-1}))$",
        -1,
        "reduction_loss",
    )
    if "time" in args.types:
        plot(times, losses, "time [s]", r"$-2\Delta \ln(L)$", None, "loss_time")
        plot(
            times,
            dlosses,
            "epoch",
            r"$-2(\ln(L_{t}) - \ln(L_{t-1}))$",
            -1,
            "reduction_loss_time",
        )

if "lcurve" in args.types:
    plot(
        tau_steps,
        lcurves,
        r"$\tau$",
        "Curvature",
        None,
        "lcurve",
        points=zip(best_tau, best_curvature),
    )
