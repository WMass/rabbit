# plot different symmetrizations

import argparse

import hist
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D

from rabbit import debugdata, inputdata, tensorwriter

from wums import output_tools, plot_tools  # isort: skip
from wums import boostHistHelpers as hh  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outpath", default="./", help="output directory")
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
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
    "--legCols", type=int, default=2, help="Number of columns in legend"
)
parser.add_argument(
    "--fluctuate",
    action="store_true",
    help="To add fluctuations to the systematic histogram",
)
args = parser.parse_args()

outdir = output_tools.make_plot_dir(args.outpath)


h = hist.Hist(hist.axis.Regular(10, -0.5, 9.5, name="x"), storage=hist.storage.Double())

rng = np.random.default_rng(seed=42)
x = rng.uniform(0, 10, 100000)

h.fill(x)

# Build tensor
writer = tensorwriter.TensorWriter(
    systematic_type=args.systematicType,
)

writer.add_channel(h.axes, "ch0")

writer.add_data(h, "ch0")

writer.add_process(h, "sig", "ch0")

# systematic uncertainties

# f(x) = a * x + b
a, b = 0.01, -0.05  # Linear coefficients
bin_centers = h.axes[0].centers  # Get bin centers
weights = a * bin_centers + b  # Compute weights

a, b = 0.0005, -0.025  # Linear coefficients
weights_asym = a * bin_centers**2 + b  # Compute weights

# Reweight the histogram values
h_syst0_up = h.copy()
h_syst0_up.values()[...] = h.values() * (1 + weights)

h_syst0_down = h.copy()
h_syst0_down.values()[...] = h.values() / (1 + weights)

h_syst1_up = h.copy()
h_syst1_up.values()[...] = h.values() * (1 + weights_asym) ** 2

# add poisson fluctuations on systematics
if args.fluctuate:
    h_syst0_up.values()[...] = rng.poisson(h_syst0_up.values())
    h_syst0_down.values()[...] = rng.poisson(h_syst0_down.values())
    h_syst1_up.values()[...] = rng.poisson(h_syst1_up.values())


writer.add_systematic(
    [h_syst0_up, h_syst0_down],
    "nominal",
    "sig",
    "ch0",
    symmetrize=None,
)

writer.add_systematic(
    [h_syst0_up, h_syst0_down],
    "average",
    "sig",
    "ch0",
    symmetrize="average",
)

writer.add_systematic(
    [h_syst0_up, h_syst0_down],
    "conservative",
    "sig",
    "ch0",
    symmetrize="conservative",
)

writer.add_systematic(
    [h_syst1_up, h_syst0_down],
    "asym",
    "sig",
    "ch0",
    symmetrize=None,
)


writer.add_systematic(
    [h_syst1_up, h_syst0_down],
    "linear",
    "sig",
    "ch0",
    symmetrize="linear",
)

writer.add_systematic(
    [h_syst1_up, h_syst0_down],
    "quadratic",
    "sig",
    "ch0",
    symmetrize="quadratic",
)

# TODO: add option to skip writing and directly use object
writer.write(outfolder=outdir, outfilename="symmetrize_tensor.hdf5")

indata = inputdata.FitInputData(f"{outdir}/symmetrize_tensor.hdf5")
debug = debugdata.FitDebugData(indata)

hist_proc = debug.nominal_hists["ch0"][{"processes": "sig"}]
hist_syst = debug.syst_hists["ch0"][{"processes": "sig"}]

# check if input histograms agree with what is actually in the indata object
test_pairs = [
    (hist_proc, h, "central"),
    (hist_syst[{"DownUp": "Up", "systs": "nominal"}], h_syst0_up, "nominal-up"),
    (hist_syst[{"DownUp": "Down", "systs": "nominal"}], h_syst0_down, "nominal-down"),
]

if not args.fluctuate:
    test_pairs.extend(
        [
            (hist_syst[{"DownUp": "Up", "systs": "average"}], h_syst0_up, "average-up"),
            (
                hist_syst[{"DownUp": "Up", "systs": "conservative"}],
                h_syst0_up,
                "conservative-up",
            ),
            (
                hist_syst[{"DownUp": "Down", "systs": "average"}],
                h_syst0_down,
                "average-down",
            ),
            (
                hist_syst[{"DownUp": "Down", "systs": "conservative"}],
                h_syst0_down,
                "conservative-down",
            ),
        ]
    )

for h1, h2, stype in test_pairs:
    if not np.all(np.isclose(h1.values(), h2.values())):
        original = h1.values()
        stored = h2.values()
        raise RuntimeError(f"""
                        '{stype}' Histograms should agree but got
                        - original: {original}
                        - stored: {stored}
                        """)


labels_dict = {
    "nominal": "Asymmetric",
    "asym": "Asymmetric",
    "average": "Average",
    "conservative": "Conservative",
    "linearSymAvg": "Linear [avg.]",
    "linearSymDiff": "Linear [diff.]",
    "quadraticSymAvg": "Quadratic [avg.]",
    "quadraticSymDiff": "Quadratic [diff.]",
}

# plot variations
ratio = True  # args.systematicType == "log_normal"

xlabel = "Bin index"
if ratio:
    ylabel = "variation / nominal"
else:
    ylabel = "variation - nominal"


def make_plot(
    name, linestyles=["-", "--", ":"], stypes=["nominal", "average", "conservative"]
):

    fig, ax1 = plot_tools.figure(h, xlabel, ylabel, width_scale=0.8, ylim=[0.94, 1.09])

    # nominal hist
    hep.histplot(
        hh.divideHists(hist_proc, hist_proc),
        xerr=False,
        yerr=False,
        histtype="step",
        color="black",
        linewidth=2,
        label=None,
        density=False,
        binwnorm=None,
        ax=ax1,
        zorder=1,
        flow="none",
    )

    for i, stype in enumerate(stypes):
        hep.histplot(
            [
                (
                    hh.divideHists(hist_syst[{"DownUp": "Up", "systs": stype}], h)
                    if ratio
                    else hh.addHists(
                        hist_syst[{"DownUp": "Up", "systs": stype}], h, scale1=-1
                    )
                ),
                (
                    hh.divideHists(hist_syst[{"DownUp": "Down", "systs": stype}], h)
                    if ratio
                    else hh.addHists(
                        hist_syst[{"DownUp": "Down", "systs": stype}], h, scale1=-1
                    )
                ),
            ],
            xerr=False,
            yerr=False,
            histtype="step",
            color=["red", "blue"],
            # label=["Up", "Down"],
            linestyle=linestyles[i],
            linewidth=2,
            density=False,
            binwnorm=None,
            ax=ax1,
            zorder=1,
            flow="none",
        )

    # --- dummy handles for colors ---
    color_handles = [
        Line2D([0], [0], color="red", lw=2, label="Up"),
        Line2D([0], [0], color="blue", lw=2, label="Down"),
    ]

    # --- dummy handles for line styles ---
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=ls, label=labels_dict[l])
        for ls, l in zip(linestyles, stypes)
    ]

    legend1 = ax1.legend(handles=color_handles, loc="upper left")
    legend2 = ax1.legend(handles=style_handles, loc="upper right")

    ax1.add_artist(legend1)

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

    # plot_tools.fix_axes(ax1, yscale=args.yscale, logy=args.logy)

    to_join = [name, args.systematicType]
    if args.fluctuate:
        to_join.append("fluctuate")
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


make_plot(
    "symmetric",
    linestyles=["-", "--", ":"],
    stypes=["nominal", "average", "conservative"],
)

make_plot(
    "linear",
    linestyles=["-", "--", ":"],
    stypes=["asym", "linearSymAvg", "linearSymDiff"],
)

make_plot(
    "quadratic",
    linestyles=["-", "--", ":"],
    stypes=["asym", "quadraticSymAvg", "quadraticSymDiff"],
)
