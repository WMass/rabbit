import argparse

import numpy as np

from rabbit import tensorwriter

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="test_tensor", help="output file name")
parser.add_argument(
    "--postfix",
    default=None,
    type=str,
    help="Postfix to append on output file name",
)
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
parser.add_argument(
    "--symmetrizeAll",
    default=False,
    action="store_true",
    help="Make fully symmetric tensor",
)
parser.add_argument(
    "--skipMaskedChannels",
    default=False,
    action="store_true",
    help="Skip adding masked channels",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)
parser.add_argument(
    "--addSystToDataCovariance",
    default=False,
    action="store_true",
    help="Add systematics to data covariance matrix, only works with '--systematicType normal'",
)
parser.add_argument(
    "--histType",
    choices=["hist", "boost", "root"],
    default="hist",
    help="Type of input histogram",
)
args = parser.parse_args()

# Make histograms using different interfaces
if args.histType == "hist":
    import hist

    def Axis(*bins, axis_type="Regular", name="h1"):
        a = getattr(hist.axis, axis_type)(*bins, name=name)
        return a

    def Histogram(*axes, weighted=False):

        h = hist.Hist(
            *axes, storage=hist.storage.Weight() if weighted else hist.storage.Double()
        )
        return h

    def fill(h, *values, weight=None):
        if weight is not None:
            h.fill(*values, weight=weight)
        else:
            h.fill(*values)

elif args.histType == "boost":
    import boost_histogram as bh

    def Axis(*bins, axis_type="Regular", name=None):
        a = getattr(bh.axis, axis_type)(*bins)
        return a

    def Histogram(*axes, weighted=False):

        h = bh.Histogram(
            *axes, storage=bh.storage.Weight() if weighted else bh.storage.Double()
        )
        return h

    def fill(h, *values, weight=None):
        if weight is not None:
            h.fill(*values, weight=weight)
        else:
            h.fill(*values)

elif args.histType == "root":
    from array import array

    import ROOT

    def Axis(*bins, axis_type="Regular", name=None):
        return bins

    def Histogram(*axes, name="h2", title="Hist 1D", weighted=False):

        # format regular or variable axes
        axs = [
            (len(a[0]) - 1, array("d", a[0])) if isinstance(a[0], list) else a
            for a in axes
        ]

        if len(axes) == 1:
            h = ROOT.TH1D(name, title, *axs[0])

        elif len(axes) == 2:
            h = ROOT.TH2D(name, title, *axs[0], *axs[1])

        return h

    def fill(h, *values, weight=None):
        n = len(values[0])
        vals = [array("d", v) for v in values]
        if weight is not None:
            wgt = array("d", weight)
        else:
            wgt = array("d", [1.0] * n)
        h.FillN(n, *vals, wgt)


bins_x = [10, -5, 5]
bins_a = [10, 0, 5]
bins_b = [0, 1, 3, 6, 10, 20]

ax_x = Axis(*bins_x, axis_type="Regular", name="x")
ax_a = Axis(*bins_a, axis_type="Regular", name="a")
ax_b = Axis(bins_b, axis_type="Variable", name="b")

h1_data = Histogram(ax_x)
h2_data = Histogram(ax_a, ax_b)

h1_sig = Histogram(ax_x, weighted=True)
h2_sig = Histogram(ax_a, ax_b, weighted=True)

h1_bkg = Histogram(ax_x, weighted=True)
h2_bkg = Histogram(ax_a, ax_b, weighted=True)

h1_bkg_2 = Histogram(ax_x, weighted=True)

# masked channel e.g. for gen level distribution
h1_sig_masked = Histogram(ax_x, weighted=True)

# for pseudodata
h1_pseudo = Histogram(ax_x, weighted=True)
h2_pseudo = Histogram(ax_a, ax_b, weighted=True)

# Generate random data for filling
np.random.seed(42)  # For reproducibility


def get_sig(factor=1):
    # gaussian distributed signal
    x = np.random.normal(0, 1, 10000 * factor)
    w_x = np.random.normal(1 / factor, 0.2, 10000 * factor)

    a = np.random.normal(2, 1, 15000 * factor)
    b = np.random.normal(10, 2.5, 15000 * factor)
    w_ab = np.random.normal(1 / factor, 0.2, 15000 * factor)
    return x, w_x, a, b, w_ab


def get_sig_masked(factor=1):
    # gaussian distributed signal
    x = np.random.normal(0, 0.8, 10000 * factor)
    w_x = np.random.normal(1 / factor, 0.1, 10000 * factor)
    return x, w_x


def get_bkg(factor=1):
    # uniform distributed background
    x = np.random.uniform(-5, 5, 5000 * factor)
    w_x = np.random.normal(1 / factor, 0.2, 5000 * factor)

    a = np.random.uniform(0, 5, 7000 * factor)
    b = np.random.uniform(0, 20, 7000 * factor)
    w_ab = np.random.normal(1 / factor, 0.2, 7000 * factor)
    return x, w_x, a, b, w_ab


def get_bkg_2():
    # uniform distributed background
    x = np.random.normal(0.5, 1.5, 5000)
    return x


# Fill histograms
x, w_x, a, b, w_ab = get_sig()
fill(h1_data, x)
fill(h2_data, a, b)

x, w_x, a, b, w_ab = get_bkg()
fill(h1_data, x)
fill(h2_data, a, b)

x = get_bkg_2()
fill(h1_data, x)

x, w_x, a, b, w_ab = get_sig(3)
fill(h1_sig, x, weight=w_x)
fill(h2_sig, a, b, weight=w_ab)

x, w_x, a, b, w_ab = get_bkg(2)
fill(h1_bkg, x, weight=w_x)
fill(h2_bkg, a, b, weight=w_ab)

x = get_bkg_2()
fill(h1_bkg_2, x)

if not args.skipMaskedChannels:
    x, w_x = get_sig_masked(3)
    fill(h1_sig_masked, x, weight=w_x)

# pseudodata as exact composition of signal and background
h1_pseudo.values()[...] = (
    h1_sig.values() + h1_bkg.values()[...] + h1_bkg_2.values()[...]
)
h2_pseudo.values()[...] = h2_sig.values() + h2_bkg.values()[...]

h1_pseudo.variances()[...] = (
    h1_sig.variances() + h1_bkg.variances()[...] + h1_bkg_2.variances()[...]
)
h2_pseudo.variances()[...] = h2_sig.variances() + h2_bkg.variances()[...]

# scale signal down signal by 10%
h1_sig.values()[...] = h1_sig.values() * 0.9
h2_sig.values()[...] = h2_sig.values() * 0.9

# scale bkg up background by 5%
h1_bkg.values()[...] = h1_bkg.values() * 1.05
h2_bkg.values()[...] = h2_bkg.values() * 1.05

# scale bkg 2 down by 10%
h1_bkg_2.values()[...] = h1_bkg_2.values() * 0.9

# data covariance matrix
variances_flat = np.concatenate(
    [h1_data.values().flatten(), h2_data.values().flatten()]
)
cov = np.diag(variances_flat)

# add fully correlated contribution
variances_bkg = np.concatenate([h1_bkg.values().flatten(), h2_bkg.values().flatten()])
cov_bkg = np.diag(variances_bkg * 0.05)

# add bin by bin stat uncertainty on diagonal elements
cov += np.diag(np.concatenate([h1_sig.values().flatten(), h2_sig.values().flatten()]))
cov += np.diag(np.concatenate([h1_bkg.values().flatten(), h2_bkg.values().flatten()]))
cov += np.diag(
    np.concatenate(
        [h1_bkg_2.values().flatten(), np.zeros_like(h2_bkg.values().flatten())]
    )
)

# Build tensor
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)

writer.add_channel(h1_data.axes, "ch0")
writer.add_channel(h2_data.axes, "ch1")

writer.add_data(h1_data, "ch0")
writer.add_data(h2_data, "ch1")

writer.add_pseudodata(h1_pseudo, "original", "ch0")
writer.add_pseudodata(h2_pseudo, "original", "ch1")

writer.add_data_covariance(cov)

writer.add_process(h1_sig, "sig", "ch0", signal=True)
writer.add_process(h2_sig, "sig", "ch1", signal=True)

writer.add_process(h1_bkg, "bkg", "ch0")
writer.add_process(h2_bkg, "bkg", "ch1")

writer.add_process(h1_bkg_2, "bkg_2", "ch0")

if not args.skipMaskedChannels:
    # add masked channel
    writer.add_channel(h1_sig_masked.axes, "ch0_masked", masked=True)
    writer.add_process(h1_sig_masked, "sig", "ch0_masked", signal=True)

# systematic uncertainties

writer.add_norm_systematic("norm", ["sig", "bkg", "bkg_2"], "ch0", 1.02)
writer.add_norm_systematic("norm", ["sig", "bkg"], "ch1", [1.02, 1.03])

writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
writer.add_norm_systematic("bkg_norm", "bkg", "ch1", 1.05)

writer.add_norm_systematic("bkg_2_norm", "bkg_2", "ch0", 1.1)

# shape systematics for channel ch0

# Apply reweighting: linear function of axis value
# f(x) = a * x + b
a, b = 0.01, -0.05  # Linear coefficients
bin_centers = h1_bkg.axes[0].centers  # Get bin centers
bin_centers -= bin_centers[0]
weights = a * bin_centers + b  # Compute weights

# Reweight the histogram values
h1_bkg_syst0 = h1_bkg.copy()
h1_bkg_syst0.values()[...] = h1_bkg.values() * (1 + weights)

# unconstrained systematic that is not an NOI
writer.add_systematic(
    h1_bkg_syst0,
    "slope_background",
    "bkg",
    "ch0",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_background"],
    constrained=False,
)

h1_sig_syst1_up = h1_sig.copy()
h1_sig_syst1_dn = h1_sig.copy()
h1_sig_syst1_up.values()[...] = h1_sig.values() * (1 + weights)
h1_sig_syst1_dn.values()[...] = h1_sig.values() * (1 - weights)

# constrained systematic that is an NOI
writer.add_systematic(
    [h1_sig_syst1_up, h1_sig_syst1_dn],
    "slope_signal_ch0",
    "sig",
    "ch0",
    groups=["slopes", "slopes_signal"],
    symmetrize="average",
    kfactor=1.2,
    constrained=True,
    noi=True,
)

# unconstrained systematic that is an NOI
writer.add_systematic(
    [h1_sig_syst1_up, h1_sig_syst1_dn],
    "slope_signal",
    "sig",
    "ch0",
    symmetrize="average",
    constrained=False,
    noi=True,
)

if not args.skipMaskedChannels:
    h1_sig_masked_syst1_up = h1_sig_masked.copy()
    h1_sig_masked_syst1_dn = h1_sig_masked.copy()
    h1_sig_masked_syst1_up.values()[...] = h1_sig_masked.values() * (1 + weights)
    h1_sig_masked_syst1_dn.values()[...] = h1_sig_masked.values() * (1 - weights)

    writer.add_systematic(
        [h1_sig_masked_syst1_up, h1_sig_masked_syst1_dn],
        "slope_signal",
        "sig",
        "ch0_masked",
        symmetrize="average",
        constrained=False,
        noi=True,
    )

h1_sig_syst2_up = h1_sig.copy()
h1_sig_syst2_dn = h1_sig.copy()
h1_sig_syst2_up.values()[...] = h1_sig.values() * (1 + weights) ** 2
h1_sig_syst2_dn.values()[...] = h1_sig.values() * (1 - weights)

writer.add_systematic(
    [h1_sig_syst2_up, h1_sig_syst2_dn],
    "slope_lin_signal_ch0",
    "sig",
    "ch0",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_signal"],
    symmetrize="linear",
)

h1_sig_syst3_up = h1_sig.copy()
h1_sig_syst3_dn = h1_sig.copy()
h1_sig_syst3_up.values()[...] = h1_sig.values() * (1 + weights) ** 3
h1_sig_syst3_dn.values()[...] = h1_sig.values() * (1 - weights) ** 2

writer.add_systematic(
    [h1_sig_syst2_up, h1_sig_syst2_dn],
    "slope_quad_signal_ch0",
    "sig",
    "ch0",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_signal"],
    symmetrize="quadratic",
)


# shape systematics for channel ch1

bin_centers = h2_bkg.axes[0].centers  # Get bin centers
bin_centers -= bin_centers[0]
weights = (a * bin_centers + b)[..., None]  # Compute weights

h2_bkg_syst0 = h2_bkg.copy()
h2_bkg_syst0.values()[...] = h2_bkg.values() * (1 + weights)
writer.add_systematic(
    h2_bkg_syst0,
    "slope_background",
    "bkg",
    "ch1",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_background"],
    constrained=False,
)

h2_sig_syst1_up = h2_sig.copy()
h2_sig_syst1_dn = h2_sig.copy()
h2_sig_syst1_up.values()[...] = h2_sig.values() * (1 + weights)
h2_sig_syst1_dn.values()[...] = h2_sig.values() * (1 - weights)

writer.add_systematic(
    [h2_sig_syst1_up, h2_sig_syst1_dn],
    "slope_signal_ch1",
    "sig",
    "ch1",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_signal"],
    symmetrize="conservative",
)

writer.add_systematic(
    [h2_sig_syst1_up, h2_sig_syst1_dn],
    "slope_signal",
    "sig",
    "ch1",
    symmetrize="average",
    constrained=False,
    noi=True,
)

# add an asymmetric uncertainty (or symmetrize)
h2_sig_syst2_up = h2_sig.copy()
h2_sig_syst2_dn = h2_sig.copy()
h2_sig_syst2_up.values()[...] = h2_sig.values() * (1 + weights) ** 2
h2_sig_syst2_dn.values()[...] = h2_sig.values() * (1 - weights)

writer.add_systematic(
    [h2_sig_syst2_up, h2_sig_syst2_dn],
    "slope_2_signal_ch1",
    "sig",
    "ch1",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_signal"],
    symmetrize="quadratic" if args.symmetrizeAll else None,
)

directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"
writer.write(outfolder=directory, outfilename=filename)
