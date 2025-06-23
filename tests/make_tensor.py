import argparse

import hist
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

args = parser.parse_args()

# Make histograms
ax_x = hist.axis.Regular(10, -5, 5, name="x")
ax_a = hist.axis.Regular(10, 0, 5, name="a")
ax_b = hist.axis.Variable([0, 1, 3, 6, 10, 20], name="b")

h1_data = hist.Hist(ax_x, storage=hist.storage.Double())
h2_data = hist.Hist(ax_a, ax_b, storage=hist.storage.Double())

h1_sig = hist.Hist(ax_x, storage=hist.storage.Weight())
h2_sig = hist.Hist(ax_a, ax_b, storage=hist.storage.Weight())

h1_bkg = hist.Hist(ax_x, storage=hist.storage.Weight())
h2_bkg = hist.Hist(ax_a, ax_b, storage=hist.storage.Weight())

h1_bkg_2 = hist.Hist(ax_x, storage=hist.storage.Weight())

# masked channel e.g. for gen level distribution
h1_sig_masked = hist.Hist(ax_x, storage=hist.storage.Weight())

# for pseudodata
h1_pseudo = hist.Hist(ax_x, storage=hist.storage.Weight())
h2_pseudo = hist.Hist(ax_a, ax_b, storage=hist.storage.Weight())

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
h1_data.fill(x)
h2_data.fill(a, b)

x, w_x, a, b, w_ab = get_bkg()
h1_data.fill(x)
h2_data.fill(a, b)

x = get_bkg_2()
h1_data.fill(x)

x, w_x, a, b, w_ab = get_sig(3)
h1_sig.fill(x, weight=w_x)
h2_sig.fill(a, b, weight=w_ab)

x, w_x, a, b, w_ab = get_bkg(2)
h1_bkg.fill(x, weight=w_x)
h2_bkg.fill(a, b, weight=w_ab)

x = get_bkg_2()
h1_bkg_2.fill(x)

if not args.skipMaskedChannels:
    x, w_x = get_sig_masked(3)
    h1_sig_masked.fill(x, weight=w_x)

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

writer.add_lnN_systematic("norm", ["sig", "bkg", "bkg_2"], "ch0", 1.02)
writer.add_lnN_systematic("norm", ["sig", "bkg"], "ch1", [1.02, 1.03])

writer.add_lnN_systematic("bkg_norm", "bkg", "ch0", 1.05)
writer.add_lnN_systematic("bkg_norm", "bkg", "ch1", 1.05)

writer.add_lnN_systematic("bkg_2_norm", "bkg_2", "ch0", 1.1)

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

writer.add_systematic(
    h1_bkg_syst0,
    "slope_background",
    "bkg",
    "ch0",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_background"],
)

h1_sig_syst1_up = h1_sig.copy()
h1_sig_syst1_dn = h1_sig.copy()
h1_sig_syst1_up.values()[...] = h1_sig.values() * (1 + weights)
h1_sig_syst1_dn.values()[...] = h1_sig.values() * (1 - weights)

writer.add_systematic(
    [h1_sig_syst1_up, h1_sig_syst1_dn],
    "slope_signal_ch0",
    "sig",
    "ch0",
    add_to_data_covariance=args.addSystToDataCovariance,
    groups=["slopes", "slopes_signal"],
    symmetrize="average",
    kfactor=1.2,
)

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
