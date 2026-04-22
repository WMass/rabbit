"""
Create a test HDF5 tensor with six ExtendedABCD regions.

Layout (x=isolation bins, y=mt bins):
  ch_Ax: extra-sideband isolation, low mt  (x=extra-sideband, y=signal)
  ch_Bx: extra-sideband isolation, high mt (x=extra-sideband, y=sideband)
  ch_A:  sideband isolation, low mt        (x=sideband, y=signal)
  ch_B:  sideband isolation, high mt       (x=sideband, y=sideband)
  ch_C:  signal isolation, low mt          (x=signal, y=signal)
  ch_D:  signal isolation, high mt         (x=signal, y=sideband) ← predicted

The ExtendedABCD formula: D = C * Ax * B² / (Bx * A²).

A signal process lives in all six channels (with small yield in control regions)
to allow a Mu POI alongside the background model. A nonprompt background lives
in all six channels.
"""

import argparse
import os

import hist
import numpy as np

from rabbit import tensorwriter

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument(
    "--outname", default="test_extended_abcd_tensor", help="output file name"
)
args = parser.parse_args()

rng = np.random.default_rng(42)

n_pt = 5  # number of pt bins
ax_pt = hist.axis.Regular(n_pt, 0, 50, name="pt")


def make_hist(n_events, mean_weight):
    """Histogram filled with uniform pt values and given mean weight."""
    h = hist.Hist(ax_pt, storage=hist.storage.Weight())
    pt_vals = rng.uniform(0, 50, n_events)
    weights = rng.normal(mean_weight, 0.1 * mean_weight, n_events)
    h.fill(pt=pt_vals, weight=weights)
    return h


def make_data(sig_h, bkg_h):
    """Poisson-fluctuated data from signal + background templates."""
    h = hist.Hist(ax_pt, storage=hist.storage.Double())
    sig_vals = sig_h.values() if sig_h is not None else 0.0
    expected = sig_vals + bkg_h.values()
    h.view()[:] = rng.poisson(np.maximum(expected, 0))
    return h


# Signal process: present in all channels (small in control regions)
h_sig_Ax = make_hist(200, 0.05)
h_sig_Bx = make_hist(200, 0.05)
h_sig_A = make_hist(500, 0.1)
h_sig_B = make_hist(500, 0.1)
h_sig_C = make_hist(5000, 1.0)
h_sig_D = make_hist(5000, 1.0)

# Nonprompt background: set up so ExtendedABCD relation holds approximately
# f(x) = B/A is the "fake rate" per bin.
# f_extra = Bx/Ax (extra sideband)
# f_sideband = B/A
# f_signal = f_sideband² / f_extra = B²*Ax / (A²*Bx)
# D ≈ C * f_signal = C * B² * Ax / (A² * Bx)
h_np_Ax = make_hist(10000, 3.0)  # extra sideband
h_np_Bx = make_hist(10000, 9.0)  # Bx ~ 3x Ax → f_extra = 3
h_np_A = make_hist(8000, 2.0)
h_np_B = make_hist(8000, 5.0)  # B ~ 2.5x A → f_sb = 2.5
h_np_C = make_hist(6000, 1.5)
# D ~ C * f_signal = C * 2.5² / 3 * correction ≈ C * 2.08
h_np_D = make_hist(6000, 3.1)

# Poisson-fluctuated data
h_data_Ax = make_data(h_sig_Ax, h_np_Ax)
h_data_Bx = make_data(h_sig_Bx, h_np_Bx)
h_data_A = make_data(h_sig_A, h_np_A)
h_data_B = make_data(h_sig_B, h_np_B)
h_data_C = make_data(h_sig_C, h_np_C)
h_data_D = make_data(h_sig_D, h_np_D)

# --- Build tensor ---
writer = tensorwriter.TensorWriter(sparse=False)

for ch in ["ch_Ax", "ch_Bx", "ch_A", "ch_B", "ch_C", "ch_D"]:
    writer.add_channel(h_sig_C.axes, ch)

# Data
writer.add_data(h_data_Ax, "ch_Ax")
writer.add_data(h_data_Bx, "ch_Bx")
writer.add_data(h_data_A, "ch_A")
writer.add_data(h_data_B, "ch_B")
writer.add_data(h_data_C, "ch_C")
writer.add_data(h_data_D, "ch_D")

# Signal process (all channels, dominant in C and D)
writer.add_process(h_sig_Ax, "signal", "ch_Ax", signal=True)
writer.add_process(h_sig_Bx, "signal", "ch_Bx", signal=True)
writer.add_process(h_sig_A, "signal", "ch_A", signal=True)
writer.add_process(h_sig_B, "signal", "ch_B", signal=True)
writer.add_process(h_sig_C, "signal", "ch_C", signal=True)
writer.add_process(h_sig_D, "signal", "ch_D", signal=True)

# Nonprompt background (all six channels)
writer.add_process(h_np_Ax, "nonprompt", "ch_Ax", signal=False)
writer.add_process(h_np_Bx, "nonprompt", "ch_Bx", signal=False)
writer.add_process(h_np_A, "nonprompt", "ch_A", signal=False)
writer.add_process(h_np_B, "nonprompt", "ch_B", signal=False)
writer.add_process(h_np_C, "nonprompt", "ch_C", signal=False)
writer.add_process(h_np_D, "nonprompt", "ch_D", signal=False)

outdir = os.path.join(args.output, args.outname)
writer.write(outdir)
print(f"Written to {outdir}/rabbit_input.hdf5")
