"""
Create a test HDF5 tensor with four ABCD regions for testing ABCDModel.

Layout:
  ch_A: pt histogram — fake rate measurement region A (low iso, low mt)
  ch_B: pt histogram — fake rate measurement region B (high iso, low mt)
  ch_C: pt histogram — application region C (low iso, high mt)
  ch_D: pt histogram — signal region D (high iso, high mt)

The ABCD method predicts D = C * A / B.
A signal process lives in ch_C and ch_D.
A nonprompt background lives in all four channels.
"""

import argparse
import os

import hist
import numpy as np

from rabbit import tensorwriter

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="test_abcd_tensor", help="output file name")
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


# Signal process: only in ch_C and ch_D
h_sig_C = make_hist(5000, 1.0)
h_sig_D = make_hist(5000, 1.0)

# Nonprompt background: present in all four channels
# Set up so the ABCD relation A*C/B ≈ D holds approximately per bin
h_np_A = make_hist(8000, 2.0)
h_np_B = make_hist(8000, 4.0)  # B ~ 2x A
h_np_C = make_hist(6000, 1.5)
h_np_D = make_hist(3000, 0.75)  # D ~ A*C/B per bin

# Poisson-fluctuated data
h_data_A = make_data(None, h_np_A)
h_data_B = make_data(None, h_np_B)
h_data_C = make_data(h_sig_C, h_np_C)
h_data_D = make_data(h_sig_D, h_np_D)

# --- Build tensor ---
writer = tensorwriter.TensorWriter(sparse=False)

writer.add_channel(h_sig_C.axes, "ch_A")
writer.add_channel(h_sig_C.axes, "ch_B")
writer.add_channel(h_sig_C.axes, "ch_C")
writer.add_channel(h_sig_C.axes, "ch_D")

# Data
writer.add_data(h_data_A, "ch_A")
writer.add_data(h_data_B, "ch_B")
writer.add_data(h_data_C, "ch_C")
writer.add_data(h_data_D, "ch_D")

# Signal process (only in C and D)
writer.add_process(h_sig_C, "signal", "ch_C", signal=True)
writer.add_process(h_sig_D, "signal", "ch_D", signal=True)

# Nonprompt background (all four channels)
writer.add_process(h_np_A, "nonprompt", "ch_A", signal=False)
writer.add_process(h_np_B, "nonprompt", "ch_B", signal=False)
writer.add_process(h_np_C, "nonprompt", "ch_C", signal=False)
writer.add_process(h_np_D, "nonprompt", "ch_D", signal=False)

outdir = os.path.join(args.output, args.outname)
writer.write(outdir)
print(f"Written to {outdir}/rabbit_input.hdf5")
