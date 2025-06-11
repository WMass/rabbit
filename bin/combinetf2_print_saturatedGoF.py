#!/usr/bin/env python3
import argparse

import scipy

import combinetf2.io_tools

parser = argparse.ArgumentParser()
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

args = parser.parse_args()

fitresult, meta = combinetf2.io_tools.get_fitresult(args.infile, args.result, meta=True)

nllvalfull = fitresult["nllvalfull"]
satnllvalfull = fitresult["satnllvalfull"]
ndf = fitresult["ndfsat"]
chi2 = 2.0 * (nllvalfull - satnllvalfull)
p_val = scipy.stats.chi2.sf(chi2, ndf)

print("Saturated chi2:")
print("    ndof: ", ndf)
print("    2*deltaNLL: ", round(chi2, 2))
print("    p-value (%): ", round(p_val * 100, 2))
