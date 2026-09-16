#!/usr/bin/env python3
"""Generate the tabulated LO parton luminosity used by :mod:`rabbit.lineshapes.zgamma`.

The Z/gamma* lineshape needs, per quark flavour ``f``, the LO parton
luminosity

    L_f(Q^2) = int_tau^1 dx (tau/x) [ f(x, Q^2) fbar(tau/x, Q^2)
                                    + fbar(x, Q^2) f(tau/x, Q^2) ] ,
    tau = Q^2 / s ,

evaluated at the factorisation scale ``mu_F = Q``. It depends on the PDF set
and the beam energy but **not** on ``m_Z``, ``Gamma_Z`` or ``sin^2(theta_W)``,
so it is a fixed input of the fit: computed once here with LHAPDF and shipped
as an ``npz`` table, which the provider splines onto its own mass grid.

The luminosity integral is imported verbatim from the reference lineshape
study (``calibration_studies/lineshape/drell_yan_xsec.py``,
``integrate_sigma_hat_prime_sm``) so the table is bit-identical to what that
code computes; the reference directory is only needed *here*, never at fit
time.

Run it in the lineshape environment (LHAPDF + numpy 2), not in the rabbit
TensorFlow environment::

    cd /work/submit/david_w/ZMass/calibration_studies && source setup_env.sh
    python /path/to/rabbit/lineshapes/make_lumi_table.py \
        --pdfset NNPDF31_nnlo_as_0118 --sqrt-s 13000 \
        --m-lo 40 --m-hi 200 --n-anchor 300 --tag nnpdf31_nnlo_13tev

Stored fields (see ``ZGammaLineshape``):

===============  =========================================================
``log_m``        (n_anchor,) natural log of the anchor masses in GeV
``log_lumi``     (5, n_anchor) natural log of ``L_f`` at those anchors,
                 ordered as ``flavors``
``flavors``      (5,) PDG quark ids [1, 2, 3, 4, 5] = d, u, s, c, b
``provenance``   1-element vlen JSON string: pdf set and member, sqrt(s),
                 factorisation scale, rapidity cut, mass range, the source
                 module and its git-less mtime, and the creation date
===============  =========================================================
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

REF_LINESHAPE_DIR = "/work/submit/david_w/ZMass/calibration_studies/lineshape"

# PDG id, electric charge, weak isospin -- the five active flavours of the
# reference calculation (top is left out, as there).
QUARKS = (
    (1, -1 / 3, -1 / 2),
    (2, 2 / 3, 1 / 2),
    (3, -1 / 3, -1 / 2),
    (4, 2 / 3, 1 / 2),
    (5, -1 / 3, -1 / 2),
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pdfset", default="NNPDF31_nnlo_as_0118")
    p.add_argument("--pdf-member", type=int, default=0)
    p.add_argument("--sqrt-s", type=float, default=13000.0, help="GeV")
    p.add_argument("--m-lo", type=float, default=40.0, help="lowest anchor mass [GeV]")
    p.add_argument(
        "--m-hi", type=float, default=200.0, help="highest anchor mass [GeV]"
    )
    p.add_argument(
        "--n-anchor",
        type=int,
        default=300,
        help="log-spaced anchors; the provider cubic-splines log L "
        "in log m between them",
    )
    p.add_argument(
        "--y-cut",
        type=float,
        default=None,
        help="restrict the boson rapidity to |Y| < y_cut (an "
        "acceptance model). Default: inclusive.",
    )
    p.add_argument(
        "--tag", default=None, help="output basename, written to data/zlumi_<tag>.npz"
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )
    p.add_argument(
        "--lineshape-dir",
        default=REF_LINESHAPE_DIR,
        help="directory holding drell_yan_xsec.py / constants.py",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.lineshape_dir):
        raise SystemExit(
            f"reference lineshape code not found at {args.lineshape_dir}; pass "
            "--lineshape-dir. This script is only needed to regenerate the "
            "shipped table -- the provider itself does not import it."
        )
    sys.path.insert(0, args.lineshape_dir)
    import drell_yan_xsec as dy  # noqa: E402
    import lhapdf  # noqa: E402

    pdf = lhapdf.mkPDF(args.pdfset, args.pdf_member)
    s = args.sqrt_s**2

    m_anchor = np.exp(np.linspace(np.log(args.m_lo), np.log(args.m_hi), args.n_anchor))
    log_lumi = np.empty((len(QUARKS), args.n_anchor))
    for i, (flavor, _, _) in enumerate(QUARKS):
        if args.y_cut is None:
            vals = dy.integrate_sigma_hat_prime_sm(s, flavor, m_anchor**2, pdf)
        else:
            vals = dy.integrate_sigma_hat_prime_sm_Ycut(
                s, flavor, m_anchor**2, pdf, args.y_cut
            )
        vals = np.asarray(vals, dtype=float)
        if not np.all(vals > 0):
            raise RuntimeError(f"non-positive luminosity for flavour {flavor}")
        log_lumi[i] = np.log(vals)
        print(
            f"  flavour {flavor}: L({args.m_lo:.0f}) = {vals[0]:.6e}, "
            f"L(91.2) = {np.exp(np.interp(np.log(91.1876), np.log(m_anchor), log_lumi[i])):.6e}, "
            f"L({args.m_hi:.0f}) = {vals[-1]:.6e}"
        )

    src = os.path.join(args.lineshape_dir, "drell_yan_xsec.py")
    provenance = dict(
        pdfset=args.pdfset,
        pdf_member=args.pdf_member,
        pdf_description=pdf.set().description if hasattr(pdf, "set") else "",
        sqrt_s_gev=args.sqrt_s,
        factorisation_scale="mu_F = Q (the dilepton mass)",
        order="LO parton luminosity (the hard ME it multiplies is LO too)",
        y_cut=args.y_cut,
        acceptance=(
            "none (fully inclusive in rapidity and lepton angles)"
            if args.y_cut is None
            else f"|Y| < {args.y_cut}, no lepton cuts"
        ),
        m_lo=args.m_lo,
        m_hi=args.m_hi,
        n_anchor=args.n_anchor,
        source=src,
        source_mtime=datetime.datetime.fromtimestamp(os.path.getmtime(src)).isoformat(
            timespec="seconds"
        ),
        source_function=(
            "drell_yan_xsec.integrate_sigma_hat_prime_sm"
            if args.y_cut is None
            else "drell_yan_xsec.integrate_sigma_hat_prime_sm_Ycut"
        ),
        generator=os.path.abspath(__file__),
        created=datetime.date.today().isoformat(),
        lhapdf_version=lhapdf.version(),
    )

    tag = args.tag or f"{args.pdfset.lower()}_{int(args.sqrt_s/1000)}tev"
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"zlumi_{tag}.npz")
    np.savez_compressed(
        out,
        log_m=np.log(m_anchor),
        log_lumi=log_lumi,
        flavors=np.array([q[0] for q in QUARKS], dtype=np.int32),
        provenance=np.array([json.dumps(provenance, indent=1)]),
    )
    print(f"wrote {out} ({os.path.getsize(out)/1024:.1f} kiB)")
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
