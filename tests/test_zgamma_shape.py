#!/usr/bin/env python3
"""The floated smooth K(m) of `ZGammaLineshape` (the LO -> MiNNLO shape).

The provider is LO in both the hard matrix element and the parton luminosity
while the samples are POWHEG MiNNLO, and the generated/model ratio runs 1.7 at
55 GeV -> 1.0 at the peak -> 1.2 at 150 GeV. Fitting the pre-FSR spectrum
without a smooth multiplicative shape gives `Gamma_Z` +75.8 MeV; five Legendre
terms close it to +1.1 MeV (`zchannel/README.md`, "Step 3"). This is the same
construction as `zchannel/fit_gen.py` `GenFit(shape=N)`, moved into the
provider so it applies to the Born spectrum BEFORE the FSR fold and so
`MassCFTerm` inherits it.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rabbit.lineshapes import ZGammaLineshape, make_provider  # noqa: E402

DT = tf.float64


def main():
    ok = True
    common = dict(window=(60.0, 130.0), nm=2048, tau_max=20.0)
    z0 = ZGammaLineshape(**common)
    z5 = ZGammaLineshape(shape=5, shape_window=(60.0, 120.0), **common)

    # ---- 1. names ---------------------------------------------------------
    p = tuple(z5.param_names) == ("m_Z", "Gamma_Z", "shape1", "shape2",
                                  "shape3", "shape4", "shape5")
    ok &= p
    print(f"1. param_names                  {'PASS' if p else 'FAIL'}   "
          f"{z5.param_names}")

    # ---- 2. c = 0 reproduces the unshaped provider bit for bit -------------
    v0 = {"m_Z": tf.constant(0.0, DT), "Gamma_Z": tf.constant(0.0, DT)}
    v5 = dict(v0, **{f"shape{k}": tf.constant(0.0, DT) for k in range(1, 6)})
    a = z0.pdf(v0).numpy()
    b = z5.pdf(v5).numpy()
    d = np.max(np.abs(a - b))
    p = d == 0.0
    ok &= p
    print(f"2. c = 0 is bit-identical       {'PASS' if p else 'FAIL'}   "
          f"max |d| {d:.3e}")

    # ---- 3. the factor is exp(sum c_k P_k) on the BORN grid ----------------
    from numpy.polynomial import legendre
    c = np.array([0.11, -0.07, 0.05, -0.02, 0.013])
    vc = dict(v0, **{f"shape{k}": tf.constant(c[k - 1], DT)
                     for k in range(1, 6)})
    y0 = z0.born_pdf(v0).numpy()
    yc = z5.born_pdf(vc).numpy()
    lo, hi = 60.0, 120.0
    uu = 2.0 * (z5.m_born - lo) / (hi - lo) - 1.0
    ref = np.exp(np.clip(sum(c[k - 1] * legendre.legval(uu, [0] * k + [1])
                             for k in range(1, 6)), -30.0, 30.0))
    d = np.max(np.abs(yc / np.maximum(y0, 1e-300) - ref))
    p = d < 1e-11
    ok &= p
    print(f"3. factor == exp(sum c_k P_k)   {'PASS' if p else 'FAIL'}   "
          f"max |d| {d:.3e}")

    # ---- 4. the shaped pdf is still normalised -----------------------------
    s = float(np.sum(z5.pdf(vc).numpy()) * z5.dm)
    p = abs(s - 1.0) < 1e-12
    ok &= p
    print(f"4. pdf still normalised         {'PASS' if p else 'FAIL'}   "
          f"sum p dm - 1 = {s-1:.3e}")

    # ---- 5. gradient w.r.t. a shape coefficient ---------------------------
    v = tf.Variable(0.11, dtype=DT)
    with tf.GradientTape() as t:
        vv = dict(v0, shape1=v, shape2=tf.constant(0.0, DT),
                  shape3=tf.constant(0.0, DT), shape4=tf.constant(0.0, DT),
                  shape5=tf.constant(0.0, DT))
        y = tf.reduce_sum(z5.pdf(vv) * tf.constant(z5.m_grid, DT))
    g = float(t.gradient(y, v).numpy())
    h = 1e-6

    def f(x):
        vv = dict(v0, shape1=tf.constant(x, DT),
                  **{f"shape{k}": tf.constant(0.0, DT) for k in range(2, 6)})
        return float(np.sum(z5.pdf(vv).numpy() * z5.m_grid))

    fd = (f(0.11 + h) - f(0.11 - h)) / (2 * h)
    d = abs(fd - g) / max(abs(fd), 1e-12)
    p = d < 1e-6
    ok &= p
    print(f"5. gradient vs FD               {'PASS' if p else 'FAIL'}   "
          f"rel {d:.3e}  (analytic {g:.6g}, FD {fd:.6g})")

    # ---- 6. config round trip ---------------------------------------------
    cfg = json.loads(json.dumps(z5.config()))
    z6 = make_provider(cfg)
    d = max(abs(np.max(np.abs(z6.pdf(vc).numpy() - z5.pdf(vc).numpy()))),
            0.0)
    p = (tuple(z6.param_names) == tuple(z5.param_names)
         and z6.config() == z5.config() and d == 0.0)
    ok &= p
    print(f"6. config round trip            {'PASS' if p else 'FAIL'}   "
          f"max |d| {d:.3e}")

    # ---- 7. declarations ---------------------------------------------------
    dec = z5.param_declarations(gz_prior=2.3, shape_prior=None)
    p = (all(f"shape{k}" in dec for k in range(1, 6))
         and all(dec[f"shape{k}"][3] == 0 for k in range(1, 6))
         and dec["Gamma_Z"][1] == 2.3 and dec["Gamma_Z"][3] == 1)
    ok &= p
    print(f"7. declarations                 {'PASS' if p else 'FAIL'}   "
          f"shape1 {dec['shape1']}")

    print("ALL TESTS PASSED" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
