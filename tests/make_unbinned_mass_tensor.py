#!/usr/bin/env python3
"""Turn the CVH mass-likelihood ``.npz`` caches into a rabbit input tensor.

The step-1 caches produced by ``cf_masspairs`` / ``cf_masskernel`` (see
``/work/submit/david_w/ZMass/calibration_studies/resolution/``) hold, per
J/psi (or B -> J/psi K) candidate, everything the unbinned mass likelihood
needs:

``pairs cache``
    ``z`` (pull), ``sigma`` (mass resolution), ``eta`` (gen mass), ``vgf``
    (Gaussian/hit variance fraction), the tabulated resolution-CF exponents
    ``Sms``, ``Sio_re``, ``Sio_im`` (and any further ``S<f>_re`` / ``S<f>_im``
    families, e.g. radiative), and the standardized quadrature grid ``tgrid``.
``kernel cache``
    ``dm``, the FSR kernel samples whose empirical characteristic function
    multiplies every candidate's CF.

This script converts them into a rabbit datacard holding one
:class:`rabbit.unbinned.MassCFTerm` (plus a 1-bin dummy channel so that the
binned part of ``FitInputData`` is satisfied) and the parameter declarations
that ``--paramModel UnbinnedParams`` picks up.

The family list is *data driven*: every ``S<name>`` / ``S<name>_re`` /
``S<name>_im`` array in the pairs cache becomes a family with its own scale
parameter, so a cache that gains a radiative family needs no code change
here. ``--model r`` points every family at a single shared scale ``r``;
``--model families`` gives each its own ``k_<name>``.

Usage::

    python tests/make_unbinned_mass_tensor.py \\
        --pairs-cache  runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz \\
        --kernel-cache runs/cf_masskernel_jpsigun_ul16_260902_m0.npz \\
        --model families -o /scratch/jpsigun_families.hdf5
"""

import argparse
import os
import time

import numpy as np

from rabbit import tensorwriter, unbinned

MJPSI = 3.0969
MWIN = 0.7  # width of the mass window carrying the uniform combinatoric floor
FBKG = 0.005
# analytic (Gaussian / hit) family; it needs only vgf, not an (n, nt) array
GAUSS_FAMILY = "hit"
# canonical ordering of the tabulated families, so the summation order of the
# exponent matches the reference implementation; unknown families follow in
# alphabetical order.
FAMILY_ORDER = ["ms", "ioni", "rad"]
# cache key -> family label used for the parameter name (k_<label>)
FAMILY_ALIAS = {"io": "ioni"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pairs-cache", required=True, help="cf_masspairs_*.npz")
    p.add_argument("--kernel-cache", required=True, help="cf_masskernel_*.npz")
    p.add_argument("-o", "--output", required=True, help="output .hdf5")
    p.add_argument(
        "--model",
        choices=["r", "families"],
        default="families",
        help="'r': one shared resolution scale for all families; "
        "'families': one scale per family",
    )
    p.add_argument("--name", default="mass", help="name of the unbinned term")
    p.add_argument("--channel", default="jpsi", help="channel label of the term")
    p.add_argument(
        "--float-bkg", action="store_true", help="float the background fraction"
    )
    p.add_argument("--fbkg", type=float, default=FBKG, help="background fraction")
    p.add_argument(
        "--background",
        choices=["uniform", "bernstein"],
        default="uniform",
        help="background pdf shape on the mass window",
    )
    p.add_argument(
        "--bernstein-degree", type=int, default=1, help="Bernstein polynomial degree"
    )
    p.add_argument("--mref", type=float, default=MJPSI, help="reference mass [GeV]")
    p.add_argument("--window", type=float, default=MWIN, help="mass window [GeV]")
    p.add_argument(
        "--maxn", type=int, default=0, help="use only the first N candidates"
    )
    p.add_argument(
        "--maxk",
        type=int,
        default=0,
        help="use only the first N kernel samples (tests: the empirical kernel "
        "CF costs len(dm) x --phik-points complex exponentials)",
    )
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--floor", choices=["softplus", "clip", "none"], default="softplus")
    p.add_argument("--floor-scale", type=float, default=unbinned.FLOOR_SCALE)
    p.add_argument(
        "--alpha0", type=float, default=0.2, help="starting value of alpha [1e-3]"
    )
    p.add_argument("--k0", type=float, default=1.0, help="starting resolution scale")
    p.add_argument(
        "--prior",
        action="append",
        default=[],
        metavar="NAME:MEAN:SIGMA",
        help="Gaussian prior on a parameter, e.g. --prior k_ms:1.0:0.005. "
        "Applied by the Fitter through the ParamModel prior mechanism "
        "(0.5 ((p - mean) / sigma)^2). Repeatable.",
    )
    p.add_argument(
        "--poi",
        default="alpha",
        help="comma separated parameters reported as POIs (default: alpha)",
    )
    p.add_argument(
        "--phik-points", type=int, default=8192, help="kernel-CF tabulation points"
    )
    p.add_argument(
        "--phik-cache",
        default=None,
        help="write (and reuse) the kernel CF tabulation here. The "
        "reference implementation's cache next to the kernel cache is always "
        "*read* when it matches, but never written to.",
    )
    return p.parse_args()


def discover_families(keys):
    """Map the pairs-cache keys to ``{family: {"re": key, "im": key}}``.

    ``Sms`` -> family "ms", real part only. ``Sio_re`` / ``Sio_im`` -> family
    "io", both parts. Anything matching the same pattern is picked up
    automatically, which is what keeps a new (e.g. radiative) family from
    needing a code change.
    """
    fams = {}
    for k in keys:
        if not k.startswith("S"):
            continue
        body = k[1:]
        if body.endswith("_re") or body.endswith("_im"):
            fam, comp = body[:-3], body[-2:]
        else:
            fam, comp = body, "re"
        fams.setdefault(FAMILY_ALIAS.get(fam, fam), {})[comp] = k
    return fams


def family_sort_key(name):
    if name in FAMILY_ORDER:
        return (0, FAMILY_ORDER.index(name), name)
    return (1, 0, name)


def build_phik_table(dm, tmax, npoints, read=(), write=None, log=print):
    """Empirical CF of the kernel samples, tabulated on ``[0, tmax]``.

    Identical to the reference implementation: ``mean_j exp(i t dm_j)``
    evaluated in row blocks (block size does not change the axis-1 mean, so
    the result is bit-identical for any blocking).
    """
    tabs = np.linspace(0.0, tmax, npoints)
    for cache in read:
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if len(z["tabs"]) == npoints and np.allclose(z["tabs"], tabs):
                log(f"kernel CF tabulation from cache {cache}")
                return tabs, z["phiK_tab"]
    t0 = time.time()
    blk = 1024
    tab = np.concatenate(
        [
            np.mean(np.exp(1j * np.outer(tabs[i : i + blk], dm)), axis=1)
            for i in range(0, npoints, blk)
        ]
    )
    log(f"kernel CF tabulation built in {time.time()-t0:.1f} s")
    if write:
        try:
            np.savez(write, tabs=tabs, phiK_tab=tab)
            log(f"kernel CF tabulation cached -> {write}")
        except OSError as e:
            log(f"kernel CF cache write failed: {e}")
    return tabs, tab


def main():
    args = parse_args()
    log = print

    d = np.load(args.pairs_cache)
    k = np.load(args.kernel_cache)
    dm = k["dm"]
    if args.maxk and args.maxk < len(dm):
        dm = dm[: args.maxk]
    tgrid = np.asarray(d["tgrid"], dtype=np.float64)

    z = d["z"].astype(np.float64)
    sigma = d["sigma"].astype(np.float64)
    eta = d["eta"].astype(np.float64)
    vgf = d["vgf"].astype(np.float64)
    n = len(z)
    if args.maxn and args.maxn < n:
        n = args.maxn
    z, sigma, eta, vgf = z[:n], sigma[:n], eta[:n], vgf[:n]
    mobs = z * sigma + (eta - args.mref)
    log(f"{n} candidates, {len(dm)} kernel samples, {len(tgrid)} t points")

    # --- families -------------------------------------------------------
    found = discover_families(d.files)
    fam_names = sorted(found, key=family_sort_key)
    log(f"families in the cache: {fam_names}")

    families = [{"name": GAUSS_FAMILY, "param": None, "kind": "gauss"}]
    datasets = {"sigma": sigma, "mobs": mobs, "vgf": vgf, "tgrid": tgrid}
    for fam in fam_names:
        families.append({"name": fam, "param": None, "kind": "tab"})
        for comp, key in sorted(found[fam].items()):
            datasets[f"S_{comp}_{fam}"] = d[key][:n]

    if args.model == "r":
        for f in families:
            f["param"] = "r"
    else:
        for f in families:
            f["param"] = f"k_{f['name']}"

    # --- kernel CF ------------------------------------------------------
    tmax = tgrid[-1] / sigma.min()
    # the reference implementation caches the same table next to the kernel
    # cache under this name; read it when it matches (it is bit-identical),
    # but never write into that tree.
    ref_cache = os.path.join(
        os.path.dirname(os.path.abspath(args.kernel_cache)),
        "masslikfit_phiKtab_%s_%d_%.12e.npz"
        % (os.path.basename(args.kernel_cache).replace(".npz", ""), len(dm), tmax),
    )
    tabs, phik = build_phik_table(
        dm,
        tmax,
        args.phik_points,
        read=(args.phik_cache, ref_cache),
        write=args.phik_cache,
        log=log,
    )
    datasets["phik_t"] = tabs
    datasets["phik_re"] = phik.real.copy()
    datasets["phik_im"] = phik.imag.copy()

    # --- background -----------------------------------------------------
    window = (args.mref - 0.5 * args.window, args.mref + 0.5 * args.window)
    if args.background == "uniform":
        background = unbinned.UniformBackground(window)
    else:
        background = unbinned.BernsteinBackground(
            window, [f"bkg_c{i}" for i in range(args.bernstein_degree + 1)]
        )

    # --- the term -------------------------------------------------------
    # Built here (with the arrays) so that it can describe itself: config(),
    # the parameter list and their order all come from the term itself, which
    # is what the reader reconstructs.
    term = unbinned.MassCFTerm(
        args.name,
        sigma=sigma,
        mobs=mobs,
        tgrid=tgrid,
        families=[
            dict(
                f,
                **{
                    c: datasets[f"S_{c}_{f['name']}"]
                    for c in ("re", "im")
                    if f"S_{c}_{f['name']}" in datasets
                },
            )
            for f in families
        ],
        vgf=vgf,
        # the kernel CF is *not* interpolated here: only config() and
        # param_names are needed from the term at write time, and the
        # interpolation onto (n, nt) costs ~2 GB and ~10 s for nothing.
        background=background,
        m_ref=args.mref,
        scale_param="alpha",
        bkg_frac_param="f_bkg" if args.float_bkg else None,
        bkg_frac=args.fbkg,
        floor=args.floor,
        floor_scale=args.floor_scale,
        chunk=args.chunk,
        channel=args.channel,
    )

    priors = {}
    for spec in args.prior:
        name, mean, sigma = spec.split(":")
        priors[name] = (float(mean), float(sigma))
    unknown = set(priors) - set(term.param_names)
    if unknown:
        raise ValueError(f"--prior for unknown parameter(s) {sorted(unknown)}; "
                         f"the term has {term.param_names}")
    poi_names = [s for s in args.poi.split(",") if s]

    defaults = []
    is_poi = []
    for p in term.param_names:
        if p == "alpha":
            defaults.append(args.alpha0)
            is_poi.append(1)
        elif p == "f_bkg":
            defaults.append(args.fbkg / unbinned.FBKG_UNIT)
            is_poi.append(0)
        elif p.startswith("bkg_c"):
            defaults.append(float(np.log(np.expm1(1.0))))  # softplus^-1(1)
            is_poi.append(0)
        else:
            defaults.append(args.k0)
            is_poi.append(0)
    is_poi = [1 if p in poi_names else 0 for p in term.param_names]
    sigmas = [priors.get(p, (0.0, np.nan))[1] for p in term.param_names]
    means = [priors.get(p, (d, 0.0))[0] for p, d in zip(term.param_names, defaults)]
    log(f"parameters {term.param_names} defaults {defaults} poi {is_poi}")
    if priors:
        log(f"priors {priors}")

    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name=f"{args.channel}_dummy")
    writer.add_unbinned_term(
        args.name,
        term.config(),
        term.param_names,
        datasets,
        param_defaults=defaults,
        param_prior_sigmas=sigmas,
        param_prior_means=means,
        param_is_poi=is_poi,
    )

    outfolder = os.path.dirname(os.path.abspath(args.output)) or "."
    outname = os.path.basename(args.output)
    if outname.endswith(".hdf5"):
        outname = outname[: -len(".hdf5")]
    os.makedirs(outfolder, exist_ok=True)
    t0 = time.time()
    writer.write(outfolder=outfolder, outfilename=outname)
    log(f"wrote {os.path.join(outfolder, outname)} in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
