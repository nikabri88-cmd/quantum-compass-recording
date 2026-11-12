"""Exploratory computations after the MAG-S3c confirmatory run (11 November 2026, UTC). Nothing here changes a verdict.

1. scale   Form K at m = 1 and the control P-0 at small CISS angles, rebuilt exactly from the reference solutions of
           the run archive with the frozen functions of mag_s3c.py (no new solves): the angle of the CISS estimate of
           Kobori et al. (2026), at most 6% for the pair with W_C and about 8% for the pair with W_B, read either as an amplitude
           (sin chi) or as a triplet share (sin^2 chi). The check row at chi = 0.1 pi must equal the run.
2. h3grid  Why the trial run of 10 November (substitute rates, 60 directions) gave H3 not met: form K at
           chi_b = 0.1 pi, m = 1, axis r of W_C and W_D on the levels of H3, rebuilt from reference solutions solved
           anew with the frozen solver on the grid of 60 directions at the Cry4a rates and at the substitute rates,
           and on the grid of 2000 directions at the substitute rates.

Usage (RUN_DIR as for results/tables/make_tables.py; the substitute rates need their own process):
  python3 -I explore_after_run.py scale  RUN_DIR OUT.json
  python3 -I explore_after_run.py h3grid RUN_DIR OUT.json [--blind] [--half N] [--workers W]
"""
import json
import math
import os
import sys
from multiprocessing import Pool

CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
import mag_s3c as c  # noqa: E402

LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6), ("std", 3e7), ("std", 1e8))
H3_LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6))


def chi_values():
    out = {"run check": 0.1}
    for share in (0.06, 0.08):
        out[f"sin chi = {share:g}"] = math.asin(share) / math.pi
        out[f"sin^2 chi = {share:g}"] = math.asin(math.sqrt(share)) / math.pi
    return out


def responses(half, yp, ym):
    row = c.summarize(half, yp, ym, check_fit=False)
    o = row["receptor"]["by_order"]["S1_p1"]
    d = row["d_odd"]
    return dict(F=float(o["eps_eff"]), F_amp=float(o["eps_E6"]), d_odd_z=float(d[2]),
                even_range=float(row["parts"]["even_range"]))


def scale(run, out):
    res = {}
    for g in ("W_C", "W_D"):
        sy = c.build_system(g)
        for rl, k in LEVELS:
            ref = c.Ref(g, rl, k)
            with open(os.path.join(run, "refs", ref.name + ".json")) as fh:
                r = json.load(fh)
            cp, cm, k_r = np.asarray(r["cp"]), np.asarray(r["cm"]), float(r["rates_used"][1])
            half = c.m2.fibonacci_sphere(len(cp))
            for ax in ("z", "x", "r"):
                for label, chi in chi_values().items():
                    for form in ("K", "P-0"):
                        cell = c.Cell(g, ax, rl, k, form, chi, 1.0)
                        yp, ym = c.rebuilt_yields(c.cell_weights(sy, cell), cp, cm, k_r)
                        v = responses(half, yp, ym)
                        v.update(chi_over_pi=chi, reading=label)
                        res[f"{g}_{ax}_{rl}{k:g}_{form}_{label}"] = v
                        print(f"{g:4s} {ax} {rl}{k:<6g} {form:4s} {label:16s} chi={chi:.4f}pi "
                              f"F={v['F']:+.4f} F_amp={v['F_amp']:+.5f} stricter={min(abs(v['F']), abs(v['F_amp'])):.5f}",
                              flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


def _solve_chunk(args):
    g, rl, k, dirs = args
    y, coef, diag = c.ref_solutions(c.Ref(g, rl, k), np.asarray(dirs))
    return coef


def h3grid(run, out, blind, half_n, workers):
    if blind:
        c.use_blind_rates()
    half = c.m2.fibonacci_sphere(half_n)
    res = {}
    with Pool(workers) as pool:
        for g in ("W_C", "W_D"):
            sy = c.build_system(g)
            for rl, k in H3_LEVELS:
                coefs = []
                for sign in (+1, -1):
                    chunks = np.array_split(sign * half, max(1, workers * 4))
                    parts = pool.map(_solve_chunk, [(g, rl, k, ch.tolist()) for ch in chunks])
                    coefs.append(np.vstack(parts))
                cp, cm = coefs
                k_r = c.rates("cry4a")[1]
                cell = c.Cell(g, "r", rl, k, "K", 0.1, 1.0)
                yp, ym = c.rebuilt_yields(c.cell_weights(sy, cell), cp, cm, k_r)
                v = responses(half, yp, ym)
                v.update(rates=list(c.rates("cry4a")), directions=2 * half_n)
                res[f"{g}_r_{rl}{k:g}"] = v
                print(f"{g:4s} r {rl}{k:<6g} rates={c.rates('cry4a')} dirs={2 * half_n} F={v['F']:+.4f} "
                      f"F_amp={v['F_amp']:+.5f} stricter={min(abs(v['F']), abs(v['F_amp'])):.5f}", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    mode, run_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    run = os.path.join(run_dir, "confirmatory_s3c", "run")
    if mode == "scale":
        scale(run, out_path)
    elif mode == "h3grid":
        args = sys.argv[4:]
        h = int(args[args.index("--half") + 1]) if "--half" in args else 30
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 2
        h3grid(run, out_path, "--blind" in args, h, w)
    else:
        sys.exit("mode: scale or h3grid")
