"""Exploratory computation after release v0.3.2, planned on 12 November 2026 (UTC) before it was run. Nothing here
changes a verdict of MAG-S3c.

The main grid of MAG-S3c applies the Cry4a rates (k_F = 5e6 s^-1, k_R = 3.4e5 s^-1) to both geometries. The value
k_R = 3.4e5 s^-1 is the estimate of Xu et al. (2021) for the pair FAD-W_D; for FAD-W_C they estimate about 1.2e7 s^-1
(Wong et al. 2021, J. R. Soc. Interface 18: 20210601, second parameter set). This step repeats the cells of note 1 of
the notes after release in W_C on the axis r at chi = 20 deg, the angle of Kobori et al. (2026), with k_R = 1.2e7 s^-1
and k_F = 5e6 s^-1 unchanged:
  direct cells P-LH and P-same without relaxation and at 1e6 s^-1 of random fields and of dephasing (6 cells);
  references of W_C at the same rates and levels (3), from which form K (chi_b = chi / 2, the same triplet
  population) and P-0 are rebuilt as in note 1.
The new rate set is added to the dictionary S3C_RATES of the frozen module at run time; the frozen files are not
changed. Readout as in note 1: the stricter of F_amp and F at p = 1 and S = 1 against the dipole threshold 0.01; both
transfers are reported.
Checks: the frozen files agree with FREEZE.txt; the cells P-LH and P-same at chi = 0.1 pi on the axis r of W_C without
relaxation at the Cry4a rates are recomputed first and must equal the archive of the run bit for bit, else nothing
else is computed.

Usage:
  python3 -I wc_rate.py test OUT          tiny grid, one cell and one reference: checks that the rate set reaches the solver
  python3 -I wc_rate.py run OUT [--workers N]
  python3 -I wc_rate.py summarize OUT
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("kd", os.path.join(HERE, "..", "kobori_and_dephasing.py"))
kd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kd)
c, np = kd.c, kd.np

RATE_NAME = "wc_xu"
RATES = (5.0e6, 1.2e7)                                   # (k_F, k_R) in s^-1
c.S3C_RATES[RATE_NAME] = RATES                           # run-time addition; the frozen file is unchanged
LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6))
CHI = 20.0 / 180.0                                       # units of pi


def cells():
    return [c.Cell("W_C", "r", rl, k, form, CHI, 1.0, RATE_NAME) for rl, k in LEVELS for form in ("P-LH", "P-same")]


def refs():
    return [c.Ref("W_C", rl, k, RATE_NAME) for rl, k in LEVELS]


def checks():
    return [x for x in kd.check_cells() if x.system == "W_C" and x.relax == "none"]


def stamp(out, name, **extra):
    data = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                script_sha256=c.file_sha256(os.path.abspath(__file__)), rates={RATE_NAME: list(RATES)}, **extra)
    with open(os.path.join(out, name), "w") as fh:
        json.dump(data, fh, indent=1)


def test(out):
    bad = kd.frozen_check()
    if bad:
        sys.exit("frozen files differ from FREEZE.txt: " + ", ".join(bad))
    os.makedirs(out, exist_ok=True)
    items = [refs()[0], cells()[0]]
    c.pair(out, 2, 0, half_n=12, items=items)
    for item in items:
        d = c.load_json(os.path.join(out, kd.sub_of(item), item.name + ".json"))
        print(item.name, "rates_used", d.get("rates_used"))
        assert [float(x) for x in d["rates_used"]] == list(RATES), "the rate set did not reach the solver"
    print("test passed")


def run(out, workers):
    bad = kd.frozen_check()
    if bad:
        sys.exit("frozen files differ from FREEZE.txt: " + ", ".join(bad))
    os.makedirs(out, exist_ok=True)
    stamp(out, "start.json", workers=workers, grid_half=c.GRID_HALF)
    dest = os.path.join(out, "check_reference")
    kd.extract_checks(os.path.join(kd.STUDY, "results", "confirmatory_run", "confirmatory_s3c.tar.gz"), dest)
    t0 = time.time()
    c.pair(out, workers, 0, half_n=c.GRID_HALF, items=checks())
    rows = kd.compare(out, dest, checks())
    print(json.dumps(rows), flush=True)
    if not all(r["identical"] for r in rows):
        stamp(out, "checks_failed.json", rows=rows)
        sys.exit("CHECK FAILED: nothing else computed")
    c.pair(out, workers, 0, half_n=c.GRID_HALF, items=refs() + cells())
    stamp(out, "done.json", checks=rows, hours=(time.time() - t0) / 3600)
    print("done", f"{(time.time() - t0) / 3600:.2f} h", flush=True)


def summarize(out):
    half = c.m2.fibonacci_sphere(c.GRID_HALF)
    sy = c.build_system("W_C")
    frozen = c.file_sha256(os.path.join(kd.STUDY, "code", "mag_s3c.py"))
    res = {}
    for rl, k in LEVELS:
        ref_path = os.path.join(out, "refs", c.Ref("W_C", rl, k, RATE_NAME).name + ".json")
        ref = kd.load_ref(ref_path)
        assert abs(ref[2] - RATES[1]) <= 1e-9 * RATES[1]
        for form in ("P-LH", "P-same"):
            d = c.load_json(os.path.join(out, "cells", c.Cell("W_C", "r", rl, k, form, CHI, 1.0, RATE_NAME).name + ".json"))
            assert d["grid_half"] == c.GRID_HALF and d["mag_s3c_sha256"] == frozen
            assert [float(x) for x in d["rates_used"]] == list(RATES)
            res[f"{form}|{rl}{k:g}"] = kd.responses(half, d["yp"], d["ym"])
        for form, chi in (("K", CHI / 2), ("P-0", CHI)):
            res[f"{form}|{rl}{k:g}"] = kd.rebuilt(sy, c.Cell("W_C", "r", rl, k, form, chi, 1.0, RATE_NAME), ref)
    with open(os.path.join(out, "wc_rate_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for form in ("P-LH", "P-same", "K", "P-0"):
        vals = [res[f"{form}|{rl}{k:g}"] for rl, k in LEVELS]
        print(form, " ".join(f"{rl}{k:g}: stricter {v['stricter']:.4f} (F {v['F']:+.4f}, F_amp {v['F_amp']:+.4f}, "
                             f"eps_mol {v['eps_mol']:.3f})" for (rl, k), v in zip(LEVELS, vals)))


if __name__ == "__main__":
    mode, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if mode == "test":
        test(args[0])
    elif mode == "run":
        run(args[0], int(args[args.index("--workers") + 1]) if "--workers" in args else (os.cpu_count() or 1))
    elif mode == "summarize":
        summarize(args[0])
    else:
        sys.exit("mode: test, run or summarize")
