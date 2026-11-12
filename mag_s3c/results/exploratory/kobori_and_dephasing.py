"""Exploratory computations after release v0.3.0 (planned on 12 November 2026, UTC, before they were run; run in Google
Colab). Nothing here changes a verdict of MAG-S3c.

A. Kobori et al. (2026, ChemRxiv, doi 10.26434/chemrxiv.15000186/v2) fit time-resolved EPR spectra of the
   cryptochrome-DASH of Xenopus laevis with the singlet-born state cos(chi/2)|S> + sin(chi/2)|T0> (their Eq. 6), the
   CISS efficiency 100 (1 - cos chi), recombination with chi_CR = -chi (Eq. 16) and the CISS axis along the axis of the
   dipolar coupling: chi = 20 deg (6%) for the pair FAD-W_C and 23 deg (8%) for FAD-W_B. Among the forms of MAG-S3c
   this is P-LH (birth at +chi, recombination at -chi); P-same recombines through its birth state. Both are computed
   as direct cells with the frozen code on the grid of the run (2000 directions, Cry4a rates) in W_C and W_D, on the
   axes r (the pair axis), z and x, without relaxation, at 1e6 s^-1 of random fields and of dephasing, and at 3e7 and
   1e8 s^-1 of dephasing: 120 cells.
B. References (the pair born in the singlet) of W_C and W_D at dephasing 4e7, 5e7, 6e7 and 8e7 s^-1, between the
   levels 3e7 and 1e8 of the run, so that k_half of form K and the odd dipole at 6e7 s^-1 need no interpolation over
   the gap 3e7-1e8: 8 references.
Checks. Twelve direct cells (P-LH and P-same at chi = 0.1 pi on the axis r, without relaxation and at 1e6 s^-1 of
both kinds, in W_C and W_D) and one reference (W_C at dephasing 3e7 s^-1) repeat items of the run. The run step
computes the twelve cells first, compares them with the archive of the run and stops before A and B when one of them
differs by more than 1e-9 relative; the reference is compared at the end.

Usage (the frozen files of code/ are checked against FREEZE.txt before anything is computed):
  python3 -I kobori_and_dephasing.py plan
  python3 -I kobori_and_dephasing.py run OUT --archive ARCHIVE.tar.gz [--workers N]
  python3 -I kobori_and_dephasing.py run OUT --test [--workers N]     tiny grid and a few items, no comparison
  python3 -I kobori_and_dephasing.py summarize OUT RUN_DIR RESULT.json
RUN_DIR is the folder confirmatory_s3c/run of the unpacked archive of the run.
"""
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                              # before NumPy loads BLAS
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(STUDY, "code"))
import numpy as np  # noqa: E402
import mag_s3c as c  # noqa: E402

ARCHIVE_SHA256 = "99e6288ba6d583181cb2e18fbf8df5bd00dbd1fce5c8568fbe68dda6f9c8446a"
KOBORI_DEG = (20.0, 23.0)
P_LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6), ("std", 3e7), ("std", 1e8))
DENSE_STD = (4e7, 5e7, 6e7, 8e7)
RUN_STD = (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)            # dephasing levels of the run for W_C and W_D (with E10)
CHECK_TOL = 1e-9
THRESHOLD = 0.01                                        # dipole test in R1 at p = 1 and S = 1


def kobori_cells():
    return [c.Cell(g, ax, rl, k, form, deg / 180.0, 1.0) for g in ("W_C", "W_D") for ax in ("r", "z", "x")
            for rl, k in P_LEVELS for form in ("P-LH", "P-same") for deg in KOBORI_DEG]


def dense_refs():
    return [c.Ref(g, "std", k) for g in ("W_C", "W_D") for k in DENSE_STD]


def check_cells():
    return [c.Cell(g, "r", rl, k, form, 0.1, 1.0) for g in ("W_C", "W_D") for rl, k in P_LEVELS[:3]
            for form in ("P-LH", "P-same")]


def check_refs():
    return [c.Ref("W_C", "std", 3e7)]


def frozen_check():
    """Files of code/ listed in FREEZE.txt whose SHA-256 differs now (empty when all agree)."""
    bad = []
    with open(os.path.join(STUDY, "FREEZE.txt")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64 and parts[1].startswith("code/"):
                path = os.path.join(STUDY, parts[1])
                if not os.path.exists(path) or c.file_sha256(path) != parts[0]:
                    bad.append(parts[1])
    return bad


def sub_of(item):
    return "refs" if isinstance(item, c.Ref) else "cells"


def extract_checks(archive, dest):
    if c.file_sha256(archive) != ARCHIVE_SHA256:
        sys.exit("the archive of the run differs from its published SHA-256")
    want = {f"confirmatory_s3c/run/{sub_of(x)}/{x.name}.json" for x in check_cells() + check_refs()}
    if all(os.path.exists(os.path.join(dest, w)) for w in want):
        return
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in want]
        if len(members) != len(want):
            sys.exit("check items missing in the archive of the run")
        tar.extractall(dest, members=members, filter="data")


def compare(out, dest, items):
    rows = []
    for item in items:
        new = c.load_json(os.path.join(out, sub_of(item), item.name + ".json"))
        old = c.load_json(os.path.join(dest, "confirmatory_s3c", "run", sub_of(item), item.name + ".json"))
        keys = ("yp", "ym", "cp", "cm") if isinstance(item, c.Ref) else ("yp", "ym")
        identical, worst = True, 0.0
        for key in keys:
            a, b = np.asarray(old[key], dtype=float), np.asarray(new[key], dtype=float)
            identical = identical and bool(np.array_equal(a, b))
            worst = max(worst, float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300)))
        rows.append(dict(item=item.name, identical=identical, max_rel_diff=worst))
    return rows


def run(out, archive, workers, test):
    bad = frozen_check()
    if bad:
        sys.exit("frozen files differ from FREEZE.txt: " + ", ".join(bad))
    os.makedirs(out, exist_ok=True)
    half_n = 12 if test else c.GRID_HALF
    start = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), workers=workers, grid_half=half_n,
                 test=test, script_sha256=c.file_sha256(os.path.abspath(__file__)), metadata=c.metadata("after-release"))
    with open(os.path.join(out, f"start_{start['start_utc']}.json"), "w") as fh:
        json.dump(start, fh, indent=1)
    print("frozen files agree with FREEZE.txt; workers", workers, "grid_half", half_n, flush=True)
    if test:
        items = check_cells()[:1] + dense_refs()[:1] + kobori_cells()[:2]
        c.pair(out, workers, 0, half_n=half_n, items=items)
        print("test done:", len(items), "items", flush=True)
        return 0
    dest = os.path.join(out, "check_reference")
    extract_checks(archive, dest)
    t0 = time.time()
    c.pair(out, workers, 0, half_n=half_n, items=check_cells())
    rows = compare(out, dest, check_cells())
    worst = max(r["max_rel_diff"] for r in rows)
    print(f"check cells: {sum(r['identical'] for r in rows)} of {len(rows)} bit-identical, largest relative "
          f"difference {worst:.2e} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if worst > CHECK_TOL:
        c.write_json(os.path.join(out, "checks.json"), dict(cells=rows, passed=False))
        print("CHECK FAILED: A and B not computed", flush=True)
        return 1
    c.pair(out, workers, 0, half_n=half_n, items=check_refs() + dense_refs() + kobori_cells())
    rows += compare(out, dest, check_refs())
    worst = max(r["max_rel_diff"] for r in rows)
    passed = worst <= CHECK_TOL
    c.write_json(os.path.join(out, "checks.json"), dict(items=rows, passed=passed))
    done = dict(end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), hours=(time.time() - t0) / 3600,
                n_kobori_cells=len(kobori_cells()), n_dense_refs=len(dense_refs()), n_checks=len(rows),
                checks_passed=passed, largest_check_difference=worst)
    c.write_json(os.path.join(out, "done.json"), done)
    print(json.dumps(done), flush=True)
    return 0 if passed else 1


# ----------------------------------------------------------------------------------------------------------------
# Summary on the calibration machine after the transfer
# ----------------------------------------------------------------------------------------------------------------
def responses(half, yp, ym):
    row = c.summarize(half, np.asarray(yp), np.asarray(ym), check_fit=False)
    o = row["receptor"]["by_order"]["S1_p1"]
    v = dict(F=float(o["eps_eff"]), F_amp=float(o["eps_E6"]), eps_mol=float(row["parts"]["eps_mol"]),
             d_odd_z=float(row["d_odd"][2]), even_range=float(row["parts"]["even_range"]))
    v["stricter"] = min(abs(v["F"]), abs(v["F_amp"]))
    v["visible"] = v["stricter"] >= THRESHOLD
    return v


def load_ref(path):
    r = c.load_json(path)
    return np.asarray(r["cp"]), np.asarray(r["cm"]), float(r["rates_used"][1])


def rebuilt(sy, cell, ref):
    cp, cm, k_r = ref
    half = c.m2.fibonacci_sphere(len(cp))
    yp, ym = c.rebuilt_yields(c.cell_weights(sy, cell), cp, cm, k_r)
    return responses(half, yp, ym)


def k_half(levels):
    """First rate at which |d| falls to half of its value without relaxation, linear in lg k between levels."""
    pts = sorted(levels)
    d0 = abs(pts[0][1])
    prev = None
    for k, d in pts[1:]:
        r = abs(d) / d0
        if r <= 0.5:
            if prev is None:
                return k
            (k1, r1) = prev
            t = (r1 - 0.5) / (r1 - r)
            return 10 ** (math.log10(k1) + t * (math.log10(k) - math.log10(k1)))
        prev = (k, r)
    return None


def summarize(out, run_dir, result):
    res = dict(checks=c.load_json(os.path.join(out, "checks.json")), done=c.load_json(os.path.join(out, "done.json")))
    half = c.m2.fibonacci_sphere(c.GRID_HALF)
    direct = {}
    for cell in kobori_cells() + check_cells():
        d = c.load_json(os.path.join(out, "cells", cell.name + ".json"))
        assert d["grid_half"] == c.GRID_HALF and d["mag_s3c_sha256"] == c.file_sha256(os.path.join(STUDY, "code", "mag_s3c.py"))
        v = responses(half, d["yp"], d["ym"])
        v.update(chi_deg=cell.chi * 180.0, seconds=d["seconds"], diagnostics=d["diagnostics"])
        direct[cell.name] = v
    res["direct"] = direct
    rb = {}
    for g in ("W_C", "W_D"):
        sy = c.build_system(g)
        for rl, k in P_LEVELS:
            ref = load_ref(os.path.join(run_dir, "refs", c.Ref(g, rl, k).name + ".json"))
            for ax in ("r", "z", "x"):
                for deg in KOBORI_DEG:
                    for form, chi in (("K", deg / 360.0), ("P-0", deg / 180.0)):
                        cell = c.Cell(g, ax, rl, k, form, chi, 1.0)
                        v = rebuilt(sy, cell, ref)
                        v.update(chi_deg=chi * 180.0, kobori_deg=deg)
                        rb[cell.name] = v
    res["rebuilt"] = rb
    deph = {}
    for g in ("W_C", "W_D"):
        sy = c.build_system(g)
        refs = {0.0: load_ref(os.path.join(run_dir, "refs", c.Ref(g, "none", 0.0).name + ".json"))}
        for k in RUN_STD:
            refs[k] = load_ref(os.path.join(run_dir, "refs", c.Ref(g, "std", k).name + ".json"))
        for k in DENSE_STD:
            refs[k] = load_ref(os.path.join(out, "refs", c.Ref(g, "std", k).name + ".json"))
        for ax in ("z", "x", "r"):
            for chi_b in (0.05, 0.1):
                lv = []
                for k, ref in sorted(refs.items()):
                    cell = c.Cell(g, ax, "none" if k == 0.0 else "std", k, "K", chi_b, 1.0)
                    lv.append((k, rebuilt(sy, cell, ref)["d_odd_z"]))
                d0 = abs(lv[0][1])
                run_only = [p for p in lv if p[0] == 0.0 or p[0] in RUN_STD]
                deph[f"{g}_{ax}_chi{chi_b:g}"] = dict(
                    levels=lv, ratio={f"{k:g}": abs(d) / d0 for k, d in lv},
                    k_half_run_levels=k_half(run_only), k_half_dense=k_half(lv))
    res["dephasing"] = deph
    with open(result, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", result)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default  # noqa: E731
    if mode == "plan":
        for item in check_cells() + check_refs() + kobori_cells() + dense_refs():
            print(item.name)
        print(len(check_cells()) + len(check_refs()), "check items,", len(kobori_cells()), "cells of A,",
              len(dense_refs()), "references of B")
    elif mode == "run":
        sys.exit(run(args[0], opt("--archive"), int(opt("--workers", os.cpu_count())), "--test" in args))
    elif mode == "summarize":
        summarize(args[0], args[1], args[2])
    else:
        sys.exit("mode: plan, run or summarize")
