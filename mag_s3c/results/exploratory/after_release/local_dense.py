"""Part B of kobori_and_dephasing.py (dense dephasing) computed on the calibration machine while Colab is unavailable.
Written on 13 November 2026 (UTC) before it was run; the plan is section 3 B of ../README.md, committed before any of
it was computed. Nothing here changes a verdict of MAG-S3c.

run: the references of W_C at dephasing 4e7, 5e7, 6e7 and 8e7 s^-1 together with the check reference of the plan
(W_C at dephasing 3e7 s^-1), then the same four references of W_D; the check reference must equal the archive of the
run to 1e-9 relative, else the summary refuses to read the new references.
summarize: k_half of form K (chi_b = 0.05 pi and 0.1 pi, m = 1) on the axes z, x and r by the rule of the protocol,
the first crossing of half, linear in lg k between adjacent levels (kd.k_half), on the levels of the run alone and
with the dense levels, beside the class of the protocol and |d_odd.z| at 6e7 s^-1 without interpolation, as in the
summary step of the plan. The values on the levels of the run are compared with results/tables/k_half.csv.

Usage:
  python3 -I local_dense.py run OUT
  python3 -I local_dense.py summarize OUT RESULT.json
"""
import csv
import importlib.util
import json
import math
import os
import sys
import tarfile
import time

E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kobori_and_dephasing.py")
spec = importlib.util.spec_from_file_location("kd", E)
kd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kd)
c = kd.c
ARCHIVE = os.path.join(kd.STUDY, "results", "confirmatory_run", "confirmatory_s3c.tar.gz")


def dense(g):
    return [x for x in kd.dense_refs() if x.system == g]


def run(out):
    bad = kd.frozen_check()
    if bad:
        sys.exit("frozen files differ from FREEZE.txt: " + ", ".join(bad))
    os.makedirs(out, exist_ok=True)
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("start", start, "script sha256", c.file_sha256(os.path.abspath(__file__)), flush=True)
    dest = os.path.join(out, "check_reference")
    kd.extract_checks(ARCHIVE, dest)
    t0 = time.time()
    c.pair(out, 2, 0, half_n=c.GRID_HALF, items=kd.check_refs() + dense("W_C"))
    rows = kd.compare(out, dest, kd.check_refs())
    print(json.dumps(rows), flush=True)
    c.pair(out, 2, 0, half_n=c.GRID_HALF, items=dense("W_D"))
    passed = max(r["max_rel_diff"] for r in rows) <= kd.CHECK_TOL
    done = dict(start_utc=start, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                hours=(time.time() - t0) / 3600, checks=rows, passed=passed, n_dense=len(kd.dense_refs()),
                script_sha256=c.file_sha256(os.path.abspath(__file__)))
    c.write_json(os.path.join(out, "local_dense_done.json"), done)
    print("done", f"{done['hours']:.2f} h", "check passed" if passed else "CHECK FAILED", flush=True)
    return 0 if passed else 1


def run_refs():
    return [c.Ref(g, "none", 0.0) for g in ("W_C", "W_D")] + [c.Ref(g, "std", k) for g in ("W_C", "W_D")
                                                                for k in kd.RUN_STD]


def extract_run_refs(dest):
    if c.file_sha256(ARCHIVE) != kd.ARCHIVE_SHA256:
        sys.exit("the archive of the run differs from its published SHA-256")
    want = {f"confirmatory_s3c/run/refs/{x.name}.json" for x in run_refs()}
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in want]
        if len(members) != len(want):
            sys.exit("references missing in the archive of the run")
        tar.extractall(dest, members=members, filter="data")
    return os.path.join(dest, "confirmatory_s3c", "run", "refs")


def band(k):
    if k is None:
        return "above 1e8"
    return "below 3e7" if k < 3e7 else ("3e7 to 6e7" if k <= 6e7 else "above 6e7")


def summarize(out, result):
    done = c.load_json(os.path.join(out, "local_dense_done.json"))
    if not done["passed"]:
        sys.exit("the check reference differs from the run: the dense references are not read")
    run_dir = extract_run_refs(os.path.join(out, "run_refs"))
    table = {(r["geometry"], r["ciss_axis"], float(r["chi_b_over_pi"])): r
             for r in csv.DictReader(open(os.path.join(kd.STUDY, "results", "tables", "k_half.csv")))
             if r["relaxation"] == "std"}
    rows, worst = {}, 0.0
    for g in ("W_C", "W_D"):
        sy = c.build_system(g)
        refs = {0.0: kd.load_ref(os.path.join(run_dir, c.Ref(g, "none", 0.0).name + ".json"))}
        for k in kd.RUN_STD:
            refs[k] = kd.load_ref(os.path.join(run_dir, c.Ref(g, "std", k).name + ".json"))
        for k in kd.DENSE_STD:
            refs[k] = kd.load_ref(os.path.join(out, "refs", c.Ref(g, "std", k).name + ".json"))
        for ax in ("z", "x", "r"):
            for chi_b in (0.05, 0.1):
                lv = []
                for k, ref in sorted(refs.items()):
                    cell = c.Cell(g, ax, "none" if k == 0.0 else "std", k, "K", chi_b, 1.0)
                    lv.append((k, kd.rebuilt(sy, cell, ref)["d_odd_z"]))
                d0 = abs(lv[0][1])
                run_only = [p for p in lv if p[0] == 0.0 or p[0] in kd.RUN_STD]
                k_run, k_dense = kd.k_half(run_only), kd.k_half(lv)
                tab = table[(g, ax, chi_b)]
                if k_run is not None:
                    worst = max(worst, abs(k_run - float(tab["k_half_per_s"])) / float(tab["k_half_per_s"]))
                rows[f"{g}_{ax}_chi{chi_b:g}"] = dict(
                    levels=lv, ratio={f"{k:g}": abs(d) / d0 for k, d in lv}, k_half_run_levels=k_run,
                    k_half_dense=k_dense, class_run_levels=band(k_run), class_dense=band(k_dense),
                    ratio_at_6e7=abs(dict(lv)[6e7]) / d0)
    res = dict(done=done, k_half_csv_largest_rel_diff=worst, dephasing=rows)
    with open(result, "w") as fh:
        json.dump(res, fh, indent=1)
    print("k_half on the levels of the run against k_half.csv: largest relative difference", f"{worst:.1e}")
    for name, r in rows.items():
        kr, kn = r["k_half_run_levels"], r["k_half_dense"]
        print(f"{name:16s} run levels {kr and f'{kr:.3g}'} ({r['class_run_levels']}), dense {kn and f'{kn:.3g}'} "
              f"({r['class_dense']}), |d_odd.z| at 6e7 / at 0: {r['ratio_at_6e7']:.3f}")
    print("written", result)


if __name__ == "__main__":
    mode, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if mode == "run":
        sys.exit(run(args[0]))
    elif mode == "summarize":
        summarize(args[0], args[1])
    else:
        sys.exit("mode: run or summarize")
