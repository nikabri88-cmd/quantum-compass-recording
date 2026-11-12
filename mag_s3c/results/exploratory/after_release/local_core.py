"""Core of part A of kobori_and_dephasing.py computed on the calibration machine while Colab is unavailable: P-LH and
P-same at 20 and 23 degrees on the axis r of W_C and W_D, without relaxation, at 1e6 s^-1 of both kinds and at 3e7 s^-1
of dephasing (32 cells), with the four check cells at chi = 0.1 pi without relaxation compared with the run first."""
import importlib.util, os, sys, json, time
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kobori_and_dephasing.py")
spec = importlib.util.spec_from_file_location("kd", E); kd = importlib.util.module_from_spec(spec); spec.loader.exec_module(kd)
c = kd.c
out = sys.argv[1]   # usage: python3 -I local_core.py OUT
bad = kd.frozen_check()
assert not bad, bad
dest = os.path.join(out, "check_reference")
kd.extract_checks(os.path.join(kd.STUDY, "results", "confirmatory_run", "confirmatory_s3c.tar.gz"), dest)
checks = [x for x in kd.check_cells() if x.relax == "none"]
t0 = time.time()
c.pair(out, 2, 0, half_n=c.GRID_HALF, items=checks)
rows = kd.compare(out, dest, checks)
print(json.dumps(rows), flush=True)
assert max(r["max_rel_diff"] for r in rows) <= kd.CHECK_TOL, "check failed"
core = [x for x in kd.kobori_cells() if x.axis == "r" and not (x.relax == "std" and x.k == 1e8)]
print(len(core), "core cells", flush=True)
c.pair(out, 2, 0, half_n=c.GRID_HALF, items=core)
json.dump(dict(checks=rows, n_core=len(core), hours=(time.time() - t0) / 3600,
               end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(os.path.join(out, "local_done.json"), "w"), indent=1)
print("done", (time.time() - t0) / 3600, "h", flush=True)
