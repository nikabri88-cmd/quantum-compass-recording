"""Summary of the core cells at the angles of Kobori et al. on the axis r, with the cells of the run at 18 degrees and
form K and P-0 rebuilt from the references of the run."""
import importlib.util, os, json, csv, sys
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kobori_and_dephasing.py")
spec = importlib.util.spec_from_file_location("kd", E); kd = importlib.util.module_from_spec(spec); spec.loader.exec_module(kd)
c, np = kd.c, kd.np
D, RUN = sys.argv[1], sys.argv[2]   # usage: python3 -I core_summary.py DIR_WITH_local_core_out RUN_DIR (confirmatory_s3c/run)
half = c.m2.fibonacci_sphere(c.GRID_HALF)
LV = (("none", 0.0), ("rf", 1e6), ("std", 1e6), ("std", 3e7))
res = {}
for g in ("W_C", "W_D"):
    sy = c.build_system(g)
    for rl, k in LV:
        ref = kd.load_ref(os.path.join(RUN, "refs", c.Ref(g, rl, k).name + ".json"))
        for form in ("P-LH", "P-same"):
            for chi in (0.1, 20 / 180, 23 / 180):
                cell = c.Cell(g, "r", rl, k, form, chi, 1.0)
                path = os.path.join(D, "local_core_out", "cells", cell.name + ".json")
                if not os.path.exists(path):
                    path = os.path.join(RUN, "cells", cell.name + ".json")
                d = c.load_json(path)
                assert d["grid_half"] == c.GRID_HALF
                res[(g, rl + f"{k:g}", form, round(chi * 180, 3))] = kd.responses(half, d["yp"], d["ym"])
        for deg in (18.0, 20.0, 23.0):
            for form, chi in (("K", deg / 360), ("P-0", deg / 180)):
                res[(g, rl + f"{k:g}", form, deg)] = kd.rebuilt(sy, c.Cell(g, "r", rl, k, form, chi, 1.0), ref)
out = {"|".join(map(str, k)): v for k, v in res.items()}
json.dump(out, open(os.path.join(D, "core_summary.json"), "w"), indent=1)
for g in ("W_C", "W_D"):
    for form in ("P-LH", "P-same", "K", "P-0"):
        for deg in (18.0, 20.0, 23.0):
            vals = [res[(g, rl + f"{k:g}", form, deg)] for rl, k in LV]
            s = " ".join(f"{rl}{k:g}:{v['stricter']:.4f}(F={v['F']:+.3f},Fa={v['F_amp']:+.4f})" for (rl, k), v in zip(LV, vals))
            print(g, form, f"{deg:g}deg", s)
