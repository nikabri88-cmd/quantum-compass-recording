"""Mutation testing of analyze.py against test_analyze.py (author's harness, not part of the study).
Every mutation changes one rule of the analysis; the synthetic tests should fail on each.
Usage: python3 -I scripts/mutate_analyze.py CODE_DIR OUT_DIR"""
import os
import shutil
import subprocess
import sys

CODE, OUT = sys.argv[1], sys.argv[2]
SRC = open(os.path.join(CODE, "analyze.py")).read()

M = [
    ("needed_floor", "return math.ceil(share * n - 1e-9)", "return math.floor(share * n)"),
    ("cp_level_090", "def cp_two(x, n, level=0.95):", "def cp_two(x, n, level=0.90):"),
    ("cp_upper_090", "def cp_upper_one(x, n, level=0.95):", "def cp_upper_one(x, n, level=0.90):"),
    ("cp_upper_two_sided", "return 1.0 if x == n else float(stats.beta.ppf(level, x + 1, n - x))",
     "return 1.0 if x == n else float(stats.beta.ppf(1 - (1 - level) / 2, x + 1, n - x))"),
    ("paired_lower", "below_zero=bool(hi < 0.5))", "below_zero=bool(lo < 0.5))"),
    ("paired_swap", "n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))", "n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))"),
    ("paired_no_scale", "    s = nd / n\n", "    s = 1.0\n"),
    ("gate_07", "    need = frac_needed(0.8, n)\n    return count(recs, lambda r: shape(r, \"rp2\") == \"rp2\") >= need and \\",
     "    need = frac_needed(0.7, n)\n    return count(recs, lambda r: shape(r, \"rp2\") == \"rp2\") >= need and \\"),
    ("gate_or", "    return count(recs, lambda r: shape(r, \"rp2\") == \"rp2\") >= need and \\",
     "    return count(recs, lambda r: shape(r, \"rp2\") == \"rp2\") >= need or \\"),
    ("I2_tol", "TOL_I2 = 1e-12", "TOL_I2 = 1e-11"),
    ("I2_skip_mags2", 'vals = (v.get("I2_ren_max_diff"), v.get("I2_mags2_max_diff"), v.get("I2_polar_max_sum"))',
     'vals = (v.get("I2_ren_max_diff"), 0.0, v.get("I2_polar_max_sum"))'),
    ("I2_nan_ok", "if not all(is_num(x) and x <= TOL_I2 for x in vals):", "if not all((not is_num(x)) or x <= TOL_I2 for x in vals):"),
    ("I4_skip_means", 'if not (v.get("I4_same_means") is True and v.get("I4_sessions_ok") is True and is_num(rb) and rb <= TOL_I4):',
     'if not (v.get("I4_sessions_ok") is True and is_num(rb) and rb <= TOL_I4):'),
    ("I4_skip_sessions", 'if not (v.get("I4_same_means") is True and v.get("I4_sessions_ok") is True and is_num(rb) and rb <= TOL_I4):',
     'if not (v.get("I4_same_means") is True and is_num(rb) and rb <= TOL_I4):'),
    ("I4_skip_rebuild", 'if not (v.get("I4_same_means") is True and v.get("I4_sessions_ok") is True and is_num(rb) and rb <= TOL_I4):',
     'if not (v.get("I4_same_means") is True and v.get("I4_sessions_ok") is True):'),
    ("I4_tol", "TOL_I4 = 1e-12", "TOL_I4 = 1e-11"),
    ("I1_no_fp", 'if i1.get("fingerprint_ok") is True:', "if True:"),
    ("I1_no_phys", 'hashes = (i1.get("mag_s2_sha256") == MAG_S2_SHA256 and i1.get("physics_sha256") == PHYSICS_SHA256',
     'hashes = (i1.get("mag_s2_sha256") == MAG_S2_SHA256'),
    ("I1_no_bird1_file", '            and i1.get("bird1_file_sha256") == M3.MAG_S2_BIRD1_SHA256)', '            )'),
    ("I1_tol_10x", 'near = (is_num(i1.get("bars_max_abs_diff")) and i1["bars_max_abs_diff"] <= M3.I1_TOL',
     'near = (is_num(i1.get("bars_max_abs_diff")) and i1["bars_max_abs_diff"] <= 10 * M3.I1_TOL'),
    ("I1_no_null", '            and is_num(i1.get("null_max_abs_diff")) and i1["null_max_abs_diff"] <= M3.I1_TOL\n', '\n'),
    ("I1_no_shape", '            and i1.get("shape") is not None and i1.get("shape") == i1.get("shape_stored"))', '            )'),
    ("I1_one_stage", 'out["I1"] = dict(verdict=OK if all(ok for ok, _ in checks) else ERROR', 'out["I1"] = dict(verdict=OK if checks[0][0] else ERROR'),
    ("I5_too_few_any", 'if not (isinstance(top.get("n_cells"), int) and top["n_cells"] < M3.MIN_CELLS_TOPOLOGY):', "if False:"),
    ("I3_counts_only", 'not (c.get("counts_ok") is True and c.get("rates_ok") is True)', 'not (c.get("counts_ok") is True)'),
    ("I5_no_relabel", "if again != top.get(\"shape\"):", "if False:"),
    ("I5_no_finite", 'if c.get("finite") is not True or not all_finite(c):', 'if c.get("finite") is not True:'),
    ("I5_no_verdict_names", 'if ap is not None and ap.get("verdict") not in ("even", "polar", "undetermined", "no_signal"):',
     "if False:"),
    ("all_finite_true", "    if isinstance(x, float):\n        return math.isfinite(x)", "    if isinstance(x, float):\n        return True"),
    ("missing_inv_ok", 'out[k] = dict(verdict=ERROR, reason="invariants.json missing")', 'out[k] = dict(verdict=OK, reason="")'),
    ("Q1_07", "need = frac_needed(0.8, n_birds(ref))", "need = frac_needed(0.7, n_birds(ref))"),
    ("Q1_09", "need = frac_needed(0.8, n_birds(ref))", "need = frac_needed(0.9, n_birds(ref))"),
    ("Q1_no_ring", 'for s in ("point", "ring", "sphere", "rp2")}', 'for s in ("point", "sphere", "rp2")}'),
    ("Q1_with_torus", 'for s in ("point", "ring", "sphere", "rp2")}', 'for s in ("point", "ring", "sphere", "rp2", "torus")}'),
    ("Q2_upper_065", "0.04 <= a <= 0.06 and b >= 0.8", "0.04 <= a <= 0.065 and b >= 0.8"),
    ("Q2_tuned_07", "0.04 <= a <= 0.06 and b >= 0.8", "0.04 <= a <= 0.06 and b >= 0.7"),
    ("Q2_all_deltas", "            if abs(d[\"delta\"] - 0.4) < 1e-12:", "            if True:"),
    ("Q3_07", "out[\"Q3\"] = dict(verdict=MET if got >= frac_needed(0.8, n_birds(q3)) else NOT_MET",
     "out[\"Q3\"] = dict(verdict=MET if got >= frac_needed(0.7, n_birds(q3)) else NOT_MET"),
    ("Q4_or", 'verdict=MET if bad_e == 0 and bad_p == 0 else NOT_MET', 'verdict=MET if bad_e == 0 or bad_p == 0 else NOT_MET'),
    ("Q4_wrong_name", 'bad_e = count(ref, lambda r: apt(r, "compass").get("verdict") == "polar")',
     'bad_e = count(ref, lambda r: apt(r, "compass").get("verdict") == "undetermined")'),
    ("H3_min_cells", "H3_MIN_CELLS = 81", "H3_MIN_CELLS = 1"),
    ("H3_point_06", 'count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.5, n_birds(recs)):\n            h3 = dict',
     'count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.6, n_birds(recs)):\n            h3 = dict'),
    ("H3_sim_cells", 'recs = cells.get(mname(0.4, n, "probe"), {})', 'recs = cells.get(mname(0.4, n, "sim"), {})'),
    ("H3_no_gate", "if n >= H3_MIN_CELLS and recs and gate_ok(recs) and \\", "if n >= H3_MIN_CELLS and recs and \\"),
    ("H4_no_half", "fit = [a for a in ANTI_GRID if a <= n // 2]", "fit = [a for a in ANTI_GRID if a <= n]"),
    ("H4_no_gate", 'if recs and gate_ok(recs) and count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.8, n_birds(recs)):',
     'if recs and count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.8, n_birds(recs)):'),
    ("H4_delta_order", "for delta in (0.4, 0.7, 1.0):", "for delta in (1.0, 0.7, 0.4):"),
    ("H4_07", 'out["H4"] = dict(verdict=MET if x >= frac_needed(0.8, n) else NOT_MET',
     'out["H4"] = dict(verdict=MET if x >= frac_needed(0.7, n) else NOT_MET'),
    ("H5_ant_verdict", 'if eps_ant is None and count(recs, lambda r: apt(r, "compass").get("odd_found") is True) >= need:',
     'if eps_ant is None and count(recs, lambda r: apt(r, "compass").get("verdict") != "even") >= need:'),
    ("H5_no_earlier", "earlier = top is None or p5[\"eps_ant\"] < top", "earlier = True"),
    ("H5_le", "earlier = top is None or p5[\"eps_ant\"] < top", "earlier = top is None or p5[\"eps_ant\"] <= top"),
    ("H5_sphere_le", "verdict=MET if (earlier and odd >= need and sph < need) else NOT_MET",
     "verdict=MET if (earlier and odd >= need and sph <= need) else NOT_MET"),
    ("H5_no_gate_ref", "if not (q4 and gate_ref):", "if not q4:"),
    ("H5_no_q4", "if not (q4 and gate_ref):", "if not gate_ref:"),
    ("H1_or", "verdict=MET if a >= need and b >= need else NOT_MET, compass_rp2=a",
     "verdict=MET if a >= need or b >= need else NOT_MET, compass_rp2=a"),
    ("H1_no_gate", "    if not gate_ref:\n        out[\"H1\"]", "    if False:\n        out[\"H1\"]"),
    ("H4_no_q4", '    if not q4:\n        out["H4"]', '    if False:\n        out["H4"]'),
    ("H2_stage2_two", "ok &= x <= math.floor(0.01 * n + 1e-9) and up < 0.05", "ok &= x <= math.floor(0.02 * n + 1e-9) and up < 0.07"),
    ("H2_cell_three", "if x > math.floor(0.2 * nb + 1e-9):", "if x > math.floor(0.3 * nb + 1e-9):"),
    ("H2_pooled_003", "ok &= nb > 0 and x / nb <= 0.02", "ok &= nb > 0 and x / nb <= 0.03"),
    ("H2_pooled_lt", "ok &= nb > 0 and x / nb <= 0.02", "ok &= nb > 0 and x / nb < 0.02"),
    ("H2_missing_ok", "                        ok = False\n                        h2[\"cells\"].setdefault(\"missing\"",
     "                        ok = ok\n                        h2[\"cells\"].setdefault(\"missing\""),
    ("H2_polar_only", 'CONTROLS = ("polar", "shuffled")', 'CONTROLS = ("polar",)'),
    ("H2_worst_ge", 'if x / nb > worst["share"]:', 'if x / nb >= worst["share"]:'),
    ("req_no_clean", "        and gate_ok(recs) and clean_controls(recs)", "        and gate_ok(recs)"),
    ("req_no_gate", "        and gate_ok(recs) and clean_controls(recs)", "        and clean_controls(recs)"),
    ("clean_02", 'shape(r, c) == "rp2") <= math.floor(0.1 * nb + 1e-9) for c in CONTROLS)', 'shape(r, c) == "rp2") <= math.floor(0.2 * nb + 1e-9) for c in CONTROLS)'),
    ("req_08_07", "if n80 is None and topology_reached(recs, 0.8):", "if n80 is None and topology_reached(recs, 0.7):"),
    ("req_n50_06", "if n50 is None and topology_reached(recs, 0.5):", "if n50 is None and topology_reached(recs, 0.6):"),
    ("bud_hours", 'if c["hours_max"] > b["max_hours"]:', "if False:"),
    ("bud_sessions", 'if "max_sessions" in b and c["n_sessions"] > b["max_sessions"]:', "if False:"),
    ("bud_recorded", 'if "max_recorded" in b and c["n_recorded"] > b["max_recorded"]:', "if False:"),
    ("bud_topo_rule", 'if prefix in topo_maps and topology_reached(recs, 0.8):',
     'if prefix in topo_maps and s.get("compass_rp2", -1) >= frac_needed(0.8, nb):'),
    ("bud_hours_first", 'key = (c["n_sessions"], c["planned_hours"], c["n_recorded"], c["hours_max"])',
     'key = (c["hours_max"], c["n_sessions"], c["planned_hours"], c["n_recorded"])'),
    ("bud_no_sessions", 'key = (c["n_sessions"], c["planned_hours"], c["n_recorded"], c["hours_max"])',
     'key = (c["planned_hours"], c["n_recorded"], c["hours_max"])'),
    ("bud_planned_off", 'key = (c["n_sessions"], c["planned_hours"], c["n_recorded"], c["hours_max"])',
     'key = (c["n_sessions"], c["n_recorded"], c["hours_max"])'),
    ("bud_anti_no_short", 'for f, topo_maps, anti_maps in ((1.0, ("M",), ("M", "A")), (0.1, ("F",), ("F", "AF"))):',
     'for f, topo_maps, anti_maps in ((1.0, ("M",), ("M",)), (0.1, ("F",), ("F",))):'),
    ("req_even_07", "if n80e is None and e >= frac_needed(0.8, nb):", "if n80e is None and e >= frac_needed(0.7, nb):"),
    ("mult_gt1", "if count(cells.get(mname(delta, n, mode), {}), lambda r, c=c: shape(r, c) == \"rp2\") > 0)",
     "if count(cells.get(mname(delta, n, mode), {}), lambda r, c=c: shape(r, c) == \"rp2\") > 1)"),
    ("val_missing", "    if missing:\n        issues.append", "    if False:\n        issues.append"),
    ("val_extra", "    if extra:\n        issues.append", "    if False:\n        issues.append"),
    ("val_seeds", "        if sorted(recs) != seeds:\n", "        if False:\n"),
    ("val_design", '            if r.get("design") != want[name]:', "            if False:"),
    ("val_code", '            if r.get("code") != code:', "            if False:"),
    ("val_label", '            if r.get("cell") != name or r.get("seed") != s:', "            if False:"),
    ("val_meta_missing", '        issues.append(f"{stage}: run metadata missing")', "        pass"),
    ("val_complete", '        if meta.get("complete") is not True:', "        if False:"),
    ("val_meta_code", '        if meta.get("code_sha256") != code["mag_s3"]:', "        if False:"),
    ("val_meta_seeds", '        if sorted(meta.get("seeds") or []) != seeds:', "        if False:"),
    ("val_points", '        if points is not None and plain(meta.get("points")) != plain(points):', "        if False:"),
    ("val_inv_missing", '        issues.append(f"{stage}: invariants.json missing")', "        pass"),
    ("val_inv_code", '        if inv.get("code") != code:', "        if False:"),
    ("val_inv_cover", "        if sorted(int(k) for k in (inv.get(\"per_seed\") or {})) != seeds:", "        if False:"),
    ("val_stored_points", "    if plain(stored) != plain(pts):", "    if False:"),
    ("val_valid_ignores", "valid=bool(not issues and all(", "valid=bool(True and all("),
    ("val_points_writes", '        if r["issues"]:\n', "        if False:\n"),
    ("confirmatory_any", "confirmatory=bool(sorted(seeds1) == STAGE1_SEEDS and sorted(seeds2) == STAGE2_SEEDS)", "confirmatory=True"),
    ("E4_lt", 'if half_w is None and s.get("compass_rp2", 99) <= s["n"] // 2:', 'if half_w is None and s.get("compass_rp2", 99) < s["n"] // 2:'),
    ("E5_compass_rp2", 'if half_eps is None and s.get("compass_sphere", -1) >= frac_needed(0.5, s["n"]):',
     'if half_eps is None and s.get("polar_sphere", -1) >= frac_needed(0.5, s["n"]):'),
]


def run_one(mid, old, new):
    n = SRC.count(old)
    if n != 1:
        return mid, f"PATTERN COUNT {n}"
    d = os.path.join(OUT, mid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "analyze.py"), "w") as fh:
        fh.write(SRC.replace(old, new))
    for f in ("test_analyze.py", "mag_s3.py", "mag_s2.py", "physics.json", "mag_s2_bird1.json"):
        shutil.copy(os.path.join(CODE, f), d)
    try:
        p = subprocess.run([sys.executable, "-I", "-B", os.path.join(d, "test_analyze.py")], capture_output=True,
                           text=True, cwd=d, timeout=600)
    except subprocess.TimeoutExpired:
        return mid, "killed (timeout)"
    fails = [ln.strip() for ln in p.stdout.splitlines() if ln.startswith("  ")]
    status = "SURVIVED" if p.returncode == 0 else ("killed" if fails else f"killed (crash rc={p.returncode})")
    return mid, status + (f"  by: {fails[0]}" + (f" (+{len(fails) - 1})" if len(fails) > 1 else "") if fails else "")


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    shutil.rmtree(OUT, ignore_errors=True)
    rows = []
    with ThreadPoolExecutor(int(os.environ.get("MUTATION_WORKERS", "2"))) as ex:
        for row in ex.map(lambda m: run_one(*m), M):
            rows.append(row)
            print("%-22s %s" % row, flush=True)
    surv = [m for m, st in rows if st.startswith("SURVIVED") or st.startswith("PATTERN")]
    print(f"\n{len(rows)} mutations, {len(rows) - len(surv)} killed, survivors or bad patterns: {surv}")
