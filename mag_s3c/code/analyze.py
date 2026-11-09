#!/usr/bin/env python3
"""
MAG-S3 confirmatory analysis, protocol version 0.4 (draft).

  I1-I5   invariants: "error" invalidates the run
  Q1-Q4   expectations: they gate the hypotheses
  H1-H5   hypotheses: "met", "not met" or "untestable"
  E1-E5   exploratory maps

A seed is one virtual bird. Stage 1 runs every map cell on birds 1-10; stage 2 checks the hypotheses on 100 new birds
(101-200) in the reference design and at the working points that this script chooses from the stage-1 maps. Shape
claims in a cell are testable only when the atlas of the same cell recognises RP2 and the sphere in at least 8 of 10
birds. Each hypothesis is gated by what it uses: H1 by the atlas gate of the reference cell, H3 by the atlas gate of
its working point, H4 by Q4, H5 by Q4 and the atlas gate of the reference cell. Q1 is reported and gates nothing. H2
is a safety claim and has no gate.

The frozen rules hold only on the full input, so both stages are checked first: every cell that mag_s3.py defines for
the stage must be present with exactly the declared birds, every record must carry the design of its cell and the
SHA-256 of the mag_s3.py next to this script, the newest run metadata must report a complete run of that code, and
the invariants file must cover every bird. `points` refuses to write working points when anything is off; `final`
recomputes the working points, checks the stage-2 designs against them and marks the result invalid on any problem.

Usage
  python3 -I analyze.py points STAGE1_DIR [--seeds1 1-10]                       writes STAGE1_DIR/working_points.json
  python3 -I analyze.py final STAGE1_DIR STAGE2_DIR [--seeds1 1-10] [--seeds2 101-200]
                                                                                writes STAGE2_DIR/results.json
Other bird sets (the blind pipeline check on 9301-9302) are analysed with the same rules and marked not confirmatory.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MAG_S2_SHA256 = "2352aefe0fe5974732c318b365f4fb5477ffc47fb78b12748bca2515465829c4"
PHYSICS_SHA256 = "647bc0ccb1312991a1b772193cac39f4ced1481c591422c50659bb822a3100c5"

MET, NOT_MET, UNTESTABLE = "met", "not met", "untestable"
OK, ERROR = "ok", "error"
STAGE1_SEEDS = list(range(1, 11))
STAGE2_SEEDS = list(range(101, 201))
DELTAS = (0.1, 0.2, 0.4, 0.7, 1.0)
N_GRID = (25, 50, 100, 200, 400)
ANTI_GRID = (6, 12, 25, 50, 100, 200, 400)
MODES = ("sim", "probe", "tetrode")
EPS_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
W_LEVELS = (0.1, 0.2, 0.3, 0.5)
H3_MIN_CELLS = 81                  # above one probe session (80 cells) the pseudo-population splits the shared noise
BUDGETS = {"B1": dict(mode="sim", max_recorded=150, max_hours=3.0),
           "B2": dict(mode="probe", max_sessions=10, max_hours=30.0),
           "B3": dict(mode="tetrode", max_sessions=40, max_hours=40.0)}
TOL_I2 = 1e-12                     # dimensionless modulation u
TOL_I4 = 1e-12                     # relative difference of lambda against its time-ordered rebuild
REF = "M_d1_N200_sim"
CONTROLS = ("polar", "shuffled")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_mag_s3():
    """The simulation module next to this script: it defines the cells of both stages and loads the frozen MAG-S2
    code (checking its hash)."""
    path = os.path.join(HERE, "mag_s3.py")
    spec = importlib.util.spec_from_file_location("mag_s3", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_mag_s2():
    return M3.m2


M3 = load_mag_s3()
FROZEN_FACTORS = (M3.P2.rel_factor, M3.P2.null_factor)


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def frac_needed(share, n):
    """Number of birds out of n that a share such as 8 of 10 or 80 of 100 asks for."""
    return math.ceil(share * n - 1e-9)


def cp_two(x, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a, x + 1, n - x))
    return [lo, hi]


def cp_upper_one(x, n, level=0.95):
    return 1.0 if x == n else float(stats.beta.ppf(level, x + 1, n - x))


def paired_difference(a, b, level=0.95):
    """Exact conditional interval for the difference of two paired shares (a - b): Clopper-Pearson for the share of
    a-only birds among the discordant ones, scaled by the discordant share. It lies below zero exactly when the
    upper bound of that share is below one half (the exact McNemar test)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    nd = n10 + n01
    if nd == 0:
        return dict(n10=n10, n01=n01, interval=[0.0, 0.0], below_zero=False)
    lo, hi = cp_two(n10, nd, level)
    s = nd / n
    return dict(n10=n10, n01=n01, interval=[s * (2 * lo - 1), s * (2 * hi - 1)], below_zero=bool(hi < 0.5))


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def all_finite(x):
    if isinstance(x, dict):
        return all(all_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(all_finite(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


# --------------------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------------------
def load_run(folder):
    """cell name -> {seed from the file name: record}."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(folder, "*", "seed_*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        seed = int(os.path.basename(path)[len("seed_"):-len(".json")])
        cells.setdefault(os.path.basename(os.path.dirname(path)), {})[seed] = rec
    return cells


def newest_meta(folder, mode):
    paths = sorted(glob.glob(os.path.join(folder, f"meta_{mode}_*.json")))
    if not paths:
        return None
    with open(paths[-1]) as fh:
        return json.load(fh)


def plain(x):
    """JSON form of a value (tuples become lists), for comparisons with what was read from disk."""
    return json.loads(json.dumps(x))


def check_stage(cells, inv, meta, expected, seeds, stage, points=None):
    """Problems that make a stage unusable for the frozen rules (empty list when the stage is complete and clean).
    cells and inv as read from the stage folder, meta its newest run metadata, expected the designs of the stage."""
    issues = []
    code = M3.CODE_HASHES
    want = {name: plain(asdict(d)) for name, d in expected.items()}
    missing, extra = sorted(set(want) - set(cells)), sorted(set(cells) - set(want))
    if missing:
        issues.append(f"{stage}: {len(missing)} expected cells missing, e.g. {missing[:5]}")
    if extra:
        issues.append(f"{stage}: {len(extra)} cells that mag_s3.py does not define, e.g. {extra[:5]}")
    seeds = sorted(seeds)
    bad = dict(seeds=[], design=[], code=[], label=[])
    for name in sorted(set(want) & set(cells)):
        recs = cells[name]
        if sorted(recs) != seeds:
            bad["seeds"].append(name)
        for s in sorted(recs):
            r = recs[s]
            if r.get("design") != want[name]:
                bad["design"].append(f"{name}/seed_{s}")
            if r.get("code") != code:
                bad["code"].append(f"{name}/seed_{s}")
            if r.get("cell") != name or r.get("seed") != s:
                bad["label"].append(f"{name}/seed_{s}")
    what = dict(seeds="cells whose birds differ from the declared ones", design="records whose design differs from "
                "the design of their cell", code="records written by other code", label="records filed under another "
                "cell or seed")
    for k, v in bad.items():
        if v:
            issues.append(f"{stage}: {len(v)} {what[k]}, e.g. {v[:3]}")
    if meta is None:
        issues.append(f"{stage}: run metadata missing")
    else:
        if meta.get("complete") is not True:
            issues.append(f"{stage}: the newest run metadata does not report a complete run")
        if meta.get("code_sha256") != code["mag_s3"]:
            issues.append(f"{stage}: the newest run metadata names other code")
        if sorted(meta.get("seeds") or []) != seeds:
            issues.append(f"{stage}: the newest run metadata declares other birds")
        if points is not None and plain(meta.get("points")) != plain(points):
            issues.append(f"{stage}: stage 2 ran at other working points than stage 1 gives")
    if inv is None:
        issues.append(f"{stage}: invariants.json missing")
    else:
        if inv.get("code") != code:
            issues.append(f"{stage}: invariants.json written by other code")
        if sorted(int(k) for k in (inv.get("per_seed") or {})) != seeds:
            issues.append(f"{stage}: invariants.json does not cover exactly the declared birds")
    return issues


def cond(rec, name):
    return rec.get("conditions", {}).get(name) or {}


def shape(rec, name):
    return (cond(rec, name).get("topology") or {}).get("shape")


def apt(rec, name):
    return cond(rec, name).get("antipodal") or {}


def count(recs, fn):
    return int(sum(1 for r in recs.values() if fn(r)))


def n_birds(recs):
    return len(recs)


def share(recs, fn):
    return count(recs, fn) / max(n_birds(recs), 1)


def gate_ok(recs):
    """Atlas of the cell recognises RP2 and the sphere in at least 8 of 10 birds."""
    n = n_birds(recs)
    if n == 0:
        return False
    need = frac_needed(0.8, n)
    return count(recs, lambda r: shape(r, "rp2") == "rp2") >= need and \
        count(recs, lambda r: shape(r, "sphere") == "sphere") >= need


def mname(delta, n, mode):
    return f"M_d{delta:g}_N{n}_{mode}"


def fname(delta, n, mode, rule="p05"):
    return f"F_d{delta:g}_N{n}_{mode}_{rule}"


def aname(delta, n, mode):
    return f"A_d{delta:g}_N{n}_{mode}"


def afname(delta, n, mode):
    return f"AF_d{delta:g}_N{n}_{mode}_p05"


def map_name(prefix, delta, n, mode):
    return {"M": mname, "F": fname, "A": aname, "AF": afname}[prefix](delta, n, mode)


# --------------------------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------------------------
def read_invariants(folder):
    path = os.path.join(folder, "invariants.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def i1_ok(i1):
    if not isinstance(i1, dict):
        return False, None
    hashes = (i1.get("mag_s2_sha256") == MAG_S2_SHA256 and i1.get("physics_sha256") == PHYSICS_SHA256
              and i1.get("bird1_file_sha256") == M3.MAG_S2_BIRD1_SHA256)
    if not hashes:
        return False, None
    if i1.get("fingerprint_ok") is True:
        return True, "bitwise"
    near = (is_num(i1.get("bars_max_abs_diff")) and i1["bars_max_abs_diff"] <= M3.I1_TOL
            and is_num(i1.get("null_max_abs_diff")) and i1["null_max_abs_diff"] <= M3.I1_TOL
            and i1.get("shape") is not None and i1.get("shape") == i1.get("shape_stored"))
    return near, ("tolerance" if near else None)


def invariants(invs, stages, m2):
    """I1, I2 and I4 from the invariants files of the run (None when a file is missing); I3 and I5 from every record of
    every stage."""
    out = {}
    invs = list(invs)
    if not invs or any(v is None for v in invs):
        for k in ("I1", "I2", "I4"):
            out[k] = dict(verdict=ERROR, reason="invariants.json missing")
    else:
        checks = [i1_ok(v.get("I1")) for v in invs]
        out["I1"] = dict(verdict=OK if all(ok for ok, _ in checks) else ERROR, how=[how for _, how in checks])
        per = {}
        for v in invs:
            per.update(v.get("per_seed") or {})
        worst = dict(ren=0.0, mags2=0.0, polar=0.0, rebuild=0.0)
        ok2, ok4 = bool(per), bool(per)
        for s, v in per.items():
            vals = (v.get("I2_ren_max_diff"), v.get("I2_mags2_max_diff"), v.get("I2_polar_max_sum"))
            if not all(is_num(x) and x <= TOL_I2 for x in vals):
                ok2 = False
            for k, x in zip(("ren", "mags2", "polar"), vals):
                if is_num(x):
                    worst[k] = max(worst[k], x)
            rb = v.get("I4_rebuild_max_rel_diff")
            if not (v.get("I4_same_means") is True and v.get("I4_sessions_ok") is True and is_num(rb) and rb <= TOL_I4):
                ok4 = False
            if is_num(rb):
                worst["rebuild"] = max(worst["rebuild"], rb)
        out["I2"] = dict(verdict=OK if ok2 else ERROR, worst={k: worst[k] for k in ("ren", "mags2", "polar")},
                         n_birds=len(per))
        out["I4"] = dict(verdict=OK if ok4 else ERROR, worst_rebuild=worst["rebuild"], n_birds=len(per))
    bad3, bad5, mismatched = [], [], []
    for cells in stages:
        for cell, recs in cells.items():
            for seed, r in recs.items():
                where = f"{cell}/seed_{seed}"
                for cname, c in (r.get("conditions") or {}).items():
                    if cname != "shuffled" and not (c.get("counts_ok") is True and c.get("rates_ok") is True):
                        bad3.append(f"{where}/{cname}")
                    if c.get("finite") is not True or not all_finite(c):
                        bad5.append(f"{where}/{cname}: not finite")
                    top = c.get("topology")
                    if top and top.get("shape") == "too_few":
                        if not (isinstance(top.get("n_cells"), int) and top["n_cells"] < M3.MIN_CELLS_TOPOLOGY):
                            bad5.append(f"{where}/{cname}: too_few with enough cells")
                    elif top:
                        try:
                            again = m2.decide(top, *FROZEN_FACTORS)["shape"]
                        except Exception as exc:                   # missing bars or null
                            bad5.append(f"{where}/{cname}: {exc!r}")
                            continue
                        if again != top.get("shape"):
                            mismatched.append(f"{where}/{cname}")
                    ap = c.get("antipodal")
                    if ap is not None and ap.get("verdict") not in ("even", "polar", "undetermined", "no_signal"):
                        bad5.append(f"{where}/{cname}: unknown antipodal verdict")
    out["I3"] = dict(verdict=OK if not bad3 else ERROR, problems=bad3[:20], n_problems=len(bad3))
    out["I5"] = dict(verdict=OK if not (bad5 or mismatched) else ERROR, problems=bad5[:20], relabelled=mismatched[:20],
                     n_problems=len(bad5) + len(mismatched))
    return out


# --------------------------------------------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------------------------------------------
def expectations(cells):
    out = {}
    ref = cells.get(REF, {})
    if ref:
        need = frac_needed(0.8, n_birds(ref))
        got = {s: count(ref, lambda r, s=s: shape(r, s) == s) for s in ("point", "ring", "sphere", "rp2")}
        out["Q1"] = dict(verdict=MET if all(v >= need for v in got.values()) else NOT_MET, recognised=got,
                         n=n_birds(ref), need=need)
    else:
        out["Q1"] = dict(verdict=UNTESTABLE, reason=f"cell {REF} missing")
    unt = tun = n_unt = n_tun = 0
    tun40 = n_tun40 = 0
    for name, recs in cells.items():
        if not (name.startswith("F_") and name.endswith("_p05")):
            continue
        for r in recs.values():
            d, c = r["design"], cond(r, "compass")
            n_t = d["n_tuned"]
            n_u = int(round(n_t * (1 - d["f"]) / d["f"]))
            unt += c.get("n_untuned_selected", 0)
            tun += c.get("n_tuned_selected", 0)
            n_unt += n_u
            n_tun += n_t
            if abs(d["delta"] - 0.4) < 1e-12:
                tun40 += c.get("n_tuned_selected", 0)
                n_tun40 += n_t
    if n_unt and n_tun40:
        a, b = unt / n_unt, tun40 / n_tun40
        out["Q2"] = dict(verdict=MET if (0.04 <= a <= 0.06 and b >= 0.8) else NOT_MET, untuned_pass=a,
                         tuned_pass_at_delta_0_4=b, tuned_pass_all=tun / max(n_tun, 1))
    else:
        out["Q2"] = dict(verdict=UNTESTABLE, reason="no f = 0.1 cells with the 5% rule")
    q3 = cells.get("Q3", {})
    if q3:
        got = count(q3, lambda r: shape(r, "compass") == "rp2")
        out["Q3"] = dict(verdict=MET if got >= frac_needed(0.8, n_birds(q3)) else NOT_MET, rp2=got, n=n_birds(q3),
                         delta=next(iter(q3.values()))["design"]["delta"])
    else:
        out["Q3"] = dict(verdict=UNTESTABLE, reason="cell Q3 missing")
    if ref and all(apt(r, "compass") and apt(r, "polar") for r in ref.values()):
        bad_e = count(ref, lambda r: apt(r, "compass").get("verdict") == "polar")
        bad_p = count(ref, lambda r: apt(r, "polar").get("verdict") == "even")
        out["Q4"] = dict(verdict=MET if bad_e == 0 and bad_p == 0 else NOT_MET, even_called_polar=bad_e,
                         polar_called_even=bad_p,
                         even_verdicts={v: count(ref, lambda r, v=v: apt(r, "compass").get("verdict") == v)
                                        for v in ("even", "polar", "undetermined", "no_signal")},
                         polar_verdicts={v: count(ref, lambda r, v=v: apt(r, "polar").get("verdict") == v)
                                         for v in ("even", "polar", "undetermined", "no_signal")})
    else:
        out["Q4"] = dict(verdict=UNTESTABLE, reason="antipodal records missing in the reference cell")
    return out


# --------------------------------------------------------------------------------------------------------------
# Working points (from stage 1)
# --------------------------------------------------------------------------------------------------------------
def working_points(cells):
    pts = {}
    h3 = None
    for n in N_GRID:
        recs = cells.get(mname(0.4, n, "probe"), {})
        if n >= H3_MIN_CELLS and recs and gate_ok(recs) and \
                count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.5, n_birds(recs)):
            h3 = dict(n_tuned=n)
            break
    pts["H3"] = h3 or dict(n_tuned=None, reason="no cell above 80 cells with an open atlas gate and RP2 in 5 of 10 "
                                                "birds at delta = 0.4")
    h4 = None
    for delta in (0.4, 0.7, 1.0):
        for n in N_GRID:
            recs = cells.get(mname(delta, n, "sim"), {})
            if recs and gate_ok(recs) and count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(0.8, n_birds(recs)):
                fit = [a for a in ANTI_GRID if a <= n // 2]
                h4 = dict(delta=delta, n_t=n, n_tuned=max(fit) if fit else None)
                break
        if h4:
            break
    pts["H4"] = h4 or dict(n_tuned=None, reason="no simultaneous cell with RP2 in 8 of 10 birds at delta 0.4, 0.7 or 1")
    eps_ant = eps_top = None
    for eps in EPS_LEVELS:
        recs = cells.get(f"E_eps{eps:g}", {})
        if not recs:
            continue
        need = frac_needed(0.8, n_birds(recs))
        if eps_ant is None and count(recs, lambda r: apt(r, "compass").get("odd_found") is True) >= need:
            eps_ant = eps
        if eps_top is None and count(recs, lambda r: shape(r, "compass") == "sphere") >= need:
            eps_top = eps
    pts["H5"] = dict(eps_ant=eps_ant, eps_top=eps_top)
    return pts


# --------------------------------------------------------------------------------------------------------------
# Hypotheses (stage 2 plus the stage-1 map for H2)
# --------------------------------------------------------------------------------------------------------------
def hypotheses(cells1, cells2, exp, pts):
    out = {}
    q4 = exp.get("Q4", {}).get("verdict") == MET
    ref1, ref2 = cells1.get(REF, {}), cells2.get("ref", {})
    gate_ref = gate_ok(ref1)
    # H1
    if not gate_ref:
        out["H1"] = dict(verdict=UNTESTABLE, reason="atlas gate of the reference cell closed")
    elif not ref2:
        out["H1"] = dict(verdict=UNTESTABLE, reason="stage-2 reference cell missing")
    else:
        n = n_birds(ref2)
        a = count(ref2, lambda r: shape(r, "compass") == "rp2")
        b = count(ref2, lambda r: shape(r, "polar") == "sphere")
        need = frac_needed(0.8, n)
        out["H1"] = dict(verdict=MET if a >= need and b >= need else NOT_MET, compass_rp2=a, polar_sphere=b, n=n,
                         compass_ci=cp_two(a, n), polar_ci=cp_two(b, n))
    # H2
    h2 = dict(stage2={}, map={}, cells={})
    ok = True
    if not ref2:
        out["H2"] = dict(verdict=UNTESTABLE, reason="stage-2 reference cell missing")
    else:
        n = n_birds(ref2)
        for c in CONTROLS:
            x = count(ref2, lambda r, c=c: shape(r, c) == "rp2")
            up = cp_upper_one(x, n)
            h2["stage2"][c] = dict(false_rp2=x, n=n, upper=up)
            ok &= x <= math.floor(0.01 * n + 1e-9) and up < 0.05
        worst = dict(share=-1.0, cell=None, control=None)
        per_mode = {m: {c: [0, 0] for c in CONTROLS} for m in MODES}
        for delta in DELTAS:
            for nn in N_GRID:
                for mode in MODES:
                    recs = cells1.get(mname(delta, nn, mode), {})
                    if not recs:
                        ok = False
                        h2["cells"].setdefault("missing", []).append(mname(delta, nn, mode))
                        continue
                    for c in CONTROLS:
                        x = count(recs, lambda r, c=c: shape(r, c) == "rp2")
                        nb = n_birds(recs)
                        per_mode[mode][c][0] += x
                        per_mode[mode][c][1] += nb
                        if x / nb > worst["share"]:
                            worst = dict(share=x / nb, cell=mname(delta, nn, mode), control=c, false_rp2=x, n=nb)
                        if x > math.floor(0.2 * nb + 1e-9):
                            ok = False
                            h2["cells"].setdefault("over", []).append(dict(cell=mname(delta, nn, mode), control=c,
                                                                           false_rp2=x, n=nb))
        for c in CONTROLS:
            x = sum(per_mode[m][c][0] for m in MODES)
            nb = sum(per_mode[m][c][1] for m in MODES)
            h2["map"][c] = dict(false_rp2=x, n=nb, share=x / max(nb, 1))
            ok &= nb > 0 and x / nb <= 0.02
        h2["worst_cell"] = worst
        h2["per_mode"] = {m: {c: dict(false_rp2=v[0], n=v[1], share=v[0] / max(v[1], 1)) for c, v in per_mode[m].items()}
                          for m in MODES}
        out["H2"] = dict(verdict=MET if ok else NOT_MET, **h2)
    # H3
    p3 = pts.get("H3") or {}
    if not p3.get("n_tuned"):
        out["H3"] = dict(verdict=UNTESTABLE, reason=p3.get("reason", "no working point"))
    else:
        a, b = cells2.get("H3_sim", {}), cells2.get("H3_probe", {})
        common = sorted(set(a) & set(b))
        if not common:
            out["H3"] = dict(verdict=UNTESTABLE, reason="stage-2 H3 cells missing")
        else:
            sa = [shape(a[s], "compass") == "rp2" for s in common]
            sb = [shape(b[s], "compass") == "rp2" for s in common]
            pdiff = paired_difference(sa, sb)
            out["H3"] = dict(verdict=MET if pdiff["below_zero"] else NOT_MET, n_tuned=p3["n_tuned"], n=len(common),
                             sim_rp2=int(sum(sa)), probe_rp2=int(sum(sb)), **pdiff)
    # H4
    p4 = pts.get("H4") or {}
    if not q4:
        out["H4"] = dict(verdict=UNTESTABLE, reason="Q4 not met")
    elif not p4.get("n_tuned"):
        out["H4"] = dict(verdict=UNTESTABLE, reason=p4.get("reason", "no working point"))
    else:
        recs = cells2.get("H4", {})
        if not recs:
            out["H4"] = dict(verdict=UNTESTABLE, reason="stage-2 H4 cell missing")
        else:
            n = n_birds(recs)
            x = count(recs, lambda r: apt(r, "compass").get("verdict") == "even")
            out["H4"] = dict(verdict=MET if x >= frac_needed(0.8, n) else NOT_MET, even=x, n=n, ci=cp_two(x, n), **p4)
    # H5
    p5 = pts.get("H5") or {}
    if not (q4 and gate_ref):
        out["H5"] = dict(verdict=UNTESTABLE, reason="Q4 not met" if not q4 else "atlas gate of the reference cell closed")
    elif p5.get("eps_ant") is None:
        out["H5"] = dict(verdict=UNTESTABLE, reason="the antipodal test reaches 8 of 10 birds at no eps of the grid", **p5)
    else:
        recs = cells2.get("H5", {})
        top = p5.get("eps_top")
        earlier = top is None or p5["eps_ant"] < top
        if not recs:
            out["H5"] = dict(verdict=UNTESTABLE, reason="stage-2 H5 cell missing", **p5)
        else:
            n = n_birds(recs)
            need = frac_needed(0.8, n)
            odd = count(recs, lambda r: apt(r, "compass").get("odd_found") is True)
            sph = count(recs, lambda r: shape(r, "compass") == "sphere")
            out["H5"] = dict(verdict=MET if (earlier and odd >= need and sph < need) else NOT_MET, odd_found=odd,
                             sphere=sph, n=n, eps_ant_below_eps_top=earlier, **p5)
    return out


# --------------------------------------------------------------------------------------------------------------
# Exploratory maps and requirements
# --------------------------------------------------------------------------------------------------------------
def clean_controls(recs):
    """At most one bird in ten with a false RP2 in each control."""
    nb = n_birds(recs)
    return all(count(recs, lambda r, c=c: shape(r, c) == "rp2") <= math.floor(0.1 * nb + 1e-9) for c in CONTROLS)


def topology_reached(recs, share):
    """The compass reads as RP2 in the given share of birds, the atlas gate of the cell is open and the controls
    are clean: the one rule behind N80, N50 and the topology column of the budget map."""
    return bool(recs) and count(recs, lambda r: shape(r, "compass") == "rp2") >= frac_needed(share, n_birds(recs)) \
        and gate_ok(recs) and clean_controls(recs)


def planned_hours(design):
    """Session length without the turns, which vary at random with the order: K x R x (window + settle)."""
    return design["K"] * design["R"] * (design["T"] + M3.SETTLE_S) / 3600.0


def cell_summary(recs):
    s = dict(n=n_birds(recs))
    if not recs:
        return s
    first = next(iter(recs.values()))
    s["cost"] = dict(n_recorded=first["cost"]["n_recorded"], n_sessions=first["cost"]["n_sessions"],
                     planned_hours=planned_hours(first["design"]),
                     hours_max=max(r["cost"]["hours"] for r in recs.values()))
    if shape(first, "compass") is not None:
        s["compass_rp2"] = count(recs, lambda r: shape(r, "compass") == "rp2")
        s["polar_sphere"] = count(recs, lambda r: shape(r, "polar") == "sphere")
        s["compass_sphere"] = count(recs, lambda r: shape(r, "compass") == "sphere")
        for c in CONTROLS:
            s[f"{c}_false_rp2"] = count(recs, lambda r, c=c: shape(r, c) == "rp2")
        if shape(first, "rp2") is not None:
            s["gate_ok"] = gate_ok(recs)
            s["atlas_rp2"] = count(recs, lambda r: shape(r, "rp2") == "rp2")
            s["atlas_sphere"] = count(recs, lambda r: shape(r, "sphere") == "sphere")
        s["compass_labels"] = {lab: count(recs, lambda r, lab=lab: shape(r, "compass") == lab)
                               for lab in sorted({shape(r, "compass") for r in recs.values()}, key=str)}
    if apt(first, "compass"):
        s["even"] = count(recs, lambda r: apt(r, "compass").get("verdict") == "even")
        s["odd_found"] = count(recs, lambda r: apt(r, "compass").get("odd_found") is True)
        s["polar_called_polar"] = count(recs, lambda r: apt(r, "polar").get("verdict") == "polar")
        rho = [apt(r, "compass").get("rho") for r in recs.values() if is_num(apt(r, "compass").get("rho"))]
        s["rho_median"] = float(np.median(rho)) if rho else None
    c = cond(first, "compass")
    if first["design"]["f"] < 1:
        s["tuned_selected_mean"] = float(np.mean([cond(r, "compass").get("n_tuned_selected", 0) for r in recs.values()]))
        s["untuned_selected_mean"] = float(np.mean([cond(r, "compass").get("n_untuned_selected", 0) for r in recs.values()]))
        s["tuned_before_selection"] = first["design"]["n_tuned"]
    elif c:
        s["tuned_before_selection"] = first["design"]["n_tuned"]
    return s


def within_budget(summary, budget):
    b = BUDGETS[budget]
    if "cost" not in summary:
        return False
    c = summary["cost"]
    if c["hours_max"] > b["max_hours"]:
        return False
    if "max_recorded" in b and c["n_recorded"] > b["max_recorded"]:
        return False
    if "max_sessions" in b and c["n_sessions"] > b["max_sessions"]:
        return False
    return True


def requirements(cells):
    """Smallest number of tuned cells for RP2 in 8 of 10 (N80) and 5 of 10 (N50) birds with an open atlas gate and
    clean controls, per depth and recording mode, and the smallest number for the even verdict in 8 of 10 birds
    (N80_even)."""
    out = {}
    for prefix, f, topo in (("M", 1.0, True), ("F", 0.1, True), ("A", 1.0, False), ("AF", 0.1, False)):
        table = {}
        for delta in DELTAS:
            for mode in MODES:
                n80 = n50 = n80e = None
                for n in ANTI_GRID:
                    recs = cells.get(map_name(prefix, delta, n, mode), {})
                    if not recs:
                        continue
                    nb = n_birds(recs)
                    first = next(iter(recs.values()))
                    if topo and shape(first, "compass") is not None:
                        if n80 is None and topology_reached(recs, 0.8):
                            n80 = n
                        if n50 is None and topology_reached(recs, 0.5):
                            n50 = n
                    if apt(first, "compass"):
                        e = count(recs, lambda r: apt(r, "compass").get("verdict") == "even")
                        if n80e is None and e >= frac_needed(0.8, nb):
                            n80e = n
                row = dict(N80_even=n80e if n80e else ">400")
                if topo:
                    row.update(N80=n80 if n80 else ">400", N50=n50 if n50 else ">400")
                table[f"d{delta:g}_{mode}"] = row
        out[f"{prefix}_f{f:g}"] = table
    return out


def budget_map(cells):
    """For each budget and depth the cheapest design within the budget that reaches 8 of 10 birds, separately for the
    topology (maps M and F, by topology_reached) and for the antipodal test (maps M, F and the short-session maps A,
    AF). Cheapest means fewest sessions, then the shortest planned session, then the fewest recorded cells, then the
    shortest longest session."""
    out = {}
    for f, topo_maps, anti_maps in ((1.0, ("M",), ("M", "A")), (0.1, ("F",), ("F", "AF"))):
        res = {}
        for budget in BUDGETS:
            mode = BUDGETS[budget]["mode"]                 # each budget belongs to one recording mode
            per_delta = {}
            first = {"topology": None, "antipodal": None}
            for delta in DELTAS:
                best = {"topology": None, "antipodal": None}
                for prefix in sorted(set(topo_maps) | set(anti_maps)):
                    for n in ANTI_GRID:
                        name = map_name(prefix, delta, n, mode)
                        recs = cells.get(name, {})
                        if not recs:
                            continue
                        s = cell_summary(recs)
                        if not within_budget(s, budget):
                            continue
                        nb = s["n"]
                        c = s["cost"]
                        cand = dict(cell=name, n_sessions=c["n_sessions"], planned_hours=c["planned_hours"],
                                    n_recorded=c["n_recorded"], hours=c["hours_max"])
                        key = (c["n_sessions"], c["planned_hours"], c["n_recorded"], c["hours_max"])
                        if prefix in topo_maps and topology_reached(recs, 0.8):
                            if best["topology"] is None or key < best["topology"]["key"]:
                                best["topology"] = dict(cand, key=key)
                        if prefix in anti_maps and s.get("even", -1) >= frac_needed(0.8, nb):
                            if best["antipodal"] is None or key < best["antipodal"]["key"]:
                                best["antipodal"] = dict(cand, key=key)
                per_delta[f"d{delta:g}"] = {k: ({kk: vv for kk, vv in v.items() if kk != "key"} if v else None)
                                            for k, v in best.items()}
                for k in best:
                    if first[k] is None and best[k] is not None:
                        first[k] = delta
            res[budget] = dict(smallest_delta=first, cheapest=per_delta)
        out[f"f{f:g}"] = res
    return out


def exploratory(cells, physics=None):
    out = {"E1": {}, "E3": {}, "E4": {}, "E5": {}}
    for name in sorted(cells):
        if name.startswith(("M_", "F_", "A_", "AF_")):
            out["E1"][name] = cell_summary(cells[name])
        elif name.startswith("S_"):
            out["E3"][name] = cell_summary(cells[name])
    out["E2"] = budget_map(cells)
    half_w = None
    for w in W_LEVELS:
        recs = cells.get(f"W_w{w:g}", {})
        if recs:
            s = cell_summary(recs)
            out["E4"][f"w{w:g}"] = s
            if half_w is None and s.get("compass_rp2", 99) <= s["n"] // 2:
                half_w = w
    out["E4"]["w_half_lose_rp2"] = half_w
    half_eps = None
    for eps in EPS_LEVELS:
        recs = cells.get(f"E_eps{eps:g}", {})
        if recs:
            s = cell_summary(recs)
            out["E5"][f"eps{eps:g}"] = s
            if half_eps is None and s.get("compass_sphere", -1) >= frac_needed(0.5, s["n"]):
                half_eps = eps
    out["E5"]["eps_half_sphere"] = half_eps
    if physics:
        out["E5"]["ciss_anchor"] = physics.get("anchor_haberkorn_forward_yield")
        out["E5"]["ciss_exponential_model"] = physics.get("anchor_exponential_model")
    return out


def multiplicity(cells):
    out = {}
    for c in CONTROLS:
        out[c] = sum(1 for delta in DELTAS for n in N_GRID for mode in MODES
                     if count(cells.get(mname(delta, n, mode), {}), lambda r, c=c: shape(r, c) == "rp2") > 0)
    return dict(cells_with_any_false_rp2=out, n_cells=len(DELTAS) * len(N_GRID) * len(MODES))


# --------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------
def run_points(s1, seeds1):
    cells1 = load_run(s1)
    inv1 = read_invariants(s1)
    issues = check_stage(cells1, inv1, newest_meta(s1, "stage1"), M3.stage1_cells(), seeds1, "stage 1")
    return dict(points=working_points(cells1), issues=issues,
                invariants=invariants([inv1], [cells1], M3.m2), expectations=expectations(cells1))


def run_final(s1, s2, seeds1, seeds2):
    cells1, cells2 = load_run(s1), load_run(s2)
    inv1, inv2 = read_invariants(s1), read_invariants(s2)
    pts = working_points(cells1)
    issues = check_stage(cells1, inv1, newest_meta(s1, "stage1"), M3.stage1_cells(), seeds1, "stage 1")
    issues += check_stage(cells2, inv2, newest_meta(s2, "stage2"), M3.stage2_cells(pts), seeds2, "stage 2",
                          points=pts)
    path = os.path.join(s1, "working_points.json")
    stored = None
    if os.path.exists(path):
        with open(path) as fh:
            stored = json.load(fh)
    if plain(stored) != plain(pts):
        issues.append("working_points.json differs from the working points recomputed from stage 1")
    inv = invariants([inv1, inv2], [cells1, cells2], M3.m2)
    physics = None
    ppath = os.path.join(s1, "physics_s3.json")
    if os.path.exists(ppath):
        with open(ppath) as fh:
            physics = json.load(fh)
    exp = expectations(cells1)
    hyp = hypotheses(cells1, cells2, exp, pts)
    return dict(valid=bool(not issues and all(v["verdict"] == OK for v in inv.values())), input_issues=issues,
                confirmatory=bool(sorted(seeds1) == STAGE1_SEEDS and sorted(seeds2) == STAGE2_SEEDS),
                seeds=dict(stage1=sorted(seeds1), stage2=sorted(seeds2)), invariants=inv, expectations=exp,
                working_points=pts, hypotheses=hyp, requirements=requirements(cells1),
                exploratory=exploratory(cells1, physics), multiplicity=multiplicity(cells1),
                analysis_sha256=sha256(os.path.abspath(__file__)), code=M3.CODE_HASHES)


def parse_seeds(text):
    return M3.parse_seeds(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MAG-S3 confirmatory analysis")
    ap.add_argument("step", choices=["points", "final"])
    ap.add_argument("stage1")
    ap.add_argument("stage2", nargs="?")
    ap.add_argument("--seeds1", default="1-10")
    ap.add_argument("--seeds2", default="101-200")
    a = ap.parse_args(argv)
    seeds1, seeds2 = parse_seeds(a.seeds1), parse_seeds(a.seeds2)
    if a.step == "points":
        r = run_points(a.stage1, seeds1)
        if r["issues"]:
            print(json.dumps(dict(refused=r["issues"]), indent=1))
            return 1
        with open(os.path.join(a.stage1, "working_points.json"), "w") as fh:
            json.dump(r["points"], fh, indent=1)
        print(json.dumps(dict(points=r["points"], invariants={k: v["verdict"] for k, v in r["invariants"].items()},
                              expectations={k: v["verdict"] for k, v in r["expectations"].items()}), indent=1))
        return 0
    if not a.stage2:
        ap.error("final needs STAGE2_DIR")
    res = run_final(a.stage1, a.stage2, seeds1, seeds2)
    with open(os.path.join(a.stage2, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(dict(valid=res["valid"], confirmatory=res["confirmatory"], input_issues=res["input_issues"],
                          invariants={k: v["verdict"] for k, v in res["invariants"].items()},
                          expectations={k: v["verdict"] for k, v in res["expectations"].items()},
                          hypotheses={k: v["verdict"] for k, v in res["hypotheses"].items()}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
