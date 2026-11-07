#!/usr/bin/env python3
"""
Synthetic tests of analyze.py: worlds of records with known answers. No simulation is run; every record is written by
hand so that each verdict, gate and working-point rule can be checked at its edge.

Usage: python3 -I test_analyze.py          (exit code 0 when every test passes)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(os.environ.get("ANALYZE_MODULE", "analyze"))
M2 = A.load_mag_s2()

BARS = {"point": ([], [], [], []), "ring": ([5.0], [], [5.0], []), "sphere": ([], [5.0], [], [5.0]),
        "rp2": ([5.0], [5.0], [], [0.5]), "torus": ([5.0, 5.0], [5.0], [5.0, 5.0], [5.0]),
        "klein": ([5.0, 5.0], [5.0], [], [0.5]), "other": ([5.0, 5.0], [], [5.0, 5.0], [])}


def topo(label):
    h1, h2, h1z3, h2z3 = BARS[label]
    raw = dict(n_points=100, bars={"Z2": {"H1": h1, "H2": h2}, "Z3": {"H1": h1z3, "H2": h2z3}},
               null={"Z2": {"H1": 0.1, "H2": 0.1}, "Z3": {"H1": 0.1, "H2": 0.1}})
    raw.update(M2.decide(raw, *A.FROZEN_FACTORS))
    assert raw["shape"] == label, (label, raw["shape"])
    return raw


def ap(verdict="even", odd=False, rho=0.01):
    return dict(n_cells=50, n_repeats=8, E_odd=0.1, E_even=10.0, rho=rho, p_odd=0.01 if odd else 0.5, odd_found=odd,
                verdict=verdict, gate_q05=1.0, d1_q95=-1.0, d2_q05=-1.0)


def design(**kw):
    d = dict(delta=1.0, n_tuned=200, mode="sim", sigma_g=0.1, K=800, R=4, T=2.0, f=1.0, select="none", pattern="ren",
             fano=1.0, w=0.0, eps=0.0, tau_a=0.0, drift=True, topology=True, antipodal=True, atlas=["rp2", "sphere"])
    d.update(kw)
    return d


def condition(label=None, antipodal=None, n_tuned_sel=0, n_untuned_sel=0):
    c = dict(counts_ok=True, rates_ok=True, finite=True, n_selected=n_tuned_sel + n_untuned_sel,
             n_tuned_selected=n_tuned_sel, n_untuned_selected=n_untuned_sel, fingerprint="x")
    if label is not None:
        c["topology"] = topo(label)
    if antipodal is not None:
        c["antipodal"] = antipodal
    return c


def record(seed, d, compass=None, polar=None, shuffled=None, atlas=None, ap_c=None, ap_p=None, sel=(0, 0),
           hours=0.73, n_sessions=1):
    conds = {}
    topo_on = d["topology"]
    conds["compass"] = condition(compass if topo_on else None, ap_c if d["antipodal"] else None, *sel)
    conds["polar"] = condition(polar if topo_on else None, ap_p if d["antipodal"] else None)
    if topo_on:
        conds["shuffled"] = dict(topology=topo(shuffled), fingerprint="y", finite=True)
        for s, lab in (atlas or {}).items():
            conds[s] = condition(lab)
    n_rec = d["n_tuned"] + (int(round(d["n_tuned"] * (1 - d["f"]) / d["f"])) if d["f"] < 1 else 0)
    return dict(seed=seed, cost=dict(n_recorded=n_rec, n_sessions=n_sessions, hours=hours), conditions=conds,
                design=d)


def seq(n, k, yes, no):
    """n labels with the first k equal to yes and the rest equal to no."""
    return [yes if i < k else no for i in range(n)]


def good_i1():
    return dict(fingerprint_ok=True, mag_s2_sha256=A.MAG_S2_SHA256, physics_sha256=A.PHYSICS_SHA256,
                bird1_file_sha256=A.M3.MAG_S2_BIRD1_SHA256, bars_max_abs_diff=0.0, null_max_abs_diff=0.0,
                shape="rp2", shape_stored="rp2")


def good_invariants(seeds):
    return dict(code=dict(A.M3.CODE_HASHES), I1=good_i1(),
                per_seed={str(s): dict(I2_ren_max_diff=0.0, I2_mags2_max_diff=8e-15, I2_polar_max_sum=0.0,
                                       I4_same_means=True, I4_rebuild_max_rel_diff=5e-16, I4_sessions_ok=True)
                          for s in seeds})


class World:
    """A full stage-1 and stage-2 data set; every cell good unless a test changes it."""

    def __init__(self):
        self.s1, self.s2 = {}, {}
        seeds = A.STAGE1_SEEDS
        for delta in A.DELTAS:
            for mode in A.MODES:
                for n in A.N_GRID:
                    d = design(delta=delta, n_tuned=n, mode=mode)
                    atlas = dict(rp2="rp2", sphere="sphere")
                    if (delta, n, mode) == (1.0, 200, "sim"):
                        d["atlas"] = list(A.SHAPES_ALL) if hasattr(A, "SHAPES_ALL") else ["point", "ring", "sphere", "torus", "rp2", "klein"]
                        atlas = {s: s for s in ["point", "ring", "sphere", "torus", "rp2", "klein"]}
                    self.s1[A.mname(delta, n, mode)] = {s: record(s, d, "rp2", "sphere", "point", atlas, ap("even"),
                                                                  ap("polar", True, 0.99)) for s in seeds}
                    df = design(delta=delta, n_tuned=n, mode=mode, f=0.1, select="p05")
                    sel = (int(round(0.85 * n)), int(round(0.05 * 9 * n)))
                    self.s1[A.fname(delta, n, mode)] = {s: record(s, df, "rp2", "sphere", "point", dict(rp2="rp2", sphere="sphere"),
                                                                  ap("even"), ap("polar", True, 0.99), sel=sel)
                                                        for s in seeds}
                for n in (6, 12):
                    d = design(delta=delta, n_tuned=n, mode=mode, topology=False, atlas=[])
                    self.s1[A.mname(delta, n, mode)] = {s: record(s, d, ap_c=ap("even"), ap_p=ap("polar", True))
                                                        for s in seeds}
                for n in A.ANTI_GRID:
                    da = design(delta=delta, n_tuned=n, mode=mode, K=100, R=8, topology=False, atlas=[])
                    self.s1[A.aname(delta, n, mode)] = {s: record(s, da, ap_c=ap("even"), ap_p=ap("polar", True),
                                                                  hours=0.5) for s in seeds}
                    daf = dict(da, f=0.1, select="p05")
                    self.s1[A.afname(delta, n, mode)] = {s: record(s, daf, ap_c=ap("even"), ap_p=ap("polar", True),
                                                                   sel=(int(round(0.85 * n)), int(round(0.45 * n))),
                                                                   hours=0.5) for s in seeds}
        for eps in A.EPS_LEVELS:
            d = design(eps=eps)
            odd = eps >= 0.05
            sph = "sphere" if eps >= 0.5 else "rp2"
            self.s1[f"E_eps{eps:g}"] = {s: record(s, d, sph, "sphere", "point", dict(rp2="rp2", sphere="sphere"),
                                                  ap("undetermined", odd), ap("polar", True)) for s in seeds}
        for w in A.W_LEVELS:
            d = design(w=w)
            lab = "rp2" if w < 0.3 else "sphere"
            self.s1[f"W_w{w:g}"] = {s: record(s, d, lab, "sphere", "point", dict(rp2="rp2", sphere="sphere"),
                                              ap("even"), ap("polar", True)) for s in seeds}
        dq = design(pattern="mags2", n_tuned=100, K=600, delta=0.8, antipodal=False, atlas=[])
        self.s1["Q3"] = {s: record(s, dq, "rp2", "sphere", "point") for s in seeds}
        self.inv = good_invariants(seeds)
        s2 = A.STAGE2_SEEDS
        dref = design(atlas=[])
        self.s2["ref"] = {s: record(s, dref, "rp2" if i < 95 else "other", "sphere" if i < 95 else "other", "point",
                                    None, ap("even"), ap("polar", True)) for i, s in enumerate(s2)}
        d3s = design(delta=0.4, n_tuned=100, sigma_g=0.2, antipodal=False, atlas=[])
        d3p = design(delta=0.4, n_tuned=100, sigma_g=0.2, mode="probe", antipodal=False, atlas=[])
        self.s2["H3_sim"] = {s: record(s, d3s, "rp2" if i < 30 else "other", "sphere", "point") for i, s in enumerate(s2)}
        self.s2["H3_probe"] = {s: record(s, d3p, "rp2" if i < 60 else "other", "sphere", "point")
                               for i, s in enumerate(s2)}
        d4 = design(delta=0.4, n_tuned=50, topology=False, atlas=[])
        self.s2["H4"] = {s: record(s, d4, ap_c=ap("even" if i < 90 else "undetermined"), ap_p=ap("polar", True))
                         for i, s in enumerate(s2)}
        d5 = design(eps=0.05, atlas=[])
        self.s2["H5"] = {s: record(s, d5, "sphere" if i < 10 else "rp2", "sphere", "point", None,
                                   ap("undetermined", i < 90), ap("polar", True)) for i, s in enumerate(s2)}
        self.inv2 = good_invariants(s2)
        # H3 working point: probe cells at delta 0.4 reach 5 of 10 at N = 100 (N <= 80 does not count)
        for n in A.N_GRID:
            k = {25: 6, 50: 6, 100: 5, 200: 9, 400: 10}[n]
            for i, s in enumerate(A.STAGE1_SEEDS):
                self.s1[A.mname(0.4, n, "probe")][s]["conditions"]["compass"]["topology"] = topo("rp2" if i < k else "other")
        # H4 working point: simultaneous cells at delta 0.4 reach 8 of 10 at N = 100
        for n in A.N_GRID:
            k = {25: 3, 50: 7, 100: 8, 200: 10, 400: 10}[n]
            for i, s in enumerate(A.STAGE1_SEEDS):
                self.s1[A.mname(0.4, n, "sim")][s]["conditions"]["compass"]["topology"] = topo("rp2" if i < k else "other")

    def set_label(self, stage, cell, cname, labels):
        cells = self.s1 if stage == 1 else self.s2
        for i, s in enumerate(sorted(cells[cell])):
            cells[cell][s]["conditions"][cname]["topology"] = topo(labels[i])

    def set_ap(self, stage, cell, cname, verdicts=None, odd=None):
        cells = self.s1 if stage == 1 else self.s2
        for i, s in enumerate(sorted(cells[cell])):
            a = cells[cell][s]["conditions"][cname]["antipodal"]
            if verdicts is not None:
                a["verdict"] = verdicts[i]
            if odd is not None:
                a["odd_found"] = odd[i]

    def write(self, root):
        d1, d2 = os.path.join(root, "s1"), os.path.join(root, "s2")
        for folder, cells in ((d1, self.s1), (d2, self.s2)):
            for name, recs in cells.items():
                os.makedirs(os.path.join(folder, name), exist_ok=True)
                for s, r in recs.items():
                    with open(os.path.join(folder, name, f"seed_{s}.json"), "w") as fh:
                        json.dump(r, fh)
        with open(os.path.join(d1, "invariants.json"), "w") as fh:
            json.dump(self.inv, fh)
        os.makedirs(d2, exist_ok=True)
        with open(os.path.join(d2, "invariants.json"), "w") as fh:
            json.dump(self.inv2, fh)
        return d1, d2

    def invariants(self):
        return A.invariants([self.inv, self.inv2], [self.s1, self.s2], M2)

    def analyse(self):
        c1, c2 = self.s1, self.s2
        exp = A.expectations(c1)
        pts = A.working_points(c1)
        hyp = A.hypotheses(c1, c2, exp, pts)
        return dict(exp=exp, pts=pts, hyp=hyp, req=A.requirements(c1), bud=A.budget_map(c1),
                    expl=A.exploratory(c1), mult=A.multiplicity(c1))

    def analyse_from_disk(self):
        root = tempfile.mkdtemp(prefix="mag_s3_test_")
        try:
            d1, d2 = self.write(root)
            c1, c2 = A.load_run(d1), A.load_run(d2)
            inv = A.invariants([A.read_invariants(d1), A.read_invariants(d2)], [c1, c2], M2)
            return dict(inv=inv, exp=A.expectations(c1), pts=A.working_points(c1), c1=c1, c2=c2)
        finally:
            shutil.rmtree(root)


BASE = World()
FAILS = []


def check(name, cond, info=""):
    if not cond:
        FAILS.append(f"{name}: {info}")


def verdicts(res, part):
    return {k: v["verdict"] for k, v in res[part].items()}


BASE_PICKLE = pickle.dumps(BASE, protocol=pickle.HIGHEST_PROTOCOL)


def world():
    return pickle.loads(BASE_PICKLE)


def inv_verdicts(w):
    return {k: v["verdict"] for k, v in w.invariants().items()}


# ----------------------------------------------------------------------------------------------------------------
def test_helpers():
    check("needed 8 of 10", A.frac_needed(0.8, 10) == 8)
    check("needed 80 of 100", A.frac_needed(0.8, 100) == 80)
    check("needed 5 of 10", A.frac_needed(0.5, 10) == 5)
    check("needed 9 of 10", A.frac_needed(0.9, 10) == 9)
    check("needed 8 of 10 with 7 birds", A.frac_needed(0.8, 7) == 6)
    check("needed 5 of 10 with 7 birds", A.frac_needed(0.5, 7) == 4)
    check("cp 80/100", abs(A.cp_two(80, 100)[0] - 0.7082) < 5e-4, A.cp_two(80, 100))
    check("cp 8/10", abs(A.cp_two(8, 10)[0] - 0.4439) < 5e-4, A.cp_two(8, 10))
    check("cp 0/10 upper", abs(A.cp_two(0, 10)[1] - 0.3085) < 5e-4, A.cp_two(0, 10))
    check("cp one-sided 0/100", abs(A.cp_upper_one(0, 100) - 0.0295) < 2e-4, A.cp_upper_one(0, 100))
    check("cp one-sided 1/100", abs(A.cp_upper_one(1, 100) - 0.0466) < 2e-4, A.cp_upper_one(1, 100))
    check("cp one-sided 2/100", abs(A.cp_upper_one(2, 100) - 0.0616) < 2e-4, A.cp_upper_one(2, 100))
    a = [True] * 2 + [False] * 20 + [True] * 10 + [False] * 68
    b = [False] * 2 + [True] * 20 + [True] * 10 + [False] * 68
    p = A.paired_difference(a, b)
    check("paired n10 n01", p["n10"] == 2 and p["n01"] == 20, p)
    check("paired below zero", p["below_zero"] and p["interval"][1] < 0, p)
    check("paired interval scaled by the discordant share", abs(p["interval"][0] + 0.2151) < 0.003 and abs(p["interval"][1] + 0.0915) < 0.003, p)
    p = A.paired_difference(b, a)
    check("paired reversed above zero", not p["below_zero"] and p["interval"][0] > 0, p)
    p = A.paired_difference([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)
    check("paired tie not below", not p["below_zero"], p)
    p = A.paired_difference([True] * 10, [True] * 10)
    check("paired no discordant", not p["below_zero"] and p["interval"] == [0.0, 0.0], p)
    p = A.paired_difference([False] * 94 + [True], [True] * 6 + [False] * 89)  # n10 = 1, n01 = 6: CP upper of 1/7 is 0.58
    check("paired 1 vs 6 not below", not p["below_zero"], p)
    p = A.paired_difference([False] * 100, [True] * 6 + [False] * 94)          # n10 = 0, n01 = 6: CP upper of 0/6 is 0.46
    check("paired 0 vs 6 below", p["below_zero"], p)


def test_disk_round_trip():
    r = BASE.analyse_from_disk()
    check("disk: all invariants ok", all(v["verdict"] == A.OK for v in r["inv"].values()), {k: v["verdict"] for k, v in r["inv"].items()})
    check("disk: same cells", sorted(r["c1"]) == sorted(BASE.s1) and sorted(r["c2"]) == sorted(BASE.s2))
    check("disk: same points", r["pts"] == BASE.analyse()["pts"])
    check("disk: same invariants", r["inv"] == BASE.invariants())
    root = tempfile.mkdtemp(prefix="mag_s3_test_")
    try:
        check("disk: missing invariants file", A.read_invariants(root) is None)
        inv = A.invariants([None], [{}], M2)
        check("disk: missing file is an error", inv["I1"]["verdict"] == A.ERROR and inv["I4"]["verdict"] == A.ERROR)
    finally:
        shutil.rmtree(root)


def test_all_good():
    r = BASE.analyse()
    check("invariants ok", all(v == A.OK for v in inv_verdicts(BASE).values()), inv_verdicts(BASE))
    check("expectations met", all(v == A.MET for v in verdicts(r, "exp").values()), verdicts(r, "exp"))
    check("hypotheses met", all(v == A.MET for v in verdicts(r, "hyp").values()), r["hyp"])
    check("H3 point", r["pts"]["H3"].get("n_tuned") == 100, r["pts"]["H3"])
    check("H4 point", r["pts"]["H4"].get("n_t") == 100 and r["pts"]["H4"].get("n_tuned") == 50, r["pts"]["H4"])
    check("H5 points", r["pts"]["H5"] == dict(eps_ant=0.05, eps_top=0.5), r["pts"]["H5"])
    check("E4 half w", r["expl"]["E4"]["w_half_lose_rp2"] == 0.3, r["expl"]["E4"]["w_half_lose_rp2"])
    w = world()
    w.set_label(1, "W_w0.2", "compass", seq(10, 5, "rp2", "other"))
    r = w.analyse()
    check("E4 half w at exactly half", r["expl"]["E4"]["w_half_lose_rp2"] == 0.2, r["expl"]["E4"]["w_half_lose_rp2"])
    check("E5 half eps", r["expl"]["E5"]["eps_half_sphere"] == 0.5, r["expl"]["E5"]["eps_half_sphere"])
    check("multiplicity zero", r["mult"]["cells_with_any_false_rp2"] == dict(polar=0, shuffled=0), r["mult"])


def test_invariants():
    w = world()
    w.inv["I1"]["fingerprint_ok"] = False
    r = w.invariants()
    check("I1 tolerance path ok", r["I1"]["verdict"] == A.OK and r["I1"]["how"][0] == "tolerance", r["I1"])
    w.inv["I1"]["bars_max_abs_diff"] = 2e-6
    check("I1 bars beyond tolerance", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv["I1"]["fingerprint_ok"] = False
    w.inv["I1"]["null_max_abs_diff"] = 2e-6
    check("I1 null beyond tolerance", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv["I1"]["fingerprint_ok"] = False
    w.inv["I1"]["shape"] = "klein"
    check("I1 other shape", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv["I1"]["fingerprint_ok"] = False
    w.inv["I1"]["bars_max_abs_diff"] = None
    check("I1 bars missing", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv["I1"]["bars_max_abs_diff"] = 1.0
    check("I1 bitwise needs no tolerance", inv_verdicts(w)["I1"] == A.OK)
    for key in ("mag_s2_sha256", "physics_sha256", "bird1_file_sha256"):
        w = world()
        w.inv["I1"][key] = "0" * 64
        check(f"I1 {key}", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv2["I1"]["fingerprint_ok"] = False
    w.inv2["I1"]["shape"] = None
    check("I1 of stage 2 counts", inv_verdicts(w)["I1"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["3"]["I2_ren_max_diff"] = 2e-12
    check("I2 ren", inv_verdicts(w)["I2"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["3"]["I2_polar_max_sum"] = float("nan")
    check("I2 nan", inv_verdicts(w)["I2"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["3"]["I2_mags2_max_diff"] = 1e-12
    check("I2 at tolerance ok", inv_verdicts(w)["I2"] == A.OK)
    w = world()
    w.inv["per_seed"]["3"]["I2_mags2_max_diff"] = 2e-12
    check("I2 mags2 above tolerance", inv_verdicts(w)["I2"] == A.ERROR)
    w = world()
    w.inv2["per_seed"]["150"]["I2_mags2_max_diff"] = 2e-12
    check("I2 stage-2 bird", inv_verdicts(w)["I2"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["7"]["I4_same_means"] = False
    check("I4 same means", inv_verdicts(w)["I4"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["7"]["I4_sessions_ok"] = False
    check("I4 sessions", inv_verdicts(w)["I4"] == A.ERROR)
    w = world()
    w.inv2["per_seed"]["120"]["I4_rebuild_max_rel_diff"] = 2e-12
    check("I4 rebuild above tolerance", inv_verdicts(w)["I4"] == A.ERROR)
    w = world()
    w.inv["per_seed"]["7"]["I4_rebuild_max_rel_diff"] = 1e-12
    check("I4 rebuild at tolerance ok", inv_verdicts(w)["I4"] == A.OK)
    w = world()
    del w.inv["per_seed"]["7"]["I4_rebuild_max_rel_diff"]
    check("I4 rebuild missing", inv_verdicts(w)["I4"] == A.ERROR)
    w = world()
    w.s1[A.REF][3]["conditions"]["polar"]["counts_ok"] = False
    check("I3 counts", inv_verdicts(w)["I3"] == A.ERROR)
    w = world()
    w.s1["Q3"][5]["conditions"]["compass"]["rates_ok"] = False
    check("I3 rates", inv_verdicts(w)["I3"] == A.ERROR)
    w = world()
    w.s2["H4"][130]["conditions"]["compass"]["counts_ok"] = False
    check("I3 stage-2 record", inv_verdicts(w)["I3"] == A.ERROR)
    w = world()
    w.s1[A.REF][3]["conditions"]["compass"]["topology"]["shape"] = "sphere"     # stored label differs from its bars
    check("I5 relabel", inv_verdicts(w)["I5"] == A.ERROR)
    w = world()
    w.s2["ref"][140]["conditions"]["polar"]["topology"]["shape"] = "rp2"
    check("I5 relabel in stage 2", inv_verdicts(w)["I5"] == A.ERROR)
    w = world()
    w.s1[A.REF][3]["conditions"]["compass"]["finite"] = False
    check("I5 finite", inv_verdicts(w)["I5"] == A.ERROR)
    w = world()
    w.s1[A.REF][4]["conditions"]["compass"]["antipodal"]["E_odd"] = float("inf")
    check("I5 inf inside", inv_verdicts(w)["I5"] == A.ERROR)
    w = world()
    w.s1[A.REF][4]["conditions"]["compass"]["antipodal"]["verdict"] = "maybe"
    check("I5 verdict name", inv_verdicts(w)["I5"] == A.ERROR)
    for n_cells, want in ((A.M3.MIN_CELLS_TOPOLOGY - 1, A.OK), (A.M3.MIN_CELLS_TOPOLOGY, A.ERROR)):
        w = world()
        w.s1[A.fname(0.1, 25, "tetrode")][2]["conditions"]["compass"]["topology"] = dict(shape="too_few", n_cells=n_cells)
        check(f"I5 too_few with {n_cells} cells", inv_verdicts(w)["I5"] == want)


def test_q1_q4():
    w = world()
    w.set_label(1, A.REF, "rp2", seq(10, 8, "rp2", "other"))
    check("Q1 rp2 8/10 passes", verdicts(w.analyse(), "exp")["Q1"] == A.MET)
    w = world()
    w.set_label(1, A.REF, "rp2", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("Q1 rp2 7/10 fails", verdicts(r, "exp")["Q1"] == A.NOT_MET)
    check("closed reference gate closes H1", verdicts(r, "hyp")["H1"] == A.UNTESTABLE)
    check("closed reference gate closes H5", verdicts(r, "hyp")["H5"] == A.UNTESTABLE)
    check("H3 has its own gate", verdicts(r, "hyp")["H3"] == A.MET, r["hyp"]["H3"])
    check("H4 needs only Q4", verdicts(r, "hyp")["H4"] == A.MET, r["hyp"]["H4"])
    check("H2 not gated", verdicts(r, "hyp")["H2"] == A.MET)
    w = world()
    w.set_label(1, A.REF, "point", seq(10, 9, "point", "other"))
    check("Q1 point 9/10 ok", verdicts(w.analyse(), "exp")["Q1"] == A.MET)
    w = world()
    w.set_label(1, A.REF, "point", seq(10, 7, "point", "other"))
    r = w.analyse()
    check("Q1 point 7/10 fails", verdicts(r, "exp")["Q1"] == A.NOT_MET)
    check("a missed point closes no hypothesis", all(v == A.MET for v in verdicts(r, "hyp").values()), verdicts(r, "hyp"))
    w = world()
    w.set_label(1, A.REF, "ring", seq(10, 7, "ring", "other"))
    r = w.analyse()
    check("Q1 ring 7/10 fails", verdicts(r, "exp")["Q1"] == A.NOT_MET)
    check("a missed ring closes no hypothesis", all(v == A.MET for v in verdicts(r, "hyp").values()), verdicts(r, "hyp"))
    w = world()
    w.set_label(1, A.REF, "sphere", seq(10, 7, "sphere", "other"))
    check("Q1 sphere 7/10 fails", verdicts(w.analyse(), "exp")["Q1"] == A.NOT_MET)
    w = world()
    w.set_label(1, A.REF, "torus", seq(10, 0, "torus", "other"))
    check("Q1 ignores torus", verdicts(w.analyse(), "exp")["Q1"] == A.MET)
    w = world()
    w.set_ap(1, A.REF, "compass", seq(10, 9, "even", "polar"))
    r = w.analyse()
    check("Q4 even called polar", verdicts(r, "exp")["Q4"] == A.NOT_MET)
    check("Q4 gates H4", verdicts(r, "hyp")["H4"] == A.UNTESTABLE)
    check("Q4 gates H5", verdicts(r, "hyp")["H5"] == A.UNTESTABLE)
    check("Q4 leaves H1", verdicts(r, "hyp")["H1"] == A.MET)
    check("Q4 leaves H3", verdicts(r, "hyp")["H3"] == A.MET)
    w = world()
    w.set_ap(1, A.REF, "polar", seq(10, 9, "polar", "even"))
    check("Q4 polar called even", verdicts(w.analyse(), "exp")["Q4"] == A.NOT_MET)
    w = world()
    w.set_ap(1, A.REF, "compass", seq(10, 5, "even", "undetermined"))
    check("Q4 undetermined is fine", verdicts(w.analyse(), "exp")["Q4"] == A.MET)


def test_q2_q3():
    w = world()
    for name, recs in w.s1.items():
        if name.startswith("F_") and name.endswith("_p05"):
            for r in recs.values():
                n = r["design"]["n_tuned"]
                r["conditions"]["compass"]["n_untuned_selected"] = int(round(0.065 * 9 * n))
    check("Q2 untuned too high", verdicts(w.analyse(), "exp")["Q2"] == A.NOT_MET)
    for share, want in ((0.035, A.NOT_MET), (0.04, A.MET), (0.06, A.MET)):
        w = world()
        for name, recs in w.s1.items():
            if name.startswith("F_") and name.endswith("_p05"):
                for r in recs.values():
                    n = r["design"]["n_tuned"]
                    r["conditions"]["compass"]["n_untuned_selected"] = share * 9 * n
        check(f"Q2 untuned share {share}", verdicts(w.analyse(), "exp")["Q2"] == want, w.analyse()["exp"]["Q2"])
    w = world()
    for name, recs in w.s1.items():
        if name.startswith("F_d0.4_") and name.endswith("_p05"):
            for r in recs.values():
                r["conditions"]["compass"]["n_tuned_selected"] = int(round(0.7 * r["design"]["n_tuned"]))
    check("Q2 tuned at 0.4 too low", verdicts(w.analyse(), "exp")["Q2"] == A.NOT_MET)
    w = world()
    for name, recs in w.s1.items():
        if name.startswith("F_d0.1_") and name.endswith("_p05"):
            for r in recs.values():
                r["conditions"]["compass"]["n_tuned_selected"] = 0
    check("Q2 ignores tuned at 0.1", verdicts(w.analyse(), "exp")["Q2"] == A.MET)
    w = world()
    w.set_label(1, "Q3", "compass", seq(10, 7, "rp2", "other"))
    check("Q3 7/10", verdicts(w.analyse(), "exp")["Q3"] == A.NOT_MET)
    w = world()
    w.set_label(1, "Q3", "compass", seq(10, 8, "rp2", "other"))
    check("Q3 8/10", verdicts(w.analyse(), "exp")["Q3"] == A.MET)


def test_h1():
    w = world()
    w.set_label(2, "ref", "compass", seq(100, 79, "rp2", "other"))
    check("H1 79", verdicts(w.analyse(), "hyp")["H1"] == A.NOT_MET)
    w = world()
    w.set_label(2, "ref", "compass", seq(100, 80, "rp2", "other"))
    check("H1 80", verdicts(w.analyse(), "hyp")["H1"] == A.MET)
    w = world()
    w.set_label(2, "ref", "polar", seq(100, 79, "sphere", "other"))
    check("H1 polar 79", verdicts(w.analyse(), "hyp")["H1"] == A.NOT_MET)
    w = world()
    w.set_label(1, A.REF, "sphere", seq(10, 7, "sphere", "other"))
    r = w.analyse()
    check("H1 gate closed", verdicts(r, "hyp")["H1"] == A.UNTESTABLE)
    w = world()
    w.set_label(1, A.REF, "ring", seq(10, 0, "ring", "other"))
    r = w.analyse()
    check("H1 open with Q1 not met", verdicts(r, "exp")["Q1"] == A.NOT_MET and verdicts(r, "hyp")["H1"] == A.MET, r["hyp"]["H1"])


def test_h2():
    w = world()
    w.set_label(2, "ref", "polar", seq(100, 2, "rp2", "sphere"))
    check("H2 stage2 polar 2", verdicts(w.analyse(), "hyp")["H2"] == A.NOT_MET)
    w = world()
    w.set_label(2, "ref", "shuffled", seq(100, 1, "rp2", "point"))
    check("H2 stage2 shuffled 1", verdicts(w.analyse(), "hyp")["H2"] == A.MET)
    w = world()
    w.set_label(2, "ref", "shuffled", seq(100, 2, "rp2", "point"))
    check("H2 stage2 shuffled 2", verdicts(w.analyse(), "hyp")["H2"] == A.NOT_MET)
    w = world()
    w.set_label(1, A.mname(0.2, 50, "tetrode"), "polar", seq(10, 3, "rp2", "sphere"))
    r = w.analyse()
    check("H2 one cell 3/10", verdicts(r, "hyp")["H2"] == A.NOT_MET)
    check("H2 worst cell named", r["hyp"]["H2"]["worst_cell"]["cell"] == A.mname(0.2, 50, "tetrode"), r["hyp"]["H2"]["worst_cell"])
    w = world()
    w.set_label(1, A.mname(0.2, 50, "tetrode"), "polar", seq(10, 2, "rp2", "sphere"))
    check("H2 one cell 2/10", verdicts(w.analyse(), "hyp")["H2"] == A.MET)
    w = world()
    w.set_label(1, A.mname(0.2, 50, "tetrode"), "polar", seq(10, 2, "rp2", "sphere"))
    w.set_label(1, A.mname(0.7, 100, "sim"), "shuffled", seq(10, 2, "rp2", "point"))
    r = w.analyse()
    check("H2 tie names the first cell of the map", r["hyp"]["H2"]["worst_cell"]["cell"] == A.mname(0.2, 50, "tetrode"),
          r["hyp"]["H2"]["worst_cell"])
    # pooled share: 15 of 750 is 2% (met), 16 is above
    for k, want in ((15, A.MET), (16, A.NOT_MET)):
        w = world()
        names = [A.mname(d, n, m) for d in A.DELTAS for n in A.N_GRID for m in A.MODES]
        left = k
        for name in names:
            if left <= 0:
                break
            take = min(2, left)
            w.set_label(1, name, "shuffled", seq(10, take, "rp2", "point"))
            left -= take
        r = w.analyse()
        check(f"H2 pooled {k}", verdicts(r, "hyp")["H2"] == want, r["hyp"]["H2"]["map"])
    w = world()
    del w.s1[A.mname(0.7, 100, "probe")]
    check("H2 missing map cell", verdicts(w.analyse(), "hyp")["H2"] == A.NOT_MET)
    w = world()
    w.set_label(1, A.mname(1.0, 25, "sim"), "polar", seq(10, 1, "rp2", "sphere"))
    r = w.analyse()
    check("multiplicity counts cells", r["mult"]["cells_with_any_false_rp2"]["polar"] == 1, r["mult"])
    check("per-mode share", abs(r["hyp"]["H2"]["per_mode"]["sim"]["polar"]["share"] - 1 / 250) < 1e-12)


def test_h3():
    w = world()
    for n in (100, 200, 400):
        w.set_label(1, A.mname(0.4, n, "probe"), "compass", seq(10, 4, "rp2", "other"))
    r = w.analyse()
    check("H3 no point", r["pts"]["H3"].get("n_tuned") is None and verdicts(r, "hyp")["H3"] == A.UNTESTABLE, r["pts"])
    w = world()
    w.set_label(1, A.mname(0.4, 100, "probe"), "rp2", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("H3 skips a closed gate", r["pts"]["H3"].get("n_tuned") == 200 and verdicts(r, "hyp")["H3"] != A.UNTESTABLE,
          (r["pts"]["H3"], r["hyp"]["H3"]))
    w = world()
    for n in (100, 200, 400):
        w.set_label(1, A.mname(0.4, n, "probe"), "sphere", seq(10, 7, "sphere", "other"))
    r = w.analyse()
    check("H3 untestable when every gate is closed", r["pts"]["H3"].get("n_tuned") is None and
          verdicts(r, "hyp")["H3"] == A.UNTESTABLE, r["pts"]["H3"])
    w = world()
    w.set_label(1, A.mname(0.4, 100, "probe"), "compass", seq(10, 4, "rp2", "other"))
    r = w.analyse()
    check("H3 next N", r["pts"]["H3"].get("n_tuned") == 200, r["pts"]["H3"])
    w = world()
    w.set_label(2, "H3_sim", "compass", seq(100, 60, "rp2", "other"))
    check("H3 equal not met", verdicts(w.analyse(), "hyp")["H3"] == A.NOT_MET)
    w = world()
    w.set_label(2, "H3_sim", "compass", seq(100, 70, "rp2", "other"))
    check("H3 reversed not met", verdicts(w.analyse(), "hyp")["H3"] == A.NOT_MET)


def test_h4():
    w = world()
    w.set_label(1, A.mname(0.4, 100, "sim"), "compass", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("H4 N_t moves to 200", r["pts"]["H4"].get("n_t") == 200 and r["pts"]["H4"].get("n_tuned") == 100, r["pts"]["H4"])
    w = world()
    for n in A.N_GRID:
        w.set_label(1, A.mname(0.4, n, "sim"), "compass", seq(10, 7, "rp2", "other"))
    w.set_label(1, A.mname(0.7, 25, "sim"), "compass", seq(10, 8, "rp2", "other"))
    r = w.analyse()
    check("H4 delta 0.7, N_t 25 -> 12", r["pts"]["H4"] == dict(delta=0.7, n_t=25, n_tuned=12), r["pts"]["H4"])
    w = world()
    w.set_label(1, A.mname(0.4, 100, "sim"), "rp2", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("H4 needs the gate", r["pts"]["H4"].get("n_t") == 200, r["pts"]["H4"])
    w = world()
    for d in (0.4, 0.7, 1.0):
        for n in A.N_GRID:
            w.set_label(1, A.mname(d, n, "sim"), "compass", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("H4 no point", verdicts(r, "hyp")["H4"] == A.UNTESTABLE, r["pts"]["H4"])
    w = world()
    w.set_ap(2, "H4", "compass", seq(100, 79, "even", "undetermined"))
    check("H4 79", verdicts(w.analyse(), "hyp")["H4"] == A.NOT_MET)
    w = world()
    w.set_ap(2, "H4", "compass", seq(100, 80, "even", "undetermined"))
    check("H4 80", verdicts(w.analyse(), "hyp")["H4"] == A.MET)


def test_h5():
    w = world()
    w.set_ap(2, "H5", "compass", odd=seq(100, 79, True, False))
    check("H5 odd 79", verdicts(w.analyse(), "hyp")["H5"] == A.NOT_MET)
    w = world()
    w.set_label(2, "H5", "compass", seq(100, 80, "sphere", "rp2"))
    check("H5 sphere 80", verdicts(w.analyse(), "hyp")["H5"] == A.NOT_MET)
    w = world()
    w.set_label(2, "H5", "compass", seq(100, 79, "sphere", "rp2"))
    check("H5 sphere 79", verdicts(w.analyse(), "hyp")["H5"] == A.MET)
    w = world()
    for eps in A.EPS_LEVELS:
        w.set_label(1, f"E_eps{eps:g}", "compass", seq(10, 10, "rp2", "rp2"))
    r = w.analyse()
    check("H5 no eps_top is later", r["pts"]["H5"]["eps_top"] is None and verdicts(r, "hyp")["H5"] == A.MET, r["pts"]["H5"])
    w = world()
    w.set_label(1, "E_eps0.05", "compass", seq(10, 8, "sphere", "rp2"))
    r = w.analyse()
    check("H5 same eps not earlier", r["pts"]["H5"]["eps_top"] == 0.05 and verdicts(r, "hyp")["H5"] == A.NOT_MET, r["pts"]["H5"])
    w = world()
    for eps in A.EPS_LEVELS:
        w.set_ap(1, f"E_eps{eps:g}", "compass", odd=seq(10, 7, True, False))
    r = w.analyse()
    check("H5 no eps_ant", verdicts(r, "hyp")["H5"] == A.UNTESTABLE, r["pts"]["H5"])
    w = world()
    w.set_ap(1, "E_eps0.02", "compass", odd=seq(10, 8, True, False))
    check("H5 eps_ant 0.02", w.analyse()["pts"]["H5"]["eps_ant"] == 0.02)
    w = world()
    w.set_label(1, A.REF, "sphere", seq(10, 7, "sphere", "other"))
    r = w.analyse()
    check("H5 needs the reference gate", verdicts(r, "hyp")["H5"] == A.UNTESTABLE and verdicts(r, "exp")["Q4"] == A.MET,
          r["hyp"]["H5"])


def test_requirements_budgets():
    w = world()
    w.set_label(1, A.mname(0.4, 25, "tetrode"), "compass", seq(10, 8, "rp2", "other"))
    w.set_label(1, A.mname(0.4, 25, "tetrode"), "polar", seq(10, 2, "rp2", "sphere"))
    r = w.analyse()
    check("N80 skips dirty cell", r["req"]["M_f1"]["d0.4_tetrode"]["N80"] == 50, r["req"]["M_f1"]["d0.4_tetrode"])
    w = world()
    for n in A.N_GRID:
        w.set_label(1, A.mname(0.1, n, "probe"), "compass", seq(10, 4, "rp2", "other"))
    r = w.analyse()
    check("N80 above grid", r["req"]["M_f1"]["d0.1_probe"]["N80"] == ">400", r["req"]["M_f1"]["d0.1_probe"])
    w = world()
    w.set_label(1, A.mname(0.7, 25, "sim"), "rp2", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("N80 skips a closed gate", r["req"]["M_f1"]["d0.7_sim"]["N80"] == 50, r["req"]["M_f1"]["d0.7_sim"])
    w = world()
    w.set_label(1, A.mname(0.2, 25, "probe"), "compass", seq(10, 5, "rp2", "other"))
    w.set_label(1, A.mname(0.2, 50, "probe"), "compass", seq(10, 5, "rp2", "other"))
    r = w.analyse()
    check("N50 at 5 of 10", r["req"]["M_f1"]["d0.2_probe"]["N50"] == 25 and r["req"]["M_f1"]["d0.2_probe"]["N80"] == 100,
          r["req"]["M_f1"]["d0.2_probe"])
    w = world()
    w.set_label(1, A.mname(0.2, 25, "probe"), "compass", seq(10, 4, "rp2", "other"))
    r = w.analyse()
    check("N50 not at 4 of 10", r["req"]["M_f1"]["d0.2_probe"]["N50"] == 50, r["req"]["M_f1"]["d0.2_probe"])
    w = world()
    w.set_label(1, A.mname(0.4, 25, "tetrode"), "compass", seq(10, 7, "rp2", "other"))
    r = w.analyse()
    check("N80 not at 7 of 10", r["req"]["M_f1"]["d0.4_tetrode"]["N80"] == 50, r["req"]["M_f1"]["d0.4_tetrode"])
    w = world()
    for n in A.ANTI_GRID:
        w.set_ap(1, A.aname(0.2, n, "sim"), "compass", seq(10, 8 if n >= 25 else 7, "even", "undetermined"))
    r = w.analyse()
    check("N80 even in the short map", r["req"]["A_f1"]["d0.2_sim"]["N80_even"] == 25, r["req"]["A_f1"]["d0.2_sim"])
    check("N80 even default 6", r["req"]["A_f1"]["d0.4_sim"]["N80_even"] == 6, r["req"]["A_f1"]["d0.4_sim"])
    check("no topology keys in the short map", "N80" not in r["req"]["A_f1"]["d0.4_sim"])
    # budgets: B1 allows 150 recorded cells and 3 hours in one simultaneous session
    w = world()
    for n in A.N_GRID:
        for s, rec in w.s1[A.mname(0.1, n, "sim")].items():
            rec["cost"]["hours"] = 0.73
    r = w.analyse()
    b1 = r["bud"]["f1"]["B1"]
    check("B1 cheapest at 0.1 within 150 cells", b1["cheapest"]["d0.1"]["topology"]["cell"] == A.mname(0.1, 25, "sim"), b1["cheapest"]["d0.1"])
    check("B1 smallest delta", b1["smallest_delta"]["topology"] == 0.1, b1["smallest_delta"])
    w = world()
    for n, h in ((25, 2.931), (50, 2.920), (100, 2.910)):            # turn times make sessions differ at random
        for rec in w.s1[A.mname(0.1, n, "sim")].values():
            rec["cost"]["hours"] = h
    r = w.analyse()
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["topology"]
    check("cheapest ignores random session lengths", ch is not None and ch["cell"] == A.mname(0.1, 25, "sim"), ch)
    w = world()
    w.set_label(1, A.mname(0.1, 25, "sim"), "polar", seq(10, 2, "rp2", "sphere"))
    r = w.analyse()
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["topology"]
    check("cheapest needs clean controls", ch is not None and ch["cell"] == A.mname(0.1, 50, "sim"), ch)
    w = world()
    for n in (6, 12):
        w.set_ap(1, A.aname(0.1, n, "sim"), "compass", seq(10, 7, "even", "undetermined"))
    r = w.analyse()
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["antipodal"]
    check("shorter planned session before fewer cells", ch is not None and ch["cell"] == A.aname(0.1, 25, "sim"), ch)
    w = world()
    for n in A.N_GRID:
        for rec in w.s1[A.fname(0.4, n, "probe")].values():
            rec["cost"]["n_sessions"] = 1 if n == 100 else 2
    r = w.analyse()
    ch = r["bud"]["f0.1"]["B2"]["cheapest"]["d0.4"]["topology"]
    check("fewest sessions first", ch is not None and ch["cell"] == A.fname(0.4, 100, "probe"), ch)
    w = world()
    for n in (6, 12, 25, 50, 100):
        for rec in w.s1[A.mname(0.1, n, "sim")].values():
            rec["cost"]["hours"] = 3.5
    r = w.analyse()
    check("B1 hours limit", r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["topology"] is None, r["bud"]["f1"]["B1"]["cheapest"]["d0.1"])
    w = world()
    for n in A.N_GRID:
        for rec in w.s1[A.fname(0.4, n, "probe")].values():
            rec["cost"]["n_sessions"] = 11
    r = w.analyse()
    check("B2 session limit", r["bud"]["f0.1"]["B2"]["cheapest"]["d0.4"]["topology"] is None, r["bud"]["f0.1"]["B2"]["cheapest"]["d0.4"])
    r = BASE.analyse()
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["antipodal"]
    check("antipodal cheapest in the short map", ch is not None and ch["cell"].startswith("A_"), ch)
    ch = r["bud"]["f0.1"]["B3"]["cheapest"]["d0.1"]["antipodal"]
    check("antipodal cheapest at f 0.1 in AF", ch is not None and ch["cell"].startswith("AF_"), ch)
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["topology"]
    check("topology never from the short map", ch is not None and ch["cell"].startswith("M_"), ch)
    w = world()
    for n in A.ANTI_GRID:
        for rec in w.s1[A.aname(0.1, n, "sim")].values():
            rec["cost"]["n_recorded"] = 151
    r = w.analyse()
    ch = r["bud"]["f1"]["B1"]["cheapest"]["d0.1"]["antipodal"]
    check("B1 recorded limit in the short map", ch is not None and ch["cell"].startswith("M_"), ch)


# ----------------------------------------------------------------------------------------------------------------
def full_run(w):
    """Both stages exactly as mag_s3.py defines them, records from the world where it has them."""
    seeds1, seeds2 = A.STAGE1_SEEDS, A.STAGE2_SEEDS
    out = dict(s1={}, s2={})
    pts = A.working_points(w.s1)
    for stage, src, expected, seeds in ((1, w.s1, A.M3.stage1_cells(), seeds1), (2, w.s2, A.M3.stage2_cells(pts), seeds2)):
        cells = out[f"s{stage}"]
        for name, d in expected.items():
            dd = A.plain(A.asdict(d))
            if name in src:
                recs = copy.deepcopy(src[name])
            else:
                atlas = {x: x for x in dd["atlas"]}
                sel = (int(round(0.85 * dd["n_tuned"])), int(round(0.45 * dd["n_tuned"]))) if dd["f"] < 1 else (0, 0)
                recs = {s: record(s, dd, "rp2", "sphere", "point", atlas, ap("even"), ap("polar", True), sel=sel)
                        for s in seeds}
            for s, r in recs.items():
                r.update(design=dd, code=dict(A.M3.CODE_HASHES), cell=name, seed=s)
            cells[name] = recs
    out["inv1"], out["inv2"] = good_invariants(seeds1), good_invariants(seeds2)
    out["meta1"] = dict(mode="stage1", code_sha256=A.M3.CODE_HASHES["mag_s3"], seeds=seeds1, complete=True)
    out["meta2"] = dict(mode="stage2", code_sha256=A.M3.CODE_HASHES["mag_s3"], seeds=seeds2, complete=True, points=pts)
    out["pts"] = pts
    return out


FULL = full_run(BASE)


def issues1(**kw):
    r = dict(cells=FULL["s1"], inv=FULL["inv1"], meta=FULL["meta1"])
    r.update(kw)
    return A.check_stage(r["cells"], r["inv"], r["meta"], A.M3.stage1_cells(), A.STAGE1_SEEDS, "stage 1")


def issues2(pts=None, **kw):
    r = dict(cells=FULL["s2"], inv=FULL["inv2"], meta=FULL["meta2"])
    r.update(kw)
    pts = FULL["pts"] if pts is None else pts
    return A.check_stage(r["cells"], r["inv"], r["meta"], A.M3.stage2_cells(pts), A.STAGE2_SEEDS, "stage 2",
                         points=pts)


def with_record(cells, name, which, **changes):
    c = dict(cells)
    c[name] = dict(c[name])
    c[name][which] = dict(c[name][which], **changes)
    return c


def test_validation():
    check("valid stage 1", issues1() == [], issues1())
    check("valid stage 2", issues2() == [], issues2())
    c = dict(FULL["s1"])
    del c[A.mname(0.4, 100, "probe")]
    check("missing cell", any("missing" in x for x in issues1(cells=c)), issues1(cells=c))
    c = dict(FULL["s1"])
    c["X_extra"] = c[A.REF]
    check("extra cell", any("does not define" in x for x in issues1(cells=c)))
    c = dict(FULL["s1"])
    c["Q3"] = {k: v for k, v in c["Q3"].items() if k != 4}
    check("missing bird", any("birds differ" in x for x in issues1(cells=c)))
    c = dict(FULL["s1"])
    q = dict(c["Q3"])
    q[11] = dict(q[10], seed=11)
    c["Q3"] = q
    check("extra bird", any("birds differ" in x for x in issues1(cells=c)))
    d = dict(FULL["s1"]["Q3"][3]["design"], delta=0.5)
    check("other depth in Q3", any("design differs" in x for x in issues1(cells=with_record(FULL["s1"], "Q3", 3, design=d))))
    d = dict(FULL["s1"][A.REF][3]["design"], atlas=["rp2", "sphere"])
    check("other atlas in the reference", any("design differs" in x for x in issues1(cells=with_record(FULL["s1"], A.REF, 3, design=d))))
    code = dict(A.M3.CODE_HASHES, mag_s3="0" * 64)
    check("record of other code", any("other code" in x for x in issues1(cells=with_record(FULL["s1"], A.REF, 5, code=code))))
    check("record without code", any("other code" in x for x in issues1(cells=with_record(FULL["s1"], A.REF, 5, code=None))))
    check("record filed under another seed", any("another cell or seed" in x for x in issues1(cells=with_record(FULL["s1"], A.REF, 5, seed=6))))
    check("record filed under another cell", any("another cell or seed" in x for x in issues1(cells=with_record(FULL["s1"], A.REF, 5, cell="Q3"))))
    check("metadata missing", any("metadata missing" in x for x in issues1(meta=None)))
    check("run not complete", any("complete" in x for x in issues1(meta=dict(FULL["meta1"], complete=False))))
    check("metadata of other code", any("other code" in x for x in issues1(meta=dict(FULL["meta1"], code_sha256="0" * 64))))
    check("metadata of other birds", any("other birds" in x for x in issues1(meta=dict(FULL["meta1"], seeds=[1, 2]))))
    inv = dict(FULL["inv1"], per_seed={k: v for k, v in FULL["inv1"]["per_seed"].items() if k != "4"})
    check("invariants miss a bird", any("cover exactly" in x for x in issues1(inv=inv)))
    check("invariants missing", any("invariants.json missing" in x for x in issues1(inv=None)))
    check("invariants of other code", any("other code" in x for x in issues1(inv=dict(FULL["inv1"], code={}))))
    pts = copy.deepcopy(FULL["pts"])
    pts["H4"]["n_tuned"] = 100
    check("stage 2 at other working points", any("other working points" in x for x in issues2(pts=pts)), issues2(pts=pts))
    check("stage-2 designs follow the points", any("design differs" in x for x in issues2(pts=pts)))
    pts = copy.deepcopy(FULL["pts"])
    pts["H5"]["eps_ant"] = 0.1
    check("H5 at another eps", any("design differs" in x for x in issues2(pts=pts)))
    pts = copy.deepcopy(FULL["pts"])
    pts["H3"] = dict(n_tuned=None, reason="x")
    check("H3 cells without a point", any("does not define" in x for x in issues2(pts=pts)))


def write_full(root, run):
    d1, d2 = os.path.join(root, "s1"), os.path.join(root, "s2")
    for folder, cells, inv, meta in ((d1, run["s1"], run["inv1"], run["meta1"]), (d2, run["s2"], run["inv2"], run["meta2"])):
        for name, recs in cells.items():
            os.makedirs(os.path.join(folder, name), exist_ok=True)
            for s, r in recs.items():
                with open(os.path.join(folder, name, f"seed_{s}.json"), "w") as fh:
                    json.dump(r, fh)
        with open(os.path.join(folder, "invariants.json"), "w") as fh:
            json.dump(inv, fh)
        with open(os.path.join(folder, f"meta_{meta['mode']}_20261107T000000Z.json"), "w") as fh:
            json.dump(meta, fh)
    return d1, d2


def test_main_disk():
    root = tempfile.mkdtemp(prefix="mag_s3_test_")
    try:
        d1, d2 = write_full(root, FULL)
        quiet = open(os.devnull, "w")
        old = sys.stdout
        sys.stdout = quiet
        try:
            rc = A.main(["points", d1])
        finally:
            sys.stdout = old
        check("points accepts a complete stage 1", rc == 0 and os.path.exists(os.path.join(d1, "working_points.json")))
        sys.stdout = quiet
        try:
            rc = A.main(["final", d1, d2])
        finally:
            sys.stdout = old
        with open(os.path.join(d2, "results.json")) as fh:
            res = json.load(fh)
        check("final on a complete run", rc == 0 and res["valid"] and res["confirmatory"] and res["input_issues"] == [],
              (res["valid"], res["input_issues"], {k: v["verdict"] for k, v in res["invariants"].items()}))
        check("final reaches every hypothesis", all(v["verdict"] == A.MET for v in res["hypotheses"].values()),
              {k: v["verdict"] for k, v in res["hypotheses"].items()})
        r = A.run_final(d1, d2, A.STAGE1_SEEDS, A.STAGE2_SEEDS)
        check("run_final agrees with main", r["hypotheses"] == res["hypotheses"])
        path = os.path.join(d2, "H4", f"seed_{A.STAGE2_SEEDS[0]}.json")
        with open(path) as fh:
            rec = json.load(fh)
        rec["conditions"]["compass"]["counts_ok"] = False
        with open(path, "w") as fh:
            json.dump(rec, fh)
        r = A.run_final(d1, d2, A.STAGE1_SEEDS, A.STAGE2_SEEDS)
        check("stage-2 I3 error makes the run invalid", not r["valid"] and r["invariants"]["I3"]["verdict"] == A.ERROR)
        rec["conditions"]["compass"]["counts_ok"] = True
        with open(path, "w") as fh:
            json.dump(rec, fh)
        with open(os.path.join(d1, "working_points.json")) as fh:
            wp = json.load(fh)
        wp["H4"]["n_tuned"] = 25
        with open(os.path.join(d1, "working_points.json"), "w") as fh:
            json.dump(wp, fh)
        r = A.run_final(d1, d2, A.STAGE1_SEEDS, A.STAGE2_SEEDS)
        check("edited working points make the run invalid", not r["valid"] and
              any("working_points.json differs" in x for x in r["input_issues"]), r["input_issues"])
        check("an input problem alone makes the run invalid", all(v["verdict"] == A.OK for v in r["invariants"].values()),
              {k: v["verdict"] for k, v in r["invariants"].items()})
        r = A.run_final(d1, d2, [1, 2], A.STAGE2_SEEDS)
        check("other birds are not confirmatory", not r["confirmatory"] and not r["valid"])
        shutil.rmtree(os.path.join(d1, A.mname(0.4, 100, "sim")))
        os.remove(os.path.join(d1, "working_points.json"))
        sys.stdout = quiet
        try:
            rc = A.main(["points", d1])
        finally:
            sys.stdout = old
        check("points refuses an incomplete stage 1", rc == 1 and not os.path.exists(os.path.join(d1, "working_points.json")))
    finally:
        shutil.rmtree(root)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILS:
        print(f"{len(FAILS)} failed")
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
