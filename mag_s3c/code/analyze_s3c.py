#!/usr/bin/env python3
"""
MAG-S3c analysis: invariants I1-I7, expectations Q1-Q4, hypotheses H1-H5 and maps E1-E10 of protocol v0.3, from the
outputs of mag_s3c.py in one run folder (pair_done.json, receptor.json, checks.json, curves.json). The verdicts follow
the frozen rules without manual steps, and every verdict carries the numbers it follows from. results.json holds
everything and is written last, so its presence marks a finished analysis; the table tables/cells.csv repeats the map
of the pair E1 cell by cell. A run with missing cells gets results_incomplete.json and exit code 1.

Usage: python3 -I analyze_s3c.py --run RUN --out RESULTS
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                              # the analysis pins BLAS itself (analyze_s3b.py did not)

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mag_s3c():
    if "mag_s3c" in sys.modules:
        return sys.modules["mag_s3c"]
    spec = importlib.util.spec_from_file_location("mag_s3c", os.path.join(HERE, "mag_s3c.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3c"] = mod
    spec.loader.exec_module(mod)
    return mod


c = load_mag_s3c()
b = c.b

# Thresholds of the protocol
TOL_EXACT = 1e-12                        # I2, I3, I7
TOL_CONSERVATION = 1e-10                 # I4
TOL_RESIDUAL = 1e-10                     # I5, every solve
TOL_DENSE, TOL_STEPS = 1e-10, 1e-6       # I5, small pair
TOL_I6_YIELD, TOL_I6_DERIVED = 1e-12, 1e-6
Q1_EVEN, Q1_DIPOLE = 0.01, 0.01
Q2_REL = 0.10
Q4_TOL = 1e-5
H1_RATIO, H2_RATIO, H3_F, H4_RATIO, H5_SHARE = 10.0, 0.5, 0.01, 0.5, 0.10
DENOM_FLOOR = 1e-9                       # denominators below 1e-9 of the even range of the molecule are excluded
PROTOCOL_RATES = (5.0e6, 3.4e5)
DIPOLE_THRESHOLD, ANTIPODAL_THRESHOLD = 0.01, 0.05
NULL_BOUND = {"R1": 0.0010, "R2": 0.0022, "R3": 0.0083}
H1_LEVELS = (("none", 0.0), ("rf", 1e5), ("rf", 1e6), ("std", 1e5), ("std", 1e6))
STRONG = (("rf", 1e7), ("std", 1e7))
MAIN_COMBOS = tuple((g, ax) for g in c.GEOMETRIES for ax in c.MAIN_AXES[g])
N_COMPARISONS = dict(H1=40, H2=24, H3=6, H4=32, H5=48)


def ratio(a, z):
    """a / z, or None when z is zero."""
    return float(a) / float(z) if z != 0 else None


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_between(u, v):
    """Angle between two vectors (with sign), degrees; None when one is zero."""
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))))


def inv(ok, **numbers):
    return dict(verdict="ok" if ok else "error", **numbers)


def comparison(label, value, threshold, sense, **extra):
    ok = value >= threshold if sense == ">=" else value <= threshold
    return dict(label=label, value=float(value), threshold=threshold, sense=sense, **{"pass": bool(ok)}, **extra)


def smallest_margin(comparisons):
    """Value over threshold (>=) or threshold over value (<=); below 1 the comparison fails."""
    out = []
    for x in comparisons:
        v, t = x["value"], x["threshold"]
        out.append(v / t if x["sense"] == ">=" else (t / v if v > 0 else math.inf))
    return float(min(out))


def verdict_rule(comparisons, excluded=()):
    """met when every comparison passes, not_met when one fails, untestable when none remains."""
    if not comparisons:
        return dict(verdict="untestable", reason="no comparison left after the denominator rule", n=0, n_failed=0,
                    smallest_margin=None, comparisons=[], excluded=list(excluded))
    ok = all(x["pass"] for x in comparisons)
    return dict(verdict="met" if ok else "not_met", n=len(comparisons),
                n_failed=sum(not x["pass"] for x in comparisons), smallest_margin=smallest_margin(comparisons),
                comparisons=comparisons, excluded=list(excluded))


class Grid:
    """Rows of receptor.json by cell name."""

    def __init__(self, rows):
        self.rows = {r["name"]: r for r in rows}

    def get(self, system, axis, relax, k, form, chi, m=1.0, rates="cry4a"):
        return self.rows.get(c.Cell(system, axis, relax, k, form, chi, m, rates).name)

    @staticmethod
    def F(row, S=1.0, p=1.0):
        return float(row["receptor"]["by_order"][f"S{S:g}_p{p:g}"]["eps_eff"])

    @staticmethod
    def dz(row):
        return float(row["d_odd"][2])

    @staticmethod
    def eps(row):
        return float(row["parts"]["eps_mol"])

    @staticmethod
    def even(row):
        return float(row["parts"]["even_range"])


# --------------------------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------------------------
def steps_by_other_code(rec, chk, curves):
    """Steps written by another mag_s3c.py than the one present now."""
    now = c.file_sha256(os.path.join(HERE, "mag_s3c.py"))
    return [name for name, js in (("receptor", rec), ("checks", chk), ("curves", curves))
            if js["metadata"]["own"].get("mag_s3c.py") != now]


def invariants(rec, chk, curves):
    rows = rec["cells"]
    out = {}
    mism = chk["I1"]["frozen_mismatches"]
    code_ok = all(r["code_ok"] for r in rows)
    other = steps_by_other_code(rec, chk, curves)
    now = c.frozen_mismatches()
    out["I1"] = inv(not mism and not now and code_ok and not other, frozen_mismatches_at_checks=mism,
                    frozen_mismatches_now=now, cells_by_this_code=code_ok, steps_by_other_code=other)
    want = {key: {f"{j[1]}_{j[2]}_{j[3]}{j[4]:g}" for j in c.odd_check_jobs() if j[0] == key} for key in ("I2", "I3")}
    w2, w3 = max(chk["I2"].values()), max(chk["I3"].values())
    out["I2"] = inv(w2 < TOL_EXACT and set(chk["I2"]) == want["I2"], worst=w2, n=len(chk["I2"]),
                    n_expected=len(want["I2"]))
    out["I3"] = inv(w3 < TOL_EXACT and set(chk["I3"]) == want["I3"], worst=w3, n=len(chk["I3"]),
                    n_expected=len(want["I3"]))
    diags = ([r["diagnostics"] for r in rows] + [chk["diagnostics"]]
             + [v["K_up"]["solver_diagnostics"] for v in curves["cases"].values()])
    cons = max(d["max_conservation_error"] for d in diags)
    res = max(d["max_rel_residual"] for d in diags)
    out["I4"] = inv(cons < TOL_CONSERVATION, worst=cons)
    small = chk["I5_small"]
    cases = c.small_pair_cases()
    small_keys_ok = (set(small["dense"]) == {f"{j[1]}_{j[2]}{j[3]:g}" for j in cases if j[0] == "dense"}
                     and set(small["steps"]) == {f"{j[1]}_{j[2]}{j[3]:g}" for j in cases if j[0] == "steps"})
    out["I5"] = inv(res < TOL_RESIDUAL and small["max_dense"] < TOL_DENSE and small["max_steps"] < TOL_STEPS
                    and small_keys_ok, worst_residual=res, small_pair_vs_dense=small["max_dense"],
                    small_pair_vs_steps=small["max_steps"], small_pair_cases_complete=small_keys_ok,
                    max_gmres_iterations=max(d["max_gmres_iterations"] for d in diags))
    i6 = chk["I6"]
    wy = max(v["max_yield_diff"] for v in i6.values())
    wd = max(max(v["eps_mol_rel"], v["eps_rec_rel"], v["receptor_even_range_rel"], v["d_odd_rel"]) for v in i6.values())
    out["I6"] = inv(wy < TOL_I6_YIELD and wd < TOL_I6_DERIVED and set(i6) == {x.name for x in c.I6_CELLS},
                    worst_yield_diff=wy, worst_derived_rel=wd, n=len(i6))
    w7 = max(max(v["p0_max_odd"], v["p05_half_max_diff"]) for r in rows for v in r["I7"].values())
    out["I7"] = inv(w7 < TOL_EXACT, worst=w7)
    return out


# --------------------------------------------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------------------------------------------
def q1(rows):
    """Even residual at degree 10 at most 1% of its SD; d_odd changes by less than 1% from degree 10 to 24 (cells
    whose |d_odd| lies below 1e-9 of the even range are listed and left out of the dipole check)."""
    worst_even, worst_dip, excluded = 0.0, 0.0, []
    for r in rows:
        worst_even = max(worst_even, r["fit_even_rms_over_sd"])
        d10, d24 = np.array(r["d_odd"]), np.array(r["fit_check"]["d_odd"])
        if np.linalg.norm(d10) < DENOM_FLOOR * Grid.even(r):
            excluded.append(r["name"])
            continue
        worst_dip = max(worst_dip, float(np.linalg.norm(d24 - d10) / np.linalg.norm(d10)))
    odd10 = [r["fit_odd_rms_over_sd"] for r in rows if r["fit_odd_rms_over_sd"] is not None]
    odd24 = [r["fit_check"]["odd_rms_over_sd"] for r in rows if r["fit_check"]["odd_rms_over_sd"] is not None]
    return dict(verdict="met" if worst_even <= Q1_EVEN and worst_dip < Q1_DIPOLE else "not_met",
                worst_even_residual=worst_even, worst_dipole_change=worst_dip, excluded_from_dipole=excluded,
                odd_residual_L10=dict(median=float(np.median(odd10)), max=float(max(odd10))) if odd10 else None,
                odd_residual_L24=dict(median=float(np.median(odd24)), max=float(max(odd24))) if odd24 else None)


def q2(G):
    comps = []
    for g, ax in MAIN_COMBOS:
        for form, chi, m in c.FORM_COMBOS:
            r0 = G.get(g, ax, "none", 0.0, form, chi, m)
            for kind in ("rf", "std"):
                r1 = G.get(g, ax, kind, 1e5, form, chi, m)
                de, dF = ratio(G.eps(r1), G.eps(r0)), ratio(G.F(r1), G.F(r0))
                de = None if de is None else de - 1.0
                dF = None if dF is None else dF - 1.0
                ok = de is not None and dF is not None and abs(de) <= Q2_REL and abs(dF) <= Q2_REL
                comps.append(dict(cell=r1["name"], eps_mol_rel_change=de, F_rel_change=dF, **{"pass": ok}))
    worst = lambda key: max([abs(x[key]) for x in comps if x[key] is not None], default=None)  # noqa: E731
    return dict(verdict="met" if all(x["pass"] for x in comps) else "not_met", n=len(comps),
                worst_eps_mol_rel_change=worst("eps_mol_rel_change"), worst_F_rel_change=worst("F_rel_change"),
                failed=[x for x in comps if not x["pass"]])


def q3(rows):
    """The even range of the molecule falls with k in every cell of form K at m = 1, for each kind of relaxation:
    per system, axis and chi_b, the cell without relaxation followed by the cells of that kind in the order of k."""
    groups = {}
    for r in rows:
        cl = r["cell"]
        if cl["form"] != "K" or cl["m"] != 1.0:
            continue
        key = (cl["system"], cl["axis"], cl["chi"])
        for kind in (("rf", "std") if cl["relax"] == "none" else (cl["relax"],)):
            groups.setdefault(key + (kind,), []).append((cl["k"], Grid.even(r)))
    fails, n = [], 0
    for key, seq in sorted(groups.items()):
        seq.sort()
        if len(seq) < 2:
            continue
        n += 1
        if not all(a[1] > z[1] for a, z in zip(seq, seq[1:])):
            fails.append(dict(system=key[0], axis=key[1], chi=key[2], kind=key[3], k_and_even_range=seq))
    return dict(verdict="met" if n and not fails else "not_met", n=n, failed=fails)


def q4(curves):
    dev = {case: abs(v["K_up"]["forward_integral"] - v["K_up"]["solver_forward_yield"])
           for case, v in curves["cases"].items()}
    return dict(verdict="met" if dev and max(dev.values()) <= Q4_TOL else "not_met",
                worst=max(dev.values(), default=None), by_case=dev)


# --------------------------------------------------------------------------------------------------------------
# Hypotheses
# --------------------------------------------------------------------------------------------------------------
def h1(G, levels=H1_LEVELS):
    comps = []
    for g, ax in MAIN_COMBOS:
        for kind, k in levels:
            num = G.eps(G.get(g, ax, kind, k, "K", 0.05, 1.0))
            den = max(G.eps(G.get(g, ax, kind, k, "P-same", 0.1)), G.eps(G.get(g, ax, kind, k, "P-LH", 0.1)))
            comps.append(comparison(f"{g}_{ax}_{kind}{k:g}", num / den if den > 0 else math.inf, H1_RATIO, ">=",
                                    eps_K=num, eps_P_max=den))
    return verdict_rule(comps)


def h2(G):
    comps, excl = [], []
    for g in ("W_C", "W_D"):
        for chi in (0.05, 0.1):
            for kind, k in c.KEY_LEVELS:
                rz = G.get(g, "z", kind, k, "K", chi, 1.0)
                den = abs(G.dz(rz))
                for ax in ("x", "r"):
                    lab = f"{g}_{ax}_chi{chi:g}_{kind}{k:g}"
                    if den < DENOM_FLOOR * G.even(rz):
                        excl.append(dict(label=lab, denominator=den, even_range=G.even(rz)))
                        continue
                    num = abs(G.dz(G.get(g, ax, kind, k, "K", chi, 1.0)))
                    comps.append(comparison(lab, num / den, H2_RATIO, ">=", dz_axis=num, dz_z=den))
    return verdict_rule(comps, excl)


def h3(G):
    """Separate verdicts for W_C and W_D; each comparison carries how many times |F| lies below the threshold."""
    out = {}
    for g in ("W_C", "W_D"):
        comps = []
        for kind, k in c.KEY_LEVELS:
            F = abs(G.F(G.get(g, "r", kind, k, "K", 0.1, 1.0)))
            comps.append(comparison(f"{g}_r_{kind}{k:g}", F, H3_F, ">=",
                                    threshold_over_F=H3_F / F if F > 0 else math.inf))
        out[g] = verdict_rule(comps)
    return out


def h4(G, k_level=1e6):
    comps, excl = [], []
    for g, ax in MAIN_COMBOS:
        for chi in (0.05, 0.1):
            r0 = G.get(g, ax, "none", 0.0, "K", chi, 1.0)
            den = G.dz(r0)
            for kind in ("rf", "std"):
                lab = f"{g}_{ax}_chi{chi:g}_{kind}{k_level:g}"
                if abs(den) < DENOM_FLOOR * G.even(r0):
                    excl.append(dict(label=lab, denominator=den, even_range=G.even(r0)))
                    continue
                num = G.dz(G.get(g, ax, kind, k_level, "K", chi, 1.0))
                comps.append(comparison(lab, num / den, H4_RATIO, ">=", dz_relaxed=num, dz_none=den))
    return verdict_rule(comps, excl)


def halving(G):
    """For H4: d_odd . z of form K at m = 1 over its value without relaxation along the scale of k of each kind, and
    the pair of levels between which it first falls below one half (the levels of E10 included where they exist)."""
    out = {}
    for g, ax in MAIN_COMBOS:
        for chi in (0.05, 0.1):
            r0 = G.get(g, ax, "none", 0.0, "K", chi, 1.0)
            d0 = G.dz(r0)
            for kind in ("rf", "std"):
                if abs(d0) < DENOM_FLOOR * G.even(r0):
                    out[f"{g}_{ax}_chi{chi:g}_{kind}"] = dict(ratios=None, falls_below_half_between=None,
                                                              excluded="denominator below 1e-9 of the even range")
                    continue
                seq = []
                for k in (1e5, 3e5, 1e6, 3e6, 1e7):
                    r = G.get(g, ax, kind, k, "K", chi, 1.0)
                    if r is not None:
                        seq.append((k, ratio(G.dz(r), d0)))
                below = next((i for i, (_, v) in enumerate(seq) if v is None or v < H4_RATIO), None)
                bracket = None if below is None else [seq[below - 1][0] if below else 0.0, seq[below][0]]
                out[f"{g}_{ax}_chi{chi:g}_{kind}"] = dict(ratios=seq, falls_below_half_between=bracket)
    return out


def h5(G, levels=c.KEY_LEVELS):
    comps, excl = [], []
    for g, ax in MAIN_COMBOS:
        for chi in (0.05, 0.1):
            for kind, k in levels:
                rk = G.get(g, ax, kind, k, "K", chi, 1.0)
                ri = G.get(g, ax, kind, k, "K-inc", chi, 1.0)
                den = abs(G.dz(rk))
                lab = f"{g}_{ax}_chi{chi:g}_{kind}{k:g}"
                if den < DENOM_FLOOR * G.even(rk):
                    excl.append(dict(label=lab, denominator=den, even_range=G.even(rk)))
                    continue
                comps.append(comparison(lab, abs(G.dz(ri)) / den, H5_SHARE, "<=", dz_inc=G.dz(ri), dz_K=G.dz(rk),
                                        coherence_share=1.0 - G.dz(ri) / G.dz(rk)))
    return verdict_rule(comps, excl)


def hypotheses(G):
    hyp = dict(H1=h1(G), H2=h2(G), H3=h3(G), H4=h4(G), H5=h5(G))
    hyp["H4"]["halving"] = halving(G)
    return hyp


def comparison_counts(hyp):
    n = lambda h: h["n"] + len(h["excluded"])  # noqa: E731
    return dict(H1=n(hyp["H1"]), H2=n(hyp["H2"]), H3=sum(n(hyp["H3"][g]) for g in ("W_C", "W_D")), H4=n(hyp["H4"]),
                H5=n(hyp["H5"]))


# --------------------------------------------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------------------------------------------
def map_e1(G, rows):
    """Every cell: eps_mol, eps_rec, F, d_odd with its modulus and direction, G (projection on the receptor axis over
    the projection of the same cell with the CISS axis z), the even range and the share of the odd dipole on the
    receptor axis that the initial coherence gives, 1 - d_inc . z / d_K . z (cells of K at m = 1 with a control)."""
    out = []
    for r in rows:
        cl = r["cell"]
        d = np.array(r["d_odd"])
        row = dict(name=r["name"], branch=cl["branch"], system=cl["system"], axis=cl["axis"], relax=cl["relax"],
                   k=cl["k"], form=cl["form"], chi=cl["chi"], m=cl["m"], eps_mol=G.eps(r),
                   eps_rec=float(r["receptor"]["eps_rec"]), F=G.F(r), d_odd=d.tolist(),
                   d_odd_norm=float(np.linalg.norm(d)), d_odd_unit=unit(d).tolist(), d_odd_z=float(d[2]),
                   even_range=G.even(r), receptor_even_range=float(r["receptor"]["receptor_even_range"]),
                   odd_share_deg3_molecule=r["odd_power_shares"]["molecule"],
                   odd_share_deg3_receptor=r["odd_power_shares"]["receptor"],
                   odd_residual_L10=r["fit_odd_rms_over_sd"], odd_residual_L24=r["fit_check"]["odd_rms_over_sd"],
                   G=None, coherence_share=None)
        rz = G.get(cl["system"], "z", cl["relax"], cl["k"], cl["form"], cl["chi"], cl["m"])
        if rz is not None:
            row["G"] = ratio(G.dz(r), G.dz(rz))
        if cl["form"] == "K" and cl["m"] == 1.0:
            ri = G.get(cl["system"], cl["axis"], cl["relax"], cl["k"], "K-inc", cl["chi"], 1.0)
            if ri is not None and G.dz(r) != 0:
                row["coherence_share"] = 1.0 - G.dz(ri) / G.dz(r)
        out.append(row)
    return out


def map_e2(G, rows):
    """Admissible combinations of Delta chi (K) or chi (P), p and S for R1-R3 and visibility to the dipole and the
    antipodal tests, per geometry, axis and level of relaxation (the control K-inc left out); for the axis r the local
    estimate through the modulus of the slope s0 = F(m = 0.1) / (0.1 chi_b), which is no general bound."""
    regions = {}
    for r in rows:
        cl = r["cell"]
        if cl["form"] == "K-inc":
            continue
        key = f"{cl['system']}_{cl['axis']}_{cl['relax']}{cl['k']:g}"
        dchi = cl["m"] * cl["chi"] if cl["form"] == "K" else None
        reg = regions.setdefault(key, dict(entries=[]))
        for p in b.P_LEVELS:
            for S in b.S_LEVELS:
                v = r["receptor"]["by_order"][f"S{S:g}_p{p:g}"]
                e, e6 = float(v["eps_eff"]), float(v["eps_E6"])
                reg["entries"].append(dict(
                    cell=r["name"], form=cl["form"], chi_over_pi=cl["chi"], m=cl["m"], delta_chi_over_pi=dchi, p=p,
                    S=S, eps_eff=e, eps_E6=e6, admissible={pt: abs(e) <= bd for pt, bd in NULL_BOUND.items()},
                    admissible_E6={pt: abs(e6) <= bd for pt, bd in NULL_BOUND.items()},
                    dipole_visible=abs(e) >= DIPOLE_THRESHOLD, antipodal_visible=abs(e) >= ANTIPODAL_THRESHOLD))
    for reg in regions.values():
        en = reg["entries"]
        reg.update(n=len(en), n_admissible={pt: sum(x["admissible"][pt] for x in en) for pt in NULL_BOUND},
                   n_dipole_visible=sum(x["dipole_visible"] for x in en),
                   n_antipodal_visible=sum(x["antipodal_visible"] for x in en))
    slopes = {}
    for g in ("W_C", "W_D"):
        for kind, k in c.LEVELS:
            for chi in (0.05, 0.1):
                F01 = G.F(G.get(g, "r", kind, k, "K", chi, 0.1))
                F1 = G.F(G.get(g, "r", kind, k, "K", chi, 1.0))
                s0 = F01 / (0.1 * chi * math.pi)
                bound = {pt: (bd / abs(s0) / math.pi if s0 != 0 else None) for pt, bd in NULL_BOUND.items()}
                slopes[f"{g}_r_{kind}{k:g}_chi{chi:g}"] = dict(
                    F_m01=F01, F_m1=F1, s0_per_rad=s0, nonlinearity_F01_over_01F1=ratio(F01, 0.1 * F1),
                    local_estimate_dchi_p_g_over_pi=bound,
                    note="local estimate at small Delta chi; the admissible region is the general result")
    return dict(regions=regions, local_slopes=slopes, bounds=NULL_BOUND, dipole_threshold=DIPOLE_THRESHOLD,
                antipodal_threshold=ANTIPODAL_THRESHOLD)


def map_e3(G):
    out = {}
    for g in ("W_C", "W_D"):
        for ax in ("z", "x", "r"):
            for kind, k in c.KEY_LEVELS:
                rd, rn = G.get(g, ax, kind, k, "K", 0.1, 1.0), G.get(g + "_noD", ax, kind, k, "K", 0.1, 1.0)
                out[f"{g}_{ax}_{kind}{k:g}"] = dict(
                    eps_mol_D=G.eps(rd), eps_mol_noD=G.eps(rn), F_D=G.F(rd), F_noD=G.F(rn),
                    eps_mol_noD_over_D=ratio(G.eps(rn), G.eps(rd)), F_noD_over_D=ratio(G.F(rn), G.F(rd)),
                    d_odd_angle_deg=angle_between(rn["d_odd"], rd["d_odd"]))
    return out


def map_e4(rows):
    return {r["name"]: dict(r["E4"], eps_x_visible=abs(r["E4"]["eps_x"]) >= DIPOLE_THRESHOLD)
            for r in rows if "E4" in r}


def map_e5(G):
    out = {}
    for g in ("W_C", "W_D"):
        Fs = []
        for i in range(c.N_E5_AXES):
            ax = f"f{i:02d}"
            r = G.get(g, ax, "none", 0.0, "K", 0.1, 1.0)
            v = c.axis_vector(g, ax)
            Fs.append(G.F(r))
            out[f"{g}_{ax}"] = dict(axis=v.tolist(), axis_to_z_deg=c.angle_deg(v, np.array([0.0, 0.0, 1.0])),
                                    F=G.F(r), eps_mol=G.eps(r), d_odd=r["d_odd"],
                                    d_odd_unit=unit(r["d_odd"]).tolist(), dipole_visible=abs(G.F(r)) >= DIPOLE_THRESHOLD)
        out[f"{g}_summary"] = dict(F_min=min(Fs), F_max=max(Fs), abs_F_min=min(map(abs, Fs)),
                                   abs_F_max=max(map(abs, Fs)),
                                   n_dipole_visible=sum(abs(x) >= DIPOLE_THRESHOLD for x in Fs), n_axes=len(Fs))
    return out


def map_e6(rows):
    """Amplitude transfer: eps_E6 at every order and visibility to the dipole test through it."""
    out = {}
    for r in rows:
        by = {k: float(v["eps_E6"]) for k, v in r["receptor"]["by_order"].items()}
        out[r["name"]] = dict(eps_amplitude=float(r["zonal"][1] / b.m3.REN_B), eps_E6_by_order=by,
                              dipole_visible_S1_p1=abs(by["S1_p1"]) >= DIPOLE_THRESHOLD)
    return out


def map_e7(G):
    out = {}
    for g, var in (("W_C", "W_C_ErC"), ("W_D", "W_D_ErD")):
        for ax in ("z", "r"):
            for kind, k in c.KEY_LEVELS:
                rr, rg = G.get(g, ax, kind, k, "K", 0.1, 1.0), G.get(var, ax, kind, k, "K", 0.1, 1.0)
                out[f"{g}_{ax}_{kind}{k:g}"] = dict(
                    F_radicalpy=G.F(rr), F_gruning=G.F(rg), F_gruning_over_radicalpy=ratio(G.F(rg), G.F(rr)),
                    eps_mol_gruning_over_radicalpy=ratio(G.eps(rg), G.eps(rr)),
                    d_odd_angle_deg=angle_between(rg["d_odd"], rr["d_odd"]))
    return out


def map_e8(G):
    return dict(H1=h1(G, STRONG), H4=h4(G, 1e7), H5=h5(G, STRONG))


def map_e9(curves):
    """Per case: survival Tr rho_K, S-T coherence of K and memory of the initial coherence |rho_K - rho_inc|
    (Frobenius, over its value at t = 0) at chosen times; the contribution of the initial coherence to the odd yield
    along the receptor axis with the time at which half of it has accumulated; with relaxation the first time at
    which the memory falls to 1/e of the memory of the same pair without relaxation (effective coherence time)."""
    out = {}
    cases = curves["cases"]
    k_f = curves["rates_used"][0]
    marks = (1e-7, 2e-7, 5e-7, 1e-6, 2e-6)
    for case, v in sorted(cases.items()):
        t = np.array(v["K_up"]["t"])
        up, dn = v["delta_up"], v["delta_down"]
        rate = 0.5 * k_f * (np.array(up["trace"]) - np.array(dn["trace"]))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        total = 0.5 * (up["forward_integral"] - dn["forward_integral"])
        half_time = float(t[np.argmax(np.abs(cum) >= 0.5 * abs(cum[-1]))]) if cum[-1] != 0 else None
        norm = np.array(up["norm"])
        idx = {f"{tt:g}": int(np.argmin(np.abs(t - tt))) for tt in marks}
        out[case] = dict(
            odd_coherence_contribution=float(total), odd_half_accumulation_time_s=half_time,
            survival_at={key: float(v["K_up"]["trace"][i]) for key, i in idx.items()},
            st_coherence_at={key: float(v["K_up"]["st_coherence"][i]) for key, i in idx.items()},
            memory_at={key: float(norm[i] / norm[0]) for key, i in idx.items()},
            survival_end=float(v["K_up"]["trace"][-1]), effective_coherence_time_s=None)
    for g in ("W_C", "W_D"):
        base_case = f"{g}_none0"
        if base_case not in cases:
            continue
        base = np.array(cases[base_case]["delta_up"]["norm"])
        t = np.array(cases[base_case]["delta_up"]["t"])
        for kind in ("rf", "std"):
            key = f"{g}_{kind}{1e6:g}"
            if key not in cases:
                continue
            rel = np.array(cases[key]["delta_up"]["norm"]) / np.where(base > 0, base, np.nan)
            below = np.where(rel <= math.exp(-1.0))[0]
            out[key]["effective_coherence_time_s"] = float(t[below[0]]) if len(below) else None
    return out


E10_KS = (0.0, 1e5, 3e5, 1e6, 3e6, 1e7)


def e10_threshold(ks, mags, threshold=DIPOLE_THRESHOLD):
    """First crossing of |F_coherence| below the threshold on the k scale, log-linear between the bracketing levels."""
    if mags[0] is None or mags[0] < threshold:
        return dict(k_star=None, note="below the dipole threshold already without relaxation")
    if all(mv is not None and mv >= threshold for mv in mags):
        return dict(k_star=None, note="at or above the dipole threshold at every level up to 1e7 s^-1")
    j = next(i for i, mv in enumerate(mags) if mv is None or mv < threshold)
    k_lo, k_hi = ks[j - 1], ks[j]
    if k_lo > 0.0 and mags[j] is not None and mags[j] > 0.0:
        a, z = math.log(mags[j - 1]), math.log(mags[j])
        f = (math.log(threshold) - a) / (z - a)
        return dict(k_star=math.exp(math.log(k_lo) + f * (math.log(k_hi) - math.log(k_lo))), bracket=[k_lo, k_hi],
                    note="log-linear interpolation between the bracketing levels")
    return dict(k_star=None, bracket=[k_lo, k_hi], note="crossing between the bracketing levels, no interpolation")


def map_e10(G):
    """Rate k* at which F (1 - d_inc . z / d_K . z), the contribution of the initial coherence to F, falls below the
    dipole threshold on the half-decade scale of k."""
    out = {}
    for g in ("W_C", "W_D"):
        for kind in ("rf", "std"):
            levels = []
            for k in E10_KS:
                rl = "none" if k == 0.0 else kind
                rk, ri = G.get(g, "r", rl, k, "K", 0.1, 1.0), G.get(g, "r", rl, k, "K-inc", 0.1, 1.0)
                share = None if G.dz(rk) == 0 else 1.0 - G.dz(ri) / G.dz(rk)
                levels.append(dict(k=k, F_K=G.F(rk), F_K_inc=G.F(ri), coherence_share=share,
                                   F_coherence=None if share is None else G.F(rk) * share))
            mags = [None if x["F_coherence"] is None else abs(x["F_coherence"]) for x in levels]
            out[f"{g}_r_{kind}"] = dict(levels=levels, **e10_threshold(list(E10_KS), mags))
    return out


def maps(G, rows, curves):
    return dict(E1=map_e1(G, rows), E2=map_e2(G, rows), E3=map_e3(G), E4=map_e4(rows), E5=map_e5(G), E6=map_e6(rows),
                E7=map_e7(G), E8=map_e8(G), E9=map_e9(curves), E10=map_e10(G))


# --------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------
CSV_FIELDS = ("name", "branch", "system", "axis", "relax", "k", "form", "chi", "m", "eps_mol", "eps_rec", "F",
              "d_odd_z", "d_odd_norm", "even_range", "G", "coherence_share")


def write_table(path, e1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for row in e1:
            w.writerow(["" if row[f] is None else (repr(row[f]) if isinstance(row[f], float) else row[f])
                        for f in CSV_FIELDS])


def blas_single_thread(*metas):
    """Every step ran with one BLAS thread (protocol, «Окружение»)."""
    return all(m.get("blas") and all(x.get("num_threads") == 1 for x in m["blas"]) for m in metas)


def curves_structure_ok(curves):
    """The 18 curves of E9: the six cases, each with K_up, delta_up and delta_down."""
    want = {f"{g}_{rl}{k:g}" for g, rl, k in c.CURVE_CASES}
    return (set(curves["cases"]) == want
            and all(set(v) == {"K_up", "delta_up", "delta_down"} for v in curves["cases"].values()))


def curves_complete(curves):
    """The 18 curves with the length, step and output interval of the protocol."""
    return (curves_structure_ok(curves) and curves["t_max"] == c.CURVE_T and curves["dt"] == c.CURVE_DT
            and curves["every"] == c.CURVE_EVERY)


def analyze(rec, chk, curves, pair_done=None):
    rows = rec["cells"]
    G = Grid(rows)
    expected = {cl.name for cl in c.all_cells()}
    present = [r["name"] for r in rows]
    cells_complete = set(present) == expected and len(present) == len(expected)
    if not cells_complete or not curves_structure_ok(curves):
        return dict(valid=False, incomplete=True, cells_complete=cells_complete, curves_full=False,
                    curves_structure_ok=curves_structure_ok(curves), missing=sorted(expected - set(present)),
                    unexpected=sorted(set(present) - expected), duplicates=len(present) - len(set(present)),
                    metadata=c.metadata("analyze"))
    grid_ok = rec["grid_half"] == c.GRID_HALF and chk["grid_half"] == c.GRID_HALF
    rates_ok = (all(tuple(r["rates_used"]) == PROTOCOL_RATES for r in rows)
                and tuple(chk["rates_used"]) == PROTOCOL_RATES and tuple(curves["rates_used"]) == PROTOCOL_RATES)
    curves_ok = curves_complete(curves)
    meta = c.metadata("analyze")
    blas_ok = blas_single_thread(rec["metadata"], chk["metadata"], curves["metadata"], meta,
                                 *([pair_done["metadata"]] if pair_done else []))
    invs = invariants(rec, chk, curves)
    exp = dict(Q1=q1(rows), Q2=q2(G), Q3=q3(rows), Q4=q4(curves))
    hyp = hypotheses(G)
    valid = (all(v["verdict"] == "ok" for v in invs.values()) and grid_ok and rates_ok and cells_complete
             and curves_ok and blas_ok and pair_done is not None)
    return dict(valid=bool(valid), grid_ok=bool(grid_ok), rates_ok=bool(rates_ok),
                cells_complete=bool(cells_complete), curves_full=bool(curves_ok), blas_single_thread=bool(blas_ok),
                pair_done=pair_done is not None, invariants=invs, expectations=exp, hypotheses=hyp,
                comparisons=dict(counted=comparison_counts(hyp), protocol=N_COMPARISONS),
                multiplicity="deterministic hypotheses; no correction applies", maps=maps(G, rows, curves),
                metadata=meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loaded = {}
    for name in ("receptor", "checks", "curves", "pair_done"):
        path = os.path.join(a.run, f"{name}.json")
        if name == "pair_done" and not os.path.exists(path):
            loaded[name] = None
            continue
        with open(path) as fh:
            loaded[name] = json.load(fh)
    res = analyze(loaded["receptor"], loaded["checks"], loaded["curves"], loaded["pair_done"])
    os.makedirs(a.out, exist_ok=True)
    if res.get("incomplete"):
        c.write_json(os.path.join(a.out, "results_incomplete.json"), res)
        sys.exit(f"INVALID: {len(res['missing'])} cells missing, {len(res['unexpected'])} unexpected, curves "
                 f"{'complete' if res['curves_structure_ok'] else 'incomplete'}")
    write_table(os.path.join(a.out, "tables", "cells.csv"), res["maps"]["E1"])
    c.write_json(os.path.join(a.out, "results.json"), res)
    verdicts = {k: (v["verdict"] if "verdict" in v else {g: w["verdict"] for g, w in v.items()})
                for k, v in res["hypotheses"].items()}
    print("valid" if res["valid"] else "INVALID", verdicts, flush=True)


if __name__ == "__main__":
    main()
