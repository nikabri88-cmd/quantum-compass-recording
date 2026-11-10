#!/usr/bin/env python3
"""
MAG-S3c analysis: invariants I1-I11, expectations Q1-Q4, hypotheses H1-H5 with their verdicts per geometry, the rate
k_half with its classes and the maps E1-E14 of protocol v0.4, from the outputs of mag_s3c.py in one run folder
(pair_done.json, composite_done.json, receptor.json with maps_E5.npz and maps_E14.npz, checks.json, curves.json).
The verdicts follow the frozen rules without manual steps, and every verdict carries the numbers it follows from.
results.json holds everything and is written last, so its presence marks a finished analysis; tables/cells.csv
repeats the map E1 cell by cell. A run with missing cells gets results_incomplete.json and exit code 1.

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
TOL_EXACT = 1e-12                        # I2, I3, I7, I9, I10, I11
TOL_CONSERVATION = 1e-10                 # I4
TOL_RESIDUAL = 1e-10                     # I5, every solve
TOL_DENSE, TOL_STEPS = 1e-10, 1e-6       # I5, small pair and small composite
TOL_I6_YIELD, TOL_I6_DERIVED = 1e-12, 1e-6
TOL_REBUILD = 1e-10                      # I8
Q1_EVEN, Q1_DIPOLE = 0.01, 0.01
Q2_REL = 0.10
Q3_K_MAX = 1e7
Q4_TOL = 1e-5
H1_RATIO, H2_RATIO, H3_F, H4_RATIO, H5_SHARE = 10.0, 0.5, 0.01, 0.5, 0.9
DENOM_FLOOR = 1e-9                       # denominators below 1e-9 of the even range of the molecule are excluded
NEAR_ZERO = 0.05                         # H4, H5: flagged below 5% of the largest |d_odd . z| of the geometry
PROTOCOL_RATES = {"cry4a": (5.0e6, 3.4e5), "cr2_1e2": (1.0e6, 1.0e8), "cr2_1e3": (1.0e6, 1.0e9)}
DIPOLE_THRESHOLD, ANTIPODAL_THRESHOLD = 0.01, 0.05
NULL_BOUND = {"R1": 0.0010, "R2": 0.0022, "R3": 0.0083}
DEPHASING_CLASSES = (3e7, 6e7)           # measured dephasing of Golesworthy et al. 2023
H1_LEVELS = (("none", 0.0), ("rf", 1e5), ("rf", 1e6), ("std", 1e5), ("std", 1e6))
STRONG = (("rf", 1e7), ("rf", 3e7), ("std", 1e7), ("std", 3e7), ("std", 1e8))
MAIN_COMBOS = tuple((g, ax) for g in c.GEOMETRIES for ax in c.MAIN_AXES[g])
N_COMPARISONS = dict(H1=80, H2=12, H3=12, H4=32, H5=48)
N_KNOWN = dict(H1=4, H5=4)
SCALE = {"rf": (1e5, 3e5, 1e6, 3e6, 1e7, 3e7), "std": (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)}


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


def wmax(values):
    """Largest value, NaN when any value is not finite (a NaN must not vanish from a worst case)."""
    v = [float(x) for x in values]
    if not v:
        return math.inf
    return float("nan") if not all(math.isfinite(x) for x in v) else max(v)


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


def verdict_rule(comparisons, excluded=(), known=()):
    """met when every comparison passes, not_met when one fails, untestable when none remains; the comparisons known
    in advance from the published cells of MAG-S3b are reported and stay out of the verdict."""
    base = dict(excluded=list(excluded), known_in_advance=list(known))
    if not comparisons:
        return dict(verdict="untestable", reason="no comparison left after the denominator rule", n=0, n_failed=0,
                    smallest_margin=None, comparisons=[], **base)
    ok = all(x["pass"] for x in comparisons)
    return dict(verdict="met" if ok else "not_met", n=len(comparisons),
                n_failed=sum(not x["pass"] for x in comparisons), smallest_margin=smallest_margin(comparisons),
                comparisons=comparisons, **base)


class Grid:
    """Rows of receptor.json by cell name (direct and rebuilt cells; the composite pair apart)."""

    def __init__(self, rec):
        self.rows = {r["name"]: r for r in rec["cells"] if r["kind"] != "composite"}
        self.comp = {r["name"]: r for r in rec["cells"] if r["kind"] == "composite"}
        self.refs = rec["refs"]

    def get(self, system, axis, relax, k, form, chi, m=1.0, rates="cry4a"):
        return self.rows.get(c.Cell(system, axis, relax, k, form, chi, m, rates).name)

    def ref(self, system, relax, k, rates="cry4a"):
        return self.refs.get(c.Ref(system, relax, k, rates).name)

    @staticmethod
    def F(row, S=1.0, p=1.0):
        return float(row["receptor"]["by_order"][f"S{S:g}_p{p:g}"]["eps_eff"])

    @staticmethod
    def Famp(row, S=1.0, p=1.0):
        return float(row["receptor"]["by_order"][f"S{S:g}_p{p:g}"]["eps_E6"])

    @staticmethod
    def strict(row, S=1.0, p=1.0):
        """The stricter of the two transfers: the smaller modulus of F and F_amp."""
        return min(abs(Grid.F(row, S, p)), abs(Grid.Famp(row, S, p)))

    @staticmethod
    def dz(row):
        return float(row["d_odd"][2])

    @staticmethod
    def eps(row):
        return float(row["parts"]["eps_mol"] or 0.0)

    @staticmethod
    def even(row):
        return float(row["parts"]["even_range"])


def s_C(rk, ri):
    """Share of the coherence in the projection of the odd dipole on the receptor axis, 1 - d_inc . z / d_K . z."""
    return None if Grid.dz(rk) == 0 else 1.0 - Grid.dz(ri) / Grid.dz(rk)


# --------------------------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------------------------
def steps_by_other_code(rec, chk, curves):
    """Steps written by another mag_s3c.py than the one present now."""
    now = c.file_sha256(os.path.join(HERE, "mag_s3c.py"))
    return [name for name, js in (("receptor", rec), ("checks", chk), ("curves", curves))
            if js["metadata"]["own"].get("mag_s3c.py") != now]


def invariants(rec, chk, curves, G):
    rows = rec["cells"]
    refs = rec["refs"]
    out = {}
    mism = chk["I1"]["frozen_mismatches"]
    code_ok = all(r["code_ok"] for r in rows) and all(v["code_ok"] for v in refs.values())
    other = steps_by_other_code(rec, chk, curves)
    now = c.frozen_mismatches()
    out["I1"] = inv(not mism and not now and code_ok and not other, frozen_mismatches_at_checks=mism,
                    frozen_mismatches_now=now, cells_by_this_code=code_ok, steps_by_other_code=other)
    w2 = wmax(v["I2"] for v in refs.values())
    out["I2"] = inv(w2 < TOL_EXACT and len(refs) == len(c.references()), worst=w2, n=len(refs))
    want3 = {f"{j[1]}_{j[2]}_{j[3]}{j[4]:g}_{j[5]}" for j in c.odd_check_jobs()}
    w3 = wmax(chk["I3"].values())
    out["I3"] = inv(w3 < TOL_EXACT and set(chk["I3"]) == want3, worst=w3, n=len(chk["I3"]), n_expected=len(want3))
    diags = ([r["diagnostics"] for r in rows] + [v["diagnostics"] for v in refs.values()] + [chk["diagnostics"]]
             + [v["K_up"]["solver_diagnostics"] for v in curves["cases"].values()])
    cons = wmax(d["max_conservation_error"] for d in diags)
    res = wmax(d["max_rel_residual"] for d in diags)
    out["I4"] = inv(cons < TOL_CONSERVATION, worst=cons)
    small = chk["I5_small"]
    cases = c.small_pair_cases()
    keys_ok = (set(small["dense"]) == {f"{j[1]}_{j[2]}{j[3]:g}" for j in cases if j[0] == "dense"}
               and set(small["steps"]) == {f"{j[1]}_{j[2]}{j[3]:g}" for j in cases if j[0] == "steps"}
               and set(small["tiny"]) == {j[1].name for j in cases if j[0] == "tiny"})
    out["I5"] = inv(res < TOL_RESIDUAL and small["max_dense"] < TOL_DENSE and small["max_steps"] < TOL_STEPS
                    and small["max_tiny"] < TOL_DENSE and keys_ok, worst_residual=res,
                    small_pair_vs_dense=small["max_dense"], small_pair_vs_steps=small["max_steps"],
                    small_composite_vs_dense=small["max_tiny"], small_cases_complete=keys_ok,
                    max_gmres_iterations=max(d["max_gmres_iterations"] for d in diags))
    i6 = chk["I6"]
    wy = wmax(v["max_yield_diff"] for v in i6.values())
    wd = wmax(x for v in i6.values() for x in (v["eps_mol_rel"], v["eps_rec_rel"], v["receptor_even_range_rel"],
                                                v["d_odd_rel"]))
    out["I6"] = inv(wy < TOL_I6_YIELD and wd < TOL_I6_DERIVED and set(i6) == {x.name for x in c.I6_CELLS},
                    worst_yield_diff=wy, worst_derived_rel=wd, n=len(i6))
    w7 = wmax(x for r in rows for v in r["I7"].values() for x in (v["p0_max_odd"], v["p05_half_max_diff"]))
    out["I7"] = inv(w7 < TOL_EXACT, worst=w7)
    i8 = {k: v for r in refs.values() for k, v in r["I8"].items()}
    want8 = {x.name for x in c.direct_cells(("main",)) if x.form in ("K", "K-inc") and x.m == 1.0
             and (x.relax, x.k) in c.LEVELS_OLD}
    w8 = wmax(i8.values())
    out["I8"] = inv(w8 < TOL_REBUILD and set(i8) == want8, worst=w8, n=len(i8), n_expected=len(want8))
    iso = [r for r in G.rows.values() if r["cell"]["form"] == "K-iso"]
    w9 = wmax(r["parts"]["odd_half_range"] for r in iso)
    out["I9"] = inv(w9 < TOL_EXACT and len(iso) == sum(x.form == "K-iso" for x in c.rebuilt_cells()), worst=w9,
                    n=len(iso))
    c10 = [r for r in G.comp.values() if r["cell"]["form"] == "none" and r["cell"]["rec"] == "C"]
    w10 = wmax(r["parts"]["odd_half_range"] for r in c10)
    out["I10"] = inv(w10 < TOL_EXACT and len(c10) == 6, worst=w10, n=len(c10))
    cr = rec["crystal"]
    rel_plane, rel_mirror = 0.0, 0.0
    for v in cr.values():
        den = max(v["odd_max"], DENOM_FLOOR * v["even_range"])
        rel_plane = max(rel_plane, v["plane_odd_max"] / den)
        rel_mirror = max(rel_mirror, v["mirror_residual"] / den)
    want11 = {r["name"] for r in rows if r["kind"] == "composite" or r["cell"]["system"][:3] in ("W_C", "W_D")}
    out["I11"] = inv(rel_plane < TOL_EXACT and rel_mirror < TOL_EXACT and set(cr) == want11,
                     worst_plane_over_odd_max=rel_plane, worst_mirror_over_odd_max=rel_mirror, n=len(cr),
                     note="odd maximum floored at 1e-9 of the even range of the crystal (numerical zero)")
    return out


# --------------------------------------------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------------------------------------------
def q1(G):
    """Even residual at degree 10 at most 1% of its SD; d_odd changes by less than 1% from degree 10 to 24 (cells
    whose |d_odd| lies below 1e-9 of the even range are listed and left out of the dipole check). All cells on the
    grid of 2000 directions, direct and rebuilt."""
    worst_even, worst_dip, excluded = 0.0, 0.0, []
    for r in G.rows.values():
        worst_even = max(worst_even, r["fit_even_rms_over_sd"])
        d10, d24 = np.array(r["d_odd"]), np.array(r["fit_check"]["d_odd"])
        if np.linalg.norm(d10) < DENOM_FLOOR * Grid.even(r):
            excluded.append(r["name"])
            continue
        worst_dip = max(worst_dip, float(np.linalg.norm(d24 - d10) / np.linalg.norm(d10)))
    return dict(verdict="met" if worst_even <= Q1_EVEN and worst_dip < Q1_DIPOLE else "not_met",
                worst_even_residual=worst_even, worst_dipole_change=worst_dip, excluded_from_dipole=excluded,
                n=len(G.rows))


def q2(G):
    """At k = 1e5 of both kinds eps_mol and F of every cell within 10% of the values without relaxation. K-iso has no
    odd part (I9) and stays out."""
    comps = []
    for name, r1 in G.rows.items():
        cl = r1["cell"]
        if cl["k"] != 1e5 or cl["form"] == "K-iso":
            continue
        r0 = G.get(cl["system"], cl["axis"], "none", 0.0, cl["form"], cl["chi"], cl["m"], cl["rates"])
        if r0 is None:
            continue
        de, dF = ratio(G.eps(r1), G.eps(r0)), ratio(G.F(r1), G.F(r0))
        de = None if de is None else de - 1.0
        dF = None if dF is None else dF - 1.0
        ok = de is not None and dF is not None and abs(de) <= Q2_REL and abs(dF) <= Q2_REL
        comps.append(dict(cell=name, eps_mol_rel_change=de, F_rel_change=dF, **{"pass": ok}))
    worst = lambda key: max([abs(x[key]) for x in comps if x[key] is not None], default=None)  # noqa: E731
    return dict(verdict="met" if comps and all(x["pass"] for x in comps) else "not_met", n=len(comps),
                worst_eps_mol_rel_change=worst("eps_mol_rel_change"), worst_F_rel_change=worst("F_rel_change"),
                failed=[x for x in comps if not x["pass"]])


def q3(G):
    """The even range of the molecule falls with k up to 1e7 s^-1 in every cell of form K at m = 1, the branches E3,
    E7 and E10 included, for each kind of relaxation: per system, axis, chi_b and rates, the cell without relaxation
    followed by the cells of that kind in the order of k."""
    groups = {}
    for r in G.rows.values():
        cl = r["cell"]
        if cl["form"] != "K" or cl["m"] != 1.0 or cl["k"] > Q3_K_MAX or cl["rates"] != "cry4a":
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
# Hypotheses, verdicts per geometry
# --------------------------------------------------------------------------------------------------------------
def max_dz_of_geometry(G, e5, g, chi, kind, k):
    """Largest |d_odd . z| of form K at m = 1 over the CISS axes of map E5 (over the main axes without the map)."""
    ref = c.Ref(g, kind, k).name
    key = f"{ref}__chi{chi:g}_d_K"
    if key in e5:
        return float(np.abs(e5[key][:, 2]).max())
    rows = [G.get(g, ax, kind, k, "K", chi, 1.0) for ax in c.MAIN_AXES[g]]
    return max(abs(Grid.dz(r)) for r in rows if r is not None)


def h1(G, levels=H1_LEVELS, geoms=c.GEOMETRIES):
    out = {}
    for g in geoms:
        comps, known = [], []
        for ax in c.MAIN_AXES[g]:
            for kind, k in levels:
                rk = G.get(g, ax, kind, k, "K", 0.05, 1.0)
                rs, rl = G.get(g, ax, kind, k, "P-same", 0.1), G.get(g, ax, kind, k, "P-LH", 0.1)
                if rk is None or rs is None or rl is None:
                    continue
                for meas, fn in (("eps_mol", Grid.eps), ("F", lambda r: abs(Grid.F(r)))):
                    num, den = fn(rk), max(fn(rs), fn(rl))
                    x = comparison(f"{g}_{ax}_{kind}{k:g}_{meas}", num / den if den > 0 else math.inf, H1_RATIO, ">=",
                                   K=num, P_max=den, measure=meas)
                    (known if g == "S2" and kind == "none" else comps).append(x)
        out[g] = verdict_rule(comps, known=known)
    return out


def h2(G):
    out = {}
    for g in ("W_C", "W_D"):
        comps, excl, extra = [], [], {}
        for chi in (0.05, 0.1):
            for kind, k in c.KEY_LEVELS:
                lab = f"{g}_chi{chi:g}_{kind}{k:g}"
                circ = G.ref(g, kind, k)["H2_circle"][f"chi{chi:g}"]
                rz = G.get(g, "z", kind, k, "K", chi, 1.0)
                den = abs(circ["z_axis_dz"])
                for ax in ("x", "r"):
                    ra = G.get(g, ax, kind, k, "K", chi, 1.0)
                    extra[f"{lab}_{ax}"] = ratio(abs(Grid.dz(ra)), abs(Grid.dz(rz)))
                if den < DENOM_FLOOR * Grid.even(rz):
                    excl.append(dict(label=lab, denominator=den, even_range=Grid.even(rz)))
                    continue
                med = float(np.median(np.abs(circ["circle_dz"])))
                comps.append(comparison(lab, med / den, H2_RATIO, ">=", median_abs_dz_circle=med, abs_dz_z=den,
                                        n_axes=len(circ["circle_dz"])))
        out[g] = verdict_rule(comps, excl)
        out[g]["ratios_at_axes_x_and_r_without_verdict"] = extra
    return out


def h3(G):
    out = {}
    for g in ("W_C", "W_D"):
        comps = []
        for kind, k in c.KEY_LEVELS:
            r = G.get(g, "r", kind, k, "K", 0.1, 1.0)
            for meas, v in (("F", abs(Grid.F(r))), ("F_amp", abs(Grid.Famp(r)))):
                comps.append(comparison(f"{g}_r_{kind}{k:g}_{meas}", v, H3_F, ">=", measure=meas,
                                        threshold_over_value=H3_F / v if v > 0 else math.inf))
        out[g] = verdict_rule(comps)
        out[g]["threshold_over_stricter"] = {f"{kind}{k:g}": H3_F / G.strict(G.get(g, "r", kind, k, "K", 0.1, 1.0))
                                             for kind, k in c.KEY_LEVELS}
    return out


def h4(G, e5, k_level=1e6, geoms=c.GEOMETRIES, kinds=("rf", "std")):
    out = {}
    for g in geoms:
        comps, excl = [], []
        for ax in c.MAIN_AXES[g]:
            for chi in (0.05, 0.1):
                r0 = G.get(g, ax, "none", 0.0, "K", chi, 1.0)
                den = Grid.dz(r0)
                for kind in kinds:
                    lab = f"{g}_{ax}_chi{chi:g}_{kind}{k_level:g}"
                    r1 = G.get(g, ax, kind, k_level, "K", chi, 1.0)
                    if r1 is None:
                        continue
                    if abs(den) < DENOM_FLOOR * Grid.even(r0):
                        excl.append(dict(label=lab, denominator=den, even_range=Grid.even(r0)))
                        continue
                    near = abs(den) < NEAR_ZERO * max_dz_of_geometry(G, e5, g, chi, "none", 0.0)
                    comps.append(comparison(lab, Grid.dz(r1) / den, H4_RATIO, ">=", dz_relaxed=Grid.dz(r1),
                                            dz_none=den, near_zero_circle=bool(near)))
        out[g] = verdict_rule(comps, excl)
    return out


def h5(G, e5, levels=c.KEY_LEVELS, geoms=c.GEOMETRIES):
    out = {}
    for g in geoms:
        comps, excl, known = [], [], []
        for ax in c.MAIN_AXES[g]:
            for chi in (0.05, 0.1):
                for kind, k in levels:
                    rk, ri = G.get(g, ax, kind, k, "K", chi, 1.0), G.get(g, ax, kind, k, "K-inc", chi, 1.0)
                    if rk is None or ri is None:
                        continue
                    den = Grid.dz(rk)
                    lab = f"{g}_{ax}_chi{chi:g}_{kind}{k:g}"
                    if abs(den) < DENOM_FLOOR * Grid.even(rk):
                        excl.append(dict(label=lab, denominator=den, even_range=Grid.even(rk)))
                        continue
                    near = abs(den) < NEAR_ZERO * max_dz_of_geometry(G, e5, g, chi, kind, k)
                    x = comparison(lab, s_C(rk, ri), H5_SHARE, ">=", dz_inc=Grid.dz(ri), dz_K=den,
                                   near_zero_circle=bool(near))
                    (known if g == "S2" and kind == "none" else comps).append(x)
        out[g] = verdict_rule(comps, excl, known)
    return out


def k_half_of(seq, top):
    """k_half from [(k, |d . z|)] with k = 0 first: the first rate at which the modulus reaches half its value without
    relaxation, linear in lg k between neighbouring levels."""
    d0 = seq[0][1]
    if d0 == 0:
        return dict(k_half=None, note="no odd dipole without relaxation")
    r = [(k, v / d0) for k, v in seq[1:]]
    for i, (k, v) in enumerate(r):
        if v <= 0.5:
            if i == 0:
                return dict(k_half=None, below=k, note=f"below {k:g}")
            (k0, v0) = r[i - 1]
            f = (v0 - 0.5) / (v0 - v)
            return dict(k_half=float(10 ** (math.log10(k0) + f * (math.log10(k) - math.log10(k0)))), bracket=[k0, k])
    return dict(k_half=None, above=top, note=f"above {top:g}")


def k_half_class(res):
    lo, hi = DEPHASING_CLASSES
    if res.get("k_half") is None:
        return "below 3e7" if "below" in res else ("above 6e7" if "above" in res else None)
    v = res["k_half"]
    return "below 3e7" if v < lo else ("3e7 to 6e7" if v <= hi else "above 6e7")


def ref_dz(G, g, axis, chi, kind, k, which="d_K"):
    ref = G.ref(g, kind, k)
    if ref is None:
        return None
    return float(ref["main_axes"][f"{axis}_chi{chi:g}"][which][2])


def k_halves(G):
    """k_half for every geometry, axis, chi_b and kind on the scale of levels (with E10 in W_C and W_D), from the
    references (on the old levels they equal the direct cells, I8), and the class of the dephasing values."""
    out = {}
    for g, ax in MAIN_COMBOS:
        for chi in (0.05, 0.1):
            for kind in ("rf", "std"):
                seq = [(0.0, abs(ref_dz(G, g, ax, chi, "none", 0.0)))]
                for k in SCALE[kind]:
                    v = ref_dz(G, g, ax, chi, kind, k)
                    if v is not None:
                        seq.append((k, abs(v)))
                res = k_half_of(seq, SCALE[kind][-1])
                res["levels"] = seq
                if kind == "std":
                    res["class"] = k_half_class(res)
                out[f"{g}_{ax}_chi{chi:g}_{kind}"] = res
    return out


def hypotheses(G, e5):
    hyp = dict(H1=h1(G), H2=h2(G), H3=h3(G), H4=h4(G, e5), H5=h5(G, e5))
    hyp["H4"]["k_half"] = k_halves(G)
    return hyp


def comparison_counts(hyp):
    n = lambda h: h["n"] + len(h["excluded"]) + len(h["known_in_advance"])  # noqa: E731
    per = {key: sum(n(v) for g, v in hyp[key].items() if g in c.GEOMETRIES) for key in ("H1", "H2", "H3", "H4", "H5")}
    known = {key: sum(len(v["known_in_advance"]) for g, v in hyp[key].items() if g in c.GEOMETRIES)
             for key in ("H1", "H5")}
    return dict(per_hypothesis=per, total=sum(per.values()), known_in_advance=known)


# --------------------------------------------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------------------------------------------
def map_e1(G):
    """Every cell, direct and rebuilt: eps_mol, eps_rec, F, F_amp, d_odd with modulus and direction, G (projection on
    the receptor axis over the projection of the same cell at the CISS axis z), the even range, s_C (K at m = 1),
    and the ratios of K at m = 0.1 to the polarisation forms and of K to P-0."""
    out = []
    for r in G.rows.values():
        cl = r["cell"]
        d = np.array(r["d_odd"])
        row = dict(name=r["name"], source=cl["source"], branch=cl["branch"], system=cl["system"], axis=cl["axis"],
                   relax=cl["relax"], k=cl["k"], form=cl["form"], chi=cl["chi"], m=cl["m"], rates=cl["rates"],
                   eps_mol=G.eps(r), eps_rec=float(r["receptor"]["eps_rec"]), F=G.F(r), F_amp=G.Famp(r),
                   d_odd=d.tolist(), d_odd_norm=float(np.linalg.norm(d)), d_odd_unit=unit(d).tolist(),
                   d_odd_z=float(d[2]), even_range=G.even(r),
                   receptor_even_range=float(r["receptor"]["receptor_even_range"]),
                   odd_share_deg3_molecule=r["odd_power_shares"]["molecule"],
                   odd_share_deg3_receptor=r["odd_power_shares"]["receptor"], G=None, s_C=None,
                   K_over_P_eps=None, K_over_P_F=None, K_over_P0_eps=None, K_over_P0_F=None)
        rz = G.get(cl["system"], "z", cl["relax"], cl["k"], cl["form"], cl["chi"], cl["m"], cl["rates"])
        if rz is not None and abs(G.dz(rz)) >= DENOM_FLOOR * G.even(rz):
            row["G"] = ratio(G.dz(r), G.dz(rz))
        args = (cl["system"], cl["axis"], cl["relax"], cl["k"])
        if cl["form"] == "K":
            ri = G.get(*args, "K-inc", cl["chi"], 1.0, cl["rates"]) if cl["m"] == 1.0 else None
            if ri is not None and abs(G.dz(r)) >= DENOM_FLOOR * G.even(r):
                row["s_C"] = s_C(r, ri)
            ps = [G.get(*args, f, 0.1, 1.0, cl["rates"]) for f in ("P-same", "P-LH")]
            if all(p is not None for p in ps):
                row["K_over_P_eps"] = ratio(G.eps(r), max(G.eps(p) for p in ps))
                row["K_over_P_F"] = ratio(abs(G.F(r)), max(abs(G.F(p)) for p in ps))
            p0 = G.get(*args, "P-0", 0.1, 1.0, cl["rates"])
            if p0 is not None:
                row["K_over_P0_eps"] = ratio(G.eps(r), G.eps(p0))
                row["K_over_P0_F"] = ratio(abs(G.F(r)), abs(G.F(p0)))
        out.append(row)
    return out


def map_e2(G):
    """Admissible combinations of Delta chi (K) or chi (P), p and S for R1-R3 by the stricter transfer and visibility
    to the dipole and the antipodal tests by both transfers, per geometry, axis, level and rates (forms K, P-same and
    P-LH); for the axis r the local estimate through the modulus of the slope s0, which is no general bound."""
    regions = {}
    for r in G.rows.values():
        cl = r["cell"]
        if cl["form"] not in ("K", "P-same", "P-LH"):
            continue
        key = f"{cl['system']}_{cl['axis']}_{cl['relax']}{cl['k']:g}_{cl['rates']}"
        dchi = cl["m"] * cl["chi"] if cl["form"] == "K" else None
        reg = regions.setdefault(key, dict(entries=[]))
        for p in b.P_LEVELS:
            for S in b.S_LEVELS:
                e, e6 = G.F(r, S, p), G.Famp(r, S, p)
                strict = min(abs(e), abs(e6))
                reg["entries"].append(dict(
                    cell=r["name"], form=cl["form"], chi_over_pi=cl["chi"], m=cl["m"], delta_chi_over_pi=dchi, p=p,
                    S=S, eps_eff=e, eps_E6=e6, stricter=strict,
                    admissible={pt: strict <= bd for pt, bd in NULL_BOUND.items()},
                    dipole_visible=strict >= DIPOLE_THRESHOLD, antipodal_visible=strict >= ANTIPODAL_THRESHOLD))
    for reg in regions.values():
        en = reg["entries"]
        reg.update(n=len(en), n_admissible={pt: sum(x["admissible"][pt] for x in en) for pt in NULL_BOUND},
                   n_dipole_visible=sum(x["dipole_visible"] for x in en),
                   n_antipodal_visible=sum(x["antipodal_visible"] for x in en))
    slopes = {}
    for g in ("W_C", "W_D"):
        for kind, k in c.LEVELS_OLD:
            for chi in (0.05, 0.1):
                r01, r1 = G.get(g, "r", kind, k, "K", chi, 0.1), G.get(g, "r", kind, k, "K", chi, 1.0)
                entry = {}
                for meas, fn in (("F", G.F), ("F_amp", G.Famp)):
                    v01 = fn(r01)
                    s0 = v01 / (0.1 * chi * math.pi)
                    entry[meas] = dict(m01=v01, m1=fn(r1), s0_per_rad=s0, nonlinearity=ratio(v01, 0.1 * fn(r1)),
                                       local_estimate_dchi_p_g_over_pi={
                                           pt: (bd / abs(s0) / math.pi if s0 != 0 else None)
                                           for pt, bd in NULL_BOUND.items()})
                slopes[f"{g}_r_{kind}{k:g}_chi{chi:g}"] = dict(
                    entry, note="local estimate at small Delta chi; the admissible region is the general result")
    return dict(regions=regions, local_slopes=slopes, bounds=NULL_BOUND, dipole_threshold=DIPOLE_THRESHOLD,
                antipodal_threshold=ANTIPODAL_THRESHOLD)


def ratios_k_over(G, system, ax, kind, k, chi=0.05):
    rk = G.get(system, ax, kind, k, "K", chi, 1.0)
    ps = [G.get(system, ax, kind, k, f, 0.1) for f in ("P-same", "P-LH")]
    p0 = G.get(system, ax, kind, k, "P-0", 0.1)
    out = {}
    if rk is not None and all(p is not None for p in ps):
        out.update(K_over_P_eps=ratio(G.eps(rk), max(G.eps(p) for p in ps)),
                   K_over_P_F=ratio(abs(G.F(rk)), max(abs(G.F(p)) for p in ps)))
    if rk is not None and p0 is not None:
        out.update(K_over_P0_eps=ratio(G.eps(rk), G.eps(p0)), K_over_P0_F=ratio(abs(G.F(rk)), abs(G.F(p0))))
    return out


def map_e3(G):
    out = {}
    for g in ("W_C", "W_D"):
        for ax in ("z", "x", "r"):
            for kind, k in c.KEY_LEVELS:
                rd, rn = G.get(g, ax, kind, k, "K", 0.1, 1.0), G.get(g + "_noD", ax, kind, k, "K", 0.1, 1.0)
                out[f"{g}_{ax}_{kind}{k:g}"] = dict(
                    eps_mol_D=G.eps(rd), eps_mol_noD=G.eps(rn), F_D=G.F(rd), F_noD=G.F(rn), F_amp_D=G.Famp(rd),
                    F_amp_noD=G.Famp(rn), eps_mol_noD_over_D=ratio(G.eps(rn), G.eps(rd)),
                    F_noD_over_D=ratio(G.F(rn), G.F(rd)), F_amp_noD_over_D=ratio(G.Famp(rn), G.Famp(rd)),
                    d_odd_angle_deg=angle_between(rn["d_odd"], rd["d_odd"]),
                    ratios_D=ratios_k_over(G, g, ax, kind, k), ratios_noD=ratios_k_over(G, g + "_noD", ax, kind, k))
    return out


def map_e4(G):
    return {r["name"]: dict(r["E4"], eps_x_visible=abs(r["E4"]["eps_x"]) >= DIPOLE_THRESHOLD)
            for r in G.rows.values() if "E4" in r}


def map_e5(G, e5):
    """Map E5 over the CISS axes: per reference and chi_b the range of F, F_amp and eps_mol, the share of axes visible
    to the dipole test by the stricter transfer, s_C, the values at the axes to W_A-W_D, and k_half per axis."""
    out = {}
    for ref in c.e5_references():
        info = G.refs[ref.name]
        named = info.get("E5_named_axes", [])
        for chi in (0.05, 0.1):
            pre = f"{ref.name}__chi{chi:g}"
            F, Fa = e5[f"{pre}_F"], e5[f"{pre}_F_amp"]
            dk, di = e5[f"{pre}_d_K"], e5[f"{pre}_d_inc"]
            sC = np.where(dk[:, 2] != 0, 1.0 - di[:, 2] / np.where(dk[:, 2] != 0, dk[:, 2], 1.0), np.nan)
            strict = np.minimum(np.abs(F), np.abs(Fa))
            n_grid = len(F) - len(named)
            q = lambda a: dict(zip(("min", "q25", "median", "q75", "max"),  # noqa: E731
                                   np.nanquantile(a[:n_grid], [0, 0.25, 0.5, 0.75, 1]).tolist()))
            out[pre] = dict(F=q(F), F_amp=q(Fa), abs_stricter=q(strict), eps_mol=q(e5[f"{pre}_eps_mol"]), s_C=q(sC),
                            share_dipole_visible=float(np.mean(strict[:n_grid] >= DIPOLE_THRESHOLD)),
                            named_axes={t: dict(F=float(F[n_grid + i]), F_amp=float(Fa[n_grid + i]),
                                                s_C=float(sC[n_grid + i]), d_odd=dk[n_grid + i].tolist())
                                        for i, t in enumerate(named)})
    # k_half per CISS axis
    for g in c.GEOMETRIES:
        for chi in (0.05, 0.1):
            base = e5.get(f"{c.Ref(g, 'none', 0.0).name}__chi{chi:g}_d_K")
            if base is None:
                continue
            for kind in ("rf", "std"):
                ks = [k for k in SCALE[kind] if f"{c.Ref(g, kind, k).name}__chi{chi:g}_d_K" in e5]
                mags = np.array([np.abs(base[:, 2])] + [np.abs(e5[f"{c.Ref(g, kind, k).name}__chi{chi:g}_d_K"][:, 2])
                                                        for k in ks])
                res = [k_half_of([(0.0, mags[0, i])] + [(k, mags[j + 1, i]) for j, k in enumerate(ks)], ks[-1])
                       for i in range(mags.shape[1])]
                vals = np.array([x["k_half"] if x["k_half"] is not None else np.nan for x in res])
                entry = dict(levels=ks, n_axes=len(res), n_below_scale=sum("below" in x for x in res),
                             n_above_scale=sum("above" in x for x in res),
                             k_half_quantiles=(np.nanquantile(vals, [0, 0.25, 0.5, 0.75, 1]).tolist()
                                               if np.isfinite(vals).any() else None))
                if kind == "std":
                    classes = [k_half_class(x) for x in res]
                    entry["class_shares"] = {cl: classes.count(cl) / len(classes)
                                             for cl in ("below 3e7", "3e7 to 6e7", "above 6e7")}
                named = G.refs[c.Ref(g, "none", 0.0).name].get("E5_named_axes", [])
                n_grid = mags.shape[1] - len(named)
                entry["named_axes"] = {t: res[n_grid + i] for i, t in enumerate(named)}
                out[f"k_half_{g}_chi{chi:g}_{kind}"] = entry
    return out


CLASS_CODE = {"below 3e7": 0, "3e7 to 6e7": 1, "above 6e7": 2, None: -1}


def axis_maps(G, e5):
    """Per CISS axis of map E5: k_half (with the class code of the dephasing: 0 below 3e7, 1 from 3e7 to 6e7, 2 above
    6e7, -1 undefined) for every geometry, chi_b and kind, and for W_C and W_D the rate of E10 at which s_C F and
    s_C F_amp of K at chi_b = 0.1 pi, m = 1 fall below the dipole threshold (stricter transfer). Arrays for an npz
    beside results.json."""
    out = {}
    for g in c.GEOMETRIES:
        pre0 = f"{c.Ref(g, 'none', 0.0).name}__chi"
        if f"{pre0}0.1_d_K" not in e5:
            continue
        out[f"axes_{g}"] = e5[f"{c.Ref(g, 'none', 0.0).name}__axes"]
        for chi in (0.05, 0.1):
            for kind in ("rf", "std"):
                ks = [k for k in SCALE[kind] if f"{c.Ref(g, kind, k).name}__chi{chi:g}_d_K" in e5]
                key = lambda k: c.Ref(g, "none", 0.0).name if k == 0.0 else c.Ref(g, kind, k).name  # noqa: E731
                mags = np.array([np.abs(e5[f"{key(k)}__chi{chi:g}_d_K"][:, 2]) for k in [0.0] + ks])
                res = [k_half_of([(0.0, mags[0, i])] + [(k, mags[j + 1, i]) for j, k in enumerate(ks)], ks[-1])
                       for i in range(mags.shape[1])]
                out[f"k_half_{g}_chi{chi:g}_{kind}"] = np.array(
                    [x["k_half"] if x["k_half"] is not None else np.nan for x in res])
                out[f"k_half_code_{g}_chi{chi:g}_{kind}"] = np.array(
                    [0 if "below" in x else (3 if "above" in x else 1) for x in res])     # 0 below, 1 inside, 3 above
                if kind == "std":
                    out[f"k_half_class_{g}_chi{chi:g}"] = np.array([CLASS_CODE[k_half_class(x)] for x in res])
                if g in ("W_C", "W_D") and chi == 0.1:
                    stricter = []
                    for k in [0.0] + ks:
                        pre = f"{key(k)}__chi0.1"
                        dk, di = e5[f"{pre}_d_K"][:, 2], e5[f"{pre}_d_inc"][:, 2]
                        sc = np.where(dk != 0, 1.0 - di / np.where(dk != 0, dk, 1.0), np.nan)
                        stricter.append(np.minimum(np.abs(sc * e5[f"{pre}_F"]), np.abs(sc * e5[f"{pre}_F_amp"])))
                    stricter = np.array(stricter)
                    ks_all = [0.0] + ks
                    out[f"e10_k_star_{g}_{kind}"] = np.array([
                        (lambda r: r["k_star"] if r["k_star"] is not None else np.nan)(first_below(
                            ks_all, [None if not np.isfinite(v) else float(v) for v in stricter[:, i]]))
                        for i in range(stricter.shape[1])])
    return out


def rates_seen(rec, chk):
    """Every pair (k_F, k_R) used per rate set and every hop scale, for the post-check of the smoke and blind runs."""
    by = {}
    for r in rec["cells"]:
        by.setdefault(r["cell"]["rates"], set()).add(tuple(r["rates_used"]))
    for v in rec["refs"].values():
        by.setdefault(v["ref"]["rates"], set()).add(tuple(v["rates_used"]))
    for k, v in chk["rates_used"].items():
        by.setdefault(k, set()).add(tuple(v))
    scales = {float(chk["hop_scale"])} | {round(r["hop_rates_used"][0] / r["cell"]["k_cd"], 12)
                                          for r in rec["cells"] if r["kind"] == "composite"}
    return dict(by_set={k: sorted(list(x) for x in v) for k, v in by.items()}, hop_scale=sorted(scales))


def map_e6(G):
    out = {}
    for r in G.rows.values():
        by = {k: dict(F=float(v["eps_eff"]), F_amp=float(v["eps_E6"])) for k, v in r["receptor"]["by_order"].items()}
        out[r["name"]] = dict(F_amp_over_F=ratio(G.Famp(r), G.F(r)), by_order=by,
                              dipole_visible_S1_p1=G.strict(r) >= DIPOLE_THRESHOLD)
    return out


def map_e7(G):
    out = {}
    for g, var in (("W_C", "W_C_ErC"), ("W_D", "W_D_ErD")):
        for ax in ("z", "r"):
            for kind, k in c.KEY_LEVELS:
                rr, rg = G.get(g, ax, kind, k, "K", 0.1, 1.0), G.get(var, ax, kind, k, "K", 0.1, 1.0)
                out[f"{g}_{ax}_{kind}{k:g}"] = dict(
                    F_radicalpy=G.F(rr), F_gruning=G.F(rg), F_amp_radicalpy=G.Famp(rr), F_amp_gruning=G.Famp(rg),
                    F_gruning_over_radicalpy=ratio(G.F(rg), G.F(rr)),
                    F_amp_gruning_over_radicalpy=ratio(G.Famp(rg), G.Famp(rr)),
                    eps_mol_gruning_over_radicalpy=ratio(G.eps(rg), G.eps(rr)),
                    d_odd_angle_deg=angle_between(rg["d_odd"], rr["d_odd"]))
    return out


def map_e8(G, e5):
    return dict(H1={f"{kind}{k:g}": h1(G, ((kind, k),), ("W_C", "W_D") if k > 1e7 else c.GEOMETRIES)
                    for kind, k in STRONG},
                H4={f"{kind}{k:g}": h4(G, e5, k, kinds=(kind,)) for kind, k in STRONG},
                H5={f"{kind}{k:g}": h5(G, e5, ((kind, k),)) for kind, k in STRONG})


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
        for key in (f"{g}_rf{1e6:g}", f"{g}_std{1e6:g}", f"{g}_std{3e7:g}"):
            if key not in cases:
                continue
            rel = np.array(cases[key]["delta_up"]["norm"]) / np.where(base > 0, base, np.nan)
            below = np.where(rel <= math.exp(-1.0))[0]
            out[key]["effective_coherence_time_s"] = float(t[below[0]]) if len(below) else None
    return out


def first_below(ks, mags, threshold=DIPOLE_THRESHOLD):
    """First crossing of the magnitude below the threshold on the scale of k, linear in lg k between the levels."""
    if mags[0] is None:
        return dict(k_star=None, note="share of the coherence undefined without relaxation (d_K . z = 0)")
    if mags[0] < threshold:
        return dict(k_star=None, note="below the dipole threshold already without relaxation")
    if all(mv is not None and mv >= threshold for mv in mags):
        return dict(k_star=None, note=f"at or above the dipole threshold up to {ks[-1]:g} s^-1")
    j = next(i for i, mv in enumerate(mags) if mv is None or mv < threshold)
    k_lo, k_hi = ks[j - 1], ks[j]
    if k_lo > 0.0 and mags[j] is not None:
        f = (mags[j - 1] - threshold) / (mags[j - 1] - mags[j])
        return dict(k_star=float(10 ** (math.log10(k_lo) + f * (math.log10(k_hi) - math.log10(k_lo)))),
                    bracket=[k_lo, k_hi], note="linear in lg k between the bracketing levels")
    return dict(k_star=None, bracket=[k_lo, k_hi], note="crossing between the bracketing levels, no interpolation")


def map_e10(G):
    """Rate at which the contribution of the initial coherence, s_C F and s_C F_amp, falls below the dipole threshold
    for K at chi_b = 0.1 pi, m = 1, axis r, on the whole scale of k; the stricter transfer decides."""
    out = {}
    for g in ("W_C", "W_D"):
        for kind in ("rf", "std"):
            ks = (0.0,) + SCALE[kind]
            levels = []
            for k in ks:
                rl = "none" if k == 0.0 else kind
                rk, ri = G.get(g, "r", rl, k, "K", 0.1, 1.0), G.get(g, "r", rl, k, "K-inc", 0.1, 1.0)
                sc = s_C(rk, ri)
                levels.append(dict(k=k, F_K=G.F(rk), F_amp_K=G.Famp(rk), s_C=sc,
                                   F_coherence=None if sc is None else G.F(rk) * sc,
                                   F_amp_coherence=None if sc is None else G.Famp(rk) * sc))
            res = {}
            for meas in ("F_coherence", "F_amp_coherence"):
                res[meas] = first_below(list(ks), [None if x[meas] is None else abs(x[meas]) for x in levels])
            stricter = [None if x["F_coherence"] is None else min(abs(x["F_coherence"]), abs(x["F_amp_coherence"]))
                        for x in levels]
            out[f"{g}_r_{kind}"] = dict(levels=levels, by_transfer=res, stricter=first_below(list(ks), stricter))
    return out


def map_e11(G):
    out = {}
    for ax in ("z", "x", "r"):
        for kind, k in c.KEY_LEVELS:
            for th in (0.05, 0.1):
                rK = G.get("W_C", ax, kind, k, "K", th, 1.0)
                ps = [G.get("W_C", ax, kind, k, f, 0.1) for f in ("P-same", "P-LH")]
                for mo in c.DENTON_K2:
                    rE = G.get("W_C", ax, kind, k, f"E-{mo}", th, 1.0)
                    out[f"W_C_{ax}_{kind}{k:g}_theta{th:g}_{mo}"] = dict(
                        eps_mol=G.eps(rE), F=G.F(rE), F_amp=G.Famp(rE), d_odd=rE["d_odd"],
                        E_over_K_eps=ratio(G.eps(rE), G.eps(rK)), E_over_K_F=ratio(G.F(rE), G.F(rK)),
                        E_over_K_F_amp=ratio(G.Famp(rE), G.Famp(rK)),
                        E_over_Pmax_eps=ratio(G.eps(rE), max(G.eps(p) for p in ps)),
                        d_odd_angle_to_K_deg=angle_between(rE["d_odd"], rK["d_odd"]),
                        eta=[c.eta(mo).real, c.eta(mo).imag])
    return out


def map_e12(G):
    out = {}
    for rn in c.E12_RATES:
        for ax in ("z", "x", "r"):
            for kind, k in c.E12_LEVELS:
                for form, chi in (("K", 0.1), ("P-0", 0.1), ("P-same", 0.1), ("P-LH", 0.1), ("E-M1", 0.1)):
                    r = G.get("W_C", ax, kind, k, form, chi, 1.0, rn)
                    if r is None:
                        continue
                    out[r["name"]] = dict(eps_mol=G.eps(r), F=G.F(r), F_amp=G.Famp(r), d_odd=r["d_odd"],
                                          even_range=G.even(r), dipole_visible=G.strict(r) >= DIPOLE_THRESHOLD)
    return out


def map_e13(G):
    """Composite pair: odd part without CISS in units of the even range and of the responses, next to form K of the
    same composite."""
    out = {}
    for name, r in sorted(G.comp.items()):
        cl = r["cell"]
        twin = G.comp.get(c.CompositeCell("K", cl["rec"], cl["k_cd"], cl["k_dc"], cl["rates"]).name)
        entry = dict(form=cl["form"], rec=cl["rec"], k_cd=cl["k_cd"], k_dc=cl["k_dc"],
                     odd_over_even_range=ratio(r["parts"]["odd_half_range"], r["parts"]["even_range"]),
                     eps_mol=G.eps(r), F=G.F(r), F_amp=G.Famp(r), d_odd=r["d_odd"], rec_C=r["rec_C"], rec_D=r["rec_D"])
        if cl["form"] == "none" and twin is not None:
            entry.update(F_over_F_of_K=ratio(G.F(r), G.F(twin)), F_amp_over_F_amp_of_K=ratio(G.Famp(r), G.Famp(twin)),
                         eps_over_eps_of_K=ratio(G.eps(r), G.eps(twin)))
        out[name] = entry
    return out


def map_e14(G, rec, e14):
    """Receptor axis over the grid for every cell of K at m = 1 (F and F_amp), and the crystal 6PU0 of every cell of
    W_C and W_D: amplitude of the xyz pattern over the even range of the crystal."""
    sweep = {}
    for r in G.rows.values():
        key = f"{r['name']}__F"
        if key not in e14:
            continue
        F, Fa = e14[key], e14[f"{r['name']}__F_amp"]
        strict = np.minimum(np.abs(F), np.abs(Fa))
        sweep[r["name"]] = dict(abs_F_max=float(np.abs(F).max()), abs_F_amp_max=float(np.abs(Fa).max()),
                                abs_stricter_quantiles=np.quantile(strict, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                                share_dipole_visible=float(np.mean(strict >= DIPOLE_THRESHOLD)))
    crystal = {name: dict(xyz_amplitude_over_even_range=v["xyz_amplitude_over_even_range"],
                          xyz_amplitude=v["xyz_amplitude"], even_range=v["even_range"], odd_max=v["odd_max"],
                          dipole_norm=v["dipole_norm"], molecule_dipole_norm=v["molecule_dipole_norm"])
               for name, v in rec["crystal"].items()}
    return dict(receptor_axis=sweep, crystal=crystal)


def maps(G, rec, e5, e14, curves):
    return dict(E1=map_e1(G), E2=map_e2(G), E3=map_e3(G), E4=map_e4(G), E5=map_e5(G, e5), E6=map_e6(G), E7=map_e7(G),
                E8=map_e8(G, e5), E9=map_e9(curves), E10=map_e10(G), E11=map_e11(G), E12=map_e12(G), E13=map_e13(G),
                E14=map_e14(G, rec, e14))


# --------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------
CSV_FIELDS = ("name", "source", "branch", "system", "axis", "relax", "k", "form", "chi", "m", "rates", "eps_mol",
              "eps_rec", "F", "F_amp", "d_odd_z", "d_odd_norm", "even_range", "G", "s_C")


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
    """The 24 curves of E9: the eight cases, each with K_up, delta_up and delta_down."""
    want = {f"{g}_{rl}{k:g}" for g, rl, k in c.CURVE_CASES}
    return (set(curves["cases"]) == want
            and all(set(v) == {"K_up", "delta_up", "delta_down"} for v in curves["cases"].values()))


def curves_complete(curves):
    return (curves_structure_ok(curves) and curves["t_max"] == c.CURVE_T and curves["dt"] == c.CURVE_DT
            and curves["every"] == c.CURVE_EVERY)


def rates_ok(rec, chk, curves):
    ok = True
    for r in rec["cells"]:
        cl = r["cell"]
        ok &= tuple(r["rates_used"]) == PROTOCOL_RATES[cl["rates"]]
        if r["kind"] == "composite":
            ok &= tuple(r["hop_rates_used"]) == (cl["k_cd"], cl["k_dc"])
    for v in rec["refs"].values():
        ok &= tuple(v["rates_used"]) == PROTOCOL_RATES[v["ref"]["rates"]]
    ok &= all(tuple(chk["rates_used"][k]) == PROTOCOL_RATES[k] for k in chk["rates_used"]) and chk["hop_scale"] == 1.0
    ok &= tuple(curves["rates_used"]) == PROTOCOL_RATES["cry4a"]
    return bool(ok)


def expected_names():
    return dict(direct={x.name for x in c.direct_cells()}, rebuilt={x.name for x in c.rebuilt_cells()},
                composite={x.name for x in c.composite_cells()}, refs={x.name for x in c.references()})


A_GRID = [None]


def analyze(rec, chk, curves, e5, e14, pair_done=None, composite_done=None):
    want = expected_names()
    have = dict(direct=[r["name"] for r in rec["cells"] if r["kind"] == "direct"],
                rebuilt=[r["name"] for r in rec["cells"] if r["kind"] == "rebuilt"],
                composite=[r["name"] for r in rec["cells"] if r["kind"] == "composite"], refs=list(rec["refs"]))
    complete = all(set(have[k]) == want[k] and len(have[k]) == len(want[k]) for k in want)
    if not complete or not curves_structure_ok(curves):
        return dict(valid=False, incomplete=True, cells_complete=complete, curves_structure_ok=curves_structure_ok(curves),
                    missing={k: sorted(want[k] - set(have[k])) for k in want},
                    unexpected={k: sorted(set(have[k]) - want[k]) for k in want}, metadata=c.metadata("analyze"))
    G = Grid(rec)
    A_GRID[:] = [G]
    grid_ok = (rec["grid_half"] == c.GRID_HALF and chk["grid_half"] == c.GRID_HALF
               and rec["e13_grid_half"] == [c.E13_GRID_HALF])
    r_ok = rates_ok(rec, chk, curves)
    curves_ok = curves_complete(curves)
    meta = c.metadata("analyze")
    steps = [rec["metadata"], chk["metadata"], curves["metadata"], meta]
    steps += [x["metadata"] for x in (pair_done, composite_done) if x]
    blas_ok = blas_single_thread(*steps)
    invs = invariants(rec, chk, curves, G)
    exp = dict(Q1=q1(G), Q2=q2(G), Q3=q3(G), Q4=q4(curves))
    hyp = hypotheses(G, e5)
    counted = comparison_counts(hyp)
    counts_ok = counted["per_hypothesis"] == N_COMPARISONS and counted["known_in_advance"] == N_KNOWN
    valid = (all(v["verdict"] == "ok" for v in invs.values()) and grid_ok and r_ok and complete and curves_ok
             and blas_ok and counts_ok and pair_done is not None and composite_done is not None)
    return dict(valid=bool(valid), grid_ok=bool(grid_ok), rates_ok=r_ok, cells_complete=bool(complete),
                curves_full=bool(curves_ok), blas_single_thread=bool(blas_ok), counts_ok=bool(counts_ok),
                pair_done=pair_done is not None, composite_done=composite_done is not None,
                grid_half=rec["grid_half"], e13_grid_half=rec["e13_grid_half"], rates_seen=rates_seen(rec, chk),
                invariants=invs, expectations=exp, hypotheses=hyp,
                comparisons=dict(counted=counted, protocol=N_COMPARISONS, known_protocol=N_KNOWN),
                multiplicity="deterministic hypotheses; no correction applies", maps=maps(G, rec, e5, e14, curves),
                metadata=meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loaded = {}
    for name in ("receptor", "checks", "curves", "pair_done", "composite_done"):
        path = os.path.join(a.run, f"{name}.json")
        if name.endswith("_done") and not os.path.exists(path):
            loaded[name] = None
            continue
        with open(path) as fh:
            loaded[name] = json.load(fh)
    e5 = dict(np.load(os.path.join(a.run, "maps_E5.npz")))
    e14 = dict(np.load(os.path.join(a.run, "maps_E14.npz")))
    res = analyze(loaded["receptor"], loaded["checks"], loaded["curves"], e5, e14, loaded["pair_done"],
                  loaded["composite_done"])
    os.makedirs(a.out, exist_ok=True)
    if res.get("incomplete"):
        c.write_json(os.path.join(a.out, "results_incomplete.json"), res)
        sys.exit(f"INVALID: missing {sum(len(v) for v in res['missing'].values())}, unexpected "
                 f"{sum(len(v) for v in res['unexpected'].values())}, curves "
                 f"{'complete' if res['curves_structure_ok'] else 'incomplete'}")
    write_table(os.path.join(a.out, "tables", "cells.csv"), res["maps"]["E1"])
    np.savez_compressed(os.path.join(a.out, "maps_axes.npz"), **axis_maps(A_GRID[0], e5))
    c.write_json(os.path.join(a.out, "results.json"), res)
    if res["rates_ok"] and res["grid_ok"]:
        verdicts = {k: {g: w["verdict"] for g, w in v.items() if g in c.GEOMETRIES} for k, v in res["hypotheses"].items()}
        print("valid" if res["valid"] else "INVALID", verdicts, flush=True)
    else:       # smoke and blind checks: technical facts only, no verdicts on the screen or in the logs
        print("valid" if res["valid"] else "INVALID", "(rates or grids of a check run; verdicts withheld)", flush=True)


if __name__ == "__main__":
    main()
