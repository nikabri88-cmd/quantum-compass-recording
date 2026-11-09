#!/usr/bin/env python3
"""
MAG-S3b analysis - invariants, expectations, hypotheses and exploratory maps. Protocol version 0.4 (draft).

Order: the pair grid and the receptor maps (mode physics and receptor of mag_s3b.py), the calibration (C1, C3), the
blind estimates (estimate.py) and only then the truth. The truth file is opened after I7 has checked that neither
the records nor the estimates folder holds a truth file, that estimates.sha256 matches estimates.json and that every
estimate read a record of the generator's manifest (SHA-256 of the record files; the manifest carries no truth).
Verdicts: invariants ok / error; expectations and hypotheses met / not_met / untestable with the reason. The run is
valid when all eight invariants are ok.

Usage
  python3 -I analyze_s3b.py --pair DIR --calibration DIR --records DIR --estimates DIR --truth DIR --out DIR
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


b = _load("mag_s3b")

TOL_EXACT, TOL_I6, TOL_Q1, TOL_Q2 = 1e-12, 1e-6, 1e-6, 0.01
H1_TOL, H2_RATIO, H3_EPS_ANT, H4_TOL = 0.15, 10.0, 0.05, 0.10
# H5 (v0.4, binomial tolerances): an estimate with exact 95% coverage meets H5 with probability 0.95, one with
# coverage 0.92 or 0.98 with probability 0.003 at most (pooled band per point, every level, zero rule two-sided)
H5_BAND, H5_PER_LEVEL, H5_ZERO_MAX = (0.925, 0.975), 87, 10
DETECT_NEED = 80
# eps_equivalent of P-same at chi = 0.1 pi, MAG-S3 v0.1.0 results/confirmatory_run_v04/physics_s3.json
# (SHA-256 6b765a69...), anchor_haberkorn_forward_yield
MAG_S3_E5 = {"mags2": 0.0011244586373918653, "cry4a": 0.00014289998826479909, "cry4a_fast": 0.00012169824179592192}
MAIN_RATES = "cry4a"


def met(ok, **kw):
    return dict(verdict="met" if ok else "not_met", **kw)


def untestable(reason, **kw):
    return dict(verdict="untestable", reason=reason, **kw)


def inv(ok, **kw):
    return dict(verdict="ok" if ok else "error", **kw)


def cp_two(x, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a, x + 1, n - x))
    return [lo, hi]


# --------------------------------------------------------------------------------------------------------------
# Pair grid and receptor
# --------------------------------------------------------------------------------------------------------------
def index_cells(rows):
    return {b.Cell(**r["cell"]).name: r for r in rows}


def K(chi, m, beta=0.0, rates=MAIN_RATES, alpha=0.0):
    return b.Cell("K", chi, m, alpha, beta, rates).name


def P(form, chi, beta=0.0, rates=MAIN_RATES):
    return b.Cell(form, chi, beta=beta, rates=rates).name


def eps_mol(ix, name):
    return ix[name]["parts"]["eps_mol"]


def eps_eff(ix, name, p=1.0, S=1.0):
    return abs(ix[name]["receptor"]["by_order"][f"S{S:g}_p{p:g}"]["eps_eff"])


def dz(ix, name):
    return ix[name]["d_odd"][2]


def h1(ix):
    rows = []
    for chi in b.CHI_LEVELS:
        e1 = eps_mol(ix, K(chi, 1.0))
        for m in (0.1, 0.25, 0.5):
            e = eps_mol(ix, K(chi, m))
            rows.append(dict(chi=chi, m=m, eps_mol=e, linear=m * e1, rel_dev=abs(e - m * e1) / (m * e1)))
    return met(all(r["rel_dev"] <= H1_TOL for r in rows), rows=rows)


def h2(ix):
    eK = eps_mol(ix, K(0.05, 1.0))
    eP = {f: eps_mol(ix, P(f, 0.1)) for f in ("P-same", "P-LH")}
    ratio = eK / max(eP.values())
    same_chi = {"K_chi0.1": eps_mol(ix, K(0.1, 1.0)), **{f"{f}_chi0.1": v for f, v in eP.items()}}
    return met(ratio >= H2_RATIO, eps_K_chi005=eK, eps_P_chi01=eP, ratio=ratio, equal_chi_descriptive=same_chi)


def h3(ix):
    vals = {m: eps_eff(ix, K(0.1, m)) for m in (0.1, 0.25, 0.5)}
    return met(max(vals.values()) >= H3_EPS_ANT, eps_eff=vals, threshold=H3_EPS_ANT)


def h4(ix):
    rows = []
    for chi in b.CHI_LEVELS:
        z0 = dz(ix, K(chi, 1.0))
        for beta in (45.0, b.BETA_K_EXTRA, 90.0):
            z = dz(ix, K(chi, 1.0, beta))
            rows.append(dict(chi=chi, beta=beta, dz=z, cos_law=z0 * np.cos(np.deg2rad(beta)),
                             dev_over_dz0=abs(z - z0 * np.cos(np.deg2rad(beta))) / abs(z0)))
    return met(all(r["dev_over_dz0"] <= H4_TOL for r in rows), rows=rows)


def q1(ix):
    rows = {r: eps_mol(ix, P("P-same", 0.1, 0.0, r)) / MAG_S3_E5[r] - 1.0 for r in ("mags2", "cry4a")}
    return met(all(abs(v) <= TOL_Q1 for v in rows.values()), rel_dev=rows)


def q2(ix):
    """Q2 (v0.4): the even part fits at degree 10 to 1% of its SD, and the odd dipole changes by less than 1% from
    degree 10 to degree 24. The odd residuals at both degrees are reported."""
    worst_even = max(r["fit_even_rms_over_sd"] for r in ix.values())
    dev = []
    for r in ix.values():
        d10, d24 = np.array(r["d_odd"]), np.array(r["fit_check"]["d_odd"])
        if np.linalg.norm(d10) > 0:
            dev.append(float(np.linalg.norm(d24 - d10) / np.linalg.norm(d10)))
    odd10 = [r["fit_odd_rms_over_sd"] for r in ix.values() if r["fit_odd_rms_over_sd"] is not None]
    odd24 = [r["fit_check"]["odd_rms_over_sd"] for r in ix.values() if r["fit_check"]["odd_rms_over_sd"] is not None]
    return met(worst_even <= TOL_Q2 and max(dev) <= TOL_Q2, worst_even=worst_even, worst_dipole_change=max(dev),
               odd_residual_L10=dict(median=float(np.median(odd10)), max=float(max(odd10))),
               odd_residual_L24=dict(median=float(np.median(odd24)), max=float(max(odd24))))


def q4(ix):
    rows, ok = [], True
    for rates in b.RATES:
        for chi in b.CHI_LEVELS:
            F = [eps_eff(ix, K(chi, m, rates=rates)) for m in b.M_LEVELS]
            mono = all(F[i] < F[i + 1] for i in range(len(F) - 1))
            ok &= mono
            rows.append(dict(rates=rates, chi=chi, F=F, increasing=mono))
    return met(ok, rows=rows)


# --------------------------------------------------------------------------------------------------------------
# Inverse problem
# --------------------------------------------------------------------------------------------------------------
def join(estimates, truth):
    tr = {(t["point"], t["seed"]): t for t in truth}
    rows = []
    for e in estimates:
        t = tr[(e["point"], e["seed"])]
        rows.append(dict(e, eps_eff=t["eps_eff"], S=t["S"], truth_record_sha256=t.get("record_sha256")))
    return rows


def covered(est, eps):
    lo, hi = est["ci95"]
    return lo <= eps <= hi


def h5(rows, gate):
    """gate: None when calibration C3 was met by the current code, else the reason."""
    if gate:
        return untestable(gate)
    out, ok = {}, True
    for point in ("R1", "R2", "R3"):
        pr = [r for r in rows if r["point"] == point]
        pos = [r for r in pr if r["eps_eff"] > 0]
        cov = [covered(r["estimate"], r["eps_eff"]) for r in pos]
        share = float(np.mean(cov))
        per_level = {f"{e:g}": int(sum(covered(r["estimate"], e) for r in pos if r["eps_eff"] == e))
                     for e in b.EPS_GRID[1:]}
        zero = [r for r in pr if r["eps_eff"] == 0]
        excl = int(sum(r["estimate"]["ci95_raw"][0] > 0 or r["estimate"]["ci95_raw"][1] < 0 for r in zero))
        good = H5_BAND[0] <= share <= H5_BAND[1] and min(per_level.values()) >= H5_PER_LEVEL and excl <= H5_ZERO_MAX
        ok &= good
        out[point] = dict(n=len(pos), coverage=share, coverage_cp95=cp_two(int(sum(cov)), len(pos)),
                          per_level_covered=per_level, zero_excluded=excl, n_zero=len(zero), met=bool(good))
    return met(ok, points=out)


def detection_threshold(counts):
    for e in b.EPS_GRID[1:]:
        if counts[f"{e:g}"] >= DETECT_NEED:
            return e
    return None


def h6(rows, gate):
    """gate: None when calibration C1 was met by the current code, else the reason."""
    if gate:
        return untestable(gate)
    pr = [r for r in rows if r["point"] == "R1"]
    dip = {f"{e:g}": int(sum(r["dipole"]["odd_found"] for r in pr if r["eps_eff"] == e)) for e in b.EPS_GRID}
    ant = {f"{e:g}": int(sum(r["antipodal"]["odd_found"] for r in pr if r["eps_eff"] == e)) for e in b.EPS_GRID}
    td, ta = detection_threshold(dip), detection_threshold(ant)
    kw = dict(dipole_found=dip, antipodal_found=ant, dipole_threshold=td, antipodal_threshold=ta)
    if td is None and ta is None:
        return untestable("neither test reaches 80 of 100 birds on the grid", **kw)
    if td is None:
        return met(False, **kw)
    if ta is None:
        # the antipodal threshold lies above the largest level, so half of it lies above half of that level
        if td <= b.EPS_GRID[-1] / 2:
            return met(True, **kw)
        return untestable("the antipodal threshold lies beyond the grid and the dipole threshold above half of its "
                          "largest level", **kw)
    return met(td <= ta / 2, **kw)


def q3(rows):
    pr = [r for r in rows if r["point"] == "R1" and r["eps_eff"] == 0.05]
    n = int(sum(r["antipodal"]["odd_found"] for r in pr))
    return met(n >= DETECT_NEED, antipodal_found=n, n=len(pr))


# --------------------------------------------------------------------------------------------------------------
# Exploratory maps
# --------------------------------------------------------------------------------------------------------------
def e1_e5_e6(ix):
    """E1 pair map, E5 high odd harmonics (the share of odd receptor power the dipole test cannot see, at degrees 10
    and 24) and E6 amplitude transfer for every cell."""
    rows = []
    for name, r in ix.items():
        rows.append(dict(cell=name, eps_mol=r["parts"]["eps_mol"], eps_rec=r["receptor"]["eps_rec"],
                         d_odd=r["d_odd"], odd_power_by_degree=r["odd_power_by_degree"],
                         odd_power_by_degree_L24=r["fit_check"]["odd_power_by_degree"],
                         odd_share_deg3_molecule=r["odd_power_shares"]["molecule"],
                         odd_share_deg3_receptor=r["odd_power_shares"]["receptor"],
                         odd_share_deg3_molecule_L24=r["fit_check"]["odd_power_shares"]["molecule"],
                         odd_share_deg3_receptor_L24=r["fit_check"]["odd_power_shares"]["receptor"],
                         dipole_test_blind_share_L24=r["fit_check"]["odd_power_shares"]["receptor"],
                         odd_residual_L10=r["fit_odd_rms_over_sd"], odd_residual_L24=r["fit_check"]["odd_rms_over_sd"],
                         eps_amplitude_E6=float(r["zonal"][1] / b.m3.REN_B),
                         E6_by_order={k: v["eps_E6"] for k, v in r["receptor"]["by_order"].items()}))
    return rows


def e2(rows, ix):
    """E2: the upper bounds at eps_eff = 0 per point and their translation into physics. For K, F(dchi) = eps_eff at
    p = 1, beta = 0, S = 1 and the slope s = F(chi_b) / chi_b give a bound on dchi p cos(beta) for every level S
    (dividing by g(S)); for P the bound is on p G(beta) g(S). The admissible region lists every cell of the grid at
    every (p, S) whose eps_eff (and, as E6, whose amplitude transfer) lies below the median bound."""
    out = {}
    g = {S: b.g_order(S) for S in b.S_LEVELS}
    for point in ("R1", "R2", "R3"):
        ub = [r["estimate"]["upper95"] for r in rows if r["point"] == point and r["eps_eff"] == 0]
        med = float(np.median(ub)) if ub else None
        trans = {}
        for rates in b.RATES:
            for chi in b.CHI_LEVELS:
                F1 = eps_eff(ix, K(chi, 1.0, rates=rates))
                slope = F1 / (chi * np.pi)
                trans[f"K_chi{chi:g}_{rates}"] = dict(slope_per_rad=slope, bound_dchi_p_cosb_by_S={
                    f"S{S:g}": (med / (slope * g[S]) if med is not None and slope > 0 else None) for S in b.S_LEVELS})
                for f in ("P-same", "P-LH"):
                    e = eps_eff(ix, P(f, chi, rates=rates))
                    trans[f"{f}_chi{chi:g}_{rates}"] = dict(eps_eff=e, bound_p_G_g=med / e if med and e else None)
        region = []
        if med is not None:
            for name, r in ix.items():
                for k, v in r["receptor"]["by_order"].items():
                    if abs(v["eps_eff"]) <= med:
                        region.append(dict(cell=name, order=k, eps_eff=abs(v["eps_eff"]),
                                           E6_admissible=bool(abs(v["eps_E6"]) <= med)))
        out[point] = dict(upper95_per_bird=ub, upper95_median=med, translation=trans,
                          admissible=dict(n=len(region), of=len(ix) * len(b.S_LEVELS) * len(b.P_LEVELS),
                                          cells=region))
    return out


def e3_map(rows):
    """E3: coverage, mean and width of the estimate without the retina against the main estimate, per point and
    level of eps_eff."""
    out = {}
    for point in ("R1", "R2", "R3"):
        pr = [r for r in rows if r["point"] == point and "estimate_e3" in r]
        lev = {}
        for e in b.EPS_GRID:
            sel = [r for r in pr if r["eps_eff"] == e]
            if not sel:
                continue
            width = lambda key: float(np.median([r[key]["ci95"][1] - r[key]["ci95"][0] for r in sel]))  # noqa: E731
            lev[f"{e:g}"] = dict(n=len(sel), covered_e3=int(sum(covered(r["estimate_e3"], e) for r in sel)),
                                 covered_main=int(sum(covered(r["estimate"], e) for r in sel)),
                                 mean_eps_e3=float(np.mean([r["estimate_e3"]["eps_hat_raw"] for r in sel])),
                                 mean_eps_main=float(np.mean([r["estimate"]["eps_hat_raw"] for r in sel])),
                                 median_width_e3=width("estimate_e3"), median_width_main=width("estimate"))
        out[point] = lev
    return out


def e4_map(pair_dir, h6_result):
    """E4: residual odd part at p = 0 with fixed azimuth, its transfer eps_x and its detection by the dipole test in
    R1, next to the detection threshold of the dipole test from H6."""
    path = os.path.join(pair_dir, "e4.json")
    if not os.path.exists(path):
        return "not computed"
    with open(path) as fh:
        e4 = json.load(fh)
    return dict(points=e4["points"], dipole_threshold_H6=h6_result.get("dipole_threshold"))


def e7(receptor_json, ix):
    """E7: D(S) and g(S) of the Ren pattern and of the pair pattern of every cell."""
    pair = {name: {k: dict(D=v["D_pair"], g=v["g_pair"]) for k, v in r["receptor"]["by_order"].items()
                   if k.endswith("_p1")} for name, r in ix.items()}
    return dict(ren=receptor_json["order"]["ren"], pair=pair)


def e8(rows):
    out = {}
    for point in ("E8_S0.8", "E8_S0.6"):
        pr = [r for r in rows if r["point"] == point]
        lev = {}
        for e in b.EPS_GRID:
            sel = [r for r in pr if r["eps_eff"] == e]
            if not sel:
                continue
            lev[f"{e:g}"] = {var: dict(mean_eps_hat=float(np.mean([r[var]["eps_hat_raw"] for r in sel])),
                                       covered=int(sum(covered(r[var], e) for r in sel)), n=len(sel))
                             for var in ("estimate", "estimate_known_S")}
        out[point] = lev
    return out


# --------------------------------------------------------------------------------------------------------------
# Invariants and the run
# --------------------------------------------------------------------------------------------------------------
def invariants(inv_phys, receptor_rows, i7):
    out = dict(I1=inv(not b.frozen_mismatches(), mismatches=b.frozen_mismatches()))
    for key in ("I2", "I3", "I4"):
        worst = max(inv_phys[key].values())
        out[key] = inv(worst < TOL_EXACT, worst=worst)
    worst5 = max(max(v["p0_max_odd"], v["p05_half_max_diff"]) for r in receptor_rows for v in r["I5"].values())
    out["I5"] = inv(worst5 < TOL_EXACT, worst=worst5)
    worst6 = max(inv_phys["I6"].values())
    out["I6"] = inv(worst6 < TOL_I6, worst=worst6)
    out["I7"] = i7
    worst8 = max(inv_phys["I8"].values())
    out["I8"] = inv(worst8 <= TOL_EXACT, worst=worst8)
    return out


def calibration_gate(path, name):
    """None when the calibration criterion was met by the current mag_s3b.py and estimate.py, else the reason."""
    with open(path) as fh:
        j = json.load(fh)
    own = j.get("metadata", {}).get("own", {})
    if any(own.get(f) != b.file_sha256(os.path.join(HERE, f)) for f in ("mag_s3b.py", "estimate.py")):
        return f"calibration {name} was written by other code"
    return None if j.get("criterion_met") else f"calibration {name} not met"


def check_i7(rec_dir, est_dir, est_json):
    """Before the truth is opened: no truth file next to the records or the estimates, estimates.sha256 matches
    estimates.json, and every estimate read a record of the generator's manifest (which carries no truth)."""
    issues = []
    for d in (rec_dir, est_dir):
        if glob.glob(os.path.join(d, "**", "*truth*"), recursive=True):
            issues.append(f"truth file in {d}")
    try:
        with open(os.path.join(est_dir, "estimates.sha256")) as fh:
            line = fh.read().split()
        if line[0] != b.file_sha256(os.path.join(est_dir, "estimates.json")):
            issues.append("estimates.sha256 does not match estimates.json")
        hashed_at = line[2] if len(line) > 2 else None
    except OSError:
        issues.append("estimates.sha256 missing")
        hashed_at = None
    with open(os.path.join(rec_dir, "manifest.json")) as fh:
        manifest = {(r["point"], r["seed"]): r["record_sha256"] for r in json.load(fh)["records"]}
    bad = [(e["point"], e["seed"]) for e in est_json["estimates"]
           if manifest.get((e["point"], e["seed"])) != e.get("record_sha256")]
    if bad:
        issues.append(f"{len(bad)} estimated records are not records of the manifest")
    if len(est_json["estimates"]) != len(manifest):
        issues.append("number of estimates differs from the number of generated records")
    return inv(not issues, issues=issues, estimates_hashed_at=hashed_at), manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pair", "calibration", "records", "estimates", "truth", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.pair, "invariants_physics.json")) as fh:
        inv_phys = json.load(fh)
    with open(os.path.join(a.pair, "receptor.json")) as fh:
        rec_json = json.load(fh)
    ix = index_cells(rec_json["cells"])
    gate = {c: calibration_gate(os.path.join(a.calibration, f"{c}.json"), c) for c in ("C1", "C3")}
    grid_ok = rec_json.get("grid_half") == b.GRID_HALF
    rates_ok = all(tuple(v) == b.RATES[k] for k, v in (rec_json.get("rates_used") or {}).items())
    with open(os.path.join(a.estimates, "estimates.json")) as fh:
        est_json = json.load(fh)
    i7, manifest = check_i7(a.records, a.estimates, est_json)
    t_open = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(a.truth, "truth.json")) as fh:
        truth = json.load(fh)["records"]
    i7["truth_opened_at"] = t_open
    if any(manifest.get((t["point"], t["seed"])) != t["record_sha256"] for t in truth):
        i7.update(verdict="error", issues=i7["issues"] + ["truth file and manifest disagree"])
    invs = invariants(inv_phys, rec_json["cells"], i7)
    rows = join(est_json["estimates"], truth)
    hyp = dict(H1=h1(ix), H2=h2(ix), H3=h3(ix), H4=h4(ix), H5=h5(rows, gate["C3"]), H6=h6(rows, gate["C1"]))
    res = dict(
        valid=all(v["verdict"] == "ok" for v in invs.values()) and grid_ok and rates_ok, invariants=invs,
        grid_ok=grid_ok, rates_ok=rates_ok, calibration_gates=gate,
        expectations=dict(Q1=q1(ix), Q2=q2(ix), Q3=q3(rows), Q4=q4(ix)), hypotheses=hyp,
        multiplicity="six hypotheses, no correction",
        maps=dict(E1_E5_E6=e1_e5_e6(ix), E2=e2(rows, ix), E3=e3_map(rows), E4=e4_map(a.pair, hyp["H6"]),
                  E7=e7(rec_json, ix), E8=e8(rows)),
        metadata=b.metadata("analyze"))
    os.makedirs(a.out, exist_ok=True)
    b.write_json(os.path.join(a.out, "results.json"), res)
    print("valid" if res["valid"] else "INVALID", {k: v["verdict"] for k, v in res["hypotheses"].items()})


if __name__ == "__main__":
    main()
