"""Mutation testing of estimate.py against test_s3b.py (author's harness, not part of the study).
Every mutation changes one rule of the dipole test or of the estimate; test_s3b.py must fail on each.
Usage: python3 -I scripts/mutate_estimate.py CODE_DIR OUT_DIR"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

CODE, OUT = sys.argv[1], sys.argv[2]
SRC = open(os.path.join(CODE, "estimate.py")).read()

M = [
    ("gamma_no_intercept", "gam = (Sxy - Sx * Sy / Sw) / (Sxx - Sx * Sx / Sw)", "gam = Sxy / Sxx"),
    ("odd_template_sign", "return s_even, b.depth(S_assumed) * (V @ N.T)", "return s_even, -b.depth(S_assumed) * (V @ N.T)"),
    ("odd_template_no_depth", "return s_even, b.depth(S_assumed) * (V @ N.T)", "return s_even, (V @ N.T)"),
    ("even_template_with_odd", "b.receptor_signal(v @ N.T, S_assumed, 1.0, 0.0)",
     "b.receptor_signal(v @ N.T, S_assumed, 1.0, 0.05)"),
    ("even_part_in_numerator", "num, den = wp @ (t * D).T, wp @ (t * t).T", "num, den = wp @ (t * E).T, wp @ (t * t).T"),
    ("bootstrap_cells_and_pairs", "boot[a:a + k] = eps_fit(wp, E, D, s_even, t, w)[0]",
     "boot[a:a + k] = eps_fit(wp, E, D, s_even, t, w * rng_boot.multinomial(len(w), np.full(len(w), 1.0 / len(w)), "
     "size=k))[0]"),
    ("interval_90", "t975, t95 = stats.t.ppf(0.975, P - 1), stats.t.ppf(0.95, P - 1)",
     "t975, t95 = stats.t.ppf(0.95, P - 1), stats.t.ppf(0.95, P - 1)"),
    ("upper_with_t975", "t975, t95 = stats.t.ppf(0.975, P - 1), stats.t.ppf(0.95, P - 1)",
     "t975, t95 = stats.t.ppf(0.975, P - 1), stats.t.ppf(0.975, P - 1)"),
    ("normal_quantiles", "t975, t95 = stats.t.ppf(0.975, P - 1), stats.t.ppf(0.95, P - 1)",
     "t975, t95 = 1.959964, 1.644854"),
    ("no_small_sample_factor", "scale = np.sqrt(P / (P - 1.0)) * se", "scale = se"),
    ("null_scaled_up", "G[s] = ((XA * wgt[sel, None]).T @ XB) * mask / (4.0 * len(A) * len(B))",
     "G[s] = ((XA * wgt[sel, None]).T @ XB) * mask / (3.6 * len(A) * len(B))"),
    ("pvalue_without_plus_one", "return float((1 + np.sum(null >= e - 1e-12 * abs(e))) / (1 + len(null)))",
     "return float(np.sum(null >= e) / len(null))"),
    ("main_estimate_knows_S", "s_even, t = templates(geo, 1.0)", "s_even, t = templates(geo, S_true)"),
    ("weights_uniform", "return 1.0 / np.where(var > 0, var, 1.0)", "return np.ones(C.shape[0])"),
    ("dipole_same_half", "c = (Dp[..., B].mean(-1) / 2) @ Q", "c = (Dp[..., A].mean(-1) / 2) @ Q"),
    ("dipole_null_unprojected", "mask = np.kron(Q @ Q.T, np.ones((len(A), len(B))))",
     "mask = np.kron(np.eye(P), np.ones((len(A), len(B))))"),
    ("alpha_010", "N_PERM, N_BOOT, ALPHA = 1000, 1000, 0.05", "N_PERM, N_BOOT, ALPHA = 1000, 1000, 0.10"),
    ("geometry_stream", "h = m2.uniform_dirs(rng_for(seed, 5), m3.MAX_PAIRS)[:K // 2]",
     "h = m2.uniform_dirs(rng_for(seed, 6), m3.MAX_PAIRS)[:K // 2]"),
    ("flips_shared_by_halves", "fb = flips[..., B].reshape(k, n_sess, -1)", "fb = flips[..., A].reshape(k, n_sess, -1)"),
]

# equivalent mutations keep a valid test or estimate and need not be killed
EQUIVALENT = {"flips_shared_by_halves": "coupled flips of the two halves are a subgroup of the flip group, so the "
                                        "permutation test stays exact"}

os.makedirs(OUT, exist_ok=True)
rows = []
for name, old, new in M:
    if SRC.count(old) != 1:
        rows.append(dict(mutation=name, status="pattern_not_unique_or_missing"))
        continue
    work = tempfile.mkdtemp(prefix=f"mut_{name}_")
    shutil.copytree(CODE, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(work, "estimate.py"), "w") as fh:
        fh.write(SRC.replace(old, new))
    r = subprocess.run([sys.executable, "-I", os.path.join(work, "test_s3b.py"), "--no-analysis"], capture_output=True,
                       text=True)
    shutil.rmtree(work)
    failed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("FAIL")]
    rows.append(dict(mutation=name, killed=r.returncode != 0, failed_tests=failed, equivalent=EQUIVALENT.get(name)))
    print(name, "killed" if r.returncode else "SURVIVED", failed, flush=True)
with open(os.path.join(OUT, "mutation_estimate.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
real = [r for r in rows if not r.get("equivalent")]
print(sum(r.get("killed", False) for r in real), "of", len(real), "mutations killed;", len(rows) - len(real), "equivalent")
