#!/usr/bin/env python3
"""
MAG-S3b estimate - the dipole test and the blind estimate of eps_eff. Protocol version 0.4 (draft).

Reads only records (counts, sessions, seed, recording point). The estimate knows the retina and the patches of the
bird, rebuilt from the frozen MAG-S2 streams of its seed (receptor axes, patch weights, ON/OFF signs, presented
directions), and does not know eps, the base rates or the modulation depths.

Counts C (cells x K x repeats); directions k and k + K/2 are opposite. Per cell and pair p, the even part
E_ip = mean over repeats of (C(n_p) + C(-n_p)) / 2 and the odd part D_ip = mean of (C(n_p) - C(-n_p)) / 2. Weights
w_i = 1 / variance of all counts of the cell, as in MAG-S3.

Dipole test. The odd part of each cell is projected on the span of the pair directions (Q, P x 3, orthonormal), so
the test reads only its dipole. Energy E = sum_i w_i (Q^T oA_i) . (Q^T oB_i) with oA, oB the odd parts of the two
halves of the repeats. p value: sign flips of n and -n per session, pair and repeat (the same flip for every cell of
a session), 1000 permutations, p = (1 + #(null >= E)) / (1 + 1000), as the antipodal test of MAG-S3.

Estimate. With the Ren pattern the expected counts are lambda_i(n) = c_i + gamma_i s_i(n), where s_i is the signed
patch sum of the receptor signal and gamma_i = r0_i delta T / range_i. The even part of s_i is the template of the
even part, its odd part is eps D(S) V_i . n with V_i = sign_i sum_k W_ik v_k. gamma_i is the least-squares slope of
E_ip on the even template (with intercept, so the unknown normalisation range_i is absorbed), and eps_hat is the
weighted least-squares fit of D_ip = eps gamma_i t_ip with t_ip = D(S) V_i . n_p:

    eps_hat = sum_i w_i gamma_i sum_p t_ip D_ip / sum_i w_i gamma_i^2 sum_p t_ip^2

Interval. 1000 bootstrap samples over direction pairs (gamma refitted in every sample) give the standard error SE of
eps_hat. The 95% interval is eps_hat +- t_{P-1}(0.975) sqrt(P / (P - 1)) SE with negative ends set to 0, and the
upper bound is eps_hat + t_{P-1}(0.95) sqrt(P / (P - 1)) SE with the estimate before clipping (clipped at 0
afterwards, so it keeps one-sided 95% coverage), the small-sample rule for P clusters of cluster-robust
inference (Cameron & Miller 2015). Cells are not resampled: eps is a property of the bird and the estimate is
conditional on its recorded cells, while resampling cells and pairs together counts the independent count noise
about three times. The percentile interval is reported alongside; it is shifted down, because every bootstrap
sample refits gamma and the noise of gamma lowers eps a second time (test_s3b.py, test_bootstrap_coverage).

The antipodal test of MAG-S3 runs unchanged on the same records (H6).

E3, the estimate without the retina. It knows the presented directions and the counts only. The cross-validated
dipole energy of the dipole test, sum_i w_i (N^T W D_A,i)^T (N^T W N)^-1 (N^T W D_B,i), has the expectation
eps^2 sum_i w_i gamma_i^2 V_i^T (N^T W N) V_i, which for a dipole direction it does not know is
eps^2 kappa^2 sum_i w_i Var(even signal of cell i) sum_p W_p / 3, with Var(even signal) = gamma_i^2 Var(s_even_i)
read from the even part of the counts (minus its count noise from the repeats) and kappa^2 = |V|^2 / Var(s_even)
averaged over the cells of calibration birds 7001-7100 (e3_geometry.json). eps^2_hat is their ratio; the interval
for eps^2 follows the same pair bootstrap and t rule and is carried to eps by the signed square root.

Usage
  python3 -I estimate.py --records DIR --out DIR [--workers 2]
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_mag_s3b():
    if "mag_s3b" in sys.modules:
        return sys.modules["mag_s3b"]
    spec = importlib.util.spec_from_file_location("mag_s3b", os.path.join(HERE, "mag_s3b.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3b"] = mod
    spec.loader.exec_module(mod)
    return mod


b = _load_mag_s3b()
m3, m2, P2, rng_for = b.m3, b.m2, b.P2, b.rng_for

N_PERM, N_BOOT, ALPHA = 1000, 1000, 0.05
STREAM_DIPOLE, STREAM_BOOT, STREAM_ANTI_PERM, STREAM_ANTI_BOOT = 40, 41, 42, 43
BOOT_CHUNK = 250


# --------------------------------------------------------------------------------------------------------------
# What the estimate knows about the bird
# --------------------------------------------------------------------------------------------------------------
def geometry(seed, K, n):
    """Receptor axes v, patch weights W and signs of the first n neurons, and the K presented directions, rebuilt
    from the streams of m3.Bird (1: retina, 2: neurons, 5: directions); base rates and drift are not drawn."""
    v, _, eyes = m2.retina(P2, rng_for(seed, 1))
    W, sign, _ = m2.neurons(replace(P2, n_neurons=m3.POOL_TUNED), rng_for(seed, 2), v, eyes)
    h = m2.uniform_dirs(rng_for(seed, 5), m3.MAX_PAIRS)[:K // 2]
    return v, W[:n], sign[:n], np.vstack([h, -h])


def templates(geo, S_assumed=1.0):
    """Even template s_even (cells x pairs) and odd template per unit eps t (cells x pairs) at assumed order S."""
    v, W, sign, dirs = geo
    N = dirs[:len(dirs) // 2]
    s_even = sign[:, None] * (W @ b.receptor_signal(v @ N.T, S_assumed, 1.0, 0.0))
    V = sign[:, None] * (W @ v)
    return s_even, b.depth(S_assumed) * (V @ N.T)


def cell_weights(C):
    var = C.reshape(C.shape[0], -1).var(1)
    return 1.0 / np.where(var > 0, var, 1.0)


# --------------------------------------------------------------------------------------------------------------
# Dipole test
# --------------------------------------------------------------------------------------------------------------
def perm_pvalue(null, e):
    """p value of the permutation test: (1 + #(null >= E)) / (1 + number of permutations), as in MAG-S3."""
    return float((1 + np.sum(null >= e - 1e-12 * abs(e))) / (1 + len(null)))


def t_interval(est, se, P):
    """95% interval and one-sided 95% upper bound of the t rule for P clusters, before clipping at zero."""
    scale = np.sqrt(P / (P - 1.0)) * se
    t975, t95 = stats.t.ppf(0.975, P - 1), stats.t.ppf(0.95, P - 1)
    return est - t975 * scale, est + t975 * scale, est + t95 * scale


def dipole_parts(C, sess, dirs):
    """Energy E of the dipole test and the matrices G_s of its permutation form: for flips f of n and -n per
    session, pair and repeat, E(f) = sum_s f_A,s^T G_s f_B,s, and E(all +1) = E."""
    n, K, Ra = C.shape
    P = K // 2
    C = C.astype(float)
    wgt = cell_weights(C)
    A, B = np.arange(Ra // 2), np.arange(Ra // 2, Ra)
    Q, _ = np.linalg.qr(dirs[:P])
    Dp = C[:, :P] - C[:, P:]
    a = (Dp[..., A].mean(-1) / 2) @ Q
    c = (Dp[..., B].mean(-1) / 2) @ Q
    e = float(np.sum(wgt[:, None] * a * c))
    _, sidx = np.unique(sess, return_inverse=True)
    n_sess = int(sidx.max()) + 1
    mask = np.kron(Q @ Q.T, np.ones((len(A), len(B))))
    G = np.empty((n_sess, P * len(A), P * len(B)))
    for s in range(n_sess):
        sel = sidx == s
        XA = Dp[sel][:, :, A].reshape(int(sel.sum()), -1)
        XB = Dp[sel][:, :, B].reshape(int(sel.sum()), -1)
        G[s] = ((XA * wgt[sel, None]).T @ XB) * mask / (4.0 * len(A) * len(B))
    return e, G, A, B


def dipole_test(C, sess, dirs, rng_perm, n_perm=N_PERM):
    n, K, Ra = C.shape
    P = K // 2
    e, G, A, B = dipole_parts(C, sess, dirs)
    n_sess = G.shape[0]
    null = np.empty(n_perm)
    chunk = max(1, int(2e6 // (n_sess * P * Ra)))
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        flips = rng_perm.integers(0, 2, size=(k, n_sess, P, Ra), dtype=np.int8).astype(float) * 2 - 1
        fa = flips[..., A].reshape(k, n_sess, -1)
        fb = flips[..., B].reshape(k, n_sess, -1)
        null[done:done + k] = np.einsum("ksa,sab,ksb->k", fa, G, fb, optimize=True)
        done += k
    p = perm_pvalue(null, e)
    return dict(E=e, p=p, odd_found=bool(p <= ALPHA), null_mean=float(null.mean()), null_sd=float(null.std()))


# --------------------------------------------------------------------------------------------------------------
# Estimate of eps_eff
# --------------------------------------------------------------------------------------------------------------
def even_odd_parts(C):
    P = C.shape[1] // 2
    C = C.astype(float)
    return (C[:, :P] + C[:, P:]).mean(-1) / 2, (C[:, :P] - C[:, P:]).mean(-1) / 2


def eps_fit(wp, E, D, s_even, t, w):
    """eps for pair weights wp (samples x pairs); gamma refitted per sample by weighted least squares with
    intercept."""
    Sw = wp.sum(1, keepdims=True)
    Sx, Sy = wp @ s_even.T, wp @ E.T
    Sxx, Sxy = wp @ (s_even ** 2).T, wp @ (s_even * E).T
    gam = (Sxy - Sx * Sy / Sw) / (Sxx - Sx * Sx / Sw)
    num, den = wp @ (t * D).T, wp @ (t * t).T
    return np.sum(w * gam * num, 1) / np.sum(w * gam * gam * den, 1), gam


def estimate_eps(C, s_even, t, rng_boot, n_boot=N_BOOT):
    E, D = even_odd_parts(C)
    w = cell_weights(np.asarray(C, dtype=float))
    P = E.shape[1]
    eps0, gam0 = eps_fit(np.ones((1, P)), E, D, s_even, t, w)
    boot = np.empty(n_boot)
    for a in range(0, n_boot, BOOT_CHUNK):
        k = min(BOOT_CHUNK, n_boot - a)
        wp = rng_boot.multinomial(P, np.full(P, 1.0 / P), size=k).astype(float)
        boot[a:a + k] = eps_fit(wp, E, D, s_even, t, w)[0]
    q = np.percentile(boot, [2.5, 50.0, 95.0, 97.5])
    raw, se = float(eps0[0]), float(boot.std())
    lo, hi, up = t_interval(raw, se, P)
    return dict(eps_hat_raw=raw, eps_hat=max(raw, 0.0), ci95=[max(lo, 0.0), max(hi, 0.0)], ci95_raw=[lo, hi],
                upper95=max(up, 0.0), upper95_raw=up, se=se, ci95_percentile=[float(q[0]), float(q[3])],
                upper95_percentile=float(q[2]), boot_median=float(q[1]),
                finite=bool(np.isfinite(raw) and np.all(np.isfinite(boot)) and np.isfinite(hi - lo)),
                n_gamma_negative=int(np.sum(gam0[0] <= 0)), n_cells=int(E.shape[0]), n_pairs=int(P))


# --------------------------------------------------------------------------------------------------------------
# One record
# --------------------------------------------------------------------------------------------------------------
def read_record(rec, with_tests=True, with_estimate=True, e3=None):
    seed, point = int(rec["seed"]), rec["point"]
    code = b.POINT_CODE[point]
    C, sess = rec["counts"], np.asarray(rec["sess"])
    n, K, _ = C.shape
    geo = geometry(seed, K, n)
    out = dict(seed=seed, point=point)
    if with_tests:
        out["dipole"] = dipole_test(C, sess, geo[3], rng_for(seed, STREAM_DIPOLE, code))
        out["antipodal"] = m3.antipodal_test(C, sess, rng_for(seed, STREAM_ANTI_PERM, code),
                                             rng_for(seed, STREAM_ANTI_BOOT, code))
    if with_estimate:
        S_true = b.POINT_S[point]
        s_even, t = templates(geo, 1.0)
        out["estimate"] = estimate_eps(C, s_even, t, rng_for(seed, STREAM_BOOT, code, 1))
        if S_true < 1.0:
            s_even, t = templates(geo, S_true)
            out["estimate_known_S"] = estimate_eps(C, s_even, t, rng_for(seed, STREAM_BOOT, code, 2))
        key = f"K{K}_n{n}"
        if e3 and S_true >= 1.0 and key in e3:
            out["estimate_e3"] = estimate_e3(C, geo[3], e3[key]["kappa2_mean"], rng_for(seed, STREAM_BOOT, code, 3))
    return out


def estimate_e3(C, dirs, kappa2, rng_boot, n_boot=N_BOOT):
    n, K, Ra = C.shape
    P = K // 2
    C = np.asarray(C, dtype=float)
    w = cell_weights(C)
    A, B = np.arange(Ra // 2), np.arange(Ra // 2, Ra)
    N = dirs[:P]
    Dp, Sp = C[:, :P] - C[:, P:], (C[:, :P] + C[:, P:]) / 2
    DA, DB = Dp[..., A].mean(-1) / 2, Dp[..., B].mean(-1) / 2
    E, v = Sp.mean(-1), Sp.var(-1, ddof=1) / Ra

    TA = (DA.T[:, :, None] * N[:, None, :]).reshape(P, -1)          # pairs x (cells * 3)
    TB = (DB.T[:, :, None] * N[:, None, :]).reshape(P, -1)
    NN = (N[:, :, None] * N[:, None, :]).reshape(P, 9)

    def fit(wp):
        k = wp.shape[0]
        Sw = wp.sum(1)
        Minv = np.linalg.inv((wp @ NN).reshape(k, 3, 3))
        XA, XB = (wp @ TA).reshape(k, n, 3), (wp @ TB).reshape(k, n, 3)
        energy = np.einsum("i,bij,bij->b", w, XA, np.einsum("bij,bkj->bik", XB, Minv))
        m1 = (wp @ E.T) / Sw[:, None]
        sig = np.maximum((wp @ (E * E).T) / Sw[:, None] - m1 * m1 - (wp @ v.T) / Sw[:, None], 0.0)
        return energy / (kappa2 * (sig @ w) * Sw / 3.0)

    e2 = float(fit(np.ones((1, P)))[0])
    boot = np.concatenate([fit(rng_boot.multinomial(P, np.full(P, 1.0 / P), size=min(BOOT_CHUNK, n_boot - a))
                               .astype(float)) for a in range(0, n_boot, BOOT_CHUNK)])
    se = float(boot.std())
    lo, hi, up = t_interval(e2, se, P)
    root = lambda x: float(np.sqrt(max(x, 0.0)))  # noqa: E731
    return dict(eps2_hat=e2, eps_hat_raw=float(np.sign(e2) * np.sqrt(abs(e2))), eps_hat=root(e2),
                ci95=[root(lo), root(hi)], ci95_eps2=[float(lo), float(hi)], upper95=root(up), se_eps2=se,
                finite=bool(np.isfinite(e2) and np.all(np.isfinite(boot))), kappa2=float(kappa2))


def _job(args):
    path, e3 = args
    r = read_record(b.load_record(path), e3=e3)
    r["record_sha256"] = b.file_sha256(path)
    return r


def main():
    from multiprocessing import get_context
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if glob.glob(os.path.join(a.records, "**", "*truth*"), recursive=True):
        sys.exit("I7: the records folder contains a truth file")
    paths = sorted(glob.glob(os.path.join(a.records, "*.npz")))
    e3_path = os.path.join(a.records, "e3_geometry.json")
    e3 = json.load(open(e3_path))["designs"] if os.path.exists(e3_path) else None
    os.makedirs(a.out, exist_ok=True)
    rows = []
    with get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, [(p, e3) for p in paths], chunksize=2):
            rows.append(r)
    rows.sort(key=lambda r: (r["point"], r["seed"]))
    path = os.path.join(a.out, "estimates.json")
    b.write_json(path, dict(estimates=rows, metadata=b.metadata("estimate")))
    digest = b.file_sha256(path)
    with open(os.path.join(a.out, "estimates.sha256"), "w") as fh:
        fh.write(f"{digest}  estimates.json  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    print(f"{len(rows)} records, estimates.json {digest}")


if __name__ == "__main__":
    main()
