#!/usr/bin/env python3
"""
Synthetic tests of MAG-S3b with known answers. The estimate is checked on hand-made worlds (linear counts with
known gamma and eps, Poisson noise and a shared gain term) and on noise-free records of the neural model; the pair
physics on exact symmetries; the receptor on closed-form moments. No test reads the cells or seeds of the hypotheses.

ESTIMATE_MODULE names the estimate file to test (default estimate); scripts/mutate_estimate.py runs these tests on
mutated copies, and each mutation must fail at least one test.

Usage: python3 -I test_s3b.py [--fast]        (exit code 0 when every test passes)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = load(os.environ.get("ESTIMATE_MODULE", "estimate"))
b = E.b
m3, m2, np = b.m3, b.m2, b.np
from dataclasses import replace  # noqa: E402

TEST_SEED = 1001          # a calibration bird (neutral checks only)


# --------------------------------------------------------------------------------------------------------------
# Hand-made worlds for the estimate
# --------------------------------------------------------------------------------------------------------------
def unit(rng, m):
    x = rng.standard_normal((m, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def world(rng, n=60, P=100, R=4, eps=0.05, shared=0.05, noise=True, ortho=False):
    """Counts C (n x 2P x R) with lambda(+-n_p) = c_i + gamma_i s_ip +- eps gamma_i t_ip, times a shared gain
    (1 + shared z) per presentation; Poisson counts when noise is True. Returns C, sess, dirs, s_even, t."""
    N = unit(rng, P)
    if ortho:
        N = np.linalg.qr(N)[0] * np.sqrt(P / 3.0)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    dirs = np.vstack([N, -N])
    s_even = rng.uniform(-1.0, 0.0, (n, P))
    t = rng.standard_normal((n, 3)) @ N.T
    gam = rng.uniform(5.0, 15.0, n)
    c = rng.uniform(25.0, 40.0, n)
    lp = c[:, None] + gam[:, None] * (s_even + eps * t)
    lm = c[:, None] + gam[:, None] * (s_even - eps * t)
    C = np.empty((n, 2 * P, R))
    for r in range(R):
        gp = np.maximum(1.0 + shared * rng.standard_normal(P), 0.0) if shared else 1.0
        gm = np.maximum(1.0 + shared * rng.standard_normal(P), 0.0) if shared else 1.0
        C[:, :P, r], C[:, P:, r] = lp * gp, lm * gm
    if noise:
        C = rng.poisson(C)
    return C, np.zeros(n, dtype=int), dirs, s_even, t


def test_estimate_exact_synthetic():
    """Noise-free linear world: eps_hat equals eps to 1e-10 with random and with orthogonal pair directions."""
    rng = np.random.default_rng(11)
    for ortho in (False, True):
        for eps in (0.0, 0.02, 0.1):
            C, _, _, s_even, t = world(rng, eps=eps, shared=0.0, noise=False, ortho=ortho)
            r = E.estimate_eps(C, s_even, t, np.random.default_rng(1), n_boot=50)
            assert abs(r["eps_hat_raw"] - eps) < 1e-10, (ortho, eps, r["eps_hat_raw"])
            assert r["finite"]


def coverage(make, est_fn, reps, eps):
    hit = 0
    for k in range(reps):
        rng = np.random.default_rng(1000 + k)
        C, _, _, s_even, t = make(rng)
        r = est_fn(C, s_even, t, np.random.default_rng(5000 + k))
        lo, hi = r["ci95_raw"]
        hit += lo <= eps <= hi
    return hit / reps


def two_way(C, s_even, t, rng_boot, n_boot=1000):
    """The pigeonhole bootstrap over cells and pairs (the rule of protocol v0.3), for comparison."""
    Ep, Dp = E.even_odd_parts(C)
    w = E.cell_weights(np.asarray(C, dtype=float))
    n, P = Ep.shape
    boot = np.empty(n_boot)
    for a in range(0, n_boot, 250):
        k = min(250, n_boot - a)
        wp = rng_boot.multinomial(P, np.full(P, 1.0 / P), size=k).astype(float)
        wc = rng_boot.multinomial(n, np.full(n, 1.0 / n), size=k).astype(float)
        boot[a:a + k] = E.eps_fit(wp, Ep, Dp, s_even, t, w * wc)[0]
    q = np.percentile(boot, [2.5, 97.5])
    return dict(ci95_raw=[float(q[0]), float(q[1])])


def test_bootstrap_coverage():
    """The 95% interval of the estimate (pair bootstrap, t rule) covers eps in 92-98% of 400 worlds with Poisson
    counts and a shared gain, in a world of 60 cells and 100 pairs and in one of 50 cells and 50 pairs like R3. The
    pigeonhole bootstrap over cells and pairs covers more than 98.5%: it counts the independent count noise once
    through the cells, once through the pairs and once through both."""
    rows = []
    for eps in (0.03, 0.1):
        big = lambda rng, e=eps: world(rng, n=60, P=100, R=4, eps=e, shared=0.05)  # noqa: E731
        small = lambda rng, e=eps: world(rng, n=50, P=50, R=8, eps=e, shared=0.05)  # noqa: E731
        rows.append((eps, coverage(big, E.estimate_eps, 400, eps), coverage(small, E.estimate_eps, 400, eps)))
    c_two = coverage(lambda rng: world(rng, n=60, P=100, R=4, eps=0.03, shared=0.05), two_way, 200, 0.03)
    print("    coverage: " + "; ".join(f"eps {e:g}: {a:.3f} (60 x 100), {s:.3f} (50 x 50)" for e, a, s in rows)
          + f"; cells and pairs {c_two:.3f}")
    assert all(0.92 <= a <= 0.98 and 0.92 <= s <= 0.98 for _, a, s in rows), rows
    assert c_two > 0.985, c_two


def test_dipole_size_and_power():
    """Even world with a shared gain (sigma 0.1): the dipole test rejects in 2-9% of 300 worlds; a planted dipole
    (eps = 0.1) is found in every one of 20 worlds."""
    rej = 0
    for k in range(300):
        rng = np.random.default_rng(20000 + k)
        C, sess, dirs, _, _ = world(rng, n=40, P=60, R=4, eps=0.0, shared=0.1)
        rej += E.dipole_test(C, sess, dirs, np.random.default_rng(30000 + k), n_perm=400)["odd_found"]
    size = rej / 300
    found = 0
    for k in range(20):
        rng = np.random.default_rng(40000 + k)
        C, sess, dirs, _, _ = world(rng, n=40, P=60, R=4, eps=0.1, shared=0.1)
        found += E.dipole_test(C, sess, dirs, np.random.default_rng(50000 + k), n_perm=400)["odd_found"]
    print(f"    dipole test: size {size:.3f}, power {found}/20 at eps 0.1")
    assert 0.02 <= size <= 0.09, size
    assert found == 20, found


def test_dipole_energy_identity():
    """The energy of the data computed cell by cell equals the energy of the dipole test, and the permutation form
    sum_s f_A^T G_s f_B with every flip +1 equals it too (three sessions)."""
    rng = np.random.default_rng(3)
    C, sess, dirs, _, _ = world(rng, n=30, P=40, R=4, eps=0.05, shared=0.1)
    sess = rng.integers(0, 3, len(sess))
    n, K, Ra = C.shape
    P = K // 2
    Cf = C.astype(float)
    w = E.cell_weights(Cf)
    Q, _ = np.linalg.qr(dirs[:P])
    Dp = Cf[:, :P] - Cf[:, P:]
    A, B = np.arange(Ra // 2), np.arange(Ra // 2, Ra)
    e_direct = 0.0
    for i in range(n):
        a = Q.T @ (Dp[i][:, A].mean(1) / 2)
        c = Q.T @ (Dp[i][:, B].mean(1) / 2)
        e_direct += w[i] * a @ c
    r = E.dipole_test(C, sess, dirs, np.random.default_rng(1), n_perm=10)
    assert abs(r["E"] - e_direct) < 1e-9 * max(1.0, abs(e_direct)), (r["E"], e_direct)
    e, G, _, _ = E.dipole_parts(C, sess, dirs)
    assert abs(G.sum() - e_direct) < 1e-9 * max(1.0, abs(e_direct)), (G.sum(), e_direct)   # all flips +1


def test_weights_as_mag_s3():
    """w_i = 1 / variance of all counts of the cell, the weights of the MAG-S3 antipodal test."""
    rng = np.random.default_rng(4)
    C = rng.poisson(20.0, (10, 8, 4)).astype(float)
    ref = 1.0 / C.reshape(10, -1).var(1)
    assert np.allclose(E.cell_weights(C), ref, rtol=0, atol=0)


# --------------------------------------------------------------------------------------------------------------
# The estimate on the neural model
# --------------------------------------------------------------------------------------------------------------
def test_geometry_matches_bird():
    bird = m3.Bird(TEST_SEED)
    for K, n in ((800, 200), (100, 50)):
        v, W, sign, dirs = E.geometry(TEST_SEED, K, n)
        assert np.array_equal(v, bird.v) and np.array_equal(W, bird.W[:n])
        assert np.array_equal(sign, bird.sign[:n]) and np.array_equal(dirs, bird.dirs(K))


def test_record_is_mag_s3_record():
    """A record at S = 1 has the counts of the compass record of m3.simulate for the same design, compared through
    the MAG-S3 fingerprint of the repeat means (bitwise)."""
    bird = m3.Bird(TEST_SEED)
    for point, eps in (("R1", 0.05), ("R3", 0.0)):
        rec, _ = b.make_record(bird, point, eps)
        d = replace(b.POINT_DESIGN[point], eps=eps)
        ref = m3.simulate(d, TEST_SEED, bird=bird, conditions=("compass",), shuffled=False)
        X = rec["counts"].astype(np.int64).mean(2).T
        assert m3.fingerprint(X) == ref["conditions"]["compass"]["fingerprint"], point


def test_receptor_at_full_order_is_ren():
    x = np.linspace(-1.0, 1.0, 20001)
    assert np.array_equal(b.receptor_signal(x, 1.0, 1.0, 0.07), m3.ren_signal(x, 0.07))
    bird = m3.Bird(TEST_SEED)
    d = replace(b.R1, eps=0.07)
    assert np.array_equal(b.modulation_order(bird, d, 1.0, 1.0, 0.07), m3.modulation(bird, d, "compass"))


def test_estimate_exact_neural():
    """Noise-free records of the neural model (expected counts, sigma_g = 0, no drift): eps_hat = eps to 1e-9 at
    S = 1 and, with the true S known, at S = 0.8. Calibration bird 1001."""
    bird = m3.Bird(TEST_SEED)
    for point, eps in (("C3", 0.05), ("T_S0.8", 0.05)):
        rec, _ = b.make_record(bird, point, eps, expected=True)
        geo = E.geometry(TEST_SEED, rec["K"], rec["n_cells"])
        s_even, t = E.templates(geo, b.POINT_S[point])
        r = E.estimate_eps(rec["counts"], s_even, t, np.random.default_rng(2), n_boot=50)
        assert abs(r["eps_hat_raw"] / eps - 1.0) < 1e-9, (point, r["eps_hat_raw"])


def test_i8_order_invisible_without_ciss():
    bird = m3.Bird(TEST_SEED)
    for S in b.S_LEVELS:
        u1 = b.modulation_order(bird, b.R1, S, 1.0, 0.0)
        for p in (0.5, 0.0):
            assert np.max(np.abs(b.modulation_order(bird, b.R1, S, p, 0.0) - u1)) <= 1e-12, (S, p)


# --------------------------------------------------------------------------------------------------------------
# Pair physics
# --------------------------------------------------------------------------------------------------------------
def test_forms_operators():
    s = b.spin()
    sim, PS = s.sim, s.PS
    M = int(round(np.trace(PS).real))
    Pc = m3.ciss_projector(sim, 0.1 * np.pi)
    assert np.max(np.abs(s.polarised(0.1 * np.pi, b.axis(0.0), +1.0) - Pc)) < 1e-14
    assert np.max(np.abs(s.turned(0.0, b.axis(30.0)) - PS)) < 1e-15
    for chi in (0.02, 0.1):
        for beta in (0.0, 45.0, 90.0):
            for Pm in (s.turned(chi * np.pi, b.axis(beta)), s.polarised(chi * np.pi, b.axis(beta), -1.0)):
                assert np.max(np.abs(Pm @ Pm - Pm)) < 1e-13 and abs(np.trace(Pm).real - M) < 1e-11
            # birth triplet share: sin^2 chi for K, sin^2 (chi/2) for P
            qK = 1.0 - np.trace(PS @ s.turned(chi * np.pi, b.axis(beta))).real / M
            qP = 1.0 - np.trace(PS @ s.polarised(chi * np.pi, b.axis(beta), 1.0)).real / M
            assert abs(qK - np.sin(chi * np.pi) ** 2) < 1e-12 and abs(qP - np.sin(chi * np.pi / 2) ** 2) < 1e-12


def test_exact_evenness():
    """Reciprocal K and the exponential model of P-same are even at a tilted axis; non-reciprocal K is not
    (MAG-S2 rates, which the hypotheses do not use)."""
    s = b.spin()
    dirs = m2.uniform_dirs(m3.rng_for(71, 1), 4)
    c = b.Cell("K", 0.1, 0.0, 0.0, 45.0, "cry4a")
    assert np.max(np.abs(b.forward_yields(s, c, dirs) - b.forward_yields(s, c, -dirs))) / 2 < 1e-12
    Pm = s.polarised(0.1 * np.pi, b.axis(45.0), 1.0)
    odd = [abs(m3.ciss_yield(s.system, Pm, n) - m3.ciss_yield(s.system, Pm, -n)) / 2 for n in dirs]
    assert max(odd) < 1e-12
    c = b.Cell("K", 0.1, 1.0, 0.0, 0.0, "mags2")
    assert np.max(np.abs(b.forward_yields(s, c, dirs) - b.forward_yields(s, c, -dirs))) / 2 > 1e-9


def test_harmonics():
    """Dipole convention, even/odd decoupling, zonal coefficients against a direct azimuthal average, and eps_mol
    against MAG-S3 pattern_parts."""
    half = m2.fibonacci_sphere(1000)
    a = np.array([0.3, -0.7, 0.2])
    f = lambda n: 0.5 + 0.1 * n[..., 2] ** 2 - 0.05 * n[..., 0] * n[..., 1] + n @ a * 1e-3  # noqa: E731
    yp, ym = f(half), f(-half)
    fit = b.harmonic_fit(half, yp, ym, 10)
    assert np.max(np.abs(b.dipole(fit) - a * 1e-3)) < 1e-14
    assert fit["even_rms_over_sd"] < 1e-12 and fit["odd_rms_over_sd"] < 1e-10
    zon = b.zonal(fit)
    for th in (0.3, 1.1, 2.5):
        ph = np.linspace(0, 2 * np.pi, 4001)[:-1]
        n = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.full_like(ph, np.cos(th))])
        assert abs(np.mean(f(n)) - b.npleg.legval(np.cos(th), zon)) < 1e-12
    pp = m3.pattern_parts(lambda n: f(n), half)
    assert abs(b.parity_parts(yp, ym)["eps_mol"] - pp["eps_equivalent"]) < 1e-15


def test_order_moments():
    from scipy import integrate
    for S in (0.8, 0.6, 0.4):
        k = b.kappa_for(S)
        mom = b.p_moments(k, 6)
        assert abs(mom[2] - S) < 1e-12
        for l in range(1, 7):
            Pl = lambda x, l=l: b.npleg.legval(x, np.eye(7)[l])  # noqa: E731
            num = integrate.quad(lambda x: Pl(x) * np.exp(k * (x - 1)), -1, 1, epsabs=1e-14, epsrel=1e-13)[0]
            den = integrate.quad(lambda x: np.exp(k * (x - 1)), -1, 1, epsabs=1e-14, epsrel=1e-13)[0]
            assert abs(mom[l] - num / den) < 1e-11, (S, l)
    assert 1.0 > b.depth(0.8) > b.depth(0.6) > b.depth(0.4) > 0.0


def test_i5_mixture():
    zon = np.array([0.5, 2e-4, -0.03, 5e-5, 0.01, -1e-5])
    r = b.i5_receptor(zon)
    for row in r.values():
        assert row["p0_max_odd"] < 1e-15 and row["p05_half_max_diff"] < 1e-15


def test_analysis_rules():
    """The verdict rules of analyze_s3b.py on hand-made inputs: a pair grid whose eps_mol is linear in m (H1 met,
    then broken), and inverse-problem rows with known coverage and detection (H5, H6, Q3)."""
    A = load("analyze_s3b")
    half = m2.fibonacci_sphere(500)
    ix = {}
    for c in b.pair_cells():
        amp = 1e-4 * (c.m if c.form == "K" else 0.01) * c.chi * (1.0 + c.alpha / 90.0) * np.cos(np.deg2rad(c.beta))
        tilt = np.array([np.sin(np.deg2rad(c.beta)), 0.0, np.cos(np.deg2rad(c.beta))])
        f = lambda n, amp=amp, tilt=tilt: 0.5 + 0.1 * n[..., 2] ** 2 + amp * (n @ tilt)  # noqa: E731
        ix[c.name] = b.cell_summary(c, half, f(half), f(-half))
    assert A.h1(ix)["verdict"] == "met"
    assert A.q2(ix)["verdict"] == "met" and A.q4(ix)["verdict"] == "met"
    broken = dict(ix)
    name = A.K(0.05, 0.25)
    broken[name] = dict(ix[name], parts=dict(ix[name]["parts"], eps_mol=ix[name]["parts"]["eps_mol"] * 1.2))
    assert A.h1(broken)["verdict"] == "not_met"
    for key in ("H2", "H3", "H4", "Q1"):
        assert getattr(A, key.lower())(ix)["verdict"] in ("met", "not_met")
    rows = []
    for e in b.EPS_GRID:
        for k in range(100):
            for point in ("R1", "R2", "R3"):
                hit = k < 95 if e > 0 else k >= 3
                ci = [e * 0.9, e * 1.1] if hit else [e + 0.01, e + 0.02]
                raw = list(ci)
                if e == 0:
                    ci, raw = ([0.0, 0.01], [-0.005, 0.01]) if hit else ([0.001, 0.01], [0.001, 0.01])
                found = e >= 0.02
                rows.append(dict(point=point, seed=k, eps_eff=e, estimate=dict(ci95=ci, ci95_raw=raw, upper95=ci[1]),
                                 dipole=dict(odd_found=found), antipodal=dict(odd_found=e >= 0.05)))
    assert A.h5(rows, None)["verdict"] == "met"
    assert A.h5(rows, "calibration C3 not met")["verdict"] == "untestable"
    r6 = A.h6(rows, None)
    assert r6["verdict"] == "met" and r6["dipole_threshold"] == 0.02 and r6["antipodal_threshold"] == 0.05
    assert A.q3(rows)["verdict"] == "met"
    for r in rows:
        if r["point"] == "R2" and r["eps_eff"] == 0.03 and r["seed"] < 20:
            r["estimate"] = dict(ci95=[0.05, 0.06], ci95_raw=[0.05, 0.06], upper95=0.06)
    assert A.h5(rows, None)["verdict"] == "not_met"


def icosa_pairs(rng, m):
    """m randomly turned icosahedra, 6 antipodal axes each: sum over the axes of n n^T is exactly (P / 3) I."""
    g = (1 + 5 ** 0.5) / 2
    ax = np.array([[0, 1, g], [0, -1, g], [1, g, 0], [-1, g, 0], [g, 0, 1], [g, 0, -1]], dtype=float)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    out = []
    for _ in range(m):
        q, r = np.linalg.qr(rng.standard_normal((3, 3)))
        out.append(ax @ (q * np.sign(np.diag(r))).T)
    return np.vstack(out)


def test_e3_exact_and_null():
    """E3 is exact without noise when every cell has the same kappa^2 and sum n n^T = (P / 3) I, and its eps^2 is
    centred on zero for an even code with Poisson noise and a shared gain."""
    rng = np.random.default_rng(21)
    N = icosa_pairs(rng, 20)
    P, n, kappa2 = len(N), 40, 9.0
    s_even = rng.uniform(-1.0, 0.0, (n, P))
    V = rng.standard_normal((n, 3))
    V *= np.sqrt(kappa2 * s_even.var(1) / np.sum(V ** 2, 1))[:, None]
    gam, c = rng.uniform(5.0, 15.0, n), rng.uniform(25.0, 40.0, n)
    for eps in (0.02, 0.1):
        t = V @ N.T
        lp = c[:, None] + gam[:, None] * (s_even + eps * t)
        lm = c[:, None] + gam[:, None] * (s_even - eps * t)
        C = np.concatenate([np.repeat(lp[:, :, None], 4, 2), np.repeat(lm[:, :, None], 4, 2)], axis=1)
        r = E.estimate_e3(C, np.vstack([N, -N]), kappa2, np.random.default_rng(1), n_boot=50)
        assert abs(r["eps_hat_raw"] / eps - 1.0) < 1e-9, (eps, r["eps_hat_raw"])
    vals = []
    for k in range(150):
        rng = np.random.default_rng(60000 + k)
        C, _, dirs, s_even, t = world(rng, n=40, P=60, R=4, eps=0.0, shared=0.1)
        vals.append(E.estimate_e3(C, dirs, kappa2, np.random.default_rng(k), n_boot=50)["eps2_hat"])
    vals = np.array(vals)
    assert abs(vals.mean()) < 3 * vals.std() / np.sqrt(len(vals)), (vals.mean(), vals.std())


def test_e4_pieces():
    """The fixed-azimuth mixture keeps only the x component of a dipole, and the E4 modulation with eps_x = 0 is the
    compass modulation of MAG-S3."""
    half = m2.fibonacci_sphere(500)
    d = np.array([0.4, -0.3, 0.8]) * 1e-3
    f = lambda m: 0.5 + 0.1 * m[..., 2] ** 2 + m @ d  # noqa: E731
    fit = b.harmonic_fit(half, f(half), f(-half), 4)
    zon = b.zonal(fit)
    r = b.e4_cell(half, fit["coef"], 4, b.dipole(fit), zon)
    grid = np.vstack([half, -half])
    expect = abs(d[0]) * np.abs(grid[:, 0]).max() / (0.1 * np.ptp(grid[:, 2] ** 2))
    assert abs(r["eps_mol_fixed_p0"] / expect - 1.0) < 1e-9, (r["eps_mol_fixed_p0"], expect)
    assert abs(r["eps_x"] - b.R_REN * d[0] / b.zonal_range(zon)) < 1e-15
    bird = m3.Bird(TEST_SEED)
    assert np.array_equal(b.modulation_fixed(bird, b.R1, 0.0), m3.modulation(bird, b.R1, "compass"))


def test_run_plan():
    """Every (point, seed) is one record, in the run (7 levels x 100 birds x 5 points, own birds per level) and in
    the smoke and blind checks."""
    full = b.run_plan()
    assert len(full) == 3500 and len({(p, s) for p, s, _ in full}) == 3500
    for e in b.EPS_GRID:
        assert len({s for _, s, x in full if x == e}) == 100
    chk = b.run_plan((9401, 9402))
    assert len({(p, s) for p, s, _ in chk}) == len(chk) == 10


def test_interval_rule_and_pvalue():
    """Known answers of the t rule, of its use inside the estimate, and of the permutation p value."""
    from scipy import stats
    lo, hi, up = E.t_interval(0.1, 0.01, 50)
    f = np.sqrt(50.0 / 49.0) * 0.01
    assert abs(lo - (0.1 - stats.t.ppf(0.975, 49) * f)) < 1e-15 and abs(hi - (0.1 + stats.t.ppf(0.975, 49) * f)) < 1e-15
    assert abs(up - (0.1 + stats.t.ppf(0.95, 49) * f)) < 1e-15
    rng = np.random.default_rng(31)
    C, _, _, s_even, t = world(rng, eps=0.0, shared=0.05)
    r = E.estimate_eps(C, s_even, t, np.random.default_rng(2), n_boot=200)
    lo, hi, up = E.t_interval(r["eps_hat_raw"], r["se"], r["n_pairs"])
    assert r["ci95_raw"] == [lo, hi] and r["upper95_raw"] == up
    assert r["ci95"] == [max(lo, 0.0), max(hi, 0.0)] and r["upper95"] == max(up, 0.0)
    assert r["eps_hat"] == max(r["eps_hat_raw"], 0.0)
    null = np.array([1.0, 2.0, 3.0])
    assert E.perm_pvalue(null, 2.0) == 0.75 and E.perm_pvalue(null, 4.0) == 0.25 and E.perm_pvalue(null, 0.0) == 1.0


def test_read_record_e8():
    """At S < 1 the main estimate assumes S = 1 and the second one knows S: on a noise-free record at S = 0.8 the
    second is exact and the first is not."""
    bird = m3.Bird(TEST_SEED)
    rec, _ = b.make_record(bird, "T_S0.8", 0.05, expected=True)
    r = E.read_record(rec, with_tests=False)
    assert abs(r["estimate_known_S"]["eps_hat_raw"] / 0.05 - 1.0) < 1e-9
    assert abs(r["estimate"]["eps_hat_raw"] / 0.05 - 1.0) > 1e-3


def fake_ix():
    return {c.name: dict(parts=dict(eps_mol=1e-4), receptor=dict(by_order={"S1_p1": dict(eps_eff=0.01)}),
                         d_odd=[0.0, 0.0, 1.0]) for c in b.pair_cells()}


def test_analysis_edges():
    """Each rule of H2, H3, H4, H6 and Q1 at its edge, with hand-made inputs."""
    A = load("analyze_s3b")
    ix = fake_ix()
    ix[A.P("P-same", 0.1)]["parts"]["eps_mol"], ix[A.P("P-LH", 0.1)]["parts"]["eps_mol"] = 1e-4, 2e-4
    ix[A.K(0.05, 1.0)]["parts"]["eps_mol"] = 2e-3
    assert A.h2(ix)["verdict"] == "met"
    ix[A.K(0.05, 1.0)]["parts"]["eps_mol"] = 1.999e-3
    assert A.h2(ix)["verdict"] == "not_met"
    for m, v in ((0.1, 0.01), (0.25, 0.03), (0.5, 0.05)):
        ix[A.K(0.1, m)]["receptor"]["by_order"]["S1_p1"]["eps_eff"] = v
    assert A.h3(ix)["verdict"] == "met"
    ix[A.K(0.1, 0.5)]["receptor"]["by_order"]["S1_p1"]["eps_eff"] = 0.0499
    assert A.h3(ix)["verdict"] == "not_met"
    for chi in b.CHI_LEVELS:
        for beta in (45.0, 80.0, 90.0):
            ix[A.K(chi, 1.0, beta)]["d_odd"] = [5.0, -3.0, np.cos(np.deg2rad(beta)) + 0.0999]
    assert A.h4(ix)["verdict"] == "met"
    ix[A.K(0.05, 1.0, 80.0)]["d_odd"][2] = np.cos(np.deg2rad(80.0)) + 0.1001
    assert A.h4(ix)["verdict"] == "not_met"
    for r_, v in A.MAG_S3_E5.items():
        ix[A.P("P-same", 0.1, 0.0, r_)]["parts"]["eps_mol"] = v * (1 + 5e-7) if r_ != "cry4a_fast" else 1.0
    assert A.q1(ix)["verdict"] == "met"
    ix[A.P("P-same", 0.1, 0.0, "mags2")]["parts"]["eps_mol"] = A.MAG_S3_E5["mags2"] * (1 + 2e-6)
    assert A.q1(ix)["verdict"] == "not_met"

    def h5_rows(per_level_hit, zero_out):
        out = []
        for point in ("R1", "R2", "R3"):
            for e in b.EPS_GRID:
                for k in range(100):
                    if e > 0:
                        ci = [e * 0.9, e * 1.1] if k < per_level_hit else [e + 0.01, e + 0.02]
                        raw = ci
                    else:
                        raw = [-0.002, -0.001] if k < zero_out else [-0.005, 0.01]
                        ci = [max(raw[0], 0.0), max(raw[1], 0.0)]
                    out.append(dict(point=point, seed=k, eps_eff=e, estimate=dict(ci95=ci, ci95_raw=raw)))
        return out
    assert A.h5(h5_rows(95, 10), None)["verdict"] == "met"          # 95% pooled, 10 below zero still allowed
    assert A.h5(h5_rows(95, 11), None)["verdict"] == "not_met"      # 11 intervals entirely below zero
    assert A.h5(h5_rows(93, 0), None)["verdict"] == "met"           # 93 per level, 93% pooled
    assert A.h5(h5_rows(92, 0), None)["verdict"] == "not_met"       # 92% pooled, under the band

    def rows(td, ta):
        out = []
        for e in b.EPS_GRID:
            nd = 100 if td is not None and e >= td else 50
            na = 100 if ta is not None and e >= ta else 50
            out += [dict(point="R1", seed=k, eps_eff=e, dipole=dict(odd_found=k < nd),
                         antipodal=dict(odd_found=k < na)) for k in range(100)]
        return out
    cases = {(0.02, 0.05): "met", (0.03, 0.05): "not_met", (0.05, None): "met", (0.1, None): "untestable",
             (None, 0.05): "not_met", (None, None): "untestable"}
    for (td, ta), want in cases.items():
        r = A.h6(rows(td, ta), None)
        assert r["verdict"] == want, (td, ta, r)


def test_blind_rates():
    """The substitute rates of the blind check differ from every rate set of the protocol and replace all three."""
    saved = dict(b.RATES)
    try:
        b.use_blind_rates()
        assert set(b.RATES) == set(saved) and not set(b.RATES.values()) & set(saved.values())
        assert not set(b.BLIND_RATES.values()) & set(b.m3.ANCHOR_RATES)
    finally:
        b.RATES.clear()
        b.RATES.update(saved)
    assert tuple(b.RATES.values()) == b.m3.ANCHOR_RATES


TESTS = [test_run_plan, test_blind_rates, test_interval_rule_and_pvalue, test_read_record_e8, test_analysis_edges, test_estimate_exact_synthetic, test_weights_as_mag_s3, test_dipole_energy_identity, test_geometry_matches_bird,
         test_receptor_at_full_order_is_ren, test_estimate_exact_neural, test_i8_order_invisible_without_ciss,
         test_forms_operators, test_harmonics, test_order_moments, test_i5_mixture, test_record_is_mag_s3_record,
         test_exact_evenness, test_e3_exact_and_null, test_e4_pieces, test_analysis_rules, test_dipole_size_and_power, test_bootstrap_coverage]
SLOW = {test_dipole_size_and_power, test_bootstrap_coverage}
ANALYSIS = {test_analysis_rules}


def main():
    """--fast skips the slow statistical tests and the analysis test; --no-analysis skips only the analysis test
    (the mutation harness of estimate.py uses it)."""
    fast, no_analysis = "--fast" in sys.argv, "--no-analysis" in sys.argv
    failed = []
    for t in TESTS:
        if (fast and t in SLOW | ANALYSIS) or (no_analysis and t in ANALYSIS):
            continue
        t0 = time.perf_counter()
        try:
            t()
            print(f"PASS {t.__name__} ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc(limit=2)
    print(f"{len(failed)} failed" if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
