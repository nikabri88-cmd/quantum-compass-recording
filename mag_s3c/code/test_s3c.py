#!/usr/bin/env python3
"""
Tests of MAG-S3c with known answers: the solver against the dense Liouville solve and exact time steps, exact zeros
and conservation of probability with relaxation, the algebra of the two relaxation superoperators, the control K-inc,
the E9 curves, the geometry and the dipolar tensor, the transfer through the receptor on patterns with known
harmonics, the mechanics of the modes, and the rules of analyze_s3c.py on synthetic rows, edges included.

Full pairs with CISS or relaxation run only at the MAG-S2 rates (k_F = k_R = 1e6 s^-1) or at the substitute rates of
the blind check (6e6, 5e5), never at the Cry4a rates of the hypotheses. The small pair N5 + N1 and the pair S2
without relaxation (the published cells of MAG-S3b) are the exceptions.

Usage: python3 -I test_s3c.py [--fast]        (exit code 0 when every test passes; --fast skips the slow tests)
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c = load("mag_s3c")
A = load("analyze_s3c")
b, m2, m3 = c.b, c.m2, c.m3
np = c.np

MAGS2 = (1.0e6, 1.0e6)
BLIND = (6.0e6, 5.0e5)
_SMALL = {}


def small():
    if "W_C" not in _SMALL:
        _SMALL["W_C"] = c.system_structure("W_C", small=True)
    return _SMALL["W_C"]


def rand_dir(seed):
    v = np.random.default_rng(seed).standard_normal(3)
    return v / np.linalg.norm(v)


def close(a, z, tol):
    a, z = np.asarray(a), np.asarray(z)
    err = float(np.max(np.abs(a - z)))
    assert err <= tol, f"difference {err:.3e} above {tol:.1e}"
    return err


# --------------------------------------------------------------------------------------------------------------
# Design and geometry
# --------------------------------------------------------------------------------------------------------------
def test_frozen_and_cells():
    assert c.frozen_mismatches() == []
    counts = {br: len(f()) for br, f in c.BRANCHES.items()}
    assert counts == {"main": 448, "E3": 18, "E5": 40, "E7": 12, "E10": 16}, counts
    names = [x.name for x in c.all_cells()]
    assert len(names) == len(set(names)) == 534
    assert len(c.E4_CELLS) == 18 and len(c.I6_CELLS) == 12 and len(c.CURVE_CASES) == 6
    jobs = c.odd_check_jobs()
    assert sum(j[0] == "I2" for j in jobs) == 41 and sum(j[0] == "I3" for j in jobs) == 94
    assert len({(j[0], j[1], j[2], j[3], j[4]) for j in jobs}) == len(jobs)
    combos = {(j[1], j[2]) for j in jobs if j[0] == "I3"}
    assert len(combos) == 18
    assert all(x.m == 1.0 for x in c.all_cells() if x.form == "K-inc")
    assert all(x.rates == "cry4a" for x in c.all_cells())


def test_geometry():
    g = c.geo()
    R = np.array(g["frame"])
    close(R @ R.T, np.eye(3), 1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    A_ = c.pdb_atoms(os.path.join(HERE, c.PDB_FILE))
    fad = {k[2]: v for k, v in A_.items() if k[0] == "FAD"}
    o = np.array([fad[n] for n in ("N10", "N5", "C9A", "C10", "C5X", "C4X")]).mean(axis=0)
    n10 = R @ (fad["N10"] - o)
    assert n10[0] > 0 and abs(n10[1]) < 1e-12 and abs(n10[2]) < 1e-12           # x points to N10
    mid = R @ (0.5 * (fad["C4X"] + fad["C10"]) - o)
    assert mid[1] > 0 and abs(mid[2]) < 1e-12                                     # y towards C4X-C10
    for t in c.TRP:
        assert abs(np.linalg.norm(g[t]["normal"]) - 1.0) < 1e-12 and g[t]["normal"][2] >= 0
        assert abs(np.linalg.norm(g[t]["r"]) - g[t]["dist"]) < 1e-12
    assert c.c4_geometry()["criterion_met"]


def test_dipolar_tensor():
    mu0_4pi, ge, muB = 1.00000000055e-7, 2.00231930436, 9.2740100783e-24
    for t, d_protocol in (("W_C", 0.3398), ("W_D", 0.1929)):
        r = c.geo()[t]["r"]
        T = c.dipolar_tensor(r)
        d = mu0_4pi * ge * muB / (np.linalg.norm(r) * 1e-10) ** 3 * 1e3            # mT
        assert abs(d - d_protocol) < 1e-4
        close(T, T.T, 0.0)
        w, v = np.linalg.eigh(T)
        close(w, [-2 * d, d, d], 1e-5 * d)
        assert abs(abs(v[:, 0] @ c.geo()[t]["r_hat"]) - 1.0) < 1e-12               # -2d along r
        close(T, d * (np.eye(3) - 3 * np.outer(c.geo()[t]["r_hat"], c.geo()[t]["r_hat"])), 1e-5 * d)


def test_tensors_of_structure():
    from radicalpy.data import Molecule
    db = {lab: np.linalg.eigvalsh(n.hfc.anisotropic) for lab, n in
          zip(("N5", "N10"), Molecule.fromdb("flavin_anion", nuclei=["N5", "N10"]).nuclei)}
    db["N1"] = np.linalg.eigvalsh(Molecule.fromdb("tryptophan_cation", nuclei=["N1"]).nuclei[0].hfc.anisotropic)
    z = np.array([0.0, 0.0, 1.0])
    worst_shift = 0.0
    for t in ("W_C", "W_D"):
        sy = c.build_system(t)
        nuclei = list(zip(("N5", "N10"), sy.sim.molecules[0].nuclei)) + [("N1", sy.sim.molecules[1].nuclei[0])]
        for lab, nuc in nuclei:
            T = np.asarray(nuc.hfc.anisotropic, dtype=float)
            w, v = np.linalg.eigh(T)
            i = int(np.argmax(np.abs(w)))
            j = int(np.argmax(np.abs(db[lab])))
            assert abs(w[i] - db[lab][j]) < 1e-12
            smalls = np.delete(db[lab], j)
            close(np.delete(w, i), [smalls.mean()] * 2, 1e-12)
            worst_shift = max(worst_shift, float(np.max(np.abs(smalls - smalls.mean()))))
            axis = z if lab != "N1" else c.geo()[t]["normal"]
            assert abs(abs(v[:, i] @ axis) - 1.0) < 1e-12
        noD = c.build_system(t + "_noD")
        close(sy.H0 - noD.H0, sy.sim.dipolar_hamiltonian(c.dipolar_tensor(c.geo()[t]["r"])), 1e-6)
        close(sy.H0, sy.H0.conj().T, 1e-6)
        gr = c.build_system({"W_C": "W_C_ErC", "W_D": "W_D_ErD"}[t])
        grn = list(zip(("N5", "N10"), gr.sim.molecules[0].nuclei)) + [("N1", gr.sim.molecules[1].nuclei[0])]
        for lab, nuc in grn:
            w, v = np.linalg.eigh(np.asarray(nuc.hfc.anisotropic, dtype=float))
            want = np.array(c.GRUNING_MHZ[t][lab]) / c.MHZ_PER_MT
            j = int(np.argmax(np.abs(want)))
            i = int(np.argmax(np.abs(w)))
            assert abs(w[i] - want[j]) < 1e-12
            close(np.delete(w, i), [np.delete(want, j).mean()] * 2, 1e-12)
            axis = z if lab != "N1" else c.geo()[t]["normal"]
            assert abs(abs(v[:, i] @ axis) - 1.0) < 1e-12
    assert worst_shift <= 0.0054 + 1e-6, worst_shift


# --------------------------------------------------------------------------------------------------------------
# Relaxation and solver
# --------------------------------------------------------------------------------------------------------------
def relax_apply(sy, kind, k, X):
    shift, terms = c.relax_terms(sy, kind, k)
    out = -2.0 * shift * X
    for coef, Op in terms:
        out = out + coef * (Op @ X @ Op)
    return out


def test_relaxation_algebra():
    sy = small()
    k = 1e6
    rng = np.random.default_rng(5)
    d = sy.dim
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Y = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    for n in (c.axis_vector("W_C", "r"), rand_dir(3)):
        sn = sy.n_sigma(n)
        CK = 1j * (sy.PS @ sn - sn @ sy.PS)                     # imaginary S-T coherence of K
        CP = sy.PS @ sn + sn @ sy.PS                            # real S-T coherence of P-same and P-LH
        for kind, rate_K, rate_P in (("rf", 2.0, 1.0), ("std", 1.0, 1.0)):
            close(relax_apply(sy, kind, k, CK), -rate_K * k * CK, 1e-9 * k)
            close(relax_apply(sy, kind, k, CP), -rate_P * k * CP, 1e-9 * k)
    for kind in ("rf", "std"):
        RX, RY = relax_apply(sy, kind, k, X), relax_apply(sy, kind, k, Y)
        assert abs(np.trace(RX)) < 1e-9 * k                                         # trace preserving
        assert abs(np.trace(X.conj().T @ RY) - np.trace(RX.conj().T @ Y)) < 1e-9 * k * d   # self-adjoint
        close(relax_apply(sy, kind, k, X.conj().T), RX.conj().T, 1e-9 * k)          # keeps Hermiticity
        close(relax_apply(sy, kind, k, sy.eye), 0.0 * sy.eye, 1e-9 * k)             # unital
    for r in (0, 1):
        for a in range(3):
            close(relax_apply(sy, "rf", k, sy.S[r][a]), -k * sy.S[r][a], 1e-9 * k)   # T1 = T2 = 1/k
    for P in (sy.PS, sy.PT, sy.PS @ X @ sy.PS):
        close(relax_apply(sy, "std", k, P), 0.0 * P, 1e-9 * k)                       # populations kept


def test_relaxation_matches_radicalpy():
    """The superoperators of the solver equal RandomFields and SingletTripletDephasing of RadicalPy 1.0.9, d rho/dt =
    -subH rho in the same row-major vectorisation (vec(A X B) = (A kron B^T) vec X)."""
    from radicalpy.data import Molecule
    from radicalpy.relaxation import RandomFields, SingletTripletDephasing
    from radicalpy.simulation import Basis, LiouvilleSimulation
    sy = small()
    ls = LiouvilleSimulation([Molecule.fromdb("flavin_anion", nuclei=["N5"]),
                              Molecule.fromdb("tryptophan_cation", nuclei=["N1"])], basis=Basis.ST)
    k = 1e6
    for kind, proc in (("rf", RandomFields(k)), ("std", SingletTripletDephasing(k))):
        proc.init(ls)
        shift, terms = c.relax_terms(sy, kind, k)
        mine = -2.0 * shift * np.eye(sy.dim ** 2) + sum(coef * np.kron(Op, Op.T) for coef, Op in terms)
        close(mine, -np.asarray(proc.subH), 1e-15 * k)


def forms_small(sy, chi=0.1 * np.pi):
    nb = c.axis_vector("W_C", "r")
    return {"K": (sy.turned(chi, nb), sy.PS), "K_m0.5": (sy.turned(chi, nb), sy.turned(0.5 * chi, nb)),
            "K-inc": (sy.dephased(sy.turned(chi, nb)), sy.PS),
            "P-same": (sy.polarised(chi, nb, 1.0), sy.polarised(chi, nb, 1.0)),
            "P-LH": (sy.polarised(chi, nb, 1.0), sy.polarised(chi, nb, -1.0))}


def test_solver_against_dense():
    sy = small()
    n = rand_dir(11)
    forms = forms_small(sy)
    cases = [(MAGS2, kind, k, f) for kind, k in (("none", 0.0), ("rf", 1e6), ("std", 1e7)) for f in forms]
    cases += [(BLIND, "rf", 1e7, "K"), (BLIND, "std", 1e6, "P-LH")]
    for (k_f, k_r), kind, k, form in cases:
        Pi, Pr = forms[form]
        f, rr, it, rel = c.solve(sy, n, Pi, Pr, kind, k, k_f, k_r)
        close(f, c.dense_yield(sy, n, Pi, Pr, kind, k, k_f, k_r), 1e-10)
        assert abs(f + rr - 1.0) < 1e-10 and rel < 1e-10, (form, kind, k, f + rr - 1.0, rel)
        assert (it == 0) == (kind == "none")


def test_solver_against_steps():
    sy = small()
    n = rand_dir(12)
    Pi, Pr = forms_small(sy)["K"]
    f = c.solve(sy, n, Pi, Pr, "rf", 1e6, *BLIND)[0]
    close(f, c.exact_steps_yield(sy, n, Pi, Pr, "rf", 1e6, *BLIND), 1e-6)


def test_gmres_failure_raises():
    sy = small()
    Pi, Pr = forms_small(sy)["K"]
    old = c.GMRES_MAXITER
    try:
        c.GMRES_MAXITER = 1
        try:
            c.solve(sy, rand_dir(4), Pi, Pr, "rf", 1e7, *BLIND, tol=1e-40)
        except RuntimeError:
            pass
        else:
            raise AssertionError("solve accepted a solution that did not converge")
    finally:
        c.GMRES_MAXITER = old


def test_frozen_regression_s2():
    """S2 without relaxation repeats the frozen solver of MAG-S3b on every I6 cell (published cells)."""
    s = b.spin()
    dirs = np.array([rand_dir(i) for i in (21, 22, 23)])
    sy = c.build_system("S2")
    close(sy.H0, s.system[1], 0.0)
    for cell in c.I6_CELLS:
        y = c.cell_yields(cell, dirs)[0]
        close(y, b.forward_yields(s, c.frozen_cell(cell), dirs), 1e-12)
        close(np.array(c.cell_operators(sy, cell)), np.array(b.cell_operators(s, c.frozen_cell(cell))), 1e-15)


def test_exact_zeros_full_pairs():
    """I2 (chi = 0) and I3 (reciprocal K) on the full pairs with relaxation, MAG-S2 rates, three directions."""
    dirs = [rand_dir(i) for i in (31, 32, 33)]
    for system in ("W_C", "W_D"):
        sy = c.build_system(system)
        P = sy.turned(0.1 * np.pi, c.axis_vector(system, "r"))
        for (Pi, Pr), kind, k in (((sy.PS, sy.PS), "rf", 1e6), ((sy.PS, sy.PS), "std", 1e7), ((P, P), "rf", 1e6)):
            for n in dirs:
                a = c.solve(sy, n, Pi, Pr, kind, k, *MAGS2)
                z = c.solve(sy, -n, Pi, Pr, kind, k, *MAGS2)
                assert abs(a[0] - z[0]) / 2 < 1e-12, (system, kind, abs(a[0] - z[0]))
                for s_ in (a, z):
                    assert abs(s_[0] + s_[1] - 1.0) < 1e-10 and s_[3] < 1e-10


# --------------------------------------------------------------------------------------------------------------
# Forms and the control K-inc
# --------------------------------------------------------------------------------------------------------------
def test_kinc_and_forms():
    sy = c.build_system("W_C")
    M = np.trace(sy.PS).real
    for chi in (0.05 * np.pi, 0.1 * np.pi):
        for n in (c.axis_vector("W_C", "r"), c.axis_vector("W_C", "x"), rand_dir(7)):
            sn = sy.n_sigma(n)
            rk = sy.turned(chi, n)
            inc = sy.dephased(rk)
            close(inc, np.cos(chi) ** 2 * sy.PS + np.sin(chi) ** 2 * sn @ sy.PS @ sn, 1e-13)
            close(sy.PS @ inc @ sy.PT, 0.0 * inc, 1e-14)                           # no S-T coherence
            for P in (sy.PS, sy.PT, sn @ sy.PS @ sn):
                assert abs(np.trace(P @ inc) - np.trace(P @ rk)) < 1e-12            # same populations
            for sign in (1.0, -1.0):
                Pp = sy.polarised(2 * chi, n, sign)
                close(sy.dephased(Pp), inc, 1e-13)                                  # equal populations at equal q
                close(Pp @ Pp, Pp, 1e-13)
                assert abs(np.trace(Pp).real - M) < 1e-10
            close(rk @ rk, rk, 1e-13)
            close(inc, inc.conj().T, 0.0)
    close(sy.turned(0.0, rand_dir(8)), sy.PS, 1e-15)
    cells = {x.form + str(x.m): x for x in c.main_cells() if x.system == "W_C" and x.axis == "r"
             and x.relax == "none" and x.chi == 0.1}
    Pi, Pr = c.cell_operators(sy, cells["K1.0"])
    close(Pr, sy.PS, 1e-15)                                  # turned(0) symmetrises the rounding of P_S
    Pi, Pr = c.cell_operators(sy, cells["K0.1"])
    close(Pr, sy.turned(0.09 * np.pi, c.axis_vector("W_C", "r")), 1e-15)
    Pi, Pr = c.cell_operators(sy, cells["K-inc1.0"])
    close(Pi, sy.dephased(sy.turned(0.1 * np.pi, c.axis_vector("W_C", "r"))), 0.0)
    close(Pr, sy.PS, 0.0)


# --------------------------------------------------------------------------------------------------------------
# Curves of E9
# --------------------------------------------------------------------------------------------------------------
def test_curves():
    sy = small()
    rho_k, rho_inc, delta = c.curve_states(sy, "W_C")
    assert abs(np.trace(rho_k).real - 1.0) < 1e-14 and abs(np.trace(delta)) < 1e-14
    obs = dict(trace=lambda r: float(np.trace(r).real), norm=lambda r: float(np.linalg.norm(r)))
    n = c.FIELD_UP
    t_max = 25.0 / BLIND[0]
    full = c.rk4_curve(sy, n, rho_k, sy.PS, "rf", 1e6, *BLIND, obs, t_max=t_max)
    close(full["forward_integral"], c.dense_yield(sy, n, rho_k, sy.PS, "rf", 1e6, *BLIND), 1e-6)
    steps = int(round(t_max / c.CURVE_DT))
    assert len(full["t"]) == steps // c.CURVE_EVERY + 1
    close(np.diff(full["t"]), c.CURVE_EVERY * c.CURVE_DT, 1e-20)
    out = {name: c.rk4_curve(sy, n, rho, sy.PS, "std", 1e6, *BLIND, obs, t_max=2e-7)
           for name, rho in (("K", rho_k), ("inc", rho_inc), ("delta", delta))}
    close(np.array(out["K"]["trace"]), np.array(out["inc"]["trace"]) + np.array(out["delta"]["trace"]), 1e-13)
    # without relaxation the memory |delta| decays no faster than the pair; with it, faster
    none = c.rk4_curve(sy, n, delta, sy.PS, "none", 0.0, *BLIND, obs, t_max=2e-7)
    rf = c.rk4_curve(sy, n, delta, sy.PS, "rf", 1e7, *BLIND, obs, t_max=2e-7)
    assert rf["norm"][-1] < none["norm"][-1]


# --------------------------------------------------------------------------------------------------------------
# Receptor transfer on a pattern with known harmonics
# --------------------------------------------------------------------------------------------------------------
def test_receptor_transfer():
    half = m2.fibonacci_sphere(c.GRID_HALF)
    a = np.array([0.001, -0.002, 0.003])

    def f(nv):
        return 0.5 + 0.1 * (nv[:, 2] ** 2 - 1.0 / 3.0) + 0.02 * (3.0 * nv[:, 0] ** 2 - 1.0) + nv @ a

    s = c.summarize(half, f(half), f(-half), e4=True)
    close(s["d_odd"], a, 1e-13)
    close(s["fit_check"]["d_odd"], a, 1e-13)
    assert s["fit_even_rms_over_sd"] < 1e-10
    close(s["zonal"][1], a[2], 1e-13)
    rng0 = 0.1 - 0.03                                             # even zonal part 0.5 - 0.1/3 + 0.01 + 0.07 x^2
    close(s["receptor"]["receptor_even_range"], rng0, 1e-9)
    close(s["receptor"]["eps_rec"], a[2] / rng0, 1e-8)
    close(s["receptor"]["by_order"]["S1_p1"]["eps_eff"], b.R_REN * a[2] / rng0, 1e-8)
    for S in b.S_LEVELS:
        for p in b.P_LEVELS:
            close(s["receptor"]["by_order"][f"S{S:g}_p{p:g}"]["eps_eff"],
                  p * b.g_order(S) * b.R_REN * a[2] / rng0, 1e-8)
    assert max(max(v["p0_max_odd"], v["p05_half_max_diff"]) for v in s["I7"].values()) < 1e-12
    close(s["E4"]["d_x"], a[0], 1e-13)
    close(s["E4"]["eps_x"], b.R_REN * a[0] / rng0, 1e-8)
    p = s["parts"]
    even = (f(half) + f(-half)) / 2
    close(p["even_range"], even.max() - even.min(), 1e-15)


# --------------------------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------------------------
def test_blind_rates():
    saved = dict(b.RATES)
    try:
        assert c.rates("cry4a") == (5.0e6, 3.4e5) and c.rates("mags2") == MAGS2
        c.use_blind_rates()
        assert c.rates("cry4a") == BLIND and c.rates("mags2") == (2.0e6, 2.0e6)
    finally:
        b.RATES.clear()
        b.RATES.update(saved)
    assert c.rates("cry4a") == (5.0e6, 3.4e5)
    tmp = tempfile.mkdtemp()
    try:
        cell = c.Cell("W_C", "r", "rf", 1e6, "K", 0.1, 1.0, "mags2", "test")
        os.makedirs(os.path.join(tmp, "cells"))
        c.use_blind_rates()
        c._cell_job((cell, os.path.join(tmp, "cells"), 2))
        with open(os.path.join(tmp, "cells", cell.name + ".json")) as fh:
            assert json.load(fh)["rates_used"] == [2.0e6, 2.0e6]
    finally:
        b.RATES.clear()
        b.RATES.update(saved)
        shutil.rmtree(tmp)
    tmp = tempfile.mkdtemp()
    try:
        for args in (["pair"], ["checks"], ["curves"], ["pair", "--blind-rates", "--confirmatory"]):
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "mag_s3c.py")] + args
                               + ["--out", os.path.join(tmp, "x")], capture_output=True, text=True)
            assert p.returncode != 0 and not os.path.exists(os.path.join(tmp, "x")), args
        for flag, want in (([], [5.0e6, 3.4e5]), (["--blind-rates"], list(BLIND))):
            out = os.path.join(tmp, "b" if flag else "a")
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "mag_s3c.py"), "calibrate", "--out", out,
                            "--checks", "C4"] + flag, check=True, capture_output=True)
            with open(os.path.join(out, "calibration.json")) as fh:
                assert json.load(fh)["metadata"]["rates"]["cry4a"] == want
    finally:
        shutil.rmtree(tmp)


def test_pair_and_receptor_modes():
    """The mechanics of pair (resumable) and receptor on two cells at the MAG-S2 rates and a grid of 2 x 6."""
    cells = [c.Cell("W_C", "r", "rf", 1e6, "K", 0.1, 1.0, "mags2", "test"),
             c.Cell("W_D", "x", "std", 1e6, "K-inc", 0.1, 1.0, "mags2", "test")]
    c.BRANCHES["test"] = lambda: list(cells)
    tmp = tempfile.mkdtemp()
    try:
        assert c.pair(tmp, 1, 0.0, half_n=6, branches=("test",))
        for x in cells:
            with open(os.path.join(tmp, "cells", x.name + ".json")) as fh:
                r = json.load(fh)
            assert r["grid_half"] == 6 and r["rates_used"] == list(MAGS2) and len(r["yp"]) == len(r["ym"]) == 6
            assert r["mag_s3c_sha256"] == c.file_sha256(os.path.join(HERE, "mag_s3c.py"))
            d = r["diagnostics"]
            assert d["max_rel_residual"] < 1e-10 and d["max_conservation_error"] < 1e-10
            y = c.cell_yields(x, m2.fibonacci_sphere(6))[0]
            close(y, r["yp"], 0.0)
        mtime = os.path.getmtime(os.path.join(tmp, "cells", cells[0].name + ".json"))
        assert c.pair(tmp, 1, 0.0, half_n=6, branches=("test",))
        assert os.path.getmtime(os.path.join(tmp, "cells", cells[0].name + ".json")) == mtime
        rows = c.receptor(tmp, branches=("test",))
        assert [r["name"] for r in rows] == [x.name for x in cells] and all(r["code_ok"] for r in rows)
        with open(os.path.join(tmp, "receptor.json")) as fh:
            rec = json.load(fh)
        assert rec["grid_half"] == 6 and rec["metadata"]["own"]["mag_s3c.py"] == c.file_sha256(
            os.path.join(HERE, "mag_s3c.py"))
    finally:
        del c.BRANCHES["test"]
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------------------------------------------
# Analysis on synthetic rows
# --------------------------------------------------------------------------------------------------------------
G_S = {1.0: 1.0, 0.8: 1.13, 0.6: 1.32, 0.4: 1.66}


def factor(relax, k):
    """Synthetic suppression by relaxation."""
    if relax == "none":
        return 1.0
    table = {1e5: 0.98, 3e5: 0.95, 1e6: 0.7 if relax == "rf" else 0.8, 3e6: 0.6, 1e7: 0.4}
    return table[k]


def synth_values(cl):
    """A world where every hypothesis is met with a known margin."""
    f = factor(cl.relax, cl.k)
    even = 0.2 / (1.0 + cl.k / 1e6) if cl.form == "K" else 0.2
    axis_share = {"z": 1.0, "x": 0.8, "r": 0.8}.get(cl.axis, 0.5)
    if cl.form in ("K", "K-inc"):
        dz = 0.004 * (cl.chi / 0.05) * axis_share * f * (1.0 if cl.m == 1.0 else 0.2)
        eps = 0.01 * (cl.chi / 0.05) * f * (1.0 if cl.m == 1.0 else 0.2)
        if cl.form == "K-inc":
            dz, eps = 0.05 * dz, 0.05 * eps
    else:
        dz, eps = 1e-5 * f, 5e-4 * f
    return dict(eps=eps, dz=dz, F=5.0 * dz, even=even)


def synth_row(cl, eps, dz, F, even):
    by = {f"S{S:g}_p{p:g}": dict(eps_eff=p * G_S[S] * F, eps_E6=0.9 * p * G_S[S] * F, pair_ratio=0.0, D_pair=1.0,
                                 g_pair=1.0) for S in b.S_LEVELS for p in b.P_LEVELS}
    d = [0.3 * dz, -0.2 * dz, dz]
    row = dict(name=cl.name, cell=asdict(cl), parts=dict(eps_mol=eps, even_range=even), d_odd=d,
               fit_even_rms_over_sd=1e-4, fit_odd_rms_over_sd=1e-3, zonal=[0.5, dz, 0.07],
               fit_check=dict(L=24, even_rms_over_sd=1e-5, odd_rms_over_sd=1e-4, d_odd=[x * (1 + 1e-4) for x in d]),
               receptor=dict(eps_rec=F / b.R_REN, receptor_even_range=0.07, by_order=by),
               odd_power_shares=dict(molecule=0.1, receptor=0.05),
               I7={f"S{S:g}": dict(p0_max_odd=0.0, p05_half_max_diff=1e-18) for S in b.S_LEVELS},
               code_ok=True, rates_used=[5.0e6, 3.4e5],
               diagnostics=dict(max_rel_residual=1e-14, max_conservation_error=1e-15, max_gmres_iterations=12))
    if cl.name in c.E4_CELLS:
        row["E4"] = dict(eps_mol_fixed_p0=0.01, d_x=d[0], eps_x=0.02, eps_y_p1=0.0, eps_z_p1=0.03)
    return row


def synth_rows(over=None):
    over = over or {}
    rows = []
    for cl in c.all_cells():
        v = synth_values(cl)
        v.update(over.get(cl.name, {}))
        rows.append(synth_row(cl, v["eps"], v["dz"], v["F"], v["even"]))
    return rows


def synth_checks():
    jobs = c.odd_check_jobs()
    key = lambda j: f"{j[1]}_{j[2]}_{j[3]}{j[4]:g}"  # noqa: E731
    return dict(I1=dict(frozen_mismatches=[]), I2={key(j): 1e-16 for j in jobs if j[0] == "I2"},
                I3={key(j): 1e-16 for j in jobs if j[0] == "I3"},
                I5_small=dict(max_dense=1e-14, max_steps=1e-7,
                              dense={f"{j[1]}_{j[2]}{j[3]:g}": 1e-14 for j in c.small_pair_cases() if j[0] == "dense"},
                              steps={f"{j[1]}_{j[2]}{j[3]:g}": 1e-7 for j in c.small_pair_cases() if j[0] == "steps"}),
                I6={x.name: dict(max_yield_diff=1e-15, eps_mol_rel=1e-12, eps_rec_rel=1e-12,
                                 receptor_even_range_rel=1e-12, d_odd_rel=1e-12) for x in c.I6_CELLS},
                diagnostics=dict(max_rel_residual=1e-14, max_conservation_error=1e-15, max_gmres_iterations=30),
                grid_half=c.GRID_HALF, rates_used=[5.0e6, 3.4e5], metadata=c.metadata("checks"))


def synth_curves():
    k_f = 5.0e6
    t = np.arange(0, 4001) * 1e-9
    cases = {}
    for g, relax, k in c.CURVE_CASES:
        kap = {"none": 0.0, "rf": 2.0 * k, "std": k}[relax]
        tr = np.exp(-k_f * t)
        cases[f"{g}_{relax}{k:g}"] = dict(
            K_up=dict(t=t.tolist(), trace=tr.tolist(), st_coherence=(0.3 * np.exp(-(k_f + kap) * t)).tolist(),
                      forward_integral=0.97, solver_forward_yield=0.97 + 1e-7, trace_end=float(tr[-1]),
                      solver_diagnostics=dict(max_rel_residual=1e-14, max_conservation_error=1e-15,
                                              max_gmres_iterations=11)),
            delta_up=dict(t=t.tolist(), trace=(1e-3 * np.exp(-(k_f + kap) * t)).tolist(),
                          norm=np.exp(-(k_f + kap) * t).tolist(), forward_integral=2e-4, trace_end=0.0),
            delta_down=dict(t=t.tolist(), trace=(-1e-3 * np.exp(-(k_f + kap) * t)).tolist(),
                            norm=np.exp(-(k_f + kap) * t).tolist(), forward_integral=-2e-4, trace_end=0.0))
    return dict(cases=cases, dt=c.CURVE_DT, t_max=c.CURVE_T, every=c.CURVE_EVERY, rates_used=[5.0e6, 3.4e5],
                metadata=c.metadata("curves"))


def synth_receptor(rows):
    return dict(cells=rows, grid_half=c.GRID_HALF, metadata=c.metadata("receptor"))


def synth_pair_done():
    return dict(n_cells=534, grid_half=c.GRID_HALF, metadata=c.metadata("pair"))


def run_analysis(rows=None, chk=None, cur=None, pair_done=True):
    return A.analyze(synth_receptor(synth_rows() if rows is None else rows), chk or synth_checks(),
                     cur or synth_curves(), synth_pair_done() if pair_done else None)


def hyp_of(over):
    return A.hypotheses(A.Grid(synth_rows(over)))


def name(*args, **kw):
    return c.Cell(*args, **kw).name


def test_analysis_world_and_counts():
    res = run_analysis()
    assert res["valid"], {k: v["verdict"] for k, v in res["invariants"].items()}
    assert res["blas_single_thread"] and res["pair_done"]
    assert res["comparisons"]["counted"] == A.N_COMPARISONS
    assert sum(A.N_COMPARISONS.values()) == 150
    h = res["hypotheses"]
    assert [h[k]["verdict"] for k in ("H1", "H2", "H4", "H5")] == ["met"] * 4
    assert h["H3"]["W_C"]["verdict"] == h["H3"]["W_D"]["verdict"] == "met"
    close(h["H1"]["smallest_margin"], 2.0, 1e-12)                  # eps_K / eps_P = 20, threshold 10
    close(h["H2"]["smallest_margin"], 1.6, 1e-12)                  # 0.8 / 0.5
    close(h["H4"]["smallest_margin"], 1.4, 1e-12)                  # 0.7 / 0.5 (rf)
    close(h["H5"]["smallest_margin"], 2.0, 1e-12)                  # 0.1 / 0.05
    hv = h["H4"]["halving"]
    assert len(hv) == 32 and hv["W_C_r_chi0.1_rf"]["falls_below_half_between"] == [3e6, 1e7]
    assert [k for k, _ in hv["W_C_r_chi0.1_rf"]["ratios"]] == [1e5, 3e5, 1e6, 3e6, 1e7]
    assert [k for k, _ in hv["S2_z_chi0.05_std"]["ratios"]] == [1e5, 1e6, 1e7]
    close(h["H3"]["W_C"]["comparisons"][1]["threshold_over_F"], 0.01 / (0.032 * 0.7), 1e-12)
    e = res["expectations"]
    assert [e[k]["verdict"] for k in ("Q1", "Q2", "Q3", "Q4")] == ["met"] * 4
    assert e["Q3"]["n"] > 0
    e9 = res["maps"]["E9"]
    assert abs(e9["W_C_rf1e+06"]["effective_coherence_time_s"] - 0.5e-6) <= 2e-9
    assert abs(e9["W_D_std1e+06"]["effective_coherence_time_s"] - 1.0e-6) <= 2e-9
    assert e9["W_C_none0"]["effective_coherence_time_s"] is None
    close(e9["W_C_none0"]["odd_coherence_contribution"], 2e-4, 1e-18)
    e10 = res["maps"]["E10"]["W_C_r_rf"]
    close([x["coherence_share"] for x in e10["levels"]], [0.95] * 6, 1e-12)
    assert e10["k_star"] is None
    tmp = tempfile.mkdtemp()
    try:
        A.write_table(os.path.join(tmp, "tables", "cells.csv"), res["maps"]["E1"])
        with open(os.path.join(tmp, "tables", "cells.csv")) as fh:
            assert len(fh.read().splitlines()) == 535
    finally:
        shutil.rmtree(tmp)


def test_analysis_edges():
    # H1: the ratio exactly at 10 passes, just below fails (binary fractions make the ratios exact)
    kc = c.Cell("W_C", "r", "rf", 1e6, "K", 0.05, 1.0)
    pc = c.Cell("W_C", "r", "rf", 1e6, "P-LH", 0.1, 1.0)
    assert hyp_of({kc.name: dict(eps=0.625), pc.name: dict(eps=0.0625)})["H1"]["verdict"] == "met"
    h = hyp_of({kc.name: dict(eps=0.625), pc.name: dict(eps=0.0625 * 1.0001)})["H1"]
    assert h["verdict"] == "not_met" and h["n_failed"] == 1 and h["smallest_margin"] < 1.0
    # H2: exactly half passes, below fails, a vanishing denominator is excluded, all excluded is untestable
    z = c.Cell("W_D", "z", "std", 1e6, "K", 0.1, 1.0)
    x = c.Cell("W_D", "x", "std", 1e6, "K", 0.1, 1.0)
    r = c.Cell("W_D", "r", "std", 1e6, "K", 0.1, 1.0)
    assert hyp_of({z.name: dict(dz=0.0625), x.name: dict(dz=0.03125), r.name: dict(dz=0.0625)})["H2"]["verdict"] \
        == "met"
    h = hyp_of({z.name: dict(dz=0.0625), x.name: dict(dz=-0.03125 * 0.9999), r.name: dict(dz=0.0625)})["H2"]
    assert h["verdict"] == "not_met" and h["n_failed"] == 1
    h = hyp_of({z.name: dict(dz=1e-12)})["H2"]                     # 1e-12 < 1e-9 x even range
    assert h["verdict"] == "met" and len(h["excluded"]) == 2 and h["n"] == 22
    zs = {name(g, "z", rl, k, "K", chi, 1.0): dict(dz=0.0) for g in ("W_C", "W_D") for chi in (0.05, 0.1)
          for rl, k in c.KEY_LEVELS}
    h = hyp_of(zs)["H2"]
    assert h["verdict"] == "untestable" and len(h["excluded"]) == 24
    # H3: |F| at the threshold passes, below fails, separately for W_C and W_D
    rr = c.Cell("W_D", "r", "rf", 1e6, "K", 0.1, 1.0)
    h = hyp_of({rr.name: dict(F=-0.01)})["H3"]
    assert h["W_D"]["verdict"] == "met"
    h = hyp_of({rr.name: dict(F=0.00999)})["H3"]
    assert h["W_D"]["verdict"] == "not_met" and h["W_C"]["verdict"] == "met"
    # H4: the ratio at 0.5 passes, a change of sign fails
    s0 = c.Cell("S2", "x", "none", 0.0, "K", 0.05, 1.0)
    s1 = c.Cell("S2", "x", "std", 1e6, "K", 0.05, 1.0)
    s2 = c.Cell("S2", "x", "rf", 1e6, "K", 0.05, 1.0)
    assert hyp_of({s0.name: dict(dz=0.0625), s1.name: dict(dz=0.03125), s2.name: dict(dz=0.0625)})["H4"]["verdict"] \
        == "met"
    h = hyp_of({s0.name: dict(dz=0.0625), s1.name: dict(dz=-0.0625), s2.name: dict(dz=0.0625)})["H4"]
    assert h["verdict"] == "not_met" and h["n_failed"] == 1
    # H5: |d_inc| at 10% passes, above fails
    kk = c.Cell("W_C", "x", "none", 0.0, "K", 0.1, 1.0)
    ki = c.Cell("W_C", "x", "none", 0.0, "K-inc", 0.1, 1.0)
    assert hyp_of({kk.name: dict(dz=1.0), ki.name: dict(dz=-0.1)})["H5"]["verdict"] == "met"
    h = hyp_of({kk.name: dict(dz=1.0), ki.name: dict(dz=0.1000001)})["H5"]
    assert h["verdict"] == "not_met" and h["n_failed"] == 1
    # Q3: an even range that grows with k fails
    q = c.Cell("W_C", "r", "rf", 1e7, "K", 0.1, 1.0)
    rows = synth_rows({q.name: dict(even=1.0)})
    assert A.q3(rows)["verdict"] == "not_met" and len(A.q3(rows)["failed"]) == 1
    # Q2: a cell with F = 0 without relaxation is a failure, not a crash
    q2c = c.Cell("W_D", "z", "none", 0.0, "P-same", 0.1, 1.0)
    out = A.q2(A.Grid(synth_rows({q2c.name: dict(F=0.0)})))
    assert out["verdict"] == "not_met" and len(out["failed"]) == 2
    # Q1: a vanishing dipole is listed and left out
    rows = synth_rows({q2c.name: dict(dz=0.0)})
    assert A.q1(rows)["excluded_from_dipole"] == [q2c.name]
    # E10: log-linear crossing between the bracketing levels
    ks = [0.0, 1e5, 3e5, 1e6, 3e6, 1e7]
    out = A.e10_threshold(ks, [0.04, 0.03, 0.02, 0.012, 0.008, 0.004])
    f = math.log(0.01 / 0.012) / math.log(0.008 / 0.012)
    close(out["k_star"] / math.exp(math.log(1e6) + f * math.log(3.0)), 1.0, 1e-12)
    assert out["bracket"] == [1e6, 3e6]
    assert A.e10_threshold(ks, [0.005] * 6)["k_star"] is None
    assert A.e10_threshold(ks, [0.02, 0.005, 0, 0, 0, 0])["k_star"] is None    # crossing at the first level
    assert A.e10_threshold(ks, [0.02] * 6)["k_star"] is None
    # invariants: one cell from other code, a missing check entry, an I6 miss
    rows = synth_rows()
    rows[5]["code_ok"] = False
    chk = synth_checks()
    chk["I2"].pop(next(iter(chk["I2"])))
    chk["I6"][c.I6_CELLS[0].name]["max_yield_diff"] = 2e-12
    res = run_analysis(rows, chk)
    inv = res["invariants"]
    assert (inv["I1"]["verdict"], inv["I2"]["verdict"], inv["I6"]["verdict"]) == ("error",) * 3 and not res["valid"]
    chk = synth_checks()
    chk["I5_small"]["steps"].pop(next(iter(chk["I5_small"]["steps"])))
    assert run_analysis(chk=chk)["invariants"]["I5"]["verdict"] == "error"
    # validity: grid, rates, curves, BLAS threads, the pair marker, missing cells
    rec = synth_receptor(synth_rows())
    rec["grid_half"] = 250
    assert not A.analyze(rec, synth_checks(), synth_curves(), synth_pair_done())["valid"]
    rows = synth_rows()
    rows[0]["rates_used"] = list(BLIND)
    assert not run_analysis(rows)["rates_ok"]
    for change in ("t_max", "every", "case", "kind"):
        cur = synth_curves()
        if change == "t_max":
            cur["t_max"] = 5e-8
        elif change == "every":
            cur["every"] = 8
        elif change == "case":
            cur["cases"].pop(next(iter(cur["cases"])))
        else:
            next(iter(cur["cases"].values())).pop("delta_down")
        res = run_analysis(cur=cur)
        assert not res["valid"] and not res["curves_full"], change
        assert res.get("incomplete", False) == (change in ("case", "kind")), change
    chk = synth_checks()
    chk["metadata"] = dict(chk["metadata"], blas=[dict(chk["metadata"]["blas"][0], num_threads=2)])
    res = run_analysis(chk=chk)
    assert not res["blas_single_thread"] and not res["valid"]
    assert not run_analysis(pair_done=False)["valid"]
    res = run_analysis(synth_rows()[:-1])
    assert not res["cells_complete"] and not res["valid"] and len(res["missing"]) == 1
    # H4 halving leaves out the combinations whose denominator vanishes
    s0 = c.Cell("W_D", "x", "none", 0.0, "K", 0.05, 1.0)
    hv = hyp_of({s0.name: dict(dz=0.0)})["H4"]["halving"]
    assert hv["W_D_x_chi0.05_rf"]["falls_below_half_between"] is None and "excluded" in hv["W_D_x_chi0.05_rf"]


def test_analysis_cli():
    tmp = tempfile.mkdtemp()
    try:
        run, out = os.path.join(tmp, "run"), os.path.join(tmp, "results")
        os.makedirs(run)
        for nm, obj in (("receptor", synth_receptor(synth_rows())), ("checks", synth_checks()),
                        ("curves", synth_curves()), ("pair_done", synth_pair_done())):
            with open(os.path.join(run, f"{nm}.json"), "w") as fh:
                json.dump(obj, fh)
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "analyze_s3c.py"), "--run", run, "--out", out],
                           check=True, capture_output=True, text=True)
        assert p.stdout.startswith("valid")
        with open(os.path.join(out, "results.json")) as fh:
            res = json.load(fh)
        assert res["valid"] and set(res["maps"]) == {f"E{i}" for i in range(1, 11)}
        assert os.path.exists(os.path.join(out, "tables", "cells.csv"))
        with open(os.path.join(run, "receptor.json"), "w") as fh:
            json.dump(synth_receptor(synth_rows()[:-3]), fh)
        out2 = os.path.join(tmp, "results2")
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "analyze_s3c.py"), "--run", run, "--out", out2],
                           capture_output=True, text=True)
        assert p.returncode != 0 and not os.path.exists(os.path.join(out2, "results.json"))
        assert os.path.exists(os.path.join(out2, "results_incomplete.json"))
    finally:
        shutil.rmtree(tmp)


def test_pipeline_gates():
    """The confirmatory run refuses to start without a confirmation of the exact files (nothing is computed)."""
    tmp = tempfile.mkdtemp()
    try:
        sh = os.path.join(HERE, "scripts", "pipeline.sh")
        p = subprocess.run(["bash", sh, "confirmatory", tmp], capture_output=True, text=True)
        assert p.returncode == 1 and "confirmation" in p.stdout + p.stderr
        hashes = {f: c.file_sha256(os.path.join(HERE, f)) for f in ("mag_s3c.py", "analyze_s3c.py")}
        hashes["mag_s3c.py"] = "0" * 64
        for answer in ("запускай", "да"):
            with open(os.path.join(tmp, "confirmation.json"), "w") as fh:
                json.dump(dict(answer=answer, time_utc="2026-11-09T00:00:00Z", sha256=hashes), fh, ensure_ascii=False)
            p = subprocess.run(["bash", sh, "confirmatory", tmp], capture_output=True, text=True)
            assert p.returncode == 1
        assert os.listdir(tmp) == ["confirmation.json"]
        p = subprocess.run(["bash", sh, "nonsense", tmp], capture_output=True, text=True)
        assert p.returncode == 2
    finally:
        shutil.rmtree(tmp)


def test_notebook_build_reproducible():
    tmp = tempfile.mkdtemp()
    try:
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "mag_s3c.py"), "calibrate", "--out",
                        os.path.join(tmp, "ref"), "--checks", "C4"], check=True, capture_output=True)
        digests = []
        for i in range(2):
            out = os.path.join(tmp, f"nb{i}.ipynb")
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "scripts", "make_colab_notebook.py"), HERE,
                            os.path.join(tmp, "ref"), out], check=True, capture_output=True)
            digests.append(c.file_sha256(out))
            time.sleep(1.1)
        assert digests[0] == digests[1]
        with open(os.path.join(tmp, "nb0.ipynb")) as fh:
            nb = json.load(fh)
        cell5 = "".join(nb["cells"][6]["source"])
        assert cell5.index("BLIND_OK") < cell5.index("input(") < cell5.index("pipeline.sh confirmatory")
    finally:
        shutil.rmtree(tmp)


TESTS = [test_frozen_and_cells, test_geometry, test_dipolar_tensor, test_tensors_of_structure,
         test_relaxation_algebra, test_relaxation_matches_radicalpy, test_solver_against_dense, test_gmres_failure_raises, test_frozen_regression_s2,
         test_exact_zeros_full_pairs, test_kinc_and_forms, test_curves, test_receptor_transfer, test_blind_rates,
         test_pair_and_receptor_modes, test_analysis_world_and_counts, test_analysis_edges, test_analysis_cli,
         test_pipeline_gates, test_notebook_build_reproducible, test_solver_against_steps]
SLOW = {test_solver_against_steps}


def main():
    fast = "--fast" in sys.argv
    failed = []
    for t in TESTS:
        if fast and t in SLOW:
            continue
        t0 = time.perf_counter()
        try:
            t()
            print(f"PASS {t.__name__} ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc(limit=3)
    print(f"{len(failed)} failed" if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
