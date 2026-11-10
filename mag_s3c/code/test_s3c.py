#!/usr/bin/env python3
"""
Tests of MAG-S3c with known answers: the solver against the dense Liouville solve and exact time steps, exact zeros
and conservation of probability with relaxation, the algebra of the two relaxation superoperators, the controls and
form E (against the formation equations of Denton et al. 2026), the reconstruction from the references and the split
of K, the two-site solver against the dense solve, the crystal 222, the E9 curves, the geometry and the dipolar tensor,
the transfer through the receptor on patterns with known harmonics, the mechanics of the modes, and the rules of
analyze_s3c.py on a synthetic world, edges included.

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
    d, r, refs, cc = c.direct_cells(), c.rebuilt_cells(), c.references(), c.composite_cells()
    assert (len(d), len(r), len(refs), len(cc)) == (574, 442, 54, 24)
    names = [x.name for x in d + r]
    assert len(names) == len(set(names)) and len({x.name for x in refs}) == 54
    assert {c.ref_of(x).name for x in r} <= {x.name for x in refs}
    assert sum(x.branch == "main" for x in d) == 484 and sum(x.branch == "main" for x in r) == 336
    assert sum(x.branch != "main" for x in d) == 90 and sum(x.branch != "main" for x in r) == 106
    assert all(x.m == 1.0 for x in d + r if x.form in ("K-inc", "K-iso", "P-0"))
    assert all(x.form in c.REBUILT_FORMS for x in r)
    assert not any(x.form in ("K-iso", "P-0") for x in d)
    assert len(c.E4_CELLS) == 24 and len(c.I6_CELLS) == 12 and len(c.CURVE_CASES) == 8
    jobs = c.odd_check_jobs()
    assert len(jobs) == len(set(jobs)) == 146 and len({(j[1], j[2]) for j in jobs}) == 18
    assert len(c.e5_references()) == 38


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
            "P-LH": (sy.polarised(chi, nb, 1.0), sy.polarised(chi, nb, -1.0)),
            "E-M1": (sy.bridge(chi, nb, c.eta("M1")), sy.turned(chi, nb))}


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
            iso = sy.isotropic(chi)
            for P in (sy.PS, sy.PT):
                assert abs(np.trace(P @ inc) - np.trace(P @ rk)) < 1e-12            # same populations
                assert abs(np.trace(P @ iso) - np.trace(P @ rk)) < 1e-12
            for sign in (1.0, -1.0):
                Pp = sy.polarised(2 * chi, n, sign)
                close(sy.dephased(Pp), inc, 1e-13)                                  # equal populations at equal q
                close(Pp @ Pp, Pp, 1e-13)
                assert abs(np.trace(Pp).real - M) < 1e-10
            close(rk @ rk, rk, 1e-13)
            C, Cr, A_ = sy.coherence_ops(n)
            close(rk, np.cos(chi) ** 2 * sy.PS + np.sin(chi) * np.cos(chi) * C + np.sin(chi) ** 2 * A_, 1e-13)
            close(sy.bridge(chi, n, 1.0 + 0.0j), rk, 1e-13)                         # eta = 1: form K
            E = sy.bridge(chi, n, c.eta("M1"))
            close(E, E.conj().T, 0.0)
            assert abs(np.trace(E).real - M) < 1e-10
    cells = {x.form + str(x.m): x for x in c.direct_cells(("main",)) if x.system == "W_C" and x.axis == "r"
             and x.relax == "none" and x.chi == 0.1}
    Pi, Pr = c.cell_operators(sy, cells["K1.0"])
    close(Pr, sy.PS, 1e-15)                                  # turned(0) symmetrises the rounding of P_S
    Pi, Pr = c.cell_operators(sy, cells["K0.1"])
    close(Pr, sy.turned(0.09 * np.pi, c.axis_vector("W_C", "r")), 1e-15)
    Pi, Pr = c.cell_operators(sy, cells["K-inc1.0"])
    close(Pi, sy.dephased(sy.turned(0.1 * np.pi, c.axis_vector("W_C", "r"))), 0.0)
    close(Pr, sy.PS, 0.0)
    e = [x for x in c.direct_cells(("E11",)) if x.axis == "z" and x.relax == "none" and x.chi == 0.1][0]
    Pi, Pr = c.cell_operators(sy, e)
    close(Pr, sy.turned(0.1 * np.pi, b.axis(0.0)), 0.0)
    assert abs(abs(c.eta("M1")) - 0.074) < 5e-4 and abs(abs(c.eta("M2")) - 0.276) < 5e-3


def test_reconstruction():
    """The yield of every rebuilt form follows from the singlet-born reference at the reversed field (C5 at the
    MAG-S2 rates), the split of K holds term by term, and the basis weights rebuild any CISS axis."""
    sy = small()
    nb = c.axis_vector("W_C", "r")
    W10 = c.basis_weights(sy)
    for kind, k in (("none", 0.0), ("rf", 1e6), ("std", 1e7), ("std", 1e8)):
        n = rand_dir(41)
        X = c.solve_X(sy, n, sy.PS, sy.PS, kind, k, *MAGS2)[0]
        cn = c.pauli_coefficients(sy, X)
        for form in c.REBUILT_FORMS:
            Pi, Pr = c.birth_operators(sy, form, 0.1 * np.pi, 1.0, nb)
            close(Pr, sy.PS, 1e-15)
            rebuilt = 1.0 - MAGS2[1] * (cn @ c.pauli_weights(sy, Pi))
            close(rebuilt, c.solve(sy, -n, Pi, sy.PS, kind, k, *MAGS2)[0], 1e-12)
        axes = np.array([rand_dir(s) for s in (42, 43, 44)])
        wk, wi = c.split_weights(W10, 0.07 * np.pi, axes)
        for j, ax in enumerate(axes):
            close(wk[j], c.pauli_weights(sy, sy.turned(0.07 * np.pi, ax)), 1e-15)
            close(wi[j], c.pauli_weights(sy, sy.dephased(sy.turned(0.07 * np.pi, ax))), 1e-15)
    # reference job and rebuilt yields on a tiny grid
    half = m2.fibonacci_sphere(4)
    ref = c.Ref("W_C", "rf", 1e6, "mags2")
    sy_f = c.build_system("W_C")
    yp, cp, dp = c.ref_solutions(ref, half)
    ym, cm, dm = c.ref_solutions(ref, -half)
    assert dp["max_rel_residual"] < 1e-10 and dm["max_conservation_error"] < 1e-10
    rp, rm = c.rebuilt_yields(c.pauli_weights(sy_f, sy_f.PS), cp, cm, MAGS2[1])
    close(rp, yp, 1e-13)
    cell = c.Cell("W_C", "x", "rf", 1e6, "P-0", 0.1, 1.0, "mags2", "test", "ref")
    rp, rm = c.rebuilt_yields(c.cell_weights(sy_f, cell), cp, cm, MAGS2[1])
    close(rp, c.cell_yields(cell, half)[0], 1e-12)
    close(rm, c.cell_yields(cell, -half)[0], 1e-12)


def test_form_E_and_denton():
    r = c.c6_denton()
    assert r["criterion_met"], r
    assert abs(r["eq3_operator_over_spin_matrices"] + 1.0) < 1e-12      # Eq. (3) as printed has the other sign


def test_composite_pair():
    cs = c.build_composite(tiny=True)
    X = np.random.default_rng(3).standard_normal((cs.D, cs.D))
    close(cs.hop(cs.hop(X)), X, 0.0)                                     # the hop is an involution
    n = rand_dir(51)
    H = cs.H(0, n)
    close(H, H.conj().T, 1e-15 * np.abs(H).max())
    nb = c.axis_vector("W_C", "r")
    for form in ("none", "K"):
        P = cs.sites[0].PS if form == "none" else cs.sites[0].turned(0.1 * np.pi, nb)
        rho0 = cs.birth(P)
        for k_out in ((2e7, 2e6), (2e8, 2e8), (1e9, 1e9), (1e9, 1e8)):
            for k_rec in ((5e5, 0.0), (5e5, 5e5)):
                f, rr, it, rel, rec = c.composite_solve(cs, n, rho0, k_rec, k_out, 6e6)
                close(f, c.composite_dense(cs, n, rho0, k_rec, k_out, 6e6), 1e-10)
                assert abs(f + rr - 1.0) < 1e-10 and rel < 1e-10
                if form == "none" and k_rec[1] == 0.0:
                    z = c.composite_solve(cs, -n, rho0, k_rec, k_out, 6e6)[0]
                    assert abs(f - z) / 2 < 1e-12
    # recombination at both sites gives an odd part without CISS
    rho0 = cs.birth(cs.sites[0].PS)
    a = c.composite_solve(cs, n, rho0, (5e5, 5e5), (2e7, 2e7), 6e6)[0]
    z = c.composite_solve(cs, -n, rho0, (5e5, 5e5), (2e7, 2e7), 6e6)[0]
    assert abs(a - z) > 1e-9


def test_gmres_real():
    rng = np.random.default_rng(9)
    M = np.eye(60) + 0.3 * rng.standard_normal((60, 60)) / np.sqrt(60)
    rhs = rng.standard_normal(60)
    x, info, it = c.gmres_real(lambda v: M @ v, rhs, 1e-13, 15, 50)
    assert info == 0 and it > 15
    close(x, np.linalg.solve(M, rhs), 1e-11)
    hp = c.Hermitian2(5)
    A_ = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    H = A_ + A_.conj().T
    v = hp.pack(H, 2 * H)
    close(hp.unpack(v)[1], 2 * H, 1e-15)
    assert abs(np.linalg.norm(v) - np.sqrt(5) * np.linalg.norm(H)) < 1e-12


def test_crystal_and_maps():
    half = m2.fibonacci_sphere(300)
    G_ = np.vstack([half, -half])
    E = c.cell_axes()
    close(E.T @ E, np.eye(3), 1e-12)
    cs = c.crystal_setup(300)
    a = np.array([0.003, -0.002, 0.001])
    xyz = lambda v: (v @ E[:, 0]) * (v @ E[:, 1]) * (v @ E[:, 2])  # noqa: E731

    def f(v):
        return 0.6 + 0.1 * (v[:, 2] ** 2 - 1 / 3) + v @ a + 0.05 * xyz(v) + 0.02 * v[:, 0] ** 3

    s = c.summarize(half, f(half), f(-half))
    cr = c.crystal_cell(cs, s["coef"])
    assert cr["dipole_norm"] < 1e-15 and cr["molecule_dipole_norm"] > 1e-3
    assert cr["plane_odd_max"] < 1e-12 * cr["odd_max"] and cr["mirror_residual"] < 1e-12 * cr["odd_max"]
    g3 = lambda v: 0.02 * v[:, 0] ** 3  # noqa: E731
    sg = c.summarize(half, g3(half), g3(-half))
    close(cr["a_xyz"], 0.05 / c.Y_XYZ + np.asarray(sg["coef"]) @ cs["coef_xyz"], 1e-12)
    assert cr["xyz_amplitude"] > 0.05 / (3 * np.sqrt(3)) * 0.5
    # zonal ranges, responses and the receptor-axis sweep agree with the frozen functions at the axis z
    zon = np.array(s["zonal"])
    close(c.zonal_range_many(zon[None, :])[0], b.zonal_range(zon), 0.0)
    close(c.coef_zonal(np.array(s["coef"])[None, :])[0], zon, 1e-15)
    close(c.coef_dipole(np.array(s["coef"])[None, :])[0], s["d_odd"], 1e-15)
    U = m2.real_sh(b.L_FIT, np.array([[0.0, 0.0, 1.0]]))[0]
    l, m = c._lm()
    F, Fa = c.receptor_axis_sweep(s["coef"], U, [l == k for k in range(b.L_FIT + 1)])
    close(F[0], s["receptor"]["by_order"]["S1_p1"]["eps_eff"], 1e-15)
    close(Fa[0], s["receptor"]["by_order"]["S1_p1"]["eps_E6"], 1e-15)

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
    saved, saved3, hs = dict(b.RATES), dict(c.S3C_RATES), c.HOP_SCALE[0]
    try:
        assert c.rates("cry4a") == (5.0e6, 3.4e5) and c.rates("mags2") == MAGS2 and c.rates("cr2_1e3") == (1e6, 1e9)
        assert c.rates("blind_cry4a") == BLIND and c.hop_rate(1e8) == 1e8
        c.use_blind_rates()
        assert c.rates("cry4a") == BLIND and c.rates("mags2") == (2.0e6, 2.0e6)
        assert c.rates("cr2_1e2") == (1.2e6, 1.2e8) and c.hop_rate(1e8) == 1.2e8
    finally:
        b.RATES.clear()
        b.RATES.update(saved)
        c.S3C_RATES.clear()
        c.S3C_RATES.update(saved3)
        c.HOP_SCALE[0] = hs
    assert c.rates("cry4a") == (5.0e6, 3.4e5) and c.rates("cr2_1e2") == (1e6, 1e8) and c.hop_rate(1e7) == 1e7
    tmp = tempfile.mkdtemp()
    try:
        for args in (["pair"], ["composite"], ["checks"], ["curves"], ["pair", "--blind-rates", "--confirmatory"]):
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "mag_s3c.py")] + args
                               + ["--out", os.path.join(tmp, "x")], capture_output=True, text=True)
            assert p.returncode != 0 and not os.path.exists(os.path.join(tmp, "x")), args
        for flag, want in (([], [5.0e6, 3.4e5]), (["--blind-rates"], list(BLIND))):
            out = os.path.join(tmp, "b" if flag else "a")
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "mag_s3c.py"), "calibrate", "--out", out,
                            "--checks", "C4"] + flag, check=True, capture_output=True)
            with open(os.path.join(out, "calibration.json")) as fh:
                meta = json.load(fh)["metadata"]
            assert meta["rates"]["cry4a"] == want and meta["hop_scale"] == (1.2 if flag else 1.0)
    finally:
        shutil.rmtree(tmp)


def test_pair_and_receptor_modes():
    """The mechanics of pair (resumable), composite and receptor on a few cells at the MAG-S2 rates, small grids."""
    refs = [c.Ref("W_C", "rf", 1e6, "mags2", "test")]
    direct = [c.Cell("W_C", "r", "rf", 1e6, "K", 0.1, 1.0, "mags2", "test"),
              c.Cell("W_C", "x", "rf", 1e6, "E-M1", 0.1, 1.0, "mags2", "test")]
    rebuilt = [c.Cell("W_C", "r", "rf", 1e6, "K-iso", 0.1, 1.0, "mags2", "test", "ref"),
               c.Cell("W_C", "z", "rf", 1e6, "K", 0.1, 1.0, "mags2", "test", "ref")]
    comp = [c.CompositeCell("none", "C", 1e7, 1e7, "mags2", "test")]
    tmp = tempfile.mkdtemp()
    old_part = c.COMPOSITE_PART
    try:
        assert c.pair(tmp, 1, 0.0, half_n=6, items=direct + refs)
        for x in direct:
            r = json.load(open(c.item_path(tmp, x)))
            assert r["grid_half"] == 6 and r["rates_used"] == list(MAGS2) and len(r["yp"]) == len(r["ym"]) == 6
            assert r["mag_s3c_sha256"] == c.file_sha256(os.path.join(HERE, "mag_s3c.py"))
            close(c.cell_yields(x, m2.fibonacci_sphere(6))[0], r["yp"], 0.0)
        rf = json.load(open(c.item_path(tmp, refs[0])))
        assert np.array(rf["cp"]).shape == (6, 16) and rf["diagnostics"]["max_rel_residual"] < 1e-10
        mtime = os.path.getmtime(c.item_path(tmp, direct[0]))
        assert c.pair(tmp, 1, 0.0, half_n=6, items=direct + refs)
        assert os.path.getmtime(c.item_path(tmp, direct[0])) == mtime
        c.COMPOSITE_PART = 4
        orig = c.composite_cells
        c.composite_cells = lambda: list(comp)
        try:
            assert c.composite(tmp, 1, 0.0, half_n=4)
        finally:
            c.composite_cells = orig
        cr = json.load(open(os.path.join(tmp, "composite", comp[0].name + ".json")))
        assert len(cr["yp"]) == len(cr["ym"]) == 4 and cr["hop_rates_used"] == [1e7, 1e7]
        y = c.composite_yields(comp[0], m2.fibonacci_sphere(4))[0]
        close(y, cr["yp"], 0.0)
        rows = c.receptor(tmp, items=dict(direct=direct, rebuilt=rebuilt, composite=comp, refs=refs))
        assert [r["name"] for r in rows] == [x.name for x in direct + rebuilt + comp] and all(r["code_ok"] for r in rows)
        rec = json.load(open(os.path.join(tmp, "receptor.json")))
        assert rec["grid_half"] == 6 and rec["e13_grid_half"] == [4]
        ri = rec["refs"][refs[0].name]
        assert ri["I2"] < 1e-12 and ri["self_rebuild"] < 1e-13 and "main_axes" in ri
        kz = [r for r in rows if r["name"] == rebuilt[1].name][0]
        close(kz["receptor"]["by_order"]["S1_p1"]["eps_eff"],
              c.summarize(m2.fibonacci_sphere(6), *[c.cell_yields(rebuilt[1], s * m2.fibonacci_sphere(6))[0]
                                                    for s in (1, -1)])["receptor"]["by_order"]["S1_p1"]["eps_eff"],
              1e-11)
        assert set(rec["crystal"]) == {x.name for x in direct + rebuilt + comp}
        e14 = dict(np.load(os.path.join(tmp, "maps_E14.npz")))
        assert {k.split("__")[0] for k in e14 if k != "axes"} == {direct[0].name, rebuilt[1].name}
    finally:
        c.COMPOSITE_PART = old_part
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------------------------------------------
# Analysis on a synthetic world
# --------------------------------------------------------------------------------------------------------------
G_S = {1.0: 1.0, 0.8: 1.13, 0.6: 1.32, 0.4: 1.66}
FACTOR = {"rf": {1e5: 0.98, 3e5: 0.95, 1e6: 0.7, 3e6: 0.6, 1e7: 0.4, 3e7: 0.3},
          "std": {1e5: 0.98, 3e5: 0.95, 1e6: 0.8, 3e6: 0.6, 1e7: 0.4, 3e7: 0.3, 1e8: 0.2}}
DIAG = dict(max_rel_residual=1e-14, max_conservation_error=1e-15, max_gmres_iterations=12)


def factor(relax, k):
    return 1.0 if relax == "none" else FACTOR[relax][k]


def synth_values(cl):
    """A world where every hypothesis is met with a known margin."""
    f = factor(cl.relax, cl.k)
    even = 0.2 / (1.0 + cl.k / 1e6) if cl.form == "K" else 0.2
    share = {"z": 1.0, "x": 0.8, "r": 0.8}[cl.axis]
    if cl.form in ("K", "K-inc") or cl.form.startswith("E-"):
        dz = 0.004 * (cl.chi / 0.05) * share * f * (1.0 if cl.m == 1.0 else 0.2)
        eps = 0.01 * (cl.chi / 0.05) * f * (1.0 if cl.m == 1.0 else 0.2)
        scale = 0.05 if cl.form == "K-inc" else (0.3 if cl.form.startswith("E-") else 1.0)
        dz, eps = scale * dz, scale * eps
    elif cl.form == "K-iso":
        dz, eps = 0.0, 0.0
    else:
        dz, eps = 1e-5 * f, 5e-4 * f
    return dict(eps=eps, dz=dz, F=5.0 * dz, even=even)


def synth_row(cl, eps, dz, F, even, kind, grid_half=None):
    by = {f"S{S:g}_p{p:g}": dict(eps_eff=p * G_S[S] * F, eps_E6=0.9 * p * G_S[S] * F, pair_ratio=0.0, D_pair=1.0,
                                 g_pair=1.0) for S in b.S_LEVELS for p in b.P_LEVELS}
    d = [0.3 * dz, -0.2 * dz, dz]
    row = dict(name=cl.name, cell=asdict(cl), kind=kind, grid_half=grid_half or c.GRID_HALF,
               parts=dict(eps_mol=eps, even_range=even, odd_half_range=eps * even), d_odd=d,
               fit_even_rms_over_sd=1e-4, fit_odd_rms_over_sd=1e-3, zonal=[0.5, dz, 0.07],
               fit_check=dict(L=24, even_rms_over_sd=1e-5, odd_rms_over_sd=1e-4, d_odd=[x * (1 + 1e-4) for x in d]),
               receptor=dict(eps_rec=F / b.R_REN, receptor_even_range=0.07, by_order=by),
               odd_power_shares=dict(molecule=0.1, receptor=0.05),
               I7={f"S{S:g}": dict(p0_max_odd=0.0, p05_half_max_diff=1e-18) for S in b.S_LEVELS},
               code_ok=True, rates_used=list(A.PROTOCOL_RATES[cl.rates]), diagnostics=dict(DIAG))
    if getattr(cl, "system", None) is not None and cl.name in c.E4_CELLS:
        row["E4"] = dict(eps_mol_fixed_p0=0.01, d_x=d[0], eps_x=0.02, eps_y_p1=0.0, eps_z_p1=0.03)
    return row


def synth_world(over=None):
    over = over or {}
    rows = {}
    for kind, cells in (("direct", c.direct_cells()), ("rebuilt", c.rebuilt_cells())):
        for cl in cells:
            v = synth_values(cl)
            v.update(over.get(cl.name, {}))
            rows[cl.name] = synth_row(cl, v["eps"], v["dz"], v["F"], v["even"], kind)
    for cc in c.composite_cells():
        odd = 1e-16 if (cc.form == "none" and cc.rec == "C") else 1e-4
        r = synth_row(cc, odd / 0.2, odd, 5 * odd, 0.2, "composite", c.E13_GRID_HALF)
        r.update(hop_rates_used=[cc.k_cd, cc.k_dc], rec_C=dict(mean=0.05, odd_half_range=odd),
                 rec_D=dict(mean=0.0, odd_half_range=0.0))
        r.update(over.get(cc.name, {}))
        rows[cc.name] = r
    get = lambda *a: rows[c.Cell(*a).name]  # noqa: E731
    refs = {}
    for ref in c.references():
        info = dict(ref=asdict(ref), name=ref.name, grid_half=c.GRID_HALF, diagnostics=dict(DIAG),
                    rates_used=list(A.PROTOCOL_RATES[ref.rates]), code_ok=True, I2=1e-16, self_rebuild=1e-16, I8={})
        if ref.branch == "main" and (ref.relax, ref.k) in c.LEVELS_OLD:
            info["I8"] = {x.name: 1e-14 for x in c.direct_cells(("main",)) if x.form in ("K", "K-inc") and x.m == 1.0
                          and (x.system, x.relax, x.k) == (ref.system, ref.relax, ref.k)}
        info["main_axes"] = {}
        for ax in c.system_axes(ref.system):
            for chi in (0.05, 0.1):
                vk = synth_values(c.Cell(ref.system, ax, ref.relax, ref.k, "K", chi, 1.0, ref.rates))
                vk.update(over.get(c.Cell(ref.system, ax, ref.relax, ref.k, "K", chi, 1.0, ref.rates).name, {}))
                info["main_axes"][f"{ax}_chi{chi:g}"] = dict(d_K=[0.0, 0.0, vk["dz"]], d_inc=[0.0, 0.0, 0.05 * vk["dz"]])
        if ref.system in ("W_C", "W_D"):
            info["E5_named_axes"] = list(c.E5_NAMED)
        if ref.system in ("W_C", "W_D") and ref.branch == "main" and (ref.relax, ref.k) in c.KEY_LEVELS:
            info["H2_circle"] = {}
            for chi in (0.05, 0.1):
                dz = get(ref.system, "z", ref.relax, ref.k, "K", chi, 1.0)["d_odd"][2]
                fac = over.get(f"circle_{ref.system}_{ref.relax}{ref.k:g}_chi{chi:g}", 0.8)
                info["H2_circle"][f"chi{chi:g}"] = dict(circle_dz=[fac * dz] * 360, z_axis_dz=dz,
                                                        median_ratio=fac if dz else None)
        refs[ref.name] = info
    crystal = {n: dict(even_range=0.2, odd_max=1e-4, plane_odd_max=1e-21, mirror_residual=1e-21, a_xyz=1e-5,
                       xyz_amplitude=2e-5, xyz_amplitude_over_even_range=1e-4, dipole_norm=1e-21,
                       molecule_dipole_norm=1e-3)
               for n, r in rows.items() if r["kind"] == "composite" or r["cell"]["system"][:3] in ("W_C", "W_D")}
    rec = dict(cells=list(rows.values()), refs=refs, crystal=crystal, grid_half=c.GRID_HALF,
               e13_grid_half=[c.E13_GRID_HALF], metadata=c.metadata("receptor"))
    e5 = {}
    for ref in c.e5_references():
        n_ax = 10 + (4 if ref.system != "S2" else 0)
        dz = refs[ref.name]["main_axes"][f"z_chi0.1"]["d_K"][2]
        for chi in (0.05, 0.1):
            dk = np.tile([0.0, 0.0, dz * chi / 0.1], (n_ax, 1))
            e5.update({f"{ref.name}__chi{chi:g}_F": np.full(n_ax, 0.02), f"{ref.name}__chi{chi:g}_F_amp": np.full(n_ax, 0.018),
                       f"{ref.name}__chi{chi:g}_eps_mol": np.full(n_ax, 0.01),
                       f"{ref.name}__chi{chi:g}_even_range": np.full(n_ax, 0.2), f"{ref.name}__chi{chi:g}_d_K": dk,
                       f"{ref.name}__chi{chi:g}_d_inc": 0.05 * dk, f"{ref.name}__chi{chi:g}_F_amp_inc": np.full(n_ax, 0.001)})
    e14 = {}
    for r in rows.values():
        if r["kind"] != "composite" and r["cell"]["form"] == "K" and r["cell"]["m"] == 1.0:
            e14[f"{r['name']}__F"], e14[f"{r['name']}__F_amp"] = np.full(10, 0.02), np.full(10, 0.018)
    return rec, e5, e14


def synth_checks():
    cases = c.small_pair_cases()
    return dict(I1=dict(frozen_mismatches=[]), I3={f"{j[1]}_{j[2]}_{j[3]}{j[4]:g}_{j[5]}": 1e-16 for j in c.odd_check_jobs()},
                I5_small=dict(max_dense=1e-14, max_steps=1e-7, max_tiny=1e-13,
                              dense={f"{j[1]}_{j[2]}{j[3]:g}": 1e-14 for j in cases if j[0] == "dense"},
                              steps={f"{j[1]}_{j[2]}{j[3]:g}": 1e-7 for j in cases if j[0] == "steps"},
                              tiny={j[1].name: 1e-13 for j in cases if j[0] == "tiny"}),
                I6={x.name: dict(max_yield_diff=1e-15, eps_mol_rel=1e-12, eps_rec_rel=1e-12,
                                 receptor_even_range_rel=1e-12, d_odd_rel=1e-12) for x in c.I6_CELLS},
                diagnostics=dict(DIAG, max_gmres_iterations=30), grid_half=c.GRID_HALF,
                rates_used={k: list(v) for k, v in A.PROTOCOL_RATES.items()}, hop_scale=1.0,
                metadata=c.metadata("checks"))


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
                      solver_diagnostics=dict(DIAG, max_gmres_iterations=11)),
            delta_up=dict(t=t.tolist(), trace=(1e-3 * np.exp(-(k_f + kap) * t)).tolist(),
                          norm=np.exp(-(k_f + kap) * t).tolist(), forward_integral=2e-4, trace_end=0.0),
            delta_down=dict(t=t.tolist(), trace=(-1e-3 * np.exp(-(k_f + kap) * t)).tolist(),
                            norm=np.exp(-(k_f + kap) * t).tolist(), forward_integral=-2e-4, trace_end=0.0))
    return dict(cases=cases, dt=c.CURVE_DT, t_max=c.CURVE_T, every=c.CURVE_EVERY, rates_used=[5.0e6, 3.4e5],
                metadata=c.metadata("curves"))


def run_analysis(world=None, chk=None, cur=None, pair_done=True, composite_done=True):
    rec, e5, e14 = world or synth_world()
    return A.analyze(rec, chk or synth_checks(), cur or synth_curves(), e5, e14,
                     dict(metadata=c.metadata("pair")) if pair_done else None,
                     dict(metadata=c.metadata("composite")) if composite_done else None)


def hyp_of(over):
    rec, e5, _ = synth_world(over)
    return A.hypotheses(A.Grid(rec), e5)


def name(*args, **kw):
    return c.Cell(*args, **kw).name


def test_analysis_world_and_counts():
    res = run_analysis()
    assert res["valid"], {k: v["verdict"] for k, v in res["invariants"].items()}
    assert res["blas_single_thread"] and res["pair_done"] and res["composite_done"]
    cnt = res["comparisons"]["counted"]
    assert cnt["per_hypothesis"] == A.N_COMPARISONS and cnt["total"] == 184 and cnt["known_in_advance"] == A.N_KNOWN
    h = res["hypotheses"]
    for key, geoms in (("H1", c.GEOMETRIES), ("H2", ("W_C", "W_D")), ("H3", ("W_C", "W_D")), ("H4", c.GEOMETRIES),
                       ("H5", c.GEOMETRIES)):
        assert [h[key][g]["verdict"] for g in geoms] == ["met"] * len(geoms), key
    close(h["H1"]["W_C"]["smallest_margin"], 2.0, 1e-12)            # eps_K / eps_P = 20, threshold 10
    assert h["H1"]["S2"]["n"] == 16 and len(h["H1"]["S2"]["known_in_advance"]) == 4
    close(h["H2"]["W_D"]["smallest_margin"], 1.6, 1e-12)            # 0.8 / 0.5
    close(h["H3"]["W_C"]["smallest_margin"], 0.9 * 5 * 0.004 * 2 * 0.8 * 0.7 / 0.01, 1e-12)
    close(h["H4"]["S2"]["smallest_margin"], 1.4, 1e-12)             # 0.7 / 0.5 (rf)
    close(h["H5"]["W_C"]["smallest_margin"], 0.95 / 0.9, 1e-12)
    assert len(h["H5"]["S2"]["known_in_advance"]) == 4
    kh = h["H4"]["k_half"]
    assert len(kh) == 32
    close(kh["W_C_r_chi0.1_rf"]["k_half"], math.sqrt(3e6 * 1e7), 1e-6)
    close(kh["S2_z_chi0.05_rf"]["k_half"], 10 ** (6 + 2 / 3), 1e-6)
    assert kh["W_D_x_chi0.1_std"]["class"] == "below 3e7"
    e = res["expectations"]
    assert [e[k]["verdict"] for k in ("Q1", "Q2", "Q3", "Q4")] == ["met"] * 4 and e["Q3"]["n"] > 0
    m = res["maps"]
    assert set(m) == {f"E{i}" for i in range(1, 15)}
    assert abs(m["E9"]["W_C_rf1e+06"]["effective_coherence_time_s"] - 0.5e-6) <= 2e-9
    e10 = m["E10"]["W_C_r_rf"]
    close([x["s_C"] for x in e10["levels"]], [0.95] * 7, 1e-12)
    assert e10["stricter"]["k_star"] is not None
    assert m["E13"][c.CompositeCell("none", "CD", 1e8, 1e8).name]["F_over_F_of_K"] == 1.0
    assert set(m["E14"]["crystal"]) == set(synth_world()[0]["crystal"])
    tmp = tempfile.mkdtemp()
    try:
        A.write_table(os.path.join(tmp, "tables", "cells.csv"), m["E1"])
        with open(os.path.join(tmp, "tables", "cells.csv")) as fh:
            assert len(fh.read().splitlines()) == 574 + 442 + 1
    finally:
        shutil.rmtree(tmp)


def test_analysis_edges():
    # H1: the ratio exactly at 10 passes, just below fails (binary fractions make the ratios exact)
    kc, pc = name("W_C", "r", "rf", 1e6, "K", 0.05, 1.0), name("W_C", "r", "rf", 1e6, "P-LH", 0.1, 1.0)
    assert hyp_of({kc: dict(eps=0.625), pc: dict(eps=0.0625)})["H1"]["W_C"]["verdict"] == "met"
    h = hyp_of({kc: dict(eps=0.625), pc: dict(eps=0.0625 * 1.0001)})["H1"]
    assert h["W_C"]["verdict"] == "not_met" and h["W_C"]["n_failed"] == 1 and h["W_D"]["verdict"] == "met"
    # known comparisons stay out of the verdict
    sk, sp = name("S2", "z", "none", 0.0, "K", 0.05, 1.0), name("S2", "z", "none", 0.0, "P-same", 0.1, 1.0)
    assert hyp_of({sk: dict(eps=0.001), sp: dict(eps=0.001)})["H1"]["S2"]["verdict"] == "met"
    # H2: the median at half passes, below fails, a vanishing denominator is excluded
    assert hyp_of({"circle_W_D_std1e+06_chi0.1": 0.5})["H2"]["W_D"]["verdict"] == "met"
    h = hyp_of({"circle_W_D_std1e+06_chi0.1": 0.4999})["H2"]
    assert h["W_D"]["verdict"] == "not_met" and h["W_C"]["verdict"] == "met"
    z = name("W_C", "z", "none", 0.0, "K", 0.1, 1.0)
    h = hyp_of({z: dict(dz=1e-12)})["H2"]["W_C"]
    assert h["verdict"] == "met" and len(h["excluded"]) == 1 and h["n"] == 5
    zs = {name(g, "z", rl, k, "K", chi, 1.0): dict(dz=0.0) for g in ("W_C",) for chi in (0.05, 0.1)
          for rl, k in c.KEY_LEVELS}
    assert hyp_of(zs)["H2"]["W_C"]["verdict"] == "untestable"
    # H3: the stricter transfer at the threshold passes, below fails, separately for W_C and W_D
    rr = name("W_D", "r", "rf", 1e6, "K", 0.1, 1.0)
    assert hyp_of({rr: dict(F=-0.0125)})["H3"]["W_D"]["verdict"] == "met"
    h = hyp_of({rr: dict(F=0.0111)})["H3"]                          # F_amp = 0.9 F below 0.01
    assert h["W_D"]["verdict"] == "not_met" and h["W_C"]["verdict"] == "met"
    # H4: the ratio at 0.5 passes, a change of sign fails
    s0, s1 = name("S2", "x", "none", 0.0, "K", 0.05, 1.0), name("S2", "x", "std", 1e6, "K", 0.05, 1.0)
    s2 = name("S2", "x", "rf", 1e6, "K", 0.05, 1.0)
    assert hyp_of({s0: dict(dz=0.0625), s1: dict(dz=0.03125), s2: dict(dz=0.0625)})["H4"]["S2"]["verdict"] == "met"
    h = hyp_of({s0: dict(dz=0.0625), s1: dict(dz=-0.0625), s2: dict(dz=0.0625)})["H4"]
    assert h["S2"]["verdict"] == "not_met" and h["S2"]["n_failed"] == 1
    # H5: s_C at 0.9 passes, below fails, above 1 passes
    kk, ki = name("W_C", "x", "none", 0.0, "K", 0.1, 1.0), name("W_C", "x", "none", 0.0, "K-inc", 0.1, 1.0)
    assert hyp_of({kk: dict(dz=1.0), ki: dict(dz=0.1)})["H5"]["W_C"]["verdict"] == "met"
    assert hyp_of({kk: dict(dz=1.0), ki: dict(dz=-0.5)})["H5"]["W_C"]["verdict"] == "met"
    h = hyp_of({kk: dict(dz=1.0), ki: dict(dz=0.1000001)})["H5"]["W_C"]
    assert h["verdict"] == "not_met" and h["n_failed"] == 1
    # k_half: crossing on the first level, none at all, interpolation in lg k
    assert "below" in A.k_half_of([(0.0, 1.0), (1e5, 0.4)], 1e5)
    assert A.k_half_of([(0.0, 1.0), (1e5, 0.9), (1e6, 0.6)], 1e6)["above"] == 1e6
    close(A.k_half_of([(0.0, 1.0), (1e6, 0.75), (1e7, 0.25)], 1e7)["k_half"], math.sqrt(1e13), 1e-3)
    assert A.k_half_class(dict(k_half=4e7)) == "3e7 to 6e7" and A.k_half_class(dict(k_half=None, above=1e8)) \
        == "above 6e7"
    out = A.first_below([0.0, 1e6, 1e7], [0.04, 0.02, 0.005])
    close(out["k_star"], 10 ** (6 + (0.02 - 0.01) / (0.02 - 0.005)), 1e-3)
    # expectations
    q = name("W_C", "r", "rf", 1e7, "K", 0.1, 1.0)
    out = A.q3(A.Grid(synth_world({q: dict(even=1.0)})[0]))
    assert out["verdict"] == "not_met" and len(out["failed"]) == 1
    q2c = name("W_D", "z", "none", 0.0, "P-same", 0.1, 1.0)
    out = A.q2(A.Grid(synth_world({q2c: dict(F=0.0)})[0]))
    assert out["verdict"] == "not_met" and len(out["failed"]) == 2
    ex = A.q1(A.Grid(synth_world({q2c: dict(dz=0.0)})[0]))["excluded_from_dipole"]
    assert q2c in ex and len(ex) == 1 + sum(x.form == "K-iso" for x in c.rebuilt_cells())
    # invariants
    rec, e5, e14 = synth_world()
    rec["cells"][5]["code_ok"] = False
    next(iter(rec["refs"].values()))["I2"] = 2e-12
    rec["crystal"][next(iter(rec["crystal"]))]["plane_odd_max"] = 1e-15
    chk = synth_checks()
    chk["I3"].pop(next(iter(chk["I3"])))
    chk["I6"][c.I6_CELLS[0].name]["max_yield_diff"] = 2e-12
    res = run_analysis((rec, e5, e14), chk)
    inv = res["invariants"]
    assert all(inv[k]["verdict"] == "error" for k in ("I1", "I2", "I3", "I6", "I11")) and not res["valid"]
    rec, e5, e14 = synth_world({c.CompositeCell("none", "C", 1e9, 1e9).name: dict(
        parts=dict(eps_mol=1e-11, even_range=0.2, odd_half_range=2e-12))})
    rec["refs"][c.Ref("S2", "none", 0.0).name]["I8"][name("S2", "z", "none", 0.0, "K", 0.1, 1.0)] = 2e-10
    next(r for r in rec["cells"] if r["cell"].get("form") == "K-iso")["parts"]["odd_half_range"] = 2e-12
    inv = run_analysis((rec, e5, e14))["invariants"]
    assert (inv["I8"]["verdict"], inv["I9"]["verdict"], inv["I10"]["verdict"]) == ("error",) * 3
    chk = synth_checks()
    chk["I5_small"]["tiny"].pop(next(iter(chk["I5_small"]["tiny"])))
    assert run_analysis(chk=chk)["invariants"]["I5"]["verdict"] == "error"
    # validity: grids, rates, curves, BLAS threads, markers, missing cells
    rec, e5, e14 = synth_world()
    rec["e13_grid_half"] = [50]
    assert not run_analysis((rec, e5, e14))["valid"]
    rec, e5, e14 = synth_world()
    rec["cells"][0]["rates_used"] = list(BLIND)
    assert not run_analysis((rec, e5, e14))["rates_ok"]
    chk = synth_checks()
    chk["hop_scale"] = 1.2
    assert not run_analysis(chk=chk)["rates_ok"]
    for change in ("t_max", "case"):
        cur = synth_curves()
        if change == "t_max":
            cur["t_max"] = 5e-8
        else:
            cur["cases"].pop(next(iter(cur["cases"])))
        res = run_analysis(cur=cur)
        assert not res["valid"] and res.get("incomplete", False) == (change == "case"), change
    chk = synth_checks()
    chk["metadata"] = dict(chk["metadata"], blas=[dict(chk["metadata"]["blas"][0], num_threads=2)])
    assert not run_analysis(chk=chk)["valid"]
    assert not run_analysis(pair_done=False)["valid"] and not run_analysis(composite_done=False)["valid"]
    rec, e5, e14 = synth_world()
    rec["cells"] = rec["cells"][:-1]
    res = run_analysis((rec, e5, e14))
    assert res["incomplete"] and len(res["missing"]["composite"]) == 1


def test_analysis_cli():
    tmp = tempfile.mkdtemp()
    try:
        run, out = os.path.join(tmp, "run"), os.path.join(tmp, "results")
        os.makedirs(run)
        rec, e5, e14 = synth_world()
        for nm, obj in (("receptor", rec), ("checks", synth_checks()), ("curves", synth_curves()),
                        ("pair_done", dict(metadata=c.metadata("pair"))),
                        ("composite_done", dict(metadata=c.metadata("composite")))):
            with open(os.path.join(run, f"{nm}.json"), "w") as fh:
                json.dump(obj, fh)
        np.savez(os.path.join(run, "maps_E5.npz"), **e5)
        np.savez(os.path.join(run, "maps_E14.npz"), **e14)
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "analyze_s3c.py"), "--run", run, "--out", out],
                           check=True, capture_output=True, text=True)
        assert p.stdout.startswith("valid")
        res = json.load(open(os.path.join(out, "results.json")))
        assert res["valid"] and set(res["maps"]) == {f"E{i}" for i in range(1, 15)}
        assert os.path.exists(os.path.join(out, "tables", "cells.csv"))
        rec["cells"] = rec["cells"][:-3]
        with open(os.path.join(run, "receptor.json"), "w") as fh:
            json.dump(rec, fh)
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
         test_relaxation_algebra, test_relaxation_matches_radicalpy, test_solver_against_dense, test_gmres_failure_raises,
         test_frozen_regression_s2, test_exact_zeros_full_pairs, test_kinc_and_forms, test_reconstruction,
         test_form_E_and_denton, test_composite_pair, test_gmres_real, test_crystal_and_maps, test_curves,
         test_receptor_transfer, test_blind_rates, test_pair_and_receptor_modes, test_analysis_world_and_counts,
         test_analysis_edges, test_analysis_cli, test_pipeline_gates, test_notebook_build_reproducible,
         test_solver_against_steps]
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
