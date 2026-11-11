"""Independent reproduction of the two-site parity results (Fable 5.2, MAG_two_site_checks), written from scratch.

Differences from her code on purpose: column-major vectorisation, full (non-axial) random hyperfine tensors, other
rates, other field directions, and a second solver for the case without back hop (two Sylvester equations solved in
sequence, no Liouville matrix at all). Invented toy system: electrons A and B, spin-1/2 nuclei a (on A at both sites),
b1 (on B at site 1 only) and b2 (on B at site 2 only). No W_C/W_D geometry, no RadicalPy tensors, no Cry4a rates.

Site 1 is where the pair is born (the bridge), site 2 is the terminal pair, the signalling product leaves site 2:
  d rho1/dt = L1 rho1 - k12 rho1 + k21 U^+ rho2 U
  d rho2/dt = L2 rho2 - (k21 + kF) rho2 + k12 U rho1 U^+
  Li rho = -i[Hi, rho] - kRi/2 {Qi, rho} + kstd (PS rho PS + PT rho PT - rho)
  rho1(0) = Qb / M,  PhiF = kF int Tr rho2 dt.
"""
import numpy as np
from functools import reduce
from scipy.linalg import solve_sylvester

GAM = 1.76085963e8            # rad s^-1 mT^-1
rng = np.random.default_rng(2026)
s = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.array([[1, 0], [0, -1]]) / 2]
e = np.eye(2)
NS = 5
DIM = 2 ** NS
M = DIM // 4


def spin(k):
    return [reduce(np.kron, [s[i] if j == k else e for j in range(NS)]).astype(complex) for i in range(3)]


SA, SB, Ia, Ib1, Ib2 = (spin(k) for k in range(NS))
ONE = np.eye(DIM, dtype=complex)
PS = ONE / 4 - sum(SA[i] @ SB[i] for i in range(3))
PT = ONE - PS


def coupling(T, X, Y):
    return sum(T[i, j] * X[i] @ Y[j] for i in range(3) for j in range(3))


def rand_tensor(scale):
    a = rng.normal(size=(3, 3))
    return scale * (a + a.T) / 2 + scale * 0.3 * np.eye(3)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def dipolar(d, axis):
    r = unit(axis)
    return d * (np.eye(3) - 3 * np.outer(r, r))


def rot(chi, n):
    """CISS rotation of radical A: cos(chi) - i sin(chi) n.sigma_A."""
    return np.cos(chi) * ONE - 2j * np.sin(chi) * sum(n[i] * SA[i] for i in range(3))


T_a, T_b1, T_b2 = rand_tensor(0.9), rand_tensor(0.6), rand_tensor(0.7)          # mT
D1, D2 = dipolar(-0.45, rng.normal(size=3)), dipolar(-0.2, rng.normal(size=3))  # mT
B0 = 0.05                                                                        # mT


def ham_site1(n, kind="full", J=0.0, de=0.0, Pth=None):
    zee = B0 * sum(n[i] * (SA[i] + SB[i]) for i in range(3))
    if kind == "full":
        H = zee + coupling(T_a, SA, Ia) + coupling(T_b1, SB, Ib1) + coupling(D1, SA, SB)
    elif kind == "zeeman_exchange":
        H = zee.copy()
    elif kind == "denton":                       # bridge of Denton et al.: de P_theta + 2 J P_S, nothing else
        H = de * Pth + 2 * J * PS
        return GAM * H
    H = H - 2 * J * sum(SA[i] @ SB[i] for i in range(3))
    return GAM * H


def ham_site2(n):
    zee = B0 * sum(n[i] * (SA[i] + SB[i]) for i in range(3))
    return GAM * (zee + coupling(T_a, SA, Ia) + coupling(T_b2, SB, Ib2) + coupling(D2, SA, SB))


I_D = np.eye(DIM)
I_L = np.eye(DIM * DIM)


def lmul(A):           # X -> A X   (column-major vec)
    return np.kron(I_D, A)


def rmul(A):           # X -> X A
    return np.kron(A.T, I_D)


def site_liouvillian(H, kR, Q, kstd):
    L = -1j * (lmul(H) - rmul(H)) - kR / 2 * (lmul(Q) + rmul(Q))
    if kstd:
        L = L + kstd * (lmul(PS) @ rmul(PS) + lmul(PT) @ rmul(PT) - I_L)
    return L


def vec(X):
    return X.reshape(-1, order="F")


def unvec(x):
    return x.reshape(DIM, DIM, order="F")


def yields(n, c):
    H1 = ham_site1(n, c["bridge"], c.get("J", 0.0), c.get("de", 0.0), c.get("Pth"))
    H2 = ham_site2(n)
    U = c["U"]
    hop12 = lmul(U) @ rmul(U.conj().T)               # rho -> U rho U^+
    hop21 = lmul(U.conj().T) @ rmul(U)
    L1 = site_liouvillian(H1, c["kR1"], c["Q1"], c.get("kstd", 0.0)) - c["k12"] * I_L
    L2 = site_liouvillian(H2, c["kR2"], c["Q2"], c.get("kstd", 0.0)) - (c["k21"] + c["kF"]) * I_L
    L = np.block([[L1, c["k21"] * hop21], [c["k12"] * hop12, L2]])
    rhs = np.concatenate([vec(c["Qb"] / M), np.zeros(DIM * DIM)])
    x = np.linalg.solve(L, -rhs)
    X1, X2 = unvec(x[:DIM * DIM]), unvec(x[DIM * DIM:])
    phiF = c["kF"] * np.trace(X2).real
    phiR1 = c["kR1"] * np.trace(c["Q1"] @ X1).real
    phiR2 = c["kR2"] * np.trace(c["Q2"] @ X2).real
    return np.array([phiF, phiR1, phiR2])


def yields_sequential(n, c):
    """No back hop: two Sylvester equations in sequence, A X + X A^+ = -source."""
    assert c["k21"] == 0.0 and not c.get("kstd")
    H1 = ham_site1(n, c["bridge"], c.get("J", 0.0), c.get("de", 0.0), c.get("Pth"))
    H2 = ham_site2(n)
    A1 = -1j * H1 - c["kR1"] / 2 * c["Q1"] - c["k12"] / 2 * ONE
    X1 = solve_sylvester(A1, A1.conj().T, -c["Qb"] / M)
    A2 = -1j * H2 - c["kR2"] / 2 * c["Q2"] - c["kF"] / 2 * ONE
    X2 = solve_sylvester(A2, A2.conj().T, -c["k12"] * c["U"] @ X1 @ c["U"].conj().T)
    return np.array([c["kF"] * np.trace(X2).real, c["kR1"] * np.trace(c["Q1"] @ X1).real,
                     c["kR2"] * np.trace(c["Q2"] @ X2).real])


DIRS = np.array([unit(v) for v in rng.normal(size=(12, 3))])


def report(label, c, solver=yields):
    yp = np.array([solver(n, c) for n in DIRS])
    ym = np.array([solver(-n, c) for n in DIRS])
    odd = np.max(np.abs(yp - ym) / 2, axis=0)
    span = yp[:, 0].max() - yp[:, 0].min()
    cons = np.max(np.abs(yp.sum(axis=1) - 1))
    print(f"{label:58s} odd F {odd[0]:.1e}  R1 {odd[1]:.1e}  R2 {odd[2]:.1e} | odd/range of F {odd[0] / span:7.2%} "
          f"| |sum-1| {cons:.0e}", flush=True)
    return yp, ym


chi_b, n_b = 0.08 * np.pi, unit([0.3, -0.5, 0.8])
chi_h, n_h = 0.06 * np.pi, unit([-0.7, 0.2, 0.4])
Ub, Uh = rot(chi_b, n_b), rot(chi_h, n_h)
Qb_ciss = Ub @ PS @ Ub.conj().T
base = dict(k12=2e7, k21=3e6, kR1=1e7, kR2=3e6, kF=2e6, bridge="full", U=ONE, Qb=PS, Q1=PS, Q2=PS)
print("toy rates:", {k: base[k] for k in ("k12", "k21", "kR1", "kR2", "kF")})

print("-- no CISS")
report("E1 recombination only at the birth site (kR2 = 0)", dict(base, kR2=0.0))
report("E2 singlet recombination at both sites", base)
report("E3 as E2, bridge = Zeeman only (no hyperfine, no dipole)", dict(base, bridge="zeeman_exchange"))
report("E3b as E3 plus exchange 0.3 mT in the bridge", dict(base, bridge="zeeman_exchange", J=0.3))
report("E2b as E2, recombination only at the terminal site (kR1 = 0)", dict(base, kR1=0.0))

print("-- CISS at birth and at the hop")
ciss = dict(base, Qb=Qb_ciss, Q1=Qb_ciss, U=Uh)
report("E4 recombination only at birth site through birth projector", dict(ciss, kR2=0.0))
report("E4r same with S-T dephasing 3e7 at both sites", dict(ciss, kR2=0.0, kstd=3e7))
report("E4n same with no back hop", dict(ciss, kR2=0.0, k21=0.0))
report("E4m birth projector rotated, recombination through P_S", dict(ciss, kR2=0.0, Q1=PS))

print("-- structure of Denton et al.: P_theta at birth and in both channels, bridge = de P_theta + 2 J P_S")
Pth = rot(0.12 * np.pi, unit([0.1, 0.9, -0.4])) @ PS @ rot(0.12 * np.pi, unit([0.1, 0.9, -0.4])).conj().T
den = dict(base, bridge="denton", Qb=Pth, Q1=Pth, Q2=Pth, Pth=Pth, k12=2e8)
report("E5a J = 0, de = 0, kR2 = 3e6", dict(den, J=0.0, de=0.0))
report("E5b J = 0, de = 4 mT, kR2 = 3e6", dict(den, J=0.0, de=4.0))
report("E5c J = 15 mT, de = 4 mT, kR2 = 0", dict(den, J=15.0, de=4.0, kR2=0.0))
report("E5d J = 15 mT, de = 4 mT, kR2 = 3e6", dict(den, J=15.0, de=4.0))
report("E5e J = 15 mT, de = 0, kR2 = 3e6", dict(den, J=15.0, de=0.0))
report("E5f theta = 0 (P_S everywhere), J = 15 mT, kR2 = 3e6", dict(den, J=15.0, de=0.0, Qb=PS, Q1=PS, Q2=PS, Pth=PS))

print("-- no CISS: odd part against the bridge lifetime (forward hop k12)")
for k12 in (2e7, 2e8, 2e9, 2e10):
    report(f"E6 k12 = {k12:.0e} (bridge lifetime {1e9 / (k12 + base['kR1']):.3g} ns)", dict(base, k12=k12))

print("-- second solver, no back hop: block Liouville vs two Sylvester equations in sequence")
seq = dict(base, k21=0.0)
yp1, ym1 = report("E7 Liouville, no CISS, both channels, k21 = 0", seq)
yp2, ym2 = report("E7 Sylvester sequence, same case", seq, solver=yields_sequential)
print(f"E7 largest difference between the two solvers: {max(np.abs(yp1 - yp2).max(), np.abs(ym1 - ym2).max()):.1e}")
