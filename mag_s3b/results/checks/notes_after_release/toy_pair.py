"""Shared toy radical pair for the notes after release v0.2.1 of MAG-S3b.

Two electrons (A, B) and two spin-1 nuclei with random anisotropic hyperfine tensors, a point dipole between the
electrons, a 50 uT field, Haberkorn recombination through a projector Q at rate kR and a spin-independent product at
rate kF (MAG-S2 rates, 1e6 s^-1 each). Everything here is invented: no W_C or W_D geometry, no RadicalPy tensors, no
Cry4a rates. The logs of the notes print residuals and pass/fail only.
"""
import numpy as np

rng = np.random.default_rng(31)
r2 = np.sqrt(2)
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
Ix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / r2
Iy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / r2
Iz = np.diag([1.0, 0.0, -1.0])
e2, e3 = np.eye(2), np.eye(3)


def kron(*a):
    out = np.array([[1.0 + 0j]])
    for x in a:
        out = np.kron(out, x)
    return out


SA = [kron(s, e2, e3, e3) for s in (sx, sy, sz)]
SB = [kron(e2, s, e3, e3) for s in (sx, sy, sz)]
I1 = [kron(e2, e2, s, e3) for s in (Ix, Iy, Iz)]
I2 = [kron(e2, e2, e3, s) for s in (Ix, Iy, Iz)]
M = 9
DIM = 4 * M
ONE = np.eye(DIM, dtype=complex)
PS = ONE / 4 - sum(SA[i] @ SB[i] for i in range(3))
PT = ONE - PS


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _tensor(scale):
    a = rng.normal(size=(3, 3))
    return scale * (a + a.T) / 2


A1, A2 = _tensor(6e7), _tensor(3e7)                       # rad/s
_dip = unit(rng.normal(size=3))
_d = 2 * np.pi * 9.5e6                                   # rad/s, about 0.34 mT
H0 = sum(A1[i, j] * SA[i] @ I1[j] for i in range(3) for j in range(3))
H0 = H0 + sum(A2[i, j] * SB[i] @ I2[j] for i in range(3) for j in range(3))
H0 = H0 - _d * (3 * sum(_dip[i] * SA[i] for i in range(3)) @ sum(_dip[i] * SB[i] for i in range(3))
                - sum(SA[i] @ SB[i] for i in range(3)))
OMEGA = 1.760859e11 * 50e-6                              # rad/s at 50 uT
KF = KR = 1e6
_I = np.eye(DIM)


def sigma_A(n):
    return 2 * sum(n[i] * SA[i] for i in range(3))


def form_K(chi, n):
    """Birth state of form K at m = 1: cos(chi) - i sin(chi) n.sigma_A applied to the singlet."""
    U = np.cos(chi) * ONE - 1j * np.sin(chi) * sigma_A(n)
    return U @ PS @ U.conj().T


def form_P(chi, n):
    """Birth state |S> + sin(chi/2)|T0 along n>, normalised (polarisation form)."""
    G = ONE + np.sin(chi / 2) * sigma_A(n)
    rho = G @ PS @ G.conj().T
    return rho / np.trace(rho).real * M


def solve_integral(n, rho0, Q=PS):
    """Time integral of rho for a pair born in rho0/M at field direction n, recombining through Q."""
    H = H0 + OMEGA * sum(n[i] * (SA[i] + SB[i]) for i in range(3))
    L = (-1j * (np.kron(_I, H) - np.kron(H.T, _I)) - KR / 2 * (np.kron(_I, Q) + np.kron(Q.T, _I))
         - KF * np.eye(DIM * DIM))
    x = np.linalg.solve(L, -(rho0 / M).reshape(-1, order="F"))
    return x.reshape(DIM, DIM, order="F")


def yield_R(n, rho0, Q=PS):
    return KR * np.trace(Q @ solve_integral(n, rho0, Q)).real


def yield_F(n, rho0, Q=PS):
    return KF * np.trace(solve_integral(n, rho0, Q)).real


def _rot_y_pi(dim_spin):
    if dim_spin == 2:
        return -2j * sy                                    # exp(-i pi S_y) for spin 1/2
    w, v = np.linalg.eigh(Iy)
    return v @ np.diag(np.exp(-1j * np.pi * w)) @ v.conj().T


U_THETA = kron(_rot_y_pi(2), _rot_y_pi(2), _rot_y_pi(3), _rot_y_pi(3))


def time_reversed(rho):
    """Theta rho Theta^-1 with Theta = exp(-i pi S_y) K on every spin."""
    return U_THETA @ rho.conj() @ U_THETA.conj().T


def random_directions(k, seed=5):
    g = np.random.default_rng(seed)
    return np.array([unit(v) for v in g.normal(size=(k, 3))])
