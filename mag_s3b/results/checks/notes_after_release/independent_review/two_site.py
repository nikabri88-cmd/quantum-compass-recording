"""Two-site toy (bridge state '*' = FAD/W_B-like pair, terminal state 'CS' = FAD/W_C-like pair).
Invented system: 2 electrons + 3 spin-1/2 nuclei (n1 on radical A at both sites, n2 on radical B at the bridge site
only, n3 on radical B at the terminal site only), invented axial tensors, invented rates (not Cry4a).

d rho_*/dt  = -i[H_*, rho_*]  - kCR/2 {P_CR, rho_*}   - k2 rho_*  + km2 U2^+ rho_CS U2  (+ relaxation)
d rho_CS/dt = -i[H_CS,rho_CS] - kCR2/2{P_CR2,rho_CS} - (km2+kSP) rho_CS + k2 U2 rho_* U2^+ (+ relaxation)
rho_*(0) = P_b / Z,  Phi_SP = kSP int Tr rho_CS dt.
"""
import numpy as np
from functools import reduce

GAM = 1.76085963e8
sx = np.array([[0, 1], [1, 0]], complex) / 2
sy = np.array([[0, -1j], [1j, 0]], complex) / 2
sz = np.array([[1, 0], [0, -1]], complex) / 2
PA = [sx, sy, sz]
I2 = np.eye(2, dtype=complex)
N = 5
DIM = 2 ** N


def op(k, i):
    lst = [I2] * N
    lst[k] = PA[i]
    return reduce(np.kron, lst)


S = [[op(k, i) for i in range(3)] for k in range(N)]
SA, SB, I1, I2n, I3 = S
ONE = np.eye(DIM, dtype=complex)
PS = ONE / 4 - sum(SA[i] @ SB[i] for i in range(3))
PT = ONE - PS
Z = DIM / 4


def dot(n, ops):
    return sum(n[i] * ops[i] for i in range(3))


def axial(apar, aperp, axis):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return aperp * np.eye(3) + (apar - aperp) * np.outer(axis, axis)


def sph(t, p):
    t, p = np.radians(t), np.radians(p)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def bil(T, X, Y):
    return sum(T[i, j] * X[i] @ Y[j] for i in range(3) for j in range(3) if T[i, j] != 0)


def dip(d, r):
    r = np.asarray(r, float); r = r / np.linalg.norm(r)
    return d * (np.eye(3) - 3 * np.outer(r, r))


def UK(chi, n):
    return np.cos(chi) * ONE - 1j * np.sin(chi) * 2 * dot(n, SA)


def projK(chi, n):
    U = UK(chi, n)
    return U @ PS @ U.conj().T


A1 = axial(1.5, 0.2, sph(0, 0))
A2 = axial(0.8, 0.1, sph(65, 130))
A3 = axial(0.9, 0.1, sph(35, 10))


def hams(b, B0=0.05, Jb=0.3, db=0.6, dcs=0.25, aniso_bridge=True):
    Zee = GAM * B0 * (dot(b, SA) + dot(b, SB))
    Hb = Zee + GAM * bil(A1, SA, I1) - GAM * 2 * Jb * sum(SA[i] @ SB[i] for i in range(3))
    if aniso_bridge:
        Hb = Hb + GAM * bil(A2, SB, I2n) + GAM * bil(dip(db, sph(80, 250)), SA, SB)
    else:
        Hb = Zee - GAM * 2 * Jb * sum(SA[i] @ SB[i] for i in range(3))   # bridge with exchange only
    Hcs = Zee + GAM * bil(A1, SA, I1) + GAM * bil(A3, SB, I3) + GAM * bil(dip(dcs, sph(70, 40)), SA, SB)
    return Hb, Hcs


E = np.eye(DIM, dtype=complex)
D2 = DIM * DIM


def sup_comm(H):
    return -1j * (np.kron(H, E) - np.kron(E, H.T))


def sup_anti(P):
    return np.kron(P, E) + np.kron(E, P.T)


def sup_sand(Aop, Bop):      # rho -> A rho B
    return np.kron(Aop, Bop.T)


R_STD = sup_sand(PS, PS) + sup_sand(PT, PT) - np.eye(D2)


def yields(b, Pb, PCR, PCR2, U2, k2, km2, kCR, kCR2, kSP, kstd=0.0, **hk):
    Hb, Hcs = hams(b, **hk)
    L = np.zeros((2 * D2, 2 * D2), complex)
    L[:D2, :D2] = sup_comm(Hb) - 0.5 * kCR * sup_anti(PCR) - k2 * np.eye(D2) + kstd * R_STD
    L[D2:, D2:] = sup_comm(Hcs) - 0.5 * kCR2 * sup_anti(PCR2) - (km2 + kSP) * np.eye(D2) + kstd * R_STD
    L[D2:, :D2] = k2 * sup_sand(U2, U2.conj().T)
    L[:D2, D2:] = km2 * sup_sand(U2.conj().T, U2)
    rhs = np.concatenate([(Pb / Z).reshape(-1), np.zeros(D2, complex)])
    X = np.linalg.solve(L, -rhs)
    Xb, Xcs = X[:D2].reshape(DIM, DIM), X[D2:].reshape(DIM, DIM)
    phi_sp = kSP * np.trace(Xcs).real
    phi_cr = kCR * np.trace(PCR @ Xb).real
    phi_cr2 = kCR2 * np.trace(PCR2 @ Xcs).real
    return phi_sp, phi_cr, phi_cr2


def odd_test(label, ndir=10, **kw):
    rng = np.random.default_rng(7)
    mx = np.zeros(3); span = []; cons = 0.0
    for _ in range(ndir):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        yp = np.array(yields(v, **kw)); ym = np.array(yields(-v, **kw))
        mx = np.maximum(mx, np.abs(yp - ym) / 2)
        span.append(yp[0]); cons = max(cons, abs(yp.sum() - 1))
    span = np.array(span)
    print(f"{label:62s} max|odd| SP {mx[0]:.2e}  CR {mx[1]:.2e}  CR2 {mx[2]:.2e} | range of Phi_SP over dirs {span.max() - span.min():.2e} | |sum-1| {cons:.1e}", flush=True)
