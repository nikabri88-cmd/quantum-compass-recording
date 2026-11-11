"""Exact check: if a radical pair is born in a time-reversal-even state Q and recombines through the same Q,
its yield is even under field inversion. Applied to the microscopic CISS state of Denton et al. (2026),
|psi> = cos(t)|S> + i sin(t)|T0>, born and recombining through |psi><psi|.
Two contrasts are reported only as pass/fail (odd part nonzero or not):
  (b) born in |psi><psi|, recombining through P_S (non-reciprocal, like form K at m = 1);
  (c) born with real S-T0 coherence (polarization), recombining through |psi><psi|.
Arbitrary anisotropic hyperfine tensors for two spin-1 nuclei, a point dipole, MAG-S2 rates k_F = k_R = 1e6 s^-1.
Nothing here uses the W_C or W_D geometry or the Cry4a rates."""
import numpy as np

rng = np.random.default_rng(7)
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
r2 = np.sqrt(2)
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
SD = [kron(e2, s, e3, e3) for s in (sx, sy, sz)]
I1 = [kron(e2, e2, s, e3) for s in (Ix, Iy, Iz)]
I2 = [kron(e2, e2, e3, s) for s in (Ix, Iy, Iz)]
M = 9
dim = 4 * M


def sym_tensor(scale):
    a = rng.normal(size=(3, 3))
    return scale * (a + a.T) / 2


A1, A2 = sym_tensor(6e7), sym_tensor(3e7)          # rad/s
dip_axis = rng.normal(size=3)
dip_axis /= np.linalg.norm(dip_axis)
d = 2 * np.pi * 9.5e6                                # rad/s, about 0.34 mT
H0 = sum(A1[i, j] * SA[i] @ I1[j] for i in range(3) for j in range(3))
H0 += sum(A2[i, j] * SD[i] @ I2[j] for i in range(3) for j in range(3))
SAd = sum(dip_axis[i] * SA[i] for i in range(3))
SDd = sum(dip_axis[i] * SD[i] for i in range(3))
H0 += -d * (3 * SAd @ SDd - sum(SA[i] @ SD[i] for i in range(3)))
omega = 1.760859e11 * 50e-6                          # rad/s at 50 uT

up, dn = np.array([1, 0]), np.array([0, 1])
S = (np.kron(up, dn) - np.kron(dn, up)) / r2
T0 = (np.kron(up, dn) + np.kron(dn, up)) / r2
theta = np.pi / 8
psi_im = np.cos(theta) * S + 1j * np.sin(theta) * T0
psi_re = np.cos(theta) * S + np.sin(theta) * T0


def full(v):
    return np.kron(np.outer(v, v.conj()), np.eye(M))


PS, Qim, Qre = full(S), full(psi_im), full(psi_re)
kF = kR = 1e6
Id = np.eye(dim)


def yield_F(n, rho0, Q):
    H = H0 + omega * sum(n[i] * (SA[i] + SD[i]) for i in range(3))
    L = (-1j * (np.kron(Id, H) - np.kron(H.T, Id)) - kR / 2 * (np.kron(Id, Q) + np.kron(Q.T, Id))
         - kF * np.eye(dim * dim))
    x = np.linalg.solve(L, -rho0.reshape(-1, order="F") / M)
    return kF * np.trace(x.reshape(dim, dim, order="F")).real


dirs = rng.normal(size=(6, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
cases = {"(a) born and recombining through |psi_im>": (Qim, Qim),
         "(b) born |psi_im>, recombining through P_S": (Qim, PS),
         "(c) born |psi_re>, recombining through |psi_im>": (Qre, Qim)}
for name, (rho0, Q) in cases.items():
    odd = max(abs(yield_F(n, rho0, Q) - yield_F(-n, rho0, Q)) for n in dirs)
    scale = max(abs(yield_F(n, rho0, Q)) for n in dirs)
    if name.startswith("(a)"):
        print(f"{name}: worst |Phi(n) - Phi(-n)| / Phi = {odd / scale:.1e}")
    else:
        print(f"{name}: odd part nonzero = {odd / scale > 1e-9}")
