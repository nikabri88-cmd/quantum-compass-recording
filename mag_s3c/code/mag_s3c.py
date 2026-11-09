#!/usr/bin/env python3
"""
MAG-S3c - real geometry of the radical pair and decay of coherence.   Protocol version 0.3.

Pair. The three-nucleus pair of MAG-S2 and MAG-S3b (flavin anion with N5 and N10, tryptophan cation with N1) in three
geometries: S2 (RadicalPy 1.0.9 database tensors, as in MAG-S2 and MAG-S3b, J = D = 0) and the FAD-W_C (W318) and
FAD-W_D (W369) pairs of ClCry4a from PDB 6PU0. In W_C and W_D the RadicalPy principal values are kept, the two small
ones replaced by their mean; the large one lies along the normal of its ring (flavin normal z for N5 and N10, indole
normal for N1). The point-dipole tensor d (1 - 3 r r^T), d = mu0 g muB / (4 pi r^3), joins the ring centres; J = 0.
Frame of the flavin as in Gruning et al. 2022, made right-handed: origin at the centre of the central ring, x to
N10, y to the midpoint of C4X-C10 made orthogonal to x, z = x cross y; the receptor axis is z.

Kinetics and relaxation. Haberkorn recombination through P_rec at k_R and a spin-independent forward reaction at k_F
(Cry4a rates of MAG-S3b), plus one of two Lindblad superoperators with Hermitian jump operators:
  rf    random fields on each electron, k sum_i (sum_a S_ia rho S_ia - 3/4 rho), T1 = T2 = 1/k
  std   singlet-triplet dephasing, k (P_S rho P_S + P_T rho P_T - rho)
The time integral X of rho solves A'X + XB' - R(X) = rho(0) with A' = iH + G', B' = A'^+, G' = (k_R P_rec + k_F)/2 +
shift; GMRES in Liouville space (dimension 108^2) is preconditioned by the Sylvester solve of A'X + XB' = Y through
Schur forms (LAPACK trsyl). Without relaxation it is the frozen Sylvester solver of MAG-S3.

CISS forms on the flavin spin (electron 0), as in MAG-S3b: K (birth U P_S U^+, U = exp(-i chi_b n . sigma_A),
recombination through the turn by chi_r = chi_b (1 - m)), P-same and P-LH (|S> +- sin(chi/2)|T0 along n>), and the
control K-inc: the birth state of K at m = 1 with the singlet-triplet coherence removed, P_S rho P_S + P_T rho P_T,
recombination through P_S.

Modes
  pair       the cells of the main grid and of the branches E3, E5, E7, E10 (resumable; exit code 3 at the time limit)
  checks     the invariant checks I2, I3, I5 (small pair) and I6 (S2 against the frozen MAG-S3b code), after pair
  curves     the time curves of branch E9 (RK4 time steps)
  receptor   harmonics, receptor maps and E4 quantities of every cell, with the solver diagnostics (I4, I5)
  calibrate  C1-C4; C1-C3 at the MAG-S2 rates, which no hypothesis, expectation or map uses

Usage
  python3 -I mag_s3c.py pair --out RUN (--blind-rates | --confirmatory) [--workers 2] [--max-hours 1] [--grid-half N]
  python3 -I mag_s3c.py checks --out RUN (--blind-rates | --confirmatory) [--workers 2]
  python3 -I mag_s3c.py curves --out RUN (--blind-rates | --confirmatory) [--workers 2] [--t-max 4e-6]
  python3 -I mag_s3c.py receptor --out RUN [--workers 2]
  python3 -I mag_s3c.py calibrate --out CAL [--checks C1,C2,C3,C4] [--workers 2]
pair, checks and curves refuse to run without one of the two flags: before the freeze nothing is computed at the
rates of the hypotheses, and scripts/pipeline.sh passes --confirmatory only after the answer «запускай» of the
notebook. --grid-half, --t-max and --blind-rates serve only the smoke and blind checks of scripts/pipeline.sh.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                              # before NumPy loads BLAS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sl  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_VERSION = "0.3"

# Frozen files of the MAG-S3b release v0.2.0 (DOI 10.5281/zenodo.24079875), which carry MAG-S3 v0.1.0 and
# MAG-S2 v0.1.2; invariant I1. The structure file is checked against the hash written at the freeze.
FROZEN = {
    "mag_s3b.py": "f88118db34507156d3204c00368ba14c88c884e45f17ada8855da1fc45cd0ee3",
    "mag_s3.py": "1df703af63a2af71a94385d33276ad39a422e9abaa040fdf4bcb618344e1f350",
    "analyze.py": "ecc6a15b77e4f4d1c07a4217439e0c97868b9a12971c9ada255f79840ef1d21f",
    "mag_s2.py": "2352aefe0fe5974732c318b365f4fb5477ffc47fb78b12748bca2515465829c4",
    "physics.json": "647bc0ccb1312991a1b772193cac39f4ced1481c591422c50659bb822a3100c5",
}
PDB_FILE = os.path.join("data", "6PU0.pdb")
PDB_SHA256 = "39b25015964ecd1f53a09391c7ccd1515c31780c96a9e8861c4346f68e445a3b"
OWN_FILES = ("mag_s3c.py", "analyze_s3c.py", "test_s3c.py")
HARNESS_FILES = ("requirements.txt", "scripts/pipeline.sh", "scripts/env_check.py", "scripts/make_colab_notebook.py")


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def frozen_mismatches():
    out = [n for n, d in FROZEN.items()
           if not os.path.exists(os.path.join(HERE, n)) or file_sha256(os.path.join(HERE, n)) != d]
    pdb = os.path.join(HERE, PDB_FILE)
    if not os.path.exists(pdb) or file_sha256(pdb) != PDB_SHA256:
        out.append(PDB_FILE)
    return out


def load_mag_s3b():
    if "mag_s3b" in sys.modules:
        return sys.modules["mag_s3b"]
    bad = frozen_mismatches()
    if bad:
        sys.exit("frozen files differ from the MAG-S3b release v0.2.0 or the frozen structure: " + ", ".join(bad))
    spec = importlib.util.spec_from_file_location("mag_s3b", os.path.join(HERE, "mag_s3b.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3b"] = mod
    spec.loader.exec_module(mod)
    return mod


b = load_mag_s3b()
m3, m2 = b.m3, b.m2


def write_json(path, obj):
    m3.write_json_atomic(path, obj, indent=1)


def rates(name):
    """(k_F, k_R) in s^-1, read at call time so that the substitute rates of the blind check reach every solve."""
    k_f, k_r = b.RATES[name]
    return float(k_f), float(k_r)


def use_blind_rates():
    b.use_blind_rates()


# --------------------------------------------------------------------------------------------------------------
# Grids of the protocol
# --------------------------------------------------------------------------------------------------------------
GEOMETRIES = ("S2", "W_C", "W_D")
VARIANTS_E3 = ("W_C_noD", "W_D_noD")
VARIANTS_E7 = ("W_C_ErC", "W_D_ErD")
SYSTEMS = GEOMETRIES + VARIANTS_E3 + VARIANTS_E7
MAIN_AXES = {"S2": ("z", "x"), "W_C": ("z", "x", "r"), "W_D": ("z", "x", "r")}
LEVELS = (("none", 0.0), ("rf", 1e5), ("rf", 1e6), ("rf", 1e7), ("std", 1e5), ("std", 1e6), ("std", 1e7))
KEY_LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6))
E10_LEVELS = (("rf", 3e5), ("rf", 3e6), ("std", 3e5), ("std", 3e6))
FORM_COMBOS = (("K", 0.05, 0.1), ("K", 0.05, 1.0), ("K", 0.1, 0.1), ("K", 0.1, 1.0),
               ("K-inc", 0.05, 1.0), ("K-inc", 0.1, 1.0), ("P-same", 0.1, 1.0), ("P-LH", 0.1, 1.0))
N_E5_AXES = 20
GRID_HALF, GRID_HALF_CHECK = b.GRID_HALF, b.GRID_HALF_CHECK      # 2000 and 4000 directions with antipodes
CHECK_DIRS = 40
CHECK_STREAM = (71, 1)
GMRES_TOL, GMRES_RESTART, GMRES_MAXITER = 1e-13, 60, 50
CURVE_DT, CURVE_T, CURVE_EVERY = 2.5e-10, 4e-6, 4                 # RK4 step 0.25 ns to 4 us, output every 1 ns

TRP = {"W_A": 395, "W_B": 372, "W_C": 318, "W_D": 369}
INDOLE = ["CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"]
# Principal values of the MD-averaged tensors of Gruning et al. 2022 (SI tables S1 and S2, MHz) for branch E7
GRUNING_MHZ = {"W_C": {"N5": (-0.6, -0.4, 50.8), "N10": (0.1, 0.4, 17.8), "N1": (0.9, 1.5, 27.7)},
               "W_D": {"N5": (-0.6, -0.3, 50.6), "N10": (0.2, 0.5, 17.9), "N1": (27.1, 1.0, 0.2)}}
G_E, MU_B, H_PLANCK = 2.00231930436, 9.2740100783e-24, 6.62607015e-34
MHZ_PER_MT = G_E * MU_B / H_PLANCK * 1e-9                         # 28.025 MHz per mT
# MolSpin listing of Gruning et al. 2022 (ErC, FAD frame, tesla): tensor of the indole N1 (C4)
LISTING_WNE1 = np.array([[7.6295e-05, 0.0001, -8.8812e-05], [0.0001, 0.0006, -0.0004], [-8.8812e-05, -0.0004, 0.0004]])


# --------------------------------------------------------------------------------------------------------------
# Geometry from 6PU0
# --------------------------------------------------------------------------------------------------------------
def pdb_atoms(path, chain="A"):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and line[21] == chain and line[16] in (" ", "A"):
                key = (line[17:20].strip(), int(line[22:26]), line[12:16].strip())
                out[key] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


def geometry(path=None):
    """FAD frame (rows x, y, z in PDB coordinates) and, for the four tryptophans, the indole normal (sign: z >= 0),
    the centroid vector r from the frame origin, its length and unit vector, all in the FAD frame (angstrom)."""
    A = pdb_atoms(path or os.path.join(HERE, PDB_FILE))
    fad = {k[2]: v for k, v in A.items() if k[0] == "FAD"}
    o = np.array([fad[n] for n in ("N10", "N5", "C9A", "C10", "C5X", "C4X")]).mean(axis=0)
    x = fad["N10"] - o
    x /= np.linalg.norm(x)
    y = 0.5 * (fad["C4X"] + fad["C10"]) - o
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.vstack([x, y, z])
    out = dict(frame=R.tolist())
    for name, num in TRP.items():
        pts = np.array([A[("TRP", num, a)] for a in INDOLE])
        c = pts.mean(axis=0)
        n = R @ np.linalg.svd(pts - c)[2][2]
        n = n if n[2] >= 0 else -n
        r = R @ (c - o)
        out[name] = dict(normal=n, r=r, dist=float(np.linalg.norm(r)), r_hat=r / np.linalg.norm(r))
    return out


_GEO = None


def geo():
    global _GEO
    if _GEO is None:
        _GEO = geometry()
    return _GEO


def angle_deg(u, v):
    """Angle between two axes (sign-free), degrees."""
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(min(1.0, c))))


def geometry_report():
    g = geo()
    z = np.array([0.0, 0.0, 1.0])
    w, v = np.linalg.eigh(LISTING_WNE1)
    ax = v[:, np.argmax(np.abs(w))]
    rep = {t: dict(dist=g[t]["dist"], normal_to_z_deg=angle_deg(g[t]["normal"], z),
                   r_hat_to_z_deg=angle_deg(g[t]["r_hat"], z)) for t in TRP}
    rep["listing_N1_vs_W318_deg"] = angle_deg(ax, g["W_C"]["normal"])
    rep["listing_N1_vs_W318_after_z_sign_deg"] = angle_deg(ax, np.diag([1.0, 1.0, -1.0]) @ g["W_C"]["normal"])
    from radicalpy.data import Molecule
    fl = Molecule.fromdb("flavin_anion", nuclei=["N5", "N10"])
    tr = Molecule.fromdb("tryptophan_cation", nuclei=["N1"])
    db = {lab: main_axis(n.hfc.anisotropic) for lab, n in zip(("N5", "N10"), fl.nuclei)}
    db["N1"] = main_axis(tr.nuclei[0].hfc.anisotropic)
    rep["S2"] = dict(N5_to_z_deg=angle_deg(db["N5"], z), N10_to_z_deg=angle_deg(db["N10"], z),
                     N1_to_z_deg=angle_deg(db["N1"], z), N5_vs_N1_deg=angle_deg(db["N5"], db["N1"]))
    return rep


def main_axis(T):
    """Principal axis of the largest |principal value| of a tensor."""
    w, v = np.linalg.eigh(np.asarray(T, dtype=float))
    return v[:, int(np.argmax(np.abs(w)))]


def axial(T, axis):
    """Axial tensor with the principal values of T: the largest |value| along axis, the mean of the other two across."""
    w = np.linalg.eigvalsh(np.asarray(T, dtype=float))
    return axial_from_values(w, axis)


def axial_from_values(values, axis):
    w = np.asarray(values, dtype=float)
    i = int(np.argmax(np.abs(w)))
    par, perp = float(w[i]), float(np.mean(np.delete(w, i)))
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return perp * np.eye(3) + (par - perp) * np.outer(a, a)


def dipolar_tensor(r_angstrom):
    """Point-dipole tensor in mT by the formula of radicalpy.estimations.dipolar_interaction_point_dipole, (2/3) D_iso
    (3 r r^T - 1) with D_iso = dipolar_interaction_isotropic(|r|) < 0, i.e. d (1 - 3 r r^T), d = mu0 g muB/(4 pi r^3).
    The RadicalPy function is not called: its check of the eigenvalues compares them in a fixed order and fails for
    most directions of r."""
    from radicalpy.estimations import dipolar_interaction_isotropic
    r = np.asarray(r_angstrom, dtype=float) * 1e-10
    u = r / np.linalg.norm(r)
    return (2.0 / 3.0) * float(dipolar_interaction_isotropic(float(np.linalg.norm(r)))) * (3.0 * np.outer(u, u) - np.eye(3))


# --------------------------------------------------------------------------------------------------------------
# Spin systems
# --------------------------------------------------------------------------------------------------------------
class Sys:
    """Spin system of one geometry: H0 without the Zeeman term, P_S, the electron spin operators and sigma_A."""

    def __init__(self, sim, H0, name):
        from radicalpy.simulation import State
        self.sim, self.name = sim, name
        self.H0 = np.asarray(H0, dtype=complex)
        self.dim = self.H0.shape[0]
        self.eye = np.eye(self.dim, dtype=complex)
        self.PS = np.asarray(sim.projection_operator(State.SINGLET), dtype=complex)
        self.PT = self.eye - self.PS
        self.S = [[np.asarray(sim.spin_operator(r, a), dtype=complex) for a in "xyz"] for r in (0, 1)]
        self.sigma_A = [2.0 * s for s in self.S[0]]

    def H(self, n):
        th, ph = m2.to_angles(np.asarray(n, dtype=float)[None, :])
        return self.H0 + self.sim.zeeman_hamiltonian_3d(m3.P2.B0_mT, th[0], ph[0])

    def n_sigma(self, n):
        return n[0] * self.sigma_A[0] + n[1] * self.sigma_A[1] + n[2] * self.sigma_A[2]

    def turned(self, chi, n):
        """U P_S U^+ with U = exp(-i chi n . sigma_A), as mag_s3b.Spin.turned."""
        U = np.cos(chi) * self.eye - 1j * np.sin(chi) * self.n_sigma(n)
        Pm = U @ self.PS @ U.conj().T
        return 0.5 * (Pm + Pm.conj().T)

    def polarised(self, chi, n, sign):
        """|psi><psi| (x) 1 with psi = cos(chi/2)|S> + sign sin(chi/2)|T0 along n>, as mag_s3b.Spin.polarised."""
        A = np.cos(chi / 2) * self.eye + sign * np.sin(chi / 2) * self.n_sigma(n)
        Pm = A @ self.PS @ A.conj().T
        return 0.5 * (Pm + Pm.conj().T)

    def dephased(self, rho):
        """rho without its singlet-triplet coherence: P_S rho P_S + P_T rho P_T (same populations)."""
        X = self.PS @ rho @ self.PS + self.PT @ rho @ self.PT
        return 0.5 * (X + X.conj().T)


def system_s2():
    sim, H0, _, _ = m2.spin_system(m3.P2)
    return Sys(sim, H0, "S2")


def system_structure(trp, with_d=True, values="radicalpy", small=False):
    """Pair FAD-trp: flavin tensors along z, the N1 tensor along the indole normal, point dipole along the centroid
    vector, J = 0. values: 'radicalpy' (database principal values) or 'gruning' (tables S1 and S2 of Gruning et al.
    2022 for ErC with W_C and ErD with W_D). small: N5 and N1 only (checks)."""
    from radicalpy.data import Hfc, Molecule
    from radicalpy.simulation import Basis, HilbertSimulation
    g = geo()
    labels_fl = ["N5"] if small else ["N5", "N10"]
    fl = Molecule.fromdb("flavin_anion", nuclei=labels_fl)
    tr = Molecule.fromdb("tryptophan_cation", nuclei=["N1"])
    z = np.array([0.0, 0.0, 1.0])
    for lab, nuc in zip(labels_fl, fl.nuclei):
        T = (axial_from_values(np.array(GRUNING_MHZ[trp][lab]) / MHZ_PER_MT, z) if values == "gruning"
             else axial(nuc.hfc.anisotropic, z))
        nuc.hfc = Hfc(T.tolist())
    for nuc in tr.nuclei:
        T = (axial_from_values(np.array(GRUNING_MHZ[trp]["N1"]) / MHZ_PER_MT, g[trp]["normal"]) if values == "gruning"
             else axial(nuc.hfc.anisotropic, g[trp]["normal"]))
        nuc.hfc = Hfc(T.tolist())
    sim = HilbertSimulation([fl, tr], basis=Basis.ST)
    H0 = np.asarray(sim.total_hamiltonian(B0=0.0, J=0.0, D=0.0, hfc_anisotropy=True), dtype=complex)
    if with_d:
        H0 = H0 + np.asarray(sim.dipolar_hamiltonian(dipolar_tensor(g[trp]["r"])), dtype=complex)
    return Sys(sim, H0, trp + ("" if with_d else "_noD") + ("_gruning" if values == "gruning" else "")
               + ("_small" if small else ""))


_SYSTEMS = {}


def build_system(name):
    """S2, W_C, W_D (main grid), W_C_noD, W_D_noD (E3), W_C_ErC, W_D_ErD (E7); cached per process."""
    if name not in _SYSTEMS:
        if name == "S2":
            _SYSTEMS[name] = system_s2()
        else:
            trp = name[:3]
            _SYSTEMS[name] = system_structure(trp, with_d=not name.endswith("_noD"),
                                              values="gruning" if name.endswith(("_ErC", "_ErD")) else "radicalpy")
    return _SYSTEMS[name]


def trp_of(system_name):
    return None if system_name == "S2" else system_name[:3]


# --------------------------------------------------------------------------------------------------------------
# Liouville-space solver
# --------------------------------------------------------------------------------------------------------------
def relax_terms(sy, kind, k):
    """(shift on each side of the Sylvester part, [(coefficient, operator)] of the sandwich terms)."""
    if kind == "none" or k == 0.0:
        return 0.0, []
    if kind == "rf":        # random fields on each electron: k sum_i (sum_a S_ia X S_ia - 3/4 X)
        return 0.75 * k, [(k, S) for r in (0, 1) for S in sy.S[r]]
    if kind == "std":       # singlet-triplet dephasing: k (P_S X P_S + P_T X P_T - X)
        return 0.5 * k, [(k, sy.PS), (k, sy.PT)]
    raise ValueError(kind)


def sylvester_parts(sy, n, P_rec, kind, k, k_f, k_r):
    H = sy.H(n)
    shift, terms = relax_terms(sy, kind, k)
    G = 0.5 * (k_r * P_rec + k_f * sy.eye) + shift * sy.eye
    return 1j * H + G, terms


def solve(sy, n, P_init, P_rec, kind, k, k_f, k_r, tol=GMRES_TOL):
    """Time integral X of rho for d rho/dt = -i[H, rho] - k_r/2 {P_rec, rho} - k_f rho + R rho, rho(0) = P_init / Tr.
    Returns (forward yield k_f Tr X, recombination yield k_r Tr P_rec X, GMRES iterations, relative residual of the
    full equation)."""
    A, terms = sylvester_parts(sy, n, P_rec, kind, k, k_f, k_r)
    B = A.conj().T
    T, Q = sl.schur(A, output="complex")
    Sb, U = sl.schur(B, output="complex")
    trsyl = sl.lapack.get_lapack_funcs("trsyl", (T,))
    QH, UH = Q.conj().T, U.conj().T

    def L0inv(Y):
        Z, scale, info = trsyl(T, Sb, QH @ Y @ U)
        if info < 0:
            raise RuntimeError(f"trsyl info {info}")
        return Q @ (Z / scale) @ UH

    def R(X):
        out = np.zeros_like(X)
        for c, Op in terms:
            out += c * (Op @ X @ Op)
        return out

    rho0 = P_init / np.trace(P_init).real
    rhs = L0inv(rho0)
    it = 0
    if terms:
        d = sy.dim
        count = [0]
        op = spla.LinearOperator((d * d, d * d), dtype=complex,
                                 matvec=lambda x: (x.reshape(d, d) - L0inv(R(x.reshape(d, d)))).ravel())
        x, info = spla.gmres(op, rhs.ravel(), rtol=tol, atol=0.0, restart=GMRES_RESTART, maxiter=GMRES_MAXITER,
                             callback=lambda r: count.__setitem__(0, count[0] + 1), callback_type="pr_norm")
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info {info})")
        X, it = x.reshape(d, d), count[0]
    else:
        X = rhs
    res = A @ X + X @ B - R(X) - rho0
    rel = float(np.linalg.norm(res) / np.linalg.norm(rho0))
    f, rr = float(k_f * np.trace(X).real), float(k_r * np.trace(P_rec @ X).real)
    if not (np.isfinite(f) and np.isfinite(rr) and np.isfinite(rel)):
        raise RuntimeError("solution is not finite")
    return f, rr, it, rel


def liouvillian(sy, n, P_rec, kind, k, k_f, k_r):
    """Dense Liouvillian of the same equation (row-major vec: vec(A X B) = (A kron B^T) vec X); d rho/dt = L rho.
    For the small check pair only."""
    A, terms = sylvester_parts(sy, n, P_rec, kind, k, k_f, k_r)
    B = A.conj().T
    L = np.kron(A, sy.eye) + np.kron(sy.eye, B.T)
    for c, Op in terms:
        L -= c * np.kron(Op, Op.T)
    return -L


def dense_yield(sy, n, P_init, P_rec, kind, k, k_f, k_r):
    L = liouvillian(sy, n, P_rec, kind, k, k_f, k_r)
    x = np.linalg.solve(-L, (P_init / np.trace(P_init).real).ravel())
    return float(k_f * np.trace(x.reshape(sy.dim, sy.dim)).real)


def exact_steps_yield(sy, n, P_init, P_rec, kind, k, k_f, k_r, dt=2.5e-10, lifetimes=25.0):
    """Forward yield by steps of the exact Liouville propagator expm(L dt) with the trapezoid rule (small pair)."""
    E = sl.expm(liouvillian(sy, n, P_rec, kind, k, k_f, k_r) * dt)
    rho = (P_init / np.trace(P_init).real).ravel()
    steps = int(round(lifetimes / (k_f * dt)))
    idx = np.arange(sy.dim) * (sy.dim + 1)
    tr = np.empty(steps + 1)
    for i in range(steps + 1):
        if i:
            rho = E @ rho
        tr[i] = rho[idx].sum().real
    return float(k_f * np.trapezoid(tr, dx=dt))


def rk4_curve(sy, n, rho0, P_rec, kind, k, k_f, k_r, observables, dt=CURVE_DT, t_max=CURVE_T, every=CURVE_EVERY):
    """RK4 steps of d rho/dt = -A' rho - rho A'^+ + R(rho) from rho0 (not normalised). observables: name -> f(rho)
    sampled every `every` steps; the trapezoid integrals of k_f Tr rho over the steps are returned with them."""
    A, terms = sylvester_parts(sy, n, P_rec, kind, k, k_f, k_r)
    AH = A.conj().T

    def L(r):
        out = -(A @ r + r @ AH)
        for c, Op in terms:
            out += c * (Op @ r @ Op)
        return out

    steps = int(round(t_max / dt))
    r = rho0.astype(complex)
    samples = {name: [f(r)] for name, f in observables.items()}
    times = [0.0]
    tr_prev = float(np.trace(r).real)
    integral = 0.0
    for i in range(1, steps + 1):
        k1 = L(r)
        k2 = L(r + 0.5 * dt * k1)
        k3 = L(r + 0.5 * dt * k2)
        k4 = L(r + dt * k3)
        r = r + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tr = float(np.trace(r).real)
        integral += 0.5 * dt * (tr_prev + tr)
        tr_prev = tr
        if i % every == 0:
            times.append(i * dt)
            for name, f in observables.items():
                samples[name].append(f(r))
    return dict(t=times, **samples, forward_integral=float(k_f * integral), trace_end=tr_prev)


# --------------------------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    system: str               # S2, W_C, W_D, or a variant of E3 or E7
    axis: str                 # z, x, r (centroid vector of the pair), or f00-f19 (E5)
    relax: str                # none, rf, std
    k: float                  # relaxation rate, s^-1
    form: str                 # K, K-inc, P-same, P-LH
    chi: float                # chi_b of K and K-inc, chi of P, units of pi
    m: float = 1.0            # K: share of divergence; K-inc only at m = 1
    rates: str = "cry4a"
    branch: str = "main"

    @property
    def name(self):
        return f"{self.system}_{self.axis}_{self.relax}{self.k:g}_{self.form}_chi{self.chi:g}_m{self.m:g}_{self.rates}"


def main_cells():
    return [Cell(g, ax, rl, k, f, chi, m) for g in GEOMETRIES for ax in MAIN_AXES[g] for rl, k in LEVELS
            for f, chi, m in FORM_COMBOS]


def e3_cells():
    return [Cell(g, ax, rl, k, "K", 0.1, 1.0, branch="E3") for g in VARIANTS_E3 for ax in ("z", "x", "r")
            for rl, k in KEY_LEVELS]


def e5_cells():
    return [Cell(g, f"f{i:02d}", "none", 0.0, "K", 0.1, 1.0, branch="E5") for g in ("W_C", "W_D")
            for i in range(N_E5_AXES)]


def e7_cells():
    return [Cell(g, ax, rl, k, "K", 0.1, 1.0, branch="E7") for g in VARIANTS_E7 for ax in ("z", "r")
            for rl, k in KEY_LEVELS]


def e10_cells():
    return [Cell(g, "r", rl, k, f, 0.1, 1.0, branch="E10") for g in ("W_C", "W_D") for rl, k in E10_LEVELS
            for f in ("K", "K-inc")]


BRANCHES = {"main": main_cells, "E3": e3_cells, "E5": e5_cells, "E7": e7_cells, "E10": e10_cells}


def all_cells(branches=tuple(BRANCHES)):
    out = []
    for br in branches:
        out += BRANCHES[br]()
    return out


E5_AXES = m2.fibonacci_sphere(N_E5_AXES)


def axis_vector(system, axis):
    if axis == "z":
        return b.axis(0.0)
    if axis == "x":
        return b.axis(90.0)
    if axis == "r":
        if system == "S2":
            raise ValueError("geometry S2 has no centroid vector")
        return np.asarray(geo()[trp_of(system)]["r_hat"], dtype=float)
    if axis.startswith("f"):
        return np.asarray(E5_AXES[int(axis[1:])], dtype=float)
    raise ValueError(axis)


def cell_operators(sy, cell):
    """(P_init, P_rec) of a cell."""
    chi = cell.chi * np.pi
    n = axis_vector(cell.system, cell.axis)
    if cell.form == "K":
        return sy.turned(chi, n), sy.turned(chi * (1.0 - cell.m), n)
    if cell.form == "K-inc":
        if cell.m != 1.0:
            raise ValueError("K-inc is defined at m = 1")
        return sy.dephased(sy.turned(chi, n)), sy.PS.copy()
    if cell.form == "P-same":
        Pm = sy.polarised(chi, n, +1.0)
        return Pm, Pm
    if cell.form == "P-LH":
        return sy.polarised(chi, n, +1.0), sy.polarised(chi, n, -1.0)
    raise ValueError(cell.form)


def cell_yields(cell, dirs):
    """Forward yields at the field directions dirs with the solver diagnostics."""
    sy = build_system(cell.system)
    P_init, P_rec = cell_operators(sy, cell)
    k_f, k_r = rates(cell.rates)
    y = np.empty(len(dirs))
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    for i, n in enumerate(dirs):
        f, rr, it, rel = solve(sy, n, P_init, P_rec, cell.relax, cell.k, k_f, k_r)
        y[i] = f
        diag = diag_merge(diag, solve_diag(f, rr, it, rel))
    return y, diag


def solve_diag(f, rr, it, rel):
    return dict(max_rel_residual=rel, max_conservation_error=abs(f + rr - 1.0), max_gmres_iterations=it)


def _worst(a, z):
    """max that keeps a NaN (Python's max drops it when it comes second)."""
    return float("nan") if (a != a or z != z) else max(a, z)


def diag_merge(a, c):
    return dict(max_rel_residual=_worst(a["max_rel_residual"], c["max_rel_residual"]),
                max_conservation_error=_worst(a["max_conservation_error"], c["max_conservation_error"]),
                max_gmres_iterations=max(a["max_gmres_iterations"], c["max_gmres_iterations"]))


# --------------------------------------------------------------------------------------------------------------
# Mode pair
# --------------------------------------------------------------------------------------------------------------
def _cell_job(args):
    cell, folder, half_n = args
    path = os.path.join(folder, cell.name + ".json")
    if os.path.exists(path):
        return cell.name, "exists"
    t0 = time.perf_counter()
    half = m2.fibonacci_sphere(half_n)
    yp, dp = cell_yields(cell, half)
    ym, dm = cell_yields(cell, -half)
    write_json(path, dict(cell=asdict(cell), grid_half=half_n, rates_used=list(rates(cell.rates)),
                          mag_s3c_sha256=file_sha256(os.path.abspath(__file__)), yp=yp.tolist(), ym=ym.tolist(),
                          diagnostics=diag_merge(dp, dm), seconds=time.perf_counter() - t0))
    return cell.name, "done"


def pair(out, workers, max_hours, half_n=GRID_HALF, branches=tuple(BRANCHES)):
    """The cells (resumable; False when stopped at the time limit). The slowest cells go first."""
    from multiprocessing import get_context
    folder = os.path.join(out, "cells")
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    todo = [c for c in all_cells(branches) if not os.path.exists(os.path.join(folder, c.name + ".json"))]
    todo.sort(key=lambda c: -(c.k if c.relax == "rf" else 0.8 * c.k))
    with get_context("fork").Pool(workers) as pool:
        for name, state in pool.imap_unordered(_cell_job, [(c, folder, half_n) for c in todo]):
            print(name, state, f"{(time.time() - t0) / 3600:.2f} h", flush=True)
            if max_hours and time.time() - t0 > max_hours * 3600:
                pool.terminate()
                print("time limit, stopped; rerun to continue", flush=True)
                return False
    cells = all_cells(branches)
    if all(os.path.exists(os.path.join(folder, c.name + ".json")) for c in cells):
        write_json(os.path.join(out, "pair_done.json"), dict(n_cells=len(cells), grid_half=half_n,
                                                             metadata=metadata("pair")))
    return True


# --------------------------------------------------------------------------------------------------------------
# Mode checks: I2, I3, I5 (small pair), I6
# --------------------------------------------------------------------------------------------------------------
def check_dirs():
    return m2.uniform_dirs(m3.rng_for(*CHECK_STREAM), CHECK_DIRS)


def system_levels(system):
    return LEVELS if system in GEOMETRIES else KEY_LEVELS


def system_axes(system):
    if system in GEOMETRIES:
        return MAIN_AXES[system]
    return ("z", "x", "r") if system in VARIANTS_E3 else ("z", "r")


def odd_check_jobs(rate_name="cry4a"):
    """I2 in the seven spin systems and I3 in the 18 combinations of geometry and axis, each on every relaxation level
    of its cells, the levels of E10 included (W_C and W_D, axis r)."""
    jobs = []
    for system in SYSTEMS:
        for relax, k in system_levels(system):
            jobs.append(("I2", system, "-", relax, k, rate_name))
            for ax in system_axes(system):
                jobs.append(("I3", system, ax, relax, k, rate_name))
    for system in ("W_C", "W_D"):
        for relax, k in E10_LEVELS:
            jobs.append(("I2", system, "-", relax, k, rate_name))
            jobs.append(("I3", system, "r", relax, k, rate_name))
    return jobs


def _odd_job(args):
    """Largest odd part on the check directions for one (system, axis, level, kind of check)."""
    kind_check, system, axis, relax, k, rate_name = args
    sy = build_system(system)
    dirs = check_dirs()
    k_f, k_r = rates(rate_name)
    if kind_check == "I2":
        P_init, P_rec = sy.PS, sy.PS
    else:                                   # I3: reciprocal K, chi_r = chi_b = 0.1 pi, n_r = n_b
        P = sy.turned(0.1 * np.pi, axis_vector(system, axis))
        P_init, P_rec = P, P
    worst = 0.0
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    for n in dirs:
        a = solve(sy, n, P_init, P_rec, relax, k, k_f, k_r)
        c = solve(sy, -n, P_init, P_rec, relax, k, k_f, k_r)
        worst = _worst(worst, abs(a[0] - c[0]) / 2)
        for s in (a, c):
            diag = diag_merge(diag, solve_diag(*s))
    return kind_check, f"{system}_{axis}_{relax}{k:g}", worst, diag


SMALL_CHI = 0.1 * np.pi


def small_pair_cases(rate_name="cry4a"):
    """I5 on the small pair N5 + N1 in the W_C geometry at one check direction: the solver against the dense
    Liouville solve (three forms, seven levels) and against exact Liouville steps (K and P-LH at k = 0, 1e6, 1e7)."""
    cases = []
    for relax, k in LEVELS:
        for f in ("K", "P-same", "P-LH"):
            cases.append(("dense", f, relax, k, rate_name))
            if f in ("K", "P-LH") and k in (0.0, 1e6, 1e7):
                cases.append(("steps", f, relax, k, rate_name))
    return cases


def small_forms(sy):
    nb = axis_vector("W_C", "r")
    return {"K": (sy.turned(SMALL_CHI, nb), sy.PS),
            "P-same": (sy.polarised(SMALL_CHI, nb, 1.0), sy.polarised(SMALL_CHI, nb, 1.0)),
            "P-LH": (sy.polarised(SMALL_CHI, nb, 1.0), sy.polarised(SMALL_CHI, nb, -1.0))}


def _small_job(args):
    kind_check, f, relax, k, rate_name = args
    if "W_C_small" not in _SYSTEMS:
        _SYSTEMS["W_C_small"] = system_structure("W_C", small=True)
    sy = _SYSTEMS["W_C_small"]
    n0 = check_dirs()[0]
    Pi, Pr = small_forms(sy)[f]
    k_f, k_r = rates(rate_name)
    s = solve(sy, n0, Pi, Pr, relax, k, k_f, k_r)
    ref = (dense_yield if kind_check == "dense" else exact_steps_yield)(sy, n0, Pi, Pr, relax, k, k_f, k_r)
    return kind_check, f"{f}_{relax}{k:g}", abs(s[0] - ref), solve_diag(*s)


I6_CELLS = [c for c in main_cells() if c.system == "S2" and c.relax == "none" and c.form != "K-inc"]


def frozen_cell(cell):
    """The mag_s3b.Cell of an S2 cell without relaxation (axis z: beta = 0, axis x: beta = 90 degrees)."""
    beta = {"z": 0.0, "x": 90.0}[cell.axis]
    if cell.form == "K":
        return b.Cell("K", cell.chi, cell.m, 0.0, beta, cell.rates)
    return b.Cell(cell.form, cell.chi, beta=beta, rates=cell.rates)


def _i6_job(args):
    cell, half_n = args[0], args[1]
    half = m2.fibonacci_sphere(half_n)
    fc = frozen_cell(cell)
    s = b.spin()
    return cell.name, b.forward_yields(s, fc, half), b.forward_yields(s, fc, -half)


def _check_job(args):
    if args[0] in ("I2", "I3"):
        return ("odd",) + _odd_job(args)
    if args[0] in ("dense", "steps"):
        return ("small",) + _small_job(args)
    return ("I6",) + _i6_job(args[1:])


def checks(out, workers):
    """I2 and I3 on 40 directions and their antipodes, I5 on the small pair, I6 against the frozen MAG-S3b code, in
    one pool (the slowest jobs first)."""
    from multiprocessing import get_context
    folder = os.path.join(out, "cells")
    i6, half_n = {}, None
    for c in I6_CELLS:
        with open(os.path.join(folder, c.name + ".json")) as fh:
            r = json.load(fh)
        half_n = int(r["grid_half"])
        i6[c.name] = dict(yp=np.array(r["yp"]), ym=np.array(r["ym"]))
    jobs = ([j for j in small_pair_cases() if j[0] == "steps"] + [("I6", c, half_n) for c in I6_CELLS]
            + sorted(odd_check_jobs("cry4a"), key=lambda j: -j[4]) + [j for j in small_pair_cases() if j[0] == "dense"])
    res = dict(I2={}, I3={})
    dense, steps = {}, {}
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    with get_context("fork").Pool(workers) as pool:
        for out_ in pool.imap_unordered(_check_job, jobs):
            if out_[0] == "odd":
                _, kind_check, key, worst, d = out_
                res[kind_check][key] = worst
                diag = diag_merge(diag, d)
            elif out_[0] == "small":
                _, kind_check, key, err, d = out_
                (dense if kind_check == "dense" else steps)[key] = err
                diag = diag_merge(diag, d)
            else:
                _, name, yp, ym = out_
                i6[name].update(yp_frozen=yp, ym_frozen=ym)
    half = m2.fibonacci_sphere(half_n)
    i6_out = {}
    for c in I6_CELLS:
        r = i6[c.name]
        new, old = summarize(half, r["yp"], r["ym"]), summarize(half, r["yp_frozen"], r["ym_frozen"])
        rel = lambda a, z: abs(a / z - 1.0) if z != 0 else abs(a)  # noqa: E731
        d_new, d_old = np.array(new["d_odd"]), np.array(old["d_odd"])
        i6_out[c.name] = dict(
            max_yield_diff=float(max(np.abs(r["yp"] - r["yp_frozen"]).max(), np.abs(r["ym"] - r["ym_frozen"]).max())),
            eps_mol_rel=rel(new["parts"]["eps_mol"], old["parts"]["eps_mol"]),
            eps_rec_rel=rel(new["receptor"]["eps_rec"], old["receptor"]["eps_rec"]),
            receptor_even_range_rel=rel(new["receptor"]["receptor_even_range"], old["receptor"]["receptor_even_range"]),
            d_odd_rel=float(np.linalg.norm(d_new - d_old) / np.linalg.norm(d_old)))
    small = dict(dense=dense, steps=steps, max_dense=max(dense.values()), max_steps=max(steps.values()))
    write_json(os.path.join(out, "checks.json"), dict(
        I1=dict(frozen_mismatches=frozen_mismatches(), frozen=FROZEN, pdb_sha256=PDB_SHA256), I2=res["I2"],
        I3=res["I3"], I5_small=small, I6=i6_out, diagnostics=diag, check_dirs=CHECK_DIRS, grid_half=half_n,
        rates_used=list(rates("cry4a")), metadata=metadata("checks")))


# --------------------------------------------------------------------------------------------------------------
# Mode curves: branch E9
# --------------------------------------------------------------------------------------------------------------
CURVE_CASES = tuple((g, rl, k) for g in ("W_C", "W_D") for rl, k in KEY_LEVELS)
FIELD_UP, FIELD_DOWN = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])


def curve_states(sy, system):
    """Birth state of K at chi_b = 0.1 pi, m = 1, axis r, normalised, its control without coherence and the
    difference of the two (the initial singlet-triplet coherence)."""
    rho_k = sy.turned(0.1 * np.pi, axis_vector(system, "r"))
    rho_k = rho_k / np.trace(rho_k).real
    rho_inc = sy.dephased(rho_k)
    return rho_k, rho_inc, rho_k - rho_inc


def _curve_job(args):
    system, relax, k, which, t_max = args
    sy = build_system(system)
    rho_k, _, delta = curve_states(sy, system)
    k_f, k_r = rates("cry4a")
    fro = lambda X: float(np.linalg.norm(X))  # noqa: E731
    if which == "K_up":
        obs = dict(trace=lambda r: float(np.trace(r).real), st_coherence=lambda r: fro(sy.PS @ r @ sy.PT))
        out = rk4_curve(sy, FIELD_UP, rho_k, sy.PS, relax, k, k_f, k_r, obs, t_max=t_max)
        f, rr, it, rel = solve(sy, FIELD_UP, rho_k, sy.PS, relax, k, k_f, k_r)
        out["solver_forward_yield"] = f
        out["solver_diagnostics"] = dict(max_rel_residual=rel, max_conservation_error=abs(f + rr - 1.0),
                                         max_gmres_iterations=it)
    else:
        n = FIELD_UP if which == "delta_up" else FIELD_DOWN
        obs = dict(trace=lambda r: float(np.trace(r).real), norm=fro)
        out = rk4_curve(sy, n, delta, sy.PS, relax, k, k_f, k_r, obs, t_max=t_max)
    return f"{system}_{relax}{k:g}", which, out


def curves(out, workers, t_max=CURVE_T):
    from multiprocessing import get_context
    jobs = [(g, rl, k, which, t_max) for g, rl, k in CURVE_CASES for which in ("K_up", "delta_up", "delta_down")]
    res = {}
    with get_context("fork").Pool(workers) as pool:
        for case, which, r in pool.imap_unordered(_curve_job, jobs):
            res.setdefault(case, {})[which] = r
    write_json(os.path.join(out, "curves.json"), dict(cases=res, dt=CURVE_DT, t_max=t_max, every=CURVE_EVERY,
                                                      rates_used=list(rates("cry4a")), metadata=metadata("curves")))


# --------------------------------------------------------------------------------------------------------------
# Mode receptor
# --------------------------------------------------------------------------------------------------------------
E4_CELLS = {c.name for c in main_cells() if c.system in ("W_C", "W_D") and c.form == "K" and c.chi == 0.1
            and c.m == 1.0 and (c.relax, c.k) in KEY_LEVELS}


def summarize(half, yp, ym, e4=False):
    """Pair and receptor quantities of one cell with the frozen functions of MAG-S3b: fit of degree 10, check fit of
    degree 24 (Q1), receptor map over p and S (F = eps_eff at S = 1, p = 1), I5 of MAG-S3b (here I7), E4 at p = 0."""
    fit = b.harmonic_fit(half, yp, ym, b.L_FIT)
    hi = b.harmonic_fit(half, yp, ym, b.L_ODD_MAP)
    zon = b.zonal(fit)
    d = b.dipole(fit)
    row = dict(parts=b.parity_parts(yp, ym), fit_L=b.L_FIT, fit_even_rms_over_sd=fit["even_rms_over_sd"],
               fit_odd_rms_over_sd=fit["odd_rms_over_sd"], d_odd=d.tolist(), zonal=zon.tolist(),
               receptor=b.receptor_order_map(zon), I7=b.i5_receptor(zon), odd_power_shares=b.odd_power_shares(fit),
               fit_check=dict(L=b.L_ODD_MAP, even_rms_over_sd=hi["even_rms_over_sd"],
                              odd_rms_over_sd=hi["odd_rms_over_sd"], d_odd=b.dipole(hi).tolist()))
    if e4:
        row["E4"] = b.e4_cell(half, hi["coef"], b.L_ODD_MAP, d, zon)
    return row


def _receptor_job(args):
    c, folder, own = args
    with open(os.path.join(folder, c.name + ".json")) as fh:
        r = json.load(fh)
    h = int(r["grid_half"])
    row = summarize(m2.fibonacci_sphere(h), np.array(r["yp"]), np.array(r["ym"]), e4=c.name in E4_CELLS)
    row.update(cell=asdict(c), name=c.name, diagnostics=r["diagnostics"], rates_used=r["rates_used"],
               code_ok=r.get("mag_s3c_sha256") == own)
    return h, row


def receptor(out, workers=1, branches=tuple(BRANCHES)):
    from multiprocessing import get_context
    folder = os.path.join(out, "cells")
    own = file_sha256(os.path.abspath(__file__))
    cells = all_cells(branches)
    with get_context("fork").Pool(workers) as pool:
        got = pool.map(_receptor_job, [(c, folder, own) for c in cells], chunksize=1)
    halves = {h for h, _ in got}
    rows = [row for _, row in got]
    if len(halves) != 1:
        sys.exit("cells were computed on different grids")
    g = geometry_report()
    write_json(os.path.join(out, "receptor.json"), dict(
        cells=rows, grid_half=next(iter(halves)), geometry=g, R_REN=b.R_REN,
        order={f"S{S:g}": dict(g=b.g_order(S), D=b.depth(S)) for S in b.S_LEVELS}, metadata=metadata("receptor")))
    return rows


# --------------------------------------------------------------------------------------------------------------
# Calibration C1-C4
# --------------------------------------------------------------------------------------------------------------
C1_CELLS = (Cell("W_C", "r", "rf", 1e6, "K", 0.1, 1.0, "mags2", "C1"),
            Cell("W_C", "r", "rf", 1e6, "P-same", 0.1, 1.0, "mags2", "C1"))
PROTOCOL_GEOMETRY = dict(W_C=dict(dist=17.61, normal_to_z_deg=50.0, r_hat_to_z_deg=72.4),
                         W_D=dict(dist=21.27, normal_to_z_deg=81.6, r_hat_to_z_deg=78.8),
                         W_A=dict(r_hat_to_z_deg=89.3), W_B=dict(r_hat_to_z_deg=57.3),
                         S2=dict(N5_to_z_deg=9.4, N10_to_z_deg=9.4, N1_to_z_deg=37.0, N5_vs_N1_deg=46.3),
                         listing_N1_vs_W318_after_z_sign_deg=3.9, listing_N1_vs_W318_deg=77.6)


def _chunk_job(args):
    cell, dirs = args
    return cell_yields(cell, dirs)


def c1_c2_c3(workers, out_dir):
    """C1: doubling the grid changes eps_mol, F and d_odd . z by less than 1%; C2: degree 12 changes eps_rec by less
    than 1%; C3: residuals and |Phi + Phi_R - 1| at least ten times below the thresholds of I5 (1e-10) and I4 (1e-10)
    in the C1 cells and on 40 directions of each relaxation level without CISS, GMRES within its iterations. MAG-S2
    rates throughout; only relative changes and margins are written, the yields stay in C1_yields.npz."""
    from multiprocessing import get_context
    path = os.path.join(out_dir, "C1_yields.npz")
    stored = dict(np.load(path)) if os.path.exists(path) else {}
    out, diag = {}, dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    as_diag = lambda v: dict(max_rel_residual=float(v[0]), max_conservation_error=float(v[1]),  # noqa: E731
                             max_gmres_iterations=int(v[2]))
    with get_context("fork").Pool(workers) as pool:
        for cell in C1_CELLS:
            row = {}
            for half_n in (GRID_HALF, GRID_HALF_CHECK):
                half = m2.fibonacci_sphere(half_n)
                key = f"{cell.name}_grid{2 * half_n}"
                if key not in stored:
                    allx = np.vstack([half, -half])
                    parts = np.array_split(allx, 4 * workers)
                    got = pool.map(_chunk_job, [(cell, p) for p in parts])
                    stored[key] = np.concatenate([g[0] for g in got])
                    d = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
                    for g in got:
                        d = diag_merge(d, g[1])
                    stored[key + "__diag"] = np.array([d["max_rel_residual"], d["max_conservation_error"],
                                                       d["max_gmres_iterations"]])
                    np.savez(path, **stored)
                diag = diag_merge(diag, as_diag(stored[key + "__diag"]))
                y = stored[key]
                s = summarize(half, y[:half_n], y[half_n:])
                row[half_n] = s
                if half_n == GRID_HALF:
                    fit12 = b.harmonic_fit(half, y[:half_n], y[half_n:], b.L_FIT_CHECK)
                    zon12 = b.zonal(fit12)
                    row["eps_rec_L12"] = float(zon12[1] / b.zonal_range(zon12))
            s1, s2 = row[GRID_HALF], row[GRID_HALF_CHECK]
            F = lambda s: s["receptor"]["by_order"]["S1_p1"]["eps_eff"]  # noqa: E731
            out[cell.name] = dict(
                C1_eps_mol_rel_change=float(s2["parts"]["eps_mol"] / s1["parts"]["eps_mol"] - 1.0),
                C1_F_rel_change=float(F(s2) / F(s1) - 1.0),
                C1_d_odd_z_rel_change=float(s2["d_odd"][2] / s1["d_odd"][2] - 1.0),
                C2_eps_rec_rel_change=float(row["eps_rec_L12"] / s1["receptor"]["eps_rec"] - 1.0))
        jobs = [("I2", "W_C", "-", rl, k, "mags2") for rl, k in LEVELS]
        for _, _, _, d in pool.imap_unordered(_odd_job, jobs):
            diag = diag_merge(diag, d)
    c1 = all(abs(v[key]) < 0.01 for v in out.values() for key in
             ("C1_eps_mol_rel_change", "C1_F_rel_change", "C1_d_odd_z_rel_change"))
    c2 = all(abs(v["C2_eps_rec_rel_change"]) < 0.01 for v in out.values())
    # solve() raises when GMRES does not converge within GMRES_MAXITER restarts, so that part holds by construction;
    # the number of restarts is reported
    restarts = -(-diag["max_gmres_iterations"] // GMRES_RESTART)
    c3 = (diag["max_rel_residual"] < 1e-11 and diag["max_conservation_error"] < 1e-11 and restarts <= GMRES_MAXITER)
    return dict(C1=dict(cells=out, criterion_met=bool(c1)), C2=dict(criterion_met=bool(c2)),
                C3=dict(diagnostics=dict(diag, max_gmres_restarts=restarts), criterion_met=bool(c3)))


def c4_geometry():
    rep = geometry_report()
    dev = {}
    for t in ("W_A", "W_B", "W_C", "W_D", "S2"):
        for key, want in PROTOCOL_GEOMETRY[t].items():
            dev[f"{t}_{key}"] = abs(rep[t][key] - want)
    for key in ("listing_N1_vs_W318_after_z_sign_deg", "listing_N1_vs_W318_deg"):
        dev[key] = abs(rep[key] - PROTOCOL_GEOMETRY[key])
    ok = all((v <= 0.01 if k.endswith("_dist") else v <= 0.1) for k, v in dev.items())
    return dict(report=rep, deviations=dev, criterion_met=bool(ok))


def calibrate(out, checks_wanted, workers):
    os.makedirs(out, exist_ok=True)
    meta = metadata("calibrate")
    res = {}
    t0 = time.perf_counter()
    if any(c in checks_wanted for c in ("C1", "C2", "C3")):
        res.update(c1_c2_c3(workers, out))
    if "C4" in checks_wanted:
        res["C4"] = c4_geometry()
    res["seconds"] = time.perf_counter() - t0
    res["metadata"] = meta
    write_json(os.path.join(out, "calibration.json"), res)
    print({k: v["criterion_met"] for k, v in res.items() if isinstance(v, dict) and "criterion_met" in v}, flush=True)


# --------------------------------------------------------------------------------------------------------------
# Metadata and command line
# --------------------------------------------------------------------------------------------------------------
def metadata(mode):
    """Environment, rates and the SHA-256 computed now of every file of the study and of its harness."""
    sha = lambda f: file_sha256(os.path.join(HERE, f)) if os.path.exists(os.path.join(HERE, f)) else None  # noqa: E731
    own = {f: sha(f) for f in OWN_FILES if sha(f)}
    files = {f: sha(f) for f in OWN_FILES + tuple(FROZEN) + (PDB_FILE,) + HARNESS_FILES}
    return dict(mode=mode, protocol=PROTOCOL_VERSION, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                python=sys.version.split()[0], numpy=np.__version__, scipy=m3.scipy.__version__,
                radicalpy=m3._pkg_version("radicalpy"), blas=m3._blas(), frozen_ok=not frozen_mismatches(),
                frozen=FROZEN, pdb_sha256=PDB_SHA256, own=own, files=files, rates=dict(b.RATES))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pair", "checks", "curves", "receptor", "calibrate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--grid-half", type=int, default=GRID_HALF, help="pair: smaller grids only for the smoke check")
    ap.add_argument("--t-max", type=float, default=CURVE_T, help="curves: shorter curves only for the smoke check")
    ap.add_argument("--checks", default="C1,C2,C3,C4")
    ap.add_argument("--blind-rates", action="store_true", help="the substitute rates of the blind check")
    ap.add_argument("--confirmatory", action="store_true",
                    help="the confirmatory run at the Cry4a rates (set by scripts/pipeline.sh after the confirmation)")
    a = ap.parse_args()
    if a.blind_rates and a.confirmatory:
        sys.exit("--blind-rates and --confirmatory exclude each other")
    if a.mode in ("pair", "checks", "curves") and not (a.blind_rates or a.confirmatory):
        sys.exit("pair, checks and curves compute at the rates of the hypotheses only in the confirmatory run "
                 "(--confirmatory, set by scripts/pipeline.sh after the answer «запускай»); before the freeze use "
                 "--blind-rates")
    if a.blind_rates:
        use_blind_rates()
    if a.mode == "pair":
        if not pair(a.out, a.workers, a.max_hours, a.grid_half):
            sys.exit(3)
    elif a.mode == "checks":
        checks(a.out, a.workers)
    elif a.mode == "curves":
        curves(a.out, a.workers, a.t_max)
    elif a.mode == "receptor":
        receptor(a.out, a.workers)
    else:
        calibrate(a.out, [c.strip() for c in a.checks.split(",") if c.strip()], a.workers)


if __name__ == "__main__":
    main()
