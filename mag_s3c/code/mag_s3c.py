#!/usr/bin/env python3
"""
MAG-S3c - real geometry of the radical pair and decay of coherence.   Protocol version 0.4.

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
recombination through the turn by chi_r = chi_b (1 - m)), P-same and P-LH (|S> +- sin(chi/2)|T0 along n>). Controls
with recombination through P_S: K-inc (the birth state of K at m = 1 without its singlet-triplet coherence,
P_S rho P_S + P_T rho P_T), K-iso (cos^2 chi P_S + sin^2 chi P_T / 3) and P-0 (the birth state of the P forms).
Form E (branch E11): the birth state of Denton et al. 2026, Eq. (7) at d_eps = 0 in the sign of K,
cos^2 P_S + sin^2 A + sin cos (Re eta C - Im eta C_r), eta = k2 / (k2 + 2i J_Ex), recombination through the state of
K; C = i(P_S s - s P_S), C_r = P_S s + s P_S, A = s P_S s, s = n . sigma_A.

References. For singlet recombination the yield of any birth state follows from the pair born in the singlet
(chi = 0): Phi_R(-n; rho) = k_R Z Tr[T rho T^-1 X_S(n)], T the time reversal of every spin. The code stores the
partial trace of X_S(n) over the nuclei by its 16 Pauli coefficients c_mn = Tr[sigma_m(0) sigma_n(1) X_S(n)] and
rebuilds K at m = 1, K-inc, K-iso and P-0 at any CISS axis without new solves.

Composite pair (branch E13): W_C <-> W_D with both tryptophan N1 nuclei (dimension 324). Site s carries the system of
its pair and the other N1 as a free spin with its nuclear Zeeman term; hops at k_CD and k_DC keep the spins (they swap
the two N1 factors of the site orderings). GMRES over the two sites, preconditioned by the Sylvester solve of each
site (eigenbases of the 108-dim part and of the free nucleus) or, at fast hops, by the two-site system with the
Hamiltonian averaged over the site populations and the hops kept exactly.

Modes
  pair       direct cells and references (resumable; exit code 3 at the time limit)
  composite  the cells of branch E13 (resumable; exit code 3 at the time limit)
  checks     the invariant checks I3, I5 (small pair and small composite) and I6, after pair
  curves     the time curves of branch E9 (RK4 time steps)
  receptor   rebuilt cells, harmonics, receptor maps, E4, the maps E5, E14 and the circle of H2, I8, I9, I11
  calibrate  C1-C6; C1-C3 at the MAG-S2 and the substitute rates, which no hypothesis, expectation or map uses

Usage
  python3 -I mag_s3c.py pair --out RUN (--blind-rates | --confirmatory) [--workers 2] [--max-hours 1] [--grid-half N]
  python3 -I mag_s3c.py composite --out RUN (--blind-rates | --confirmatory) [--workers 2] [--max-hours 1]
          [--grid-half N]
  python3 -I mag_s3c.py checks --out RUN (--blind-rates | --confirmatory) [--workers 2]
  python3 -I mag_s3c.py curves --out RUN (--blind-rates | --confirmatory) [--workers 2] [--t-max 4e-6]
  python3 -I mag_s3c.py receptor --out RUN [--workers 2]
  python3 -I mag_s3c.py calibrate --out CAL [--checks C1,C2,C3,C4,C5,C6] [--workers 2]
pair, composite, checks and curves refuse to run without one of the two flags: before the freeze nothing is computed
at the rates of the hypotheses, and scripts/pipeline.sh passes --confirmatory only after the answer «запускай» of the
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
PROTOCOL_VERSION = "0.4"

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


# Rates of branch E12, the regime of the largest anisotropy of Denton et al. 2026 (k_SP = 1 us^-1, k_CR2 = 1e2 and
# 1e3 us^-1), and the hop rates of branch E13; the blind check multiplies both by BLIND_FACTOR. "blind_cry4a" is the
# substitute Cry4a rate pair at every time, for the cell of C1 at strong dephasing.
S3C_RATES_NOMINAL = {"cr2_1e2": (1.0e6, 1.0e8), "cr2_1e3": (1.0e6, 1.0e9)}
S3C_RATES = dict(S3C_RATES_NOMINAL)
BLIND_FACTOR = 1.2
HOP_SCALE = [1.0]


def rates(name):
    """(k_F, k_R) in s^-1, read at call time so that the substitute rates of the blind check reach every solve."""
    if name == "blind_cry4a":
        k_f, k_r = b.BLIND_RATES["cry4a"]
    elif name in S3C_RATES:
        k_f, k_r = S3C_RATES[name]
    else:
        k_f, k_r = b.RATES[name]
    return float(k_f), float(k_r)


def hop_rate(k):
    """Hop rate in s^-1 of a nominal rate of branch E13 (times BLIND_FACTOR in the blind check)."""
    return float(k) * HOP_SCALE[0]


def use_blind_rates():
    b.use_blind_rates()
    for key, (k_f, k_r) in S3C_RATES_NOMINAL.items():
        S3C_RATES[key] = (BLIND_FACTOR * k_f, BLIND_FACTOR * k_r)
    HOP_SCALE[0] = BLIND_FACTOR


# --------------------------------------------------------------------------------------------------------------
# Grids of the protocol
# --------------------------------------------------------------------------------------------------------------
GEOMETRIES = ("S2", "W_C", "W_D")
VARIANTS_E3 = ("W_C_noD", "W_D_noD")
VARIANTS_E7 = ("W_C_ErC", "W_D_ErD")
SYSTEMS = GEOMETRIES + VARIANTS_E3 + VARIANTS_E7
MAIN_AXES = {"S2": ("z", "x"), "W_C": ("z", "x", "r"), "W_D": ("z", "x", "r")}
LEVELS_OLD = (("none", 0.0), ("rf", 1e5), ("rf", 1e6), ("rf", 1e7), ("std", 1e5), ("std", 1e6), ("std", 1e7))
LEVELS_NEW = (("rf", 3e7), ("std", 3e7), ("std", 1e8))
LEVELS = LEVELS_OLD + LEVELS_NEW
KEY_LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6))
E10_LEVELS = (("rf", 3e5), ("rf", 3e6), ("std", 3e5), ("std", 3e6))
E12_LEVELS = (("none", 0.0), ("std", 3e7))
E12_RATES = ("cr2_1e2", "cr2_1e3")
# direct cells: (form, chi in units of pi, m)
FORM_COMBOS = (("K", 0.05, 0.1), ("K", 0.05, 1.0), ("K", 0.1, 0.1), ("K", 0.1, 1.0),
               ("K-inc", 0.05, 1.0), ("K-inc", 0.1, 1.0), ("P-same", 0.1, 1.0), ("P-LH", 0.1, 1.0))
NEW_LEVEL_FORMS = (("P-same", 0.1, 1.0), ("P-LH", 0.1, 1.0))
# rebuilt cells: controls on every level, K at m = 1 and K-inc on the new levels
REBUILT_ALL_LEVELS = (("K-iso", 0.05, 1.0), ("K-iso", 0.1, 1.0), ("P-0", 0.1, 1.0))
REBUILT_NEW_LEVELS = (("K", 0.05, 1.0), ("K", 0.1, 1.0), ("K-inc", 0.05, 1.0), ("K-inc", 0.1, 1.0))
REBUILT_FORMS = ("K", "K-inc", "K-iso", "P-0")
# form E: exchange in the bridge and its lifetime, models 1 and 2 of table I of Denton et al. 2026
DENTON_J = 2.0 * np.pi * 1.39e10                                  # J_Ex = 1.39e4 MHz in rad s^-1
DENTON_K2 = {"M1": 1.3e10, "M2": 5.0e10}                          # s^-1
# composite pair of branch E13
HOP_RATES = (1e7, 1e8, 1e9)                                       # k_CD, s^-1 (nominal)
HOP_RATIOS = (1.0, 0.1)                                           # k_DC / k_CD
E13_GRID_HALF = 250                                               # 500 directions with antipodes
H2_CIRCLE = 360                                                   # CISS axes across the receptor axis (H2)
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
        self._pauli = None

    def pauli(self):
        """The 16 electron Pauli products sigma_m(0) sigma_n(1) of the full space, sigma_0 = 1, index 4 m + n."""
        if self._pauli is None:
            sig = [[self.eye] + [2.0 * S for S in self.S[r]] for r in (0, 1)]
            self._pauli = np.array([sig[0][a] @ sig[1][c] for a in range(4) for c in range(4)])
        return self._pauli

    def isotropic(self, chi):
        """Birth state of K-iso: the populations of K with an isotropic triplet."""
        return np.cos(chi) ** 2 * self.PS + np.sin(chi) ** 2 * self.PT / 3.0

    def coherence_ops(self, n):
        """(C, C_r, A) along n: imaginary and real singlet-triplet coherence and the triplet aligned along n."""
        s = self.n_sigma(n)
        return 1j * (self.PS @ s - s @ self.PS), self.PS @ s + s @ self.PS, s @ self.PS @ s

    def bridge(self, theta, n, eta):
        """Birth state of form E, Eq. (7) of Denton et al. 2026 at d_eps = 0 in the sign of K (theta -> -theta):
        cos^2 P_S + sin^2 A + sin cos (Re eta C - Im eta C_r); eta = 1 gives the birth state of K."""
        C, Cr, A = self.coherence_ops(n)
        X = (np.cos(theta) ** 2 * self.PS + np.sin(theta) ** 2 * A
             + np.sin(theta) * np.cos(theta) * (eta.real * C - eta.imag * Cr))
        return 0.5 * (X + X.conj().T)

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


def system_structure(trp, with_d=True, values="radicalpy", small=False, tiny=False):
    """Pair FAD-trp: flavin tensors along z, the N1 tensor along the indole normal, point dipole along the centroid
    vector, J = 0. values: 'radicalpy' (database principal values) or 'gruning' (tables S1 and S2 of Gruning et al.
    2022 for ErC with W_C and ErD with W_D). small: N5 and N1 only (checks); tiny: N1 only (sites of the small
    composite pair)."""
    from radicalpy.data import Hfc, Molecule
    from radicalpy.simulation import Basis, HilbertSimulation
    g = geo()
    labels_fl = [] if tiny else (["N5"] if small else ["N5", "N10"])
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
               + ("_tiny" if tiny else ("_small" if small else "")))


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
    return solve_X(sy, n, P_init, P_rec, kind, k, k_f, k_r, tol)[1:]


def solve_X(sy, n, P_init, P_rec, kind, k, k_f, k_r, tol=GMRES_TOL):
    """As solve, with the time integral X first: (X, forward yield, recombination yield, iterations, residual)."""
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
    return X, f, rr, it, rel


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
    axis: str                 # z, x, r (centroid vector of the pair)
    relax: str                # none, rf, std
    k: float                  # relaxation rate, s^-1
    form: str                 # K, K-inc, K-iso, P-same, P-LH, P-0, E-M1, E-M2
    chi: float                # chi_b of K, K-inc and K-iso, chi of the P forms, theta of E, units of pi
    m: float = 1.0            # K: share of divergence; the controls only at m = 1
    rates: str = "cry4a"
    branch: str = "main"
    source: str = "direct"    # direct (own solves) or ref (rebuilt from the reference of its system and level)

    @property
    def name(self):
        return f"{self.system}_{self.axis}_{self.relax}{self.k:g}_{self.form}_chi{self.chi:g}_m{self.m:g}_{self.rates}"


@dataclass(frozen=True)
class Ref:
    """Reference: the pair born in the singlet and recombining through P_S, on the whole grid."""
    system: str
    relax: str
    k: float
    rates: str = "cry4a"
    branch: str = "main"

    @property
    def name(self):
        return f"ref_{self.system}_{self.relax}{self.k:g}_{self.rates}"


def ref_of(cell):
    return Ref(cell.system, cell.relax, cell.k, cell.rates)


def _grid(systems, axes, levels, combos, branch, source, rate_names=("cry4a",)):
    return [Cell(g, ax, rl, k, f, chi, m, rn, branch, source) for g in systems for ax in axes(g) for rn in rate_names
            for rl, k in levels for f, chi, m in combos]


def direct_cells(branches=None):
    """574 cells with their own solves: 484 of the main grid, 36 of E3, 36 of E11, 18 of E12."""
    main = _grid(GEOMETRIES, MAIN_AXES.get, LEVELS_OLD, FORM_COMBOS, "main", "direct")
    main += _grid(("W_C", "W_D"), MAIN_AXES.get, LEVELS_NEW, NEW_LEVEL_FORMS, "main", "direct")
    out = dict(main=main,
               E3=_grid(VARIANTS_E3, lambda g: ("z", "x", "r"), KEY_LEVELS, NEW_LEVEL_FORMS, "E3", "direct"),
               E11=_grid(("W_C",), lambda g: ("z", "x", "r"), KEY_LEVELS,
                         [(f"E-{mo}", th, 1.0) for mo in DENTON_K2 for th in (0.05, 0.1)], "E11", "direct"),
               E12=_grid(("W_C",), lambda g: ("z", "x", "r"), (("none", 0.0),),
                         (("P-same", 0.1, 1.0), ("P-LH", 0.1, 1.0), ("E-M1", 0.1, 1.0)), "E12", "direct", E12_RATES))
    return [c for br, cs in out.items() if branches is None or br in branches for c in cs]


def rebuilt_cells(branches=None):
    """442 cells rebuilt from the references: 336 of the main grid, 54 of E3, 12 of E7, 16 of E10, 24 of E12."""
    main = _grid(GEOMETRIES, MAIN_AXES.get, LEVELS, REBUILT_ALL_LEVELS, "main", "ref")
    main += _grid(GEOMETRIES, MAIN_AXES.get, LEVELS_NEW, REBUILT_NEW_LEVELS, "main", "ref")
    k_p0 = (("K", 0.05, 1.0), ("K", 0.1, 1.0), ("P-0", 0.1, 1.0))
    out = dict(main=main,
               E3=_grid(VARIANTS_E3, lambda g: ("z", "x", "r"), KEY_LEVELS, k_p0, "E3", "ref"),
               E7=_grid(VARIANTS_E7, lambda g: ("z", "r"), KEY_LEVELS, (("K", 0.1, 1.0),), "E7", "ref"),
               E10=_grid(("W_C", "W_D"), lambda g: ("r",), E10_LEVELS, (("K", 0.1, 1.0), ("K-inc", 0.1, 1.0)), "E10",
                         "ref"),
               E12=_grid(("W_C",), lambda g: ("z", "x", "r"), E12_LEVELS, (("K", 0.1, 1.0), ("P-0", 0.1, 1.0)), "E12",
                         "ref", E12_RATES))
    return [c for br, cs in out.items() if branches is None or br in branches for c in cs]


def references():
    """54 references: 30 of the main grid, 6 of E3, 6 of E7, 8 of E10, 4 of E12."""
    out = [Ref(g, rl, k) for g in GEOMETRIES for rl, k in LEVELS]
    out += [Ref(g, rl, k, branch="E3") for g in VARIANTS_E3 for rl, k in KEY_LEVELS]
    out += [Ref(g, rl, k, branch="E7") for g in VARIANTS_E7 for rl, k in KEY_LEVELS]
    out += [Ref(g, rl, k, branch="E10") for g in ("W_C", "W_D") for rl, k in E10_LEVELS]
    out += [Ref("W_C", rl, k, rn, "E12") for rn in E12_RATES for rl, k in E12_LEVELS]
    return out


def e5_references():
    """References of the map E5: every level of the three geometries, with the levels of E10 for W_C and W_D."""
    return [r for r in references() if r.branch in ("main", "E10")]


def axis_vector(system, axis):
    if axis == "z":
        return b.axis(0.0)
    if axis == "x":
        return b.axis(90.0)
    if axis == "r":
        if system == "S2":
            raise ValueError("geometry S2 has no centroid vector")
        return np.asarray(geo()[trp_of(system)]["r_hat"], dtype=float)
    raise ValueError(axis)


def eta(model):
    """Bridge factor of form E, k2 / (k2 + 2i J_Ex)."""
    k2 = DENTON_K2[model]
    return k2 / (k2 + 2j * DENTON_J)


def birth_operators(sy, form, chi, m, n):
    """(P_init, P_rec) of a form at the CISS axis n; chi in radians."""
    if form == "K":
        return sy.turned(chi, n), sy.turned(chi * (1.0 - m), n)
    if form in ("K-inc", "K-iso", "P-0") and m != 1.0:
        raise ValueError(f"{form} is defined at m = 1")
    if form == "K-inc":
        return sy.dephased(sy.turned(chi, n)), sy.PS.copy()
    if form == "K-iso":
        return sy.isotropic(chi), sy.PS.copy()
    if form == "P-0":
        return sy.polarised(chi, n, +1.0), sy.PS.copy()
    if form == "P-same":
        Pm = sy.polarised(chi, n, +1.0)
        return Pm, Pm
    if form == "P-LH":
        return sy.polarised(chi, n, +1.0), sy.polarised(chi, n, -1.0)
    if form.startswith("E-"):
        return sy.bridge(chi, n, eta(form[2:])), sy.turned(chi, n)
    raise ValueError(form)


def cell_operators(sy, cell):
    """(P_init, P_rec) of a cell."""
    return birth_operators(sy, cell.form, cell.chi * np.pi, cell.m, axis_vector(cell.system, cell.axis))


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


# --------------------------------------------------------------------------------------------------------------
# References and reconstruction
# --------------------------------------------------------------------------------------------------------------
# time reversal flips every electron Pauli matrix: sign (-1)^(number of non-identity factors) of sigma_m sigma_n
SIGNS = np.array([(1.0 if a == 0 else -1.0) * (1.0 if c == 0 else -1.0) for a in range(4) for c in range(4)])


def pauli_coefficients(sy, X):
    """c_mn = Tr[sigma_m(0) sigma_n(1) X] (real for Hermitian X): the partial trace of X over the nuclei."""
    return np.einsum("aij,ji->a", sy.pauli(), X).real


def pauli_weights(sy, rho):
    """Weights w of a birth state acting on the electrons only: Phi_R(-n; rho) = k_R w . c(n), with c(n) the
    coefficients of the reference at n, w = s_m s_n Tr[sigma_m sigma_n rho] / (4 Tr rho)."""
    return SIGNS * pauli_coefficients(sy, rho) / (4.0 * np.trace(rho).real)


def ref_solutions(ref, dirs):
    """Forward yields of the singlet-born pair and the 16 coefficients c(n) at the directions dirs."""
    sy = build_system(ref.system)
    k_f, k_r = rates(ref.rates)
    y, coef = np.empty(len(dirs)), np.empty((len(dirs), 16))
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    for i, n in enumerate(dirs):
        X, f, rr, it, rel = solve_X(sy, n, sy.PS, sy.PS, ref.relax, ref.k, k_f, k_r)
        y[i], coef[i] = f, pauli_coefficients(sy, X)
        diag = diag_merge(diag, solve_diag(f, rr, it, rel))
    return y, coef, diag


def rebuilt_yields(w, cp, cm, k_r):
    """(yp, ym) of a birth state with weights w from the coefficients of the reference at +half (cp) and -half (cm):
    the yield at +n needs the reference at -n, and Phi = 1 - Phi_R."""
    return 1.0 - k_r * (cm @ w), 1.0 - k_r * (cp @ w)


def cell_weights(sy, cell):
    if cell.form not in REBUILT_FORMS:
        raise ValueError(f"form {cell.form} is not rebuilt")
    P_init, P_rec = cell_operators(sy, cell)
    if not np.allclose(P_rec, sy.PS, atol=1e-14):
        raise ValueError("only cells recombining through P_S are rebuilt")
    return pauli_weights(sy, P_init)


def basis_weights(sy):
    """Weights of the terms of the exact split of K (Tr = Z): P_S, C_i along x, y, z, and the aligned triplets
    A_ij = (s_i P_S s_j + s_j P_S s_i) / 2 for ij in xx, yy, zz, xy, xz, yz (s_i = sigma_A along axis i)."""
    Z = np.trace(sy.PS).real
    ax = np.eye(3)
    s = [sy.n_sigma(a) for a in ax]
    ops = [sy.PS] + [1j * (sy.PS @ si - si @ sy.PS) for si in s]
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        ops.append(0.5 * (s[i] @ sy.PS @ s[j] + s[j] @ sy.PS @ s[i]))
    return np.array([SIGNS * pauli_coefficients(sy, Op) / (4.0 * Z) for Op in ops])


def split_weights(W10, chi, axes):
    """Weights of K (and of K-inc) at m = 1 for CISS axes (N, 3) from the basis weights W10 (10, 16): rho_K =
    cos^2 P_S + sin cos sum n_i C_i + sin^2 sum n_i n_j A_ij, rho_inc without the C term."""
    n = np.asarray(axes, dtype=float)
    quad = np.stack([n[:, 0] ** 2, n[:, 1] ** 2, n[:, 2] ** 2, 2 * n[:, 0] * n[:, 1], 2 * n[:, 0] * n[:, 2],
                     2 * n[:, 1] * n[:, 2]], axis=1)
    c2, sc, s2 = np.cos(chi) ** 2, np.sin(chi) * np.cos(chi), np.sin(chi) ** 2
    w_inc = c2 * W10[0][None, :] + s2 * quad @ W10[4:]
    return w_inc + sc * n @ W10[1:4], w_inc


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
# Mode pair: direct cells and references
# --------------------------------------------------------------------------------------------------------------
def pair_items():
    return direct_cells() + references()


def item_path(out, item):
    return os.path.join(out, "refs" if isinstance(item, Ref) else "cells", item.name + ".json")


def _cost(item):
    """Relative cost of a solve (slowest first): GMRES iterations grow with k, dephasing at 1e8 above all."""
    return item.k if item.relax == "rf" else 0.8 * item.k


def _cell_job(args):
    item, out, half_n = args
    path = item_path(out, item)
    if os.path.exists(path):
        return item.name, "exists"
    t0 = time.perf_counter()
    half = m2.fibonacci_sphere(half_n)
    common = dict(grid_half=half_n, rates_used=list(rates(item.rates)), mag_s3c_sha256=file_sha256(os.path.abspath(__file__)))
    if isinstance(item, Ref):
        yp, cp, dp = ref_solutions(item, half)
        ym, cm, dm = ref_solutions(item, -half)
        write_json(path, dict(ref=asdict(item), **common, yp=yp.tolist(), ym=ym.tolist(), cp=cp.tolist(),
                              cm=cm.tolist(), diagnostics=diag_merge(dp, dm), seconds=time.perf_counter() - t0))
    else:
        yp, dp = cell_yields(item, half)
        ym, dm = cell_yields(item, -half)
        write_json(path, dict(cell=asdict(item), **common, yp=yp.tolist(), ym=ym.tolist(),
                              diagnostics=diag_merge(dp, dm), seconds=time.perf_counter() - t0))
    return item.name, "done"


def pair(out, workers, max_hours, half_n=GRID_HALF, items=None):
    """Direct cells and references (resumable; False when stopped at the time limit). The slowest go first. items:
    a subset for the tests."""
    from multiprocessing import get_context
    for sub in ("cells", "refs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    t0 = time.time()
    items = pair_items() if items is None else list(items)
    todo = sorted([c for c in items if not os.path.exists(item_path(out, c))], key=lambda c: -_cost(c))
    with get_context("fork").Pool(workers) as pool:
        for name, state in pool.imap_unordered(_cell_job, [(c, out, half_n) for c in todo]):
            print(name, state, f"{(time.time() - t0) / 3600:.2f} h", flush=True)
            if max_hours and time.time() - t0 > max_hours * 3600:
                pool.terminate()
                print("time limit, stopped; rerun to continue", flush=True)
                return False
    if all(os.path.exists(item_path(out, c)) for c in items):
        write_json(os.path.join(out, "pair_done.json"), dict(
            n_cells=sum(not isinstance(c, Ref) for c in items), n_refs=sum(isinstance(c, Ref) for c in items),
            grid_half=half_n, metadata=metadata("pair")))
    return True


# --------------------------------------------------------------------------------------------------------------
# Composite pair W_C <-> W_D (branch E13)
# --------------------------------------------------------------------------------------------------------------
class Composite:
    """Two sites with the same particles in different orders: site s carries its own system (electrons, N5, N10, the
    N1 of its tryptophan; dimension d) and the N1 of the other tryptophan as a free spin with its nuclear Zeeman term,
    as the last factor. The hop keeps every spin and swaps the two N1 factors between the site orderings."""

    def __init__(self, sy_c, sy_d):
        if not np.array_equal(sy_c.PS, sy_d.PS) or sy_c.dim != sy_d.dim:
            raise ValueError("the two sites need the same electron and nuclear structure")
        self.sites, self.d = (sy_c, sy_d), sy_c.dim
        self.D = 3 * self.d
        last = len(sy_c.sim.particles) - 1
        nuc = sy_c.sim.particles[last]
        if nuc.multiplicity != 3:
            raise ValueError("the free spin is the N1 of a tryptophan")
        self.gamma = float(nuc.gamma_mT)
        self.I = []
        for a in "xyz":
            full = np.asarray(sy_c.sim.spin_operator(last, a), dtype=complex)
            Ia = full[:3, :3].copy()
            if np.abs(np.kron(np.eye(self.d // 3), Ia) - full).max() > 1e-14:
                raise ValueError("the N1 spin is not the last factor")
            self.I.append(Ia)
        self.perm = np.arange(self.D).reshape(self.d // 3, 3, 3).transpose(0, 2, 1).ravel()
        self.PS = np.kron(sy_c.PS, np.eye(3))
        self.eye = np.eye(self.D, dtype=complex)

    def h_free(self, n):
        """Nuclear Zeeman term of the free N1, as RadicalPy writes it: -B0 gamma (rotation . I)."""
        from radicalpy import utils
        th, ph = m2.to_angles(np.asarray(n, dtype=float)[None, :])
        rot = utils.spherical_to_cartesian(th[0], ph[0])
        return -m3.P2.B0_mT * self.gamma * sum(r * Ia for r, Ia in zip(rot, self.I))

    def H(self, s, n):
        return np.kron(self.sites[s].H(n), np.eye(3)) + np.kron(np.eye(self.d), self.h_free(n))

    def hop(self, X):
        """The same state in the ordering of the other site."""
        return X[np.ix_(self.perm, self.perm)]

    def birth(self, P_site_c):
        rho = np.kron(P_site_c, np.eye(3))
        return rho / np.trace(rho).real


_COMPOSITES = {}


def build_composite(tiny=False):
    key = "tiny" if tiny else "full"
    if key not in _COMPOSITES:
        if tiny:
            _COMPOSITES[key] = Composite(system_structure("W_C", tiny=True), system_structure("W_D", tiny=True))
        else:
            _COMPOSITES[key] = Composite(build_system("W_C"), build_system("W_D"))
    return _COMPOSITES[key]


COMPOSITE_RESTART, COMPOSITE_MAXITER = 120, 50
COMPOSITE_AVERAGED_ABOVE = 4e8      # k_CD + k_DC above which the averaged two-site system preconditions


def _eig_sylvester(A, spec_vals, spec_vecs):
    """Solver of A_f X + X A_f^+ = Y for A_f = A (x) 1 + 1 (x) i h, from the eigenvectors of A and of the Hermitian
    h (spec_vals, spec_vecs): X = W [(W^-1 Y W^-+) / (L_i + L_j^*)] W^+, W = V_A (x) V_h. A is the site matrix of a
    radical pair, nearly normal (its non-Hermitian part is k_R P_S / 2 plus a constant)."""
    d = A.shape[0]
    lam, Vb = np.linalg.eig(A)
    Vi = np.linalg.inv(Vb)
    V, VH = spec_vecs, spec_vecs.conj().T
    ev = (lam[:, None] + 1j * spec_vals[None, :]).ravel()
    den = (ev[:, None] + ev.conj()[None, :]).reshape(d, 3, d, 3)
    ViH, VbH = Vi.conj().T, Vb.conj().T

    def inv(Y):
        Y4 = np.einsum("jp,apbq,qk->ajbk", VH, Y.reshape(d, 3, d, 3), V, optimize=True)
        Y4 = np.tensordot(np.tensordot(Vi, Y4, axes=([1], [0])), ViH, axes=([2], [0])).transpose(0, 1, 3, 2)
        Z4 = Y4 / den
        Z4 = np.tensordot(np.tensordot(Vb, Z4, axes=([1], [0])), VbH, axes=([2], [0])).transpose(0, 1, 3, 2)
        return np.einsum("pj,ajbk,kq->apbq", V, Z4, VH, optimize=True).reshape(3 * d, 3 * d)
    return inv


class Hermitian2:
    """Real coordinates of a pair of Hermitian D x D matrices (the two sites): the real part on and above the
    diagonal and the imaginary part above it, off-diagonal entries times sqrt 2, so that the Euclidean norm of the
    coordinates is the Frobenius norm of the pair."""

    def __init__(self, D):
        self.D = D
        self.iu, self.iu1 = np.triu_indices(D), np.triu_indices(D, 1)
        self.w = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.n1 = len(self.iu[0]) + len(self.iu1[0])          # D^2 per site

    def pack1(self, X):
        return np.concatenate([X.real[self.iu] * self.w, X.imag[self.iu1] * np.sqrt(2.0)])

    def unpack1(self, v):
        D, m = self.D, len(self.iu[0])
        R = np.zeros((D, D))
        R[self.iu] = v[:m] / self.w
        R = R + R.T - np.diag(np.diag(R))
        Im = np.zeros((D, D))
        Im[self.iu1] = v[m:] / np.sqrt(2.0)
        return R + 1j * (Im - Im.T)

    def pack(self, XC, XD):
        return np.concatenate([self.pack1(XC), self.pack1(XD)])

    def unpack(self, v):
        return self.unpack1(v[:self.n1]), self.unpack1(v[self.n1:])


def gmres_real(matvec, rhs, tol, restart, maxiter):
    """Restarted GMRES on real vectors: classical Gram-Schmidt with one reorthogonalisation and Givens rotations.
    Converged when the true residual is at most tol ||rhs||. Returns (x, info, iterations): info 0 when converged,
    else the number of restarts done."""
    from scipy.linalg import solve_triangular
    n = len(rhs)
    x = np.zeros(n)
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return x, 0, 0
    it = 0
    r = rhs.copy()
    for _ in range(maxiter):
        beta = float(np.linalg.norm(r))
        if beta <= tol * bnorm:
            return x, 0, it
        V = np.empty((restart + 1, n))
        V[0] = r / beta
        H = np.zeros((restart + 1, restart))
        c, s = np.zeros(restart), np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = beta
        k = 0
        for j in range(restart):
            w = matvec(V[j])
            it += 1
            h = V[:j + 1] @ w
            w -= h @ V[:j + 1]
            h2 = V[:j + 1] @ w
            w -= h2 @ V[:j + 1]
            h += h2
            hn = float(np.linalg.norm(w))
            H[:j + 1, j], H[j + 1, j] = h, hn
            for i in range(j):
                t = c[i] * H[i, j] + s[i] * H[i + 1, j]
                H[i + 1, j] = -s[i] * H[i, j] + c[i] * H[i + 1, j]
                H[i, j] = t
            den = float(np.hypot(H[j, j], H[j + 1, j]))
            c[j], s[j] = H[j, j] / den, H[j + 1, j] / den
            H[j, j], H[j + 1, j] = den, 0.0
            g[j + 1], g[j] = -s[j] * g[j], c[j] * g[j]
            k = j + 1
            if hn == 0.0 or abs(g[j + 1]) <= 0.5 * tol * bnorm:
                break
            V[j + 1] = w / hn
        y = solve_triangular(H[:k, :k], g[:k])
        x = x + y @ V[:k]
        r = rhs - matvec(x)
    return x, (0 if float(np.linalg.norm(r)) <= tol * bnorm else maxiter), it


def composite_solve(cs, n, rho0, k_rec, k_out, k_f, tol=GMRES_TOL):
    """Time integrals (X_C, X_D) of the two sites: d rho_s/dt = -i[H_s, rho_s] - k_rec_s/2 {P_S, rho_s}
    - (k_f + k_out_s) rho_s + k_out_s' hop(rho_s'), rho_C(0) = rho0, rho_D(0) = 0 (k_out = (k_CD, k_DC)).
    GMRES in the real coordinates of Hermitian matrices. Below COMPOSITE_AVERAGED_ABOVE: the block system
    preconditioned by the Sylvester solve of each site (left), with X_D eliminated; above it: the two sites,
    preconditioned (right) by the two-site system with the Hamiltonian and recombination averaged over the site populations
    k_DC : k_CD and the hops kept exactly, which splits into the modes 0 and k_CD + k_DC of the hop matrix (right).
    Returns (forward yield, recombination yield, GMRES iterations, relative residual of the full equations,
    (recombination yield at C, at D))."""
    d, D = cs.d, cs.D
    hv, hV = np.linalg.eigh(cs.h_free(n))
    k_cd, k_dc = k_out
    A_site = [1j * cs.sites[s].H(n) + 0.5 * (k_rec[s] * cs.sites[s].PS + (k_f + k_out[s]) * cs.sites[s].eye)
              for s in (0, 1)]
    ih = 1j * cs.h_free(n)

    def A_times(s, X):
        """A_f X with A_f = A_site (x) 1 + 1 (x) i h."""
        X4 = X.reshape(d, 3, D)
        return (np.tensordot(A_site[s], X4, axes=([1], [0])) + np.einsum("jp,apy->ajy", ih, X4, optimize=True)
                ).reshape(D, D)

    def L_site(s, X):
        """A_f X + X A_f^+ = A_f X + (A_f X^+)^+ (X need not be Hermitian inside GMRES)."""
        return A_times(s, X) + A_times(s, X.conj().T).conj().T

    def L(XC, XD):
        return L_site(0, XC) - k_dc * cs.hop(XD), L_site(1, XD) - k_cd * cs.hop(XC)

    hp = Hermitian2(D)
    zero = np.zeros((D, D), dtype=complex)
    if k_cd + k_dc <= COMPOSITE_AVERAGED_ABOVE:
        site_inv = [_eig_sylvester(A_site[s], hv, hV) for s in (0, 1)]
        to_d = lambda XC: k_cd * site_inv[1](cs.hop(XC))  # noqa: E731   X_D from X_C (rho_D(0) = 0)

        def matvec(v):
            """X_C - k_DC S_C^-1 hop(X_D(X_C)): the block system with X_D eliminated."""
            XC = hp.unpack1(v)
            return hp.pack1(XC - k_dc * site_inv[0](cs.hop(to_d(XC))))
        y, info, iters = gmres_real(matvec, hp.pack1(site_inv[0](rho0)), tol, COMPOSITE_RESTART, COMPOSITE_MAXITER)
        XC = hp.unpack1(y)
        X = (XC, to_d(XC))
    else:
        pc, pd = k_dc / (k_cd + k_dc), k_cd / (k_cd + k_dc)
        Hbar = pc * cs.H(0, n) + pd * cs.hop(cs.H(1, n))
        Abar = 1j * Hbar + 0.5 * ((pc * k_rec[0] + pd * k_rec[1]) * cs.PS + k_f * cs.eye)
        lam, W = np.linalg.eig(Abar)
        Wi = np.linalg.inv(W)
        WH, WiH = W.conj().T, Wi.conj().T
        den = lam[:, None] + lam.conj()[None, :]
        mu = k_cd + k_dc

        def P_inv(RC, RD):
            RD = cs.hop(RD)
            Z1 = W @ ((Wi @ ((RC + RD) / mu) @ WiH) / den) @ WH
            Z2 = W @ ((Wi @ ((k_cd * RC - k_dc * RD) / mu) @ WiH) / (den + mu)) @ WH
            return k_dc * Z1 + Z2, cs.hop(k_cd * Z1 - Z2)

        y, info, iters = gmres_real(lambda v: hp.pack(*L(*P_inv(*hp.unpack(v)))), hp.pack(rho0, zero), tol,
                                    COMPOSITE_RESTART, COMPOSITE_MAXITER)
        X = P_inv(*hp.unpack(y))
        X = tuple(0.5 * (Xs + Xs.conj().T) for Xs in X)
    count = [iters]
    if info != 0:
        raise RuntimeError(f"GMRES of the composite pair did not converge (info {info})")
    rC, rD = L(*X)
    rel = float(np.sqrt(np.linalg.norm(rC - rho0) ** 2 + np.linalg.norm(rD) ** 2) / np.linalg.norm(rho0))
    f = float(k_f * (np.trace(X[0]).real + np.trace(X[1]).real))
    rec = tuple(float(k_rec[s] * np.trace(cs.PS @ X[s]).real) for s in (0, 1))
    if not all(np.isfinite(v) for v in (f, rel) + rec):
        raise RuntimeError("solution is not finite")
    return f, rec[0] + rec[1], count[0], rel, rec


def composite_dense(cs, n, rho0, k_rec, k_out, k_f):
    """Forward yield of the same equations by a dense solve in Liouville space (small composite only)."""
    D = cs.D
    L = []
    for s in (0, 1):
        A = 1j * cs.H(s, n) + 0.5 * (k_rec[s] * cs.PS + (k_f + k_out[s]) * cs.eye)
        L.append(np.kron(A, cs.eye) + np.kron(cs.eye, A.conj()))      # row-major vec of A X + X A^+
    i = np.arange(D)
    P = np.zeros((D * D, D * D))
    P[(i[:, None] * D + i[None, :]).ravel(), (cs.perm[:, None] * D + cs.perm[None, :]).ravel()] = 1.0
    M = np.block([[L[0], -k_out[1] * P], [-k_out[0] * P, L[1]]])
    x = np.linalg.solve(M, np.concatenate([rho0.ravel(), np.zeros(D * D)]))
    return float(k_f * (np.trace(x[:D * D].reshape(D, D)).real + np.trace(x[D * D:].reshape(D, D)).real))


@dataclass(frozen=True)
class CompositeCell:
    form: str                 # none (born in the singlet) or K (chi_b = 0.1 pi, m = 1, axis r of W_C)
    rec: str                  # C: singlet recombination only at W_C; CD: at both sites with the same k_R
    k_cd: float               # nominal hop rate W_C -> W_D, s^-1
    k_dc: float               # nominal hop rate W_D -> W_C, s^-1
    rates: str = "cry4a"
    branch: str = "E13"

    @property
    def name(self):
        return f"E13_{self.form}_rec{self.rec}_kcd{self.k_cd:g}_kdc{self.k_dc:g}_{self.rates}"


def composite_cells():
    return [CompositeCell(f, rc, k, k * q) for f in ("none", "K") for rc in ("C", "CD") for k in HOP_RATES
            for q in HOP_RATIOS]


def composite_setup(cc, tiny=False):
    cs = build_composite(tiny)
    sy = cs.sites[0]
    P = sy.PS if cc.form == "none" else sy.turned(0.1 * np.pi, axis_vector("W_C", "r"))
    k_f, k_r = rates(cc.rates)
    return cs, cs.birth(P), (k_r, k_r if cc.rec == "CD" else 0.0), (hop_rate(cc.k_cd), hop_rate(cc.k_dc)), k_f


def composite_yields(cc, dirs):
    cs, rho0, k_rec, k_out, k_f = composite_setup(cc)
    y, rc, rd = np.empty(len(dirs)), np.empty(len(dirs)), np.empty(len(dirs))
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    for i, n in enumerate(dirs):
        f, rr, it, rel, rec = composite_solve(cs, n, rho0, k_rec, k_out, k_f)
        y[i], rc[i], rd[i] = f, rec[0], rec[1]
        diag = diag_merge(diag, solve_diag(f, rr, it, rel))
    return y, rc, rd, diag


COMPOSITE_PART = 25                                               # directions per job of mode composite


def composite_parts(half_n):
    """The directions +half then -half, split into jobs of about COMPOSITE_PART directions."""
    half = m2.fibonacci_sphere(half_n)
    allx = np.vstack([half, -half])
    return np.array_split(allx, max(1, len(allx) // COMPOSITE_PART))


def _composite_job(args):
    cc, out, half_n, part = args
    path = os.path.join(out, "composite", "parts", f"{cc.name}__{part:03d}.json")
    if os.path.exists(path):
        return cc.name, part, "exists"
    t0 = time.perf_counter()
    dirs = composite_parts(half_n)[part]
    y, rc, rd, diag = composite_yields(cc, dirs)
    write_json(path, dict(part=part, y=y.tolist(), rec_C=rc.tolist(), rec_D=rd.tolist(), diagnostics=diag,
                          seconds=time.perf_counter() - t0))
    return cc.name, part, "done"


def composite_merge(out, cc, half_n):
    """Join the parts of one cell into composite/NAME.json (yields at +half, then at -half)."""
    n_parts = len(composite_parts(half_n))
    got = []
    for part in range(n_parts):
        with open(os.path.join(out, "composite", "parts", f"{cc.name}__{part:03d}.json")) as fh:
            got.append(json.load(fh))
    cat = lambda key: np.concatenate([np.asarray(g[key], dtype=float) for g in got])  # noqa: E731
    y, rc, rd = cat("y"), cat("rec_C"), cat("rec_D")
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    for g in got:
        diag = diag_merge(diag, g["diagnostics"])
    write_json(os.path.join(out, "composite", cc.name + ".json"), dict(
        cell=asdict(cc), grid_half=half_n, rates_used=list(rates(cc.rates)),
        hop_rates_used=[hop_rate(cc.k_cd), hop_rate(cc.k_dc)], mag_s3c_sha256=file_sha256(os.path.abspath(__file__)),
        yp=y[:half_n].tolist(), ym=y[half_n:].tolist(), rec_C_p=rc[:half_n].tolist(), rec_C_m=rc[half_n:].tolist(),
        rec_D_p=rd[:half_n].tolist(), rec_D_m=rd[half_n:].tolist(), diagnostics=diag,
        seconds=float(sum(g["seconds"] for g in got))))


def composite(out, workers, max_hours, half_n=E13_GRID_HALF):
    """The 24 cells of branch E13 (resumable): jobs of COMPOSITE_PART directions, the fastest hops last."""
    from multiprocessing import get_context
    os.makedirs(os.path.join(out, "composite", "parts"), exist_ok=True)
    t0 = time.time()
    cells = composite_cells()
    n_parts = len(composite_parts(half_n))
    jobs = [(c, out, half_n, p) for c in sorted(cells, key=lambda c: -max(c.k_cd, c.k_dc)) for p in range(n_parts)
            if not os.path.exists(os.path.join(out, "composite", "parts", f"{c.name}__{p:03d}.json"))]
    with get_context("fork").Pool(workers) as pool:
        for name, part, state in pool.imap_unordered(_composite_job, jobs):
            print(name, part, state, f"{(time.time() - t0) / 3600:.2f} h", flush=True)
            if max_hours and time.time() - t0 > max_hours * 3600:
                pool.terminate()
                print("time limit, stopped; rerun to continue", flush=True)
                return False
    for c in cells:
        if not os.path.exists(os.path.join(out, "composite", c.name + ".json")):
            composite_merge(out, c, half_n)
    write_json(os.path.join(out, "composite_done.json"), dict(n_cells=len(cells), grid_half=half_n,
                                                              metadata=metadata("composite")))
    return True


# --------------------------------------------------------------------------------------------------------------
# Mode checks: I3, I5 (small pair and small composite), I6
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
    """I3 in the 18 combinations of spin system and axis on the levels of their branch, in W_C and W_D also on the
    levels of E10 and, in W_C, at the rates of E12: 146 sets."""
    jobs = [("I3", s, ax, rl, k, rate_name) for s in SYSTEMS for rl, k in system_levels(s) for ax in system_axes(s)]
    jobs += [("I3", s, ax, rl, k, rate_name) for s in ("W_C", "W_D") for rl, k in E10_LEVELS for ax in ("z", "x", "r")]
    jobs += [("I3", "W_C", ax, rl, k, rn) for rn in E12_RATES for rl, k in E12_LEVELS for ax in ("z", "x", "r")]
    return jobs


def _odd_job(args):
    """Largest odd part on the check directions for one (system, axis, level, rates): I2 (born in the singlet) or I3
    (reciprocal K, chi_r = chi_b = 0.1 pi, n_r = n_b)."""
    kind_check, system, axis, relax, k, rate_name = args
    sy = build_system(system)
    dirs = check_dirs()
    k_f, k_r = rates(rate_name)
    if kind_check == "I2":
        P_init, P_rec = sy.PS, sy.PS
    else:
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
    return kind_check, f"{system}_{axis}_{relax}{k:g}_{rate_name}", worst, diag


SMALL_CHI = 0.1 * np.pi
SMALL_FORMS = ("K", "P-same", "P-LH", "E-M1", "E-M2")


def small_pair_cases(rate_name="cry4a"):
    """I5 on the small pair N5 + N1 in the W_C geometry at one check direction: the solver against the dense
    Liouville solve (K at m = 1, P-same, P-LH and E on the ten levels) and against exact Liouville steps (K and P-LH
    at k = 0, 1e6, 1e7); the two-site solver on the small composite pair against the dense solve."""
    cases = []
    for relax, k in LEVELS:
        for f in SMALL_FORMS:
            cases.append(("dense", f, relax, k, rate_name))
            if f in ("K", "P-LH") and k in (0.0, 1e6, 1e7):
                cases.append(("steps", f, relax, k, rate_name))
    cases += [("tiny", cc) for cc in composite_cells()]
    return cases


def small_forms(sy):
    nb = axis_vector("W_C", "r")
    return {f: birth_operators(sy, f, SMALL_CHI, 1.0, nb) for f in SMALL_FORMS}


def small_system():
    if "W_C_small" not in _SYSTEMS:
        _SYSTEMS["W_C_small"] = system_structure("W_C", small=True)
    return _SYSTEMS["W_C_small"]


def _small_job(args):
    if args[0] == "tiny":
        cc = args[1]
        cs, rho0, k_rec, k_out, k_f = composite_setup(cc, tiny=True)
        n0 = check_dirs()[0]
        s = composite_solve(cs, n0, rho0, k_rec, k_out, k_f)
        return "tiny", cc.name, abs(s[0] - composite_dense(cs, n0, rho0, k_rec, k_out, k_f)), solve_diag(*s[:4])
    kind_check, f, relax, k, rate_name = args
    sy = small_system()
    n0 = check_dirs()[0]
    Pi, Pr = small_forms(sy)[f]
    k_f, k_r = rates(rate_name)
    s = solve(sy, n0, Pi, Pr, relax, k, k_f, k_r)
    ref = (dense_yield if kind_check == "dense" else exact_steps_yield)(sy, n0, Pi, Pr, relax, k, k_f, k_r)
    return kind_check, f"{f}_{relax}{k:g}", abs(s[0] - ref), solve_diag(*s)


I6_CELLS = [c for c in direct_cells(("main",)) if c.system == "S2" and c.relax == "none" and c.form != "K-inc"]


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
    if args[0] in ("dense", "steps", "tiny"):
        return ("small",) + _small_job(args)
    return ("I6",) + _i6_job(args[1:])


def checks(out, workers):
    """I3 on 40 directions and their antipodes, I5 on the small pair and the small composite, I6 against the frozen
    MAG-S3b code, in one pool (the slowest jobs first). I2 is read from the references in mode receptor."""
    from multiprocessing import get_context
    folder = os.path.join(out, "cells")
    i6, half_n = {}, None
    for c in I6_CELLS:
        with open(os.path.join(folder, c.name + ".json")) as fh:
            r = json.load(fh)
        half_n = int(r["grid_half"])
        i6[c.name] = dict(yp=np.array(r["yp"]), ym=np.array(r["ym"]))
    small = small_pair_cases()
    jobs = ([j for j in small if j[0] in ("steps", "tiny")] + [("I6", c, half_n) for c in I6_CELLS]
            + sorted(odd_check_jobs("cry4a"), key=lambda j: -j[4]) + [j for j in small if j[0] == "dense"])
    res = dict(I3={})
    found = dict(dense={}, steps={}, tiny={})
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    with get_context("fork").Pool(workers) as pool:
        for out_ in pool.imap_unordered(_check_job, jobs):
            if out_[0] == "odd":
                _, kind_check, key, worst, d = out_
                res[kind_check][key] = worst
                diag = diag_merge(diag, d)
            elif out_[0] == "small":
                _, kind_check, key, err, d = out_
                found[kind_check][key] = err
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
    i5 = dict(found, max_dense=max(found["dense"].values()), max_steps=max(found["steps"].values()),
              max_tiny=max(found["tiny"].values()))
    write_json(os.path.join(out, "checks.json"), dict(
        I1=dict(frozen_mismatches=frozen_mismatches(), frozen=FROZEN, pdb_sha256=PDB_SHA256), I3=res["I3"],
        I5_small=i5, I6=i6_out, diagnostics=diag, check_dirs=CHECK_DIRS, grid_half=half_n,
        rates_used={rn: list(rates(rn)) for rn in ("cry4a",) + E12_RATES}, hop_scale=HOP_SCALE[0],
        metadata=metadata("checks")))


# --------------------------------------------------------------------------------------------------------------
# Mode curves: branch E9
# --------------------------------------------------------------------------------------------------------------
CURVE_LEVELS = KEY_LEVELS + (("std", 3e7),)
CURVE_CASES = tuple((g, rl, k) for g in ("W_C", "W_D") for rl, k in CURVE_LEVELS)
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
# Mode receptor: rebuilt cells, harmonics, receptor, E4, maps E5 and E14, circle of H2, I2, I8, I9, I11
# --------------------------------------------------------------------------------------------------------------
E4_LEVELS = KEY_LEVELS + (("std", 3e7),)
E4_CELLS = {c.name for c in direct_cells(("main",)) + rebuilt_cells(("main",))
            if c.system in ("W_C", "W_D") and c.form == "K" and c.chi == 0.1 and c.m == 1.0
            and (c.relax, c.k) in E4_LEVELS}
E5_CHI = (0.05, 0.1)
E5_NAMED = ("W_A", "W_B", "W_C", "W_D")      # CISS axes along the vectors from the flavin to the four tryptophans
Y_XYZ = np.sqrt(105.0 / (4.0 * np.pi))       # Y_xyz = sqrt(105 / 4 pi) x y z, largest |x y z| = 1 / (3 sqrt 3)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def summarize(half, yp, ym, e4=False):
    """Pair and receptor quantities of one cell with the frozen functions of MAG-S3b: fit of degree 10, check fit of
    degree 24 (Q1), receptor map over p and S (F = eps_eff at S = 1, p = 1; F_amp = eps_E6 at S = 1, p = 1), I5 of
    MAG-S3b (here I7), E4 at p = 0, and the coefficients of degree 10 for the maps."""
    fit = b.harmonic_fit(half, yp, ym, b.L_FIT)
    hi = b.harmonic_fit(half, yp, ym, b.L_ODD_MAP)
    zon = b.zonal(fit)
    d = b.dipole(fit)
    row = dict(parts=b.parity_parts(yp, ym), fit_L=b.L_FIT, fit_even_rms_over_sd=fit["even_rms_over_sd"],
               fit_odd_rms_over_sd=fit["odd_rms_over_sd"], d_odd=d.tolist(), zonal=zon.tolist(),
               receptor=b.receptor_order_map(zon), I7=b.i5_receptor(zon), odd_power_shares=b.odd_power_shares(fit),
               fit_check=dict(L=b.L_ODD_MAP, even_rms_over_sd=hi["even_rms_over_sd"],
                              odd_rms_over_sd=hi["odd_rms_over_sd"], d_odd=b.dipole(hi).tolist()),
               coef=fit["coef"].tolist())
    if e4:
        row["E4"] = b.e4_cell(half, hi["coef"], b.L_ODD_MAP, d, zon)
    return row


def zonal_range_many(A, chunk=48):
    """b.zonal_range of every row of A (N, L+1), the same arithmetic on the same cos(theta) grid."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ae = np.where(np.arange(A.shape[1]) % 2 == 0, A, 0.0)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        f = b.npleg.legval(b.X_FINE, ae[i:i + chunk].T)
        out[i:i + chunk] = f.max(axis=1) - f.min(axis=1)
    return out


def _lm(L=None):
    return b.sh_degrees(b.L_FIT if L is None else L)


def coef_zonal(C):
    """Zonal coefficients of P_l about the molecular z axis for coefficient rows C (N, ncoef), as b.zonal."""
    l, m = _lm()
    C = np.atleast_2d(C)
    return np.stack([C[:, (l == k) & (m == 0)][:, 0] * np.sqrt((2 * k + 1) / (4.0 * np.pi))
                     for k in range(b.L_FIT + 1)], axis=1)


def coef_dipole(C):
    l, m = _lm()
    C = np.atleast_2d(C)
    get = lambda mm: C[:, (l == 1) & (m == mm)][:, 0]  # noqa: E731
    return np.sqrt(3.0 / (4.0 * np.pi)) * np.stack([-get(1), -get(-1), get(0)], axis=1)


def responses(zon):
    """F = R_REN eps_rec and F_amp = c_1 / b at S = 1, p = 1 for zonal rows (N, L+1)."""
    zon = np.atleast_2d(zon)
    return b.R_REN * zon[:, 1] / zonal_range_many(zon), zon[:, 1] / m3.REN_B


_FIT_PINV = {}


def fit_coefficients(half, YP, YM, L=None):
    """Coefficients of the harmonic fit of b.harmonic_fit (same harmonics, even degrees to the even part and odd
    degrees to the odd part on the closed grid) for the columns of YP and YM (n, K) at once, without the residual
    ratios of b.harmonic_fit, so that constant or vanishing columns pass."""
    L = b.L_FIT if L is None else L
    key = (len(half), float(half[0, 0]), float(half[-1, 2]), L)
    if key not in _FIT_PINV:
        B = m2.real_sh(L, np.vstack([half, -half]))[0]
        l, _ = b.sh_degrees(L)
        _FIT_PINV[key] = (l, np.linalg.pinv(B[:, l % 2 == 0]), np.linalg.pinv(B[:, l % 2 == 1]))
    l, Pe, Po = _FIT_PINV[key]
    YP, YM = np.atleast_2d(YP.T).T, np.atleast_2d(YM.T).T
    even = np.vstack([(YP + YM) / 2] * 2)
    odd = np.vstack([(YP - YM) / 2, (YM - YP) / 2])
    coef = np.zeros((len(l), YP.shape[1]))
    coef[l % 2 == 0], coef[l % 2 == 1] = Pe @ even, Po @ odd
    return coef.T


def ref_fits(half, cp, cm, k_r):
    """Degree-10 coefficients of the constant 1 and of the 16 Pauli components -k_R c_m(-n): the coefficients of
    a rebuilt yield are const + w @ comp."""
    one = np.ones((len(half), 1))
    const = fit_coefficients(half, one, one)[0]
    comp = fit_coefficients(half, -k_r * cm, -k_r * cp)
    return const, comp


def e5_axes(system, half_n):
    """CISS axes of map E5: the closed grid of the fields, and for W_C and W_D the vectors to the four tryptophans."""
    half = m2.fibonacci_sphere(half_n)
    G = np.vstack([half, -half])
    if system == "S2":
        return G, ()
    g = geo()
    return np.vstack([G] + [np.asarray(g[t]["r_hat"])[None, :] for t in E5_NAMED]), E5_NAMED


def e5_map(ref, sy, half, cp, cm, k_r, const, comp):
    """Map E5: K at m = 1 and K-inc for every CISS axis of the grid (and the named axes) at chi_b of E5_CHI."""
    AX, named = e5_axes(ref.system, len(half))
    W10 = basis_weights(sy)
    out = dict(axes=AX)
    for chi in E5_CHI:
        wk, wi = split_weights(W10, chi * np.pi, AX)
        ck, ci = const[None, :] + wk @ comp, const[None, :] + wi @ comp
        zk = coef_zonal(ck)
        F, Fa = responses(zk)
        yp, ym = 1.0 - k_r * (wk @ cm.T), 1.0 - k_r * (wk @ cp.T)
        even, odd = (yp + ym) / 2, (yp - ym) / 2
        er = even.max(axis=1) - even.min(axis=1)
        tag = f"chi{chi:g}"
        out.update({f"{tag}_F": F, f"{tag}_F_amp": Fa, f"{tag}_eps_mol": np.abs(odd).max(axis=1) / er,
                    f"{tag}_even_range": er, f"{tag}_d_K": coef_dipole(ck), f"{tag}_d_inc": coef_dipole(ci),
                    f"{tag}_F_amp_inc": coef_zonal(ci)[:, 1] / m3.REN_B})
    return out, list(named)


def h2_circle(sy, const, comp):
    """d_odd . z of K at m = 1 on the circle of H2_CIRCLE CISS axes across the receptor axis and at the axis z."""
    phi = 2.0 * np.pi * np.arange(H2_CIRCLE) / H2_CIRCLE
    AX = np.vstack([np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1), [[0.0, 0.0, 1.0]]])
    W10 = basis_weights(sy)
    out = {}
    for chi in E5_CHI:
        wk, _ = split_weights(W10, chi * np.pi, AX)
        dz = coef_dipole(const[None, :] + wk @ comp)[:, 2]
        out[f"chi{chi:g}"] = dict(circle_dz=dz[:-1].tolist(), z_axis_dz=float(dz[-1]),
                                  median_ratio=float(np.median(np.abs(dz[:-1])) / abs(dz[-1])) if dz[-1] != 0 else None)
    return out


def _ref_job(args):
    """I2 and I8 of one reference, its map E5 and, for W_C and W_D on the key levels, the circle of H2."""
    ref, out, own, direct_names = args
    r = load_json(item_path(out, ref))
    yp, ym, cp, cm = (np.asarray(r[k], dtype=float) for k in ("yp", "ym", "cp", "cm"))
    k_r = float(r["rates_used"][1])
    half_n = len(yp)
    half = m2.fibonacci_sphere(half_n)
    sy = build_system(ref.system)
    info = dict(ref=asdict(ref), name=ref.name, grid_half=half_n, diagnostics=r["diagnostics"],
                rates_used=r["rates_used"], code_ok=r.get("mag_s3c_sha256") == own,
                I2=float(np.abs(yp - ym).max() / 2), parts=b.parity_parts(yp, ym), I8={})
    rp, rm = rebuilt_yields(pauli_weights(sy, sy.PS), cp, cm, k_r)
    info["self_rebuild"] = float(max(np.abs(rp - yp).max(), np.abs(rm - ym).max()))
    if ref.branch == "main" and (ref.relax, ref.k) in LEVELS_OLD:
        for c in direct_cells(("main",)):
            if c.name not in direct_names:
                continue
            if (c.system, c.relax, c.k) == (ref.system, ref.relax, ref.k) and c.form in ("K", "K-inc") and c.m == 1.0:
                d = load_json(item_path(out, c))
                rp, rm = rebuilt_yields(cell_weights(sy, c), cp, cm, k_r)
                info["I8"][c.name] = float(max(np.abs(rp - np.asarray(d["yp"])).max(),
                                               np.abs(rm - np.asarray(d["ym"])).max()))
    maps = {}
    const, comp = ref_fits(half, cp, cm, k_r)
    if ref.system in GEOMETRIES or ref.system in VARIANTS_E3 + VARIANTS_E7:
        W10 = basis_weights(sy)
        axes = system_axes(ref.system)
        AX = np.array([axis_vector(ref.system, ax) for ax in axes])
        info["main_axes"] = {}
        for chi in E5_CHI:
            wk, wi = split_weights(W10, chi * np.pi, AX)
            dk, di = coef_dipole(const[None, :] + wk @ comp), coef_dipole(const[None, :] + wi @ comp)
            for j, ax in enumerate(axes):
                info["main_axes"][f"{ax}_chi{chi:g}"] = dict(d_K=dk[j].tolist(), d_inc=di[j].tolist())
    if ref in e5_references():
        maps, info["E5_named_axes"] = e5_map(ref, sy, half, cp, cm, k_r, const, comp)
        if ref.system in ("W_C", "W_D") and ref.branch == "main" and (ref.relax, ref.k) in KEY_LEVELS:
            info["H2_circle"] = h2_circle(sy, const, comp)
    return ref.name, info, maps


def _receptor_job(args):
    kind, c, out, own = args
    if kind == "rebuilt":
        r = load_json(item_path(out, ref_of(c)))
        sy = build_system(c.system)
        yp, ym = rebuilt_yields(cell_weights(sy, c), np.asarray(r["cp"]), np.asarray(r["cm"]),
                                float(r["rates_used"][1]))
    else:
        r = load_json(os.path.join(out, "composite", c.name + ".json") if kind == "composite" else item_path(out, c))
        yp, ym = np.asarray(r["yp"]), np.asarray(r["ym"])
    h = len(yp)
    row = summarize(m2.fibonacci_sphere(h), yp, ym, e4=c.name in E4_CELLS)
    row.update(cell=asdict(c), name=c.name, kind=kind, grid_half=h, diagnostics=r["diagnostics"],
               rates_used=r["rates_used"], code_ok=r.get("mag_s3c_sha256") == own)
    if kind == "composite":
        row["hop_rates_used"] = r["hop_rates_used"]
        for key in ("rec_C", "rec_D"):
            p, m = np.asarray(r[key + "_p"]), np.asarray(r[key + "_m"])
            row[key] = dict(mean=float((p + m).mean() / 2), odd_half_range=float(np.abs(p - m).max() / 2))
    return row


# crystal 6PU0, space group P2_1 2_1 2_1: the four molecules of the cell are related by 2-fold rotations about the
# cell axes a, b, c, which are the PDB axes x, y, z (right angles, C4); in the flavin frame they are the columns of
# the frame matrix
def cell_axes():
    return np.asarray(geo()["frame"], dtype=float).T          # columns: a, b, c in the flavin frame


def crystal_rotations():
    E = cell_axes()
    return [np.eye(3)] + [2.0 * np.outer(E[:, i], E[:, i]) - np.eye(3) for i in range(3)]


def crystal_basis(dirs):
    """Harmonics of degree 10 of the crystal average: f_cr(n) = 1/4 sum_k f(R_k n) (R_k symmetric)."""
    return sum(m2.real_sh(b.L_FIT, np.asarray(dirs) @ R)[0] for R in crystal_rotations()) / 4.0


def crystal_setup(half_n):
    half = m2.fibonacci_sphere(half_n)
    G = np.vstack([half, -half])
    E = cell_axes()
    phi = 2.0 * np.pi * np.arange(360) / 360
    planes = [np.cos(phi)[:, None] * E[:, i][None, :] + np.sin(phi)[:, None] * E[:, j][None, :]
              for i, j in ((0, 1), (1, 2), (2, 0))]
    P = np.vstack(planes)
    mirrors = [np.eye(3) - 2.0 * np.outer(E[:, i], E[:, i]) for i in range(3)]
    Bg = m2.real_sh(b.L_FIT, G)[0]
    xyz = Y_XYZ * (G @ E[:, 0]) * (G @ E[:, 1]) * (G @ E[:, 2])
    coef_xyz = np.linalg.lstsq(Bg, xyz, rcond=None)[0]
    return dict(n=len(half), G=crystal_basis(G), P=crystal_basis(P), Pm=crystal_basis(-P),
                M=[crystal_basis(G @ Mi) for Mi in mirrors], coef_xyz=coef_xyz)


def crystal_cell(cs, coef):
    """Crystal pattern of one cell: even range, odd part, its dipole (zero for 222), the amplitude of the xyz pattern
    and the I11 quantities (odd part on fields in the three cell planes; residual of the sign change at the mirrors).
    The odd quantities use the odd degrees only, so that rounding of the constant and the even part stays out."""
    c = np.asarray(coef, dtype=float)
    l, m = _lm()
    ce, co = np.where(l % 2 == 0, c, 0.0), np.where(l % 2 == 1, c, 0.0)
    h = cs["n"]
    even = (cs["G"] @ ce)[:h]
    fo = cs["G"] @ co
    odd = (fo[:h] - fo[h:]) / 2
    er = float(even.max() - even.min())
    plane_odd = (cs["P"] @ co - cs["Pm"] @ co) / 2
    mirror_res = 0.0
    for Bm in cs["M"]:
        fm = Bm @ co
        mirror_res = max(mirror_res, float(np.abs((fm[:h] - fm[h:]) / 2 + odd).max()))
    a_xyz = float(c @ cs["coef_xyz"])
    amp = abs(a_xyz) * Y_XYZ / (3.0 * np.sqrt(3.0))
    d = coef_dipole(c[None, :])[0]
    d_cr = sum(R @ d for R in crystal_rotations()) / 4.0
    return dict(even_range=er, odd_max=float(np.abs(odd).max()), plane_odd_max=float(np.abs(plane_odd).max()),
                mirror_residual=mirror_res, a_xyz=a_xyz, xyz_amplitude=amp,
                xyz_amplitude_over_even_range=amp / er if er > 0 else None, dipole_norm=float(np.linalg.norm(d_cr)),
                molecule_dipole_norm=float(np.linalg.norm(d)))


def receptor_axis_sweep(coef, U_basis, blocks):
    """Map E14: F and F_amp of one cell for receptor axes u: a_l(u) = sum_m c_lm Y_lm(u)."""
    c = np.asarray(coef, dtype=float)
    M = U_basis * c[None, :]
    zon = np.stack([M[:, blk].sum(axis=1) for blk in blocks], axis=1)
    return responses(zon)


def _e14_job(args):
    name, coef = args
    U = _E14["U"]
    F, Fa = receptor_axis_sweep(coef, U, _E14["blocks"])
    return name, F, Fa


_E14 = {}


def receptor(out, workers=1, items=None):
    """items: dict(direct=, rebuilt=, composite=, refs=) with subsets for the tests."""
    from multiprocessing import get_context
    own = file_sha256(os.path.abspath(__file__))
    items = items or dict(direct=direct_cells(), rebuilt=rebuilt_cells(), composite=composite_cells(),
                          refs=references())
    jobs = ([("direct", c, out, own) for c in items["direct"]] + [("rebuilt", c, out, own) for c in items["rebuilt"]]
            + [("composite", c, out, own) for c in items["composite"]])
    direct_names = {c.name for c in items["direct"]}
    half_n = int(load_json(item_path(out, items["refs"][0]))["grid_half"])
    half = m2.fibonacci_sphere(half_n)
    G = np.vstack([half, -half])
    l, m = _lm()
    _E14.update(U=m2.real_sh(b.L_FIT, G)[0], blocks=[l == k for k in range(b.L_FIT + 1)])
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(_receptor_job, jobs, chunksize=2)
        refs = pool.map(_ref_job, [(r, out, own, direct_names) for r in items["refs"]], chunksize=1)
        sweep_rows = [r for r in rows if r["kind"] != "composite" and r["cell"]["form"] == "K"
                      and r["cell"]["m"] == 1.0]
        sweeps = pool.map(_e14_job, [(r["name"], r["coef"]) for r in sweep_rows], chunksize=4)
    halves = {r["grid_half"] for r in rows if r["kind"] != "composite"} | {i["grid_half"] for _, i, _ in refs}
    if len(halves) != 1:
        sys.exit("cells and references were computed on different grids")
    maps_e5 = {f"{name}__{k}": v for name, _, mp in refs for k, v in mp.items()}
    np.savez_compressed(os.path.join(out, "maps_E5.npz"), **maps_e5)
    np.savez_compressed(os.path.join(out, "maps_E14.npz"), axes=G,
                        **{f"{n}__F": F for n, F, _ in sweeps}, **{f"{n}__F_amp": Fa for n, _, Fa in sweeps})
    crystal, setups = {}, {}
    for r in rows:
        if r["kind"] == "composite" or r["cell"]["system"][:3] in ("W_C", "W_D"):
            h = r["grid_half"]
            if h not in setups:
                setups[h] = crystal_setup(h)
            crystal[r["name"]] = crystal_cell(setups[h], r["coef"])
    g = geometry_report()
    write_json(os.path.join(out, "receptor.json"), dict(
        cells=rows, refs={name: info for name, info, _ in refs}, crystal=crystal, grid_half=next(iter(halves)),
        e13_grid_half=sorted({r["grid_half"] for r in rows if r["kind"] == "composite"}), geometry=g, R_REN=b.R_REN,
        REN_B=m3.REN_B, cell_axes=cell_axes().tolist(),
        order={f"S{S:g}": dict(g=b.g_order(S), D=b.depth(S)) for S in b.S_LEVELS}, metadata=metadata("receptor")))
    return rows


# --------------------------------------------------------------------------------------------------------------
# Calibration C1-C6
# --------------------------------------------------------------------------------------------------------------
C1_CELLS = (Cell("W_C", "r", "rf", 1e6, "K", 0.1, 1.0, "mags2", "C1"),
            Cell("W_C", "r", "rf", 1e6, "P-same", 0.1, 1.0, "mags2", "C1"),
            Cell("W_C", "r", "std", 1e8, "K", 0.1, 1.0, "blind_cry4a", "C1"))
PROTOCOL_GEOMETRY = dict(W_C=dict(dist=17.61, normal_to_z_deg=50.0, r_hat_to_z_deg=72.4),
                         W_D=dict(dist=21.27, normal_to_z_deg=81.6, r_hat_to_z_deg=78.8),
                         W_A=dict(r_hat_to_z_deg=89.3), W_B=dict(r_hat_to_z_deg=57.3),
                         S2=dict(N5_to_z_deg=9.4, N10_to_z_deg=9.4, N1_to_z_deg=37.0, N5_vs_N1_deg=46.3),
                         listing_N1_vs_W318_after_z_sign_deg=3.9, listing_N1_vs_W318_deg=77.6)


def blind_hops(cc):
    """(k_rec, k_out) of a composite cell at the substitute rates of the blind check, whatever the mode."""
    k_f, k_r = rates("blind_cry4a")
    return k_f, (k_r, k_r if cc.rec == "CD" else 0.0), (BLIND_FACTOR * cc.k_cd, BLIND_FACTOR * cc.k_dc)


def _chunk_job(args):
    cell, dirs = args
    return cell_yields(cell, dirs)


def _composite_check_job(args):
    """Diagnostics of the composite pair without CISS at the substitute rates on the check directions (C3)."""
    cc, dirs = args
    cs = build_composite()
    k_f, k_rec, k_out = blind_hops(cc)
    rho0 = cs.birth(cs.sites[0].PS)
    diag = dict(max_rel_residual=0.0, max_conservation_error=0.0, max_gmres_iterations=0)
    t0 = time.perf_counter()
    for n in dirs:
        diag = diag_merge(diag, solve_diag(*composite_solve(cs, n, rho0, k_rec, k_out, k_f)[:4]))
    return cc.name, diag, (time.perf_counter() - t0) / len(dirs)


def c1_c2_c3(workers, out_dir):
    """C1: doubling the grid changes eps_mol, F, F_amp and d_odd . z by less than 1%; C2: degree 12 changes eps_rec by
    less than 1%; C3: residuals and |Phi + Phi_R - 1| at least ten times below the thresholds of I5 (1e-10) and I4
    (1e-10) in the C1 cells, on 40 directions of each relaxation level without CISS and for the composite pair
    without CISS on 40 directions at every hop rate, GMRES within its restarts. MAG-S2 rates and the substitute rates
    only; only relative changes, margins and times are written, the yields stay in C1_yields.npz."""
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
            F = lambda s, k: s["receptor"]["by_order"]["S1_p1"][k]  # noqa: E731
            out[cell.name] = dict(
                C1_eps_mol_rel_change=float(s2["parts"]["eps_mol"] / s1["parts"]["eps_mol"] - 1.0),
                C1_F_rel_change=float(F(s2, "eps_eff") / F(s1, "eps_eff") - 1.0),
                C1_F_amp_rel_change=float(F(s2, "eps_E6") / F(s1, "eps_E6") - 1.0),
                C1_d_odd_z_rel_change=float(s2["d_odd"][2] / s1["d_odd"][2] - 1.0),
                C2_eps_rec_rel_change=float(row["eps_rec_L12"] / s1["receptor"]["eps_rec"] - 1.0))
        jobs = [("I2", "W_C", "-", rl, k, "mags2") for rl, k in LEVELS]
        for _, _, _, d in pool.imap_unordered(_odd_job, jobs):
            diag = diag_merge(diag, d)
        comp = {}
        cdirs = check_dirs()
        cjobs = [(cc, part) for cc in composite_cells() if cc.form == "none" for part in np.array_split(cdirs, 4)]
        for name, d, sec in pool.imap_unordered(_composite_check_job, cjobs):
            r = comp.setdefault(name, dict(diagnostics=dict(max_rel_residual=0.0, max_conservation_error=0.0,
                                                            max_gmres_iterations=0), seconds_per_direction=[]))
            r["diagnostics"] = diag_merge(r["diagnostics"], d)
            r["seconds_per_direction"].append(sec)
        for r in comp.values():
            r["seconds_per_direction"] = float(np.mean(r["seconds_per_direction"]))
            r["max_gmres_restarts"] = -(-r["diagnostics"]["max_gmres_iterations"] // COMPOSITE_RESTART)
            diag_c = r["diagnostics"]
            diag = dict(diag, max_rel_residual=_worst(diag["max_rel_residual"], diag_c["max_rel_residual"]),
                        max_conservation_error=_worst(diag["max_conservation_error"],
                                                      diag_c["max_conservation_error"]))
    keys = ("C1_eps_mol_rel_change", "C1_F_rel_change", "C1_F_amp_rel_change", "C1_d_odd_z_rel_change")
    c1 = all(abs(v[key]) < 0.01 for v in out.values() for key in keys)
    c2 = all(abs(v["C2_eps_rec_rel_change"]) < 0.01 for v in out.values())
    # solve() and composite_solve() raise when GMRES does not converge within its restarts, so that part holds by
    # construction; the numbers of restarts are reported
    restarts = -(-diag["max_gmres_iterations"] // GMRES_RESTART)
    c3 = (diag["max_rel_residual"] < 1e-11 and diag["max_conservation_error"] < 1e-11 and restarts <= GMRES_MAXITER
          and all(r["max_gmres_restarts"] <= COMPOSITE_MAXITER for r in comp.values()))
    return dict(C1=dict(cells=out, criterion_met=bool(c1)), C2=dict(criterion_met=bool(c2)),
                C3=dict(diagnostics=dict(diag, max_gmres_restarts=restarts), composite=comp, criterion_met=bool(c3)))


def cryst1(path=None):
    with open(path or os.path.join(HERE, PDB_FILE)) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                return dict(a=float(line[6:15]), b=float(line[15:24]), c=float(line[24:33]), alpha=float(line[33:40]),
                            beta=float(line[40:47]), gamma=float(line[47:54]), space_group=line[55:66].strip())
    return None


def c4_geometry():
    rep = geometry_report()
    dev = {}
    for t in ("W_A", "W_B", "W_C", "W_D", "S2"):
        for key, want in PROTOCOL_GEOMETRY[t].items():
            dev[f"{t}_{key}"] = abs(rep[t][key] - want)
    for key in ("listing_N1_vs_W318_after_z_sign_deg", "listing_N1_vs_W318_deg"):
        dev[key] = abs(rep[key] - PROTOCOL_GEOMETRY[key])
    cr = cryst1()
    cell_ok = (cr is not None and cr["space_group"] == "P 21 21 21"
               and all(abs(cr[k] - 90.0) < 1e-9 for k in ("alpha", "beta", "gamma")))
    ok = all((v <= 0.01 if k.endswith("_dist") else v <= 0.1) for k, v in dev.items()) and cell_ok
    return dict(report=rep, deviations=dev, cryst1=cr, cell_ok=bool(cell_ok), criterion_met=bool(ok))


def c5_rebuild():
    """C5 at the substitute rates: on the small pair the reconstruction from the reference against direct solves at
    the reversed field for K at m = 1, K-inc, K-iso and P-0 on the ten levels, and the split of K; on the small
    composite the two-site solver against the dense solve for both recombination schemes and all hop rates, and
    the odd part without CISS with recombination only at W_C."""
    sy = small_system()
    k_f, k_r = rates("blind_cry4a")
    nb = axis_vector("W_C", "r")
    W10 = basis_weights(sy)
    rebuild, split = 0.0, 0.0
    for relax, k in LEVELS:
        for n in check_dirs()[:2]:
            X = solve_X(sy, n, sy.PS, sy.PS, relax, k, k_f, k_r)[0]
            cn = pauli_coefficients(sy, X)
            for form, chi in (("K", 0.05), ("K", 0.1), ("K-inc", 0.1), ("K-iso", 0.1), ("P-0", 0.1)):
                P_init, _ = birth_operators(sy, form, chi * np.pi, 1.0, nb)
                f_rebuilt = 1.0 - k_r * (cn @ pauli_weights(sy, P_init))
                f_direct = solve(sy, -n, P_init, sy.PS, relax, k, k_f, k_r)[0]
                rebuild = max(rebuild, abs(f_rebuilt - f_direct))
            for chi in (0.05, 0.1):
                wk, wi = split_weights(W10, chi * np.pi, nb[None, :])
                for w, P in ((wk[0], sy.turned(chi * np.pi, nb)), (wi[0], sy.dephased(sy.turned(chi * np.pi, nb)))):
                    split = max(split, abs(k_r * cn @ (w - pauli_weights(sy, P))))
    tiny, odd = {}, 0.0
    for cc in composite_cells():
        cs = build_composite(tiny=True)
        k_f2, k_rec, k_out = blind_hops(cc)
        P = cs.sites[0].PS if cc.form == "none" else cs.sites[0].turned(0.1 * np.pi, nb)
        rho0 = cs.birth(P)
        n0 = check_dirs()[0]
        s = composite_solve(cs, n0, rho0, k_rec, k_out, k_f2)
        tiny[cc.name] = dict(error=abs(s[0] - composite_dense(cs, n0, rho0, k_rec, k_out, k_f2)), iterations=s[2],
                             rel_residual=s[3], conservation=abs(s[0] + s[1] - 1.0))
        if cc.form == "none" and cc.rec == "C":
            for n in check_dirs()[:4]:
                a = composite_solve(cs, n, rho0, k_rec, k_out, k_f2)[0]
                z = composite_solve(cs, -n, rho0, k_rec, k_out, k_f2)[0]
                odd = max(odd, abs(a - z) / 2)
    worst_tiny = max(v["error"] for v in tiny.values())
    ok = rebuild < 1e-12 and split < 1e-12 and worst_tiny < 1e-10 and odd < 1e-12
    return dict(rebuild_max_error=rebuild, split_max_error=split, tiny_composite=tiny,
                tiny_composite_max_error=worst_tiny, tiny_composite_odd_no_ciss_rec_C=odd, criterion_met=bool(ok))


def denton_eq7(theta, J, k2, de):
    """Eq. (7) of Denton et al. 2026 in the basis (S, T0), with N and A of Eqs. (8) and (9)."""
    N = de ** 2 + 4 * J ** 2 + k2 ** 2 + 4 * de * J * np.cos(2 * theta)
    A = de ** 2 + k2 ** 2 + 2 * de * J * np.cos(2 * theta)
    r = np.array([[np.cos(theta) ** 2 * (A + 2 * de * J + 4 * J ** 2), -0.5j * np.sin(2 * theta) * (A - 2j * J * k2)],
                  [0.5j * np.sin(2 * theta) * (A + 2j * J * k2), np.sin(theta) ** 2 * (A - 2 * de * J + 4 * J ** 2)]])
    return r / N, N, A


def formation(H, P, k2):
    """k2 (k2 + i[H, .])^-1 P: the state formed through the bridge, Eqs. (A7) and (A8) of Denton et al. 2026."""
    A = 1j * H + 0.5 * k2 * np.eye(len(H))
    return k2 * sl.solve_sylvester(A, A.conj().T, P)


def c6_denton():
    """C6: the formation equations of Denton et al. 2026 against their Eqs. (7)-(11) on the electron block (S, T0)
    at d_eps = 0 and d_eps != 0, and form E of the code against the formation equations in the full space of the
    small pair at d_eps = 0 in the sign of K. Eq. (11) is taken with the coherence operator as Eq. (3) writes it;
    the report also gives the operator S_Ax S_Dy - S_Ay S_Dx from the spin matrices in the same basis."""
    eq7 = eq10 = eq11 = 0.0
    S, T0 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    P_op = np.outer(S, T0) + np.outer(T0, S)                         # Eq. (2): S_Az - S_Dz
    C_op = -0.5j * (np.outer(S, T0) - np.outer(T0, S))               # Eq. (3) as written
    for theta in (0.05 * np.pi, 0.1 * np.pi, np.pi / 8, np.pi / 4):
        psi = np.cos(theta) * S + 1j * np.sin(theta) * T0
        Pp = np.outer(psi, psi.conj())
        for model in DENTON_K2:
            for de in (0.0, 0.3 * DENTON_J, -1.1 * DENTON_K2[model]):
                J, k2 = DENTON_J, DENTON_K2[model]
                rho = formation(de * Pp + 2 * J * np.diag([1.0, 0.0]), Pp, k2)
                ref, N, A = denton_eq7(theta, J, k2, de)
                eq7 = max(eq7, float(np.abs(rho - ref).max()))
                eq10 = max(eq10, abs(np.trace(P_op @ rho).real - (-2 * J * k2 / N * np.sin(2 * theta))))
                eq11 = max(eq11, abs(np.trace(C_op @ rho).real - A / (2 * N) * np.sin(2 * theta)))
    # S_Ax S_Dy - S_Ay S_Dx in the basis (S, T0) from the spin matrices of two electrons
    sx, sy_, sz = (np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.array([[1, 0], [0, -1]]) / 2)
    up_dn, dn_up = np.kron([1, 0], [0, 1]), np.kron([0, 1], [1, 0])
    B = np.stack([(up_dn - dn_up) / np.sqrt(2), (up_dn + dn_up) / np.sqrt(2)], axis=1)
    O = np.kron(sx, sy_) - np.kron(sy_, sx)
    O_st = B.conj().T @ O @ B
    sign_eq3 = float(np.real(O_st[0, 1] / C_op[0, 1]))
    # form E of the code in the full space of the small pair against the formation equations (sign of K)
    sy = small_system()
    bridge = 0.0
    for axis in ("z", "x", "r"):
        n = axis_vector("W_C", axis)
        for model in DENTON_K2:
            for th in (0.05, 0.1):
                rho_k = sy.turned(th * np.pi, n)
                formed = formation(2 * DENTON_J * sy.PS, rho_k, DENTON_K2[model])
                bridge = max(bridge, float(np.abs(sy.bridge(th * np.pi, n, eta(model)) - formed).max()))
    ok = eq7 < 1e-12 and eq10 < 1e-12 and eq11 < 1e-12 and bridge < 1e-12
    return dict(eq7_max_error=eq7, eq10_max_error=eq10, eq11_max_error=eq11, form_E_vs_formation=bridge,
                eq3_operator_over_spin_matrices=sign_eq3, eta={m: [eta(m).real, eta(m).imag] for m in DENTON_K2},
                criterion_met=bool(ok))


def calibrate(out, checks_wanted, workers):
    os.makedirs(out, exist_ok=True)
    meta = metadata("calibrate")
    res = {}
    t0 = time.perf_counter()
    if any(c in checks_wanted for c in ("C1", "C2", "C3")):
        res.update(c1_c2_c3(workers, out))
    if "C4" in checks_wanted:
        res["C4"] = c4_geometry()
    if "C5" in checks_wanted:
        res["C5"] = c5_rebuild()
    if "C6" in checks_wanted:
        res["C6"] = c6_denton()
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
                frozen=FROZEN, pdb_sha256=PDB_SHA256, own=own, files=files, rates=dict(b.RATES),
                s3c_rates=dict(S3C_RATES), hop_scale=HOP_SCALE[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("pair", "composite", "checks", "curves", "receptor", "calibrate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--grid-half", type=int, default=None,
                    help="pair and composite: smaller grids only for the smoke and blind checks")
    ap.add_argument("--t-max", type=float, default=CURVE_T, help="curves: shorter curves only for the smoke check")
    ap.add_argument("--checks", default="C1,C2,C3,C4,C5,C6")
    ap.add_argument("--blind-rates", action="store_true", help="the substitute rates of the blind check")
    ap.add_argument("--confirmatory", action="store_true",
                    help="the confirmatory run at the Cry4a rates (set by scripts/pipeline.sh after the confirmation)")
    a = ap.parse_args()
    if a.blind_rates and a.confirmatory:
        sys.exit("--blind-rates and --confirmatory exclude each other")
    if a.mode in ("pair", "composite", "checks", "curves") and not (a.blind_rates or a.confirmatory):
        sys.exit("pair, composite, checks and curves compute at the rates of the hypotheses only in the confirmatory run "
                 "(--confirmatory, set by scripts/pipeline.sh after the answer «запускай»); before the freeze use "
                 "--blind-rates")
    if a.blind_rates:
        use_blind_rates()
    if a.mode == "pair":
        if not pair(a.out, a.workers, a.max_hours, a.grid_half or GRID_HALF):
            sys.exit(3)
    elif a.mode == "composite":
        if not composite(a.out, a.workers, a.max_hours, a.grid_half or E13_GRID_HALF):
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
