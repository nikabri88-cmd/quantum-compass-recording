#!/usr/bin/env python3
"""
MAG-S3b - non-reciprocity of CISS, order of cryptochromes and the inverse problem.   Protocol version 0.4 (draft).

Pair. The pair of MAG-S2 v0.1.2 (flavin anion with N5 and N10, tryptophan cation with N1, anisotropic tensors from
RadicalPy 1.0.9, 50 uT, J = D = 0) with the kinetics of the MAG-S3 anchor: recombination through P_rec at rate k_R
and a spin-independent forward reaction at rate k_F from every state. The receptor signal is the forward yield,
Phi = 1 - Phi_R. CISS enters in two forms with separate parameters for birth and recombination:

  K       birth U_b P_S U_b^+, U_b = exp(-i chi_b n_b . sigma_A); recombination U_r P_S U_r^+ with chi_r and n_r
  P-same  birth and recombination through cos(chi/2)|S> + sin(chi/2)|T0>, T0 quantised along n_b (MAG-S3 anchor)
  P-LH    birth as P-same, recombination through cos(chi/2)|S> - sin(chi/2)|T0> (Luo & Hore 2021)

sigma_A = 2 S_A are the Pauli matrices of electron A, the flavin electron (particle 0, as in the MAG-S3 anchor).
n_b = (sin beta, 0, cos beta) in the molecular frame; for K, chi_r = chi_b (1 - m) and n_r is n_b turned by alpha
in the xz plane. |T0 along n> = (n . sigma_A)|S>, so the P state is (cos(chi/2) + sin(chi/2) n . sigma_A)|S>.

Receptor. With a free azimuth around the receptor axis only the m = 0 harmonics of the molecular pattern remain.
Axial order S = <P2> (von Mises - Fisher distribution of the molecular z axis) multiplies degree l by
<P_l>_kappa = I_{l+1/2}(kappa) / I_{1/2}(kappa); polar order p multiplies the odd degrees by p, because a share
(1 - p)/2 of the molecules is turned by 180 degrees about an in-plane axis (theta -> pi - theta).

Transfer to the neural model of MAG-S3. eps_rec is the coefficient of cos(theta) of the receptor at S = 1 divided by
the range of its even part. The Ren pattern of MAG-S3 has the even range R_REN b (R_REN = 1 - exp(-pi^2 / 8c^2) =
0.9975), so the ratio is carried over exactly by eps_eff = p g(S) R_REN eps_rec with g(S) = <P1>_kappa / D(S), where
D(S) is the share of the depth of the Ren dip left at order S. At S < 1 the Ren pattern is expanded in Legendre
polynomials up to degree 200 and degree l is multiplied by <P_l>_kappa; the odd admixture is eps_eff D(S) cos(theta),
so eps_eff keeps its meaning at every S. (Calibration round 1: at degree 60 the residual was 3.1e-4 b at the poles,
where the Ren pattern has a cone because theta - 90 deg = -arcsin(cos theta) is not smooth on the sphere; the residual
falls as 1/L there and is 9.5e-5 b at degree 200. At S < 1 the series of degree 60 and 300 are bitwise equal.)

Inverse problem. A virtual bird gets eps_eff and passes the frozen MAG-S3 recording without changes (retina,
neurons, schedule, drift, shared gain noise, counts). Records and the truth file are written separately;
estimate.py reads only the records.

Modes
  physics    the pair grid (225 cells on the closed grid of 2000 directions) and the invariants I2-I4, I6 and I8
             (resumable; exit code 3 when it stops at the time limit)
  receptor   harmonics, receptor maps, D(S), g(S) and the E4 quantities from the stored pair grid, no new solutions
  e3geo      the calibration averages of E3 (birds 7001-7100) into the records folder; they carry no truth
  generate   records of the inverse problem in one folder and the truth in another
  e4         E4 records at fixed azimuth (R1, birds of eps_eff = 0) and their dipole tests, after receptor
  calibrate  C1-C5 (C1 on seeds 6001-7000, C3 on 1001-1010)

Usage
  python3 -I mag_s3b.py physics --out PAIR [--workers 2] [--max-hours 1]
  python3 -I mag_s3b.py receptor --out PAIR
  python3 -I mag_s3b.py e3geo --records DIR
  python3 -I mag_s3b.py generate --records DIR --truth DIR [--workers 2]
  python3 -I mag_s3b.py e4 --out PAIR [--workers 2]
  python3 -I mag_s3b.py calibrate --out OUT [--checks C1,C2,C3,C4,C5] [--workers 2]
scripts/pipeline.sh runs them in order; --seeds and --grid-half serve only its smoke and blind checks.
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
from dataclasses import asdict, dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial import legendre as npleg  # noqa: E402
from scipy import optimize, special  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_VERSION = "0.4"

# Frozen files of MAG-S3 v0.1.0 (DOI 10.5281/zenodo.24043500), which carry MAG-S2 v0.1.2; invariant I1.
FROZEN = {
    "mag_s3.py": "1df703af63a2af71a94385d33276ad39a422e9abaa040fdf4bcb618344e1f350",
    "analyze.py": "ecc6a15b77e4f4d1c07a4217439e0c97868b9a12971c9ada255f79840ef1d21f",
    "mag_s2.py": "2352aefe0fe5974732c318b365f4fb5477ffc47fb78b12748bca2515465829c4",
    "physics.json": "647bc0ccb1312991a1b772193cac39f4ced1481c591422c50659bb822a3100c5",
}
OWN_FILES = ("mag_s3b.py", "estimate.py", "analyze_s3b.py", "test_s3b.py")


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def frozen_mismatches():
    out = []
    for name, digest in FROZEN.items():
        path = os.path.join(HERE, name)
        if not os.path.exists(path) or file_sha256(path) != digest:
            out.append(name)
    return out


def load_mag_s3():
    if "mag_s3" in sys.modules:
        return sys.modules["mag_s3"]
    bad = frozen_mismatches()
    if bad:
        sys.exit("frozen files differ from MAG-S3 v0.1.0: " + ", ".join(bad))
    spec = importlib.util.spec_from_file_location("mag_s3", os.path.join(HERE, "mag_s3.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3"] = mod
    spec.loader.exec_module(mod)
    return mod


m3 = load_mag_s3()
m2, P2, rng_for = m3.m2, m3.P2, m3.rng_for


def write_json(path, obj):
    m3.write_json_atomic(path, obj, indent=1)


# --------------------------------------------------------------------------------------------------------------
# Grids of the protocol
# --------------------------------------------------------------------------------------------------------------
RATES = {"mags2": (1.0e6, 1.0e6), "cry4a": (5.0e6, 3.4e5), "cry4a_fast": (1.0e7, 3.4e5)}   # (k_F, k_R), s^-1
assert tuple(RATES.values()) == m3.ANCHOR_RATES
# The blind check of the pipeline computes the grid at rates the protocol uses nowhere, at the same cost, so that it
# never holds a number of H1-H4 or Q1-Q4 before the run.
BLIND_RATES = {"mags2": (2.0e6, 2.0e6), "cry4a": (6.0e6, 5.0e5), "cry4a_fast": (8.0e6, 5.0e5)}


def use_blind_rates():
    RATES.update(BLIND_RATES)
FORMS = ("K", "P-same", "P-LH")
CHI_LEVELS = (0.02, 0.05, 0.1)          # chi_b of K and chi of P, units of pi
M_LEVELS = (0.1, 0.25, 0.5, 1.0)        # share of divergence of K, chi_r = chi_b (1 - m)
ALPHA_LEVELS = (30.0, 90.0)             # angle between n_b and n_r of K at m = 0, degrees
BETA_LEVELS = (0.0, 45.0, 90.0)         # tilt of the CISS axis from the molecular z axis, degrees
BETA_K_EXTRA = 80.0                     # one more tilt for K at m = 1 (H4)
P_LEVELS = (1.0, 0.5, 0.0)
S_LEVELS = (1.0, 0.8, 0.6, 0.4)
L_FIT, L_FIT_CHECK = 10, 12             # spherical-harmonic degree of the fits (C4 compares 12 with 10)
L_ODD_MAP = 24                          # second fit for Q2 (stability of d_odd) and E5 (odd power above degree 9)
LEG_DEG, LEG_DEG_CHECK = 200, 240       # Legendre degree of the Ren pattern (C5 compares 240 with 200)
LEG_NODES = 4000                        # Gauss-Legendre nodes for the Legendre coefficients
GRID_HALF, GRID_HALF_CHECK = m3.GRID_HALF, 2 * m3.GRID_HALF   # 2000 and 4000 directions with antipodes
CHECK_DIRS = 40                         # directions of the exact checks I2-I4
CHECK_STREAM = (61, 1)
X_FINE = np.linspace(-1.0, 1.0, 200001)  # cos(theta) grid for ranges of zonal patterns (contains 0 and +-1)

EPS_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
LEVEL_SEEDS = {e: tuple(range(5001 + 100 * i, 5101 + 100 * i)) for i, e in enumerate(EPS_GRID)}
CAL_SEEDS = m3.CAL_SEEDS                 # 1001-1010
C1_SEEDS = tuple(range(6001, 7001))
E3_SEEDS = tuple(range(7001, 7101))
BLIND_SEEDS = (9401, 9402)
E8_S = (0.8, 0.6)

REN_C = np.deg2rad(m3.REN_C_DEG)
R_REN = float(1.0 - np.exp(-np.pi ** 2 / (8.0 * REN_C ** 2)))   # even range of m3.ren_signal in units of b

# Recording points: MAG-S3 designs with the readouts of MAG-S3b (no topology, no atlas)
R1 = replace(m3.REFERENCE, topology=False, atlas=())
POINT_DESIGN = {
    "R1": R1,
    "R2": replace(R1, delta=0.4),
    "R3": replace(m3.ANTIPODAL_MAP, delta=0.4, n_tuned=50),
    "E8_S0.8": R1,
    "E8_S0.6": R1,
    "C1_d1": R1,
    "C1_d0.1": replace(R1, delta=0.1),
    "C3": replace(R1, sigma_g=0.0, drift=False),
    "BLIND": R1,
    "T_S0.8": replace(R1, sigma_g=0.0, drift=False),     # test_s3b.py only: noise-free record at S = 0.8
    "E4_b0": R1, "E4_b45": R1, "E4_b90": R1,               # E4 records (mode e4), fixed azimuth at p = 0
}
POINT_S = {k: 1.0 for k in POINT_DESIGN}
POINT_S.update({"E8_S0.8": 0.8, "E8_S0.6": 0.6, "T_S0.8": 0.8})
POINT_CODE = {"R1": 1, "R2": 2, "R3": 3, "E8_S0.8": 4, "E8_S0.6": 5, "C1_d1": 11, "C1_d0.1": 12, "C3": 13,
              "BLIND": 14, "T_S0.8": 21, "E4_b0": 31, "E4_b45": 32, "E4_b90": 33}
RUN_POINTS = ("R1", "R2", "R3", "E8_S0.8", "E8_S0.6")


# --------------------------------------------------------------------------------------------------------------
# Pair physics
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    form: str                 # K, P-same, P-LH
    chi: float                # chi_b of K or chi of P, units of pi
    m: float = 0.0            # K: share of divergence
    alpha: float = 0.0        # K: angle between n_b and n_r, degrees
    beta: float = 0.0         # tilt of n_b from the molecular z axis in the xz plane, degrees
    rates: str = "cry4a"

    @property
    def name(self):
        if self.form == "K":
            return f"K_chi{self.chi:g}_m{self.m:g}_a{self.alpha:g}_b{self.beta:g}_{self.rates}"
        return f"{self.form}_chi{self.chi:g}_b{self.beta:g}_{self.rates}"


def pair_cells():
    """The 225 cells of the pair grid: K 3 chi x 6 (m, alpha) x 3 beta x 3 rates + 9 with beta = 80 deg at m = 1,
    P-same and P-LH 3 chi x 3 beta x 3 rates each."""
    cells = []
    for rates in RATES:
        for chi in CHI_LEVELS:
            for beta in BETA_LEVELS:
                cells += [Cell("K", chi, m, 0.0, beta, rates) for m in M_LEVELS]
                cells += [Cell("K", chi, 0.0, a, beta, rates) for a in ALPHA_LEVELS]
                cells += [Cell("P-same", chi, beta=beta, rates=rates), Cell("P-LH", chi, beta=beta, rates=rates)]
            cells.append(Cell("K", chi, 1.0, 0.0, BETA_K_EXTRA, rates))
    return cells


def reciprocity_cells():
    """I3: K with chi_r = chi_b and n_r = n_b, every chi_b, beta and rate set (27 checks)."""
    return [Cell("K", chi, 0.0, 0.0, beta, rates) for rates in RATES for chi in CHI_LEVELS for beta in BETA_LEVELS]


def axis(deg):
    b = np.deg2rad(deg)
    return np.array([np.sin(b), 0.0, np.cos(b)])


class Spin:
    """The spin system of MAG-S2 and the CISS operators on electron A."""

    def __init__(self):
        self.system = m2.spin_system(P2)
        sim, H0, PS, _ = self.system
        self.sim = sim
        self.PS = np.asarray(PS, dtype=complex)
        self.sigma_A = [2.0 * np.asarray(sim.spin_operator(0, a), dtype=complex) for a in "xyz"]
        self.eye = np.eye(H0.shape[0], dtype=complex)

    def n_sigma(self, n):
        return n[0] * self.sigma_A[0] + n[1] * self.sigma_A[1] + n[2] * self.sigma_A[2]

    def turned(self, chi, n):
        """U P_S U^+ with U = exp(-i chi n . sigma_A) = cos(chi) - i sin(chi) n . sigma_A, since (n . sigma)^2 = 1."""
        U = np.cos(chi) * self.eye - 1j * np.sin(chi) * self.n_sigma(n)
        Pm = U @ self.PS @ U.conj().T
        return 0.5 * (Pm + Pm.conj().T)

    def polarised(self, chi, n, sign):
        """|psi><psi| (x) 1 with psi = cos(chi/2)|S> + sign sin(chi/2)|T0 along n>, |T0 along n> = (n . sigma_A)|S>."""
        A = np.cos(chi / 2) * self.eye + sign * np.sin(chi / 2) * self.n_sigma(n)
        Pm = A @ self.PS @ A.conj().T
        return 0.5 * (Pm + Pm.conj().T)


def cell_operators(spin: Spin, cell: Cell):
    """(P_init, P_rec) of a cell."""
    chi = cell.chi * np.pi
    nb = axis(cell.beta)
    if cell.form == "K":
        return spin.turned(chi, nb), spin.turned(chi * (1.0 - cell.m), axis(cell.beta + cell.alpha))
    if cell.form == "P-same":
        Pm = spin.polarised(chi, nb, +1.0)
        return Pm, Pm
    if cell.form == "P-LH":
        return spin.polarised(chi, nb, +1.0), spin.polarised(chi, nb, -1.0)
    raise ValueError(cell.form)


def forward_yields(spin: Spin, cell: Cell, dirs):
    """Forward (signal) yield of the Haberkorn solver of MAG-S3 at field directions dirs (molecular frame)."""
    P_init, P_rec = cell_operators(spin, cell)
    k_f, k_r = RATES[cell.rates]
    return np.array([m3.haberkorn_yields(spin.system, n, P_init, P_rec, k_r, 0.0, k_f)[2] for n in dirs])


def check_dirs():
    return m2.uniform_dirs(rng_for(*CHECK_STREAM), CHECK_DIRS)


def parity_parts(yp, ym):
    """Even and odd parts over the closed grid (yp at n, ym at -n). eps_mol is half the range of the odd part over
    the range of the even part, as eps_equivalent of MAG-S3 (pattern_parts)."""
    even, odd = (yp + ym) / 2, (yp - ym) / 2
    er = float(even.max() - even.min())
    om = float(np.abs(odd).max())
    return dict(mean=float(even.mean()), even_range=er, odd_half_range=om, eps_mol=om / er if er > 0 else None,
                odd_sd=float(np.concatenate([odd, -odd]).std()), even_sd=float(even.std()))


def sh_degrees(L):
    l = np.concatenate([np.full(2 * k + 1, k) for k in range(L + 1)])
    m = np.concatenate([np.arange(-k, k + 1) for k in range(L + 1)])
    return l, m


def harmonic_fit(half, yp, ym, L):
    """Real spherical harmonics of MAG-S2 up to degree L, fitted on the closed grid to the even part with the even
    degrees and to the odd part with the odd degrees (on a closed grid the two sets are exactly orthogonal)."""
    grid = np.vstack([half, -half])
    B, degs = m2.real_sh(L, grid)
    l, m = sh_degrees(L)
    assert np.array_equal(l, degs)
    even = np.concatenate([(yp + ym) / 2] * 2)
    odd = np.concatenate([(yp - ym) / 2, (ym - yp) / 2])
    ev, od = l % 2 == 0, l % 2 == 1
    ce, *_ = np.linalg.lstsq(B[:, ev], even, rcond=None)
    co, *_ = np.linalg.lstsq(B[:, od], odd, rcond=None)
    coef = np.zeros(len(l))
    coef[ev], coef[od] = ce, co
    re, ro = B[:, ev] @ ce - even, B[:, od] @ co - odd
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))  # noqa: E731
    odd_sd = float(odd.std())
    return dict(L=L, coef=coef, l=l, m=m, even_rms_over_sd=rms(re) / float(even.std()),
                odd_rms_over_sd=rms(ro) / odd_sd if odd_sd > 1e-12 * float(even.std()) else None)


def dipole(fit):
    """Odd dipole d of the molecule, f_1(n) = d . n, from the degree-1 coefficients. With the Condon-Shortley phase
    of SciPy the real harmonics of degree 1 are -c x, -c y and c z for m = 1, -1 and 0, with c = sqrt(3 / 4 pi)."""
    c, l, m = fit["coef"], fit["l"], fit["m"]
    get = lambda mm: float(c[(l == 1) & (m == mm)][0])  # noqa: E731
    return np.sqrt(3.0 / (4.0 * np.pi)) * np.array([-get(1), -get(-1), get(0)])


def zonal(fit):
    """Coefficients of P_l(cos theta) of the azimuthal average around the molecular z axis."""
    c, l, m = fit["coef"], fit["l"], fit["m"]
    L = int(fit["L"])
    return np.array([c[(l == k) & (m == 0)][0] * np.sqrt((2 * k + 1) / (4.0 * np.pi)) for k in range(L + 1)])


def odd_power_by_degree(fit):
    """Share of the odd power of the molecule (all m) and of the receptor (m = 0) in each odd degree."""
    c, l, m = fit["coef"], fit["l"], fit["m"]
    od = l % 2 == 1
    tot, tot0 = float(np.sum(c[od] ** 2)), float(np.sum(c[od & (m == 0)] ** 2))
    degs = sorted(set(l[od].tolist()))
    return dict(molecule={str(k): float(np.sum(c[l == k] ** 2)) / tot if tot > 0 else None for k in degs},
                receptor={str(k): float(np.sum(c[(l == k) & (m == 0)] ** 2)) / tot0 if tot0 > 0 else None
                          for k in degs})


def odd_power_shares(fit):
    """Share of the odd power in degrees 3 and above: of the molecule (all m) and of the receptor (m = 0)."""
    c, l, m = fit["coef"], fit["l"], fit["m"]
    od = l % 2 == 1
    tot, hi = float(np.sum(c[od] ** 2)), float(np.sum(c[od & (l >= 3)] ** 2))
    tot0, hi0 = float(np.sum(c[od & (m == 0)] ** 2)), float(np.sum(c[od & (m == 0) & (l >= 3)] ** 2))
    return dict(molecule=hi / tot if tot > 0 else None, receptor=hi0 / tot0 if tot0 > 0 else None)


def zonal_range(a):
    """Range over cos(theta) of the even part of the zonal series sum_l a_l P_l."""
    ae = np.where(np.arange(len(a)) % 2 == 0, a, 0.0)
    f = npleg.legval(X_FINE, ae)
    return float(f.max() - f.min())


# --------------------------------------------------------------------------------------------------------------
# Order of the receptor
# --------------------------------------------------------------------------------------------------------------
def p_moments(kappa, lmax):
    """<P_l(cos theta)> of the von Mises - Fisher distribution, I_{l+1/2}(kappa) / I_{1/2}(kappa), l = 0..lmax."""
    if not np.isfinite(kappa):
        return np.ones(lmax + 1)
    l = np.arange(lmax + 1, dtype=float)
    return special.ive(l + 0.5, kappa) / special.ive(0.5, kappa)


_KAPPA = {}


def kappa_for(S):
    """Concentration kappa with <P2>_kappa = S (S = 1 gives infinity)."""
    if S >= 1.0:
        return np.inf
    if S not in _KAPPA:
        _KAPPA[S] = float(optimize.brentq(lambda k: p_moments(k, 2)[2] - S, 1e-8, 1e5, xtol=1e-14, rtol=1e-15))
    return _KAPPA[S]


_REN_LEG = {}


def ren_legendre(lmax=LEG_DEG, nodes=LEG_NODES):
    """Legendre coefficients of the even Ren signal -exp(-arcsin(x)^2 / 2c^2) (units of b), odd ones set to zero."""
    key = (lmax, nodes)
    if key not in _REN_LEG:
        x, w = special.roots_legendre(nodes)
        g = m3.ren_signal(x)
        V = npleg.legvander(x, lmax)
        a = (2.0 * np.arange(lmax + 1) + 1.0) / 2.0 * ((w * g) @ V)
        a[1::2] = 0.0
        _REN_LEG[key] = a
    return _REN_LEG[key]


def ren_smoothed_coef(S, lmax=LEG_DEG):
    return ren_legendre(lmax) * p_moments(kappa_for(S), lmax)


_DEPTH = {}


def depth(S, lmax=LEG_DEG):
    """D(S): range of the Ren pattern smoothed by order S over the range of the Ren pattern itself (D(1) = 1)."""
    if S >= 1.0:
        return 1.0
    if (S, lmax) not in _DEPTH:
        _DEPTH[(S, lmax)] = zonal_range(ren_smoothed_coef(S, lmax)) / R_REN
    return _DEPTH[(S, lmax)]


def g_order(S, lmax=LEG_DEG):
    return float(p_moments(kappa_for(S), 1)[1] / depth(S, lmax))


def receptor_signal(cos_t, S=1.0, p=1.0, a=0.0):
    """Receptor signal of the neural model in units of b at cos(theta) between field and receptor axis: the Ren
    pattern smoothed by axial order S with the odd admixture a <P1> cos(theta) of the molecule, mixed over the two
    sides of the folds with polar order p. At S = 1 and p = 1 it is m3.ren_signal(cos_t, a) bitwise."""
    if S >= 1.0:
        up, down = m3.ren_signal(cos_t, a), m3.ren_signal(-cos_t, a)
    else:
        coef = ren_smoothed_coef(S)
        p1 = p_moments(kappa_for(S), 1)[1]
        up = npleg.legval(cos_t, coef) + a * p1 * cos_t
        down = npleg.legval(-cos_t, coef) - a * p1 * cos_t
    return (1.0 + p) / 2.0 * up + (1.0 - p) / 2.0 * down


def admixture_for(eps_eff, S, p=1.0):
    """Molecular admixture a that gives eps_eff at order (S, p): eps_eff = p <P1> a / D(S)."""
    return eps_eff * depth(S) / (p * p_moments(kappa_for(S), 1)[1])


def receptor_order_map(zon):
    """Receptor quantities of one cell for all (p, S): eps_rec (S = 1, p = 1), the transferred eps_eff and the
    pair's own depth D_pair(S) and odd-to-even ratio at (p, S). zon: zonal coefficients of P_l of the molecule."""
    L = len(zon) - 1
    rng0 = zonal_range(zon)
    eps_rec = float(zon[1] / rng0)
    out = dict(eps_rec=eps_rec, receptor_even_range=rng0, by_order={})
    for S in S_LEVELS:
        mom = p_moments(kappa_for(S), L)
        rng_s = zonal_range(zon * mom)
        for p in P_LEVELS:
            out["by_order"][f"S{S:g}_p{p:g}"] = dict(
                eps_eff=float(p * g_order(S) * R_REN * eps_rec),
                eps_E6=float(p * mom[1] * zon[1] / (m3.REN_B * depth(S))),
                pair_ratio=float(p * mom[1] * zon[1] / rng_s), D_pair=rng_s / rng0, g_pair=float(mom[1] * rng0 / rng_s))
    return out


def i5_receptor(zon):
    """I5 through the explicit mixture of the two sides of the folds: f_p(x) = (1+p)/2 F(x) + (1-p)/2 F(-x)."""
    L = len(zon) - 1
    res = {}
    for S in S_LEVELS:
        a = zon * p_moments(kappa_for(S), L)
        F = lambda x: npleg.legval(x, a)  # noqa: E731
        odd = {}
        for p in (1.0, 0.5, 0.0):
            fp = (1 + p) / 2 * F(X_FINE) + (1 - p) / 2 * F(-X_FINE)
            fm = (1 + p) / 2 * F(-X_FINE) + (1 - p) / 2 * F(X_FINE)
            odd[p] = (fp - fm) / 2
        res[f"S{S:g}"] = dict(p0_max_odd=float(np.abs(odd[0.0]).max()),
                              p05_half_max_diff=float(np.abs(odd[0.5] - odd[1.0] / 2).max()))
    return res


# --------------------------------------------------------------------------------------------------------------
# Records of the inverse problem
# --------------------------------------------------------------------------------------------------------------
def order_key(d, S, p):
    text = json.dumps(dict(design=asdict(d), S=S, p=p), sort_keys=True)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def modulation_order(bird, d, S, p, a):
    """u (tuned cells x K) of the compass with the receptor signal at order (S, p) and molecular admixture a; the
    normalisation over the closed grid and the presented directions is that of m3.modulation."""
    dirs = bird.dirs(d.K)
    allx = np.vstack([bird.grid, dirs])
    rec = receptor_signal(bird.v @ allx.T, S, p, a)
    s = bird.sign[:d.n_tuned, None] * (bird.W[:d.n_tuned] @ rec)
    return m3.normalised(s)[:, len(bird.grid):]


def make_record(bird, point, eps_eff, expected=False):
    """One record and its truth. S = 1 goes through m3.modulation and m3.record_condition unchanged, with the odd
    admixture eps = eps_eff of the MAG-S3 Ren pattern; S < 1 goes through modulation_order with the same expected
    counts and count draw of MAG-S3. expected=True returns the expected counts (C3)."""
    S = POINT_S[point]
    d = replace(POINT_DESIGN[point], eps=float(eps_eff))
    sched = m3.schedule(bird, d)
    if S >= 1.0:
        u = m3.modulation(bird, d, "compass")
        if expected:
            C = m3.expected_counts(bird, d, u, sched)
        else:
            C, _, _, checks = m3.record_condition(bird, d, u, sched, m3.COND_CODE["compass"], m3.design_key(d))
            if not (checks["counts_ok"] and checks["rates_ok"]):
                raise RuntimeError(f"bad counts or rates: seed {bird.seed}, point {point}")
    else:
        u = modulation_order(bird, d, S, 1.0, admixture_for(eps_eff, S))
        lam = m3.expected_counts(bird, d, u, sched)
        if not (np.all(np.isfinite(lam)) and lam.min() >= 0):
            raise RuntimeError(f"bad rates: seed {bird.seed}, point {point}")
        C = lam if expected else m3.draw_counts(lam, d.fano, rng_for(bird.seed, 50, order_key(d, S, 1.0), 1))
    if not expected:
        if C.max() >= 2 ** 16:
            raise RuntimeError("count above the uint16 range")
        C = C.astype(np.uint16)
    rec = dict(seed=int(bird.seed), point=point, K=int(d.K), R=int(d.R), n_cells=int(d.n_tuned), counts=C,
               sess=sched[0].astype(np.int16))
    truth = dict(seed=int(bird.seed), point=point, eps_eff=float(eps_eff), S=float(S), p=1.0, delta=float(d.delta),
                 design=asdict(d))
    return rec, truth


def record_path(folder, point, seed):
    return os.path.join(folder, f"{point}_{seed}.npz")


def save_record(folder, rec):
    path = record_path(folder, rec["point"], rec["seed"])
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, counts=rec["counts"], sess=rec["sess"],
                        meta=np.array(json.dumps({k: rec[k] for k in ("seed", "point", "K", "R", "n_cells")})))
    os.replace(tmp, path)
    return path


def load_record(path):
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        meta.update(counts=z["counts"], sess=z["sess"].astype(int))
    return meta


EPS_CHECK = (0.0, 0.05)                 # levels of the birds of the smoke and blind checks, in turn


def run_plan(seeds=None):
    """(point, seed, eps_eff) of every record of the run: 7 levels x 100 birds x (R1-R3 and E8 at S 0.8 and 0.6).
    With seeds (smoke and blind checks) bird j gets the level EPS_CHECK[j mod 2], so every (point, seed) is one
    record, as in the run."""
    if seeds:
        return [(pt, s, EPS_CHECK[j % len(EPS_CHECK)]) for j, s in enumerate(seeds) for pt in RUN_POINTS]
    return [(pt, s, e) for e in EPS_GRID for s in LEVEL_SEEDS[e] for pt in RUN_POINTS]


# --------------------------------------------------------------------------------------------------------------
# E3: what the estimate takes from calibration birds when it does not know the retina
# --------------------------------------------------------------------------------------------------------------
E3_DESIGNS = {"K800_n200": R1, "K100_n50": POINT_DESIGN["R3"]}


def e3_geometry(seeds=E3_SEEDS):
    """kappa_i^2 = |V_i|^2 / variance over the presented pairs of the even template s_even_i, averaged over every
    cell of the calibration birds 7001-7100, for the two recording sizes. With it the dipole energy of a cell has the
    expectation eps^2 kappa^2 Var(even signal) P / 3 for a dipole direction the estimate does not know."""
    out = {}
    for key, d in E3_DESIGNS.items():
        k2 = []
        for seed in seeds:
            bird = m3.Bird(seed)
            N = bird.dirs(d.K)[:d.K // 2]
            s_even = bird.sign[:d.n_tuned, None] * (bird.W[:d.n_tuned] @ m3.ren_signal(bird.v @ N.T))
            V = bird.sign[:d.n_tuned, None] * (bird.W[:d.n_tuned] @ bird.v)
            k2.append(np.sum(V ** 2, 1) / s_even.var(1))
        k2 = np.concatenate(k2)
        out[key] = dict(kappa2_mean=float(k2.mean()), kappa2_sd=float(k2.std()), n_cells=int(len(k2)),
                        seeds=[int(seeds[0]), int(seeds[-1])])
    return out


# --------------------------------------------------------------------------------------------------------------
# E4: fixed azimuth
# --------------------------------------------------------------------------------------------------------------
E4_CELLS = tuple(Cell("K", 0.1, 1.0, 0.0, beta, "cry4a") for beta in BETA_LEVELS)
FLIP_X = np.diag([1.0, -1.0, -1.0])      # 180 degrees about the molecular x axis (the local meridian)


def e4_cell(half, coef, L, d_odd, zon):
    """Fixed azimuth, p = 0: half of the molecules turned by 180 degrees about their x axis, f_mix(m) = (f(m) +
    f(F m)) / 2. Returns the odd residual of the mixture over its even range (from the fit of degree L on the closed
    grid) and the surviving dipole d_x with its transfer eps_x = R_REN d_x / range of the zonal even part, which is
    the normalisation of eps_eff."""
    grid = np.vstack([half, -half])
    B, _ = m2.real_sh(L, grid)
    Bf, _ = m2.real_sh(L, grid @ FLIP_X.T)
    f_mix = 0.5 * (B @ coef + Bf @ coef)
    k = len(half)
    odd = 0.5 * (f_mix[:k] - f_mix[k:])
    even = 0.5 * (f_mix[:k] + f_mix[k:])
    rng0 = zonal_range(zon)
    return dict(eps_mol_fixed_p0=float(np.abs(odd).max() / (even.max() - even.min())), d_x=float(d_odd[0]),
                eps_x=float(R_REN * d_odd[0] / rng0), eps_y_p1=float(R_REN * d_odd[1] / rng0),
                eps_z_p1=float(R_REN * d_odd[2] / rng0))


def modulation_fixed(bird, d, eps_x):
    """u (tuned cells x K) at fixed azimuth and p = 0: the even Ren pattern around each receptor axis and the odd part
    eps_x (x_k . n) along its molecular x axis, the only dipole component the flip about x keeps."""
    dirs = bird.dirs(d.K)
    allx = np.vstack([bird.grid, dirs])
    rec = m3.ren_signal(bird.v @ allx.T) + eps_x * (bird.R[:, :, 0] @ allx.T)
    s = bird.sign[:d.n_tuned, None] * (bird.W[:d.n_tuned] @ rec)
    return m3.normalised(s)[:, len(bird.grid):]


def _e4_job(args):
    seed, eps_by_point = args
    est = load_estimate()
    bird = m3.Bird(seed)
    sched = m3.schedule(bird, R1)
    out = {}
    for point, eps_x in eps_by_point.items():
        u = modulation_fixed(bird, R1, eps_x)
        lam = m3.expected_counts(bird, R1, u, sched)
        C = m3.draw_counts(lam, R1.fano, rng_for(seed, 51, POINT_CODE[point]))
        r = est.dipole_test(C, sched[0], bird.dirs(R1.K), rng_for(seed, est.STREAM_DIPOLE, POINT_CODE[point]))
        out[point] = dict(p=r["p"], odd_found=r["odd_found"])
    return seed, out


def e4(out, workers, seeds=None):
    """E4 records in R1 (birds of eps_eff = 0 unless seeds are given) and their dipole tests."""
    from multiprocessing import get_context
    with open(os.path.join(out, "receptor.json")) as fh:
        rows = {Cell(**r["cell"]).name: r for r in json.load(fh)["cells"]}
    eps = {f"E4_b{c.beta:g}": rows[c.name]["E4"]["eps_x"] for c in E4_CELLS}
    seeds = list(seeds or LEVEL_SEEDS[0.0])
    res = {}
    with get_context("fork").Pool(workers) as pool:
        for seed, r in pool.imap_unordered(_e4_job, [(s, eps) for s in seeds], chunksize=4):
            res[seed] = r
    summary = {pt: dict(eps_x=e, n=len(seeds), dipole_found=int(sum(res[s][pt]["odd_found"] for s in seeds)),
                        receptor=rows[c.name]["E4"]) for (pt, e), c in zip(eps.items(), E4_CELLS)}
    write_json(os.path.join(out, "e4.json"), dict(points=summary, birds={str(s): res[s] for s in seeds},
                                                  metadata=metadata("e4")))
    return summary


def _gen_job(args):
    seed, items, folder = args
    bird = m3.Bird(seed)
    truths = []
    for point, eps in items:
        rec, truth = make_record(bird, point, eps)
        truth["record_sha256"] = file_sha256(save_record(folder, rec))
        truths.append(truth)
    return truths


def generate(records_dir, truth_dir, workers, plan=None):
    from multiprocessing import get_context
    if os.path.abspath(truth_dir).startswith(os.path.abspath(records_dir)):
        sys.exit("the truth folder must lie outside the records folder")
    os.makedirs(records_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)
    plan = plan or run_plan()
    by_seed = {}
    for point, seed, eps in plan:
        by_seed.setdefault(seed, []).append((point, eps))
    jobs = [(s, by_seed[s], records_dir) for s in sorted(by_seed)]
    truths = []
    with get_context("fork").Pool(workers) as pool:
        for t in pool.imap_unordered(_gen_job, jobs, chunksize=2):
            truths += t
    truths.sort(key=lambda t: (t["point"], t["seed"]))
    meta = metadata("generate")
    # the manifest carries no truth: I7 compares it with the estimates before the truth file is opened
    write_json(os.path.join(records_dir, "manifest.json"), dict(
        protocol=PROTOCOL_VERSION, records=[dict(point=t["point"], seed=t["seed"], record_sha256=t["record_sha256"])
                                            for t in truths], metadata=meta))
    write_json(os.path.join(truth_dir, "truth.json"), dict(protocol=PROTOCOL_VERSION, records=truths, metadata=meta))
    return truths


# --------------------------------------------------------------------------------------------------------------
# Pair grid and invariants (mode physics)
# --------------------------------------------------------------------------------------------------------------
_SPIN = None


def spin():
    global _SPIN
    if _SPIN is None:
        _SPIN = Spin()
    return _SPIN


def cell_grid_yields(cell, half_n=GRID_HALF):
    half = m2.fibonacci_sphere(half_n)
    s = spin()
    return half, forward_yields(s, cell, half), forward_yields(s, cell, -half)


def _cell_job(args):
    cell, folder, half_n = args
    path = os.path.join(folder, cell.name + ".json")
    if os.path.exists(path):
        return cell.name, "exists"
    t0 = time.perf_counter()
    _, yp, ym = cell_grid_yields(cell, half_n)
    write_json(path, dict(cell=asdict(cell), grid_half=half_n, rates_used=list(RATES[cell.rates]),
                          mag_s3b_sha256=file_sha256(os.path.abspath(__file__)), yp=yp.tolist(), ym=ym.tolist(),
                          seconds=time.perf_counter() - t0))
    return cell.name, "done"


def invariants_physics():
    """I2 (chi = 0, every form and rate set), I3 (reciprocal K), I4 (exponential model of P-same) on 40 directions,
    I6 (Haberkorn against time steps, one cell of each form) and I8 (without CISS the neural input at p = 1 and
    p = 0 is the same at every S; calibration bird 1001, reference design)."""
    s = spin()
    dirs = check_dirs()
    out = dict(I2={}, I3={}, I4={}, I6={}, I8={})
    bird = m3.Bird(CAL_SEEDS[0])
    for S in S_LEVELS:
        u1 = modulation_order(bird, R1, S, 1.0, 0.0)
        out["I8"][f"S{S:g}"] = float(max(np.max(np.abs(modulation_order(bird, R1, S, p, 0.0) - u1))
                                         for p in (0.5, 0.0)))
    for form in FORMS:
        for rates in RATES:
            c = Cell(form, 0.0, 0.0, 0.0, 0.0, rates)
            out["I2"][c.name] = float(np.max(np.abs(forward_yields(s, c, dirs) - forward_yields(s, c, -dirs))) / 2)
    for c in reciprocity_cells():
        out["I3"][c.name] = float(np.max(np.abs(forward_yields(s, c, dirs) - forward_yields(s, c, -dirs))) / 2)
    for chi in CHI_LEVELS:
        for beta in BETA_LEVELS:
            Pm = s.polarised(chi * np.pi, axis(beta), +1.0)
            a = np.array([m3.ciss_yield(s.system, Pm, n) for n in dirs])
            b = np.array([m3.ciss_yield(s.system, Pm, -n) for n in dirs])
            out["I4"][f"chi{chi:g}_b{beta:g}"] = float(np.max(np.abs(a - b)) / 2)
    n0 = dirs[0]
    for cell in (Cell("K", 0.1, 0.5, 0.0, 45.0, "cry4a"), Cell("P-same", 0.1, beta=45.0, rates="cry4a"),
                 Cell("P-LH", 0.1, beta=45.0, rates="cry4a")):
        P_init, P_rec = cell_operators(s, cell)
        k_f, k_r = RATES[cell.rates]
        syl = m3.haberkorn_yields(s.system, n0, P_init, P_rec, k_r, 0.0, k_f)[2]
        td = m3.haberkorn_time_domain(s.system, n0, P_init, P_rec, k_r, k_f, dt=2.5e-10, lifetimes=25.0)
        out["I6"][cell.name] = float(abs(td - syl))
    return out


def physics(out, workers, max_hours, half_n=GRID_HALF):
    """The pair grid (resumable; False when it stopped at the time limit) and the invariants I2-I4, I6, I8. A smaller
    grid (half_n < 1000) serves only the smoke test of the pipeline."""
    from multiprocessing import get_context
    folder = os.path.join(out, "cells")
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    todo = [c for c in pair_cells() if not os.path.exists(os.path.join(folder, c.name + ".json"))]
    with get_context("fork").Pool(workers) as pool:
        for name, state in pool.imap_unordered(_cell_job, [(c, folder, half_n) for c in todo]):
            print(name, state, f"{(time.time() - t0) / 3600:.2f} h", flush=True)
            if max_hours and time.time() - t0 > max_hours * 3600:
                pool.terminate()
                print("time limit, stopped; rerun to continue", flush=True)
                return False
    inv_path = os.path.join(out, "invariants_physics.json")
    if not os.path.exists(inv_path):
        write_json(inv_path, dict(invariants_physics(), metadata=metadata("physics")))
    return True


def cell_summary(cell, half, yp, ym, L=L_FIT):
    """Pair and receptor quantities of one cell from the fit of degree L; a second fit of degree L_ODD_MAP gives the
    stability of the dipole (Q2) and the odd power above degree 9 (E5). Calibration round 2 found that the odd part
    carries angular structure above degree 30 (K cell of C2: residual 17% of its SD at degree 10, 4% at 28), while
    the dipole stays within 0.02% from degree 10 to 32."""
    fit = harmonic_fit(half, yp, ym, L)
    hi = harmonic_fit(half, yp, ym, L_ODD_MAP)
    zon = zonal(fit)
    row = dict(cell=asdict(cell), parts=parity_parts(yp, ym), fit_L=L,
               fit_even_rms_over_sd=fit["even_rms_over_sd"], fit_odd_rms_over_sd=fit["odd_rms_over_sd"],
               d_odd=dipole(fit).tolist(), odd_power_shares=odd_power_shares(fit),
               odd_power_by_degree=odd_power_by_degree(fit), zonal=zon.tolist(),
               receptor=receptor_order_map(zon), I5=i5_receptor(zon),
               fit_check=dict(L=L_ODD_MAP, even_rms_over_sd=hi["even_rms_over_sd"],
                              odd_rms_over_sd=hi["odd_rms_over_sd"], d_odd=dipole(hi).tolist(),
                              odd_power_shares=odd_power_shares(hi), odd_power_by_degree=odd_power_by_degree(hi)))
    if cell in E4_CELLS:
        row["E4"] = e4_cell(half, hi["coef"], L_ODD_MAP, dipole(fit), zon)
    return row


def receptor(out):
    folder = os.path.join(out, "cells")
    rows, halves, cell_files = [], {}, []
    for c in pair_cells():
        with open(os.path.join(folder, c.name + ".json")) as fh:
            r = json.load(fh)
        cell_files.append(dict(mag_s3b_sha256=r.get("mag_s3b_sha256"), rates_used=r.get("rates_used")))
        h = int(r["grid_half"])
        if h not in halves:
            halves[h] = m2.fibonacci_sphere(h)
        rows.append(cell_summary(c, halves[h], np.array(r["yp"]), np.array(r["ym"])))
    if len(halves) != 1:
        sys.exit("cells were computed on different grids")
    if any(r.get("mag_s3b_sha256") != file_sha256(os.path.abspath(__file__)) for r in cell_files):
        sys.exit("cells were computed by another version of mag_s3b.py")
    order = dict(ren={f"S{S:g}": dict(kappa=kappa_for(S), P1=float(p_moments(kappa_for(S), 1)[1]), D=depth(S),
                                      g=g_order(S)) for S in S_LEVELS}, R_REN=R_REN)
    write_json(os.path.join(out, "receptor.json"), dict(cells=rows, order=order, grid_half=next(iter(halves)),
                                                        rates_used=cell_files[0]["rates_used"] and
                                                        {c.rates: f["rates_used"] for c, f in zip(pair_cells(), cell_files)},
                                                        metadata=metadata("receptor")))
    return rows


# --------------------------------------------------------------------------------------------------------------
# Calibration C1-C5
# --------------------------------------------------------------------------------------------------------------
C2_CELLS = (Cell("P-same", 0.1, beta=0.0, rates="mags2"), Cell("K", 0.02, 1.0, 0.0, 0.0, "mags2"))
# eps_equivalent of P-same at chi = 0.1 pi in MAG-S3 v0.1.0, results/confirmatory_run_v04/physics_s3.json (E5)
MAG_S3_E5_ANCHOR = {"mags2": 0.0011244586373918653, "cry4a": 0.00014289998826479909,
                    "cry4a_fast": 0.00012169824179592192}


def c5_ren():
    """C5: the Legendre expansion of the Ren pattern up to LEG_DEG reproduces it to 1e-4 b, and LEG_DEG_CHECK changes
    D(S) by less than 0.1% at S = 0.4. D(S), g(S), kappa and <P1> are reported for every S (they do not use the
    pair), with the residual away from the poles and the difference of the smoothed series of degree 60 and LEG_DEG."""
    exact = m3.ren_signal(X_FINE)
    res = {}
    for L in (60, 80, LEG_DEG, LEG_DEG_CHECK):
        r = npleg.legval(X_FINE, ren_legendre(L)) - exact
        res[f"L{L}"] = dict(max_abs=float(np.abs(r).max()), rms_sphere=float(np.sqrt(np.mean(r ** 2))),
                            max_abs_at_cos=float(X_FINE[np.argmax(np.abs(r))]),
                            max_abs_cos_below_099=float(np.abs(r[np.abs(X_FINE) < 0.99]).max()))
    d1, d2 = depth(0.4, LEG_DEG), depth(0.4, LEG_DEG_CHECK)
    order = {}
    for S in S_LEVELS:
        k = kappa_for(S)
        lo = np.where(np.arange(LEG_DEG + 1) <= 60, ren_smoothed_coef(S, LEG_DEG), 0.0)
        order[f"S{S:g}"] = dict(kappa=k, P1=float(p_moments(k, 1)[1]), P2=float(p_moments(k, 2)[2]),
                                D=depth(S, LEG_DEG), D_check=depth(S, LEG_DEG_CHECK), g=g_order(S, LEG_DEG),
                                smoothed_deg60_vs_full_max=float(np.abs(npleg.legval(X_FINE, lo) - npleg.legval(
                                    X_FINE, ren_smoothed_coef(S, LEG_DEG))).max()) if S < 1 else None)
    ok = res[f"L{LEG_DEG}"]["max_abs"] <= 1e-4 and abs(d2 / d1 - 1.0) < 1e-3
    return dict(degree=LEG_DEG, degree_check=LEG_DEG_CHECK, residual=res, D_S04=d1, D_S04_check=d2,
                D_rel_change=float(d2 / d1 - 1.0), order=order, R_REN=R_REN, criterion_met=bool(ok))


def _yield_chunk(args):
    cell, dirs = args
    return forward_yields(spin(), cell, dirs)


def c2_c4(workers, out_dir=None):
    """C2: doubling the grid to 4000 directions changes eps_mol by less than 1%; C4: degree 12 changes eps_rec by less
    than 1% (both in the C2 cells: P-same at chi = 0.1 pi and K at m = 1, chi_b = 0.02 pi, MAG-S2 rates, beta = 0).
    The yields are kept in C2_yields.npz when out_dir is given (an existing file is reused)."""
    from multiprocessing import get_context
    out = {}
    path = os.path.join(out_dir, "C2_yields.npz") if out_dir else None
    stored = dict(np.load(path)) if path and os.path.exists(path) else {}
    with get_context("fork").Pool(workers) as pool:
        for cell in C2_CELLS:
            row = {}
            for half_n in (GRID_HALF, GRID_HALF_CHECK):
                half = m2.fibonacci_sphere(half_n)
                key = f"{cell.name}_grid{2 * half_n}"
                if key not in stored:
                    allx = np.vstack([half, -half])
                    parts = np.array_split(allx, 4 * workers)
                    stored[key] = np.concatenate(pool.map(_yield_chunk, [(cell, p) for p in parts]))
                    if path:
                        np.savez(path, **stored)
                y = stored[key]
                yp, ym = y[:half_n], y[half_n:]
                row[f"grid{2 * half_n}"] = dict(parts=parity_parts(yp, ym))
                if half_n == GRID_HALF:
                    for L in (L_FIT, L_FIT_CHECK):
                        fit = harmonic_fit(half, yp, ym, L)
                        zon = zonal(fit)
                        row[f"grid{2 * half_n}"][f"L{L}"] = dict(
                            eps_rec=float(zon[1] / zonal_range(zon)), d_odd=dipole(fit).tolist(),
                            even_rms_over_sd=fit["even_rms_over_sd"], odd_rms_over_sd=fit["odd_rms_over_sd"])
            e2, e4 = row["grid2000"]["parts"]["eps_mol"], row["grid4000"]["parts"]["eps_mol"]
            r10, r12 = row["grid2000"][f"L{L_FIT}"]["eps_rec"], row["grid2000"][f"L{L_FIT_CHECK}"]["eps_rec"]
            row["C2_rel_change"] = float(e4 / e2 - 1.0)
            row["C4_rel_change"] = float(r12 / r10 - 1.0)
            if cell.form == "P-same":
                row["anchor_check_vs_mag_s3_E5"] = float(e2 / MAG_S3_E5_ANCHOR[cell.rates] - 1.0)
            out[cell.name] = row
    out["C2_criterion_met"] = bool(all(abs(out[c.name]["C2_rel_change"]) < 0.01 for c in C2_CELLS))
    out["C4_criterion_met"] = bool(all(abs(out[c.name]["C4_rel_change"]) < 0.01 for c in C2_CELLS))
    return out


def _c1_job(seed):
    est = load_estimate()
    bird = m3.Bird(seed)
    res = {}
    for point in ("C1_d1", "C1_d0.1"):
        rec, _ = make_record(bird, point, 0.0)
        r = est.read_record(rec, with_estimate=False)
        res[point] = dict(p_dipole=r["dipole"]["p"], odd_found=r["dipole"]["odd_found"],
                          p_antipodal=r["antipodal"]["p_odd"], antipodal_odd_found=r["antipodal"]["odd_found"])
    return seed, res


def c1_false_odd(workers, seeds=C1_SEEDS):
    """C1: on 1000 birds with an even code (seeds 6001-7000) the dipole test finds an odd part in 3.6-6.4% of the
    birds, in R1 and in R1 at delta = 10%. The antipodal test of MAG-S3 on the same records is reported alongside."""
    from multiprocessing import get_context
    res = {}
    with get_context("fork").Pool(workers) as pool:
        for seed, r in pool.imap_unordered(_c1_job, seeds, chunksize=5):
            res[seed] = r
    out = {}
    for point in ("C1_d1", "C1_d0.1"):
        rows = [res[s][point] for s in seeds]
        n_found = int(sum(r["odd_found"] for r in rows))
        p = np.array([r["p_dipole"] for r in rows])
        out[point] = dict(n=len(rows), dipole_odd_found=n_found, share=n_found / len(rows),
                          antipodal_odd_found=int(sum(r["antipodal_odd_found"] for r in rows)),
                          p_dipole_deciles=np.percentile(p, np.arange(10, 100, 10)).tolist(),
                          criterion_met=bool(36 <= n_found <= 64), p_dipole=p.tolist())
    out["criterion_met"] = bool(all(out[pt]["criterion_met"] for pt in ("C1_d1", "C1_d0.1")))
    return out


def c3_exact(seeds=CAL_SEEDS):
    """C3: with expected counts, sigma_g = 0 and no drift, eps_hat matches eps_eff to 2% at 0.02 and 0.1 in R1 and the
    bootstrap gives finite intervals in every record."""
    est = load_estimate()
    rows = []
    for seed in seeds:
        bird = m3.Bird(seed)
        for eps in (0.02, 0.1):
            rec, _ = make_record(bird, "C3", eps, expected=True)
            r = est.read_record(rec, with_tests=False)["estimate"]
            rows.append(dict(seed=int(seed), eps_eff=eps, eps_hat=r["eps_hat_raw"],
                             rel_err=float(r["eps_hat_raw"] / eps - 1.0), ci95=r["ci95"], upper95=r["upper95"],
                             finite=bool(r["finite"])))
    ok = all(abs(r["rel_err"]) <= 0.02 and r["finite"] for r in rows)
    return dict(rows=rows, max_abs_rel_err=float(max(abs(r["rel_err"]) for r in rows)), criterion_met=bool(ok))


_ESTIMATE = None


def load_estimate():
    global _ESTIMATE
    if _ESTIMATE is None:
        if "estimate" in sys.modules:
            _ESTIMATE = sys.modules["estimate"]
        else:
            spec = importlib.util.spec_from_file_location("estimate", os.path.join(HERE, "estimate.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["estimate"] = mod
            spec.loader.exec_module(mod)
            _ESTIMATE = mod
    return _ESTIMATE


def calibrate(out, checks, workers):
    os.makedirs(out, exist_ok=True)
    meta = metadata("calibrate")
    for chk in checks:
        t0 = time.perf_counter()
        if chk == "C5":
            res = c5_ren()
        elif chk in ("C2", "C4"):
            if os.path.exists(os.path.join(out, "C2_C4.json")):
                continue
            res, chk = c2_c4(workers, out), "C2_C4"
        elif chk == "C3":
            res = c3_exact()
        elif chk == "C1":
            res = c1_false_odd(workers)
        else:
            raise ValueError(chk)
        res["seconds"] = time.perf_counter() - t0
        res["metadata"] = meta
        write_json(os.path.join(out, f"{chk}.json"), res)
        print(chk, "done", f"{res['seconds']:.0f} s", flush=True)


# --------------------------------------------------------------------------------------------------------------
# Metadata and command line
# --------------------------------------------------------------------------------------------------------------
def metadata(mode):
    own = {f: file_sha256(os.path.join(HERE, f)) for f in OWN_FILES if os.path.exists(os.path.join(HERE, f))}
    return dict(mode=mode, protocol=PROTOCOL_VERSION, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                python=sys.version.split()[0], numpy=np.__version__, scipy=m3.scipy.__version__,
                radicalpy=m3._pkg_version("radicalpy"), blas=m3._blas(), frozen_ok=not frozen_mismatches(),
                frozen=FROZEN, own=own)


def parse_seeds(text):
    """'9401-9402' or '7001,7005' into a tuple of seeds."""
    out = []
    for part in text.split(","):
        a, _, c = part.partition("-")
        out += list(range(int(a), int(c) + 1)) if c else [int(a)]
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("physics", "receptor", "e3geo", "generate", "e4", "calibrate"))
    ap.add_argument("--out")
    ap.add_argument("--records")
    ap.add_argument("--truth")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--checks", default="C5,C2,C4,C3,C1")
    ap.add_argument("--grid-half", type=int, default=GRID_HALF, help="physics: smaller grids only for the smoke test")
    ap.add_argument("--seeds", help="generate, e3geo, e4: birds for the smoke and blind checks, e.g. 9401-9402")
    ap.add_argument("--blind-rates", action="store_true", help="physics: the substitute rates of the blind check")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds) if a.seeds else None
    if a.blind_rates:
        use_blind_rates()
    if a.mode == "physics":
        if not physics(a.out, a.workers, a.max_hours, a.grid_half):
            sys.exit(3)
    elif a.mode == "receptor":
        receptor(a.out)
    elif a.mode == "e3geo":
        os.makedirs(a.records, exist_ok=True)
        write_json(os.path.join(a.records, "e3_geometry.json"),
                   dict(designs=e3_geometry(seeds or E3_SEEDS), metadata=metadata("e3geo")))
    elif a.mode == "generate":
        generate(a.records, a.truth, a.workers, run_plan(seeds))
    elif a.mode == "e4":
        e4(a.out, a.workers, seeds)
    else:
        calibrate(a.out, [c.strip() for c in a.checks.split(",") if c.strip()], a.workers)


if __name__ == "__main__":
    main()
