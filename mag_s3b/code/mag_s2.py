#!/usr/bin/env python3
"""
MAG-S2 - the shape of the quantum compass signal.   Protocol version 0.1.

Physics. A radical pair (flavin anion with N5 and N10, tryptophan cation with N1; anisotropic hyperfine tensors from
the RadicalPy database) starts in the singlet state in a 50 uT field and recombines with rate k (exponential model).
Its singlet yield Phi(n) depends on the field direction n in the molecular frame. Phi is computed exactly from the
eigenbasis of the spin Hamiltonian, checked against a resolvent solution that needs no eigenbasis and against a
time-domain propagation with RadicalPy's own initial state, propagator and projection, and fitted by real spherical
harmonics of all degrees up to L (odd degrees included, so the fit itself can show any asymmetry).

Eye and neurons. Two lateral eyes; cryptochromes in each photoreceptor have their molecular z axis along the receptor
axis and their x axis along the local meridian with a small jitter (Ritz, Adem & Schulten 2000). Model neurons read
Gaussian patches of receptors (ON or OFF), pass the summed signal through a softplus and carry Gaussian noise whose SD
is a fixed fraction of each neuron's signal SD.

Measurement. The field is rotated over directions in the head frame (as with three-axis coils). The population
activity is z-scored, projected on its first six principal components and passed to ripser with Z2 and Z3
coefficients. A bar counts when it outlives both a share of the longest bar of the dataset and a multiple of the
longest bar of its dimension in shuffled copies (both factors calibrated on the atlas of known shapes). The Z2 Betti
numbers and orientability (the longest H2 bar keeps its length under Z3 only on an orientable surface) name the shape:
point, ring, sphere, torus, projective plane (RP2) or Klein bottle. Every seed also draws the atlas of known shapes,
plus two rings whose neurons carry the noise share of the two yaw conditions; these two rings gate the yaw hypothesis.

Modes
  physics      compute and store the yield pattern (physics.json) with its invariants
  calibrate    neutral calibration of the topology pipeline on the atlas of known shapes only
  smoke        tiny end-to-end technical check on seed 9307 (no shape verdicts on compass conditions)
  run          confirmatory run (only after the protocol is approved; needs an empty output folder)

BLAS runs on one thread: results differ between thread counts at the 1e-15 level, and one thread keeps reruns
bitwise identical on the same BLAS build and CPU kernel (both are recorded in the metadata).

Usage
  python3 -I mag_s2.py physics --out OUT
  python3 -I mag_s2.py calibrate --seeds 1001-1010 --out OUT
  python3 -I mag_s2.py smoke --out OUT
  python3 -I mag_s2.py run --seeds 1-10 --out OUT
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                              # before NumPy loads BLAS

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.special import sph_harm_y  # noqa: E402

SHAPES = ("point", "ring", "sphere", "torus", "rp2", "klein")
YAW_CONDITIONS = ("yaw_inc66", "yaw_inc0")
# Z2 Betti numbers (b1, b2) count the holes. Orientability comes from the longest H2 bar: an orientable surface has
# its fundamental class under every field, so the bar is the same under Z3; a non-orientable surface has it only
# under Z2. Z3 counts are reported but not used for the label.
SIGNATURES = {(0, 0, None): "point", (1, 0, None): "ring", (0, 1, True): "sphere", (2, 1, True): "torus",
              (1, 1, False): "rp2", (2, 1, False): "klein"}
ORIENTABLE_MIN = 0.9   # an orientable surface keeps its longest H2 bar under Z3 (ratio 1); a non-orientable one loses it
D_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Params:
    # radical pair
    B0_mT: float = 0.05            # geomagnetic field strength, 50 uT
    k_rec: float = 1.0e6           # recombination rate, s^-1 (exponential model)
    J: float = 0.0                 # exchange coupling
    D: float = 0.0                 # dipolar coupling
    L_sh: int = 10                 # maximal spherical-harmonic degree of the fit (all degrees)
    n_fit: int = 1000              # Fibonacci directions for the fit (antipodes added)
    # eyes
    eye_azimuth_deg: float = 60.0  # optical axes at +-60 degrees from forward, horizontal
    fov_half_deg: float = 80.0     # retina covers a cap of this half-angle around the optical axis
    n_receptors: int = 400         # receptor patches over both eyes
    jitter_deg: float = 10.0       # in-plane jitter of the molecular x axis
    # neurons
    n_neurons: int = 100
    rf_deg: float = 15.0           # SD of the Gaussian receptive field on the retina
    gain: float = 2.0
    bias_sd: float = 0.5
    noise: float = 0.5             # noise SD as a fraction of each neuron's signal SD
    # stimulus
    n_dirs: int = 600              # field directions per condition
    inclination_deg: float = 66.0  # field inclination for yaw-only rotations (central Europe)
    # topology
    pca_dim: int = 6
    smooth_k: int = 1              # neighbours averaged before thinning (1 = off)
    knn: int = 10
    density_drop: float = 0.0      # fraction of points with the sparsest neighbourhood removed (0 = off)
    n_perm: int = 200              # ripser greedy subsample
    n_shuffles: int = 2
    rel_factor: float = 0.4        # a bar counts when it outlives rel_factor x the longest bar of the dataset ...
    null_factor: float = 4.0       # ... and null_factor x the longest bar of its dimension in shuffled copies (calibrated)


def rng_for(*keys):
    """Independent stream per key tuple; all keys >= 1."""
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in keys]))


# --------------------------------------------------------------------------------------------------------------
# Radical-pair physics
# --------------------------------------------------------------------------------------------------------------
def spin_system(P: Params):
    from radicalpy.data import Molecule
    from radicalpy.simulation import Basis, HilbertSimulation, State
    flavin = Molecule.fromdb("flavin_anion", nuclei=["N5", "N10"])
    trp = Molecule.fromdb("tryptophan_cation", nuclei=["N1"])
    sim = HilbertSimulation([flavin, trp], basis=Basis.ST)
    H0 = sim.total_hamiltonian(B0=0.0, J=P.J, D=P.D, hfc_anisotropy=True)
    PS = sim.projection_operator(State.SINGLET)
    M = int(round(np.trace(PS).real))       # number of nuclear spin states
    return sim, H0, PS, M


def singlet_yield(system, P: Params, theta, phi):
    """Exact singlet yield of the exponential model: (1/M) sum_mn |<m|P_S|n>|^2 k^2 / (k^2 + w_mn^2)."""
    sim, H0, PS, M = system
    H = H0 + sim.zeeman_hamiltonian_3d(P.B0_mT, theta, phi)
    E, V = np.linalg.eigh(H)
    Pm = V.conj().T @ PS @ V
    w = E[:, None] - E[None, :]
    return float(np.sum(np.abs(Pm) ** 2 * P.k_rec ** 2 / (P.k_rec ** 2 + w ** 2)).real / M)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    ang = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([r * np.cos(ang), r * np.sin(ang), z])


def to_angles(n):
    theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))
    phi = np.arctan2(n[:, 1], n[:, 0])
    return theta, phi


def real_sh(L, n):
    """Real spherical harmonics up to degree L at unit vectors n; returns (basis, degree of each column)."""
    theta, phi = to_angles(n)
    cols, degs = [], []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                col = np.sqrt(2.0) * Y.real
            elif m < 0:
                col = np.sqrt(2.0) * Y.imag
            else:
                col = Y.real
            cols.append(col)
            degs.append(l)
    return np.column_stack(cols), np.array(degs)


def zeeman_cartesian(sim, P: Params, n):
    """Zeeman term built from the field vector n itself, without angles: -B0 sum_i gamma_i (n . S_i)."""
    return -P.B0_mT * sum(p.gamma_mT * sum(n[a] * sim.spin_operator(i, "xyz"[a]) for a in range(3))
                          for i, p in enumerate(sim.particles))


def yield_resolvent(system, P: Params, n):
    """Singlet yield from the resolvent rho = k (k + i[H, .])^-1 rho0, solved as a Sylvester equation (no eigenbasis)."""
    from scipy.linalg import solve_sylvester
    sim, H0, PS, M = system
    H = H0 + zeeman_cartesian(sim, P, n)
    eye = np.eye(H.shape[0])
    rho = solve_sylvester(P.k_rec / 2 * eye + 1j * H, P.k_rec / 2 * eye - 1j * H, P.k_rec * PS / np.trace(PS).real)
    return float(np.trace(PS @ rho).real)


def yield_time_domain(system, P: Params, theta, phi, lifetimes=20.0, dt=0.5e-9):
    """Singlet yield by stepping the density matrix with RadicalPy's own initial state, propagator and projection,
    weighting by k exp(-k t) and integrating with the trapezoid rule over `lifetimes` recombination lifetimes
    (the truncated tail is below exp(-lifetimes); one matrix in memory at a time)."""
    from radicalpy.simulation import State
    sim, H0, PS, M = system
    H = H0 + sim.zeeman_hamiltonian_3d(P.B0_mT, theta, phi)
    Up, Um = sim.unitary_propagator(H, dt)
    rho = sim.initial_density_matrix(State.SINGLET, H)
    Q = sim.observable_projection_operator(State.SINGLET)
    n_steps = int(round(lifetimes / (P.k_rec * dt)))
    p = np.empty(n_steps + 1)
    for i in range(n_steps + 1):
        if i:
            rho = Um @ rho @ Up
        p[i] = np.real(np.trace(Q @ rho))
    t = np.arange(n_steps + 1) * dt
    return float(P.k_rec * np.trapezoid(p * np.exp(-P.k_rec * t), dx=dt))


def yield_pattern(P: Params, check=True):
    """Yield over an antipodally closed grid, its spherical-harmonic fit and the physics invariants."""
    system = spin_system(P)
    base = fibonacci_sphere(P.n_fit)
    grid = np.vstack([base, -base])
    theta, phi = to_angles(grid)
    vals = np.array([singlet_yield(system, P, t, f) for t, f in zip(theta, phi)])
    antipodal = float(np.max(np.abs(vals[:P.n_fit] - vals[P.n_fit:])))
    B, degs = real_sh(P.L_sh, grid)
    coef, *_ = np.linalg.lstsq(B, vals, rcond=None)
    fit = B @ coef
    resid = float(np.sqrt(np.mean((fit - vals) ** 2)))
    var_parts = {l: float(np.sum(coef[degs == l] ** 2)) for l in range(1, P.L_sh + 1)}
    total = sum(var_parts.values())
    odd_fraction = float(sum(v for l, v in var_parts.items() if l % 2 == 1) / total) if total > 0 else float("nan")
    out = dict(coef=coef.tolist(), degrees=degs.tolist(), mean=float(vals.mean()), sd=float(vals.std()),
               min=float(vals.min()), max=float(vals.max()), relative_anisotropy=float((vals.max() - vals.min()) / vals.mean()),
               antipodal_max_diff=antipodal, fit_rms=resid, fit_rms_over_sd=float(resid / vals.std()),
               odd_power_fraction=odd_fraction, degree_power={str(l): v for l, v in var_parts.items()})
    if check:
        sim = system[0]
        test = uniform_dirs(rng_for(31, 1), 400)          # directions outside the fit grid, fixed stream
        t1, f1 = to_angles(test)
        t2, f2 = to_angles(-test)
        exact = np.array([singlet_yield(system, P, a, b) for a, b in zip(t1, f1)])
        exact_neg = np.array([singlet_yield(system, P, a, b) for a, b in zip(t2, f2)])
        fit_oos = real_sh(P.L_sh, test)[0] @ coef
        out["antipodal_max_diff"] = max(antipodal, float(np.max(np.abs(exact - exact_neg))))
        out["fit_oos_rms_over_sd"] = float(np.sqrt(np.mean((fit_oos - exact) ** 2)) / vals.std())
        zc = []
        for v in test[:5]:                                # the (theta, phi) convention against the field vector itself
            th, ph = to_angles(v[None, :])
            Hc = zeeman_cartesian(sim, P, v)
            zc.append(float(np.max(np.abs(sim.zeeman_hamiltonian_3d(P.B0_mT, th[0], ph[0]) - Hc)) / np.max(np.abs(Hc))))
        out["zeeman_cartesian_max_rel_diff"] = max(zc)
        out["resolvent_max_diff"] = float(max(abs(yield_resolvent(system, P, v) - exact[i]) for i, v in enumerate(test[:12])))
        td = [(np.pi / 2, 0.0), (np.pi / 2, np.pi / 2), (np.pi / 3, 1.0), (0.2, 0.5)]   # the last one near the pole
        out["radicalpy_max_diff"] = float(max(abs(yield_time_domain(system, P, t, f) - singlet_yield(system, P, t, f))
                                              for t, f in td))
    return out


class Pattern:
    def __init__(self, phys, P: Params):
        self.coef = np.array(phys["coef"])
        self.mean, self.sd, self.L = phys["mean"], phys["sd"], P.L_sh

    def __call__(self, n):
        """Yield at unit vectors n (..., 3) in the molecular frame."""
        shape = n.shape[:-1]
        B, _ = real_sh(self.L, n.reshape(-1, 3))
        return (B @ self.coef).reshape(shape)


# --------------------------------------------------------------------------------------------------------------
# Eyes, receptors and neurons
# --------------------------------------------------------------------------------------------------------------
def rotation_to(axis):
    """Rotation matrix taking the z axis to the unit vector axis."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    s, c = np.linalg.norm(v), float(np.dot(z, axis))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def retina(P: Params, rng):
    """Receptor axes v (head frame), molecular frames R (columns x, y, z in the head frame) and eye index."""
    az = np.deg2rad(P.eye_azimuth_deg)
    axes = [np.array([np.cos(az), np.sin(az), 0.0]), np.array([np.cos(az), -np.sin(az), 0.0])]
    per = P.n_receptors // 2
    cmin = np.cos(np.deg2rad(P.fov_half_deg))
    vs, eyes = [], []
    for e, a in enumerate(axes):
        cz = rng.uniform(cmin, 1.0, per)
        ang = rng.uniform(0.0, 2 * np.pi, per)
        sz = np.sqrt(1.0 - cz ** 2)
        local = np.column_stack([sz * np.cos(ang), sz * np.sin(ang), cz])
        vs.append(local @ rotation_to(a).T)
        eyes += [e] * per
    v = np.vstack(vs)
    up = np.array([0.0, 0.0, 1.0])
    R = np.zeros((len(v), 3, 3))
    jit = np.deg2rad(P.jitter_deg) * rng.standard_normal(len(v))
    for i, zi in enumerate(v):
        x0 = up - np.dot(up, zi) * zi
        x0 /= np.linalg.norm(x0)
        y0 = np.cross(zi, x0)
        xi = np.cos(jit[i]) * x0 + np.sin(jit[i]) * y0
        R[i] = np.column_stack([xi, np.cross(zi, xi), zi])
    return v, R, np.array(eyes)


def neurons(P: Params, rng, v, eyes):
    """Patch weights W (neurons x receptors), ON/OFF sign and bias."""
    W = np.zeros((P.n_neurons, len(v)))
    sigma = np.deg2rad(P.rf_deg)
    for j in range(P.n_neurons):
        e = rng.integers(2)
        idx = np.flatnonzero(eyes == e)
        c = v[rng.choice(idx)]
        ang = np.arccos(np.clip(v[idx] @ c, -1.0, 1.0))
        w = np.exp(-ang ** 2 / (2 * sigma ** 2))
        W[j, idx] = w / w.sum()
    sign = rng.choice([-1.0, 1.0], P.n_neurons)
    bias = P.bias_sd * rng.standard_normal(P.n_neurons)
    return W, sign, bias


def receptor_signals(kind, dirs, R, pattern):
    """Centred, unit-scale receptor signals (receptors x directions) for field directions dirs (M x 3, head frame)."""
    n_mol = np.einsum("ikj,mk->imj", R, dirs)      # R_i^T n for every receptor i and direction m
    if kind == "rp":
        return (pattern(n_mol) - pattern.mean) / pattern.sd
    if kind == "polar":
        return n_mol[..., 2] * np.sqrt(3.0)            # u_i . n, scaled to unit SD over the sphere
    raise ValueError(kind)


def softplus(x):
    return np.logaddexp(0.0, x)


def rates(z, W, sign, bias, P: Params):
    """Noiseless firing rates (directions x neurons)."""
    u = sign[:, None] * (W @ z)
    return softplus(P.gain * u + bias[:, None]).T


def uniform_dirs(rng, m):
    n = rng.standard_normal((m, 3))
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def yaw_dirs(psi, inclination_deg):
    inc = np.deg2rad(inclination_deg)
    return np.column_stack([np.cos(inc) * np.cos(psi), np.cos(inc) * np.sin(psi), -np.sin(inc) * np.ones_like(psi)])


# --------------------------------------------------------------------------------------------------------------
# Atlas of known shapes (synthetic population codes)
# --------------------------------------------------------------------------------------------------------------
def atlas_features(shape, m, rng):
    if shape == "ring":
        u = rng.uniform(0, 2 * np.pi, m)
        return np.column_stack([np.cos(u), np.sin(u), np.cos(2 * u), np.sin(2 * u)])
    if shape == "sphere":                               # linear in the direction, like the polar control
        return uniform_dirs(rng, m)
    if shape == "rp2":                                  # quadratic forms of the direction (Veronese surface)
        x, y, z = uniform_dirs(rng, m).T
        return np.column_stack([x * y, y * z, x * z, (x * x - y * y) / 2, (3 * z * z - 1) / (2 * np.sqrt(3))])
    if shape == "torus":
        u, v = rng.uniform(0, 2 * np.pi, (2, m))
        return np.column_stack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])
    if shape == "klein":
        u, v = rng.uniform(0, 2 * np.pi, (2, m))
        return np.column_stack([np.cos(2 * u), np.sin(2 * u), np.cos(v), np.sin(v) * np.cos(u), np.sin(v) * np.sin(u)])
    if shape == "point":
        return rng.standard_normal((m, 3))
    raise ValueError(shape)


def atlas_population(shape, P: Params, rng, noise_frac=None, noise_rng=None):
    """Synthetic population code of a known shape. Each neuron's noise SD is noise_frac (one number per neuron, or
    P.noise for all) times its own signal SD. Features and weights come from rng; the noise comes from noise_rng when
    given, otherwise from rng after the weights. So the same rng with its own noise_rng gives the same noiseless code
    with independent noise."""
    F = atlas_features(shape, P.n_dirs, rng)
    F = (F - F.mean(0)) / F.std(0)
    A = rng.standard_normal((P.n_neurons, F.shape[1])) / np.sqrt(F.shape[1])
    b = P.bias_sd * rng.standard_normal(P.n_neurons)
    r = softplus(P.gain * (F @ A.T) + b)
    frac = P.noise if noise_frac is None else np.asarray(noise_frac, float)
    return add_noise(r, frac * r.std(0), rng if noise_rng is None else noise_rng)


def add_noise(r, sd, rng):
    return r + sd[None, :] * rng.standard_normal(r.shape)


# --------------------------------------------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------------------------------------------
def preprocess(X, P: Params):
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    Y = Z @ Vt[:P.pca_dim].T
    if P.smooth_k > 1:                                  # local averaging over nearest neighbours
        idx = cKDTree(Y).query(Y, k=P.smooth_k)[1]
        Y = Y[idx].mean(axis=1)
    dk = cKDTree(Y).query(Y, k=P.knn + 1)[0][:, -1]
    keep = dk <= np.quantile(dk, 1.0 - P.density_drop)
    return Y[keep]


def lifetimes(Y, coeff, P: Params):
    from ripser import ripser
    dg = ripser(Y, maxdim=2, coeff=coeff, n_perm=min(P.n_perm, len(Y)))["dgms"]
    out = {}
    for k in (1, 2):
        d = dg[k]
        d = d[np.isfinite(d[:, 1])]
        out[k] = np.sort(d[:, 1] - d[:, 0])[::-1] if len(d) else np.zeros(0)
    return out


N_KEEP = 8   # longest bars kept per dimension; no signature needs more than two


def topo_raw(X, P: Params, rng):
    """Longest bar lifetimes of the data and the longest bar of each dimension in shuffled copies (Z2 and Z3)."""
    Y = preprocess(X, P)
    raw = dict(n_points=int(len(Y)), bars={}, null={})
    shuffled = [preprocess(np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])]), P)
                for _ in range(P.n_shuffles)]
    for p in (2, 3):
        life = lifetimes(Y, p, P)
        null = {1: 0.0, 2: 0.0}
        for Ys in shuffled:
            ls = lifetimes(Ys, p, P)
            for k in (1, 2):
                null[k] = max(null[k], float(ls[k][0]) if len(ls[k]) else 0.0)
        raw["bars"][f"Z{p}"] = {f"H{k}": [float(x) for x in life[k][:N_KEEP]] for k in (1, 2)}
        raw["null"][f"Z{p}"] = {f"H{k}": null[k] for k in (1, 2)}
    return raw


def decide(raw, rel_factor, null_factor):
    """Betti numbers and the shape they name. A bar counts when it outlives rel_factor x the longest bar of the
    dataset (any dimension, either field) and null_factor x the longest bar of its dimension in the shuffled copies."""
    longest = max([x for z in ("Z2", "Z3") for h in ("H1", "H2") for x in raw["bars"][z][h]] or [0.0])
    out = dict(thresholds={}, betti={}, longest=float(longest))
    for z in ("Z2", "Z3"):
        out["thresholds"][z] = {h: max(rel_factor * longest, null_factor * raw["null"][z][h]) for h in ("H1", "H2")}
        out["betti"][z] = {h: int(sum(x > out["thresholds"][z][h] for x in raw["bars"][z][h])) for h in ("H1", "H2")}
    b1, b2 = out["betti"]["Z2"]["H1"], out["betti"]["Z2"]["H2"]
    h2z2, h2z3 = raw["bars"]["Z2"]["H2"], raw["bars"]["Z3"]["H2"]
    ratio = float(h2z3[0] / h2z2[0]) if h2z2 and h2z3 and h2z2[0] > 0 else 0.0
    out["orientability_ratio"] = ratio
    orientable = None if b2 == 0 else bool(ratio >= ORIENTABLE_MIN)
    out["signature"] = [b1, b2, out["betti"]["Z3"]["H1"], out["betti"]["Z3"]["H2"]]
    out["shape"] = SIGNATURES.get((b1, b2, orientable), "other")
    return out


def classify(X, P: Params, rng):
    raw = topo_raw(X, P, rng)
    return {**raw, **decide(raw, P.rel_factor, P.null_factor)}


def heading_confusion(X, psi, rng):
    """Nearest-neighbour heading decoder on held-out samples: share of errors larger than 90 degrees."""
    idx = rng.permutation(len(psi))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    mu, sd = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1.0)
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    nn = cKDTree(A).query(B, k=1)[1]
    err = np.angle(np.exp(1j * (psi[tr][nn] - psi[te])))
    return float(np.mean(np.abs(err) > np.pi / 2))


def fingerprint(X):
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------------------------------------------
def compass_populations(P: Params, pattern, seed):
    """All compass conditions of one seed (one bird): name -> (rates with noise, extra info)."""
    v, R, eyes = retina(P, rng_for(seed, 1))
    W, sign, bias = neurons(P, rng_for(seed, 2), v, eyes)
    dirs = uniform_dirs(rng_for(seed, 3), P.n_dirs)
    z_rp = receptor_signals("rp", dirs, R, pattern)
    r_rp = rates(z_rp, W, sign, bias, P)
    sd_rp = P.noise * r_rp.std(0)                      # each neuron's noise, fixed by its intact signal
    out = {}
    for di, d in enumerate(D_LEVELS, start=1):
        r = rates((1.0 - d) * z_rp, W, sign, bias, P)
        out[f"rp_d{d:g}"] = add_noise(r, sd_rp, rng_for(seed, 4, di))
    out["rp_above"] = add_noise(r_rp, sd_rp, rng_for(seed, 4, 1))   # above threshold: same stream as d = 0
    z_pol = receptor_signals("polar", dirs, R, pattern)
    r_pol = rates(z_pol, W, sign, bias, P)
    out["polar"] = add_noise(r_pol, P.noise * r_pol.std(0), rng_for(seed, 5))
    psi = rng_for(seed, 6).uniform(0.0, 2 * np.pi, P.n_dirs)
    noise_frac = {}
    for hi, inc in enumerate((P.inclination_deg, 0.0), start=1):
        name = f"yaw_inc{inc:g}"
        r = rates(receptor_signals("rp", yaw_dirs(psi, inc), R, pattern), W, sign, bias, P)
        out[name] = add_noise(r, sd_rp, rng_for(seed, 7, hi))
        noise_frac[name] = sd_rp / np.maximum(r.std(0), 1e-12)   # noise SD in units of each neuron's own yaw signal SD
    noiseless_antipodal = float(np.max(np.abs(rates(receptor_signals("rp", -dirs, R, pattern), W, sign, bias, P) - r_rp)))
    return out, psi, noiseless_antipodal, noise_frac


DESIGN_NEURONS = (25, 50, 100)
DESIGN_NOISE = (0.25, 0.5, 1.0)


def design_populations(P: Params, pattern, seed, key):
    """Intact radical-pair compass and polar control for one design point (its own bird and noise)."""
    v, R, eyes = retina(P, rng_for(seed, 20, key, 1))
    W, sign, bias = neurons(P, rng_for(seed, 20, key, 2), v, eyes)
    dirs = uniform_dirs(rng_for(seed, 20, key, 3), P.n_dirs)
    out = {}
    for ki, kind in enumerate(("rp", "polar"), start=1):
        r = rates(receptor_signals(kind, dirs, R, pattern), W, sign, bias, P)
        out[kind] = add_noise(r, P.noise * r.std(0), rng_for(seed, 20, key, 3 + ki))
    return out


def run_seed(P: Params, phys, seed, with_atlas=True):
    pattern = Pattern(phys, P)
    pops, psi, antip, noise_frac = compass_populations(P, pattern, seed)
    res = dict(seed=seed, conditions={}, noiseless_antipodal_max_diff=antip)
    for ci, (name, X) in enumerate(pops.items(), start=1):
        rec = dict(fingerprint=fingerprint(X), finite=bool(np.all(np.isfinite(X))))
        if name != "rp_above":
            rec.update(classify(X, P, rng_for(seed, 8, ci)))
        if name.startswith("yaw"):
            rec["confusion_180"] = heading_confusion(X, psi, rng_for(seed, 9, ci))
        res["conditions"][name] = rec
    res["design"] = {}                                  # exploratory design map (E2), every seed
    for ni, nn in enumerate(DESIGN_NEURONS, start=1):
        for xi, nz in enumerate(DESIGN_NOISE, start=1):
            Pd = replace(P, n_neurons=nn, noise=nz)
            pops_d = design_populations(Pd, pattern, seed, 10 * ni + xi)
            for ki, (kind, X) in enumerate(pops_d.items(), start=1):
                rec = classify(X, Pd, rng_for(seed, 21, 10 * ni + xi, ki))
                res["design"][f"N{nn}_noise{nz:g}_{kind}"] = dict(shape=rec["shape"], signature=rec["signature"],
                                                                   orientability_ratio=rec["orientability_ratio"])
    if with_atlas:
        res["atlas"] = {}
        for si, shape in enumerate(SHAPES, start=1):
            X = atlas_population(shape, P, rng_for(seed, 10, si))
            rec = classify(X, P, rng_for(seed, 11, si))
            rec["finite"] = bool(np.all(np.isfinite(X)))
            res["atlas"][shape] = rec
        # gate rings for H3: the same noiseless ring as above (same stream) with independent noise from its own stream,
        # each neuron's noise set to the share that a randomly paired neuron of the yaw condition carries relative to
        # its own yaw signal
        for hi, name in enumerate(YAW_CONDITIONS, start=1):
            frac = noise_frac[name][rng_for(seed, 12, hi).permutation(P.n_neurons)]
            X = atlas_population("ring", P, rng_for(seed, 10, SHAPES.index("ring") + 1), noise_frac=frac,
                                 noise_rng=rng_for(seed, 13, hi))
            rec = classify(X, P, rng_for(seed, 11, len(SHAPES) + hi))
            rec["finite"] = bool(np.all(np.isfinite(X)))
            rec["median_snr"] = float(np.median(1.0 / frac))
            res["atlas"][f"ring_{name}"] = rec
    return res


# --------------------------------------------------------------------------------------------------------------
# Calibration (atlas only) and smoke test
# --------------------------------------------------------------------------------------------------------------
REL_FACTORS = (0.25, 0.30, 0.35, 0.40)
NULL_FACTORS = (2.0, 3.0, 4.0, 6.0)


HYPOTHESIS_SHAPES = ("point", "ring", "sphere", "rp2")


def calibrate(P: Params, seeds):
    """Atlas accuracy on a grid of the two threshold factors. Rule (round 3): among pairs where the shapes the
    hypotheses name (point, ring, sphere, rp2) all reach 90%, take the pair with the highest minimum accuracy over all
    six shapes; ties go to the pair closest to the grid centre (rel 0.325, null 4.0) in grid steps, then to the
    smaller factors."""
    raws = {shape: [] for shape in SHAPES}
    for seed in seeds:
        for si, shape in enumerate(SHAPES, start=1):
            X = atlas_population(shape, P, rng_for(seed, 10, si))
            raws[shape].append(topo_raw(X, P, rng_for(seed, 11, si)))
    rows, labels, ok = {}, {}, []
    for ri, rf in enumerate(REL_FACTORS):
        for ni, nf in enumerate(NULL_FACTORS):
            key = f"rel{rf:g}_null{nf:g}"
            labs = {shape: [decide(r, rf, nf)["shape"] for r in raws[shape]] for shape in SHAPES}
            acc = {shape: float(np.mean([lab == shape for lab in labs[shape]])) for shape in SHAPES}
            rows[key], labels[key] = acc, labs
            if all(acc[s] >= 0.9 for s in HYPOTHESIS_SHAPES):
                ok.append((-min(acc.values()), abs(ri - 1.5) + abs(ni - 2), rf, nf))
    chosen = sorted(ok)[0][2:] if ok else None
    return rows, labels, chosen, raws


SMOKE_SEED = 9307   # outside every calibration, pipeline-test and confirmatory seed


def smoke(P: Params, phys):
    issues = []
    sm = SMOKE_SEED
    Ps = replace(P, n_dirs=160, n_neurons=40, n_receptors=120, n_perm=80, n_shuffles=1)
    pattern = Pattern(phys, Ps)
    if not phys["antipodal_max_diff"] <= 1e-12:
        issues.append("radical-pair yield is not antipodally symmetric")
    pops, psi, antip, noise_frac = compass_populations(Ps, pattern, sm)
    if not antip <= 1e-9:
        issues.append("noiseless radical-pair population differs between n and -n")
    if fingerprint(pops["rp_above"]) != fingerprint(pops["rp_d0"]):
        issues.append("above-threshold control differs from the intact compass")
    for name, X in pops.items():
        if X.shape != (Ps.n_dirs, Ps.n_neurons) or not np.all(np.isfinite(X)):
            issues.append(f"bad population {name}")
    for name in YAW_CONDITIONS:
        f = noise_frac.get(name)
        if f is None or f.shape != (Ps.n_neurons,) or not np.all(np.isfinite(f)) or np.any(f <= 0):
            issues.append(f"bad noise share for {name}")
    same = atlas_population("ring", Ps, rng_for(sm, 3))
    rescaled = atlas_population("ring", Ps, rng_for(sm, 3), noise_frac=np.full(Ps.n_neurons, Ps.noise))
    if fingerprint(same) != fingerprint(rescaled):
        issues.append("a gate ring with the standard noise share differs from the atlas ring")
    v, R, eyes = retina(Ps, rng_for(sm, 1))
    W, sign, bias = neurons(Ps, rng_for(sm, 2), v, eyes)
    r_off = rates(np.zeros((Ps.n_receptors, 5)), W, sign, bias, Ps)
    if not np.allclose(r_off, r_off[0][None, :], atol=0.0, rtol=0.0):
        issues.append("fully disrupted compass still carries a signal")
    if not np.allclose(np.einsum("ikj,ikl->ijl", R, R), np.eye(3)[None], atol=1e-12):
        issues.append("molecular frames are not orthonormal")
    if not np.allclose(R[:, :, 2], v):
        issues.append("molecular z axis differs from the receptor axis")
    for shape in ("ring", "rp2"):
        rec = classify(atlas_population(shape, Ps, rng_for(sm, 3)), Ps, rng_for(sm, 4))
        if rec["shape"] not in SHAPES + ("other",):
            issues.append("classifier returned an unknown label")
    # the decoder on synthetic heading codes with known answers (no compass population involved)
    hd = rng_for(sm, 5).uniform(0.0, 2 * np.pi, 400)
    G = rng_for(sm, 6).standard_normal((2, 30))
    jitter = 0.05 * rng_for(sm, 7).standard_normal((400, 30))
    once = np.column_stack([np.cos(hd), np.sin(hd)]) @ G + jitter
    twice = np.column_stack([np.cos(2 * hd), np.sin(2 * hd)]) @ G + jitter
    if not heading_confusion(once, hd, rng_for(sm, 8)) < 0.05:
        issues.append("decoder confuses opposite headings on a once-around code")
    if not 0.35 <= heading_confusion(twice, hd, rng_for(sm, 9)) <= 0.65:
        issues.append("decoder separates opposite headings that a twice-around code merges")
    dp = design_populations(replace(Ps, n_neurons=25), pattern, sm, 11)
    if any(X.shape != (Ps.n_dirs, 25) or not np.all(np.isfinite(X)) for X in dp.values()):
        issues.append("bad design population")
    # orientability rule on a known answer: identical H2 bars under both fields mean an orientable surface
    raw = dict(bars={"Z2": {"H1": [], "H2": [5.0]}, "Z3": {"H1": [], "H2": [5.0]}},
               null={"Z2": {"H1": 0.1, "H2": 0.1}, "Z3": {"H1": 0.1, "H2": 0.1}})
    if decide(raw, 0.3, 3.0)["shape"] != "sphere":
        issues.append("orientable surface not labelled sphere")
    raw["bars"]["Z2"]["H1"] = [5.0]
    raw["bars"]["Z3"]["H2"] = [0.5]
    if decide(raw, 0.3, 3.0)["shape"] != "rp2":
        issues.append("non-orientable surface with one loop not labelled rp2")
    return issues


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def parse_seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_physics(here, P):
    path = os.path.join(here, "physics.json")
    with open(path) as fh:
        phys = json.load(fh)
    if phys.get("params_physics") != physics_params(P):
        sys.exit("physics.json was computed with other physics parameters; rerun the physics mode")
    return phys, file_sha256(path)


def physics_params(P):
    return {k: v for k, v in asdict(P).items() if k in ("B0_mT", "k_rec", "J", "D", "L_sh", "n_fit")}


def _seed_job(args):
    P, phys, seed, out = args
    r = run_seed(P, phys, seed)
    with open(os.path.join(out, f"seed_{seed}.json"), "w") as fh:
        json.dump(r, fh)
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["physics", "calibrate", "smoke", "run"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.mode == "run" and not a.seeds:
        sys.exit("run needs --seeds; the confirmatory run is --seeds 1-10")
    if a.mode == "run" and (glob.glob(os.path.join(a.out, "seed_*.json")) or os.path.exists(os.path.join(a.out, "meta_run.json"))):
        sys.exit(f"{a.out} already holds a run; use an empty folder")
    os.makedirs(a.out, exist_ok=True)
    P = Params()
    here = os.path.dirname(os.path.abspath(__file__))

    def sha_if(name):
        path = os.path.join(here, name)
        return file_sha256(path) if os.path.exists(path) else None

    import radicalpy
    import ripser
    meta = dict(code_sha256=sha_if("mag_s2.py"), analysis_sha256=sha_if("analyze.py"), tests_sha256=sha_if("test_analyze.py"),
                params=asdict(P), numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
                radicalpy=getattr(radicalpy, "__version__", None) or _pkg_version("radicalpy"),
                ripser=getattr(ripser, "__version__", None) or _pkg_version("ripser"), mode=a.mode,
                blas_threads=_blas_threads(), blas_libraries=_blas_libraries())
    t0 = time.time()
    if a.mode == "physics":
        phys = yield_pattern(P)
        phys["params_physics"] = physics_params(P)
        with open(os.path.join(here, "physics.json"), "w") as fh:
            json.dump(phys, fh, indent=1)
        meta["physics_sha256"] = sha_if("physics.json")
        print(json.dumps({k: v for k, v in phys.items() if k not in ("coef", "degrees")}, indent=1))
    elif a.mode == "calibrate":
        seeds = parse_seeds(a.seeds or "1001-1010")
        rows, labels, chosen, raws = calibrate(P, seeds)
        meta.update(seeds=seeds, atlas_accuracy=rows, atlas_labels=labels, chosen_factors=chosen, atlas_raw=raws)
        print(json.dumps(dict(atlas_accuracy=rows, chosen_factors=chosen), indent=1))
    elif a.mode == "smoke":
        phys, meta["physics_sha256"] = load_physics(here, P)
        issues = smoke(P, phys)
        meta["issues"] = issues
        print(json.dumps(dict(issues=issues), indent=1))
    else:
        phys, meta["physics_sha256"] = load_physics(here, P)
        seeds = parse_seeds(a.seeds)
        meta["seeds"] = seeds
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for s in pool.imap_unordered(_seed_job, [(P, phys, s, a.out) for s in seeds]):
                print(f"seed {s} done in {time.time() - t0:.0f}s", flush=True)
    meta["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, f"meta_{a.mode}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _blas_threads():
    """Thread counts of the BLAS libraries actually loaded (threadpoolctl), e.g. [1]."""
    try:
        from threadpoolctl import threadpool_info
        return sorted({int(d.get("num_threads", 0)) for d in threadpool_info() if d.get("user_api") == "blas"})
    except Exception:
        return None


def _blas_libraries():
    """Build, version and CPU kernel of each loaded BLAS library, for reproducibility records."""
    try:
        from threadpoolctl import threadpool_info
        keys = ("internal_api", "version", "architecture", "num_threads", "filepath")
        return [{k: d.get(k) for k in keys} for d in threadpool_info() if d.get("user_api") == "blas"]
    except Exception:
        return None


if __name__ == "__main__":
    main()
