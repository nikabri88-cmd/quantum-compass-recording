#!/usr/bin/env python3
"""
MAG-S3 - what an experiment needs to see the shape of the quantum compass.   Protocol version 0.4 (draft).

The bird is the bird of MAG-S2 (release v0.1.2, DOI 10.5281/zenodo.24030614): two lateral eyes, 400 receptor patches,
cryptochrome axes along the receptor axes, neurons reading Gaussian patches of one eye with an ON or OFF sign. MAG-S3
records it the way an electrophysiology experiment would and asks how many neurons, field directions and repeats are
needed to tell the radical-pair code (projective plane RP2) from a polar code (sphere).

Physics of the receptor. The central yield pattern follows Ren, Hiscock & Hore (2021): a dip of depth b and width c at
right angles to the molecular axis, Phi(theta) = a - b exp(-(theta - 90 deg)^2 / (2 c^2)), plus an odd admixture
eps b cos(theta) standing for chirality-induced spin selectivity (CISS). The exact three-nucleus pattern of MAG-S2 is the
comparison pattern. A CISS anchor recomputes the MAG-S2 pair with the initial and recombination state
cos(chi/2)|S> + sin(chi/2)|T0> (Luo & Hore 2021).

Neurons. The input of neuron i is the signed patch sum s_i(n) of receptor yields. Its rate in the counting window is
r_i(n) = r0_i (1 + delta u_i(n)), with u_i = (s_i - mid_i) / range_i over an antipodally closed grid of 2000 directions
and the presented directions, so the rate swings by delta r0_i from the worst to the best direction. Base rates are
log-normal (median 10 Hz, spread factor 2, clipped to 1-80 Hz). A share f of the recorded cells is tuned. Counts are
Poisson, or negative binomial with Fano factor 1.5.

Experiment. Head fixed; coils hold the field in one of K directions (K/2 opposite pairs), then turn it along a great
circle at constant 50 uT and at most 100 uT/s (2 rad/s). After each turn 0.5 s settle, then a counting window of T s.
Each block of K directions is repeated R times in a new random order. Recording modes: one simultaneous probe session,
a pseudo-population of probe sessions (80 cells each) or of tetrode sessions (6 cells each); every session has its own
order, its own shared gain noise (1 + g, g ~ N(0, sigma_g^2), per presentation) and a slow drift of each base rate.
The reference design has K = 800 directions (400 axes), R = 4 and T = 2 s, about 2.9 h per session (set by the
atlas calibration C6); the short-session map for the antipodal test has K = 100 and R = 8, about 0.7 h.

Readouts. With f < 1 cells are selected on the even repeats (ANOVA across directions with the repeats as blocks)
and read on the odd repeats. Topology: the frozen MAG-S2 pipeline (z-score, six principal components, ripser with Z2
and Z3, frozen bar thresholds, orientability ratio). Antipodal test: cross-validated odd and even energies over
opposite pairs, a sign-flip permutation test for the odd part and a bootstrap over cells and direction pairs for the
even/polar verdict.

Modes
  physics     Ren pattern checks (C1), CISS anchor and its check at chi = 0 (C2)
  calibrate   C3-C7 on seeds 1001-1010 (C7 on 2001-3000)
  smoke       small end-to-end technical check on seed 9311
  stage1      maps on seeds 1-10, Q3 at the depth frozen from C3 (resumable; --max-hours stops cleanly)
  stage2      hypothesis checks on seeds 101-200 at the working points chosen by analyze.py

Every record carries the SHA-256 of the code that wrote it, and every run folder gets invariants.json: I1 once, I2
and I4 for each of its birds. BLAS runs on one thread so that reruns are bitwise identical on the same build.

Usage
  python3 -I mag_s3.py physics --out OUT
  python3 -I mag_s3.py calibrate --out OUT [--checks C3,C4,C5,C6,C7]
  python3 -I mag_s3.py smoke --out OUT
  python3 -I mag_s3.py stage1 --out OUT [--workers 2] [--max-hours 1.8]
  python3 -I mag_s3.py stage2 --out OUT --points working_points.json [--workers 2] [--max-hours 1.8]
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
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen inputs from MAG-S2 v0.1.2. The fingerprint is the noisy intact-compass population of bird 1 as stored in
# results/confirmatory_run_v01/seed_1.json of that release (file SHA-256 9c7ed0e3ed949da6...).
MAG_S2_SHA256 = "2352aefe0fe5974732c318b365f4fb5477ffc47fb78b12748bca2515465829c4"
PHYSICS_SHA256 = "647bc0ccb1312991a1b772193cac39f4ced1481c591422c50659bb822a3100c5"
MAG_S2_BIRD1_FINGERPRINT = "570ca0da7267ef09bbdad03987dad068bf530aa2c2a66ce54a0e4492e3922bf9"
# Barcode, null bars and shape of that population as stored in the same file (extract: mag_s2_bird1.json). A build whose
# floating-point kernels differ in the last bits fails the fingerprint and is then judged by these within I1_TOL.
MAG_S2_BIRD1_FILE, MAG_S2_BIRD1_SHA256 = "mag_s2_bird1.json", "0b603a65362e999a0c1da09ee0d6edc0e2864dcd690a2f277978d541943b2d86"
I1_TOL = 1e-6


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_json_atomic(path, obj, **kw):
    """Write through a temporary file of this process, so that readers and other workers never see a partial file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def load_mag_s2():
    path = os.path.join(HERE, "mag_s2.py")
    if file_sha256(path) != MAG_S2_SHA256:
        sys.exit("mag_s2.py differs from the frozen MAG-S2 v0.1.2 file")
    if file_sha256(os.path.join(HERE, "physics.json")) != PHYSICS_SHA256:
        sys.exit("physics.json differs from the frozen MAG-S2 v0.1.2 file")
    spec = importlib.util.spec_from_file_location("mag_s2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mag_s2"] = mod
    spec.loader.exec_module(mod)
    return mod


m2 = load_mag_s2()
P2 = m2.Params()                  # frozen MAG-S2 parameters: eyes, receptors, patches, topology pipeline
rng_for = m2.rng_for
CODE_HASHES = dict(mag_s3=file_sha256(os.path.abspath(__file__)), mag_s2=MAG_S2_SHA256, physics=PHYSICS_SHA256)

# --------------------------------------------------------------------------------------------------------------
# Constants of the model and of the readouts
# --------------------------------------------------------------------------------------------------------------
REN_A, REN_B, REN_C_DEG = 0.5, 1.27e-3, 26.0     # Ren, Hiscock & Hore 2021
B_UT, DBDT_MAX_UT_S = 50.0, 100.0
OMEGA_MAX = DBDT_MAX_UT_S / B_UT                 # rad/s along a great circle at constant modulus
SETTLE_S = 0.5
CELLS_PER_SESSION = {"probe": 80, "tetrode": 6}
MODE_CODE = {"sim": 1, "probe": 2, "tetrode": 3}
R0_MEDIAN_HZ, R0_SPREAD, R0_MIN_HZ, R0_MAX_HZ = 10.0, 2.0, 1.0, 80.0
DRIFT_SD = 0.2                                   # SD of the change of log base rate over one session
GRID_HALF = 1000                                 # Fibonacci directions; with their antipodes 2000 grid directions
POOL_TUNED, POOL_UNTUNED, MAX_PAIRS = 400, 3600, 600
N_PERM, N_BOOT = 1000, 1000
RHO_EVEN_MAX, RHO_POLAR_MIN, ALPHA_ODD = 0.1, 0.9, 0.05
SELECT_ALPHA = {"p05": 0.05, "p01": 0.01}
BH_Q = 0.10
MIN_CELLS_TOPOLOGY = P2.pca_dim
ATLAS_ALL = ("point", "ring", "sphere", "torus", "rp2", "klein")
COND_CODE = {"compass": 1, "polar": 2, "point": 3, "ring": 4, "sphere": 5, "torus": 6, "rp2": 7, "klein": 8}
SHAPE_CODE = {s: i for i, s in enumerate(ATLAS_ALL, start=1)}
VERDICTS = ("even", "polar", "undetermined", "no_signal")


@dataclass(frozen=True)
class Design:
    delta: float = 1.0            # peak-to-peak rate modulation as a share of r0
    n_tuned: int = 200            # tuned cells recorded
    mode: str = "sim"             # sim, probe, tetrode
    sigma_g: float = 0.1          # shared gain noise per presentation within a session
    K: int = 800                  # directions (K/2 opposite pairs)
    R: int = 4                    # repeats of the block of K directions
    T: float = 2.0                # counting window, s
    f: float = 1.0                # share of tuned cells among the recorded ones
    select: str = "none"          # p05, p01 or bh10 when f < 1
    pattern: str = "ren"          # ren or mags2
    fano: float = 1.0
    w: float = 0.0                # polar admixture of the compass input
    eps: float = 0.0              # odd admixture of the Ren pattern (CISS)
    tau_a: float = 0.0            # decay of the field response during the hold, s (0 = none)
    drift: bool = True
    topology: bool = True
    antipodal: bool = True
    atlas: tuple = ("rp2", "sphere")

    @property
    def n_untuned(self):
        return int(round(self.n_tuned * (1.0 - self.f) / self.f)) if self.f < 1.0 else 0

    @property
    def n_rec(self):
        return self.n_tuned + self.n_untuned


def design_key(d: Design):
    """Stable integer of a design; keys the noise streams of its cell."""
    text = json.dumps(asdict(d), sort_keys=True)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


REFERENCE = Design()   # K = 800, R = 4, T = 2 s, Ren pattern, simultaneous with sigma_g = 0.1, delta = 1, 200 cells
ANTIPODAL_MAP = replace(REFERENCE, K=100, R=8, topology=False, atlas=())   # short sessions for the antipodal test


# --------------------------------------------------------------------------------------------------------------
# The bird
# --------------------------------------------------------------------------------------------------------------
def base_rates(rng, n):
    return np.clip(R0_MEDIAN_HZ * R0_SPREAD ** rng.standard_normal(n), R0_MIN_HZ, R0_MAX_HZ)


def closed_grid():
    g = m2.fibonacci_sphere(GRID_HALF)
    return np.vstack([g, -g])


class Bird:
    """Everything that belongs to one virtual bird (one seed) and is shared by all designs."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.v, self.R, self.eyes = m2.retina(P2, rng_for(seed, 1))
        self.W, self.sign, _ = m2.neurons(replace(P2, n_neurons=POOL_TUNED), rng_for(seed, 2), self.v, self.eyes)
        self.r0_tuned = base_rates(rng_for(seed, 3), POOL_TUNED)
        self.r0_untuned = base_rates(rng_for(seed, 4), POOL_UNTUNED)
        self.first = m2.uniform_dirs(rng_for(seed, 5), MAX_PAIRS)
        self.eta_tuned = rng_for(seed, 6).standard_normal(POOL_TUNED)
        self.eta_untuned = rng_for(seed, 7).standard_normal(POOL_UNTUNED)
        self.grid = closed_grid()
        self._pattern2 = None

    def dirs(self, K):
        h = self.first[:K // 2]
        return np.vstack([h, -h])                       # directions k and k + K/2 are opposite

    def r0(self, d: Design):
        return np.concatenate([self.r0_tuned[:d.n_tuned], self.r0_untuned[:d.n_untuned]])

    def eta(self, d: Design):
        return np.concatenate([self.eta_tuned[:d.n_tuned], self.eta_untuned[:d.n_untuned]])

    def pattern2(self):
        if self._pattern2 is None:
            with open(os.path.join(HERE, "physics.json")) as fh:
                self._pattern2 = m2.Pattern(json.load(fh), P2)
        return self._pattern2


# --------------------------------------------------------------------------------------------------------------
# Receptor yields, neuron inputs and modulation
# --------------------------------------------------------------------------------------------------------------
def ren_yield(cos_t, eps=0.0):
    """Ren et al. pattern of the angle theta between field and molecular axis. theta - 90 deg = -arcsin(cos theta), and
    arcsin is odd in IEEE arithmetic, so the even part is exactly even."""
    dev = np.arcsin(np.clip(cos_t, -1.0, 1.0))
    c = np.deg2rad(REN_C_DEG)
    return REN_A - REN_B * np.exp(-dev ** 2 / (2 * c * c)) + eps * REN_B * cos_t


def ren_signal(cos_t, eps=0.0):
    """(Phi - a) / b: the Ren et al. pattern without its constant, in units of the dip depth. The constant a is 400 times
    the dip and would cost precision in the patch sums; the normalised modulation does not depend on it."""
    dev = np.arcsin(np.clip(cos_t, -1.0, 1.0))
    c = np.deg2rad(REN_C_DEG)
    return -np.exp(-dev ** 2 / (2 * c * c)) + eps * cos_t


PATTERN_CHUNK = 200     # directions per call of the MAG-S2 pattern: about 0.3 GB instead of 3 GB, bitwise the same


def receptor_matrix(bird: Bird, kind, dirs, eps=0.0):
    """Receptor signals (receptors x directions)."""
    cos_t = bird.v @ dirs.T
    if kind == "ren":
        return ren_signal(cos_t, eps)
    if kind == "mags2":
        out = np.empty((len(bird.R), len(dirs)))
        for a in range(0, len(dirs), PATTERN_CHUNK):
            out[:, a:a + PATTERN_CHUNK] = m2.receptor_signals("rp", dirs[a:a + PATTERN_CHUNK], bird.R, bird.pattern2())
        return out
    if kind == "polar":
        return cos_t
    raise ValueError(kind)


def normalised(s):
    """(s - mid) / range per row; rows with no range give zero."""
    hi, lo = s.max(1, keepdims=True), s.min(1, keepdims=True)
    rng = hi - lo
    return np.where(rng > 0, (s - (hi + lo) / 2) / np.where(rng > 0, rng, 1.0), 0.0)


def neuron_input(bird: Bird, n, kind, dirs, eps=0.0):
    return bird.sign[:n, None] * (bird.W[:n] @ receptor_matrix(bird, kind, dirs, eps))


def modulation(bird: Bird, d: Design, cond):
    """u (tuned cells x K) for the compass (with polar and odd admixtures) or the polar control. The range is taken over
    the closed grid together with the presented directions, so |u| <= 1/2 at every presented direction."""
    dirs = bird.dirs(d.K)
    allx = np.vstack([bird.grid, dirs])
    n = d.n_tuned
    if cond == "polar":
        s = neuron_input(bird, n, "polar", allx)
    elif cond == "compass":
        s = neuron_input(bird, n, d.pattern, allx, d.eps if d.pattern == "ren" else 0.0)
        if d.w > 0:
            sp = neuron_input(bird, n, "polar", allx)
            ra = s.max(1, keepdims=True) - s.min(1, keepdims=True)
            rp = sp.max(1, keepdims=True) - sp.min(1, keepdims=True)
            s = (1 - d.w) * s / ra + d.w * sp / rp
    else:
        raise ValueError(cond)
    return normalised(s)[:, len(bird.grid):]


def atlas_modulation(bird: Bird, d: Design, shape):
    """u (tuned cells x K) of a synthetic code of known shape: random linear read-outs of standardised features.
    Sphere and RP2 use the bird's own directions (linear and quadratic forms); the other shapes draw latent samples."""
    K = d.K
    if shape == "sphere":
        F = bird.dirs(K)
    elif shape == "rp2":
        x, y, z = bird.dirs(K).T
        F = np.column_stack([x * y, y * z, x * z, (x * x - y * y) / 2, (3 * z * z - 1) / (2 * np.sqrt(3))])
    else:
        F = m2.atlas_features(shape, K, rng_for(bird.seed, 8, SHAPE_CODE[shape]))
    F = (F - F.mean(0)) / F.std(0)
    A = rng_for(bird.seed, 9, SHAPE_CODE[shape]).standard_normal((POOL_TUNED, F.shape[1]))[:d.n_tuned]
    return normalised(A @ F.T / np.sqrt(F.shape[1]))


# --------------------------------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------------------------------
def schedule(bird: Bird, d: Design):
    """Session of every recorded cell; per session the window-midpoint time of each presentation (k, r) as a share of the
    session, the unit gain noise z of that presentation, and the session duration in seconds."""
    n_rec, K, R = d.n_rec, d.K, d.R
    if d.mode == "sim":
        sess = np.zeros(n_rec, dtype=int)
    else:
        per = CELLS_PER_SESSION[d.mode]
        perm = rng_for(bird.seed, 21, MODE_CODE[d.mode], n_rec).permutation(n_rec)
        sess = np.empty(n_rec, dtype=int)
        sess[perm] = np.arange(n_rec) // per
    n_sess = int(sess.max()) + 1
    dirs = bird.dirs(K)
    frac = np.empty((n_sess, K, R))
    gz = np.empty((n_sess, K, R))
    dur = np.empty(n_sess)
    rep = np.repeat(np.arange(R), K)
    for s in range(n_sess):
        rng = rng_for(bird.seed, 20, MODE_CODE[d.mode], K, R, n_rec, s + 1)
        seq = np.concatenate([rng.permutation(K) for _ in range(R)])
        cosang = np.clip(np.sum(dirs[seq[1:]] * dirs[seq[:-1]], axis=1), -1.0, 1.0)
        turn = np.concatenate([[0.0], np.arccos(cosang) / OMEGA_MAX])
        start = np.cumsum(turn + SETTLE_S) + np.arange(len(seq)) * d.T
        total = float(np.sum(turn + SETTLE_S + d.T))
        frac[s, seq, rep] = (start + d.T / 2) / total
        gz[s, seq, rep] = rng.standard_normal(len(seq))
        dur[s] = total
    return sess, frac, gz, dur


def expected_counts(bird: Bird, d: Design, u, sched):
    """lambda (recorded cells x K x R); u covers the tuned cells, untuned cells have no modulation."""
    sess, frac, gz, _ = sched
    n_rec = d.n_rec
    mod = np.zeros((n_rec, d.K))
    if d.tau_a > 0:
        keep = d.tau_a * (np.exp(-SETTLE_S / d.tau_a) - np.exp(-(SETTLE_S + d.T) / d.tau_a)) / d.T
        mod[:d.n_tuned] = d.delta * u * keep
    else:
        mod[:d.n_tuned] = d.delta * u
    base = bird.r0(d)[:, None, None]
    if d.drift:
        base = base * np.exp(DRIFT_SD * bird.eta(d)[:, None, None] * (frac[sess] - 0.5))
    gain = np.maximum(1.0 + d.sigma_g * gz[sess], 0.0)
    return base * (1.0 + mod[:, :, None]) * gain * d.T


def draw_counts(lam, fano, rng):
    if fano == 1.0:
        return rng.poisson(lam)
    return rng.poisson(rng.gamma(lam / (fano - 1.0), fano - 1.0))


# --------------------------------------------------------------------------------------------------------------
# Readouts
# --------------------------------------------------------------------------------------------------------------
def anova_pvalues(C):
    """ANOVA across directions per cell with the repeats as blocks (two-way, no interaction); C is (cells x K x
    repeats). Each repeat covers its own part of the session, so a drift of the base rate shifts whole repeats; taking
    the repeat means out of the residual keeps that shift out of the test."""
    n, K, r = C.shape
    grand = C.mean(axis=(1, 2), keepdims=True)
    mk = C.mean(axis=2, keepdims=True)
    mr = C.mean(axis=1, keepdims=True)
    ssb = r * np.sum((mk - grand) ** 2, axis=(1, 2))
    ssr = np.sum((C - mk - mr + grand) ** 2, axis=(1, 2))
    df1, df2 = K - 1, (K - 1) * (r - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / df1) / (ssr / df2)
    F = np.where(ssr > 0, F, np.where(ssb > 0, np.inf, 0.0))
    return stats.f.sf(F, df1, df2)


def bh_reject(p, q):
    m = len(p)
    order = np.argsort(p, kind="stable")
    ok = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, dtype=bool)
    if ok.any():
        keep[order[:np.flatnonzero(ok).max() + 1]] = True
    return keep


def select_cells(C, d: Design):
    """Selected cells and the repeats left for reading. With f = 1 every cell and every repeat is read."""
    if d.f >= 1.0 or d.select == "none":
        return np.ones(C.shape[0], dtype=bool), np.arange(d.R), None
    sel_rep, ana_rep = np.arange(0, d.R, 2), np.arange(1, d.R, 2)
    p = anova_pvalues(C[:, :, sel_rep].astype(float))
    if d.select in SELECT_ALPHA:
        keep = p < SELECT_ALPHA[d.select]
    elif d.select == "bh10":
        keep = bh_reject(p, BH_Q)
    else:
        raise ValueError(d.select)
    return keep, ana_rep, p


def fingerprint(X):
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


def topology_record(X, rng):
    if X.shape[1] < MIN_CELLS_TOPOLOGY:
        return dict(shape="too_few", n_cells=int(X.shape[1]))
    rec = m2.classify(X, P2, rng)
    rec["n_cells"] = int(X.shape[1])
    return rec


def antipodal_test(C, sess, rng_perm, rng_boot, n_perm=N_PERM, n_boot=N_BOOT):
    """C: counts (cells x K x analysis repeats); sess: session of each cell. Returns energies, the permutation p value of
    the odd part and the verdict even / polar / undetermined / no_signal.

    For each cell and opposite pair, the even part is half the sum and the odd part half the difference of the counts at
    n and -n. Energies are cross products of the two halves of the repeats (A and B), each half centred over directions.
    Each cell is weighted by the inverse variance of all its analysis counts; swapping the labels n and -n inside a
    repeat leaves that variance unchanged, so one fixed weight per cell serves the data and every permutation."""
    n, K, Ra = C.shape
    out = dict(n_cells=int(n), n_repeats=int(Ra))
    if n == 0 or Ra < 2:
        out.update(E_odd=0.0, E_even=0.0, rho=None, p_odd=1.0, odd_found=False, verdict="no_signal",
                   gate_q05=None, d1_q95=None, d2_q05=None)
        return out
    P = K // 2
    A, B = np.arange(Ra // 2), np.arange(Ra // 2, Ra)
    C = C.astype(float)
    var = C.reshape(n, -1).var(1)
    wgt = 1.0 / np.where(var > 0, var, 1.0)
    Sp, Dp = C[:, :P] + C[:, P:], C[:, :P] - C[:, P:]
    SA, SB = Sp[..., A].mean(-1), Sp[..., B].mean(-1)
    eA = SA / 2 - SA.mean(-1, keepdims=True) / 2
    eB = SB / 2 - SB.mean(-1, keepdims=True) / 2
    oA, oB = Dp[..., A].mean(-1) / 2, Dp[..., B].mean(-1) / 2
    Modd = wgt[:, None] * oA * oB
    Meven = wgt[:, None] * eA * eB
    e_odd, e_even = float(Modd.sum()), float(Meven.sum())
    # sign-flip permutations: per session, pair and repeat, the same flip for every cell of the session.
    # E_odd under flips eps is sum_s,p eps_A^T G_sp eps_B with G_sp = sum over the cells of s of w D_A D_B^T / (4 |A| |B|).
    _, sess_idx = np.unique(sess, return_inverse=True)
    n_sess = int(sess_idx.max()) + 1
    G = np.zeros((n_sess, P, len(A), len(B)))
    np.add.at(G, sess_idx, wgt[:, None, None, None] * Dp[:, :, A, None] * Dp[:, :, None, B])
    G /= 4.0 * len(A) * len(B)
    chunk = max(1, int(2e6 // (n_sess * P * Ra)))
    null = np.empty(n_perm)
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        flips = rng_perm.integers(0, 2, size=(b, n_sess, P, Ra), dtype=np.int8).astype(float) * 2 - 1
        fa, fb = flips[..., A], flips[..., B]
        null[done:done + b] = np.einsum("bspa,spac,bspc->b", fa, G, fb, optimize=True)
        done += b
    p_odd = float((1 + np.sum(null >= e_odd - 1e-12 * abs(e_odd))) / (1 + n_perm))
    # bootstrap over cells and over direction pairs
    wc = rng_boot.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    wp = rng_boot.multinomial(P, np.full(P, 1.0 / P), size=n_boot).astype(float)
    eo = np.sum((wc @ Modd) * wp, axis=1)
    ee = np.sum((wc @ Meven) * wp, axis=1)
    gate = float(np.percentile(eo + ee, 5))
    d1 = float(np.percentile((1 - RHO_EVEN_MAX) * eo - RHO_EVEN_MAX * ee, 95))
    d2 = float(np.percentile((1 - RHO_POLAR_MIN) * eo - RHO_POLAR_MIN * ee, 5))
    if gate <= 0:
        verdict = "no_signal"
    elif d1 < 0:
        verdict = "even"
    elif d2 > 0:
        verdict = "polar"
    else:
        verdict = "undetermined"
    total = e_odd + e_even
    out.update(E_odd=e_odd, E_even=e_even, rho=float(e_odd / total) if total > 0 else None, p_odd=p_odd,
               odd_found=bool(p_odd <= ALPHA_ODD), verdict=verdict, gate_q05=gate, d1_q95=d1, d2_q05=d2)
    return out


# --------------------------------------------------------------------------------------------------------------
# One design for one bird
# --------------------------------------------------------------------------------------------------------------
def record_condition(bird: Bird, d: Design, u, sched, code, key):
    lam = expected_counts(bird, d, u, sched)
    C = draw_counts(lam, d.fano, rng_for(bird.seed, 30, key, code))
    noiseless = bird.r0(d)[:d.n_tuned, None] * (1.0 + d.delta * u)
    checks = dict(counts_ok=bool(np.issubdtype(C.dtype, np.integer) and C.min() >= 0),
                  rates_ok=bool(np.all(np.isfinite(lam)) and lam.min() >= 0 and noiseless.min() > 0))
    keep, ana, _ = select_cells(C, d)
    return C, keep, ana, checks


def simulate(d: Design, seed, bird=None, conditions=("compass", "polar"), shuffled=True):
    """Record one design for one bird and read it out."""
    bird = bird or Bird(seed)
    key = design_key(d)
    sched = schedule(bird, d)
    res = dict(seed=int(seed), cost=dict(n_recorded=d.n_rec, n_sessions=int(len(sched[3])),
                                         hours=float(sched[3].sum() / 3600.0)), conditions={})
    todo = list(conditions) + [s for s in d.atlas if d.topology]
    for cond in todo:
        code = COND_CODE[cond]
        u = modulation(bird, d, cond) if cond in ("compass", "polar") else atlas_modulation(bird, d, cond)
        C, keep, ana, checks = record_condition(bird, d, u, sched, code, key)
        Csel = C[keep][:, :, ana]
        X = Csel.mean(2).T
        rec = dict(checks, n_selected=int(keep.sum()), n_tuned_selected=int(keep[:d.n_tuned].sum()),
                   n_untuned_selected=int(keep[d.n_tuned:].sum()), fingerprint=fingerprint(X),
                   finite=bool(np.all(np.isfinite(X))))
        if d.topology:
            rec["topology"] = topology_record(X, rng_for(seed, 31, key, code))
        if d.antipodal and cond in ("compass", "polar"):
            rec["antipodal"] = antipodal_test(Csel, sched[0][keep], rng_for(seed, 33, key, code),
                                              rng_for(seed, 34, key, code))
        res["conditions"][cond] = rec
        if cond == "compass" and d.topology and shuffled:
            prng = rng_for(seed, 32, key)
            Xs = np.column_stack([X[prng.permutation(X.shape[0]), j] for j in range(X.shape[1])]) \
                if X.shape[1] else X
            res["conditions"]["shuffled"] = dict(topology=topology_record(Xs, rng_for(seed, 35, key)),
                                                 fingerprint=fingerprint(Xs), finite=bool(np.all(np.isfinite(Xs))))
    return res


# --------------------------------------------------------------------------------------------------------------
# Invariants computed with the run
# --------------------------------------------------------------------------------------------------------------
def mag_s2_bird1_check():
    """I1: the frozen MAG-S2 code reproduces bird 1 of the MAG-S2 confirmatory run in this environment. The noisy
    intact-compass population is compared bitwise with its stored fingerprint, and its barcode, null bars and shape,
    recomputed with the MAG-S2 stream of that condition, are compared with the stored ones."""
    with open(os.path.join(HERE, "physics.json")) as fh:
        phys = json.load(fh)
    pops, _, _, _ = m2.compass_populations(P2, m2.Pattern(phys, P2), 1)
    X = pops["rp_d0"]
    out = dict(fingerprint_ok=bool(m2.fingerprint(X) == MAG_S2_BIRD1_FINGERPRINT),
               mag_s2_sha256=file_sha256(os.path.join(HERE, "mag_s2.py")),
               physics_sha256=file_sha256(os.path.join(HERE, "physics.json")),
               bird1_file_sha256=None, bars_max_abs_diff=None, null_max_abs_diff=None, shape=None, shape_stored=None)
    path = os.path.join(HERE, MAG_S2_BIRD1_FILE)
    if not os.path.exists(path):
        return out
    out["bird1_file_sha256"] = file_sha256(path)
    with open(path) as fh:
        ref = json.load(fh)
    rec = m2.classify(X, P2, rng_for(1, 8, list(pops).index("rp_d0") + 1))
    worst = 0.0
    for z in ("Z2", "Z3"):
        for h in ("H1", "H2"):
            a, b = rec["bars"][z][h], ref["bars"][z][h]
            worst = max(worst, float("inf") if len(a) != len(b) else max([abs(x - y) for x, y in zip(a, b)] or [0.0]))
    out["bars_max_abs_diff"] = worst
    out["null_max_abs_diff"] = max(abs(rec["null"][z][h] - ref["null"][z][h]) for z in ("Z2", "Z3") for h in ("H1", "H2"))
    out["shape"], out["shape_stored"] = rec["shape"], ref["shape"]
    return out


def lambda_by_time(bird: Bird, d: Design, u):
    """Expected counts rebuilt presentation by presentation in time order, straight from the random streams of the
    protocol and without schedule(): session of every cell, the R block orders, turns along great circles at OMEGA_MAX,
    the settle time, window midpoints, each base rate's drift and the gain noise of every presentation (I4)."""
    n_rec, K, R = d.n_rec, d.K, d.R
    if d.mode == "sim":
        groups = [np.arange(n_rec)]
    else:
        per = CELLS_PER_SESSION[d.mode]
        perm = rng_for(bird.seed, 21, MODE_CODE[d.mode], n_rec).permutation(n_rec)
        groups = [perm[i:i + per] for i in range(0, n_rec, per)]
    dirs = bird.dirs(K)
    r0, eta = bird.r0(d), bird.eta(d)
    keep = 1.0
    if d.tau_a > 0:
        keep = d.tau_a * (np.exp(-SETTLE_S / d.tau_a) - np.exp(-(SETTLE_S + d.T) / d.tau_a)) / d.T
    mod = np.zeros((n_rec, K))
    mod[:d.n_tuned] = d.delta * u * keep
    lam = np.full((n_rec, K, R), np.nan)
    gain = np.full((n_rec, K, R), np.nan)
    for s, cells in enumerate(groups):
        rng = rng_for(bird.seed, 20, MODE_CODE[d.mode], K, R, n_rec, s + 1)
        orders = [rng.permutation(K) for _ in range(R)]
        z = rng.standard_normal(K * R)
        t, prev, slots = 0.0, None, []
        for r, order in enumerate(orders):
            for k in order:
                if prev is not None:          # the cosine as schedule() forms it: arccos is ill-conditioned at -1
                    t += float(np.arccos(np.clip(np.sum(dirs[k] * dirs[prev]), -1.0, 1.0))) / OMEGA_MAX
                t += SETTLE_S
                slots.append((int(k), r, t + d.T / 2))
                t += d.T
                prev = k
        for j, (k, r, mid) in enumerate(slots):
            g = max(1.0 + d.sigma_g * float(z[j]), 0.0)
            drift = np.exp(DRIFT_SD * eta[cells] * (mid / t - 0.5)) if d.drift else 1.0
            lam[cells, k, r] = r0[cells] * drift * (1.0 + mod[cells, k]) * g * d.T
            gain[cells, k, r] = g
    return lam, gain, groups


def i4_rebuild(bird: Bird, d: Design):
    """Compare the vectorised recording with the time-ordered rebuild: largest relative difference of lambda, and
    whether sessions, the gain factors and their sharing (same within a session, different between sessions) agree."""
    u = modulation(bird, d, "compass")
    sched = schedule(bird, d)
    sess, _, gz, _ = sched
    lam_v = expected_counts(bird, d, u, sched)
    lam_t, gain_t, groups = lambda_by_time(bird, d, u)
    rel = float(np.max(np.abs(lam_v - lam_t)) / np.max(np.abs(lam_t)))
    gain_v = np.maximum(1.0 + d.sigma_g * gz[sess], 0.0)
    ok = bool(sum(len(g) for g in groups) == d.n_rec and all(np.all(sess[g] == s) for s, g in enumerate(groups)))
    ok &= bool(np.array_equal(gain_v, gain_t))
    ok &= all(bool(np.all(gain_v[g] == gain_v[g[0]])) for g in groups)
    if d.sigma_g > 0:
        firsts = [gain_v[g[0]] for g in groups]
        ok &= all(not np.array_equal(firsts[a], firsts[b]) for a in range(len(firsts)) for b in range(a))
    return rel, ok


I4_DESIGNS = tuple(replace(REFERENCE, mode=m) for m in MODE_CODE) + \
    (replace(REFERENCE, mode="tetrode", n_tuned=20, f=0.1, select="p05", K=100, tau_a=2.0),)


def invariants_for_seed(seed):
    """I2 and I4 for one bird (I1 once per run, I3 and I5 from every record)."""
    bird = Bird(seed)
    d = REFERENCE
    out = {}
    K2 = d.K // 2
    for name, dd in (("ren", d), ("mags2", replace(d, pattern="mags2"))):
        u = modulation(bird, dd, "compass")
        out[f"I2_{name}_max_diff"] = float(np.max(np.abs(u[:, :K2] - u[:, K2:])))
    up = modulation(bird, d, "polar")
    out["I2_polar_max_sum"] = float(np.max(np.abs(up[:, :K2] + up[:, K2:])))
    # I4a: no gain noise, no drift -> the three recording modes give the same mean responses
    base = replace(d, sigma_g=0.0, drift=False, topology=False, antipodal=False)
    X = {}
    for mode in MODE_CODE:
        dm = replace(base, mode=mode)
        sched = schedule(bird, dm)
        lam = expected_counts(bird, dm, modulation(bird, dm, "compass"), sched)
        X[mode] = draw_counts(lam, 1.0, rng_for(seed, 40)).mean(2)
    out["I4_same_means"] = bool(np.array_equal(X["sim"], X["probe"]) and np.array_equal(X["sim"], X["tetrode"]))
    # I4b: with drift and gain noise on, the recording equals its time-ordered rebuild
    rels, oks = zip(*(i4_rebuild(bird, dd) for dd in I4_DESIGNS))
    out["I4_rebuild_max_rel_diff"] = float(max(rels))
    out["I4_sessions_ok"] = bool(all(oks))
    return out


def _inv_job(seed):
    return seed, invariants_for_seed(seed)


def write_invariants(folder, seeds, workers):
    """invariants.json of a run folder: I1 once, I2 and I4 for every bird of the run. Entries computed by the same code
    are kept, so a resumed or extended run adds only the missing birds; another code version starts afresh."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    path = os.path.join(folder, "invariants.json")
    inv = {}
    if os.path.exists(path):
        with open(path) as fh:
            inv = json.load(fh)
    if inv.get("code") != CODE_HASHES:
        inv = dict(code=CODE_HASHES)
    if "I1" not in inv:
        inv["I1"] = mag_s2_bird1_check()
    per = inv.setdefault("per_seed", {})
    todo = [s for s in seeds if str(s) not in per]
    if todo:
        try:
            with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as ex:
                for seed, v in ex.map(_inv_job, todo):
                    per[str(seed)] = v
        except BrokenProcessPool:
            sys.exit("a worker process died while computing the invariants (out of memory?); rerun with fewer --workers")
    inv["per_seed"] = {k: per[k] for k in sorted(per, key=int)}
    write_json_atomic(path, inv, indent=1)
    return inv


# --------------------------------------------------------------------------------------------------------------
# Physics: Ren pattern (C1) and the CISS anchor (C2, E5)
# --------------------------------------------------------------------------------------------------------------
CHI_LEVELS = (0.0, 0.02, 0.05, 0.1)          # in units of pi


def ciss_projector(sim, chi):
    """|psi><psi| (x) 1 for psi = cos(chi/2)|S> + sin(chi/2)|T0>, with S_Az - S_Bz = |S><T0| + |T0><S|."""
    PS = sim.projection_operator(m2_state("SINGLET"))
    PT0 = sim.projection_operator(m2_state("TRIPLET_ZERO"))
    cross = sim.spin_operator(0, "z") - sim.spin_operator(1, "z")
    c, s = np.cos(chi / 2), np.sin(chi / 2)
    return c * c * PS + s * s * PT0 + c * s * cross


def m2_state(name):
    from radicalpy.simulation import State
    return getattr(State, name)


def hamiltonian(system, n):
    sim, H0, _, _ = system
    th, ph = m2.to_angles(n[None, :])
    return H0 + sim.zeeman_hamiltonian_3d(P2.B0_mT, th[0], ph[0])


def ciss_yield(system, Pchi, n):
    """Yield of the exponential model (spin-independent decay at rate k) for initial and recombination state Pchi / M at
    field direction n (molecular frame): (1/M) sum_mn |<m|Pchi|n>|^2 k^2 / (k^2 + w_mn^2)."""
    M = system[3]
    E, V = np.linalg.eigh(hamiltonian(system, n))
    Pm = V.conj().T @ Pchi @ V
    w = E[:, None] - E[None, :]
    return float(np.sum(np.abs(Pm) ** 2 * P2.k_rec ** 2 / (P2.k_rec ** 2 + w ** 2)).real / M)


def haberkorn_yields(system, n, P_init, P_rec, k_rec, k_rest, k_fwd):
    """Haberkorn master equation d rho/dt = -i[H, rho] - k_rec/2 {P_rec, rho} - k_rest/2 {1 - P_rec, rho} - k_fwd rho with
    rho(0) = P_init / Tr P_init. The time integral of rho solves a Sylvester equation. Returns the yields of the three
    channels (recombination through P_rec, the complementary channel, the spin-independent forward reaction)."""
    from scipy.linalg import solve_sylvester
    H = hamiltonian(system, n)
    eye = np.eye(H.shape[0])
    G = 0.5 * (k_rec * P_rec + k_rest * (eye - P_rec) + k_fwd * eye)
    rho_int = solve_sylvester(1j * H + G, -1j * H + G, P_init / np.trace(P_init).real)
    return (float(k_rec * np.trace(P_rec @ rho_int).real), float(k_rest * np.trace((eye - P_rec) @ rho_int).real),
            float(k_fwd * np.trace(rho_int).real))


def haberkorn_time_domain(system, n, P_init, P_rec, k_rec, k_fwd, dt=5e-10, lifetimes=25.0):
    """Forward yield by stepping the same master equation in time with the exact step of the effective Hamiltonian
    H - i k_rec/2 P_rec (complementary channel off) and integrating with the trapezoid rule; checks the Sylvester
    solution."""
    from scipy.linalg import expm
    H = hamiltonian(system, n)
    Heff = H - 0.5j * k_rec * P_rec
    U = expm(-1j * Heff * dt) * np.exp(-0.5 * k_fwd * dt)
    rho = P_init / np.trace(P_init).real
    k_tot = k_fwd
    steps = int(round(lifetimes / (k_tot * dt)))
    tr = np.empty(steps + 1)
    for i in range(steps + 1):
        if i:
            rho = U @ rho @ U.conj().T
        tr[i] = np.trace(rho).real
    return float(k_fwd * np.trapezoid(tr, dx=dt))


# (k_fwd, k_rec) in s^-1: the MAG-S2 rate for both channels, and the FAD-TrpD pair of ErCry4a with recombination
# 3.4e5 s^-1 and a forward reaction of 5e6-1e7 s^-1 (Xu et al. 2021, rates as summarised by Wong et al. 2021)
ANCHOR_RATES = ((1.0e6, 1.0e6), (5.0e6, 3.4e5), (1.0e7, 3.4e5))


def pattern_parts(f, grid):
    """Even and odd parts of a yield function over an antipodally closed grid and the eps of the Ren model with the same
    ratio: (odd range / 2) / even range."""
    yp = np.array([f(n) for n in grid])
    ym = np.array([f(-n) for n in grid])
    even = np.concatenate([(yp + ym) / 2] * 2)
    odd = np.concatenate([(yp - ym) / 2, (ym - yp) / 2])
    er, orr = float(even.max() - even.min()), float(odd.max() - odd.min())
    return dict(mean=float(even.mean()), even_range=er, odd_range=orr, eps_equivalent=orr / 2 / er if er > 0 else None,
                odd_to_even_sd=float(odd.std() / even.std()) if even.std() > 0 else None)


def physics_checks():
    out = {}
    th = np.linspace(0.0, np.pi, 180001)
    phi = ren_yield(np.cos(th))
    out["C1_min_theta_deg"] = float(np.rad2deg(th[np.argmin(phi)]))
    out["C1_depth_over_a"] = REN_B / REN_A
    out["C1_relative_range"] = float((phi.max() - phi.min()) / phi.mean())
    x = np.linspace(-1.0, 1.0, 200001)
    out["C1_even_max_diff"] = float(np.max(np.abs(ren_yield(x) - ren_yield(-x))))
    out["C1_signal_even_max_diff"] = float(np.max(np.abs(ren_signal(x) - ren_signal(-x))))
    system = m2.spin_system(P2)
    sim, PS = system[0], system[2]
    test = m2.uniform_dirs(rng_for(41, 1), 100)
    P0 = ciss_projector(sim, 0.0)
    out["C2_projector_max_diff_chi0"] = float(np.max(np.abs(P0 - PS)))
    th2, ph2 = m2.to_angles(test)
    ref = np.array([m2.singlet_yield(system, P2, a, b) for a, b in zip(th2, ph2)])
    # the Haberkorn solver with equal rates in the singlet and triplet channels is the exponential model of MAG-S2
    eq = np.array([haberkorn_yields(system, n, P0, P0, P2.k_rec, P2.k_rec, 0.0)[0] for n in test])
    out["C2_equal_rate_limit_max_diff_vs_mag_s2"] = float(np.max(np.abs(eq - ref)))
    kf, kr = ANCHOR_RATES[0]
    lh = np.array([haberkorn_yields(system, n, P0, P0, kr, 0.0, kf)[2] for n in test[:20]])
    lhn = np.array([haberkorn_yields(system, -n, P0, P0, kr, 0.0, kf)[2] for n in test[:20]])
    out["C2_odd_max_chi0"] = float(np.max(np.abs(lh - lhn)) / 2)
    Pc = ciss_projector(sim, 0.1 * np.pi)
    sy = haberkorn_yields(system, test[0], Pc, Pc, kr, 0.0, kf)
    out["C2_channels_sum_minus_one"] = float(sum(sy) - 1.0)
    out["C2_time_domain_diff_chi0.1"] = float(abs(haberkorn_time_domain(system, test[0], Pc, Pc, kr, kf) - sy[2]))
    grid = m2.fibonacci_sphere(GRID_HALF)
    expo, hab = {}, {}
    for chi in CHI_LEVELS:
        Pc = ciss_projector(sim, chi * np.pi)
        key = f"{chi:g}"
        expo[key] = pattern_parts(lambda n: ciss_yield(system, Pc, n), grid[:200])
        expo[key]["projector_idempotence_max"] = float(np.max(np.abs(Pc @ Pc - Pc)))
        for kf, kr in ANCHOR_RATES:
            hab[f"chi{key}_kf{kf:g}_kr{kr:g}"] = pattern_parts(
                lambda n: haberkorn_yields(system, n, Pc, Pc, kr, 0.0, kf)[2], grid)
    out["anchor_exponential_model"] = expo
    out["anchor_haberkorn_forward_yield"] = hab
    return out


# --------------------------------------------------------------------------------------------------------------
# Calibration (C3-C7)
# --------------------------------------------------------------------------------------------------------------
CAL_SEEDS = tuple(range(1001, 1011))
C7_SEEDS = tuple(range(2001, 3001))
# Q3 repeats the MAG-S2 condition: 100 neurons and 600 distinct field axes (1200 directions in opposite pairs)
Q3_DESIGN = replace(REFERENCE, pattern="mags2", n_tuned=100, K=1200, atlas=(), antipodal=False)
# C3, calibration round 2 (seeds 1001-1010): the depth at which the median neuron of Q3 has a signal-to-noise ratio
# of 2. Frozen here so that stage 1 runs Q3 at exactly this depth; `calibrate --checks C3` recomputes and compares it.
DELTA2_Q3 = 0.8786485075914896


_SNR_CACHE = {}


def median_snr(d: Design, seeds, delta):
    """Median over neurons of SD over directions of the expected repeat-averaged count divided by the root mean
    variance of that average (Poisson and shared gain noise, no drift)."""
    vals = []
    for seed in seeds:
        key = (design_key(replace(d, delta=1.0)), seed)
        if key not in _SNR_CACHE:
            bird = Bird(seed)
            _SNR_CACHE[key] = (bird.r0(d)[:d.n_tuned], modulation(bird, d, "compass"))
        r0, u = _SNR_CACHE[key]
        mu = r0[:, None] * (1.0 + delta * u) * d.T
        var = (mu * d.fano + (d.sigma_g * mu) ** 2) / d.R
        vals.append(mu.std(1) / np.sqrt(var.mean(1)))
    return float(np.median(np.concatenate(vals)))


def find_delta2(d: Design, seeds, target=2.0):
    lo, hi = 1e-4, 1.0
    if median_snr(d, seeds, hi) < target:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if median_snr(d, seeds, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def c4_selection(seeds):
    """Pass rates of untuned and tuned cells at f = 0.1 with the 5% rule, in all three recording modes."""
    out = {}
    for mode in MODE_CODE:
        for delta in (0.4, 1.0):
            d = replace(REFERENCE, mode=mode, delta=delta, f=0.1, select="p05", topology=False, antipodal=False,
                        atlas=())
            unt = tun = n_unt = n_tun = 0
            for seed in seeds:
                bird = Bird(seed)
                sched = schedule(bird, d)
                u = modulation(bird, d, "compass")
                _, keep, _, _ = record_condition(bird, d, u, sched, COND_CODE["compass"], design_key(d))
                tun += int(keep[:d.n_tuned].sum())
                unt += int(keep[d.n_tuned:].sum())
                n_tun += d.n_tuned
                n_unt += d.n_untuned
            out[f"{mode}_delta{delta:g}"] = dict(untuned_pass=unt / n_unt, tuned_pass=tun / n_tun,
                                                 untuned_share_of_selected=unt / max(unt + tun, 1),
                                                 n_untuned=n_unt, n_tuned=n_tun)
    return out


def c5_antipodal(seeds):
    hi = replace(REFERENCE, T=5.0, R=16, topology=False, atlas=())
    out = dict(even=[], polar=[], noise=[])
    for seed in seeds:
        r = simulate(hi, seed, conditions=("compass", "polar"))
        out["even"].append(r["conditions"]["compass"]["antipodal"])
        out["polar"].append(r["conditions"]["polar"]["antipodal"])
        rn = simulate(replace(REFERENCE, delta=0.0, topology=False, atlas=()), seed, conditions=("compass",))
        out["noise"].append(rn["conditions"]["compass"]["antipodal"])
    return out


def c6_atlas(seeds):
    d = replace(REFERENCE, antipodal=False, atlas=ATLAS_ALL)
    labels = {s: [] for s in ATLAS_ALL}
    for seed in seeds:
        r = simulate(d, seed, conditions=())
        for s in ATLAS_ALL:
            labels[s].append(r["conditions"][s]["topology"]["shape"])
    return labels


def _c7_job(args):
    seed, d = args
    r = simulate(d, seed, conditions=("compass", "polar"))
    return seed, {c: r["conditions"][c]["antipodal"] for c in ("compass", "polar")}


def c7_false_odd(seeds, workers):
    from multiprocessing import Pool
    out = {}
    for delta in (1.0, 0.1):
        d = replace(REFERENCE, delta=delta, topology=False, atlas=())
        res = {}
        with Pool(workers) as pool:
            for seed, r in pool.imap_unordered(_c7_job, [(s, d) for s in seeds], chunksize=4):
                res[seed] = r
        rows = [res[s] for s in seeds]
        out[f"delta{delta:g}"] = dict(
            even_odd_found=int(sum(r["compass"]["odd_found"] for r in rows)),
            even_polar_verdict=int(sum(r["compass"]["verdict"] == "polar" for r in rows)),
            polar_even_verdict=int(sum(r["polar"]["verdict"] == "even" for r in rows)),
            n=len(rows),
            even_verdicts={v: int(sum(r["compass"]["verdict"] == v for r in rows)) for v in VERDICTS},
            polar_verdicts={v: int(sum(r["polar"]["verdict"] == v for r in rows)) for v in VERDICTS},
            even_p_odd=[r["compass"]["p_odd"] for r in rows])
    return out


# --------------------------------------------------------------------------------------------------------------
# Stage 1 and stage 2 designs
# --------------------------------------------------------------------------------------------------------------
DELTAS = (0.1, 0.2, 0.4, 0.7, 1.0)
N_GRID = (25, 50, 100, 200, 400)
N_ANTIPODAL_ONLY = (6, 12)
W_LEVELS = (0.1, 0.2, 0.3, 0.5)
EPS_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def stage1_cells(delta2=DELTA2_Q3):
    """Named designs of the stage-1 maps; delta2 is the depth found by C3."""
    ref = REFERENCE
    cells = {}
    for delta in DELTAS:
        for mode in MODE_CODE:
            for n in N_GRID:
                atlas = ATLAS_ALL if (delta == 1.0 and n == 200 and mode == "sim") else ("rp2", "sphere")
                cells[f"M_d{delta:g}_N{n}_{mode}"] = replace(ref, delta=delta, n_tuned=n, mode=mode, atlas=atlas)
                cells[f"F_d{delta:g}_N{n}_{mode}_p05"] = replace(ref, delta=delta, n_tuned=n, mode=mode, f=0.1,
                                                                 select="p05")
            for n in N_ANTIPODAL_ONLY:
                cells[f"M_d{delta:g}_N{n}_{mode}"] = replace(ref, delta=delta, n_tuned=n, mode=mode, topology=False,
                                                             atlas=())
                cells[f"F_d{delta:g}_N{n}_{mode}_p05"] = replace(ref, delta=delta, n_tuned=n, mode=mode, f=0.1,
                                                                 select="p05", topology=False, atlas=())
            for n in N_ANTIPODAL_ONLY + N_GRID:
                cells[f"A_d{delta:g}_N{n}_{mode}"] = replace(ANTIPODAL_MAP, delta=delta, n_tuned=n, mode=mode)
                cells[f"AF_d{delta:g}_N{n}_{mode}_p05"] = replace(ANTIPODAL_MAP, delta=delta, n_tuned=n, mode=mode,
                                                                  f=0.1, select="p05")
        for rule in ("p01", "bh10"):
            cells[f"F_d{delta:g}_N200_sim_{rule}"] = replace(ref, delta=delta, f=0.1, select=rule)
        for K in (100, 200, 400, 600):
            cells[f"S_K{K}_d{delta:g}"] = replace(ref, delta=delta, K=K)
        for K in (10, 20):
            cells[f"S_K{K}_d{delta:g}"] = replace(ref, delta=delta, K=K, topology=False, atlas=())
        for R in (2, 8, 16):
            cells[f"S_R{R}_d{delta:g}"] = replace(ref, delta=delta, R=R)
        for T in (1.0, 5.0):
            cells[f"S_T{T:g}_d{delta:g}"] = replace(ref, delta=delta, T=T)
        cells[f"S_fano1.5_d{delta:g}"] = replace(ref, delta=delta, fano=1.5)
        cells[f"S_tau2_d{delta:g}"] = replace(ref, delta=delta, tau_a=2.0)
        cells[f"S_mags2_d{delta:g}"] = replace(ref, delta=delta, pattern="mags2")
        for sg in (0.0, 0.2):
            cells[f"S_sg{sg:g}_d{delta:g}"] = replace(ref, delta=delta, sigma_g=sg)
    for w in W_LEVELS:
        cells[f"W_w{w:g}"] = replace(ref, w=w)
    for eps in EPS_LEVELS:
        cells[f"E_eps{eps:g}"] = replace(ref, eps=eps)
    cells["Q3"] = replace(Q3_DESIGN, delta=float(delta2))
    return cells


def stage2_cells(points):
    """Designs of stage 2 from the working points chosen by analyze.py."""
    ref = replace(REFERENCE, atlas=())
    cells = {"ref": ref}
    h3 = points.get("H3") or {}
    if h3.get("n_tuned"):
        for mode in ("sim", "probe"):
            cells[f"H3_{mode}"] = replace(ref, delta=0.4, n_tuned=int(h3["n_tuned"]), mode=mode, sigma_g=0.2,
                                          antipodal=False)
    h4 = points.get("H4") or {}
    if h4.get("n_tuned"):
        cells["H4"] = replace(ref, delta=float(h4["delta"]), n_tuned=int(h4["n_tuned"]), topology=False)
    h5 = points.get("H5") or {}
    if h5.get("eps_ant") is not None:
        cells["H5"] = replace(ref, eps=float(h5["eps_ant"]))
    return cells


# --------------------------------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------------------------------
SMOKE_SEED = 9311


def smoke():
    issues = []
    s = SMOKE_SEED
    bird = Bird(s)
    d = replace(REFERENCE, n_tuned=30, K=20, R=4, atlas=("rp2",))
    u = modulation(bird, d, "compass")
    if not np.all(np.abs(u) <= 0.5 + 1e-12):
        issues.append("modulation outside [-1/2, 1/2]")
    if np.max(np.abs(u[:, :10] - u[:, 10:])) > 1e-12:
        issues.append("compass modulation differs between n and -n")
    up = modulation(bird, d, "polar")
    if np.max(np.abs(up[:, :10] + up[:, 10:])) > 1e-12:
        issues.append("polar modulation is not odd")
    for mode in MODE_CODE:
        dm = replace(d, mode=mode, f=0.1, select="p05")
        sess, frac, gz, dur = schedule(bird, dm)
        if not (np.all((frac > 0) & (frac < 1)) and np.all(np.isfinite(gz))):
            issues.append(f"bad schedule in {mode}")
        if mode != "sim" and np.bincount(sess).max() > CELLS_PER_SESSION[mode]:
            issues.append(f"session larger than allowed in {mode}")
        r = simulate(dm, s)
        for cond, rec in r["conditions"].items():
            if cond != "shuffled" and not (rec["counts_ok"] and rec["rates_ok"] and rec["finite"]):
                issues.append(f"bad record {mode} {cond}")
    # one turn of 180 degrees takes pi/2 s
    if abs(np.pi / OMEGA_MAX - np.pi / 2) > 1e-12:
        issues.append("turn speed is not 2 rad/s")
    # antipodal test on synthetic codes with known answers
    rng = rng_for(s, 50)
    K, n, R = 40, 60, 8
    even = np.tile(rng.uniform(5, 15, (n, K // 2)), 2)
    odd_part = rng.uniform(-3, 3, (n, K // 2))
    polar = 10 + np.concatenate([odd_part, -odd_part], axis=1)
    for name, mu, want in (("even", even, "even"), ("polar", polar, "polar")):
        Cn = rng.poisson(np.repeat(mu[:, :, None] * 5, R, axis=2))
        t = antipodal_test(Cn, np.zeros(n, dtype=int), rng_for(s, 51), rng_for(s, 52), n_perm=200, n_boot=200)
        if t["verdict"] != want:
            issues.append(f"antipodal test calls a {name} code {t['verdict']}")
    flat = rng.poisson(np.full((n, K, R), 20.0))
    t = antipodal_test(flat, np.zeros(n, dtype=int), rng_for(s, 53), rng_for(s, 54), n_perm=200, n_boot=200)
    if t["verdict"] in ("even", "polar"):
        issues.append("antipodal test finds a code in pure noise")
    if anova_pvalues(np.ones((3, 5, 4))).min() < 1.0:
        issues.append("ANOVA rejects constant cells")
    if not np.array_equal(bh_reject(np.array([0.001, 0.2, 0.03, 0.04]), 0.1), np.array([True, False, True, True])):
        issues.append("Benjamini-Hochberg rule wrong")
    inv = invariants_for_seed(s)
    if not (inv["I2_ren_max_diff"] <= 1e-12 and inv["I2_mags2_max_diff"] <= 1e-12 and inv["I2_polar_max_sum"] <= 1e-12
            and inv["I4_same_means"] and inv["I4_rebuild_max_rel_diff"] <= 1e-12 and inv["I4_sessions_ok"]):
        issues.append(f"invariant check failed {inv}")
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


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _blas():
    try:
        from threadpoolctl import threadpool_info
        keys = ("internal_api", "version", "architecture", "num_threads", "filepath")
        return [{k: x.get(k) for k in keys} for x in threadpool_info() if x.get("user_api") == "blas"]
    except Exception:
        return None


def metadata(mode):
    def sha_if(name):
        path = os.path.join(HERE, name)
        return file_sha256(path) if os.path.exists(path) else None
    return dict(mode=mode, code_sha256=sha_if("mag_s3.py"), analysis_sha256=sha_if("analyze.py"),
                tests_sha256=sha_if("test_analyze.py"), mag_s2_sha256=sha_if("mag_s2.py"),
                physics_sha256=sha_if("physics.json"), python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, ripser=_pkg_version("ripser"), radicalpy=_pkg_version("radicalpy"),
                blas=_blas(), reference=asdict(REFERENCE))


def _cell_job(args):
    name, d, seed, out = args
    path = os.path.join(out, name, f"seed_{seed}.json")
    if os.path.exists(path):
        return name, seed, 0.0
    t0 = time.time()
    r = simulate(d, seed)
    r["cell"] = name
    r["design"] = asdict(d)
    r["code"] = CODE_HASHES
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json_atomic(path, r)
    return name, seed, time.time() - t0


def run_jobs(cells, seeds, out, workers, max_hours):
    """Run the missing (cell, seed) jobs. At the time limit no new job starts and the running ones finish; a worker
    that dies (for example out of memory) stops the run with an error instead of leaving it waiting."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    jobs = [(name, d, s, out) for name, d in cells.items() for s in seeds
            if not os.path.exists(os.path.join(out, name, f"seed_{s}.json"))]
    t0 = time.time()
    done, nxt = 0, 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = set()
        try:
            while nxt < len(jobs) or pending:
                late = bool(max_hours and time.time() - t0 > max_hours * 3600)
                while not late and nxt < len(jobs) and len(pending) < 2 * workers:
                    pending.add(ex.submit(_cell_job, jobs[nxt]))
                    nxt += 1
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result()
                    done += 1
                    if done % 25 == 0 or done == len(jobs):
                        print(f"{done}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)
        except BrokenProcessPool:
            sys.exit("a worker process died (out of memory?); rerun the same command, with fewer --workers if it repeats")
    if done < len(jobs):
        print("stopped at the time limit; rerun the same command to continue", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["physics", "calibrate", "smoke", "stage1", "stage2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--checks", default="C3,C4,C5,C6,C7")
    ap.add_argument("--points", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = metadata(a.mode)
    t0 = time.time()
    if a.mode == "physics":
        res = physics_checks()
        write_json_atomic(os.path.join(a.out, "physics_s3.json"), res, indent=1)
        print(json.dumps(res, indent=1))
    elif a.mode == "calibrate":
        checks = a.checks.split(",")
        res = {}
        path = os.path.join(a.out, "calibration.json")
        if os.path.exists(path):
            with open(path) as fh:
                res = json.load(fh)
        if "C3" in checks:
            dq = find_delta2(Q3_DESIGN, CAL_SEEDS)
            res["C3"] = dict(delta2=dq, median_snr_at_delta2=median_snr(Q3_DESIGN, CAL_SEEDS, dq) if dq else None,
                             median_snr_reference_delta1=median_snr(REFERENCE, CAL_SEEDS, 1.0),
                             frozen_delta2=DELTA2_Q3, matches_frozen=bool(dq == DELTA2_Q3))
        if "C4" in checks:
            res["C4"] = c4_selection(CAL_SEEDS)
        if "C5" in checks:
            res["C5"] = c5_antipodal(CAL_SEEDS)
        if "C6" in checks:
            res["C6"] = dict(K=REFERENCE.K, R=REFERENCE.R, labels=c6_atlas(CAL_SEEDS))
        if "C7" in checks:
            res["C7"] = c7_false_odd(C7_SEEDS, a.workers)
        write_json_atomic(path, res, indent=1)
    elif a.mode == "smoke":
        issues = smoke()
        meta["issues"] = issues
        print(json.dumps(dict(issues=issues), indent=1))
    elif a.mode == "stage1":
        seeds = parse_seeds(a.seeds or "1-10")
        cells = stage1_cells()
        meta.update(seeds=seeds, delta2=DELTA2_Q3, cells={k: asdict(v) for k, v in cells.items()})
        write_invariants(a.out, seeds, a.workers)
        meta["complete"] = run_jobs(cells, seeds, a.out, a.workers, a.max_hours)
    else:
        if not a.points:
            sys.exit("stage2 needs --points from analyze.py")
        with open(a.points) as fh:
            points = json.load(fh)
        seeds = parse_seeds(a.seeds or "101-200")
        cells = stage2_cells(points)
        meta.update(seeds=seeds, points=points, cells={k: asdict(v) for k, v in cells.items()})
        write_invariants(a.out, seeds, a.workers)
        meta["complete"] = run_jobs(cells, seeds, a.out, a.workers, a.max_hours)
    meta["seconds"] = round(time.time() - t0, 1)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_json_atomic(os.path.join(a.out, f"meta_{a.mode}_{stamp}.json"), meta, indent=1)


if __name__ == "__main__":
    main()
