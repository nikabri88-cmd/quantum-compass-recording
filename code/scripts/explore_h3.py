#!/usr/bin/env python3
"""Exploratory check of H3 after the confirmatory run (post hoc; outside the frozen analysis).

For birds of stage 2 the script rebuilds the compass recording of both H3 designs (simultaneous recording and probe
pseudo-population, sigma_g = 0.2, delta = 0.4, 200 cells) with the frozen mag_s3.py, confirms each rebuild by the
fingerprint stored in the run, and reports
  1. the share of variance carried by the first principal components of the z-scored cloud, as in the pipeline;
  2. for each of the first six components, the squared projection on the span of the per-session gain series
     (the gain of each presentation averaged over the analysis repeats, centred over directions);
  3. the label of the frozen MAG-S2 pipeline after the counts are divided by the true gain of each presentation
     (oracle) or by the gain estimated from the session itself: the summed count of its cells in that presentation
     over the mean of that sum.

  python3 -I scripts/explore_h3.py RUN_DIR SEEDS OUT.json      e.g. RUN_DIR = extracted confirmatory_v04, SEEDS = 101-110
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location("mag_s3", os.path.join(CODE, "mag_s3.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3"] = m
    spec.loader.exec_module(m)
    return m


M = load()
RUN = None


def design_of(run, cell, seed):
    with open(os.path.join(run, "s2", cell, f"seed_{seed}.json")) as fh:
        r = json.load(fh)
    d = dict(r["design"])
    d["atlas"] = tuple(d["atlas"])
    return M.Design(**d), r["conditions"]["compass"]


def zscore(X):
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)


def gain_basis(sched, d, ana):
    """Orthonormal basis of the per-session gain series over the K directions."""
    g = np.maximum(1.0 + d.sigma_g * sched[2], 0.0)[:, :, ana].mean(2)      # sessions x K
    Q, _ = np.linalg.qr((g - g.mean(1, keepdims=True)).T)
    return Q


def label(X, seed, key):
    return M.topology_record(X, M.rng_for(seed, 31, key, M.COND_CODE["compass"]))["shape"]


def one(args):
    run, cell, seed = args
    d, stored = design_of(run, cell, seed)
    bird = M.Bird(seed)
    key = M.design_key(d)
    sched = M.schedule(bird, d)
    u = M.modulation(bird, d, "compass")
    C, keep, ana, _ = M.record_condition(bird, d, u, sched, M.COND_CODE["compass"], key)
    X = C[keep][:, :, ana].mean(2).T
    U, S, _ = np.linalg.svd(zscore(X), full_matrices=False)
    Q = gain_basis(sched, d, ana)
    sess = sched[0]
    gain = np.maximum(1.0 + d.sigma_g * sched[2][sess], 0.0)                 # cells x K x R
    Cf = C.astype(float)
    Xo = (Cf / np.where(gain > 0, gain, 1.0))[keep][:, :, ana].mean(2).T
    est = np.empty_like(gain)
    for s in np.unique(sess):
        tot = Cf[sess == s].sum(0)                                            # K x R
        est[sess == s] = (tot / tot.mean())[None]
    Xe = (Cf / np.where(est > 0, est, 1.0))[keep][:, :, ana].mean(2).T
    return dict(cell=cell, seed=seed, n_sessions=int(sess.max()) + 1, rebuilt=M.fingerprint(X) == stored["fingerprint"],
                stored_label=stored["topology"]["shape"], pc_share=[float(x) for x in (S ** 2 / np.sum(S ** 2))[:8]],
                pc_in_gain_span=[float(np.sum((Q.T @ U[:, j]) ** 2)) for j in range(6)],
                label_oracle=label(Xo, seed, key), label_estimated=label(Xe, seed, key))


if __name__ == "__main__":
    run, seeds, out = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = (int(x) for x in seeds.split("-"))
    jobs = [(run, c, s) for s in range(a, b + 1) for c in ("H3_sim", "H3_probe")]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", os.cpu_count()))) as ex:
        res = list(ex.map(one, jobs))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res:
        print(r["cell"], r["seed"], "sessions", r["n_sessions"], "rebuilt", r["rebuilt"], "stored", r["stored_label"],
              "oracle", r["label_oracle"], "estimated", r["label_estimated"],
              "pc%", [round(100 * x, 1) for x in r["pc_share"][:6]],
              "in_gain", [round(x, 2) for x in r["pc_in_gain_span"]])
