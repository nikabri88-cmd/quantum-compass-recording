"""Interval of the difference for H3 of MAG-S3, recomputed after release v0.3.3 (13 November 2026).

The frozen analyze.py reports for H3 the Clopper-Pearson interval of the share of simultaneous-only birds among the
discordant ones, scaled by the observed discordant share. The scaling treats that share as known, so the interval of
the difference comes out too narrow. The verdict uses only whether the upper bound of the conditional share lies below
one half, the exact McNemar test, which the scaling does not touch.

This script reads results.json of the run and
  reproduces the reported interval with the frozen function paired_difference;
  gives the Newcombe hybrid score interval for a difference of paired shares (Newcombe 1998, method 10) and the
  Clopper-Pearson interval of the share of simultaneous recordings with RP², which is exact for the difference here
  because no pseudo-population of probe sessions got RP²;
  computes the exact coverage of the reported interval and of the Newcombe interval for true shares 0.44 and 0.

Usage: python3 -I h3_interval.py RESULTS_JSON > h3_interval.txt
"""
import importlib.util
import json
import math
import os
import sys

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("analyze", os.path.join(HERE, "..", "analyze.py"))
A = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.join(HERE, ".."))
_spec.loader.exec_module(A)

Z = stats.norm.ppf(0.975)


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(n11, n10, n01, n00):
    """Hybrid score interval for p1 - p2 of paired shares, p1 = (n11 + n10)/n, p2 = (n11 + n01)/n."""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    a = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = 0.0 if a == 0 else (n11 * n00 - n10 * n01) / math.sqrt(a)
    d = p1 - p2
    lo = d - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = d + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return lo, hi


def frozen(n10, n01, n):
    a = [True] * n10 + [False] * n01 + [False] * (n - n10 - n01)
    b = [False] * n10 + [True] * n01 + [False] * (n - n10 - n01)
    return A.paired_difference(a, b)


def main(path):
    h3 = json.load(open(path))["hypotheses"]["H3"]
    n, n10, n01 = h3["n"], h3["n10"], h3["n01"]
    n11 = h3["probe_rp2"] - n01
    n00 = n - n11 - n10 - n01
    rep = frozen(n10, n01, n)
    print(f"H3 of the run: n = {n}, simultaneous RP2 {h3['sim_rp2']}, probe RP2 {h3['probe_rp2']}, "
          f"n10 = {n10}, n01 = {n01}, n11 = {n11}, n00 = {n00}, verdict {h3['verdict']}")
    print("reported interval of the difference:", [round(v, 6) for v in h3["interval"]])
    print("frozen paired_difference reproduces it:", rep["interval"] == h3["interval"])
    lo, hi = A.cp_two(n10, n10 + n01)
    print(f"conditional Clopper-Pearson interval of the simultaneous-only share among discordant birds: "
          f"{lo:.4f}-{hi:.4f}; upper bound below 1/2: {hi < 0.5} (the verdict rule)")
    nl, nh = newcombe(n11, n10, n01, n00)
    print(f"Newcombe hybrid score interval of the difference: {nl:.4f}-{nh:.4f}")
    if n11 + n01 == 0:
        cl, ch = A.cp_two(n10 + n11, n)
        print(f"Clopper-Pearson interval of the simultaneous share (exact for the difference, no probe RP2): "
              f"{cl:.4f}-{ch:.4f}")
    p1 = (n11 + n10) / n
    cov_rep = cov_new = 0.0
    for x in range(n + 1):
        w = stats.binom.pmf(x, n, p1)
        r = frozen(x, 0, n)["interval"]
        cov_rep += w * (r[0] <= p1 <= r[1])
        a, b = newcombe(0, x, 0, n - x)
        cov_new += w * (a <= p1 <= b)
    print(f"exact coverage for true shares {p1:g} and 0: reported interval {cov_rep:.4f}, Newcombe {cov_new:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
