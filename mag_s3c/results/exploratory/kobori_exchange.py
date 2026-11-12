"""Exploratory computation after release v0.3.0, added to the plan of section 3 of README.md on 12 November 2026 (UTC)
before it was run. Nothing here changes a verdict of MAG-S3c.

C. Kobori et al. (2026, ChemRxiv, doi 10.26434/chemrxiv.15000186/v2) fitted the exchange coupling J = 22 uT for the
   pair FAD-W_C, in the convention 2J = E_S - E_T0, where the run of MAG-S3c took J = 0. This script adds
   -J |gamma_e| (2 S_A.S_B + 1/2), the convention of RadicalPy (singlet at +J, triplets at -J), to the Hamiltonian H0 of
   the frozen systems W_C and W_D, with J = +22 uT and -22 uT, and computes P-LH and P-same at chi = 20 deg on the
   axis r, without relaxation and at 1e6 s^-1 of random fields and of dephasing: 24 direct cells with the frozen
   solver on the grid of the run. The frozen code is used unchanged; the systems with J are registered in its cache
   before the solves. A check cell at J = 0 must equal the cell of the frozen system W_C bit for bit.

Usage:
  python3 -I kobori_exchange.py run OUT FROZEN_CELLS [--workers N]   FROZEN_CELLS holds the cell of the frozen W_C
  python3 -I kobori_exchange.py summarize OUT RESULT.json
"""
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"                              # before NumPy loads BLAS
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("kobori_and_dephasing", os.path.join(HERE, "kobori_and_dephasing.py"))
kd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kd)
c, np = kd.c, kd.np

J_MT = (0.022, -0.022)
LEVELS = (("none", 0.0), ("rf", 1e6), ("std", 1e6))
DEG = 20.0


def name_of(base, j_mt):
    return f"{base}_J{j_mt:+g}"


def exchange_system(base, j_mt):
    sy = c.build_system(base)
    SS = sum(sy.S[0][a] @ sy.S[1][a] for a in range(3))
    H_ex = -j_mt * abs(sy.sim.radicals[0].gamma_mT) * (2.0 * SS + 0.5 * sy.eye)
    return c.Sys(sy.sim, sy.H0 + H_ex, name_of(base, j_mt)), H_ex


def register():
    """Put the systems with J into the cache of the frozen code (inherited by the forked workers) and return the
    singlet-triplet gap of the added term, E_S - E_T in units of J |gamma_e|, for each."""
    gaps = {}
    for base in ("W_C", "W_D"):
        for j in J_MT + (0.0,):
            sy, H_ex = exchange_system(base, j)
            c._SYSTEMS[sy.name] = sy
            if j:
                es = np.trace(sy.PS @ H_ex).real / np.trace(sy.PS).real
                et = np.trace(sy.PT @ H_ex).real / np.trace(sy.PT).real
                gaps[sy.name] = (es - et) / (j * abs(sy.sim.radicals[0].gamma_mT))
    return gaps


def cells():
    return [c.Cell(name_of(g, j), "r", rl, k, form, DEG / 180.0, 1.0) for g in ("W_C", "W_D") for j in J_MT
            for rl, k in LEVELS for form in ("P-LH", "P-same")]


def check_cell():
    return c.Cell(name_of("W_C", 0.0), "r", "none", 0.0, "P-LH", DEG / 180.0, 1.0)


def run(out, frozen_cells, workers):
    bad = kd.frozen_check()
    if bad:
        sys.exit("frozen files differ from FREEZE.txt: " + ", ".join(bad))
    gaps = register()
    print("E_S - E_T of the added term in units of J |gamma_e| (2 expected):", gaps, flush=True)
    t0 = time.time()
    c.pair(out, workers, 0, half_n=c.GRID_HALF, items=[check_cell()])
    ref_name = c.Cell("W_C", "r", "none", 0.0, "P-LH", DEG / 180.0, 1.0).name
    old = c.load_json(os.path.join(frozen_cells, ref_name + ".json"))
    new = c.load_json(os.path.join(out, "cells", check_cell().name + ".json"))
    identical = all(np.array_equal(np.asarray(old[k]), np.asarray(new[k])) for k in ("yp", "ym"))
    print("check cell at J = 0 equal to the frozen W_C bit for bit:", identical, flush=True)
    if not identical:
        sys.exit(1)
    c.pair(out, workers, 0, half_n=c.GRID_HALF, items=cells())
    c.write_json(os.path.join(out, "exchange_done.json"), dict(
        gaps=gaps, check_identical=identical, n_cells=len(cells()), hours=(time.time() - t0) / 3600,
        end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script_sha256=c.file_sha256(os.path.abspath(__file__)), metadata=c.metadata("after-release-exchange")))
    return 0


def summarize(out, result):
    half = c.m2.fibonacci_sphere(c.GRID_HALF)
    res = dict(done=c.load_json(os.path.join(out, "exchange_done.json")), cells={})
    for cell in cells():
        d = c.load_json(os.path.join(out, "cells", cell.name + ".json"))
        v = kd.responses(half, d["yp"], d["ym"])
        v.update(seconds=d["seconds"], diagnostics=d["diagnostics"])
        res["cells"][cell.name] = v
    with open(result, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", result)


if __name__ == "__main__":
    mode, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if mode == "run":
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else os.cpu_count()
        sys.exit(run(args[0], args[1], w))
    elif mode == "summarize":
        summarize(args[0], args[1])
    else:
        sys.exit("mode: run or summarize")
