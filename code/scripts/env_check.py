#!/usr/bin/env python3
"""Environment check of MAG-S3 (author's harness): the same numbers computed in two environments must agree.

  python3 -I scripts/env_check.py CHECK_DIR      reads CHECK_DIR/calibration.json and CHECK_DIR/physics_s3.json,
                                                 writes CHECK_DIR/env_check.json and prints it
  python3 -I scripts/env_check.py --compare A.json B.json

The check uses only calibration birds (1001-1010, C7 on 2001-3000), never a bird of the confirmatory run.
"""
import hashlib
import importlib.util
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location("mag_s3", os.path.join(HERE, "mag_s3.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["mag_s3"] = m
    spec.loader.exec_module(m)
    return m


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def summary(check_dir):
    m = load()
    from dataclasses import replace
    out = dict(environment=dict(python=platform.python_version(), machine=platform.machine(),
                                processor=platform.processor(), cpus=os.cpu_count()),
               packages=m.metadata("env_check"), files={})
    for name in ("mag_s3.py", "analyze.py", "test_analyze.py", "mag_s2.py", "physics.json", "mag_s2_bird1.json",
                 "scripts/pipeline.sh", "scripts/env_check.py"):
        path = os.path.join(HERE, name)
        out["files"][name] = sha(path) if os.path.exists(path) else None
    out["I1"] = m.mag_s2_bird1_check()
    out["invariants_bird_1001"] = m.invariants_for_seed(1001)
    rec = m.simulate(replace(m.REFERENCE, atlas=("rp2", "sphere")), 1001)
    out["reference_bird_1001"] = {
        c: dict(fingerprint=v.get("fingerprint"), shape=(v.get("topology") or {}).get("shape"),
                longest=(v.get("topology") or {}).get("longest"),
                antipodal={k: (v.get("antipodal") or {}).get(k) for k in ("E_odd", "E_even", "p_odd", "verdict")}
                if v.get("antipodal") else None)
        for c, v in rec["conditions"].items()}
    out["reference_bird_1001"]["cost"] = rec["cost"]
    with open(os.path.join(check_dir, "calibration.json")) as fh:
        cal = json.load(fh)
    c5 = {k: [r["verdict"] for r in v] for k, v in cal["C5"].items()}
    c7 = {k: {kk: vv for kk, vv in v.items() if kk != "even_p_odd"} for k, v in cal["C7"].items()}
    c7_p = {k: hashlib.sha256(json.dumps(v["even_p_odd"]).encode()).hexdigest() for k, v in cal["C7"].items()}
    out["calibration"] = dict(C3=cal["C3"], C4=cal["C4"], C5=c5, C6=cal["C6"], C7=c7, C7_p_values_sha256=c7_p)
    with open(os.path.join(check_dir, "physics_s3.json")) as fh:
        phys = json.load(fh)
    out["physics"] = {k: v for k, v in phys.items() if k.startswith(("C1", "C2"))}
    out["physics"]["anchor_eps_equivalent"] = {
        k: v["eps_equivalent"] for k, v in phys["anchor_haberkorn_forward_yield"].items()}
    return out


def flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def compare(a, b):
    A, B = dict(flat(a)), dict(flat(b))
    skip = ("/environment", "/packages", "/files/scripts/env_check.py")
    same, close, differ = 0, [], []
    for k in sorted(set(A) | set(B)):
        if k.startswith(skip):
            continue
        x, y = A.get(k), B.get(k)
        if x == y:
            same += 1
        elif isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y)):
            close.append((k, x, y))
        else:
            differ.append((k, x, y))
    print(f"{same} values identical, {len(close)} equal to 1e-9, {len(differ)} different")
    for k, x, y in close[:20]:
        print("  close  ", k, x, y)
    for k, x, y in differ[:60]:
        print("  DIFFER ", k, x, y)
    return not differ


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        with open(sys.argv[2]) as fa, open(sys.argv[3]) as fb:
            sys.exit(0 if compare(json.load(fa), json.load(fb)) else 1)
    res = summary(sys.argv[1])
    with open(os.path.join(sys.argv[1], "env_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
