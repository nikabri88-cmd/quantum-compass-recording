"""Environment check of MAG-S3c (author's harness, not part of the study): compares the calibration C1-C6 of a new
environment with the reference calibration, number by number, and prints the environment.
Usage: python3 -I scripts/env_check.py REF_DIR NEW_DIR
Exit code 0 when every number agrees to 1e-9 relative or 1e-12 absolute (round-off quantities such as the residuals
of C3), the relative changes of C1 and C2 to 1e-6 absolute (GMRES stops at a relative residual of 1e-13, so their last
digits depend on the BLAS of the machine), every count of GMRES iterations to 2 and of restarts to 1, and every verdict
is the same, 1 otherwise. Times are left out."""
import json
import os
import platform
import sys

FILES = ("calibration.json",)
SKIP = {"metadata", "seconds", "seconds_per_direction"}
ITERATIONS = ("max_gmres_iterations", "/iterations")
RESTARTS = "max_gmres_restarts"
CHANGE_TOL = 1e-6                        # relative changes of C1 and C2 (keys C1_..., C2_...)


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            if k not in a or k not in b:
                out.append(("missing", f"{path}/{k}", a.get(k), b.get(k)))
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or not all(isinstance(v, (int, float)) for v in (a, b)):
        out.append(("same" if a == b else "DIFFER", path, a, b))
    elif a == b:
        out.append(("same", path, a, b))
    elif path.endswith(ITERATIONS):
        out.append(("close" if abs(a - b) <= 2 else "DIFFER", path, a, b))
    elif path.endswith(RESTARTS):
        out.append(("close" if abs(a - b) <= 1 else "DIFFER", path, a, b))
    elif path.rsplit("/", 1)[-1].startswith(("C1_", "C2_")):
        out.append(("close" if abs(a - b) <= CHANGE_TOL else "DIFFER", path, a, b))
    else:
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        out.append(("close" if rel <= 1e-9 or abs(a - b) <= 1e-12 else "DIFFER", path, a, b))


def main():
    ref, new = sys.argv[1], sys.argv[2]
    print(f"python {platform.python_version()}, {platform.machine()}, {os.cpu_count()} processors")
    meta = json.load(open(os.path.join(new, FILES[0])))["metadata"]
    print("packages:", {k: meta.get(k) for k in ("numpy", "scipy", "radicalpy")})
    print("BLAS:", [(x.get("version"), x.get("architecture")) for x in (meta.get("blas") or [])])
    rows = []
    for f in FILES:
        walk(json.load(open(os.path.join(ref, f))), json.load(open(os.path.join(new, f))), f, rows)
    counts = {k: sum(r[0] == k for r in rows) for k in ("same", "close", "DIFFER", "missing")}
    print(f"{counts['same']} values identical, {counts['close']} equal within the tolerances, {counts['DIFFER']} "
          f"different, {counts['missing']} missing")
    for r in rows:
        if r[0] in ("close", "DIFFER", "missing"):
            print(f"  {r[0]:7s} {r[1]} {r[2]} {r[3]}")
    sys.exit(0 if counts["DIFFER"] == 0 and counts["missing"] == 0 else 1)


if __name__ == "__main__":
    main()
