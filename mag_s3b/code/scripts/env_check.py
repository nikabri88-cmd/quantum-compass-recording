"""Environment check of MAG-S3b (author's harness, not part of the study): compares the calibration C1-C5 of a new
environment with the reference calibration, number by number, and prints the environment.
Usage: python3 -I scripts/env_check.py REF_DIR NEW_DIR
Exit code 0 when every number agrees to 1e-9 relative or 1e-12 absolute (round-off quantities such as C3 rel_err)
and every verdict is the same, 1 otherwise."""
import json
import os
import platform
import sys

FILES = ("C1.json", "C2_C4.json", "C3.json", "C5.json")
SKIP = {"metadata", "seconds"}


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
    else:
        if a == b:
            out.append(("same", path, a, b))
        else:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
            out.append(("close" if rel <= 1e-9 or abs(a - b) <= 1e-12 else "DIFFER", path, a, b))


def main():
    ref, new = sys.argv[1], sys.argv[2]
    print(f"python {platform.python_version()}, {platform.machine()}, {os.cpu_count()} processors")
    meta = json.load(open(os.path.join(new, "C5.json")))["metadata"]
    print("packages:", {k: meta.get(k) for k in ("numpy", "scipy", "radicalpy")})
    print("BLAS:", [(x.get("version"), x.get("architecture")) for x in (meta.get("blas") or [])])
    rows = []
    for f in FILES:
        walk(json.load(open(os.path.join(ref, f))), json.load(open(os.path.join(new, f))), f, rows)
    counts = {k: sum(r[0] == k for r in rows) for k in ("same", "close", "DIFFER", "missing")}
    print(f"{counts['same']} values identical, {counts['close']} equal to 1e-9, {counts['DIFFER']} different, "
          f"{counts['missing']} missing")
    for r in rows:
        if r[0] in ("close", "DIFFER", "missing") and not r[1].endswith("]"):
            print(f"  {r[0]:7s} {r[1]} {r[2]} {r[3]}")
    n_list = sum(r[0] in ("close", "DIFFER") and r[1].endswith("]") for r in rows)
    if n_list:
        print(f"  {n_list} list entries close or different (p values, rows), first:",
              next(r for r in rows if r[0] in ("close", "DIFFER") and r[1].endswith("]"))[1:])
    sys.exit(0 if counts["DIFFER"] == 0 and counts["missing"] == 0 else 1)


if __name__ == "__main__":
    main()
