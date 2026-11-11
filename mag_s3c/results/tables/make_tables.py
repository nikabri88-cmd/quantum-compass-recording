"""Tables of the MAG-S3c confirmatory run (written after the run; it only reformats frozen outputs).

The protocol asks the report for every one of the 184 comparisons of H1-H5 with the smallest margin per hypothesis
and geometry, for k_half of every combination with its class, and for the admissible region of every geometry, axis
and level of relaxation. This script writes them from results.json of the run archive, with no new computation:

  comparisons_H1_H5.csv  one row per comparison (184, of them 8 known in advance from MAG-S3b): hypothesis,
                         geometry, label, value, threshold, verdict of the comparison, and the numbers it is made of
  k_half.csv             one row per geometry, CISS axis, chi_b and kind of relaxation (32): k_half or the side of
                         the scale it lies beyond, the bracketing levels, the class of the dephasing and d_odd . z of
                         form K at m = 1 on every level of the scale
  e2_regions.csv         one row per cell, polar order p and axial order S of map E2 (6432): eps_eff by the ratio
                         transfer and by the amplitude transfer, the stricter one, admissibility after a null result in
                         R1, R2 and R3, and visibility to the dipole and antipodal tests
  cells.csv              the table of all 1016 cells written by analyze_s3c.py (copied from the archive)

Usage (the archive unpacks into RUN_DIR/confirmatory_s3c and RUN_DIR/confirmation.json):
  mkdir RUN_DIR && tar -xzf ../confirmatory_run/confirmatory_s3c.tar.gz -C RUN_DIR
  python3 -I make_tables.py RUN_DIR OUT_DIR
"""
import csv
import json
import os
import shutil
import sys

RUN, OUT = sys.argv[1], sys.argv[2]
RES = os.path.join(RUN, "confirmatory_s3c", "results")
GEOMS = ("S2", "W_C", "W_D")
HYPS = ("H1", "H2", "H3", "H4", "H5")
EXTRA = ("K", "P_max", "measure", "dz_none", "dz_relaxed", "dz_K", "dz_inc", "near_zero_circle")


def write(name, fields, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else ("" if v is None else v)) for k, v in row.items()})
    print(name, len(rows))


def comparisons(res):
    rows = []
    for h in HYPS:
        for g in GEOMS:
            w = res["hypotheses"][h].get(g)
            if not w:
                continue
            for known, items in ((False, w["comparisons"]), (True, w.get("known_in_advance", []))):
                for c in items:
                    row = dict(hypothesis=h, geometry=g, label=c["label"], value=c["value"], threshold=c["threshold"],
                               sense=c["sense"], passed=c["pass"], known_in_advance=known,
                               verdict_of_geometry=w["verdict"], smallest_margin_of_geometry=w["smallest_margin"])
                    row.update({k: c.get(k) for k in EXTRA})
                    rows.append(row)
    fields = ["hypothesis", "geometry", "label", "value", "threshold", "sense", "passed", "known_in_advance",
              "verdict_of_geometry", "smallest_margin_of_geometry", *EXTRA]
    write("comparisons_H1_H5.csv", fields, rows)
    return rows


def k_half(res):
    levels = sorted({k for v in res["hypotheses"]["H4"]["k_half"].values() for k, _ in v["levels"]})
    rows = []
    for key, v in res["hypotheses"]["H4"]["k_half"].items():
        geom, rest = (key[:3], key[4:]) if key.startswith("W_") else (key[:2], key[3:])
        axis, chi, kind = rest.split("_")
        row = dict(geometry=geom, ciss_axis=axis, chi_b_over_pi=float(chi[3:]), relaxation=kind,
                   k_half_per_s=v.get("k_half"), beyond_scale=v.get("note", ""),
                   bracket_low=(v.get("bracket") or [None, None])[0], bracket_high=(v.get("bracket") or [None, None])[1],
                   dephasing_class=v.get("class", ""))
        seen = dict((k, d) for k, d in v["levels"])
        for k in levels:
            row[f"d_odd_z_at_{k:g}"] = seen.get(k)
        rows.append(row)
    fields = ["geometry", "ciss_axis", "chi_b_over_pi", "relaxation", "k_half_per_s", "beyond_scale", "bracket_low",
              "bracket_high", "dephasing_class"] + [f"d_odd_z_at_{k:g}" for k in levels]
    write("k_half.csv", fields, rows)


def regions(res):
    rows = []
    for region, v in res["maps"]["E2"]["regions"].items():
        for x in v["entries"]:
            rows.append(dict(region=region, cell=x["cell"], form=x["form"], chi_over_pi=x["chi_over_pi"], m=x["m"],
                             delta_chi_over_pi=x["delta_chi_over_pi"], p=x["p"], S=x["S"], eps_eff_ratio=x["eps_eff"],
                             eps_eff_amplitude=x["eps_E6"], eps_eff_stricter=x["stricter"],
                             admissible_R1=x["admissible"]["R1"], admissible_R2=x["admissible"]["R2"],
                             admissible_R3=x["admissible"]["R3"], dipole_visible=x["dipole_visible"],
                             antipodal_visible=x["antipodal_visible"]))
    fields = ["region", "cell", "form", "chi_over_pi", "m", "delta_chi_over_pi", "p", "S", "eps_eff_ratio",
              "eps_eff_amplitude", "eps_eff_stricter", "admissible_R1", "admissible_R2", "admissible_R3",
              "dipole_visible", "antipodal_visible"]
    write("e2_regions.csv", fields, rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(RES, "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    rows = comparisons(res)
    n_known = sum(r["known_in_advance"] for r in rows)
    print(f"comparisons {len(rows)}, known in advance {n_known}")
    k_half(res)
    regions(res)
    shutil.copyfile(os.path.join(RES, "tables", "cells.csv"), os.path.join(OUT, "cells.csv"))
    print("cells.csv copied")


if __name__ == "__main__":
    main()
