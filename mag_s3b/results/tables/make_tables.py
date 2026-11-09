"""Tables of the MAG-S3b confirmatory run (written after the run; it only reformats frozen outputs).

The protocol asks the report for a table of every cell of the pair grid behind H1-H4 and for the upper 95% bound of
every bird at eps_eff = 0. This script writes both from the run archive, with no new computation:

  cells_H1_H4.csv            one row per cell of the pair grid (225): the cell, eps_mol, eps_rec, the odd dipole,
                             the odd shares above degree 3, the fit residuals and eps_eff at p = 1 for every S,
                             with the ratio transfer and the amplitude transfer (E6)
  upper_bounds_per_bird.csv  one row per bird at eps_eff = 0 in R1, R2 and R3 (300): the raw estimate, the raw 95%
                             interval, the upper bound, and the p values of the dipole and antipodal tests

Usage (the archive unpacks the run folder's contents into RUN_DIR):
  mkdir RUN_DIR && tar -xzf ../confirmatory_run/confirmatory_s3b_light.tar.gz -C RUN_DIR
  python3 -I make_tables.py RUN_DIR OUT_DIR
"""
import csv
import json
import os
import sys

RUN, OUT = sys.argv[1], sys.argv[2]
ORDERS = ("S1_p1", "S0.8_p1", "S0.6_p1", "S0.4_p1")


def load(*parts):
    with open(os.path.join(RUN, *parts)) as fh:
        return json.load(fh)


def cells_table():
    receptor = load("pair", "receptor.json")["cells"]
    maps = {c["cell"]: c for c in load("results", "results.json")["maps"]["E1_E5_E6"]}
    rows = []
    for c in receptor:
        x = c["cell"]
        name = f'{x["form"]}_chi{x["chi"]:g}' + (f'_m{x["m"]:g}_a{x["alpha"]:g}' if x["form"] == "K" else "") \
            + f'_b{x["beta"]:g}_{x["rates"]}'
        e = maps[name]
        d = c["d_odd"]
        row = dict(cell=name, form=x["form"], chi_b_over_pi=x["chi"], m=x.get("m"), alpha_deg=x.get("alpha"),
                   beta_deg=x["beta"], rates=x["rates"], eps_mol=c["parts"]["eps_mol"],
                   eps_rec=c["receptor"]["eps_rec"], receptor_even_range=c["receptor"]["receptor_even_range"],
                   d_odd_x=d[0], d_odd_y=d[1], d_odd_z=d[2],
                   odd_share_deg3_molecule_L24=e["odd_share_deg3_molecule_L24"],
                   odd_share_deg3_receptor_L24=e["odd_share_deg3_receptor_L24"],
                   even_residual_L10=c["fit_even_rms_over_sd"], odd_residual_L10=c["fit_odd_rms_over_sd"],
                   odd_residual_L24=c["fit_check"]["odd_rms_over_sd"])
        for k in ORDERS:
            row[f"eps_eff_{k}"] = c["receptor"]["by_order"][k]["eps_eff"]
        for k in ORDERS:
            row[f"eps_E6_{k}"] = c["receptor"]["by_order"][k]["eps_E6"]
        rows.append(row)
    rows.sort(key=lambda r: (r["form"], r["rates"], r["chi_b_over_pi"], r["m"] if r["m"] is not None else -1,
                             r["alpha_deg"] or 0, r["beta_deg"]))
    return rows


def bounds_table():
    truth = {(r["seed"], r["point"]): r["eps_eff"] for r in load("truth", "truth.json")["records"]}
    rows = []
    for r in load("estimates", "estimates.json")["estimates"]:
        if r["point"] in ("R1", "R2", "R3") and truth[(r["seed"], r["point"])] == 0:
            est = r["estimate"]
            rows.append(dict(point=r["point"], seed=r["seed"], eps_hat_raw=est["eps_hat_raw"],
                             ci95_raw_low=est["ci95_raw"][0], ci95_raw_high=est["ci95_raw"][1],
                             upper95=est["upper95"], dipole_p=r["dipole"]["p"], antipodal_p=r["antipodal"]["p_odd"]))
    rows.sort(key=lambda r: (r["point"], r["seed"]))
    return rows


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print(name, len(rows), "rows")


write("cells_H1_H4.csv", cells_table())
write("upper_bounds_per_bird.csv", bounds_table())
