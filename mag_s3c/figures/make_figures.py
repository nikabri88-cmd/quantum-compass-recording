"""Figures of the MAG-S3c confirmatory run (written after the run; they only draw frozen outputs of results.json).

  k_half.png                     |d_odd . z| of form K at chi_b = 0.1 pi and m = 1 over its value without relaxation,
                                 on the scale of rates of both kinds of relaxation (H4, k_half), for S2 with the
                                 CISS axis z and W_C and W_D with the axis r
  coherence_vs_polarisation.png  eps_mol of form K at chi_b = 0.05 pi over the larger of the polarisation forms at
                                 chi = 0.1 pi without relaxation (H1), with the dipolar coupling as in the run and
                                 without it (map E3)

Usage: python3 -I make_figures.py RUN_DIR OUT_DIR   (RUN_DIR as for results/tables/make_tables.py)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

RUN, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = {"S2": "#2a78d6", "W_C": "#eb6834", "W_D": "#1baf7a"}          # categorical slots 1-3, validated
plt.rcParams.update({"font.family": "Inter", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def comma(x, _=None):
    s = f"{x:g}"
    return s.replace(".", ",")


def power10(x, _=None):
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    for e in range(-3, 10):
        if abs(x - 10.0 ** e) < 1e-9 * 10.0 ** e:
            return "10" + str(e).translate(sup)
    m, e = f"{x:.0e}".split("e")
    return f"{m}·10" + str(int(e)).translate(sup)


def sci_comma(x):
    m, e = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return m.replace(".", ",") + "·10" + str(int(e)).translate(sup)


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, head, sub):
    fig.text(0.035, 0.955, head, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.885, sub, fontsize=10, color=INK2, va="top")


def figure_k_half(res):
    kh = res["hypotheses"]["H4"]["k_half"]
    cases = (("S2", "z", "S2, ось z"), ("W_C", "r", "W_C, ось r̂"), ("W_D", "r", "W_D, ось r̂"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.66, bottom=0.2, wspace=0.08)
    title(fig, "Дефазировка гасит нечётный диполь вдвое при 3–7·10⁷ с⁻¹",
          "|d_odd·z| формы K при χ_b = 0,1π и m = 1 в долях значения без релаксации")
    for ax, kind, head in ((axes[0], "std", "дефазировка синглета и триплета"), (axes[1], "rf", "случайные поля")):
        style(ax)
        ax.set_xscale("log")
        if kind == "std":
            ax.axvspan(3e7, 6e7, color="#e9e7e1", zorder=0, linewidth=0)
            ax.text(4.25e7, 1.27, "по опытам\nв Cry4a", ha="center", va="top", fontsize=8.5, color=INK2)
        ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        for g, axis, label in cases:
            v = kh[f"{g}_{axis}_chi0.1_{kind}"]
            pts = [(k, d) for k, d in v["levels"]]
            d0 = abs(pts[0][1])
            xs = [k for k, _ in pts[1:]]
            ys = [abs(d) / d0 for _, d in pts[1:]]
            ax.plot(xs, ys, color=SERIES[g], linewidth=2, marker="o", markersize=4.5, zorder=3, label=label)
            k12 = v.get("k_half")
            if k12:
                ax.plot([k12], [0.5], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=SERIES[g],
                        markeredgewidth=1.4, zorder=4)
        ax.set_xlim(7e4, 1.4e8 if kind == "std" else 4.2e7)
        ax.set_ylim(0, 1.3)
        ticks = [1e5, 1e6, 1e7, 1e8] if kind == "std" else [1e5, 1e6, 1e7]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(power10))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(comma))
        ax.set_title(head, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("скорость k, с⁻¹", fontsize=9.5)
    axes[0].text(8e4, 0.53, "половина", fontsize=8.5, color=INK2, va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.025, 0.835), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.6, columnspacing=1.6)
    fig.text(0.035, 0.025, "Кружки отмечают k½. Без релаксации отношение равно 1. "
             "Источник: results.json подтверждающего прогона, H4.", fontsize=8, color=MUTED)
    path = os.path.join(OUT, "k_half.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


def figure_h1(res):
    e3 = res["maps"]["E3"]
    h1 = {c["label"]: c for c in res["hypotheses"]["H1"]["S2"].get("known_in_advance", [])}
    cats = [("S2", "z"), ("S2", "x"), ("W_C", "z"), ("W_C", "x"), ("W_C", "r"), ("W_D", "z"), ("W_D", "x"),
            ("W_D", "r")]
    axis_name = {"z": "z", "x": "x", "r": "r̂"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.1, right=0.975, top=0.7, bottom=0.21)
    title(fig, "Диполь-дипольная связь снимает перевес формы когерентности",
          "ε_mol формы K в долях большей из форм поляризации, без релаксации, логарифмическая шкала")
    style(ax)
    ax.set_yscale("log")
    ax.axhline(10, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(1, color="#c9c8c2", linewidth=0.9, zorder=1)
    xs_d, ys_d, xs_n, ys_n = [], [], [], []
    for i, (g, a) in enumerate(cats):
        x = i + (0.6 if g != "S2" else 0) + (0.6 if g == "W_D" else 0)
        if g == "S2":
            ys = h1[f"S2_{a}_none0_eps_mol"]["value"]
            xs_n.append(x)
            ys_n.append(ys)
            continue
        v = e3[f"{g}_{a}_none0"]
        with_d, no_d = v["ratios_D"]["K_over_P_eps"], v["ratios_noD"]["K_over_P_eps"]
        ax.plot([x, x], [with_d, no_d], color="#c9c8c2", linewidth=1.6, zorder=2)
        xs_d.append(x)
        ys_d.append(with_d)
        xs_n.append(x)
        ys_n.append(no_d)
    ax.scatter(xs_n, ys_n, s=46, color=SERIES["W_C"], zorder=3, label="без диполь-дипольной связи (D = 0)",
               edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(xs_d, ys_d, s=46, color=SERIES["S2"], zorder=3, label="с диполь-дипольной связью, как в прогоне",
               edgecolors=SURFACE, linewidths=1.5)
    ticks = [i + (0.6 if g != "S2" else 0) + (0.6 if g == "W_D" else 0) for i, (g, _) in enumerate(cats)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([axis_name[a] for _, a in cats])
    for g, centre in (("S2", 0.5), ("W_C", 3.6), ("W_D", 7.2)):
        ax.text(centre, -0.17, g + (" (тензоры MAG-S2, D = 0)" if g == "S2" else ""), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_ylim(0.01, 1e4)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000, 1e4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: comma(x) if x < 1000 else power10(x)))
    ax.text(1.8, 10 * 1.15, "порог H1: 10", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(1.8, 1 * 1.15, "поровну", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(-0.5, 8.9)
    fig.legend(loc="upper left", bbox_to_anchor=(0.025, 0.835), ncol=2, frameon=False, fontsize=9.5,
               handletextpad=0.3, columnspacing=1.6)
    fig.text(0.035, 0.025, "Ось CISS под каждой точкой. Источник: results.json подтверждающего прогона, "
             "H1 и карта E3.", fontsize=8, color=MUTED)
    path = os.path.join(OUT, "coherence_vs_polarisation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(RUN, "confirmatory_s3c", "results", "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    figure_k_half(res)
    figure_h1(res)


if __name__ == "__main__":
    main()
