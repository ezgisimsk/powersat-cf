"""Graphical abstract for the EMS submission: pipeline on the left, within-region and cross-country results on the right.

Labelled plants (1,458 = 1,235 US CEMS + 223 ENTSO-E) as in the cohort table of the paper.
Numbers are read from results/state_folds_oracle/state_folds_oracle.json (held-out US states, all plants; Europe).
Output: results/figures/graphical_abstract.{pdf,png} (13.3 x 5.3 cm aspect, 300 dpi PNG).

usage: python 40_graphical_abstract.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path(__file__).resolve().parents[1] / "results"
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#ecebe8"
BLUE, ORANGE, GRAY, TEAL = "#2a78d6", "#eb6834", "#8a8984", "#2a9d8f"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=color, ec="none", alpha=0.15, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, transform=ax.transAxes)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.9,
                                 transform=ax.transAxes))


def bars(ax, labels, values, colors, hatches, title, ylim):
    xs = range(len(values))
    for x, v, c, hch in zip(xs, values, colors, hatches):
        ax.bar(x, v, color=c if not hch else "white", edgecolor=c, hatch=hch, lw=1.0, width=0.65)
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(*ylim)
    ax.set_ylabel("MAE of capacity factor", fontsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main():
    r = json.load(open(R / "state_folds_oracle" / "state_folds_oracle.json"))
    us, eu = r["loso"]["all"]["MAE"], r["europe"]["all"]["MAE"]
    # unrounded Europe prior (state_folds_oracle.json stores 4 decimals, 0.1695, which would print as 0.170)
    eu = dict(eu, natl_prior=json.load(open(R / "multiregion" / "multiregion.json"))["cf_europe"]["MAE"]["natl_prior"])
    fig = plt.figure(figsize=(7.2, 2.88))
    a = fig.add_axes([0.0, 0.0, 0.40, 1.0])
    a.axis("off")
    box(a, 0.03, 0.70, 0.42, 0.20, "Monthly Sentinel-2\nframes, 2017–2019\n(all 2,626 plants)", BLUE)
    box(a, 0.55, 0.70, 0.42, 0.20, "Overpass-hour load,\n1,458 plants\n(US CEMS, ENTSO-E)", GRAY)
    box(a, 0.15, 0.40, 0.70, 0.18, "Image model: is the plant running?", TEAL)
    box(a, 0.15, 0.10, 0.70, 0.18, "On-rate + subtype + national prior\n→ annual capacity factor", ORANGE)
    arrow(a, 0.24, 0.70, 0.40, 0.58)
    arrow(a, 0.76, 0.70, 0.60, 0.58)
    arrow(a, 0.50, 0.40, 0.50, 0.28)
    a.text(0.03, 0.97, "Coal and gas power plants", fontsize=8, fontweight="bold", va="top", transform=a.transAxes)

    b = fig.add_axes([0.47, 0.20, 0.23, 0.62])
    bars(b, ["national\nprior", "+ subtype", "+ subtype\n+ on-rate"],
         [us["natl_prior"], us["+sub"], us["+sub+onrate [state_folds, 3 seeds]"]],
         [GRAY, BLUE, ORANGE], [None, None, None], "Within the US\n(held-out states)", (0, 0.28))
    c = fig.add_axes([0.76, 0.20, 0.23, 0.62])
    bars(c, ["national\nprior", "+ subtype\n+ predicted\non-rate", "+ subtype\n+ true\non-rate*"],
         [eu["natl_prior"], eu["+sub+onrate [predicted]"], eu["+sub+onrate [oracle]"]],
         [GRAY, ORANGE, INK], [None, None, "////"], "Across countries\n(train US, test Europe)", (0, 0.28))
    c.set_ylabel("")
    fig.text(0.735, -0.10, "Satellite activity helps within a region; across countries, detection is the main loss.\n"
             "* true on-rate: upper bound, not available in deployment",
             ha="center", fontsize=7.5, color=INK)
    out = R / "figures" / "graphical_abstract"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
