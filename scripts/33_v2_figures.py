"""Figure for the revised paper: within-region vs cross-country results (coal + gas).

(a) Leave-one-US-state-out: per-state MAE of the national prior vs the residual GBT + subtype + on-rate
    (marker area ~ plants in the state); points below the diagonal = model better.
(b) Europe (train US): gain in MAE over the national prior with plant-bootstrap 95 % CIs, for every approach.

usage: python 33_v2_figures.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#ecebe8"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def main():
    loso = pd.read_csv(R / "revision_cg" / "loso_by_state.csv")
    w = loso.pivot(index="state", columns="model", values="MAE")
    n = loso.pivot(index="state", columns="model", values="n")["natl_prior"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})

    lim = (0.05, 0.36)
    a.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=1)
    a.scatter(w["natl_prior"], w["+sub+onrate_matched"], s=8 + n * 0.25, color=ORANGE, alpha=0.75,
              edgecolor="#fcfcfb", lw=0.5, zorder=2)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("MAE, national prior")
    a.set_ylabel("MAE, GBT + subtype + on-rate")
    better = int((w["+sub+onrate_matched"] < w["natl_prior"]).sum())
    a.set_title(f"(a) Within the US: held-out states\n{better} of {len(w)} states below the diagonal", loc="left")
    a.text(0.30, 0.07, "model better", color=MUTED, fontsize=7, ha="center")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(color=GRID, lw=0.5)

    ci_rev = pd.read_csv(R / "revision_cg" / "ci.csv")
    ci_b = pd.read_csv(R / "relative_fewshot" / "B_ci.csv")
    rows = [("+ subtype", ci_b, "+sub"), ("+ subtype + on-rate", ci_b, "+sub+onrate (raw)"),
            ("+ subtype + relative on-rate (B1)", ci_b, "B1 +sub+relative"),
            ("allocation, no training (B2)", ci_b, "B2 allocation"),
            ("shrunk allocation (B3)", ci_b, "B3 shrunk allocation")]
    ys = np.arange(len(rows))[::-1]
    for y, (label, df, key) in zip(ys, rows):
        r = df[(df.subset == "Europe") & (df.model == key) & (df["resample"] == "plant")].iloc[0]
        color = ORANGE if "on-rate" in label or "allocation" in label else BLUE
        b.errorbar(r.gain_vs_prior, y, xerr=[[r.gain_vs_prior - r.ci_low], [r.ci_high - r.gain_vs_prior]],
                   fmt="o", color=color, ms=5, elinewidth=1.2, capsize=2)
    b.axvline(0, color=GRAY, lw=0.9, ls="--")
    b.set_yticks(ys)
    b.set_yticklabels([r[0] for r in rows])
    b.set_xlabel("MAE gain over national prior (95% CI)")
    b.set_title("(b) Across countries: train US, test Europe\n(240 coal and gas plants, 21 countries)", loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(R / "figures" / f"within_vs_cross.{ext}", dpi=220, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
