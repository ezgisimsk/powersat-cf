"""Figure for the revised paper: within-region vs cross-country results (coal + gas).

(a) Leave-one-US-state-out: per-state MAE of the national prior vs the residual GBT + subtype + on-rate
    (marker area ~ plants in the state); points below the diagonal = model better.
(b) Europe (train US): gain in MAE over the national prior with plant-bootstrap 95 % CIs, for every approach.

usage: python 33_v2_figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#ecebe8"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def main():
    # held-out states: on-rates from activity models trained on state-grouped folds, three seeds (scripts/38)
    lp = pd.read_csv(R / "state_folds_oracle" / "loso_preds.csv")
    m = {"natl_prior": "natl_prior", "+sub+onrate_matched": "+sub+onrate [state_folds, 3 seeds]"}
    w = pd.DataFrame({k: (lp[v] - lp.cf_target).abs().groupby(lp.state).mean() for k, v in m.items()})
    n = lp.groupby("state").size()
    # drawn at the final text width (5.4 in) so that no text prints below 7 pt
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 3.0), gridspec_kw={"width_ratios": [1, 1.1]})

    lim = (0.05, 0.36)
    a.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=1)
    a.scatter(w["natl_prior"], w["+sub+onrate_matched"], s=5 + n * 0.15, color=ORANGE, alpha=0.75,
              edgecolor="#fcfcfb", lw=0.5, zorder=2)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("MAE, national prior")
    a.set_ylabel("MAE, GBT + subtype + on-rate")
    better = int((w["+sub+onrate_matched"] < w["natl_prior"]).sum())
    a.set_title(f"(a) Within the US: held-out states\n{better} of {len(w)} below the diagonal", loc="left")
    a.text(0.355, 0.20, "model\nbetter", color=MUTED, fontsize=7, ha="right", va="bottom")  # empty area below the diagonal
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(color=GRID, lw=0.5)

    ci_rev = pd.read_csv(R / "revision_cg" / "ci.csv")
    ci_b = pd.read_csv(R / "relative_fewshot" / "B_ci.csv")
    rows = [("+ subtype", ci_b, "+sub"), ("+ subtype + on-rate", ci_b, "+sub+onrate (raw)"),
            ("+ subtype + relative on-rate", ci_b, "B1 +sub+relative"),
            ("allocation by relative on-rate", ci_b, "B2 allocation"),
            ("shrunk allocation", ci_b, "B3 shrunk allocation")]
    ys = np.arange(len(rows))[::-1] + 1.4
    for y, (label, df, key) in zip(ys, rows):
        r = df[(df.subset == "Europe") & (df.model == key) & (df["resample"] == "plant")].iloc[0]
        color = ORANGE
        b.errorbar(r.gain_vs_prior, y, xerr=[[r.gain_vs_prior - r.ci_low], [r.ci_high - r.gain_vs_prior]],
                   fmt="o", color=color, ms=4, elinewidth=1.0, capsize=2)
    # oracle: true on-rate from ENTSO-E labels, an upper bound not available in deployment (scripts/38);
    # hollow marker below a divider, country-cluster 95% CI
    g, lo, hi = json.load(open(R / "state_folds_oracle" / "state_folds_oracle.json"))["europe"]["all"][
        "gain_ci_country"]["+sub+onrate [oracle] over natl_prior"]
    b.axhline(0.7, color=MUTED, lw=0.6)
    b.errorbar(g, 0, xerr=[[g - lo], [hi - g]], fmt="o", mfc="white", mec=INK, ecolor=INK, ms=4, mew=1.0,
               elinewidth=1.0, capsize=2)
    b.axvline(0, color=GRAY, lw=0.9, ls="--")
    b.set_yticks(list(ys) + [0])
    b.set_yticklabels([r[0] for r in rows] + ["true on-rate (oracle,\nupper bound)"])
    b.set_xlabel("MAE reduction vs national prior\n(positive = better)")
    b.set_title("(b) Train US, test Europe\n(240 plants, 21 countries)", loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(R / "figures" / f"within_vs_cross.{ext}", dpi=220, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
