"""Paper figures F4 (benchmark), F5 (per-fuel error), F8 (activity detection).

Fixed colours per model family across all figures: national prior gray, GBT blue, tabular NN violet,
image NN aqua, fusion orange. Saved as PDF + PNG in results/figures/.

usage: python 26_paper_figures.py
"""
from pathlib import Path

import matplotlib
import matplotlib.dates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1] / "results"
FIG = R / "figures"
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#ecebe8"
FAM = {"prior": "#8a8984", "gbt": "#2a78d6", "tabnn": "#6250d6", "img": "#1baf7a", "fusion": "#eb6834"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

MODELS = [  # (label, comparison key, family)
    ("National prior", "natl_cf", "prior"),
    ("WRI published*", "wri_published*", "prior"),
    ("GBT (WRI features)", "gbt_wri", "gbt"),
    ("GBT + lat/lon", "gbt_plus", "gbt"),
    ("GBT residual", "gbt_wri_res", "gbt"),
    ("GBT residual + subtype", "SUBTYPE", "gbt"),
    ("Tabular NN", "tab_aux0", "tabnn"),
    ("Tabular NN residual", "tab-res_aux0", "tabnn"),
    ("Image NN", "img_aux0.3", "img"),
    ("Image NN residual", "img-res_aux0.3", "img"),
    ("Fusion NN", "fusion_concat_aux0.3", "fusion"),
    ("Fusion NN residual", "fusion_concat-res_aux0.3", "fusion"),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_benchmark():
    sub = pd.read_csv(R / "subtypes" / "subtype_cf_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, split, title in zip(axes, ("random", "geo"), ("(a) In distribution (random split)", "(b) Transfer: train US, test non-US")):
        c = pd.read_csv(R / "comparison_v2" / f"comparison_{split}.csv").query("subset == 'all'").set_index("model")
        prior = c.loc["natl_cf", "MAE"]
        for i, (label, key, fam) in enumerate(MODELS):
            if key == "SUBTYPE":
                mae, sd = sub[(sub.split == split) & (sub.set == "+db")].MAE.iloc[0], np.nan
            else:
                mae, sd = c.loc[key, "MAE"], c.loc[key, "MAE_sd"]
            y = len(MODELS) - 1 - i
            if np.isfinite(sd) and sd > 0:
                ax.errorbar(mae, y, xerr=sd, fmt="none", ecolor=FAM[fam], elinewidth=1, capsize=2)
            ax.plot(mae, y, "o", ms=6, color=FAM[fam], mec="#fcfcfb", mew=0.8)
            ax.annotate(f"{mae:.3f}", (mae, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=6.5, color=MUTED)
        ax.axvline(prior, color=FAM["prior"], lw=0.8, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("MAE of capacity factor (lower is better)")
        style(ax)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([m[0] for m in MODELS][::-1])
    axes[1].set_xlim(0.17, 0.235)
    axes[0].set_xlim(0.085, 0.155)
    fig.text(0.01, -0.02, "Dashed line: national prior. Error bars: ±1 s.d. over 3 seeds (neural models). "
             "* WRI estimates were partly fitted on these plants.", fontsize=6.5, color=MUTED)
    save(fig, "benchmark")


def fig_per_fuel():
    show = [("National prior", "natl_cf", "prior", "s"), ("GBT (WRI features)", "gbt_wri", "gbt", "o"),
            ("Fusion NN residual", "fusion_concat-res_aux0.3", "fusion", "D")]
    fuels = ["Waste", "Oil", "Nuclear", "Coal", "Wind", "Hydro", "Gas", "Solar"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, split, title in zip(axes, ("random", "geo"), ("(a) In distribution", "(b) Transfer (non-US)")):
        pf = pd.read_csv(R / "comparison_v2" / f"per_fuel_{split}.csv").set_index("primary_fuel").reindex(fuels)
        ys = np.arange(len(fuels))
        ok = (pf.n >= 10).values
        for k, (label, key, fam, mk) in enumerate(show):
            ax.plot(pf[key][ok], ys[ok] + (k - 1) * 0.22, mk, ms=5, color=FAM[fam], mec="#fcfcfb", mew=0.6, label=label, ls="none")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{f} (n={int(n)})" if n >= 10 else f"{f} (n<10)" for f, n in zip(fuels, pf.n.fillna(0))])
        ax.set_title(title, loc="left")
        ax.set_xlabel("MAE of capacity factor")
        style(ax)
    fig.subplots_adjust(wspace=0.45)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.2, -0.17), ncol=3)
    save(fig, "per_fuel")


def fig_activity():
    d = pd.read_parquet(R / "onoff" / "s0" / "frame_preds.parquet")
    us = d[d.country == "USA"].dropna(subset=["on"])
    rows = []
    for pid, g in us.groupby("gppd_idnr"):
        if (g.on == 1).sum() >= 3 and (g.on == 0).sum() >= 3:
            rows.append((pid, roc_auc_score(g.on, g.p_on), g.primary_fuel.iloc[0], len(g)))
    w = pd.DataFrame(rows, columns=["gppd_idnr", "auc", "fuel", "n"])
    fig = plt.figure(figsize=(7.2, 3.6))
    ax0 = fig.add_axes([0.06, 0.15, 0.26, 0.72])
    ax0.hist(w.auc, bins=np.linspace(0.2, 1, 17), color=FAM["fusion"], edgecolor="#fcfcfb", lw=1)
    ax0.axvline(w.auc.median(), color=INK, lw=1, ls="--")
    ax0.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax0.annotate(f"median {w.auc.median():.2f}", (w.auc.median(), ax0.get_ylim()[1] * 0.92), xytext=(-4, 0),
                 textcoords="offset points", ha="right", fontsize=6.5)
    ax0.set_xlabel("within-plant AUROC (on vs off)")
    ax0.set_ylabel("US test plants")
    ax0.set_title(f"(a) n = {len(w)} plants", loc="left")
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    # (b, c) two example plants: one coal near the 75th percentile, one gas near the median
    picks = []
    for fuel, q in (("Coal", 0.5), ("Gas", 0.5)):
        cand = w[(w.fuel == fuel) & (w.n >= 20)]
        picks.append(cand.iloc[(cand.auc - cand.auc.quantile(q)).abs().argsort().iloc[0]])
    for j, r in enumerate(picks):
        ax = fig.add_axes([0.42, 0.57 - j * 0.45, 0.56, 0.32])
        g = us[us.gppd_idnr == r.gppd_idnr].sort_values("date")
        t = pd.to_datetime(g.date)
        ax.plot(t, g.inst_util, "s", ms=3.5, color=FAM["prior"], label="CEMS utilisation at overpass hour")
        ax.plot(t, g.p_on, "o-", ms=3, lw=0.7, color=FAM["fusion"], label="predicted P(on)")
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"({'bc'[j]}) {r.fuel} plant, {int(g.capacity_mw.iloc[0])} MW, within-plant AUROC {r.auc:.2f}", loc="left", fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2)
    save(fig, "activity")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_benchmark()
    fig_per_fuel()
    fig_activity()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
