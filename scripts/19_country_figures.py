"""Maps and fuel-mix-adjusted error analysis from 18_country_analysis.py output.

Expected MAE of a region = sum over fuels of (region's share of plants of that fuel) x (pooled MAE of that fuel
in the same regime). ratio = actual / expected: < 1 the region is easier than its fuel mix implies, > 1 harder.

Figures (PDF + PNG in <fig_dir>):
  map_error.*      (a) world, leave-one-country-out MAE + Europe inset; (b) contiguous US states, out-of-fold MAE
  map_ratio.*      same layout, actual / fuel-mix-expected MAE (diverging, log scale, gray = as expected)
  expected_vs_actual.*  scatter of expected vs actual MAE per region
Tables: <out_dir>/region_adjusted.csv, correlation of the ratio with region diagnostics.

usage: python 19_country_figures.py <country_dir> <plants.parquet> <naturalearth_dir> <fig_dir>
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

MODEL = "gbt_wri_res"
MIN_N = 9
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
NODATA, EDGE, INK, MUTED = "#e6e5e1", "#ffffff", "#2b2b29", "#6b6b66"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": "DejaVu Sans", "text.color": INK})


def expected_mae(P, plants, key, regime):
    d = P[(P.regime == regime) & (P.model == MODEL)].merge(plants[["gppd_idnr", "primary_fuel"]], on="gppd_idnr")
    d["ae"] = (d.pred - d.cf_target).abs()
    fuel_mae = d.groupby("primary_fuel").ae.mean()
    d["fuel_mae"] = d.primary_fuel.map(fuel_mae)
    g = d.dropna(subset=[key]).groupby(key)
    return pd.DataFrame({"n": g.size(), "actual": g.ae.mean(), "expected": g.fuel_mae.mean()}).assign(
        ratio=lambda x: x.actual / x.expected, regime=regime, level=key)


def draw(ax, shapes, col, cmap, norm, label_n=None):
    shapes.plot(ax=ax, color=NODATA, edgecolor=EDGE, linewidth=0.3)
    has = shapes[shapes[col].notna()]
    has.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=EDGE, linewidth=0.4)
    ax.set_axis_off()


def figure(world, states, cty, st, col, cmap, norm, cbar_label, title_a, title_b, path, ticks=None):
    fig = plt.figure(figsize=(7.2, 6.2))
    ax_w = fig.add_axes([0.0, 0.47, 0.72, 0.50])
    ax_e = fig.add_axes([0.70, 0.52, 0.30, 0.42])
    ax_u = fig.add_axes([0.05, 0.03, 0.70, 0.42])
    w = world.merge(cty[[col, "n"]], left_on="ISO_A3_EH", right_index=True, how="left")
    draw(ax_w, w[w.CONTINENT != "Antarctica"], col, cmap, norm)
    ax_w.set_title(title_a, loc="left")
    draw(ax_e, w, col, cmap, norm)
    ax_e.set_xlim(-11, 30)
    ax_e.set_ylim(35, 70)
    ax_e.set_title("Europe", loc="left", fontsize=8, color=MUTED)
    for _, r in w[w[col].notna()].iterrows():  # sample sizes on the inset / world
        ax = ax_e if (-11 < r.geometry.centroid.x < 30 and 35 < r.geometry.centroid.y < 70) else ax_w
        p = r.geometry.representative_point()
        dark = cmap(norm(r[col]))[:3] if np.isfinite(r[col]) else (1, 1, 1)
        txt = "#ffffff" if 0.2126 * dark[0] + 0.7152 * dark[1] + 0.0722 * dark[2] < 0.45 else INK
        ax.annotate(f"{int(r['n'])}", (p.x, p.y), ha="center", va="center", fontsize=5.5, color=txt)
    s = states.merge(st[[col, "n"]], left_on="postal", right_index=True, how="left")
    s = s[~s.postal.isin(["AK", "HI"])].to_crs("EPSG:5070")
    draw(ax_u, s, col, cmap, norm)
    ax_u.set_title(title_b, loc="left")
    cax = fig.add_axes([0.80, 0.12, 0.02, 0.28])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.ax.minorticks_off()
    fig.text(0.80, 0.07, "gray: no labeled plants\nnumbers: labeled plants", fontsize=6, color=MUTED)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    cdir, plants_pq, ne, fig_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    fig_dir.mkdir(parents=True, exist_ok=True)
    plants = pd.read_parquet(plants_pq)
    P = pd.read_parquet(cdir / "preds.parquet")
    P = P.merge(plants[["gppd_idnr", "cf_target", "country"]], on="gppd_idnr")
    eia = Path(plants_pq).parents[1] / "raw" / "eia" / "eia8602019" / "2___Plant_Y2019.xlsx"
    stt = pd.read_excel(eia, skiprows=1)[["Plant Code", "State"]]
    stt = stt.set_index(pd.to_numeric(stt["Plant Code"], errors="coerce")).State
    P["state"] = [stt.get(int(g[3:])) if g.startswith("USA") and g[3:].isdigit() else None for g in P.gppd_idnr]

    adj = pd.concat([expected_mae(P, plants, "country", "loco"), expected_mae(P, plants, "country", "oof"),
                     expected_mae(P, plants, "state", "oof")])
    adj.round(4).to_csv(cdir / "region_adjusted.csv")
    cty = adj[(adj.level == "country") & (adj.regime == "loco") & (adj.n >= MIN_N)]
    st = adj[(adj.level == "state") & (adj.regime == "oof") & (adj.n >= MIN_N)]
    print("countries (loco):\n", cty.sort_values("ratio").round(3).to_string())
    print("\nstates (oof), best/worst 8 by ratio:\n", pd.concat([st.sort_values("ratio").head(8), st.sort_values("ratio").tail(8)]).round(3).to_string())

    world = gpd.read_file(ne / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp")
    states = gpd.read_file(ne / "ne_110m_admin_1_states_provinces" / "ne_110m_admin_1_states_provinces.shp")
    lo = min(cty.actual.min(), st.actual.min())
    hi = max(cty.actual.max(), st.actual.max())
    figure(world, states, cty.rename(columns={"actual": "v"}), st.rename(columns={"actual": "v"}), "v", SEQ,
           Normalize(np.floor(lo * 50) / 50, np.ceil(hi * 50) / 50), "MAE of capacity factor",
           "(a) Countries: leave-one-country-out", "(b) US states: 5-fold out-of-fold", fig_dir / "map_error")
    figure(world, states, cty.rename(columns={"ratio": "v"}), st.rename(columns={"ratio": "v"}), "v", DIV,
           LogNorm(0.5, 2.0), "actual / fuel-mix-expected MAE",
           "(a) Countries: leave-one-country-out", "(b) US states: 5-fold out-of-fold", fig_dir / "map_ratio",
           ticks=[0.5, 0.75, 1, 1.5, 2])

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0.05, 0.3], [0.05, 0.3], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(st.expected, st.actual, s=14, color="#86b6ef", edgecolor="#fcfcfb", lw=0.5, label="US states (out-of-fold)", zorder=2)
    ax.scatter(cty.expected, cty.actual, s=22, color="#eb6834", edgecolor="#fcfcfb", lw=0.5, label="countries (leave-one-out)", zorder=3)
    for iso, r in cty.iterrows():
        if abs(r.ratio - 1) > 0.15 or r.n > 200:
            dy = -7 if iso == "GRC" else 2
            ax.annotate(iso, (r.expected, r.actual), xytext=(3, dy), textcoords="offset points", fontsize=6, color=INK)
    ax.set_xlabel("expected MAE from fuel mix")
    ax.set_ylabel("actual MAE")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#ecebe8", lw=0.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"expected_vs_actual.{ext}", dpi=200)

    # what explains the residual difficulty (ratio)? Spearman with region diagnostics
    for key, regime, fname in (("country", "loco", "country_metrics.csv"), ("state", "oof", "state_metrics.csv")):
        D = pd.read_csv(cdir / fname, index_col=0)
        a = adj[(adj.level == key) & (adj.regime == regime) & (adj.n >= MIN_N)].join(D)
        diag = ["within_fuel_cf_sd", "prior_gap", "thermal_share", "vre_share", "hydro_share", "median_mw", "age_known", "n"]
        print(f"\nSpearman with ratio ({key}, {regime}, n={len(a)}):",
              {c: round(a.ratio.corr(a[c], method="spearman"), 2) for c in diag})
        print(f"Spearman with actual MAE ({key}):", {c: round(a.actual.corr(a[c], method="spearman"), 2) for c in diag})


if __name__ == "__main__":
    main()
