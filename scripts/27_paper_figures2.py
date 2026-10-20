"""Paper figures F2 (label coverage) and F3 (imagery examples).

F2: (a) world map of labeled plants per country (log scale), (b) fuel composition of labeled plants vs all GPPD.
F3: (a) 2019 Sentinel-2 patch of the largest labeled plant of eight fuels (true colour, 2.56 km);
    (b) monthly frames of one US coal plant, four with the plant on and four off at the overpass hour (EPA CEMS).

usage: python 27_paper_figures2.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
INK, MUTED, NODATA = "#2b2b29", "#6b6b66", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rgb(x, lo=None, hi=None):
    """True-colour composite with a per-image 2-98 % stretch."""
    img = x[[2, 1, 0]].astype(np.float32).transpose(1, 2, 0)
    lo = np.percentile(img, 2) if lo is None else lo
    hi = np.percentile(img, 98) if hi is None else hi
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


def fig_coverage(plants):
    L = plants[plants.split_random.notna()]
    world = gpd.read_file(ROOT / "data/raw/naturalearth/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    n = L.groupby("country").size().rename("n")
    w = world.merge(n, left_on="ISO_A3_EH", right_index=True, how="left")
    w = w[w.CONTINENT != "Antarctica"]
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_axes([0.0, 0.42, 0.86, 0.56])
    w.plot(ax=ax, color=NODATA, edgecolor="#ffffff", linewidth=0.3)
    w[w.n.notna()].plot(ax=ax, column="n", cmap=SEQ, norm=LogNorm(1, 10000), edgecolor="#ffffff", linewidth=0.3)
    ax.set_axis_off()
    ax.set_title(f"(a) Plants with reported generation: {len(L):,} in {L.country.nunique()} countries "
                 f"(of {len(plants):,} plants in {plants.country.nunique()} countries)", loc="left")
    cax = fig.add_axes([0.87, 0.52, 0.015, 0.36])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1, 10000), cmap=SEQ), cax=cax)
    cb.set_label("labeled plants")
    cb.outline.set_visible(False)

    ax2 = fig.add_axes([0.16, 0.06, 0.78, 0.30])
    fuels = plants.primary_fuel.value_counts().index[:12][::-1]
    share_all = plants.primary_fuel.value_counts(normalize=True).reindex(fuels)
    share_lab = L.primary_fuel.value_counts(normalize=True).reindex(fuels).fillna(0)
    y = np.arange(len(fuels))
    ax2.barh(y + 0.2, share_all, height=0.38, color="#b5b4af", label="all GPPD plants")
    ax2.barh(y - 0.2, share_lab, height=0.38, color="#2a78d6", label="labeled plants")
    ax2.set_yticks(y)
    ax2.set_yticklabels(fuels)
    ax2.set_xlabel("share of plants")
    ax2.set_title("(b) Fuel composition", loc="left")
    ax2.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="x", color="#ecebe8", lw=0.6)
    ax2.set_axisbelow(True)
    save(fig, "coverage")


def fig_imagery(plants):
    L = plants[plants.split_random.notna()]
    idx = pd.read_csv(ROOT / "data/raw/s2_2019/index.csv").drop_duplicates("gppd_idnr", keep="last")
    ok = set(idx.gppd_idnr[idx.status == "ok"])
    fuels = ["Coal", "Gas", "Nuclear", "Hydro", "Solar", "Wind", "Oil", "Waste"]
    fig = plt.figure(figsize=(7.2, 3.0))
    for i, f in enumerate(fuels):
        c = L[(L.primary_fuel == f) & L.gppd_idnr.isin(ok)].sort_values("capacity_mw", ascending=False)
        r = c.iloc[0]
        x = np.load(ROOT / "data/raw/s2_2019/patches" / f"{r.gppd_idnr}.npy")
        ax = fig.add_axes([i * 0.125, 0.52, 0.12, 0.40])
        ax.imshow(rgb(x))
        ax.set_axis_off()
        ax.set_title(f"{f}\n{r.capacity_mw:,.0f} MW, CF {r.cf_target:.2f}", fontsize=6.5)
    fig.text(0.0, 0.97, "(a) Sentinel-2, 2019, 2.56 km around the largest labeled plant of each fuel", fontsize=8.5)

    # (b) one US coal plant: clear monthly frames with the plant on vs off at the overpass hour
    cems = pd.read_parquet(ROOT / "data/processed/cems_overpass.parquet")
    coal = set(L[(L.primary_fuel == "Coal") & L.is_us].gppd_idnr)
    best = None
    for pid in sorted(coal, key=lambda p: -float(L.set_index("gppd_idnr").capacity_mw.get(p, 0))):
        fp = ROOT / "data/raw/s2_ts/series" / f"{pid}.npz"
        if not fp.exists():
            continue
        d = np.load(fp)
        lab = pd.DataFrame({"date": d["dates"], "cloud": d["cloud"]}).merge(cems[cems.gppd_idnr == pid], on="date")
        lab = lab[lab.cloud < 5]
        if (lab.on == 1).sum() >= 4 and (lab.on == 0).sum() >= 4:
            best = (pid, d, lab)
            break
    pid, d, lab = best
    sel = pd.concat([lab[lab.on == 1].head(4), lab[lab.on == 0].head(4)]).sort_values("date")
    pos = {dt: k for k, dt in enumerate(d["dates"])}
    stack = np.stack([d["x"][pos[dt]] for dt in sel.date])
    lo, hi = np.percentile(stack[:, [2, 1, 0]], 2), np.percentile(stack[:, [2, 1, 0]], 98)  # common stretch
    for i, (_, r) in enumerate(sel.iterrows()):
        ax = fig.add_axes([i * 0.125, 0.02, 0.12, 0.40])
        ax.imshow(rgb(d["x"][pos[r.date]], lo, hi))
        ax.set_axis_off()
        state = "on" if r.on == 1 else "off"
        ax.set_title(f"{r.date}\n{state} (load {r.inst_util:.0%})", fontsize=6.5,
                     color="#c2410c" if r.on == 1 else MUTED)
    cap = L.set_index("gppd_idnr").capacity_mw[pid]
    fig.text(0.0, 0.47, f"(b) Monthly frames (960 m) of a {cap:,.0f} MW US coal plant; state at the overpass hour from EPA CEMS",
             fontsize=8.5)
    save(fig, "imagery_examples")
    print("F3 series plant:", pid)


def main():
    plants = pd.read_parquet(ROOT / "data/processed/plants.parquet")
    fig_coverage(plants)
    fig_imagery(plants)


if __name__ == "__main__":
    main()
