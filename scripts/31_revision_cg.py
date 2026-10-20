"""Revision experiments for the coal + gas paper (reviewer response).

Cohort: labeled plants whose primary fuel is Coal or Gas.
Priors: national (Ember country x fuel), regional (Ember region median), global (Ember median), training-set fuel mean.
Models (LightGBM, 3 seeds):
  gbt_noprior        WRI-style features without the national prior
  gbt_prior_feat     national prior as an input feature
  gbt_res            national prior as residual anchor (trees learn cf - prior)
  +sub               + database technology subtype and cooling type
  +onrate_all        + satellite on-rate from all 2017-2019 clear frames
  +onrate_matched    + on-rate only from frames in the plant's label years (non-US labels mostly 2017-2018)
  +sub+onrate_all / +sub+onrate_matched
On-rate models are run once per on/off seed (results/onoff, onoff_s1, onoff_s2) and reported as mean +- s.d.;
confidence intervals use the seed-averaged predictions.
Protocols: random (cohort plants of split_random), geo (train US -> test non-US; Europe = main, India and
Australia = exploratory), loso (leave-one-US-state-out, key models only).
Uncertainty: plant-paired bootstrap and country-cluster bootstrap of MAE differences (5,000 resamples).

usage: python 31_revision_cg.py <project_root> <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = load("bl", "07_baselines.py")
ds = load("ds", "06_build_dataset.py")
EUROPE = bl.EUROPE
FUELS = ["Coal", "Gas"]
ONOFF_DIRS = ["onoff", "onoff_s1", "onoff_s2"]
SUB = ["subtype", "cooling"]
ON = ["on_rate", "util", "n_clear"]
CLEAR = 10


def onrates(root, run_dir, label_years):
    parts = []
    for k in range(5):
        fp = pd.read_parquet(root / "results" / run_dir / f"fold{k}" / "frame_preds.parquet",
                             columns=["gppd_idnr", "date", "scene_cloud", "p_on", "util_pred"])
        parts.append(fp.assign(fold=k))
    fp = pd.concat(parts)
    fp = fp[fp.scene_cloud < CLEAR]
    fp["year"] = pd.to_datetime(fp.date).dt.year

    def agg(d):
        g = d.groupby(["gppd_idnr", "fold"]).agg(on_rate=("p_on", "mean"), util=("util_pred", "mean"), n_clear=("p_on", "size"))
        return g.groupby(level=0).mean()  # US: one fold each; non-US: mean over the five fold models

    all_years = agg(fp)
    yrs = fp.gppd_idnr.map(label_years)
    in_years = np.array([y in s if isinstance(s, set) else True for y, s in zip(fp.year, yrs)])
    matched = agg(fp[in_years]).reindex(all_years.index)
    matched = matched.fillna(all_years)  # plants without frames in their label years fall back to all frames
    return all_years, matched


def evaluate(y, p, cap):
    e = np.abs(y - p)
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "MAE": e.mean(), "R2": 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan,
            "wMAPE_gen": (e * cap).sum() / (y * cap).sum()}


def boot_diff(te, pa, pb, by_country, n=5000, seed=0):
    """Mean of |err_a| - |err_b| (positive = b better) with a 95 % CI, resampling plants or countries."""
    d = pd.Series(np.abs(te.cf_target.values - pa) - np.abs(te.cf_target.values - pb), index=te.index)
    rng = np.random.default_rng(seed)
    if by_country:
        groups = [d[te.country.values == c].values for c in te.country.unique()]
        stats = []
        for _ in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            stats.append(np.concatenate([groups[i] for i in pick]).mean())
    else:
        v = d.values
        stats = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return d.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    plants = pd.read_parquet(root / "data/processed/plants.parquet")
    plants = plants.merge(pd.read_parquet(root / "data/processed/subtypes.parquet")[["gppd_idnr", *SUB]], on="gppd_idnr", how="left")
    L = plants[plants.split_random.notna() & plants.primary_fuel.isin(FUELS)].copy()

    # priors from Ember
    ctry, region, glob, iso_region = ds.ember_cf(root / "data/raw/ember/ember_yearly_long.csv")
    ef = L.primary_fuel.map(ds.EMBER_FUEL)
    L["region_cf"] = [region.get((iso_region.get(c), f), np.nan) for c, f in zip(L.country, ef)]
    L["global_cf"] = ef.map(glob)
    L["region_cf"] = L.region_cf.fillna(L.global_cf)

    # label years per plant (for year-matched on-rates); US uses GPPD 2017-2019 as for cf_target
    lab = pd.read_parquet(root / "data/processed/labels.parquet").query("valid & source == 'gppd' & year >= 2017 & year <= 2019")
    label_years = lab.groupby("gppd_idnr").year.apply(set)

    # state for leave-one-state-out
    st = pd.read_excel(root / "data/raw/eia/eia8602019/2___Plant_Y2019.xlsx", skiprows=1)[["Plant Code", "State"]]
    st = st.set_index(pd.to_numeric(st["Plant Code"], errors="coerce")).State
    L["state"] = [st.get(int(g[3:])) if g.startswith("USA") and g[3:].isdigit() else None for g in L.gppd_idnr]

    for c in ["primary_fuel", "other_fuel1", *SUB]:
        L[c] = L[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", *SUB]
    noprior = [f for f in bl.F_WRI if f != "natl_cf"]

    rates = [onrates(root, d, label_years) for d in ONOFF_DIRS]
    base_sets = {"gbt_noprior": (noprior, False), "gbt_prior_feat": (bl.F_WRI, False), "gbt_res": (bl.F_WRI, True),
                 "+sub": (bl.F_WRI + SUB, True)}
    on_sets = {"+onrate_all": (bl.F_WRI + ON, "all"), "+onrate_matched": (bl.F_WRI + ON, "matched"),
               "+sub+onrate_all": (bl.F_WRI + SUB + ON, "all"), "+sub+onrate_matched": (bl.F_WRI + SUB + ON, "matched")}
    print("cohort:", len(L), L.primary_fuel.value_counts().to_dict(), flush=True)

    rows, ci_rows, pred_store = [], [], {}
    for split in ("random", "geo"):
        col = f"split_{split}"
        tr, va, te = (L[L[col] == s] for s in ("train", "val", "test"))
        preds = {"natl_prior": te.natl_cf.clip(0, 1).values, "region_prior": te.region_cf.clip(0, 1).values,
                 "global_prior": te.global_cf.clip(0, 1).values,
                 "fuel_mean": te.primary_fuel.map(tr.groupby("primary_fuel", observed=True).cf_target.mean()).astype(float).values}
        seed_sd = {}
        for name, (feats, res) in base_sets.items():
            preds[name] = bl.gbt_predict(tr, va, te, feats, residual=res)
        for name, (feats, kind) in on_sets.items():
            per_seed = []
            for all_r, matched_r in rates:
                r = all_r if kind == "all" else matched_r
                f = lambda d: d.drop(columns=ON, errors="ignore").join(r, on="gppd_idnr")
                per_seed.append(bl.gbt_predict(f(tr), f(va), f(te), feats, residual=True))
            preds[name] = np.mean(per_seed, axis=0)
            seed_sd[name] = np.std([np.abs(te.cf_target.values - p).mean() for p in per_seed])
        pred_store[split] = pd.DataFrame(preds).assign(gppd_idnr=te.gppd_idnr.values)

        region = np.where(te.country == "USA", "US", np.where(te.country.isin(EUROPE), "Europe",
                          np.where(te.country == "IND", "India", np.where(te.country == "AUS", "Australia", "other"))))
        subsets = [("all", np.ones(len(te), bool))] + [(f, (te.primary_fuel == f).values) for f in FUELS]
        if split == "geo":
            subsets += [(f"{r}", region == r) for r in ("Europe", "India", "Australia")]
            subsets += [(f"Europe:{f}", (region == "Europe") & (te.primary_fuel == f).values) for f in FUELS]
        for sub, m in subsets:
            if m.sum() < 10:
                continue
            for k, p in preds.items():
                r = evaluate(te.cf_target.values[m], p[m], te.capacity_mw.values[m])
                if split == "geo" and sub in ("all", "Europe"):
                    tm = te[m]
                    r["MAE_country_macro"] = pd.Series(np.abs(tm.cf_target.values - p[m]), index=tm.index).groupby(tm.country.values).mean().mean()
                rows.append({"split": split, "subset": sub, "model": k, "seed_sd": seed_sd.get(k, np.nan) if sub == "all" else np.nan, **r})
        # confidence intervals for the key comparisons
        comps = [("natl_prior", "+sub+onrate_matched"), ("natl_prior", "+sub+onrate_all"), ("+sub", "+sub+onrate_matched"),
                 ("gbt_res", "+sub+onrate_matched"), ("natl_prior", "gbt_res"), ("gbt_noprior", "gbt_prior_feat"),
                 ("gbt_prior_feat", "gbt_res")]
        for sub, m in subsets:
            if m.sum() < 30 or sub not in ("all", "Coal", "Gas", "Europe", "Europe:Coal", "Europe:Gas"):
                continue
            for a, b in comps:
                for by_c in ((False, True) if split == "geo" else (False,)):
                    mean, lo, hi = boot_diff(te[m], preds[a][m], preds[b][m], by_c)
                    ci_rows.append({"split": split, "subset": sub, "a": a, "b": b, "resample": "country" if by_c else "plant",
                                    "gain_b_over_a": mean, "ci_low": lo, "ci_high": hi})
        print(split, "done", flush=True)

    # leave-one-US-state-out (key models)
    us = L[L.country == "USA"].dropna(subset=["state"])
    loso_rows = []
    rate0_all, rate0_matched = rates[0]
    for s in sorted(us.state.unique()):
        te = us[us.state == s]
        if len(te) < 10:
            continue
        trv = us[us.state != s]
        va = trv.sample(frac=0.1, random_state=0)
        tr = trv.drop(va.index)
        f = lambda d: d.join(rate0_matched, on="gppd_idnr")
        p = {"natl_prior": te.natl_cf.clip(0, 1).values,
             "gbt_res": bl.gbt_predict(tr, va, te, bl.F_WRI, residual=True),
             "+sub": bl.gbt_predict(tr, va, te, bl.F_WRI + SUB, residual=True),
             "+sub+onrate_matched": bl.gbt_predict(f(tr), f(va), f(te), bl.F_WRI + SUB + ON, residual=True)}
        for k, v in p.items():
            loso_rows.append({"state": s, "model": k, **evaluate(te.cf_target.values, v, te.capacity_mw.values)})
    loso = pd.DataFrame(loso_rows)
    loso_sum = loso.groupby("model").agg(states=("state", "nunique"), plants=("n", "sum"),
                                         MAE_state_macro=("MAE", "mean"), MAE_state_sd=("MAE", "std"))
    loso_sum["MAE_plant_weighted"] = loso.assign(w=loso.MAE * loso.n).groupby("model").w.sum() / loso.groupby("model").n.sum()
    loso_sum["states_beating_prior"] = loso.pivot(index="state", columns="model", values="MAE").pipe(
        lambda t: pd.Series({m: int((t[m] < t["natl_prior"]).sum()) for m in t.columns}))

    res = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows)
    res.round(4).to_csv(out / "results.csv", index=False)
    ci.round(4).to_csv(out / "ci.csv", index=False)
    loso.round(4).to_csv(out / "loso_by_state.csv", index=False)
    loso_sum.round(4).to_csv(out / "loso_summary.csv")
    for s, p in pred_store.items():
        p.to_csv(out / f"preds_{s}.csv", index=False)

    pd.set_option("display.width", 250)
    order = ["natl_prior", "region_prior", "global_prior", "fuel_mean", "gbt_noprior", "gbt_prior_feat", "gbt_res", "+sub",
             "+onrate_all", "+onrate_matched", "+sub+onrate_all", "+sub+onrate_matched"]
    piv = res.pivot_table(index="model", columns=["split", "subset"], values="MAE").reindex(order)
    print("\nMAE:\n", piv.round(3).to_string())
    print("\nseed s.d. (all):\n", res[res.subset == "all"].pivot_table(index="model", columns="split", values="seed_sd").reindex(order).round(4).to_string())
    print("\ncountry-macro MAE (geo):\n", res[(res.split == "geo") & res.subset.isin(["all", "Europe"])].pivot_table(index="model", columns="subset", values="MAE_country_macro").reindex(order).round(3).to_string())
    print("\nwMAPE_gen (all):\n", res[res.subset == "all"].pivot_table(index="model", columns="split", values="wMAPE_gen").reindex(order).round(3).to_string())
    print("\nCIs:\n", ci.round(4).to_string(index=False))
    print("\nLOSO:\n", loso_sum.round(4).to_string())


if __name__ == "__main__":
    main()
