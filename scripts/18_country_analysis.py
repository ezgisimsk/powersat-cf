"""Where do we estimate plant capacity factors well, and why?  Per-country (and per-US-state) analysis.

Regimes
  oof    5-fold out-of-fold predictions over all labeled plants (folds stratified by country x fuel):
         every country also contributes its own plants to training ("in-distribution")
  loco   leave-one-country-out: each country is predicted by a model trained on all other countries
         (transfer difficulty); the US is predicted from the ~1.1k non-US plants only
Models: natl_cf (Ember national prior), gbt_wri, gbt_wri_res (residual on natl_cf), gbt_wri_img_res
(+ frozen SSL4EO embeddings).

Per region (country, and US state for oof) we report n, MAE, R2, bias, skill vs natl_cf, and diagnostics
that may explain the error: fuel mix, within-region CF dispersion (overall and within fuel), consistency of
the national prior with the plant labels, share of labels from 2017-18 only, plant size.

usage: python 18_country_analysis.py <plants.parquet> <embeddings.parquet> <labels.parquet> <eia_dir> <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

spec = importlib.util.spec_from_file_location("bl", Path(__file__).with_name("07_baselines.py"))
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)

THERMAL = {"Coal", "Gas", "Oil", "Nuclear", "Biomass", "Waste", "Petcoke", "Cogeneration", "Geothermal"}
VRE = {"Solar", "Wind"}


def models(tr, te, emb):
    va = tr.sample(frac=0.1, random_state=0)
    tr = tr.drop(va.index)
    return {
        "natl_cf": te.natl_cf.clip(0, 1).values,
        "gbt_wri": bl.gbt_predict(tr, va, te, bl.F_WRI),
        "gbt_wri_res": bl.gbt_predict(tr, va, te, bl.F_WRI, residual=True),
        "gbt_wri_img_res": bl.gbt_predict(tr, va, te, bl.F_WRI + emb, residual=True),
    }


def predictions(L, emb):
    rows = []
    strat = L.country + "_" + L.primary_fuel.astype(str)
    strat = strat.where(strat.map(strat.value_counts()) >= 5, "rare")
    for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(L, strat)):
        tr, te = L.iloc[tri], L.iloc[tei]
        for m, p in models(tr, te, emb).items():
            rows.append(pd.DataFrame({"gppd_idnr": te.gppd_idnr.values, "regime": "oof", "model": m, "pred": p}))
        print("oof fold", k, flush=True)
    for c in L.country.unique():
        tr, te = L[L.country != c], L[L.country == c]
        for m, p in models(tr, te, emb).items():
            rows.append(pd.DataFrame({"gppd_idnr": te.gppd_idnr.values, "regime": "loco", "model": m, "pred": p}))
    print("loco done", flush=True)
    return pd.concat(rows, ignore_index=True)


def region_metrics(d):
    y, p = d.cf_target.values, d.pred.values
    ss = ((y - y.mean()) ** 2).sum()
    return pd.Series({"n": len(d), "MAE": np.abs(y - p).mean(), "bias": (p - y).mean(),
                      "R2": 1 - ((y - p) ** 2).sum() / ss if ss > 0 and len(d) > 2 else np.nan})


def diagnostics(L, key):
    g = L.groupby(key)
    fuel = L.primary_fuel.astype(str)
    within_fuel_sd = L.groupby([key, fuel]).cf_target.std().groupby(level=0).mean()
    # prior consistency: |mean plant CF - national CF| per (region, fuel), averaged with plant counts as weights
    cf_gap = L.assign(gap=L.cf_target - L.natl_cf).groupby([key, fuel]).agg(gap=("gap", "mean"), n=("gap", "size"))
    prior_gap = cf_gap.assign(w=cf_gap.gap.abs() * cf_gap.n).groupby(level=0).apply(lambda x: x.w.sum() / x.n.sum())
    return pd.DataFrame({
        "mean_cf": g.cf_target.mean(),
        "cf_sd": g.cf_target.std(),
        "within_fuel_cf_sd": within_fuel_sd,
        "prior_gap": prior_gap,
        "thermal_share": g.primary_fuel.apply(lambda s: s.astype(str).isin(THERMAL).mean()),
        "vre_share": g.primary_fuel.apply(lambda s: s.astype(str).isin(VRE).mean()),
        "hydro_share": g.primary_fuel.apply(lambda s: (s.astype(str) == "Hydro").mean()),
        "median_mw": g.capacity_mw.median(),
        "age_known": g.age.apply(lambda s: s.notna().mean()),
        "labels_2017_18_only": g.only_1718.mean(),
        "top_fuels": g.primary_fuel.apply(lambda s: ", ".join(f"{k}:{v}" for k, v in s.astype(str).value_counts().head(3).items())),
    })


def main():
    plants, emb_pq, labels_pq, eia_dir, out = sys.argv[1:6]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(plants).merge(pd.read_parquet(emb_pq), on="gppd_idnr", how="left")
    for c in bl.CATS:
        df[c] = df[c].astype("category")
    L = df[df.split_random.notna()].copy()
    emb = [c for c in L.columns if c.startswith("emb_")]

    lab = pd.read_parquet(labels_pq).query("valid & source == 'gppd' & year >= 2017 & year <= 2019")
    L["only_1718"] = L.gppd_idnr.map(lab.groupby("gppd_idnr").year.max() < 2019).fillna(False)
    st = pd.read_excel(Path(eia_dir) / "eia8602019" / "2___Plant_Y2019.xlsx", skiprows=1)[["Plant Code", "State"]]
    st = st.set_index(pd.to_numeric(st["Plant Code"], errors="coerce")).State
    L["state"] = np.where(L.is_us, L.gppd_idnr.str[3:].map(lambda s: st.get(int(s)) if s.isdigit() else None), None)

    pred_path = out / "preds.parquet"
    if pred_path.exists():
        P = pd.read_parquet(pred_path)
    else:
        P = predictions(L, emb)
        P.to_parquet(pred_path, index=False)
    P = P.merge(L[["gppd_idnr", "cf_target", "country", "country_long", "state"]], on="gppd_idnr")

    for key, regimes in (("country", ("oof", "loco")), ("state", ("oof",))):
        D = diagnostics(L.dropna(subset=[key]), key)
        tabs = []
        for r in regimes:
            M = P[P.regime == r].dropna(subset=[key]).groupby(["model", key]).apply(region_metrics).unstack(0)
            M.columns = [f"{r}_{m}_{s}" for s, m in M.columns]
            tabs.append(M)
        T = pd.concat(tabs, axis=1).join(D)
        for r in regimes:
            T[f"{r}_skill"] = 1 - T[f"{r}_gbt_wri_res_MAE"] / T[f"{r}_natl_cf_MAE"]
        if key == "country":
            T = T.join(L.groupby("country").country_long.first())
        T.sort_values(f"{regimes[0]}_natl_cf_n", ascending=False).round(4).to_csv(out / f"{key}_metrics.csv")
        print(f"\n===== by {key} =====")
        cols = [c for c in T.columns if c.endswith(("gbt_wri_res_MAE", "natl_cf_MAE", "gbt_wri_img_res_MAE",
                                                       "gbt_wri_res_R2", "gbt_wri_res_bias", "skill"))]
        show = T[T[f"oof_natl_cf_n"] >= 9].sort_values(f"{regimes[-1]}_gbt_wri_res_MAE")
        print(show[["oof_natl_cf_n", *cols, "within_fuel_cf_sd", "prior_gap", "labels_2017_18_only", "top_fuels"]]
              .round(3).to_string())


if __name__ == "__main__":
    main()
