"""Tabular baselines for plant-level capacity-factor estimation.

Baselines
  fuel_mean          training-set mean CF per primary fuel
  country_fuel_mean  training-set mean CF per (country, fuel), fallback fuel_mean
  natl_cf            Ember national CF for the plant's fuel (no training; Climate TRACE-style fallback)
  wri_published      WRI's published 2017 estimate (only where available; WRI trained on many of these plants)
  gbt_wri            LightGBM on WRI-style features: log capacity, age, fuel, #other fuels, national CF
  gbt_plus           gbt_wri + latitude/longitude + secondary fuel

Evaluation: split_random test, split_geo test (non-US, also per region), fold_spatial 5-fold CV.

usage: python 07_baselines.py <plants.parquet> <out_dir> [embeddings.parquet]
  *_res: residual variants (trees learn cf - natl_cf); gbt_wri_img*: + frozen image embeddings (stacking)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEEDS = (0, 1, 2)
F_WRI = ["log_capacity", "age", "primary_fuel", "n_other_fuels", "natl_cf"]
F_PLUS = F_WRI + ["latitude", "longitude", "other_fuel1"]
CATS = ["primary_fuel", "other_fuel1"]
EUROPE = {"ITA", "DEU", "ESP", "FRA", "POL", "GRC", "SWE", "AUT", "BEL", "ROU", "CZE", "DNK", "PRT", "NLD",
          "FIN", "HUN", "IRL", "BGR", "SVK", "SVN", "GBR", "EST", "LTU", "LVA", "HRV", "LUX", "NOR", "CHE"}


def metrics(y, p, cap):
    gen_err = np.abs(y - p) * cap * 8.76  # GWh
    return {"n": len(y), "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "MAE_GWh": gen_err.mean(), "wMAPE_gen": gen_err.sum() / (y * cap * 8.76).sum()}


def fit_gbt(tr, va, feats, seed, residual=False):
    """residual=True: the national CF is the starting point (init_score) and trees learn cf - natl_cf."""
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1)
    init = (lambda d: d.natl_cf.values) if residual else (lambda d: None)
    dtr = lgb.Dataset(tr[feats], tr.cf_target, init_score=init(tr), categorical_feature=[c for c in feats if c in CATS])
    dva = lgb.Dataset(va[feats], va.cf_target, init_score=init(va), reference=dtr)
    return lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def gbt_predict(tr, va, te, feats, residual=False):
    preds = [fit_gbt(tr, va, feats, s, residual).predict(te[feats]) for s in SEEDS]
    p = np.mean(preds, axis=0) + (te.natl_cf.values if residual else 0)
    return np.clip(p, 0, 1)


def predict_all(tr, va, te):
    """Return {baseline: predictions for te}."""
    out = {}
    fm = tr.groupby("primary_fuel").cf_target.mean()
    out["fuel_mean"] = te.primary_fuel.map(fm).fillna(tr.cf_target.mean()).values
    cfm = tr.groupby(["country", "primary_fuel"]).cf_target.mean()
    cf = pd.Series([cfm.get(k, np.nan) for k in zip(te.country, te.primary_fuel)], index=te.index)
    out["country_fuel_mean"] = cf.fillna(pd.Series(out["fuel_mean"], index=te.index)).values
    out["natl_cf"] = te.natl_cf.values
    emb = [c for c in te.columns if c.startswith("emb_")]
    sets = [("gbt_wri", F_WRI), ("gbt_plus", F_PLUS)] + ([("gbt_wri_img", F_WRI + emb)] if emb else [])
    for name, feats in sets:
        out[name] = gbt_predict(tr, va, te, feats)
        out[name + "_res"] = gbt_predict(tr, va, te, feats, residual=True)
    return out


def save_preds(te, preds, path):
    d = te[["gppd_idnr"]].copy()
    for b, p in preds.items():
        d[b] = p
    d["wri_published*"] = te.wri_est_cf_2017.clip(0, 1).values
    d.to_csv(path, index=False)


def evaluate(te, preds, tag):
    rows = []
    for b, p in preds.items():
        rows.append({"eval": tag, "baseline": b, **metrics(te.cf_target.values, p, te.capacity_mw.values)})
    m = te.wri_est_cf_2017.notna()
    if m.any():
        rows.append({"eval": tag, "baseline": "wri_published*",
                     **metrics(te.cf_target[m].values, te.wri_est_cf_2017[m].clip(0, 1).values, te.capacity_mw[m].values)})
    return rows


def main():
    df = pd.read_parquet(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:  # frozen image embeddings (emb_*), merged by plant; missing patches stay NaN
        df = df.merge(pd.read_parquet(sys.argv[3]), on="gppd_idnr", how="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in CATS:
        df[c] = df[c].astype("category")
    L = df[df.split_random.notna()].copy()
    rows, per_fuel = [], []

    # random split
    tr, va, te = (L[L.split_random == s] for s in ("train", "val", "test"))
    p = predict_all(tr, va, te)
    save_preds(te, p, out_dir / "preds_random.csv")
    rows += evaluate(te, p, "random")
    for b in ("natl_cf", "gbt_wri", "gbt_plus"):
        e = (te.cf_target - p[b]).abs()
        per_fuel.append(e.groupby(te.primary_fuel.astype(str)).mean().rename(b))
    pd.concat(per_fuel, axis=1).assign(n=te.primary_fuel.astype(str).value_counts()).round(3) \
        .sort_values("n", ascending=False).to_csv(out_dir / "baselines_per_fuel_random.csv")

    # geographic OOD: train US, test non-US (overall + per region)
    tr, va, te = (L[L.split_geo == s] for s in ("train", "val", "test"))
    p = predict_all(tr, va, te)
    save_preds(te, p, out_dir / "preds_geo.csv")
    rows += evaluate(te, p, "geo_nonUS")
    for region, m in (("geo_India", te.country == "IND"), ("geo_Australia", te.country == "AUS"),
                      ("geo_Europe", te.country.isin(EUROPE))):
        rows += evaluate(te[m.values], {b: v[m.values] for b, v in p.items()}, region)

    # spatial block CV (inner 10% of training folds for early stopping)
    cv = []
    for k in range(5):
        trv, te = L[L.fold_spatial != k], L[L.fold_spatial == k]
        va = trv.sample(frac=0.1, random_state=k)
        tr = trv.drop(va.index)
        cv += evaluate(te, predict_all(tr, va, te), f"spatial_fold{k}")
    cv = pd.DataFrame(cv)
    agg = cv.groupby("baseline")[["MAE", "RMSE", "R2", "MAE_GWh", "wMAPE_gen"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.round(4).to_csv(out_dir / "baselines_spatial_cv.csv")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(out_dir / "baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print("\nspatial 5-fold CV (mean/std):\n", agg.round(3).to_string())
    print("\nper-fuel MAE (random split):\n", pd.read_csv(out_dir / "baselines_per_fuel_random.csv").to_string(index=False))


if __name__ == "__main__":
    main()
