"""GBT models on interpretable time-series activity features (thermal plants only).

Models (all residual on the national CF prior, i.e. trees learn cf - natl_cf; 3 seeds averaged)
  tab          WRI-style tabular features only (same thermal training plants -> fair reference)
  tab+ts       + plant-level aggregates of the frame features (mean / max / std over all frames,
               mean over clear frames, 'on-rate' = share of clear frames with a plume signal)
  frame        frame-level model with *monthly* supervision: each frame is labelled with the plant's
               CF in that month (EIA-923, US) or its annual CF when no monthly label exists; the plant
               estimate is the mean of its frame predictions.
Evaluated on the thermal plants of split_random test and split_geo test (non-US, also per region).

usage: python 15_ts_gbt.py <plants.parquet> <ts_frame_features.parquet> <labels_monthly.parquet> <out_dir>
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEEDS = (0, 1, 2)
TAB = ["log_capacity", "age", "primary_fuel", "n_other_fuels", "natl_cf"]
EUROPE = {"ITA", "DEU", "ESP", "FRA", "POL", "GRC", "SWE", "AUT", "BEL", "ROU", "CZE", "DNK", "PRT", "NLD",
          "FIN", "HUN", "IRL", "BGR", "SVK", "SVN", "GBR", "EST", "LTU", "LVA", "HRV", "LUX", "NOR", "CHE"}
CLEAR = 10  # scene cloud cover (%) below which local brightness is attributed to the plant, not weather


def metrics(y, p, cap):
    e = np.abs(y - p) * cap * 8.76
    return {"n": len(y), "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "MAE_GWh": e.mean(), "wMAPE_gen": e.sum() / (y * cap * 8.76).sum()}


def aggregate(fr):
    feats = [c for c in fr.columns if c.startswith(("c_", "f_")) or c == "cloud_excess"]
    g = fr.groupby("gppd_idnr")[feats]
    agg = pd.concat([g.mean().add_suffix("_mean"), g.max().add_suffix("_max"), g.std().add_suffix("_std")], axis=1)
    clear = fr[fr.scene_cloud < CLEAR]
    gc = clear.groupby("gppd_idnr")
    agg = agg.join(gc[feats].mean().add_suffix("_clearmean"))
    on = (clear.cloud_excess > 0.05) | (clear.c_white_frac > 0.02)
    agg["on_rate"] = on.groupby(clear.gppd_idnr).mean()
    agg["n_clear"] = gc.size()
    agg["n_frames"] = fr.groupby("gppd_idnr").size()
    return agg.reset_index()


def fit_predict(tr, va, te, feats, ytr, yva, seed):
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1)
    cats = [c for c in feats if str(tr[c].dtype) == "category"]
    dtr = lgb.Dataset(tr[feats], ytr, init_score=tr.natl_cf.values, categorical_feature=cats)
    dva = lgb.Dataset(va[feats], yva, init_score=va.natl_cf.values, reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(te[feats]) + te.natl_cf.values, m


def main():
    plants, frames, monthly, out = (pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2]),
                                    pd.read_parquet(sys.argv[3]), Path(sys.argv[4]))
    out.mkdir(parents=True, exist_ok=True)
    plants["primary_fuel"] = plants.primary_fuel.astype("category")
    P = plants[plants.gppd_idnr.isin(frames.gppd_idnr.unique())].merge(aggregate(frames), on="gppd_idnr")
    ts_feats = [c for c in P.columns if c.endswith(("_mean", "_max", "_std", "_clearmean")) or c in ("on_rate", "n_clear", "n_frames")]
    ts_feats = [c for c in ts_feats if c not in TAB]

    # frame table with monthly labels (fallback: annual CF)
    m = monthly[monthly.valid][["gppd_idnr", "year", "month", "cf"]].rename(columns={"cf": "cf_month"})
    F = frames.merge(m, on=["gppd_idnr", "year", "month"], how="left")
    F = F.merge(plants[["gppd_idnr", *TAB, "cf_target"]], on="gppd_idnr")
    F["y_frame"] = F.cf_month.fillna(F.cf_target)
    frame_feats = TAB + [c for c in frames.columns if c.startswith(("c_", "f_"))] + ["cloud_excess", "scene_cloud", "month"]
    print(f"thermal plants: {len(P)}, frames: {len(F)}, frames with monthly label: {F.cf_month.notna().mean():.2f}")

    rows, preds = [], {}
    for split in ("random", "geo"):
        col = f"split_{split}"
        tr, va, te = (P[P[col] == s] for s in ("train", "val", "test"))
        res = {"natl_cf": te.natl_cf.clip(0, 1).values}
        for name, feats in (("tab", TAB), ("tab+ts", TAB + ts_feats)):
            res[name] = np.clip(np.mean([fit_predict(tr, va, te, feats, tr.cf_target, va.cf_target, s)[0]
                                         for s in SEEDS], 0), 0, 1)
        Ftr, Fva, Fte = (F[F.gppd_idnr.isin(d.gppd_idnr)] for d in (tr, va, te))
        fp = np.mean([fit_predict(Ftr, Fva, Fte, frame_feats, Ftr.y_frame, Fva.y_frame, s)[0] for s in SEEDS], 0)
        plant_pred = pd.Series(np.clip(fp, 0, 1), index=Fte.gppd_idnr.values).groupby(level=0).mean()
        res["frame(monthly)"] = plant_pred.reindex(te.gppd_idnr).fillna(te.set_index("gppd_idnr").natl_cf).values
        preds[split] = pd.DataFrame(res).assign(gppd_idnr=te.gppd_idnr.values)
        subs = [("all", np.ones(len(te), bool))]
        if split == "geo":
            subs += [("India", (te.country == "IND").values), ("Australia", (te.country == "AUS").values),
                     ("Europe", te.country.isin(EUROPE).values)]
        for sub, msk in subs:
            for k, p in res.items():
                rows.append({"split": split, "subset": sub, "model": k,
                             **metrics(te.cf_target.values[msk], p[msk], te.capacity_mw.values[msk])})
        # per-fuel MAE
        pf = pd.DataFrame({k: np.abs(te.cf_target.values - p) for k, p in res.items()}).groupby(te.primary_fuel.astype(str).values).mean()
        pf.assign(n=te.primary_fuel.astype(str).value_counts()).round(3).to_csv(out / f"per_fuel_{split}.csv")
        print(f"\nper-fuel MAE ({split}):\n", pf.assign(n=te.primary_fuel.astype(str).value_counts()).round(3).to_string())
    r = pd.DataFrame(rows)
    r.round(4).to_csv(out / "ts_gbt.csv", index=False)
    for s, p in preds.items():
        p.to_csv(out / f"preds_{s}.csv", index=False)
    print("\n", r.round(3).to_string(index=False))

    # feature importance of the plant-level model (random split, seed 0) for interpretation
    tr, va = P[P.split_random == "train"], P[P.split_random == "val"]
    _, mdl = fit_predict(tr, va, va, TAB + ts_feats, tr.cf_target, va.cf_target, 0)
    imp = pd.Series(mdl.feature_importance("gain"), index=TAB + ts_feats).sort_values(ascending=False)
    (imp / imp.sum()).round(4).to_csv(out / "importance_tab_ts.csv")
    print("\ntop features (gain share):\n", (imp / imp.sum()).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
