"""Do technology subtypes fix the transfer failures, and can imagery supply them where databases cannot?

1) Image subtype classifiers (LightGBM on frozen SSL4EO embeddings + water metrics + log capacity [+ age]):
   hydro_class (ror / storage / pumped; labels: JRC Europe + EIA pumped storage), gas_class (ccgt / ocgt /
   ic / steam; labels: EIA US), tower (cooling tower yes/no; EIA + JRC). Predictions are out-of-fold by
   country (US split into 5 random pseudo-countries), so a plant's predicted subtype never uses subtype labels
   from its own country -- i.e. what we would get for a country without a plant database.
2) Residual CF GBTs (3 seeds) with feature sets
     base      WRI-style features (gbt_wri_res)
     +db       + database subtype and cooling type (where a database covers the plant)
     +img      + image-predicted subtype probabilities and water metrics
     +db+img   both
   evaluated on split_random, split_geo (train US -> non-US) and leave-one-country-out (LOCO).

usage: python 22_subtype_experiments.py <plants.parquet> <subtypes.parquet> <img_feats_all.parquet> <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

spec = importlib.util.spec_from_file_location("bl", Path(__file__).with_name("07_baselines.py"))
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)

WATER = ["water_frac", "water_frac_center", "water_largest", "water_n_bodies"]
THERMAL = {"Coal", "Gas", "Oil", "Nuclear", "Biomass", "Waste", "Petcoke", "Cogeneration", "Geothermal"}
FOCUS = ["AUT", "AUS", "BEL", "FRA", "IND", "DEU", "ITA", "ESP"]


def oof_classifier(df, target, mask_train, mask_apply, feats, name):
    """Country-grouped out-of-fold class probabilities for rows in mask_apply."""
    d = df[mask_train | mask_apply].copy()
    rng = np.random.default_rng(0)
    d["grp"] = np.where(d.country == "USA", "USA" + pd.Series(rng.integers(0, 5, len(d)), index=d.index).astype(str), d.country)
    lab = d[mask_train[d.index]]
    classes = sorted(lab[target].unique())
    y = lab[target].map({c: i for i, c in enumerate(classes)})
    binary = len(classes) == 2
    params = dict(objective="binary" if binary else "multiclass", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=10, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                  **({} if binary else {"num_class": len(classes)}))
    probs = pd.DataFrame(np.nan, index=d.index, columns=[f"p_{name}_{c}" for c in classes])
    groups = d.grp.unique()
    n_splits = min(10, len(groups))
    gkf = GroupKFold(n_splits)
    for tr_g, te_g in gkf.split(groups, groups=groups):
        test_groups = set(groups[te_g])
        tr = lab[~lab.grp.isin(test_groups)]
        te = d[d.grp.isin(test_groups)]
        m = lgb.train(params, lgb.Dataset(tr[feats], y[tr.index]), 300)
        p = m.predict(te[feats])
        probs.loc[te.index] = np.c_[1 - p, p] if binary else p
    pred = probs.loc[lab.index].values.argmax(1)
    report = {"classifier": name, "n_labeled": len(lab), "classes": dict(lab[target].value_counts()),
              "balanced_acc": balanced_accuracy_score(y, pred)}
    if binary:
        report["auroc"] = roc_auc_score(y, probs.loc[lab.index].values[:, 1])
    return probs, report


def evaluate(te, p):
    y = te.cf_target.values
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(te), "MAE": np.abs(y - p).mean(), "R2": 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan}


def main():
    plants, sub_pq, img_pq, out = sys.argv[1:5]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(plants).merge(pd.read_parquet(sub_pq), on="gppd_idnr", how="left") \
        .merge(pd.read_parquet(img_pq), on="gppd_idnr", how="left").set_index("gppd_idnr", drop=False)
    emb = [c for c in df.columns if c.startswith("emb_")]
    has_img = df.emb_0.notna()

    # ---- 1) image subtype classifiers
    base_f = emb + WATER + ["log_capacity"]
    hyd = df.primary_fuel == "Hydro"
    gas = df.primary_fuel.isin(["Gas", "Cogeneration"])
    thermal = df.primary_fuel.isin(THERMAL)
    reports, prob_frames = [], []
    for name, target, scope, feats in (
        ("hydro", "hydro_class", hyd, base_f),
        ("gas", "gas_class", gas, base_f + ["age"]),
        ("tower", "tower", thermal, base_f + ["age"]),
    ):
        train_mask = scope & has_img & df[target].notna()
        apply_mask = scope & has_img
        probs, rep = oof_classifier(df, target, train_mask, apply_mask, feats, name)
        reports.append(rep)
        prob_frames.append(probs)
        print(rep, flush=True)
    P = pd.concat(prob_frames, axis=1)
    if "p_tower_0.0" in P:
        P = P.drop(columns=["p_tower_0.0"])
    df = df.join(P)
    pd.DataFrame(reports).to_csv(out / "subtype_classifiers.csv", index=False)
    img_sub = list(P.columns) + WATER

    # ---- 2) residual CF models
    for c in ["primary_fuel", "other_fuel1", "subtype", "cooling"]:
        df[c] = df[c].astype("category")
    sets = {"base": bl.F_WRI, "+db": bl.F_WRI + ["subtype", "cooling"], "+img": bl.F_WRI + img_sub,
            "+db+img": bl.F_WRI + ["subtype", "cooling"] + img_sub}
    bl.CATS = ["primary_fuel", "other_fuel1", "subtype", "cooling"]
    L = df[df.split_random.notna()].copy()

    rows, loco_rows = [], []
    for split in ("random", "geo"):
        col = f"split_{split}"
        tr, va, te = (L[L[col] == s] for s in ("train", "val", "test"))
        for name, feats in sets.items():
            p = bl.gbt_predict(tr, va, te, feats, residual=True)
            rows.append({"split": split, "set": name, **evaluate(te, p)})
            for fuel in ("Hydro", "Gas", "Coal", "Waste"):
                m = (te.primary_fuel == fuel).values
                rows.append({"split": f"{split}:{fuel}", "set": name, **evaluate(te[m], p[m])})
        print(split, "done", flush=True)
    for c in L.country.unique():
        trv, te = L[L.country != c], L[L.country == c]
        va = trv.sample(frac=0.1, random_state=0)
        tr = trv.drop(va.index)
        for name, feats in sets.items():
            p = bl.gbt_predict(tr, va, te, feats, residual=True)
            loco_rows.append(pd.DataFrame({"gppd_idnr": te.gppd_idnr.values, "set": name, "pred": p}))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "subtype_cf_results.csv", index=False)
    LP = pd.concat(loco_rows).merge(L[["gppd_idnr", "cf_target", "country", "primary_fuel"]].reset_index(drop=True), on="gppd_idnr")
    LP.to_parquet(out / "loco_preds.parquet", index=False)
    LP["ae"] = (LP.pred - LP.cf_target).abs()
    by_c = LP.pivot_table(index="country", columns="set", values="ae", aggfunc="mean")
    by_c["n"] = LP[LP.set == "base"].groupby("country").size()
    by_c = by_c[by_c.n >= 9].sort_values("n", ascending=False)[["n", "base", "+db", "+img", "+db+img"]]
    by_c.round(3).to_csv(out / "loco_by_country.csv")
    nonus = LP[LP.country != "USA"]
    print("\n", res.pivot_table(index="split", columns="set", values="MAE")[["base", "+db", "+img", "+db+img"]].round(3).to_string())
    print("\nLOCO MAE by country:\n", by_c.round(3).to_string())
    print("\nLOCO pooled non-US MAE:", nonus.groupby("set").ae.mean().round(3).to_dict())
    fuel_c = LP[LP.country.isin(FOCUS)].pivot_table(index=["country", "primary_fuel"], columns="set", values="ae", aggfunc="mean")
    fuel_c["n"] = LP[(LP.set == "base") & LP.country.isin(FOCUS)].groupby(["country", "primary_fuel"]).size()
    print("\nLOCO focus countries by fuel:\n", fuel_c[fuel_c.n >= 5][["n", "base", "+db", "+img", "+db+img"]].round(3).to_string())


if __name__ == "__main__":
    main()
