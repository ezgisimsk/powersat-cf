"""Does the satellite on-rate help plant CF estimation for thermal plants?

Plant features from the 5-fold on/off model (24_train_onoff.py --fold k): for US plants the prediction of the
fold that held them out; for non-US plants the mean over the 5 folds. on_rate = mean P(on) over clear frames
(scene cloud < 10 %), util = mean predicted overpass utilisation, plus n clear frames.

Residual GBTs (3 seeds) on thermal plants: tab (WRI features), tab+sub (+ database subtype/cooling),
tab+onrate, tab+sub+onrate. Evaluated on split_random and split_geo test; idle-plant AUROC (CF < 0.1) for
Indian coal plants.

usage: python 25_onrate_gbt.py <plants.parquet> <subtypes.parquet> <onoff_dir> <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("bl", Path(__file__).with_name("07_baselines.py"))
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)
THERMAL = {"Coal", "Gas", "Oil", "Nuclear", "Biomass", "Waste", "Petcoke", "Cogeneration", "Geothermal"}
EUROPE = bl.EUROPE


def load_onrate(onoff_dir):
    parts = []
    for k in range(5):
        fp = pd.read_parquet(Path(onoff_dir) / f"fold{k}" / "frame_preds.parquet")
        fp["fold"] = k
        parts.append(fp)
    fp = pd.concat(parts)
    clear = fp[fp.scene_cloud < 10]
    g = clear.groupby(["gppd_idnr", "fold"]).agg(on_rate=("p_on", "mean"), util=("util_pred", "mean"), n_clear=("p_on", "size"))
    return g.groupby(level=0).mean().reset_index()  # US: one fold each; non-US: mean over folds


def evaluate(te, p):
    y = te.cf_target.values
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(te), "MAE": np.abs(y - p).mean(), "R2": 1 - ((y - p) ** 2).sum() / ss}


def main():
    plants, sub_pq, onoff_dir, out = sys.argv[1:5]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(plants).merge(pd.read_parquet(sub_pq)[["gppd_idnr", "subtype", "cooling"]], on="gppd_idnr", how="left")
    df = df.merge(load_onrate(onoff_dir), on="gppd_idnr", how="left")
    df = df[df.split_random.notna() & df.primary_fuel.isin(THERMAL) & df.on_rate.notna()].copy()
    for c in ["primary_fuel", "other_fuel1", "subtype", "cooling"]:
        df[c] = df[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", "subtype", "cooling"]
    on = ["on_rate", "util", "n_clear"]
    sets = {"tab": bl.F_WRI, "tab+sub": bl.F_WRI + ["subtype", "cooling"], "tab+onrate": bl.F_WRI + on,
            "tab+sub+onrate": bl.F_WRI + ["subtype", "cooling"] + on}
    rows, preds = [], {}
    for split in ("random", "geo"):
        col = f"split_{split}"
        tr, va, te = (df[df[col] == s] for s in ("train", "val", "test"))
        res = {"natl_cf": te.natl_cf.clip(0, 1).values}
        res.update({k: bl.gbt_predict(tr, va, te, f, residual=True) for k, f in sets.items()})
        preds[split] = pd.DataFrame(res).assign(gppd_idnr=te.gppd_idnr.values)
        subs = [("all", np.ones(len(te), bool))] + [(f, (te.primary_fuel == f).values) for f in ("Coal", "Gas")]
        if split == "geo":
            subs += [("India", (te.country == "IND").values), ("Australia", (te.country == "AUS").values),
                     ("Europe", te.country.isin(EUROPE).values)]
        for sub, m in subs:
            for k, p in res.items():
                rows.append({"split": split, "subset": sub, "model": k, **evaluate(te[m], p[m])})
        if split == "geo":
            ind = te.assign(**{k: v for k, v in res.items()})
            ind = ind[(ind.country == "IND") & (ind.primary_fuel == "Coal")]
            idle = (ind.cf_target < 0.1).astype(int)
            print(f"India coal: n={len(ind)}, idle={idle.sum()}")
            for k in ["on_rate", *res]:
                print(f"  idle AUROC from {k}: {roc_auc_score(idle, -ind[k]):.3f}")
    r = pd.DataFrame(rows)
    r.round(4).to_csv(out / "onrate_gbt.csv", index=False)
    for s, p in preds.items():
        p.to_csv(out / f"preds_{s}.csv", index=False)
    print(r.pivot_table(index=["split", "subset"], columns="model", values="MAE")[["natl_cf", *sets]].round(3).to_string())


if __name__ == "__main__":
    main()
