"""Experiment D (pre-registered): does the US-trained activity model detect running plants in Europe?

Ground truth: data/processed/entsoe_overpass.parquet (ENTSO-E unit generation at the overpass hour).
Primary: the US-trained ResNet-50 model (results/onoff_ablation/full, seed 0), unchanged; within-plant AUROC
median over European plants with >= 3 on and >= 3 off frames; success if median > 0.5 and the bootstrap 95 % CI
lower bound > 0.5.
Secondary: frame AUROC; plant on-rate Spearman (clear frames, >= 5 per plant); paired image-over-context gain
(context-only model); coal vs gas; ViT-S/16; mean of the 15 fold models (3 seeds x 5 folds).

usage: python 36_europe_activity_eval.py <project_root> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLEAR = 10
MODELS = {"ResNet-50 (primary)": ["onoff_ablation/full"], "context only": ["onoff_ablation/context"],
          "image only": ["onoff_ablation/image"], "ViT-S/16": ["onoff_vit/split"],
          "ResNet-50, 15-model mean": [f"{d}/fold{k}" for d in ("onoff", "onoff_s1", "onoff_s2") for k in range(5)]}


def preds(root, dirs):
    parts = [pd.read_parquet(root / "results" / d / "frame_preds.parquet",
                             columns=["gppd_idnr", "date", "scene_cloud", "primary_fuel", "country", "p_on", "util_pred"])
             for d in dirs]
    p = pd.concat(parts)
    return p.groupby(["gppd_idnr", "date"], as_index=False).agg(
        p_on=("p_on", "mean"), util_pred=("util_pred", "mean"), scene_cloud=("scene_cloud", "first"),
        primary_fuel=("primary_fuel", "first"), country=("country", "first"))


def within(d):
    return pd.Series({pid: roc_auc_score(g.on, g.p_on) for pid, g in d.groupby("gppd_idnr")
                      if (g.on == 1).sum() >= 3 and (g.on == 0).sum() >= 3})


def boot_median(v, n=5000):
    rng = np.random.default_rng(0)
    s = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(root / "data/processed/entsoe_overpass.parquet")
    report, wp = {}, {}
    for name, dirs in MODELS.items():
        d = preds(root, dirs).merge(lab[["gppd_idnr", "date", "on", "inst_util"]], on=["gppd_idnr", "date"])
        if d.empty:
            continue
        w = within(d)
        wp[name] = w
        r = {"frames": int(len(d)), "plants": int(d.gppd_idnr.nunique()), "share_on": float(d.on.mean()),
             "frame_auroc": float(roc_auc_score(d.on, d.p_on)),
             "within_plant_median": float(w.median()), "within_plant_ci": boot_median(w.values),
             "within_plant_n": int(len(w)), "within_plant_excluded": int(d.gppd_idnr.nunique() - len(w))}
        c = d[d.scene_cloud < CLEAR].groupby("gppd_idnr").agg(pred=("p_on", "mean"), true=("on", "mean"), n=("on", "size"))
        c = c[c.n >= 5]
        r["onrate_spearman"] = float(c.pred.corr(c.true, method="spearman"))
        r["onrate_plants"] = int(len(c))
        for fuel in ("Coal", "Gas"):
            wf = w[w.index.isin(d[d.primary_fuel == fuel].gppd_idnr)]
            if len(wf) >= 5:
                r[f"within_plant_median_{fuel}"] = float(wf.median())
                r[f"within_plant_n_{fuel}"] = int(len(wf))
        report[name] = r
    prim = report["ResNet-50 (primary)"]
    report["success_criterion_met"] = bool(prim["within_plant_median"] > 0.5 and prim["within_plant_ci"][0] > 0.5)
    for a, b in (("ResNet-50 (primary)", "context only"), ("image only", "context only"), ("ViT-S/16", "ResNet-50 (primary)")):
        if a in wp and b in wp:
            diff = (wp[a] - wp[b]).dropna().values
            report[f"paired: {a} - {b}"] = {"median": float(np.median(diff)), "ci": boot_median(diff),
                                            "plants": int(len(diff)), "first_better": int((diff > 0).sum())}
    # per-country within-plant medians for the primary model (countries with >= 5 plants)
    d = preds(root, MODELS["ResNet-50 (primary)"]).merge(lab[["gppd_idnr", "date", "on"]], on=["gppd_idnr", "date"])
    w = wp["ResNet-50 (primary)"]
    country = d.drop_duplicates("gppd_idnr").set_index("gppd_idnr").country
    per_c = w.groupby(country.reindex(w.index).values).agg(["size", "median"])
    report["by_country_primary"] = {k: {"plants": int(v["size"]), "median": float(v["median"])} for k, v in per_c.iterrows() if v["size"] >= 5}
    json.dump(report, open(out / "europe_activity.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
