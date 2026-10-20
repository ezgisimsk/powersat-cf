"""Pre-registered comparison of activity-model backbones: SSL4EO ResNet-50 vs SSL4EO ViT-S/16 (seed 0 each).

Primary: (1) within-plant AUROC on held-out US plants, paired ViT - ResNet difference with bootstrap CI;
(2) leave-one-US-state-out gain of the on-rate over subtypes (+sub -> +sub+on-rate), per backbone, state bootstrap.
Secondary: frame AUROC, plant on-rate Spearman, Europe transfer MAE and gain over the national prior.
ResNet runs: results/onoff_ablation/full (split) and results/onoff/fold* ; ViT: results/onoff_vit/split and fold*.

usage: python 34_backbone_compare.py <project_root> <out_dir>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("rev", HERE / "31_revision_cg.py")
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)
bl = rev.bl
SUB, ON = rev.SUB, rev.ON
RUNS = {"ResNet-50": ("onoff_ablation/full", "onoff"), "ViT-S/16": ("onoff_vit/split", "onoff_vit")}


def within_plant(fp):
    d = pd.read_parquet(fp)
    d = d[(d.country == "USA") & d.on.notna()]
    return pd.Series({pid: roc_auc_score(g.on, g.p_on) for pid, g in d.groupby("gppd_idnr")
                      if (g.on == 1).sum() >= 3 and (g.on == 0).sum() >= 3})


def state_boot(d, w, n=5000):
    rng = np.random.default_rng(0)
    s = [(d[i] * w[i]).sum() / w[i].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(n))]
    return (d * w).sum() / w.sum(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # (1) within-plant AUROC and secondary activity metrics
    wp = {}
    for name, (split_dir, _) in RUNS.items():
        m = json.load(open(root / "results" / split_dir / "metrics.json"))
        wp[name] = within_plant(root / "results" / split_dir / "frame_preds.parquet")
        report[name] = {"frame_auroc": m["us_test_auroc_all"], "within_plant_median": float(wp[name].median()),
                        "onrate_spearman": m["us_plant_spearman_onrate"]}
    diff = (wp["ViT-S/16"] - wp["ResNet-50"]).dropna().values
    rng = np.random.default_rng(0)
    bs = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(5000)]
    report["within_plant_diff_vit_minus_resnet"] = {"median": float(np.median(diff)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                    "plants": int(len(diff)), "vit_better": int((diff > 0).sum())}

    # cohort as in 31 (coal + gas, subtypes), on-rates from each backbone's 5 folds (year-matched)
    plants = pd.read_parquet(root / "data/processed/plants.parquet")
    plants = plants.merge(pd.read_parquet(root / "data/processed/subtypes.parquet")[["gppd_idnr", *SUB]], on="gppd_idnr", how="left")
    L = plants[plants.split_random.notna() & plants.primary_fuel.isin(rev.FUELS)].copy()
    lab = pd.read_parquet(root / "data/processed/labels.parquet").query("valid & source == 'gppd' & year >= 2017 & year <= 2019")
    label_years = lab.groupby("gppd_idnr").year.apply(set)
    st = pd.read_excel(root / "data/raw/eia/eia8602019/2___Plant_Y2019.xlsx", skiprows=1)[["Plant Code", "State"]]
    st = st.set_index(pd.to_numeric(st["Plant Code"], errors="coerce")).State
    L["state"] = [st.get(int(g[3:])) if g.startswith("USA") and g[3:].isdigit() else None for g in L.gppd_idnr]
    for c in ["primary_fuel", "other_fuel1", *SUB]:
        L[c] = L[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", *SUB]

    # (2) leave-one-state-out: +sub (shared) vs +sub+on-rate per backbone
    us = L[L.country == "USA"].dropna(subset=["state"])
    rates = {name: rev.onrates(root, fold_dir, label_years)[1] for name, (_, fold_dir) in RUNS.items()}
    rows = []
    for s in sorted(us.state.unique()):
        te = us[us.state == s]
        if len(te) < 10:
            continue
        trv = us[us.state != s]
        va = trv.sample(frac=0.1, random_state=0)
        tr = trv.drop(va.index)
        row = {"state": s, "n": len(te), "+sub": np.abs(te.cf_target - bl.gbt_predict(tr, va, te, bl.F_WRI + SUB, residual=True)).mean()}
        for name, r in rates.items():
            f = lambda d: d.join(r, on="gppd_idnr")
            row[name] = np.abs(te.cf_target - bl.gbt_predict(f(tr), f(va), f(te), bl.F_WRI + SUB + ON, residual=True)).mean()
        rows.append(row)
    loso = pd.DataFrame(rows)
    loso.to_csv(out / "loso_backbones.csv", index=False)
    w = loso.n.values
    for name in rates:
        mean, lo, hi = state_boot((loso["+sub"] - loso[name]).values, w)
        report[name]["loso_mae"] = float((loso[name] * w).sum() / w.sum())
        report[name]["loso_gain_over_sub"] = {"mean": float(mean), "ci": [float(lo), float(hi)]}
    mean, lo, hi = state_boot((loso["ResNet-50"] - loso["ViT-S/16"]).values, w)
    report["loso_diff_vit_vs_resnet"] = {"mean": float(mean), "ci": [float(lo), float(hi)]}
    report["loso_sub_mae"] = float((loso["+sub"] * w).sum() / w.sum())

    # secondary: Europe transfer
    tr, va, te = (L[L.split_geo == s] for s in ("train", "val", "test"))
    E = te.country.isin(rev.EUROPE).values
    for name, r in rates.items():
        f = lambda d: d.join(r, on="gppd_idnr")
        p = bl.gbt_predict(f(tr), f(va), f(te), bl.F_WRI + SUB + ON, residual=True)
        mean, lo, hi = rev.boot_diff(te[E], te.natl_cf.clip(0, 1).values[E], p[E], by_country=False)
        report[name]["europe_mae"] = float(np.abs(te.cf_target.values[E] - p[E]).mean())
        report[name]["europe_gain_over_prior"] = {"mean": float(mean), "ci": [float(lo), float(hi)]}

    json.dump(report, open(out / "backbone_compare.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
