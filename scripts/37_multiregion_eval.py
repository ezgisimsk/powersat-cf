"""Experiment E (pre-registered): activity model trained on US + European labels, evaluated leave-one-country-group-out.

Models (results/onoff_multi): full_g0..4 (US + Europe outside group k), context_g0..4 (same, month + cloud only),
control (same US data, no European frames). Comparator from the pre-registration: US-only model
results/onoff_ablation/full (US split_random train).
Primary: paired within-plant AUROC difference (multi-region - US-only) on European plants with >= 3 on and
>= 3 off frames; median and plant bootstrap 95 % CI; success if the CI lower bound > 0.
Secondary: multi-region - control (contribution of European labels), image over context, coal vs gas, on-rate
Spearman, and downstream Europe CF (US-trained CF model with Europe on-rates from the held-out-country models).

usage: python 37_multiregion_eval.py <project_root> <out_dir>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
CLEAR = 10


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frames(root, dirs):
    p = pd.concat([pd.read_parquet(root / "results" / d / "frame_preds.parquet",
                                   columns=["gppd_idnr", "date", "scene_cloud", "primary_fuel", "country", "p_on", "util_pred"])
                   for d in dirs])
    # a plant from an unlabelled country is predicted by every group model: average them
    return p.groupby(["gppd_idnr", "date"], as_index=False).agg(
        p_on=("p_on", "mean"), util_pred=("util_pred", "mean"), scene_cloud=("scene_cloud", "first"),
        primary_fuel=("primary_fuel", "first"), country=("country", "first"))


def within(d):
    return pd.Series({pid: roc_auc_score(g.on, g.p_on) for pid, g in d.groupby("gppd_idnr")
                      if (g.on == 1).sum() >= 3 and (g.on == 0).sum() >= 3})


def boot_median(v, n=5000):
    rng = np.random.default_rng(0)
    s = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def onrate(fp):
    c = fp[fp.scene_cloud < CLEAR]
    return c.groupby("gppd_idnr").agg(on_rate=("p_on", "mean"), util=("util_pred", "mean"), n_clear=("p_on", "size"))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(root / "data/processed/entsoe_overpass.parquet")[["gppd_idnr", "date", "on"]]
    models = {"US-only (pre-registered comparator)": ["onoff_ablation/full"],
              "control (all US, no Europe)": ["onoff_multi/control"],
              "multi-region": [f"onoff_multi/full_g{k}" for k in range(5)],
              "multi-region context only": [f"onoff_multi/context_g{k}" for k in range(5)]}
    fp, wp, rep = {}, {}, {}
    for name, dirs in models.items():
        fp[name] = frames(root, dirs)
        d = fp[name].merge(lab, on=["gppd_idnr", "date"])
        w = within(d)
        wp[name] = w
        r = {"frames": int(len(d)), "plants": int(d.gppd_idnr.nunique()), "frame_auroc": float(roc_auc_score(d.on, d.p_on)),
             "within_plant_median": float(w.median()), "within_plant_ci": boot_median(w.values), "within_plant_n": int(len(w))}
        c = d[d.scene_cloud < CLEAR].groupby("gppd_idnr").agg(pred=("p_on", "mean"), true=("on", "mean"), n=("on", "size"))
        c = c[c.n >= 5]
        r["onrate_spearman"], r["onrate_plants"] = float(c.pred.corr(c.true, method="spearman")), int(len(c))
        fuel = d.drop_duplicates("gppd_idnr").set_index("gppd_idnr").primary_fuel
        for f in ("Coal", "Gas"):
            wf = w[fuel.reindex(w.index).values == f]
            r[f"within_plant_median_{f}"], r[f"within_plant_n_{f}"] = float(wf.median()), int(len(wf))
        rep[name] = r
    pairs = [("multi-region", "US-only (pre-registered comparator)"), ("multi-region", "control (all US, no Europe)"),
             ("control (all US, no Europe)", "US-only (pre-registered comparator)"), ("multi-region", "multi-region context only")]
    for a, b in pairs:
        diff = (wp[a] - wp[b]).dropna()
        rep[f"paired: {a} - {b}"] = {"median": float(diff.median()), "ci": boot_median(diff.values), "plants": int(len(diff)),
                                     "first_better": int((diff > 0).sum())}
    prim = rep["paired: multi-region - US-only (pre-registered comparator)"]
    rep["success_criterion_met"] = bool(prim["ci"][0] > 0)
    country = fp["multi-region"].drop_duplicates("gppd_idnr").set_index("gppd_idnr").country
    dm = (wp["multi-region"] - wp["control (all US, no Europe)"]).dropna()
    by_c = pd.DataFrame({"multi": wp["multi-region"], "control": wp["control (all US, no Europe)"], "gain": dm})
    by_c = by_c.groupby(country.reindex(by_c.index).values).agg(plants=("gain", "size"), multi=("multi", "median"),
                                                                 control=("control", "median"), gain=("gain", "median"))
    rep["by_country"] = {k: {kk: (int(vv) if kk == "plants" else float(vv)) for kk, vv in v.items()}
                         for k, v in by_c[by_c.plants >= 5].iterrows()}

    # downstream CF: US-trained CF model (geo split, +sub+on-rate, residual on the national prior); US training plants
    # keep their out-of-fold on-rates (seed 0); European plants take on-rates from each activity model in turn
    rv = load("rv", "31_revision_cg.py")
    bl, ds = rv.bl, rv.ds
    plants = pd.read_parquet(root / "data/processed/plants.parquet")
    plants = plants.merge(pd.read_parquet(root / "data/processed/subtypes.parquet")[["gppd_idnr", *rv.SUB]], on="gppd_idnr", how="left")
    L = plants[plants.split_random.notna() & plants.primary_fuel.isin(rv.FUELS)].copy()
    for c in ["primary_fuel", "other_fuel1", *rv.SUB]:
        L[c] = L[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", *rv.SUB]
    us_rates, _ = rv.onrates(root, "onoff", {})
    tr, va, te = (L[L.split_geo == s] for s in ("train", "val", "test"))
    te = te[te.country.isin(rv.EUROPE)]
    feats = bl.F_WRI + rv.SUB + rv.ON
    j = lambda d, r: d.drop(columns=rv.ON, errors="ignore").join(r, on="gppd_idnr")
    cf = {"natl_prior": te.natl_cf.clip(0, 1).values,
          "+sub": bl.gbt_predict(tr, va, te, bl.F_WRI + rv.SUB, residual=True)}
    for name in ("US-only (pre-registered comparator)", "control (all US, no Europe)", "multi-region"):
        eu = onrate(fp[name])
        cf[f"+sub+onrate [{name}]"] = bl.gbt_predict(j(tr, us_rates), j(va, us_rates), j(te, eu), feats, residual=True)
    rep["cf_europe"] = {"plants": int(len(te)), "countries": int(te.country.nunique()),
                        "MAE": {k: float(np.abs(te.cf_target.values - p).mean()) for k, p in cf.items()}}
    gains = {}
    for a, b in [("natl_prior", "+sub+onrate [multi-region]"), ("+sub", "+sub+onrate [multi-region]"),
                 ("+sub+onrate [US-only (pre-registered comparator)]", "+sub+onrate [multi-region]"),
                 ("+sub+onrate [control (all US, no Europe)]", "+sub+onrate [multi-region]")]:
        for by in (False, True):
            m, lo, hi = rv.boot_diff(te, cf[a], cf[b], by)
            gains[f"{b} over {a} ({'country' if by else 'plant'})"] = [float(m), float(lo), float(hi)]
    rep["cf_europe"]["gain_ci"] = gains
    json.dump(rep, open(out / "multiregion.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
