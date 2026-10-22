"""Experiments F and G (pre-registered after review round 1).

F  Leave-one-US-state-out with on-rates from activity models trained on state-grouped folds (scripts/24 --state_folds),
   so a held-out state's plants never trained the model that scores them; three activity seeds, one capacity-factor
   model per seed, predictions averaged. The earlier protocol (random plant folds, seed 0) is rerun for comparison.
   Primary: MAE gain of +sub+onrate over +sub, state-cluster bootstrap 95 % CI; success if the lower bound > 0.
G  Oracle: the same capacity-factor model with the true on-rate (mean overpass-hour on/off and utilisation over clear
   labelled frames; CEMS in the US, ENTSO-E in Europe) in place of the predicted one. Leave-one-state-out (US) and
   US -> Europe. Plants without labels keep a missing value. Descriptive; reported on all plants and on labelled ones.

usage: python 38_state_folds_oracle.py <project_root> <out_dir>
"""
import importlib.util
import json
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


rv = load("rv", "31_revision_cg.py")
bl, ds = rv.bl, rv.ds
SUB, ON = rv.SUB, rv.ON


def true_rates(root, labels):
    fr = pd.read_parquet(root / "data/processed/ts_frames64.frames.parquet", columns=["gppd_idnr", "date", "scene_cloud"])
    d = fr.merge(labels[["gppd_idnr", "date", "on", "inst_util"]], on=["gppd_idnr", "date"])
    d = d[d.scene_cloud < rv.CLEAR]
    return d.groupby("gppd_idnr").agg(on_rate=("on", "mean"), util=("inst_util", "mean"), n_clear=("on", "size"))


def gain(te, pa, pb, group_col):
    """Mean |err_a| - |err_b| (positive = b better), cluster bootstrap over group_col."""
    m, lo, hi = rv.boot_diff(te.assign(country=te[group_col].values), pa, pb, True)
    return [round(float(m), 4), round(float(lo), 4), round(float(hi), 4)]


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    plants = pd.read_parquet(root / "data/processed/plants.parquet")
    plants = plants.merge(pd.read_parquet(root / "data/processed/subtypes.parquet")[["gppd_idnr", *SUB]], on="gppd_idnr", how="left")
    L = plants[plants.split_random.notna() & plants.primary_fuel.isin(rv.FUELS)].copy()
    for c in ["primary_fuel", "other_fuel1", *SUB]:
        L[c] = L[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", *SUB]
    lab = pd.read_parquet(root / "data/processed/labels.parquet").query("valid & source == 'gppd' & year >= 2017 & year <= 2019")
    label_years = lab.groupby("gppd_idnr").year.apply(set)
    st = pd.read_excel(root / "data/raw/eia/eia8602019/2___Plant_Y2019.xlsx", skiprows=1)[["Plant Code", "State"]]
    st = st.set_index(pd.to_numeric(st["Plant Code"], errors="coerce")).State
    L["state"] = [st.get(int(g[3:])) if g.startswith("USA") and g[3:].isdigit() else None for g in L.gppd_idnr]

    rates = {"state_folds": [rv.onrates(root, d, label_years)[1] for d in ("onoff_state", "onoff_state_s1", "onoff_state_s2")],
             "random_folds": [rv.onrates(root, d, label_years)[1] for d in ("onoff", "onoff_s1", "onoff_s2")]}
    oracle = pd.concat([true_rates(root, pd.read_parquet(root / "data/processed/cems_overpass.parquet")),
                        true_rates(root, pd.read_parquet(root / "data/processed/entsoe_overpass.parquet"))])
    j = lambda d, r: d.drop(columns=ON, errors="ignore").join(r, on="gppd_idnr")
    F_SUB, F_ON = bl.F_WRI + SUB, bl.F_WRI + SUB + ON

    # ---- leave-one-state-out
    us = L[L.country == "USA"].dropna(subset=["state"])
    rows = []
    for s in sorted(us.state.unique()):
        te = us[us.state == s]
        if len(te) < 10:
            continue
        trv = us[us.state != s]
        va = trv.sample(frac=0.1, random_state=0)
        tr = trv.drop(va.index)
        p = {"natl_prior": te.natl_cf.clip(0, 1).values, "+sub": bl.gbt_predict(tr, va, te, F_SUB, residual=True)}
        for name, rs in rates.items():
            p[f"+sub+onrate [{name}, 3 seeds]"] = np.mean([bl.gbt_predict(j(tr, r), j(va, r), j(te, r), F_ON, residual=True) for r in rs], axis=0)
            p[f"+sub+onrate [{name}, seed 0]"] = bl.gbt_predict(j(tr, rs[0]), j(va, rs[0]), j(te, rs[0]), F_ON, residual=True)
        p["+sub+onrate [oracle]"] = bl.gbt_predict(j(tr, oracle), j(va, oracle), j(te, oracle), F_ON, residual=True)
        rows.append(pd.DataFrame(p).assign(state=s, gppd_idnr=te.gppd_idnr.values, cf_target=te.cf_target.values,
                                           labelled=te.gppd_idnr.isin(oracle.index).values))
    lo = pd.concat(rows, ignore_index=True)
    models = [c for c in lo.columns if c == "natl_prior" or c.startswith("+")]
    rep = {"loso": {"plants": int(len(lo)), "states": int(lo.state.nunique()), "labelled_plants": int(lo.labelled.sum())}}
    for subset, d in (("all", lo), ("labelled", lo[lo.labelled])):
        mae = {m: round(float(np.abs(d.cf_target - d[m]).mean()), 4) for m in models}
        g = {f"{b} over {a}": gain(d, d[a].values, d[b].values, "state") for a, b in [
            ("+sub", "+sub+onrate [state_folds, 3 seeds]"), ("+sub", "+sub+onrate [random_folds, 3 seeds]"),
            ("+sub", "+sub+onrate [random_folds, seed 0]"), ("+sub", "+sub+onrate [oracle]"),
            ("natl_prior", "+sub+onrate [state_folds, 3 seeds]")]}
        per_state = d.assign(**{m: np.abs(d.cf_target - d[m]) for m in models}).groupby("state")[models].mean()
        beats = {m: int((per_state[m] < per_state["+sub"]).sum()) for m in models if m.startswith("+sub+")}
        beats_prior = {m: int((per_state[m] < per_state["natl_prior"]).sum()) for m in models if m != "natl_prior"}
        rep["loso"][subset] = {"MAE": mae, "gain_ci": g, "states_better_than_sub": beats, "states_better_than_prior": beats_prior}
        if subset == "labelled":
            o = mae["+sub"] - mae["+sub+onrate [oracle]"]
            pr = mae["+sub"] - mae["+sub+onrate [state_folds, 3 seeds]"]
            rep["loso"][subset]["share_of_oracle_gain_captured"] = round(pr / o, 3) if o > 0 else None
    rep["success_criterion_met"] = bool(rep["loso"]["all"]["gain_ci"]["+sub+onrate [state_folds, 3 seeds] over +sub"][1] > 0)

    # ---- US -> Europe with the oracle
    tr, va, te = (L[L.split_geo == s] for s in ("train", "val", "test"))
    te = te[te.country.isin(rv.EUROPE)]
    pred_rates = rates["random_folds"]
    e = {"natl_prior": te.natl_cf.clip(0, 1).values, "+sub": bl.gbt_predict(tr, va, te, F_SUB, residual=True),
         "+sub+onrate [predicted]": np.mean([bl.gbt_predict(j(tr, r), j(va, r), j(te, r), F_ON, residual=True) for r in pred_rates], axis=0),
         "+sub+onrate [oracle]": bl.gbt_predict(j(tr, oracle), j(va, oracle), j(te, oracle), F_ON, residual=True)}
    ed = pd.DataFrame(e).assign(country=te.country.values, cf_target=te.cf_target.values, labelled=te.gppd_idnr.isin(oracle.index).values)
    rep["europe"] = {"plants": int(len(ed)), "labelled_plants": int(ed.labelled.sum())}
    for subset, d in (("all", ed), ("labelled", ed[ed.labelled])):
        rep["europe"][subset] = {"MAE": {m: round(float(np.abs(d.cf_target - d[m]).mean()), 4) for m in e},
                                 "gain_ci_country": {f"{b} over natl_prior": gain(d, d["natl_prior"].values, d[b].values, "country")
                                                     for b in ("+sub", "+sub+onrate [predicted]", "+sub+onrate [oracle]")}}
    lo.to_csv(out / "loso_preds.csv", index=False)
    json.dump(rep, open(out / "state_folds_oracle.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
