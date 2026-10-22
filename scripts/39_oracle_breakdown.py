"""Breakdowns requested in review round 2 (no new training; reads outputs of scripts/38).

1. Held-out states: oracle and predicted on-rate gains on labelled vs unlabelled plants, and the predicted on-rate's
   gain over subtypes by capacity class (<=25, 25-100, 100-500, >500 MW), state-cluster bootstrap.
2. Raw true utilisation as a direct estimate: MAE of the plant's mean overpass-hour utilisation (clear frames) against
   the capacity-factor target, for labelled US and European plants, next to the model fed with the true on-rate.

usage: python 39_oracle_breakdown.py <project_root>   (writes results/state_folds_oracle/breakdown.json)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("so", HERE / "38_state_folds_oracle.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)
PRED, SUB, ORA = "+sub+onrate [state_folds, 3 seeds]", "+sub", "+sub+onrate [oracle]"


def mae(y, p):
    return round(float(np.abs(y - p).mean()), 4)


def main():
    root = Path(sys.argv[1])
    out = root / "results/state_folds_oracle"
    lo = pd.read_csv(out / "loso_preds.csv")
    plants = pd.read_parquet(root / "data/processed/plants.parquet", columns=["gppd_idnr", "capacity_mw", "country"])
    lo = lo.merge(plants[["gppd_idnr", "capacity_mw"]], on="gppd_idnr", how="left")
    rep = {"loso_by_labelled": {}, "loso_by_capacity": {}, "raw_utilisation": {}}
    for name, d in (("labelled", lo[lo.labelled]), ("unlabelled", lo[~lo.labelled])):
        rep["loso_by_labelled"][name] = {"plants": int(len(d)), "MAE": {m: mae(d.cf_target, d[m]) for m in ("natl_prior", SUB, PRED, ORA)}}
    bins = [0, 25, 100, 500, np.inf]
    lo["cap_class"] = pd.cut(lo.capacity_mw, bins, labels=["<=25", "25-100", "100-500", ">500"])
    for c, d in lo.groupby("cap_class", observed=True):
        g = so.gain(d, d[SUB].values, d[PRED].values, "state") if d.state.nunique() > 3 else None
        rep["loso_by_capacity"][str(c)] = {"plants": int(len(d)), "labelled_share": round(float(d.labelled.mean()), 3),
                                           "MAE_sub": mae(d.cf_target, d[SUB]), "MAE_sub_onrate": mae(d.cf_target, d[PRED]),
                                           "gain_ci_state": g}
    # raw true utilisation as an estimate of the capacity factor
    tr = pd.concat([so.true_rates(root, pd.read_parquet(root / "data/processed/cems_overpass.parquet")),
                    so.true_rates(root, pd.read_parquet(root / "data/processed/entsoe_overpass.parquet"))])
    L = pd.read_parquet(root / "data/processed/plants.parquet", columns=["gppd_idnr", "country", "primary_fuel", "cf_target", "split_random", "natl_cf"])
    L = L[L.split_random.notna() & L.primary_fuel.isin(["Coal", "Gas"])].join(tr, on="gppd_idnr", how="inner")
    eu = L[L.country.isin(so.rv.EUROPE)]
    us = L[L.country == "USA"]
    rep["raw_utilisation"]["europe_labelled"] = {"plants": int(len(eu)), "MAE_raw_util": mae(eu.cf_target, eu.util),
                                                 "MAE_raw_onrate": mae(eu.cf_target, eu.on_rate), "MAE_natl_prior": mae(eu.cf_target, eu.natl_cf.clip(0, 1))}
    lu = lo[lo.labelled].join(tr, on="gppd_idnr")
    rep["raw_utilisation"]["us_heldout_states_labelled"] = {"plants": int(len(lu)), "MAE_raw_util": mae(lu.cf_target, lu.util),
                                                           "MAE_raw_onrate": mae(lu.cf_target, lu.on_rate), "MAE_oracle_model": mae(lu.cf_target, lu[ORA])}
    json.dump(rep, open(out / "breakdown.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
