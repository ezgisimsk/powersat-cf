"""Compare trained models with tabular baselines on identical test plants.

Run dirs are named <split>__<variant>__aux<w>__s<seed>; seeds are averaged as mean +- std.
Baseline predictions come from 07_baselines.py (results/baselines/preds_<split>.csv).

usage: python 10_compare.py <plants.parquet> <baselines_dir> <runs_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EUROPE = {"ITA", "DEU", "ESP", "FRA", "POL", "GRC", "SWE", "AUT", "BEL", "ROU", "CZE", "DNK", "PRT", "NLD",
          "FIN", "HUN", "IRL", "BGR", "SVK", "SVN", "GBR", "EST", "LTU", "LVA", "HRV", "LUX", "NOR", "CHE"}


def metrics(y, p, cap):
    gen_err = np.abs(y - p) * cap * 8.76
    return {"n": len(y), "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "MAE_GWh": gen_err.mean(), "wMAPE_gen": gen_err.sum() / (y * cap * 8.76).sum()}


def subsets(split, d):
    yield "all", np.ones(len(d), bool)
    if split == "geo":
        yield "India", (d.country == "IND").values
        yield "Australia", (d.country == "AUS").values
        yield "Europe", d.country.isin(EUROPE).values


def main():
    plants, bdir, rdir, out = pd.read_parquet(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    plants = plants.set_index("gppd_idnr")
    for split in ("random", "geo"):
        runs = {}
        for r in sorted(rdir.glob(f"{split}__*")):
            if (r / "preds_test.csv").exists():
                runs[r.name] = pd.read_csv(r / "preds_test.csv").set_index("gppd_idnr").pred
        if not runs:
            continue
        base = pd.read_csv(bdir / f"preds_{split}.csv").set_index("gppd_idnr")
        common = base.index
        for p in runs.values():
            common = common.intersection(p.index)
        d = plants.loc[common]
        rows = []
        for sub, m in subsets(split, d):
            if m.sum() < 10:
                continue
            y, cap = d.cf_target.values[m], d.capacity_mw.values[m]
            for b in base.columns:
                bp = base.loc[common, b].values[m]
                ok = ~np.isnan(bp)
                if ok.sum() < 10:
                    continue
                rows.append({"subset": sub, "model": b, "seed": 0, **metrics(y[ok], bp[ok], cap[ok])})
            for name, p in runs.items():
                _, variant, aux, seed = name.split("__")
                rows.append({"subset": sub, "model": f"{variant}_{aux}", "seed": int(seed[1:]),
                             **metrics(y, p.loc[common].values[m], cap)})
        res = pd.DataFrame(rows)
        agg = res.groupby(["subset", "model"]).agg(
            n=("n", "first"), seeds=("seed", "nunique"),
            MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"),
            R2=("R2", "mean"), R2_sd=("R2", "std"), MAE_GWh=("MAE_GWh", "mean"), wMAPE_gen=("wMAPE_gen", "mean"),
        ).reset_index().sort_values(["subset", "MAE"])
        agg.round(4).to_csv(out / f"comparison_{split}.csv", index=False)

        # per-fuel MAE on the full test set (seed-averaged predictions)
        pf = {}
        for b in ("natl_cf", "gbt_wri", "gbt_plus"):
            pf[b] = (d.cf_target - base.loc[common, b]).abs().groupby(d.primary_fuel).mean()
        for v in sorted({n.split("__", 1)[1].rsplit("__", 1)[0] for n in runs}):
            p = np.mean([runs[n].loc[common].values for n in runs if n.split("__", 1)[1].startswith(v + "__")], 0)
            pf[v.replace("__", "_")] = (d.cf_target - p).abs().groupby(d.primary_fuel).mean()
        pf = pd.DataFrame(pf).assign(n=d.primary_fuel.value_counts()).sort_values("n", ascending=False)
        pf.round(3).to_csv(out / f"per_fuel_{split}.csv")

        pd.set_option("display.width", 220)
        print(f"\n===== {split} (n common = {len(common)}) =====")
        print(agg.round(3).to_string(index=False))
        print(pf.round(3).to_string())


if __name__ == "__main__":
    main()
