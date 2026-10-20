"""Spatial-block CV summary for neural models: MAE per fold (2-degree cells), mean +- s.d. over the 5 folds.

Run dirs: results/runs_spatial/spatial<k>__<variant>__aux<w>__s<seed>/preds_test.csv
usage: python 28_spatial_nn.py <plants.parquet> <runs_dir> <out.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    plants, runs, out = pd.read_parquet(sys.argv[1]).set_index("gppd_idnr"), Path(sys.argv[2]), sys.argv[3]
    rows = []
    for d in sorted(runs.glob("spatial*__*")):
        f = d / "preds_test.csv"
        if not d.is_dir() or not f.exists():
            continue
        fold, variant, aux, seed = d.name.split("__")
        p = pd.read_csv(f).set_index("gppd_idnr").pred
        y = plants.cf_target.reindex(p.index)
        rows.append({"model": f"{variant}_{aux}", "fold": int(fold.replace("spatial", "")), "n": len(p),
                     "MAE": float(np.abs(p - y).mean()),
                     "R2": float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())})
    r = pd.DataFrame(rows)
    s = r.groupby("model").agg(folds=("fold", "nunique"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
                                R2=("R2", "mean"), R2_sd=("R2", "std")).round(4)
    s.to_csv(out)
    print(s.to_string())


if __name__ == "__main__":
    main()
