"""Interpretable per-frame activity features from the monthly Sentinel-2 series of thermal plants.

For every frame (plant x month) and two regions -- the central 32x32 px (~320 m) and the full 96x96 px
patch -- we compute:
  white_frac   bright, low-saturation pixels (water-vapour plume candidates): mean RGB > 0.2, (max-min)/mean < 0.25
  scl_cloud    fraction of SCL cloud classes (8, 9, 10) -- plumes are usually labelled cloud
  bright_mean  mean visible reflectance
  b12_p99      99th percentile of SWIR-2 reflectance (hot / combustion-related surfaces)
  swir_ratio   mean B12 / B11
  water_frac   NDWI > 0.2 (cooling ponds; mostly static context)
plus cloud_excess = scl_cloud(center) - scene_cloud/100: local cloudiness beyond the scene average,
the cleanest single plume indicator when the scene itself is clear.

Output: <out.parquet>, one row per frame (gppd_idnr, date, year, month, scene_cloud, features).
usage: python 14_ts_features.py <series_dir> <out.parquet> [--workers 6]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CLOUD_SCL = np.array([8, 9, 10])


def region_feats(x, prefix):
    rgb = x[[2, 1, 0]] / 10000.0
    mean = rgb.mean(0)
    sat = (rgb.max(0) - rgb.min(0)) / np.maximum(mean, 1e-4)
    b11, b12 = x[4] / 10000.0, x[5] / 10000.0
    green, nir = x[1] / 10000.0, x[3] / 10000.0
    ndwi = (green - nir) / np.maximum(green + nir, 1e-4)
    return {
        f"{prefix}_white_frac": float(((mean > 0.2) & (sat < 0.25)).mean()),
        f"{prefix}_scl_cloud": float(np.isin(x[6], CLOUD_SCL).mean()),
        f"{prefix}_bright_mean": float(mean.mean()),
        f"{prefix}_b12_p99": float(np.percentile(b12, 99)),
        f"{prefix}_swir_ratio": float(b12.mean() / max(b11.mean(), 1e-4)),
        f"{prefix}_water_frac": float((ndwi > 0.2).mean()),
    }


def plant_frames(path):
    d = np.load(path)
    x, dates, cloud = d["x"].astype(np.float32), d["dates"], d["cloud"]
    s = x.shape[-1]
    c0, c1 = s // 2 - 16, s // 2 + 16
    rows = []
    for t in range(len(x)):
        f = {"gppd_idnr": path.stem, "date": str(dates[t]), "scene_cloud": float(cloud[t])}
        f.update(region_feats(x[t, :, c0:c1, c0:c1], "c"))
        f.update(region_feats(x[t], "f"))
        f["cloud_excess"] = f["c_scl_cloud"] - f["scene_cloud"] / 100
        rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(Path(a.series_dir).glob("*.npz"))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(plant_frames, files, chunksize=8)):
            rows += r
            if i % 500 == 0:
                print(i, len(files), flush=True)
    df = pd.DataFrame(rows)
    dt = pd.to_datetime(df.date)
    df["year"], df["month"] = dt.dt.year, dt.dt.month
    df.to_parquet(a.out, index=False)
    print(df.shape)
    print(df.describe().T[["mean", "50%", "max"]].round(3))


if __name__ == "__main__":
    main()
