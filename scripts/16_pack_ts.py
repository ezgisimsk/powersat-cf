"""Pack the monthly thermal-plant frames into one memmap of central crops for the frame-level CNN.

Writes <prefix>.npy (N x 7 x C x C uint16; channels B02 B03 B04 B08 B11 B12 SCL) and <prefix>.frames.parquet
(row, gppd_idnr, date, year, month, scene_cloud) in the same row order.

usage: python 16_pack_ts.py <series_dir> <out_prefix> [--crop 64]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir")
    ap.add_argument("prefix")
    ap.add_argument("--crop", type=int, default=64)
    a = ap.parse_args()
    files = sorted(Path(a.series_dir).glob("*.npz"))
    counts = [len(np.load(f)["dates"]) for f in files]
    n, c = sum(counts), a.crop
    mm = np.lib.format.open_memmap(a.prefix + ".npy", mode="w+", dtype=np.uint16, shape=(n, 7, c, c))
    rows, i = [], 0
    for k, f in enumerate(files):
        d = np.load(f)
        x = d["x"]
        s = x.shape[-1]
        o = (s - c) // 2
        mm[i : i + len(x)] = x[:, :, o : o + c, o : o + c]
        for t in range(len(x)):
            rows.append((i + t, f.stem, str(d["dates"][t]), float(d["cloud"][t])))
        i += len(x)
        if k % 500 == 0:
            print(k, len(files), flush=True)
    mm.flush()
    df = pd.DataFrame(rows, columns=["row", "gppd_idnr", "date", "scene_cloud"])
    dt = pd.to_datetime(df.date)
    df["year"], df["month"] = dt.dt.year, dt.dt.month
    df.to_parquet(a.prefix + ".frames.parquet", index=False)
    print("packed", mm.shape)


if __name__ == "__main__":
    main()
