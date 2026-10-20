"""Plant-level gross load around the Sentinel-2 overpass from EPA CEMS hourly data (2017-2019).

CEMS hours are local standard time; Sentinel-2 passes at ~10:30 local solar time, so the label is the plant's
summed unit gross load in hour 10 (10:00-10:59). Reads the PUDL raw archive zips (quarterly CSVs) as streams,
keeping hours 9-12 only. A plant's reference load is its maximum summed daytime load over 2017-2019 (covers
exactly the CEMS units, unlike nameplate capacity), and
  inst_util = load_h10 / reference load,  on = inst_util > 0.05.

Output: <out.parquet> with gppd_idnr, date, load_h9..load_h12, ref_load, inst_util, on
usage: python 23_cems_overpass.py <cems_dir> <out.parquet>
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

COLS = ["Facility ID", "Date", "Hour", "Gross Load (MW)"]


def main():
    cems, out = Path(sys.argv[1]), sys.argv[2]
    parts = []
    for z in sorted(cems.glob("epacems-201[789].zip")):
        with zipfile.ZipFile(z) as zf:
            for name in sorted(zf.namelist()):
                with zf.open(name) as f:
                    for ch in pd.read_csv(f, usecols=COLS, chunksize=2_000_000, dtype={"Facility ID": "int32"}):
                        ch = ch[ch.Hour.between(9, 12)]
                        g = ch.groupby(["Facility ID", "Date", "Hour"])["Gross Load (MW)"].sum(min_count=1)
                        parts.append(g)
                print(z.name, name, flush=True)
    s = pd.concat(parts)
    s = s.groupby(level=[0, 1, 2]).sum(min_count=1)  # chunks can split a plant-hour
    d = s.unstack("Hour").fillna(0)
    d.columns = [f"load_h{h}" for h in d.columns]
    d = d.reset_index().rename(columns={"Facility ID": "plant", "Date": "date"})
    d["ref_load"] = d.groupby("plant")[[c for c in d.columns if c.startswith("load_h")]].transform("max").max(axis=1)
    d = d[d.ref_load > 0]
    d["inst_util"] = (d.load_h10 / d.ref_load).clip(0, 1)
    d["on"] = (d.inst_util > 0.05).astype("int8")
    d["gppd_idnr"] = "USA" + d.plant.astype(str).str.zfill(7)
    d = d.drop(columns="plant")
    d.to_parquet(out, index=False)
    print(d.shape, "plants:", d.gppd_idnr.nunique(), "share on:", round(d.on.mean(), 3))


if __name__ == "__main__":
    main()
