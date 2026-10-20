"""Monthly capacity factors for US plants, 2017-2019 (EIA-923 monthly net generation / EIA-860 nameplate).

Used to supervise individual monthly Sentinel-2 frames (one label per frame instead of one per plant).

Output: <out.parquet> with gppd_idnr, year, month, gen_mwh, capacity_mw, cf, valid
usage: python 13_monthly_labels.py <gppd.csv> <eia_dir> <out.parquet>
"""
import calendar
import sys
from pathlib import Path

import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def read_923_monthly(eia_dir, y):
    f = next((eia_dir / f"f923_{y}").glob("EIA923_Schedules_2_3_4_5_M_12_*.xlsx"))
    d = pd.read_excel(f, sheet_name="Page 1 Generation and Fuel Data", skiprows=5)
    cols = {c: i for c in d.columns for i, m in enumerate(MONTHS, 1)
            if str(c).replace("\n", " ").lower() == f"netgen {m.lower()}"}
    d = d[["Plant Id", *cols]].rename(columns={"Plant Id": "plant", **cols})
    d = d[pd.to_numeric(d.plant, errors="coerce").notna()]
    d["plant"] = d.plant.astype(int)
    d = d[d.plant != 99999]
    d = d.melt(id_vars="plant", var_name="month", value_name="mwh")
    d["mwh"] = pd.to_numeric(d.mwh, errors="coerce")
    return d.groupby(["plant", "month"]).mwh.sum(min_count=1).reset_index()


def read_860(eia_dir, y):
    f = eia_dir / f"eia860{y}" / f"3_1_Generator_Y{y}.xlsx"
    d = pd.read_excel(f, sheet_name="Operable", skiprows=1)
    d = d[["Plant Code", "Nameplate Capacity (MW)"]].rename(columns={"Plant Code": "plant", "Nameplate Capacity (MW)": "mw"})
    d = d[pd.to_numeric(d.plant, errors="coerce").notna()]
    d["plant"] = d.plant.astype(int)
    d["mw"] = pd.to_numeric(d.mw, errors="coerce")
    return d.groupby("plant").mw.sum(min_count=1)


def main():
    gppd_csv, eia_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    g = pd.read_csv(gppd_csv, low_memory=False)
    us = g[g.gppd_idnr.str.match(r"^USA\d+$")][["gppd_idnr"]].copy()
    us["plant"] = us.gppd_idnr.str[3:].astype(int)
    rows = []
    for y in (2017, 2018, 2019):
        m, cap = read_923_monthly(eia_dir, y), read_860(eia_dir, y)
        m = m.merge(us, on="plant")
        m["year"], m["capacity_mw"] = y, m.plant.map(cap)
        rows.append(m)
        print(y, "plant-months:", len(m), flush=True)
    d = pd.concat(rows).dropna(subset=["mwh", "capacity_mw"])
    hours = d.apply(lambda r: calendar.monthrange(int(r.year), int(r.month))[1] * 24, axis=1)
    d["cf"] = d.mwh / (d.capacity_mw * hours)
    d["valid"] = d.cf.between(0, 1) & (d.capacity_mw > 0)
    d = d.rename(columns={"mwh": "gen_mwh"})[["gppd_idnr", "year", "month", "gen_mwh", "capacity_mw", "cf", "valid"]]
    d.to_parquet(out, index=False)
    print(d.valid.mean().round(3), "valid;", d[d.valid].gppd_idnr.nunique(), "plants")
    print(d[d.valid].groupby("month").cf.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
