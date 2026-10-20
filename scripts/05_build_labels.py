"""Build a long plant-year capacity-factor label table.

Sources
  * GPPD v1.3 reported generation 2013-2019 (all countries), capacity = GPPD capacity_mw
  * EIA-923 net generation + EIA-860 operable nameplate capacity, 2019-2025 (US plants;
    GPPD US id 'USA00xxxxx' -> EIA plant code xxxxx)

Output: <out_dir>/labels.parquet with columns
  gppd_idnr, year, source, gen_gwh, capacity_mw, cf, valid
and <out_dir>/eia_vs_gppd_2019.txt (agreement check on the overlapping year).

usage: python 05_build_labels.py <gppd.csv> <eia_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEARS_EIA = range(2019, 2026)


def read_923(eia_dir, y):
    f = next((eia_dir / f"f923_{y}").glob("EIA923_Schedules_2_3_4_5_M_12_*.xlsx"))
    d = pd.read_excel(f, sheet_name="Page 1 Generation and Fuel Data", skiprows=5)
    gen_col = next(c for c in d.columns if str(c).startswith("Net Generation"))
    d = d[["Plant Id", gen_col]].rename(columns={"Plant Id": "plant", gen_col: "mwh"})
    d = d[pd.to_numeric(d.plant, errors="coerce").notna()]
    d["plant"] = d.plant.astype(int)
    d = d[d.plant != 99999]  # state-level imputation rows
    d["mwh"] = pd.to_numeric(d.mwh, errors="coerce")
    return d.groupby("plant").mwh.sum(min_count=1)


def read_860(eia_dir, y):
    f = eia_dir / f"eia860{y}" / f"3_1_Generator_Y{y}.xlsx"
    d = pd.read_excel(f, sheet_name="Operable", skiprows=1)
    d = d[["Plant Code", "Nameplate Capacity (MW)"]].rename(columns={"Plant Code": "plant", "Nameplate Capacity (MW)": "mw"})
    d = d[pd.to_numeric(d.plant, errors="coerce").notna()]
    d["plant"] = d.plant.astype(int)
    d["mw"] = pd.to_numeric(d.mw, errors="coerce")
    return d.groupby("plant").mw.sum(min_count=1)


def main():
    gppd_csv, eia_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(gppd_csv, low_memory=False)

    # --- GPPD reported generation
    rows = []
    for y in range(2013, 2020):
        d = g[["gppd_idnr", "capacity_mw", f"generation_gwh_{y}"]].dropna()
        d = d.rename(columns={f"generation_gwh_{y}": "gen_gwh"}).assign(year=y, source="gppd")
        rows.append(d)
    gppd = pd.concat(rows)

    # --- EIA
    us = g[g.gppd_idnr.str.match(r"^USA\d+$")][["gppd_idnr"]].copy()
    us["plant"] = us.gppd_idnr.str[3:].astype(int)
    rows = []
    for y in YEARS_EIA:
        gen, cap = read_923(eia_dir, y), read_860(eia_dir, y)
        d = us.assign(year=y, source="eia")
        d["gen_gwh"] = d.plant.map(gen) / 1000
        d["capacity_mw"] = d.plant.map(cap)
        rows.append(d.dropna(subset=["gen_gwh", "capacity_mw"]).drop(columns="plant"))
        print(y, "EIA plants matched:", rows[-1].shape[0], flush=True)
    eia = pd.concat(rows)

    lab = pd.concat([gppd, eia], ignore_index=True)
    lab["cf"] = lab.gen_gwh * 1000 / (lab.capacity_mw * 8760)
    lab["valid"] = lab.cf.between(0, 1) & (lab.capacity_mw > 0)
    lab = lab[["gppd_idnr", "year", "source", "gen_gwh", "capacity_mw", "cf", "valid"]]
    lab.to_parquet(out_dir / "labels.parquet", index=False)

    # --- agreement check on 2019 (GPPD's US numbers come from EIA too)
    a = lab[(lab.year == 2019) & lab.valid].pivot_table(index="gppd_idnr", columns="source", values="cf")
    a = a.dropna()
    with open(out_dir / "eia_vs_gppd_2019.txt", "w") as f:
        msg = (f"plants with both: {len(a)}\n"
               f"CF corr: {a.corr().iloc[0, 1]:.4f}\n"
               f"|dCF| median: {(a.eia - a.gppd).abs().median():.4f}  p95: {(a.eia - a.gppd).abs().quantile(.95):.4f}\n")
        f.write(msg)
    print(msg)
    print(lab.groupby(["source", "year"]).valid.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
