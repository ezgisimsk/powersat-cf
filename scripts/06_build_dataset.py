"""One row per GPPD plant: tabular features, targets and evaluation splits.

Targets
  cf_target  mean valid CF over 2017-2019 (GPPD; non-US plants mostly 2017-2018)   <- main target
  cf_2019    CF in 2019 (US only, same year as imagery)                          <- robustness
  cf_future  mean valid CF over 2023-2025 (EIA, US only)                         <- temporal shift test

Splits (all at plant level, only plants with cf_target; Storage / Wave and Tidal excluded)
  split_random  train/val/test 70/15/15, stratified by fuel x (US / non-US)
  split_geo     US -> train/val (85/15); non-US -> test (geographic out-of-distribution)
  fold_spatial  0..4, GroupKFold over 2x2 degree grid cells (spatial block CV)

usage: python 06_build_dataset.py <gppd.csv> <labels.parquet> <ember.csv> <out.parquet>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

SEED = 42
EXCLUDE_FUELS = {"Storage", "Wave and Tidal"}
EMBER_FUEL = {
    "Biomass": "Bioenergy", "Waste": "Bioenergy", "Coal": "Coal", "Petcoke": "Coal", "Gas": "Gas",
    "Cogeneration": "Gas", "Oil": "Other Fossil", "Other": "Other Fossil", "Hydro": "Hydro",
    "Nuclear": "Nuclear", "Solar": "Solar", "Wind": "Wind", "Geothermal": "Other Renewables",
    "Wave and Tidal": "Other Renewables", "Storage": None,
}


def ember_cf(path):
    """National fuel-level CF averaged over 2017-2019, from Ember generation (TWh) / capacity (GW)."""
    e = pd.read_csv(path, low_memory=False)
    e = e[(e.Subcategory == "Fuel") & e.Year.between(2017, 2019) & (e["Area type"] == "Country or economy")]
    gen = e[(e.Category == "Electricity generation") & (e.Unit == "TWh")]
    cap = e[(e.Category == "Capacity") & (e.Unit == "GW")]
    k = ["ISO 3 code", "Year", "Variable"]
    m = gen[k + ["Value", "Ember region"]].merge(cap[k + ["Value"]], on=k, suffixes=("_twh", "_gw"))
    m = m[m.Value_gw > 0.05]
    m["cf"] = m.Value_twh / (m.Value_gw * 8.76)
    m = m[m.cf.between(0.01, 1)]
    ctry = m.groupby(["ISO 3 code", "Variable"]).cf.mean()
    region = m.groupby(["Ember region", "Variable"]).cf.median()
    glob = m.groupby("Variable").cf.median()
    iso_region = e.drop_duplicates("ISO 3 code").set_index("ISO 3 code")["Ember region"]
    return ctry, region, glob, iso_region


def main():
    gppd_csv, labels_pq, ember_csv, out = sys.argv[1:5]
    g = pd.read_csv(gppd_csv, low_memory=False)
    lab = pd.read_parquet(labels_pq)
    lab = lab[lab.valid]

    # --- targets
    t = lab[(lab.source == "gppd") & lab.year.between(2017, 2019)].groupby("gppd_idnr").cf.agg(["mean", "size"])
    g["cf_target"] = g.gppd_idnr.map(t["mean"])
    g["n_target_years"] = g.gppd_idnr.map(t["size"]).fillna(0).astype(int)
    g["cf_2019"] = g.gppd_idnr.map(lab[(lab.source == "gppd") & (lab.year == 2019)].set_index("gppd_idnr").cf)
    g["cf_future"] = g.gppd_idnr.map(lab[(lab.source == "eia") & lab.year.between(2023, 2025)].groupby("gppd_idnr").cf.mean())
    g["gen_gwh_target"] = g.cf_target * g.capacity_mw * 8.76

    # --- tabular features
    g["log_capacity"] = np.log1p(g.capacity_mw)
    g["age"] = 2018 - g.commissioning_year
    g["n_other_fuels"] = g[["other_fuel1", "other_fuel2", "other_fuel3"]].notna().sum(axis=1)
    g["is_us"] = g.country == "USA"
    ctry, region, glob, iso_region = ember_cf(ember_csv)
    ef = g.primary_fuel.map(EMBER_FUEL)
    g["ember_fuel"] = ef
    keys_c = list(zip(g.country, ef))
    keys_r = list(zip(g.country.map(iso_region), ef))
    g["natl_cf"] = [ctry.get(k, np.nan) for k in keys_c]
    g["natl_cf_level"] = np.where(g.natl_cf.notna(), "country", "region")
    g.loc[g.natl_cf.isna(), "natl_cf"] = [region.get(k, np.nan) for k, isna in zip(keys_r, g.natl_cf.isna()) if isna]
    g.loc[g.natl_cf.isna(), "natl_cf_level"] = "global"
    g.loc[g.natl_cf.isna(), "natl_cf"] = ef[g.natl_cf.isna()].map(glob)
    g["wri_est_cf_2017"] = g.estimated_generation_gwh_2017 * 1000 / (g.capacity_mw * 8760)

    # --- splits
    lab_mask = g.cf_target.notna() & ~g.primary_fuel.isin(EXCLUDE_FUELS)
    L = g[lab_mask]
    strat = L.primary_fuel + L.is_us.astype(str)
    strat = strat.where(strat.map(strat.value_counts()) >= 20, "rare")
    tr, rest = train_test_split(L.index, test_size=0.30, stratify=strat, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=strat[rest], random_state=SEED)
    g["split_random"] = None
    g.loc[tr, "split_random"], g.loc[va, "split_random"], g.loc[te, "split_random"] = "train", "val", "test"

    us, nonus = L.index[L.is_us], L.index[~L.is_us]
    tr, va = train_test_split(us, test_size=0.15, stratify=strat[us], random_state=SEED)
    g["split_geo"] = None
    g.loc[tr, "split_geo"], g.loc[va, "split_geo"], g.loc[nonus, "split_geo"] = "train", "val", "test"

    cell = (np.floor(L.latitude / 2).astype(int).astype(str) + "_" + np.floor(L.longitude / 2).astype(int).astype(str))
    g["fold_spatial"] = -1
    for k, (_, idx) in enumerate(GroupKFold(5, shuffle=True, random_state=SEED).split(L, groups=cell)):
        g.loc[L.index[idx], "fold_spatial"] = k

    g.to_parquet(out, index=False)
    print("labeled plants:", lab_mask.sum(), "| US:", L.is_us.sum(), "non-US:", (~L.is_us).sum())
    print("natl_cf source:", g[lab_mask].natl_cf_level.value_counts().to_dict())
    print(g.split_random.value_counts().to_dict(), g.split_geo.value_counts().to_dict())
    print("spatial fold sizes:", g[lab_mask].fold_spatial.value_counts().sort_index().to_dict())
    print("cf_future available:", g.cf_future.notna().sum(), " cf_2019:", g.cf_2019.notna().sum())


if __name__ == "__main__":
    main()
