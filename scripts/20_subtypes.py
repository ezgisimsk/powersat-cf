"""Plant-level technology subtype and cooling type from EIA-860 (US) and JRC-PPDB-OPEN (Europe).

Per plant the capacity-weighted dominant generator technology is mapped to a harmonized subtype, e.g.
gas_ccgt / gas_ocgt / gas_ic / gas_steam, hydro_ror / hydro_reservoir / hydro_pumped / hydro_conv (US: no
run-of-river vs storage split), coal_hard / coal_lignite / coal_steam, bio_landfill / bio_msw / bio_wood ...
Cooling (thermal plants) is harmonized to tower_natural / tower_mech / once_through / pond / dry / none.
JRC units are linked to GPPD ids through JRC_OPEN_LINKAGES.WRI_id.

Output: <out.parquet> with gppd_idnr, subtype, subtype_src, cooling, plus classifier targets
hydro_class (ror / storage / pumped), gas_class (ccgt / ocgt / ic / steam), tower (0/1).
usage: python 20_subtypes.py <eia860_2019_dir> <jrc_dir> <out.parquet>
"""
import sys
from pathlib import Path

import pandas as pd

EIA_TECH = {
    "Natural Gas Fired Combined Cycle": "gas_ccgt", "Natural Gas Fired Combustion Turbine": "gas_ocgt",
    "Natural Gas Internal Combustion Engine": "gas_ic", "Natural Gas Steam Turbine": "gas_steam",
    "Other Natural Gas": "gas_other", "Conventional Hydroelectric": "hydro_conv",
    "Hydroelectric Pumped Storage": "hydro_pumped", "Conventional Steam Coal": "coal_steam",
    "Coal Integrated Gasification Combined Cycle": "coal_igcc", "Landfill Gas": "bio_landfill",
    "Municipal Solid Waste": "bio_msw", "Wood/Wood Waste Biomass": "bio_wood", "Other Waste Biomass": "bio_other",
    "Petroleum Liquids": "oil_liquids", "Petroleum Coke": "coal_petcoke", "Nuclear": "nuclear",
    "Geothermal": "geothermal", "Onshore Wind Turbine": "wind_onshore", "Offshore Wind Turbine": "wind_offshore",
    "Solar Photovoltaic": "solar_pv", "Solar Thermal without Energy Storage": "solar_thermal",
    "Solar Thermal with Energy Storage": "solar_thermal",
}
JRC_TYPE = {
    "Hydro Water Reservoir": "hydro_reservoir", "Hydro Run-of-river and poundage": "hydro_ror",
    "Hydro Pumped Storage": "hydro_pumped", "Fossil Hard coal": "coal_hard", "Fossil Brown coal/Lignite": "coal_lignite",
    "Fossil Gas": "gas_unknown", "Fossil Oil": "oil_liquids", "Fossil Coal-derived gas": "gas_other",
    "Nuclear": "nuclear", "Biomass": "bio_wood", "Waste": "bio_msw", "Wind Onshore": "wind_onshore",
    "Wind Offshore": "wind_offshore", "Solar": "solar_pv", "Geothermal": "geothermal",
}
EIA_COOL = {"RN": "tower_natural", "RF": "tower_mech", "RI": "tower_mech", "HRF": "tower_mech", "HRI": "tower_mech",
            "HT": "tower_mech", "ON": "once_through", "OC": "once_through", "OF": "once_through", "OS": "once_through",
            "RC": "pond", "DC": "dry", "HRC": "pond"}
JRC_COOL = {"Natural Draught Tower": "tower_natural", "Mechanical Draught Tower": "tower_mech",
            "Once-through": "once_through", "Air Cooling": "dry", "No Cooling": "none"}


def dominant(df, key, col, w):
    s = df.dropna(subset=[col]).groupby([key, col])[w].sum().reset_index()
    return s.sort_values(w, ascending=False).drop_duplicates(key).set_index(key)[col]


def main():
    eia, jrc, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

    g = pd.read_excel(eia / "3_1_Generator_Y2019.xlsx", sheet_name="Operable", skiprows=1,
                      usecols=["Plant Code", "Technology", "Nameplate Capacity (MW)"])
    g = g[pd.to_numeric(g["Plant Code"], errors="coerce").notna()]
    g["plant"] = g["Plant Code"].astype(int)
    g["mw"] = pd.to_numeric(g["Nameplate Capacity (MW)"], errors="coerce").fillna(0)
    g["subtype"] = g.Technology.map(EIA_TECH)
    us_sub = dominant(g, "plant", "subtype", "mw")

    c = pd.read_excel(eia / "6_2_EnviroEquip_Y2019.xlsx", sheet_name="Cooling", skiprows=1,
                      usecols=["Plant Code", "Cooling Type 1"])
    c = c[pd.to_numeric(c["Plant Code"], errors="coerce").notna()]
    c["plant"] = c["Plant Code"].astype(int)
    c["cooling"] = c["Cooling Type 1"].astype(str).str.strip().map(EIA_COOL)
    us_cool = c.dropna(subset=["cooling"]).groupby("plant").cooling.agg(lambda s: s.value_counts().index[0])
    us = pd.DataFrame({"subtype": us_sub}).join(us_cool, how="outer")
    us.index = [f"USA{p:07d}" for p in us.index]
    us["subtype_src"] = "eia"

    u = pd.read_csv(jrc / "JRC_OPEN_UNITS.csv", low_memory=False)
    lk = pd.read_csv(jrc / "JRC_OPEN_LINKAGES.csv").dropna(subset=["WRI_id"])[["eic_g", "WRI_id"]]
    u = u.merge(lk, on="eic_g")
    u["subtype"] = u.type_g.map(JRC_TYPE)
    u["cooling"] = u.cooling_type.map(JRC_COOL)
    eu = pd.DataFrame({"subtype": dominant(u, "WRI_id", "subtype", "capacity_g"),
                       "cooling": dominant(u, "WRI_id", "cooling", "capacity_g")})
    eu["subtype_src"] = "jrc"

    d = pd.concat([us, eu])
    d = d[~d.index.duplicated()].rename_axis("gppd_idnr").reset_index()
    d["hydro_class"] = d.subtype.map({"hydro_ror": "ror", "hydro_reservoir": "storage", "hydro_pumped": "pumped"})
    d["gas_class"] = d.subtype.map({"gas_ccgt": "ccgt", "gas_ocgt": "ocgt", "gas_ic": "ic", "gas_steam": "steam"})
    d["tower"] = d.cooling.map(lambda x: None if pd.isna(x) else float(str(x).startswith("tower")))
    d.to_parquet(out, index=False)
    print(d.subtype_src.value_counts().to_dict())
    print(d.groupby("subtype_src").subtype.value_counts().head(40).to_string())
    print(d.groupby("subtype_src").cooling.value_counts().to_string())


if __name__ == "__main__":
    main()
