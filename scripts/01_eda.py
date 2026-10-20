"""Exploratory analysis of the Global Power Plant Database v1.3."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
print("shape:", df.shape)
print("\ncolumns:", list(df.columns))
print("\nfuel counts:\n", df.primary_fuel.value_counts())
print("\ncapacity by fuel (GW):\n", (df.groupby("primary_fuel").capacity_mw.sum() / 1000).sort_values(ascending=False).round(1))
print("\ntop countries:\n", df.country_long.value_counts().head(15))

gen = [f"generation_gwh_{y}" for y in range(2013, 2020)]
est = [c for c in df.columns if c.startswith("estimated_generation_gwh")]
print("\nreported generation non-null per year:\n", df[gen].notna().sum())
print("\nestimated generation non-null:\n", df[est].notna().sum())
print("\nyear_of_capacity_data:\n", df.year_of_capacity_data.value_counts().head())
print("\ngeneration_data_source:\n", df.generation_data_source.value_counts().head(10))

n_years = df[gen].notna().sum(axis=1)
print("\nplants by # reported gen years:\n", n_years.value_counts().sort_index())
full = df[n_years >= 5]
print("\nplants with >=5 years gen, by country:\n", full.country_long.value_counts().head(10))
print("\n...by fuel:\n", full.primary_fuel.value_counts())

# capacity factor where computable
for y in (2017, 2018, 2019):
    cf = df[f"generation_gwh_{y}"] * 1000 / (df.capacity_mw * 8760)
    print(f"\nCF {y}:", cf.describe().round(3).to_dict())
cf = df["generation_gwh_2019"] * 1000 / (df.capacity_mw * 8760)
print("\nmedian CF 2019 by fuel:\n", cf.groupby(df.primary_fuel).median().round(3).sort_values())
print("\ncommissioning_year non-null:", df.commissioning_year.notna().sum(), " owner non-null:", df.owner.notna().sum())
print("\ncapacity_mw describe:\n", df.capacity_mw.describe())
