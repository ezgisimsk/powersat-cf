"""Build the public dataset release (Zenodo): tables, data card and Sentinel-2 patch shards.

Output in <out_dir>:
  README.md                      data card (contents, columns, licences, attribution)
  plants.parquet                 one row per GPPD plant: GPPD fields, targets, national prior, splits
  labels_annual.parquet          plant-year capacity factors (GPPD 2013-2019, EIA 2019-2025)
  labels_monthly_us.parquet      US plant-month capacity factors 2017-2019 (EIA-923)
  subtypes.parquet               technology subtype and cooling type (EIA-860, JRC-PPDB-OPEN)
  cems_overpass_us.parquet       US plant gross load at the Sentinel-2 overpass hours 2017-2019 (EPA CEMS)
  s2_2019_index.csv              Sentinel-2 scene id, date and cloud metrics per patch
  s2_2019_patches_partXX.zip     uncompressed zips of <gppd_idnr>.npy patches (uint16, 6 x 256 x 256)

usage: python 29_build_release.py <project_root> <out_dir> [--shard_gb 5]
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

CARD = """# PowerSat-CF: power plants, reported generation and Sentinel-2 imagery

Multimodal benchmark for estimating plant-level capacity factors. It links the 34,936 plants of the Global Power
Plant Database (GPPD v1.3) to reported generation (10,632 labeled plants in 29 countries), national fuel-level capacity
factors, technology subtypes, EPA CEMS gross load at the satellite overpass hour (US), and a 2019 Sentinel-2 L2A
image patch for {n_patches:,} plants.

Code: https://github.com/ezgisimsk/powersat-cf

## Files

| File | Rows | Description |
|---|---|---|
| `plants.parquet` | {n_plants:,} | One row per GPPD plant: original GPPD fields, targets, features, national prior and evaluation splits |
| `labels_annual.parquet` | {n_annual:,} | Plant-year capacity factors (`source` = `gppd` for 2013-2019 or `eia` for US 2019-2025) |
| `labels_monthly_us.parquet` | {n_monthly:,} | US plant-month capacity factors 2017-2019 from EIA-923 / EIA-860 |
| `subtypes.parquet` | {n_sub:,} | Technology subtype and cooling type (EIA-860 2019 for US, JRC-PPDB-OPEN v1.0 for Europe) |
| `cems_overpass_us.parquet` | {n_cems:,} | US plant-day summed unit gross load (MW) in hours 9-12 local standard time, 2017-2019 |
| `s2_2019_index.csv` | {n_idx:,} | Per-plant Sentinel-2 scene id, acquisition date, scene cloud cover, local bad-pixel share |
| `s2_2019_patches_partXX.zip` | {n_patches:,} files | `<gppd_idnr>.npy`: uint16 array (6, 256, 256), bands B02, B03, B04, B08 (10 m) and B11, B12 (20 m, nearest x2), reflectance x 10,000, centred on the GPPD coordinates (2.56 km) |

## Key columns of `plants.parquet`

- `gppd_idnr`: GPPD plant id (US plants: `USA` + EIA plant code). GPPD columns are kept unchanged.
- `cf_target`: mean valid capacity factor 2017-2019 (main target; non-US plants mostly 2017-2018). `n_target_years`: years averaged.
- `cf_2019`: capacity factor in 2019 (US). `cf_future`: mean capacity factor 2023-2025 (US, EIA). `gen_gwh_target`: `cf_target` x capacity x 8.76.
- `natl_cf`: national capacity factor of the plant's fuel (Ember, 2017-2019 average; `natl_cf_level` = country / region / global fallback).
- `wri_est_cf_2017`: WRI's published 2017 generation estimate converted to a capacity factor.
- `log_capacity`, `age` (2018 - commissioning year), `n_other_fuels`, `is_us`, `ember_fuel`.
- Splits (labeled plants only): `split_random` (train/val/test 70/15/15), `split_geo` (US train/val, non-US test), `fold_spatial` (0-4, 2-degree cells; -1 = unlabeled).

Capacity factor = generation / (capacity x 8,760 h); labels outside [0, 1] are flagged `valid = False` in the label tables.
In `cems_overpass_us.parquet`: `ref_load` = plant's maximum daytime load 2017-2019, `inst_util` = `load_h10` / `ref_load`, `on` = `inst_util` > 0.05.

## Sources, licences and attribution

This dataset is released under **CC BY 4.0**. It contains or is derived from:
- Global Power Plant Database v1.3, World Resources Institute (CC BY 4.0).
- Ember Yearly Electricity Data (CC BY 4.0).
- U.S. Energy Information Administration, Form EIA-923 and EIA-860 (public domain).
- U.S. EPA Clean Air Markets Program Data, hourly emissions (public domain), via the Catalyst Cooperative PUDL raw archive.
- JRC Open Power Plants Database (JRC-PPDB-OPEN) v1.0 (CC BY 4.0).
- Contains modified Copernicus Sentinel data 2019, processed by ESA and accessed through Microsoft Planetary Computer.

## Known limitations

GPPD coordinates can be off by up to ~1 km; non-US labels are mostly 2017-2018 averages while images are from 2019;
9 plants have no suitable scene; GPPD v1.3 is frozen (2021).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--shard_gb", type=float, default=5.0)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = root / "data" / "processed"

    tables = {"plants.parquet": "plants.parquet", "labels.parquet": "labels_annual.parquet",
              "labels_monthly.parquet": "labels_monthly_us.parquet", "subtypes.parquet": "subtypes.parquet",
              "cems_overpass.parquet": "cems_overpass_us.parquet"}
    for src, dst in tables.items():
        shutil.copy2(proc / src, out / dst)
    idx = pd.read_csv(root / "data/raw/s2_2019/index.csv").drop_duplicates("gppd_idnr", keep="last")
    idx = idx[idx.status == "ok"].drop(columns=["status", "error"])
    idx.to_csv(out / "s2_2019_index.csv", index=False)

    patches = sorted((root / "data/raw/s2_2019/patches").glob("*.npy"))
    ok_ids = set(idx.gppd_idnr)
    patches = [p for p in patches if p.stem in ok_ids]
    limit, part, size, zf = a.shard_gb * 1e9, 0, 0, None
    for p in patches:
        if zf is None or size > limit:
            if zf:
                zf.close()
            part += 1
            zf = zipfile.ZipFile(out / f"s2_2019_patches_part{part:02d}.zip", "w", zipfile.ZIP_STORED)
            size = 0
            print("part", part, flush=True)
        zf.write(p, arcname=p.name)
        size += p.stat().st_size
    zf.close()

    counts = dict(n_plants=len(pd.read_parquet(out / "plants.parquet")),
                  n_annual=len(pd.read_parquet(out / "labels_annual.parquet")),
                  n_monthly=len(pd.read_parquet(out / "labels_monthly_us.parquet")),
                  n_sub=len(pd.read_parquet(out / "subtypes.parquet")),
                  n_cems=len(pd.read_parquet(out / "cems_overpass_us.parquet")),
                  n_idx=len(idx), n_patches=len(patches))
    (out / "README.md").write_text(CARD.format(**counts), encoding="utf-8")
    print(counts, "shards:", part)


if __name__ == "__main__":
    main()
