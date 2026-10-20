# PowerSat-CF

Code for **"Can satellites tell how much a power plant generates? A multimodal benchmark and geographic-transfer analysis of plant-level capacity factors"** (İrem Ezgi Bayar, Ömer Karal; manuscript in preparation).

The pipeline links the [Global Power Plant Database](https://datasets.wri.org/dataset/globalpowerplantdatabase) (v1.3) to reported generation (EIA-923/860, CEA, Australian CER, JRC-PPDB-OPEN), national fuel-level capacity factors (Ember), technology subtypes (EIA-860, JRC-PPDB-OPEN), Sentinel-2 L2A imagery (Microsoft Planetary Computer) and EPA CEMS hourly gross load. It then benchmarks tabular, image, fusion, residual-prior, temporal and activity-detection models under random, spatial-block and geographic-transfer protocols.

The dataset is not part of this repository; it will be released separately.

## Requirements

Python 3.13, a CUDA GPU (developed on an 8 GB RTX 4060 Ti) and ~17 GB RAM.

```bash
pip install -r requirements.txt
```

Pretrained encoder: SSL4EO-S12 ResNet-50 MoCo weights (`resnet50_sentinel2_all_moco`, torchgeo on Hugging Face, CC BY 4.0), saved as `models/pretrained/resnet50_s2_all_moco.pth`.

## Pipeline

Scripts are numbered in execution order. Each script's docstring documents its inputs, outputs and usage.

| Step | Scripts | Purpose |
|---|---|---|
| Exploration | `01_eda.py`, `02_wri_baseline.py` | GPPD statistics; WRI published estimates vs reported capacity factors |
| Imagery | `03_s2_patch.py`, `04_download_s2.py` | Sentinel-2 2019 patch (256 px, 6 bands) for every plant |
| Labels & features | `05_build_labels.py`, `06_build_dataset.py`, `13_monthly_labels.py` | Annual/monthly capacity factors, national priors, splits (random, spatial block, geographic) |
| Tabular baselines | `07_baselines.py` | National prior, fuel means, LightGBM (incl. residual and embedding variants) |
| Neural models | `08_pack_patches.py`, `09_train.py`, `10_compare.py`, `11_extract_embeddings.py`, `run_experiments.sh`, `28_spatial_nn.py` | SSL4EO ResNet-50 image, tabular and fusion models; residual prior; auxiliary fuel task |
| Monthly series | `12_download_s2_timeseries.py`, `14_ts_features.py`, `15_ts_gbt.py`, `16_pack_ts.py`, `17_train_ts.py`, `run_ts_experiments.sh` | 2017–2019 monthly frames of thermal plants; plume features; frame-level CNN |
| Country analysis | `18_country_analysis.py`, `19_country_figures.py` | Leave-one-country-out and out-of-fold errors; fuel-mix-adjusted maps |
| Technology subtypes | `20_subtypes.py`, `21_image_features.py`, `22_subtype_experiments.py` | EIA-860/JRC subtypes and cooling; image subtype classifiers |
| Activity detection | `23_cems_overpass.py`, `24_train_onoff.py`, `25_onrate_gbt.py` | Overpass-hour labels from EPA CEMS; on/off CNN; satellite on-rate as a feature |
| Figures | `26_paper_figures.py`, `27_paper_figures2.py` | Paper figures |

## Data sources and licences

GPPD (CC BY 4.0), Ember yearly electricity data (CC BY 4.0), EIA-923/860 and EPA CEMS (US public domain; CEMS via the PUDL raw archive on Zenodo), JRC-PPDB-OPEN (CC BY 4.0), Copernicus Sentinel-2 data (free and open), Natural Earth (public domain).
