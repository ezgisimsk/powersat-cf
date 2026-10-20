"""Monthly Sentinel-2 L2A time series (2017-2019) around labeled thermal plants, for activity (plume / SWIR) signals.

Per plant and calendar month: the scene with the lowest *scene-level* cloud cover (< --max_cloud).
Scenes are deliberately NOT filtered by local cloud: water-vapour plumes look like cloud to the SCL
classifier, so local filtering would drop exactly the images where the plant is running. SCL is stored
as the 7th channel instead so models can learn the difference.

Output per plant: <out>/series/<gppd_idnr>.npz with
  x      uint16 [T, 7, S, S]  (B02 B03 B04 B08 @10 m; B11 B12 SCL @20 m upsampled x2)
  dates  str [T], cloud float [T]
and one row per plant in <out>/index.csv (status, n_scenes). Resumable. Uses single-threaded worker
processes (the GDAL/HTTP memory growth seen earlier only occurred with many threads per process).
Do not use max_tasks_per_child: on Windows it deadlocks once workers hit the limit; run batches with
--limit from an outer loop instead.

usage: python 12_download_s2_timeseries.py <plants.parquet> <out_dir> [--size 96] [--workers 10] [--limit N]
"""
import argparse
import csv
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

THERMAL = {"Coal", "Gas", "Oil", "Nuclear", "Biomass", "Waste", "Petcoke", "Cogeneration", "Geothermal"}
BANDS_10 = ["B02", "B03", "B04", "B08"]
BANDS_20 = ["B11", "B12", "SCL"]
_cat = None


def catalog():
    global _cat
    if _cat is None:
        import planetary_computer
        import pystac_client
        _cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                         modifier=planetary_computer.sign_inplace)
    return _cat


def read_window(href, lon, lat, size):
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    with rasterio.open(href) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        return src.read(1, window=Window(c - size // 2, r - size // 2, size, size), boundless=True, fill_value=0)


def fetch_series(pid, lat, lon, size, dates, max_cloud):
    items = catalog().search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                             datetime=dates, query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    by_month = defaultdict(list)
    for it in items:
        by_month[it.properties["datetime"][:7]].append(it)
    chosen = [min(v, key=lambda i: i.properties["eo:cloud_cover"]) for _, v in sorted(by_month.items())]
    xs, ds, cl = [], [], []
    for it in chosen:
        try:
            arr = [read_window(it.assets[b].href, lon, lat, size) for b in BANDS_10]
            arr += [np.repeat(np.repeat(read_window(it.assets[b].href, lon, lat, size // 2), 2, 0), 2, 1) for b in BANDS_20]
        except Exception:  # a single broken asset should not lose the whole plant
            continue
        a = np.stack(arr).astype(np.uint16)
        if (a[:4] == 0).mean() > 0.5:  # outside the swath / nodata
            continue
        xs.append(a)
        ds.append(it.properties["datetime"][:10])
        cl.append(it.properties["eo:cloud_cover"])
    return xs, ds, cl


def work(pid, lat, lon, out_dir, size, dates, max_cloud):
    err = ""
    for attempt in range(3):
        try:
            xs, ds, cl = fetch_series(pid, lat, lon, size, dates, max_cloud)
            if not xs:
                return pid, "no_scene", 0, ""
            np.savez(out_dir / f"{pid}.npz", x=np.stack(xs), dates=np.array(ds), cloud=np.array(cl, np.float32))
            return pid, "ok", len(xs), ""
        except Exception as e:
            err = repr(e)[:200]
            time.sleep(5 * (attempt + 1))
    return pid, "error", 0, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--dates", default="2017-01-01/2019-12-31")
    ap.add_argument("--max_cloud", type=float, default=40)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / "series").mkdir(parents=True, exist_ok=True)
    log_path = out / "index.csv"
    p = pd.read_parquet(a.plants)
    p = p[p.split_random.notna() & p.primary_fuel.isin(THERMAL)]
    p = p.sort_values(["is_us", "capacity_mw"], ascending=[True, False])  # geo-test plants first, then large plants
    done = set()
    if log_path.exists():
        prev = pd.read_csv(log_path)
        done = set(prev.gppd_idnr[prev.status.isin(["ok", "no_scene"])])
    todo = [r for r in p.itertuples() if r.gppd_idnr not in done][: a.limit]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)

    new = not log_path.exists()
    t0 = time.time()
    with open(log_path, "a", newline="") as f, \
            ProcessPoolExecutor(a.workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["gppd_idnr", "status", "n_scenes", "error"])
        futs = [ex.submit(work, r.gppd_idnr, r.latitude, r.longitude, out / "series", a.size, a.dates, a.max_cloud)
                for r in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            w.writerow(fut.result())
            f.flush()
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)}  {rate * 3600:.0f} plants/h  eta {(len(todo) - n) / rate / 3600:.1f} h", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
