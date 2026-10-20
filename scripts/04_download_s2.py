"""Bulk-download Sentinel-2 L2A patches for every GPPD plant (resumable, multithreaded).

Per plant: 6 bands (B02 B03 B04 B08 @10 m, B11 B12 @20 m upsampled x2) as uint16 [6, S, S],
from the least-cloudy scene in the date window whose *local* cloud/shadow fraction (SCL) is low.
Plants with reported generation are downloaded first.

usage: python 04_download_s2.py <gppd.csv> <out_dir> [--size 256] [--dates 2019-04-01/2019-10-31]
                                 [--workers 12] [--limit N]
"""
import argparse
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

BANDS_10 = ["B02", "B03", "B04", "B08"]
BANDS_20 = ["B11", "B12"]
BAD_SCL = {0, 1, 3, 8, 9, 10}  # nodata, saturated, cloud shadow, cloud med/high, cirrus
MAX_LOCAL_CLOUD = 0.05
MAX_TRIES = 4

_tls = threading.local()


def catalog():
    if not hasattr(_tls, "cat"):
        _tls.cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace
        )
    return _tls.cat


def read_window(href, lon, lat, size):
    with rasterio.open(href) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        return src.read(1, window=Window(c - size // 2, r - size // 2, size, size), boundless=True, fill_value=0)


def fetch(lat, lon, size, dates):
    items = catalog().search(
        collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=dates, query={"eo:cloud_cover": {"lt": 30}},
    ).item_collection()
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])
    best = None
    for item in items[:MAX_TRIES]:
        scl = read_window(item.assets["SCL"].href, lon, lat, size // 2)
        bad = np.isin(scl, list(BAD_SCL)).mean()
        if best is None or bad < best[1]:
            best = (item, bad)
        if bad <= MAX_LOCAL_CLOUD:
            break
    if best is None:
        return None
    item, bad = best
    arr = [read_window(item.assets[b].href, lon, lat, size) for b in BANDS_10]
    arr += [np.repeat(np.repeat(read_window(item.assets[b].href, lon, lat, size // 2), 2, 0), 2, 1) for b in BANDS_20]
    meta = {"scene": item.id, "date": item.properties["datetime"][:10],
            "scene_cloud": item.properties["eo:cloud_cover"], "local_bad": round(float(bad), 4)}
    return np.stack(arr).astype(np.uint16), meta


def work(row, out_dir, size, dates):
    path = out_dir / f"{row.gppd_idnr}.npy"
    for attempt in range(3):
        try:
            res = fetch(row.latitude, row.longitude, size, dates)
            if res is None:
                return row.gppd_idnr, {"status": "no_scene"}
            arr, meta = res
            np.save(path, arr)
            return row.gppd_idnr, {"status": "ok", **meta}
        except Exception as e:  # network hiccups, expired SAS tokens, etc.
            err = repr(e)[:200]
            time.sleep(2 * (attempt + 1))
    return row.gppd_idnr, {"status": "error", "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dates", default="2019-04-01/2019-10-31")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "index.csv"

    df = pd.read_csv(a.csv, low_memory=False)
    gen = [f"generation_gwh_{y}" for y in range(2013, 2020)]
    df["_labeled"] = df[gen].notna().any(axis=1)
    df = df.sort_values("_labeled", ascending=False)
    done = set()
    if log_path.exists():  # errors are retried; readers should keep the last row per plant
        prev = pd.read_csv(log_path)
        done = set(prev.gppd_idnr[prev.status.isin(["ok", "no_scene"])])
    todo = [r for r in df.itertuples() if r.gppd_idnr not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)

    fields = ["gppd_idnr", "status", "scene", "date", "scene_cloud", "local_bad", "error"]
    new = not log_path.exists()
    t0 = time.time()
    with open(log_path, "a", newline="") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        futs = [ex.submit(work, r, out_dir / "patches", a.size, a.dates) for r in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            pid, meta = fut.result()
            w.writerow({"gppd_idnr": pid, **meta})
            if n % 50 == 0:
                f.flush()
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 3600:.1f} h", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
