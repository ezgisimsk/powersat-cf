"""Fetch a Sentinel-2 L2A patch (least-cloudy scene in a date window) around a plant via Planetary Computer.

usage: python 03_s2_patch.py <gppd.csv> <gppd_idnr> <out_dir> [size_px=256] [dates=2019-04-01/2019-09-30]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

BANDS = ["B02", "B03", "B04", "B08"]  # blue, green, red, nir @ 10 m


def fetch(lat, lon, size=256, dates="2019-04-01/2019-09-30"):
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace
    )
    items = cat.search(
        collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=dates, query={"eo:cloud_cover": {"lt": 20}},
    ).item_collection()
    if not items:
        return None, None
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    out = []
    for b in BANDS:
        with rasterio.open(item.assets[b].href) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            r, c = src.index(xs[0], ys[0])
            out.append(src.read(1, window=Window(c - size // 2, r - size // 2, size, size), boundless=True, fill_value=0))
    return np.stack(out), item


if __name__ == "__main__":
    csv, pid, out_dir = sys.argv[1:4]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    dates = sys.argv[5] if len(sys.argv) > 5 else "2019-04-01/2019-09-30"
    row = pd.read_csv(csv, low_memory=False).set_index("gppd_idnr").loc[pid]
    arr, item = fetch(row.latitude, row.longitude, size, dates)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(out_dir) / f"{pid}.npy", arr)
    print(pid, row["name"], row.primary_fuel, arr.shape, arr.dtype, item.id, item.properties["eo:cloud_cover"])
    import matplotlib.pyplot as plt
    rgb = np.clip(arr[[2, 1, 0]].transpose(1, 2, 0) / 3000, 0, 1)
    plt.imsave(Path(out_dir) / f"{pid}.png", rgb)
