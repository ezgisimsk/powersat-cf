"""Image features for every downloaded 2019 patch (labeled or not): frozen SSL4EO embeddings + water metrics.

Embeddings: as in 11_extract_embeddings.py (mean of the 8x8 ResNet-50 map and of its central 4x4, 4096-d),
reduced by PCA fitted on labeled plants in the training set of both split_random and split_geo.
Water (MNDWI = (B03 - B11) / (B03 + B11) > 0): fraction over the full 2.56 km patch and the central 640 m,
area share of the largest connected water body, and number of water bodies -- a reservoir shows up as one
large body, a run-of-river barrage as a narrow river.

usage: python 21_image_features.py <plants.parquet> <s2_dir> <weights.pth> <out.parquet> [--dims 32]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.decomposition import PCA

spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("09_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def water_feats(x):
    g, s1 = x[1].astype(np.float32), x[4].astype(np.float32)
    w = (g - s1) / np.maximum(g + s1, 1) > 0
    c = x.shape[-1] // 2
    lab, n = ndimage.label(w)
    largest = np.bincount(lab.ravel())[1:].max() / w.size if n else 0.0
    return {"water_frac": w.mean(), "water_frac_center": w[c - 32 : c + 32, c - 32 : c + 32].mean(),
            "water_largest": largest, "water_n_bodies": float(n)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants")
    ap.add_argument("s2_dir")
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--dims", type=int, default=32)
    a = ap.parse_args()
    idx = pd.read_csv(Path(a.s2_dir) / "index.csv").drop_duplicates("gppd_idnr", keep="last")
    ids = [p for p in idx.gppd_idnr[idx.status == "ok"] if (Path(a.s2_dir) / "patches" / f"{p}.npy").exists()]
    enc = T.ssl4eo_resnet50(a.weights).cuda().eval()
    feats, water = [], []
    for s in range(0, len(ids), 64):
        xs = np.stack([np.load(Path(a.s2_dir) / "patches" / f"{p}.npy") for p in ids[s : s + 64]])
        water += [water_feats(x) for x in xs]
        x = torch.from_numpy(np.minimum(xs, 32767).astype(np.float32) / 10000.0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = enc.forward_features(x).float()
        feats.append(torch.cat([f.mean((2, 3)), f[:, :, 2:6, 2:6].mean((2, 3))], 1).cpu().numpy())
        if s % 3200 == 0:
            print(s, len(ids), flush=True)
    feats = np.concatenate(feats)
    p = pd.read_parquet(a.plants).set_index("gppd_idnr").reindex(ids)
    fit = ((p.split_random == "train") & (p.split_geo == "train")).values
    z = PCA(a.dims, random_state=0).fit(feats[fit]).transform(feats)
    out = pd.DataFrame(z, columns=[f"emb_{i}" for i in range(a.dims)]).assign(gppd_idnr=ids)
    out = out.join(pd.DataFrame(water))
    out.to_parquet(a.out, index=False)
    print(len(out), "plants;", out[["water_frac", "water_frac_center", "water_largest", "water_n_bodies"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
