"""Frozen SSL4EO-S12 ResNet-50 embeddings for stacking with tabular GBTs (no labels involved).

Per patch: mean of the 8x8 feature map (whole 2.56 km) and of its central 4x4 cells (~1.3 km),
concatenated (4096-d) and reduced with PCA to --dims components. PCA is fitted only on plants that are
in the training set of both split_random and split_geo, so no test image shapes the projection.

usage: python 11_extract_embeddings.py <plants.parquet> <s2_labeled.npy> <weights.pth> <out.parquet> [--dims 32]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("09_train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants")
    ap.add_argument("mm")
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--dims", type=int, default=32)
    a = ap.parse_args()

    enc = train.ssl4eo_resnet50(a.weights).cuda().eval()
    mm = np.load(a.mm, mmap_mode="r")
    feats = []
    for s in range(0, len(mm), 64):  # sequential reads only
        x = torch.from_numpy(np.minimum(mm[s : s + 64], 32767).astype(np.float32) / 10000.0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = enc.forward_features(x).float()  # B, 2048, 8, 8
        feats.append(torch.cat([f.mean((2, 3)), f[:, :, 2:6, 2:6].mean((2, 3))], 1).cpu().numpy())
        if s % 2048 == 0:
            print(s, len(mm), flush=True)
    feats = np.concatenate(feats)

    ids = pd.read_csv(a.mm.replace(".npy", ".ids.csv")).sort_values("row").gppd_idnr.values
    p = pd.read_parquet(a.plants).set_index("gppd_idnr").loc[ids]
    fit = ((p.split_random == "train") & (p.split_geo == "train")).values
    pca = PCA(a.dims, random_state=0).fit(feats[fit])
    z = pca.transform(feats)
    out = pd.DataFrame(z, columns=[f"emb_{i}" for i in range(a.dims)]).assign(gppd_idnr=ids)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} embeddings, PCA fitted on {fit.sum()}, explained var {pca.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
