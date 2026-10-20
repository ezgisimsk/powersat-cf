"""Frame-level CNN with monthly supervision for thermal plants; plant CF = mean of its frame predictions.

Each monthly Sentinel-2 frame is labelled with the plant's CF in that month (EIA-923, US) or its annual CF
when no monthly label exists. Prediction = sigmoid(logit(natl_cf) + delta(image, tabular, month, cloud)).

Variants
  frame_img    SSL4EO-S12 ResNet-50 frame encoder (6 bands, 64 px crop upsampled to 128) + tabular + context
  frame_noimg  tabular + context only (ablation: what month / cloud / plant metadata explain without pixels)

Reports plant-level annual metrics on thermal test plants and, for US test plants, the median within-plant
Spearman correlation between predicted and reported monthly CF (does the model track activity over time?).

usage: python 17_train_ts.py --variant frame_img --split random --seed 0 --out results/ts_runs/NAME
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("09_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
NUM = T.NUM


def load_frames(a):
    plants = pd.read_parquet(a.plants)
    plants["age_missing"] = plants.age.isna().astype(float)
    plants["age"] = plants.age.fillna(plants.age.median()).clip(-5, 100)
    plants["fuel_id"] = plants.primary_fuel.astype("category").cat.codes
    fr = pd.read_parquet(a.frames)
    m = pd.read_parquet(a.monthly).query("valid")[["gppd_idnr", "year", "month", "cf"]].rename(columns={"cf": "cf_month"})
    fr = fr.merge(m, on=["gppd_idnr", "year", "month"], how="left")
    feats = pd.read_parquet(a.features)[["gppd_idnr", "date", "c_scl_cloud"]]
    fr = fr.merge(feats, on=["gppd_idnr", "date"], how="left")
    cols = ["gppd_idnr", *NUM, "fuel_id", "cf_target", "capacity_mw", "country", "primary_fuel",
            "split_random", "split_geo"]
    fr = fr.merge(plants[cols], on="gppd_idnr")
    fr["y"] = fr.cf_month.fillna(fr.cf_target)
    return fr, int(plants.fuel_id.max()) + 1


class FrameModel(nn.Module):
    def __init__(self, variant, n_fuels, weights, d=256):
        super().__init__()
        self.use_img = variant == "frame_img"
        if self.use_img:
            self.enc = T.ssl4eo_resnet50(weights)
            self.img_proj = nn.Linear(2048, d)
        self.fuel_emb = nn.Embedding(n_fuels, 16)
        self.tab = nn.Sequential(nn.Linear(len(NUM) + 16, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.month_emb = nn.Embedding(13, 16)
        self.ctx = nn.Sequential(nn.Linear(16 + 2, 32), nn.GELU())
        n_in = d * (2 if self.use_img else 1) + 32
        self.head = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, img, x, fuel, month, cloud, prior):
        parts = [self.tab(torch.cat([x, self.fuel_emb(fuel)], 1)),
                 self.ctx(torch.cat([self.month_emb(month), cloud], 1))]
        if self.use_img:
            img = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False)
            parts.insert(0, self.img_proj(self.enc.forward_features(img).mean((2, 3))))
        return torch.sigmoid(self.head(torch.cat(parts, 1)).squeeze(1) + prior)


class FrameBatches:
    def __init__(self, d, images, stats, bs, train, dev):
        self.d, self.images, self.bs, self.train, self.dev = d.reset_index(drop=True), images, bs, train, dev
        self.x = torch.tensor(((d[NUM].values - stats[0]) / stats[1]).astype(np.float32))
        self.fuel = torch.tensor(d.fuel_id.values, dtype=torch.long)
        self.month = torch.tensor(d.month.values, dtype=torch.long)
        self.y = torch.tensor(d.y.values, dtype=torch.float32)
        pr = np.clip(d.natl_cf.values, 0.01, 0.99)
        self.prior = torch.tensor(np.log(pr / (1 - pr)), dtype=torch.float32)
        self.rows = d.row.values
        self.cloud = torch.tensor(np.stack([d.c_scl_cloud.fillna(0).values, d.scene_cloud.values / 100], 1), dtype=torch.float32)

    def __len__(self):
        return len(self.d) // self.bs if self.train else -(-len(self.d) // self.bs)

    def __iter__(self):
        order = torch.randperm(len(self.d)) if self.train else torch.arange(len(self.d))
        for b in range(len(self)):
            i = order[b * self.bs : (b + 1) * self.bs]
            img = None
            if self.images is not None:
                r = self.rows[i.numpy()]
                srt = np.argsort(r)
                buf = torch.from_numpy(np.minimum(self.images[r[srt], :6], 32767).astype(np.int16))
                img = torch.empty_like(buf)
                img[torch.from_numpy(srt)] = buf
                img = img.to(self.dev, non_blocking=True).float() / 10000.0
                if self.train:
                    img = T.augment(img)
            dv = self.dev
            yield (img, self.x[i].to(dv), self.fuel[i].to(dv), self.month[i].to(dv), self.cloud[i].to(dv),
                   self.prior[i].to(dv), self.y[i].to(dv))


@torch.no_grad()
def predict(model, batches):
    model.eval()
    out = []
    for img, x, fuel, month, cloud, prior, _ in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(img, x, fuel, month, cloud, prior).float().cpu())
    return torch.cat(out).numpy()


def plant_level(d, p):
    s = pd.Series(p, index=d.gppd_idnr.values).groupby(level=0).mean()
    t = d.drop_duplicates("gppd_idnr").set_index("gppd_idnr")
    return t.assign(pred=s.reindex(t.index))


def within_plant_spearman(d, p):
    d = d.assign(pred=p)
    d = d[d.cf_month.notna()]
    cs = [g.pred.corr(g.cf_month, method="spearman") for _, g in d.groupby("gppd_idnr")
          if len(g) >= 8 and g.cf_month.std() > 0.05 and g.pred.std() > 0]
    return float(np.nanmedian(cs)) if cs else float("nan"), len(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["frame_img", "frame_noimg"])
    ap.add_argument("--split", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=1e-4)
    ap.add_argument("--plants", default=str(ROOT / "data/processed/plants.parquet"))
    ap.add_argument("--frames", default=str(ROOT / "data/processed/ts_frames64.frames.parquet"))
    ap.add_argument("--mm", default=str(ROOT / "data/processed/ts_frames64.npy"))
    ap.add_argument("--monthly", default=str(ROOT / "data/processed/labels_monthly.parquet"))
    ap.add_argument("--features", default=str(ROOT / "data/processed/ts_frame_features.parquet"))
    ap.add_argument("--weights", default=str(ROOT / "models/pretrained/resnet50_s2_all_moco.pth"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    fr, n_fuels = load_frames(a)
    col = f"split_{a.split}"
    tr, va, te = (fr[fr[col] == s] for s in ("train", "val", "test"))
    images = None
    if a.variant == "frame_img":
        t0 = time.time()
        images = np.load(a.mm)  # ~8 GB, read sequentially once
        print(f"loaded frames {images.shape} in {time.time() - t0:.0f}s", flush=True)
    stats = (tr.drop_duplicates("gppd_idnr")[NUM].mean().values, tr.drop_duplicates("gppd_idnr")[NUM].std().values + 1e-6)
    mk = lambda d, train: FrameBatches(d, images, stats, a.bs, train, dev)
    b_tr, b_va, b_te = mk(tr, True), mk(va, False), mk(te, False)

    model = FrameModel(a.variant, n_fuels, a.weights).to(dev)
    enc = list(model.enc.parameters()) if model.use_img else []
    ids = {id(p) for p in enc}
    groups = [{"params": [p for p in model.parameters() if id(p) not in ids], "lr": a.lr}]
    if enc:
        groups.append({"params": enc, "lr": a.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=a.epochs * len(b_tr), pct_start=0.1)
    best, hist = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for img, x, fuel, month, cloud, prior, y in b_tr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(model(img, x, fuel, month, cloud, prior).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl, n = tl + loss.item() * len(y), n + len(y)
        pv = plant_level(va, predict(model, b_va))
        mae = float((pv.pred - pv.cf_target).abs().mean())
        hist.append({"epoch": ep, "train_loss": tl / n, "val_plant_mae": mae, "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if mae < best:
            best = mae
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    p = predict(model, b_te)
    pl = plant_level(te, p)
    pl[["cf_target", "pred", "natl_cf", "capacity_mw", "country", "primary_fuel"]].to_csv(out / "preds_test_plant.csv")
    te.assign(pred=p)[["gppd_idnr", "date", "cf_month", "y", "pred"]].to_csv(out / "preds_test_frame.csv", index=False)
    rho, n_rho = within_plant_spearman(te, p)
    res = {"plant_MAE": float((pl.pred - pl.cf_target).abs().mean()),
           "plant_R2": float(1 - ((pl.pred - pl.cf_target) ** 2).sum() / ((pl.cf_target - pl.cf_target.mean()) ** 2).sum()),
           "within_plant_spearman_median": rho, "n_plants_spearman": n_rho}
    json.dump({**vars(a), **res, "best_val_plant_mae": best, "history": hist}, open(out / "run.json", "w"), indent=1)
    (out / "best.pt").unlink()
    print("done", out, json.dumps(res))


if __name__ == "__main__":
    main()
