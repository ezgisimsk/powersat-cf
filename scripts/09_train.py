"""Train image / tabular / fusion models for plant capacity-factor (CF) estimation.

Variants
  tab            tabular MLP only (same features as gbt_wri; no coordinates)
  img            SSL4EO-S12 ResNet-50 + attention pooling -> CF
  fusion_concat  late fusion: [tabular emb, mean-pooled image emb] -> CF
  fusion_xattn   tabular embedding queries image tokens (cross-attention) -> CF
Auxiliary task (--aux > 0): primary-fuel classification from the *image* embedding only
(fuel is a tabular input, so the aux head must not see the tabular branch).

usage: python 09_train.py --variant fusion_xattn --split random --aux 0.3 --seed 0 --out results/runs/NAME
       --split: random | geo | spatial0..spatial4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
S2_IDX = [1, 2, 3, 7, 11, 12]  # B02 B03 B04 B08 B11 B12 within the 13-band SSL4EO order
NUM = ["log_capacity", "age", "age_missing", "n_other_fuels", "natl_cf"]


# ----------------------------------------------------------------------------- data
def load_table(plants_pq, ids_csv):
    df = pd.read_parquet(plants_pq)
    df = df[df.split_random.notna()].reset_index(drop=True)
    rows = pd.read_csv(ids_csv).set_index("gppd_idnr").row
    df["img_row"] = df.gppd_idnr.map(rows).fillna(-1).astype(int)
    df["age_missing"] = df.age.isna().astype(float)
    df["age"] = df.age.fillna(df.age.median()).clip(-5, 100)
    df["fuel_id"] = df.primary_fuel.astype("category").cat.codes
    return df


def make_split(df, split, seed):
    df = df[df.img_row >= 0]  # image and fusion models need a patch; baselines are re-scored on the same subset
    if split in ("random", "geo"):
        col = f"split_{split}"
        return (df[df[col] == s] for s in ("train", "val", "test"))
    k = int(split.replace("spatial", ""))
    trv, te = df[df.fold_spatial != k], df[df.fold_spatial == k]
    va = trv.sample(frac=0.1, random_state=seed)
    return trv.drop(va.index), va, te


def load_images(mm_path, rows):
    """Read the needed patches sequentially into RAM once (random memmap reads on HDD are far too slow).
    Returns (int16 array [n, C, H, W], mapping img_row -> position)."""
    mm = np.load(mm_path, mmap_mode="r")
    rows = np.sort(np.unique(rows))
    arr = np.empty((len(rows), *mm.shape[1:]), np.int16)
    for s in range(0, len(rows), 512):
        r = rows[s : s + 512]
        chunk = mm[r[0] : r[-1] + 1][r - r[0]]  # contiguous read, then select
        arr[s : s + len(r)] = np.minimum(chunk, 32767).astype(np.int16)
    return arr, {int(r): i for i, r in enumerate(rows)}


class Batches:
    """Minimal in-memory batch iterator; augmentation (flip / rot90 per sample) runs on the GPU."""

    def __init__(self, d, images, pos, stats, bs, train, dev, residual=False):
        self.d, self.images, self.bs, self.train, self.dev = d.reset_index(drop=True), images, bs, train, dev
        self.x = torch.tensor(((d[NUM].values - stats[0]) / stats[1]).astype(np.float32))
        self.fuel = torch.tensor(d.fuel_id.values, dtype=torch.long)
        self.y = torch.tensor(d.cf_target.values, dtype=torch.float32)
        # residual mode: the model starts from logit(national CF) and learns the deviation
        prior = np.clip(d.natl_cf.values, 0.01, 0.99)
        self.prior = torch.tensor(np.log(prior / (1 - prior)) if residual else np.zeros(len(d)), dtype=torch.float32)
        self.pos = None if images is None else np.array([pos[r] for r in d.img_row])

    def __len__(self):
        return len(self.d) // self.bs if self.train else -(-len(self.d) // self.bs)

    def __iter__(self):
        order = torch.randperm(len(self.d)) if self.train else torch.arange(len(self.d))
        for b in range(len(self)):
            i = order[b * self.bs : (b + 1) * self.bs]
            img = None
            if self.images is not None:
                p = self.pos[i.numpy()]
                srt = np.argsort(p)  # sorted fancy-indexing is faster; undo afterwards
                buf = torch.from_numpy(self.images[p[srt]])
                img = torch.empty_like(buf)
                img[torch.from_numpy(srt)] = buf
                img = img.to(self.dev, non_blocking=True).float() / 10000.0
                if self.train:
                    img = augment(img)
            yield img, self.x[i].to(self.dev), self.fuel[i].to(self.dev), self.prior[i].to(self.dev), self.y[i].to(self.dev)


def augment(img):
    flip = torch.rand(img.size(0), device=img.device) < 0.5
    img = torch.where(flip[:, None, None, None], img.flip(3), img)
    k = torch.randint(4, (img.size(0),), device=img.device)
    out = img.clone()
    for r in range(1, 4):
        m = k == r
        if m.any():
            out[m] = torch.rot90(img[m], r, (2, 3))
    return out


# ----------------------------------------------------------------------------- model
def ssl4eo_resnet50(weights):
    m = timm.create_model("resnet50", in_chans=13, num_classes=0, global_pool="")
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    w = m.conv1.weight.data[:, S2_IDX] * (13 / len(S2_IDX))
    m.conv1 = nn.Conv2d(len(S2_IDX), 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    return m


class AttnPool(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, tokens, query=None):
        q = self.q.expand(tokens.size(0), -1, -1) if query is None else query.unsqueeze(1) + self.q
        out, w = self.attn(q, tokens, tokens)
        return out.squeeze(1), w


class CFModel(nn.Module):
    def __init__(self, variant, n_fuels, weights, d=256):
        super().__init__()
        self.variant = variant
        self.use_img, self.use_tab = variant != "tab", variant != "img"
        if self.use_img:
            self.enc = ssl4eo_resnet50(weights)
            self.proj = nn.Conv2d(2048, d, 1)
            self.pos = nn.Parameter(torch.zeros(1, 64, d))
            self.pool = AttnPool(d)
            self.aux = nn.Linear(d, n_fuels)
        if self.use_tab:
            self.fuel_emb = nn.Embedding(n_fuels, 16)
            self.tab = nn.Sequential(nn.Linear(len(NUM) + 16, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        if variant == "fusion_xattn":
            self.xattn = AttnPool(d)
        n_in = {"tab": 1, "img": 1, "fusion_concat": 2, "fusion_xattn": 3}[variant]
        self.head = nn.Sequential(nn.LayerNorm(n_in * d), nn.Linear(n_in * d, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, img, x, fuel, prior):
        parts, aux = [], None
        if self.use_tab:
            z_tab = self.tab(torch.cat([x, self.fuel_emb(fuel)], 1))
            parts.append(z_tab)
        if self.use_img:
            fmap = self.proj(self.enc.forward_features(img))  # B, d, h, w
            tokens = fmap.flatten(2).transpose(1, 2)
            tokens = tokens + self.pos[:, : tokens.size(1)]
            z_img, _ = self.pool(tokens)
            aux = self.aux(z_img)
            if self.variant == "img":
                parts.append(z_img)
            elif self.variant == "fusion_concat":
                parts.append(tokens.mean(1))
            else:
                z_x, _ = self.xattn(tokens, query=z_tab)
                parts += [z_x, z_img]
        cf = torch.sigmoid(self.head(torch.cat(parts, 1)).squeeze(1) + prior)
        return cf, aux


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for img, x, fuel, prior, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cf, _ = model(img, x, fuel, prior)
        out.append(cf.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["tab", "img", "fusion_concat", "fusion_xattn"])
    ap.add_argument("--split", default="random")
    ap.add_argument("--aux", type=float, default=0.3)
    ap.add_argument("--residual", action="store_true", help="predict deviation from national CF (logit space)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=100)  # OneCycle improves late; keep best of all epochs
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, help="debug: subsample train/val/test")
    ap.add_argument("--plants", default=str(ROOT / "data/processed/plants.parquet"))
    ap.add_argument("--mm", default=str(ROOT / "data/processed/s2_labeled.npy"))
    ap.add_argument("--weights", default=str(ROOT / "models/pretrained/resnet50_s2_all_moco.pth"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    df = load_table(a.plants, a.mm.replace(".npy", ".ids.csv"))
    n_fuels = int(df.fuel_id.max()) + 1
    tr, va, te = make_split(df, a.split, a.seed)
    if a.limit:
        tr, va, te = tr.sample(min(a.limit, len(tr)), random_state=0), va.head(a.limit // 4), te.head(a.limit // 4)
    stats = (tr[NUM].mean().values, tr[NUM].std().values + 1e-6)
    images, pos = None, None
    if a.variant != "tab":
        t0 = time.time()
        images, pos = load_images(a.mm, np.concatenate([tr.img_row, va.img_row, te.img_row]))
        print(f"loaded {len(images)} patches in {time.time() - t0:.0f}s", flush=True)
    mk = lambda d, train: Batches(d, images, pos, stats, a.bs, train, dev, a.residual)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    model = CFModel(a.variant, n_fuels, a.weights).to(dev)
    enc_params = list(model.enc.parameters()) if model.use_img else []
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": other, "lr": a.lr}] + ([{"params": enc_params, "lr": a.lr_enc}] if enc_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=a.epochs * len(dl_tr), pct_start=0.1)

    best, best_ep, hist = 1e9, -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for img, x, fuel, prior, y in dl_tr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cf, aux = model(img, x, fuel, prior)
                loss = F.mse_loss(cf.float(), y)
                if aux is not None and a.aux > 0:
                    loss = loss + a.aux * F.cross_entropy(aux.float(), fuel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl, n = tl + loss.item() * len(y), n + len(y)
        pv = predict(model, dl_va, dev)
        mae = float(np.abs(pv - va.cf_target.values).mean())
        hist.append({"epoch": ep, "train_loss": tl / n, "val_mae": mae, "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if mae < best:
            best, best_ep = mae, ep
            torch.save(model.state_dict(), out / "best.pt")
        elif ep - best_ep >= a.patience:
            break

    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    for name, d, dl in (("val", va, dl_va), ("test", te, dl_te)):
        pd.DataFrame({"gppd_idnr": d.gppd_idnr.values, "pred": predict(model, dl, dev)}).to_csv(out / f"preds_{name}.csv", index=False)
    json.dump({**vars(a), "best_epoch": best_ep, "best_val_mae": best, "history": hist}, open(out / "run.json", "w"), indent=1)
    (out / "best.pt").unlink()  # keep disk usage low; predictions are what we evaluate
    print("done", out, "best val MAE", round(best, 4), "epoch", best_ep)


if __name__ == "__main__":
    main()
