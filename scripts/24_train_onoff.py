"""Is the plant running? Frame-level on/off detection from monthly Sentinel-2 frames, supervised with EPA CEMS
gross load at the overpass hour (US), then transferred to India to flag idle plants.

Model: SSL4EO-S12 ResNet-50 on the central 64 px (upsampled x2) + month + cloud context. Deliberately NO plant
metadata, so the output reflects visual evidence, not a prior of how plants usually run. Two heads:
on/off (BCE) and instantaneous utilisation at the overpass hour (MSE).
Train/val/test = US plants of split_random train/val/test that have CEMS labels.

Outputs (results/onoff/<name>/): frame predictions for US test and all non-US thermal plants, metrics.json.
Plant on-rate = mean P(on) over frames with scene cloud < 10 %.

Ablation (--inputs): full = image + month + cloud; image = pixels only; context = month + cloud only;
month / cloud = that context alone. Disabled inputs are zeroed (the encoder is skipped without pixels).

Backbone (--backbone): resnet50 (default) or vit_s = ViT-S/16 with the same SSL4EO-S12 MoCo pretraining;
for the ViT the 64 px crop is resized to 224 px and patch tokens are averaged.

Multi-region mode (--eu_group k): ENTSO-E overpass-hour labels for European plants are added. European countries
with labels are split into 5 groups balanced by labelled plants (greedy, largest country first). Training uses all
US labelled frames plus the European frames outside group k; predictions are written for every non-US frame whose
country is not in training (so each labelled European plant is predicted by a model that never saw its country).

usage: python 24_train_onoff.py --out results/onoff/NAME [--epochs 8] [--seed 0] [--inputs full] [--backbone resnet50]
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
import timm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("09_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
CLEAR = 10


INPUTS = {"full": (1, 1, 1), "image": (1, 0, 0), "context": (0, 1, 1), "month": (0, 1, 0), "cloud": (0, 0, 1)}


def ssl4eo_vit_s(weights):
    """ViT-S/16 pretrained with MoCo on SSL4EO-S12 (13 bands), patch embedding restricted to our 6 bands."""
    m = timm.create_model("vit_small_patch16_224", in_chans=13, num_classes=0)
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    old = m.patch_embed.proj
    new = nn.Conv2d(len(T.S2_IDX), old.out_channels, old.kernel_size, old.stride)
    new.weight.data = old.weight.data[:, T.S2_IDX] * (13 / len(T.S2_IDX))
    new.bias.data = old.bias.data.clone()
    m.patch_embed.proj = new
    return m


class OnOffModel(nn.Module):
    def __init__(self, weights, d=256, inputs="full", backbone="resnet50"):
        super().__init__()
        self.use_img, self.use_month, self.use_cloud = INPUTS[inputs]
        self.backbone = backbone
        if backbone == "vit_s":
            self.enc = ssl4eo_vit_s(weights)
            self.img = nn.Linear(self.enc.embed_dim, d)
        else:
            self.enc = T.ssl4eo_resnet50(weights)
            self.img = nn.Linear(2048, d)
        self.month = nn.Embedding(13, 16)
        self.ctx = nn.Sequential(nn.Linear(16 + 2, 32), nn.GELU())
        self.trunk = nn.Sequential(nn.LayerNorm(d + 32), nn.Linear(d + 32, d), nn.GELU(), nn.Dropout(0.2))
        self.on_head, self.util_head = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, img, month, cloud):
        if self.use_img and self.backbone == "vit_s":
            img = F.interpolate(img, size=224, mode="bilinear", align_corners=False)
            tokens = self.enc.forward_features(img)  # B, 1 + 196, embed_dim
            z_img = self.img(tokens[:, self.enc.num_prefix_tokens:].mean(1))
        elif self.use_img:
            img = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False)
            z_img = self.img(self.enc.forward_features(img).mean((2, 3)))
        else:
            z_img = torch.zeros(month.size(0), self.img.out_features, device=month.device)
        month = month * self.use_month  # index 0 is never a real month
        cloud = cloud * self.use_cloud
        z = torch.cat([z_img, self.ctx(torch.cat([self.month(month), cloud], 1))], 1)
        h = self.trunk(z)
        return self.on_head(h).squeeze(1), torch.sigmoid(self.util_head(h).squeeze(1))


def batches(d, images, bs, train, dev):
    rows = d.row.values
    month = torch.tensor(d.month.values, dtype=torch.long)
    cloud = torch.tensor(np.stack([d.c_scl_cloud.fillna(0).values, d.scene_cloud.values / 100], 1), dtype=torch.float32)
    on = torch.tensor(d.on.fillna(-1).values, dtype=torch.float32)
    util = torch.tensor(d.inst_util.fillna(-1).values, dtype=torch.float32)
    order = torch.randperm(len(d)) if train else torch.arange(len(d))
    n = len(d) // bs if train else -(-len(d) // bs)
    for b in range(n):
        i = order[b * bs : (b + 1) * bs]
        r = rows[i.numpy()]
        srt = np.argsort(r)
        buf = torch.from_numpy(np.minimum(images[r[srt], :6], 32767).astype(np.int16))
        img = torch.empty_like(buf)
        img[torch.from_numpy(srt)] = buf
        img = img.to(dev, non_blocking=True).float() / 10000.0
        if train:
            img = T.augment(img)
        yield img, month[i].to(dev), cloud[i].to(dev), on[i].to(dev), util[i].to(dev)


@torch.no_grad()
def predict(model, d, images, dev):
    model.eval()
    ps, us = [], []
    for img, month, cloud, _, _ in batches(d, images, 256, False, dev):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, u = model(img, month, cloud)
        ps.append(torch.sigmoid(lo.float()).cpu())
        us.append(u.float().cpu())
    return torch.cat(ps).numpy(), torch.cat(us).numpy()


def frame_metrics(d, prefix):
    m = d.on.notna()
    out = {}
    for name, sel in (("all", m), ("clear", m & (d.scene_cloud < CLEAR))):
        x = d[sel]
        if x.on.nunique() == 2:
            out[f"{prefix}_auroc_{name}"] = roc_auc_score(x.on, x.p_on)
        out[f"{prefix}_util_spearman_{name}"] = x.inst_util.corr(x.util_pred, method="spearman")
        out[f"{prefix}_n_{name}"] = int(len(x))
    return out


def within_plant(d, n_boot=2000):
    """AUROC of on vs off frames inside each plant (plants with >= 3 of each), median + bootstrap CI."""
    d = d[d.on.notna()]
    aucs, excluded = [], 0
    for _, g in d.groupby("gppd_idnr"):
        if (g.on == 1).sum() >= 3 and (g.on == 0).sum() >= 3:
            aucs.append(roc_auc_score(g.on, g.p_on))
        else:
            excluded += 1
    aucs = np.array(aucs)
    rng = np.random.default_rng(0)
    meds = [np.median(rng.choice(aucs, len(aucs))) for _ in range(n_boot)]
    return {"within_plant_auroc_median": float(np.median(aucs)), "within_plant_auroc_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "within_plant_n_plants": int(len(aucs)), "within_plant_n_excluded": int(excluded)}


def eu_groups(labels, n=5):
    """Greedy balance of labelled countries into n groups by number of labelled plants."""
    counts = labels.groupby("country").gppd_idnr.nunique().sort_values(ascending=False)
    groups, size = [[] for _ in range(n)], [0] * n
    for c, k in counts.items():
        j = int(np.argmin(size))
        groups[j].append(c)
        size[j] += k
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--plants", default=str(ROOT / "data/processed/plants.parquet"))
    ap.add_argument("--frames", default=str(ROOT / "data/processed/ts_frames64.frames.parquet"))
    ap.add_argument("--mm", default=str(ROOT / "data/processed/ts_frames64.npy"))
    ap.add_argument("--features", default=str(ROOT / "data/processed/ts_frame_features.parquet"))
    ap.add_argument("--cems", default=str(ROOT / "data/processed/cems_overpass.parquet"))
    ap.add_argument("--subtypes", default=str(ROOT / "data/processed/subtypes.parquet"))
    ap.add_argument("--weights", default=str(ROOT / "models/pretrained/resnet50_s2_all_moco.pth"))
    ap.add_argument("--inputs", default="full", choices=list(INPUTS))
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_s"])
    ap.add_argument("--vit_weights", default=str(ROOT / "models/pretrained/vit_s16_s2_all_moco.pth"))
    ap.add_argument("--fold", type=int, help="k-fold mode: train on US plants outside fold k (5 folds by plant), predict fold k + non-US")
    ap.add_argument("--state_folds", action="store_true", help="with --fold: folds are groups of US states (balanced by plants), not random plants")
    ap.add_argument("--eia860", default=str(ROOT / "data/raw/eia/eia8602019/2___Plant_Y2019.xlsx"))
    ap.add_argument("--eu_group", type=int, help="multi-region mode: hold out European country group k (0-4)")
    ap.add_argument("--no_eu_train", action="store_true", help="control: same US data as multi-region mode, no European training frames")
    ap.add_argument("--entsoe", default=str(ROOT / "data/processed/entsoe_overpass.parquet"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    plants = pd.read_parquet(a.plants)[["gppd_idnr", "country", "primary_fuel", "capacity_mw", "cf_target", "split_random", "is_us"]]
    fr = pd.read_parquet(a.frames)
    fr = fr.merge(pd.read_parquet(a.features)[["gppd_idnr", "date", "c_scl_cloud"]], on=["gppd_idnr", "date"], how="left")
    lab = pd.read_parquet(a.cems)[["gppd_idnr", "date", "on", "inst_util"]]
    if a.eu_group is not None:
        lab = pd.concat([lab, pd.read_parquet(a.entsoe)[["gppd_idnr", "date", "on", "inst_util"]]])
    fr = fr.merge(lab, on=["gppd_idnr", "date"], how="left")
    fr = fr.merge(plants, on="gppd_idnr").merge(pd.read_parquet(a.subtypes)[["gppd_idnr", "cooling", "tower"]], on="gppd_idnr", how="left")
    us_lab = fr[fr.is_us & fr.on.notna()]
    if a.eu_group is not None:
        eu_lab = fr[~fr.is_us & fr.on.notna()]
        groups = eu_groups(eu_lab)
        hold = sum(groups, []) if a.no_eu_train else groups[a.eu_group]
        train_c = [c for g in groups for c in g if c not in hold]
        print(f"groups {groups}; holding out {hold}", flush=True)
        rest = pd.concat([us_lab, eu_lab[eu_lab.country.isin(train_c)]])
        va_ids = set(pd.Series(rest.gppd_idnr.unique()).sample(frac=0.1, random_state=0))
        tr, va = rest[~rest.gppd_idnr.isin(va_ids)], rest[rest.gppd_idnr.isin(va_ids)]
        te = eu_lab[eu_lab.country.isin(hold)]
        test_mask = pd.Series(False, index=fr.index)
        a.groups, a.holdout = groups, hold
    elif a.fold is None:
        tr, va, te = (us_lab[us_lab.split_random == s] for s in ("train", "val", "test"))
        test_mask = fr.is_us & (fr.split_random == "test")
    else:  # out-of-fold on-rates for every US plant (feature for the CF model without leakage)
        us_ids = np.array(sorted(fr[fr.is_us].gppd_idnr.unique()))
        fold_of = pd.Series(np.random.default_rng(0).integers(0, 5, len(us_ids)), index=us_ids)
        if a.state_folds:  # whole states per fold, so a held-out state's plants never train the model that scores them
            st = pd.read_excel(a.eia860, skiprows=1)[["Plant Code", "State"]]
            st = st.set_index(pd.to_numeric(st["Plant Code"], errors="coerce")).State
            state = pd.Series([st.get(int(g[3:])) if g[3:].isdigit() else None for g in us_ids], index=us_ids)
            counts = state.value_counts()
            size, group = [0] * 5, {}
            for s_, k in counts.items():
                j = int(np.argmin(size))
                group[s_], size[j] = j, size[j] + k
            fold_of = state.map(group).fillna(fold_of).astype(int)  # plants without a state keep their random fold
            print(f"state folds: plants per fold {fold_of.value_counts().sort_index().tolist()}", flush=True)
        f = us_lab.gppd_idnr.map(fold_of)
        rest = us_lab[f != a.fold]
        va_ids = set(pd.Series(rest.gppd_idnr.unique()).sample(frac=0.1, random_state=0))
        tr, va, te = rest[~rest.gppd_idnr.isin(va_ids)], rest[rest.gppd_idnr.isin(va_ids)], us_lab[f == a.fold]
        test_mask = fr.is_us & (fr.gppd_idnr.map(fold_of) == a.fold)
    print(f"US labeled frames: train {len(tr)}, val {len(va)}, test {len(te)}; share on {us_lab.on.mean():.3f}", flush=True)

    # load only the frames this run touches (train/val + frames to predict), remapping row indices
    apply_mask = test_mask | ~fr.is_us
    if a.eu_group is not None:
        apply_mask = ~fr.is_us & ~fr.country.isin(train_c)
    need = np.unique(np.concatenate([tr.row.values, va.row.values, fr[apply_mask].row.values]))
    mm = np.load(a.mm, mmap_mode="r")
    images = np.empty((len(need), *mm.shape[1:]), mm.dtype)
    for s0 in range(0, len(need), 4096):  # sorted indices -> mostly sequential reads
        images[s0 : s0 + 4096] = mm[need[s0 : s0 + 4096]]
    remap = pd.Series(np.arange(len(need)), index=need)
    tr, va = tr.assign(row=tr.row.map(remap).values), va.assign(row=va.row.map(remap).values)
    fr = fr.assign(row=fr.row.map(remap).fillna(-1).astype(int).values)
    print(f"loaded {len(need)} of {len(mm)} frames", flush=True)
    w = a.vit_weights if a.backbone == "vit_s" else a.weights
    model = OnOffModel(w, inputs=a.inputs, backbone=a.backbone).to(dev)
    enc_ids = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": 1e-3},
                             {"params": list(model.enc.parameters()), "lr": 1e-4}], weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 1e-4], total_steps=steps, pct_start=0.1)
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for img, month, cloud, on, util in batches(tr, images, a.bs, True, dev):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, u = model(img, month, cloud)
                loss = F.binary_cross_entropy_with_logits(lo.float(), on) + F.mse_loss(u.float(), util)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl, n = tl + loss.item() * len(on), n + len(on)
        p, _ = predict(model, va, images, dev)
        auc = roc_auc_score(va.on, p)
        hist.append({"epoch": ep, "loss": tl / n, "val_auroc": auc, "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))

    # US test frames (all frames of test plants, CEMS label where available) + all non-US thermal frames
    apply = fr[apply_mask].copy()
    apply["p_on"], apply["util_pred"] = predict(model, apply, images, dev)
    keep = ["gppd_idnr", "date", "month", "scene_cloud", "country", "primary_fuel", "capacity_mw", "cf_target",
            "cooling", "tower", "on", "inst_util", "p_on", "util_pred"]
    apply[keep].to_parquet(out / "frame_preds.parquet", index=False)

    m = {"best_val_auroc": best}
    if a.eu_group is not None:  # report on the held-out European labelled frames and stop
        eud = apply[apply.on.notna()]
        m.update(frame_metrics(eud, "eu_holdout"))
        m.update(within_plant(eud))
        json.dump({**vars(a), **m, "history": hist}, open(out / "metrics.json", "w"), indent=1)
        (out / "best.pt").unlink()
        print(json.dumps(m, indent=1))
        return
    usd = apply[apply.is_us]
    m.update(frame_metrics(usd, "us_test"))
    m.update(within_plant(usd))
    for name, sel in (("tower", usd.tower == 1), ("no_tower", usd.tower == 0), ("cooling_unknown", usd.tower.isna())):
        m.update(frame_metrics(usd[sel], f"us_test_{name}"))
    clear = apply[apply.scene_cloud < CLEAR]
    pl = clear.groupby("gppd_idnr").agg(on_rate_pred=("p_on", "mean"), on_rate_true=("on", "mean"),
                                         util_pred=("util_pred", "mean"), n=("p_on", "size"))
    pl = pl.join(apply.drop_duplicates("gppd_idnr").set_index("gppd_idnr")[["country", "primary_fuel", "cf_target", "tower", "capacity_mw"]])
    pl.to_csv(out / "plant_onrate.csv")
    us_pl = pl[(pl.country == "USA") & pl.on_rate_true.notna() & (pl.n >= 5)]
    m["us_plant_spearman_onrate"] = us_pl.on_rate_pred.corr(us_pl.on_rate_true, method="spearman")
    m["us_plant_spearman_cf"] = us_pl.util_pred.corr(us_pl.cf_target, method="spearman")
    ind = pl[(pl.country == "IND") & (pl.primary_fuel == "Coal") & (pl.n >= 5)]
    ind_idle = (ind.cf_target < 0.1).astype(int)
    if ind_idle.nunique() == 2:
        m["ind_coal_idle_auroc_onrate"] = roc_auc_score(ind_idle, -ind.on_rate_pred)
        m["ind_coal_idle_auroc_util"] = roc_auc_score(ind_idle, -ind.util_pred)
        m["ind_coal_n"], m["ind_coal_n_idle"] = int(len(ind)), int(ind_idle.sum())
        m["ind_coal_spearman_cf"] = ind.util_pred.corr(ind.cf_target, method="spearman")
    json.dump({**vars(a), **m, "history": hist}, open(out / "metrics.json", "w"), indent=1)
    (out / "best.pt").unlink()
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
