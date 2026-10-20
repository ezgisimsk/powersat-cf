"""Pre-registered experiments B and C (see paper/review_response.md), coal + gas cohort.

B  Within-country relative activity (train US, test non-US; Europe = main):
   B1  residual GBT + subtype + relative on-rate (on-rate / mean on-rate of the same country and fuel)
   B2  allocation without training: CF = national prior x relative on-rate, clipped to [0, 1]
   B3  shrunk allocation: CF = prior x (1 + alpha (relative on-rate - 1)), alpha fitted by least squares on US train
   compared with the national prior, residual GBT + subtype, and + subtype + raw on-rate.
   Metrics: MAE (all / Europe / coal / gas), country-macro MAE, median within-country Spearman (countries with
   >= 10 test plants), gains vs the prior with plant- and country-cluster bootstrap CIs.
C  Few-shot adaptation to Europe: add k in {0, 5, 10, 20, 50} random European plants to the US training data,
   test on the remaining European plants, 10 random draws per k; models: prior, + subtype, + subtype + on-rate.

On-rates: year-matched, averaged over the three on/off seeds.
usage: python 32_relative_fewshot.py <project_root> <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("rev", HERE / "31_revision_cg.py")
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)
bl, ds = rev.bl, rev.ds
SUB, ON, EUROPE = rev.SUB, rev.ON, rev.EUROPE
REL = ["rel_on", "rel_util"]


def cohort(root):
    plants = pd.read_parquet(root / "data/processed/plants.parquet")
    plants = plants.merge(pd.read_parquet(root / "data/processed/subtypes.parquet")[["gppd_idnr", *SUB]], on="gppd_idnr", how="left")
    L = plants[plants.split_random.notna() & plants.primary_fuel.isin(rev.FUELS)].copy()
    lab = pd.read_parquet(root / "data/processed/labels.parquet").query("valid & source == 'gppd' & year >= 2017 & year <= 2019")
    label_years = lab.groupby("gppd_idnr").year.apply(set)
    matched = [rev.onrates(root, d, label_years)[1] for d in rev.ONOFF_DIRS]
    rate = sum(matched) / len(matched)
    L = L.join(rate, on="gppd_idnr")
    grp = L.groupby(["country", "primary_fuel"])
    L["rel_on"] = L.on_rate / grp.on_rate.transform("mean")
    L["rel_util"] = L.util / grp.util.transform("mean")
    for c in ["primary_fuel", "other_fuel1", *SUB]:
        L[c] = L[c].astype("category")
    bl.CATS = ["primary_fuel", "other_fuel1", *SUB]
    return L


def within_country_spearman(te, p, min_n=10):
    vals = []
    for c in te.country.unique():
        m = (te.country == c).values
        if m.sum() >= min_n and np.std(p[m]) > 0:
            vals.append(pd.Series(p[m]).corr(pd.Series(te.cf_target.values[m]), method="spearman"))
    return np.median(vals) if vals else np.nan, len(vals)


def experiment_b(L, out):
    tr, va, te = (L[L.split_geo == s] for s in ("train", "val", "test"))
    x = tr.natl_cf * (tr.rel_on - 1)
    alpha = float((x * (tr.cf_target - tr.natl_cf)).sum() / (x ** 2).sum())
    preds = {
        "natl_prior": te.natl_cf.clip(0, 1).values,
        "+sub": bl.gbt_predict(tr, va, te, bl.F_WRI + SUB, residual=True),
        "+sub+onrate (raw)": bl.gbt_predict(tr, va, te, bl.F_WRI + SUB + ON, residual=True),
        "B1 +sub+relative": bl.gbt_predict(tr, va, te, bl.F_WRI + SUB + REL, residual=True),
        "B2 allocation": (te.natl_cf * te.rel_on.fillna(1)).clip(0, 1).values,
        "B3 shrunk allocation": (te.natl_cf * (1 + alpha * (te.rel_on.fillna(1) - 1))).clip(0, 1).values,
    }
    region = np.where(te.country.isin(EUROPE), "Europe", np.where(te.country == "IND", "India",
                      np.where(te.country == "AUS", "Australia", "other")))
    subsets = [("all", np.ones(len(te), bool)), ("Europe", region == "Europe"), ("India", region == "India"),
               ("Australia", region == "Australia"), ("Coal", (te.primary_fuel == "Coal").values),
               ("Gas", (te.primary_fuel == "Gas").values)]
    rows, ci = [], []
    for sub, m in subsets:
        for k, p in preds.items():
            r = rev.evaluate(te.cf_target.values[m], p[m], te.capacity_mw.values[m])
            tm = te[m]
            r["MAE_country_macro"] = pd.Series(np.abs(tm.cf_target.values - p[m])).groupby(tm.country.values).mean().mean()
            r["within_country_spearman"], r["n_countries"] = within_country_spearman(tm, p[m])
            rows.append({"subset": sub, "model": k, **r})
        if sub in ("all", "Europe", "Coal", "Gas"):
            for k in preds:
                if k == "natl_prior":
                    continue
                for by_c in (False, True):
                    mean, lo, hi = rev.boot_diff(te[m], preds["natl_prior"][m], preds[k][m], by_c)
                    ci.append({"subset": sub, "model": k, "resample": "country" if by_c else "plant",
                               "gain_vs_prior": mean, "ci_low": lo, "ci_high": hi})
    res, ci = pd.DataFrame(rows), pd.DataFrame(ci)
    res.round(4).to_csv(out / "B_results.csv", index=False)
    ci.round(4).to_csv(out / "B_ci.csv", index=False)
    print(f"alpha (B3) = {alpha:.3f}")
    print(res.pivot_table(index="model", columns="subset", values="MAE").round(3).to_string())
    print(res[res.subset.isin(["all", "Europe"])].pivot_table(index="model", columns="subset",
          values=["MAE_country_macro", "within_country_spearman"]).round(3).to_string())
    print(ci.round(4).to_string(index=False))


def experiment_c(L, out, ks=(0, 5, 10, 20, 50), reps=10):
    us_tr, us_va = L[L.split_geo == "train"], L[L.split_geo == "val"]
    E = L[(L.split_geo == "test") & L.country.isin(EUROPE)]
    rows = []
    for k in ks:
        for r in range(reps if k > 0 else 1):
            pick = E.sample(k, random_state=r) if k else E.iloc[:0]
            te = E.drop(pick.index)
            tr = pd.concat([us_tr, pick])
            p = {"natl_prior": te.natl_cf.clip(0, 1).values,
                 "+sub": bl.gbt_predict(tr, us_va, te, bl.F_WRI + SUB, residual=True),
                 "+sub+onrate": bl.gbt_predict(tr, us_va, te, bl.F_WRI + SUB + ON, residual=True)}
            for name, v in p.items():
                rows.append({"k": k, "rep": r, "model": name, "n_test": len(te), "MAE": np.abs(te.cf_target.values - v).mean()})
        print("k", k, "done", flush=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "C_fewshot.csv", index=False)
    summ = res.groupby(["k", "model"]).MAE.agg(["mean", "std"]).unstack("model")
    print(summ.round(4).to_string())


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    L = cohort(root)
    print("cohort", len(L), "| with on-rate", L.on_rate.notna().sum(), flush=True)
    experiment_b(L, out)
    experiment_c(L, out)


if __name__ == "__main__":
    main()
