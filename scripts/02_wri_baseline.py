"""How good is WRI's own generation estimate (2017) vs reported generation? -> baseline to beat."""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
m = df.generation_gwh_2017.notna() & df.estimated_generation_gwh_2017.notna() & (df.generation_gwh_2017 > 0)
d = df[m]
cf_true = d.generation_gwh_2017 * 1000 / (d.capacity_mw * 8760)
cf_est = d.estimated_generation_gwh_2017 * 1000 / (d.capacity_mw * 8760)
ok = cf_true.between(0, 1)
cf_true, cf_est, d = cf_true[ok], cf_est[ok], d[ok]
print("n =", len(d))
print("CF MAE overall:", round((cf_true - cf_est).abs().mean(), 3))
r2 = 1 - ((cf_true - cf_est) ** 2).sum() / ((cf_true - cf_true.mean()) ** 2).sum()
print("CF R2 overall:", round(r2, 3))
g = pd.DataFrame({"fuel": d.primary_fuel, "ae": (cf_true - cf_est).abs()})
print(g.groupby("fuel").ae.agg(["count", "mean"]).round(3).sort_values("count", ascending=False))
