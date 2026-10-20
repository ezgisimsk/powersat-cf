"""Pack downloaded Sentinel-2 patches of the selected plants into one uint16 memmap for fast training I/O.

Writes <out_prefix>.u16 (N x 6 x S x S, C-order) and <out_prefix>.ids.csv (row -> gppd_idnr).
Plants whose patch is missing are skipped (has_image = False downstream).

usage: python 08_pack_patches.py <plants.parquet> <s2_dir> <out_prefix> [--all]
       (default: only plants with a cf_target label)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    plants, s2_dir, prefix = pd.read_parquet(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if "--all" not in sys.argv:
        plants = plants[plants.split_random.notna()]
    idx = pd.read_csv(s2_dir / "index.csv").drop_duplicates("gppd_idnr", keep="last")
    ok = set(idx.gppd_idnr[idx.status == "ok"])
    ids = [p for p in plants.gppd_idnr if p in ok and (s2_dir / "patches" / f"{p}.npy").exists()]
    first = np.load(s2_dir / "patches" / f"{ids[0]}.npy")
    mm = np.lib.format.open_memmap(prefix + ".npy", mode="w+", dtype=np.uint16, shape=(len(ids), *first.shape))
    for i, p in enumerate(ids):
        mm[i] = np.load(s2_dir / "patches" / f"{p}.npy")
        if i % 2000 == 0:
            print(i, len(ids), flush=True)
    mm.flush()
    pd.DataFrame({"gppd_idnr": ids}).to_csv(prefix + ".ids.csv", index_label="row")
    print(f"packed {len(ids)} / {len(plants)} plants -> {prefix}.npy {mm.shape}")


if __name__ == "__main__":
    main()
