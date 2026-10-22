"""Overpass-hour activity labels for European coal and gas plants from ENTSO-E unit-level generation (16.1.A).

Plants are linked to generation units through JRC-PPDB-OPEN (eic_g). For every country domain and every date on
which one of its plants has a monthly Sentinel-2 frame, the day of per-unit actual generation is downloaded
(cached as parquet, resumable). The overpass time is approximated per plant from longitude: Sentinel-2 crosses at
about 10:30 local mean solar time, i.e. 10.5 - lon/15 hours UTC; the label is the summed unit output in that UTC
hour divided by the plant's maximum daytime output over all fetched days; on = utilisation > 0.05.

The API key is read from the ENTSOE_KEY environment variable and never printed.

usage:
  python 35_entsoe_overpass.py probe <project_root>        # which domains return our units
  python 35_entsoe_overpass.py fetch <project_root>        # download unit generation on frame dates
  python 35_entsoe_overpass.py label <project_root>        # build data/processed/entsoe_overpass.parquet
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = {  # GPPD country -> ENTSO-E areas that return unit-level data (probe 2018-07-15); none for DNK, SWE
    "AUT": ["AT"], "BEL": ["BE"], "BGR": ["BG"], "CZE": ["CZ"],
    "DEU": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"], "ESP": ["ES"], "FIN": ["FI"], "FRA": ["FR"],
    "GRC": ["GR"], "HUN": ["HU"], "ITA": ["IT"], "LTU": ["LT"], "LVA": ["LV"], "NLD": ["NL"], "POL": ["PL"],
    "PRT": ["PT"], "ROU": ["RO"], "SVK": ["SK"], "SVN": ["SI"],
}
EU = set(DOMAINS)


def client():
    from entsoe import EntsoeRawClient
    key = os.environ.get("ENTSOE_KEY")
    if not key:
        sys.exit("ENTSOE_KEY is not set")
    return EntsoeRawClient(api_key=key, retry_count=3, retry_delay=10)


def parse(xml):
    """Parse a 16.1.A GL_MarketDocument: generation (inBiddingZone) series only; curveType A01 (all points)
    and A03 (points only where the value changes; missing positions repeat the previous value)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    ns = {"n": root.tag.split("}")[0].strip("{")}
    rows = []
    for ts in root.findall("n:TimeSeries", ns):
        if ts.find("n:inBiddingZone_Domain.mRID", ns) is None:
            continue  # consumption (e.g. pumped storage) series
        res_eic = ts.findtext("n:registeredResource.mRID", namespaces=ns)
        psr_eic = ts.findtext("n:MktPSRType/n:PowerSystemResources/n:mRID", namespaces=ns)
        curve = ts.findtext("n:curveType", default="A01", namespaces=ns)
        for per in ts.findall("n:Period", ns):
            start = pd.Timestamp(per.findtext("n:timeInterval/n:start", namespaces=ns))
            end = pd.Timestamp(per.findtext("n:timeInterval/n:end", namespaces=ns))
            step = pd.Timedelta(per.findtext("n:resolution", namespaces=ns).replace("PT", "").lower().replace("m", "min"))
            n = int((end - start) / step)
            vals = np.full(n, np.nan)
            for pt in per.findall("n:Point", ns):
                vals[int(pt.findtext("n:position", namespaces=ns)) - 1] = float(pt.findtext("n:quantity", namespaces=ns))
            if curve == "A03":
                vals = pd.Series(vals).ffill().values
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    rows.append((start + i * step, res_eic, psr_eic, v))
    df = pd.DataFrame(rows, columns=["time", "res_eic", "psr_eic", "mw"])
    if len(df):  # hourly means (some areas report 15-min data)
        df["time"] = df.time.dt.floor("h")
        df = df.groupby(["time", "res_eic", "psr_eic"], as_index=False).mw.mean()
    return df


def fetch_day(c, area, day):
    s = pd.Timestamp(day, tz="UTC")
    return parse(c.query_generation_per_plant(area, start=s, end=s + pd.Timedelta(days=1)))


def units(root):
    return pd.read_csv(root / "data/interim/europe_units.csv")


def frame_dates(root, ids):
    fr = pd.read_parquet(root / "data/processed/ts_frames64.frames.parquet", columns=["gppd_idnr", "date"])
    return fr[fr.gppd_idnr.isin(ids)]


def probe(root):
    c, u = client(), units(root)
    ours = set(u.eic_g) | set(u.eic_p.dropna())
    rows = []
    for country, areas in DOMAINS.items():
        for a in areas:
            try:
                d = fetch_day(c, a, "2018-07-15")
                got = set(d.res_eic) | set(d.psr_eic)
                rows.append((country, a, d.psr_eic.nunique(), len(got & ours), ""))
            except Exception as ex:
                rows.append((country, a, 0, 0, type(ex).__name__))
            time.sleep(0.5)
    print(pd.DataFrame(rows, columns=["country", "area", "units_returned", "our_eics", "error"]).to_string(index=False))


def fetch(root):
    c = client()
    u = units(root)
    plants = pd.read_parquet(root / "data/processed/plants.parquet").set_index("gppd_idnr")
    u["iso3"] = u.WRI_id.map(plants.country)
    fd = frame_dates(root, set(u.WRI_id))
    fd["iso3"] = fd.gppd_idnr.map(plants.country)
    cache = root / "data/raw/entsoe"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = sorted({(a, d) for iso, d in zip(fd.iso3, fd.date) for a in DOMAINS.get(iso, [])})
    print(len(jobs), "area-days to fetch", flush=True)
    done = 0
    for a, d in jobs:
        f = cache / f"{a}_{d}.parquet"
        if f.exists() or (cache / f"{a}_{d}.empty").exists():
            continue
        try:
            fetch_day(c, a, d).to_parquet(f, index=False)
        except Exception as ex:
            (cache / f"{a}_{d}.empty").write_text(type(ex).__name__)
        done += 1
        if done % 50 == 0:
            print(done, "fetched", flush=True)
        time.sleep(0.2)
    print("fetch finished", flush=True)


def label(root):
    u = units(root)
    plants = pd.read_parquet(root / "data/processed/plants.parquet").set_index("gppd_idnr")
    files = sorted((root / "data/raw/entsoe").glob("*.parquet"))
    g = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["time", "res_eic", "psr_eic"])
    by_unit = g.merge(u[["eic_g", "WRI_id"]], left_on="psr_eic", right_on="eic_g")
    by_res = g.merge(u[["eic_g", "WRI_id"]], left_on="res_eic", right_on="eic_g")
    m = pd.concat([by_unit, by_res]).drop_duplicates(["time", "res_eic", "psr_eic"])
    plant_hour = m.groupby(["WRI_id", "time"]).mw.sum().reset_index()
    plant_hour["time"] = pd.to_datetime(plant_hour.time, utc=True)
    lon = plants.longitude
    plant_hour["date"] = plant_hour.time.dt.strftime("%Y-%m-%d")
    plant_hour["hour"] = plant_hour.time.dt.hour
    overpass = (10.5 - plant_hour.WRI_id.map(lon) / 15).astype(float)
    plant_hour["is_overpass"] = plant_hour.hour == np.floor(overpass).astype(int)
    day = plant_hour[plant_hour.hour.between(6, 15)]
    ref = day.groupby("WRI_id").mw.max().rename("ref_load")
    lab = plant_hour[plant_hour.is_overpass].join(ref, on="WRI_id")
    lab = lab[lab.ref_load > 0]
    lab["inst_util"] = (lab.mw / lab.ref_load).clip(0, 1)
    lab["on"] = (lab.inst_util > 0.05).astype("int8")
    lab = lab.rename(columns={"WRI_id": "gppd_idnr"})[["gppd_idnr", "date", "mw", "ref_load", "inst_util", "on"]]
    fd = frame_dates(root, set(lab.gppd_idnr))
    lab = lab.merge(fd, on=["gppd_idnr", "date"])
    lab.to_parquet(root / "data/processed/entsoe_overpass.parquet", index=False)
    print(len(lab), "labeled frames,", lab.gppd_idnr.nunique(), "plants, share on", round(lab.on.mean(), 3))
    print(lab.assign(country=lab.gppd_idnr.map(plants.country)).groupby("country").gppd_idnr.nunique().to_dict())


if __name__ == "__main__":
    {"probe": probe, "fetch": fetch, "label": label}[sys.argv[1]](Path(sys.argv[2]))
