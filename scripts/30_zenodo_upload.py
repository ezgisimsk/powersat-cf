"""Create (or resume) the Zenodo deposition for the dataset release, upload files, optionally publish.

The access token is read from the ZENODO_TOKEN environment variable and never printed. The deposition id is kept
in <release_dir>/zenodo_deposition.json, so an interrupted upload resumes (files already uploaded with the same
size are skipped). Publishing is irreversible and only happens with --publish.

usage: ZENODO_TOKEN=... python 30_zenodo_upload.py <release_dir> [--publish]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"
METADATA = {
    "title": "PowerSat-CF: power plants with reported generation, national priors, technology subtypes and Sentinel-2 imagery",
    "upload_type": "dataset",
    "description": (
        "<p>Multimodal benchmark for estimating plant-level capacity factors. It links the 34,936 plants of the "
        "Global Power Plant Database (GPPD v1.3) to reported generation (10,632 labeled plants in 29 countries; "
        "annual 2013-2025 and US monthly 2017-2019), national fuel-level capacity factors (Ember), technology "
        "subtypes and cooling types (EIA-860, JRC-PPDB-OPEN), US gross load at the Sentinel-2 overpass hour "
        "(EPA CEMS), and a 2019 Sentinel-2 L2A patch (6 bands, 2.56 km) for 34,927 plants, with random, "
        "spatial-block and geographic-transfer evaluation splits. See README.md for file descriptions, columns, "
        "sources and attribution.</p><p>Code: https://github.com/ezgisimsk/powersat-cf</p>"),
    "creators": [{"name": "Bayar, İrem Ezgi", "affiliation": "Eaton Corporation"}],
    "license": "cc-by-4.0",
    "access_right": "open",
    "version": "1.0",
    "keywords": ["power plants", "capacity factor", "electricity generation", "Sentinel-2", "remote sensing",
                 "benchmark", "Global Power Plant Database", "machine learning"],
    "related_identifiers": [{"identifier": "https://github.com/ezgisimsk/powersat-cf", "relation": "isSupplementedBy",
                             "resource_type": "software"}],
}


def session():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        sys.exit("ZENODO_TOKEN is not set")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release_dir")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--init-only", action="store_true", help="create the deposition and reserve the DOI, upload nothing")
    a = ap.parse_args()
    rel = Path(a.release_dir)
    state_f = rel / "zenodo_deposition.json"
    s = session()

    if state_f.exists():
        dep = s.get(f"{API}/deposit/depositions/{json.loads(state_f.read_text())['id']}").json()
    else:
        r = s.post(f"{API}/deposit/depositions", json={})
        r.raise_for_status()
        dep = r.json()
        state_f.write_text(json.dumps({"id": dep["id"]}))
    r = s.put(f"{API}/deposit/depositions/{dep['id']}", json={"metadata": METADATA})
    r.raise_for_status()
    dep = r.json()
    print("deposition", dep["id"], "| reserved DOI", dep["metadata"].get("prereserve_doi", {}).get("doi"), flush=True)
    if a.init_only:
        return

    bucket = dep["links"]["bucket"]
    have = {f["filename"]: int(f["filesize"]) for f in s.get(f"{API}/deposit/depositions/{dep['id']}/files").json()}
    files = sorted(p for p in rel.iterdir() if p.is_file() and p.name != state_f.name)
    for p in files:
        if have.get(p.name) == p.stat().st_size:
            print("skip", p.name, flush=True)
            continue
        for attempt in range(1, 6):  # long uploads can drop; retry the whole file
            try:
                with open(p, "rb") as fh:
                    r = s.put(f"{bucket}/{p.name}", data=fh, timeout=(30, 3 * 3600))
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"retry {attempt} {p.name}: {type(e).__name__}", flush=True)
                time.sleep(30 * attempt)
        else:
            sys.exit(f"giving up on {p.name}")
        print("uploaded", p.name, round(p.stat().st_size / 1e9, 2), "GB", flush=True)

    if a.publish:
        r = s.post(f"{API}/deposit/depositions/{dep['id']}/actions/publish")
        r.raise_for_status()
        d = r.json()
        print("PUBLISHED doi:", d.get("doi"), "| url:", d.get("links", {}).get("record_html") or d.get("links", {}).get("html"))
    else:
        print("draft ready (not published):", dep["links"].get("html"))


if __name__ == "__main__":
    main()
