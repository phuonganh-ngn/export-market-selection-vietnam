from __future__ import annotations

import datetime as dt
from html.parser import HTMLParser
import io
import json
import math
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from common import CFG, ROOT, mkdirs, path, sha256_file, wdi_indicator_codes, write_csv


def download(url: str, destination: Path, min_bytes: int = 1000) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {}
    mode = "wb"
    if partial.exists():
        headers["Range"] = f"bytes={partial.stat().st_size}-"
        mode = "ab"
    response = requests.get(url, stream=True, timeout=(30, 300), headers=headers)
    response.raise_for_status()
    if mode == "ab" and response.status_code != 206:
        mode = "wb"
    with partial.open(mode) as out:
        for block in response.iter_content(1024 * 1024):
            if block:
                out.write(block)
    if partial.stat().st_size < min_bytes:
        raise RuntimeError(f"Download too small ({partial.stat().st_size} bytes): {url}")
    partial.replace(destination)


def acquire_baci() -> Path:
    archive = path("raw_baci_dir") / "BACI_HS12_V202601.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        _download_baci_ranges(archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"BACI archive is not a valid ZIP after download: {archive}")
    return archive


def _download_baci_ranges(archive: Path, workers: int = 16) -> None:
    """Resume a throttled large BACI ZIP with bounded, verified HTTP byte ranges."""
    url = CFG["sources"]["baci_url"]
    head = requests.head(url, timeout=(30, 90), allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    if not total or not head.headers.get("Accept-Ranges", "").lower().startswith("bytes"):
        raise RuntimeError("CEPII BACI source did not advertise a byte-range download")
    prefix = archive.stat().st_size if archive.exists() else 0
    if prefix >= total:
        return
    parts_dir = archive.parent / ".BACI_HS12_V202601.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    remaining = total - prefix
    n = min(workers, max(1, math.ceil(remaining / (64 * 1024 * 1024))))
    size = math.ceil(remaining / n)
    ranges = []
    cursor = prefix
    for idx in range(n):
        start = cursor
        end = min(total - 1, start + size - 1)
        if start <= end:
            ranges.append((idx, start, end))
        cursor = end + 1

    def fetch(item):
        idx, start, end = item
        dest = parts_dir / f"part_{idx:02d}.bin"
        expected = end - start + 1
        existing = dest.stat().st_size if dest.exists() else 0
        if existing > expected:
            dest.write_bytes(b"")
            existing = 0
        for attempt in range(8):
            existing = dest.stat().st_size if dest.exists() else 0
            if existing > expected:
                dest.write_bytes(b"")
                existing = 0
            cur = start + existing
            if cur > end:
                break
            response = None
            try:
                response = requests.get(url, headers={"Range": f"bytes={cur}-{end}"}, stream=True, timeout=(30, 600))
                if response.status_code != 206:
                    raise RuntimeError(f"BACI range request returned HTTP {response.status_code} for {cur}-{end}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {cur}-"):
                    raise RuntimeError(f"Unexpected BACI Content-Range: {content_range}")
                with dest.open("ab") as out:
                    for block in response.iter_content(256 * 1024):
                        if block:
                            out.write(block)
            except (requests.exceptions.RequestException, RuntimeError) as exc:
                if attempt == 7:
                    raise
                # CEPII's IIS endpoint occasionally ignores a Range header and
                # answers with a full-response 200. Do not append that body;
                # close it and retry the same byte range instead.
                print(f"Retrying BACI range {idx+1}: attempt {attempt+2}/8 after {type(exc).__name__}: {exc}")
                time.sleep(min(30, 2 ** attempt))
            finally:
                if response is not None:
                    response.close()
        actual = dest.stat().st_size if dest.exists() else 0
        if actual != expected:
            raise RuntimeError(f"BACI range {idx} incomplete: {actual} bytes; expected {expected}")
        return idx, actual

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, r) for r in ranges]
        for future in as_completed(futures):
            idx, amount = future.result()
            completed += amount
            print(f"BACI range {idx+1}/{len(ranges)} complete ({completed/1e6:.0f}/{remaining/1e6:.0f} MB fetched)")
    assembled = archive.with_suffix(".assembling")
    with assembled.open("wb") as out:
        if archive.exists():
            with archive.open("rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
        for idx, _, _ in ranges:
            with (parts_dir / f"part_{idx:02d}.bin").open("rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    if assembled.stat().st_size != total or not zipfile.is_zipfile(assembled):
        raise RuntimeError("Assembled BACI archive failed size or ZIP validation")
    assembled.replace(archive)
    shutil.rmtree(parts_dir)


def acquire_wdi() -> list[Path]:
    out_files = []
    first, last = CFG["study"]["first_year"], CFG["study"]["last_year"] + 1
    for label, code in wdi_indicator_codes().items():
        destination = path("raw_wdi_dir") / f"wdi_{label}_{first}_{last}.csv"
        if destination.exists() and destination.stat().st_size > 100:
            out_files.append(destination)
            continue
        url = f"{CFG['sources']['wdi_api']}/country/all/indicator/{code}"
        params = {"format": "json", "date": f"{first}:{last}", "per_page": 20000, "page": 1}
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError(f"Unexpected World Bank API response for {code}: {str(payload)[:500]}")
        meta = payload[0]
        records = payload[1]
        rows = []
        while int(meta.get("page", 1)) < int(meta.get("pages", 1)):
            params["page"] = int(meta["page"]) + 1
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            page_payload = response.json()
            meta, records = page_payload[0], page_payload[1]
            rows.extend(records)
        # The loop above appended only later pages; add page 1 exactly once.
        page1 = requests.get(url, params={**params, "page": 1}, timeout=90).json()
        rows = (page1[1] or []) + rows
        data = []
        for row in rows:
            country = row.get("country", {}) or {}
            if (country.get("region", {}) or {}).get("id") == "NA":
                continue
            iso3 = row.get("countryiso3code")
            if not iso3 or len(iso3) != 3:
                continue
            data.append({"iso3": iso3.upper(), "country_name": country.get("value"),
                         "year": int(row["date"]), "indicator": code,
                         "indicator_label": (row.get("indicator", {}) or {}).get("value"),
                         "value": row.get("value")})
        write_csv(pd.DataFrame(data), destination)
        out_files.append(destination)
    return out_files


def acquire_wgi() -> Path:
    destination = path("raw_wgi_dir") / "wgidataset_with_sourcedata-2026.xlsx"
    if not destination.exists() or destination.stat().st_size < 100_000:
        download(CFG["sources"]["wgi_url"], destination, min_bytes=100_000)
    if destination.read_bytes()[:2] != b"PK":
        raise RuntimeError("WGI download is not an XLSX/ZIP file; check the official source URL")
    return destination


def acquire_geodist() -> list[Path]:
    files = [(CFG["sources"]["geodist_url"], path("raw_cepii_dir") / "dist_cepii.zip"),
             (CFG["sources"]["geodist_country_url"], path("raw_cepii_dir") / "geo_cepii.xls")]
    for url, dest in files:
        if not dest.exists() or dest.stat().st_size < 10_000:
            download(url, dest, min_bytes=10_000)
    if not zipfile.is_zipfile(files[0][1]):
        raise RuntimeError("CEPII GeoDist dyadic archive failed ZIP validation")
    return [p for _, p in files]


class _WCOChapterRows(HTMLParser):
    """Read the chapter-number and title columns from WCO's HS 2012 table."""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_row = False
        self.in_cell = False
        self.cells = []
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_row = True
            self.cells = []
        elif tag == "td" and self.in_row:
            self.in_cell = True
            self.buffer = []

    def handle_data(self, data):
        if self.in_cell:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.cells.append(" ".join(" ".join(self.buffer).split()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self.rows.append(self.cells)
            self.in_row = False


def acquire_hs12_chapter_headings() -> Path:
    url = CFG["sources"]["hs12_headings_url"]
    response = requests.get(url, timeout=(30, 120))
    response.raise_for_status()
    parser = _WCOChapterRows()
    parser.feed(response.text)
    rows = []
    for cells in parser.rows:
        if len(cells) >= 2 and re.fullmatch(r"\d{1,2}", cells[0].strip()):
            number = int(cells[0])
            if 1 <= number <= 97 and number != 77:
                rows.append({"hs2": f"{number:02d}", "hs2_description": cells[1].strip(),
                    "classification": "Harmonized System", "edition": "2012", "source_url": url,
                    "retrieval_date": dt.date.today().isoformat()})
    frame = pd.DataFrame(rows).drop_duplicates("hs2")
    if len(frame) != 96 or frame.hs2.duplicated().any():
        raise RuntimeError(f"WCO HS 2012 heading table did not yield 96 unique active chapters; got {len(frame)}")
    destination = path("processed_dir") / "hs12_chapter_descriptions.csv"
    write_csv(frame, destination)
    return destination


def registry() -> pd.DataFrame:
    source_items = [
        ("CEPII BACI HS12", "CEPII", CFG["sources"]["baci_url"], "V202601 HS12", path("raw_baci_dir") / "BACI_HS12_V202601.zip", "2012–2024", "thousand current USD (v); metric tons (q)", "Etalab Open Licence 2.0", "Official BACI release page; ZIP validated"),
        ("World Bank WGI", "World Bank", CFG["sources"]["wgi_url"], "2026", path("raw_wgi_dir") / "wgidataset_with_sourcedata-2026.xlsx", "1996–2025", "standard normal estimates / absolute scores", "CC BY 4.0", "Official WGI 2026 release; XLSX validated"),
        ("CEPII GeoDist dyadic", "CEPII", CFG["sources"]["geodist_url"], "GeoDist download", path("raw_cepii_dir") / "dist_cepii.zip", "time-invariant", "distance in km and dyadic controls", "Etalab 2.0", "Official CEPII download; ZIP validated"),
        ("CEPII GeoDist country", "CEPII", CFG["sources"]["geodist_country_url"], "GeoDist download", path("raw_cepii_dir") / "geo_cepii.xls", "time-invariant", "country geography and historical controls", "Etalab 2.0", "Official CEPII download; legacy XLS signature validated"),
        ("HS 2012 chapter headings", "World Customs Organization", CFG["sources"]["hs12_headings_url"], "HS Nomenclature 2012 Edition", path("processed_dir") / "hs12_chapter_descriptions.csv", "2012 classification", "HS2 chapter descriptions", "WCO nomenclature; see source terms", "Official WCO table of contents; 96 active chapters, chapter 77 reserved"),
    ]
    wdi_definitions = {"gdp_usd":"GDP, current US$", "gdp_per_capita_usd":"GDP per capita, current US$",
        "gdp_growth":"GDP growth, annual percent", "population":"Population, total persons",
        "urban_population_share":"Urban population, percent of total", "internet_usage":"Individuals using the Internet, percent of population",
        "inflation":"Inflation, consumer prices, annual percent", "trade_openness":"Trade, percent of GDP"}
    for label, code in wdi_indicator_codes().items():
        p = path("raw_wdi_dir") / f"wdi_{label}_{CFG['study']['first_year']}_{CFG['study']['last_year'] + 1}.csv"
        source_items.append((f"World Bank WDI {code}", "World Bank", f"{CFG['sources']['wdi_api']}/country/all/indicator/{code}", "API vintage accessed at run", p,
                             f"{CFG['study']['first_year']}–{CFG['study']['last_year'] + 1}", wdi_definitions[label], "World Bank open data terms", "World Bank API JSON response persisted as CSV"))
    stamp = dt.date.today().isoformat()
    rows = []
    for dataset, provider, url, version, fp, years, units, license_name, status in source_items:
        exists = fp.exists() and fp.stat().st_size > 0
        verified = "verified"
        if not exists:
            verified = "not acquired"
        elif fp.suffix == ".xlsx" and fp.read_bytes()[:2] != b"PK":
            verified = "invalid download"
        elif fp.suffix == ".zip" and not zipfile.is_zipfile(fp):
            verified = "invalid archive"
        elif fp.suffix == ".xls" and fp.read_bytes()[:8] != bytes.fromhex("D0CF11E0A1B11AE1"):
            verified = "invalid XLS"
        elif dataset == "HS 2012 chapter headings":
            verified = "verified official WCO table; 96 active chapters parsed"
        elif fp.suffix == ".csv":
            verified = "verified API response; CSV persisted"
        notes = status
        if dataset == "HS 2012 chapter headings":
            notes = "Official HS Nomenclature 2012 table of contents; chapter 77 is reserved."
        elif fp.suffix == ".csv":
            notes = status + "; historical World Bank series can be revised after the reference year."
        rows.append({"dataset": dataset, "provider": provider, "source_url": url, "version": version,
                     "download_date": stamp if exists else "", "file_name": fp.name,
                     "file_size": fp.stat().st_size if exists else 0, "sha256": sha256_file(fp) if exists else "",
                     "years_covered": years, "units": units, "license": license_name,
                     "verification_status": verified, "notes": notes})
    result = pd.DataFrame(rows)
    write_csv(result, path("reports_dir") / "data_source_registry.csv")
    return result


def run() -> None:
    mkdirs()
    acquire_baci()
    acquire_wdi()
    acquire_wgi()
    acquire_geodist()
    acquire_hs12_chapter_headings()
    registry()
    print("Acquisition complete and source registry written.")


if __name__ == "__main__":
    run()
