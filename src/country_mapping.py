from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from common import CFG, hs2_chapter_descriptions, hs2_code, path, write_csv


def _find_member(names: list[str], token: str) -> str:
    matches = [n for n in names if token in Path(n).name.lower() and n.lower().endswith(".csv")]
    if not matches:
        raise FileNotFoundError(f"No BACI metadata CSV found for {token}")
    return matches[0]


def load_baci_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = path("raw_baci_dir") / "BACI_HS12_V202601.zip"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        country_member = _find_member(names, "country_codes_v202601")
        product_member = _find_member(names, "product_codes_hs12_v202601")
        countries = pd.read_csv(zf.open(country_member), dtype=str)
        products = pd.read_csv(zf.open(product_member), dtype=str)
    countries.columns = [c.strip().lower() for c in countries.columns]
    products.columns = [c.strip().lower() for c in products.columns]
    numeric_col = next((c for c in ["country_code", "code", "i"] if c in countries.columns), None)
    iso_col = next((c for c in ["country_iso3", "iso3", "country_iso3_code", "iso_3digit_alpha"] if c in countries.columns), None)
    name_col = next((c for c in ["country_name_full", "country_name_abbreviation", "country_name", "name"] if c in countries.columns), None)
    if numeric_col is None or iso_col is None:
        raise ValueError(f"Unrecognized BACI country metadata schema: {countries.columns.tolist()}")
    if name_col is None:
        name_col = iso_col
    countries = countries.rename(columns={numeric_col: "baci_id", iso_col: "iso3", name_col: "country_name"})
    countries["baci_id"] = pd.to_numeric(countries["baci_id"], errors="coerce").astype("Int64")
    countries["iso3"] = countries["iso3"].astype(str).str.upper().str.strip()
    countries["mapping_status"] = countries["iso3"].where(countries["iso3"].str.fullmatch(r"[A-Z]{3}"), "unmapped/non-ISO3")
    countries = countries[["baci_id", "iso3", "country_name", "mapping_status"]].drop_duplicates()
    pcode = next((c for c in ["code", "product_code", "k"] if c in products.columns), None)
    pdesc = next((c for c in ["description", "product_description", "name"] if c in products.columns), None)
    if pcode is None:
        raise ValueError(f"Unrecognized BACI product metadata schema: {products.columns.tolist()}")
    products = products.rename(columns={pcode: "hs6"})
    products["hs6"] = products["hs6"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if pdesc and pdesc in products:
        products = products.rename(columns={pdesc: "hs6_description"})
    else:
        products["hs6_description"] = ""
    products = products[["hs6", "hs6_description"]].drop_duplicates()
    path("processed_dir").mkdir(parents=True, exist_ok=True)
    countries.to_parquet(path("processed_dir") / "country_mapping.parquet", index=False)
    products.to_parquet(path("processed_dir") / "hs6_codes.parquet", index=False)
    return countries, products


def run() -> None:
    countries, products = load_baci_metadata()
    write_csv(countries, path("reports_dir") / "country_coverage.csv")
    chapters = products.assign(hs2=products.hs6.str[:2]).drop_duplicates("hs2").copy()
    descriptions = hs2_chapter_descriptions()
    chapters["hs2_description"] = chapters.hs2.map(lambda x: descriptions[hs2_code(x)])
    write_csv(chapters, path("reports_dir") / "product_coverage.csv")
    print(f"Loaded {len(countries)} BACI country mappings and {len(products)} HS6 codes.")


if __name__ == "__main__":
    run()
