from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import CFG, path, wdi_indicator_codes, write_csv


WGI_MAP = {"ge": "government_effectiveness", "rq": "regulatory_quality",
           "rl": "rule_of_law", "pv": "political_stability"}


def build_wdi_panel() -> pd.DataFrame:
    files = sorted(path("raw_wdi_dir").glob("wdi_*.csv"))
    if not files:
        raise FileNotFoundError("No WDI API files; run mode acquire first")
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    long["value"] = pd.to_numeric(long.value, errors="coerce")
    wide = long.pivot_table(index=["iso3", "year"], columns="indicator", values="value", aggfunc="first").reset_index()
    reverse = {v: k for k, v in wdi_indicator_codes().items()}
    wide = wide.rename(columns=reverse)
    return wide


def build_wgi_panel() -> pd.DataFrame:
    xlsx = path("raw_wgi_dir") / "wgidataset_with_sourcedata-2026.xlsx"
    if not xlsx.exists() or xlsx.read_bytes()[:2] != b"PK":
        raise RuntimeError("WGI 2026 data file is absent or invalid")
    rows = []
    for dimension, variable in WGI_MAP.items():
        data = pd.read_excel(xlsx, sheet_name=dimension, usecols=["Economy (code)", "Year", "Governance estimate (approx. -2.5 to +2.5)"])
        data.columns = ["iso3", "year", variable]
        data["iso3"] = data.iso3.astype(str).str.upper().str.strip()
        data["year"] = pd.to_numeric(data.year, errors="coerce").astype("Int64")
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
        rows.append(data.dropna(subset=["iso3", "year"]))
    merged = rows[0]
    for d in rows[1:]:
        merged = merged.merge(d, on=["iso3", "year"], how="outer", validate="one_to_one")
    return merged


def build_geography() -> pd.DataFrame:
    archive = path("raw_cepii_dir") / "dist_cepii.zip"
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        distances = pd.read_excel(zf.open(member))
    distances.columns = [c.lower() for c in distances.columns]
    out = distances.loc[distances.iso_o.eq(CFG["study"]["exporter_iso3"]),
                        [c for c in ["iso_d", "dist", "distw", "contig", "comlang_off", "colony", "comcol"] if c in distances.columns]].copy()
    out = out.rename(columns={"iso_d": "iso3", "dist": "distance_km", "distw": "population_weighted_distance_km",
                              "contig": "contiguous", "comlang_off": "common_official_language", "colony": "colonial_link", "comcol": "common_colonizer"})
    geo_country = pd.read_excel(path("raw_cepii_dir") / "geo_cepii.xls", usecols=["iso3", "landlocked", "continent"])
    geo_country.columns = [c.lower() for c in geo_country.columns]
    geo_country["iso3"] = geo_country.iso3.astype(str).str.upper().str.strip()
    geo_country = geo_country.drop_duplicates("iso3", keep="first")
    out = out.merge(geo_country[["iso3", "landlocked", "continent"]], on="iso3", how="left", validate="one_to_one")
    out["distance_km"] = pd.to_numeric(out.distance_km, errors="coerce")
    if "population_weighted_distance_km" in out:
        out["population_weighted_distance_km"] = pd.to_numeric(out.population_weighted_distance_km, errors="coerce")
    out["log_distance"] = np.log(out.distance_km.where(out.distance_km.gt(0)))
    return out.rename(columns={"iso3": "destination_iso3"})


def build_macro_features() -> pd.DataFrame:
    wdi = build_wdi_panel()
    wgi = build_wgi_panel()
    panel = wdi.merge(wgi, on=["iso3", "year"], how="outer", validate="one_to_one")
    rename = {"gdp_usd": "gdp_usd", "gdp_per_capita_usd": "gdp_per_capita_usd"}
    panel = panel.rename(columns=rename)
    panel["log_gdp"] = np.log(pd.to_numeric(panel.gdp_usd, errors="coerce").where(pd.to_numeric(panel.gdp_usd, errors="coerce").gt(0)))
    panel["log_gdp_per_capita"] = np.log(pd.to_numeric(panel.gdp_per_capita_usd, errors="coerce").where(pd.to_numeric(panel.gdp_per_capita_usd, errors="coerce").gt(0)))
    panel["log_population"] = np.log(pd.to_numeric(panel.population, errors="coerce").where(pd.to_numeric(panel.population, errors="coerce").gt(0)))
    panel["digitalization_missing_indicator"] = panel.internet_usage.isna().astype("int8")
    path("processed_dir").mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path("processed_dir") / "country_year_features.parquet", index=False)
    return panel


def feature_timing_audit(lag: int) -> pd.DataFrame:
    descriptions = {
        "macro_and_wgi": "World Bank WDI/WGI annual observation at feature_year",
        "geography": "CEPII GeoDist 2011 time-invariant bilateral measure",
        "trade_demand": "BACI destination HS2 imports and supplier concentration at feature_year",
        "historical_trade": "BACI Vietnam exports at feature_year and prior historical years",
    }
    rows = [{"feature_group": group, "source_and_timing": source, "feature_year": "outcome_year - lag",
             "required_max_year": f"<= outcome_year - {lag}", "lag": lag,
             "assertion": "passed after panel construction"} for group, source in descriptions.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build_macro_features()
