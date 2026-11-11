from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import CFG, path, write_csv
from country_mapping import load_baci_metadata


def inspect_baci_archive(full_scan: bool = True) -> tuple[pd.DataFrame, dict]:
    archive = path("raw_baci_dir") / "BACI_HS12_V202601.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Mandatory BACI archive unavailable or invalid: {archive}")
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
    years_in_archive = sorted({int(m.group(1)) for n in members if (m := re.search(r"BACI_HS12_Y(\d{4})_V202601\.csv$", n))})
    first, last = CFG["study"]["first_year"], CFG["study"]["last_year"]
    missing = sorted(set(range(first, last + 1)) - set(years_in_archive))
    if missing:
        raise RuntimeError(f"BACI archive is missing required HS12 years: {missing}; contains {years_in_archive}")
    countries, products = load_baci_metadata()
    ids = set(countries.baci_id.dropna().astype(int))
    hs6s = set(products.hs6)
    yearly = []
    with zipfile.ZipFile(archive) as zf:
        for year in range(first, last + 1):
            member = next(n for n in members if n.endswith(f"BACI_HS12_Y{year}_V202601.csv"))
            if not full_scan:
                with zf.open(member) as stream:
                    sample = pd.read_csv(stream, nrows=5, dtype={"k":"string"})
                required={"t","i","j","k","v"}
                if not required.issubset(sample.columns):
                    raise ValueError(f"Missing BACI columns in {member}: {required-set(sample.columns)}")
                yearly.append({"year":year,"raw_records":np.nan,"trade_value_thousand_usd":np.nan,
                    "invalid_hs6_records":np.nan,"negative_value_records":np.nan,"missing_field_records":np.nan,
                    "source_member":member,"header_validation":"passed"})
                continue
            total = nrows = invalid = negative = nulls = 0
            hashes=[]
            usecols = ["t", "i", "j", "k", "v"]
            for chunk in pd.read_csv(zf.open(member), usecols=usecols, dtype={"k": "string"}, chunksize=CFG["compute"]["chunk_rows"]):
                nrows += len(chunk)
                values = pd.to_numeric(chunk["v"], errors="coerce")
                total += float(values.fillna(0).sum())
                hashes.append(pd.util.hash_pandas_object(chunk[["i","j","k"]].astype({"k":"string"}),index=False).to_numpy(dtype="uint64"))
                invalid += int((~chunk["k"].astype(str).str.zfill(6).isin(hs6s)).sum())
                negative += int((values < 0).sum())
                nulls += int(chunk[["t", "i", "j", "k", "v"]].isna().any(axis=1).sum())
                if full_scan and ((~pd.to_numeric(chunk.i, errors="coerce").isin(ids)).any() or (~pd.to_numeric(chunk.j, errors="coerce").isin(ids)).any()):
                    raise ValueError(f"BACI country identifier not found in release codebook ({year})")
                if full_scan and (values.isna().any() or (values < 0).any()):
                    raise ValueError(f"BACI has invalid/negative trade value in {year}")
            digest=np.concatenate(hashes) if hashes else np.array([],dtype="uint64")
            duplicates=int(len(digest)-len(np.unique(digest)))
            yearly.append({"year": year, "raw_records": nrows, "trade_value_thousand_usd": total,
                           "invalid_hs6_records": invalid, "negative_value_records": negative, "missing_field_records": nulls,
                           "duplicate_records":duplicates,"source_member": member})
    audit = pd.DataFrame(yearly)
    def safe_sum(column):
        value=audit[column].sum(min_count=1) if column in audit else np.nan
        return int(value) if pd.notna(value) else None
    summary = {"release": "V202601 HS12", "year_min": int(audit.year.min()), "year_max": int(audit.year.max()),
               "country_codes": len(countries), "product_codes_hs6": len(products),
               "records": safe_sum("raw_records"), "invalid_hs6_records": safe_sum("invalid_hs6_records"),
               "negative_value_records": safe_sum("negative_value_records"), "missing_field_records": safe_sum("missing_field_records"),
               "duplicate_records": safe_sum("duplicate_records")}
    write_csv(audit, path("reports_dir") / "year_coverage.csv")
    return audit, summary


def run() -> None:
    _, summary = inspect_baci_archive(full_scan=True)
    if summary["records"] == 0 or summary["invalid_hs6_records"] or summary["negative_value_records"] or summary["duplicate_records"]:
        raise RuntimeError(f"BACI validation did not pass: {summary}")
    write_csv(pd.DataFrame([summary]), path("reports_dir") / "aggregation_checks.csv")
    print(summary)


if __name__ == "__main__":
    run()
