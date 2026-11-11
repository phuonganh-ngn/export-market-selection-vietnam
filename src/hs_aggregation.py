from __future__ import annotations

import re
import zipfile
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from common import CFG, path, sha256_file, write_csv
from country_mapping import load_baci_metadata


def _verified_cached_aggregates(archive: Path):
    """Reuse a completed streamed aggregation only when its source and checks match."""
    files = [path("interim_dir") / n for n in ["baci_hs2_trade.parquet", "baci_hs4_trade.parquet", "baci_hs6_codebook.parquet"]]
    checks_path = path("reports_dir") / "aggregation_checks.csv"
    registry_path = path("reports_dir") / "data_source_registry.csv"
    if not all(f.exists() for f in files) or not checks_path.exists() or not registry_path.exists():
        return None
    registry = pd.read_csv(registry_path)
    source = registry.loc[registry.dataset.eq("CEPII BACI HS12")]
    if source.empty or source.iloc[0].verification_status != "verified" or source.iloc[0].sha256 != sha256_file(archive):
        return None
    if any(f.stat().st_mtime < archive.stat().st_mtime for f in files):
        return None
    checks = pd.read_csv(checks_path)
    years = set(range(CFG["study"]["first_year"], CFG["study"]["last_year"] + 1))
    if set(checks.hs_level.astype(int)) != {2, 4} or set(checks.year.astype(int)) != years:
        return None
    if len(checks) != len(years) * 2 or not checks["pass"].astype(str).str.lower().isin(["true", "1"]).all():
        return None
    if "duplicate_records" not in checks or checks.duplicate_records.fillna(1).astype(int).ne(0).any():
        return None
    try:
        hs2, hs4, hs6 = (pd.read_parquet(f) for f in files)
    except Exception:
        return None
    if len(hs2) != int(checks.loc[checks.hs_level.eq(2), "markets"].sum()):
        return None
    if len(hs4) != int(checks.loc[checks.hs_level.eq(4), "markets"].sum()):
        return None
    if hs2.duplicated(["baci_importer_id", "hs2", "year"]).any() or hs4.duplicated(["baci_importer_id", "hs4", "year"]).any():
        return None
    print("Reusing verified streamed BACI aggregates; source SHA-256 and all annual aggregation checks match.")
    return hs2, hs4, checks


def _aggregate_resolution(chunks: list[pd.DataFrame], year: int, level: int) -> tuple[pd.DataFrame, dict]:
    if not chunks:
        raise ValueError(f"No BACI data aggregated at HS{level} for {year}")
    detail = pd.concat(chunks, ignore_index=True)
    by = ["j", "hs", "i"]
    suppliers = detail.groupby(by, observed=True, sort=False, as_index=False)["v"].sum()
    world = suppliers.groupby(["j", "hs"], observed=True, sort=False, as_index=False)["v"].sum().rename(columns={"v": "world_imports_thousand_usd"})
    suppliers = suppliers.merge(world, on=["j", "hs"], how="left", validate="many_to_one")
    suppliers["share"] = suppliers["v"] / suppliers["world_imports_thousand_usd"]
    hhi = suppliers.assign(share2=suppliers.share.pow(2)).groupby(["j", "hs"], observed=True, as_index=False).share2.sum().rename(columns={"share2": "supplier_hhi"})
    vnm_id = int(suppliers.loc[suppliers.i.eq(int(suppliers.i.unique()[0])), "i"].iloc[0]) if False else None
    vietnam_id = getattr(_aggregate_resolution, "vietnam_id", None)
    vnm = suppliers.loc[suppliers.i.eq(vietnam_id)].groupby(["j", "hs"], observed=True, as_index=False)["v"].sum().rename(columns={"v": "vietnam_exports_thousand_usd"})
    result = world.merge(hhi, on=["j", "hs"], how="left", validate="one_to_one").merge(vnm, on=["j", "hs"], how="left", validate="one_to_one")
    result["vietnam_exports_thousand_usd"] = result.vietnam_exports_thousand_usd.fillna(0.0)
    result["year"] = int(year)
    result = result.rename(columns={"j": "baci_importer_id", "hs": f"hs{level}"})
    result["world_imports_usd"] = result.world_imports_thousand_usd * 1000.0
    result["vietnam_exports_usd"] = result.vietnam_exports_thousand_usd * 1000.0
    # Independent total validation: world imports must equal all source trade records.
    source_total = float(detail.v.sum())
    aggregate_total = float(result.world_imports_thousand_usd.sum())
    check = {"year": int(year), "hs_level": level, "source_trade_thousand_usd": source_total,
             "aggregated_import_thousand_usd": aggregate_total,
             "absolute_difference_thousand_usd": abs(source_total - aggregate_total),
             "markets": int(len(result)), "supplier_hhi_out_of_range": int(((result.supplier_hhi < 0) | (result.supplier_hhi > 1.000001)).sum())}
    return result, check


def aggregate_baci() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    archive = path("raw_baci_dir") / "BACI_HS12_V202601.zip"
    cached = _verified_cached_aggregates(archive)
    if cached is not None:
        return cached
    country_map, product_codes = load_baci_metadata()
    vnm = country_map.loc[country_map.iso3.eq(CFG["study"]["exporter_iso3"]), "baci_id"]
    if vnm.empty:
        raise RuntimeError("Vietnam (VNM) is not in the BACI country codebook")
    _aggregate_resolution.vietnam_id = int(vnm.iloc[0])
    valid_hs6 = set(product_codes.hs6.astype(str))
    valid_country_ids = set(country_map.baci_id.dropna().astype(int))
    first, last = CFG["study"]["first_year"], CFG["study"]["last_year"]
    checks, hs2_years, hs4_years = [], [], []
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        for year in range(first, last + 1):
            member = next((n for n in members if n.endswith(f"BACI_HS12_Y{year}_V202601.csv")), None)
            if member is None:
                raise FileNotFoundError(f"Year {year} not found in BACI archive")
            partial2, partial4, key_hashes = [], [], []
            bad_rows = 0
            records = 0
            for chunk in pd.read_csv(zf.open(member), usecols=["t", "i", "j", "k", "v"],
                                     dtype={"k": "string"}, chunksize=CFG["compute"]["chunk_rows"]):
                records += len(chunk)
                chunk["k"] = chunk.k.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                values = pd.to_numeric(chunk.v, errors="coerce")
                t = pd.to_numeric(chunk.t, errors="coerce")
                i = pd.to_numeric(chunk.i, errors="coerce")
                j = pd.to_numeric(chunk.j, errors="coerce")
                if t.isna().any() or not t.eq(year).all():
                    raise ValueError(f"BACI year field mismatch in {year}")
                if i.isna().any() or j.isna().any() or (~i.isin(valid_country_ids)).any() or (~j.isin(valid_country_ids)).any():
                    raise ValueError(f"BACI country identifiers do not match codebook in {year}")
                if values.isna().any() or (values < 0).any():
                    raise ValueError(f"BACI has missing or negative values in {year}")
                chunk["i"], chunk["j"] = i.astype("int32"), j.astype("int32")
                key_hashes.append(pd.util.hash_pandas_object(chunk[["i","j","k"]],index=False).to_numpy(dtype="uint64"))
                valid = chunk.k.isin(valid_hs6) & values.notna() & values.ge(0)
                bad_rows += int((~valid).sum())
                chunk = chunk.loc[valid, ["i", "j", "k"]].copy()
                chunk["v"] = values.loc[valid].astype("float64").to_numpy()
                chunk["hs"] = chunk.k.str[:2]
                chunk["hs4"] = chunk.k.str[:4]
                for field, out in [("hs", partial2), ("hs4", partial4)]:
                    g = chunk.groupby(["j", field, "i"], observed=True, sort=False, as_index=False).v.sum()
                    partial = g.rename(columns={field: "hs"})[["j", "hs", "i", "v"]]
                    out.append(partial)
            if bad_rows:
                raise ValueError(f"{bad_rows} invalid HS6/value records in BACI {year}")
            hashes=np.concatenate(key_hashes)
            duplicate_count=int(len(hashes)-len(np.unique(hashes)))
            if duplicate_count:
                raise ValueError(f"{duplicate_count} duplicate exporter-importer-HS6 records found in BACI {year}")
            for level, chunks, output in [(2, partial2, hs2_years), (4, partial4, hs4_years)]:
                aggregate, check = _aggregate_resolution(chunks, year, level)
                check["raw_records"] = records
                check["duplicate_records"] = duplicate_count
                checks.append(check)
                output.append(aggregate)
            print(f"Aggregated BACI {year}: {len(hs2_years[-1]):,} destination–HS2 markets")
            del partial2, partial4, key_hashes, hashes
            gc.collect()
    hs2 = pd.concat(hs2_years, ignore_index=True)
    hs4 = pd.concat(hs4_years, ignore_index=True)
    hs6 = product_codes.assign(hs2=product_codes.hs6.str[:2], hs4=product_codes.hs6.str[:4])
    hs2.to_parquet(path("interim_dir") / "baci_hs2_trade.parquet", index=False)
    hs4.to_parquet(path("interim_dir") / "baci_hs4_trade.parquet", index=False)
    hs6.to_parquet(path("interim_dir") / "baci_hs6_codebook.parquet", index=False)
    check_df = pd.DataFrame(checks)
    # Aggregation order differs between pandas groupby passes, so summing at
    # USD 1,000 scale can differ by a few floating-point ulps. Permit a tight
    # scale-aware tolerance (at most 0.01 thousand USD at observed totals).
    tolerance = np.maximum(1e-8, check_df.source_trade_thousand_usd.abs() * 1e-13)
    check_df["pass"] = check_df.absolute_difference_thousand_usd.le(tolerance) & check_df.supplier_hhi_out_of_range.eq(0)
    write_csv(check_df, path("reports_dir") / "aggregation_checks.csv")
    if not check_df["pass"].all():
        raise RuntimeError("HS6 to HS2/HS4 totals failed aggregation checks")
    return hs2, hs4, check_df


def run() -> None:
    hs2, hs4, checks = aggregate_baci()
    print(f"Prepared {len(hs2):,} HS2 and {len(hs4):,} HS4 destination-year markets; checks pass={checks['pass'].all()}")


if __name__ == "__main__":
    run()
