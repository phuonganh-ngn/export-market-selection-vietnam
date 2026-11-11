# Data dictionary

## Analytical units and identifiers

The primary analytical key is `destination_iso3 × hs2 × outcome_year`. BACI country IDs and names are preserved in the country mapping table alongside the harmonized ISO3 code and mapping status. `hs2` is the two-digit HS 2012 chapter code; chapter descriptions are copied from the World Customs Organization's official HS Nomenclature 2012 table of contents and stored in `data/processed/hs12_chapter_descriptions.csv`. Chapter 77 is reserved and absent from the 96 observed product chapters.

## Trade data

| Field | Definition | Unit / treatment |
|---|---|---|
| `t` | BACI trade year | Calendar year |
| `i` | BACI exporter identifier | BACI country code; joined through `country_mapping.parquet` |
| `j` | BACI importer identifier | BACI country code; joined through `country_mapping.parquet` |
| `k` | BACI HS6 product code | HS 2012; zero-padded to six digits |
| `v` | BACI bilateral trade value | Thousand current USD; multiplied by 1,000 before outcomes and aggregation |
| `q` | BACI quantity when available | Metric tons; retained for source validation, not used as the primary target |
| `export_value_usd` | Vietnam exports to a destination and HS2 chapter | USD; observed flow or verified zero under the rule below |
| `destination_import_demand` | Imports by the destination from all exporters in HS2 | USD; world supply includes Vietnam and all other reporters |
| `historical_market_share` | Vietnam exports divided by destination world imports | Unit interval where the denominator is positive |
| `destination_supplier_concentration` | Herfindahl concentration of exporter shares in destination–HS2 imports | Unit interval; missing if the trade universe cannot support the calculation |

HS6 observations are aggregated additively to HS4 and HS2 within destination and year. Vietnam exports are separately aggregated by destination, and destination import demand sums all exporters. The full source is kept at HS6; aggregation reconciliation is recorded in `reports/aggregation_checks.csv`.

## Macroeconomic and institutional fields

World Development Indicators are obtained from the World Bank Indicators API. Values retain each indicator's native unit: GDP and GDP per capita in current US dollars, GDP growth/inflation/urban population/internet/trade openness in percent, and population in persons. Indicator codes and definitions are recorded in `reports/data_source_registry.csv` and `src/common.py`.

Worldwide Governance Indicators use the 2026 release and include government effectiveness, regulatory quality, rule of law, and political stability. WGI historical estimates are revised across release vintages. CEPII GeoDist contributes bilateral distance in kilometers and time-invariant dyadic controls; the original CEPII identifiers and geographic fields are preserved in intermediate data.

## Time-varying predictors and timing

`outcome_year` is the year whose Vietnam export value is predicted. `feature_year` is the latest year permitted for annual time-varying predictors. The primary specification has `outcome_year - feature_year >= 2`; the sensitivity specification permits a one-year gap. Lagged export value, presence, market share, demand growth, supplier concentration, volatility, macroeconomic values, governance, and digitalization all obey the configured cutoff. A missingness indicator is used for internet access when unavailable.

## Missing observations and zeros

BACI's sparse flow file contains positive reported trade flows and does not enumerate every zero. A missing Vietnam bilateral record is assigned zero only when BACI verifies positive destination–HS2 imports for that outcome year. If destination–product imports are absent or cannot be verified, the target remains unavailable and is excluded from the evaluated outcome sample. Missing WDI/WGI values remain missing until imputation is fit on the training fold; unavailable geographies are not inferred from country names alone.

`outcome_observed` distinguishes a verified outcome from an unavailable cell. `historical_export_presence` is constructed only from flows at or before the information cutoff. Entry outcomes require verified historical inactivity and the specified USD threshold; an unknown relationship is never treated as a zero-history candidate.

## Reproducibility and source provenance

Every acquired file or reference classification appears in `reports/data_source_registry.csv`, including provider, release/version, retrieval date, units, file hash, and verification status. Raw source files are retained under `data/raw/`; derived country and product mappings and the HS chapter-heading table are under `data/processed/`.
