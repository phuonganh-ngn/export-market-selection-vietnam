# Input audit and frozen evaluation population

**Status: VERIFIED** for key-level reproduction of the audited 2022–2024 two-year-lag candidate set. This is not a point-in-time macro-data vintage.

## Inputs

- Processed panel: `data/processed/panel_lag2.parquet`; rows = 272,113; columns = 54.
- Forecast comparators: `audit_v2/corrected_exante_forecast_predictions.parquet`; rows = 313,985; models = Gravity_PPML, LightGBM, Persistence, RandomForest, XGBoost.
- Entry comparators: `audit_v2/corrected_E2_candidates.parquet`; rows = 616,965; models = LightGBM, RandomForest, XGBoost.
- Primary feature cutoff: `feature_year = outcome_year - 2`; training outcomes stop at `evaluation_year - 2`.
- Entry target: three years of verified reporting through the origin; exports below USD 50,000 in each; success iff observed outcome-year exports reach USD 50,000.
- Missing outcome labels are preserved as NA. An absent BACI destination-HS2 outcome import aggregate is not treated as a true zero.
- Revised current WDI API and WGI 2026 historical values are retrospective and do not establish point-in-time publication availability.

## Reconstructed annual population

|   outcome_year |     n |   observed |   missing |   inactive |   events |
|---------------:|------:|-----------:|----------:|-----------:|---------:|
|           2022 | 20921 |      20734 |       187 |      13791 |      576 |
|           2023 | 20899 |      20749 |       150 |      13687 |      520 |
|           2024 | 20977 |      20755 |       222 |      13653 |      362 |

Total candidates = 62,797; observed outcomes = 62,238; unobserved outcomes = 559; inactive entry candidates = 41,131.

## Neural implementation

PyTorch 2.14.1+cpu; official `tabm` package 0.0.3 (installed distribution 0.0.3).
TabM uses the official package class, categorical one-hot handling, and averages member probabilities after sigmoid for classification; regression members are averaged before inverse transform.
MLP and TabM use the same listed numeric features, destination identity, and HS2 identity. All imputers and scalers are fitted within each chronological training fold.

## Unresolved integrity boundary

The source audit cannot distinguish a true zero-import destination-product outcome from unavailable/incomplete reporting for 559 candidate keys. Complete-case performance is therefore conditional on observed outcomes. See `reports/MISSING_LABEL_SENSITIVITY.md`.
