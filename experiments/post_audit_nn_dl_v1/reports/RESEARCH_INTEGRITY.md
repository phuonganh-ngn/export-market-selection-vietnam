# Research integrity

**Final status: COMPLETE_REDUCED_SPECIFICATION.**

## Data and timing

The extension reads the processed HS2 two-year-lag panel and the saved audited baseline prediction files; no raw data were modified or downloaded. Candidate eligibility is based on lagged positive destination import demand and verified historical inactivity. Outcome-year absent BACI records are not recoded to zero; the 559 unknown labels remain NA.

Annual WDI and WGI historical values come from revised current vintages; BACI reference-year release availability is not verified. The exercise is retrospective lag-based, not a real-time forecast evaluation. The one-year lag panel yields a different ex-ante universe and its neural performance sensitivity is not run.

Saved baseline E1 predictions were reconciled against `audit_v2/corrected_exante_forecast_predictions.parquet` over all 62,238 known outcome keys per model. Persistence, PPML, XGBoost, and LightGBM matched exactly; Random Forest's maximum seed-mean difference was 0.00005 USD (floating-point serialization scale). See `results/baseline_reconciliation.csv`.

## Models and evaluation

MLP uses training-fitted median imputation/missing indicators, training-only scaling, and train-fitted one-hot destination/HS2 identity. TabM uses the official `tabm` distribution, k=8, train-fitted categorical cardinalities with unknown index 0; classification aggregates member probabilities after sigmoid, regression averages member log-scale outputs before inversion. Both train on identical declared features and chronological folds.

Hyperparameters and epochs are selected on inner chronological validation only. Neural regression is trained on log1p USD and uses a validation-selected multiplicative calibration for the USD target. Predictions are saved on log and USD scales; the calibration is not claimed to recover an unbiased conditional mean.

## Statistical assumptions

Seeds describe algorithmic variability and are averaged within economic evaluation units. E2 and E3 macro comparisons use HS2-block bootstrap/sign-flip procedures; E1 uses destination-block bootstrap and destination-level sign-rank tests. E3 also reports a destination-block bootstrap that re-ranks weighted candidates jointly across pools. These procedures address cross-sectional dependence imperfectly and do not provide precise uncertainty across only three years.

## Scope and interpretation

E3 Task A is binary entry; Task B is existing-market expansion. Two earlier E3 implementations were invalidated and are preserved outside the final-result tables; only the corrected per-HS2-year pools are reported. Missing labels make complete-case metrics conditional and bounds may be wide. Feature ablation uses one year and one seed with full-feature-selected hyperparameters; it is exploratory. No analysis is causal. Market rankings are country–product historical priorities, not firm-level recommendations, because firm capacity, costs, tariffs, and product quality are not modeled.

## Completed and incomplete work

Primary two-year-lag E2, E3, and E1 run completion is determined by saved predictions and the final run manifest. The one-year-lag performance sensitivity is not completed. Any failed seed, missing model, or test status is listed in `FINAL_STATUS.md`; no unfinished analysis is labeled passed.
