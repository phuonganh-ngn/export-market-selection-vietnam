# Feature-level XAI results

## Model fidelity

All 2022, 2023 and 2024 Task A candidate rows were checked against the saved E2/E6 score archive and the final outer score archive. The strict-zero candidate universe contains 29,737 rows in 288 HS2-year pools, with 29,181 observed labels and 556 unknown labels. All 21/21 prediction/rank comparisons are exact; the worst score error is 0. The largest ensemble SHAP additivity residual is 5.69e-09. The PPML log-link identity has maximum error 0.

The ensemble LightGBM explanations are on the positive-class probability scale. The five fitted seed models are exact and their averaged probability attributions reconstruct the saved mean score. PPML terms are on eta = log(expected exports in USD millions); they are not dollar contributions. PPML estimator objects were archived; their preprocessing objects were deterministically reconstructed from the original code and training folds.

## Global LightGBM attributions

On the outcome-blind candidate sample, the largest mean absolute feature attributions are log_destination_import_demand (0.1120); log_lagged_export_value (0.0349); gdp_growth__missing_indicator (0.0252); destination_supplier_concentration (0.0196); log_population (0.0143). The leading aggregated groups are trade_demand (0.1254); historical_trade (0.0419); predictor_missingness (0.0321); product_indicator (0.0309); macroeconomic_size (0.0289). These are predictive SHAP contributions relative to the declared background, not causal effects. Destination identifiers and correlated macroeconomic measures require particular caution.

## PPML coefficient contributions

The PPML grouped signed contribution means on eta scale are trade_demand (+5.080); macroeconomic_size (+2.138); geography (-0.989); predictor_missingness (+0.353); purchasing_power (+0.219). Group and feature contributions are `x_j beta_j` terms from the fitted linear predictor. They reconstruct eta together with the intercept, but they are not additive USD effects or causal coefficients.

## Decision-level results

The swap archive contains 3,502 candidate-pool members across the union of the LightGBM and Hybrid Top-10 lists. There are 622 deterministic boundary-pair comparisons. Across all 288 pools, the Hybrid adds 25 observed successful entries and displaces 13; net change is +0.0417 per pool. The partial observed qualified-export difference is $11,274 per pool; unknown outcomes remain unknown. The NDCG bridge allocates +0.029223 to membership and +0.007962 to ordering under the LightGBM-order bridge, using the same relevance denominator.

There are 51 pools where Hybrid ranks 10 and 11 have exactly equal fused scores; ISO3 tie-breaking determines membership in these pools. In the Bulgaria HS2 27, 2022 case, Bulgaria and Iran tie at the Hybrid boundary and Bulgaria is selected because BGR precedes IRN alphabetically, even though the PPML rank term favors Bulgaria and LightGBM rank term favors Iran by equal opposing amounts. The complete membership swap is recorded in the candidate archive.

Authentic contrastive cases are in `contrastive_case_studies.csv`; full pairwise feature contributions are in `feature_to_rank_explanations.csv` and Parquet. Outcomes describe later observed trade, not causal replacement effects. All outcome-linked analysis is exploratory for previously inspected 2022–2024 years.
