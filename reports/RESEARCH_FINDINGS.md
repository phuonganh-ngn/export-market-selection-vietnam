# Research findings

1. **Does ML outperform gravity?** Best ML (RandomForest) MAE USD 9,062,253.98; PPML MAE USD 7,857,280.40; difference (ML−PPML) USD 1,204,973.58.

2. **Which model performs best overall?** Persistence by pooled MAE; product-specific heterogeneity is in E1.

3. **Does performance differ across products?** See per-HS2 results; no uniform winner is assumed.

4. **Can ML discover new destinations?** Best discovery model by macro NDCG@10: LightGBM. Pool-level prevalence and event counts are in E2.

5. **Does forecast accuracy translate into market selection?** Lowest pooled MAE: Persistence; highest macro NDCG@10: Gravity_PPML_increment. Same model family wins both tasks: False.

6. **Which predictors contribute most?** Highest aggregated mean absolute SHAP predictors: log_lagged_export_value, log_destination_import_demand, historical_export_volatility, log_gdp, government_effectiveness. These are predictive associations, not causal effects.

7. **Is clustering useful?** Equal-cluster mean MAE is USD 5,737,950.24 cluster-specific vs USD 5,804,094.35 pooled (difference USD -66,144.11); mean within-cluster NDCG@10 is 0.481 vs 0.491. The larger K-means cluster NDCG falls from 0.491 to 0.481, while the small near-zero-flow cluster has undefined NDCG; segmentation shows no consistent decision-ranking gain.

8. **Are findings statistically and economically meaningful?** Read paired confidence intervals and Holm-adjusted tests in `confidence_intervals.csv` and `hypothesis_tests.csv`; three outer years and country/product dependence limit precision.

9. **Which conclusions survive robustness?** See `ROBUSTNESS.md`; only completed checks count as evidence.

Negative and null findings are retained in the result tables. No causal economic benefit is claimed.
