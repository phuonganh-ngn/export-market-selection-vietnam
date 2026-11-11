# E4 — Numerical explainability

SHAP, permutation importance, feature-group ablation and numerical ALE are predictive diagnostics, not causal estimates. The explanatory sample is a fixed-seed random sample from out-of-time predictions. Feature-group ablations use the pre-specified seed 11 to limit computation; the primary E1 comparisons retain all five seeds.

## Global SHAP

| feature                                            | feature_group             |   mean_abs_shap |   sd_across_years |   evaluation_years |   sample_n |
|:---------------------------------------------------|:--------------------------|----------------:|------------------:|-------------------:|-----------:|
| log_lagged_export_value                            | historical_trade          |         4.528   |          0.07715  |                  3 |       6000 |
| log_destination_import_demand                      | trade_demand              |         0.5183  |          0.01751  |                  3 |       6000 |
| historical_export_volatility                       | historical_trade          |         0.2278  |          0.001766 |                  3 |       6000 |
| log_gdp                                            | macroeconomic_size        |         0.2109  |          0.004403 |                  3 |       6000 |
| government_effectiveness                           | governance                |         0.1334  |          0.02332  |                  3 |       6000 |
| historical_export_presence                         | historical_trade          |         0.0948  |          0.06223  |                  3 |       6000 |
| destination_supplier_concentration                 | trade_demand              |         0.09352 |          0.01368  |                  3 |       6000 |
| historical_export_volatility_missing_indicator     | historical_trade          |         0.08392 |          0.01351  |                  3 |       6000 |
| regulatory_quality                                 | governance                |         0.06603 |          0.00794  |                  3 |       6000 |
| log_population                                     | macroeconomic_size        |         0.05779 |          0.009783 |                  3 |       6000 |
| destination fixed effects                          | destination_fixed_effects |         0.05264 |          0.00206  |                  3 |       6000 |
| log_distance                                       | geography                 |         0.04905 |          0.005001 |                  3 |       6000 |
| gdp_growth                                         | macroeconomic_size        |         0.04597 |          0.009996 |                  3 |       6000 |
| HS2 fixed effects                                  | product                   |         0.04416 |          0.002579 |                  3 |       6000 |
| destination_import_demand_growth_missing_indicator | trade_demand              |         0.04354 |          0.004162 |                  3 |       6000 |

## Feature ablation

| model        | ablation                   |   evaluation_year | dropped_group      |   mae_usd |   rmse_usd |   mae_log1p |   wape |   mae_zero_usd |   mae_positive_usd |     n |   zero_share |   seeds |
|:-------------|:---------------------------|------------------:|:-------------------|----------:|-----------:|------------:|-------:|---------------:|-------------------:|------:|-------------:|--------:|
| RandomForest | full                       |              2024 | none               | 1.094e+07 |  3.228e+08 |       2.062 | 0.4567 |      1.145e+06 |          2.633e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_macroeconomic_size |              2024 | macroeconomic_size | 1.113e+07 |  3.234e+08 |       2.013 | 0.4646 |      1.168e+06 |          2.678e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_purchasing_power   |              2024 | purchasing_power   | 1.094e+07 |  3.227e+08 |       2.06  | 0.4566 |      1.139e+06 |          2.633e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_geography          |              2024 | geography          | 1.095e+07 |  3.228e+08 |       2.077 | 0.4569 |      1.148e+06 |          2.634e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_trade_demand       |              2024 | trade_demand       | 1.11e+07  |  3.36e+08  |       2.084 | 0.4631 |      1.08e+06  |          2.682e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_governance         |              2024 | governance         | 1.096e+07 |  3.233e+08 |       2.096 | 0.4575 |      1.194e+06 |          2.63e+07  | 20755 |       0.6109 |       1 |
| RandomForest | without_digitalization     |              2024 | digitalization     | 1.094e+07 |  3.228e+08 |       2.066 | 0.4564 |      1.145e+06 |          2.631e+07 | 20755 |       0.6109 |       1 |
| RandomForest | without_historical_trade   |              2024 | historical_trade   | 1.673e+07 |  3.724e+08 |       2.799 | 0.6982 |      8.518e+05 |          4.166e+07 | 20755 |       0.6109 |       1 |

No figures were generated.
