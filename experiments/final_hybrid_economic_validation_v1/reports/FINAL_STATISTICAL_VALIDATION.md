# Final statistical validation

Primary uncertainty intervals use paired HS2-block resampling, keeping the three calendar years together within each chapter. The paired sign-flip test uses HS2 blocks as its unit. Holm adjustments are applied within the named metric-specific families. These cross-sectional uncertainty procedures do not provide three-year time-series inference. USD comparisons are especially sensitive to concentration in large flows; raw and development-capped results are both retained.

The 556 missing Task A labels remain unknown. For a fixed candidate pool, fixed predicted Top-10, and binary entry gains, the reported regret bounds enumerate every common completion of unknown labels and are sharp for that fixed-ranking binary regret estimand. They do not bound NDCG under all completions or uncapped USD opportunity. No economically defensible finite USD upper bound is available.

Rank predictions were archived as seed-mean/archived scores rather than candidate-level seed-specific outer scores; a new multi-seed expert retraining was not performed. The existing risk gate is deterministic conditional on those base scores. WDI/WGI are revised vintages.

## Robustness coverage

| analysis                             | task                           |   rows |
|:-------------------------------------|:-------------------------------|-------:|
| HS_section                           | Task_A_new_entry               |   1440 |
| HS_section                           | Task_B_existing_underexploited |   1440 |
| K_and_blend_weight                   | Task_A_new_entry               |   1440 |
| K_and_blend_weight                   | Task_B_existing_underexploited |   1440 |
| alternative_risk_calibration_windows | both                           |      0 |
| destination_identity_ablation        | both                           |      0 |
| gain_binary_50k                      | Task_B_existing_underexploited |    864 |
| gain_log1p_usd                       | Task_B_existing_underexploited |    864 |
| gain_raw_usd                         | Task_B_existing_underexploited |    864 |
| multi_seed_outer_predictions         | both                           |      0 |
| year                                 | Task_A_new_entry               |   1440 |
| year                                 | Task_B_existing_underexploited |   1440 |
