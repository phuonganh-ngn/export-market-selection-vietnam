# Expert routing results

The primary fixed rule chooses LightGBM for strict new-entry pools and PPML for existing-market expansion pools, reflecting prior M1 development evidence. This is exploratory, not an untouched protocol. The adaptive ridge gate predicts each model's future pool regret using historical rolling-origin pool outcomes and pre-origin characteristics.

| task                           | policy                    |   pools |   mean_ndcg10 |   mean_precision10 |   mean_regret10 |   mean_observed_positive_usd_top10 |
|:-------------------------------|:--------------------------|--------:|--------------:|-------------------:|----------------:|-----------------------------------:|
| Task_A_new_entry               | ADAPTIVE_ROUTING          |     288 |      0.281714 |          0.0395833 |        0.458333 |                   445913           |
| Task_A_new_entry               | HISTORICAL_WINNER_ROUTING |     288 |      0.280091 |          0.0378472 |        0.475694 |                   351834           |
| Task_A_new_entry               | STATIC_TASK_ROUTING       |     288 |      0.294694 |          0.0395833 |        0.458333 |                   546676           |
| Task_B_existing_underexploited | ADAPTIVE_ROUTING          |     288 |      0.579031 |          0.482102  |        2.25     |                        1.99635e+07 |
| Task_B_existing_underexploited | HISTORICAL_WINNER_ROUTING |     288 |      0.592524 |          0.484532  |        2.15278  |                        2.13721e+07 |
| Task_B_existing_underexploited | STATIC_TASK_ROUTING       |     288 |      0.592524 |          0.484532  |        2.15278  |                        2.13721e+07 |

| task                           |   outcome_year | policy           |   selected_lightgbm_share |   selected_ppml_share |   pool_count |
|:-------------------------------|---------------:|:-----------------|--------------------------:|----------------------:|-------------:|
| Task_A_new_entry               |           2022 | ADAPTIVE_ROUTING |                 0.489583  |              0.510417 |           96 |
| Task_A_new_entry               |           2023 | ADAPTIVE_ROUTING |                 0.0104167 |              0.989583 |           96 |
| Task_A_new_entry               |           2024 | ADAPTIVE_ROUTING |                 0.697917  |              0.302083 |           96 |
| Task_B_existing_underexploited |           2022 | ADAPTIVE_ROUTING |                 0.25      |              0.75     |           96 |
| Task_B_existing_underexploited |           2023 | ADAPTIVE_ROUTING |                 0.385417  |              0.614583 |           96 |
| Task_B_existing_underexploited |           2024 | ADAPTIVE_ROUTING |                 0.25      |              0.75     |           96 |

`T3_expert_selection_frequency.csv` reports the routed expert share. The rank-average blend is formed from within-pool percentiles, avoiding direct averaging of raw scores on different scales. Neither adaptive routing nor blending is causal evidence.

The gate's hindsight agreement share and the association between model disagreement and subsequent regret are reported in `T4_disagreement_diagnostics.csv`. Hindsight agreement is descriptive and not available at the decision origin.

Historical OOF model fits completed: 12 model-folds across 4 outcome years. See `results/historical_oof_model_diagnostics.csv`.
