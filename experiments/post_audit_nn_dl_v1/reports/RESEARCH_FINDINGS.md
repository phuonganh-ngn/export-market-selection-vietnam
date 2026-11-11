# Research findings

**Status: COMPLETE_REDUCED_SPECIFICATION.**

## H1 — Neural market discovery

Observed macro PR-AUC winner: LightGBM at 0.3827; MLP and TabM are reported beside LightGBM and PPML in `results/E2_neural_discovery.csv`. Paired metric-specific inference is in `results/statistical_comparisons_E2.csv`; interpret adjusted intervals and p-values, not winner labels alone.

| model                      |   pr_auc |   precision_at_10 |   ndcg_at_10 |   valid_pools |   valid_pr_auc_pools |   valid_ndcg_at_10_pools |
|:---------------------------|---------:|------------------:|-------------:|--------------:|---------------------:|-------------------------:|
| LightGBM                   |   0.3827 |            0.2465 |       0.4495 |           288 |                  281 |                      281 |
| XGBoost                    |   0.3812 |            0.2458 |       0.4479 |           288 |                  281 |                      281 |
| Gravity_PPML               |   0.3773 |            0.2309 |       0.4368 |           288 |                  281 |                      281 |
| RandomForest               |   0.3632 |            0.237  |       0.4257 |           288 |                  281 |                      281 |
| TabM                       |   0.3576 |            0.2297 |       0.4168 |           288 |                  281 |                      281 |
| MLP                        |   0.3451 |            0.2278 |       0.4056 |           288 |                  281 |                      281 |
| LogisticRegression         |   0.3433 |            0.2285 |       0.4071 |           288 |                  281 |                      281 |
| World_import_demand        |   0.2526 |            0.1674 |       0.2859 |           288 |                  281 |                      281 |
| Historical_entry_frequency |   0.0391 |            0.0372 |       0.0456 |           288 |                  281 |                      281 |

## H2 — Neural market ranking

Task_A_new_entry: Gravity_PPML ndcg10_entry_binary=0.4370. Task_B_existing_underexploited: Gravity_PPML ndcg10_capped_grade=0.5925. 
Task A (new entry) uses binary entry NDCG; Task B (existing underexploited markets) uses capped-grade incremental-export NDCG. Log1p/raw-gain metrics and destination-block sensitivity are also reported. Pool-specific sample sizes and no-relevance NA rules are explicit.

| ranking_model            | task                            | ndcg10_entry_binary   |   ndcg10_capped_grade |   ndcg10_log1p_gain |   observed_incremental_usd_top10_full_rank_partial |   valid_product_year_pools |   valid_ndcg10_entry_binary_pools |   valid_ndcg10_capped_grade_pools |
|:-------------------------|:--------------------------------|:----------------------|----------------------:|--------------------:|---------------------------------------------------:|---------------------------:|----------------------------------:|----------------------------------:|
| GDP                      | Task_A_new_entry                | 0.2931                |                0.2823 |              0.384  |                                        1.07595e+06 |                        288 |                               281 |                               112 |
| GDP                      | Task_B_all_existing_sensitivity | NA                    |                0.5073 |              0.5853 |                                        6.56978e+08 |                        288 |                                 0 |                               283 |
| GDP                      | Task_B_existing_underexploited  | NA                    |                0.531  |              0.6191 |                                        1.79006e+07 |                        288 |                                 0 |                               241 |
| Gravity_PPML             | Task_A_new_entry                | 0.4370                |                0.3429 |              0.4591 |                                        1.51122e+06 |                        288 |                               281 |                               112 |
| Gravity_PPML             | Task_B_all_existing_sensitivity | NA                    |                0.5216 |              0.5809 |                                        5.04349e+08 |                        288 |                                 0 |                               283 |
| Gravity_PPML             | Task_B_existing_underexploited  | NA                    |                0.5925 |              0.6312 |                                        2.13721e+07 |                        288 |                                 0 |                               241 |
| Historical_export_trend  | Task_A_new_entry                | 0.2287                |                0.1394 |              0.3059 |                                   421115           |                        288 |                               281 |                               112 |
| Historical_export_trend  | Task_B_all_existing_sensitivity | NA                    |                0.0882 |              0.2752 |                                        1.27812e+07 |                        288 |                                 0 |                               283 |
| Historical_export_trend  | Task_B_existing_underexploited  | NA                    |                0.1368 |              0.3804 |                                        3.99295e+06 |                        288 |                                 0 |                               241 |
| Historical_import_demand | Task_A_new_entry                | 0.2859                |                0.3235 |              0.376  |                                        1.26017e+06 |                        288 |                               281 |                               112 |
| Historical_import_demand | Task_B_all_existing_sensitivity | NA                    |                0.4927 |              0.568  |                                        6.92148e+08 |                        288 |                                 0 |                               283 |
| Historical_import_demand | Task_B_existing_underexploited  | NA                    |                0.5812 |              0.6268 |                                        1.88088e+07 |                        288 |                                 0 |                               241 |
| LightGBM                 | Task_A_new_entry                | 0.3029                |                0.3052 |              0.4601 |                                        1.13016e+06 |                        288 |                               281 |                               112 |
| LightGBM                 | Task_B_all_existing_sensitivity | NA                    |                0.5043 |              0.5638 |                                        2.09363e+08 |                        288 |                                 0 |                               283 |
| LightGBM                 | Task_B_existing_underexploited  | NA                    |                0.5406 |              0.6216 |                                        1.79641e+07 |                        288 |                                 0 |                               241 |
| MLP                      | Task_A_new_entry                | 0.2801                |                0.2854 |              0.4664 |                                        1.15668e+06 |                        288 |                               281 |                               112 |
| MLP                      | Task_B_all_existing_sensitivity | NA                    |                0.291  |              0.5159 |                                        2.87644e+08 |                        288 |                                 0 |                               283 |
| MLP                      | Task_B_existing_underexploited  | NA                    |                0.3654 |              0.5867 |                                        1.20906e+07 |                        288 |                                 0 |                               241 |
| MLP_entry_probability    | Task_A_new_entry                | 0.4056                |                0.3398 |              0.4611 |                                        1.62218e+06 |                        288 |                               281 |                               112 |
| Persistence              | Task_A_new_entry                | 0.0456                |                0.0229 |              0.0916 |                                   112603           |                        288 |                               281 |                               112 |
| Persistence              | Task_B_all_existing_sensitivity | NA                    |                0.1056 |              0.359  |                                        5.42662e+07 |                        288 |                                 0 |                               283 |
| Persistence              | Task_B_existing_underexploited  | NA                    |                0.153  |              0.4359 |                                        5.04516e+06 |                        288 |                                 0 |                               241 |
| RandomForest             | Task_A_new_entry                | 0.2267                |                0.1567 |              0.4082 |                                   789908           |                        288 |                               281 |                               112 |
| RandomForest             | Task_B_all_existing_sensitivity | NA                    |                0.3367 |              0.4745 |                                        4.21411e+08 |                        288 |                                 0 |                               283 |
| RandomForest             | Task_B_existing_underexploited  | NA                    |                0.247  |              0.5206 |                                        8.58903e+06 |                        288 |                                 0 |                               241 |
| TabM                     | Task_A_new_entry                | 0.2935                |                0.2858 |              0.4825 |                                        1.09724e+06 |                        288 |                               281 |                               112 |
| TabM                     | Task_B_all_existing_sensitivity | NA                    |                0.2586 |              0.4919 |                                        3.21086e+08 |                        288 |                                 0 |                               283 |
| TabM                     | Task_B_existing_underexploited  | NA                    |                0.3324 |              0.5733 |                                        9.74762e+06 |                        288 |                                 0 |                               241 |
| TabM_entry_probability   | Task_A_new_entry                | 0.4168                |                0.3426 |              0.4794 |                                        1.65396e+06 |                        288 |                               281 |                               112 |
| XGBoost                  | Task_A_new_entry                | 0.2944                |                0.2804 |              0.4598 |                                        1.04277e+06 |                        288 |                               281 |                               112 |
| XGBoost                  | Task_B_all_existing_sensitivity | NA                    |                0.4942 |              0.5546 |                                        2.62192e+08 |                        288 |                                 0 |                               283 |
| XGBoost                  | Task_B_existing_underexploited  | NA                    |                0.5092 |              0.6168 |                                        1.66737e+07 |                        288 |                                 0 |                               241 |

## H3 — Export value forecasting

USD-MAE winner: Persistence (6,493,933 USD). Discovery PR-AUC winner: LightGBM (0.3827).

| model        |    mae_usd |    rmse_usd |   mae_log1p |   wape |   mae_zero_flow_usd |   mae_positive_flow_usd | mae_usd_seed_sd   | rmse_usd_seed_sd   | mae_log1p_seed_sd   | wape_seed_sd   | mae_zero_flow_usd_seed_sd   | mae_positive_flow_usd_seed_sd   |     n |   seeds |   macro_hs2_year_mae_usd |
|:-------------|-----------:|------------:|------------:|-------:|--------------------:|------------------------:|:------------------|:-------------------|:--------------------|:---------------|:----------------------------|:--------------------------------|------:|--------:|-------------------------:|
| Persistence  |  6,493,933 | 137,126,652 |      1.4804 | 0.2962 |              399454 |              14,223,142 | NA                | NA                 | NA                  | NA             | NA                          | NA                              | 62238 |       1 |                6,298,260 |
| Gravity_PPML |  7,857,280 | 155,718,524 |      4.8028 | 0.3584 |              429246 |              17,277,747 | NA                | NA                 | NA                  | NA             | NA                          | NA                              | 62238 |       1 |                7,630,959 |
| RandomForest |  9,083,048 | 278,022,756 |      1.7577 | 0.4143 |              422682 |              20,066,395 | 74,688.3          | 587,769.6          | 0.0027              | 0.0034         | 8,961.3                     | 159,301.0                       | 62238 |       5 |                8,800,065 |
| MLP          | 10,285,795 | 241,626,223 |      1.6732 | 0.4692 |              330488 |              22,911,427 | 585,194.3         | 34,026,677         | 0.0188              | 0.0267         | 96,117.3                    | 1,236,491                       | 62238 |       5 |                9,967,765 |
| TabM         | 11,069,578 | 254,710,761 |      1.6978 | 0.5049 |              367761 |              24,641,959 | 1,879,316         | 69,101,889         | 0.0204              | 0.0857         | 43,083.2                    | 4,218,474                       | 62238 |       5 |               10,727,244 |
| XGBoost      | 13,617,138 | 465,065,506 |      1.8113 | 0.6211 |              355926 |              30,435,423 | 79,426.9          | 3,707,538          | 0.0039              | 0.0036         | 17,064.8                    | 170,950.3                       | 62238 |       5 |               13,178,244 |
| LightGBM     | 14,473,854 | 478,610,629 |      1.7875 | 0.6602 |              373194 |              32,356,753 | 0.0000            | 0.0000             | 0.0000              | 0.0000         | 0.0000                      | 0.0000                          | 62238 |       5 |               14,006,549 |

USD MAE is the primary criterion. Log1p MAE is a separate scale and does not supersede USD performance. See matched observed-label comparisons and paired destination-block inference.

## H4 — Task-specific neural value

E2, E3 Task A, E3 Task B, and E1 are separate estimands. Differences between a forecasting winner and a ranking winner alone do not prove a lack of association; matched-population MAE/NDCG values and paired task-specific MAE intervals are in `results/E3_matched_forecast_ranking.csv` and `results/statistical_comparisons_E3_matched_mae.csv`.

| ranking_model   |   forecast_mae_usd | forecast_mae_seed_sd   |   seeds |   forecast_macro_hs2_year_mae_usd | forecast_macro_seed_sd   |   ndcg10 | ndcg10_seed_sd   |   valid_rank_pools |   ranking_seeds | task                           |   n_observed_candidates | forecast_metric          | ranking_metric                        |   across_model_spearman_mae_vs_ndcg |
|:----------------|-------------------:|:-----------------------|--------:|----------------------------------:|:-------------------------|---------:|:-----------------|-------------------:|----------------:|:-------------------------------|------------------------:|:-------------------------|:--------------------------------------|------------------------------------:|
| Gravity_PPML    |    33821.8         | NA                     |       1 |                    33998.9        | NA                       |   0.437  | NA               |                281 |               1 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| LightGBM        |    22640.4         | NA                     |       1 |                    22927.6        | NA                       |   0.3029 | NA               |                281 |               1 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| MLP             |    23199.9         | 487.4347               |       5 |                    23611.9        | 583.5298                 |   0.2801 | 0.0126           |                281 |               5 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| Persistence     |    22763.5         | NA                     |       1 |                    23046.3        | NA                       |   0.0456 | NA               |                281 |               1 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| RandomForest    |    22689.7         | NA                     |       1 |                    22985.8        | NA                       |   0.2267 | NA               |                281 |               1 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| TabM            |    22710           | 9.8203                 |       5 |                    23021.9        | 14.0886                  |   0.2935 | 0.0070           |                281 |               5 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| XGBoost         |    22635.5         | NA                     |       1 |                    22924.2        | NA                       |   0.2944 | NA               |                281 |               1 | Task_A_new_entry               |                   40572 | MAE USD; lower is better | ndcg10_entry_binary; higher is better |                             -0.0357 |
| Gravity_PPML    |        1.38505e+06 | NA                     |       1 |                        1.2233e+06 | NA                       |   0.5925 | NA               |                241 |               1 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| LightGBM        |   899517           | NA                     |       1 |                   769036          | NA                       |   0.5406 | NA               |                241 |               1 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| MLP             |   965806           | 31,699.4               |       5 |                   810477          | 21,742.0                 |   0.3654 | 0.0320           |                241 |               5 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| Persistence     |   816160           | NA                     |       1 |                   708192          | NA                       |   0.153  | NA               |                241 |               1 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| RandomForest    |   860550           | NA                     |       1 |                   739743          | NA                       |   0.247  | NA               |                241 |               1 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| TabM            |   947778           | 50,537.1               |       5 |                   800148          | 34,118.7                 |   0.3324 | 0.0404           |                241 |               5 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |
| XGBoost         |   883555           | NA                     |       1 |                   756860          | NA                       |   0.5092 | NA               |                241 |               1 | Task_B_existing_underexploited |                   12962 | MAE USD; lower is better | ndcg10_capped_grade; higher is better |                              0.7143 |

## H5 — Information value and heterogeneity

Feature ablation rows: 120; heterogeneity rows: 122,543. The feature ablation is an exploratory 2024 single-seed analysis with F4-selected hyperparameters held fixed; it does not establish causal effects or optimal performance for every feature set. Heterogeneity estimates are descriptive and small-cell uncertainty is limited.

## Paired comparative conclusions

E2 PR-AUC: MLP vs LightGBM: Δ=-0.0376, pHolm=0.024, statistically distinguishable after Holm; favors LightGBM; MLP vs Gravity_PPML: Δ=-0.0322, pHolm=0.006, statistically distinguishable after Holm; favors Gravity_PPML; TabM vs LightGBM: Δ=-0.0252, pHolm=0.154, not statistically distinguishable after Holm; TabM vs Gravity_PPML: Δ=-0.0197, pHolm=0.154, not statistically distinguishable after Holm; TabM vs MLP: Δ=0.0125, pHolm=0.024, statistically distinguishable after Holm; favors TabM

E3 Task A binary-entry NDCG@10: TabM vs Gravity_PPML: Δ=-0.1435, pHolm=0.001, statistically distinguishable after Holm; favors Gravity_PPML; TabM vs LightGBM: Δ=-0.0094, pHolm=0.394, not statistically distinguishable after Holm; MLP vs Gravity_PPML: Δ=-0.1569, pHolm=0.001, statistically distinguishable after Holm; favors Gravity_PPML; MLP vs LightGBM: Δ=-0.0228, pHolm=0.134, not statistically distinguishable after Holm; TabM vs MLP: Δ=0.0134, pHolm=0.137, not statistically distinguishable after Holm

E3 Task B capped-gain NDCG@10: TabM vs Gravity_PPML: Δ=-0.2601, pHolm=0.001, statistically distinguishable after Holm; favors Gravity_PPML; TabM vs LightGBM: Δ=-0.2081, pHolm=0.001, statistically distinguishable after Holm; favors LightGBM; MLP vs Gravity_PPML: Δ=-0.2272, pHolm=0.001, statistically distinguishable after Holm; favors Gravity_PPML; MLP vs LightGBM: Δ=-0.1752, pHolm=0.001, statistically distinguishable after Holm; favors LightGBM; TabM vs MLP: Δ=-0.0329, pHolm=0.001, statistically distinguishable after Holm; favors MLP

E1 all-candidate USD MAE: MLP vs Persistence: Δ=3,791,862, pHolm=0.000, statistically distinguishable after Holm; favors MLP; TabM vs Persistence: Δ=4,575,646, pHolm=0.000, statistically distinguishable after Holm; favors TabM; MLP vs Gravity_PPML: Δ=2,428,515, pHolm=0.548, not statistically distinguishable after Holm; TabM vs Gravity_PPML: Δ=3,212,298, pHolm=0.308, not statistically distinguishable after Holm

E3 task-matched USD MAE: MLP vs Persistence: Δ=436.4344, pHolm=0.477, not statistically distinguishable after Holm; TabM vs Persistence: Δ=-53.4692, pHolm=0.432, not statistically distinguishable after Holm; MLP vs Gravity_PPML: Δ=-10,621.9, pHolm=0.000, statistically distinguishable after Holm; favors Gravity_PPML; TabM vs Gravity_PPML: Δ=-11,111.8, pHolm=0.000, statistically distinguishable after Holm; favors Gravity_PPML; MLP vs Persistence: Δ=149,646.0, pHolm=0.000, statistically distinguishable after Holm; favors MLP; TabM vs Persistence: Δ=131,618.4, pHolm=0.000, statistically distinguishable after Holm; favors TabM; MLP vs Gravity_PPML: Δ=-419,247.6, pHolm=0.000, statistically distinguishable after Holm; favors Gravity_PPML; TabM vs Gravity_PPML: Δ=-437,275.3, pHolm=0.000, statistically distinguishable after Holm; favors Gravity_PPML

## Inference and limits

The frozen ex-ante universe has 62,797 rows, 62,238 known outcomes and 559 unknown labels. Complete-case performance is conditional on label availability. Current WDI/WGI vintages are revised, the one-year-lag model sensitivity was not run because the population changes, and three evaluation years do not support precise temporal generalization.

## Feature information and market groups

| task                           | model   | feature_set           | metric                      |   mean_across_hs2 |   valid_hs2_chapters |
|:-------------------------------|:--------|:----------------------|:----------------------------|------------------:|---------------------:|
| Task_A_new_entry               | MLP     | F0_history_only       | NDCG@10_capped              |       0.2542      |                   96 |
| Task_A_new_entry               | MLP     | F0_history_only       | NDCG@10_entry_binary        |       0.301       |                   96 |
| Task_A_new_entry               | MLP     | F0_history_only       | NDCG@10_log1p_gain          |       0.4061      |                   96 |
| Task_A_new_entry               | MLP     | F0_history_only       | NDCG@10_raw_gain            |       0.2837      |                   96 |
| Task_A_new_entry               | MLP     | F0_history_only       | Top10_known_incremental_USD |  582777           |                   96 |
| Task_A_new_entry               | MLP     | F1_history_plus_size  | NDCG@10_capped              |       0.2259      |                   96 |
| Task_A_new_entry               | MLP     | F1_history_plus_size  | NDCG@10_entry_binary        |       0.3105      |                   96 |
| Task_A_new_entry               | MLP     | F1_history_plus_size  | NDCG@10_log1p_gain          |       0.4189      |                   96 |
| Task_A_new_entry               | MLP     | F1_history_plus_size  | NDCG@10_raw_gain            |       0.2884      |                   96 |
| Task_A_new_entry               | MLP     | F1_history_plus_size  | Top10_known_incremental_USD |  538884           |                   96 |
| Task_A_new_entry               | MLP     | F2_plus_import_demand | NDCG@10_capped              |       0.2277      |                   96 |
| Task_A_new_entry               | MLP     | F2_plus_import_demand | NDCG@10_entry_binary        |       0.2922      |                   96 |
| Task_A_new_entry               | MLP     | F2_plus_import_demand | NDCG@10_log1p_gain          |       0.4167      |                   96 |
| Task_A_new_entry               | MLP     | F2_plus_import_demand | NDCG@10_raw_gain            |       0.2937      |                   96 |
| Task_A_new_entry               | MLP     | F2_plus_import_demand | Top10_known_incremental_USD |  567650           |                   96 |
| Task_A_new_entry               | MLP     | F3_plus_governance    | NDCG@10_capped              |       0.2601      |                   96 |
| Task_A_new_entry               | MLP     | F3_plus_governance    | NDCG@10_entry_binary        |       0.3227      |                   96 |
| Task_A_new_entry               | MLP     | F3_plus_governance    | NDCG@10_log1p_gain          |       0.4354      |                   96 |
| Task_A_new_entry               | MLP     | F3_plus_governance    | NDCG@10_raw_gain            |       0.3087      |                   96 |
| Task_A_new_entry               | MLP     | F3_plus_governance    | Top10_known_incremental_USD |  521407           |                   96 |
| Task_A_new_entry               | MLP     | F4_full               | NDCG@10_capped              |       0.3475      |                   96 |
| Task_A_new_entry               | MLP     | F4_full               | NDCG@10_entry_binary        |       0.3139      |                   96 |
| Task_A_new_entry               | MLP     | F4_full               | NDCG@10_log1p_gain          |       0.435       |                   96 |
| Task_A_new_entry               | MLP     | F4_full               | NDCG@10_raw_gain            |       0.3163      |                   96 |
| Task_A_new_entry               | MLP     | F4_full               | Top10_known_incremental_USD |  611671           |                   96 |
| Task_A_new_entry               | TabM    | F0_history_only       | NDCG@10_capped              |       0.2979      |                   96 |
| Task_A_new_entry               | TabM    | F0_history_only       | NDCG@10_entry_binary        |       0.3293      |                   96 |
| Task_A_new_entry               | TabM    | F0_history_only       | NDCG@10_log1p_gain          |       0.4182      |                   96 |
| Task_A_new_entry               | TabM    | F0_history_only       | NDCG@10_raw_gain            |       0.3015      |                   96 |
| Task_A_new_entry               | TabM    | F0_history_only       | Top10_known_incremental_USD |  633425           |                   96 |
| Task_A_new_entry               | TabM    | F1_history_plus_size  | NDCG@10_capped              |       0.3416      |                   96 |
| Task_A_new_entry               | TabM    | F1_history_plus_size  | NDCG@10_entry_binary        |       0.358       |                   96 |
| Task_A_new_entry               | TabM    | F1_history_plus_size  | NDCG@10_log1p_gain          |       0.4324      |                   96 |
| Task_A_new_entry               | TabM    | F1_history_plus_size  | NDCG@10_raw_gain            |       0.3198      |                   96 |
| Task_A_new_entry               | TabM    | F1_history_plus_size  | Top10_known_incremental_USD |  683072           |                   96 |
| Task_A_new_entry               | TabM    | F2_plus_import_demand | NDCG@10_capped              |       0.382       |                   96 |
| Task_A_new_entry               | TabM    | F2_plus_import_demand | NDCG@10_entry_binary        |       0.3488      |                   96 |
| Task_A_new_entry               | TabM    | F2_plus_import_demand | NDCG@10_log1p_gain          |       0.4304      |                   96 |
| Task_A_new_entry               | TabM    | F2_plus_import_demand | NDCG@10_raw_gain            |       0.3301      |                   96 |
| Task_A_new_entry               | TabM    | F2_plus_import_demand | Top10_known_incremental_USD |  660131           |                   96 |
| Task_A_new_entry               | TabM    | F3_plus_governance    | NDCG@10_capped              |       0.4053      |                   96 |
| Task_A_new_entry               | TabM    | F3_plus_governance    | NDCG@10_entry_binary        |       0.3577      |                   96 |
| Task_A_new_entry               | TabM    | F3_plus_governance    | NDCG@10_log1p_gain          |       0.4434      |                   96 |
| Task_A_new_entry               | TabM    | F3_plus_governance    | NDCG@10_raw_gain            |       0.3346      |                   96 |
| Task_A_new_entry               | TabM    | F3_plus_governance    | Top10_known_incremental_USD |  683407           |                   96 |
| Task_A_new_entry               | TabM    | F4_full               | NDCG@10_capped              |       0.345       |                   96 |
| Task_A_new_entry               | TabM    | F4_full               | NDCG@10_entry_binary        |       0.3441      |                   96 |
| Task_A_new_entry               | TabM    | F4_full               | NDCG@10_log1p_gain          |       0.4361      |                   96 |
| Task_A_new_entry               | TabM    | F4_full               | NDCG@10_raw_gain            |       0.3255      |                   96 |
| Task_A_new_entry               | TabM    | F4_full               | Top10_known_incremental_USD |  641380           |                   96 |
| Task_B_existing_underexploited | MLP     | F0_history_only       | NDCG@10_capped              |       0.3656      |                   96 |
| Task_B_existing_underexploited | MLP     | F0_history_only       | NDCG@10_log1p_gain          |       0.5172      |                   96 |
| Task_B_existing_underexploited | MLP     | F0_history_only       | NDCG@10_raw_gain            |       0.378       |                   96 |
| Task_B_existing_underexploited | MLP     | F0_history_only       | Top10_known_incremental_USD |       9.04231e+06 |                   96 |
| Task_B_existing_underexploited | MLP     | F1_history_plus_size  | NDCG@10_capped              |       0.4112      |                   96 |
| Task_B_existing_underexploited | MLP     | F1_history_plus_size  | NDCG@10_log1p_gain          |       0.5126      |                   96 |
| Task_B_existing_underexploited | MLP     | F1_history_plus_size  | NDCG@10_raw_gain            |       0.4054      |                   96 |
| Task_B_existing_underexploited | MLP     | F1_history_plus_size  | Top10_known_incremental_USD |       9.01014e+06 |                   96 |
| Task_B_existing_underexploited | MLP     | F2_plus_import_demand | NDCG@10_capped              |       0.3092      |                   96 |
| Task_B_existing_underexploited | MLP     | F2_plus_import_demand | NDCG@10_log1p_gain          |       0.5232      |                   96 |
| Task_B_existing_underexploited | MLP     | F2_plus_import_demand | NDCG@10_raw_gain            |       0.3336      |                   96 |
| Task_B_existing_underexploited | MLP     | F2_plus_import_demand | Top10_known_incremental_USD |       7.57345e+06 |                   96 |
| Task_B_existing_underexploited | MLP     | F3_plus_governance    | NDCG@10_capped              |       0.4518      |                   96 |
| Task_B_existing_underexploited | MLP     | F3_plus_governance    | NDCG@10_log1p_gain          |       0.5304      |                   96 |
| Task_B_existing_underexploited | MLP     | F3_plus_governance    | NDCG@10_raw_gain            |       0.4213      |                   96 |
| Task_B_existing_underexploited | MLP     | F3_plus_governance    | Top10_known_incremental_USD |       1.01635e+07 |                   96 |
| Task_B_existing_underexploited | MLP     | F4_full               | NDCG@10_capped              |       0.391       |                   96 |
| Task_B_existing_underexploited | MLP     | F4_full               | NDCG@10_log1p_gain          |       0.539       |                   96 |
| Task_B_existing_underexploited | MLP     | F4_full               | NDCG@10_raw_gain            |       0.3821      |                   96 |
| Task_B_existing_underexploited | MLP     | F4_full               | Top10_known_incremental_USD |       8.79378e+06 |                   96 |
| Task_B_existing_underexploited | TabM    | F0_history_only       | NDCG@10_capped              |       0.46        |                   96 |
| Task_B_existing_underexploited | TabM    | F0_history_only       | NDCG@10_log1p_gain          |       0.5149      |                   96 |
| Task_B_existing_underexploited | TabM    | F0_history_only       | NDCG@10_raw_gain            |       0.4573      |                   96 |
| Task_B_existing_underexploited | TabM    | F0_history_only       | Top10_known_incremental_USD |       1.38943e+07 |                   96 |
| Task_B_existing_underexploited | TabM    | F1_history_plus_size  | NDCG@10_capped              |       0.4703      |                   96 |
| Task_B_existing_underexploited | TabM    | F1_history_plus_size  | NDCG@10_log1p_gain          |       0.51        |                   96 |
| Task_B_existing_underexploited | TabM    | F1_history_plus_size  | NDCG@10_raw_gain            |       0.4625      |                   96 |
| Task_B_existing_underexploited | TabM    | F1_history_plus_size  | Top10_known_incremental_USD |       1.18904e+07 |                   96 |
| Task_B_existing_underexploited | TabM    | F2_plus_import_demand | NDCG@10_capped              |       0.4724      |                   96 |
| Task_B_existing_underexploited | TabM    | F2_plus_import_demand | NDCG@10_log1p_gain          |       0.5221      |                   96 |
| Task_B_existing_underexploited | TabM    | F2_plus_import_demand | NDCG@10_raw_gain            |       0.465       |                   96 |
| Task_B_existing_underexploited | TabM    | F2_plus_import_demand | Top10_known_incremental_USD |       1.25996e+07 |                   96 |
| Task_B_existing_underexploited | TabM    | F3_plus_governance    | NDCG@10_capped              |       0.4629      |                   96 |
| Task_B_existing_underexploited | TabM    | F3_plus_governance    | NDCG@10_log1p_gain          |       0.5254      |                   96 |
| Task_B_existing_underexploited | TabM    | F3_plus_governance    | NDCG@10_raw_gain            |       0.4605      |                   96 |
| Task_B_existing_underexploited | TabM    | F3_plus_governance    | Top10_known_incremental_USD |       9.92895e+06 |                   96 |
| Task_B_existing_underexploited | TabM    | F4_full               | NDCG@10_capped              |       0.4811      |                   96 |
| Task_B_existing_underexploited | TabM    | F4_full               | NDCG@10_log1p_gain          |       0.5336      |                   96 |
| Task_B_existing_underexploited | TabM    | F4_full               | NDCG@10_raw_gain            |       0.478       |                   96 |
| Task_B_existing_underexploited | TabM    | F4_full               | Top10_known_incremental_USD |       1.03846e+07 |                   96 |
| discovery                      | MLP     | F0_history_only       | NDCG@10                     |       0.3379      |                   96 |
| discovery                      | MLP     | F0_history_only       | PR_AUC                      |       0.2801      |                   96 |
| discovery                      | MLP     | F0_history_only       | Precision@10                |       0.1667      |                   96 |
| discovery                      | MLP     | F1_history_plus_size  | NDCG@10                     |       0.345       |                   96 |
| discovery                      | MLP     | F1_history_plus_size  | PR_AUC                      |       0.2859      |                   96 |
| discovery                      | MLP     | F1_history_plus_size  | Precision@10                |       0.1667      |                   96 |
| discovery                      | MLP     | F2_plus_import_demand | NDCG@10                     |       0.3654      |                   96 |
| discovery                      | MLP     | F2_plus_import_demand | PR_AUC                      |       0.3072      |                   96 |
| discovery                      | MLP     | F2_plus_import_demand | Precision@10                |       0.174       |                   96 |
| discovery                      | MLP     | F3_plus_governance    | NDCG@10                     |       0.3694      |                   96 |
| discovery                      | MLP     | F3_plus_governance    | PR_AUC                      |       0.3068      |                   96 |
| discovery                      | MLP     | F3_plus_governance    | Precision@10                |       0.1771      |                   96 |
| discovery                      | MLP     | F4_full               | NDCG@10                     |       0.3642      |                   96 |
| discovery                      | MLP     | F4_full               | PR_AUC                      |       0.296       |                   96 |
| discovery                      | MLP     | F4_full               | Precision@10                |       0.1833      |                   96 |
| discovery                      | TabM    | F0_history_only       | NDCG@10                     |       0.3272      |                   96 |
| discovery                      | TabM    | F0_history_only       | PR_AUC                      |       0.2666      |                   96 |
| discovery                      | TabM    | F0_history_only       | Precision@10                |       0.1698      |                   96 |
| discovery                      | TabM    | F1_history_plus_size  | NDCG@10                     |       0.3731      |                   96 |
| discovery                      | TabM    | F1_history_plus_size  | PR_AUC                      |       0.3014      |                   96 |
| discovery                      | TabM    | F1_history_plus_size  | Precision@10                |       0.1771      |                   96 |
| discovery                      | TabM    | F2_plus_import_demand | NDCG@10                     |       0.3588      |                   96 |
| discovery                      | TabM    | F2_plus_import_demand | PR_AUC                      |       0.3022      |                   96 |
| discovery                      | TabM    | F2_plus_import_demand | Precision@10                |       0.1729      |                   96 |
| discovery                      | TabM    | F3_plus_governance    | NDCG@10                     |       0.4012      |                   96 |
| discovery                      | TabM    | F3_plus_governance    | PR_AUC                      |       0.3335      |                   96 |
| discovery                      | TabM    | F3_plus_governance    | Precision@10                |       0.1854      |                   96 |
| discovery                      | TabM    | F4_full               | NDCG@10                     |       0.3767      |                   96 |
| discovery                      | TabM    | F4_full               | PR_AUC                      |       0.3116      |                   96 |
| discovery                      | TabM    | F4_full               | Precision@10                |       0.1833      |                   96 |

Prespecified subgroup estimates and paired neural-minus-baseline differences are in `results/heterogeneity.csv` and `results/heterogeneity_comparisons.csv`. These are descriptive, with no causal interpretation.

No observational ranking result is a causal export gain or firm-level recommendation. Statistical and economic importance should be judged separately; see compute costs and missing-label bounds.
