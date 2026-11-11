# Decision robustness

## Top-K stability

| task                           |   k | comparison             | model_a    | model_b                    |   mean_jaccard |   median_jaccard |   pool_pairs |
|:-------------------------------|----:|:-----------------------|:-----------|:---------------------------|---------------:|-----------------:|-------------:|
| Task_A_new_entry               |   5 | LambdaMART_seed_pair   | LambdaMART | LambdaMART                 |     0.720581   |         0.833333 |         2880 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | Gravity_PPML               |     0.411334   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | Historical_entry_frequency |     0.00462963 |         0        |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | LightGBM                   |     0.470555   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | LogisticRegression         |     0.424824   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | MLP                        |     0.425182   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | RandomForest               |     0.439222   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | TabM                       |     0.410204   |         0.428571 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | World_import_demand        |     0.172481   |         0.111111 |          288 |
| Task_A_new_entry               |   5 | LambdaMART_vs_baseline | LambdaMART | XGBoost                    |     0.462798   |         0.428571 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_seed_pair   | LambdaMART | LambdaMART                 |     0.739355   |         0.818182 |         2880 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | Gravity_PPML               |     0.500245   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | Historical_entry_frequency |     0.0254128  |         0        |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | LightGBM                   |     0.556077   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | LogisticRegression         |     0.526066   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | MLP                        |     0.532605   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | RandomForest               |     0.535364   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | TabM                       |     0.518546   |         0.538462 |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | World_import_demand        |     0.255523   |         0.25     |          288 |
| Task_A_new_entry               |  10 | LambdaMART_vs_baseline | LambdaMART | XGBoost                    |     0.553444   |         0.538462 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_seed_pair   | LambdaMART | LambdaMART                 |     0.798086   |         0.666667 |         2880 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | GDP                        |     0.426642   |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | Gravity_PPML               |     0.52724    |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | Historical_export_trend    |     0.0673914  |         0        |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | Historical_import_demand   |     0.448178   |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | LightGBM                   |     0.483011   |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | MLP                        |     0.40636    |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | Persistence                |     0.0822586  |         0        |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | RandomForest               |     0.19275    |         0.111111 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | TabM                       |     0.390997   |         0.428571 |          288 |
| Task_B_existing_underexploited |   5 | LambdaMART_vs_baseline | LambdaMART | XGBoost                    |     0.468102   |         0.428571 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_seed_pair   | LambdaMART | LambdaMART                 |     0.863429   |         0.818182 |         2880 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | GDP                        |     0.544884   |         0.538462 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | Gravity_PPML               |     0.636702   |         0.666667 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | Historical_export_trend    |     0.172087   |         0.111111 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | Historical_import_demand   |     0.564969   |         0.538462 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | LightGBM                   |     0.549508   |         0.538462 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | MLP                        |     0.461531   |         0.428571 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | Persistence                |     0.187121   |         0.111111 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | RandomForest               |     0.269268   |         0.176471 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | TabM                       |     0.429537   |         0.428571 |          288 |
| Task_B_existing_underexploited |  10 | LambdaMART_vs_baseline | LambdaMART | XGBoost                    |     0.534101   |         0.538462 |          288 |

## Seed-frequency consensus policy

| task                           |   mean_consensus_ndcg |   mean_score_policy_ndcg |   pools |
|:-------------------------------|----------------------:|-------------------------:|--------:|
| Task_A_new_entry               |              0.414263 |                 0.414785 |     281 |
| Task_B_existing_underexploited |              0.558501 |                 0.559187 |     241 |

Rankings are also accompanied by `E7_rank_sensitivity.csv` with capped, log1p, raw USD, and binary relevance sensitivity where applicable. Candidate universes are kept fixed within each pool; unknown outcomes are retained in rankings and reported separately. The consensus is exploratory and is not a confirmatory model selected on the test set.

## Historical high-demand eligibility sensitivity

This analysis restricts each HS2-year pool to candidates with y−2 destination import demand at or above the HS2-specific median computed from historical panel rows through y−2. It deliberately changes the candidate universe; its NDCG and regret values are not directly compared to primary-population levels.

| task                           | model                      |   mean_candidate_fraction |   mean_ndcg10_change |   mean_top10_jaccard |   mean_restricted_regret10 |   pools |
|:-------------------------------|:---------------------------|--------------------------:|---------------------:|---------------------:|---------------------------:|--------:|
| Task_A_new_entry               | Gravity_PPML               |                  0.304725 |           0.0646633  |             0.734818 |                    1.58333 |     256 |
| Task_A_new_entry               | Historical_entry_frequency |                  0.304725 |           0.159785   |             0.161838 |                    2.61806 |     256 |
| Task_A_new_entry               | LambdaMART                 |                  0.304725 |           0.0767641  |             0.700149 |                    1.59722 |     256 |
| Task_A_new_entry               | LightGBM                   |                  0.304725 |           0.063945   |             0.725002 |                    1.46181 |     256 |
| Task_A_new_entry               | LogisticRegression         |                  0.304725 |           0.0718746  |             0.757477 |                    1.54861 |     256 |
| Task_A_new_entry               | MLP                        |                  0.304725 |           0.0714468  |             0.757829 |                    1.52778 |     256 |
| Task_A_new_entry               | RandomForest               |                  0.304725 |           0.0690636  |             0.733618 |                    1.47917 |     256 |
| Task_A_new_entry               | TabM                       |                  0.304725 |           0.0674888  |             0.772717 |                    1.54861 |     256 |
| Task_A_new_entry               | World_import_demand        |                  0.304725 |           0.0550748  |             0.966319 |                    2.00347 |     256 |
| Task_A_new_entry               | XGBoost                    |                  0.304725 |           0.0651576  |             0.725359 |                    1.46528 |     256 |
| Task_B_existing_underexploited | GDP                        |                  0.803395 |           0.00430208 |             0.991924 |                    2.50694 |     240 |
| Task_B_existing_underexploited | Gravity_PPML               |                  0.803395 |           0.00463089 |             0.996475 |                    2.03819 |     240 |
| Task_B_existing_underexploited | Historical_export_trend    |                  0.803395 |           0.0415109  |             0.726884 |                    4.90625 |     240 |
| Task_B_existing_underexploited | Historical_import_demand   |                  0.803395 |           0.0044589  |             0.999306 |                    2.25347 |     240 |
| Task_B_existing_underexploited | LambdaMART                 |                  0.803395 |           0.00473328 |             0.991367 |                    2.23611 |     240 |
| Task_B_existing_underexploited | LightGBM                   |                  0.803395 |           0.00569184 |             0.975631 |                    2.54167 |     240 |
| Task_B_existing_underexploited | MLP                        |                  0.803395 |           0.00997391 |             0.931178 |                    3.17361 |     240 |
| Task_B_existing_underexploited | Persistence                |                  0.803395 |           0.0553569  |             0.67556  |                    4.57639 |     240 |
| Task_B_existing_underexploited | RandomForest               |                  0.803395 |           0.0357354  |             0.793272 |                    4.22569 |     240 |
| Task_B_existing_underexploited | TabM                       |                  0.803395 |           0.0155474  |             0.893132 |                    3.53125 |     240 |
| Task_B_existing_underexploited | XGBoost                    |                  0.803395 |           0.00761191 |             0.96321  |                    2.63542 |     240 |
