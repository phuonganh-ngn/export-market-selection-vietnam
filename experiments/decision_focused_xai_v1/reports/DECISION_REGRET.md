# Decision regret

For each HS2-year pool, the oracle is the top K observed candidates under the same gain function and candidate-label intersection. Realized-outcome opportunity regret is oracle gain minus selected gain; it is not profit forgone or a causal loss. Normalized regret is undefined when oracle gain is zero.

## New-market entry, binary regret@10

| task             | model                      |   seed |   k |   mean_regret |   mean_normalized_regret |   mean_selected_gain |   pools |
|:-----------------|:---------------------------|-------:|----:|--------------:|-------------------------:|---------------------:|--------:|
| Task_A_new_entry | Gravity_PPML               |     -1 |  10 |       2.54861 |                 0.497386 |             2.30903  |     288 |
| Task_A_new_entry | Historical_entry_frequency |     -1 |  10 |       4.48611 |                 0.927243 |             0.371528 |     288 |
| Task_A_new_entry | LambdaMART                 |     -2 |  10 |       2.61111 |                 0.488556 |             2.24653  |     288 |
| Task_A_new_entry | LightGBM                   |     -1 |  10 |       2.39236 |                 0.461451 |             2.46528  |     288 |
| Task_A_new_entry | LogisticRegression         |     -1 |  10 |       2.57292 |                 0.486288 |             2.28472  |     288 |
| Task_A_new_entry | MLP                        |     -1 |  10 |       2.54167 |                 0.480983 |             2.31597  |     288 |
| Task_A_new_entry | RandomForest               |     -1 |  10 |       2.46875 |                 0.477517 |             2.38889  |     288 |
| Task_A_new_entry | TabM                       |     -1 |  10 |       2.55903 |                 0.482514 |             2.29861  |     288 |
| Task_A_new_entry | World_import_demand        |     -1 |  10 |       3.18403 |                 0.632016 |             1.67361  |     288 |
| Task_A_new_entry | XGBoost                    |     -1 |  10 |       2.39236 |                 0.455893 |             2.46528  |     288 |

## Existing-market expansion, capped-gain regret@10

| task                           | model                    |   seed |   k |   mean_regret |   mean_normalized_regret |   mean_selected_gain |   pools |
|:-------------------------------|:-------------------------|-------:|----:|--------------:|-------------------------:|---------------------:|--------:|
| Task_B_existing_underexploited | GDP                      |     -1 |  10 |       2.625   |                 0.332204 |              4.17708 |     288 |
| Task_B_existing_underexploited | Gravity_PPML             |     -1 |  10 |       2.15278 |                 0.254285 |              4.64931 |     288 |
| Task_B_existing_underexploited | Historical_export_trend  |     -1 |  10 |       5.58333 |                 0.766361 |              1.21875 |     288 |
| Task_B_existing_underexploited | Historical_import_demand |     -1 |  10 |       2.36806 |                 0.283874 |              4.43403 |     288 |
| Task_B_existing_underexploited | LambdaMART               |     -2 |  10 |       2.35417 |                 0.282173 |              4.44792 |     288 |
| Task_B_existing_underexploited | LightGBM                 |     -1 |  10 |       2.67014 |                 0.340714 |              4.13194 |     288 |
| Task_B_existing_underexploited | MLP                      |     -1 |  10 |       3.36458 |                 0.448089 |              3.4375  |     288 |
| Task_B_existing_underexploited | Persistence              |     -1 |  10 |       5.35764 |                 0.75133  |              1.44444 |     288 |
| Task_B_existing_underexploited | RandomForest             |     -1 |  10 |       4.82292 |                 0.668876 |              1.97917 |     288 |
| Task_B_existing_underexploited | TabM                     |     -1 |  10 |       3.8125  |                 0.502786 |              2.98958 |     288 |
| Task_B_existing_underexploited | XGBoost                  |     -1 |  10 |       2.79167 |                 0.368088 |              4.01042 |     288 |

Pool-specific metrics for K=5 and K=10, plus Task B raw-USD and binary-expansion gain sensitivities, are preserved in `E7_regret_at_5_10.csv`. The Task B USD increment sensitivity is an observed historical outcome metric; missing labels are excluded from oracle and model outcome gains. Cross-pool sums should not be read as economic returns.
