# Economic value and opportunity regret

The primary USD outcome is known-label qualified export opportunity captured in the Top-10: observed outcome-year exports for candidates with exports ≥ USD 50,000. Total observed exports are reported separately. Missing outcomes are never treated as zero. The ranking comparison on observed labels is complete-case and uses the same candidate keys for each paired model comparison.

| model                            |   qualified_usd10_mean_per_pool |   qualified_usd10_median_per_pool |   total_usd10_mean_per_pool |   qualified_usd_regret10_mean |
|:---------------------------------|--------------------------------:|----------------------------------:|----------------------------:|------------------------------:|
| Gravity_PPML                     |                          341927 |                                 0 |                      351834 |                        498498 |
| LightGBM                         |                          534664 |                                 0 |                      546676 |                        305761 |
| Equal_Rank_Blend                 |                          545938 |                                 0 |                      558123 |                        294487 |
| Two_Part                         |                          443233 |                                 0 |                      454746 |                        397192 |
| LGBM_TEMPORAL_MATCHED_RAW        |                          503757 |                                 0 |                      514976 |                        336667 |
| LGBM_TEMPORAL_MATCHED_SUMMARY    |                          487932 |                                 0 |                      498688 |                        352493 |
| GRU_SEQ_ONLY_L5                  |                          471314 |                                 0 |                      482034 |                        369111 |
| GRU_SEQ_PLUS_STATIC_L5           |                          415819 |                                 0 |                      427602 |                        424606 |
| GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3 |                          426209 |                                 0 |                      437911 |                        414216 |

These retrospective amounts describe opportunity captured in historical trade records; they are not profit, deployment ROI, or causal model effects. The full candidate ranking can include unknown outcomes; their count is saved separately and makes the observed sum partial.
