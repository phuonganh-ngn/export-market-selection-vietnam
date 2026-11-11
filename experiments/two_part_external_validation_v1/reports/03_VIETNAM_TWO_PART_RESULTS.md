# 03 — Vietnam two-part results

| model                                             |   macro_ndcg10 |   macro_precision10 |   mean_successes_top10 |   mean_qualified_usd |   mean_total_export_usd |   valid_ndcg_pools |
|:--------------------------------------------------|---------------:|--------------------:|-----------------------:|---------------------:|------------------------:|-------------------:|
| Gravity_PPML                                      |       0.280091 |           0.0378472 |               0.378472 |               341927 |                  351834 |                138 |
| LightGBM_probability                              |       0.294694 |           0.0395833 |               0.395833 |               534664 |                  546676 |                138 |
| Original_equal_rank_blend                         |       0.331878 |           0.04375   |               0.4375   |               545938 |                  558123 |                138 |
| TwoPart_calibrated_probability_x_conditional_PPML |       0.277417 |           0.0371528 |               0.371528 |               443233 |                  454746 |                138 |

All models use the same 29,737 strict-zero candidate rows and 288 HS2-year pools. NDCG is macro-averaged over 138 pools with at least one observed positive entry among known-label candidates; unknown outcomes remain NA. USD@10 is the observed-label sum among the full ranking's Top-10, so is a lower bound if an unknown is selected. VNM 2022–24 were repeatedly inspected and the evidence is exploratory. The two-part model is below LightGBM and rank blend on both NDCG and qualified USD; it does not improve the primary economic target.


## Final independent check and paired evidence

The earlier hybrid comparison (same retrospective 2022–24 period) found equal-weight rank blending above LightGBM by 0.03718 NDCG@10 (95% HS2-block bootstrap CI 0.00584 to 0.07100; Holm p=0.0272; 138 positive-relevance pools, 75 HS2 blocks). Precision@10 differed by +0.00417 (Holm p=0.1036), and successful entries by +0.0417 per pool (Holm p=0.1372). The qualified-USD@10 mean difference was +USD 11,274 for blend versus LightGBM, with CI [−USD 89,488, +USD 118,503] and Holm p=1.00 in the five-contrast family. The separate raw total-export comparison reported +USD 11,448 (95% CI [−USD 89,534, +USD 118,882], Holm p=0.708); it is not the qualified target. Thus the evidence supports a ranking-metric improvement over LightGBM in this inspected sample, not an economic-value improvement.

The new Two-Part score had NDCG@10 0.27742 versus 0.33188 for the rank blend; paired difference −0.05446 (95% CI [−0.09394, −0.01550], Holm p=0.0237). Its qualified-USD@10 difference versus the blend was −USD 102,705 (95% CI [−USD 301,717, +USD 46,849], Holm p=1.00). Mean observed-label oracle qualified-USD regret per pool was USD 294,487 for the blend, USD 305,761 for LightGBM, USD 397,192 for Two-Part and USD 498,498 for PPML. This is regret against the observed-label oracle only; missing candidate labels could change the full-population oracle. No model selected an unknown-label market in its VNM Top-10.

The conditional value stage used 878, 969 and 1,060 qualifying training observations at cutoffs 2020, 2021 and 2022. This is a small, skewed conditional sample. NDCG uses 138/288 pools with an observed positive entry; the other pools have no identified positive gain, not proven zero opportunity.
