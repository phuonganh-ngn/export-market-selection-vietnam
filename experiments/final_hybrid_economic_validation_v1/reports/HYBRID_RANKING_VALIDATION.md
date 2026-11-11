# Hybrid ranking validation

The audited Task A sample has 29,737 ex-ante candidate rows, 556 unknown labels, and 288 HS2–year pools. The reproduced LightGBM NDCG@10 is 0.294694; PPML is 0.280091; equal-weight normalized-rank blend is 0.331878. The descriptive blend change versus LightGBM is +0.037185 (+12.62%).

The blend is the mean of within-pool normalized percentile ranks, with ascending ISO3 as the deterministic tie rule. It does not average raw PPML and LightGBM scores. Borda aggregation is rank-equivalent to this blend for two experts; the saved blend was reconstructed numerically. Task A's PPML component is an export-value forecast score, whereas LightGBM is an entry probability score. Their rank aggregation is scale-invariant but combines outputs trained for different target tasks; this limits interpretation as a calibrated hybrid probability.

Paired inference resamples HS2 blocks and retains each chapter's three years. The inferential population remains 288 previously inspected HS2–year pools, with only 96 independent product blocks and three calendar years. See `F1_blend_paired_tests.csv` for confidence intervals and Holm-adjusted family tests. These intervals do not establish generalization across future cycles.

## Key contrast table

| policy                      |   candidate_pools |   valid_ndcg10_pools |   macro_ndcg10 |   macro_precision10 |   successful_count_top10 |
|:----------------------------|------------------:|---------------------:|---------------:|--------------------:|-------------------------:|
| BORDA_RANK_AGGREGATION      |               288 |                  138 |       0.331686 |           0.04375   |                 0.4375   |
| EQUAL_WEIGHT_RANK_BLEND     |               288 |                  138 |       0.331878 |           0.04375   |                 0.4375   |
| Gravity_PPML                |               288 |                  138 |       0.280091 |           0.0378472 |                 0.378472 |
| LightGBM                    |               288 |                  138 |       0.294694 |           0.0395833 |                 0.395833 |
| SAVED_ORIGINAL_STATIC_BLEND |               288 |                  138 |       0.331878 |           0.04375   |                 0.4375   |
