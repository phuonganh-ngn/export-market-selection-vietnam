# E3 pool construction correction log

## Final status: CORRECTED

Independent inspection found two E3 implementation defects after the first full pipeline run:

1. The initial run ranked candidates from multiple HS2 chapters in one combined pool. That run's E3 metrics, matched forecast-ranking comparison, and E3 inference are **INVALIDATED** because the estimand requires separate HS2 × outcome-year pools.
2. The first corrective run grouped by HS2, but computed the underexploited-market threshold from an evaluation-only panel slice. Consequently, it had no historical training observations for 2022 and 2023. Those Task B metrics and comparisons are **INVALIDATED**.

Both sets of outputs are preserved under `results/invalidated_first_pass/` and `predictions/invalidated_first_pass/`, with separate subdirectories. They are not used in final reports.

The corrected run ranks within each HS2 × outcome-year pool. Task B's HS2 median historical market-share threshold is computed from the full panel using observed historical records through `outcome_year - 2`; only the evaluation candidates are drawn from the outer test year. GDP and historical import-demand indicator baselines use training-period median imputation for missing scores, and the number imputed is reported. Missing outcomes remain NA.

The corrected E3 run completed through `runner.py --stage full`. It contains 288 candidate pools for Task A, 288 all-existing sensitivity pools, and 288 Task B pools (96 HS2 × 3 years) using the full historical panel to set each training-period threshold. Binary-entry NDCG is defined in 281 Task A pools; capped-gain NDCG is defined in 241 Task B pools because the remaining observed-label pools have zero relevance. Unknown-label rows remain missing and are counted separately.

The corrected run also records the number of simple-indicator candidate scores filled from a training-only median. Across the macro averages, this is 19.35 candidate scores per Task A pool for GDP and 0.125 per Task B pool; the import-demand indicator has no missing candidate scores. These counts are diagnostics, not added observations.

The destination-block bootstrap implementation was separately checked for model-specific rank order and then vectorized. Its weighted NDCG results were unit-tested against explicit duplicated-destination expansion; the final sensitivity uses 1,000 shared-destination resamples for each of the ten task/model contrasts. An earlier in-progress bootstrap was stopped before writing results and is not reported.

No original M1 artifacts or `audit_v2` files were changed. The E2 and E1 results are unaffected by these E3 corrections.
