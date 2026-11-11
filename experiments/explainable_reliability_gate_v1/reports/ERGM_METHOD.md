# ERGM method and evaluation

ERGM is evaluated as a pool-level two-stage system. A task × HS2 × outcome-year pool contains one complete destination candidate set. The first stage selects PPML or LightGBM; the second stage accepts the whole Top-10 recommendation or marks the pool `DEFER_TO_EXPERT`.

## Estimands

Task A primary candidate eligibility is zero exports in each of the three reporting-verified origin years; the success label is observed exports ≥ USD 50,000 in the outcome year. Task B requires positive historical exports and historical share no larger than the HS2 median based on data through the feature cutoff. The corrected sets are mutually exclusive.

For each expert e and pool q, the learning target is retrospective complete-case Top-10 regret: oracle gain among observed-label candidates minus the gain selected by e, bounded below by zero. Task A gain is binary entry success. Task B gain is the audited capped grade: `grade=min(4,floor(log10(1+max(0,increment)/50,000)))`, with DCG gain `2^grade−1`; pool regret uses the difference in undiscounted Top-10 total capped gain. NDCG discounts position by `log2(rank+1)`.

The gate uses a fixed Ridge model with alpha 10 to predict both experts' task-specific regret from pre-origin pool summaries, model-score disagreement and HS2 identity. Prior-expert regret features include only outcomes through y−2 for a y decision. The expert with lower predicted regret is selected. The predicted regret is a ranking score, not a calibrated probability.

For operational coverage thresholds, risk quantiles are derived from chronological held-out historical pool predictions: a pool in historical year t is scored by a gate trained only on outcomes through t−2. This calibration history is short, especially for the 2022 fold, and is not a formal risk guarantee. Matched-coverage estimates select the requested fraction by test-time risk scores only (no test labels), solely to standardize workloads across methods. They are not deployable frozen thresholds.

Human review labels, review cost and post-review decisions do not exist. Deferral is therefore a review-workload flag, not validated human-AI collaboration.
