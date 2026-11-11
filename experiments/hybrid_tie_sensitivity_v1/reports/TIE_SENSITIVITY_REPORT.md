# Hybrid rank-fusion tie sensitivity

## Protocol

This audit uses the original strict-zero Task A 2022--2024 candidate-score archive without retraining models, changing eligibility, changing weights, or using outcomes to select a tie rule. The hybrid score is the equal-weight mean of within-pool normalized inverse ranks. Exact ties use equality of the archived/reconstructed score from the original floating-point expression. The integer rank sum $r_P+r_L$ is tracked separately to reveal algebraically equal values that differ only by floating-point rounding. Original ties use ascending ISO3. PPML-first orders a tied group by ascending PPML rank, then LightGBM rank, then ISO3; LightGBM-first reverses the model priority. Random ties uniformly permute ISO3-sorted candidates within each exact-score group under fixed seeds 10,000--10,999 (1,000 draws). These repetitions measure tie-order uncertainty, not independent evaluation samples.

For statistical uncertainty, paired HS2 blocks are resampled 10,000 times, keeping all three years together. Sign-flip tests and Holm adjustment use the existing Task A comparison families expanded to include the tie sensitivity contrasts. All sensitivity inference is exploratory because 2022--2024 outcomes were previously inspected. NDCG follows the original complete-case, binary-entry definition and is averaged over positive-relevance pools. Selection outcomes retain unknown labels; USD paired inference uses complete-outcome pools and the reports separately expose selected unknown labels.

## Reproduction checks

- Candidate rows: 29,737; pools: 288; HS2 blocks: 96.
- Maximum reconstructed hybrid-score deviation from the saved score: 0; PPML/LightGBM rank deviations: 0/0.
- Original ISO3 Hybrid NDCG@10 minus LightGBM: +0.037184513; saved reference: +0.037184513.
- Exact Hybrid-score tie groups across all pools: 4,557 (9,875 candidate-pool records); groups wholly inside Top-10: 379; boundary-tie pools: 51, of which 33 are among the 138 positive-relevance evaluation pools.
- Integer rank-sum arithmetic identifies 60 boundary equalities across 60 pools; 9 pools are algebraically equal but not exact float-score ties under the saved expression. These are reported as a numerical-rounding diagnostic and are not reclassified as exact ties.

## Performance by tie rule

| Rule | NDCG@10 | Precision@10 | Successful entries / pool | Qualified USD / pool | Raw USD / pool | Pools changed vs original | Mean Jaccard vs LightGBM |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM_baseline | 0.294694 | 0.039583 | 0.395833 | $534,664.18 | $546,675.50 | 281 | 1.000000 |
| Gravity_PPML | 0.280091 | 0.037847 | 0.378472 | $341,926.56 | $351,833.66 | 281 | 0.401979 |
| ISO3_ascending_original | 0.331878 | 0.043750 | 0.437500 | $545,937.84 | $558,123.34 | 0 | 0.655864 |
| PPML_first_then_LightGBM_then_ISO3 | 0.334387 | 0.043403 | 0.434028 | $545,861.03 | $558,046.51 | 25 | 0.647006 |
| LightGBM_first_then_PPML_then_ISO3 | 0.331170 | 0.043750 | 0.437500 | $509,184.88 | $521,392.82 | 30 | 0.664739 |
| uniform_random_ties_expected | 0.333098 | 0.043654 | 0.436542 | $528,666.87 | $540,812.51 | 51 | 0.656063 |

The seeded-random row is the per-pool mean over the 1,000 tie-order draws. `TIE_SENSITIVITY_RESULTS.csv` also contains each seed's aggregate metrics; their 2.5th--97.5th percentiles describe Monte Carlo tie-order variation only, not a confidence interval over historical evaluation samples. Across seeds, aggregate NDCG ranges from 0.324569 to 0.341755 (2.5th--97.5th percentiles 0.327056--0.339444); 0/1,000 draws fall below the fixed LightGBM NDCG. The corresponding tie-order percentile ranges are 0.043056--0.044444 for Precision@10, 0.430556--0.444444 successful entries per pool, $508,386--$547,043 qualified USD per pool, and $520,403--$559,237 raw USD per pool. The USD ranges span the fixed LightGBM values, so seed-level USD contrasts change sign; these are Monte Carlo ranges, not confidence intervals.

## Selection dependence and cases

There are 118 boundary-tied candidate-pool selections whose inclusion varies across the 1,000 random orders: 6 are observed successful entries, 112 observed failures, and 0 have unknown outcomes. The observed success/failure composition is descriptive and does not select or validate a tie-break. PPML-first changes Top-10 membership in 25 pools relative to ISO3; LightGBM-first changes it in 30. See `TIE_AFFECTED_POOLS.csv` for all tied groups and pool-level additions/removals.

The boundary-case file contains all 51 boundary-tie pools plus the seven prior authentic XAI case-study pairs. Bulgaria--Iran (HS2 27, 2022) is an exact equal-score rank-10/11 tie: PPML and LightGBM rank contributions offset, and ISO3 makes BGR the original selected member. Any observed Bulgarian advantage over Iran is a consequence of outcomes after prediction; it is not evidence that ISO3 or a model component predicts better. The other prior case pairs are individually flagged for whether either candidate participates in any exact tie group in its pool.

## Inference and interpretation

`TIE_SENSITIVITY_INFERENCE.csv` reports paired HS2-block percentile intervals and sign-flip p-values for each sensitivity rule versus the fixed LightGBM ranking. Holm adjustment is computed within the corresponding Task A metric family while retaining the original Hybrid-vs-LightGBM and Hybrid-vs-PPML comparisons as family members. Random-policy inference averages each pool over tie seeds before resampling; it does not count seeds as economic observations.

The NDCG advantage remains positive under all prespecified tie rules, though its size varies.

The NDCG difference versus LightGBM ranges from +0.036476 to +0.039693 across the four Hybrid tie policies. The original two-contrast inference family retains its published Holm $p=0.0272$. In the broader exploratory sensitivity family that includes ISO3, PPML-first, LightGBM-first, randomized expected tie order, and the original PPML reference contrast, the ISO3-vs-LightGBM Holm-adjusted $p$ is 0.0940. Thus the positive point estimate is robust to these tie choices, while conventional significance is not robust to expanding the post hoc comparison family. Precision and successful-entry intervals remain compatible with no difference.

Tie-break changes also alter dollar outcomes. Relative to LightGBM, qualified USD differences are $+11,197 per pool for PPML-first, $-25,479 for LightGBM-first, and $-5,997 for the random-tie expectation; all corresponding bootstrap intervals span zero. The dollar sign of the descriptive comparison is therefore tie-sensitive and is not an economic advantage claim.

For the changed memberships, PPML-first adds 1 successful and 24 failed destination-pool recommendations while displacing 2 successful and 23 failed recommendations. LightGBM-first adds 1 successful and 29 unsuccessful candidates, while displacing 1 successful and 29 unsuccessful candidates. Across random orders, 118 candidate-pool memberships vary, comprising 6 observed successes and 112 observed failures.

**Recommendation:** retain the original observed NDCG estimate as a result under its stated ISO3 convention, but qualify the main statistical claim: the ranking advantage remains positive across tie rules, whereas its significance depends on the multiplicity family; observed USD opportunity is not robust to tie order and has no established advantage. Report the tie sensitivity alongside the primary result. Do not interpret the outcome composition of ISO3-selected ties as predictive or economic superiority.
