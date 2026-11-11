# Partial identification with unavailable labels

The broad ex-ante panel has 559 outcome-year labels unavailable in BACI-derived processed records. They remain missing in every primary table. All are in Task A; Task B has no unknown labels under its audited eligibility rule.

For each selected set, observed utility sums use only observed outcomes and report the selected-observed count. An unknown selected market is not counted as failure or as zero. A missing selected item makes that policy-pool incomplete for full-K complete-case inference.

For binary entry relevance, each unknown outcome is bounded by 0 and 1. The paired PR-EMS–baseline utility-difference bounds use a common completion: coefficients are +1 for markets selected only by PR-EMS, −1 for those selected only by the baseline, and 0 for common selections. This yields exact bounds for the chosen rectangular [0,1] support relaxation. The full hindsight regret bounds use a common completion for policy and oracle; lower regret 0 is conservative, upper bounds are support-based.

Task A's actual utility has linked components: entry must succeed before its scale utility is positive. Thus the rectangular [0,1] support relaxation can be wider than the actual feasible entry-utility support. It is conservative, not sharp for the full joint outcome. Task B's capped scale avoids unbounded utility, but it does not identify uncapped USD increments. No finite cap is asserted for raw exports.

The aggregate paired-bound table is `results/R4_partial_identification_summary.csv`; pool-level calculations are `results/R4_partial_identification.csv` and `results/R4_missing_outcome_bounds.csv`. A conclusion is robust to missing outcomes only if the paired bound excludes zero; otherwise the data do not identify a uniform winner under this support assumption.

Missingness may be informative. Its observed association with historical market size is summarized in `results/R4_missingness_by_market_size.csv`. These are descriptive associations and do not establish MAR or MNAR.
