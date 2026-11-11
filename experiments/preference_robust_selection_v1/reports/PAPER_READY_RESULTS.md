# Paper-ready methods and results draft

## Research design

We formulate product–destination market selection as a cardinality-constrained multiobjective decision problem. New-market entry and existing-market expansion are treated separately. At each HS2-product × year pool, policies select five or ten historically eligible destinations using predictors dated no later than two years before the outcome year. Candidate eligibility is constructed without outcome-year trade performance.

## Robust decision rules

We compare fixed equal-weight utility, rank aggregation, preference-only maximin and minimax regret, a Weber-style relative-robust multicriteria rule, predictive maximin, finite-scenario joint minimax regret, and saved PPML, LightGBM and LambdaMART rankings. The joint rule, labeled PR-EMS for convenience, minimizes maximum absolute regret over the Cartesian product of a prespecified preference interval and historical model-disagreement scenarios. It is an application of established robust optimization, not a new method.

## Outcomes and uncertainty

For entry, the objectives are success probability and conditional successful-entry scale. For expansion, they are the probability of a positive increment and bounded utility of its positive magnitude. Probabilities and scale utilities remain distinct; USD and log-USD are not counted as separate criteria. All scales are estimated from training-period outcomes. Finite scenarios are not calibrated predictive distributions.

## Empirical results

The broad evaluation manifest contains 62,797 candidate keys, of which 41,131 meet entry eligibility and 12,962 meet expansion eligibility. There are 559 missing outcome labels, all within Task A. Missing labels remain unobserved. Complete-case metrics and common-completion bounds are reported separately. The empirical policy comparison, macro outcomes, regret and paired uncertainty are in `ROBUST_SELECTION_RESULTS.md` and the CSV tables.

## Interpretation and limitations

The 2022–2024 test years have already informed earlier M1 model and research choices; they are not independent confirmation. Historical WDI/WGI values are revised vintages. The dataset contains trade flows rather than exporter costs, profits or randomized market-promotion treatments. Results therefore describe retrospective associations and decisions under the stated utility/scenario assumptions, not causal or firm-level recommendations. A manuscript should not claim a newly invented robust method or independent future performance.
