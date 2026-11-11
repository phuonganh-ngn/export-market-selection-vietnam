# Experimental design and evidence status

## Frozen protocol

- Existing audited HS12 BACI V202601-based panel; no raw data redownload.
- Candidate origin cutoff: outcome year minus two years.
- Outer years: 2022, 2023 and 2024; all have already appeared in M1 reports and method-development decisions.
- Product × year × task is the ranking pool; the same historically eligible candidate universe and K are used for all policies.
- K = 10 primary and K = 5 sensitivity.
- Preference ranges: narrow [0.40, 0.60], primary [0.25, 0.75], broad [0, 1], fixed before result evaluation.
- Utility scaling and predictive scenario calibration use training-period outcomes through y−2 only.
- Missing labels stay NA. Candidate eligibility and label observability are separate.
- Primary results use the finite-scenario comparisons documented in `METHOD_FORMULATION.md`.

## Tasks and populations

The broad audited panel contains 62,797 unique eligible destination–HS2–year rows. Task A has 41,131 rows; Task B has 12,962; 6,511 satisfy both and 15,215 neither. The two task row counts sum to 54,093; their union has 47,582 unique destination–HS2–year keys. All 559 unavailable outcome labels are in Task A; Task B has none in its selected set. The 15,215 neither-task rows are not discarded from the broad manifest; they do not satisfy either specific decision rule. The overlap is due to the distinct inactivity and underexploitation criteria, not duplicate rows within a task.

The task-expanded data are not pooled into a common ranking. Task A evaluates entry probability and conditional entry scale separately. Task B evaluates expansion likelihood and bounded positive increment scale separately.

## Predictive scenarios and preference assumptions

The model-derived scenarios use saved audited rolling-origin predictions. Historical conditional size quantiles and Task B probability models are estimated from data available by each fold's y−2 cutoff. Scenario sets are finite stress cases; no predictive distribution or calibrated coverage is assumed. Scenario envelope coverage is calculated only after policy/scenario freeze and is labelled descriptive.

Preference sets are assumed ranges for sensitivity analysis. They are not population estimates or elicited values. The study does not pick weights based on evaluation performance.

## Policy comparisons

The joint policy called PR-EMS is standard finite-scenario minimax absolute regret. It is compared with equal-weight sum, Borda, maximin preference, standard preference-only minimax regret, Weber-style relative robustness, predictive maximin, uniform scenario-average utility, PPML, LightGBM and LambdaMART score rankings. All portfolios are selected from the same frozen task/product/year pool, and their solver outputs are checked.

## Evaluation estimands

Outcome utility is summarized over observed selected markets, with the count of missing selected outcomes and evaluated slots shown. Complete-case regret compares the policy's observed selected utility to the hindsight top-K among observed candidates, with K reduced to the number of observed selections only when a selected market has a missing label; those rows are not silently treated as full-K evaluations. Full-K inferential comparisons are separately restricted to paired pools where both policies' selected labels are observed. Partial-identification tables retain all pools and bound the paired differences under common missing-outcome completions.

For Task A, retrospective entry is binary and capped scale utility is conditional on success. For Task B, realized expansion is positive incremental exports under a training-only cap. These metrics are observational and do not identify the effect of recommending a destination.

## Inference

Paired comparisons are made on common task–HS2–year–K pools. The planned macro estimand is a mean paired policy difference across product-year pools, with HS2-block bootstrap resampling all years together within sampled HS2 chapters. A destination-block bootstrap is not treated as the primary ranking uncertainty procedure because it changes the candidate composition and hence the top-K decision problem; if reported as sensitivity, it is explicitly an alternative candidate-population estimand. Three evaluation years do not identify uncertainty over future cycles.

Holm correction is limited to named primary pairwise families. Model seeds are not economic replications. Any inference from 2022–2024 is exploratory, not independent confirmation.

## Independent holdout

The CEPII BACI release page identifies the current V202601 HS12 series as 2012–2024. No verified, untouched HS12 2025 release exists in the current input, and all 2022–2024 years have informed prior M1 study development. Status: **NO_FRESH_INDEPENDENT_HOLDOUT**. WDI/WGI are revised vintages, further preventing a real-time historical-availability claim.

## Confirmatory versus exploratory

This extension is exploratory in its empirical claims. The robust-optimization problem and hypotheses are specified before rerunning the new computations, but the proposed research direction itself was selected after inspecting prior results from the same 2022–2024 period. Thus no test-period result is independent confirmation. The mathematical comparisons are confirmatory only with respect to the fixed scenario sets and chosen normalized utility, not to actual future market returns.
