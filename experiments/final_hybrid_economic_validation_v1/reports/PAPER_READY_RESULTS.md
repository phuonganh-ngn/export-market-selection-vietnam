# Paper-ready results draft

## Methods

We evaluate a strict-zero new-market task and a separate existing-market expansion task using a destination–HS2–year panel. Candidate eligibility uses only historical information available by the two-year lag origin. New-market candidates have verified zero Vietnam exports in each of the three years ending at that origin and a successful entry is defined as outcome-year exports of at least USD 50,000. Expansion candidates have positive historical exports and historical market share no greater than the training-period HS2 median. The two task populations are disjoint. Unknown outcome labels remain missing.

Gravity–PPML and LightGBM predictions are ranked within HS2–year pools. The hybrid score is the equal-weight average of within-pool normalized inverse ranks; raw scores are not averaged. NDCG@10 and Precision@10 are evaluated on observed-label candidates. Economic opportunity is the sum of observed exports for new-market selection and positive incremental exports for expansion, with unknown outcomes reported separately. Statistical comparisons use paired HS2 block resampling and sign-flip tests, with Holm correction within prespecified metric families.

A Ridge model predicts historical Top-10 regret from pool-level information and strictly rolling-origin OOF expert outcomes. Thresholds are quantiles of cross-fitted development predictions, frozen before each outer year; achieved coverage is allowed to differ from the requested operating point. The outcome years have been repeatedly inspected in prior M1 development and are treated as exploratory.

## Results

For strict-zero entry, macro NDCG@10 was 0.2947 for LightGBM, 0.2801 for PPML, and 0.3319 for equal-weight rank blending. The blend-minus-LightGBM difference was +0.0372 (95% HS2-block interval 0.0058 to 0.0710; Holm p=0.0272); Precision@10 and successful-entry count differences were not statistically distinguishable. The mean observed USD opportunity contrast versus LightGBM was $11,448 per pool (95% interval $-89,534 to $118,882; Holm p=0.708), so the NDCG gain did not yield statistically established dollar opportunity. For expansion, PPML (0.5925) exceeded the equal-weight blend (0.5713) on capped-gain NDCG@10 and observed USD opportunity. These retrospective ranking metrics do not imply model-induced changes in trade.

The decision-risk analysis estimates pool-level retrospective regret, not a probability of failure. Coverage and conditional accepted risk varied under frozen development thresholds. Because no expert-review decisions or review costs are observed, deferral performance is limited to triage and selective-automation diagnostics.

## Limitations

The 2022–2024 outcomes are not an untouched holdout. Task A has 556 missing outcome labels, and complete-case NDCG does not automatically identify performance on all ex-ante candidates. Historical WDI/WGI values are revised vintages. Export opportunity is not profit, and the observed relationship between model-selected destinations and future trade is not causal. The archived outer predictions do not provide independent seed-level model runs. A fresh holdout is required before claiming prospective reliability.
