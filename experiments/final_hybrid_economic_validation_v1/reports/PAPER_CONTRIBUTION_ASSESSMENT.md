# Paper contribution assessment

## Findings and status

1. The previously reported descriptive Task A blend NDCG advantage reproduces: LightGBM 0.2947, PPML 0.2801, blend 0.3319. Its inferential status is given by the paired HS2-block estimates below; it remains exploratory because the evaluation years informed prior M1 iteration.
2. Blend-minus-LightGBM NDCG@10 difference is +0.0372 with 95% interval [+0.0058, +0.0710] and Holm p=0.0272. Precision@10 difference is +0.0042 (Holm p=0.1036); successful Top-10 count difference is +0.0417 (Holm p=0.1372). The NDCG difference is distinguishable in this exploratory, previously inspected sample; precision and event count are not.
3. Blend-minus-LightGBM observed USD opportunity is $11,448 per pool (95% interval $-89,534 to $118,882, Holm p=0.7084); the median paired difference is $0, with 154/288 ties. This does not establish an economic gain. For Task B, the blend loses to PPML on capped-gain NDCG and observed USD expansion opportunity.
4. Task B PPML NDCG@10 is 0.5925 versus blend 0.5713; the USD contrast is $-1,974,569 per pool (95% interval $-3,766,574 to $-450,490, Holm p=0.0456).
5. Risk scores rank later regret most clearly for Task B, but they are not calibrated probabilities and clipping creates many ties in some folds. At nominal 80%, average achieved coverage is about 95.5% for Task A blend-risk gating and 74.3% for Task B's PPML gate.
6. Matched-coverage diagnostics are exploratory and use the evaluation-year score distribution. Task A's blend-risk gate has lower descriptive regret than simple gates at this matched coverage; the Task B gate is roughly tied with historical HS2-error screening and defers economically large pools. Deferred opportunity is unresolved, not lost; no human review outcome or cost data exist.
7. No fresh independent holdout was found. Official BACI V202601 coverage ends in 2024; the public UN Comtrade availability API returned zero Vietnam annual HS records for 2024 and 2025 when queried on 2026-10-13.

## Contribution classification

The most defensible contribution is a narrow, exploratory empirical result: in this repeatedly inspected Vietnam HS2 sample, equal-weight rank aggregation improves macro NDCG@10 for strict-zero market entry, but not measured USD opportunity relative to LightGBM; PPML remains stronger for expansion. Historical regret scores show retrospective risk ordering, especially for expansion, while deployment thresholds and deferral trade-offs remain uncertain. Rank averaging, PPML, boosted trees, Ridge risk estimation, risk–coverage analysis, and selective prediction are established methods. This work does not establish a novel algorithm, causal export gains, human-review value, future-cycle reliability, or independently validated superiority.

A journal manuscript could be drafted as an exploratory empirical paper if it foregrounds the task distinction, common candidate universe, complete-case limits, statistical uncertainty, and the lack of untouched validation. A demanding journal submission should wait for a genuinely fresh validation period or an external market/product context and stronger evidence that USD opportunity improves.
