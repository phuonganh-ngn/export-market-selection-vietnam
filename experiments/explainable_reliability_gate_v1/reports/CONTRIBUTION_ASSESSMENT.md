# Contribution assessment

## Findings

- Adaptive routing vs fixed task routing: see `T3_adaptive_routing.csv` and `T1_static_task_routing.csv`; results are actual historical Top-10 regret and NDCG on common corrected task pools.
- Reliability score: see `T4_risk_calibration.csv` and `T5_high_regret_detection.csv`. It is a predicted regret score; no probability calibration guarantee is claimed.
- Matched-coverage deferral: ERGM has 6 task-year matched 80% summary rows; simple gates have 30. See paired estimates in `inference.csv`.
- Expert review value is unidentified because no human recommendations or post-review outcomes are available.
- The selected method components—expert routing, selective prediction, risk–coverage, confidence thresholding and abstention—are established in prior work; see `LITERATURE_AND_NOVELTY_AUDIT.md`.

## Contribution classification

The evidence supports an exploratory empirical application of pool-level routing and selective automation. It does not support a new algorithm, a formal reliability guarantee, or validated human-AI collaboration.
The specific domain adaptation is a pool-level gate trained to predict Top-10 regret between gravity and ML experts. This is an empirical decision-unit adaptation, not a new general-purpose learning-to-defer method. The results use previously inspected 2022–2024 years and cannot support independent validation. Any potentially useful evidence must be confirmed on a genuinely new trade vintage and with expert review outcomes if human-AI claims are intended.
