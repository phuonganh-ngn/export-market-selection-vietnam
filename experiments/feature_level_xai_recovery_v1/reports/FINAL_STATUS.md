# Final status

Engineering: **COMPLETE_VERIFIED** for model recovery, prediction equivalence, feature-attribution reconstruction, and rank-level decision linkage, subject to the scope below.

Model fidelity classification: **PREDICTION_EQUIVALENT_RECONSTRUCTION** overall. The LightGBM outer-fold checkpoints and preprocessors are exact and reproduce the saved seed-mean scores, ranks, and Top-10 lists. PPML estimator objects and coefficients are recovered; their preprocessing was reconstructed deterministically from archived code/training folds and matches saved predictions and ranks exactly.

XAI classification: **VERIFIED_COMPONENT_AND_DECISION_LEVEL_XAI**. The five-seed LightGBM probability ensemble has additive TreeSHAP reconstruction, PPML eta contributions reconstruct the log-link predictor, and the rank-fusion score and Top-10 decisions are exact.

Scientific contribution: **FEATURE_LEVEL_EVIDENCE_ADDS_EXPLANATORY_DEPTH**. This adds faithful predictive attribution connected to actual Top-10 swaps; it is not a new explainability algorithm, causal account, or evidence of improved trade outcomes.

Rows: 29,737 Task A candidates, 288 pools, 29,181 observed outcomes, 556 unknown. Exact score/rank equivalence rows: 21/21. Boundary pairs: 622. Case-study rows: 7. Hybrid rank-10/11 exact-score ties: 51 pools; the ISO3 tie rule is decision-relevant.

The outcome-linked analysis is exploratory because the outer years 2022–2024 have been repeatedly inspected. No feature-group perturbation was used.

Reproduction: `.venv_uv/bin/python experiments/feature_level_xai_recovery_v1/src/recover_feature_xai.py` from the M1 project root.
