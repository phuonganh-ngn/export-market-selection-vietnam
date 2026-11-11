# Final status

Status: **COMPLETE_REDUCED_SPECIFICATION**

Primary GRU and lag-matched LightGBM outer predictions were executed on the audited Task-A population. The 2022–2024 outcomes are exploratory retrospective evidence. Missing labels remain unknown. Non-primary temporal ablations use one prespecified seed to conserve CPU resources; this is a reduced specification relative to full multi-seed ablation.

Reproduce from the M1 root with `.venv_uv/bin/python experiments/gru_temporal_market_selection_v1/run_gru_experiment.py`.

## Final independent verification

A separate reconstruction from `predictions/gru_predictions.parquet`, `predictions/matched_lgbm_predictions.parquet`, and the saved audited candidate prediction archive reproduced the reported pool metrics to floating-point tolerance. All three models were evaluated on 288 HS2–year pools and 138 pools with observed positive entry relevance. The GRU file contains 446,055 prediction rows and the matched LightGBM file contains 59,474; the 8,340 and 1,112 unobserved rows respectively retain missing entry labels. No prediction has `training_outcome_max` later than its prediction cutoff or later than `outcome_year - 2`.

The extension tests passed when run from this experiment directory: `../../.venv_uv/bin/python -m pytest -q tests` (4 passed). No PNG, SVG, or PDF files were generated. Main GRU specifications used five seeds across three outer years; the prespecified feature-group ablations used one seed and are therefore exploratory. Historical WDI/WGI revisions and the previously inspected 2022–2024 outcomes remain limits on point-in-time and independent validation claims.
