# Revision changes (2026-11)

Copy of the original project under git. Commit `530305a` is the untouched baseline;
`git diff 530305a` shows every change. Raw data and archived models are symlinks to the original read-only repository.

## Bugs fixed

| Where | Problem | Fix |
|---|---|---|
| `src/statistics.py` `ndcg_at_k` | Returned NaN when the top k held no relevant item, so pools where the model missed every success were dropped and averages inflated (used by E2, E3, clustering, robustness R14). The paper's main numbers come from `final_validation.py`, which was already correct. | 0 when the top k misses; NaN only when the pool has no relevant item |
| LightGBM (`src/ml_models.py`) | No row/column subsampling, so `random_state` had no effect: the five "seeds" were identical models (repo's own seed SD = 0) | `compute.lightgbm_subsample`, `_subsample_freq`, `_colsample_bytree` (default 0.8 / 1 / 0.8) |
| `src/gravity_ppml.py` | Fallback `alpha` 0.01 differed from config and archived models (0.1) | fallback 0.1; config is the single source |
| Feature set | `common_official_language` is zero for every Vietnam-destination pair | `study.excluded_features` |
| Rank fusion | Normalisation (n-r+1)/(n-1) exceeds 1; floating-point noise (~1e-17) decided some ties instead of the tie rule (13 of 576 frozen-window pools change Top-10 membership once ties are exact) | `src/fusion.py`: (n-r)/(n-1), rounding before tie-breaking, `study.fusion_tie_rule` |

## Options and modules added

- `study.entry_training_population`: `threshold_inactive` (archived: history below USD 50,000) or `strict_inactive` (zero history, the evaluation population); `market_discovery.entry_strict_inactive`, `entry_training_mask`.
- `src/statistics.py`: `hs2_block_inference`, `twoway_cluster_t`, `year_signflip_p`, `paired_contrast_inference`.
- `M1_CONFIG` environment variable selects the config; `config/legacy_archived.yaml` reproduces the archived models.
- `experiments/paper_revision_v1/`: 2018-2024 evaluation, LightGBM re-fit, LambdaMART check, LaTeX table generation (see its README).
- `tests/test_revision_core.py`: NDCG fix, fusion order and ties, LightGBM seeds with and without bagging, year-level inference, training population. All 24 tests pass.

## Unchanged on purpose

- `experiments/final_hybrid_economic_validation_v1` and the other archived experiments are not modified; their stored outputs remain the archived evidence. Their reconstruction checks rely on the archived float formula and full feature list, so re-run them with `M1_CONFIG=config/legacy_archived.yaml`.
- Hard-coded LightGBM parameters inside some experiments (`historical_oof._base_parameters`, `two_part_external_validation_v1`) are left as they were.

## Not done yet

Pool-relative features and monotone constraints for LightGBM, rolling-origin tuning over a wider grid, probability-scale fusion,
serialising full preprocessing pipelines, a lock file, the country-mapping gaps (ROU, COD, SRB, MNE, SSD, TLS, PSE), and a
`predict_2025` path for the out-of-time test.

## XAI (added after the list above)

- `refit_lightgbm.py --save-models` stores every seed model (preprocessor + estimator) per year and task.
- `experiments/paper_revision_v1/src/xai_decision.py`: SHAP fidelity and stability, swap attributions,
  theory consistency, faithfulness deletion test, decision margins (results in the experiment README).
- `xai_expansion.py`: the same analysis for expansion (PPML vs LightGBM Top-10 swaps).

## Later revision rounds

- Expansion relevance grading now starts at the USD 50,000 success threshold (`revision_lib.expansion_grade`);
  all expansion results were regenerated and the earlier grading is kept as a robustness check.
- Added `aligned_baselines.py`, `aligned_regrade.py`, `subperiods.py`, `grading_sensitivity.py`, `mechanism_checks.py`,
  `stacking.py`, `robustness_inference_penalty_scale.py`, `robustness_threshold_equivalence.py` and
  `xai_hybrid_faithfulness.py` (see the README of `experiments/paper_revision_v1`).
- Removed absolute local paths from the scripts; `aligned_regrade.py` takes the previous run's files as arguments.

