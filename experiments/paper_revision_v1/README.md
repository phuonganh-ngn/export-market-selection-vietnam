# paper_revision_v1

Evaluation behind the revised manuscript: seven rolling-origin outcome years (2018-2024), Gravity-PPML,
a re-fitted LightGBM with real bagging, their equal-weight rank fusion, and a naive import-demand rule.

## Run

```bash
cd <project root>
PYTHONPATH=src python experiments/paper_revision_v1/run_revision.py              # refit + evaluate + LambdaMART + tables (~3 min)
PYTHONPATH=src python experiments/paper_revision_v1/run_revision.py --skip-refit # reuse predictions/lgbm_refit_scores.parquet
```

Reproduction of the archived deterministic models (writes to `*/legacy_check/`, leaves the main outputs alone):

```bash
M1_CONFIG=config/legacy_archived.yaml REVISION_TAG=legacy_check PYTHONPATH=src python experiments/paper_revision_v1/run_revision.py
```

With the legacy config the re-fitted LightGBM scores match the archived ones exactly (Spearman 1.0 in all
14 task-years, `results/legacy_check/stored_vs_refit_check.csv`).

## Steps and outputs

| Script | Output |
|---|---|
| `src/refit_lightgbm.py` | `predictions/lgbm_refit_scores.parquet`: per-seed and mean LightGBM scores, 2018-2024, both tasks |
| `src/evaluate.py` | `results/pool_metrics.csv`, `results/main_by_year.csv`, `results/main_contrasts.csv`, `results/stored_vs_refit_check.csv` |
| `src/lambdamart_check.py` | `results/lambdamart_check.csv` (LambdaMART alone and in the hybrid, frozen window) |
| `src/make_tables.py` | `reports/tables/tab_main_year.tex`, `tab_main_contrasts.tex`, `tab_main_secondary.tex` (the manuscript `\input`s these) |

Inputs (read only): stored PPML scores in `experiments/explainable_reliability_gate_v1/predictions/ergm_candidate_predictions_all_periods.parquet`,
the processed panel, and the stored LambdaMART predictions of `decision_focused_xai_v1`.

## Inference

Every contrast reports three layers (`src/statistics.py`): HS2-block bootstrap interval with block sign-flip p,
a t-statistic with standard errors clustered by HS2 chapter and year, and an exact year-level sign-flip test
(floor 2/2^T; 0.016 for seven years, 0.25 for three). Holm adjustment within a family of three contrasts per task.

## Results with the revised configuration (bagging 0.8/1/0.8, five seeds, constant feature excluded, exact ties)

NDCG@10, 2018-2024:

| Task | Contrast | Difference | 95% CI | Years > 0 |
|---|---|---|---|---|
| Discovery (364 pools) | Hybrid - LightGBM | +0.065 | [0.044, 0.086] | 7/7 |
| | Hybrid - PPML | +0.050 | [0.031, 0.070] | 7/7 |
| | Hybrid - import demand | +0.063 | [0.042, 0.085] | 7/7 |
| Expansion (562 pools) | PPML - Hybrid | +0.026 | [0.010, 0.044] | 6/7 |
| | PPML - LightGBM | +0.063 | [0.040, 0.086] | 7/7 |
| | PPML - import demand | +0.022 | [0.005, 0.039] | 6/7 |

LambdaMART (stored, frozen window) is worse than LightGBM in discovery (-0.060, negative in all three years) and
a PPML + LambdaMART hybrid is worse than the PPML + LightGBM hybrid (-0.038); in expansion it only matches PPML.
Some stored LambdaMART fits stopped after two boosting rounds. It is not adopted.

Sensitivity: an earlier bagged run that still contained the constant `common_official_language` column gave
+0.059 for Hybrid - LightGBM (column sampling changes when a column is removed). Treat differences of about
0.006 between bagged specifications as noise.
