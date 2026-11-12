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

## Decision-level explanations (`src/xai_decision.py`, discovery, 2018-2024)

Needs `refit_lightgbm.py --save-models` first (70 seed models in `models/lightgbm/`, git-ignored); PPML is
re-fitted per year and cached in `models/ppml/`. Outputs in `results/xai/`.

- Fidelity: rebuilt LightGBM scores equal the evaluated scores (max diff 0); re-fitted PPML ranks equal the stored
  PPML ranks (Spearman 1.000). Interventional TreeSHAP (log-odds, background of 100 same-year candidates):
  additivity error 0 for three seeds, max 0.04 and 0.10 log-odds for seeds 11 and 37 (share of affected rows not yet checked).
- Stability of feature-group importance: Spearman 0.98-0.995 across seeds within a year, 0.92 on average across
  years (min 0.73). Trade demand is the top group in every year.
- Swaps (1,558 pairs where the hybrid adds a destination to the LightGBM Top-10 and displaces another): PPML's
  preference for the added destination comes mainly from trade demand (49% of the absolute log-mean difference,
  favouring the added market in 94% of pairs) and market size (21%). LightGBM ranks the added market lower mainly
  through governance (17%, against the added market in 74% of pairs) and purchasing power (12%, 72%) while still
  favouring it on trade demand (67%). The hybrid therefore adds high-demand markets that LightGBM marks down for
  weaker institutions and lower income.
- Theory consistency of LightGBM attributions in the hybrid Top-10: 88% of absolute attribution has the expected
  sign (61% of feature-row cases). High for import demand (99.6%), supplier concentration, population, GDP,
  landlocked, distance (85%); low for GDP per capita (8%), internet use (19%), trade openness (29%): for first
  entries LightGBM rewards poorer, less connected markets. PPML coefficients are conditional on destination
  indicators, so time-invariant variables (colonial link, landlocked, contiguity) are not identified and their signs
  are not interpretable; import demand, distance, GDP, population, regulatory quality and government effectiveness
  have the expected sign in every year.
- Faithfulness: neutralising (pool median) the group SHAP names as the main reason for a Top-10 placement removes
  the destination from the Top-10 in 89.5% of 6,720 cases, against 15.4% for a random group.
- Decision margins (3,360 destinations at hybrid ranks 11-15): an increase in destination import demand, with both
  component models re-scored, brings 6% into the Top-10 within +10%, 47% within +50% and 94% within x4.5; median
  needed +49%.
