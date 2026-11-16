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
| Expansion (653 pools) | PPML - Hybrid | +0.021 | [0.008, 0.034] | 6/7 |
| | PPML - LightGBM | +0.054 | [0.035, 0.075] | 7/7 |
| | PPML - import demand | +0.022 | [0.009, 0.036] | 6/7 |

Expansion relevance is graded from the USD 50,000 success threshold: grade 0 below it, then
`min(4, 1 + floor(log10(increment / 50,000)))`, so grades start at USD 50k, 500k, 5m and 50m
(`revision_lib.expansion_grade`). An earlier grading, `floor(log10(1 + increment / 50,000))`, gave its first grade
only at about USD 450k; results under it are kept as a robustness check (`src/grading_sensitivity.py`).

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
  additivity error 0 for three seeds, max 0.04 and 0.10 log-odds for seeds 11 and 37. The affected rows are 15 of 15,687
  (`src/xai_additivity_sensitivity.py`); excluding them changes group shares by at most 0.00014.
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

## Decision-level explanations for expansion (`src/xai_expansion.py`)

Same checks, LightGBM regressor (log1p USD), swaps between the PPML and LightGBM Top-10. Outputs in `results/xai_expansion/`.

- Fidelity: rebuilt LightGBM scores and re-fitted PPML ranks match exactly; SHAP additivity error above 0.01 in
  0.04% of rows for two seeds, zero for the others.
- Stability: Spearman 0.97-0.995 across seeds, 0.93 across years (min 0.79). Export history is the top group
  every year, then trade demand.
- Swaps (1,953 pairs): disagreement between the models is mostly about export history (59% of PPML's and 62% of
  LightGBM's absolute differences) and then trade demand (18% / 14%). PPML-only destinations have a median
  historical export of USD 119k against USD 6k for LightGBM-only ones, and reach the USD 50k gain in 40% of cases
  against 29%: PPML backs markets where Vietnam already sells something, LightGBM chases very small footholds.
- Theory consistency (LightGBM attributions in the PPML Top-10): 89% of absolute attribution has the expected sign
  (68% of cases); low for demand growth (25%), rule of law and trade openness (44%), GDP per capita (50%).
- Faithfulness: neutralising the main attributed group removes the destination from the LightGBM Top-10 in 75.6%
  of 6,677 cases, against 13.8% for a random group.
- Margins (3,227 destinations at PPML ranks 11-15): +10% import demand is enough for 2.5%, +50% for 23%; median
  needed +92%; 27% not reachable within x4.5. Expansion shortlists are harder to move with demand alone than
  discovery shortlists.

## Further analyses

Run from the project root with `PYTHONPATH=src:experiments/paper_revision_v1/src:.`. Outputs are written under
`experiments/paper_revision_v1/results/`.

| Script | What it does | Output |
|---|---|---|
| `src/aligned_baselines.py` | LightGBM re-fitted under objectives aligned with the decisions (LambdaRank, Poisson on value, L2 and Poisson on the increment), each also fused with PPML | `aligned/` |
| `src/aligned_regrade.py` | Updates the aligned baselines after a change of the expansion grading without a full re-fit | `aligned/` |
| `src/subperiods.py` | Main contrasts for windows starting in 2018 to 2022 (length of training history) | `robustness/subperiods.csv` |
| `src/grading_sensitivity.py` | Expansion NDCG@10 under the primary, the earlier, a binary and a continuous relevance | `robustness/expansion_grading_*.csv` |
| `src/mechanism_checks.py` | Rank fusions of each estimated model with the import-demand rule; gain against training-set size | `robustness/mechanism_*.csv` |
| `src/stacking.py` | Entry classifier with out-of-time PPML predictions as features, against the rank fusion | `robustness/stacking_*.csv` |
| `src/robustness_inference_penalty_scale.py` | Few-cluster p-values for the two-way t, PPML penalty sensitivity, economic scale of entries | `robustness/ppml_penalty.csv` and console |
| `src/robustness_threshold_equivalence.py` | Success threshold at constant 2018 prices; 90% intervals for equivalence in expansion | `robustness/real_threshold.csv` and console |
| `src/robustness_refit.py` | Membership split, re-trained thresholds, strict-zero training population, seed sets | `robustness/` |
| `src/group_ablation.py` | Re-fits all models without each feature group | `ablation/` |
| `src/vietnam_specific.py` | Trade agreements, main markets, product groups | `vietnam/` |
| `src/decision_utility.py` | Screening yield and capped realised value | `utility/` |
| `src/pointwise_vs_decision.py` | Pointwise accuracy metrics against shortlist quality | `pointwise_vs_decision*.csv` |
| `src/xai_hybrid_faithfulness.py` | Deletion test at the level of the hybrid decision: both component scores recomputed | `xai/hybrid_faithfulness_*.csv` |
| `src/sealed_shortlists.py` | Prospective shortlists for outcome years 2025 and 2026 with a hash manifest | `sealed/` |

Main findings of these analyses (discovery NDCG@10 unless stated):

- Objective alignment: the hybrid leads a LambdaRank LightGBM by 0.050 (HS2-block interval above zero, year-level Holm 0.19).
  In expansion PPML is not distinguishable from a Poisson LightGBM on the export level (+0.002) or from LambdaRank
  on the graded gain (+0.004).
- Training history: the hybrid's gain over LightGBM is 0.039 for 2020-2024 and 0.042 for 2021-2024.
- Source of the gain: fusing the classifier with the import-demand rule gives +0.042 of the hybrid's +0.065;
  PPML adds a further 0.023 over 2018-2024 but 0.014 [-0.004, 0.032] over 2020-2024. Stacking gives +0.008.
- Hybrid-level deletion test: replacing the group with the largest combined attribution removes the destination
  from the hybrid Top-10 in 97.5% of 6,720 cases, against 14.1% for a random group (49.5% against 9.5% with the
  values of the nearest real candidate).

The evaluation is retrospective: the 2022-2024 outcomes were inspected during development, and the mechanism
checks, the expansion grading and the subgroup analyses were specified after the main results had been seen.
