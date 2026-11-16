# Export market selection for Vietnam: gravity, machine learning and hybrid ranking

Replication code for the manuscript *Matching Models to Market Decisions: Gravity, Machine Learning and Hybrid
Ranking for Export Market Entry and Expansion in Vietnam* (under review).

The study asks which model should build a Top-10 shortlist of destinations for each HS2 chapter of Vietnam's
exports, and shows that the answer depends on the decision:

- **New-market discovery** (extensive margin): destinations with no Vietnamese exports in a chapter for three years.
  Success is an entry of at least USD 50,000 in the outcome year.
- **Existing-market expansion** (intensive margin): destinations with positive exports and a below-median market
  share. The outcome is the positive two-year increment, graded from the USD 50,000 threshold.

Three approaches are compared as shortlist producers: a regularised Poisson gravity-type model (Gravity-PPML),
a bagged LightGBM, and their equal-weight rank combination (the hybrid), against naive rules such as ranking by
import demand. Evaluation is rolling-origin over the outcome years 2018-2024 with a two-year information lag,
and the main metric is NDCG@10 within HS2-by-year pools.

## Main results

NDCG@10, 2018-2024, paired differences with HS2-block bootstrap 95% intervals:

| Task | Contrast | Difference | 95% CI | Years > 0 |
|---|---|---|---|---|
| Discovery (364 pools) | Hybrid - LightGBM | +0.065 | [0.044, 0.086] | 7/7 |
| | Hybrid - PPML | +0.050 | [0.031, 0.070] | 7/7 |
| | Hybrid - import demand | +0.063 | [0.042, 0.085] | 7/7 |
| Expansion (653 pools) | PPML - Hybrid | +0.021 | [0.008, 0.034] | 6/7 |
| | PPML - LightGBM | +0.054 | [0.035, 0.075] | 7/7 |
| | PPML - import demand | +0.022 | [0.009, 0.036] | 6/7 |

Qualifications that the manuscript reports alongside these numbers:

- The discovery gain over LightGBM is about 0.04 once the classifier has three or more training years.
- Fusing the classifier with the one-variable import-demand rule already gives about two-thirds of the gain
  (+0.042); the further contribution of PPML is not distinguishable from zero over 2020-2024.
- In expansion PPML is not distinguishable from a LightGBM trained on export levels with a Poisson objective or
  directly on the ranking gain, so the training target matters more than the model family. With a binary
  relevance at USD 50,000 its lead over the hybrid is 0.009 and not significant.
- The evaluation is retrospective: the 2022-2024 outcomes were inspected during development. Shortlists for
  the outcome years 2025 and 2026 are sealed with a hash manifest in `experiments/paper_revision_v1/sealed/`.
- The evidence concerns Vietnam. The discovery gain is not replicated for Indonesia under country-specific
  re-fitting.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Core library: data acquisition and validation, panel construction, features, Gravity-PPML (`gravity_ppml.py`), tree models (`ml_models.py`), rank fusion (`fusion.py`), inference (`statistics.py`) |
| `run_pipeline.py` | Base pipeline: acquire, validate, prepare, experiments, robustness, report |
| `config/config.yaml` | Study configuration; `config/legacy_archived.yaml` reproduces the archived deterministic models |
| `experiments/paper_revision_v1/` | **Evaluation behind the manuscript**: seven-year rolling-origin evaluation, aligned baselines, explanation tests, robustness checks, sealed shortlists. See its `README.md` for a script-by-script table |
| `experiments/<other>_v1/` | Earlier experiment rounds, kept as the archived record; their stored predictions are inputs to the revision scripts |
| `tests/` | Unit tests for aggregation, leakage, ranking, fusion, NDCG and inference |
| `docs/` | Research design, methodology and data dictionary |
| `reports/` | Data audit, coverage tables and source registry with file hashes |
| `CHANGELOG_REVISION.md` | What changed during the revision and why |

## Data

No raw data are stored in the repository. The pipeline downloads public sources:

- CEPII BACI, HS 2012 nomenclature, release V202601 (bilateral trade flows);
- World Bank World Development Indicators and Worldwide Governance Indicators;
- CEPII Gravity / GeoDist (bilateral variables).

`reports/data_source_registry.csv` records the files used and their hashes. Trade values are converted from
BACI's thousand current US dollars to US dollars. A bilateral outcome is set to zero only when the importer
reports positive world imports of the chapter in that year; otherwise it is left missing.

The WDI and WGI series are the vintages downloaded in 2026, so the evaluation reproduces the publication lag of
trade data but is not a real-time exercise.

## Setup

Python 3.11 was used.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Base pipeline (downloads the data, builds the panel and the archived experiments):

```bash
python run_pipeline.py --mode acquire
python run_pipeline.py --mode validate
python run_pipeline.py --mode prepare
python run_pipeline.py --mode full
```

Evaluation behind the manuscript:

```bash
export PYTHONPATH=src:experiments/paper_revision_v1/src:.
python experiments/paper_revision_v1/run_revision.py               # re-fit LightGBM, evaluate, write the main tables
python experiments/paper_revision_v1/run_revision.py --skip-refit  # reuse stored LightGBM scores
python experiments/paper_revision_v1/src/aligned_baselines.py      # objective-aligned LightGBM variants
python experiments/paper_revision_v1/src/subperiods.py             # length of training history
python experiments/paper_revision_v1/src/grading_sensitivity.py    # alternative relevance gradings
python experiments/paper_revision_v1/src/mechanism_checks.py       # fusions with the import-demand rule
python experiments/paper_revision_v1/src/stacking.py               # stacked classifier
python experiments/paper_revision_v1/src/xai_decision.py           # explanation tests, discovery
python experiments/paper_revision_v1/src/xai_hybrid_faithfulness.py
```

`experiments/paper_revision_v1/README.md` lists every script with its outputs.

The revision scripts read intermediate files that are not in the repository because they are large or
regenerable: the processed panel, the stored predictions of the archived experiments and the saved models. They
are produced by the base pipeline and by `refit_lightgbm.py --save-models`. Result tables in CSV form are
committed under `experiments/paper_revision_v1/results/`, so the reported numbers can be inspected without
re-running anything.

## Tests

```bash
PYTHONPATH=src:experiments/paper_revision_v1/src:. python -m pytest -q tests
```

## Inference

Every contrast is paired within pools and reported with three layers (`src/statistics.py`): an HS2-block
bootstrap interval with a block sign-flip p-value, a t-statistic with standard errors clustered by chapter and
year, and an exact year-level sign-flip test, whose smallest attainable p-value is 0.016 with seven years.
Holm adjustment is applied within a family of three contrasts per task.

## Status and licence

The manuscript is under review, and no licence has been chosen for this repository yet. Please contact the
authors before reusing the code or the results.
