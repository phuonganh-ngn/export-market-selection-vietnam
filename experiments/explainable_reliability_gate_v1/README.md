# Explainable Reliability-Gated Export Market Selection (ERGM v1)

Versioned empirical extension. It reuses the M1 processed HS2 panel and saved 2022–2024 Gravity–PPML/LightGBM predictions, and reconstructs leakage-safe historical rolling-origin expert predictions for 2018–2021. It does not modify earlier research folders or download BACI again.

## Primary task definitions

- **New entry:** no verified Vietnam exports in each of the three origin-known years; success is outcome-year exports at least USD 50,000. This corrects the prior overlap with expansion. The earlier below-USD-50,000 inactivity rule is retained as an audit sensitivity.
- **Existing expansion:** positive historical Vietnam exports and historical market share at or below the HS2 median estimated from observations no later than the feature cutoff.
- The pool unit is task × HS2 × outcome year. The gate selects a model for the whole Top-10 list and may defer the whole product-year pool.

## Reproduction

From `M1_Export_Market_Selection/`:

```bash
.venv_uv/bin/python -m experiments.explainable_reliability_gate_v1.run_ergm --stage audit
.venv_uv/bin/python -m experiments.explainable_reliability_gate_v1.run_ergm --stage oof --force-oof
.venv_uv/bin/python -m experiments.explainable_reliability_gate_v1.run_ergm --stage full
.venv_uv/bin/python -m pytest experiments/explainable_reliability_gate_v1/tests -q
```

The full run saves fold models and prediction tables, then writes the reports and `ERGM_FULL_RESEARCH_RESULTS.xlsx`. It is CPU-only, fixes LightGBM threads at four, uses chronological M1 inner validation, and uses deterministic Ridge routing. No figures are generated.

## Interpretation limits

The routing “experts” are PPML and LightGBM, not people. The reliability output is a predicted regret score, not a calibrated probability or formal safety guarantee. Deferred pools are a review queue; actual review decisions and human accuracy are unavailable. Complete-case retrospective metrics preserve unknown outcomes as missing. The 2022–2024 outcomes were already inspected in prior M1 development, so this extension is exploratory rather than an independent confirmation.
