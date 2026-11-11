# M1 Final Hybrid and Decision Reliability Validation

This versioned extension verifies the strict-zero Task A definition and the two-year information cutoff, reconstructs the Gravity–PPML/LightGBM rank blend from saved predictions, evaluates observed opportunity capture, and rechecks historical decision-risk gates. It preserves all earlier experiments.

Run from `M1_Export_Market_Selection/`:

```bash
.venv_uv/bin/python experiments/final_hybrid_economic_validation_v1/run_final_validation.py
.venv_uv/bin/python -m pytest experiments/final_hybrid_economic_validation_v1/tests -q
```

All 2022–2024 evaluations are exploratory retrospective evidence because those outcomes informed earlier M1 development. BACI V202601 ends in 2024. No figures are produced. Missing outcomes remain missing; complete-case metrics are labeled as such.
