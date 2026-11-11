# Final status

Engineering status: **COMPLETE_REDUCED_SPECIFICATION**. The audited candidate universe was reconstructed, the blend was reproduced from saved expert scores, F1–F7 numerical outputs and reports were generated, ridge risk gates and operational thresholds were fitted from OOF development data, and the final extension integrity suite passed (11 tests); the original ERGM suite also passed (7 tests). Base PPML/LightGBM experts were reused; no historical neural, LambdaMART, or robust-selection experiments were rerun.

Evidence status: **EXPLORATORY_ONLY**. The 2022–2024 outcomes have informed previous method development. The official CEPII BACI V202601 source ends in 2024; the tested official UN Comtrade public availability endpoint returned no annual 2024/2025 Vietnam records on 2026-10-13.

Contribution classification: **EMPIRICAL_CONTRIBUTION_SUPPORTED** only in the narrow sense of a reproducible, task-specific retrospective evaluation; no novel method or independently validated superiority is established.

Known limitations: 556 unknown Task A labels; revised WDI/WGI vintages; only three previously inspected outer years; archived base predictions do not contain independent outer seed runs; clipped risk predictions create ties; no human review outcomes or costs; observed trade is not profit or causal model benefit. Outer seed checks, destination identity ablation, and alternative risk-calibration windows are explicitly not run.

Reproduction from the project root: `.venv_uv/bin/python experiments/final_hybrid_economic_validation_v1/run_final_validation.py` and `.venv_uv/bin/python -m pytest experiments/final_hybrid_economic_validation_v1/tests -q`. No figures were created.
