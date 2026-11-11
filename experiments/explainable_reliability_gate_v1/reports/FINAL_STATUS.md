# Final status: COMPLETE_REDUCED_SPECIFICATION

Research evidence status: **EXPLORATORY_ONLY**. The 2022–2024 outcomes were previously inspected. The local BACI V202601 outcomes end in 2024; no fresh independent holdout was found.

Contribution classification: **EMPIRICAL_CONTRIBUTION_ONLY**. Expert routing and selective automation are established methods. This work is a pool-level empirical application; human-review performance is not observed.

Missing required outputs: none.

Task A strict-zero candidate rows: 29737; Task B rows: 12962. Unknown labels are retained as NA.

Historical LightGBM OOF predictions use seed 11 only; a multi-seed stability experiment was not run. The Ridge gate is deterministic. This is one reason for the reduced-specification status.

Excel workbook: `results/ERGM_FULL_RESEARCH_RESULTS.xlsx`.

Reproduction: `.venv_uv/bin/python -m experiments.explainable_reliability_gate_v1.run_ergm --stage full`, followed by `.venv_uv/bin/python -m pytest experiments/explainable_reliability_gate_v1/tests -q`.

There are no figures. Human-AI review efficacy, true prospective reliability, future-cycle performance, and causal economic returns remain unverified.
