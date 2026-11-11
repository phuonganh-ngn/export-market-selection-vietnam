# Post-audit neural learning extension

This versioned experiment adds an MLP and the official Yandex Research TabM implementation to the audited HS2 study. It reads existing processed inputs and `audit_v2` prediction files; it does not modify original research artifacts or raw data.

## Reproduction

From `M1_Export_Market_Selection/`:

```bash
uv pip install --python .venv_uv/bin/python -r experiments/post_audit_nn_dl_v1/requirements_nn.txt
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.runner --stage audit
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.runner --stage discovery
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.runner --stage regression
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.runner --stage full
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.supplementary --bootstrap-reps 1000
.venv_uv/bin/python -m experiments.post_audit_nn_dl_v1.src.finalize
.venv_uv/bin/python -m pytest experiments/post_audit_nn_dl_v1/tests -q
PYTHONPATH=src .venv_uv/bin/python -m pytest tests -q
```

The full stage reuses verified per-fold checkpoints and prediction files. Use `--force` only to refit files inside this versioned extension. Training is sequential, CPU-only, capped at four PyTorch threads, and uses at most 100 epochs with 12-epoch early stopping. The primary runs use five seeds: 11, 23, 37, 51, and 73.

## Scope and integrity

- Frozen primary evaluation universe: 62,797 ex-ante destination–HS2–year candidates, including 559 records with unknown outcome labels.
- Known-outcome forecast evaluation: 62,238 observations. New-entry evaluation: 41,131 ex-ante inactive candidates, of which 40,572 outcomes are observed.
- Entry is defined using three years of verified prior reporting, exports below USD 50,000 in each year, and outcome exports of at least USD 50,000. Unknown outcomes remain missing.
- Outer years are 2022–2024. Every model uses the audited two-year feature and training-outcome cutoff. Tuning uses an earlier chronological validation fold.
- `audit_v2` contains the prior verified PPML, tree, and corrected candidate predictions used as baselines. New model predictions are stored separately by task, fold, and seed.
- Current revised WDI and WGI historical values do not support a genuinely point-in-time publication-vintage claim.

See `reports/INPUT_AUDIT.md`, `reports/TEMPORAL_VALIDATION.md`, and `reports/RESEARCH_INTEGRITY.md` for actual completion and limitations.
