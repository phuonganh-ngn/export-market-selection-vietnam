# M1 Two-Part Export Opportunity and Cross-Country Validation

This versioned extension reuses BACI HS12 V202601 and audited M1 predictions. It separates qualifying market entry from conditional successful-entry scale, then evaluates VNM development-period results and frozen THA/IDN transfer/replication. It never changes the original research artifacts.

## Reproduction

From `M1_Export_Market_Selection/`:

```bash
.venv_uv/bin/python experiments/two_part_external_validation_v1/run_experiments.py --stage all
.venv_uv/bin/python -m pytest experiments/two_part_external_validation_v1/tests -q
```

The run is checkpointed by stage. Use `--stage audit|vietnam|extract|external-predict|external-evaluate`. `--force` reruns a stage. Before a fresh all-stage run, the existing source extraction and prediction manifests can be reused.

## Main limitations

The two-part score estimates expected exports above the USD 50,000 qualifying threshold, not total export value. VNM 2022–24 is previously inspected. THA had limited outcome exposure during candidate-panel debugging and is exploratory. IDN is the prespecified external geographic replication. Neither is a new temporal-cycle holdout. WDI/WGI historical values are revised vintages. All USD performance is observational and noncausal.

Independent metric reconstruction from saved predictions: `.venv_uv/bin/python experiments/two_part_external_validation_v1/src/reconstruct_metrics.py`. This checks VNM summaries and each external HS2-year-model metric against the stored prediction files.
