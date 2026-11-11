# Final status

**Engineering: COMPLETE_REDUCED_SPECIFICATION.** The Vietnam two-part experiment, annual streamed BACI HS2 extraction, external VNM-to-THA/IDN transfer, country-specific replication, paired inference, workbook, and 26 automated integrity tests were executed. A separate reconstruction program recalculated 16,144 metric cells from saved prediction-level files; all matched saved VNM and external result tables within numerical tolerance. The external sample-flow table was corrected to deduplicate candidates repeated across B1/B2, then checked against BACI coverage counts. The optional nonlinear conditional-value model was not fit because an arithmetic-mean retransformation was not justified. Estimator objects were not checkpointed; prediction files and full re-runnable code are saved. External per-component runtime was not instrumented, so cost reporting is reduced.

**Evidence: GEOGRAPHIC_REPLICATION_ONLY.** VNM 2022–24 are previously inspected exploratory results. THA results are exploratory because a few outcome examples were exposed during panel debugging before prediction freeze. IDN was prespecified and the prediction manifest was sealed before evaluation labels were merged, so it is a predeclared geographic replication; it still uses the same 2022–24 world years and is not independent temporal-cycle validation. WDI/WGI historical vintages are revised.

**Contribution: DESCRIPTIVE_FINDINGS_ONLY.** Two-Part/hurdle modeling is established and did not outperform the leading original ranking methods consistently. Rank blending improves Task A NDCG relative to LightGBM in the repeatedly inspected VNM period, but a meaningful qualified-USD gain is not statistically established. PPML remains strong in Task B. Results support task- and exporter-dependent performance rather than a new method or universal economic advantage.

**Sample:** VNM Task A: 29,737 candidates, 29,181 observed and 556 unknown. THA Task A: 19,887 candidates, 532 unknown. IDN Task A: 24,079 candidates, 548 unknown. Task B totals: THA 19,287 (one unknown), IDN 17,111 (none unknown). Each country has 96 HS2 chapters and 288 HS2-year pools per task.

**Checks:** 26 tests pass; 16,144 independently reconstructed metric cells pass; external prediction manifest SHA-256 matches `data_manifest/external_prediction_manifest.json`; no figures were generated.

From the M1 root, see `experiments/two_part_external_validation_v1/README.md` for reproduction commands.
