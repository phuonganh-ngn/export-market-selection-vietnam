# External protocol freeze and exposure audit

Initial SHA-256, recorded before external archive extraction: `af7ea0e77835082703c8e4d20cc95bdd841b79b6a953229ed5d473d96be5b179`.

Current full protocol SHA-256 (Task B regression and local OOF calibration details made explicit before any model fit or ranking evaluation): `16f7d097f74f584422137068e14616b5fd3c8254a80d58a3efb5e1c4dc1d3688`. The protocol fixes VNM as reference, THA as primary and IDN as secondary; outer years 2022–2024; the y−2 cutoff; the $50,000 entry threshold; the model families; fixed hyperparameters; top-K budgets; and evaluation definitions.

## Exposure deviation

During a source-panel debugging check after extraction, the THA 2022 candidate panel printed a few example rows containing observed 2022 exports and aggregate counts of known/qualifying labels. No model predictions, score comparisons, ranking metrics or outcome-based country choices were computed or inspected at that point. This is nevertheless direct exposure to THA evaluation outcomes before prediction freezing. Therefore **THA must be labeled exploratory geographic replication**, not a clean untouched holdout. The country choice and protocol were fixed before this exposure and were not changed in response.

IDN's evaluation outcomes were not displayed or summarized before the protocol was frozen; the raw archive was processed once to aggregate the three fixed exporter series, but IDN labels were not inspected by the analyst before this specification. IDN remains the only potentially predeclared external evaluation, subject to successful checksum-protected prediction-before-outcome scoring. Cross-country replication is not new temporal-cycle validation.

No model tuning, threshold selection, feature selection or country selection will use outer evaluation metrics. The protocol amendment only records the already intended Task B regression baseline and temporal OOF calibration mechanics.

Prediction manifest checksum: `9945395caa95d1821169b953a0918f7dc22bec339605822f76ada4ba061ad06b`. External predictions were saved before outcome labels were merged for metrics. The score columns contain calibrated probabilities under `entry_probability_calibrated`; the legacy `_raw` name is documented in `PREDICTION_MANIFEST_NOTE.md`.
