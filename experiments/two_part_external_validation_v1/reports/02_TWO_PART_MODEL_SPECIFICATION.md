# 02 — Two-Part model specification

For eligible strict-zero candidate i, let Vᵢ be outcome-year exports and Yᵢ=1(Vᵢ≥$50,000). The primary two-part score is pᵢmᵢ, with pᵢ=P(Yᵢ=1|Xᵢ) and mᵢ=E[Vᵢ|Yᵢ=1,Xᵢ]. Thus pᵢmᵢ=E[Vᵢ1(Vᵢ≥$50,000)|Xᵢ], a qualified-export-value expectation. It is not E[Vᵢ|Xᵢ] when positive sub-threshold trade occurs.

The entry probability is the archived LightGBM classifier score, Platt-calibrated on historical rolling-origin OOF Task A scores and labels only. For each test year y, calibration uses OOF outcomes no later than y−2 (bounded by the OOF archive ending in 2021). Conditional value is a pooled log-link Poisson pseudo-maximum-likelihood model fitted only on verified strict-zero candidates with observed exports ≥$50,000 and outcomes through y−2. Predictions are projected to the support lower bound $50,000. This is a constrained conditional-mean estimate, not a model of total exports.

The nonlinear conditional-value model was not fit: the project did not establish a defensible retransformation to an arithmetic conditional mean under the available time and validation protocol. No alternative was selected by final test performance.
