# Feature-level XAI limitations

1. Model attribution is predictive, not causal. A SHAP contribution is relative to the declared training-derived background and can be sensitive to correlated predictors and one-hot country indicators.
2. The original LightGBM component score is a five-seed mean probability. Explanations average the exact five saved outer-fold classifiers in probability space; they are faithful to this ensemble score, while an individual seed's raw-margin SHAP is not substituted for it.
3. PPML estimator objects were archived, but preprocessing objects were not serialized alongside them. We deterministically reconstructed the transform from the original source function and fold training rows, and required prediction and rank agreement with the archive. This is prediction-equivalent pipeline recovery, not evidence that a preprocessing object itself was originally checkpointed.
4. PPML contributions are additive only on eta, the log expected mean in USD millions. They are not additive USD amounts and are not causal coefficients.
5. Hybrid rank-fusion terms are contributions of normalized ranks. SHAP and PPML eta terms use different scales and are never summed into a single feature-level Hybrid attribution. The exact final decision explanation is the combination of (i) component score attributions and (ii) the audited rank-fusion arithmetic.
6. The selected/displaced comparisons use observed later outcomes retrospectively. They are not causal replacement effects or deployment gains. Unknown labels remain unknown.
7. The 2022–2024 years have already informed M1 development; all outcome-linked explanations are exploratory.
8. Feature-group perturbation was not run because changing GDP, demand, governance, or geography independently would create an underjustified, potentially implausible synthetic covariate profile. No causal or intervention claim is made.
