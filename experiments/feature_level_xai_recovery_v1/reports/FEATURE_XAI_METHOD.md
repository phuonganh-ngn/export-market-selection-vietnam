# Feature-level XAI method and model recovery

## Outer-fold model recovery

For each outer year, the five archived LightGBM binary-classification checkpoints were loaded with their fitted preprocessing objects. Their probabilities were recomputed over the frozen strict-zero Task A candidate manifest and matched against the seed-specific E6 archive; the seed-mean probability was also matched against the archived score used in the final hybrid. We use probability-scale TreeSHAP with a deterministic 128-row training-only background. For the five-seed score, feature attributions are averaged across the five fitted models; because each seed explanation is additive on the probability scale, the averaged attributions reconstruct the archived seed-mean probability.

The fitted PPML estimator is available for every year. Its original preprocessing object was not serialized with the estimator, so the preprocessing design was rebuilt using the archived `_design` implementation and the exact fold training rows. The transformed schema was checked against the coefficient count. Candidate predictions are `exp(eta)` in USD millions, converted to USD, then transformed by `log1p` for archived Task A ranking. The coefficient contributions are reported only on the PPML linear-predictor scale.

## Linking features and ranking

Within each HS2-year pool, candidates are ranked by descending component scores with ISO3 ascending as the deterministic tie-break. Each component rank is converted to the audited normalized inverse-rank contribution `(n-rank+1)/(n-1)`, and the Hybrid score is their equal-weight mean. Membership pairs are matched from the decision boundary inward: the weakest Hybrid-added candidate is paired with the strongest Hybrid-excluded candidate, then the next weakest/strongest pair, and so on. The exact pairwise Hybrid score difference is decomposed into its PPML and LightGBM rank terms. LightGBM SHAP differences explain the archived ensemble probability contrast; PPML coefficient differences explain the candidate contrast in log expected exports. These distinct scales are never added together.

SHAP global summaries use a deterministic candidate sample selected without reference to evaluation labels; all Top-10 membership-change candidates are included for local pair explanations. Outcome-conditioned cohort summaries are post-prediction diagnostics. Neither SHAP nor PPML contributions are causal effects.
