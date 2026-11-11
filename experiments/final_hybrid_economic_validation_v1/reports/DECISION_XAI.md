# Decision XAI

The blend explanation decomposes the within-pool rank movement across PPML, LightGBM, and the equal-weight blend for destinations selected by at least one expert. It shows ranking complementarity and subsequent observed outcomes separately; outcomes are retrospective and do not explain the model at decision time. The gate explanation uses coefficient contributions from the fitted imputation/scaling/one-hot Ridge pipeline.

The outer risk-gate score reconstruction maximum absolute error is 1.78e-15; decision reproduction is 1.000 on nominal-80% records. Explanations are predictive diagnostics, not causal economic effects.

A compact blend example set (pools where the blend's Top-10 differs from at least one expert) is saved at `results/F6_blend_rank_explanations.csv`; the full candidate-level record supports verification.
