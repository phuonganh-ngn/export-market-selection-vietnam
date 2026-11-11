# Hybrid rank-fusion tie sensitivity

This standalone robustness audit evaluates the exact Task A score archive from the frozen final hybrid validation. It does not retrain models, alter candidates, tune weights, or use outcomes to select a tie rule.

The integer tie key is the sum of the component within-pool ranks. For a fixed pool, equal rank sums are mathematically identical to equal equal-weight normalized-rank scores and avoid float-comparison ambiguity. Alternative component-first rules only order candidates within a tied Hybrid score: PPML rank then LightGBM rank then ISO3, or the reverse. Random ties uniformly permute ISO3-sorted members of each exact-score tie group, using 1,000 fixed seeds (10,000–10,999).

Run from the M1 project root:

```bash
.venv_uv/bin/python experiments/hybrid_tie_sensitivity_v1/src/run_tie_sensitivity.py
```

The 1,000 random repetitions quantify tie-order uncertainty only. They are not independent economic observations. Confidence intervals resample HS2 blocks and keep the three years together.
