# Decision-focused ranking and explanation extension

Versioned extension of M1. It uses the existing lag-2 processed panel and verified audit/neural prediction files; it does not rewrite them or download BACI again. The frozen population and all output files are local to this directory.

## Reproduction

From `M1_Export_Market_Selection/` with `.venv_uv` active:

```bash
.venv_uv/bin/python -m experiments.decision_focused_xai_v1.run_decision_study --stage audit
.venv_uv/bin/python -m experiments.decision_focused_xai_v1.run_decision_study --stage train
.venv_uv/bin/python -m experiments.decision_focused_xai_v1.run_decision_study --stage rank_xendcg
.venv_uv/bin/python -m experiments.decision_focused_xai_v1.run_decision_study --stage evaluate
.venv_uv/bin/python -m experiments.decision_focused_xai_v1.run_decision_study --stage full
.venv_uv/bin/python -m pytest experiments/decision_focused_xai_v1/tests -q
```

The outer folds are chronological and use the audited two-year feature and outcome cutoffs. Hyperparameter selection uses the last admissible historical validation query year. Five seeds are fitted. The stage commands write checkpoints and are safe to resume.

`rank_xendcg` is an optional objective sensitivity. It holds the Lambdarank-selected settings and iteration count fixed, changes only the LightGBM ranking objective, and is reported as exploratory rather than separately tuned confirmatory evidence.

## Model and explanation provenance

LambdaMART uses the installed LightGBM `LGBMRanker` with `objective="lambdarank"`, query group sizes, integer labels and explicit gain arrays. Product-year queries are supplied in contiguous sorted order, with group sizes summing to the training row count. Native LightGBM `pred_contrib` explains additive raw ranking scores; pairwise score differences are decomposed by differencing those contributions. This is a score explanation, not an explanation of causal effects or of realized listwise utility.

Task A relevance is the audited binary USD 50,000 entry outcome. Task B training labels use fold-specific, training-only positive-increment quantile cutoffs; the primary evaluation remains the audited capped relevance grade. A secondary NDCG metric uses the fold-specific training relevance cutoffs. This distinction is explicit in all outputs.

All reported test metrics use observed-label candidates only, after ranking candidates within the full ex-ante pool. The 559 missing outcome labels remain NA. Rankings of the complete candidate universe and missing-label positions are retained separately.

WDI and WGI historical values are revised vintages; results are retrospective rather than point-in-time forecasts. Three outer years cannot establish future-cycle stability. No figures are generated.
