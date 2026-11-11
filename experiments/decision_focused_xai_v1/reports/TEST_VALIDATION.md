# Test and syntax validation

Executed from the M1 project root:

```text
.venv_uv/bin/python -m pytest experiments/decision_focused_xai_v1/tests -q
14 passed in 0.75s
.venv_uv/bin/python -m compileall -q experiments/decision_focused_xai_v1
exit code 0
```

The tests cover temporal leakage assertions, query-group ordering, integer relevance, NDCG edge cases and ties, nonnegative regret, missing-label preservation, additive pairwise explanation reconstruction, matched pool comparisons, and consistency of the cluster-bootstrap point estimand.
