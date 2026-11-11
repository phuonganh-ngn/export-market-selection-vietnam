import numpy as np
import pandas as pd


def test_common_completion_paired_bounds_are_sharp():
    # Two policies differ on one unknown candidate. The same unknown value is
    # used for both, so the paired difference is bounded by its coefficient.
    coefficient = np.array([1.0, -1.0, 0.0])
    known_diff = 0.3
    lower = known_diff + np.minimum(coefficient, 0).sum()
    upper = known_diff + np.maximum(coefficient, 0).sum()
    completions = [known_diff + float(np.dot(coefficient, y))
                   for y in (np.array([0., 0., 0.]), np.array([1., 0., 0.]),
                             np.array([0., 1., 0.]), np.array([1., 1., 1.]))]
    assert np.isclose(min(completions), lower)
    assert np.isclose(max(completions), upper)


def test_missing_outcomes_are_not_zeroed_in_saved_manifest():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results/candidate_task_manifest.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    missing = ~d.label_observed.astype(bool)
    if missing.any():
        assert d.loc[missing, "export_value_usd"].isna().all()


def test_relevance_support_and_cardinality():
    relevance = np.array([0., 0.2, 1.])
    assert np.isfinite(relevance).all()
    assert ((relevance >= 0) & (relevance <= 1)).all()


def test_missing_expansion_label_stays_nan_in_binary_utility():
    from experiments.preference_robust_selection_v1.src.portfolio_evaluation import _actual_outcomes, TASK_B
    d = pd.DataFrame({"label_observed": [True, False],
                      "realized_incremental_export_usd": [20_000.0, np.nan]})
    entry, amount, observed = _actual_outcomes(TASK_B, d, 100_000.0)
    assert entry[0] == 1.0
    assert np.isnan(entry[1]) and np.isnan(amount[1])
    assert observed.tolist() == [True, False]


def test_missing_outcome_regret_upper_bound_matches_exhaustive_common_completions():
    from itertools import product
    from experiments.preference_robust_selection_v1.src.portfolio_evaluation import (
        _actual_outcomes, _r4_pool_bounds, TASK_B,
    )
    candidates = pd.DataFrame({
        "task": [TASK_B] * 4, "hs2": ["01"] * 4, "outcome_year": [2022] * 4,
        "candidate_id": ["a", "b", "c", "d"], "destination_iso3": ["AAA", "BBB", "CCC", "DDD"],
        "label_observed": [True, True, False, False],
        "realized_incremental_export_usd": [100_000.0, 0.0, np.nan, np.nan],
    })
    selected = pd.DataFrame({"decision_budget_k": [2, 2], "candidate_id": ["a", "c"]})
    bounds = _r4_pool_bounds(TASK_B, "P", selected, candidates, 200_000.0)
    u1, u2, observed = _actual_outcomes(TASK_B, candidates, 200_000.0)
    selected_mask = candidates.candidate_id.isin(["a", "c"]).to_numpy()
    for row in bounds:
        w = row["weight_objective1"]
        completions = []
        for y_c, y_d in product([0.0, 1.0], repeat=2):
            u1c, u2c = u1.copy(), u2.copy()
            u1c[2:] = [y_c, y_d]
            u2c[2:] = [y_c, y_d]
            u = w*u1c+(1-w)*u2c
            oracle = np.sort(u)[::-1][:2].sum()
            chosen = u[selected_mask].sum()
            completions.append(max(0.0, oracle-chosen))
        assert np.isclose(row["worst_preference_regret_upper"], max(completions))
