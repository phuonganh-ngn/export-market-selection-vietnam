import numpy as np

from experiments.preference_robust_selection_v1.src.robust_optimization import (
    brute_force_regret, solve_maximin, solve_minimax_regret, solve_relative_robust, verify_solver_small_instances,
)
from itertools import combinations


def test_solver_matches_bruteforce():
    rows = verify_solver_small_instances()
    assert all(x["matches_enumeration"] for x in rows)
    assert all(x["absolute_mip_gap"] <= 1e-6 for x in rows)
    assert all(x["max_constraint_violation"] <= 2e-6 for x in rows)


def test_regret_nonnegative_and_collapse_to_single_preference():
    u = np.array([[0.2, 0.9, 0.5], [0.8, 0.3, 0.4]])
    r = solve_minimax_regret(u, 1)
    assert r.objective >= -1e-10
    oracle = np.sort(u, axis=1)[:, ::-1][:, :1].sum(axis=1)
    actual = float(np.max(oracle - u[:, r.selected].sum(axis=1)))
    assert abs(actual - r.objective) < 1e-7


def test_bruteforce_returns_feasible_portfolio():
    u = np.array([[0.1, 0.4, 0.2, 0.7], [0.5, 0.2, 0.6, 0.1]])
    val, chosen = brute_force_regret(u, 2)
    assert len(chosen) == 2
    assert np.isfinite(val) and val >= 0


def test_maximin_matches_exhaustive_portfolio_search():
    u = np.array([[.8, .2, .6, .4, .5], [.3, .9, .4, .7, .2], [.5, .4, .8, .3, .6]])
    k = 2
    result = solve_maximin(u, k)
    values = [float(np.min(u[:, list(chosen)].sum(axis=1)))
              for chosen in combinations(range(u.shape[1]), k)]
    assert np.isclose(result.objective, max(values), atol=1e-7)
    assert result.absolute_gap <= 1e-6


def test_relative_robust_matches_exhaustive_enumeration():
    a = np.array([.9, .4, .7, .2])
    b = np.array([.1, .8, .5, .6])
    weights = (.25, .75)
    k = 2
    result = solve_relative_robust(a, b, weights, k)
    ratios = []
    for chosen in combinations(range(len(a)), k):
        current = []
        for w in weights:
            values = w*a + (1-w)*b
            chosen_sum = values[list(chosen)].sum()
            oracle = np.sort(values)[::-1][:k].sum()
            current.append(chosen_sum / oracle)
        ratios.append(min(current))
    assert np.isclose(result.objective, max(ratios), atol=1e-7)


def test_relative_ratio_endpoint_reduction_matches_dense_preference_grid():
    rng = np.random.default_rng(1942)
    weights = np.linspace(.25, .75, 1001)
    for _ in range(30):
        a = rng.uniform(.01, 1, size=7)
        b = rng.uniform(.01, 1, size=7)
        chosen = rng.choice(len(a), size=3, replace=False)
        ratios = []
        for w in weights:
            u = w*a + (1-w)*b
            ratios.append(u[chosen].sum()/np.sort(u)[-3:].sum())
        endpoint = [ratios[0], ratios[-1]]
        assert min(ratios) >= min(endpoint) - 1e-10


def test_singleton_scenario_has_zero_regret_and_scenario_expansion_is_monotone():
    base = np.array([[.2, .9, .5, .4], [.8, .3, .4, .6]])
    expanded = np.vstack([base, [.1, .2, .95, .3]])
    narrow = solve_minimax_regret(base, 1)
    broad = solve_minimax_regret(expanded, 1)
    assert broad.objective + 1e-8 >= narrow.objective
    singleton = solve_minimax_regret(np.array([[.2, .9, .5, .4]]), 1)
    assert abs(singleton.objective) < 1e-8
