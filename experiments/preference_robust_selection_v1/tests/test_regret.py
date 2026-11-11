import numpy as np

from experiments.preference_robust_selection_v1.src.decision_explanations import _best_one_swap


def test_vectorized_one_swap_matches_exhaustive_recomputation():
    rng = np.random.default_rng(114)
    util = rng.random((7, 9))
    oracle = np.sort(util, axis=1)[:, ::-1]
    selected = np.array([0, 2, 5])
    candidate = 8
    ids = [f"D{i}" for i in range(util.shape[1])]

    expected = []
    for removed in selected:
        trial = np.r_[selected[selected != removed], candidate]
        regret = oracle[:, :len(trial)].sum(axis=1) - util[:, trial].sum(axis=1)
        expected.append((float(regret.max()), str(ids[removed]), int(removed)))
    expected.sort()

    removed, worst_regret = _best_one_swap(util, oracle, selected, candidate, ids)
    assert removed == expected[0][2]
    assert np.isclose(worst_regret, expected[0][0])
