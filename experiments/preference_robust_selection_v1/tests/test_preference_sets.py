from experiments.preference_robust_selection_v1.src.preference_sets import PREFERENCE_SETS, vertices


def test_preference_sets_are_nested_and_valid():
    assert vertices(PREFERENCE_SETS["W_narrow"]) == (0.4, 0.6)
    assert vertices(PREFERENCE_SETS["W_primary"]) == (0.25, 0.75)
    assert vertices(PREFERENCE_SETS["W_broad"]) == (0.0, 1.0)
    assert PREFERENCE_SETS["W_narrow"][0] >= PREFERENCE_SETS["W_primary"][0]
    assert PREFERENCE_SETS["W_primary"][1] <= PREFERENCE_SETS["W_broad"][1]
