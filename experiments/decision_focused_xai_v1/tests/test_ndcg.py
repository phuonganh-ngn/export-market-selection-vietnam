import numpy as np
import pytest

from experiments.decision_focused_xai_v1.src.ranking_metrics import ndcg_from_gains, order_indices


def test_perfect_and_reversed_ndcg():
    gains = np.array([3.0, 1.0, 0.0])
    assert ndcg_from_gains(gains, gains, 10) == pytest.approx(1.0)
    assert ndcg_from_gains(gains[::-1], gains, 10) < 1.0


def test_zero_relevance_is_undefined_and_short_lists_work():
    assert np.isnan(ndcg_from_gains([0, 0], [0, 0], 10))
    assert ndcg_from_gains([1], [1], 10) == pytest.approx(1.0)


def test_tied_scores_use_deterministic_iso3_order():
    assert order_indices([1, 1, 2], ["NZL", "AUS", "USA"]).tolist() == [2, 1, 0]
