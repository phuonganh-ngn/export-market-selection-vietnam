import numpy as np
import pandas as pd
import pytest

from experiments.post_audit_nn_dl_v1.src.metrics import (
    capped_export_grade, discovery_pool_metrics, ndcg_at_k, ndcg_from_gain,
    precision_at_k, rank_pool,
)


def test_ndcg_perfect_reversed_and_zero_relevance():
    relevance = np.array([3, 2, 0])
    assert ndcg_at_k(relevance, 3) == pytest.approx(1.0)
    assert ndcg_at_k(relevance[::-1], 3) < 1.0
    assert np.isnan(ndcg_at_k(np.zeros(4), 10))


def test_precision_uses_short_pool_denominator():
    assert precision_at_k([1, 0, 1], 10) == pytest.approx(2 / 3)
    assert precision_at_k([], 10) is np.nan or np.isnan(precision_at_k([], 10))


def test_missing_labels_are_not_valid_metric_inputs():
    with pytest.raises(ValueError):
        discovery_pool_metrics([1, np.nan, 0], [0.8, 0.4, 0.1])


def test_score_ties_are_deterministically_broken_by_iso3():
    frame = pd.DataFrame({"destination_iso3": ["VNM", "AUS", "CAN"], "score": [1.0, 1.0, 1.0]})
    assert rank_pool(frame, "score").destination_iso3.tolist() == ["AUS", "CAN", "VNM"]


def test_gain_definitions_are_nonnegative_and_cap_is_bounded():
    grade = capped_export_grade([-1, 0, 50_000, 10**12])
    assert grade.tolist()[0:2] == [0, 0]
    assert grade.min() >= 0 and grade.max() <= 4
    assert ndcg_from_gain([1, 0], [0, 1], 10) == pytest.approx(1.0)


def test_new_entry_relevance_is_binary_and_distinct_from_expansion_grade():
    # USD 50,000 is an entry success, while the separately defined expansion grade
    # need not assign positive graded relevance at that exact boundary.
    assert ndcg_at_k([1, 0], 10) == pytest.approx(1.0)
    assert capped_export_grade([50_000], threshold=50_000).tolist() == [0]


def test_standard_ndcg_is_zero_when_ranker_misses_all_positives_at_k():
    # The positive item exists in the pool but lies below the selected top-1.
    assert ndcg_at_k([0, 1], 1) == 0.0
