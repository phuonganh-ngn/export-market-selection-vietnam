import numpy as np
import pandas as pd

from experiments.decision_focused_xai_v1.src.ranking_targets import task_b_quantile_grade, training_increment_cutoffs


def test_task_b_cutoffs_are_derived_from_training_only():
    train = pd.DataFrame({"realized_incremental_export_usd": [0, 10, 20, 30, 40, 50]})
    test = pd.DataFrame({"realized_incremental_export_usd": [1e12]})
    cuts = training_increment_cutoffs(train)
    labels = task_b_quantile_grade(test.realized_incremental_export_usd, cuts)
    assert np.allclose(cuts, [20, 30, 40])
    assert labels.tolist() == [4]


def test_negative_and_zero_increment_have_relevance_zero():
    assert task_b_quantile_grade([-10, 0, 1], [2, 3, 4]).tolist() == [0, 0, 1]
