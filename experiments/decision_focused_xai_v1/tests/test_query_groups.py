import numpy as np
import pandas as pd

from experiments.decision_focused_xai_v1.src.ranking_targets import assert_query_groups, query_sort


def test_group_sizes_are_contiguous_and_sum_to_rows():
    frame = pd.DataFrame({"hs2": ["02", "01", "01", "02"], "outcome_year": [2020, 2020, 2020, 2020],
                          "destination_iso3": ["CAN", "USA", "CAN", "USA"], "x": [1, 2, 3, 4]})
    sorted_frame, _, labels, groups = query_sort(frame, np.array([0, 1, 0, 1]), ["x"])
    assert groups.tolist() == [2, 2]
    assert groups.sum() == len(frame)
    assert_query_groups(sorted_frame, groups, labels)
