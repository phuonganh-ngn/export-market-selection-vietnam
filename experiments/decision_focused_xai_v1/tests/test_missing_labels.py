import numpy as np
import pandas as pd

from experiments.decision_focused_xai_v1.src.missing_label_sensitivity import observed_complete_cases


def test_unknown_labels_remain_unknown_in_primary_population():
    frame = pd.DataFrame({"label_observed": [True, False], "entry_success": [1.0, np.nan]})
    known = observed_complete_cases(frame)
    assert len(known) == 1
    assert np.isnan(frame.loc[1, "entry_success"])
