import pandas as pd
import pytest

from experiments.decision_focused_xai_v1.src.candidate_protocol import assert_outer_timing


def test_two_year_cutoff_passes():
    train = pd.DataFrame({"outcome_year": [2018, 2020], "feature_year": [2016, 2018]})
    test = pd.DataFrame({"outcome_year": [2022, 2022], "feature_year": [2020, 2020]})
    assert_outer_timing(train, test, 2022)


def test_future_training_outcome_fails():
    train = pd.DataFrame({"outcome_year": [2021], "feature_year": [2019]})
    test = pd.DataFrame({"outcome_year": [2022], "feature_year": [2020]})
    with pytest.raises(AssertionError):
        assert_outer_timing(train, test, 2022)


def test_future_feature_fails():
    train = pd.DataFrame({"outcome_year": [2018], "feature_year": [2017]})
    test = pd.DataFrame({"outcome_year": [2022], "feature_year": [2021]})
    with pytest.raises(AssertionError):
        assert_outer_timing(train, test, 2022)
