import numpy as np
import pandas as pd

from src.pipeline import THRESHOLD, feature_years, _ndcg, _rank


def test_sequence_ends_at_two_year_information_cutoff():
    assert feature_years(2022, 5) == [2016, 2017, 2018, 2019, 2020]
    assert feature_years(2023, 5)[-1] == 2021
    assert feature_years(2024, 3) == [2020, 2021, 2022]
    assert max(feature_years(2022, 5)) <= 2022 - 2


def test_entry_threshold_and_unknown_policy():
    exports = pd.Series([0.0, 49999.0, 50000.0, np.nan])
    y = np.where(exports.notna(), exports.ge(THRESHOLD).astype(float), np.nan)
    assert list(y[:3]) == [0.0, 0.0, 1.0]
    assert np.isnan(y[3])


def test_deterministic_ranking_ties_use_iso3():
    d = pd.DataFrame({"score": [0.5, 0.5, 0.7], "destination_iso3": ["USA", "ABW", "CAN"]})
    assert _rank(d, "score").destination_iso3.tolist() == ["CAN", "ABW", "USA"]


def test_ndcg_zero_positive_pool_is_undefined():
    assert np.isnan(_ndcg(np.zeros(5), 10))
    assert _ndcg(np.array([1, 0, 0]), 3) == 1.0

