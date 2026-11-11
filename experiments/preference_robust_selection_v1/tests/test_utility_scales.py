import numpy as np

from experiments.preference_robust_selection_v1.src.portfolio_evaluation import _scale


def test_log_scale_is_invariant_to_currency_units_when_constants_match():
    values_usd = np.array([50_000.0, 125_000.0, 800_000.0, 2_500_000.0])
    cap_usd = 2_000_000.0
    values_thousands = values_usd / 1_000.0
    cap_thousands = cap_usd / 1_000.0
    threshold_thousands = 50_000.0 / 1_000.0
    in_usd = np.clip(np.log1p(np.minimum(values_usd, cap_usd) / 50_000.0)
                     / np.log1p(cap_usd / 50_000.0), 0, 1)
    in_thousands = np.clip(np.log1p(np.minimum(values_thousands, cap_thousands) / threshold_thousands)
                           / np.log1p(cap_thousands / threshold_thousands), 0, 1)
    np.testing.assert_allclose(in_usd, in_thousands, rtol=0, atol=1e-14)


def test_utility_is_bounded_and_monotone():
    values = np.array([-1.0, 0.0, 50_000.0, 250_000.0, 1e9])
    utility = _scale(values, 500_000.0)
    assert np.all((utility >= 0) & (utility <= 1))
    assert np.all(np.diff(utility) >= 0)
