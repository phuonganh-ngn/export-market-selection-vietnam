import pandas as pd
import pytest


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_audited_candidate_cutoff(year):
    from experiments.preference_robust_selection_v1.src.data_audit import HERE
    path = HERE / "results/candidate_manifest.csv"
    if not path.exists():
        pytest.skip("Run candidate audit first")
    d = pd.read_csv(path)
    d = d.loc[d.outcome_year.eq(year)]
    assert not d.empty
    assert (d.historical_information_cutoff <= year - 2).all()


def test_holm_is_monotone():
    from statsmodels.stats.multitest import multipletests
    p = [0.001, 0.02, 0.04]
    adjusted = multipletests(p, method="holm")[1]
    assert (adjusted >= p).all()
    assert (adjusted[:-1] <= adjusted[1:]).all()
