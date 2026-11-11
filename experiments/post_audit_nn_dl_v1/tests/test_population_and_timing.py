import numpy as np
import pandas as pd

from experiments.post_audit_nn_dl_v1.src.data import load_frozen_population
from experiments.post_audit_nn_dl_v1.src.pipeline import (
    temporal_outer, _impute_indicator_score, _make_task_populations,
)


def test_rebuilt_candidate_manifest_keeps_all_missing_labels_missing():
    _, candidates, observed = load_frozen_population(write=False)
    assert len(candidates) == 62_797
    assert len(observed) == 62_238
    assert int((~candidates.label_observed).sum()) == 559
    assert candidates.loc[~candidates.label_observed, "export_value_usd"].isna().all()
    assert candidates.candidate_id.is_unique


def test_outer_split_keeps_unknown_outcome_candidate_in_test_universe():
    frame = pd.DataFrame({
        "outcome_year": [2018, 2020, 2022], "feature_year": [2016, 2018, 2020],
        "outcome_observed": [True, True, False], "export_value_usd": [0.0, 10.0, np.nan],
        "destination_iso3": ["AAA", "AAA", "AAA"], "hs2": ["01", "01", "01"],
    })
    train, test = temporal_outer(frame, 2022, "regression")
    assert train.outcome_year.max() == 2020
    assert len(test) == 1 and not bool(test.outcome_observed.iloc[0])
    assert pd.isna(test.export_value_usd.iloc[0])


def test_indicator_score_imputation_uses_training_cutoff_only():
    panel = pd.DataFrame({
        "destination_iso3": ["AAA", "BBB", "AAA", "BBB"],
        "outcome_year": [2020, 2020, 2022, 2022],
        "hs2": ["01", "01", "01", "01"],
        "log_gdp": [1.0, 3.0, 100.0, 200.0],
        "log_destination_import_demand": [4.0, 6.0, 100.0, 200.0],
    })
    pool = panel.loc[panel.outcome_year.eq(2022)].copy()
    pool.loc[pool.destination_iso3.eq("AAA"), ["log_gdp", "log_destination_import_demand"]] = np.nan
    gdp, ngdp = _impute_indicator_score(panel, pool, 2022, "GDP")
    demand, ndemand = _impute_indicator_score(panel, pool, 2022, "Historical_import_demand")
    assert ngdp == ndemand == 1
    assert gdp.iloc[0] == 2.0
    assert demand.iloc[0] == 5.0


def test_underexploited_market_cutoff_uses_full_history_not_eval_slice():
    rows = []
    for hs2, shares in [("01", [0.10, 0.20]), ("02", [0.05, 0.15])]:
        for i, share in enumerate(shares):
            rows.append({"outcome_year": 2020, "hs2": hs2, "destination_iso3": f"H{i}{hs2}",
                         "outcome_observed": True, "historical_export_value": 1.0,
                         "historical_market_share": share, "entry_candidate_ex_ante": False})
        # This later historical row must be excluded from the 2022 (y-2=2020) cutoff.
        rows.append({"outcome_year": 2021, "hs2": hs2, "destination_iso3": f"L{hs2}",
                     "outcome_observed": True, "historical_export_value": 1.0,
                     "historical_market_share": 0.0, "entry_candidate_ex_ante": False})
        for i, share in enumerate(shares):
            rows.append({"outcome_year": 2022, "hs2": hs2, "destination_iso3": f"T{i}{hs2}",
                         "outcome_observed": True, "historical_export_value": 1.0,
                         "historical_market_share": share, "entry_candidate_ex_ante": False})
    full = pd.DataFrame(rows)
    evaluation = full.loc[full.outcome_year.eq(2022)].copy()
    populations = _make_task_populations(evaluation, 2022, history_panel=full)
    under = populations["Task_B_existing_underexploited"]
    assert len(under) == 2
    assert set(under.hs2) == {"01", "02"}
