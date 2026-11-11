import numpy as np
import pandas as pd
import pytest

from experiments.post_audit_nn_dl_v1.src import analysis
from experiments.post_audit_nn_dl_v1.src.metrics import destination_block_bootstrap
from experiments.post_audit_nn_dl_v1.src.metrics import ndcg_at_k


def test_destination_block_bootstrap_preserves_paired_pools_and_task_relevance(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(analysis, "OUT", tmp_path)
    rows = []
    models = {"TabM": [11, 23, 37, 51, 73], "MLP": [11, 23, 37, 51, 73],
              "Gravity_PPML": [0], "LightGBM": [0]}
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for model, seeds in models.items():
            for seed in seeds:
                for i in range(20):
                    inc = 500_000.0 if i % 4 == 0 else 0.0
                    rows.append({"task": task, "evaluation_year": 2022, "hs2": "01",
                                 "ranking_model": model, "seed": seed,
                                 "destination_iso3": f"C{i:02d}", "candidate_id": f"{task}-{i}",
                                 "label_observed": True,
                                 "opportunity_score": float(20 - i + (1 if model == "TabM" and i < 5 else 0)),
                                 "realized_incremental_export_usd": inc,
                                 "entry_success": int(inc >= 50_000)})
    result = analysis.destination_block_ranking_bootstrap(pd.DataFrame(rows), reps=40, seed=12)
    assert len(result) == 10
    assert result.valid_paired_pools.eq(1).all()
    assert result.bootstrap_reps_valid.eq(40).all()
    assert result.ci95_low.notna().all() and result.ci95_high.notna().all()


def test_destination_block_bootstrap_uses_destination_level_paired_errors():
    rows = []
    for i in range(12):
        for hs2 in ["01", "02"]:
            for year in [2022, 2023]:
                rows.append({"destination_iso3": f"D{i:02d}", "hs2": hs2, "year": year,
                             "err_a": float(i + 1), "err_b": float(i + 2)})
    lo, hi, p, blocks = destination_block_bootstrap(pd.DataFrame(rows), reps=200, seed=7)
    assert blocks == 12
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(-1.0)
    assert 0 <= p <= 1


def test_destination_bootstrap_multiplicity_uses_each_models_own_rank_order():
    multiplicity = np.array([0, 2, 1], dtype=int)
    a = analysis._resample_ranked_gains(
        np.array([0, 1, 2]), np.array([1, 0, 1]), multiplicity)
    b = analysis._resample_ranked_gains(
        np.array([2, 0, 1]), np.array([1, 1, 0]), multiplicity)
    np.testing.assert_array_equal(a, np.array([0, 0, 1]))
    np.testing.assert_array_equal(b, np.array([1, 0, 0]))


def test_vectorized_weighted_ndcg_matches_expanded_destination_draws():
    ids = np.array([2, 0, 1], dtype=int)
    relevance = np.array([2, 0, 1], dtype=int)
    draws = np.array([[1, 0, 2, 0], [0, 1, 1, 1], [0, 0, 0, 3]], dtype=int)
    actual = analysis._weighted_ndcg_draws(ids, relevance, draws, k=10)
    expected = []
    for weights in draws:
        expanded = np.repeat(relevance, weights[ids])
        expected.append(ndcg_at_k(expanded, 10))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
