import pandas as pd
import pytest

from experiments.decision_focused_xai_v1.src.statistical_inference import (
    _cluster_inference,
    align_paired_pool_metrics,
)


def test_pairing_uses_only_identical_product_year_pools():
    a = pd.DataFrame({"hs2": ["01", "02"], "outcome_year": [2022, 2022], "ndcg": [0.7, 0.2]})
    b = pd.DataFrame({"hs2": ["01", "03"], "outcome_year": [2022, 2022], "ndcg": [0.5, 0.8]})
    pair = align_paired_pool_metrics(a, b, "ndcg")
    assert pair[["hs2", "outcome_year"]].values.tolist() == [["01", 2022]]
    assert pair.difference.iloc[0] == pytest.approx(0.2)


def test_cluster_bootstrap_point_estimate_matches_pool_weighted_macro():
    # HS2 01 contributes one valid pool, HS2 02 contributes three. The point
    # estimate must match the pool-weighted macro estimand used in E6 tables.
    frame = pd.DataFrame({"hs2": ["01", "02", "02", "02"],
                          "difference": [0.1, 0.3, 0.3, 0.3]})
    out = _cluster_inference(frame, "difference", seed=11)
    assert out["estimate"] == pytest.approx(0.25)
