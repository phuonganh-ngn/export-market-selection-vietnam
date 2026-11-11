from pathlib import Path

import numpy as np
import pandas as pd

from experiments.explainable_reliability_gate_v1.src.decision_metrics import pool_decision_metrics
from experiments.explainable_reliability_gate_v1.src.ergm_pipeline import _cutoff_accept
from experiments.explainable_reliability_gate_v1.src.ergm_pipeline import _entry_partial_id_bounds

ROOT = Path(__file__).resolve().parents[1]


def _toy_entry(success, labels=None):
    n = len(success)
    if labels is None:
        labels = [True] * n
    return pd.DataFrame({
        "destination_iso3": [f"C{i:02d}" for i in range(n)],
        "label_observed": labels,
        "entry_success": [float(x) if obs else np.nan for x, obs in zip(success, labels)],
        "export_value_usd": [50_000.0 if x else 0.0 for x in success],
        "realized_incremental_export_usd": [0.0] * n,
        "score": list(range(n, 0, -1)),
    })


def test_primary_task_pools_are_disjoint_and_unknown_labels_remain_missing():
    manifest = pd.read_csv(ROOT / "results/candidate_manifest.csv")
    a = manifest.loc[manifest.task.eq("Task_A_new_entry")]
    b = manifest.loc[manifest.task.eq("Task_B_existing_underexploited")]
    assert not a.candidate_id.isin(b.candidate_id).any()
    assert int((~a.label_observed.astype(bool)).sum()) == 556
    assert int((~b.label_observed.astype(bool)).sum()) == 0
    assert a.loc[~a.label_observed.astype(bool), "entry_success"].isna().all()


def test_decision_metric_perfect_reverse_and_zero_relevance():
    perfect = _toy_entry([1, 1, 0, 0])
    reverse = _toy_entry([0, 0, 1, 1])
    p = pool_decision_metrics(perfect, "score", "Task_A_new_entry")
    r = pool_decision_metrics(reverse, "score", "Task_A_new_entry")
    z = pool_decision_metrics(_toy_entry([0, 0, 0, 0]), "score", "Task_A_new_entry")
    assert p["ndcg@5"] == 1.0
    assert p["precision@5"] == 0.5
    assert r["ndcg@5"] < p["ndcg@5"]
    assert np.isnan(z["ndcg@10"])
    assert z["regret@10"] == 0.0


def test_unknown_outcome_is_not_given_a_zero_label():
    d = _toy_entry([1, 0, 1], [True, False, True])
    m = pool_decision_metrics(d, "score", "Task_A_new_entry")
    assert m["unknown_label_count"] == 1
    assert m["observed_label_count"] == 2
    assert m["missing_in_full_score_top5"] == 1
    assert np.isfinite(m["ndcg@5"])


def test_matched_coverage_uses_pool_features_and_exact_cardinality():
    n = 11
    risk = np.arange(n, dtype=float)
    accepted, _ = _cutoff_accept(risk, risk, .8, pd.Series([f"p{i}" for i in range(n)]), matched=True)
    assert int(accepted.sum()) == int(np.ceil(.8 * n))
    assert np.array_equal(np.flatnonzero(accepted), np.arange(9))


def test_binary_missing_label_bounds_match_joint_exhaustive_completions():
    bounds = _entry_partial_id_bounds(known_total_success=1, selected_known_success=0,
                                      selected_missing=1, unselected_missing=2, k=2)
    regrets=[]; selected=[]
    for s in [0,1]:
        for u in [0,1,2]:
            selected.append(s)
            regrets.append(max(0,min(2,1+s+u)-s))
    assert bounds["selected_lower"] == min(selected)
    assert bounds["selected_upper"] == max(selected)
    assert bounds["regret_lower"] == min(regrets)
    assert bounds["regret_upper"] == max(regrets)


def test_temporal_model_cutoffs_and_explanation_reconstruction():
    temporal = pd.read_csv(ROOT / "results/temporal_validation.csv")
    hist = temporal.loc[temporal.train_outcome_max.notna()]
    assert (hist.train_outcome_max.astype(int) <= hist.year.astype(int) - 2).all()
    oof = pd.read_parquet(ROOT / "predictions/historical_oof_expert_predictions.parquet")
    assert oof.prediction_provenance.astype(str).str.contains("rolling-origin").all()
    fidelity_path = ROOT / "results/T7_explanation_fidelity.csv"
    if fidelity_path.exists():
        fidelity = pd.read_csv(fidelity_path)
        assert len(fidelity) > 0
        assert fidelity.raw_gate_score_reconstructed.astype(bool).all()
        assert fidelity.route_decision_reproduced.astype(bool).all()


def test_all_frozen_test_candidates_have_both_expert_scores_and_exact_bounds():
    manifest = pd.read_csv(ROOT / "results/candidate_manifest.csv")
    predictions = pd.read_parquet(ROOT / "predictions/ergm_candidate_predictions.parquet")
    keys = ["candidate_id", "task"]
    assert len(predictions) == len(manifest) == 42_699
    assert not predictions.duplicated(keys).any()
    assert set(map(tuple, predictions[keys].to_numpy())) == set(map(tuple, manifest[keys].to_numpy()))
    assert predictions[["ppml_score", "lightgbm_score"]].notna().all().all()
    bounds = pd.read_csv(ROOT / "results/T10_missing_outcome_sensitivity.csv")
    entry = bounds.loc[bounds.task.eq("Task_A_new_entry")]
    assert len(entry) == 576
    assert entry.entry_bound_status.astype(str).str.contains("sharp exact bound").all()

