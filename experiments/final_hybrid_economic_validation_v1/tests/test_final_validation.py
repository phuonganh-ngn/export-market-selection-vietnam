from pathlib import Path

import numpy as np
import pandas as pd

from experiments.final_hybrid_economic_validation_v1.src.final_validation import (
    TASK_A,
    TASK_B,
    _add_blend_scores,
    _task_gain,
    pool_metrics,
)


def entry_pool(successes, scores=None, observed=None):
    n = len(successes)
    return pd.DataFrame({
        "destination_iso3": [f"C{i:03d}" for i in range(n)],
        "entry_success": successes,
        "export_value_usd": np.asarray(successes, dtype=float) * 100_000,
        "realized_incremental_export_usd": np.zeros(n),
        "label_observed": [True] * n if observed is None else observed,
        "score": np.arange(n, 0, -1, dtype=float) if scores is None else scores,
    })


def test_entry_ndcg_perfect_and_reversed_rankings():
    frame = entry_pool([1, 1, 0, 0])
    perfect = pool_metrics(frame, "score", TASK_A)["ndcg@10"]
    frame["score"] = [1, 2, 3, 4]
    reversed_value = pool_metrics(frame, "score", TASK_A)["ndcg@10"]
    assert perfect == 1.0
    assert 0 <= reversed_value < perfect


def test_all_zero_relevance_ndcg_is_undefined():
    frame = entry_pool([0, 0, 0])
    result = pool_metrics(frame, "score", TASK_A)
    assert np.isnan(result["ndcg@10"])
    assert result["successful_or_meaningful_count@10"] == 0


def test_fewer_than_k_candidates_and_missing_labels_remain_unknown():
    frame = entry_pool([1, 0, 1], observed=[True, False, True])
    result = pool_metrics(frame, "score", TASK_A)
    assert result["candidate_count"] == 3
    assert result["observed_label_count"] == 2
    assert result["unknown_label_count"] == 1
    assert result["unknown_in_full_score_top10"] == 1
    assert result["partial_usd_is_complete"] is False
    assert result["successful_or_meaningful_count@10"] == 2


def test_blend_is_deterministic_and_rank_based():
    frame = pd.DataFrame({
        "task": [TASK_A] * 3,
        "hs2": ["01"] * 3,
        "outcome_year": [2022] * 3,
        "destination_iso3": ["BBB", "AAA", "CCC"],
        "ppml_score": [3.0, 2.0, 1.0],
        "lightgbm_score": [1.0, 2.0, 3.0],
    })
    first = _add_blend_scores(frame)
    second = _add_blend_scores(frame)
    assert np.allclose(first.blend_rank_score, second.blend_rank_score)
    # With opposing rankings, the destination ordering is determined by the
    # documented deterministic ISO3 tie-break.
    top = first.sort_values(["blend_rank_score", "destination_iso3"], ascending=[False, True])
    assert top.destination_iso3.tolist() == ["AAA", "BBB", "CCC"]


def test_expansion_gains_are_nonnegative_and_bounded_when_primary():
    frame = pd.DataFrame({"realized_incremental_export_usd": [-100, 0, 50_000, 1_000_000_000]})
    gains = _task_gain(frame, TASK_B, "primary")
    assert np.all(gains >= 0)
    assert gains.max() <= 15


def test_saved_candidate_manifest_is_disjoint_and_unknowns_are_not_zero():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results/final_candidate_manifest.csv"
    assert manifest_path.exists(), "Run --mode all before integration tests"
    manifest = pd.read_csv(manifest_path)
    keys = ["destination_iso3", "hs2", "outcome_year"]
    a = set(map(tuple, manifest.loc[manifest.task.eq(TASK_A), keys].to_numpy()))
    b = set(map(tuple, manifest.loc[manifest.task.eq(TASK_B), keys].to_numpy()))
    assert not (a & b)
    unknown = manifest.loc[~manifest.label_observed.astype(bool)]
    assert len(unknown) > 0
    assert unknown.export_value_usd.isna().all()
    assert not manifest.duplicated(["task", *keys]).any()
    assert (manifest.historical_information_cutoff == manifest.outcome_year - 2).all()
    assert (manifest.history_end_year <= manifest.historical_information_cutoff).all()


def test_risk_predictions_respect_information_cutoff():
    root = Path(__file__).resolve().parents[1]
    path = root / "predictions/F3_risk_scores.parquet"
    assert path.exists(), "Run --mode all before integration tests"
    scores = pd.read_parquet(path)
    learned = scores.loc[scores.gate_name.isin(["RIDGE_STATIC_EXPERT", "RIDGE_BLEND"])]
    assert (learned.risk_training_year_max <= learned.outcome_year - 2).all()
    # Risk models use the same historical endpoint recorded at fit time.
    assert (learned.risk_training_year_max <= learned.risk_training_max_allowed).all()


def test_reconstructed_blend_matches_archived_blend_and_summary():
    root = Path(__file__).resolve().parents[1]
    scores = pd.read_parquet(root / "predictions/F1_F2_reconstructed_outer_scores.parquet")
    assert np.max(np.abs(scores.blend_rank_score - scores.saved_blend_score)) < 1e-12
    table = pd.read_csv(root / "results/F1_blend_model_comparison.csv")
    row = table.loc[table.task.eq(TASK_A) & table.policy.eq("EQUAL_WEIGHT_RANK_BLEND")].iloc[0]
    assert row.candidate_pools == 288
    assert row.valid_ndcg10_pools == 138
    assert np.isclose(row.macro_ndcg10, 0.331878, atol=1e-6)


def test_random_deferral_and_operational_coverage_are_present():
    root = Path(__file__).resolve().parents[1]
    operational = pd.read_csv(root / "results/F4_operational_coverage.csv")
    assert operational.policy.eq("RANDOM_DEFERRAL").any()
    fixed = operational.loc[operational.policy.eq("HISTORICAL_HS2_ERROR")]
    assert len(fixed) > 0
    assert fixed.actual_coverage.notna().all()
    assert (fixed.actual_coverage > 0).all()
    assert (fixed.actual_coverage <= 1).all()


def test_risk_clipping_and_matched_gate_diagnostic_are_reported():
    root = Path(__file__).resolve().parents[1]
    clipping = pd.read_csv(root / "results/F3_risk_clipping_ties.csv")
    assert (clipping.negative_raw_predictions_clipped >= 0).all()
    assert (clipping.largest_tie_group >= 1).all()
    comparisons = pd.read_csv(root / "results/F4_simple_gate_inference.csv")
    assert comparisons.interpretation.str.contains("exploratory HS2-macro paired diagnostic").all()
    assert comparisons.p_holm.notna().all()


def test_binary_missing_label_bounds_are_ordered_and_usd_is_unbounded():
    root = Path(__file__).resolve().parents[1]
    bounds = pd.read_csv(root / "results/F7_missing_outcome_sensitivity.csv")
    binary = bounds.loc[bounds.task.eq(TASK_A)]
    assert (binary.common_completion_regret_lower_bound <= binary.common_completion_regret_upper_bound).all()
    assert binary.unknown_values_remain_missing.astype(bool).all()
    assert binary.usd_upper_bound.isna().all()

