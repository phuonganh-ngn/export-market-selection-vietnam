from pathlib import Path
import hashlib,sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"src"))
from pipeline import _rank_metrics

RESULTS=HERE/"results"

def test_binary_target_and_missing_not_zero():
    x=pd.DataFrame({"score":[.9,.1,.2],"destination_iso3":["A","B","C"],"y":[1.,0.,np.nan]})
    m=_rank_metrics(x,"score","y",2)
    assert m["entries"]==1 and m["n_known"]==2 and m["n_positive"]==1

def test_strict_keys_reconcile_vietnam():
    m=pd.read_csv(ROOT/"experiments/final_hybrid_economic_validation_v1/results/final_candidate_manifest.csv",dtype={"hs2":str})
    a=m.loc[m.task.eq("Task_A_new_entry")]
    assert len(a)==29737
    assert not a.duplicated(["destination_iso3","hs2","outcome_year"]).any()
    assert int((~a.label_observed.astype(bool)).sum())==556

def test_external_predictions_are_label_free_and_frozen():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    forbidden={"export_value_usd","entry_success","label_observed","qualified_export_value_usd","_world_imports"}
    assert forbidden.isdisjoint(d.columns)
    meta=pd.read_json(HERE/"data_manifest/external_prediction_manifest.json",typ="series")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta.sha256

def test_external_y2_cutoff_and_unique_pool_candidate():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    assert (d.feature_year==d.outcome_year-2).all()
    assert (d.training_cutoff<=d.outcome_year-2).all()
    assert not d.duplicated(["country","validation_mode","task","destination_iso3","hs2","outcome_year"]).any()

def test_two_part_product_is_qualified_not_total_value():
    p=HERE/"predictions/A_candidate_predictions.parquet"
    if not p.exists():pytest.skip("Vietnam model stage not run")
    d=pd.read_parquet(p)
    assert np.allclose(d.two_part_qualified_usd_score,d.prob_calibrated*d.conditional_ppml_usd,equal_nan=True)
    assert (d.prob_calibrated.dropna().between(0,1)).all()
    assert (d.conditional_ppml_usd.dropna()>=50000).all()
    assert d.loc[~d.label_observed.astype(bool),"qualified_export_value_usd"].isna().all()

def test_source_hs2_unique_and_valid_units():
    p=HERE/"data_manifest/baci_hs2_external_source.parquet"
    if not p.exists():pytest.skip("BACI source extraction not run")
    d=pd.read_parquet(p)
    assert not d.duplicated(["destination_iso3","hs2","year"]).any()
    assert d.hs2.astype(str).str.fullmatch(r"\d{2}").all()
    assert (d.world_imports_usd>0).all()

def test_vietnam_feature_information_cutoff():
    p=HERE/"predictions/A_candidate_predictions.parquet"
    if not p.exists():pytest.skip("Vietnam model stage not run")
    d=pd.read_parquet(p)
    assert (d.historical_information_cutoff==d.outcome_year-2).all()
    assert (d.training_outcome_max<=d.outcome_year-2).all()

def test_vietnam_tasks_are_disjoint():
    p=ROOT/"experiments/final_hybrid_economic_validation_v1/results/final_candidate_manifest.csv"
    d=pd.read_csv(p,dtype={"hs2":str})
    a=d.loc[d.task.eq("Task_A_new_entry"),["destination_iso3","hs2","outcome_year"]]
    b=d.loc[d.task.eq("Task_B_existing_underexploited"),["destination_iso3","hs2","outcome_year"]]
    assert len(a.merge(b,on=["destination_iso3","hs2","outcome_year"]))==0

def test_threshold_labels_match_observed_exports():
    p=HERE/"predictions/A_candidate_predictions.parquet"
    if not p.exists():pytest.skip("Vietnam model stage not run")
    d=pd.read_parquet(p);q=d.loc[d.label_observed.astype(bool)]
    assert np.array_equal(q.entry_success.astype(int).to_numpy(),q.export_value_usd.ge(50000).astype(int).to_numpy())

def test_unknown_vietnam_targets_are_nan_not_zero():
    p=HERE/"predictions/A_candidate_predictions.parquet"
    if not p.exists():pytest.skip("Vietnam model stage not run")
    d=pd.read_parquet(p);q=d.loc[~d.label_observed.astype(bool)]
    assert q.export_value_usd.isna().all()
    assert q.entry_success.isna().all()
    assert q.qualified_export_value_usd.isna().all()

def test_conditional_mean_fit_uses_only_qualifying_labels():
    p=RESULTS/"A_conditional_value_fit_diagnostics.csv"
    if not p.exists():pytest.skip("fit diagnostics absent")
    d=pd.read_csv(p)
    assert (d.conditional_train_n==d.conditional_success_n).all()
    assert (d.train_outcome_max<=d.training_cutoff).all()
    assert d.conditional_ppml_converged.astype(bool).all()

def test_external_task_a_and_b_exclusive():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    for keys,g in d.groupby(["country","validation_mode","outcome_year"]):
        a=g.loc[g.task.eq("Task_A_new_entry"),["destination_iso3","hs2"]]
        b=g.loc[g.task.eq("Task_B_existing_underexploited"),["destination_iso3","hs2"]]
        assert len(a.merge(b,on=["destination_iso3","hs2"]))==0

def test_external_scores_are_finite_and_probability_bounded():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p);a=d.loc[d.task.eq("Task_A_new_entry")]
    assert np.isfinite(a.entry_probability_calibrated).all()
    assert a.entry_probability_calibrated.between(0,1).all()
    assert np.isfinite(a.two_part_qualified_value_score).all()
    assert np.isfinite(a.ppml_export_value_score).all()

def test_external_scores_have_complete_candidate_keys():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    assert not d.duplicated(["country","validation_mode","task","destination_iso3","hs2","outcome_year"]).any()
    counts=d.groupby(["country","validation_mode","task","outcome_year"]).size()
    assert counts.gt(0).all()

def test_external_prediction_cutoff_feature_max_and_training_max():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    assert (d.historical_information_cutoff==d.outcome_year-2).all()
    assert (d.training_cutoff<=d.outcome_year-2).all()

def test_prediction_manifest_sha256_is_reproducible():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    meta=pd.read_json(HERE/"data_manifest/external_prediction_manifest.json",typ="series")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta.sha256

def test_complete_case_ndcg_denominator_reported():
    p=RESULTS/"A_two_part_model_comparison.csv"
    if not p.exists():pytest.skip("Vietnam comparison absent")
    d=pd.read_csv(p)
    assert d.pools_total.eq(288).all()
    assert d.valid_ndcg_pools.eq(138).all()

def test_holm_adjustment_not_smaller_than_raw_p():
    p=RESULTS/"A_two_part_paired_tests.csv"
    if not p.exists():pytest.skip("paired tests absent")
    d=pd.read_csv(p)
    assert (d.holm_p_metric_family+1e-12>=d.paired_signflip_p_raw).all()

def test_pairwise_external_comparisons_have_common_hs2_year_pools():
    p=RESULTS/"B_country_paired_tests.csv"
    if not p.exists():pytest.skip("external paired results absent")
    d=pd.read_csv(p)
    assert d.paired_pools.gt(0).all()
    assert d.hs2_blocks.le(96).all()
    assert d.year_count.eq(3).all()

def test_external_labels_not_present_in_frozen_score_file():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    assert {"export_value_usd","entry_success","label_observed"}.isdisjoint(d.columns)

def test_external_unknown_outcomes_remain_missing_after_evaluation_merge():
    for c in ["THA","IDN"]:
        p=HERE/f"predictions/B_{c}_evaluated.parquet"
        if not p.exists():pytest.skip("external evaluation absent")
        d=pd.read_parquet(p);q=d.loc[~d.label_observed]
        assert q.export_value_usd.isna().all()
        assert q.entry_success.isna().all()

def test_external_label_selection_did_not_change_fixed_countries():
    p=HERE/"config/FROZEN_EXTERNAL_PROTOCOL.yaml"
    text=p.read_text()
    assert "primary_external: THA" in text and "secondary_external: IDN" in text
    assert "external_outcome_years: [2022, 2023, 2024]" in text

def test_development_cap_does_not_exceed_raw_observed_selected_values():
    p=RESULTS/"A_value_tail_robustness.csv"
    if not p.exists():pytest.skip("tail table absent")
    d=pd.read_csv(p)
    assert (d.capped_qualified_usd_top10_observed<=d.uncapped_qualified_usd_top10_observed+1e-6).all()

def test_prediction_source_provenance_mentions_y_minus_2():
    p=HERE/"predictions/B_country_prediction_manifest.parquet"
    if not p.exists():pytest.skip("external prediction stage not run")
    d=pd.read_parquet(p)
    assert d.score_provenance.str.contains("y-2",case=False).all()

def test_external_sample_flow_counts_unique_candidates_not_modes():
    flow=RESULTS/"B_external_candidate_flow.csv"
    quality=RESULTS/"B_country_data_quality.csv"
    if not flow.exists() or not quality.exists():pytest.skip("external finalization absent")
    f=pd.read_csv(flow);q=pd.read_csv(quality)
    a=f.loc[f.task.eq("Task_A_new_entry")].merge(q,on=["country","year"],suffixes=("_flow","_quality"),validate="one_to_one")
    assert (a.candidate_rows==a.strict_zero_candidates).all()
    assert (a.unknown_labels_flow==a.unknown_labels_quality).all()
    assert f.groupby(["country","year","task"]).size().eq(1).all()

def test_qualified_usd_regret_matches_observed_label_oracle():
    p=RESULTS/"A_topk_pool_economics.csv"
    if not p.exists():pytest.skip("economic regret table absent")
    d=pd.read_csv(p);d=d.loc[d.k.eq(10)]
    assert (d.qualified_usd_regret_observed_label_oracle>=0).all()
    expected=(d.oracle_qualified_usd_observed_labels-d.qualified_usd_observed_lower_bound).clip(lower=0)
    assert np.allclose(expected,d.qualified_usd_regret_observed_label_oracle)
    assert d.unknown_selected.eq(0).all()
