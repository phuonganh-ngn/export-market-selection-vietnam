"""Independently reconstruct selected result tables from persisted predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def _ndcg(relevance: np.ndarray, values_are_gains: bool = False) -> float:
    n = min(10, len(relevance))
    if not n:
        return np.nan
    discount = 1 / np.log2(np.arange(2, n + 2))
    gains = relevance[:n] if values_are_gains else 2 ** relevance[:n] - 1
    ideal_values = np.sort(relevance)[::-1][:n]
    ideal = ideal_values if values_are_gains else 2 ** ideal_values - 1
    denominator = float(np.sum(ideal * discount))
    return float(np.sum(gains * discount) / denominator) if denominator > 0 else np.nan


def reconstruct() -> pd.DataFrame:
    rows = []
    vnm = pd.read_parquet(HERE / "predictions/A_candidate_predictions.parquet")
    summary = pd.read_csv(RESULTS / "A_two_part_model_comparison.csv").set_index("model")
    scores = {
        "Gravity_PPML": "ppml_score",
        "LightGBM_probability": "lightgbm_score",
        "Original_equal_rank_blend": "blend_rank_score",
        "TwoPart_calibrated_probability_x_conditional_PPML": "two_part_qualified_usd_score",
    }
    for model, score in scores.items():
        ndcg, precision, entries, usd = [], [], [], []
        for _, group in vnm.groupby(["hs2", "outcome_year"], sort=True):
            known = group.loc[group.label_observed.astype(bool)].sort_values(
                [score, "destination_iso3"], ascending=[False, True], kind="mergesort"
            )
            rel = known.entry_success.astype(float).to_numpy()
            value = _ndcg(rel)
            if np.isfinite(value):
                ndcg.append(value)
            top_known = known.head(10)
            n_success = float(top_known.entry_success.sum())
            entries.append(n_success)
            precision.append(n_success / 10)
            top_full = group.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").head(10)
            usd.append(float(top_full.loc[top_full.label_observed.astype(bool), "qualified_export_value_usd"].sum()))
        for metric, computed in [
            ("macro_ndcg10", np.mean(ndcg)),
            ("macro_precision10", np.mean(precision)),
            ("mean_successes_top10", np.mean(entries)),
            ("mean_qualified_usd", np.mean(usd)),
        ]:
            stored = float(summary.loc[model, metric])
            rows.append({"scope": "VNM", "model": model, "metric": metric,
                         "independent_reconstruction": computed, "saved_table": stored,
                         "absolute_difference": abs(computed - stored),
                         "match": bool(np.isclose(computed, stored, atol=1e-7, rtol=1e-9))})

    external_summary = pd.read_csv(RESULTS / "B_external_pool_metrics.csv")
    for country in ["THA", "IDN"]:
        evaluated = pd.read_parquet(HERE / f"predictions/B_{country}_evaluated.parquet")
        for (task, mode, year, hs2), group in evaluated.groupby(
            ["task", "validation_mode", "outcome_year", "hs2"], sort=True
        ):
            models = ({
                "LightGBM_probability": "lightgbm_entry_score",
                "TwoPart": "two_part_qualified_value_score",
                "Gravity_PPML": "ppml_export_value_score",
                "RankBlend": "rank_blend_score",
            } if task == "Task_A_new_entry" else {
                "LightGBM_incremental": "lightgbm_incremental_score",
                "Gravity_PPML_incremental": "ppml_incremental_score",
                "RankBlend": "rank_blend_score",
            })
            known = group.loc[group.label_observed].copy()
            for model, score in models.items():
                ranked = known.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort")
                if task == "Task_A_new_entry":
                    relevance = ranked.entry_success.astype(float).to_numpy()
                    usd_col = "qualified_usd_top10_observed_lower_bound"
                    selected_usd = group.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").head(10)
                    observed_usd = float(selected_usd.loc[selected_usd.label_observed, "qualified_export_value_usd"].sum())
                else:
                    increment = ranked.realized_incremental_export_usd.fillna(0).clip(lower=0).to_numpy(float)
                    grade = np.minimum(4, np.floor(np.log10(1 + increment / 50_000))).astype(int)
                    relevance = 2 ** grade - 1
                    usd_col = "positive_incremental_usd_top10_observed_lower_bound"
                    selected_usd = group.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").head(10)
                    observed_usd = float(selected_usd.loc[selected_usd.label_observed, "realized_incremental_export_usd"].clip(lower=0).sum())
                reconstructed_ndcg = _ndcg(relevance, values_are_gains=(task == "Task_B_existing_underexploited"))
                match = external_summary.loc[
                    external_summary.country.eq(country)
                    & external_summary.task.eq(task)
                    & external_summary["mode"].eq(mode)
                    & external_summary.year.eq(year)
                    & external_summary.hs2.astype(str).str.zfill(2).eq(str(hs2).zfill(2))
                    & external_summary.model.eq(model)
                ]
                if len(match) != 1:
                    rows.append({"scope": f"{country}-{mode}-{task}-{year}-{hs2}", "model": model,
                                 "metric": "pool_key", "match": False})
                    continue
                stored = match.iloc[0]
                for metric, computed, stored_value in [
                    ("ndcg10_complete_case", reconstructed_ndcg, stored.ndcg10_complete_case),
                    ("observed_selected_usd", observed_usd, stored[usd_col]),
                ]:
                    same = (pd.isna(computed) and pd.isna(stored_value)) or np.isclose(
                        computed, stored_value, atol=1e-7, rtol=1e-9
                    )
                    rows.append({"scope": f"{country}-{mode}-{task}-{year}-{hs2}", "model": model,
                                 "metric": metric, "independent_reconstruction": computed,
                                 "saved_table": stored_value,
                                 "absolute_difference": abs(computed - stored_value)
                                 if np.isfinite(computed) and np.isfinite(stored_value) else np.nan,
                                 "match": bool(same)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = reconstruct()
    result.to_csv(RESULTS / "independent_metric_reconstruction.csv", index=False)
    print(f"Reconstructed {len(result):,} metric cells; all match: {result.match.all()}")
    if not result.match.all():
        raise SystemExit(1)
