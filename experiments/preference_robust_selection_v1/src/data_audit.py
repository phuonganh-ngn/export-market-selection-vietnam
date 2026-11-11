from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments" / "decision_focused_xai_v1" / "src"))
from candidate_protocol import KEY, YEARS, load_panel_and_candidates  # noqa: E402


def run_audit() -> dict:
    panel, broad, task_manifest, counts = load_panel_and_candidates(write=False)
    key = ["destination_iso3", "hs2", "outcome_year"]
    broad = broad.copy()
    a = broad.entry_candidate_ex_ante.astype(bool)
    b = broad.task_b_eligible.astype(bool)
    broad["task_a_eligible"] = a
    broad["task_b_eligible"] = b
    broad["eligible_in_either_task"] = a | b
    broad["eligible_in_both_tasks"] = a & b
    broad["eligible_in_neither_task"] = ~(a | b)
    broad["historical_information_cutoff"] = broad.feature_year.astype(int)
    broad["label_observed"] = broad.label_observed.astype(bool)

    ka = pd.MultiIndex.from_frame(broad.loc[a, key].drop_duplicates())
    kb = pd.MultiIndex.from_frame(broad.loc[b, key].drop_duplicates())
    ku = pd.MultiIndex.from_frame(broad[key].drop_duplicates())
    overlap = len(ka.intersection(kb))
    neither = len(ku.difference(ka.union(kb)))
    if len(broad) != 62_797 or int((~broad.label_observed).sum()) != 559:
        raise AssertionError("Broad audited evaluation universe no longer reconciles to Audit V2")
    if int(a.sum()) != 41_131 or int(b.sum()) != 12_962 or overlap != 6_511 or neither != 15_215:
        raise AssertionError(f"Task sample flow changed: A={a.sum()}, B={b.sum()}, overlap={overlap}, neither={neither}")

    sample_rows = []
    for y, g in broad.groupby("outcome_year", sort=True):
        ga, gb = g.task_a_eligible, g.task_b_eligible
        sample_rows.append({
            "outcome_year": int(y), "broad_candidates": len(g),
            "observed_labels": int(g.label_observed.sum()), "unknown_labels": int((~g.label_observed).sum()),
            "task_a_rows": int(ga.sum()), "task_a_unknown": int((~g.loc[ga, "label_observed"]).sum()),
            "task_b_rows": int(gb.sum()), "task_b_unknown": int((~g.loc[gb, "label_observed"]).sum()),
            "in_both_tasks": int((ga & gb).sum()), "in_neither_task": int((~(ga | gb)).sum()),
        })
    sample_flow = pd.DataFrame(sample_rows)
    broad.to_csv(HERE / "results/candidate_manifest.csv", index=False, na_rep="")
    task_manifest.to_csv(HERE / "results/candidate_task_manifest.csv", index=False, na_rep="")
    sample_flow.to_csv(HERE / "results/sample_flow.csv", index=False, na_rep="")
    counts.to_csv(HERE / "results/audited_population_by_year.csv", index=False, na_rep="")

    cutoff = broad.outcome_year.astype(int) - 2
    if (broad.feature_year.astype(int) > cutoff).any():
        raise AssertionError("Outer features exceed the audited y-2 information cutoff")
    if broad.duplicated(key).any():
        raise AssertionError("Broad candidate manifest duplicates destination-product-year keys")
    if broad.export_value_usd[broad.label_observed].isna().any():
        raise AssertionError("An observed label has missing export value")
    if broad.export_value_usd[~broad.label_observed].notna().any():
        # The source panel's outcome value is explicitly unavailable for these rows.
        raise AssertionError("Unknown labels unexpectedly carry export outcomes")

    ranking_path = HERE.parent / "decision_focused_xai_v1" / "predictions" / "E6_ranking_predictions.parquet"
    ranked = pd.read_parquet(ranking_path)
    if ranked.groupby(["task", "candidate_id"]).model.nunique().min() < 1:
        raise AssertionError("Saved candidate prediction file has no model scores")
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        pred = ranked.loc[ranked.task.eq(task)]
        expected = int(task_manifest.task.eq(task).sum())
        if pred.loc[pred.model.eq("LambdaMART")].candidate_id.nunique() != expected:
            raise AssertionError(f"Saved LambdaMART population mismatch for {task}")
    audit = [
        "# Input and candidate audit", "",
        "## Sources and scope", "",
        "The extension reuses the audited HS12 BACI V202601-derived panel and saved rolling-origin model predictions. It does not redownload trade data or alter prior artifacts. BACI HS12 V202601 ends in 2024, so no unexamined 2025 HS12 outcome is available in that release.", "",
        f"The evaluation panel has {len(broad):,} unique ex-ante destination–HS2–year keys across {broad.destination_iso3.nunique()} destinations and {broad.hs2.nunique()} HS2 chapters, for 2022–2024. It contains {int(broad.label_observed.sum()):,} observed outcome labels and {int((~broad.label_observed).sum()):,} unknown labels.", "",
        "## Population reconciliation", "",
        f"Task A contains {int(a.sum()):,} keys; Task B contains {int(b.sum()):,}; {overlap:,} keys satisfy both audited rules; {neither:,} keys satisfy neither rule. Therefore the task-expanded manifest has {len(task_manifest):,} rows but only {len(ka.union(kb)):,} distinct keys. The overlap arises because entry inactivity means each of the previous three verified annual flows is below USD 50,000, while expansion eligibility requires positive lagged export and below-median historical share. A sub-threshold positive relationship can satisfy both definitions. The rules are preserved rather than silently forced into mutually exclusive classes.", "",
        "All 559 unknown outcome labels are retained as missing. Task A contains all 559 unknowns (187, 150, and 222 by evaluation year); Task B has no unknown labels in its eligible sample. The full universe includes 15,215 keys in neither task, mainly existing relationships that do not meet the audited underexploited rule or the entry inactivity rule.", "",
        "## Temporal and source-vintage audit", "",
        "Each outer row uses `feature_year = outcome_year - 2`. Reused prediction files are those from the previously verified two-year-lag rolling-origin runs. WDI/WGI historical series are revised vintages, so the evaluation is retrospective rather than point-in-time. Since 2022–2024 informed the prior research and this extension, this analysis is exploratory; no independent holdout exists.", "",
        "## Eligibility and missing outcomes", "",
        "Candidate eligibility is reconstructed from the audited panel and historical rules before reading the test outcomes. Outcome observability is stored separately. Unknown outcomes are not converted to zero. The full candidate manifest includes eligibility flags, historical cutoff, label-observed status, and task overlap.", "",
        "## Decision interpretation", "",
        "The 10-market and 5-market constraints are cardinality budgets, not monetary promotion budgets. No country-specific cost data are available. Predictive scenarios are finite model-disagreement stress cases, not a calibrated probability law.",
    ]
    (HERE / "reports/INPUT_AUDIT.md").write_text("\n".join(audit) + "\n")
    return {"panel_rows": len(panel), "broad_candidates": len(broad), "task_a": int(a.sum()), "task_b": int(b.sum()),
            "overlap": overlap, "neither": neither, "unknown": int((~broad.label_observed).sum())}
