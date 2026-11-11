from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(HERE/"src"))
from pipeline import audit_inputs,run_vietnam_two_part,extract_baci_targets,run_external_predictions,evaluate_external
from finalize_results import finalize_all

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["audit","vietnam","extract","external-predict","external-evaluate","all"],default="all");ap.add_argument("--force",action="store_true")
    a=ap.parse_args()
    if a.stage in ["audit","vietnam","all"]:
        manifest,scores,oof=audit_inputs()
    if a.stage in ["vietnam","all"]:
        path=HERE/"predictions/A_candidate_predictions.parquet"
        if a.force or not path.exists():run_vietnam_two_part(manifest,scores,oof)
    source=None
    if a.stage in ["extract","external-predict","external-evaluate","all"]:
        source=extract_baci_targets(force=a.force)
    if a.stage in ["external-predict","all"]:
        predpath=HERE/"predictions/B_country_prediction_manifest.parquet"
        if a.force or not predpath.exists():
            proto=HERE/"config/FROZEN_EXTERNAL_PROTOCOL.yaml"
            h=hashlib.sha256(proto.read_bytes()).hexdigest()
            run_external_predictions(source,h)
    if a.stage in ["external-evaluate","all"]:
        predpath=HERE/"predictions/B_country_prediction_manifest.parquet"
        if not predpath.exists():raise FileNotFoundError("Run --stage external-predict before external evaluation")
        evaluate_external(pd.read_parquet(predpath),source)
    if a.stage=="all":
        final=HERE/"predictions/A_candidate_predictions.parquet"
        if not final.exists():raise FileNotFoundError("Vietnam model predictions absent")
        wb=finalize_all();print(f"Completed reduced specification; workbook: {wb}")

if __name__=="__main__":main()
