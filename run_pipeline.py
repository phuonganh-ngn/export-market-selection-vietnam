from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
os.chdir(ROOT)

from common import CFG, mkdirs, path, write_json


def _state_file(): return path("reports_dir")/"pipeline_state.json"


def _load_state():
    p=_state_file()
    return json.loads(p.read_text()) if p.exists() else {"stages":{},"environment":{}}


def _log(message:str):
    p=path("reports_dir")/"pipeline.log"
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as stream:
        stream.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}\n")
    print(message,flush=True)


def _stage(name:str,func,resume:bool,state:dict,expected:Path|None=None):
    if resume and expected is not None and expected.exists() and state.get("stages",{}).get(name,{}).get("status")=="completed":
        _log(f"SKIP {name}: completed artifact exists: {expected.relative_to(ROOT)}")
        return None
    proc=psutil.Process(); rss_before=proc.memory_info().rss/1024**2; t0=time.time()
    _log(f"START {name}; rss_mb={rss_before:.1f}")
    try:
        result=func()
        elapsed=time.time()-t0; rss_after=proc.memory_info().rss/1024**2
        state.setdefault("stages",{})[name]={"status":"completed","started_at":dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "elapsed_seconds":elapsed,"rss_mb_before":rss_before,"rss_mb_after":rss_after}
        write_json(_state_file(),state)
        _log(f"DONE {name}; elapsed_s={elapsed:.1f}; rss_mb={rss_after:.1f}")
        return result
    except Exception as exc:
        elapsed=time.time()-t0
        state.setdefault("stages",{})[name]={"status":"failed","elapsed_seconds":elapsed,"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}
        write_json(_state_file(),state)
        _log(f"FAILED {name}; {type(exc).__name__}: {exc}")
        return exc


def main():
    parser=argparse.ArgumentParser(description="Vietnam export forecasting and market selection research pipeline")
    parser.add_argument("--mode",required=True,choices=["acquire","validate","prepare","experiment1","experiment2","experiment3","experiment4","experiment5","robustness","report","full"])
    parser.add_argument("--quick",action="store_true",help="Engineering smoke test only; never scientific final evidence")
    parser.add_argument("--resume",action="store_true",help="Skip stages recorded as complete in reports/pipeline_state.json")
    args=parser.parse_args()
    mkdirs(); state=_load_state()
    state["environment"]={"python":sys.version,"platform":platform.platform(),"cpu_count":os.cpu_count(),
        "memory_total_gb":round(psutil.virtual_memory().total/1024**3,2),"quick_mode":args.quick,"started_at":dt.datetime.now().isoformat(timespec="seconds")}
    write_json(_state_file(),state)

    from acquire import run as acquire_run, registry
    from validate import run as validate_run, inspect_baci_archive
    from country_mapping import run as country_mapping_run
    from hs_aggregation import run as aggregation_run
    from panel import run as panel_run
    from temporal_validation import run as leakage_run
    from experiment1 import run as e1_run
    from market_discovery import run as e2_run
    from market_ranking import run as e3_run
    from explainability import run as e4_run
    from clustering import run as e5_run
    from robustness import run as robustness_run
    from inference import run as inference_run
    from reporting import reporting_run, blocked_report

    def acquire_stage():
        acquire_run(); registry()
        return True
    def validate_headers():
        audit,summary=inspect_baci_archive(full_scan=False)
        if summary.get("year_min")!=CFG["study"]["first_year"] or summary.get("year_max")!=CFG["study"]["last_year"]:
            raise RuntimeError(f"Mandatory BACI year coverage differs from config: {summary}")
        return summary
    def prepare():
        country_mapping_run(); aggregation_run(); panel_run(); leakage_run()
        return True
    quick=args.quick
    if args.mode=="acquire":
        _stage("acquire",acquire_stage,args.resume,state,path("reports_dir")/"data_source_registry.csv")
        return
    if args.mode=="validate":
        _stage("validate_full",validate_run,args.resume,state,path("reports_dir")/"aggregation_checks.csv")
        registry()
        return
    if args.mode=="prepare":
        _stage("validate_headers",validate_headers,args.resume,state,path("reports_dir")/"year_coverage.csv")
        _stage("prepare",prepare,args.resume,state,path("processed_dir")/"panel_lag2.parquet")
        return
    if args.mode=="experiment1":
        _stage("experiment1",lambda:e1_run(quick=quick),args.resume,state,path("results_dir")/"E1_forecasting_overall.csv"); return
    if args.mode=="experiment2":
        _stage("experiment2",lambda:e2_run(quick=quick),args.resume,state,path("results_dir")/"E2_market_discovery.csv"); return
    if args.mode=="experiment3":
        _stage("experiment3",lambda:e3_run(quick=quick),args.resume,state,path("results_dir")/"E3_market_ranking.csv"); return
    if args.mode=="experiment4":
        _stage("experiment4",lambda:e4_run(quick=quick),args.resume,state,path("results_dir")/"E4_global_shap.csv"); return
    if args.mode=="experiment5":
        _stage("experiment5",lambda:e5_run(quick=quick),args.resume,state,path("results_dir")/"E5_clusters.csv"); return
    if args.mode=="robustness":
        _stage("robustness",lambda:robustness_run(quick=quick),args.resume,state,path("results_dir")/"robustness_summary.csv"); return
    if args.mode=="report":
        inference_run(); reporting_run(quick=quick); return

    # Full run. Data preparation validates every streamed BACI record and HS aggregation totals.
    acquisition=_stage("acquire",acquire_stage,args.resume,state,path("reports_dir")/"data_source_registry.csv")
    if isinstance(acquisition,Exception):
        from reporting import blocked_report
        blocked_report(acquisition,quick=quick)
        return
    validation=_stage("validate_headers",validate_headers,args.resume,state,path("reports_dir")/"year_coverage.csv")
    if isinstance(validation,Exception):
        from reporting import blocked_report
        blocked_report(validation,quick=quick)
        return
    prep=_stage("prepare",prepare,args.resume,state,path("processed_dir")/"panel_lag2.parquet")
    if isinstance(prep,Exception):
        state.setdefault("stages",{})["mandatory_data_validation"]={"status":"failed","error":str(prep)}
        write_json(_state_file(),state)
        from reporting import blocked_report
        blocked_report(prep,quick=quick)
        return
    stage_calls=[("experiment1",lambda:e1_run(quick=quick),path("results_dir")/"E1_forecasting_overall.csv"),
        ("experiment2",lambda:e2_run(quick=quick),path("results_dir")/"E2_market_discovery.csv"),
        ("experiment3",lambda:e3_run(quick=quick),path("results_dir")/"E3_market_ranking.csv"),
        ("experiment4",lambda:e4_run(quick=quick),path("results_dir")/"E4_global_shap.csv"),
        ("experiment5",lambda:e5_run(quick=quick),path("results_dir")/"E5_clusters.csv"),
        ("robustness",lambda:robustness_run(quick=quick),path("results_dir")/"robustness_summary.csv")]
    failures=[]
    for name,call,expected in stage_calls:
        result=_stage(name,call,args.resume,state,expected)
        if isinstance(result,Exception): failures.append({"stage":name,"error":str(result)})
    try:
        inference_result=_stage("inference",inference_run,args.resume,state,path("results_dir")/"hypothesis_tests.csv")
        if isinstance(inference_result,Exception): failures.append({"stage":"inference","error":str(inference_result)})
    except Exception as exc:
        failures.append({"stage":"inference","error":str(exc)})
    try:
        reporting_run(quick=quick,failures=failures)
    except Exception as exc:
        _log(f"FAILED reporting: {type(exc).__name__}: {exc}")
        state.setdefault("stages",{})["reporting"]={"status":"failed","error":str(exc),"traceback":traceback.format_exc()}
        write_json(_state_file(),state)
        raise


if __name__=="__main__": main()
