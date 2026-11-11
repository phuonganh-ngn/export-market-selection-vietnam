from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
# The configuration file can be switched with M1_CONFIG (absolute path, or relative to the project
# root), e.g. M1_CONFIG=config/legacy_archived.yaml to reproduce the archived models exactly.
CONFIG_PATH = Path(os.environ.get("M1_CONFIG", "config/config.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH
CFG = yaml.safe_load(CONFIG_PATH.read_text())


def path(key: str) -> Path:
    return ROOT / CFG["paths"][key]


def mkdirs() -> None:
    for p in [path("raw_baci_dir"), path("raw_wdi_dir"), path("raw_wgi_dir"),
              path("raw_cepii_dir"), path("interim_dir"), path("processed_dir"),
              path("results_dir"), path("reports_dir")]:
        p.mkdir(parents=True, exist_ok=True)
    for p in [path("results_dir") / "models", path("results_dir") / "predictions",
              path("results_dir") / "tables"]:
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(file_path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(file_path, index=False, na_rep="")


def read_table(name: str) -> pd.DataFrame:
    file_path = path("processed_dir") / name
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def hs2_code(value) -> str:
    """Normalize an HS2 chapter identifier, including numeric CSV inference."""
    return f"{int(float(value)):02d}"


@lru_cache(maxsize=1)
def hs2_chapter_descriptions() -> dict[str, str]:
    """Load verified HS 2012 chapter headings bundled with the processed inputs."""
    file_path = path("processed_dir") / "hs12_chapter_descriptions.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing official HS12 chapter headings: {file_path}")
    frame = pd.read_csv(file_path, dtype={"hs2": str})
    frame["hs2"] = frame.hs2.map(hs2_code)
    if frame.hs2.duplicated().any() or len(frame) != 96:
        raise ValueError("HS12 chapter heading file must contain 96 unique chapters (chapter 77 is reserved).")
    return dict(zip(frame.hs2, frame.hs2_description))


def hs2_chapter_description(value) -> str:
    code = hs2_code(value)
    return hs2_chapter_descriptions()[code]


def write_json(file_path: Path, obj: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(obj, indent=2, default=str))


def wdi_indicator_codes() -> dict[str, str]:
    return {
        "gdp_usd": "NY.GDP.MKTP.CD",
        "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "population": "SP.POP.TOTL",
        "urban_population_share": "SP.URB.TOTL.IN.ZS",
        "internet_usage": "IT.NET.USER.ZS",
        "inflation": "FP.CPI.TOTL.ZG",
        "trade_openness": "NE.TRD.GNFS.ZS",
    }
