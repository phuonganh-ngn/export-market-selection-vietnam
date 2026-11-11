# Vietnam export forecasting and market selection

This repository implements a reproducible, CPU-oriented empirical pipeline using CEPII BACI HS12, World Bank WDI/WGI and CEPII GeoDist. The primary unit is destination × HS2 × outcome year. The study uses chronological outer evaluation in 2022–2024 and a two-year information lag, with one-year lag sensitivity. Trade values are converted from BACI's thousand-current-USD units to USD.

## Run

```bash
python run_pipeline.py --mode acquire
python run_pipeline.py --mode validate
python run_pipeline.py --mode prepare
python run_pipeline.py --mode experiment1
python run_pipeline.py --mode experiment2
python run_pipeline.py --mode experiment3
python run_pipeline.py --mode experiment4
python run_pipeline.py --mode experiment5
python run_pipeline.py --mode robustness
python run_pipeline.py --mode report
python run_pipeline.py --mode full
```

Install `requirements.txt` first. Configuration is in `config/config.yaml`. A quick engineering mode is available with `--quick`; smoke-test outputs must not be presented as empirical evidence.

## Integrity rules

The processed outcome is set to zero only for an importer–HS2–year with positive BACI world imports for that product chapter; otherwise the bilateral outcome is left missing. Market candidates are formed from positive historical world import demand. This avoids treating an absent flow in an unobserved destination–product cell as a verified zero. All feature joins use the declared lag. The pipeline does not generate figures.

The downloaded BACI archive is large and remains in `data/raw/baci`. Raw files are not committed. `reports/data_source_registry.csv` records file hashes and source metadata.
