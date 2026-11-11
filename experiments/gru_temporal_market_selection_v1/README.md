# GRU temporal market-selection experiment

This extension evaluates a small GRU on the audited Vietnam Task A population. It reuses the existing BACI-derived panel and saved baseline predictions. It does not alter earlier M1 experiments or download trade data.

The primary sequence for outcome year `y` uses annual feature vintages `y-6,...,y-2`. Every input is at or before the audited `y-2` information cutoff. The primary comparison is a lag-matched LightGBM trained on the same sequence and static information. The 2022–2024 evaluation years have been repeatedly inspected during M1 development, so results are exploratory retrospective evidence.

Run from the M1 repository root with:

```bash
.venv_uv/bin/python experiments/gru_temporal_market_selection_v1/run_gru_experiment.py
```

All outputs are confined to this directory. No figures are produced.
