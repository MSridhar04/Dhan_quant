# dhan_quant

Quant research and live market odds for NIFTY and SENSEX. Formerly `dhan_prob` (renamed 15 Nov 2026).

**What it is, in one line:** we turn 5 years of minute-by-minute market data into honest probabilities
("the next 30 minutes have a 52% chance of being busy, normal is 25%"), test every idea on data it never saw,
and show the live odds in the Control Center's **Quant** tab. It never places orders.

## Read this first (plain English, with examples)
| Doc | What you'll learn |
|---|---|
| [docs/00_overview.md](docs/00_overview.md) | What "quant" means for us, why not HFT, the roadmap |
| [docs/01_data_and_features.md](docs/01_data_and_features.md) | The data we have and every "feature" the models look at |
| [docs/02_how_we_test.md](docs/02_how_we_test.md) | How we avoid fooling ourselves: walk-forward, costs, calibration, luck discount |
| [docs/03_live_dashboard.md](docs/03_live_dashboard.md) | How the live engine works and how to read the Quant tab |
| [docs/04_results_so_far.md](docs/04_results_so_far.md) | Every test so far and what it taught us |
| [docs/05_research_plan.md](docs/05_research_plan.md) | The quant strategies we'll research next, explained simply |
| [docs/06_vrp.md](docs/06_vrp.md) | Research #1: volatility risk premium, the method and the verdict |
| [docs/07_regimes.md](docs/07_regimes.md) | Research #2: regime detection, the method and the verdict |
| [docs/glossary.md](docs/glossary.md) | Words like basis point, AUC, calibration, implied volatility |

## Folder map
```
quant/features.py        the ONE feature definition (research + live use the same code)
quant/dataset.py         builds the 5-year minute dataset from dhan_algo's cache (read-only)
quant/harness.py         testing toolkit: walk-forward, real costs, luck discount (deflated Sharpe)
quant/model.py, step1*.py, quietday.py, analysis.py    research done so far (see docs/04)
quant/live/train.py      trains the models the live engine uses
quant/live/engine.py     live engine (runs 09:14-15:31 on trading days) + replay mode for testing
live/state.json          what the Quant tab shows (written every minute)
live/predictions/        every live prediction and, 30 minutes later, what really happened
models/                  saved models     reports/   research results     data/   datasets (not in git)
```

## Everyday commands
```powershell
.venv\Scripts\python.exe -m quant.live.engine --replay 2026-11-13   # replay a past day through the live engine
.venv\Scripts\python.exe -m quant.dataset                           # rebuild the research dataset
.venv\Scripts\python.exe -m quant.live.train                        # retrain the live models
```
The live engine starts by itself: Windows scheduled task **"DhanQuant Live"**, weekdays 09:05 (waits for 09:14).
It uses the Dhan token in `dhan_algo\.env`, its own market-data connection, and never touches paper trading.
