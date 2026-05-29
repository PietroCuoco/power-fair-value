# European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View

> Daily fundamentals-driven fair-value model for German Day-Ahead power,
> translated into front-week baseload curve signals.

## Headline result
_Forecast-vs-actual figure and skill table go here (added Day 5)._

## TL;DR
- _skill vs baseline with DM-test significance_
- _where it works / where it fails_
- _backtested signal edge_

## Architecture
data (SMARD) -> QA -> features (point-in-time) -> model (LightGBM + quantiles)
-> validation (walk-forward, DM test) -> DA-to-curve signal | + LLM news features

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
python scripts/run_pipeline.py --stage all
```

## Data sources
| Source | Series | Licence | Access |
|---|---|---|---|
| SMARD.de (Bundesnetzagentur) | DA price, load (actual+forecast), wind/solar/residual-load forecast, generation | CC BY 4.0 | Token-free JSON |
| ENTSO-E Transparency | Cross-check (optional) | — | Token (email request) |

**Point-in-time note:** features use only information available before the
12:00 D-1 day-ahead gate closure (day-ahead *forecasts*, not realized actuals).
Enforced by a leakage-guard test.

## Limitations
_Filled in Day 5._