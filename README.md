# Bolivia Grid Intelligence

Forecasting, backtesting, and dispatch-support tools for Bolivia's electricity system using CNDC demand, forecast, generation, marginal-cost, and capacity data.

This project started as a practical research prototype: collect operational energy data, build a full-day-ahead demand forecaster, evaluate it against CNDC forecasts, and frame dispatch support as a reinforcement learning problem.

## What It Does

- Collects CNDC demand, demand forecast, generation, marginal-cost, capacity, and plant-dispatch data.
- Generates full-day-ahead demand forecasts at 15-minute resolution with the Bolivia Demand Model.
- Runs strict recursive day-ahead backtests.
- Trains a Q-learning reserve-adjustment policy for aggregate dispatch support.
- Simulates technology-level dispatch using transparent cost assumptions and CNDC capacity limits.
- Serves a local dashboard for forecasting, optimization, RL experiments, and saved results.

## Key Results

The current evidence package is based on a 40-day evaluation window from `2026-04-17` to `2026-05-26`.

| Area | Result |
|---|---:|
| Full-day-ahead forecast MAE | 35.81 MW |
| CNDC forecast MAE on same period | 83.93 MW |
| Forecast MAE improvement vs CNDC | 57.34% |
| Daily win rate vs CNDC | 95.00% |
| Q-learning held-out mismatch reduction | 58.07% |
| RL imbalance cost-proxy saving | 189,928 USD proxy |
| Technology dispatch cost-proxy saving | 12.79M USD proxy |
| Renewable share, historical to simulated optimized | 33.43% to 60.81% |

See [`docs/research_metrics_40d/research_output_40d_report.md`](docs/research_metrics_40d/research_output_40d_report.md) for the full report and caveats.

## Repository Structure

```text
app/                         Local dashboard server
src/                         Data collection, forecasting, RL, and dispatch modules
scripts/                     Backtest and report generation scripts
docs/research_metrics_40d/   Final 40-day report and KPI summaries
docs/research_metrics/       Compact evidence used by the 40-day report
runs/                        Small committed summary files plus ignored generated runs
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The local dashboard can run data collection, forecasts, plant-level optimization, real-time dispatch recommendations, and RL experiments.

## Forecasting

The Bolivia Demand Model predicts a full next-day curve at 15-minute resolution. The model uses:

- CNDC regional demand forecast,
- calendar and cyclic time features,
- previous-day and previous-week demand lags,
- rolling demand statistics,
- CNDC forecast-error lags,
- regional demand lags.

Expected local model files:

```text
models/bolivia_demand_model.pkl
models/bolivia_demand_features.pkl
```

Example:

```bash
python run_project.py forecast --date 2026-06-06 --lookback-days 21 --chart
```

## Data Collection

Example:

```bash
python run_project.py collect \
  --start-date 2026-05-01 \
  --end-date 2026-05-07 \
  --name may_sample \
  --skip-token
```

Plant dispatch collection can be requested with `--plant`, but use it carefully because it requires many endpoint calls.

## Reinforcement Learning

The RL component is a research framing for dispatch support. It is not an operational grid controller.

State includes time, demand forecast, real demand context, generation context, forecast error, and marginal-cost signal. The action is a discrete reserve adjustment around the forecast. The reward penalizes imbalance and cost pressure.

Example:

```bash
python run_project.py rl \
  --start-date 2026-04-17 \
  --end-date 2026-05-26 \
  --name apr17_may26_40d_aggregate \
  --episodes 400
```

## Dispatch Simulation

The dispatch optimizer is capacity-aware and uses transparent technology cost assumptions. It is designed for research comparison and visualization, not market settlement or operational dispatch.

Important limitations:

- no full unit commitment,
- no hydrological reservoir constraints,
- no outage schedule,
- no ramp-rate constraints,
- no AC power-flow or transmission constraint model,
- cost values are proxy assumptions unless replaced with official plant-level cost data.

## Reports

The main report is here:

- [`docs/research_metrics_40d/research_output_40d_report.md`](docs/research_metrics_40d/research_output_40d_report.md)
- [`docs/research_metrics_40d/research_output_40d_kpis.csv`](docs/research_metrics_40d/research_output_40d_kpis.csv)

## Notes

The public repository intentionally excludes private local datasets, raw CNDC caches, and model binaries. See [`docs/DATA_AND_MODEL_NOTES.md`](docs/DATA_AND_MODEL_NOTES.md).
