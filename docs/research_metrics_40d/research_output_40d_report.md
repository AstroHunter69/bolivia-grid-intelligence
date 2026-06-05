# 40-Day Research Output Report

Report date: 2026-06-05.

## Scope

This report replaces the earlier 10-day evidence summary with wider 40-day experiments where the local data allows it.

The demand model is a **full-day-ahead forecaster**: it predicts an entire next-day demand curve at **15-minute resolution**. The phrase "15-minute model" refers to output granularity, not a 15-minute forecast horizon.

## 1. Strict 40-Day Full-Day-Ahead Forecast Backtest

Methodology: for each target date, actual demand for that date was masked. The model predicted the full day recursively at 15-minute resolution using only historical demand before the target date and the target-day CNDC forecast.

Period: `2026-04-17 00:00:00` to `2026-05-26 23:15:00`  
Days: **40**  
Rows: **3781**

| Metric | BDM full-day-ahead | CNDC forecast | Improvement |
|---|---:|---:|---:|
| MAE | 35.81 MW | 83.93 MW | 48.12 MW lower (57.34%) |
| RMSE | 48.89 MW | 98.16 MW | - |
| Bias | -19.77 MW | -80.86 MW | Lower absolute bias |
| Daily win rate | 95.00% | - | BDM beat CNDC on 38/40 days |
| Same-weekday naive MAE | 76.83 MW | - | BDM also beats this baseline in the strict 40-day test |

Interpretation: the model is not merely doing one-step-ahead prediction. It can generate a complete next-day demand curve at 15-minute resolution, and in this 40-day recursive test it reduced MAE vs CNDC by **57.34%**.

## 2. 40-Day Aggregate RL Backtest

The RL experiment was rerun on aggregate CNDC data from `2026-04-17` to `2026-05-26`. The agent learns a discrete reserve adjustment around the demand forecast.

Held-out test period: `2026-05-14 23:00:00` to `2026-05-26 23:15:00`  
Held-out rows: **1135**

| Metric | No-adjustment policy | Q-learning policy | Improvement |
|---|---:|---:|---:|
| Test MAE | 73.18 MW | 30.68 MW | 42.49 MW lower (58.07%) |
| Avg dispatch | 1,110.80 MW | 1,187.60 MW | Closer to real demand |
| Avg real demand | 1,183.09 MW | 1,183.09 MW | - |
| Imbalance cost proxy | $326,956.15 | $137,027.93 | $189,928.22 lower (58.09%) |

Interpretation: the RL agent improved the aggregate grid-balancing proxy by reducing forecast-dispatch mismatch. This is a reserve-support result, not a full plant-control policy.

## 3. 40-Day Technology-Level Dispatch and Renewable Integration Simulation

This simulation uses the 40-day aggregate generation-by-type data, CNDC static capacity detail, demand, and simplified technology cost assumptions. It is less granular than the one-day plant-level simulation, but it gives a 40-day view without requiring thousands of plant endpoint calls.

Period: `2026-04-17 00:00:00` to `2026-05-26 23:15:00`  
Rows: **3781**  
Days: **40**

| KPI | Historical aggregate dispatch | Optimized technology dispatch | Change |
|---|---:|---:|---:|
| Cost proxy | $51,828,695.41 | $39,039,285.82 | $12,789,409.59 lower (24.68%) |
| Renewable share | 33.43% | 60.81% | +27.39 percentage points |
| Renewable energy | 413,632.46 MWh | 747,852.43 MWh | +334,219.97 MWh (80.80% relative MWh increase) |
| Thermal energy | - | - | -341,915.80 MWh |
| Max unserved MW | - | 0.000 MW | No unserved load in simplified simulation |

Technology-level capacity audit:

| Technology | Historical MWh | Optimized MWh | Delta MWh | Optimized peak | CNDC static capacity | Peak utilization |
|---|---:|---:|---:|---:|---:|---:|
| BIO | 26.8 | 26.8 | 0.0 | 10.5 MW | 125.0 MW | 8.41% |
| EOL | 33,441.8 | 33,441.8 | 0.0 | 88.5 MW | 135.0 MW | 65.55% |
| HID | 342,371.7 | 676,591.7 | 334,220.0 | 715.8 MW | 715.8 MW | 100.00% |
| SOL | 37,792.1 | 37,792.1 | 0.0 | 151.4 MW | 167.6 MW | 90.37% |
| TER | 823,809.3 | 481,893.5 | -341,915.8 | 2,387.5 MW | 2,387.5 MW | 100.00% |

Interpretation: the optimized dispatch is backed by CNDC static capacity at the **power capacity** level. For example, optimized hydro peaks at the CNDC hydro capacity boundary, so the model is not exceeding stated installed/effective capacity. However, capacity alone does not prove water availability, reservoir constraints, or hydro scheduling feasibility across the full period.

## 4. What Is Marginal Cost Used For?

CNDC marginal cost is useful, but it is not identical to plant-by-plant marginal cost.

In this project:

- The **RL environment** uses marginal cost as a system-level cost-pressure signal in the reward.
- The **imbalance cost proxy** uses marginal cost to estimate the cost of forecast/dispatch mismatch.
- The **plant/technology dispatch simulator** uses simplified technology cost assumptions because CNDC marginal cost forecast is not a plant-specific bid curve.

For the 2026-05-31 sample, CNDC marginal cost forecast ranged from **15.56** to **16.99 USD/MWh**, with an average of **16.32 USD/MWh**.

Best interpretation: marginal cost helps score system conditions and imbalance risk; plant-specific economic dispatch still needs plant-level cost data or official bid/marginal-cost curves.

## 5. Is the Hydro Increase Physically Backed?

Partially, yes.

The CNDC static capacity detail reports **715.81 MW** of hydro capacity. In the one-day plant-level simulation, optimized hydro peak was **597.27 MW**, which is **83.44%** of CNDC hydro capacity. In the 40-day aggregate simulation, optimized hydro can reach the hydro capacity boundary of **715.81 MW**, so the power-capacity constraint is respected.

But there is an important caveat:

- Capacity confirms maximum MW capability.
- It does **not** confirm reservoir/water availability, hydrological constraints, ramp limits, maintenance status, minimum generation, or transmission constraints.

So the correct claim is:

> The simulation is capacity-backed at the installed/effective MW level using CNDC capacity detail, but it is not yet hydro-energy-constrained or network-constrained.

## 6. Updated CV-Safe Claims

**Forecasting**

> Built a full-day-ahead 15-minute electricity demand forecasting model for Bolivia, reducing MAE vs CNDC by **57.34%** over a strict 40-day recursive backtest.

**RL**

> Formulated grid balancing as a Q-learning reserve-adjustment problem and reduced held-out forecast-dispatch mismatch by **58.07%** over a 40-day aggregate CNDC dataset.

**Dispatch and renewables**

> Built a capacity-aware technology-level dispatch simulator showing **$12,789,409.59** 40-day cost-proxy savings and renewable share increase from **33.43%** to **60.81%** under simplified cost assumptions.

## 7. Caveats

- Monetary values are cost-proxy savings, not audited financial savings.
- Technology dispatch is capacity-aware but not unit-commitment, hydrology, ramp-rate, outage, or transmission constrained.
- CNDC marginal cost is used as a system-level signal, not as plant-specific cost data.
- The RL result is a reserve-policy improvement, not a final real-time plant-control agent.
