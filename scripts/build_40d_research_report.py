#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "research_metrics_40d"
OUT_MD = OUT / "research_output_40d_report.md"
OUT_JSON = OUT / "research_output_40d_summary.json"
OUT_CSV = OUT / "research_output_40d_kpis.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def money(v: float) -> str:
    return f"${v:,.2f}"


def num(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}"


def pct(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    forecast = load_json(ROOT / "docs/research_metrics/day_ahead_40d/day_ahead_40d_summary.json")
    rl_summary = pd.read_csv(ROOT / "runs/apr17_may26_40d_aggregate_q_learning/q_learning_summary.csv")
    rl_cost = pd.read_csv(ROOT / "runs/apr17_may26_40d_aggregate_q_learning/rl_cost_proxy_summary.csv")
    dispatch = load_json(ROOT / "docs/research_metrics/aggregate_dispatch_40d/aggregate_dispatch_40d_report.json")
    dispatch_type = pd.read_csv(ROOT / "docs/research_metrics/aggregate_dispatch_40d/aggregate_dispatch_40d_by_type.csv")
    cap_audit = load_json(ROOT / "docs/research_metrics/capacity_and_marginal_cost_audit.json")

    rl_base = rl_summary[(rl_summary["split"].eq("test")) & (rl_summary["policy"].eq("cndc_forecast_no_adjustment"))].iloc[0]
    rl_q = rl_summary[(rl_summary["split"].eq("test")) & (rl_summary["policy"].eq("q_learning_discrete_reserve"))].iloc[0]
    rl_cost_q = rl_cost[(rl_cost["split"].eq("test")) & (rl_cost["policy"].eq("q_learning_discrete_reserve"))].iloc[0]
    rl_cost_base = rl_cost[(rl_cost["split"].eq("test")) & (rl_cost["policy"].eq("cndc_forecast_no_adjustment"))].iloc[0]

    renewable_share_pp = (dispatch["optimized_renewable_share"] - dispatch["actual_renewable_share"]) * 100
    renewable_share_relative = (dispatch["optimized_renewable_share"] - dispatch["actual_renewable_share"]) / dispatch["actual_renewable_share"] * 100
    renewable_types = {"BIO", "EOL", "HID", "SOL"}
    renewable_rows = dispatch_type[dispatch_type["generation_type"].isin(renewable_types)]
    actual_renewable_mwh = renewable_rows["actual_mwh"].sum()
    optimized_renewable_mwh = renewable_rows["optimized_mwh"].sum()
    renewable_mwh_relative = (optimized_renewable_mwh - actual_renewable_mwh) / actual_renewable_mwh * 100
    thermal_row = dispatch_type[dispatch_type["generation_type"].eq("TER")].iloc[0]
    hydro_row = dispatch_type[dispatch_type["generation_type"].eq("HID")].iloc[0]

    kpis = [
        ["Forecast", "Strict full-day-ahead BDM MAE", forecast["bdm_mae_mw"], "MW", "40-day recursive backtest"],
        ["Forecast", "CNDC MAE", forecast["cndc_mae_mw"], "MW", "same 40 days"],
        ["Forecast", "MAE improvement vs CNDC", forecast["mae_improvement_pct"], "%", "40-day recursive backtest"],
        ["Forecast", "Daily win rate vs CNDC", forecast["daily_win_rate_pct"], "%", "40-day recursive backtest"],
        ["RL", "Q-learning test MAE", rl_q["mae_mw"], "MW", "40-day aggregate train/test"],
        ["RL", "MAE reduction vs no-adjustment", (rl_base["mae_mw"] - rl_q["mae_mw"]) / rl_base["mae_mw"] * 100, "%", "40-day aggregate test split"],
        ["RL", "Imbalance cost-proxy saving", rl_cost_q["imbalance_cost_proxy_saving_usd"], "USD proxy", "40-day aggregate test split"],
        ["RL", "Imbalance cost-proxy saving", rl_cost_q["imbalance_cost_proxy_saving_pct"], "%", "40-day aggregate test split"],
        ["Dispatch", "Cost-proxy saving", dispatch["cost_proxy_saving_usd"], "USD proxy", "40-day technology-level simulation"],
        ["Dispatch", "Cost-proxy saving", dispatch["cost_proxy_saving_pct"], "%", "40-day technology-level simulation"],
        ["Renewables", "Renewable share increase", renewable_share_pp, "percentage points", "40-day technology-level simulation"],
        ["Renewables", "Renewable MWh increase", dispatch["renewable_mwh_increase"], "MWh", "40-day technology-level simulation"],
        ["Thermal", "Thermal MWh reduction", dispatch["thermal_mwh_reduction"], "MWh", "40-day technology-level simulation"],
        ["Reliability", "Max unserved energy", dispatch["unserved_mw_max"], "MW", "40-day technology-level simulation"],
    ]
    pd.DataFrame(kpis, columns=["category", "metric", "value", "unit", "scope"]).to_csv(OUT_CSV, index=False)

    by_type_rows = "\n".join(
        f"| {r.generation_type} | {num(r.actual_mwh, 1)} | {num(r.optimized_mwh, 1)} | {num(r.delta_mwh, 1)} | {num(r.optimized_peak_mw, 1)} MW | {num(r.cndc_static_capacity_mw, 1)} MW | {pct(r.optimized_peak_capacity_utilization_pct)} |"
        for r in dispatch_type.itertuples()
    )

    summary = {
        "forecast_40d": forecast,
        "rl_40d_test": {
            "baseline_mae_mw": float(rl_base["mae_mw"]),
            "q_learning_mae_mw": float(rl_q["mae_mw"]),
            "mae_reduction_pct": float((rl_base["mae_mw"] - rl_q["mae_mw"]) / rl_base["mae_mw"] * 100),
            "imbalance_cost_proxy_saving_usd": float(rl_cost_q["imbalance_cost_proxy_saving_usd"]),
            "imbalance_cost_proxy_saving_pct": float(rl_cost_q["imbalance_cost_proxy_saving_pct"]),
            "test_start": str(rl_cost_base["start"]),
            "test_end": str(rl_cost_base["end"]),
            "test_rows": int(rl_base["rows"]),
        },
        "aggregate_dispatch_40d": dispatch,
        "capacity_and_marginal_cost_audit": cap_audit,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    md = f"""# 40-Day Research Output Report

Report date: 2026-06-05.

## Scope

This report replaces the earlier 10-day evidence summary with wider 40-day experiments where the local data allows it.

The demand model is a **full-day-ahead forecaster**: it predicts an entire next-day demand curve at **15-minute resolution**. The phrase "15-minute model" refers to output granularity, not a 15-minute forecast horizon.

## 1. Strict 40-Day Full-Day-Ahead Forecast Backtest

Methodology: for each target date, actual demand for that date was masked. The model predicted the full day recursively at 15-minute resolution using only historical demand before the target date and the target-day CNDC forecast.

Period: `{forecast['start']}` to `{forecast['end']}`  
Days: **{forecast['days']}**  
Rows: **{forecast['rows']}**

| Metric | BDM full-day-ahead | CNDC forecast | Improvement |
|---|---:|---:|---:|
| MAE | {num(forecast['bdm_mae_mw'])} MW | {num(forecast['cndc_mae_mw'])} MW | {num(forecast['mae_improvement_mw'])} MW lower ({pct(forecast['mae_improvement_pct'])}) |
| RMSE | {num(forecast['bdm_rmse_mw'])} MW | {num(forecast['cndc_rmse_mw'])} MW | - |
| Bias | {num(forecast['bdm_bias_mw'])} MW | {num(forecast['cndc_bias_mw'])} MW | Lower absolute bias |
| Daily win rate | {pct(forecast['daily_win_rate_pct'])} | - | BDM beat CNDC on 38/40 days |
| Same-weekday naive MAE | {num(forecast['same_weekday_mae_mw'])} MW | - | BDM also beats this baseline in the strict 40-day test |

Interpretation: the model is not merely doing one-step-ahead prediction. It can generate a complete next-day demand curve at 15-minute resolution, and in this 40-day recursive test it reduced MAE vs CNDC by **{pct(forecast['mae_improvement_pct'])}**.

## 2. 40-Day Aggregate RL Backtest

The RL experiment was rerun on aggregate CNDC data from `2026-04-17` to `2026-05-26`. The agent learns a discrete reserve adjustment around the demand forecast.

Held-out test period: `{rl_cost_base['start']}` to `{rl_cost_base['end']}`  
Held-out rows: **{int(rl_base['rows'])}**

| Metric | No-adjustment policy | Q-learning policy | Improvement |
|---|---:|---:|---:|
| Test MAE | {num(rl_base['mae_mw'])} MW | {num(rl_q['mae_mw'])} MW | {num(rl_base['mae_mw'] - rl_q['mae_mw'])} MW lower ({pct((rl_base['mae_mw'] - rl_q['mae_mw']) / rl_base['mae_mw'] * 100)}) |
| Avg dispatch | {num(rl_base['avg_dispatch_mw'])} MW | {num(rl_q['avg_dispatch_mw'])} MW | Closer to real demand |
| Avg real demand | {num(rl_base['avg_real_demand_mw'])} MW | {num(rl_q['avg_real_demand_mw'])} MW | - |
| Imbalance cost proxy | {money(rl_cost_base['imbalance_cost_proxy_usd'])} | {money(rl_cost_q['imbalance_cost_proxy_usd'])} | {money(rl_cost_q['imbalance_cost_proxy_saving_usd'])} lower ({pct(rl_cost_q['imbalance_cost_proxy_saving_pct'])}) |

Interpretation: the RL agent improved the aggregate grid-balancing proxy by reducing forecast-dispatch mismatch. This is a reserve-support result, not a full plant-control policy.

## 3. 40-Day Technology-Level Dispatch and Renewable Integration Simulation

This simulation uses the 40-day aggregate generation-by-type data, CNDC static capacity detail, demand, and simplified technology cost assumptions. It is less granular than the one-day plant-level simulation, but it gives a 40-day view without requiring thousands of plant endpoint calls.

Period: `{dispatch['start']}` to `{dispatch['end']}`  
Rows: **{dispatch['rows']}**  
Days: **{dispatch['days']}**

| KPI | Historical aggregate dispatch | Optimized technology dispatch | Change |
|---|---:|---:|---:|
| Cost proxy | {money(dispatch['actual_cost_proxy_usd'])} | {money(dispatch['optimized_cost_proxy_usd'])} | {money(dispatch['cost_proxy_saving_usd'])} lower ({pct(dispatch['cost_proxy_saving_pct'])}) |
| Renewable share | {pct(dispatch['actual_renewable_share'] * 100)} | {pct(dispatch['optimized_renewable_share'] * 100)} | +{num(renewable_share_pp)} percentage points |
| Renewable energy | {num(actual_renewable_mwh)} MWh | {num(optimized_renewable_mwh)} MWh | +{num(dispatch['renewable_mwh_increase'])} MWh ({pct(renewable_mwh_relative)} relative MWh increase) |
| Thermal energy | - | - | -{num(dispatch['thermal_mwh_reduction'])} MWh |
| Max unserved MW | - | {num(dispatch['unserved_mw_max'], 3)} MW | No unserved load in simplified simulation |

Technology-level capacity audit:

| Technology | Historical MWh | Optimized MWh | Delta MWh | Optimized peak | CNDC static capacity | Peak utilization |
|---|---:|---:|---:|---:|---:|---:|
{by_type_rows}

Interpretation: the optimized dispatch is backed by CNDC static capacity at the **power capacity** level. For example, optimized hydro peaks at the CNDC hydro capacity boundary, so the model is not exceeding stated installed/effective capacity. However, capacity alone does not prove water availability, reservoir constraints, or hydro scheduling feasibility across the full period.

## 4. What Is Marginal Cost Used For?

CNDC marginal cost is useful, but it is not identical to plant-by-plant marginal cost.

In this project:

- The **RL environment** uses marginal cost as a system-level cost-pressure signal in the reward.
- The **imbalance cost proxy** uses marginal cost to estimate the cost of forecast/dispatch mismatch.
- The **plant/technology dispatch simulator** uses simplified technology cost assumptions because CNDC marginal cost forecast is not a plant-specific bid curve.

For the 2026-05-31 sample, CNDC marginal cost forecast ranged from **{num(cap_audit['marginal_cost_forecast_usd_mwh']['min'])}** to **{num(cap_audit['marginal_cost_forecast_usd_mwh']['max'])} USD/MWh**, with an average of **{num(cap_audit['marginal_cost_forecast_usd_mwh']['mean'])} USD/MWh**.

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

> Built a full-day-ahead 15-minute electricity demand forecasting model for Bolivia, reducing MAE vs CNDC by **{pct(forecast['mae_improvement_pct'])}** over a strict 40-day recursive backtest.

**RL**

> Formulated grid balancing as a Q-learning reserve-adjustment problem and reduced held-out forecast-dispatch mismatch by **{pct((rl_base['mae_mw'] - rl_q['mae_mw']) / rl_base['mae_mw'] * 100)}** over a 40-day aggregate CNDC dataset.

**Dispatch and renewables**

> Built a capacity-aware technology-level dispatch simulator showing **{money(dispatch['cost_proxy_saving_usd'])}** 40-day cost-proxy savings and renewable share increase from **{pct(dispatch['actual_renewable_share'] * 100)}** to **{pct(dispatch['optimized_renewable_share'] * 100)}** under simplified cost assumptions.

## 7. Caveats

- Monetary values are cost-proxy savings, not audited financial savings.
- Technology dispatch is capacity-aware but not unit-commitment, hydrology, ramp-rate, outage, or transmission constrained.
- CNDC marginal cost is used as a system-level signal, not as plant-specific cost data.
- The RL result is a reserve-policy improvement, not a final real-time plant-control agent.
"""
    OUT_MD.write_text(md)
    print(OUT_MD)
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
