#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apr17_may26_40d_aggregate"
OUT = ROOT / "docs" / "research_metrics" / "aggregate_dispatch_40d"

COSTS = {"EOL": 5.0, "SOL": 5.0, "BIO": 12.0, "HID": 18.0, "TER": 55.0}
RENEWABLE = {"EOL", "SOL", "BIO", "HID"}
GEN_COLS = {"eol": "EOL", "solar": "SOL", "bagazo": "BIO", "hidro": "HID", "termo": "TER"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    demand = pd.read_csv(DATA / "demand_15min.csv", parse_dates=["timestamp"])[["timestamp", "total"]].rename(columns={"total": "demand_mw"})
    gen = pd.read_csv(DATA / "generation_15min.csv", parse_dates=["timestamp"])
    cap = pd.read_csv(DATA / "capacity_detail_static.csv")
    capacity = cap.groupby("description")["value"].sum().to_dict()

    base = demand.merge(gen[["timestamp", *GEN_COLS.keys()]], on="timestamp", how="inner").dropna(subset=["demand_mw"])
    rows = []
    reserve_margin = 0.03

    for _, row in base.iterrows():
        target = float(row["demand_mw"]) * (1 + reserve_margin)
        remaining = target
        actual = {typ: float(row[col]) for col, typ in GEN_COLS.items()}
        optimized = {}
        for typ in sorted(COSTS, key=lambda k: COSTS[k]):
            if typ in {"EOL", "SOL", "BIO"}:
                available = max(actual.get(typ, 0.0), 0.0)
            else:
                available = max(float(capacity.get(typ, 0.0)), 0.0)
            dispatch = min(available, max(remaining, 0.0))
            optimized[typ] = dispatch
            remaining -= dispatch

        for typ in COSTS:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "generation_type": typ,
                    "demand_mw": row["demand_mw"],
                    "actual_dispatch_mw": actual.get(typ, 0.0),
                    "optimized_dispatch_mw": optimized.get(typ, 0.0),
                    "cost_usd_mwh": COSTS[typ],
                    "is_renewable": typ in RENEWABLE,
                    "unserved_mw": max(remaining, 0.0),
                }
            )

    long = pd.DataFrame(rows)
    for col in ["actual_dispatch_mw", "optimized_dispatch_mw"]:
        long[col.replace("_dispatch_mw", "_mwh")] = long[col] * 0.25
    long["actual_cost_proxy_usd"] = long["actual_mwh"] * long["cost_usd_mwh"]
    long["optimized_cost_proxy_usd"] = long["optimized_mwh"] * long["cost_usd_mwh"]
    long["actual_renewable_mwh"] = np.where(long["is_renewable"], long["actual_mwh"], 0.0)
    long["optimized_renewable_mwh"] = np.where(long["is_renewable"], long["optimized_mwh"], 0.0)
    long.to_csv(OUT / "aggregate_dispatch_40d_long.csv", index=False)

    interval = (
        long.groupby("timestamp", as_index=False)
        .agg(
            demand_mw=("demand_mw", "first"),
            actual_total_mwh=("actual_mwh", "sum"),
            optimized_total_mwh=("optimized_mwh", "sum"),
            actual_cost_proxy_usd=("actual_cost_proxy_usd", "sum"),
            optimized_cost_proxy_usd=("optimized_cost_proxy_usd", "sum"),
            actual_renewable_mwh=("actual_renewable_mwh", "sum"),
            optimized_renewable_mwh=("optimized_renewable_mwh", "sum"),
            unserved_mw=("unserved_mw", "max"),
        )
    )
    interval["actual_renewable_share"] = interval["actual_renewable_mwh"] / interval["actual_total_mwh"]
    interval["optimized_renewable_share"] = interval["optimized_renewable_mwh"] / interval["optimized_total_mwh"]
    interval["cost_proxy_saving_usd"] = interval["actual_cost_proxy_usd"] - interval["optimized_cost_proxy_usd"]
    interval.to_csv(OUT / "aggregate_dispatch_40d_interval_summary.csv", index=False)

    by_type = (
        long.groupby("generation_type", as_index=False)
        .agg(
            actual_mwh=("actual_mwh", "sum"),
            optimized_mwh=("optimized_mwh", "sum"),
            actual_cost_proxy_usd=("actual_cost_proxy_usd", "sum"),
            optimized_cost_proxy_usd=("optimized_cost_proxy_usd", "sum"),
            actual_peak_mw=("actual_dispatch_mw", "max"),
            optimized_peak_mw=("optimized_dispatch_mw", "max"),
        )
    )
    by_type["delta_mwh"] = by_type["optimized_mwh"] - by_type["actual_mwh"]
    by_type["cost_proxy_saving_usd"] = by_type["actual_cost_proxy_usd"] - by_type["optimized_cost_proxy_usd"]
    by_type["cndc_static_capacity_mw"] = by_type["generation_type"].map(capacity)
    by_type["optimized_peak_capacity_utilization_pct"] = by_type["optimized_peak_mw"] / by_type["cndc_static_capacity_mw"] * 100
    by_type.to_csv(OUT / "aggregate_dispatch_40d_by_type.csv", index=False)

    report = {
        "start": str(interval["timestamp"].min()),
        "end": str(interval["timestamp"].max()),
        "rows": int(len(interval)),
        "days": int(interval["timestamp"].dt.date.nunique()),
        "reserve_margin": reserve_margin,
        "actual_cost_proxy_usd": float(interval["actual_cost_proxy_usd"].sum()),
        "optimized_cost_proxy_usd": float(interval["optimized_cost_proxy_usd"].sum()),
        "cost_proxy_saving_usd": float(interval["cost_proxy_saving_usd"].sum()),
        "cost_proxy_saving_pct": float(interval["cost_proxy_saving_usd"].sum() / interval["actual_cost_proxy_usd"].sum() * 100),
        "actual_renewable_share": float(interval["actual_renewable_mwh"].sum() / interval["actual_total_mwh"].sum()),
        "optimized_renewable_share": float(interval["optimized_renewable_mwh"].sum() / interval["optimized_total_mwh"].sum()),
        "renewable_mwh_increase": float(interval["optimized_renewable_mwh"].sum() - interval["actual_renewable_mwh"].sum()),
        "thermal_mwh_reduction": float(
            by_type.loc[by_type["generation_type"].eq("TER"), "actual_mwh"].iloc[0]
            - by_type.loc[by_type["generation_type"].eq("TER"), "optimized_mwh"].iloc[0]
        ),
        "unserved_mw_max": float(interval["unserved_mw"].max()),
        "capacity_by_type_mw": {k: float(v) for k, v in capacity.items()},
        "cost_assumptions_usd_mwh": COSTS,
        "note": "Technology-level 40-day dispatch simulation using aggregate generation-by-type data and CNDC static capacity detail. Cost values are simplified technology proxies.",
    }
    (OUT / "aggregate_dispatch_40d_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
