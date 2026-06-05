#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COSTS = {
    "EOL": 5.0,
    "SOL": 5.0,
    "BIO": 12.0,
    "HID": 18.0,
    "INT": 35.0,
    "TER": 55.0,
}

RENEWABLE_TYPES = {"EOL", "SOL", "BIO", "HID"}


def normalize_generation_type(value: object) -> str:
    return str(value or "UNK").strip().upper()


def add_aggregate_variable_renewables(data_dir: Path, plant: pd.DataFrame) -> pd.DataFrame:
    generation_path = data_dir / "generation_15min.csv"
    if not generation_path.exists():
        return plant

    generation = pd.read_csv(generation_path, parse_dates=["timestamp"])
    pseudo_specs = {
        "eol": ("AGG_EOL", "EOL"),
        "solar": ("AGG_SOL", "SOL"),
    }
    rows = []
    for column, (code, gen_type) in pseudo_specs.items():
        if column not in generation.columns:
            continue
        for _, row in generation[["timestamp", column]].dropna().iterrows():
            value = float(row[column])
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "date": row["timestamp"].date().isoformat(),
                    "interval": int((row["timestamp"].hour * 60 + row["timestamp"].minute) / 15),
                    "plant_code": code,
                    "generation_type": gen_type,
                    "value_mw": max(value, 0.0),
                    "is_aggregate_pseudo_plant": True,
                    "availability_mw": max(value, 0.0),
                }
            )

    if not rows:
        plant["is_aggregate_pseudo_plant"] = False
        plant["availability_mw"] = np.nan
        return plant

    plant = plant.copy()
    plant["is_aggregate_pseudo_plant"] = False
    plant["availability_mw"] = np.nan
    return pd.concat([plant, pd.DataFrame(rows)], ignore_index=True)


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plant = pd.read_csv(data_dir / "plant_dispatch_15min_long.csv", parse_dates=["timestamp"])
    demand = pd.read_csv(data_dir / "demand_15min.csv", parse_dates=["timestamp"])
    capacity = pd.read_csv(data_dir / "capacity_detail_static.csv")

    plant["generation_type"] = plant["generation_type"].map(normalize_generation_type)
    plant["value_mw"] = pd.to_numeric(plant["value_mw"], errors="coerce").fillna(0)
    plant = add_aggregate_variable_renewables(data_dir, plant)
    demand = demand[["timestamp", "total"]].rename(columns={"total": "demand_mw"})
    demand["demand_mw"] = pd.to_numeric(demand["demand_mw"], errors="coerce")
    demand = demand.dropna(subset=["timestamp", "demand_mw"])
    return plant, demand, capacity


def infer_capacity(plant: pd.DataFrame, capacity_margin: float) -> pd.DataFrame:
    observed = (
        plant.groupby(["plant_code", "generation_type"], as_index=False)
        .agg(
            observed_max_mw=("value_mw", "max"),
            is_aggregate_pseudo_plant=("is_aggregate_pseudo_plant", "max"),
        )
    )
    observed["capacity_mw"] = observed["observed_max_mw"] * capacity_margin
    observed["capacity_mw"] = observed["capacity_mw"].clip(lower=0.01)
    observed["cost_usd_mwh"] = observed["generation_type"].map(DEFAULT_COSTS).fillna(60.0)
    observed["is_renewable"] = observed["generation_type"].isin(RENEWABLE_TYPES)
    return observed.sort_values(["cost_usd_mwh", "generation_type", "plant_code"]).reset_index(drop=True)


def merit_order_dispatch(
    plants: pd.DataFrame,
    demand_mw: float,
    reserve_margin: float,
    availability: dict[str, float],
) -> pd.DataFrame:
    target = demand_mw * (1 + reserve_margin)
    remaining = target
    rows = []

    for _, plant in plants.iterrows():
        capacity = float(plant["capacity_mw"])
        if bool(plant.get("is_aggregate_pseudo_plant", False)):
            capacity = min(capacity, float(availability.get(plant["plant_code"], 0.0)))
        dispatch = min(capacity, max(remaining, 0.0))
        remaining -= dispatch
        rows.append(
            {
                "plant_code": plant["plant_code"],
                "generation_type": plant["generation_type"],
                "optimized_dispatch_mw": dispatch,
                "cost_usd_mwh": plant["cost_usd_mwh"],
                "is_renewable": plant["is_renewable"],
            }
        )

    out = pd.DataFrame(rows)
    out["unserved_mw"] = max(remaining, 0.0)
    return out


def run_optimization(
    plant: pd.DataFrame,
    demand: pd.DataFrame,
    capacity_margin: float,
    reserve_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plants = infer_capacity(plant, capacity_margin)
    optimized_frames = []

    for _, row in demand.iterrows():
        ts = row["timestamp"]
        availability = (
            plant[
                (plant["timestamp"] == ts)
                & (plant["is_aggregate_pseudo_plant"].fillna(False))
            ]
            .set_index("plant_code")["availability_mw"]
            .to_dict()
        )
        dispatch = merit_order_dispatch(plants, float(row["demand_mw"]), reserve_margin, availability)
        dispatch["timestamp"] = row["timestamp"]
        dispatch["demand_mw"] = row["demand_mw"]
        optimized_frames.append(dispatch)

    optimized = pd.concat(optimized_frames, ignore_index=True)

    actual = plant.rename(columns={"value_mw": "actual_dispatch_mw"}).copy()
    actual["cost_usd_mwh"] = actual["generation_type"].map(DEFAULT_COSTS).fillna(60.0)
    actual["is_renewable"] = actual["generation_type"].isin(RENEWABLE_TYPES)

    merged = optimized.merge(
        actual[["timestamp", "plant_code", "actual_dispatch_mw"]],
        on=["timestamp", "plant_code"],
        how="left",
    )
    merged["actual_dispatch_mw"] = merged["actual_dispatch_mw"].fillna(0)
    merged["optimized_cost_proxy_usd"] = merged["optimized_dispatch_mw"] * merged["cost_usd_mwh"] * 0.25
    merged["actual_cost_proxy_usd"] = merged["actual_dispatch_mw"] * merged["cost_usd_mwh"] * 0.25
    merged["optimized_renewable_mwh"] = np.where(
        merged["is_renewable"], merged["optimized_dispatch_mw"] * 0.25, 0.0
    )
    merged["actual_renewable_mwh"] = np.where(merged["is_renewable"], merged["actual_dispatch_mw"] * 0.25, 0.0)
    merged["optimized_total_mwh"] = merged["optimized_dispatch_mw"] * 0.25
    merged["actual_total_mwh"] = merged["actual_dispatch_mw"] * 0.25
    return merged, plants


def summarize_interval(merged: pd.DataFrame) -> pd.DataFrame:
    summary = (
        merged.groupby("timestamp", as_index=False)
        .agg(
            demand_mw=("demand_mw", "first"),
            actual_dispatch_mw=("actual_dispatch_mw", "sum"),
            optimized_dispatch_mw=("optimized_dispatch_mw", "sum"),
            actual_cost_proxy_usd=("actual_cost_proxy_usd", "sum"),
            optimized_cost_proxy_usd=("optimized_cost_proxy_usd", "sum"),
            actual_renewable_mwh=("actual_renewable_mwh", "sum"),
            optimized_renewable_mwh=("optimized_renewable_mwh", "sum"),
            actual_total_mwh=("actual_total_mwh", "sum"),
            optimized_total_mwh=("optimized_total_mwh", "sum"),
            unserved_mw=("unserved_mw", "max"),
        )
        .sort_values("timestamp")
    )
    summary["actual_renewable_share"] = summary["actual_renewable_mwh"] / summary["actual_total_mwh"]
    summary["optimized_renewable_share"] = summary["optimized_renewable_mwh"] / summary["optimized_total_mwh"]
    summary["cost_proxy_saving_usd"] = summary["actual_cost_proxy_usd"] - summary["optimized_cost_proxy_usd"]
    return summary


def summarize_by_type(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby("generation_type", as_index=False)
        .agg(
            actual_mwh=("actual_total_mwh", "sum"),
            optimized_mwh=("optimized_total_mwh", "sum"),
            actual_cost_proxy_usd=("actual_cost_proxy_usd", "sum"),
            optimized_cost_proxy_usd=("optimized_cost_proxy_usd", "sum"),
        )
        .sort_values("generation_type")
    )


def summarize_daily(interval: pd.DataFrame) -> pd.DataFrame:
    daily = interval.copy()
    daily["date"] = daily["timestamp"].dt.date
    out = (
        daily.groupby("date", as_index=False)
        .agg(
            intervals=("timestamp", "count"),
            demand_mwh=("demand_mw", lambda s: s.sum() * 0.25),
            actual_dispatch_mwh=("actual_total_mwh", "sum"),
            optimized_dispatch_mwh=("optimized_total_mwh", "sum"),
            actual_cost_proxy_usd=("actual_cost_proxy_usd", "sum"),
            optimized_cost_proxy_usd=("optimized_cost_proxy_usd", "sum"),
            actual_renewable_mwh=("actual_renewable_mwh", "sum"),
            optimized_renewable_mwh=("optimized_renewable_mwh", "sum"),
            unserved_mw_max=("unserved_mw", "max"),
        )
        .sort_values("date")
    )
    out["cost_proxy_saving_usd"] = out["actual_cost_proxy_usd"] - out["optimized_cost_proxy_usd"]
    out["cost_proxy_saving_pct"] = out["cost_proxy_saving_usd"] / out["actual_cost_proxy_usd"] * 100
    out["actual_renewable_share"] = out["actual_renewable_mwh"] / out["actual_dispatch_mwh"]
    out["optimized_renewable_share"] = out["optimized_renewable_mwh"] / out["optimized_dispatch_mwh"]
    return out


def make_charts(interval: pd.DataFrame, by_type: pd.DataFrame, daily: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 16, "axes.titleweight": "bold"})

    fig, ax = plt.subplots(figsize=(13, 6), dpi=170)
    ax.plot(interval["timestamp"], interval["demand_mw"], label="Demand", color="#212529", linewidth=2.2)
    ax.plot(
        interval["timestamp"],
        interval["actual_dispatch_mw"],
        label="Historical dispatch",
        color="#868e96",
        linewidth=2.1,
        linestyle=(0, (4, 3)),
    )
    ax.plot(
        interval["timestamp"],
        interval["optimized_dispatch_mw"],
        label="Optimized merit-order dispatch",
        color="#0b7285",
        linewidth=2.4,
    )
    start = interval["timestamp"].min()
    end = interval["timestamp"].max()
    title_suffix = f"{start:%Y-%m-%d}" if start.date() == end.date() else f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    ax.set_title(f"Plant-Level Dispatch Simulation ({title_suffix})")
    ax.set_ylabel("MW")
    ax.set_xlabel("Time")
    ax.grid(True, color="#e9ecef")
    ax.legend(frameon=False)
    if start.date() == end.date():
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (end.date() - start.date()).days // 8)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(outdir / "dispatch_vs_demand.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=170)
    x = np.arange(len(by_type))
    width = 0.38
    ax.bar(x - width / 2, by_type["actual_mwh"], width, label="Historical", color="#868e96")
    ax.bar(x + width / 2, by_type["optimized_mwh"], width, label="Optimized", color="#0b7285")
    ax.set_xticks(x)
    ax.set_xticklabels(by_type["generation_type"])
    ax.set_title(f"Energy Mix by Generation Type ({title_suffix})")
    ax.set_ylabel("MWh")
    ax.grid(True, axis="y", color="#e9ecef")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "energy_mix_by_type.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=170)
    ax.plot(
        interval["timestamp"],
        interval["actual_renewable_share"] * 100,
        label="Historical renewable share",
        color="#868e96",
        linewidth=2.1,
        linestyle=(0, (4, 3)),
    )
    ax.plot(
        interval["timestamp"],
        interval["optimized_renewable_share"] * 100,
        label="Optimized renewable share",
        color="#2b8a3e",
        linewidth=2.4,
    )
    ax.set_title(f"Renewable Share in Dispatch ({title_suffix})")
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Time")
    ax.grid(True, color="#e9ecef")
    ax.legend(frameon=False)
    if start.date() == end.date():
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (end.date() - start.date()).days // 8)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(outdir / "renewable_share.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=170)
    ax.plot(
        interval["timestamp"],
        interval["cost_proxy_saving_usd"].cumsum(),
        color="#0b7285",
        linewidth=2.5,
    )
    ax.set_title(f"Cumulative Dispatch Cost-Proxy Saving ({title_suffix})")
    ax.set_ylabel("USD proxy")
    ax.set_xlabel("Time")
    ax.grid(True, color="#e9ecef")
    if start.date() == end.date():
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (end.date() - start.date()).days // 8)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(outdir / "cost_proxy_savings.png", bbox_inches="tight")
    plt.close(fig)

    if len(daily) > 1:
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=170)
        dates = pd.to_datetime(daily["date"])
        ax1.bar(dates, daily["cost_proxy_saving_usd"], color="#0b7285", alpha=0.82, label="Daily cost-proxy saving")
        ax1.set_ylabel("USD proxy")
        ax1.set_title(f"Daily Optimization Impact ({title_suffix})")
        ax1.grid(True, axis="y", color="#e9ecef")
        ax2 = ax1.twinx()
        ax2.plot(
            dates,
            daily["optimized_renewable_share"] * 100,
            color="#2b8a3e",
            linewidth=2.5,
            label="Optimized renewable share",
        )
        ax2.plot(
            dates,
            daily["actual_renewable_share"] * 100,
            color="#868e96",
            linewidth=2.0,
            linestyle=(0, (4, 3)),
            label="Historical renewable share",
        )
        ax2.set_ylabel("Renewable share (%)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate(rotation=0)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(outdir / "daily_optimization_impact.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple plant-level economic dispatch simulator")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--outdir", default=Path("runs/plant_dispatch"), type=Path)
    parser.add_argument("--capacity-margin", default=1.10, type=float)
    parser.add_argument("--reserve-margin", default=0.03, type=float)
    parser.add_argument("--model-prediction", type=Path)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    plant, demand, capacity = load_inputs(args.data_dir)
    merged, plants = run_optimization(plant, demand, args.capacity_margin, args.reserve_margin)
    interval = summarize_interval(merged)
    by_type = summarize_by_type(merged)
    daily = summarize_daily(interval)

    merged.to_csv(args.outdir / "plant_dispatch_comparison_long.csv", index=False)
    interval.to_csv(args.outdir / "dispatch_interval_summary.csv", index=False)
    by_type.to_csv(args.outdir / "dispatch_by_type_summary.csv", index=False)
    daily.to_csv(args.outdir / "dispatch_daily_summary.csv", index=False)
    plants.to_csv(args.outdir / "inferred_plant_capacities_and_costs.csv", index=False)
    capacity.to_csv(args.outdir / "source_capacity_detail_static.csv", index=False)

    total_actual_cost = interval["actual_cost_proxy_usd"].sum()
    total_optimized_cost = interval["optimized_cost_proxy_usd"].sum()
    total_saving = total_actual_cost - total_optimized_cost
    actual_renewable_share = interval["actual_renewable_mwh"].sum() / interval["actual_total_mwh"].sum()
    optimized_renewable_share = interval["optimized_renewable_mwh"].sum() / interval["optimized_total_mwh"].sum()

    report = {
        "rows": int(len(interval)),
        "start": str(interval["timestamp"].min()),
        "end": str(interval["timestamp"].max()),
        "reserve_margin": args.reserve_margin,
        "capacity_margin": args.capacity_margin,
        "actual_cost_proxy_usd": float(total_actual_cost),
        "optimized_cost_proxy_usd": float(total_optimized_cost),
        "cost_proxy_saving_usd": float(total_saving),
        "cost_proxy_saving_pct": float(total_saving / total_actual_cost * 100) if total_actual_cost else 0.0,
        "actual_renewable_share": float(actual_renewable_share),
        "optimized_renewable_share": float(optimized_renewable_share),
        "unserved_mw_max": float(interval["unserved_mw"].max()),
        "cost_assumptions_usd_mwh": DEFAULT_COSTS,
        "note": "Cost values are technology-level assumptions for a dispatch-cost proxy, not audited plant marginal costs.",
    }
    (args.outdir / "optimization_report.json").write_text(json.dumps(report, indent=2))

    if not args.no_charts:
        make_charts(interval, by_type, daily, args.outdir)

    if not args.no_dashboard:
        dashboard_script = Path(__file__).with_name("generate_interactive_dashboard.py")
        if dashboard_script.exists():
            cmd = [
                sys.executable,
                str(dashboard_script),
                "--run-dir",
                str(args.outdir),
                "--data-dir",
                str(args.data_dir),
            ]
            if args.model_prediction:
                cmd.extend(["--model-prediction", str(args.model_prediction)])
            subprocess.run(cmd, check=False)

    print("================================")
    print("Plant Dispatch Optimization")
    print("================================")
    print(f"Intervals:              {report['rows']}")
    print(f"Actual cost proxy:      ${report['actual_cost_proxy_usd']:,.2f}")
    print(f"Optimized cost proxy:   ${report['optimized_cost_proxy_usd']:,.2f}")
    print(f"Proxy saving:           ${report['cost_proxy_saving_usd']:,.2f} ({report['cost_proxy_saving_pct']:.1f}%)")
    print(f"Historical renewable:   {report['actual_renewable_share'] * 100:.1f}%")
    print(f"Optimized renewable:    {report['optimized_renewable_share'] * 100:.1f}%")
    print(f"Max unserved MW:        {report['unserved_mw_max']:.3f}")
    print()
    print("Note: this is a constrained merit-order simulation with assumed technology costs.")
    print(f"Saved: {args.outdir}")


if __name__ == "__main__":
    main()
