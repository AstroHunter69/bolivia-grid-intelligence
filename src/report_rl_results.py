#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_trace(path: Path, policy: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["policy"] = policy
    df["split"] = split
    df["interval_hours"] = 0.25
    df["imbalance_cost_proxy_usd"] = df["absolute_error_mw"] * df["marginal_cost"] * df["interval_hours"]
    df["dispatch_cost_proxy_usd"] = df["dispatch_mw"] * df["marginal_cost"] * df["interval_hours"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "policy"])
        .agg(
            rows=("timestamp", "count"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            mae_mw=("absolute_error_mw", "mean"),
            total_reward=("reward", "sum"),
            imbalance_cost_proxy_usd=("imbalance_cost_proxy_usd", "sum"),
            dispatch_cost_proxy_usd=("dispatch_cost_proxy_usd", "sum"),
            avg_dispatch_mw=("dispatch_mw", "mean"),
            avg_real_demand_mw=("real_demand_mw", "mean"),
        )
        .reset_index()
    )


def add_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in summary.groupby("split"):
        baseline = group[group["policy"] == "cndc_forecast_no_adjustment"].iloc[0]
        for _, row in group.iterrows():
            item = row.to_dict()
            item["mae_improvement_mw"] = baseline["mae_mw"] - row["mae_mw"]
            item["imbalance_cost_proxy_saving_usd"] = (
                baseline["imbalance_cost_proxy_usd"] - row["imbalance_cost_proxy_usd"]
            )
            item["imbalance_cost_proxy_saving_pct"] = (
                item["imbalance_cost_proxy_saving_usd"] / baseline["imbalance_cost_proxy_usd"] * 100
                if baseline["imbalance_cost_proxy_usd"]
                else 0
            )
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    traces = pd.concat(
        [
            load_trace(args.run_dir / "baseline_train_policy_trace.csv", "cndc_forecast_no_adjustment", "train"),
            load_trace(args.run_dir / "q_learning_train_policy_trace.csv", "q_learning_discrete_reserve", "train"),
            load_trace(args.run_dir / "baseline_test_policy_trace.csv", "cndc_forecast_no_adjustment", "test"),
            load_trace(args.run_dir / "q_learning_test_policy_trace.csv", "q_learning_discrete_reserve", "test"),
        ],
        ignore_index=True,
    )

    summary = add_improvements(summarize(traces))
    summary_path = args.run_dir / "rl_cost_proxy_summary.csv"
    traces_path = args.run_dir / "rl_policy_traces_with_cost_proxy.csv"
    summary.to_csv(summary_path, index=False)
    traces.to_csv(traces_path, index=False)

    test = summary[summary["split"] == "test"].copy()
    print("================================")
    print("RL Cost-Proxy Report")
    print("================================")
    print(test.round(3).to_string(index=False))
    print()
    print("Cost proxy = absolute error MW * marginal cost * 0.25h.")
    print("This is an avoided imbalance-cost proxy, not an audited national savings claim.")
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {traces_path}")


if __name__ == "__main__":
    main()
