#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET = "target_demand_mw"
REGIONS = ["beni", "chuquisaca", "cochabamba", "la_paz", "oruro", "potosi", "santa_cruz", "tarija"]


def add_recursive_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    ts = frame["timestamp"]
    demand = frame[TARGET]

    frame["hour"] = ts.dt.hour
    frame["minute"] = ts.dt.minute
    frame["day_of_week"] = ts.dt.dayofweek
    frame["month"] = ts.dt.month
    frame["day_of_year"] = ts.dt.dayofyear
    frame["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)

    frame["target_lag_96"] = demand.shift(96)
    frame["target_lag_192"] = demand.shift(192)
    frame["target_lag_672"] = demand.shift(672)

    history = demand.shift(1)
    for window in (96, 672):
        rolling = history.rolling(window=window, min_periods=window)
        frame[f"rolling_mean_{window}"] = rolling.mean()
        frame[f"rolling_std_{window}"] = rolling.std()
        frame[f"rolling_min_{window}"] = rolling.min()
        frame[f"rolling_max_{window}"] = rolling.max()

    forecast_error = frame[TARGET] - frame["cndc_forecast_total_mw"]
    frame["forecast_error_lag_96"] = forecast_error.shift(96)
    frame["forecast_error_lag_672"] = forecast_error.shift(672)
    frame["safe_ramp_96"] = frame["target_lag_96"] - frame["target_lag_192"]
    frame["safe_ramp_672"] = frame["target_lag_672"] - demand.shift(1344)

    return frame


def choose_target_dates(df: pd.DataFrame, days: int, end_date: str | None) -> list[pd.Timestamp]:
    counts = (
        df.assign(date=df["timestamp"].dt.normalize())
        .groupby("date")
        .agg(rows=("timestamp", "count"), end=("timestamp", "max"))
        .reset_index()
    )
    complete = counts[counts["rows"] >= 94].copy()
    if end_date:
        complete = complete[complete["date"] <= pd.Timestamp(end_date)]
    else:
        complete = complete[complete["date"] <= pd.Timestamp("2026-05-26")]
    selected = complete.tail(days)["date"].tolist()
    if len(selected) < days:
        raise ValueError(f"Only found {len(selected)} complete target dates, requested {days}.")
    return selected


def predict_day(
    source: pd.DataFrame,
    target_date: pd.Timestamp,
    model,
    features: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    start = target_date - pd.Timedelta(days=lookback_days)
    end = target_date + pd.Timedelta(hours=23, minutes=45)
    base = source[(source["timestamp"] >= start) & (source["timestamp"] <= end)].copy()
    base = base.sort_values("timestamp").reset_index(drop=True)

    target_mask = base["timestamp"].dt.normalize().eq(target_date)
    truth = base.loc[target_mask, ["timestamp", TARGET, "cndc_forecast_total_mw"]].rename(columns={TARGET: "actual_demand_mw"})
    base.loc[target_mask, TARGET] = np.nan

    target_indices = base.index[target_mask].tolist()
    if not target_indices:
        return pd.DataFrame()

    for idx in target_indices:
        base = add_recursive_features(base)
        row = base.loc[[idx], features]
        missing = row.columns[row.isna().any()].tolist()
        if missing:
            raise ValueError(f"{target_date.date()} {base.at[idx, 'timestamp']} missing features: {missing}")
        pred = float(model.predict(row)[0])
        base.at[idx, TARGET] = pred

    out = base.loc[target_indices, ["timestamp", TARGET]].rename(columns={TARGET: "bdm_prediction_mw"})
    out = out.merge(truth, on="timestamp", how="left")
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["bdm_error_mw"] = out["bdm_prediction_mw"] - out["actual_demand_mw"]
    out["cndc_error_mw"] = out["cndc_forecast_total_mw"] - out["actual_demand_mw"]
    out["is_day_ahead_recursive"] = True
    return out


def summarize(pred: pd.DataFrame, label: str) -> dict:
    bdm_abs = pred["bdm_error_mw"].abs()
    cndc_abs = pred["cndc_error_mw"].abs()
    return {
        "label": label,
        "rows": int(len(pred)),
        "days": int(pred["date"].nunique()),
        "start": str(pred["timestamp"].min()),
        "end": str(pred["timestamp"].max()),
        "bdm_mae_mw": float(bdm_abs.mean()),
        "cndc_mae_mw": float(cndc_abs.mean()),
        "mae_improvement_mw": float(cndc_abs.mean() - bdm_abs.mean()),
        "mae_improvement_pct": float((cndc_abs.mean() - bdm_abs.mean()) / cndc_abs.mean() * 100),
        "bdm_rmse_mw": float(np.sqrt(mean_squared_error(pred["actual_demand_mw"], pred["bdm_prediction_mw"]))),
        "cndc_rmse_mw": float(np.sqrt(mean_squared_error(pred["actual_demand_mw"], pred["cndc_forecast_total_mw"]))),
        "bdm_bias_mw": float(pred["bdm_error_mw"].mean()),
        "cndc_bias_mw": float(pred["cndc_error_mw"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict recursive one-day-ahead BDM backtest from historical ML dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/cndc_ml_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/bolivia_demand_model.pkl"))
    parser.add_argument("--features", type=Path, default=Path("models/bolivia_demand_features.pkl"))
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--end-date")
    parser.add_argument("--outdir", type=Path, default=Path("docs/research_metrics/day_ahead_40d"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    features = joblib.load(args.features)
    model = joblib.load(args.model)

    target_dates = choose_target_dates(df, args.days, args.end_date)
    frames = []
    errors = []
    for day in target_dates:
        try:
            frames.append(predict_day(df, day, model, features, args.lookback_days))
        except Exception as exc:
            errors.append({"date": str(day.date()), "error": str(exc)})

    if not frames:
        raise RuntimeError(f"No successful day-ahead predictions. Errors: {errors[:3]}")

    pred = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    pred.to_csv(args.outdir / "day_ahead_40d_predictions.csv", index=False)

    daily = (
        pred.groupby("date")
        .apply(lambda g: pd.Series({
            "rows": len(g),
            "bdm_mae_mw": mean_absolute_error(g["actual_demand_mw"], g["bdm_prediction_mw"]),
            "cndc_mae_mw": mean_absolute_error(g["actual_demand_mw"], g["cndc_forecast_total_mw"]),
            "bdm_bias_mw": (g["bdm_prediction_mw"] - g["actual_demand_mw"]).mean(),
            "cndc_bias_mw": (g["cndc_forecast_total_mw"] - g["actual_demand_mw"]).mean(),
        }), include_groups=False)
        .reset_index()
    )
    daily["improvement_mw"] = daily["cndc_mae_mw"] - daily["bdm_mae_mw"]
    daily["improvement_pct"] = daily["improvement_mw"] / daily["cndc_mae_mw"] * 100
    daily["bdm_beats_cndc"] = daily["bdm_mae_mw"] < daily["cndc_mae_mw"]
    daily.to_csv(args.outdir / "day_ahead_40d_daily_summary.csv", index=False)

    actual_by_time = pred.set_index("timestamp")["actual_demand_mw"]
    pred["same_weekday_baseline_mw"] = pred["timestamp"].map(lambda ts: actual_by_time.get(ts - pd.Timedelta(days=7), np.nan))
    pred["same_weekday_error_mw"] = pred["same_weekday_baseline_mw"] - pred["actual_demand_mw"]

    summary = summarize(pred, "strict_recursive_day_ahead_40d")
    summary["target_dates"] = [str(d.date()) for d in target_dates]
    summary["successful_days"] = int(pred["date"].nunique())
    summary["failed_days"] = errors
    summary["daily_win_rate_pct"] = float(daily["bdm_beats_cndc"].mean() * 100)
    summary["same_weekday_mae_mw"] = float(pred["same_weekday_error_mw"].abs().mean())
    summary["methodology"] = (
        "For each target date, actual demand on that date is masked. The model predicts the full day "
        "recursively at 15-minute resolution using target-day CNDC forecast and historical demand before the target date."
    )
    (args.outdir / "day_ahead_40d_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
