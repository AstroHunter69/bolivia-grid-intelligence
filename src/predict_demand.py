#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://cndcapi.cndc.bo/WebApi"
TARGET = "target_demand_mw"

REGIONS = [
    "beni",
    "chuquisaca",
    "cochabamba",
    "la_paz",
    "oruro",
    "potosi",
    "santa_cruz",
    "tarija",
]

DEMAND_CODE_MAP = {
    "BENI": "demand_beni_mw",
    "CHUQUISACA": "demand_chuquisaca_mw",
    "COCHABAMBA": "demand_cochabamba_mw",
    "LA PAZ": "demand_la_paz_mw",
    "ORURO": "demand_oruro_mw",
    "POTOSI": "demand_potosi_mw",
    "SANTA CRUZ": "demand_santa_cruz_mw",
    "TARIJA": "demand_tarija_mw",
}

FORECAST_CODE_MAP = {
    "BENI": "forecast_beni_mw",
    "CHUQUISACA": "forecast_chuquisaca_mw",
    "COCHABAMBA": "forecast_cochabamba_mw",
    "LA PAZ": "forecast_la_paz_mw",
    "ORURO": "forecast_oruro_mw",
    "POTOSI": "forecast_potosi_mw",
    "SANTA CRUZ": "forecast_santa_cruz_mw",
    "TARIJA": "forecast_tarija_mw",
}


@dataclass
class PredictionPaths:
    run_dir: Path
    prediction_csv: Path
    prediction_json: Path
    features_csv: Path
    history_csv: Path
    chart_png: Path
    metadata_json: Path


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def bolivia_date_string(date: pd.Timestamp) -> str:
    return date.strftime("%d.%m.%Y")


def clean_api_code(value: object) -> str:
    return str(value or "").strip().upper()


def fetch_json(code: int, date: pd.Timestamp, cache_dir: Path, refresh: bool = False) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"code{code}_{date:%Y-%m-%d}.json"
    collector_cache = Path(__file__).resolve().parents[1] / "data" / "cache" / f"webapi_code{code}_{date:%Y-%m-%d}.json"

    if cache_file.exists() and not refresh:
        payload = json.loads(cache_file.read_text())
        if payload or not collector_cache.exists():
            return payload
        collector_payload = json.loads(collector_cache.read_text())
        if collector_payload:
            cache_file.write_text(json.dumps(collector_payload, ensure_ascii=False, indent=2))
            return collector_payload

    response = requests.get(
        BASE_URL,
        params={"code": code, "Fecha": bolivia_date_string(date)},
        timeout=30,
        headers={
            "Accept": "*/*",
            "Origin": "https://www.cndc.bo",
            "Referer": "https://www.cndc.bo/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    response.raise_for_status()
    payload = response.json()
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    time.sleep(0.15)
    return payload


def demand_frame_for_day(date: pd.Timestamp, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    payload = fetch_json(code=1, date=date, cache_dir=cache_dir, refresh=refresh)
    data: dict[str, pd.Series] = {}

    for record in payload:
        code = clean_api_code(record.get("codigo"))
        if code.startswith("PREV"):
            continue
        col = DEMAND_CODE_MAP.get(code)
        if not col:
            continue
        values = pd.Series(record.get("valores", []), dtype="float64").replace(-1, np.nan)
        data[col] = values

    missing = [col for col in DEMAND_CODE_MAP.values() if col not in data]
    if missing:
        rows = pd.DataFrame({"timestamp": pd.date_range(start=date, periods=96, freq="15min")})
        for col in DEMAND_CODE_MAP.values():
            rows[col] = np.nan
        rows[TARGET] = np.nan
        return rows[["timestamp", TARGET, *DEMAND_CODE_MAP.values()]]

    rows = pd.DataFrame(data)
    rows["timestamp"] = pd.date_range(start=date, periods=len(rows), freq="15min")
    demand_cols = [f"demand_{region}_mw" for region in REGIONS]
    rows[TARGET] = rows[demand_cols].sum(axis=1, min_count=len(demand_cols))
    return rows[["timestamp", TARGET, *demand_cols]]


def forecast_frame_for_day(date: pd.Timestamp, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    payload = fetch_json(code=4, date=date, cache_dir=cache_dir, refresh=refresh)
    data: dict[str, pd.Series] = {}

    for record in payload:
        col = FORECAST_CODE_MAP.get(clean_api_code(record.get("codigo")))
        if not col:
            continue
        data[col] = pd.Series(record.get("valores", []), dtype="float64").replace(-1, np.nan)

    missing = [col for col in FORECAST_CODE_MAP.values() if col not in data]
    if missing:
        raise ValueError(f"Forecast response for {date:%Y-%m-%d} is missing columns: {missing}")

    hourly = pd.DataFrame(data)
    if len(hourly) != 24:
        raise ValueError(f"Forecast response for {date:%Y-%m-%d} has {len(hourly)} hourly rows, expected 24")

    hourly["timestamp"] = pd.date_range(start=date, periods=24, freq="h")
    hourly = hourly.set_index("timestamp")
    full_index = pd.date_range(start=date, periods=96, freq="15min")
    quarter_hour = (
        hourly.reindex(hourly.index.union(full_index))
        .interpolate("linear")
        .ffill()
        .reindex(full_index)
        .reset_index(names="timestamp")
    )

    forecast_cols = [f"forecast_{region}_mw" for region in REGIONS]
    quarter_hour["cndc_forecast_total_mw"] = quarter_hour[forecast_cols].sum(axis=1)
    return quarter_hour[["timestamp", *forecast_cols, "cndc_forecast_total_mw"]]


def fallback_forecast_frame_for_day(
    date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    forecast_cols = [f"forecast_{region}_mw" for region in REGIONS]
    for days_back in (7, 1, 2, 3):
        source_date = date - pd.Timedelta(days=days_back)
        source = demand_frame_for_day(source_date, cache_dir, refresh=refresh)
        demand_cols = [f"demand_{region}_mw" for region in REGIONS]
        if source[demand_cols].notna().all(axis=1).sum() < 80:
            continue
        source = source.set_index("timestamp")[demand_cols].interpolate(limit_direction="both").ffill().bfill()
        target_index = pd.date_range(start=date, periods=96, freq="15min")
        fallback = pd.DataFrame({"timestamp": target_index})
        for region in REGIONS:
            fallback[f"forecast_{region}_mw"] = source[f"demand_{region}_mw"].to_numpy()[:96]
        fallback["cndc_forecast_total_mw"] = fallback[forecast_cols].sum(axis=1)
        return fallback[["timestamp", *forecast_cols, "cndc_forecast_total_mw"]], (
            f"fallback_from_actual_demand_{source_date:%Y-%m-%d}"
        )
    raise ValueError(f"CNDC forecast is unavailable for {date:%Y-%m-%d} and no fallback demand profile was found.")


def date_range_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="D"))


def fetch_input_data(
    target_date: pd.Timestamp,
    lookback_days: int,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    start_date = target_date - pd.Timedelta(days=lookback_days)
    dates = date_range_days(start_date, target_date)

    demand_frames = []
    forecast_frames = []
    forecast_sources: dict[str, str] = {}

    for date in dates:
        try:
            forecast_frames.append(forecast_frame_for_day(date, cache_dir, refresh=refresh))
            forecast_sources[f"{date:%Y-%m-%d}"] = "cndc"
        except ValueError:
            if date != target_date:
                raise
            fallback, source = fallback_forecast_frame_for_day(date, cache_dir, refresh=refresh)
            forecast_frames.append(fallback)
            forecast_sources[f"{date:%Y-%m-%d}"] = source
        if date < target_date:
            demand_frames.append(demand_frame_for_day(date, cache_dir, refresh=refresh))

    actuals = pd.concat(demand_frames, ignore_index=True)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    return actuals, forecasts, forecast_sources


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    demand = df[TARGET]
    df["target_lag_96"] = demand.shift(96)
    df["target_lag_192"] = demand.shift(192)
    df["target_lag_672"] = demand.shift(672)

    history = demand.shift(1)
    for window in (96, 672):
        rolling = history.rolling(window=window, min_periods=window)
        df[f"rolling_mean_{window}"] = rolling.mean()
        df[f"rolling_std_{window}"] = rolling.std()
        df[f"rolling_min_{window}"] = rolling.min()
        df[f"rolling_max_{window}"] = rolling.max()

    forecast_error = df[TARGET] - df["cndc_forecast_total_mw"]
    df["forecast_error_lag_96"] = forecast_error.shift(96)
    df["forecast_error_lag_672"] = forecast_error.shift(672)

    df["safe_ramp_96"] = df["target_lag_96"] - df["target_lag_192"]
    df["safe_ramp_672"] = df["target_lag_672"] - demand.shift(1344)

    for region in REGIONS:
        series = df[f"demand_{region}_mw"]
        df[f"demand_{region}_mw_lag_96"] = series.shift(96)
        df[f"demand_{region}_mw_lag_672"] = series.shift(672)

    return df


def initialize_base_frame(actuals: pd.DataFrame, forecasts: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    end_ts = target_date + pd.Timedelta(hours=23, minutes=45)
    grid = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                forecasts["timestamp"].min(),
                end_ts,
                freq="15min",
            )
        }
    )
    base = (
        grid.merge(forecasts, on="timestamp", how="left")
        .merge(actuals, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    missing_forecasts = base["cndc_forecast_total_mw"].isna().sum()
    if missing_forecasts:
        raise ValueError(f"Missing {missing_forecasts} forecast rows. CNDC forecast data is incomplete.")
    return base


def repair_historical_actual_gaps(base: pd.DataFrame, latest_actual_ts: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    if pd.isna(latest_actual_ts):
        return base, 0

    base = base.copy()
    demand_cols = [TARGET, *[f"demand_{region}_mw" for region in REGIONS]]
    history_mask = base["timestamp"] <= latest_actual_ts
    before_missing = int(base.loc[history_mask, TARGET].isna().sum())

    repaired = (
        base.loc[history_mask, demand_cols]
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )
    base.loc[history_mask, demand_cols] = repaired
    after_missing = int(base.loc[history_mask, TARGET].isna().sum())
    return base, before_missing - after_missing


def set_predicted_regional_demands(base: pd.DataFrame, row_idx: int, prediction: float) -> None:
    total_forecast = base.at[row_idx, "cndc_forecast_total_mw"]
    if not total_forecast or math.isnan(total_forecast):
        shares = {region: 1 / len(REGIONS) for region in REGIONS}
    else:
        shares = {
            region: base.at[row_idx, f"forecast_{region}_mw"] / total_forecast
            for region in REGIONS
        }

    for region in REGIONS:
        base.at[row_idx, f"demand_{region}_mw"] = prediction * shares[region]


def predict_day(
    target_date: pd.Timestamp,
    model_path: Path,
    features_path: Path,
    cache_dir: Path,
    lookback_days: int,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    model = joblib.load(model_path)
    features = joblib.load(features_path)

    actuals, forecasts, forecast_sources = fetch_input_data(target_date, lookback_days, cache_dir, refresh=refresh)
    base = initialize_base_frame(actuals, forecasts, target_date)
    latest_actual_ts = actuals.dropna(subset=[TARGET])["timestamp"].max()
    base, repaired_history_rows = repair_historical_actual_gaps(base, latest_actual_ts)

    target_start = target_date
    target_end = target_date + pd.Timedelta(hours=23, minutes=45)

    predictions = []
    nowcast_count = 0
    target_count = 0

    prediction_mask = (base[TARGET].isna() & (base["timestamp"] > latest_actual_ts)) | (
        (base["timestamp"] >= target_start) & (base["timestamp"] <= target_end)
    )
    prediction_indices = base.index[prediction_mask]

    if len(prediction_indices) == 0:
        raise ValueError("Nothing to predict. Check target date and input data.")

    for row_idx in prediction_indices:
        ts = base.at[row_idx, "timestamp"]
        base = add_features(base)

        row = base.loc[[row_idx], features]
        missing = row.columns[row.isna().any()].tolist()
        if missing:
            raise ValueError(
                f"Cannot predict {ts}; missing required features {missing}. "
                f"Try increasing --lookback-days."
            )

        pred = float(model.predict(row)[0])
        base.at[row_idx, TARGET] = pred
        set_predicted_regional_demands(base, row_idx, pred)

        is_target = target_start <= ts <= target_end
        if is_target:
            target_count += 1
            predictions.append(
                {
                    "timestamp": ts,
                    "predicted_demand_mw": pred,
                    "cndc_forecast_total_mw": float(base.at[row_idx, "cndc_forecast_total_mw"]),
                    "is_recursive_fill": bool(pd.isna(actuals.loc[actuals["timestamp"].eq(ts), TARGET]).all())
                    if ts in set(actuals["timestamp"])
                    else True,
                }
            )
        else:
            nowcast_count += 1

    featured = add_features(base)
    prediction_df = pd.DataFrame(predictions)
    target_features = featured[(featured["timestamp"] >= target_start) & (featured["timestamp"] <= target_end)]

    metadata = {
        "target_date": str(target_date.date()),
        "lookback_days": lookback_days,
        "history_start": str(base["timestamp"].min()),
        "history_end": str((target_start - pd.Timedelta(minutes=15))),
        "target_start": str(target_start),
        "target_end": str(target_end),
        "target_predictions": target_count,
        "nowcast_rows_filled_before_target": nowcast_count,
        "historical_gap_rows_repaired": repaired_history_rows,
        "latest_real_actual_timestamp": str(latest_actual_ts),
        "forecast_sources": forecast_sources,
        "model_path": str(model_path),
        "features_path": str(features_path),
    }
    return prediction_df, target_features[["timestamp", *features]], featured, metadata


def output_paths(outdir: Path, target_date: pd.Timestamp) -> PredictionPaths:
    run_dir = outdir / f"prediction_{target_date:%Y-%m-%d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bdm_prediction_{target_date:%Y-%m-%d}"
    return PredictionPaths(
        run_dir=run_dir,
        prediction_csv=run_dir / f"{stem}.csv",
        prediction_json=run_dir / f"{stem}.json",
        features_csv=run_dir / f"{stem}_features.csv",
        history_csv=run_dir / f"{stem}_history_used.csv",
        chart_png=run_dir / f"{stem}.png",
        metadata_json=run_dir / f"{stem}_metadata.json",
    )


def save_chart(prediction_df: pd.DataFrame, path: Path, target_date: pd.Timestamp) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(14, 7), dpi=170)
    fig.patch.set_facecolor("#f8fbfc")
    ax.set_facecolor("#f8fbfc")

    ax.plot(
        prediction_df["timestamp"],
        prediction_df["cndc_forecast_total_mw"],
        color="#8b949e",
        linewidth=2.2,
        linestyle=(0, (4, 3)),
        label="CNDC forecast",
    )
    ax.plot(
        prediction_df["timestamp"],
        prediction_df["predicted_demand_mw"],
        color="#0b7285",
        linewidth=3.0,
        label="Bolivia Demand Model prediction",
    )

    peak = prediction_df.loc[prediction_df["predicted_demand_mw"].idxmax()]
    low = prediction_df.loc[prediction_df["predicted_demand_mw"].idxmin()]
    ax.scatter([peak["timestamp"]], [peak["predicted_demand_mw"]], color="#0b7285", s=55, zorder=4)
    ax.scatter([low["timestamp"]], [low["predicted_demand_mw"]], color="#d9480f", s=45, zorder=4)

    ax.set_title(f"Bolivia Demand Forecast - {target_date:%Y-%m-%d}", loc="left", fontsize=20, weight="bold")
    ax.text(
        0,
        1.015,
        f"Average {prediction_df['predicted_demand_mw'].mean():.1f} MW | "
        f"Peak {peak['predicted_demand_mw']:.1f} MW at {peak['timestamp']:%H:%M} | "
        f"Low {low['predicted_demand_mw']:.1f} MW at {low['timestamp']:%H:%M}",
        transform=ax.transAxes,
        color="#51636b",
        fontsize=11,
    )
    ax.set_ylabel("MW")
    ax.set_xlabel("Time")
    ax.grid(True, axis="y", color="#d9e2e7", linewidth=0.9)
    ax.grid(True, axis="x", color="#eef3f5", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    paths: PredictionPaths,
    prediction_df: pd.DataFrame,
    target_features: pd.DataFrame,
    featured_history: pd.DataFrame,
    metadata: dict,
    target_date: pd.Timestamp,
    make_chart: bool,
) -> None:
    prediction_df.to_csv(paths.prediction_csv, index=False)
    paths.prediction_json.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "predictions": prediction_df.assign(
                    timestamp=prediction_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
                ).to_dict(orient="records"),
            },
            indent=2,
        )
    )
    target_features.to_csv(paths.features_csv, index=False)
    featured_history.to_csv(paths.history_csv, index=False)
    paths.metadata_json.write_text(json.dumps(metadata, indent=2))
    if make_chart:
        try:
            save_chart(prediction_df, paths.chart_png, target_date)
        except Exception as exc:
            metadata["chart_error"] = str(exc)
            paths.metadata_json.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Bolivia Demand Model CNDC day-ahead demand predictor")
    parser.add_argument("--target-date", required=True, help="Date to predict, e.g. 2026-07-04")
    parser.add_argument("--model", default=Path("models/bolivia_demand_model.pkl"), type=Path)
    parser.add_argument("--features", default=Path("models/bolivia_demand_features.pkl"), type=Path)
    parser.add_argument("--lookback-days", default=21, type=int)
    parser.add_argument("--cache-dir", default=Path("data/cache"), type=Path)
    parser.add_argument("--outdir", default=Path("runs"), type=Path)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--chart", action="store_true", help="Also render a PNG chart for the prediction")
    parser.add_argument("--no-chart", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    target_date = parse_date(args.target_date)
    paths = output_paths(args.outdir, target_date)

    prediction_df, target_features, featured_history, metadata = predict_day(
        target_date=target_date,
        model_path=args.model,
        features_path=args.features,
        cache_dir=args.cache_dir,
        lookback_days=args.lookback_days,
        refresh=args.refresh_cache,
    )
    write_outputs(
        paths,
        prediction_df,
        target_features,
        featured_history,
        metadata,
        target_date,
        make_chart=args.chart and not args.no_chart,
    )

    peak = prediction_df.loc[prediction_df["predicted_demand_mw"].idxmax()]
    low = prediction_df.loc[prediction_df["predicted_demand_mw"].idxmin()]

    print("================================")
    print("Bolivia Demand Model Local Prediction Complete")
    print("================================")
    print(f"Target date:       {target_date:%Y-%m-%d}")
    print(f"Rows predicted:    {len(prediction_df)}")
    print(f"Average demand:    {prediction_df['predicted_demand_mw'].mean():.3f} MW")
    print(f"Peak demand:       {peak['predicted_demand_mw']:.3f} MW at {peak['timestamp']:%Y-%m-%d %H:%M}")
    print(f"Minimum demand:    {low['predicted_demand_mw']:.3f} MW at {low['timestamp']:%Y-%m-%d %H:%M}")
    print(f"Nowcast fills:     {metadata['nowcast_rows_filled_before_target']}")
    print(f"Latest real data:  {metadata['latest_real_actual_timestamp']}")
    print()
    print(f"CSV:               {paths.prediction_csv}")
    print(f"JSON:              {paths.prediction_json}")
    if args.chart and not args.no_chart and paths.chart_png.exists():
        print(f"Chart:             {paths.chart_png}")


if __name__ == "__main__":
    main()
