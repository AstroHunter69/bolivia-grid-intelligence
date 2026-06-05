#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://cndcapi.cndc.bo"

WEBAPI_DATASETS = {
    0: {
        "name": "generation",
        "resolution": "15min",
        "wide_output": "generation_15min.csv",
        "long_output": "generation_15min_long.csv",
        "value_name": "value_mw",
        "total_components": ["termo", "hidro", "solar", "eol", "bagazo"],
    },
    1: {
        "name": "demand",
        "resolution": "15min",
        "wide_output": "demand_15min.csv",
        "long_output": "demand_15min_long.csv",
        "value_name": "demand_mw",
        "skip_prefixes": ["prev"],
    },
    3: {
        "name": "marginal_cost_forecast",
        "resolution": "hourly",
        "wide_output": "marginal_cost_forecast_hourly.csv",
        "long_output": "marginal_cost_forecast_hourly_long.csv",
        "value_name": "cost_usd_mwh",
    },
    4: {
        "name": "demand_forecast",
        "resolution": "hourly",
        "wide_output": "demand_forecast_hourly.csv",
        "long_output": "demand_forecast_hourly_long.csv",
        "value_name": "forecast_mw",
    },
    6: {
        "name": "injected_energy",
        "resolution": "hourly",
        "wide_output": "injected_energy_hourly.csv",
        "long_output": "injected_energy_hourly_long.csv",
        "value_name": "injected_mw",
    },
}

STATIC_DATASETS = {
    1: "effective_capacity_static.csv",
    2: "capacity_detail_static.csv",
    3: "transmission_lines_static.csv",
}

TOKEN_DATASETS = {
    0: "token_export_argentina_snapshot.csv",
    1: "token_import_argentina_snapshot.csv",
}


def clean_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    for old, new in {
        " ": "_",
        "-": "_",
        "/": "_",
        ".": "",
        "(": "",
        ")": "",
        "*": "",
    }.items():
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def cndc_date(date: pd.Timestamp) -> str:
    return date.strftime("%d.%m.%Y")


def date_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"))


def get_json(
    endpoint: str,
    params: dict,
    cache_dir: Path,
    cache_key: str,
    refresh: bool,
    sleep_seconds: float,
) -> list | dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
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
    time.sleep(sleep_seconds)
    return payload


def timestamps_for_day(date: pd.Timestamp, n_values: int, resolution: str) -> pd.DatetimeIndex:
    if resolution == "hourly":
        return pd.date_range(start=date, periods=n_values, freq="h")
    return pd.date_range(start=date, periods=n_values, freq="15min")


def fetch_webapi_day(
    code: int,
    date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    cfg = WEBAPI_DATASETS[code]
    payload = get_json(
        "/WebApi",
        {"code": code, "Fecha": cndc_date(date)},
        cache_dir,
        f"webapi_code{code}_{date:%Y-%m-%d}",
        refresh,
        sleep_seconds,
    )

    rows = []
    for record in payload:
        raw_code = str(record.get("codigo", "")).strip()
        if any(clean_name(raw_code).startswith(prefix) for prefix in cfg.get("skip_prefixes", [])):
            continue
        values = record.get("valores", [])
        timestamps = timestamps_for_day(date, len(values), cfg["resolution"])
        feature = clean_name(raw_code)

        for ts, value in zip(timestamps, values):
            rows.append(
                {
                    "timestamp": ts,
                    "date": date.date().isoformat(),
                    "source_code": code,
                    "dataset": cfg["name"],
                    "series_code": raw_code,
                    "series": feature,
                    cfg["value_name"]: np.nan if value == -1 else value,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "date",
            "source_code",
            "dataset",
            "series_code",
            "series",
            cfg["value_name"],
        ],
    )


def webapi_wide(long_df: pd.DataFrame, value_name: str, cfg: dict) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["timestamp", "total"])
    timestamps = pd.DataFrame({"timestamp": sorted(long_df["timestamp"].dropna().unique())})
    wide = (
        long_df.pivot_table(
            index="timestamp",
            columns="series",
            values=value_name,
            aggfunc="last",
            dropna=False,
        )
        .reset_index()
        .sort_values("timestamp")
    )
    wide = timestamps.merge(wide, on="timestamp", how="left")
    wide.columns.name = None
    numeric_cols = [c for c in wide.columns if c != "timestamp"]
    total_components = [c for c in cfg.get("total_components", []) if c in wide.columns]
    if "tot" in wide.columns:
        wide["total"] = wide["tot"]
    elif total_components:
        wide["total"] = wide[total_components].sum(axis=1, min_count=1)
    else:
        wide["total"] = wide[numeric_cols].sum(axis=1, min_count=1)
    return wide


def fetch_static(code: int, cache_dir: Path, refresh: bool, sleep_seconds: float) -> pd.DataFrame:
    payload = get_json(
        "/WebNoDate",
        {"code": code},
        cache_dir,
        f"webnodate_code{code}",
        refresh,
        sleep_seconds,
    )
    rows = []
    for record in payload:
        rows.append(
            {
                "title": record.get("titulo"),
                "description": record.get("descripcion"),
                "value": np.nan if record.get("valor") == -1 else record.get("valor"),
                "timestamp": pd.to_datetime(record.get("fecha"), errors="coerce"),
                "code": code,
            }
        )
    return pd.DataFrame(rows)


def fetch_token_snapshot(code: int, cache_dir: Path, refresh: bool, sleep_seconds: float) -> pd.DataFrame:
    payload = get_json(
        "/WebApiToken",
        {"code": code},
        cache_dir,
        f"webapitoken_code{code}",
        refresh,
        sleep_seconds,
    )
    rows = []
    for record in payload:
        rows.append(
            {
                "title": record.get("titulo"),
                "description": record.get("descripcion"),
                "value": np.nan if record.get("valor") == -1 else record.get("valor"),
                "timestamp": pd.to_datetime(record.get("fecha"), errors="coerce"),
                "code": code,
            }
        )
    return pd.DataFrame(rows, columns=["title", "description", "value", "timestamp", "code"])


def parse_intervals(value: str) -> list[int]:
    if value == "all":
        return list(range(96))
    intervals = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            intervals.extend(range(int(start), int(end) + 1))
        else:
            intervals.append(int(part))
    invalid = [i for i in intervals if i < 0 or i > 95]
    if invalid:
        raise ValueError(f"Invalid plant intervals: {invalid}. Expected 0..95.")
    return sorted(set(intervals))


def fetch_plant_dispatch_interval(
    date: pd.Timestamp,
    interval: int,
    cache_dir: Path,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    payload = get_json(
        "/WebApiGetDatos",
        {"fecha": cndc_date(date), "intervalo": interval},
        cache_dir,
        f"webapigetdatos_{date:%Y-%m-%d}_interval{interval:02d}",
        refresh,
        sleep_seconds,
    )

    rows = []
    for record in payload:
        time_text = str(record.get("fecha", "")).strip()
        timestamp = pd.to_datetime(f"{date.date()} {time_text}", errors="coerce")
        rows.append(
            {
                "timestamp": timestamp,
                "date": date.date().isoformat(),
                "interval": interval,
                "plant_code": record.get("codigo"),
                "generation_type": record.get("gen"),
                "value_mw": np.nan if record.get("valor") == -1 else record.get("valor"),
            }
        )
    return pd.DataFrame(rows)


def generation_by_type(plant_df: pd.DataFrame) -> pd.DataFrame:
    if plant_df.empty:
        return pd.DataFrame()
    return (
        plant_df.pivot_table(
            index="timestamp",
            columns="generation_type",
            values="value_mw",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values("timestamp")
    )


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and normalize CNDC energy-system data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--outdir", default=Path("data/cndc_energy_system"), type=Path)
    parser.add_argument("--cache-dir", default=Path("data/cache"), type=Path)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sleep-seconds", default=0.15, type=float)
    parser.add_argument("--include-plant-dispatch", action="store_true")
    parser.add_argument(
        "--plant-intervals",
        default="all",
        help="Plant intervals to fetch, e.g. all, 0, 0-3, or 0,24,48,72",
    )
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-token", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    dates = date_range(args.start_date, args.end_date)

    manifest = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "outputs": [],
        "notes": [],
    }

    for code, cfg in WEBAPI_DATASETS.items():
        frames = [
            fetch_webapi_day(code, day, args.cache_dir, args.refresh_cache, args.sleep_seconds)
            for day in dates
        ]
        long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        wide_df = webapi_wide(long_df, cfg["value_name"], cfg)

        long_path = args.outdir / cfg["long_output"]
        wide_path = args.outdir / cfg["wide_output"]
        write_df(long_df, long_path)
        write_df(wide_df, wide_path)
        manifest["outputs"].extend([str(long_path), str(wide_path)])

    if args.include_plant_dispatch:
        intervals = parse_intervals(args.plant_intervals)
        frames = []
        for day in dates:
            for interval in intervals:
                frames.append(
                    fetch_plant_dispatch_interval(
                        day,
                        interval,
                        args.cache_dir,
                        args.refresh_cache,
                        args.sleep_seconds,
                    )
                )

        plant_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        plant_path = args.outdir / "plant_dispatch_15min_long.csv"
        by_type_path = args.outdir / "plant_dispatch_by_type_15min.csv"
        write_df(plant_df, plant_path)
        write_df(generation_by_type(plant_df), by_type_path)
        manifest["outputs"].extend([str(plant_path), str(by_type_path)])
    else:
        manifest["notes"].append("Plant dispatch not fetched. Use --include-plant-dispatch to fetch WebApiGetDatos.")

    if not args.skip_static:
        for code, filename in STATIC_DATASETS.items():
            df = fetch_static(code, args.cache_dir, args.refresh_cache, args.sleep_seconds)
            path = args.outdir / filename
            write_df(df, path)
            manifest["outputs"].append(str(path))

    if not args.skip_token:
        for code, filename in TOKEN_DATASETS.items():
            df = fetch_token_snapshot(code, args.cache_dir, args.refresh_cache, args.sleep_seconds)
            path = args.outdir / filename
            write_df(df, path)
            manifest["outputs"].append(str(path))
        manifest["notes"].append("WebApiToken appears to be live snapshot data, not historical data.")

    manifest_path = args.outdir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("================================")
    print("CNDC Energy Data Collection")
    print("================================")
    print(f"Dates:       {args.start_date} to {args.end_date}")
    print(f"Output dir:  {args.outdir}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Files:       {len(manifest['outputs'])}")
    print()
    for path in manifest["outputs"]:
        print(path)
    if manifest["notes"]:
        print()
        print("Notes:")
        for note in manifest["notes"]:
            print(f"- {note}")


if __name__ == "__main__":
    main()
