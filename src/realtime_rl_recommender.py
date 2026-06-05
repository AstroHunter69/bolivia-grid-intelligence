#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://cndcapi.cndc.bo"
TYPE_COSTS = {"EOL": 5.0, "SOL": 5.0, "BIO": 12.0, "HID": 18.0, "INT": 35.0, "TER": 55.0}
TYPE_LABELS = {"EOL": "Wind", "SOL": "Solar", "BIO": "Biomass", "HID": "Hydro", "INT": "Interconnection", "TER": "Thermal"}
GENERATION_TO_TYPE = {"eol": "EOL", "solar": "SOL", "bagazo": "BIO", "hidro": "HID", "termo": "TER"}


@dataclass
class Recommendation:
    action: float
    reason: str
    confidence: str


def cndc_date(day: pd.Timestamp) -> str:
    return day.strftime("%d.%m.%Y")


def get_json(endpoint: str, params: dict, cache_dir: Path, refresh: bool = False) -> list | dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = endpoint.strip("/").lower() + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    cache_file = cache_dir / f"{key}.json"
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
    time.sleep(0.15)
    return payload


def series_total(payload: list[dict], skip_prefix: str | None = None) -> pd.Series:
    series = []
    for record in payload:
        code = str(record.get("codigo") or "").strip().lower()
        if skip_prefix and code.startswith(skip_prefix):
            continue
        values = pd.Series(record.get("valores", []), dtype="float64").replace(-1, np.nan)
        if values.notna().any():
            series.append(values)
    if not series:
        return pd.Series(dtype="float64")
    return pd.concat(series, axis=1).sum(axis=1, min_count=1)


def latest_value(values: pd.Series) -> tuple[int, float]:
    clean = values.dropna()
    if clean.empty:
        raise ValueError("No real-time values are available from CNDC.")
    idx = int(clean.index[-1])
    return idx, float(clean.iloc[-1])


def generation_mix(payload: list[dict], interval_index: int) -> dict[str, float]:
    out = {k: 0.0 for k in TYPE_COSTS}
    for record in payload:
        code = str(record.get("codigo") or "").strip().lower()
        gen_type = GENERATION_TO_TYPE.get(code)
        values = record.get("valores", [])
        if not gen_type or interval_index >= len(values):
            continue
        value = values[interval_index]
        if value is not None and value >= 0:
            out[gen_type] += float(value)
    return out


def demand_forecast_next_hour(payload: list[dict], now_ts: pd.Timestamp, current_demand: float) -> list[dict]:
    total = series_total(payload)
    if total.empty:
        return [
            {"timestamp": (now_ts + pd.Timedelta(minutes=15 * step)).strftime("%Y-%m-%d %H:%M"), "forecast_mw": current_demand}
            for step in range(1, 5)
        ]
    hourly = pd.DataFrame(
        {
            "timestamp": pd.date_range(now_ts.normalize(), periods=len(total), freq="h"),
            "forecast_mw": total.to_numpy(),
        }
    ).set_index("timestamp")
    target_index = pd.date_range(now_ts.ceil("15min"), periods=5, freq="15min")[1:]
    expanded = hourly.reindex(hourly.index.union(target_index)).interpolate("linear").ffill().bfill()
    return [
        {"timestamp": ts.strftime("%Y-%m-%d %H:%M"), "forecast_mw": float(expanded.loc[ts, "forecast_mw"])}
        for ts in target_index
    ]


def bdm_prediction_frame(day: pd.Timestamp) -> pd.DataFrame | None:
    root = Path(__file__).resolve().parents[1]
    predictor_dir = root / "src"
    pred_path = (
        root
        / "runs"
        / "forecasts"
        / f"prediction_{day:%Y-%m-%d}"
        / f"bdm_prediction_{day:%Y-%m-%d}.csv"
    )
    metadata_path = pred_path.with_name(f"bdm_prediction_{day:%Y-%m-%d}_metadata.json")
    if not is_bdm_day_ahead_usable(day, metadata_path):
        subprocess.run(
            [
                sys.executable,
                "predict_demand.py",
                "--target-date",
                f"{day:%Y-%m-%d}",
                "--no-chart",
                "--outdir",
                str(root / "runs" / "forecasts"),
                "--cache-dir",
                str(root / "data" / "cache"),
            ],
            cwd=predictor_dir,
            check=True,
        )
    if not pred_path.exists() or not is_bdm_day_ahead_usable(day, metadata_path):
        return None
    pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
    return pred.set_index("timestamp").sort_index()


def is_bdm_day_ahead_usable(day: pd.Timestamp, metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        latest = pd.Timestamp(metadata.get("latest_real_actual_timestamp"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    if pd.isna(latest):
        return False
    previous_day = day - pd.Timedelta(days=1)
    minimum_latest = previous_day + pd.Timedelta(hours=20)
    return previous_day <= latest.normalize() < day and latest >= minimum_latest


def bdm_forecast_next_hour(day: pd.Timestamp, now_ts: pd.Timestamp) -> list[dict] | None:
    pred = bdm_prediction_frame(day)
    if pred is None:
        return None
    target_index = pd.date_range(now_ts.ceil("15min"), periods=5, freq="15min")[1:]
    rows = []
    for ts in target_index:
        if ts not in pred.index:
            return None
        rows.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "forecast_mw": float(pred.loc[ts, "predicted_demand_mw"]),
            }
        )
    return rows


def load_capacity_by_type(payload: list[dict]) -> dict[str, float]:
    capacity = {k: 0.0 for k in TYPE_COSTS}
    for record in payload:
        gen_type = str(record.get("descripcion") or "").strip().upper()
        if gen_type in capacity:
            value = record.get("valor")
            if value is not None and value >= 0:
                capacity[gen_type] += float(value)
    return capacity


def choose_policy_action(current_demand: float, forecast: list[dict], generation_total: float) -> Recommendation:
    forecast_peak = max(row["forecast_mw"] for row in forecast)
    ramp_pct = (forecast_peak - current_demand) / max(current_demand, 1.0)
    supply_gap_pct = (current_demand - generation_total) / max(current_demand, 1.0)
    if ramp_pct > 0.07 or supply_gap_pct > 0.04:
        return Recommendation(0.10, "Rising short-term demand or supply gap detected; hold a higher reserve target.", "medium")
    if ramp_pct > 0.03:
        return Recommendation(0.05, "Moderate 1-hour upward ramp; increase reserve target slightly.", "medium")
    if ramp_pct < -0.05 and supply_gap_pct < -0.04:
        return Recommendation(-0.05, "Demand forecast is easing and generation is above demand; reduce reserve target.", "low")
    return Recommendation(0.0, "Short-term demand looks stable; keep neutral reserve target.", "medium")


def dispatch_by_type(target_mw: float, current_mix: dict[str, float], capacity: dict[str, float]) -> list[dict]:
    remaining = target_mw
    rows = []
    for gen_type in sorted(TYPE_COSTS, key=lambda k: TYPE_COSTS[k]):
        if gen_type in {"EOL", "SOL", "BIO"}:
            available = current_mix.get(gen_type, 0.0)
        elif gen_type == "INT":
            available = max(current_mix.get(gen_type, 0.0), 0.0)
        else:
            available = max(capacity.get(gen_type, 0.0), current_mix.get(gen_type, 0.0))
        dispatch = min(max(available, 0.0), max(remaining, 0.0))
        remaining -= dispatch
        rows.append(
            {
                "type": gen_type,
                "label": TYPE_LABELS.get(gen_type, gen_type),
                "current_mw": round(float(current_mix.get(gen_type, 0.0)), 3),
                "recommended_mw": round(float(dispatch), 3),
                "delta_mw": round(float(dispatch - current_mix.get(gen_type, 0.0)), 3),
                "cost_usd_mwh": TYPE_COSTS[gen_type],
            }
        )
    return rows


def make_dashboard(payload: dict, out: Path) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real-Time RL Dispatch</title>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f4f7f9; color:#17212b; }
    header { background:white; border-bottom:1px solid #d9e3e9; padding:22px 28px 16px; }
    h1 { margin:0; font-size:27px; }
    main { max-width:1180px; margin:0 auto; padding:20px 28px 34px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)); gap:12px; margin-bottom:16px; }
    .metric,.panel { background:white; border:1px solid #d9e3e9; border-radius:8px; padding:16px; }
    .label { color:#60717d; font-size:12px; }
    .value { font-size:22px; font-weight:750; margin-top:4px; }
    .grid { display:grid; grid-template-columns:1.1fr .9fr; gap:16px; align-items:start; }
    svg { width:100%; height:280px; display:block; }
    #curve { height:330px; }
    .controls { display:flex; gap:12px; align-items:center; margin-top:10px; }
    input[type=range] { width:100%; }
    .time { min-width:150px; color:#0b7285; font-weight:750; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; color:#60717d; font-size:12px; margin-top:8px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { border-bottom:1px solid #edf2f5; padding:7px 6px; text-align:right; }
    th:first-child,td:first-child { text-align:left; }
    .note { color:#60717d; font-size:12px; line-height:1.45; margin-top:10px; }
    @media (max-width:900px) { .grid,.metrics { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header><h1>Real-Time RL Dispatch Recommender</h1></header>
  <main>
    <div class="metrics" id="metrics"></div>
    <section class="panel" style="margin-bottom:16px">
      <h2>Real Demand, Bolivia Demand Model Day Forecast, and RL Next-Hour Correction</h2>
      <svg id="curve" viewBox="0 0 900 330" preserveAspectRatio="none"></svg>
      <div class="controls"><input id="slider" type="range" min="0" max="0" value="0"><div class="time" id="timeLabel"></div></div>
      <div class="legend" id="curveLegend"></div>
    </section>
    <div class="grid">
      <section class="panel"><h2>Recommended Mix</h2><svg id="bars" viewBox="0 0 720 280" preserveAspectRatio="none"></svg><div class="note" id="note"></div></section>
      <section class="panel"><h2>Technology Recommendation</h2><table id="table"></table></section>
    </div>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    function fmt(n,d=1){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d});}
    const metrics=[
      ["Snapshot", data.snapshot_time],
      ["Current demand", fmt(data.current_demand_mw)+" MW"],
      ["1h forecast peak", fmt(data.forecast_peak_mw)+" MW"],
      ["RL reserve action", (data.action*100).toFixed(1)+"%"]
    ];
    document.getElementById("metrics").innerHTML=metrics.map(m=>`<div class="metric"><div class="label">${m[0]}</div><div class="value">${m[1]}</div></div>`).join("");
    const curve=data.curve;
    const slider=document.getElementById("slider");
    let selected=data.default_curve_index || 0;
    slider.max=curve.length-1;
    slider.value=selected;
    function pathFor(key,x0,y0,w,h,minY,maxY){
      let out="", drawing=false;
      curve.forEach((r,i)=>{
        if(r[key]==null){ drawing=false; return; }
        const x=x0+(curve.length>1?i/(curve.length-1):0)*w;
        const y=y0+h-(r[key]-minY)/(maxY-minY)*h;
        out += `${drawing?"L":"M"}${x.toFixed(1)},${y.toFixed(1)}`;
        drawing=true;
      });
      return out;
    }
    function drawCurve(){
      const series=[
        ["actual_mw","#17212b","Real demand","Actual"],
        ["forecast_mw","#2b8a3e","Bolivia Demand Model forecast through next hour","Bolivia Demand Model"],
        ["rl_target_mw","#0b7285","RL next-hour correction","RL target"]
      ];
      const vals=curve.flatMap(r=>series.map(s=>r[s[0]]).filter(v=>v!=null));
      const minY=Math.min(...vals)*0.94, maxY=Math.max(...vals)*1.04;
      const x0=54,y0=24,w=810,h=220;
      const sx=x0+(curve.length>1?selected/(curve.length-1):0)*w;
      const p=curve[selected];
      document.getElementById("curve").innerHTML=`
        <rect x="0" y="0" width="900" height="330" fill="#fff"/>
        ${[0,1,2,3,4].map(i=>`<line x1="${x0}" x2="${x0+w}" y1="${y0+i*h/4}" y2="${y0+i*h/4}" stroke="#e9eef2"/>`).join("")}
        <path d="${pathFor("forecast_mw",x0,y0,w,h,minY,maxY)}" fill="none" stroke="#2b8a3e" stroke-width="3"/>
        <path d="${pathFor("rl_target_mw",x0,y0,w,h,minY,maxY)}" fill="none" stroke="#0b7285" stroke-width="3" stroke-dasharray="6 4"/>
        <path d="${pathFor("actual_mw",x0,y0,w,h,minY,maxY)}" fill="none" stroke="#17212b" stroke-width="3"/>
        <line x1="${sx}" x2="${sx}" y1="${y0}" y2="${y0+h}" stroke="#0b7285" stroke-width="1.5" opacity="0.55"/>
        <rect x="54" y="260" width="810" height="48" rx="5" fill="#fff" stroke="#d9e3e9"/>
        ${series.map((s,i)=>`<circle cx="${74+i*250}" cy="288" r="3.2" fill="${s[1]}"/><text x="${86+i*250}" y="292" font-size="12" fill="#17212b">${s[3]}: ${p[s[0]]==null?"n/a":fmt(p[s[0]])+" MW"}</text>`).join("")}
        <text x="8" y="${y0+4}" font-size="11" fill="#60717d">${fmt(maxY,0)} MW</text>
        <text x="8" y="${y0+h}" font-size="11" fill="#60717d">${fmt(minY,0)} MW</text>
      `;
      document.getElementById("timeLabel").textContent=p.timestamp;
      document.getElementById("curveLegend").innerHTML=series.map(s=>`<span><span class="dot" style="background:${s[1]}"></span>${s[2]}</span>`).join("");
    }
    slider.addEventListener("input",()=>{selected=Number(slider.value); drawCurve();});
    drawCurve();
    const rows=data.dispatch;
    const max=Math.max(...rows.flatMap(r=>[r.current_mw,r.recommended_mw]),1);
    document.getElementById("bars").innerHTML=`
      <rect x="0" y="0" width="720" height="280" fill="#fff"/>
      ${rows.map((r,i)=>{
        const y=28+i*38, cW=r.current_mw/max*220, oW=r.recommended_mw/max*220;
        return `<text x="0" y="${y+17}" font-size="12" fill="#34434d">${r.label}</text>
          <rect x="130" y="${y}" width="${cW}" height="12" fill="#adb5bd"><title>Current ${fmt(r.current_mw)} MW</title></rect>
          <rect x="130" y="${y+16}" width="${oW}" height="12" fill="#0b7285"><title>Recommended ${fmt(r.recommended_mw)} MW</title></rect>
          <text x="370" y="${y+24}" font-size="12" fill="${r.delta_mw>=0?'#2b8a3e':'#c92a2a'}">${r.delta_mw>=0?'+':''}${fmt(r.delta_mw)} MW</text>`;
      }).join("")}
      <rect x="520" y="28" width="12" height="12" fill="#adb5bd"/><text x="540" y="39" font-size="12" fill="#60717d">Current</text>
      <rect x="520" y="50" width="12" height="12" fill="#0b7285"/><text x="540" y="61" font-size="12" fill="#60717d">RL recommendation</text>
    `;
    document.getElementById("table").innerHTML=`<tr><th>Type</th><th>Current</th><th>RL target</th><th>Delta</th></tr>`+
      rows.map(r=>`<tr><td>${r.label}</td><td>${fmt(r.current_mw)}</td><td>${fmt(r.recommended_mw)}</td><td style="color:${r.delta_mw>=0?'#2b8a3e':'#c92a2a'}">${r.delta_mw>=0?'+':''}${fmt(r.delta_mw)}</td></tr>`).join("");
    document.getElementById("note").textContent = data.reason + " This is a decision-support simulation, not an operational dispatch instruction.";
  </script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload))
    out.write_text(html)


def run_recommender(args: argparse.Namespace) -> Path:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz="America/La_Paz").tz_localize(None).normalize()
    cache_dir = args.cache_dir

    demand_payload = get_json("/WebApi", {"code": 1, "Fecha": cndc_date(today)}, cache_dir, args.refresh)
    generation_payload = get_json("/WebApi", {"code": 0, "Fecha": cndc_date(today)}, cache_dir, args.refresh)
    forecast_payload = get_json("/WebApi", {"code": 4, "Fecha": cndc_date(today)}, cache_dir, args.refresh)
    capacity_payload = get_json("/WebNoDate", {"code": 2}, cache_dir, args.refresh)

    demand = series_total(demand_payload, skip_prefix="prev")
    interval_idx, current_demand = latest_value(demand)
    snapshot_ts = today + pd.Timedelta(minutes=15 * interval_idx)
    current_mix = generation_mix(generation_payload, interval_idx)
    generation_total = sum(current_mix.values())
    bdm_day = bdm_prediction_frame(today)
    bdm_forecast = bdm_forecast_next_hour(today, snapshot_ts)
    forecast = bdm_forecast or demand_forecast_next_hour(forecast_payload, snapshot_ts, current_demand)
    forecast_source = "bdm" if bdm_forecast else "cndc_or_flat_fallback"
    capacity = load_capacity_by_type(capacity_payload)

    rec = choose_policy_action(current_demand, forecast, generation_total)
    target_mw = max(current_demand, max(row["forecast_mw"] for row in forecast)) * (1.0 + rec.action)
    dispatch = dispatch_by_type(target_mw, current_mix, capacity)
    current_cost = sum(row["current_mw"] * row["cost_usd_mwh"] * 0.25 for row in dispatch)
    recommended_cost = sum(row["recommended_mw"] * row["cost_usd_mwh"] * 0.25 for row in dispatch)
    forecast_by_ts = {
        pd.Timestamp(row["timestamp"]): float(row["forecast_mw"])
        for row in forecast
    }
    curve_index = pd.date_range(today, snapshot_ts + pd.Timedelta(hours=1), freq="15min")
    curve = []
    default_curve_index = 0
    for pos, ts in enumerate(curve_index):
        actual_mw = None
        forecast_mw = None
        rl_target_mw = None
        demand_idx = int((ts - today).total_seconds() // (15 * 60))
        if 0 <= demand_idx < len(demand):
            value = demand.iloc[demand_idx]
            if pd.notna(value) and ts <= snapshot_ts:
                actual_mw = round(float(value), 3)
        if bdm_day is not None and ts in bdm_day.index:
            forecast_mw = round(float(bdm_day.loc[ts, "predicted_demand_mw"]), 3)
        elif ts in forecast_by_ts:
            forecast_mw = round(forecast_by_ts[ts], 3)
        if ts == snapshot_ts:
            default_curve_index = pos
            rl_target_mw = round(current_demand * (1.0 + rec.action), 3)
        elif ts > snapshot_ts and ts in forecast_by_ts:
            rl_target_mw = round(forecast_by_ts[ts] * (1.0 + rec.action), 3)
        curve.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "actual_mw": actual_mw,
                "forecast_mw": forecast_mw,
                "rl_target_mw": rl_target_mw,
            }
        )

    report = {
        "snapshot_time": snapshot_ts.strftime("%Y-%m-%d %H:%M"),
        "current_demand_mw": round(current_demand, 3),
        "current_generation_mw": round(generation_total, 3),
        "forecast": forecast,
        "forecast_source": forecast_source,
        "curve": curve,
        "default_curve_index": default_curve_index,
        "forecast_peak_mw": round(max(row["forecast_mw"] for row in forecast), 3),
        "action": rec.action,
        "reason": rec.reason,
        "confidence": rec.confidence,
        "target_mw": round(target_mw, 3),
        "dispatch": dispatch,
        "current_cost_proxy_usd_15min": round(current_cost, 3),
        "recommended_cost_proxy_usd_15min": round(recommended_cost, 3),
        "note": "Research prototype: RL-style reserve policy plus constrained technology dispatch proxy.",
    }
    (outdir / "realtime_rl_recommendation.json").write_text(json.dumps(report, indent=2))
    dashboard = outdir / "realtime_rl_dashboard.html"
    make_dashboard(report, dashboard)
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time RL-style dispatch recommender")
    parser.add_argument("--date", help="YYYY-MM-DD. Defaults to current Bolivia date.")
    parser.add_argument("--outdir", type=Path, default=Path("runs/realtime_rl_latest"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    dashboard = run_recommender(args)
    print("================================")
    print("Real-Time RL Dispatch Recommendation")
    print("================================")
    print(f"Saved dashboard: {dashboard}")


if __name__ == "__main__":
    main()
