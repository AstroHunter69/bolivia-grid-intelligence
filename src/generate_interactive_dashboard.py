#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TYPE_STYLE = {
    "HID": {"label": "Hydro", "color": "#1971c2", "anchor": (42, 47)},
    "TER": {"label": "Thermal", "color": "#e8590c", "anchor": (63, 58)},
    "BIO": {"label": "Biomass", "color": "#2b8a3e", "anchor": (66, 50)},
    "EOL": {"label": "Wind", "color": "#15aabf", "anchor": (50, 38)},
    "SOL": {"label": "Solar", "color": "#f59f00", "anchor": (56, 44)},
    "INT": {"label": "Interconnection", "color": "#7048e8", "anchor": (34, 68)},
    "UNK": {"label": "Other", "color": "#495057", "anchor": (50, 55)},
}

TYPE_ICON = {
    "HID": "water",
    "TER": "thermal",
    "BIO": "leaf",
    "EOL": "wind",
    "SOL": "solar",
    "INT": "grid",
    "UNK": "dot",
}

PLANT_REGION_ANCHORS = [
    (["WARNES", "SANTA CRUZ", "GUARACACHI", "SCZ", "EL DORADO", "SAN JULIAN"], 66, 53, "Santa Cruz"),
    (["ENTRE RIOS", "ENTRE", "ERI", "BULO", "CARRASCO", "CORANI", "MISICUNI", "QOLLPANA"], 50, 54, "Cochabamba"),
    (["ZONGO", "TAQUESI", "EL ALTO"], 36, 38, "La Paz"),
    (["UYUNI", "YUNCHARA", "YUNCHARÁ"], 45, 76, "Potosi/Tarija"),
    (["ORURO"], 43, 56, "Oruro"),
    (["SAN JACINTO"], 53, 84, "Tarija"),
    (["ARANJUEZ", "DEL SUR"], 54, 68, "Chuquisaca"),
    (["MOXOS"], 56, 30, "Beni"),
]


def plant_region(code: str) -> str:
    upper = code.upper()
    for keys, _, _, region in PLANT_REGION_ANCHORS:
        if any(key in upper for key in keys):
            return region
    if code == "AGG_SOL":
        return "Solar fleet"
    if code == "AGG_EOL":
        return "Wind fleet"
    return "Approx. regional cluster"


def stable_jitter(code: str, span: float = 10.0) -> tuple[float, float]:
    digest = hashlib.sha1(code.encode()).hexdigest()
    a = int(digest[:4], 16) / 0xFFFF - 0.5
    b = int(digest[4:8], 16) / 0xFFFF - 0.5
    return a * span, b * span


def plant_position(code: str, gen_type: str) -> tuple[float, float]:
    fixed = {
        "AGG_SOL": (43, 70),
        "AGG_EOL": (62, 46),
    }
    if code in fixed:
        return fixed[code]
    upper = code.upper()
    for keys, x, y, _ in PLANT_REGION_ANCHORS:
        if any(key in upper for key in keys):
            jx, jy = stable_jitter(code, span=5.5)
            return round(min(max(x + jx, 20), 78), 2), round(min(max(y + jy, 18), 86), 2)
    anchor = TYPE_STYLE.get(gen_type, TYPE_STYLE["UNK"])["anchor"]
    jx, jy = stable_jitter(code, span=12.0)
    x = min(max(anchor[0] + jx, 20), 78)
    y = min(max(anchor[1] + jy, 18), 82)
    return round(x, 2), round(y, 2)


def load_cndc_forecast(data_dir: Path | None) -> pd.DataFrame | None:
    if not data_dir:
        return None
    path = data_dir / "demand_forecast_hourly.csv"
    if not path.exists():
        return None
    fc = pd.read_csv(path, parse_dates=["timestamp"])
    if "total" not in fc.columns:
        return None
    fc = fc[["timestamp", "total"]].rename(columns={"total": "forecast_demand_mw"})
    return fc


def load_model_prediction(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    pred = pd.read_csv(path, parse_dates=["timestamp"])
    if "predicted_demand_mw" not in pred.columns:
        return None
    return pred[["timestamp", "predicted_demand_mw"]].rename(columns={"predicted_demand_mw": "bdm_prediction_mw"})


def build_payload(run_dir: Path, data_dir: Path | None, model_prediction: Path | None) -> dict:
    interval = pd.read_csv(run_dir / "dispatch_interval_summary.csv", parse_dates=["timestamp"])
    plants = pd.read_csv(run_dir / "plant_dispatch_comparison_long.csv", parse_dates=["timestamp"])
    by_type = pd.read_csv(run_dir / "dispatch_by_type_summary.csv")
    report = json.loads((run_dir / "optimization_report.json").read_text())

    forecast = load_cndc_forecast(data_dir)
    if forecast is not None:
        interval = interval.merge(forecast, on="timestamp", how="left")
        interval["forecast_demand_mw"] = interval["forecast_demand_mw"].ffill().bfill()
    else:
        interval["forecast_demand_mw"] = None

    model_pred = load_model_prediction(model_prediction)
    if model_pred is not None:
        interval = interval.merge(model_pred, on="timestamp", how="left")
    else:
        interval["bdm_prediction_mw"] = None

    plants["delta_mw"] = plants["optimized_dispatch_mw"] - plants["actual_dispatch_mw"]
    plant_meta = {}
    for _, row in plants[["plant_code", "generation_type"]].drop_duplicates().iterrows():
        x, y = plant_position(str(row["plant_code"]), str(row["generation_type"]))
        style = TYPE_STYLE.get(str(row["generation_type"]), TYPE_STYLE["UNK"])
        plant_meta[str(row["plant_code"])] = {
            "x": x,
            "y": y,
            "region": plant_region(str(row["plant_code"])),
            "type": str(row["generation_type"]),
            "label": style["label"],
            "color": style["color"],
        }

    time_payload = []
    for ts, group in plants.groupby("timestamp"):
        group = group.copy()
        group = group[(group["actual_dispatch_mw"] > 0.01) | (group["optimized_dispatch_mw"] > 0.01)]
        rows = []
        for _, row in group.iterrows():
            code = str(row["plant_code"])
            meta = plant_meta[code]
            rows.append(
                {
                    "code": code,
                    "type": meta["type"],
                    "label": meta["label"],
                    "color": meta["color"],
                    "icon": TYPE_ICON.get(meta["type"], "dot"),
                    "x": meta["x"],
                    "y": meta["y"],
                    "region": meta["region"],
                    "actual": round(float(row["actual_dispatch_mw"]), 3),
                    "optimized": round(float(row["optimized_dispatch_mw"]), 3),
                    "delta": round(float(row["delta_mw"]), 3),
                    "cost": round(float(row["cost_usd_mwh"]), 2),
                }
            )
        time_payload.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M"), "plants": rows})

    interval_payload = []
    for _, row in interval.iterrows():
        interval_payload.append(
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "demand": round(float(row["demand_mw"]), 3),
                "forecast": None
                if pd.isna(row.get("forecast_demand_mw"))
                else round(float(row["forecast_demand_mw"]), 3),
                "bdm": None
                if pd.isna(row.get("bdm_prediction_mw"))
                else round(float(row["bdm_prediction_mw"]), 3),
                "actual_dispatch": round(float(row["actual_dispatch_mw"]), 3),
                "optimized_dispatch": round(float(row["optimized_dispatch_mw"]), 3),
                "saving": round(float(row["cost_proxy_saving_usd"]), 3),
                "actual_renewable": round(float(row["actual_renewable_share"]) * 100, 2),
                "optimized_renewable": round(float(row["optimized_renewable_share"]) * 100, 2),
            }
        )

    return {
        "report": report,
        "intervals": interval_payload,
        "plantsByTime": time_payload,
        "byType": by_type.to_dict(orient="records"),
        "types": TYPE_STYLE,
        "hasBdm": bool(interval["bdm_prediction_mw"].notna().any()),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bolivia Grid AI Dashboard</title>
  <style>
    :root { --bg:#f4f7f9; --ink:#17212b; --muted:#60717d; --line:#d9e3e9; --blue:#0b7285; --green:#2b8a3e; --orange:#e8590c; --red:#c92a2a; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
    header { background:white; border-bottom:1px solid var(--line); padding:20px 28px 14px; position:sticky; top:0; z-index:5; }
    h1 { margin:0; font-size:27px; letter-spacing:0; }
    .subtitle { margin-top:6px; color:var(--muted); max-width:1120px; line-height:1.4; }
    main { padding:18px 28px 34px; max-width:1420px; margin:0 auto; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; }
    .tab { border:1px solid var(--line); background:#f8fbfc; color:#25313a; border-radius:6px; padding:8px 12px; font-weight:700; cursor:pointer; }
    .tab.active { background:var(--blue); color:white; border-color:var(--blue); }
    body[data-tab="prediction"] .kpis { display:none; }
    .view { display:none; }
    .view.active { display:block; }
    .kpis { display:grid; grid-template-columns:repeat(5,minmax(160px,1fr)); gap:12px; margin-bottom:16px; }
    .kpi { background:white; border:1px solid var(--line); border-radius:8px; padding:13px 14px; }
    .kpi .label { color:var(--muted); font-size:12px; line-height:1.25; }
    .kpi .value { font-size:21px; font-weight:750; margin-top:5px; }
    .panel { background:white; border:1px solid var(--line); border-radius:8px; padding:16px; }
    .panel h2 { margin:0 0 10px; font-size:18px; }
    .grid2 { display:grid; grid-template-columns:1.15fr 0.85fr; gap:16px; align-items:start; }
    .grid3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }
    svg.chart { width:100%; height:372px; display:block; }
    .mapWrap { position:relative; width:min(720px,100%); aspect-ratio:1280/1352; margin:0 auto; background:#eef4f5; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    #mapSvg { width:100%; height:100%; display:block; }
    #mixShiftChart { width:100%; height:220px; display:block; margin-bottom:12px; }
    .controls { display:flex; gap:12px; align-items:center; margin:12px 0 0; }
    .datebar { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-bottom:14px; }
    .datebar label { display:grid; gap:4px; font-size:12px; color:var(--muted); font-weight:650; }
    .datebar input { height:34px; border:1px solid #b8c4ca; border-radius:6px; padding:0 9px; background:white; }
    .datebar button { height:34px; border:0; border-radius:6px; padding:0 12px; background:var(--blue); color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.62; cursor:wait; }
    input[type=range] { width:100%; }
    .time { min-width:150px; font-weight:750; color:var(--blue); }
    .legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; color:var(--muted); font-size:12px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    .switches { display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 10px; color:var(--muted); font-size:13px; }
    .switches label { display:flex; gap:6px; align-items:center; }
    .switches input { height:34px; border:1px solid #b8c4ca; border-radius:6px; padding:0 9px; background:white; }
    .switches button { height:34px; border:0; border-radius:6px; padding:0 12px; background:var(--blue); color:white; font-weight:700; cursor:pointer; }
    .metrics { display:grid; grid-template-columns:repeat(3,minmax(130px,1fr)); gap:10px; margin-top:12px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fbfdfe; }
    .metric .label { color:var(--muted); font-size:12px; }
    .metric .value { font-size:18px; font-weight:750; margin-top:3px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid #edf2f5; padding:7px 6px; text-align:right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
    th { color:var(--muted); font-weight:650; }
    .note { color:var(--muted); font-size:12px; line-height:1.45; margin-top:10px; }
    .method { color:#34434d; line-height:1.55; }
    .embedFrame { width:100%; height:820px; border:1px solid var(--line); border-radius:8px; background:white; }
    @media (max-width: 1000px) { .grid2, .grid3 { grid-template-columns:1fr; } .kpis { grid-template-columns:1fr 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Bolivia Grid AI Dashboard</h1>
    <div class="subtitle">One-day or range-based research dashboard for Bolivia Demand Model forecasting, CNDC comparison, and an RL-oriented dispatch optimization simulation.</div>
    <nav class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="prediction">Prediction</button>
      <button class="tab" data-tab="plantMap">Plant Map</button>
      <button class="tab" data-tab="optimization">Optimization</button>
      <button class="tab" data-tab="realtimeRl">Real-Time RL</button>
      <button class="tab" data-tab="method">Method</button>
    </nav>
  </header>
  <main>
    <div class="kpis" id="kpis"></div>

    <section class="view active" id="overview">
      <div class="datebar">
        <label>Analysis Date<input id="analysisDate" type="date"></label>
        <button id="runAnalysisDate" type="button">Run This Date</button>
        <span class="note" id="analysisDateNote"></span>
      </div>
      <div class="grid2">
        <div class="panel">
          <h2>Dispatch Impact</h2>
          <svg id="overviewChart" class="chart" viewBox="0 0 900 372" preserveAspectRatio="none"></svg>
          <div class="controls"><input class="timeSlider" type="range" min="0" max="0"><div class="time" id="overviewTime"></div></div>
          <div class="legend" id="overviewLegend"></div>
        </div>
        <div class="panel">
          <h2>What This Period Means</h2>
          <div class="method" id="periodText"></div>
          <table id="mixTable"></table>
        </div>
      </div>
    </section>

    <section class="view" id="prediction">
      <div class="panel">
        <h2>Demand Forecast Comparison</h2>
        <div class="switches">
          <label>Forecast Date<input id="predictionDate" type="date"></label>
          <button id="runPredictionDate" type="button">Run Forecast</button>
        </div>
        <svg id="predictionChart" class="chart" viewBox="0 0 900 372" preserveAspectRatio="none"></svg>
        <div class="controls"><input class="timeSlider" type="range" min="0" max="0"><div class="time" id="predictionTime"></div></div>
        <div class="legend" id="predictionLegend"></div>
        <div class="metrics" id="predictionMetrics"></div>
      </div>
    </section>

    <section class="view" id="plantMap">
      <div class="grid2">
        <div class="panel">
          <h2>Regional Generation Map</h2>
          <div class="controls"><input class="timeSlider" type="range" min="0" max="0"><div class="time" id="mapTime"></div></div>
          <div class="mapWrap"><svg id="mapSvg" viewBox="0 0 100 105.6" preserveAspectRatio="xMidYMid meet"></svg></div>
          <div class="legend" id="typeLegend"></div>
          <div class="note">Markers are aggregated by approximate region and technology to avoid overplotting individual units. Hover over a bubble for current historical and RL-oriented optimized MW.</div>
        </div>
        <div class="panel">
          <h2>Largest Regional Clusters</h2>
          <svg id="mixShiftChart" viewBox="0 0 520 220" preserveAspectRatio="none"></svg>
          <table id="clusterTable"></table>
        </div>
      </div>
    </section>

    <section class="view" id="optimization">
      <div class="grid2">
        <div class="panel">
          <h2>Historical vs RL-Oriented Dispatch</h2>
          <svg id="dispatchChart" class="chart" viewBox="0 0 900 372" preserveAspectRatio="none"></svg>
          <div class="controls"><input class="timeSlider" type="range" min="0" max="0"><div class="time" id="dispatchTime"></div></div>
          <div class="legend" id="dispatchLegend"></div>
        </div>
        <div class="panel">
          <h2>Largest Unit-Level Adjustments</h2>
          <table id="plantTable"></table>
        </div>
      </div>
    </section>

    <section class="view" id="realtimeRl">
      <div class="panel">
        <h2>Real-Time RL Dispatch Recommender</h2>
        <iframe class="embedFrame" src="../realtime_rl_latest/realtime_rl_dashboard.html"></iframe>
        <div class="note">Run the Real-Time RL tab in the local app to refresh this embedded recommendation.</div>
      </div>
    </section>

    <section class="view" id="method">
      <div class="grid3">
        <div class="panel"><h2>Forecasting</h2><p class="method">Bolivia Demand Model is an XGBoost demand model at 15-minute resolution. It uses CNDC regional demand forecasts, calendar signals, lagged demand, rolling windows, forecast-error lags, and regional demand lags to estimate the next demand curve.</p></div>
        <div class="panel"><h2>Backtesting</h2><p class="method">For past dates, the Prediction tab compares CNDC forecast and Bolivia Demand Model against measured demand using MAE. For future dates, measured demand is intentionally hidden because it is not known yet.</p></div>
        <div class="panel"><h2>Optimization</h2><p class="method">The RL-oriented dispatch target is a constrained merit-order simulation using transparent technology-level cost assumptions. It estimates how a learned control policy could shift dispatch toward lower proxy cost and higher renewable utilization.</p></div>
        <div class="panel"><h2>Scope</h2><p class="method">The cost saving is a research proxy for the selected analysis date or period only. It is not audited national system saving, and it does not yet model full power-flow, unit commitment, ramp limits, or transmission constraints.</p></div>
        <div class="panel"><h2>Map Source</h2><p class="method">Physical map: Bolivia physical map.svg by Urutseg, Wikimedia Commons. Plant markers are approximate regional clusters by technology, not official GPS locations.</p></div>
      </div>
    </section>
  </main>

  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    const intervals = data.intervals;
    const plantsByTime = data.plantsByTime;
    let selected = Math.floor((intervals.length - 1) / 2);
    let activeTab = "overview";
    const sliders = [...document.querySelectorAll(".timeSlider")];
    sliders.forEach(s => { s.max = intervals.length - 1; s.value = selected; s.addEventListener("input", e => { selected = Number(e.target.value); sync(); }); });

    function fmt(n, d=1) { return Number(n).toLocaleString(undefined, {maximumFractionDigits:d}); }
    function money(n) { return "$" + Number(n).toLocaleString(undefined, {maximumFractionDigits:0}); }
    function absErr(a, b) { return a == null || b == null ? null : Math.abs(a - b); }
    function mae(key) {
      const errors = intervals.map(d => absErr(d.demand, d[key])).filter(v => v != null && Number.isFinite(v));
      return errors.length ? errors.reduce((a,b) => a + b, 0) / errors.length : null;
    }
    function periodLabel() {
      const r = data.report;
      const start = r.start.slice(0, 10), end = r.end.slice(0, 10);
      return start === end ? start : `${start} to ${end}`;
    }
    function periodStartDate() { return data.report.start.slice(0, 10); }
    function tomorrowIso() {
      const d = new Date();
      d.setDate(d.getDate() + 1);
      return d.toISOString().slice(0, 10);
    }
    function isFuturePeriod() {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      return new Date(periodStartDate() + "T00:00:00") >= today;
    }

    function renderKpis() {
      const r = data.report;
      const items = [
        ["Analysis period", periodLabel()],
        ["Cost-proxy saving for period", money(r.cost_proxy_saving_usd)],
        ["Saving vs historical proxy", fmt(r.cost_proxy_saving_pct, 1) + "%"],
        ["Historical renewable share", fmt(r.actual_renewable_share * 100, 1) + "%"],
        ["RL-optimized renewable share", fmt(r.optimized_renewable_share * 100, 1) + "%"],
      ];
      document.getElementById("kpis").innerHTML = items.map(([a,b]) => `<div class="kpi"><div class="label">${a}</div><div class="value">${b}</div></div>`).join("");
      document.getElementById("periodText").innerHTML = `All KPI values summarize <b>${periodLabel()}</b>. The ${money(r.cost_proxy_saving_usd)} figure is the estimated saving across this selected period only, using the technology-level cost proxy documented in the Method tab.`;
      document.getElementById("analysisDate").value = periodStartDate();
      document.getElementById("analysisDate").max = tomorrowIso();
      document.getElementById("predictionDate").value = periodStartDate();
      document.getElementById("predictionDate").max = tomorrowIso();
      document.getElementById("analysisDateNote").textContent = "Changing the calendar runs a fresh local optimization for that day.";
    }

    function pathFor(values, key, x0, y0, w, h, minY, maxY) {
      return values.map((d, i) => {
        if (d[key] == null) return "";
        const x = x0 + i / (values.length - 1) * w;
        const y = y0 + h - (d[key] - minY) / (maxY - minY) * h;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
    }

    function renderLineChart(svgId, legendId, timeId, series) {
      const svg = document.getElementById(svgId);
      const active = series.filter(s => s.show !== false);
      const vals = intervals.flatMap(d => active.map(s => d[s.key]).filter(v => v != null));
      const minY = Math.min(...vals) * 0.94, maxY = Math.max(...vals) * 1.04;
      const x0 = 52, y0 = 24, w = 815, h = 238;
      const sx = x0 + selected / (intervals.length - 1) * w;
      const d = intervals[selected];
      svg.innerHTML = `
        <rect x="0" y="0" width="900" height="372" fill="#fff"/>
        ${[0,1,2,3,4].map(i => `<line x1="${x0}" x2="${x0+w}" y1="${y0+i*h/4}" y2="${y0+i*h/4}" stroke="#e9eef2"/>`).join("")}
        ${active.map(s => `<path d="${pathFor(intervals,s.key,x0,y0,w,h,minY,maxY)}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.7}" stroke-dasharray="${s.dash || ""}"/>`).join("")}
        <line x1="${sx}" x2="${sx}" y1="${y0}" y2="${y0+h}" stroke="#0b7285" stroke-width="1.5" opacity="0.55"/>
        <rect x="52" y="282" width="815" height="70" rx="5" fill="#ffffff" stroke="#dbe4ea"/>
        <text x="66" y="303" font-size="12" font-weight="700" fill="#17212b">${d.timestamp}</text>
        ${active.map((s,idx) => `<circle cx="${70 + (idx % 3)*265}" cy="${idx < 3 ? 322 : 340}" r="3.2" fill="${s.color}"/><text x="${82 + (idx % 3)*265}" y="${idx < 3 ? 326 : 344}" font-size="12" fill="#17212b">${s.short}: ${d[s.key] == null ? "n/a" : fmt(d[s.key]) + " MW"}</text>`).join("")}
        <text x="5" y="${y0+5}" font-size="11" fill="#5a6b76">${fmt(maxY,0)} MW</text>
        <text x="5" y="${y0+h}" font-size="11" fill="#5a6b76">${fmt(minY,0)} MW</text>
      `;
      document.getElementById(legendId).innerHTML = active.map(s => `<span><span class="dot" style="background:${s.color}"></span>${s.label}</span>`).join("");
      document.getElementById(timeId).textContent = d.timestamp;
    }

    function mapGroups() {
      const groups = new Map();
      for (const p of plantsByTime[selected].plants) {
        const key = `${p.region}|${p.type}`;
        if (!groups.has(key)) groups.set(key, {region:p.region, type:p.type, label:p.label, color:p.color, icon:p.icon, x:0, y:0, weight:0, actual:0, optimized:0, delta:0, count:0});
        const g = groups.get(key);
        const weight = Math.max(p.actual, p.optimized, 1);
        g.x += p.x * weight; g.y += p.y * weight; g.weight += weight;
        g.actual += p.actual; g.optimized += p.optimized; g.delta += p.delta; g.count += 1;
      }
      return [...groups.values()].map(g => ({...g, x:g.x/g.weight, y:g.y/g.weight}));
    }
    function radius(mw) { return Math.min(6.5, Math.max(1.5, Math.sqrt(Math.max(mw, 0)) * 0.24)); }
    function iconFor(type, x, y) {
      if (type === "SOL") return `<g transform="translate(${x-1.6},${y-1.6})"><rect x="0" y="0.7" width="3.2" height="2.1" rx="0.2" fill="#1864ab"/><path d="M0 1.75h3.2M1.05 0.7v2.1M2.1 0.7v2.1" stroke="#d0ebff" stroke-width="0.18"/><circle cx="3.2" cy="0" r="0.55" fill="#ffd43b"/></g>`;
      if (type === "EOL") return `<g transform="translate(${x},${y})"><line x1="0" y1="0" x2="0" y2="2.9" stroke="#075985" stroke-width="0.35"/><circle cx="0" cy="0" r="0.35" fill="#075985"/><path d="M0 0 L0 -2.2 M0 0 L1.9 1.1 M0 0 L-1.9 1.1" stroke="#075985" stroke-width="0.35"/></g>`;
      if (type === "HID") return `<g transform="translate(${x-1.5},${y-1.5})"><path d="M1.5 0 C0.2 1.5 0.2 2.5 1.5 3.2 C2.8 2.5 2.8 1.5 1.5 0Z" fill="#1971c2"/><path d="M0.5 2.4 C1.2 2 2 2.8 2.6 2.3" stroke="#d0ebff" stroke-width="0.22" fill="none"/></g>`;
      if (type === "TER") return `<g transform="translate(${x-1.6},${y-1.5})"><rect x="0" y="1.5" width="3.2" height="1.6" fill="#e8590c"/><path d="M0.4 1.5 L0.4 0.3 L1.3 1 L1.3 0.3 L2.2 1 L2.2 0.3 L3 1.5Z" fill="#ff922b"/><rect x="2.5" y="0" width="0.5" height="1.5" fill="#495057"/></g>`;
      if (type === "BIO") return `<g transform="translate(${x-1.5},${y-1.5})"><path d="M0.4 2.8 C0.4 0.8 2 0 3 0.4 C2.8 2.1 1.4 3.2 0.4 2.8Z" fill="#2b8a3e"/><path d="M0.8 2.4 C1.4 1.6 2 1 2.6 0.7" stroke="#d3f9d8" stroke-width="0.22"/></g>`;
      return `<circle cx="${x}" cy="${y}" r="1.1" fill="#fff"/>`;
    }

    function renderMap() {
      const svg = document.getElementById("mapSvg");
      const groups = mapGroups();
      svg.innerHTML = `
        <image href="../../assets/bolivia_physical_map.png" x="0" y="0" width="100" height="105.6" preserveAspectRatio="xMidYMid meet"/>
        <rect x="0" y="0" width="100" height="105.6" fill="rgba(255,255,255,0.08)"/>
        ${groups.map(g => {
          const rA = radius(g.actual), rO = radius(g.optimized);
          return `<g>
            <circle cx="${g.x}" cy="${g.y}" r="${rO}" fill="none" stroke="${g.color}" stroke-width="1.2" opacity="0.95"/>
            <circle cx="${g.x}" cy="${g.y}" r="${rA}" fill="${g.color}" opacity="0.48"/>
            ${iconFor(g.type, g.x, g.y)}
            <title>${g.region} ${g.label}: historical ${fmt(g.actual)} MW, RL-optimized ${fmt(g.optimized)} MW, units ${g.count}</title>
          </g>`;
        }).join("")}
      `;
      const types = Object.entries(data.types).filter(([k]) => groups.some(g => g.type === k));
      document.getElementById("typeLegend").innerHTML = types.map(([k,v]) => `<span><span class="dot" style="background:${v.color}"></span>${v.label}</span>`).join("");
      document.getElementById("mapTime").textContent = intervals[selected].timestamp;
      document.getElementById("clusterTable").innerHTML = `<tr><th>Region</th><th>Type</th><th>Historical</th><th>RL target</th><th>Units</th></tr>` +
        groups.sort((a,b) => Math.max(b.actual,b.optimized)-Math.max(a.actual,a.optimized)).slice(0,12)
        .map(g => `<tr><td>${g.region}</td><td>${g.type}</td><td>${fmt(g.actual)}</td><td>${fmt(g.optimized)}</td><td>${g.count}</td></tr>`).join("");
      renderMixShift(groups);
    }

    function renderMixShift(groups) {
      const svg = document.getElementById("mixShiftChart");
      const byType = new Map();
      for (const g of groups) {
        if (!byType.has(g.type)) byType.set(g.type, {type:g.type, label:g.label, color:g.color, actual:0, optimized:0});
        const row = byType.get(g.type);
        row.actual += g.actual;
        row.optimized += g.optimized;
      }
      const rows = [...byType.values()].sort((a,b) => b.optimized - a.optimized);
      const actualTotal = rows.reduce((a,r) => a + r.actual, 0);
      const optTotal = rows.reduce((a,r) => a + r.optimized, 0);
      const x0 = 110, y0 = 42, w = 330, h = 34, gap = 54;
      function segments(key, total, y) {
        let x = x0;
        return rows.map(r => {
          const sw = total ? (r[key] / total) * w : 0;
          const out = `<rect x="${x}" y="${y}" width="${sw}" height="${h}" fill="${r.color}" opacity="0.88"><title>${r.label}: ${fmt(r[key])} MW</title></rect>`;
          x += sw;
          return out;
        }).join("");
      }
      svg.innerHTML = `
        <rect x="0" y="0" width="520" height="220" fill="#fff"/>
        <text x="0" y="18" font-size="15" font-weight="750" fill="#17212b">Generation Mix at Selected Time</text>
        <text x="0" y="${y0+22}" font-size="12" font-weight="700" fill="#60717d">Historical</text>
        <text x="0" y="${y0+gap+22}" font-size="12" font-weight="700" fill="#60717d">RL target</text>
        <rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="#f1f5f8" stroke="#d9e3e9"/>
        <rect x="${x0}" y="${y0+gap}" width="${w}" height="${h}" fill="#f1f5f8" stroke="#d9e3e9"/>
        ${segments("actual", actualTotal, y0)}
        ${segments("optimized", optTotal, y0 + gap)}
        <text x="${x0+w+10}" y="${y0+22}" font-size="12" fill="#60717d">${fmt(actualTotal)} MW</text>
        <text x="${x0+w+10}" y="${y0+gap+22}" font-size="12" fill="#60717d">${fmt(optTotal)} MW</text>
        ${rows.map((r,i) => `<circle cx="${18 + (i%3)*155}" cy="${158 + Math.floor(i/3)*24}" r="5" fill="${r.color}"/><text x="${30 + (i%3)*155}" y="${162 + Math.floor(i/3)*24}" font-size="12" fill="#34434d">${r.label}</text>`).join("")}
      `;
    }

    function renderTables() {
      const plants = [...plantsByTime[selected].plants].sort((a,b) => Math.abs(b.delta)-Math.abs(a.delta)).slice(0,12);
      document.getElementById("plantTable").innerHTML = `<tr><th>Plant</th><th>Type</th><th>Historical</th><th>RL target</th><th>Delta</th></tr>` +
        plants.map(p => `<tr><td>${p.code}</td><td>${p.type}</td><td>${fmt(p.actual)}</td><td>${fmt(p.optimized)}</td><td style="color:${p.delta>=0?'#2b8a3e':'#c92a2a'}">${p.delta>=0?'+':''}${fmt(p.delta)}</td></tr>`).join("");
      document.getElementById("mixTable").innerHTML = `<tr><th>Type</th><th>Historical MWh</th><th>RL target MWh</th><th>Change</th></tr>` +
        data.byType.map(r => `<tr><td>${r.generation_type}</td><td>${fmt(r.actual_mwh)}</td><td>${fmt(r.optimized_mwh)}</td><td style="color:${r.optimized_mwh-r.actual_mwh>=0?'#2b8a3e':'#c92a2a'}">${r.optimized_mwh-r.actual_mwh>=0?'+':''}${fmt(r.optimized_mwh-r.actual_mwh)}</td></tr>`).join("");
    }

    function sync() {
      sliders.forEach(s => { if (Number(s.value) !== selected) s.value = selected; });
      renderLineChart("overviewChart", "overviewLegend", "overviewTime", [
        {key:"demand", short:"Demand", label:"Real demand", color:"#17212b", width:3},
        {key:"actual_dispatch", short:"Historical", label:"Historical dispatch", color:"#e8590c", dash:"4 4"},
        {key:"optimized_dispatch", short:"RL target", label:"RL-optimized dispatch", color:"#0b7285", width:3}
      ]);
      const future = isFuturePeriod();
      const predictionSeries = [
        {key:"forecast", short:"CNDC", label:"CNDC forecast", color:"#868e96", dash:"6 5"},
        {key:"bdm", short:"Bolivia Demand Model", label:"Bolivia Demand Model prediction", color:"#2b8a3e", show:data.hasBdm}
      ];
      if (!future) predictionSeries.unshift({key:"demand", short:"Actual", label:"Real demand", color:"#17212b", width:3});
      renderLineChart("predictionChart", "predictionLegend", "predictionTime", predictionSeries);
      renderPredictionMetrics(future);
      renderLineChart("dispatchChart", "dispatchLegend", "dispatchTime", [
        {key:"demand", short:"Demand", label:"Real demand", color:"#17212b", width:3},
        {key:"actual_dispatch", short:"Historical", label:"Historical dispatch", color:"#e8590c", dash:"4 4"},
        {key:"optimized_dispatch", short:"RL target", label:"RL-optimized dispatch", color:"#0b7285", width:3}
      ]);
      if (activeTab === "plantMap") renderMap();
      renderTables();
    }

    function renderPredictionMetrics(future) {
      const box = document.getElementById("predictionMetrics");
      if (future) {
        box.innerHTML = `<div class="metric"><div class="label">Forecast date</div><div class="value">${periodLabel()}</div></div><div class="metric"><div class="label">Real demand</div><div class="value">Pending</div></div>`;
        return;
      }
      const cndc = mae("forecast");
      const bdm = mae("bdm");
      const rows = [
        ["CNDC MAE", cndc == null ? "n/a" : fmt(cndc, 2) + " MW"],
        ["BDM MAE", bdm == null ? "n/a" : fmt(bdm, 2) + " MW"],
        ["Best on this day", cndc != null && bdm != null ? (bdm < cndc ? "Bolivia Demand Model" : "CNDC") : "n/a"],
      ];
      box.innerHTML = rows.map(([a,b]) => `<div class="metric"><div class="label">${a}</div><div class="value">${b}</div></div>`).join("");
    }

    async function runLocal(action, payload, button) {
      const oldText = button.textContent;
      button.disabled = true;
      button.textContent = "Running...";
      try {
        const res = await fetch(`/run/${action}`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        const result = await res.json();
        if (result.error) throw new Error(result.error);
        const primary = result.links?.find(item => item.primary) || result.links?.[result.links.length - 1];
        if (primary) window.location.href = primary.href;
      } catch (err) {
        alert(err.message || String(err));
        button.disabled = false;
        button.textContent = oldText;
      }
    }

    document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
      activeTab = button.dataset.tab;
      document.body.dataset.tab = activeTab;
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === button));
      document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === button.dataset.tab));
      sync();
    }));
    document.getElementById("runAnalysisDate").addEventListener("click", event => {
      const date = document.getElementById("analysisDate").value;
      runLocal("optimize", {start_date:date, end_date:date, name:`plant_${date.replaceAll("-", "")}`, plant_intervals:"all", reserve_margin:"0.03", capacity_margin:"1.10"}, event.currentTarget);
    });
    document.getElementById("runPredictionDate").addEventListener("click", event => {
      const date = document.getElementById("predictionDate").value;
      runLocal("forecast", {date, lookback_days:"21", chart:"0"}, event.currentTarget);
    });
    document.body.dataset.tab = activeTab;
    renderKpis();
    sync();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-prediction", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = build_payload(args.run_dir, args.data_dir, args.model_prediction)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out = args.out or args.run_dir / "interactive_dispatch_dashboard.html"
    out.write_text(html)
    print(f"Saved dashboard: {out}")


if __name__ == "__main__":
    main()
