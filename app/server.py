#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request
import csv
import sys
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = APP_ROOT
PYTHON = sys.executable
CNDC_API = "https://cndcapi.cndc.bo/WebApi"


INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bolivia AI Energy Lab</title>
  <style>
    :root { color-scheme: light; --ink:#1f2933; --muted:#5c6b73; --line:rgba(111,130,139,0.24); --blue:#087f8c; --blue2:#125f74; --green:#2f9e44; --bg:#eef4f1; --panel:rgba(255,255,255,0.80); --shadow:0 18px 50px rgba(33,54,61,0.12); }
    body {
      margin:0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(120deg, rgba(8,127,140,0.10), rgba(47,158,68,0.08) 42%, rgba(230,119,0,0.07)),
        linear-gradient(rgba(255,255,255,0.34) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.34) 1px, transparent 1px),
        var(--bg);
      background-size:auto,44px 44px,44px 44px,auto;
      color:var(--ink);
    }
    header { padding:28px 34px 18px; border-bottom:1px solid var(--line); background:rgba(255,255,255,0.84); backdrop-filter:blur(18px); box-shadow:0 10px 34px rgba(34,55,63,0.08); }
    h1 { margin:0; font-size:30px; letter-spacing:0; }
    header p { margin:8px 0 0; color:var(--muted); max-width:980px; line-height:1.45; }
    main { padding:24px 34px 44px; max-width:1180px; }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
    .tab { border:1px solid rgba(18,95,116,0.16); background:rgba(248,252,251,0.84); color:#25313a; border-radius:8px; padding:9px 13px; font-weight:750; cursor:pointer; box-shadow:0 4px 16px rgba(34,55,63,0.06); }
    .tab.active { background:linear-gradient(135deg,var(--blue),var(--blue2)); color:white; border-color:rgba(8,127,140,0.55); }
    section { display:none; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:var(--shadow); backdrop-filter:blur(14px); }
    section.active { display:block; }
    h2 { font-size:19px; margin:0 0 12px; }
    form { display:grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap:12px; align-items:end; }
    label { display:grid; gap:5px; font-size:12px; color:var(--muted); }
    input, select { height:36px; border:1px solid rgba(102,120,129,0.34); border-radius:8px; padding:0 10px; font-size:14px; background:rgba(255,255,255,0.88); }
    button:not(.tab) { height:38px; border:0; border-radius:8px; padding:0 14px; background:linear-gradient(135deg,var(--blue),var(--blue2)); color:white; font-weight:700; cursor:pointer; box-shadow:0 8px 20px rgba(8,127,140,0.20); }
    button:disabled { opacity:.62; cursor:wait; }
    button.secondary:not(.tab) { background:linear-gradient(135deg,var(--green),#1b7f4a); }
    pre { white-space:pre-wrap; background:rgba(16,24,32,0.94); color:#d9f7ef; padding:16px; border-radius:8px; min-height:160px; overflow:auto; box-shadow:var(--shadow); }
    .wide { grid-column: span 2; }
    .hint { color:var(--muted); font-size:13px; line-height:1.45; margin-top:8px; }
    .links a { display:inline-block; margin:6px 12px 0 0; color:var(--blue); font-weight:650; text-decoration:none; }
    .output { margin-top:18px; }
    @media (max-width:900px) { form { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Bolivia AI Energy Lab</h1>
    <p>Local research prototype for CNDC data collection, Bolivia Demand Model forecasting, reinforcement learning dispatch support, and plant-level optimization.</p>
    <nav class="tabs">
      <button class="tab active" data-tab="forecast">Forecast</button>
      <button class="tab" data-tab="optimize">Optimization</button>
      <button class="tab" data-tab="realtime">Real-Time RL</button>
      <button class="tab" data-tab="rl">RL Experiment</button>
      <button class="tab" data-tab="evidence">Research Evidence</button>
      <button class="tab" data-tab="files">Results</button>
    </nav>
  </header>
  <main>
    <section id="forecast" class="active">
      <h2>Demand Forecast</h2>
      <form data-action="forecast">
        <label>Target Date<input name="date" type="date" value="__TOMORROW__"></label>
        <label>Lookback Days<input name="lookback_days" type="number" value="21"></label>
        <label>Chart<select name="chart"><option value="0">No</option><option value="1">Yes</option></select></label>
        <button>Run Forecast</button>
      </form>
      <p class="hint">Predicts a full day at 15-minute resolution using the Bolivia Demand Model and live CNDC forecast/history.</p>
    </section>

    <section id="optimize">
      <h2>Plant-Level Dispatch Optimization</h2>
      <form data-action="optimize">
        <label>Start Date<input name="start_date" type="date" value="2026-05-30"></label>
        <label>End Date<input name="end_date" type="date" value="2026-05-30"></label>
        <label>Name<input name="name" value="plant_may30"></label>
        <label>Intervals<input name="plant_intervals" value="all"></label>
        <label>Reserve<input name="reserve_margin" type="number" step="0.01" value="0.03"></label>
        <label>Capacity<input name="capacity_margin" type="number" step="0.05" value="1.10"></label>
        <button>Run Optimization</button>
      </form>
      <p class="hint">Choose any past day or range. The dashboard labels the analysis period and, for single-day runs, attaches the Bolivia Demand Model prediction automatically when that prediction file exists.</p>
    </section>

    <section id="realtime">
      <h2>Real-Time RL Dispatch</h2>
      <form data-action="realtime-rl">
        <label>Snapshot Date<input name="date" type="date" value="__TODAY__"></label>
        <label>Name<input name="name" value="realtime_rl_latest"></label>
        <label>Refresh CNDC<select name="refresh"><option value="0">No</option><option value="1">Yes</option></select></label>
        <button class="secondary">Run Real-Time RL</button>
      </form>
      <p class="hint">Builds a short-horizon dispatch recommendation from latest demand, generation, and forecast data. This is a decision-support simulation, not an operational control command.</p>
    </section>

    <section id="rl">
      <h2>RL Dispatch-Support Experiment</h2>
      <form data-action="rl">
        <label>Start Date<input name="start_date" type="date" value="2026-05-01"></label>
        <label>End Date<input name="end_date" type="date" value="2026-05-29"></label>
        <label>Name<input name="name" value="may_rl_experiment"></label>
        <label>Episodes<input name="episodes" type="number" value="400"></label>
        <label>Train Fraction<input name="train_fraction" type="number" step="0.05" value="0.70"></label>
        <button class="secondary">Run RL</button>
      </form>
      <p class="hint">Runs a train/test Q-learning experiment where the agent learns reserve adjustments around the CNDC forecast.</p>
    </section>

    <section id="evidence">
      <h2>Research Evidence</h2>
      <form data-action="evidence">
        <label>Report Name<input name="name" value="latest"></label>
        <label>Neural Run<input name="neural_name" value="neural_mlp_baseline"></label>
        <button class="secondary">Build Evidence Report</button>
      </form>
      <p class="hint">Builds a research evidence dashboard with backtesting, naive baselines, leakage checks, RL environment framing, and optional neural-baseline results.</p>
      <form data-action="neural" style="margin-top:14px">
        <label>Run Name<input name="name" value="neural_mlp_baseline"></label>
        <label>Max Iterations<input name="max_iter" type="number" value="120"></label>
        <button>Run Neural Baseline</button>
      </form>
      <p class="hint">The neural model is a compact MLP baseline on the existing Bolivia Demand Model features and is used as a comparison point.</p>
    </section>

    <section id="files">
      <h2>Results</h2>
      <p class="hint">Open saved outputs, charts, CSVs, and dashboards from previous runs.</p>
      <div class="links">
        <a target="_blank" href="/files/runs/forecasts/">Forecast runs</a>
        <a target="_blank" href="/files/runs/">Optimization runs</a>
        <a target="_blank" href="/files/runs/">RL runs</a>
        <a target="_blank" href="/files/runs/">Research evidence</a>
      </div>
    </section>

    <div class="output">
      <h2>Output</h2>
      <pre id="log">Ready.</pre>
      <div class="links" id="links"></div>
    </div>
  </main>
  <script>
    const log = document.getElementById("log");
    const links = document.getElementById("links");
    document.querySelectorAll(".tab").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === button));
        document.querySelectorAll("main > section").forEach(s => s.classList.toggle("active", s.id === button.dataset.tab));
      });
    });
    document.querySelectorAll("form").forEach(form => {
      form.addEventListener("submit", async event => {
        event.preventDefault();
        const button = form.querySelector("button");
        const oldText = button.textContent;
        button.disabled = true;
        button.textContent = "Running...";
        log.textContent = "Running... this can take a few minutes for plant-level optimization.";
        links.innerHTML = "";
        const data = Object.fromEntries(new FormData(form).entries());
        try {
          const response = await fetch("/run/" + form.dataset.action, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
          });
          const result = await response.json();
          log.textContent = result.output || result.error || "Done.";
          if (result.links) {
            links.innerHTML = result.links.map(item => `<a target="_blank" href="${item.href}">${item.label}</a>`).join("");
            const primary = result.links.find(item => item.primary);
            if (primary) window.location.href = primary.href;
          }
        } finally {
          button.disabled = false;
          button.textContent = oldText;
        }
      });
    });
  </script>
</body>
</html>
"""


def dashboard_for_date(target_date: date) -> Path | None:
    name = f"plant_{target_date:%Y%m%d}_optimization"
    path = OUTPUTS_ROOT / "runs" / name / "interactive_dispatch_dashboard.html"
    return path if path.exists() else None


def default_dashboard_path() -> Path | None:
    yesterday = date.today() - timedelta(days=1)
    dashboard = dashboard_for_date(yesterday)
    if dashboard:
        return dashboard
    candidates = sorted(
        (OUTPUTS_ROOT / "runs").glob("*/interactive_dispatch_dashboard.html"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def files_href(path: Path) -> str:
    rel = urllib.parse.quote(str(path.relative_to(OUTPUTS_ROOT)))
    return f"/files/{rel}"


def parse_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def fetch_actual_demand(date_str: str) -> list[float | None]:
    api_date = date.fromisoformat(date_str).strftime("%d.%m.%Y")
    url = CNDC_API + "?" + urllib.parse.urlencode({"code": 1, "Fecha": api_date})
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "*/*",
            "Origin": "https://www.cndc.bo",
            "Referer": "https://www.cndc.bo/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    totals = [0.0] * 96
    counts = [0] * 96
    for record in payload:
        code = str(record.get("codigo") or "").strip().upper()
        if code.startswith("PREV"):
            continue
        for idx, value in enumerate(record.get("valores", [])[:96]):
            number = parse_float(value)
            if number is not None:
                totals[idx] += number
                counts[idx] += 1
    return [totals[i] if counts[i] else None for i in range(96)]


def local_actual_demand(target_date: str) -> list[float | None] | None:
    compact = target_date.replace("-", "")
    candidates = [
        OUTPUTS_ROOT / "data" / f"plant_{compact}" / "demand_15min.csv",
        OUTPUTS_ROOT / "data" / target_date / "demand_15min.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        rows: list[float | None] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("total") or row.get("real_total_demand_mw") or row.get("demand_mw")
                rows.append(parse_float(value))
        if rows:
            return (rows + [None] * 96)[:96]
    return None


def forecast_payload(target_date: str) -> dict | None:
    run_dir = OUTPUTS_ROOT / "runs" / "forecasts" / f"prediction_{target_date}"
    csv_path = run_dir / f"bdm_prediction_{target_date}.csv"
    if not csv_path.exists():
        return None

    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp": row["timestamp"][:16],
                    "bdm": float(row["predicted_demand_mw"]),
                    "forecast": float(row["cndc_forecast_total_mw"]),
                    "demand": None,
                }
            )

    if date.fromisoformat(target_date) <= date.today():
        actual = local_actual_demand(target_date)
        if actual is None:
            try:
                actual = fetch_actual_demand(target_date)
            except Exception:
                actual = None
        if actual is not None:
            for row, value in zip(rows, actual):
                row["demand"] = value

    return {"targetDate": target_date, "rows": rows}


def make_forecast_dashboard(target_date: str) -> Path | None:
    payload_data = forecast_payload(target_date)
    if payload_data is None:
        return None

    run_dir = OUTPUTS_ROOT / "runs" / "forecasts" / f"prediction_{target_date}"
    rows = [
        {
            "timestamp": row["timestamp"],
            "model": row["bdm"],
            "cndc": row["forecast"],
            "actual": row["demand"],
        }
        for row in payload_data["rows"]
    ]
    payload = json.dumps({"targetDate": target_date, "rows": rows})
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bolivia Demand Model Forecast</title>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f4f7f9; color:#17212b; }
    header { background:white; border-bottom:1px solid #d9e3e9; padding:22px 28px 16px; }
    h1 { margin:0; font-size:27px; }
    main { max-width:1180px; margin:0 auto; padding:20px 28px 34px; }
    .panel,.metric { background:white; border:1px solid #d9e3e9; border-radius:8px; padding:16px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)); gap:12px; margin-bottom:16px; }
    .label { color:#60717d; font-size:12px; }
    .value { font-size:22px; font-weight:750; margin-top:4px; }
    svg { width:100%; height:390px; display:block; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; color:#60717d; font-size:13px; margin-top:10px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    .controls { display:flex; gap:12px; align-items:center; margin-top:12px; }
    input[type=range] { width:100%; }
    .time { min-width:150px; color:#0b7285; font-weight:750; }
    .datebar { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-bottom:16px; }
    .datebar label { display:grid; gap:4px; color:#60717d; font-size:12px; font-weight:650; }
    .datebar input { height:34px; border:1px solid #b8c4ca; border-radius:6px; padding:0 9px; background:white; }
    .datebar button { height:34px; border:0; border-radius:6px; padding:0 12px; background:#0b7285; color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.62; cursor:wait; }
    @media (max-width:900px) { .metrics { grid-template-columns:1fr 1fr; } }
  </style>
</head>
<body>
  <header><h1>Bolivia Demand Forecast</h1></header>
  <main>
    <div class="datebar">
      <label>Forecast Date<input id="forecastDate" type="date"></label>
      <button id="runForecast" type="button">Run Forecast</button>
    </div>
    <div class="metrics" id="metrics"></div>
    <section class="panel">
      <svg id="chart" viewBox="0 0 900 390" preserveAspectRatio="none"></svg>
      <div class="controls"><input id="slider" type="range" min="0" max="0" value="0"><div class="time" id="timeLabel"></div></div>
      <div class="legend" id="legend"></div>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    const rows = data.rows;
    const hasActual = rows.some(r => r.actual != null);
    const slider = document.getElementById("slider");
    let selected = Math.floor((rows.length - 1) / 2);
    slider.max = rows.length - 1;
    slider.value = selected;
    function tomorrowIso(){const d=new Date(); d.setDate(d.getDate()+1); return d.toISOString().slice(0,10);}
    function fmt(n,d=1){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d});}
    function mae(key){const e=rows.filter(r=>r.actual!=null&&r[key]!=null).map(r=>Math.abs(r.actual-r[key])); return e.length?e.reduce((a,b)=>a+b,0)/e.length:null;}
    function path(key,x0,y0,w,h,minY,maxY){return rows.map((r,i)=>{if(r[key]==null)return ""; const x=x0+i/(rows.length-1)*w; const y=y0+h-(r[key]-minY)/(maxY-minY)*h; return `${i?"L":"M"}${x.toFixed(1)},${y.toFixed(1)}`}).join(" ");}
    function draw(){
      const series=[["cndc","#868e96","CNDC forecast","6 5"],["model","#2b8a3e","Bolivia Demand Model forecast",""]];
      if(hasActual) series.unshift(["actual","#17212b","Real demand",""]);
      const vals=rows.flatMap(r=>series.map(s=>r[s[0]]).filter(v=>v!=null));
      const minY=Math.min(...vals)*0.94, maxY=Math.max(...vals)*1.04;
      const x0=54,y0=26,w=810,h=280;
      const sx=x0+selected/(rows.length-1)*w;
      const point=rows[selected];
      document.getElementById("chart").innerHTML=`
        <rect x="0" y="0" width="900" height="390" fill="#fff"/>
        ${[0,1,2,3,4].map(i=>`<line x1="${x0}" x2="${x0+w}" y1="${y0+i*h/4}" y2="${y0+i*h/4}" stroke="#e9eef2"/>`).join("")}
        ${series.map(s=>`<path d="${path(s[0],x0,y0,w,h,minY,maxY)}" fill="none" stroke="${s[1]}" stroke-width="3" stroke-dasharray="${s[3]}"/>`).join("")}
        <line x1="${sx}" x2="${sx}" y1="${y0}" y2="${y0+h}" stroke="#0b7285" stroke-width="1.5" opacity="0.55"/>
        <rect x="54" y="322" width="810" height="48" rx="5" fill="#fff" stroke="#d9e3e9"/>
        ${series.map((s,i)=>`<circle cx="${74+i*250}" cy="350" r="3.2" fill="${s[1]}"/><text x="${86+i*250}" y="354" font-size="12" fill="#17212b">${s[2].replace(" forecast","")}: ${point[s[0]]==null?"n/a":fmt(point[s[0]])+" MW"}</text>`).join("")}
        <text x="8" y="${y0+4}" font-size="11" fill="#60717d">${fmt(maxY,0)} MW</text>
        <text x="8" y="${y0+h}" font-size="11" fill="#60717d">${fmt(minY,0)} MW</text>
      `;
      document.getElementById("timeLabel").textContent = point.timestamp;
      document.getElementById("legend").innerHTML=series.map(s=>`<span><span class="dot" style="background:${s[1]}"></span>${s[2]}</span>`).join("");
    }
    const cndcMae=mae("cndc"), modelMae=mae("model");
    const avg=rows.reduce((a,r)=>a+r.model,0)/rows.length;
    const peak=rows.reduce((a,r)=>r.model>a.model?r:a,rows[0]);
    const metrics=[
      ["Forecast date", data.targetDate],
      ["Average BDM", fmt(avg)+" MW"],
      ["Peak BDM", fmt(peak.model)+" MW"],
      [hasActual?"BDM MAE":"Real demand", hasActual?fmt(modelMae,2)+" MW":"Pending"]
    ];
    if(hasActual) metrics.push(["CNDC MAE", fmt(cndcMae,2)+" MW"]);
    document.getElementById("metrics").innerHTML=metrics.map(m=>`<div class="metric"><div class="label">${m[0]}</div><div class="value">${m[1]}</div></div>`).join("");
    slider.addEventListener("input", () => { selected = Number(slider.value); draw(); });
    document.getElementById("forecastDate").value=data.targetDate;
    document.getElementById("forecastDate").max=tomorrowIso();
    document.getElementById("runForecast").addEventListener("click", async event => {
      const button=event.currentTarget;
      const old=button.textContent;
      button.disabled=true;
      button.textContent="Running...";
      try {
        const date=document.getElementById("forecastDate").value;
        const res=await fetch("/run/forecast",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({date,lookback_days:"21",chart:"0"})});
        const result=await res.json();
        if(result.error) throw new Error(result.error);
        const primary=result.links?.find(item=>item.primary)||result.links?.[0];
        if(primary) window.location.href=primary.href;
      } catch(err) {
        alert(err.message||String(err));
        button.disabled=false;
        button.textContent=old;
      }
    });
    draw();
  </script>
</body>
</html>""".replace("__PAYLOAD__", payload)
    out = run_dir / "forecast_dashboard.html"
    out.write_text(html)
    return out


def run_command(action: str, data: dict) -> tuple[str, list[dict]]:
    base = [PYTHON, str(APP_ROOT / "run_project.py")]
    links: list[dict] = []
    if action == "forecast":
        cmd = base + ["forecast", "--date", data["date"], "--lookback-days", data.get("lookback_days", "21")]
        if data.get("chart") == "1":
            cmd.append("--chart")
        date = data["date"]
        links.append({"label": "Forecast Files", "href": f"/files/runs/forecasts/prediction_{date}/"})
    elif action == "rl":
        name = data.get("name") or "rl_experiment"
        cmd = base + [
            "rl", "--start-date", data["start_date"], "--end-date", data["end_date"],
            "--name", name, "--episodes", data.get("episodes", "400"),
            "--train-fraction", data.get("train_fraction", "0.70")
        ]
        links.append({"label": "RL Runs", "href": "/files/runs/"})
    elif action == "realtime-rl":
        name = data.get("name") or "realtime_rl_latest"
        cmd = base + ["realtime-rl", "--name", name]
        if data.get("date"):
            cmd.extend(["--date", data["date"]])
        if data.get("refresh") == "1":
            cmd.append("--refresh")
        links.append({
            "label": "Real-Time RL Dashboard",
            "href": f"/files/runs/{name}/realtime_rl_dashboard.html",
            "primary": True,
        })
    elif action == "neural":
        name = data.get("name") or "neural_mlp_baseline"
        cmd = base + ["neural", "--name", name, "--max-iter", data.get("max_iter", "120")]
        links.append({"label": "Neural Baseline Files", "href": f"/files/runs/{name}/"})
    elif action == "evidence":
        name = data.get("name") or "latest"
        cmd = base + ["evidence", "--name", name, "--neural-name", data.get("neural_name", "neural_mlp_baseline")]
        links.append({
            "label": "Research Evidence Dashboard",
            "href": f"/files/runs/{name}/research_evidence_dashboard.html",
            "primary": True,
        })
        links.append({"label": "Research Evidence Files", "href": f"/files/runs/{name}/"})
    elif action == "optimize":
        name = data.get("name") or "plant_optimization"
        cmd = base + [
            "optimize", "--start-date", data["start_date"], "--end-date", data["end_date"],
            "--name", name, "--plant-intervals", data.get("plant_intervals", "all"),
            "--reserve-margin", data.get("reserve_margin", "0.03"),
            "--capacity-margin", data.get("capacity_margin", "1.10")
        ]
        links.append({"label": "Optimization Runs", "href": "/files/runs/"})
        links.append({
            "label": "Interactive Dispatch Dashboard",
            "href": f"/files/runs/{name}_optimization/interactive_dispatch_dashboard.html",
        })
    else:
        raise ValueError("Unknown action")

    proc = subprocess.run(cmd, cwd=APP_ROOT, text=True, capture_output=True)
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode:
        raise RuntimeError(output)
    if action == "forecast":
        dashboard = make_forecast_dashboard(data["date"])
        if dashboard:
            rel = urllib.parse.quote(str(dashboard.relative_to(OUTPUTS_ROOT)))
            links.insert(0, {"label": "Forecast Dashboard", "href": f"/files/{rel}", "primary": True})
    return output, links


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/forecast":
            params = urllib.parse.parse_qs(parsed.query)
            target_date = params.get("date", [""])[0]
            try:
                data = forecast_payload(target_date)
                if data is None:
                    self._send(404, json.dumps({"error": "Forecast output not found"}).encode(), "application/json")
                    return
                self._send(200, json.dumps(data).encode(), "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            return
        if self.path == "/":
            dashboard = default_dashboard_path()
            if dashboard:
                self.send_response(302)
                self.send_header("Location", files_href(dashboard))
                self.end_headers()
                return
            tomorrow = (date.today() + timedelta(days=1)).isoformat()
            today = date.today().isoformat()
            self._send(200, INDEX.replace("__TOMORROW__", tomorrow).replace("__TODAY__", today).encode(), "text/html")
            return
        if self.path == "/lab":
            tomorrow = (date.today() + timedelta(days=1)).isoformat()
            today = date.today().isoformat()
            self._send(200, INDEX.replace("__TOMORROW__", tomorrow).replace("__TODAY__", today).encode(), "text/html")
            return
        if self.path.startswith("/files/"):
            rel = urllib.parse.unquote(self.path.removeprefix("/files/"))
            target = (OUTPUTS_ROOT / rel).resolve()
            if not str(target).startswith(str(OUTPUTS_ROOT.resolve())):
                self._send(403, b"Forbidden", "text/plain")
                return
            if target.is_dir():
                body = "<h1>Files</h1>" + "".join(
                    f'<p><a href="/files/{urllib.parse.quote(str((p.relative_to(OUTPUTS_ROOT))))}">{p.name}</a></p>'
                    for p in sorted(target.iterdir())
                )
                self._send(200, body.encode(), "text/html")
                return
            if target.exists():
                content_type = "application/octet-stream"
                if target.suffix == ".html":
                    content_type = "text/html"
                elif target.suffix == ".png":
                    content_type = "image/png"
                elif target.suffix == ".json":
                    content_type = "application/json"
                elif target.suffix == ".csv":
                    content_type = "text/csv"
                self._send(200, target.read_bytes(), content_type)
                return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not self.path.startswith("/run/"):
            self._send(404, b"Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length) or b"{}")
        action = self.path.rsplit("/", 1)[-1]
        try:
            output, links = run_command(action, data)
            payload = {"output": output, "links": links}
        except Exception as exc:
            payload = {"error": str(exc), "output": str(exc)}
        self._send(200, json.dumps(payload).encode(), "application/json")

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Bolivia AI Energy Lab running at http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
