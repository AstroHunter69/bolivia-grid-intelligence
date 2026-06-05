#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def mae(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.abs().mean()) if len(clean) else float("nan")


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        pass
    return f"{value:,.{digits}f}"


def build_forecast_evidence() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    predictions = read_csv(ROOT / "backtest_predictions.csv")
    daily = read_csv(ROOT / "backtest_daily_summary.csv")
    if predictions is None or daily is None:
        return {}, pd.DataFrame(), pd.DataFrame()

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions = predictions.sort_values("timestamp")
    predictions["model_abs_error_mw"] = predictions["model_error_mw"].abs()
    predictions["cndc_abs_error_mw"] = predictions["cndc_error_mw"].abs()
    predictions["hour"] = predictions["timestamp"].dt.hour
    predictions["date"] = predictions["timestamp"].dt.date.astype(str)

    actual_by_time = predictions.set_index("timestamp")["target_demand_mw"]
    predictions["previous_day_baseline_mw"] = predictions["timestamp"].map(
        lambda ts: actual_by_time.get(ts - pd.Timedelta(days=1), np.nan)
    )
    predictions["same_weekday_baseline_mw"] = predictions["timestamp"].map(
        lambda ts: actual_by_time.get(ts - pd.Timedelta(days=7), np.nan)
    )
    predictions["previous_day_error_mw"] = predictions["previous_day_baseline_mw"] - predictions["target_demand_mw"]
    predictions["same_weekday_error_mw"] = predictions["same_weekday_baseline_mw"] - predictions["target_demand_mw"]

    baseline_rows = [
        {"model": "Bolivia Demand Model", "mae_mw": mae(predictions["model_error_mw"])},
        {"model": "CNDC forecast", "mae_mw": mae(predictions["cndc_error_mw"])},
        {"model": "Previous-day naive", "mae_mw": mae(predictions["previous_day_error_mw"])},
        {"model": "Same-weekday naive", "mae_mw": mae(predictions["same_weekday_error_mw"])},
    ]
    baseline = pd.DataFrame(baseline_rows)
    best_baseline = baseline.loc[baseline["mae_mw"].idxmin()].to_dict() if baseline["mae_mw"].notna().any() else {}

    hourly = (
        predictions.groupby("hour")
        .agg(
            model_mae_mw=("model_abs_error_mw", "mean"),
            cndc_mae_mw=("cndc_abs_error_mw", "mean"),
            model_bias_mw=("model_error_mw", "mean"),
            cndc_bias_mw=("cndc_error_mw", "mean"),
        )
        .reset_index()
    )
    daily = daily.copy()
    summary = {
        "rows": int(len(predictions)),
        "days": int(daily["date"].nunique()),
        "model_mae_mw": float(predictions["model_abs_error_mw"].mean()),
        "cndc_mae_mw": float(predictions["cndc_abs_error_mw"].mean()),
        "improvement_mw": float(predictions["cndc_abs_error_mw"].mean() - predictions["model_abs_error_mw"].mean()),
        "improvement_pct": float(
            100
            * (predictions["cndc_abs_error_mw"].mean() - predictions["model_abs_error_mw"].mean())
            / max(predictions["cndc_abs_error_mw"].mean(), 1e-9)
        ),
        "model_bias_mw": float(predictions["model_error_mw"].mean()),
        "best_baseline": best_baseline,
    }
    return summary, baseline, hourly


def build_audit_evidence() -> dict:
    leakage = read_csv(ROOT / "docs" / "research_metrics" / "leakage_audit.csv")
    stability = read_csv(ROOT / "docs" / "research_metrics" / "daily_stability.csv")
    optuna_report = ROOT / "runs" / "optuna" / "optuna_report.json"
    out = {}
    if leakage is not None:
        out["leakage_features_checked"] = int(len(leakage))
        out["leakage_features_passing"] = int(leakage["passes"].fillna(False).sum())
        out["leakage_passed"] = bool(leakage["passes"].fillna(False).all())
    if stability is not None:
        out["stability_days"] = int(stability["date"].nunique())
        out["stability_model_mae_mw"] = float(stability["model_mae_mw"].mean())
        out["stability_cndc_mae_mw"] = float(stability["cndc_mae_mw"].mean())
    if optuna_report.exists():
        report = json.loads(optuna_report.read_text())
        out["optuna_note"] = report.get("note", "Optuna smoke test available.")
        baseline = report.get("baseline", {})
        best = report.get("best", {})
        out["optuna_baseline_score"] = baseline.get("score")
        out["optuna_best_score"] = best.get("score")
    return out


def build_rl_evidence() -> dict:
    candidates = [
        ROOT / "runs" / "apr17_may26_40d_aggregate_q_learning" / "q_learning_summary.csv",
        ROOT / "runs" / "may_rl_experiment_q_learning" / "q_learning_summary.csv",
    ]
    summary = next((read_csv(path) for path in candidates if path.exists()), None)
    out = {
        "state": [
            "time of day and day of week",
            "CNDC demand forecast",
            "real-time generation level",
            "marginal cost proxy",
            "previous-day forecast error",
            "previous-day demand",
        ],
        "action": "choose a reserve adjustment around the forecast: -10%, -5%, 0%, +5%, +10%",
        "reward": "negative weighted penalty for demand mismatch, generation cost, and ramping",
        "constraints": "capacity limits, current renewable availability, reserve margin, and non-operational prototype status",
    }
    if summary is not None:
        out["summary_rows"] = summary.to_dict("records")
        test = summary[summary["split"].eq("test")]
        if len(test):
            baseline = test[test["policy"].eq("cndc_forecast_no_adjustment")]
            learned = test[test["policy"].str.contains("q_learning", case=False)]
            if len(baseline) and len(learned):
                out["test_baseline_mae_mw"] = float(baseline["mae_mw"].iloc[0])
                out["test_q_learning_mae_mw"] = float(learned["mae_mw"].iloc[0])
                out["test_improvement_mw"] = out["test_baseline_mae_mw"] - out["test_q_learning_mae_mw"]
    return out


def svg_bar_chart(rows: list[dict], label_key: str, value_key: str, color: str = "#0b7285") -> str:
    rows = [r for r in rows if r.get(value_key) is not None and not np.isnan(r.get(value_key))]
    if not rows:
        return "<p class='muted'>No comparable rows available.</p>"
    max_value = max(r[value_key] for r in rows) or 1
    h = max(190, 42 * len(rows) + 40)
    parts = [f'<svg viewBox="0 0 760 {h}" class="chart">']
    parts.append(f'<rect width="760" height="{h}" fill="#fff"/>')
    for i, row in enumerate(rows):
        y = 26 + i * 42
        w = 470 * row[value_key] / max_value
        parts.append(f'<text x="0" y="{y+16}" font-size="13" fill="#334">{row[label_key]}</text>')
        parts.append(f'<rect x="190" y="{y}" width="{w:.1f}" height="18" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{200+w:.1f}" y="{y+15}" font-size="12" fill="#334">{fmt(row[value_key])} MW</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_hourly_chart(hourly: pd.DataFrame) -> str:
    if hourly.empty:
        return "<p class='muted'>No hourly diagnostics available.</p>"
    max_y = float(hourly[["model_mae_mw", "cndc_mae_mw"]].max().max()) or 1
    parts = ['<svg viewBox="0 0 900 330" class="chart">', '<rect width="900" height="330" fill="#fff"/>']
    x0, y0, w, h = 48, 24, 820, 230
    for i in range(5):
        y = y0 + i * h / 4
        parts.append(f'<line x1="{x0}" x2="{x0+w}" y1="{y}" y2="{y}" stroke="#e9eef2"/>')
    for key, color, dash in [("cndc_mae_mw", "#868e96", "6 5"), ("model_mae_mw", "#2b8a3e", "")]:
        pts = []
        for _, row in hourly.iterrows():
            x = x0 + row["hour"] / 23 * w
            y = y0 + h - row[key] / max_y * h
            pts.append(f'{x:.1f},{y:.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
    for hour in (0, 6, 12, 18, 23):
        x = x0 + hour / 23 * w
        parts.append(f'<text x="{x-8:.1f}" y="286" font-size="11" fill="#60717d">{hour:02d}</text>')
    parts.append('<circle cx="610" cy="302" r="4" fill="#2b8a3e"/><text x="620" y="306" font-size="12" fill="#334">Bolivia Demand Model</text>')
    parts.append('<circle cx="690" cy="302" r="4" fill="#868e96"/><text x="700" y="306" font-size="12" fill="#334">CNDC</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(forecast: dict, baseline: pd.DataFrame, hourly: pd.DataFrame, audit: dict, rl: dict, neural: dict | None) -> str:
    baseline_rows = baseline.to_dict("records") if not baseline.empty else []
    neural_card = ""
    if neural:
        neural_card = f"""
        <div class="card">
          <div class="label">Neural Baseline</div>
          <div class="value">{neural.get('status', 'available')}</div>
          <p>{neural.get('summary', 'A compact MLP baseline is included as a comparison, not a replacement for XGBoost.')}</p>
        </div>"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bolivia AI Energy Research Evidence</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f8fa; color:#17212b; }}
    header {{ background:white; border-bottom:1px solid #d9e3e9; padding:26px 32px 18px; }}
    main {{ max-width:1180px; margin:0 auto; padding:22px 28px 40px; }}
    h1 {{ margin:0; font-size:30px; }}
    h2 {{ margin:0 0 12px; font-size:20px; }}
    p {{ color:#5f6f79; line-height:1.48; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .card,.panel {{ background:white; border:1px solid #d9e3e9; border-radius:8px; padding:16px; }}
    .label {{ color:#60717d; font-size:12px; }}
    .value {{ font-size:23px; font-weight:760; margin-top:4px; }}
    .chart {{ width:100%; height:auto; display:block; }}
    .muted {{ color:#60717d; font-size:13px; }}
    code {{ background:#edf3f6; padding:2px 5px; border-radius:4px; }}
    ul {{ color:#4d5f69; line-height:1.5; }}
    @media (max-width:900px) {{ .grid,.two {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Bolivia AI Energy Research Evidence</h1>
    <p>Forecasting, backtesting, RL framing, and dispatch-support evidence for Bolivia electricity demand and dispatch analysis.</p>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div class="label">Backtest Rows</div><div class="value">{forecast.get('rows', 0)}</div></div>
      <div class="card"><div class="label">BDM MAE</div><div class="value">{fmt(forecast.get('model_mae_mw'))} MW</div></div>
      <div class="card"><div class="label">CNDC MAE</div><div class="value">{fmt(forecast.get('cndc_mae_mw'))} MW</div></div>
      <div class="card"><div class="label">Improvement</div><div class="value">{fmt(forecast.get('improvement_pct'))}%</div></div>
    </div>

    <section class="panel">
      <h2>Baselines</h2>
      <p>Bolivia Demand Model is compared with CNDC and simple naive baselines. This matters because a model only earns credibility when it beats simple alternatives, not just because it is more complex.</p>
      {svg_bar_chart(baseline_rows, "model", "mae_mw", "#0b7285")}
    </section>

    <div class="two" style="margin-top:16px">
      <section class="panel">
        <h2>Error By Hour</h2>
        <p>Hourly MAE highlights where the model is strongest and where future work should focus.</p>
        {svg_hourly_chart(hourly)}
      </section>
      <section class="panel">
        <h2>Audit Status</h2>
        <div class="grid" style="grid-template-columns:1fr 1fr">
          <div class="card"><div class="label">Leakage Features Passed</div><div class="value">{audit.get('leakage_features_passing', 'n/a')}/{audit.get('leakage_features_checked', 'n/a')}</div></div>
          <div class="card"><div class="label">Stability Days</div><div class="value">{audit.get('stability_days', 'n/a')}</div></div>
        </div>
        <p>Leakage checks compare lag and rolling features against independently recomputed values. Stability testing records daily behavior over a broader holdout period.</p>
        <p class="muted">Optuna note: {audit.get('optuna_note', 'not run yet')}</p>
      </section>
    </div>

    <section class="panel" style="margin-top:16px">
      <h2>RL Environment Framing</h2>
      <p>The RL part is deliberately framed as a dispatch-support simulation. It does not claim to operate Bolivia's grid; it defines a tractable environment for studying cost, reserve, and renewable-integration tradeoffs.</p>
      <div class="two">
        <div>
          <p><b>State:</b></p>
          <ul>{"".join(f"<li>{item}</li>" for item in rl.get("state", []))}</ul>
        </div>
        <div>
          <p><b>Action:</b> {rl.get("action")}</p>
          <p><b>Reward:</b> {rl.get("reward")}</p>
          <p><b>Constraints:</b> {rl.get("constraints")}</p>
        </div>
      </div>
    </section>

    <div class="grid" style="margin-top:16px">
      <div class="card"><div class="label">RL Test Baseline MAE</div><div class="value">{fmt(rl.get('test_baseline_mae_mw'))} MW</div></div>
      <div class="card"><div class="label">RL Test Policy MAE</div><div class="value">{fmt(rl.get('test_q_learning_mae_mw'))} MW</div></div>
      <div class="card"><div class="label">RL Improvement</div><div class="value">{fmt(rl.get('test_improvement_mw'))} MW</div></div>
      {neural_card or '<div class="card"><div class="label">Neural Baseline</div><div class="value">script added</div><p>Run the compact MLP baseline to compare against XGBoost on the same tabular feature set.</p></div>'}
    </div>
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research evidence report")
    parser.add_argument("--outdir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--neural-report", type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    forecast, baseline, hourly = build_forecast_evidence()
    audit = build_audit_evidence()
    rl = build_rl_evidence()
    neural = None
    if args.neural_report and args.neural_report.exists():
        neural = json.loads(args.neural_report.read_text())

    summary = {"forecast": forecast, "baselines": baseline.to_dict("records"), "audit": audit, "rl": rl, "neural": neural}
    (args.outdir / "research_evidence_summary.json").write_text(json.dumps(summary, indent=2))
    (args.outdir / "research_evidence_dashboard.html").write_text(render_html(forecast, baseline, hourly, audit, rl, neural))
    print("================================")
    print("Research Evidence Report")
    print("================================")
    print(f"Dashboard: {args.outdir / 'research_evidence_dashboard.html'}")
    print(f"Summary:   {args.outdir / 'research_evidence_summary.json'}")


if __name__ == "__main__":
    main()
