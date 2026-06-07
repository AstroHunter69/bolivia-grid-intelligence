#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PYTHON = Path(sys.executable)


def run(cmd: list[str], cwd: Path) -> None:
    print()
    print(">", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cndc-energy")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def collector_dir() -> Path:
    return SRC


def predictor_dir() -> Path:
    return SRC


def rl_dir() -> Path:
    return SRC


def optimizer_dir() -> Path:
    return SRC


def evidence_dir() -> Path:
    return SRC


def find_model_prediction(date: str) -> Path | None:
    candidates = [
        ROOT / "runs" / "forecasts" / f"prediction_{date}" / f"bdm_prediction_{date}.csv",
        ROOT / "runs" / f"prediction_{date}" / f"bdm_prediction_{date}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def cmd_forecast(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON),
        "predict_demand.py",
        "--target-date",
        args.date,
        "--lookback-days",
        str(args.lookback_days),
        "--model",
        str(ROOT / "models" / "bolivia_demand_model.pkl"),
        "--features",
        str(ROOT / "models" / "bolivia_demand_features.pkl"),
        "--cache-dir",
        str(ROOT / "data" / "cache"),
        "--outdir",
        str(ROOT / "runs" / "forecasts"),
    ]
    if args.chart:
        command.append("--chart")
    run(command, predictor_dir())


def cmd_collect(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON),
        "collect_cndc_energy_data.py",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--outdir",
        str(ROOT / "data" / args.name),
        "--cache-dir",
        str(ROOT / "data" / "cache"),
    ]
    if args.plant:
        command.extend(["--include-plant-dispatch", "--plant-intervals", args.plant_intervals])
    if args.skip_token:
        command.append("--skip-token")
    run(command, collector_dir())


def cmd_rl(args: argparse.Namespace) -> None:
    dataset_name = args.name or f"rl_{args.start_date}_{args.end_date}".replace("-", "")
    run(
        [
            str(PYTHON),
            "collect_cndc_energy_data.py",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--outdir",
            str(ROOT / "data" / dataset_name),
            "--cache-dir",
            str(ROOT / "data" / "cache"),
            "--skip-token",
        ],
        collector_dir(),
    )
    run(
        [
            str(PYTHON),
            "train_q_learning.py",
            "--data-dir",
            str(ROOT / "data" / dataset_name),
            "--outdir",
            str(ROOT / "runs" / f"{dataset_name}_q_learning"),
            "--episodes",
            str(args.episodes),
            "--train-fraction",
            str(args.train_fraction),
        ],
        rl_dir(),
    )
    run(
        [
            str(PYTHON),
            "report_rl_results.py",
            "--run-dir",
            str(ROOT / "runs" / f"{dataset_name}_q_learning"),
        ],
        rl_dir(),
    )


def cmd_realtime_rl(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON),
        "realtime_rl_recommender.py",
        "--outdir",
        str(ROOT / "runs" / args.name),
        "--cache-dir",
        str(ROOT / "data" / "cache"),
    ]
    if args.date:
        command.extend(["--date", args.date])
    if args.refresh:
        command.append("--refresh")
    run(command, rl_dir())


def cmd_neural(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON),
        "train_neural_mlp_baseline.py",
        "--outdir",
        str(ROOT / "runs" / args.name),
        "--max-iter",
        str(args.max_iter),
    ]
    run(command, predictor_dir())


def cmd_evidence(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON),
        "build_research_report.py",
        "--outdir",
        str(ROOT / "runs" / args.name),
    ]
    neural_report = ROOT / "runs" / args.neural_name / "neural_mlp_report.json"
    if neural_report.exists():
        command.extend(["--neural-report", str(neural_report)])
    run(command, evidence_dir())


def cmd_optimize(args: argparse.Namespace) -> None:
    dataset_name = args.name or f"plant_{args.start_date}_{args.end_date}".replace("-", "")
    run(
        [
            str(PYTHON),
            "collect_cndc_energy_data.py",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--outdir",
            str(ROOT / "data" / dataset_name),
            "--cache-dir",
            str(ROOT / "data" / "cache"),
            "--include-plant-dispatch",
            "--plant-intervals",
            args.plant_intervals,
            "--skip-token",
        ],
        collector_dir(),
    )
    command = [
        str(PYTHON),
        "optimize_plant_dispatch.py",
        "--data-dir",
        str(ROOT / "data" / dataset_name),
        "--outdir",
        str(ROOT / "runs" / f"{dataset_name}_optimization"),
        "--capacity-margin",
        str(args.capacity_margin),
        "--reserve-margin",
        str(args.reserve_margin),
    ]
    model_prediction = args.model_prediction
    if model_prediction is None and args.start_date == args.end_date:
        model_prediction = find_model_prediction(args.start_date)
    if model_prediction:
        command.extend(["--model-prediction", str(model_prediction)])
    run(command, optimizer_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bolivia AI Energy local project runner")
    sub = parser.add_subparsers(required=True)

    forecast = sub.add_parser("forecast", help="Run Bolivia Demand Model demand forecast for a target date")
    forecast.add_argument("--date", required=True)
    forecast.add_argument("--lookback-days", default=21, type=int)
    forecast.add_argument("--chart", action="store_true")
    forecast.set_defaults(func=cmd_forecast)

    collect = sub.add_parser("collect", help="Collect CNDC data for a date range")
    collect.add_argument("--start-date", required=True)
    collect.add_argument("--end-date", required=True)
    collect.add_argument("--name", required=True)
    collect.add_argument("--plant", action="store_true")
    collect.add_argument("--plant-intervals", default="all")
    collect.add_argument("--skip-token", action="store_true")
    collect.set_defaults(func=cmd_collect)

    rl = sub.add_parser("rl", help="Run aggregate RL train/test experiment")
    rl.add_argument("--start-date", required=True)
    rl.add_argument("--end-date", required=True)
    rl.add_argument("--name")
    rl.add_argument("--episodes", default=400, type=int)
    rl.add_argument("--train-fraction", default=0.70, type=float)
    rl.set_defaults(func=cmd_rl)

    rt = sub.add_parser("realtime-rl", help="Run real-time RL-style dispatch recommendation")
    rt.add_argument("--date")
    rt.add_argument("--name", default="realtime_rl_latest")
    rt.add_argument("--refresh", action="store_true")
    rt.set_defaults(func=cmd_realtime_rl)

    neural = sub.add_parser("neural", help="Run compact neural MLP baseline")
    neural.add_argument("--name", default="neural_mlp_baseline")
    neural.add_argument("--max-iter", default=120, type=int)
    neural.set_defaults(func=cmd_neural)

    evidence = sub.add_parser("evidence", help="Build research evidence report")
    evidence.add_argument("--name", default="latest")
    evidence.add_argument("--neural-name", default="neural_mlp_baseline")
    evidence.set_defaults(func=cmd_evidence)

    opt = sub.add_parser("optimize", help="Run plant-level dispatch optimization")
    opt.add_argument("--start-date", required=True)
    opt.add_argument("--end-date", required=True)
    opt.add_argument("--name")
    opt.add_argument("--plant-intervals", default="all")
    opt.add_argument("--capacity-margin", default=1.10, type=float)
    opt.add_argument("--reserve-margin", default=0.03, type=float)
    opt.add_argument("--model-prediction", type=Path)
    opt.set_defaults(func=cmd_optimize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
