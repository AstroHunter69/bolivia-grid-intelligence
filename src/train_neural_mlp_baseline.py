#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "target_demand_mw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact neural MLP baseline for Bolivia Demand Model feature set")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/cndc_ml_dataset.csv"))
    parser.add_argument("--features", type=Path, default=Path("models/bolivia_demand_features.pkl"))
    parser.add_argument("--outdir", type=Path, default=Path("runs/neural_mlp_baseline"))
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--val-end", default="2026-01-31")
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset, parse_dates=["timestamp"]).sort_values("timestamp")
    features = joblib.load(args.features)

    train = df[df["timestamp"] <= args.train_end].copy()
    val = df[(df["timestamp"] > args.train_end) & (df["timestamp"] <= args.val_end)].copy()
    test = df[df["timestamp"] > args.val_end].copy()

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=args.max_iter,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=12,
            random_state=42,
            verbose=False,
        ),
    )
    model.fit(train[features], train[TARGET])

    rows = []
    for split, frame in [("validation", val), ("test", test)]:
        if frame.empty:
            continue
        pred = model.predict(frame[features])
        rows.append(
            {
                "split": split,
                "rows": int(len(frame)),
                "mae_mw": float(mean_absolute_error(frame[TARGET], pred)),
                "bias_mw": float(np.mean(pred - frame[TARGET])),
                "start": str(frame["timestamp"].min()),
                "end": str(frame["timestamp"].max()),
            }
        )
        out = frame[["timestamp", TARGET]].copy()
        out["neural_mlp_prediction_mw"] = pred
        out["error_mw"] = out["neural_mlp_prediction_mw"] - out[TARGET]
        out.to_csv(args.outdir / f"neural_mlp_{split}_predictions.csv", index=False)

    report = {
        "status": "trained",
        "summary": "Compact neural MLP baseline trained on the same engineered Bolivia Demand Model tabular features. It is a comparison point, not a replacement unless it outperforms XGBoost in backtests.",
        "dataset": str(args.dataset),
        "features": len(features),
        "architecture": "StandardScaler + MLPRegressor(hidden_layer_sizes=(96, 48))",
        "splits": rows,
    }
    (args.outdir / "neural_mlp_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(args.outdir / "neural_mlp_summary.csv", index=False)
    joblib.dump(model, args.outdir / "neural_mlp_baseline.pkl")

    print("================================")
    print("Neural MLP Baseline")
    print("================================")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print()
    print(f"Saved: {args.outdir}")


if __name__ == "__main__":
    main()
