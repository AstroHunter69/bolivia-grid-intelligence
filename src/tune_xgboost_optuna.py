#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


TARGET = "target_demand_mw"
DEFAULT_DATASET = Path("data/processed/cndc_ml_dataset.csv")
BASELINE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def load_data(dataset: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset, parse_dates=["timestamp"]).sort_values("timestamp")
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ["timestamp", TARGET]]


def make_folds(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    candidates = [
        ("fold_jul_aug_2025", "2025-06-30", "2025-08-31"),
        ("fold_sep_oct_2025", "2025-08-31", "2025-10-31"),
        ("fold_nov_dec_2025", "2025-10-31", "2025-12-31"),
        ("fold_jan_2026", "2025-12-31", "2026-01-31"),
    ]
    max_ts = df["timestamp"].max()
    folds = []
    for name, train_end, val_end in candidates:
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
        if val_end_ts <= max_ts and (df["timestamp"] <= train_end_ts).any():
            folds.append((name, train_end_ts, val_end_ts))
    if not folds:
        split = int(len(df) * 0.8)
        train_end = df.iloc[split]["timestamp"]
        val_end = df["timestamp"].max()
        folds.append(("fallback_last_20pct", train_end, val_end))
    return folds


def weighted_score(y_true: pd.Series, pred: np.ndarray, ts: pd.Series) -> dict[str, float]:
    errors = np.abs(y_true.to_numpy() - pred)
    hours = ts.dt.hour + ts.dt.minute / 60
    morning = (hours >= 6) & (hours <= 11)
    evening = (hours >= 18) & (hours <= 22)
    ramp = morning | evening

    mae = float(errors.mean())
    morning_mae = float(errors[morning].mean()) if morning.any() else mae
    ramp_mae = float(errors[ramp].mean()) if ramp.any() else mae

    score = 0.65 * mae + 0.25 * morning_mae + 0.10 * ramp_mae
    return {
        "score": float(score),
        "mae": mae,
        "morning_mae": morning_mae,
        "ramp_mae": ramp_mae,
    }


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators", 500, 1800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 8.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def evaluate_params(df: pd.DataFrame, features: list[str], folds, params: dict) -> dict:
    fold_results = []
    for name, train_end, val_end in folds:
        train = df[df["timestamp"] <= train_end]
        val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
        if train.empty or val.empty:
            continue
        model = XGBRegressor(**params)
        model.fit(train[features], train[TARGET], verbose=False)
        pred = model.predict(val[features])
        metrics = weighted_score(val[TARGET], pred, val["timestamp"])
        fold_results.append({"fold": name, **metrics})

    score = float(np.mean([r["score"] for r in fold_results]))
    mae = float(np.mean([r["mae"] for r in fold_results]))
    morning_mae = float(np.mean([r["morning_mae"] for r in fold_results]))
    ramp_mae = float(np.mean([r["ramp_mae"] for r in fold_results]))
    return {
        "score": score,
        "mae": mae,
        "morning_mae": morning_mae,
        "ramp_mae": ramp_mae,
        "folds": fold_results,
    }


def train_final(df: pd.DataFrame, features: list[str], params: dict, train_until: str) -> XGBRegressor:
    train = df[df["timestamp"] <= pd.Timestamp(train_until)]
    model = XGBRegressor(**params)
    model.fit(train[features], train[TARGET], verbose=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Bolivia Demand Model demand model with Optuna")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--outdir", type=Path, default=Path("runs/optuna"))
    parser.add_argument("--train-final-until", default="2026-01-31")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.dataset)
    features = feature_columns(df)
    folds = make_folds(df)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = evaluate_params(df, features, folds, params)
        trial.set_user_attr("mae", result["mae"])
        trial.set_user_attr("morning_mae", result["morning_mae"])
        trial.set_user_attr("ramp_mae", result["ramp_mae"])
        return result["score"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_params = suggest_params(study.best_trial)
    best_result = evaluate_params(df, features, folds, best_params)
    baseline_result = evaluate_params(df, features, folds, BASELINE_PARAMS)
    report = {
        "dataset": str(args.dataset),
        "features": features,
        "folds": [
            {"name": name, "train_until": str(train_end), "validate_until": str(val_end)}
            for name, train_end, val_end in folds
        ],
        "best_value": study.best_value,
        "best_params": best_params,
        "best_result": best_result,
        "baseline_params": BASELINE_PARAMS,
        "baseline_result": baseline_result,
        "note": "score = 0.65*overall_MAE + 0.25*morning_MAE + 0.10*ramp_MAE",
    }
    (args.outdir / "optuna_report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(features, args.outdir / "bolivia_demand_features.pkl")

    trials = study.trials_dataframe()
    trials.to_csv(args.outdir / "optuna_trials.csv", index=False)

    if args.save_model:
        final_model = train_final(df, features, best_params, args.train_final_until)
        joblib.dump(final_model, args.outdir / "bolivia_demand_model_optuna.pkl")

    print("================================")
    print("Optuna Bolivia Demand Model Tuning Complete")
    print("================================")
    print(f"Trials:          {args.trials}")
    print(f"Best score:      {best_result['score']:.3f}")
    print(f"Overall MAE:     {best_result['mae']:.3f} MW")
    print(f"Morning MAE:     {best_result['morning_mae']:.3f} MW")
    print(f"Ramp MAE:        {best_result['ramp_mae']:.3f} MW")
    print(f"Baseline score:  {baseline_result['score']:.3f}")
    print(f"Baseline MAE:    {baseline_result['mae']:.3f} MW")
    print(f"Saved:           {args.outdir}")


if __name__ == "__main__":
    main()
