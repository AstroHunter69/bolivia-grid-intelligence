#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from bolivia_grid_env import BoliviaGridEnv


def evaluate_policy(env: BoliviaGridEnv, policy: dict | None = None, default_action: int = 2) -> pd.DataFrame:
    state = env.reset()
    rows = []
    done = False
    while not done:
        key = env.discretize_state(state)
        if policy is None:
            action = default_action
        else:
            action = int(np.argmax(policy[key]))
        state, reward, done, info = env.step(action)
        info["reward"] = reward
        rows.append(info)
    return pd.DataFrame(rows)


def split_env(env: BoliviaGridEnv, train_fraction: float) -> tuple[BoliviaGridEnv, BoliviaGridEnv]:
    split_idx = int(len(env.df) * train_fraction)
    split_idx = max(1, min(split_idx, len(env.df) - 1))
    train = BoliviaGridEnv(env.df.iloc[:split_idx].copy(), config=env.config)
    test = BoliviaGridEnv(env.df.iloc[split_idx:].copy(), config=env.config)
    return train, test


def train_q_learning(
    env: BoliviaGridEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
) -> dict:
    q = defaultdict(lambda: np.zeros(env.n_actions, dtype=float))
    rng = np.random.default_rng(42)

    for _ in range(episodes):
        state = env.reset()
        done = False
        eps = epsilon

        while not done:
            key = env.discretize_state(state)
            if rng.random() < eps:
                action = int(rng.integers(0, env.n_actions))
            else:
                action = int(np.argmax(q[key]))

            next_state, reward, done, _ = env.step(action)
            next_key = env.discretize_state(next_state)
            target = reward if done else reward + gamma * float(np.max(q[next_key]))
            q[key][action] += alpha * (target - q[key][action])
            state = next_state

        epsilon *= epsilon_decay

    return q


def summarize(label: str, df: pd.DataFrame) -> dict:
    return {
        "policy": label,
        "rows": len(df),
        "mae_mw": df["absolute_error_mw"].mean(),
        "reward_sum": df["reward"].sum(),
        "avg_dispatch_mw": df["dispatch_mw"].mean(),
        "avg_real_demand_mw": df["real_demand_mw"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple Q-learning dispatch-support baseline")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--outdir", default=Path("runs/q_learning"), type=Path)
    parser.add_argument("--episodes", default=250, type=int)
    parser.add_argument("--alpha", default=0.10, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--epsilon", default=0.25, type=float)
    parser.add_argument("--epsilon-decay", default=0.995, type=float)
    parser.add_argument("--train-fraction", default=0.70, type=float)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    full_env = BoliviaGridEnv.from_collector_dir(args.data_dir)
    train_env, test_env = split_env(full_env, args.train_fraction)
    q = train_q_learning(
        train_env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
    )

    baseline_train_env = BoliviaGridEnv(train_env.df.copy(), config=train_env.config)
    learned_train_env = BoliviaGridEnv(train_env.df.copy(), config=train_env.config)
    baseline_test_env = BoliviaGridEnv(test_env.df.copy(), config=test_env.config)
    learned_test_env = BoliviaGridEnv(test_env.df.copy(), config=test_env.config)

    baseline_train = evaluate_policy(baseline_train_env, policy=None, default_action=2)
    learned_train = evaluate_policy(learned_train_env, policy=q)
    baseline_test = evaluate_policy(baseline_test_env, policy=None, default_action=2)
    learned_test = evaluate_policy(learned_test_env, policy=q)

    baseline_train.to_csv(args.outdir / "baseline_train_policy_trace.csv", index=False)
    learned_train.to_csv(args.outdir / "q_learning_train_policy_trace.csv", index=False)
    baseline_test.to_csv(args.outdir / "baseline_test_policy_trace.csv", index=False)
    learned_test.to_csv(args.outdir / "q_learning_test_policy_trace.csv", index=False)

    summary = pd.DataFrame(
        [
            {"split": "train", **summarize("cndc_forecast_no_adjustment", baseline_train)},
            {"split": "train", **summarize("q_learning_discrete_reserve", learned_train)},
            {"split": "test", **summarize("cndc_forecast_no_adjustment", baseline_test)},
            {"split": "test", **summarize("q_learning_discrete_reserve", learned_test)},
        ]
    )
    summary["mae_improvement_mw"] = 0.0
    for split in summary["split"].unique():
        baseline_mae = summary[(summary["split"] == split) & (summary["policy"] == "cndc_forecast_no_adjustment")][
            "mae_mw"
        ].iloc[0]
        summary.loc[summary["split"] == split, "mae_improvement_mw"] = baseline_mae - summary.loc[
            summary["split"] == split, "mae_mw"
        ]
    summary.to_csv(args.outdir / "q_learning_summary.csv", index=False)

    print("================================")
    print("BoliviaGridEnv Q-Learning Smoke Test")
    print("================================")
    print(f"Train rows: {len(train_env.df)}")
    print(f"Test rows:  {len(test_env.df)}")
    print()
    print(summary.round(3).to_string(index=False))
    print()
    print(f"Saved: {args.outdir}")


if __name__ == "__main__":
    main()
