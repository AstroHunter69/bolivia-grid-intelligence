from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    mismatch_weight: float = 1.0
    cost_weight: float = 0.025
    ramp_weight: float = 0.10
    reserve_actions: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)


class BoliviaGridEnv:
    """
    A lightweight Gym-like environment for dispatch-support experiments.

    The agent receives a forecast-driven state and chooses a reserve/dispatch
    adjustment around the CNDC demand forecast. The environment scores that
    action against real demand, marginal cost, and ramping.

    This is intentionally a research prototype, not an operational grid model.
    """

    def __init__(self, frame: pd.DataFrame, config: EnvConfig | None = None):
        self.df = frame.sort_values("timestamp").reset_index(drop=True)
        self.config = config or EnvConfig()
        self.actions = np.array(self.config.reserve_actions, dtype=float)
        self.idx = 0
        self.previous_dispatch = None

        self.scale_mw = max(float(self.df["real_demand_mw"].mean()), 1.0)
        self.max_cost = max(float(self.df["marginal_cost"].max(skipna=True) or 1.0), 1.0)

    @classmethod
    def from_collector_dir(cls, data_dir: str | Path, config: EnvConfig | None = None) -> "BoliviaGridEnv":
        data_dir = Path(data_dir)

        demand = pd.read_csv(data_dir / "demand_15min.csv", parse_dates=["timestamp"])
        demand = demand.rename(columns={"total": "real_demand_mw"})

        demand_forecast = pd.read_csv(data_dir / "demand_forecast_hourly.csv", parse_dates=["timestamp"])
        demand_forecast = demand_forecast[["timestamp", "total"]].rename(columns={"total": "cndc_forecast_mw"})

        generation = pd.read_csv(data_dir / "generation_15min.csv", parse_dates=["timestamp"])
        generation = generation[["timestamp", "total"]].rename(columns={"total": "generation_mw"})

        marginal = pd.read_csv(data_dir / "marginal_cost_forecast_hourly.csv", parse_dates=["timestamp"])
        if "total" in marginal.columns:
            marginal = marginal[["timestamp", "total"]].rename(columns={"total": "marginal_cost"})
        else:
            value_cols = [c for c in marginal.columns if c != "timestamp"]
            marginal["marginal_cost"] = marginal[value_cols].mean(axis=1)
            marginal = marginal[["timestamp", "marginal_cost"]]

        base = demand[["timestamp", "real_demand_mw"]].merge(generation, on="timestamp", how="left")
        base = base.merge(demand_forecast, on="timestamp", how="left")
        base = base.merge(marginal, on="timestamp", how="left")
        base = base.sort_values("timestamp").reset_index(drop=True)

        hourly_cols = ["cndc_forecast_mw", "marginal_cost"]
        base[hourly_cols] = base[hourly_cols].ffill().bfill()
        base["generation_mw"] = base["generation_mw"].interpolate(limit_direction="both")
        base = base.dropna(subset=["real_demand_mw", "cndc_forecast_mw", "generation_mw", "marginal_cost"])

        base["hour"] = base["timestamp"].dt.hour
        base["day_of_week"] = base["timestamp"].dt.dayofweek
        base["forecast_error_lag_96"] = (
            base["real_demand_mw"] - base["cndc_forecast_mw"]
        ).shift(96).fillna(0)
        base["demand_lag_96"] = base["real_demand_mw"].shift(96).bfill()
        base["ramp_lag_1"] = base["real_demand_mw"].diff().fillna(0)

        return cls(base, config=config)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def reset(self, start_index: int = 0) -> np.ndarray:
        self.idx = start_index
        self.previous_dispatch = None
        return self._state()

    def done(self) -> bool:
        return self.idx >= len(self.df)

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict]:
        row = self.df.iloc[self.idx]
        action = float(self.actions[action_index])
        dispatch = float(row["cndc_forecast_mw"] * (1.0 + action))
        real_demand = float(row["real_demand_mw"])
        marginal_cost = float(row["marginal_cost"])

        mismatch_penalty = abs(dispatch - real_demand) / self.scale_mw
        cost_penalty = (dispatch / self.scale_mw) * (marginal_cost / self.max_cost)
        if self.previous_dispatch is None:
            ramp_penalty = 0.0
        else:
            ramp_penalty = abs(dispatch - self.previous_dispatch) / self.scale_mw

        reward = -(
            self.config.mismatch_weight * mismatch_penalty
            + self.config.cost_weight * cost_penalty
            + self.config.ramp_weight * ramp_penalty
        )

        info = {
            "timestamp": row["timestamp"],
            "action": action,
            "dispatch_mw": dispatch,
            "real_demand_mw": real_demand,
            "cndc_forecast_mw": float(row["cndc_forecast_mw"]),
            "absolute_error_mw": abs(dispatch - real_demand),
            "marginal_cost": marginal_cost,
        }

        self.previous_dispatch = dispatch
        self.idx += 1
        done = self.done()
        next_state = np.zeros(7, dtype=float) if done else self._state()
        return next_state, float(reward), done, info

    def _state(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        return np.array(
            [
                row["hour"] / 23.0,
                row["day_of_week"] / 6.0,
                row["cndc_forecast_mw"] / self.scale_mw,
                row["generation_mw"] / self.scale_mw,
                row["marginal_cost"] / self.max_cost,
                row["forecast_error_lag_96"] / self.scale_mw,
                row["demand_lag_96"] / self.scale_mw,
            ],
            dtype=float,
        )

    def discretize_state(self, state: np.ndarray) -> tuple[int, ...]:
        bins = [
            np.digitize(state[0], [0.25, 0.5, 0.75]),
            np.digitize(state[1], [0.5]),
            np.digitize(state[2], [0.85, 1.0, 1.15]),
            np.digitize(state[3], [0.85, 1.0, 1.15]),
            np.digitize(state[4], [0.33, 0.66]),
            np.digitize(state[5], [-0.05, 0.0, 0.05]),
            np.digitize(state[6], [0.85, 1.0, 1.15]),
        ]
        return tuple(int(x) for x in bins)
