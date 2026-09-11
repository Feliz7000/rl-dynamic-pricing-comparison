"""Gymnasium environment for the dynamic-pricing MDP shared across all four
algorithms: state = product features (own attributes, recent sales/traffic,
pricing history, competitiveness), action = a price (discrete bucket or
continuous), reward = DRCR (Difference of Revenue Conversion Rate).

Each episode follows one product's rolled-forward trajectory. The first
WARMUP_MONTHS of real history seed rolling/lag/RCR context; the decision
horizon then proceeds using the demand model to simulate the outcome of
whatever price the agent chooses (which may differ from history). If real
months run out before `episode_horizon` steps, the episode extends via a
small-noise bootstrapped rollforward of exogenous context (competitor
prices, category traffic) -- a training-time sample-efficiency trick, never
used when evaluating on held-out test products (see evaluation/evaluate.py).

Performance note: per-step state is kept as a plain dict (`self._context`),
not a pandas Series/DataFrame row. Mutating a DataFrame row in place via
`.iloc[...] = ...` every step was measured at ~30ms/call (pandas' block
manager has to split/re-align columns of mixed dtype on each partial
assignment) -- using a dict for the hot path avoids that entirely and is
what makes a 50k-step training run practical. `self._history` (a DataFrame)
is only ever read from, never mutated, for the real logged months.
"""
from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from env.data_loader import DRCR_TAU, STATE_FEATURE_COLUMNS, WARMUP_MONTHS, StateScaler
from env.demand_model import DemandModel

REWARD_SCALE = 100.0


class PricingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        demand_model: DemandModel,
        scaler: StateScaler,
        product_ids: list[str],
        action_mode: Literal["discrete", "continuous"] = "continuous",
        k_buckets: int = 11,
        price_bound_frac: float = 0.25,
        episode_horizon: int = 20,
        allow_bootstrap_extension: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        if not product_ids:
            raise ValueError("product_ids must be non-empty")

        self.demand_model = demand_model
        self.scaler = scaler
        self.product_ids = list(product_ids)
        self.action_mode = action_mode
        self.k_buckets = k_buckets
        self.price_bound_frac = price_bound_frac
        self.episode_horizon = episode_horizon
        self.allow_bootstrap_extension = allow_bootstrap_extension

        self._rng = np.random.default_rng(seed)

        obs_dim = scaler.feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(k_buckets)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Precompute each product's real logged history as a list of plain
        # dicts once (at construction, not per-episode) -- reset() just
        # indexes into this, never touches pandas on the hot path.
        self._product_rows: dict[str, list[dict]] = {
            pid: df[df["product_id"] == pid].sort_values("t").to_dict("records") for pid in self.product_ids
        }

        # episode state
        self._pid: str | None = None
        self._real_history: list[dict] | None = None  # read-only real months for this episode
        self._context: dict | None = None  # current row's feature values (mutated each step)
        self._t_ptr: int = 0
        self._step_count: int = 0
        self._rcr_history: list[float] = []
        # Rolling window of *realized* (qty, revenue) each step, seeded from
        # real warmup months, then the demand model's own qty_hat/revenue_hat
        # thereafter -- independent of whether real history has run out, so
        # qty_roll3/revenue_roll3 stay well-defined through bootstrap
        # extension without re-indexing into the (fixed-length) real history.
        self._qty_history: list[float] = []
        self._revenue_history: list[float] = []
        self._num_clipped: int = 0

    def _decode_action(self, action, entering_price: float) -> float:
        if self.action_mode == "discrete":
            frac_grid = np.linspace(-self.price_bound_frac, self.price_bound_frac, self.k_buckets)
            multiplier = 1.0 + frac_grid[int(action)]
        else:
            a = float(np.clip(action[0] if hasattr(action, "__len__") else action, -1.0, 1.0))
            multiplier = 1.0 + a * self.price_bound_frac
        return entering_price * multiplier

    def _make_observation(self, context: dict) -> np.ndarray:
        return self.scaler.transform_row(context)

    def _bootstrap_next_context(self, last: dict) -> dict:
        """Synthesizes one more exogenous-context row past the real logged
        history by perturbing the last known row -- training-time only."""
        new_row = dict(last)
        for k in (1, 2, 3):
            new_row[f"comp_{k}"] = max(0.01, last[f"comp_{k}"] * (1 + self._rng.normal(0, 0.03)))
        new_row["traffic"] = max(1e-6, last["traffic"] * (1 + self._rng.normal(0, 0.05)))
        new_row["t"] = last["t"] + 1
        return new_row

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        pid = options.get("product_id")
        if pid is None:
            pid = self.product_ids[self._rng.integers(len(self.product_ids))]
        self._pid = pid
        self._real_history = self._product_rows[pid]

        if len(self._real_history) <= WARMUP_MONTHS:
            raise ValueError(f"Product {pid} has only {len(self._real_history)} rows, need > {WARMUP_MONTHS} for warmup")

        self._t_ptr = WARMUP_MONTHS
        self._step_count = 0
        self._num_clipped = 0
        self._rcr_history = [row["rcr"] for row in self._real_history[: self._t_ptr + 1]]
        self._qty_history = [row["qty"] for row in self._real_history[: self._t_ptr + 1]]
        self._revenue_history = [row["total_price"] for row in self._real_history[: self._t_ptr + 1]]
        self._context = dict(self._real_history[self._t_ptr])

        obs = self._make_observation(self._context)
        info = {"product_id": pid, "t": self._t_ptr}
        return obs, info

    def step(self, action):
        current = self._context
        entering_price = float(current["lag_price"])
        candidate_price = self._decode_action(action, entering_price)

        qty_hat, clipped_price, was_clipped = self.demand_model.predict_qty_single(current, candidate_price)
        if was_clipped:
            self._num_clipped += 1
        revenue_hat = clipped_price * qty_hat
        traffic = max(float(current["traffic"]), 1e-6)
        rcr_t = revenue_hat / traffic
        self._rcr_history.append(rcr_t)

        lookback_idx = len(self._rcr_history) - 1 - DRCR_TAU
        rcr_prev = self._rcr_history[lookback_idx] if lookback_idx >= 0 else rcr_t
        drcr = rcr_t - rcr_prev
        reward = REWARD_SCALE * drcr

        self._step_count += 1
        terminated = self._step_count >= self.episode_horizon
        truncated = False

        next_t = self._t_ptr + 1
        if next_t < len(self._real_history):
            next_context = dict(self._real_history[next_t])
        elif self.allow_bootstrap_extension and not terminated:
            next_context = self._bootstrap_next_context(current)
        else:
            terminated = True
            next_context = dict(current)

        # roll forward own-history-dependent fields for the next decision
        next_context["lag_price_2"] = current["lag_price"]
        next_context["lag_price"] = clipped_price
        next_context["price_change_pct"] = (
            (next_context["lag_price"] - next_context["lag_price_2"]) / next_context["lag_price_2"]
            if next_context["lag_price_2"]
            else 0.0
        )
        for k in (1, 2, 3):
            next_context[f"price_gap_{k}"] = next_context["lag_price"] - next_context[f"comp_{k}"]
        next_context["price_gap_mean"] = float(
            np.mean([next_context[f"price_gap_{k}"] for k in (1, 2, 3)])
        )
        prices = [next_context["lag_price"], next_context["comp_1"], next_context["comp_2"], next_context["comp_3"]]
        next_context["price_rank"] = float(np.argsort(np.argsort(prices))[0]) / (len(prices) - 1)

        self._qty_history.append(qty_hat)
        self._revenue_history.append(revenue_hat)
        next_context["qty_roll3"] = float(np.mean(self._qty_history[-3:]))
        next_context["revenue_roll3"] = float(np.mean(self._revenue_history[-3:]))
        next_context["prev_drcr"] = drcr

        self._context = next_context
        self._t_ptr = next_t if next_t < len(self._real_history) else self._t_ptr + 1

        obs = self._make_observation(next_context)
        info = {
            "product_id": self._pid,
            "t": self._t_ptr,
            "price": clipped_price,
            "qty_hat": qty_hat,
            "rcr": rcr_t,
            "drcr": drcr,
            "clipped": was_clipped,
        }
        return obs, reward, terminated, truncated, info
