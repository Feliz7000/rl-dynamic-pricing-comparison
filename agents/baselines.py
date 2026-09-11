"""Non-learning reference agents. Essential lower-bound comparators: without
them, an algorithm's measured DRCR has no reference point to be judged
against."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Samples a uniformly random action every step."""

    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def select_action(self, state, eval_mode: bool = False):
        return self.action_space.sample()

    def observe(self, state, action, reward, next_state, done) -> None:
        pass

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass


class StaticPriceAgent(BaseAgent):
    """Always repeats the price the product entered the step at (i.e. no
    price change) -- a "do nothing" / keep-current-price reference, distinct
    from acting randomly."""

    def __init__(self, action_mode: str, k_buckets: int = 11):
        self.action_mode = action_mode
        # bucket grid is symmetric around the entering price (see
        # PricingEnv._decode_action); the middle index is a 0% change.
        self.middle_bucket = k_buckets // 2

    def select_action(self, state, eval_mode: bool = False):
        if self.action_mode == "discrete":
            return self.middle_bucket
        return np.array([0.0], dtype=np.float32)

    def observe(self, state, action, reward, next_state, done) -> None:
        pass

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass
