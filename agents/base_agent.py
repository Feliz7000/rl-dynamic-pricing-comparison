"""Shared agent interface so DQN, DDPG, PPO, and SAC are trained and
evaluated identically -- the report's whole point in Section 7 is a fair,
like-for-like comparison of algorithm design choices, not of differences in
training setup."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, eval_mode: bool = False):
        """Returns an action for the given state. In eval_mode, exploration
        (noise / sampling / epsilon) is disabled and the policy acts
        deterministically (or greedily)."""

    @abstractmethod
    def observe(self, state, action, reward, next_state, done) -> None:
        """Records one transition (replay-buffer push for off-policy agents,
        rollout-buffer append for PPO)."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Performs one learning update (if enough data is available) and
        returns a dict of scalar loss/metric values for logging. Returns an
        empty dict if no update was performed (e.g. still warming up)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        ...
