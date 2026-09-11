"""On-policy rollout buffer with Generalized Advantage Estimation, used only
by PPO (the other three agents are off-policy and use ReplayBuffer)."""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0

    def push(self, state, action, log_prob, reward, value, done) -> None:
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr >= self.capacity

    def __len__(self) -> int:
        return self._ptr

    def compute_advantages(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        n = self._ptr
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = self.values[t]
        returns = advantages + self.values[:n]
        return advantages, returns

    def get(self, last_value: float) -> dict[str, np.ndarray]:
        advantages, returns = self.compute_advantages(last_value)
        n = self._ptr
        return {
            "states": self.states[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "advantages": advantages,
            "returns": returns,
        }

    def reset(self) -> None:
        self._ptr = 0
