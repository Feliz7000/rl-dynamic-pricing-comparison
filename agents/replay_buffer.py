"""Uniform experience replay buffer shared by DQN, DDPG, and SAC."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, discrete: bool = False):
        self.capacity = capacity
        self.discrete = discrete
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,) if discrete else (capacity, action_dim), dtype=np.int64 if discrete else np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done) -> None:
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }
