"""DQN (Mnih et al., 2015): value-based, off-policy, discrete price/discount
buckets. Q(s,a|theta) trained via TD learning against a periodically-updated
target network, with experience replay to decorrelate updates. See report
Section 3 (Davis)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.networks import QNetwork, hard_update
from agents.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        target_update_interval: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 25_000,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.n_actions = n_actions

        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        hard_update(self.target_net, self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim=1, discrete=True)
        self.rng = np.random.default_rng(seed)

        self._global_step = 0
        self._update_count = 0

    def _epsilon(self) -> float:
        frac = min(1.0, self._global_step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, eval_mode: bool = False):
        eps = 0.0 if eval_mode else self._epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=-1).item())

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self._global_step += 1

    def update(self) -> dict[str, float]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}

        batch = self.buffer.sample(self.batch_size, self.rng)
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device).long()
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        next_states = torch.as_tensor(batch["next_states"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=-1).values
            target = rewards + self.gamma * (1.0 - dones) * next_q

        q_values = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.huber_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_interval == 0:
            hard_update(self.target_net, self.q_net)

        return {"loss": loss.item(), "epsilon": self._epsilon(), "q_mean": q_values.mean().item()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        hard_update(self.target_net, self.q_net)
