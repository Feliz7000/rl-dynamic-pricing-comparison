"""DDPG (Lillicrap et al., 2015): off-policy, deterministic actor for exact
continuous prices, critic trained by TD/Bellman error, soft target networks
for both actor and critic. Exploration is external (Ornstein-Uhlenbeck
noise, annealed over training) since the actor itself is deterministic. See
report Section 2 (Shawn)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.networks import DeterministicActor, QCritic, hard_update, soft_update
from agents.replay_buffer import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration
    noise, annealed linearly from sigma_start to sigma_end."""

    def __init__(self, action_dim: int, theta: float = 0.15, sigma_start: float = 0.2, sigma_end: float = 0.05, decay_steps: int = 25_000, seed: int = 0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(action_dim, dtype=np.float32)
        self._step = 0

    def _sigma(self) -> float:
        frac = min(1.0, self._step / self.decay_steps)
        return self.sigma_start + frac * (self.sigma_end - self.sigma_start)

    def sample(self) -> np.ndarray:
        sigma = self._sigma()
        dx = self.theta * (-self.state) + sigma * self.rng.standard_normal(self.action_dim)
        self.state = self.state + dx
        self._step += 1
        return self.state.copy()

    def reset(self) -> None:
        self.state = np.zeros(self.action_dim, dtype=np.float32)


class DDPGAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        tau: float = 0.005,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_dim = action_dim

        self.actor = DeterministicActor(state_dim, action_dim).to(self.device)
        self.actor_target = DeterministicActor(state_dim, action_dim).to(self.device)
        hard_update(self.actor_target, self.actor)
        self.critic = QCritic(state_dim, action_dim).to(self.device)
        self.critic_target = QCritic(state_dim, action_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        self.noise = OUNoise(action_dim, seed=seed)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, eval_mode: bool = False):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_t).cpu().numpy()[0]
        if not eval_mode:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        if done:
            self.noise.reset()

    def update(self) -> dict[str, float]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}

        batch = self.buffer.sample(self.batch_size, self.rng)
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        next_states = torch.as_tensor(batch["next_states"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q_values.mean().item(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
