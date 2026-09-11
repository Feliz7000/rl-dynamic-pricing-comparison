"""SAC (Haarnoja et al., 2018): off-policy, maximum-entropy actor-critic for
continuous prices. Stochastic tanh-squashed Gaussian policy, twin critics
(minimum taken to reduce overestimation bias), and an automatically-tuned
temperature alpha -- directly answering the exploration-scheduling problem
DDPG solves manually with external noise. See report Section 5 (Felix).

This is the author's own assigned algorithm in the team report, so the math
here is deliberately spelled out step by step and covered by extra unit
tests (tests/test_agents_smoke.py) rather than compressed into helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.networks import GaussianPolicy, QCritic, hard_update, soft_update
from agents.replay_buffer import ReplayBuffer

LOG_PROB_EPS = 1e-6


class SACAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        target_entropy: float | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_dim = action_dim

        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.critic_1 = QCritic(state_dim, action_dim).to(self.device)
        self.critic_2 = QCritic(state_dim, action_dim).to(self.device)
        self.critic_1_target = QCritic(state_dim, action_dim).to(self.device)
        self.critic_2_target = QCritic(state_dim, action_dim).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr
        )

        # Auto-tuned temperature: target entropy defaults to -action_dim, the
        # standard heuristic (Haarnoja et al., 2018 App. D).
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _sample_action(self, states: torch.Tensor, deterministic: bool = False):
        """Reparameterized tanh-squashed Gaussian sample. Returns (action,
        log_prob) with the standard tanh Jacobian correction:
            log_prob = log N(u|mean,std) - sum(log(1 - tanh(u)^2 + eps))
        where u is the pre-squash sample and action = tanh(u)."""
        mean, log_std = self.actor(states)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mean), None
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + LOG_PROB_EPS).sum(dim=-1)
        return action, log_prob

    def select_action(self, state, eval_mode: bool = False):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self._sample_action(state_t, deterministic=eval_mode)
            return action.cpu().numpy()[0].astype(np.float32)

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict[str, float]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}

        batch = self.buffer.sample(self.batch_size, self.rng)
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        next_states = torch.as_tensor(batch["next_states"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        # --- critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            q1_next = self.critic_1_target(next_states, next_actions)
            q2_next = self.critic_2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
            target = rewards + self.gamma * (1.0 - dones) * min_q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- actor update (reparameterized) ---
        new_actions, log_probs = self._sample_action(states)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- temperature update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -log_probs.mean().item(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)
