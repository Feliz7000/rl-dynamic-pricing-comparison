"""PPO (Schulman et al., 2017): on-policy actor-critic with a clipped
surrogate objective, trained on fresh on-policy rollouts (no replay buffer).
Run in continuous mode here (see plan) so it's directly comparable to
DDPG/SAC on price fidelity. See report Section 4 (Akhil)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.base_agent import BaseAgent
from agents.networks import GaussianPolicy, ValueNetwork
from agents.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        rollout_steps: int = 2048,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps

        # State-independent log_std (a single learned per-action-dim
        # parameter, not a function of state), matching standard continuous
        # PPO implementations (e.g. SB3's DiagGaussianDistribution) -- a
        # state-dependent std head can amplify noise into the variance
        # prediction while the advantage signal is still weak early in
        # training.
        self.actor = GaussianPolicy(state_dim, action_dim, state_dependent_std=False).to(self.device)
        self.critic = ValueNetwork(state_dim).to(self.device)
        # Separate optimizers (and separate grad-norm clipping, see update())
        # for actor vs. critic: value-loss gradients can be orders of
        # magnitude larger than policy-loss gradients on environments with
        # large return magnitudes, and clipping them as one combined norm
        # would silently crush the policy gradient down to near zero.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer(rollout_steps, state_dim, action_dim, gamma, gae_lambda)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self._last_state = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._pending_next_state = None

    def _dist(self, state_t: torch.Tensor):
        mean, log_std = self.actor(state_t)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def select_action(self, state, eval_mode: bool = False):
        """Standard (unsquashed) Gaussian policy, as in the reference PPO
        implementations (Schulman et al., 2017; SB3; CleanRL) -- unlike
        DDPG/SAC, PPO doesn't need a tanh-squashed reparameterized sample, so
        we sample a raw Gaussian action and only clip it at the environment
        boundary. The buffer stores the raw pre-clip sample and its log_prob
        under the sampling policy; log_prob is recomputed on that same raw
        value during update() (clipping's effect on the density is the
        standard, near-universal approximation ignored here too)."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(state_t)
            raw_action = dist.mean if eval_mode else dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            value = self.critic(state_t)

        self._last_state = np.asarray(state, dtype=np.float32)
        self._last_action = raw_action.cpu().numpy()[0]
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        clipped_action = np.clip(self._last_action, -1.0, 1.0)
        return clipped_action.astype(np.float32)

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(self._last_state, self._last_action, self._last_log_prob, reward, self._last_value, done)
        self._pending_next_state = None if done else next_state

    def _bootstrap_value(self, next_state) -> float:
        with torch.no_grad():
            state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(state_t).item()

    def update(self) -> dict[str, float]:
        if not self.buffer.full():
            return {}

        last_value = self._bootstrap_value(self._pending_next_state) if self._pending_next_state is not None else 0.0
        data = self.buffer.get(last_value)

        states = torch.as_tensor(data["states"], device=self.device)
        actions = torch.as_tensor(data["actions"], device=self.device)
        old_log_probs = torch.as_tensor(data["log_probs"], device=self.device)
        advantages = torch.as_tensor(data["advantages"], device=self.device)
        returns = torch.as_tensor(data["returns"], device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idx[start : start + self.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                dist = self._dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                values = self.critic(mb_states)
                value_loss = torch.nn.functional.mse_loss(values, mb_returns)

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                n_updates += 1

        self.buffer.reset()
        return {k: v / max(n_updates, 1) for k, v in metrics.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
