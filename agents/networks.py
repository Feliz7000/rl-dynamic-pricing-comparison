"""Shared MLP building blocks used by all four agents, so architecture
capacity is held constant across the comparison (see report Section 7 -- the
fairness of the comparison is the whole point)."""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_SIZES = (128, 128)


def mlp(input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...] = HIDDEN_SIZES, output_activation=None) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """State -> Q-value per discrete action (DQN)."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = mlp(state_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DeterministicActor(nn.Module):
    """State -> exact continuous action in [-1, 1] (DDPG)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp(state_dim, action_dim, output_activation=nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class QCritic(nn.Module):
    """(State, action) -> scalar Q-value (DDPG / SAC critics)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """State -> tanh-squashed Gaussian policy over continuous actions
    (shared by PPO and SAC). Returns (mean, log_std) pre-squash."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, state_dependent_std: bool = True):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(HIDDEN_SIZES[-1], action_dim)
        if state_dependent_std:
            self.log_std_head = nn.Linear(HIDDEN_SIZES[-1], action_dim)
        else:
            self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean_head(h)
        if self.state_dependent_std:
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        else:
            log_std = self.log_std_param.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        return mean, log_std


class ValueNetwork(nn.Module):
    """State -> scalar value estimate (PPO critic)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = mlp(state_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
