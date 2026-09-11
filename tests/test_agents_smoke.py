"""Smoke tests on standard Gymnasium environments, run BEFORE any agent
touches PricingEnv (see build order in the plan) -- isolates "my RL math is
broken" from "my custom env/reward is broken". Not full convergence proofs,
but each test checks for a real, measurable learning signal on an easy
classic-control benchmark, not just "did not crash"."""
from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from agents.ddpg_agent import DDPGAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent


def _run_episode_return(env, agent, eval_mode, action_transform=None):
    state, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action = agent.select_action(state, eval_mode=eval_mode)
        env_action = action_transform(action) if action_transform else action
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        total += reward
        state = next_state
    return total


def test_dqn_learns_cartpole():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        buffer_capacity=10_000,
        warmup_steps=200,
        target_update_interval=50,
        eps_decay_steps=3000,
        seed=0,
    )

    state, _ = env.reset(seed=0)
    for _ in range(4000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.observe(state, action, reward, next_state, done)
        metrics = agent.update()
        if metrics:
            assert np.isfinite(metrics["loss"])
        state = next_state if not done else env.reset()[0]

    eval_env = gym.make("CartPole-v1")
    returns = [_run_episode_return(eval_env, agent, eval_mode=True) for _ in range(10)]
    assert np.mean(returns) > 100, f"DQN did not learn CartPole, mean eval return={np.mean(returns)}"


def test_ddpg_learns_pendulum():
    env = gym.make("Pendulum-v1")
    action_scale = float(env.action_space.high[0])
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        buffer_capacity=20_000,
        warmup_steps=200,
        seed=0,
    )

    early_returns, late_returns = [], []
    state, _ = env.reset(seed=0)
    ep_return = 0.0
    ep_idx = 0
    for step in range(6000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action * action_scale)
        done = terminated or truncated
        agent.observe(state, action, reward, next_state, done)
        agent.update()
        ep_return += reward
        state = next_state
        if done:
            (early_returns if ep_idx < 10 else late_returns).append(ep_return)
            ep_return = 0.0
            ep_idx += 1
            state, _ = env.reset()

    assert len(late_returns) > 0
    assert np.mean(late_returns[-10:]) > np.mean(early_returns) + 100, (
        f"DDPG did not improve on Pendulum: early={np.mean(early_returns):.1f}, "
        f"late={np.mean(late_returns[-10:]):.1f}"
    )


def test_sac_learns_pendulum():
    env = gym.make("Pendulum-v1")
    action_scale = float(env.action_space.high[0])
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        buffer_capacity=20_000,
        warmup_steps=200,
        seed=0,
    )

    early_returns, late_returns = [], []
    state, _ = env.reset(seed=0)
    ep_return = 0.0
    ep_idx = 0
    for step in range(6000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action * action_scale)
        done = terminated or truncated
        agent.observe(state, action, reward, next_state, done)
        metrics = agent.update()
        if metrics:
            assert np.isfinite(metrics["critic_loss"])
            assert np.isfinite(metrics["alpha"])
        ep_return += reward
        state = next_state
        if done:
            (early_returns if ep_idx < 10 else late_returns).append(ep_return)
            ep_return = 0.0
            ep_idx += 1
            state, _ = env.reset()

    assert len(late_returns) > 0
    assert np.mean(late_returns[-10:]) > np.mean(early_returns) + 100, (
        f"SAC did not improve on Pendulum: early={np.mean(early_returns):.1f}, "
        f"late={np.mean(late_returns[-10:]):.1f}"
    )


def test_ppo_learns_pendulum():
    """PPO needs a much larger step budget than the off-policy agents to
    show improvement on Pendulum: with gamma=0.99 and a near-zero-initialized
    critic, the GAE-bootstrapped return estimate takes many rollouts to
    calibrate to its true (large) magnitude before advantages carry any real
    signal -- confirmed by inspecting value-function R^2 across rollouts,
    which starts strongly negative and only turns positive after ~10-15
    rollouts. This is the same on-policy sample-inefficiency the report
    itself attributes to PPO vs. DQN/DDPG (Section 4.3), not a bug -- it
    just means this smoke test needs ~100k steps, not ~6k, to see it."""
    env = gym.make("Pendulum-v1")
    action_scale = float(env.action_space.high[0])
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        rollout_steps=1024,
        minibatch_size=64,
        seed=0,
    )

    returns = []
    state, _ = env.reset(seed=0)
    ep_return = 0.0
    for step in range(100_000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action * action_scale)
        done = terminated or truncated
        agent.observe(state, action, reward, next_state, done)
        agent.update()
        ep_return += reward
        state = next_state
        if done:
            returns.append(ep_return)
            ep_return = 0.0
            state, _ = env.reset()

    early = np.mean(returns[:10])
    late = np.mean(returns[int(len(returns) * 0.6) :])
    assert late > early + 300, f"PPO did not improve on Pendulum: early={early:.1f}, late={late:.1f}"
