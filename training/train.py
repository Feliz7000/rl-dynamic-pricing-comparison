"""Trains one (agent, seed) run on PricingEnv using training-split products
and writes per-step logs + a checkpoint. Identical training protocol is used
for every agent (same env construction, same total env-step budget, same
logging) so the resulting comparison is fair -- see report Section 7.

Usage:
    python -m training.train --agent sac --seed 0 --config configs/sac.yaml
    python -m training.train --agent random --seed 0 --steps 5000 --action-mode continuous
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# CPU-only, tiny MLPs: PyTorch's default multi-threaded dispatch adds more
# overhead than it saves at this scale (measured ~20% faster single-threaded).
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent
from agents.baselines import RandomAgent, StaticPriceAgent
from agents.ddpg_agent import DDPGAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from env.data_loader import StateScaler, engineer_features, load_raw
from env.demand_model import DemandModel
from env.pricing_env import PricingEnv
from training.config import load_config, set_global_seed

ROOT = Path(__file__).resolve().parent.parent
LEARNING_AGENTS = {"dqn", "ddpg", "ppo", "sac"}
BASELINE_AGENTS = {"random", "static"}

# Union of the scalar metrics the four agents' update() methods can return.
METRIC_KEYS = (
    "loss",
    "critic_loss",
    "actor_loss",
    "policy_loss",
    "value_loss",
    "alpha_loss",
    "alpha",
    "entropy",
    "q_mean",
    "epsilon",
)


def build_agent(name: str, state_dim: int, action_mode: str, k_buckets: int, seed: int, hyperparams: dict) -> BaseAgent:
    if name == "dqn":
        return DQNAgent(state_dim=state_dim, n_actions=k_buckets, seed=seed, **hyperparams)
    if name == "ddpg":
        return DDPGAgent(state_dim=state_dim, action_dim=1, seed=seed, **hyperparams)
    if name == "sac":
        return SACAgent(state_dim=state_dim, action_dim=1, seed=seed, **hyperparams)
    if name == "ppo":
        return PPOAgent(state_dim=state_dim, action_dim=1, seed=seed, **hyperparams)
    if name == "random":
        import gymnasium.spaces as spaces

        space = spaces.Discrete(k_buckets) if action_mode == "discrete" else spaces.Box(-1.0, 1.0, shape=(1,))
        return RandomAgent(action_space=space, seed=seed)
    if name == "static":
        return StaticPriceAgent(action_mode=action_mode, k_buckets=k_buckets)
    raise ValueError(f"Unknown agent: {name}")


def load_env_artifacts(source: str):
    data_path = ROOT / "data" / source / "retail_price.csv"
    df = load_raw(data_path)
    df = engineer_features(df)

    split_path = ROOT / "data" / "processed" / "product_split.json"
    with open(split_path) as f:
        split = json.load(f)

    demand_model = DemandModel.load(ROOT / "results" / "models" / "demand_model.pkl")
    scaler = StateScaler.load(ROOT / "results" / "models" / "state_scaler.pkl")
    return df, split["train_product_ids"], split["test_product_ids"], demand_model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=sorted(LEARNING_AGENTS | BASELINE_AGENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps from config")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default=None)
    parser.add_argument("--data-source", choices=["synthetic", "raw"], default="synthetic")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    action_mode = args.action_mode or cfg.get("action_mode", "continuous")
    total_steps = args.steps or cfg.get("total_steps", 50_000)
    episode_horizon = cfg.get("episode_horizon", 20)
    k_buckets = cfg.get("k_buckets", 11)
    hyperparams = cfg.get("hyperparams", {})

    set_global_seed(args.seed)

    df, train_ids, _, demand_model, scaler = load_env_artifacts(args.data_source)

    env = PricingEnv(
        df=df,
        demand_model=demand_model,
        scaler=scaler,
        product_ids=train_ids,
        action_mode=action_mode,
        k_buckets=k_buckets,
        episode_horizon=episode_horizon,
        allow_bootstrap_extension=True,
        seed=args.seed,
    )

    agent = build_agent(args.agent, scaler.feature_dim, action_mode, k_buckets, args.seed, hyperparams)

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.agent}_{args.seed}.csv"

    fieldnames = ["global_step", "episode", "product_id", "price", "qty_hat", "rcr", "drcr", "reward"]
    # Optional learning metrics returned by agent.update(); each algorithm
    # fills the subset it has, blanks elsewhere. Kept as a fixed superset so
    # the CSV header is known up front.
    metric_fields = [f"m_{k}" for k in METRIC_KEYS]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + metric_fields, restval="")
        writer.writeheader()

        episode = 0
        state, info = env.reset(seed=args.seed)
        for global_step in range(total_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            agent.observe(state, action, reward, next_state, done)
            metrics = agent.update()

            row = {
                "global_step": global_step,
                "episode": episode,
                "product_id": step_info["product_id"],
                "price": step_info["price"],
                "qty_hat": step_info["qty_hat"],
                "rcr": step_info["rcr"],
                "drcr": step_info["drcr"],
                "reward": reward,
            }
            for k, v in metrics.items():
                if k in METRIC_KEYS:
                    row[f"m_{k}"] = v
            writer.writerow(row)

            state = next_state
            if done:
                episode += 1
                state, info = env.reset()

    model_dir = ROOT / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.agent in LEARNING_AGENTS:
        agent.save(model_dir / f"{args.agent}_{args.seed}.pt")

    print(f"Finished {args.agent} seed={args.seed}: {total_steps} steps, {episode} episodes -> {log_path}")


if __name__ == "__main__":
    main()
