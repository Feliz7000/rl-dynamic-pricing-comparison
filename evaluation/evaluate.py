"""Deterministic evaluation of a trained checkpoint on held-out TEST
products only, over their real logged months only (bootstrap extension is a
training-time trick and is disabled here -- see PricingEnv docstring).

Usage:
    python -m evaluation.evaluate --agent sac --seed 0 --config configs/sac.yaml
    python -m evaluation.evaluate --agent random --seed 0 --action-mode continuous
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config, set_global_seed
from training.train import LEARNING_AGENTS, build_agent, load_env_artifacts
from env.pricing_env import PricingEnv

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default=None)
    parser.add_argument("--data-source", choices=["synthetic", "raw"], default="synthetic")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    action_mode = args.action_mode or cfg.get("action_mode", "continuous")
    episode_horizon = cfg.get("episode_horizon", 20)
    k_buckets = cfg.get("k_buckets", 11)
    hyperparams = cfg.get("hyperparams", {})

    set_global_seed(args.seed)

    df, _, test_ids, demand_model, scaler = load_env_artifacts(args.data_source)

    env = PricingEnv(
        df=df,
        demand_model=demand_model,
        scaler=scaler,
        product_ids=test_ids,
        action_mode=action_mode,
        k_buckets=k_buckets,
        episode_horizon=episode_horizon,
        allow_bootstrap_extension=False,
        seed=args.seed,
    )

    agent = build_agent(args.agent, scaler.feature_dim, action_mode, k_buckets, args.seed, hyperparams)
    if args.agent in LEARNING_AGENTS:
        ckpt_path = ROOT / "results" / "models" / f"{args.agent}_{args.seed}.pt"
        agent.load(ckpt_path)

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eval_{args.agent}_{args.seed}.csv"

    fieldnames = ["product_id", "step", "price", "qty_hat", "rcr", "drcr", "reward"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pid in test_ids:
            # each test product's real-months window may be shorter than the
            # training episode_horizon; PricingEnv naturally terminates early
            # (bootstrap disabled) once real months run out.
            state, info = env.reset(options={"product_id": pid})
            step = 0
            done = False
            while not done:
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                writer.writerow(
                    {
                        "product_id": pid,
                        "step": step,
                        "price": step_info["price"],
                        "qty_hat": step_info["qty_hat"],
                        "rcr": step_info["rcr"],
                        "drcr": step_info["drcr"],
                        "reward": reward,
                    }
                )
                state = next_state
                step += 1

    print(f"Finished evaluating {args.agent} seed={args.seed} on {len(test_ids)} test products -> {log_path}")


if __name__ == "__main__":
    main()
