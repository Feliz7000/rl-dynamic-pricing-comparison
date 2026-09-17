"""Measure training throughput and per-step inference cost of each agent on
PricingEnv, so the report's computational-performance comparison uses
measured numbers. Writes results/compute_benchmark.json.

    python scripts/benchmark_compute.py [--steps 1500]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.pricing_env import PricingEnv  # noqa: E402
from training.config import load_config, set_global_seed  # noqa: E402
from training.train import build_agent, load_env_artifacts  # noqa: E402

AGENTS = ["dqn", "ddpg", "ppo", "sac"]


def count_params(agent) -> int:
    total = 0
    for attr in ("q_net", "actor", "critic", "critic_1", "critic_2"):
        module = getattr(agent, attr, None)
        if module is not None:
            total += sum(p.numel() for p in module.parameters())
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    df, train_ids, test_ids, demand_model, scaler = load_env_artifacts("synthetic")
    results = {}
    for key in AGENTS:
        cfg = load_config(ROOT / "configs" / f"{key}.yaml")
        mode = cfg.get("action_mode", "continuous")
        hp = dict(cfg.get("hyperparams", {}))
        if "warmup_steps" in hp:
            hp["warmup_steps"] = 100  # start learning early so the timing includes updates
        set_global_seed(0)
        env = PricingEnv(df=df, demand_model=demand_model, scaler=scaler, product_ids=train_ids, action_mode=mode, k_buckets=cfg.get("k_buckets", 11), episode_horizon=cfg.get("episode_horizon", 20), allow_bootstrap_extension=True, seed=0)
        agent = build_agent(key, scaler.feature_dim, mode, cfg.get("k_buckets", 11), 0, hp)

        state, _ = env.reset(seed=0)
        t0 = time.perf_counter()
        n_updates = 0
        for _ in range(args.steps):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            agent.observe(state, action, reward, next_state, term or trunc)
            if agent.update():
                n_updates += 1
            state = next_state if not (term or trunc) else env.reset()[0]
        train_time = time.perf_counter() - t0

        # pure inference cost (deterministic action selection)
        t1 = time.perf_counter()
        for _ in range(500):
            agent.select_action(state, eval_mode=True)
        infer_us = (time.perf_counter() - t1) / 500 * 1e6

        results[key] = {
            "steps": args.steps,
            "train_seconds": round(train_time, 2),
            "steps_per_second": round(args.steps / train_time, 1),
            "seconds_per_1k_steps": round(train_time / args.steps * 1000, 2),
            "gradient_updates": n_updates,
            "inference_us_per_action": round(infer_us, 1),
            "parameters": count_params(agent),
        }
        print(f"{key:5s} {args.steps} steps in {train_time:6.1f}s  ({results[key]['steps_per_second']:.0f} steps/s, {n_updates} updates, {infer_us:.0f} us/action, {results[key]['parameters']:,} params)")

    # env-only cost for reference
    env = PricingEnv(df=df, demand_model=demand_model, scaler=scaler, product_ids=train_ids, action_mode="continuous", allow_bootstrap_extension=True, seed=0)
    state, _ = env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(500):
        state, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        if term or trunc:
            state, _ = env.reset()
    env_ms = (time.perf_counter() - t0) / 500 * 1e3

    payload = {
        "machine": {"processor": platform.processor(), "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(), "os": platform.platform()},
        "env_step_ms": round(env_ms, 2),
        "agents": results,
    }
    out = ROOT / "results" / "compute_benchmark.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"env step alone: {env_ms:.2f} ms -> wrote {out}")


if __name__ == "__main__":
    main()
