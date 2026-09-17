"""Export everything the presentation dashboard needs into one JSON file,
computed from the *real* artifacts in this repo: trained checkpoints, the
fitted demand model, the training/eval logs and the comparison summary.
Nothing in the dashboard is synthesized -- if a number is on screen, it was
produced here from those artifacts.

Usage:
    python scripts/export_dashboard_data.py                 # -> dashboard/data/dashboard_data.json
    python scripts/export_dashboard_data.py --data-source raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import DRCR_TAU, WARMUP_MONTHS  # noqa: E402
from env.demand_model import PRICE_CLIP_HIGH, PRICE_CLIP_LOW  # noqa: E402
from env.pricing_env import REWARD_SCALE, PricingEnv  # noqa: E402
from scripts.fit_demand_model import time_based_holdout  # noqa: E402
from training.config import load_config, set_global_seed  # noqa: E402
from training.train import LEARNING_AGENTS, build_agent, load_env_artifacts  # noqa: E402

LOG_DIR = ROOT / "results" / "logs"
MODEL_DIR = ROOT / "results" / "models"
OUT_PATH = ROOT / "dashboard" / "data" / "dashboard_data.json"

# Display metadata. Colours follow the presentation palette; labels are
# plain algorithm names.
ALGORITHMS = [
    {"key": "dqn", "label": "DQN", "color": "#5B8DEF", "action_space": "Discrete price buckets", "policy": "Off-policy, value-based", "config": "configs/dqn.yaml"},
    {"key": "ddpg", "label": "DDPG", "color": "#E1A339", "action_space": "Continuous price", "policy": "Off-policy, deterministic actor-critic", "config": "configs/ddpg.yaml"},
    {"key": "ppo", "label": "PPO", "color": "#C77DFF", "action_space": "Continuous price", "policy": "On-policy, clipped actor-critic", "config": "configs/ppo.yaml"},
    {"key": "sac", "label": "SAC", "color": "#4FD1A5", "action_space": "Continuous price", "policy": "Off-policy, max-entropy actor-critic", "config": "configs/sac.yaml"},
]
BASELINES = [
    {"key": "static", "label": "Static (no change)", "color": "#6E8FB0"},
    {"key": "random", "label": "Random", "color": "#8C97A3"},
]
HISTORICAL_COLOR = "#7C8896"

ROLLING_WINDOW = 200
CURVE_POINTS = 300
PRICE_GRID_POINTS = 40
ROUND = 4


def r(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), ROUND)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def detect_seeds() -> list[int]:
    seeds = set()
    for p in MODEL_DIR.glob("sac_*.pt"):
        try:
            seeds.add(int(p.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(seeds)


# --------------------------------------------------------------------------
# Rollouts on held-out products
# --------------------------------------------------------------------------

def rollout(env: PricingEnv, agent, pid: str) -> dict:
    state, _ = env.reset(options={"product_id": pid})
    out = {"price": [], "qty": [], "revenue": [], "rcr": [], "drcr": [], "clipped": []}
    done = False
    while not done:
        action = agent.select_action(state, eval_mode=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out["price"].append(info["price"])
        out["qty"].append(info["qty_hat"])
        out["revenue"].append(info["price"] * info["qty_hat"])
        out["rcr"].append(info["rcr"])
        out["drcr"].append(info["drcr"])
        out["clipped"].append(bool(info["clipped"]))
    return out


def historical_path(rows: list[dict], demand_model) -> dict:
    """The 'manual pricing' reference: the seller's actual month-by-month
    prices, pushed through the SAME demand model the agents are scored
    with, so the comparison is apples-to-apples inside the simulator.
    Mirrors PricingEnv's RCR-history seeding and DRCR lookback exactly."""
    rcr_hist = [row["rcr"] for row in rows[: WARMUP_MONTHS + 1]]
    out = {"price": [], "qty_actual": [], "revenue_actual": [], "qty_model": [], "revenue_model": [], "rcr": [], "drcr": []}
    for t in range(WARMUP_MONTHS, len(rows)):
        row = rows[t]
        price = float(row["unit_price"])
        qty_model, _, _ = demand_model.predict_qty_single(row, price)
        revenue_model = price * qty_model
        rcr_t = revenue_model / max(float(row["traffic"]), 1e-6)
        rcr_hist.append(rcr_t)
        lookback = len(rcr_hist) - 1 - DRCR_TAU
        drcr = rcr_t - rcr_hist[lookback]
        out["price"].append(price)
        out["qty_actual"].append(float(row["qty"]))
        out["revenue_actual"].append(float(row["total_price"]))
        out["qty_model"].append(qty_model)
        out["revenue_model"].append(revenue_model)
        out["rcr"].append(rcr_t)
        out["drcr"].append(drcr)
    return out


def price_grid(rows: list[dict], demand_model, hist: dict) -> list[dict]:
    """For every decision month: demand / revenue / DRCR across a grid of
    candidate prices, evaluated at that month's real context. DRCR here is
    relative to the previous month's in-model historical RCR ('vs. last
    month'), so it does not depend on any agent's path."""
    pid = rows[0]["product_id"]
    low, high = demand_model.price_bounds[pid]
    prices = np.linspace(low, high, PRICE_GRID_POINTS)
    grid = []
    for i, t in enumerate(range(WARMUP_MONTHS, len(rows))):
        row = rows[t]
        traffic = max(float(row["traffic"]), 1e-6)
        ref_rcr = hist["rcr"][i - 1] if i > 0 else float(rows[t - 1]["rcr"])
        qty = np.array([demand_model.predict_qty_single(row, float(p))[0] for p in prices])
        revenue = prices * qty
        drcr = revenue / traffic - ref_rcr
        grid.append({"prices": r(prices), "qty": r(qty), "revenue": r(revenue), "drcr": r(drcr)})
    return grid


def export_products(df, test_ids, demand_model, scaler, seeds, configs) -> list[dict]:
    products = []
    agent_specs = [(a["key"], configs[a["key"]]) for a in ALGORITHMS] + [("static", {}), ("random", {})]

    # one env per action mode; product_ids covers all test products
    envs = {}
    for mode in ("discrete", "continuous"):
        envs[mode] = PricingEnv(
            df=df,
            demand_model=demand_model,
            scaler=scaler,
            product_ids=test_ids,
            action_mode=mode,
            k_buckets=configs["dqn"].get("k_buckets", 11),
            episode_horizon=max(c.get("episode_horizon", 20) for c in configs.values()),
            allow_bootstrap_extension=False,
            seed=0,
        )

    # build every (agent, seed) once
    agents = {}
    for key, cfg in agent_specs:
        mode = cfg.get("action_mode", "continuous")
        for seed in seeds:
            set_global_seed(seed)
            agent = build_agent(key, scaler.feature_dim, mode, configs["dqn"].get("k_buckets", 11), seed, cfg.get("hyperparams", {}))
            if key in LEARNING_AGENTS:
                agent.load(MODEL_DIR / f"{key}_{seed}.pt")
            agents[(key, seed)] = (agent, mode)

    for pid in test_ids:
        rows = [dict(rw) for rw in df[df["product_id"] == pid].sort_values("t").to_dict("records")]
        decision_rows = rows[WARMUP_MONTHS:]
        hist = historical_path(rows, demand_model)

        product = {
            "id": pid,
            "category": rows[0]["product_category_name"],
            "months": [rw["month_year"] for rw in decision_rows],
            "traffic": r([rw["traffic"] for rw in decision_rows]),
            "competitors": {f"comp_{k}": r([rw[f"comp_{k}"] for rw in decision_rows]) for k in (1, 2, 3)},
            "price_bounds": r(list(demand_model.price_bounds[pid])),
            "historical": {k: r(v) for k, v in hist.items()},
            "agents": {},
            "price_grid": price_grid(rows, demand_model, hist),
        }
        for (key, seed), (agent, mode) in agents.items():
            traj = rollout(envs[mode], agent, pid)
            product["agents"].setdefault(key, {})[str(seed)] = {k: (v if k == "clipped" else r(v)) for k, v in traj.items()}
        products.append(product)
        print(f"  exported {pid}: {len(product['months'])} months")
    return products


# --------------------------------------------------------------------------
# Training curves
# --------------------------------------------------------------------------

def downsample(arr: np.ndarray, n: int) -> list:
    if len(arr) <= n:
        return r(arr)
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return r(arr[idx])


def export_training(seeds) -> dict:
    training = {}
    loss_pref = ["critic_loss", "loss", "value_loss"]
    for algo in ALGORITHMS:
        key = algo["key"]
        per_seed_roll, per_seed_std, per_seed_run, per_seed_loss = [], [], [], []
        steps = None
        for seed in seeds:
            path = LOG_DIR / f"{key}_{seed}.csv"
            if not path.exists():
                continue
            log = pd.read_csv(path)
            reward = log["reward"].to_numpy()
            roll = pd.Series(reward).rolling(ROLLING_WINDOW, min_periods=1).mean().to_numpy()
            std = pd.Series(reward).rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0).to_numpy()
            running = np.cumsum(reward) / np.arange(1, len(reward) + 1)
            per_seed_roll.append(roll)
            per_seed_std.append(std)
            per_seed_run.append(running)
            steps = log["global_step"].to_numpy()
            for col in loss_pref:
                c = f"m_{col}"
                if c in log.columns and log[c].notna().any():
                    per_seed_loss.append((col, log[c].rolling(ROLLING_WINDOW, min_periods=1).mean().to_numpy()))
                    break
        if not per_seed_roll:
            continue
        n = min(len(x) for x in per_seed_roll)
        roll_mat = np.stack([x[:n] for x in per_seed_roll])
        std_mat = np.stack([x[:n] for x in per_seed_std])
        run_mat = np.stack([x[:n] for x in per_seed_run])
        entry = {
            "step": downsample(steps[:n].astype(float), CURVE_POINTS),
            "seeds": {
                str(s): {
                    "reward_rolling": downsample(roll_mat[i], CURVE_POINTS),
                    "stability": downsample(std_mat[i], CURVE_POINTS),
                    "running_mean": downsample(run_mat[i], CURVE_POINTS),
                }
                for i, s in enumerate(seeds[: len(per_seed_roll)])
            },
            "mean": downsample(roll_mat.mean(axis=0), CURVE_POINTS),
            "std": downsample(roll_mat.std(axis=0), CURVE_POINTS),
            "stability_mean": downsample(std_mat.mean(axis=0), CURVE_POINTS),
            "running_mean": downsample(run_mat.mean(axis=0), CURVE_POINTS),
            "running_std": downsample(run_mat.std(axis=0), CURVE_POINTS),
        }
        if per_seed_loss and len(per_seed_loss) == len(per_seed_roll):
            m = min(len(x) for _, x in per_seed_loss)
            loss_mat = np.stack([x[:m] for _, x in per_seed_loss])
            entry["loss"] = {"name": per_seed_loss[0][0], "mean": downsample(np.nan_to_num(loss_mat.mean(axis=0)), CURVE_POINTS)}
        training[key] = entry
    return training


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def export_summary(seeds) -> dict:
    summary = {}
    label_to_key = {a["label"]: a["key"] for a in ALGORITHMS + BASELINES}
    csv_path = ROOT / "results" / "comparison_summary.csv"
    table = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    for _, row in table.iterrows():
        key = label_to_key.get(row["agent"])
        if key is None:
            continue
        summary[key] = {
            "mean_test_drcr": r(row["mean_test_drcr"]),
            "std_test_drcr": r(row["std_test_drcr"]),
            "mean_final_reward": None if pd.isna(row.get("mean_final_reward", np.nan)) else r(row["mean_final_reward"]),
            "std_final_reward": None if pd.isna(row.get("std_across_seeds", np.nan)) else r(row["std_across_seeds"]),
            "per_seed_test_drcr": [],
        }
    for key in summary:
        for seed in seeds:
            path = LOG_DIR / f"eval_{key}_{seed}.csv"
            if path.exists():
                summary[key]["per_seed_test_drcr"].append(r(pd.read_csv(path)["drcr"].mean()))
    return summary


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["synthetic", "raw"], default="synthetic")
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    seeds = detect_seeds()
    if not seeds:
        raise SystemExit("No checkpoints found in results/models -- run the training pipeline first.")
    configs = {a["key"]: load_config(ROOT / a["config"]) for a in ALGORITHMS}

    df, train_ids, test_ids, demand_model, scaler = load_env_artifacts(args.data_source)

    _, holdout = time_based_holdout(df, holdout_months=2)
    dm_metrics = demand_model.evaluate(holdout)

    sample_log = LOG_DIR / f"sac_{seeds[0]}.csv"
    total_steps = int(pd.read_csv(sample_log, usecols=["global_step"])["global_step"].max() + 1) if sample_log.exists() else None

    print(f"Exporting {len(test_ids)} held-out products x {len(seeds)} seeds ...")
    products = export_products(df, test_ids, demand_model, scaler, seeds, configs)

    payload = {
        "meta": {
            "data_source": args.data_source,
            "seeds": seeds,
            "total_steps": total_steps,
            "n_train_products": len(train_ids),
            "n_test_products": len(test_ids),
            "warmup_months": WARMUP_MONTHS,
            "tau": DRCR_TAU,
            "reward_scale": REWARD_SCALE,
            "k_buckets": configs["dqn"].get("k_buckets", 11),
            "price_bound_frac": 0.25,
            "price_clip": [PRICE_CLIP_LOW, PRICE_CLIP_HIGH],
            "dqn_eps_decay_steps": configs["dqn"].get("hyperparams", {}).get("eps_decay_steps"),
            "demand_model": {
                "r2_holdout": r(dm_metrics["r2_log"]),
                "mae_log_holdout": r(dm_metrics["mae_log"]),
                "elasticity_coef": r(float(demand_model.baseline_coefs_[0])),
            },
            "algorithms": [{k: v for k, v in a.items() if k != "config"} for a in ALGORITHMS],
            "baselines": BASELINES,
            "historical_color": HISTORICAL_COLOR,
        },
        "products": products,
        "training": export_training(seeds),
        "summary": export_summary(seeds),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
