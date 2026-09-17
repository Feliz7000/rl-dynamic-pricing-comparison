"""Aggregates all training + evaluation logs into a measured version of the
report's Section 6 comparison table, plus learning-curve / DRCR / stability
plots for the report and presentation.

Usage:
    python -m evaluation.compare
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

AGENT_ORDER = ["static", "random", "dqn", "ddpg", "ppo", "sac"]
AGENT_LABELS = {
    "dqn": "DQN",
    "ddpg": "DDPG",
    "ppo": "PPO",
    "sac": "SAC",
    "random": "Random",
    "static": "Static (no change)",
}


def _find_runs(pattern: str) -> dict[str, list[int]]:
    """Returns {agent: [seeds]} for files in LOG_DIR matching `pattern`
    (e.g. r"eval_(\\w+)_(\\d+)\\.csv" or r"(\\w+)_(\\d+)\\.csv")."""
    runs: dict[str, list[int]] = {}
    regex = re.compile(pattern)
    for path in LOG_DIR.glob("*.csv"):
        m = regex.match(path.name)
        if m:
            agent, seed = m.group(1), int(m.group(2))
            runs.setdefault(agent, []).append(seed)
    return {a: sorted(seeds) for a, seeds in runs.items()}


def summarize_eval() -> pd.DataFrame:
    runs = _find_runs(r"eval_(\w+)_(\d+)\.csv")
    rows = []
    for agent in AGENT_ORDER:
        if agent not in runs:
            continue
        per_seed_drcr = []
        for seed in runs[agent]:
            df = pd.read_csv(LOG_DIR / f"eval_{agent}_{seed}.csv")
            per_seed_drcr.append(df["drcr"].mean())
        rows.append(
            {
                "agent": AGENT_LABELS.get(agent, agent),
                "n_seeds": len(per_seed_drcr),
                "mean_test_drcr": np.mean(per_seed_drcr),
                "std_test_drcr": np.std(per_seed_drcr),
            }
        )
    return pd.DataFrame(rows)


def summarize_training_stability() -> pd.DataFrame:
    runs = _find_runs(r"(\w+)_(\d+)\.csv")
    rows = []
    for agent in AGENT_ORDER:
        if agent not in runs or agent not in ("dqn", "ddpg", "ppo", "sac"):
            continue
        per_seed_final_reward = []
        for seed in runs[agent]:
            df = pd.read_csv(LOG_DIR / f"{agent}_{seed}.csv")
            last_20pct = df.iloc[int(len(df) * 0.8) :]
            per_seed_final_reward.append(last_20pct["reward"].mean())
        rows.append(
            {
                "agent": AGENT_LABELS.get(agent, agent),
                "mean_final_reward": np.mean(per_seed_final_reward),
                "std_across_seeds": np.std(per_seed_final_reward),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(smoothing_window: int = 200) -> None:
    runs = _find_runs(r"(\w+)_(\d+)\.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for agent in ["dqn", "ddpg", "ppo", "sac"]:
        if agent not in runs:
            continue
        curves = []
        for seed in runs[agent]:
            df = pd.read_csv(LOG_DIR / f"{agent}_{seed}.csv")
            smoothed = df["reward"].rolling(smoothing_window, min_periods=1).mean()
            curves.append(smoothed.to_numpy())
        min_len = min(len(c) for c in curves)
        curves = np.stack([c[:min_len] for c in curves])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_curve, label=AGENT_LABELS[agent])
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    ax.set_xlabel("environment step")
    ax.set_ylabel(f"reward (rolling mean, window={smoothing_window})")
    ax.set_title("Training learning curves (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_drcr_bar(eval_summary: pd.DataFrame) -> None:
    if eval_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(eval_summary))
    ax.bar(x, eval_summary["mean_test_drcr"], yerr=eval_summary["std_test_drcr"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_summary["agent"], rotation=20, ha="right")
    ax.set_ylabel("mean test DRCR (held-out products)")
    ax.set_title("Held-out test DRCR by algorithm (mean ± std across seeds)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "test_drcr_comparison.png", dpi=150)
    plt.close(fig)


def plot_drcr_stability(eval_summary_agents: list[str]) -> None:
    runs = _find_runs(r"eval_(\w+)_(\d+)\.csv")
    data, labels = [], []
    for agent in AGENT_ORDER:
        if agent not in runs:
            continue
        per_seed = [pd.read_csv(LOG_DIR / f"eval_{agent}_{seed}.csv")["drcr"].mean() for seed in runs[agent]]
        if len(per_seed) < 2:
            continue
        data.append(per_seed)
        labels.append(AGENT_LABELS.get(agent, agent))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("test DRCR (per-seed mean)")
    ax.set_title("Cross-seed stability of held-out test DRCR")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "drcr_stability.png", dpi=150)
    plt.close(fig)


def main():
    eval_summary = summarize_eval()
    stability_summary = summarize_training_stability()

    print("=== Held-out test DRCR summary ===")
    print(eval_summary.to_string(index=False) if not eval_summary.empty else "(no eval logs found)")
    print()
    print("=== Training-tail reward summary (last 20% of steps) ===")
    print(stability_summary.to_string(index=False) if not stability_summary.empty else "(no training logs found)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / "comparison_summary.csv"
    merged = eval_summary.merge(stability_summary, on="agent", how="outer")
    merged.to_csv(summary_path, index=False)
    print(f"\nSaved combined summary to {summary_path}")

    plot_learning_curves()
    plot_drcr_bar(eval_summary)
    plot_drcr_stability(eval_summary["agent"].tolist() if not eval_summary.empty else [])
    print(f"Saved plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
