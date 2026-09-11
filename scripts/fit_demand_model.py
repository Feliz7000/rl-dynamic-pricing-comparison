"""Fit and validate the demand model + state scaler, and persist both as
artifacts consumed by PricingEnv.

Usage:
    python scripts/fit_demand_model.py --source synthetic
    python scripts/fit_demand_model.py --source raw   # once the real Kaggle CSV is in data/raw/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.data_loader import (
    StateScaler,
    engineer_features,
    load_raw,
    train_test_product_split,
)
from env.demand_model import DemandModel

ROOT = Path(__file__).resolve().parent.parent


def time_based_holdout(df: pd.DataFrame, holdout_months: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-product time-based split: last `holdout_months` rows of each
    product's history are held out for validation, never a random row
    split (avoids leaking future context into training)."""
    max_t = df.groupby("product_id")["t"].transform("max")
    is_holdout = df["t"] > (max_t - holdout_months)
    return df[~is_holdout].copy(), df[is_holdout].copy()


def monotonicity_check(model: DemandModel, df: pd.DataFrame, out_path: Path, n_products: int = 6) -> None:
    """Sanity plot: predicted qty vs. a price grid, holding other features
    fixed, for a handful of sample products. Should be monotonically
    non-increasing given the elasticity-anchored baseline."""
    sample_ids = df["product_id"].drop_duplicates().sample(n=min(n_products, df["product_id"].nunique()), random_state=0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pid in zip(axes.flat, sample_ids):
        row = df[df["product_id"] == pid].iloc[-1]
        low, high = model.price_bounds.get(pid, (row["unit_price"] * 0.5, row["unit_price"] * 1.5))
        prices = np.linspace(low, high, 30)
        qtys = []
        for p in prices:
            qty, _, _ = model.predict_qty_single(row, p)
            qtys.append(qty)
        ax.plot(prices, qtys)
        ax.set_title(pid)
        ax.set_xlabel("candidate price")
        ax.set_ylabel("predicted qty")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "raw"], default="synthetic")
    parser.add_argument("--holdout-months", type=int, default=2)
    args = parser.parse_args()

    data_path = ROOT / "data" / args.source / "retail_price.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/generate_synthetic_data.py first "
            "(or scripts/download_data.py for the real dataset)."
        )

    df = load_raw(data_path)
    df = engineer_features(df)

    train_df, holdout_df = time_based_holdout(df, holdout_months=args.holdout_months)

    model = DemandModel()
    model.fit(train_df)

    train_metrics = model.evaluate(train_df)
    holdout_metrics = model.evaluate(holdout_df)
    print("Train metrics (log-qty space):", train_metrics)
    print("Holdout metrics (log-qty space):", holdout_metrics)

    monotonicity_check(model, df, ROOT / "results" / "figures" / "demand_model_monotonicity.png")
    print("Wrote monotonicity sanity plot to results/figures/demand_model_monotonicity.png")

    model_path = ROOT / "results" / "models" / "demand_model.pkl"
    model.save(model_path)
    print(f"Saved demand model to {model_path}")

    train_ids, test_ids = train_test_product_split(df)
    scaler = StateScaler().fit(df, train_ids)
    scaler.save(ROOT / "results" / "models" / "state_scaler.pkl")
    print(f"Saved state scaler to results/models/state_scaler.pkl (feature_dim={scaler.feature_dim})")

    split_manifest = {"train_product_ids": train_ids, "test_product_ids": test_ids}
    import json

    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "processed" / "product_split.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Saved product split ({len(train_ids)} train / {len(test_ids)} test) to data/processed/product_split.json")

    processed_path = ROOT / "data" / "processed" / f"{args.source}_engineered.csv"
    df.to_csv(processed_path, index=False)
    print(f"Saved engineered feature table to {processed_path}")


if __name__ == "__main__":
    main()
