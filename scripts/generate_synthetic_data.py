"""Generate a synthetic retail-pricing panel matching the schema of the
Kaggle "Retail Price Optimization" dataset (Olist-derived).

This is a permanent fallback data source, not a throwaway stub: the demand
model and environment are built entirely against this schema, so dropping in
the real Kaggle CSV later (see scripts/download_data.py) requires no code
changes anywhere downstream.

Columns produced (matching the real dataset):
    product_id, product_category_name, month_year, qty, total_price,
    freight_price, unit_price, product_score, customers, weekday, weekend,
    holiday, month, year, comp_1, comp_2, comp_3, ps1, ps2, ps3, fp1, fp2,
    fp3, lag_price

Demand is generated from an explicit negative price-elasticity law so the
data has a learnable, monotonic price -> quantity relationship by
construction, which the demand model (scripts/fit_demand_model.py) is then
validated against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "electronics",
    "home_appliances",
    "fashion",
    "beauty",
    "toys",
    "sports",
    "furniture",
    "groceries",
    "books",
    "garden",
]


def _month_range(start_year: int, start_month: int, n_months: int) -> pd.PeriodIndex:
    start = pd.Period(f"{start_year}-{start_month:02d}", freq="M")
    return pd.period_range(start=start, periods=n_months, freq="M")


def generate(
    n_categories: int = 10,
    products_per_category: int = 8,
    n_months: int = 15,
    start_year: int = 2017,
    start_month: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = CATEGORIES[:n_categories]
    months = _month_range(start_year, start_month, n_months)

    rows = []
    product_meta = {}
    pid_counter = 0

    for cat in categories:
        cat_elasticity = rng.uniform(1.0, 2.5)
        cat_base_price = rng.lognormal(mean=np.log(50), sigma=0.5)

        for _ in range(products_per_category):
            pid_counter += 1
            product_id = f"P{pid_counter:04d}"
            base_price = cat_base_price * rng.lognormal(mean=0, sigma=0.3)
            base_demand = rng.lognormal(mean=np.log(40), sigma=0.6)
            product_score = float(np.clip(rng.normal(4.0, 0.5), 1.0, 5.0))
            freight_frac = rng.uniform(0.05, 0.20)

            product_meta[product_id] = dict(
                category=cat,
                elasticity=cat_elasticity * rng.uniform(0.85, 1.15),
                base_price=base_price,
                base_demand=base_demand,
                product_score=product_score,
                freight_frac=freight_frac,
            )

    # price random walk (mean-reverting toward each product's base_price)
    price_paths = {}
    for pid, meta in product_meta.items():
        price = meta["base_price"]
        path = []
        for _ in range(n_months):
            price = price + 0.15 * (meta["base_price"] - price) + rng.normal(0, meta["base_price"] * 0.03)
            price = max(price, meta["base_price"] * 0.3)
            path.append(price)
        price_paths[pid] = path

    for t, period in enumerate(months):
        month_num = period.month
        year_num = period.year
        seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * (month_num - 1) / 12.0)
        is_holiday = int(month_num in (11, 12))
        weekday_count = 20 if month_num != 2 else 18
        weekend_count = 8 if month_num != 2 else 8

        for pid, meta in product_meta.items():
            price = price_paths[pid][t]
            lag_price = price_paths[pid][t - 1] if t > 0 else price

            noise = rng.normal(0, 0.15)
            qty = (
                meta["base_demand"]
                * (price / meta["base_price"]) ** (-meta["elasticity"])
                * seasonality
                * (1.15 if is_holiday else 1.0)
                * np.exp(noise)
            )
            qty = max(qty, 0.0)

            freight_price = round(price * meta["freight_frac"] * rng.uniform(0.9, 1.1), 2)
            total_price = qty * price
            customers = max(1, int(rng.poisson(lam=max(qty * rng.uniform(0.6, 0.9), 1))))

            rows.append(
                dict(
                    product_id=pid,
                    product_category_name=meta["category"],
                    month_year=str(period),
                    qty=round(qty, 2),
                    total_price=round(total_price, 2),
                    freight_price=freight_price,
                    unit_price=round(price, 2),
                    product_score=round(meta["product_score"], 2),
                    customers=customers,
                    weekday=weekday_count,
                    weekend=weekend_count,
                    holiday=is_holiday,
                    month=month_num,
                    year=year_num,
                    lag_price=round(lag_price, 2),
                )
            )

    df = pd.DataFrame(rows)

    # competitor columns: sample 3 other products in the same category+month
    comp_cols = {f"comp_{k}": [] for k in (1, 2, 3)}
    ps_cols = {f"ps{k}": [] for k in (1, 2, 3)}
    fp_cols = {f"fp{k}": [] for k in (1, 2, 3)}

    grouped = df.groupby(["product_category_name", "month_year"])
    for _, group_idx in grouped.groups.items():
        idx = list(group_idx)
        for row_i in idx:
            others = [j for j in idx if j != row_i]
            picks = rng.choice(others, size=min(3, len(others)), replace=len(others) < 3) if others else []
            for k in range(3):
                if k < len(picks):
                    j = picks[k]
                    comp_price = df.at[j, "unit_price"] * rng.uniform(0.95, 1.05)
                    comp_score = df.at[j, "product_score"] * rng.uniform(0.97, 1.03)
                    comp_freight = df.at[j, "freight_price"] * rng.uniform(0.95, 1.05)
                else:
                    comp_price = df.at[row_i, "unit_price"] * rng.uniform(0.9, 1.1)
                    comp_score = df.at[row_i, "product_score"] * rng.uniform(0.9, 1.1)
                    comp_freight = df.at[row_i, "freight_price"] * rng.uniform(0.9, 1.1)
                comp_cols[f"comp_{k + 1}"].append(round(float(comp_price), 2))
                ps_cols[f"ps{k + 1}"].append(round(float(comp_score), 2))
                fp_cols[f"fp{k + 1}"].append(round(float(comp_freight), 2))

    for k in (1, 2, 3):
        df[f"comp_{k}"] = comp_cols[f"comp_{k}"]
        df[f"ps{k}"] = ps_cols[f"ps{k}"]
        df[f"fp{k}"] = fp_cols[f"fp{k}"]

    df = df.sort_values(["product_id", "month_year"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", type=int, default=10)
    parser.add_argument("--products-per-category", type=int, default=8)
    parser.add_argument("--months", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "synthetic" / "retail_price.csv"),
    )
    args = parser.parse_args()

    df = generate(
        n_categories=args.categories,
        products_per_category=args.products_per_category,
        n_months=args.months,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {df['product_id'].nunique()} products to {out_path}")


if __name__ == "__main__":
    main()
