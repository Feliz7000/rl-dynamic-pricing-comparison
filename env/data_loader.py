"""Load raw retail-pricing panels (real or synthetic) and engineer the MDP
state features described in the report: own attributes, recent sales/traffic
performance, pricing history, and competitiveness vs. similar products.

Leakage discipline: any feature that represents "recent performance" (rolling
qty/revenue/traffic, previous DRCR) is computed from data strictly BEFORE the
current row's month, since the agent must choose its action before that
month's outcome is known. Competitor columns (comp_k/ps_k/fp_k) and the
product's own entering price (lag_price) are treated as contemporaneous but
exogenous -- known at decision time and unaffected by the agent's own choice.
The row's own realized `unit_price`/`qty`/`total_price` are the *outcome* of
whatever price was actually charged historically and are never used as input
features (only as demand-model fitting targets / reward-computation inputs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "product_id",
    "product_category_name",
    "month_year",
    "qty",
    "total_price",
    "freight_price",
    "unit_price",
    "product_score",
    "customers",
    "weekday",
    "weekend",
    "holiday",
    "month",
    "year",
    "lag_price",
    "comp_1",
    "comp_2",
    "comp_3",
    "ps1",
    "ps2",
    "ps3",
    "fp1",
    "fp2",
    "fp3",
]

# Minimum number of prior months needed before all rolling/lag/DRCR features
# are well-defined (see module docstring). Episodes should not start their
# decision horizon before this many months of context exist.
WARMUP_MONTHS = 3
DRCR_TAU = 1

# Continuous state feature columns fed to agents (pre-standardization).
STATE_FEATURE_COLUMNS = [
    "lag_price",
    "lag_price_2",
    "price_change_pct",
    "product_score",
    "freight_price",
    "qty_roll3",
    "revenue_roll3",
    "traffic_roll3",
    "prev_drcr",
    "price_gap_1",
    "price_gap_2",
    "price_gap_3",
    "price_gap_mean",
    "score_gap_1",
    "score_gap_2",
    "score_gap_3",
    "freight_gap_1",
    "freight_gap_2",
    "freight_gap_3",
    "price_rank",
    "month_sin",
    "month_cos",
    "weekday",
    "weekend",
    "holiday",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input data at {path} is missing required columns: {sorted(missing)}")
    df["month_year"] = df["month_year"].astype(str)
    df = df.sort_values(["product_id", "month_year"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all engineered state-feature columns plus reward-helper columns
    (traffic, rcr, drcr) to a copy of df. Does not standardize or one-hot
    encode -- see FeatureScaler for that."""
    df = df.sort_values(["product_id", "month_year"]).reset_index(drop=True)
    df["t"] = df.groupby("product_id").cumcount()

    # --- traffic proxy & RCR/DRCR (reward-helper columns) ---
    traffic = df.groupby(["product_category_name", "month_year"])["qty"].transform("sum")
    df["traffic"] = traffic.clip(lower=1e-6)
    df["rcr"] = df["total_price"] / df["traffic"]
    g = df.groupby("product_id")
    df["drcr"] = df["rcr"] - g["rcr"].shift(DRCR_TAU)

    # --- recent sales/traffic performance (shifted -> no leakage) ---
    df["qty_prev"] = g["qty"].shift(1)
    df["revenue_prev"] = g["total_price"].shift(1)
    df["traffic_prev"] = g["traffic"].shift(1)
    g2 = df.groupby("product_id")
    df["qty_roll3"] = g2["qty_prev"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["revenue_roll3"] = g2["revenue_prev"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["traffic_roll3"] = g2["traffic_prev"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["prev_drcr"] = df.groupby("product_id")["drcr"].shift(1)

    # --- pricing history (lag_price already = price at t-1 in raw data) ---
    df["lag_price_2"] = df.groupby("product_id")["lag_price"].shift(1)
    df["price_change_pct"] = (df["lag_price"] - df["lag_price_2"]) / df["lag_price_2"].replace(0, np.nan)

    # --- competitiveness vs. comp_1/2/3 (contemporaneous, exogenous) ---
    for k in (1, 2, 3):
        df[f"price_gap_{k}"] = df["lag_price"] - df[f"comp_{k}"]
        df[f"score_gap_{k}"] = df["product_score"] - df[f"ps{k}"]
        df[f"freight_gap_{k}"] = df["freight_price"] - df[f"fp{k}"]
    df["price_gap_mean"] = df[["price_gap_1", "price_gap_2", "price_gap_3"]].mean(axis=1)

    def _price_rank(row):
        prices = [row["lag_price"], row["comp_1"], row["comp_2"], row["comp_3"]]
        order = np.argsort(np.argsort(prices))
        return order[0] / (len(prices) - 1)

    df["price_rank"] = df.apply(_price_rank, axis=1)

    # --- cyclical month encoding ---
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Safety net: any remaining NaNs are in warmup rows (t < WARMUP_MONTHS)
    # that episodes never start decisions on; fill defensively so the demand
    # model (fit on the full table) never chokes on NaNs.
    df[STATE_FEATURE_COLUMNS] = df[STATE_FEATURE_COLUMNS].fillna(0.0)
    df["drcr"] = df["drcr"].fillna(0.0)
    df["prev_drcr"] = df["prev_drcr"].fillna(0.0)

    return df


@dataclass
class StateScaler:
    """Standardizes STATE_FEATURE_COLUMNS and one-hot encodes category, fit
    only on training-product rows so evaluation never leaks test-product
    statistics into the observation normalization."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    categories_: list[str] | None = None

    def fit(self, df: pd.DataFrame, train_product_ids: list[str]) -> "StateScaler":
        train_df = df[df["product_id"].isin(train_product_ids)]
        values = train_df[STATE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.categories_ = sorted(df["product_category_name"].unique().tolist())
        return self

    @property
    def feature_dim(self) -> int:
        return len(STATE_FEATURE_COLUMNS) + len(self.categories_)

    def transform_row(self, row) -> np.ndarray:
        """`row` may be a pandas Series or a plain dict (PricingEnv's hot
        path uses plain dicts to avoid per-step pandas overhead)."""
        cont = np.array([row[c] for c in STATE_FEATURE_COLUMNS], dtype=np.float64)
        cont = (cont - self.mean_) / self.std_
        one_hot = np.zeros(len(self.categories_), dtype=np.float64)
        cat = row["product_category_name"]
        if cat in self.categories_:
            one_hot[self.categories_.index(cat)] = 1.0
        return np.concatenate([cont, one_hot]).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StateScaler":
        import joblib

        return joblib.load(path)


def train_test_product_split(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Product-level split (not row-level) to avoid leakage across time
    within a product. Stratified by category so both splits see every
    category."""
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for _, group in df.groupby("product_category_name"):
        pids = sorted(group["product_id"].unique())
        rng.shuffle(pids)
        n_test = max(1, int(round(len(pids) * test_frac)))
        test_ids.extend(pids[:n_test])
        train_ids.extend(pids[n_test:])
    return train_ids, test_ids
