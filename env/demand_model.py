"""Hybrid demand-response model: predicts qty sold as a function of a
*candidate* price (which may differ from what was historically charged) plus
context. This is what lets PricingEnv answer "what would have happened at
this price?" for a logged/observational dataset -- it IS the environment's
transition/reward function.

Design (see plan): a log-log price-elasticity baseline plus a
gradient-boosted residual correction (captures nonlinear/competitor
interactions the linear baseline misses), with the residual capped to a
correction role.

The baseline is fit as a panel *fixed-effects* (within-product demeaned)
regression rather than a plain pooled regression. This matters: pooled
cross-sectional price levels are confounded with unrelated per-product/
per-category demand differences (a product priced high for reasons that have
nothing to do with price sensitivity), which can bias -- even flip the sign
of -- a single pooled log_price coefficient. Demeaning each product's
regressors by that product's own mean isolates the elasticity from each
product's own price movements over time, which is the economically
meaningful and monotonicity-safe quantity to simulate counterfactual prices
with.

Performance note: `predict_qty_single` is called once per PricingEnv.step()
-- i.e. many times per RL training run -- so it deliberately builds a raw
numpy row and calls the residual model with a bare ndarray, never a pandas
DataFrame. Constructing a pandas object (and paying sklearn's DataFrame
column-validation path) per call was measured at ~100ms/call, which made
even a single 50k-step training run impractical; the numpy fast path is
~1000x cheaper. Category is therefore encoded as an integer code (not a
pandas 'category' dtype) so HistGradientBoostingRegressor's categorical
support can be told about it via explicit column position instead of
`categorical_features="from_dtype"` (which requires a DataFrame both at fit
and predict time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

CATEGORY_COL = "product_category_name"

BASELINE_NUMERIC = [
    "log_price",
    "product_score",
    "freight_price",
    "qty_roll3",
    "month_sin",
    "month_cos",
    "weekday",
    "weekend",
    "holiday",
]

RESIDUAL_NUMERIC = BASELINE_NUMERIC + [
    "price_gap_1",
    "price_gap_2",
    "price_gap_3",
    "score_gap_1",
    "score_gap_2",
    "score_gap_3",
    "freight_gap_1",
    "freight_gap_2",
    "freight_gap_3",
    "revenue_roll3",
    "traffic_roll3",
]
_RESIDUAL_CAT_COL_IDX = len(RESIDUAL_NUMERIC)  # category code appended as the last column

# Extrapolation guard: candidate prices are clipped to this multiple of each
# product's own observed historical price range before querying the model.
PRICE_CLIP_LOW = 0.5
PRICE_CLIP_HIGH = 1.5

_TARGET_COL = "__log_qty__"


def _row_value(row, feat: str, price: float, log_price: float):
    """Looks up (or derives, for price-dependent features) a single feature
    value from a row that may be a pandas Series or a plain dict -- used by
    both the batch design matrix and the single-row fast path so the two
    stay in lockstep."""
    if feat == "log_price":
        return log_price
    if feat.startswith("price_gap_"):
        k = feat[-1]
        return price - row[f"comp_{k}"]
    return row[feat]


def _single_row_features(row, price: float, feature_list: list[str]) -> np.ndarray:
    price = max(price, 0.01)
    log_price = np.log(price)
    return np.array([_row_value(row, f, price, log_price) for f in feature_list], dtype=np.float64)


def _design_matrix(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Batch design matrix (used only at fit/evaluate time, where per-call
    overhead doesn't matter -- see module docstring for why the online
    per-step path avoids this entirely)."""
    price = df[price_col].clip(lower=0.01)
    out = pd.DataFrame(index=df.index)
    out["log_price"] = np.log(price)
    out["product_score"] = df["product_score"]
    out["freight_price"] = df["freight_price"]
    for k in (1, 2, 3):
        out[f"price_gap_{k}"] = price - df[f"comp_{k}"]
        out[f"score_gap_{k}"] = df["product_score"] - df[f"ps{k}"]
        out[f"freight_gap_{k}"] = df["freight_price"] - df[f"fp{k}"]
    out["qty_roll3"] = df["qty_roll3"]
    out["revenue_roll3"] = df["revenue_roll3"]
    out["traffic_roll3"] = df["traffic_roll3"]
    out["month_sin"] = df["month_sin"]
    out["month_cos"] = df["month_cos"]
    out["weekday"] = df["weekday"]
    out["weekend"] = df["weekend"]
    out["holiday"] = df["holiday"]
    out[CATEGORY_COL] = df[CATEGORY_COL].to_numpy()
    out["product_id"] = df["product_id"].to_numpy()
    return out


def compute_price_bounds(df_engineered: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-product [low, high] clip range for candidate prices, derived from
    that product's own observed historical price range."""
    agg = df_engineered.groupby("product_id")["unit_price"].agg(["min", "max"])
    bounds = {}
    for pid, row in agg.iterrows():
        bounds[pid] = (row["min"] * PRICE_CLIP_LOW, row["max"] * PRICE_CLIP_HIGH)
    return bounds


@dataclass
class DemandModel:
    baseline_coefs_: np.ndarray | None = None
    product_fe_: dict[str, dict] = field(default_factory=dict)
    global_means_: dict | None = None
    residual_model: HistGradientBoostingRegressor | None = None
    residual_cap_: float = 0.0
    price_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    category_to_code_: dict[str, int] = field(default_factory=dict)

    def _category_code(self, cat: str) -> int:
        return self.category_to_code_.get(cat, len(self.category_to_code_))

    def _fit_fe_baseline(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        fe_df = X[BASELINE_NUMERIC].copy()
        fe_df[_TARGET_COL] = y
        fe_df["product_id"] = X["product_id"].to_numpy()

        group_means = fe_df.groupby("product_id")[BASELINE_NUMERIC + [_TARGET_COL]].transform("mean")
        demeaned = fe_df[BASELINE_NUMERIC + [_TARGET_COL]] - group_means

        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(demeaned[BASELINE_NUMERIC].to_numpy(), demeaned[_TARGET_COL].to_numpy())
        self.baseline_coefs_ = ridge.coef_

        pid_means = fe_df.groupby("product_id")[BASELINE_NUMERIC + [_TARGET_COL]].mean()
        self.product_fe_ = {pid: row.to_dict() for pid, row in pid_means.iterrows()}
        self.global_means_ = fe_df[BASELINE_NUMERIC + [_TARGET_COL]].mean().to_dict()

        return self._baseline_predict(X)

    def _baseline_predict(self, X: pd.DataFrame) -> np.ndarray:
        pids = X["product_id"].to_numpy()
        fe_rows = [self.product_fe_.get(pid, self.global_means_) for pid in pids]
        fe_df = pd.DataFrame(fe_rows, index=X.index)
        demeaned = X[BASELINE_NUMERIC].to_numpy(dtype=float) - fe_df[BASELINE_NUMERIC].to_numpy(dtype=float)
        return fe_df[_TARGET_COL].to_numpy() + demeaned @ self.baseline_coefs_

    def fit(self, df_engineered: pd.DataFrame, price_col: str = "unit_price", target_col: str = "qty") -> None:
        X = _design_matrix(df_engineered, price_col)
        y = np.log1p(df_engineered[target_col].to_numpy())

        self.category_to_code_ = {
            cat: i for i, cat in enumerate(sorted(df_engineered[CATEGORY_COL].unique()))
        }

        baseline_pred = self._fit_fe_baseline(X, y)

        residual_target = y - baseline_pred
        cat_codes = X[CATEGORY_COL].map(self.category_to_code_).to_numpy(dtype=np.float64)
        X_res = np.column_stack([X[RESIDUAL_NUMERIC].to_numpy(dtype=np.float64), cat_codes])

        # Shallow, early-stopped, and L2-regularized so the residual model
        # can only make local corrections -- it must not be able to override
        # the baseline's monotonic price->demand direction, especially when
        # extrapolating to the clipped price range's edges on sparse data.
        self.residual_model = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            categorical_features=[_RESIDUAL_CAT_COL_IDX],
            random_state=0,
        )
        self.residual_model.fit(X_res, residual_target)

        # Hard cap on the residual's contribution (in log-qty space), sized
        # to the in-sample residual spread, so it corrects the baseline
        # rather than being able to flip its slope out-of-sample.
        residual_fit = self.residual_model.predict(X_res)
        self.residual_cap_ = float(3.0 * np.std(residual_fit - residual_fit.mean())) or 1.0

        self.price_bounds = compute_price_bounds(df_engineered)

    def _predict_log_qty(self, df: pd.DataFrame, price_col: str) -> np.ndarray:
        X = _design_matrix(df, price_col)
        baseline_pred = self._baseline_predict(X)
        cat_codes = X[CATEGORY_COL].map(lambda c: self._category_code(c)).to_numpy(dtype=np.float64)
        X_res = np.column_stack([X[RESIDUAL_NUMERIC].to_numpy(dtype=np.float64), cat_codes])
        residual_pred = self.residual_model.predict(X_res)
        residual_pred = np.clip(residual_pred, -self.residual_cap_, self.residual_cap_)
        return baseline_pred + residual_pred

    def predict_qty(self, df: pd.DataFrame, price_col: str = "unit_price") -> np.ndarray:
        """Batch prediction. `df[price_col]` holds the (possibly
        counterfactual) price to evaluate for each row."""
        log_qty = self._predict_log_qty(df, price_col)
        return np.clip(np.expm1(log_qty), 0.0, None)

    def predict_qty_single(self, context_row, candidate_price: float) -> tuple[float, float, bool]:
        """Single-row inference used by PricingEnv.step(): evaluates the
        model at a candidate price, clipped to the product's guarded price
        range (see PRICE_CLIP_LOW/HIGH), holding all other context fixed.
        `context_row` may be a pandas Series or a plain dict. Deliberately
        avoids building any pandas object -- see module docstring."""
        pid = context_row["product_id"]
        low, high = self.price_bounds.get(pid, (candidate_price * PRICE_CLIP_LOW, candidate_price * PRICE_CLIP_HIGH))
        clipped_price = float(np.clip(candidate_price, low, high))

        baseline_feat = _single_row_features(context_row, clipped_price, BASELINE_NUMERIC)
        fe = self.product_fe_.get(pid, self.global_means_)
        fe_means = np.array([fe[c] for c in BASELINE_NUMERIC], dtype=np.float64)
        baseline_pred = fe[_TARGET_COL] + float(self.baseline_coefs_ @ (baseline_feat - fe_means))

        residual_feat = _single_row_features(context_row, clipped_price, RESIDUAL_NUMERIC)
        cat_code = self._category_code(context_row[CATEGORY_COL])
        residual_input = np.concatenate([residual_feat, [cat_code]]).reshape(1, -1)
        residual_pred = float(self.residual_model.predict(residual_input)[0])
        residual_pred = float(np.clip(residual_pred, -self.residual_cap_, self.residual_cap_))

        log_qty = baseline_pred + residual_pred
        qty = max(float(np.expm1(log_qty)), 0.0)
        was_clipped = not np.isclose(clipped_price, candidate_price)
        return qty, clipped_price, was_clipped

    def evaluate(self, df_engineered: pd.DataFrame, price_col: str = "unit_price", target_col: str = "qty") -> dict:
        log_qty_true = np.log1p(df_engineered[target_col].to_numpy())
        log_qty_pred = self._predict_log_qty(df_engineered, price_col)
        return {
            "mae_log": mean_absolute_error(log_qty_true, log_qty_pred),
            "r2_log": r2_score(log_qty_true, log_qty_pred),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DemandModel":
        return joblib.load(path)
