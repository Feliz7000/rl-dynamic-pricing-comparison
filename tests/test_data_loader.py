import numpy as np

from env.data_loader import (
    STATE_FEATURE_COLUMNS,
    StateScaler,
    engineer_features,
    train_test_product_split,
)
from scripts.generate_synthetic_data import generate


def _small_df():
    return generate(n_categories=3, products_per_category=4, n_months=10, seed=1)


def test_engineer_features_no_nans():
    df = engineer_features(_small_df())
    assert df[STATE_FEATURE_COLUMNS].isna().sum().sum() == 0
    assert df["drcr"].isna().sum() == 0


def test_rcr_drcr_sane():
    df = engineer_features(_small_df())
    assert (df["rcr"] >= 0).all()
    # DRCR should be exactly zero for each product's first row (t=0, no lookback)
    first_rows = df[df["t"] == 0]
    assert np.allclose(first_rows["drcr"], 0.0)


def test_train_test_product_split_disjoint_and_covers_categories():
    df = engineer_features(_small_df())
    train_ids, test_ids = train_test_product_split(df, test_frac=0.25, seed=0)
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(train_ids) | set(test_ids) == set(df["product_id"].unique())
    for cat, group in df.groupby("product_category_name"):
        cat_pids = set(group["product_id"].unique())
        assert cat_pids & set(test_ids), f"category {cat} has no test products"


def test_state_scaler_shapes():
    df = engineer_features(_small_df())
    train_ids, _ = train_test_product_split(df)
    scaler = StateScaler().fit(df, train_ids)
    row = df.iloc[0]
    vec = scaler.transform_row(row)
    assert vec.shape == (scaler.feature_dim,)
    assert np.isfinite(vec).all()
