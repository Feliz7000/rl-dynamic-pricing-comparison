import numpy as np

from env.data_loader import engineer_features
from env.demand_model import DemandModel
from scripts.generate_synthetic_data import generate


def _engineered_df():
    df = generate(n_categories=3, products_per_category=4, n_months=12, seed=2)
    return engineer_features(df)


def test_demand_model_fits_and_predicts_reasonably():
    df = _engineered_df()
    model = DemandModel()
    model.fit(df)
    metrics = model.evaluate(df)
    assert metrics["r2_log"] > 0.5


def test_demand_model_price_bounds_present_for_all_products():
    df = _engineered_df()
    model = DemandModel()
    model.fit(df)
    for pid in df["product_id"].unique():
        assert pid in model.price_bounds
        low, high = model.price_bounds[pid]
        assert low < high


def test_demand_model_monotonic_in_price_on_average():
    df = _engineered_df()
    model = DemandModel()
    model.fit(df)

    pid = df["product_id"].iloc[0]
    row = df[df["product_id"] == pid].iloc[-1]
    low, high = model.price_bounds[pid]
    prices = np.linspace(low, high, 15)
    qtys = [model.predict_qty_single(row, p)[0] for p in prices]

    # allow small local non-monotonicity from the residual GBM, but the
    # overall trend across the full range must be clearly decreasing
    assert qtys[0] > qtys[-1]
    # majority of consecutive steps should be non-increasing
    diffs = np.diff(qtys)
    assert (diffs <= 0).mean() > 0.6


def test_demand_model_save_load_roundtrip(tmp_path):
    df = _engineered_df()
    model = DemandModel()
    model.fit(df)
    path = tmp_path / "demand_model.pkl"
    model.save(path)
    loaded = DemandModel.load(path)

    row = df.iloc[0]
    qty_a, _, _ = model.predict_qty_single(row, float(row["unit_price"]))
    qty_b, _, _ = loaded.predict_qty_single(row, float(row["unit_price"]))
    assert np.isclose(qty_a, qty_b)
