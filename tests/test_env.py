import numpy as np
import pytest

from env.data_loader import StateScaler, engineer_features, train_test_product_split
from env.demand_model import DemandModel
from env.pricing_env import PricingEnv
from scripts.generate_synthetic_data import generate


def _build_env(action_mode):
    df = generate(n_categories=3, products_per_category=4, n_months=12, seed=3)
    df = engineer_features(df)
    train_ids, _ = train_test_product_split(df)

    model = DemandModel()
    model.fit(df[df["product_id"].isin(train_ids)])

    scaler = StateScaler().fit(df, train_ids)

    return PricingEnv(
        df=df,
        demand_model=model,
        scaler=scaler,
        product_ids=train_ids,
        action_mode=action_mode,
        episode_horizon=8,
        seed=0,
    )


@pytest.mark.parametrize("action_mode", ["discrete", "continuous"])
def test_random_rollout_no_crash_and_finite(action_mode):
    env = _build_env(action_mode)
    for _ in range(5):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            done = terminated or truncated
            steps += 1
        assert steps == env.episode_horizon


def test_reset_with_specific_product():
    env = _build_env("continuous")
    pid = env.product_ids[0]
    obs, info = env.reset(options={"product_id": pid})
    assert info["product_id"] == pid


def test_bootstrap_extension_disabled_terminates_early_if_history_short():
    df = generate(n_categories=1, products_per_category=2, n_months=5, seed=4)
    df = engineer_features(df)
    train_ids = list(df["product_id"].unique())
    model = DemandModel()
    model.fit(df)
    scaler = StateScaler().fit(df, train_ids)

    env = PricingEnv(
        df=df,
        demand_model=model,
        scaler=scaler,
        product_ids=train_ids,
        action_mode="continuous",
        episode_horizon=20,
        allow_bootstrap_extension=False,
        seed=0,
    )
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps < 20  # ran out of real history well before episode_horizon
