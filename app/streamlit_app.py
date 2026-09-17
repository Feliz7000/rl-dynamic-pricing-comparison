"""Live companion to the static dashboard: runs the real trained agents and
the fitted demand model on the spot, so "what if" questions during a
presentation can be answered with the actual models rather than precomputed
data.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import DRCR_TAU, WARMUP_MONTHS  # noqa: E402
from env.pricing_env import REWARD_SCALE, PricingEnv  # noqa: E402
from training.config import load_config, set_global_seed  # noqa: E402
from training.train import LEARNING_AGENTS, build_agent, load_env_artifacts  # noqa: E402

ALGOS = [
    ("dqn", "DQN", "#5B8DEF"),
    ("ddpg", "DDPG", "#E1A339"),
    ("ppo", "PPO", "#C77DFF"),
    ("sac", "SAC", "#4FD1A5"),
]
HIST_COLOR = "#7C8896"

st.set_page_config(page_title="Dynamic Pricing RL — live", layout="wide")


@st.cache_resource(show_spinner="Loading data, demand model and checkpoints…")
def load_everything(source: str, seed: int):
    df, train_ids, test_ids, demand_model, scaler = load_env_artifacts(source)
    configs = {k: load_config(ROOT / "configs" / f"{k}.yaml") for k, _, _ in ALGOS}
    agents = {}
    for key, _, _ in ALGOS:
        cfg = configs[key]
        set_global_seed(seed)
        agent = build_agent(key, scaler.feature_dim, cfg.get("action_mode", "continuous"), cfg.get("k_buckets", 11), seed, cfg.get("hyperparams", {}))
        ckpt = ROOT / "results" / "models" / f"{key}_{seed}.pt"
        if key in LEARNING_AGENTS and ckpt.exists():
            agent.load(ckpt)
        agents[key] = (agent, cfg.get("action_mode", "continuous"))
    return df, train_ids, test_ids, demand_model, scaler, configs, agents


def make_env(df, demand_model, scaler, product_ids, mode, k_buckets):
    return PricingEnv(df=df, demand_model=demand_model, scaler=scaler, product_ids=product_ids, action_mode=mode, k_buckets=k_buckets, episode_horizon=50, allow_bootstrap_extension=False, seed=0)


def historical_path(rows, demand_model):
    rcr_hist = [r["rcr"] for r in rows[: WARMUP_MONTHS + 1]]
    out = []
    for t in range(WARMUP_MONTHS, len(rows)):
        row = rows[t]
        price = float(row["unit_price"])
        qty, _, _ = demand_model.predict_qty_single(row, price)
        rev = price * qty
        rcr = rev / max(float(row["traffic"]), 1e-6)
        rcr_hist.append(rcr)
        out.append({"month": row["month_year"], "price": price, "qty": qty, "revenue": rev, "rcr": rcr, "drcr": rcr - rcr_hist[len(rcr_hist) - 1 - DRCR_TAU]})
    return out


# --------------------------------------------------------------------------
st.sidebar.markdown("### Dynamic Pricing RL — live")
source = st.sidebar.selectbox("Data source", ["synthetic", "raw"], index=0)
available_seeds = sorted({int(p.stem.split("_")[1]) for p in (ROOT / "results" / "models").glob("sac_*.pt")})
seed = st.sidebar.selectbox("Checkpoint seed", available_seeds or [0], index=0)

try:
    df, train_ids, test_ids, demand_model, scaler, configs, agents = load_everything(source, seed)
except FileNotFoundError as exc:
    st.error(f"Missing artifacts: {exc}. Run scripts/fit_demand_model.py and the training pipeline first.")
    st.stop()

split = st.sidebar.radio("Products", ["held-out (test)", "training"], index=0)
product_pool = test_ids if split.startswith("held") else train_ids
pid = st.sidebar.selectbox("Product", product_pool, index=0)
rows = [dict(r) for r in df[df["product_id"] == pid].sort_values("t").to_dict("records")]
decision_rows = rows[WARMUP_MONTHS:]
category = rows[0]["product_category_name"].replace("_", " ")
st.sidebar.caption(f"{len(train_ids)} training / {len(test_ids)} held-out products · seed {seed} · {source} data")

st.title(f"{pid} · {category}")
tab_whatif, tab_live, tab_results = st.tabs(["Live what-if", "Live agent decisions", "Results"])

# --------------------------------------------------------------------------
with tab_whatif:
    st.markdown("Drag the price and the **fitted demand model** is evaluated on the spot at this month's real context (competitor prices, traffic, seasonality).")
    month_labels = [r["month_year"] for r in decision_rows]
    c1, c2 = st.columns([1, 3])
    with c1:
        m_idx = st.selectbox("Month", range(len(month_labels)), format_func=lambda i: month_labels[i], key="wi_month")
    row = decision_rows[m_idx]
    low, high = demand_model.price_bounds[pid]
    hist_price = float(row["unit_price"])
    with c2:
        price = st.slider("Candidate price", float(round(low, 2)), float(round(high, 2)), float(round(hist_price, 2)), step=0.05, key="wi_price")

    grid = np.linspace(low, high, 60)
    qty_grid = np.array([demand_model.predict_qty_single(row, float(p))[0] for p in grid])
    rev_grid = grid * qty_grid
    qty_now, _, _ = demand_model.predict_qty_single(row, price)
    qty_hist, _, _ = demand_model.predict_qty_single(row, hist_price)
    traffic = max(float(row["traffic"]), 1e-6)
    ref_rcr = float(rows[WARMUP_MONTHS + m_idx - 1]["rcr"]) if m_idx == 0 else historical_path(rows, demand_model)[m_idx - 1]["rcr"]
    drcr_now = price * qty_now / traffic - ref_rcr

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted demand", f"{qty_now:.1f} units", f"{qty_now - qty_hist:+.1f} vs seller's price")
    k2.metric("Predicted revenue", f"${price * qty_now:,.0f}", f"{(price * qty_now - hist_price * qty_hist) / max(hist_price * qty_hist, 1e-9) * 100:+.1f}% vs seller")
    k3.metric("DRCR vs last month", f"{drcr_now:.3f}", f"reward {REWARD_SCALE * drcr_now:+.1f}")
    k4.metric("Seller's price", f"${hist_price:.2f}", f"competitors {row['comp_1']:.0f} / {row['comp_2']:.0f} / {row['comp_3']:.0f}")

    chart_df = pd.DataFrame({"price": grid, "predicted revenue": rev_grid, "predicted units": qty_grid}).set_index("price")
    g1, g2 = st.columns(2)
    g1.line_chart(chart_df[["predicted revenue"]], color="#E1A339")
    g2.line_chart(chart_df[["predicted units"]], color="#6E8FB0")

    with st.expander("Where did each agent price this month?"):
        env_cache = {}
        marks = []
        for key, label, color in ALGOS:
            agent, mode = agents[key]
            env = env_cache.setdefault(mode, make_env(df, demand_model, scaler, product_pool, mode, configs["dqn"].get("k_buckets", 11)))
            state, _ = env.reset(options={"product_id": pid})
            chosen = None
            for step in range(m_idx + 1):
                action = agent.select_action(state, eval_mode=True)
                state, _, term, trunc, info = env.step(action)
                chosen = info["price"]
                if term or trunc:
                    break
            q, _, _ = demand_model.predict_qty_single(row, chosen)
            marks.append({"agent": label, "price": round(chosen, 2), "predicted units (at this month's context)": round(q, 1), "predicted revenue": round(chosen * q, 0)})
        st.dataframe(pd.DataFrame(marks), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
with tab_live:
    st.markdown("All four checkpoints price this product **month by month**, each from its own rolled-forward state. Advance one month at a time or run the whole window.")
    state_key = f"live_{source}_{seed}_{pid}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {"step": 0, "rows": []}
    live = st.session_state[state_key]

    def run_months(n_months: int):
        env_cache = {}
        traj = {key: [] for key, _, _ in ALGOS}
        for key, label, color in ALGOS:
            agent, mode = agents[key]
            env = env_cache.setdefault(mode, make_env(df, demand_model, scaler, product_pool, mode, configs["dqn"].get("k_buckets", 11)))
            state, _ = env.reset(options={"product_id": pid})
            for step in range(n_months):
                action = agent.select_action(state, eval_mode=True)
                state, reward, term, trunc, info = env.step(action)
                traj[key].append({"month": decision_rows[step]["month_year"], "price": info["price"], "qty": info["qty_hat"], "revenue": info["price"] * info["qty_hat"], "drcr": info["drcr"], "clipped": info["clipped"]})
                if term or trunc:
                    break
        return traj

    b1, b2, b3, _ = st.columns([1, 1, 1, 4])
    if b1.button("Next month ›", key="next"):
        live["step"] = min(live["step"] + 1, len(decision_rows))
    if b2.button("Run all months", key="all"):
        live["step"] = len(decision_rows)
    if b3.button("Reset", key="reset"):
        live["step"] = 0

    n = live["step"]
    hist = historical_path(rows, demand_model)
    if n == 0:
        st.info("Press **Next month** to let the agents make their first pricing decision.")
    else:
        traj = run_months(n)
        st.caption(f"Month {n} / {len(decision_rows)} · {decision_rows[n - 1]['month_year']}")
        table = []
        for key, label, color in ALGOS:
            last = traj[key][-1]
            cum_rev = sum(r["revenue"] for r in traj[key])
            hist_cum = sum(h["revenue"] for h in hist[:n])
            table.append({"agent": label, "price this month": f"${last['price']:.2f}", "predicted units": f"{last['qty']:.1f}", "DRCR": f"{last['drcr']:+.3f}", "cumulative revenue": f"${cum_rev:,.0f}", "vs seller": f"{(cum_rev - hist_cum) / hist_cum * 100:+.2f}%", "guard hit": "yes" if last["clipped"] else ""})
        table.append({"agent": "Seller (historical)", "price this month": f"${hist[n - 1]['price']:.2f}", "predicted units": f"{hist[n - 1]['qty']:.1f}", "DRCR": f"{hist[n - 1]['drcr']:+.3f}", "cumulative revenue": f"${sum(h['revenue'] for h in hist[:n]):,.0f}", "vs seller": "—", "guard hit": ""})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

        price_df = pd.DataFrame({label: [r["price"] for r in traj[key]] for key, label, _ in ALGOS}, index=[r["month"] for r in traj["sac"]])
        price_df["Seller"] = [h["price"] for h in hist[:n]]
        st.markdown("**Price trajectory**")
        st.line_chart(price_df, color=[c for _, _, c in ALGOS] + [HIST_COLOR])

        gain_df = pd.DataFrame({label: np.cumsum([r["revenue"] for r in traj[key]]) - np.cumsum([h["revenue"] for h in hist[:n]]) for key, label, _ in ALGOS}, index=price_df.index)
        st.markdown("**Cumulative revenue gained vs. the seller**")
        st.line_chart(gain_df, color=[c for _, _, c in ALGOS])

# --------------------------------------------------------------------------
with tab_results:
    summary_path = ROOT / "results" / "comparison_summary.csv"
    if summary_path.exists():
        st.markdown("**Held-out comparison** (from `results/comparison_summary.csv`)")
        st.dataframe(pd.read_csv(summary_path), hide_index=True, width="stretch")
    fig_dir = ROOT / "results" / "figures"
    cols = st.columns(2)
    for i, name in enumerate(["test_drcr_comparison.png", "drcr_stability.png", "learning_curves.png", "demand_model_monotonicity.png"]):
        p = fig_dir / name
        if p.exists():
            cols[i % 2].image(str(p), caption=name.replace("_", " ").replace(".png", ""), width="stretch")
    st.caption("Everything above is computed from the trained checkpoints, the fitted demand model and the logged runs in this repository — nothing is synthesized.")
