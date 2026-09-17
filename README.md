# Dynamic Pricing RL Comparison — DQN vs DDPG vs PPO vs SAC

Companion codebase to `CIA3_Algorithm_Report_TeamOfFour.docx`, which compares four
reinforcement learning algorithms applied to the *same* dynamic-pricing MDP from
Liu et al. (2019), "Dynamic Pricing on E-commerce Platform with Deep Reinforcement
Learning: A Field Experiment" (arXiv:1912.02572) — the field-tested system Alibaba
built for Tmall.com. The four algorithms compared:

| Algorithm | Action space | On/off-policy |
|---|---|---|
| DQN  | Discrete price buckets | Off-policy |
| DDPG | Continuous price | Off-policy |
| PPO  | Continuous price | On-policy |
| SAC  | Continuous price | Off-policy |

This project builds a real, runnable comparison of all four on a shared environment,
grounded in a public retail-pricing dataset, to back the report with measured
numbers rather than the source paper's field-experiment figures.

## The core design problem

The dataset is *logged/historical* — one row is one product-month at the price the
seller actually charged. If an RL agent proposes a price that was never charged, we
have no ground truth for what would have happened. The fix: fit a supervised
**demand-response model** on the historical data (`env/demand_model.py`) and use it
as the environment's reward/transition function, so any agent-chosen price yields a
plausible simulated outcome. See that file's docstring for the full design
(fixed-effects elasticity baseline + capped GBM residual correction).

## Repository layout

```
data/{raw,synthetic,processed}/   # raw/synthetic CSVs, engineered feature tables, product split
env/                              # data_loader (feature engineering), demand_model, PricingEnv (Gymnasium)
agents/                           # base_agent interface, networks, replay/rollout buffers, all 4 algorithms + baselines
training/                         # train.py (CLI), config.py (YAML loader + seeding)
evaluation/                       # evaluate.py (held-out test rollout), compare.py (aggregate table + plots)
scripts/                          # data generation/download, demand model fitting, full-comparison batch runner
configs/                          # per-algorithm hyperparameters (dqn/ddpg/ppo/sac.yaml)
tests/                            # pytest suite: data loader, demand model, env, agent smoke tests
results/{logs,models,figures}/    # training/eval logs, checkpoints + demand model, output plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The real dataset is Kaggle's "Retail Price Optimization" (Olist-derived) —
`comp_1/2/3`, `ps1/2/3`, `fp1/2/3`, `lag_price`, `qty`, `customers` etc. already
closely match the MDP's state (own attributes, sales/traffic, price history,
competitiveness). To use it:

```bash
python scripts/download_data.py       # tries the Kaggle CLI, else prints manual steps
```

Until then (or always, for reproducible offline development), a schema-identical
**synthetic generator** is a permanent part of the pipeline, not a stub:

```bash
python scripts/generate_synthetic_data.py
```

Swapping real for synthetic data requires **no code changes** — everything
downstream (`data_loader.py`, `demand_model.py`, `pricing_env.py`) is schema-driven.

## Pipeline

```bash
# 1. Fit the demand model + state scaler + product train/test split (per data source)
python scripts/fit_demand_model.py --source synthetic

# 2. Train one (agent, seed)
python -m training.train --agent sac --seed 0 --config configs/sac.yaml

# 3. Evaluate a checkpoint on held-out test products (real months only, no bootstrap)
python -m evaluation.evaluate --agent sac --seed 0 --config configs/sac.yaml

# 4. Or run the whole multi-seed, multi-algorithm comparison + aggregate report:
TOTAL_STEPS=50000 SEEDS="0 1 2 3 4" bash scripts/run_full_comparison.sh
```

`evaluation/compare.py` aggregates all logs into `results/comparison_summary.csv`
(a measured version of the report's Section 6 table) and plots in
`results/figures/` (learning curves, held-out test DRCR bar chart, cross-seed
stability box plot).

## Presentation dashboard

Two front-ends, both driven by the real artifacts above — nothing on screen is
synthesized:

**1. Self-contained dashboard** (`docs/index.html`, ~1.3 MB, no server, no Python):
open it directly in a browser, or serve it through GitHub Pages (Settings → Pages →
branch `master`, folder `/docs`). Four tabs, deep-linkable by URL hash:

| Tab | What it shows |
|---|---|
| `#simulate` | 20 held-out products ranked by revenue uplift vs. the seller's own prices; per-product price trajectories (historical, static baseline, 4 agents, guard-rail hits); a ▶ Play control that replays the month-by-month **race** with a live scoreboard; revenue by product and the headline aggregate-uplift chart. |
| `#whatif` | Pick a product and month, drag a candidate price: the demand model's revenue and demand response curves, with the seller's price and each agent's chosen price marked. |
| `#training` | Rolling reward with cross-seed bands and the running mean, per algorithm and per seed (loss curves appear automatically for runs that logged them). |
| `#results` | The measured comparison table and chart, the MDP definition, the demand model's fit and guard rails. |

It is a React + Recharts app in `dashboard/` (UI adapted from a teammate's prototype),
built into one file with Vite:

```bash
python scripts/export_dashboard_data.py   # checkpoints + demand model + logs -> dashboard/data/dashboard_data.json
python scripts/build_dashboard.py         # export + npm run build -> docs/index.html   (needs Node 18+)
cd dashboard && npm run dev               # live-reload development server
```

**2. Live Streamlit app** (`app/streamlit_app.py`) for Q&A: the same what-if slider
and month-by-month agent decisions, but evaluated on the spot with the actual
checkpoints and demand model (any product, training or held-out).

```bash
streamlit run app/streamlit_app.py
```

## Design notes worth knowing before extending this

- **Leakage discipline**: state features representing "recent performance" are
  computed from data strictly before the decision month; competitor prices are
  treated as contemporaneous-but-exogenous. See `env/data_loader.py` docstring.
- **Fixed-effects demand baseline**: the elasticity regression is fit *within
  product* (demeaned), not pooled — a pooled fit is confounded by unrelated
  cross-product price-level differences and can flip the elasticity's sign. See
  `env/demand_model.py` docstring.
- **Reward-hacking guard**: candidate prices are clipped to `[0.5x, 1.5x]` of each
  product's observed historical range, and the GBM residual correction is capped
  relative to its in-sample spread, so it can only locally correct the baseline's
  monotonic elasticity, never override its direction out-of-distribution.
- **Fair comparison**: all four agents share a `BaseAgent` interface, identical
  network sizes, and the *same total environment-step budget* — the report's
  Section 7 point is a comparison of algorithm design, not training setup.
- **PPO's continuous mode**: run continuous (not discretized) here, unlike the
  report's "optionally discrete" framing, so DDPG/PPO/SAC are directly comparable
  on price fidelity. PPO also needs a markedly larger step budget than the
  off-policy agents before its value-bootstrap estimate calibrates and advantages
  become informative — confirmed on the Pendulum-v1 smoke test
  (`tests/test_agents_smoke.py`) and consistent with the report's own point about
  PPO's lower sample efficiency (Section 4.3).
- **Performance**: `PricingEnv.step()` and `DemandModel.predict_qty_single()`
  deliberately avoid constructing any pandas object on the hot path (a bare-numpy
  fast path is ~1000x cheaper than a pandas-DataFrame-per-step call) — this is
  what makes a real training run practical. See both files' docstrings.

## Tests

```bash
pytest tests/                                    # data loader, demand model, env (fast)
pytest tests/test_agents_smoke.py -v              # all 4 algorithms on standard Gymnasium envs (slower, ~6 min)
```

The smoke tests validate each algorithm's RL math (Bellman/TD loss, deterministic
policy gradient, clipped surrogate + GAE, max-entropy twin-critic) on standard
benchmarks (CartPole-v1, Pendulum-v1) *before* it ever touches the custom pricing
environment — isolating "the RL implementation is broken" from "the environment/
reward is broken".
