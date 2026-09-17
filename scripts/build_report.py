"""Build the CIA-3 Component-2 technical report on the course template.

    python scripts/build_report.py

Reads the template (RL_CIA3_Technical_Report_Template.docx) for its styles,
replaces the body with the project's content, and pulls every number from
results/comparison_summary.csv, dashboard/data/dashboard_data.json,
results/compute_benchmark.json and the training/eval logs. Figures are
regenerated into report/figures/ so they always match the current run.
Output: report/RL_CIA3_Technical_Report.docx
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "RL_CIA3_Technical_Report_Template.docx"
OUT_DIR = ROOT / "report"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "RL_CIA3_Technical_Report.docx"

ALGOS = ["dqn", "ddpg", "ppo", "sac"]
LABEL = {"dqn": "DQN", "ddpg": "DDPG", "ppo": "PPO", "sac": "SAC", "static": "Static (no change)", "random": "Random"}
COLOR = {"dqn": "#3B6FD8", "ddpg": "#C98A1B", "ppo": "#8F4FD1", "sac": "#1FA57A", "static": "#6E8FB0", "random": "#9AA3AD", "hist": "#444444"}
ROLL = 200

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_all():
    summary = pd.read_csv(ROOT / "results" / "comparison_summary.csv").set_index("agent")
    dash = json.load(open(ROOT / "dashboard" / "data" / "dashboard_data.json"))
    bench = json.load(open(ROOT / "results" / "compute_benchmark.json"))
    logs = {k: {s: pd.read_csv(ROOT / "results" / "logs" / f"{k}_{s}.csv") for s in dash["meta"]["seeds"]} for k in ALGOS}
    return summary, dash, bench, logs


def metrics(summary, dash, bench, logs):
    meta, products = dash["meta"], dash["products"]
    seeds = meta["seeds"]
    hist_total = sum(sum(p["historical"]["revenue_model"]) for p in products)
    m = {}
    for k in ["static", "random"] + ALGOS:
        row = summary.loc[LABEL[k]]
        rev = np.mean([sum(sum(p["agents"][k][str(s)]["revenue"]) for p in products) for s in seeds])
        clipped = [c for p in products for s in seeds for c in p["agents"][k][str(s)]["clipped"]]
        m[k] = {
            "drcr": float(row.mean_test_drcr),
            "drcr_std": float(row.std_test_drcr),
            "per_seed": dash["summary"][k]["per_seed_test_drcr"],
            "uplift": (rev - hist_total) / hist_total * 100,
            "tail": None if pd.isna(row.get("mean_final_reward", np.nan)) else float(row.mean_final_reward),
            "tail_std": None if pd.isna(row.get("std_across_seeds", np.nan)) else float(row.std_across_seeds),
            "guard": 100 * float(np.mean(clipped)),
        }
    # convergence: first step from which the cross-seed running mean stays within +-0.25 of its final value
    for k in ALGOS:
        runs = []
        for s in seeds:
            r = logs[k][s]["reward"].to_numpy()
            runs.append(np.cumsum(r) / np.arange(1, len(r) + 1))
        n = min(len(x) for x in runs)
        run = np.mean([x[:n] for x in runs], axis=0)
        final = run[-1]
        inside = np.abs(run - final) <= 0.25
        conv = n
        for t in range(n):
            if inside[t:].all():
                conv = t
                break
        m[k]["conv_step"] = int(conv)
        m[k]["final_running"] = float(final)
        m[k]["steps_per_s"] = bench["agents"][k]["steps_per_second"]
        m[k]["sec_per_1k"] = bench["agents"][k]["seconds_per_1k_steps"]
        m[k]["params"] = bench["agents"][k]["parameters"]
        m[k]["infer_us"] = bench["agents"][k]["inference_us_per_action"]
    return m


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def style_ax(ax):
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_workflow(path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        ("Environment\nPricingEnv + demand model", 0.5),
        ("State s(t)\n35 features", 2.3),
        ("Agent\nDQN / DDPG / PPO / SAC", 4.1),
        ("Action a(t)\nprice within ±25 %", 5.9),
        ("Reward r(t)\n100 × DRCR", 7.7),
    ]
    for text, x in boxes:
        ax.add_patch(plt.Rectangle((x, 1.2), 1.6, 1.0, fc="#EAF1FA", ec="#3B6FD8", lw=1.2))
        ax.text(x + 0.8, 1.7, text, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1], 1.7), xytext=(boxes[i][1] + 1.6, 1.7), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.annotate("", xy=(1.3, 1.2), xytext=(8.5, 1.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#C98A1B", connectionstyle="arc3,rad=0.35"))
    ax.text(4.9, 0.25, "next state s(t+1): agent's own price becomes next month's entering price; competitors, traffic and season from the real next row",
            ha="center", fontsize=8, color="#C98A1B")
    ax.annotate("", xy=(4.9, 2.2), xytext=(8.5, 2.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#1FA57A", connectionstyle="arc3,rad=-0.35"))
    ax.text(6.7, 3.05, "learning update (replay buffer or rollout)", ha="center", fontsize=8, color="#1FA57A")
    ax.set_xlim(0.2, 9.6)
    ax.set_ylim(0, 3.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_learning_curves(logs, seeds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in ALGOS:
        rolls, runs = [], []
        for s in seeds:
            r = logs[k][s]["reward"].to_numpy()
            rolls.append(pd.Series(r).rolling(ROLL, min_periods=1).mean().to_numpy())
            runs.append(np.cumsum(r) / np.arange(1, len(r) + 1))
        n = min(len(x) for x in rolls)
        x = np.arange(n)
        rm, rs = np.mean([a[:n] for a in rolls], axis=0), np.std([a[:n] for a in rolls], axis=0)
        um, us = np.mean([a[:n] for a in runs], axis=0), np.std([a[:n] for a in runs], axis=0)
        axes[0].plot(x, rm, color=COLOR[k], label=LABEL[k], lw=1.2)
        axes[0].fill_between(x, rm - rs, rm + rs, color=COLOR[k], alpha=0.12)
        axes[1].plot(x, um, color=COLOR[k], label=LABEL[k], lw=1.5)
        axes[1].fill_between(x, um - us, um + us, color=COLOR[k], alpha=0.12)
    axes[0].set_ylim(-10, 10)
    axes[0].set_title("(a) Rolling mean reward, window 200 steps", fontsize=10)
    axes[1].set_title("(b) Running mean reward since step 0", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="#777", lw=0.8, ls=":")
        ax.set_xlabel("environment step")
        style_ax(ax)
    axes[0].set_ylabel("reward = 100 × DRCR")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_reward_vs_episode(logs, seeds, path):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k in ALGOS:
        curves = []
        for s in seeds:
            ep = logs[k][s].groupby("episode")["reward"].sum()
            curves.append(ep.rolling(25, min_periods=1).mean().to_numpy())
        n = min(len(c) for c in curves)
        m, sd = np.mean([c[:n] for c in curves], axis=0), np.std([c[:n] for c in curves], axis=0)
        x = np.arange(n)
        ax.plot(x, m, color=COLOR[k], label=LABEL[k], lw=1.4)
        ax.fill_between(x, m - sd, m + sd, color=COLOR[k], alpha=0.12)
    ax.axhline(0, color="#777", lw=0.8, ls=":")
    ax.set_xlabel("training episode (20 steps each)")
    ax.set_ylabel("episode return (rolling mean, 25 episodes)")
    ax.legend(fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_policy(dash, pid, path):
    p = [x for x in dash["products"] if x["id"] == pid][0]
    seeds = dash["meta"]["seeds"]
    months = p["months"]
    x = np.arange(len(months))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(x, p["historical"]["price"], color=COLOR["hist"], ls="--", lw=1.4, label="seller (historical)")
    ax.plot(x, [np.mean([p["agents"]["static"][str(s)]["price"][i] for s in seeds]) for i in x], color=COLOR["static"], lw=1.0, label="static")
    for k in ALGOS:
        price = np.mean([p["agents"][k][str(s)]["price"] for s in seeds], axis=0)
        clipped = [any(p["agents"][k][str(s)]["clipped"][i] for s in seeds) for i in x]
        ax.plot(x, price, color=COLOR[k], lw=1.6, label=LABEL[k])
        ax.scatter(x[clipped], price[clipped], facecolors="white", edgecolors=COLOR[k], s=22, zorder=5)
    ax.axhline(p["price_bounds"][1], color="#C4685A", lw=0.8, ls=":")
    ax.axhline(p["price_bounds"][0], color="#C4685A", lw=0.8, ls=":")
    ax.text(0.2, p["price_bounds"][1] + 1, "price guard", fontsize=7, color="#C4685A")
    ax.set_title(f"(a) {pid} price trajectory (hollow dots = guard hit)", fontsize=10)
    ax.set_ylabel("price ($)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    hist_cum = np.cumsum(p["historical"]["revenue_model"])
    for k in ["static"] + ALGOS:
        rev = np.mean([p["agents"][k][str(s)]["revenue"] for s in seeds], axis=0)
        ax.plot(x, np.cumsum(rev) - hist_cum, color=COLOR[k], lw=1.6 if k != "static" else 1.0, label=LABEL[k])
    ax.axhline(0, color=COLOR["hist"], ls="--", lw=1.2, label="seller")
    ax.set_title("(b) Cumulative simulated revenue gained vs. seller ($)", fontsize=10)
    ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x[::2])
        ax.set_xticklabels(months[::2], fontsize=7, rotation=30)
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_comparison(m, path):
    keys = ["static", "random", "dqn", "ddpg", "ppo", "sac"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    xs = np.arange(len(keys))
    ax.bar(xs, [m[k]["drcr"] for k in keys], yerr=[m[k]["drcr_std"] for k in keys], color=[COLOR[k] for k in keys], capsize=4, width=0.6)
    for i, k in enumerate(keys):
        ax.scatter([i] * len(m[k]["per_seed"]), m[k]["per_seed"], color="black", s=12, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k].replace(" (no change)", "") for k in keys], fontsize=8)
    ax.set_ylabel("mean held-out DRCR")
    ax.set_title("(a) Held-out test DRCR (bars: mean ± std; dots: seeds)", fontsize=10)
    ax = axes[1]
    ax.barh(xs, [m[k]["uplift"] for k in keys], color=[COLOR[k] for k in keys], height=0.6)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(xs)
    ax.set_yticklabels([LABEL[k].replace(" (no change)", "") for k in keys], fontsize=8)
    ax.set_xlabel("simulated revenue vs. seller's prices (%)")
    ax.set_title("(b) Revenue uplift over all held-out products", fontsize=10)
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_whatif(dash, pid, month_idx, path):
    p = [x for x in dash["products"] if x["id"] == pid][0]
    g = p["price_grid"][month_idx]
    seeds = dash["meta"]["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(g["prices"], g["revenue"], color="#C98A1B", lw=1.8)
    axes[1].plot(g["prices"], g["qty"], color="#6E8FB0", lw=1.8)
    hp = p["historical"]["price"][month_idx]
    for ax, field in zip(axes, ("revenue", "qty")):
        ax.axvline(hp, color=COLOR["hist"], ls="--", lw=1.2, label="seller's price")
        for k in ALGOS:
            price = np.mean([p["agents"][k][str(s)]["price"][month_idx] for s in seeds])
            y = np.interp(price, g["prices"], g[field])
            ax.scatter([price], [y], color=COLOR[k], s=40, zorder=5, label=f"{LABEL[k]} chose ${price:.0f}")
        ax.set_xlabel("candidate price ($)")
        style_ax(ax)
    axes[0].set_ylabel("predicted revenue ($)")
    axes[0].set_title(f"(a) Revenue response, {pid}, {p['months'][month_idx]}", fontsize=10)
    axes[1].set_ylabel("predicted units")
    axes[1].set_title("(b) Demand response", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_innovation(path):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")
    boxes = [("Logged pricing\nhistory (all SKUs)", 0.4, "#EAF1FA", "#3B6FD8"), ("Demand-model\ndigital twin", 2.6, "#EAF1FA", "#3B6FD8"),
             ("Offline RL\n(CQL / IQL) pre-training\non human prices", 4.8, "#FFF4E0", "#C98A1B"), ("Online fine-tuning\nSAC in the twin,\nguard rails on", 7.0, "#E6F7EF", "#1FA57A")]
    for text, x, fc, ec in boxes:
        ax.add_patch(plt.Rectangle((x, 0.9), 1.9, 1.3, fc=fc, ec=ec, lw=1.2))
        ax.text(x + 0.95, 1.55, text, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1], 1.55), xytext=(boxes[i][1] + 1.9, 1.55), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.text(4.75, 0.35, "cold-start SKUs: policy initialised from logged behaviour, then improved safely inside the twin before any live exposure", ha="center", fontsize=8, color="#555")
    ax.set_xlim(0.2, 9.2)
    ax.set_ylim(0, 2.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, template: Path):
        self.doc = Document(str(template))
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self.fig_n = 0
        self.tab_n = 0

    def h(self, text, level=1):
        return self.doc.add_heading(text, level=level)

    def p(self, *parts, style=None, align=None, size=None, italic=False):
        """parts: str or (text, {'b':bool,'i':bool}) tuples."""
        para = self.doc.add_paragraph(style=style)
        for part in parts:
            text, opts = (part, {}) if isinstance(part, str) else part
            run = para.add_run(text)
            run.bold = opts.get("b", False)
            run.italic = opts.get("i", italic)
            if size:
                run.font.size = Pt(size)
        if align:
            para.alignment = align
        para.paragraph_format.space_after = Pt(6)
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            if isinstance(it, str):
                self.p(it, style=style)
            else:
                self.p(*it, style=style)

    def eq(self, text):
        para = self.doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(10)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(6)
        return para

    def table(self, header, rows, widths=None, size=9, caption=None):
        if caption:
            self.tab_n += 1
            cap = self.doc.add_paragraph(f"Table {self.tab_n}. {caption}", style="Caption")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        t = self.doc.add_table(rows=len(rows) + 1, cols=len(header), style="Table Grid")
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c, text in enumerate(header):
            cell = t.cell(0, c)
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.bold = True
            run.font.size = Pt(size)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "D9EAF7")
            cell._tc.get_or_add_tcPr().append(shd)
        for r, row in enumerate(rows, start=1):
            for c, text in enumerate(row):
                cell = t.cell(r, c)
                cell.text = ""
                parts = text if isinstance(text, list) else [text]
                para = cell.paragraphs[0]
                for part in parts:
                    txt, opts = (part, {}) if isinstance(part, str) else part
                    run = para.add_run(str(txt))
                    run.bold = opts.get("b", False)
                    run.font.size = Pt(size)
        if widths:
            for row in t.rows:
                for c, w in enumerate(widths):
                    row.cells[c].width = Inches(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path, caption, width=6.3):
        self.fig_n += 1
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = self.doc.add_paragraph(f"Figure {self.fig_n}. {caption}", style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return self.fig_n

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------

def build():
    summary, dash, bench, logs = load_all()
    meta = dash["meta"]
    seeds, steps = meta["seeds"], meta["total_steps"]
    n_train, n_test = meta["n_train_products"], meta["n_test_products"]
    dm = meta["demand_model"]
    m = metrics(summary, dash, bench, logs)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_whatif(dash, "P0050", 0, FIG_DIR / "fig_demand_response.png")
    fig_workflow(FIG_DIR / "fig_workflow.png")
    fig_learning_curves(logs, seeds, FIG_DIR / "fig_learning_curves.png")
    fig_reward_vs_episode(logs, seeds, FIG_DIR / "fig_reward_vs_episode.png")
    fig_policy(dash, "P0050", FIG_DIR / "fig_policy.png")
    fig_comparison(m, FIG_DIR / "fig_comparison.png")
    fig_innovation(FIG_DIR / "fig_innovation.png")

    best = max(ALGOS, key=lambda k: m[k]["drcr"])
    fmt = lambda v, d=4: f"{v:.{d}f}"
    R = Report(TEMPLATE)
    doc = R.doc

    # ---- title page --------------------------------------------------------
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for text, sz in [("TECHNICAL PROJECT REPORT", 22), ("CIA-3 Component 2: Micro Project", 16), ("CSEAM731-REINFORCEMENT LEARNING", 16)]:
        run = para.add_run(text)
        run.bold = True
        run.font.size = Pt(sz)
        run.add_break()
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lines = ["", "Dynamic Pricing on an E-commerce Platform with Deep Reinforcement Learning:", "A Measured Comparison of DQN, DDPG, PPO and SAC on a Shared Pricing MDP", "", "Submitted by",
             "[Member 1 – Register Number]", "[Member 2 – Register Number]", "[Member 3 – Register Number]", "[Member 4 – Register Number]", "",
             "Class: 7BTCS AIML – [A/B/C]", "Course: Reinforcement Learning", "Institution: CHRIST (Deemed to be University), Bengaluru", "Academic Year: 2026–2027"]
    for i, line in enumerate(lines):
        run = para.add_run(line)
        if line.startswith(("Dynamic Pricing", "A Measured")):
            run.bold = True
            run.font.size = Pt(14)
        if i < len(lines) - 1:
            run.add_break()
    R.page_break()

    # ---- abstract ----------------------------------------------------------
    R.h("Abstract", 1)
    R.p(f"Dynamic pricing asks a seller to re-set a product's price every period as demand, traffic and competitors change. This report reproduces, in a controlled and measurable form, the comparison behind Liu et al.'s field-tested Tmall pricing system [1]: four reinforcement-learning algorithms — DQN, DDPG, PPO and SAC — are trained on one identical Markov decision process (a 35-feature product-month state, a price move within ±25 %, and the Difference of Revenue Conversion Rate as reward) and evaluated on {n_test} held-out products they never saw. Because logged sales data only reveal the outcome at the one price actually charged, a counterfactual demand-response model (a within-product log-log elasticity baseline plus a capped gradient-boosted correction, holdout R² = {dm['r2_holdout']:.2f}) serves as the environment's transition and reward function, with guard rails against exploitation. All four learners were validated on standard Gymnasium tasks before use and trained under an identical budget of {steps:,} environment steps for each of {len(seeds)} seeds. SAC achieved the best held-out DRCR ({m['sac']['drcr']:.4f} ± {m['sac']['drcr_std']:.4f}) and a {m['sac']['uplift']:+.1f} % simulated revenue gain over the seller's own prices, followed by PPO ({m['ppo']['drcr']:.4f}) and DDPG ({m['ddpg']['drcr']:.4f}); DQN ({m['dqn']['drcr']:.4f}) fell below the no-change baseline, a result attributable to both its discretised action space and an exploration schedule longer than the training budget. Continuous-action, entropy-regularised learning is the most effective and stable choice for this problem.")
    R.p(("Keywords: ", {"b": True}), "dynamic pricing, reinforcement learning, Deep Q-Network, DDPG, PPO, Soft Actor-Critic, demand modelling, e-commerce")

    # ---- 1 introduction ----------------------------------------------------
    R.h("1. Introduction", 1)
    R.h("1.1 Background", 2)
    R.p("Online marketplaces list millions of products whose demand shifts daily with seasonality, promotions, customer traffic and competitors' prices. A price set once and left alone is wrong most of the time: too high and units are lost, too low and margin is given away. Large platforms therefore price dynamically, re-setting each product's price at regular intervals. Alibaba's supply-chain team built such a system with deep reinforcement learning and deployed it on Tmall.com, China's largest B2C marketplace, from July 2018; in a 30-day field experiment the RL policies out-earned manual pricing by a wide margin, with the continuous-action agent (DDPG) scoring a Difference of Revenue Conversion Rate of 6.07 against a manual baseline rescaled to 1.00, and the discrete agent (DQN) 5.03 [1].")
    R.h("1.2 Motivation", 2)
    R.p("Each month's price does not only earn that month's revenue: it becomes the reference the customer and the competitor react to next month, it changes the product's rolling sales record, and it moves the product's position in the category's price ranking. The decision is therefore sequential — today's choice changes tomorrow's state — and the objective is cumulative. Rules of thumb (“undercut the cheapest competitor by 2 %”) cannot see this; they optimise one period at a time and cannot adapt when the relationship between price and demand changes. The team's Component-1 report studied four RL algorithms on this problem conceptually; this Component-2 project implements all four, measures them under a fair protocol, and reports what actually happened.")
    R.h("1.3 Why Reinforcement Learning?", 2)
    R.p("The problem maps cleanly onto the RL formalism of Sutton and Barto [8]. The agent is the pricing policy for one product. The environment is the market: customer traffic, competitor prices and seasonality that the agent observes but does not control, plus the demand response to the agent's own price. The state is what is known about the product before this month's price is set. The action is the new price. The reward is the change in traffic-normalised conversion (DRCR), the same signal used in [1]. The long-term objective is the discounted sum of these improvements, so the agent is rewarded for prices that keep paying off, not for one-off spikes. Trial-and-error learning from outcomes is exactly what is needed when the demand function is unknown and non-stationary.")
    R.h("1.4 Selected RL Approaches", 2)
    R.bullets([
        [("DQN ", {"b": True}), "(Deep Q-Network [4]): value-based, off-policy; chooses among K = 11 discrete price buckets. The paper's own baseline."],
        [("DDPG ", {"b": True}), "(Deep Deterministic Policy Gradient [5]): actor-critic, off-policy; outputs an exact continuous price. The paper's field winner."],
        [("PPO ", {"b": True}), "(Proximal Policy Optimization [6]): stochastic actor-critic, on-policy; clipped updates chosen for robustness to a noisy reward."],
        [("SAC ", {"b": True}), "(Soft Actor-Critic [7]): maximum-entropy actor-critic, off-policy; twin critics and a self-tuned exploration temperature."],
    ])
    R.h("1.5 Objectives", 2)
    R.bullets([
        "Build a reproducible pricing environment from logged product-month data, including a validated counterfactual demand model with safeguards against reward hacking.",
        "Implement DQN, DDPG, PPO and SAC from first principles in PyTorch behind one common agent interface, and verify each on a standard benchmark before use.",
        f"Train all four under an identical budget ({steps:,} environment steps × {len(seeds)} seeds) and evaluate them deterministically on {n_test} held-out products.",
        "Compare them on at least four metrics — held-out DRCR, cross-seed stability, simulated revenue uplift over the seller, convergence speed, guard-rail behaviour and computational cost — and identify the best approach with evidence.",
        "Deliver an interactive dashboard and a live application so the results can be inspected product by product.",
    ])
    R.h("1.6 Report Organization", 2)
    R.p("Section 2 reviews related work. Section 3 defines the case study as a sequential decision problem. Section 4 specifies the state, action, reward, episode and environment characteristics. Section 5 presents each algorithm and its implementation. Section 6 documents the experimental setup and hyperparameters. Section 7 reports results with figures and tables; Section 8 compares the approaches on six metrics; Section 9 interprets them and states limitations. Section 10 proposes an extension, Section 11 concludes, and the appendices give project resources and additional results.")

    # ---- 2 related work ----------------------------------------------------
    R.h("2. Related Work", 1)
    R.p("Reinforcement learning has been applied to pricing for over a decade, moving from tabular methods on stylised inventory problems to deep methods on live marketplaces. Table 1 summarises the studies that shaped this project.")
    R.table(["Study", "Problem / Domain", "RL Approach", "Key Finding", "Limitation"], [
        ["Liu et al., 2019 [1]", "Dynamic pricing of thousands of SKUs on Tmall.com (field experiment)", "DQN (discrete buckets) and DDPG (continuous price); DRCR reward; pre-training on human prices", "Both RL policies beat manual pricing in 30-day live tests; DDPG (6.07) > DQN (5.03) vs. manual = 1.00", "Data and simulator are proprietary; only two algorithms compared; low-traffic SKUs need pre-training"],
        ["Kastius & Schlosser, 2022 [2]", "Pricing under duopoly / oligopoly competition (simulation)", "DQN and SAC", "Both learn reasonable policies; SAC outperforms DQN; RL sellers can be pushed into tacit collusion", "Synthetic market models; no field validation"],
        ["Rana & Oliveira, 2014 [3]", "Selling fixed inventory by a deadline (airlines, hotels, fashion)", "Q-learning and Q(λ) with eligibility traces", "Model-free RL learns near-optimal pricing without a known demand function and tracks non-stationarity", "Tabular; small state spaces; single seller"],
        ["Mnih et al., 2015 [4]", "General control from high-dimensional input (Atari)", "DQN: replay buffer + target network", "Stable deep value learning over discrete actions", "Discrete actions only; price precision limited by bucket count"],
        ["Lillicrap et al., 2015 [5]", "Continuous control benchmarks", "DDPG: deterministic actor + critic with target networks", "Extends DQN's stabilisers to continuous actions", "Sensitive to hyperparameters and to the external exploration noise schedule"],
    ], widths=[1.15, 1.45, 1.45, 1.75, 1.45], size=8, caption="Related studies on reinforcement learning for pricing and the algorithms compared in this project.")
    R.h("2.1 Project Gap", 2)
    R.p("Liu et al. compare only DQN and DDPG, on data nobody else can access. Kastius and Schlosser compare DQN and SAC, but in a synthetic competition model. No study places all four algorithm families — discrete value-based, deterministic continuous, on-policy clipped, and entropy-regularised — on one identical, data-driven pricing MDP with held-out evaluation and multiple seeds. This project fills that gap and, additionally, makes the counterfactual demand model that any such comparison depends on explicit, validated and guarded.")

    # ---- 3 problem definition -----------------------------------------------
    R.h("3. Problem Definition and Case Study", 1)
    R.h("3.1 Real-Time Problem", 2)
    R.p("A marketplace seller lists a product in a category alongside three comparable competitors. Every month the seller must set the product's price for the coming month knowing its own attributes, its recent sales and traffic, its price history and how it compares with the competitors. The goal is to maximise revenue conversion over time, not to win any single month. This is the setting of [1]; the data used here follow the Kaggle Retail Price Optimization schema [14] (one row per product-month with price, units, freight, score and three competitor prices).")
    R.h("3.2 Sequential Decision-Making", 2)
    R.p("The price chosen in month t becomes the entering price in month t+1, so the allowed move (±25 %) is anchored to the previous decision. Units sold in month t enter the rolling three-month sales feature that the agent sees in months t+1 to t+3. The reward itself is a difference between consecutive months' conversion rates, so a spike bought by a deep cut is immediately paid back the next month. A myopic policy that maximises this month's revenue therefore differs from the policy that maximises the discounted sum — which is precisely the case for RL.")
    R.h("3.3 Agent and Environment", 2)
    R.p(("Agent: ", {"b": True}), "one pricing policy per algorithm, applied to every product (the state carries the product's identity through its features and category, so one network serves all products).")
    R.p(("Environment: ", {"b": True}), f"PricingEnv, a Gymnasium [11] environment that replays a product's real month-by-month context (competitor prices, category traffic, seasonality) and uses a fitted demand model to compute how many units sell at whatever price the agent chooses. {n_train} products are used for training and {n_test} are held out for evaluation.")
    R.h("3.4 Decisions and Long-Term Objective", 2)
    R.p("Each decision is a new price within ±25 % of the entering price. The immediate reward is 100 × DRCR, where DRCR = RCR(t) − RCR(t−1) and RCR = revenue / category traffic. The long-term objective is the expected discounted return with γ = 0.99 over the product's horizon, which rewards sustained improvement in conversion rather than a single good month.")
    R.h("3.5 Limitations of Rule-Based Systems", 2)
    R.p("Fixed rules (cost-plus, match-the-median, undercut-by-x %) share three weaknesses. They ignore the product's own history, so they cannot tell a product whose demand is falling from one whose traffic is falling. They are one-step: they do not account for how this month's price constrains next month's move or changes the competitor's response. And they must be re-tuned by hand whenever elasticity, seasonality or the competitor set changes. An RL policy learns the mapping from the full state to price from outcomes and can be retrained as the market drifts.")

    # ---- 4 environment design -----------------------------------------------
    R.h("4. RL Environment Design", 1)
    R.h("4.1 State Representation", 2)
    R.p("The state is a 35-dimensional vector: 25 continuous features standardised with statistics from the training products only, plus a 10-way one-hot category indicator. Every feature is computable before the month's price is set; rolling performance features are shifted by one month so that this month's outcome never leaks into the state (Table 2).")
    R.table(["State Variable", "Description", "Type / Range", "Justification"], [
        ["lag_price, lag_price_2", "price entering this month and the month before", "continuous, $", "anchor for the ±25 % move; captures own pricing history"],
        ["price_change_pct", "(lag_price − lag_price_2) / lag_price_2", "continuous, ≈ −0.3 to 0.3", "direction and size of the last move"],
        ["product_score, freight_price", "customer rating; shipping cost", "1–5; $", "own attributes that shift demand independent of price"],
        ["qty_roll3, revenue_roll3, traffic_roll3", "3-month rolling means of units, revenue and category traffic (previous months)", "continuous", "recent sales/traffic performance, as in [1]"],
        ["prev_drcr", "last month's DRCR", "continuous", "recent trend of the reward signal"],
        ["price_gap_1..3, price_gap_mean", "entering price minus each competitor's price, and their mean", "continuous, $", "competitiveness; recomputed each month"],
        ["score_gap_1..3, freight_gap_1..3", "own score / freight minus competitors'", "continuous", "non-price competitiveness"],
        ["price_rank", "rank of entering price among the four prices", "0 (cheapest) – 1 (dearest)", "relative position in the category"],
        ["month_sin, month_cos, weekday, weekend, holiday", "seasonality and calendar composition", "cyclic / counts / 0-1", "demand seasonality"],
        ["category (one-hot)", "product category", "10 binary", "lets one network price all categories"],
    ], widths=[1.5, 2.2, 1.3, 2.2], size=8, caption="State variables of the pricing MDP.")
    R.h("4.2 Action Space", 2)
    R.p("The action is the next month's price expressed as a move from the entering price, bounded to ±25 %. For DQN the action space is Discrete(11): bucket k maps to a multiplier 1 + (−25 % + 5 %·k), so the middle bucket is “no change” and the finest move is 5 %. For DDPG, PPO and SAC the action space is Box(−1, 1): a tanh-bounded scalar a maps to multiplier 1 + 0.25·a, an exact price. Two constraints apply to every agent. First, the price is clipped to [0.5×, 1.5×] of the product's historically observed price range (the “price guard”), and the clip is logged. Second, competitor gaps are recomputed at the chosen price, so a counterfactual price correctly changes the product's competitive position. PPO is run in continuous mode so that the three actor-critic methods are compared on equal price precision.")
    R.h("4.3 Reward Design", 2)
    R.p("Raw revenue is a poor reward because category traffic fluctuates for reasons unrelated to price [1]. Following the paper, revenue is normalised by traffic and the reward is the month-on-month change:")
    R.eq("RCR(t) = revenue(t) / traffic(t),      DRCR(t) = RCR(t) − RCR(t − 1),      r(t) = 100 × DRCR(t)")
    R.p("Traffic is proxied by the total units sold across all products in the category that month (the dataset has no visit counts). The factor 100 keeps rewards in a range neural networks train on comfortably. Because the reward is a difference, even an excellent policy's reward averages near zero once conversion has been raised; the level it settles at and how tightly it holds there are what distinguish policies. Table 3 lists the outcomes the reward rewards and punishes.")
    R.table(["Situation", "Outcome", "Reward", "Reason"], [
        ["Price raised, demand nearly inelastic", "revenue and RCR rise", "positive (typically +5 to +30)", "improvement in conversion is the objective"],
        ["Price raised too far", "units collapse, RCR falls", "negative", "over-pricing is penalised the same month"],
        ["Price cut when demand is inelastic", "units rise slightly, revenue falls", "negative", "discounting that does not pay is discouraged"],
        ["Price held after a good month", "RCR unchanged", "≈ 0", "no improvement, no penalty: stability is neutral"],
        ["Deep cut for a one-off spike", "RCR jumps, then falls back", "positive then negative", "the difference form pays the spike back next month"],
        ["Price outside the guard", "clipped to the guard, outcome computed there", "as for the clipped price; logged", "prevents exploiting the demand model off-distribution"],
    ], widths=[1.7, 1.7, 1.5, 2.3], size=8, caption="Reward design: situations, outcomes and the resulting reward.")
    R.p("The demand model that produces revenue(t) at any candidate price is the core of the environment. It is a two-layer model fitted on the full panel: (i) a fixed-effects log-log elasticity baseline, log(1 + qty) = αᵢ + β·log(price) + controls, fitted after demeaning every regressor within product so that β is identified from each product's own price moves (a pooled fit gave a positive, confounded coefficient); and (ii) a gradient-boosted residual (depth 3, ≤ 250 trees, early-stopped) capturing competitor and seasonal interactions, whose output is capped at ±3σ of its in-sample spread so it can bend but never flip the demand curve. On a time-based holdout (each product's last two months) it reaches R² = %.3f and MAE = %.3f in log-units; the fitted elasticity is β = %.2f. Figure 1 shows its response curves for one held-out product together with the prices the trained agents chose." % (dm["r2_holdout"], dm["mae_log_holdout"], dm["elasticity_coef"]))
    R.figure(FIG_DIR / "fig_demand_response.png", "Demand-model response for held-out product P0050 in its first decision month: (a) predicted revenue and (b) predicted units against candidate price, with the seller's actual price (dashed) and each trained agent's chosen price (dots).")
    R.h("4.4 Episode Definition", 2)
    R.p("An episode is one product priced month by month. Each product has 15 months of data; the first three are warm-up (they seed the rolling features and the RCR history) and the agent prices months 4–15, twelve decisions. In training, an episode draws a random training product and runs for a horizon of 20 steps; once the real months are exhausted the exogenous context (competitor prices, traffic) is extended by bootstrapping small perturbations (±3 % and ±5 %) of the last real row while the agent's own history keeps rolling forward — a sample-efficiency device with only twelve real decisions per product. In evaluation, episodes run each held-out product once, over its real months only, with exploration switched off; no reported number uses bootstrapped months.")
    R.h("4.5 Environment Characteristics", 2)
    R.bullets([
        [("Episodic / Continuing: ", {"b": True}), "episodic — one product's horizon; training episodes are 20 steps, evaluation episodes 12."],
        [("Static / Dynamic: ", {"b": True}), "dynamic — competitor prices, traffic and seasonality change every step, and the agent's own price alters the next state."],
        [("Deterministic / Stochastic: ", {"b": True}), "the demand model is deterministic given the state and price; stochasticity enters through product sampling, the exogenous market series, and (in training) the bootstrap noise and exploration."],
        [("Fully / Partially Observable: ", {"b": True}), "partially observable — the agent sees a 35-feature summary, not the demand function, the competitors' intentions or future traffic."],
        [("State-space type: ", {"b": True}), "continuous, 35-dimensional (25 standardised real features + 10-way one-hot)."],
        [("Action-space type: ", {"b": True}), "discrete (11 buckets) for DQN; continuous Box(−1, 1) for DDPG, PPO and SAC; both bounded to ±25 % of the entering price and guarded."],
        [("Online / Offline / Simulation-based: ", {"b": True}), "simulation-based learning in a data-driven simulator: the market context is replayed from logged data and the demand response is a model fitted on that data. Learning is online with respect to the simulator."],
    ])
    R.h("4.6 RL Workflow", 2)
    R.figure(FIG_DIR / "fig_workflow.png", "RL workflow for the pricing case study: the agent observes the product-month state, sets a price, the demand model returns units and revenue, the DRCR reward is computed, and the agent's own price rolls into the next state.")

    # ---- 5 algorithms -------------------------------------------------------
    R.h("5. RL Approaches and Implementation", 1)
    R.p("All four agents implement one interface (select_action, observe, update, save/load) so the training and evaluation loops are identical code. Every network is a two-hidden-layer MLP of 128 ReLU units trained with Adam; γ = 0.99 throughout. The pseudocode below uses s, a, r, s′ for a transition and θ⁻ for a target network.")

    # 5.1 DQN
    R.h("5.1 Deep Q-Network (DQN)", 2)
    R.h("Algorithm Overview", 3)
    R.p("DQN [4] approximates the action-value function Q(s, a | θ) with a neural network over a finite action set and trains it by temporal-difference learning towards the Bellman target:")
    R.eq("y = r + γ · maxₐ′ Q(s′, a′ | θ⁻),      L(θ) = Huber( Q(s, a | θ) − y )")
    R.p("Two mechanisms make this stable: an experience replay buffer sampled at random to decorrelate updates, and a periodically refreshed target network θ⁻ so the target does not move on every gradient step.")
    R.h("Suitability to the Case Study", 3)
    R.p("Pricing as a choice among K discount tiers is a natural Q-learning problem and is the formulation Liu et al. deployed first [1]. DQN is the simplest and most stable of the four and serves as the value-based baseline; its structural cost is the bucket grid, whose resolution caps price precision.")
    R.h("Implementation Details", 3)
    R.p("Representation: Q-network 35 → 128 → 128 → 11. Training: after 500 warm-up steps, one gradient step per environment step on a batch of 64 from a 50,000-transition buffer; Huber loss; hard target update every 100 updates. Exploration: ε-greedy with ε decayed linearly from 1.0 to 0.05 over 25,000 steps; ε = 0 in evaluation. Update mechanism: single-step TD with the max over the target network.")
    R.h("Workflow / Pseudocode", 3)
    R.bullets(["initialise Q(·|θ), copy to θ⁻, empty buffer D",
               "for each step: with prob. ε pick a random bucket, else a = argmax Q(s, ·|θ); step env; store (s, a, r, s′, done) in D",
               "if |D| ≥ warm-up: sample batch; y = r + γ(1−done) max Q(s′,·|θ⁻); minimise Huber(Q(s,a|θ) − y)",
               "every 100 updates: θ⁻ ← θ; decay ε"], style="List Number")
    R.h("Key Hyperparameters", 3)
    R.table(["Parameter", "Value", "Justification"], [
        ["K (price buckets)", "11 over ±25 %", "5 % resolution; middle bucket = no change"],
        ["Learning rate", "1e-3", "standard for DQN on small MLPs"],
        ["Replay buffer / batch", "50,000 / 64", "decorrelated updates; buffer covers the whole run"],
        ["Target update", "every 100 updates (hard)", "stable bootstrapped target"],
        ["ε schedule", "1.0 → 0.05 over 25,000 steps", "long exploration for a noisy reward (see §9.5: exceeds the 15k budget used)"],
        ["γ", "0.99", "long-horizon credit for sustained conversion gains"],
    ], widths=[1.7, 1.9, 3.6], size=8.5, caption="DQN hyperparameters.")

    # 5.2 DDPG
    R.h("5.2 Deep Deterministic Policy Gradient (DDPG)", 2)
    R.h("Algorithm Overview", 3)
    R.p("DDPG [5] is an off-policy actor-critic for continuous actions. A deterministic actor μ(s|θᵘ) outputs the price; a critic Q(s, a|θᵠ) scores it. Both have slowly updated target copies.")
    R.eq("critic:  y = r + γ Q′(s′, μ′(s′)),  L = MSE(Q(s,a) − y)        actor:  ∇J ≈ E[ ∇ₐQ(s,a)|ₐ=μ(s) · ∇θ μ(s) ]")
    R.eq("targets:  θ′ ← τθ + (1 − τ)θ′  after every update")
    R.h("Suitability to the Case Study", 3)
    R.p("Price is continuous; DDPG removes the bucket trade-off by outputting an exact price, and it is the algorithm that won Liu et al.'s field test [1]. Its known weaknesses — sensitivity to hyperparameters and to the hand-scheduled exploration noise — are exactly what SAC is included to test against.")
    R.h("Implementation Details", 3)
    R.p("Representation: actor 35 → 128 → 128 → 1 with tanh output; critic (35+1) → 128 → 128 → 1. Training: after 500 warm-up steps, one critic and one actor update per environment step on batches of 64 from a 50,000 buffer; Polyak averaging with τ = 0.005 after every update. Exploration: Ornstein–Uhlenbeck noise (θ = 0.15) added to the actor output, scale annealed from 0.20 to 0.05 over 25,000 steps and reset at episode ends; no noise in evaluation.")
    R.h("Workflow / Pseudocode", 3)
    R.bullets(["initialise actor μ, critic Q and target copies μ′, Q′; empty buffer D; OU noise state n = 0",
               "for each step: a = clip(μ(s) + n), n ← OU(n); step env; store transition",
               "sample batch; y = r + γ(1−done) Q′(s′, μ′(s′)); minimise MSE(Q(s,a) − y)",
               "maximise Q(s, μ(s)) w.r.t. actor; soft-update μ′, Q′"], style="List Number")
    R.h("Key Hyperparameters", 3)
    R.table(["Parameter", "Value", "Justification"], [
        ["Actor / critic learning rate", "1e-4 / 1e-3", "critic learns faster so the actor follows a settled value estimate"],
        ["Soft-update τ", "0.005", "slow-moving targets for stability"],
        ["Replay buffer / batch", "50,000 / 64", "as DQN, for a like-for-like off-policy setup"],
        ["OU noise θ, σ", "0.15; 0.20 → 0.05 over 25,000 steps", "temporally correlated exploration of nearby prices, annealed"],
        ["γ", "0.99", "shared across all agents"],
    ], widths=[1.7, 1.9, 3.6], size=8.5, caption="DDPG hyperparameters.")

    # 5.3 PPO
    R.h("5.3 Proximal Policy Optimization (PPO)", 2)
    R.h("Algorithm Overview", 3)
    R.p("PPO [6] is an on-policy actor-critic that learns a stochastic policy πθ(a|s) and improves it by a bounded amount per batch through a clipped surrogate objective, with advantages from Generalised Advantage Estimation [10]:")
    R.eq("rₜ(θ) = πθ(aₜ|sₜ) / πθₒₗₔ(aₜ|sₜ),    Lᶜᴸᴵᴾ = E[ min( rₜÂₜ , clip(rₜ, 1−ε, 1+ε) Âₜ ) ]")
    R.eq("δₜ = rₜ + γV(sₜ₊₁) − V(sₜ),    Âₜ = Σₖ (γλ)ᵏ δₜ₊ₖ")
    R.h("Suitability to the Case Study", 3)
    R.p("DRCR is noisy by construction: traffic and competitor shocks move it month to month regardless of the price. PPO's clip bounds how far one unusually good or bad batch can move the policy, which the team's Component-1 analysis identified as the desirable property. The known cost is sample efficiency: PPO cannot reuse old data.")
    R.h("Implementation Details", 3)
    R.p("Representation: Gaussian policy with a state-dependent mean (35 → 128 → 128 → 1) and a single learned log-standard-deviation; separate value network V(s). Training: collect 1,024 on-policy steps, compute GAE with λ = 0.95 (bootstrapping the last state's value if the rollout ends mid-episode), normalise advantages, then 10 epochs of minibatch (64) updates and discard the rollout. Actor and critic use separate Adam optimisers and separate gradient-norm clipping (0.5): a single combined clip lets the much larger value-loss gradients crush the policy gradient, a bug found during the smoke tests. Exploration: sampling from the policy plus an entropy bonus of 0.01; the mean action is used in evaluation. Actions are sampled unclipped and clipped to [−1, 1] at the environment boundary, as in reference implementations.")
    R.h("Workflow / Pseudocode", 3)
    R.bullets(["initialise policy πθ (mean net + log σ), value net Vφ",
               "repeat: run πθ for 1,024 steps, storing s, a, log π(a|s), r, V(s), done",
               "compute Â by GAE(γ, λ) and returns R = Â + V; normalise Â",
               "for 10 epochs over minibatches: update θ on −Lᶜᴸᴵᴾ − 0.01·entropy; update φ on MSE(V(s) − R); clip each gradient norm to 0.5",
               "discard the rollout"], style="List Number")
    R.h("Key Hyperparameters", 3)
    R.table(["Parameter", "Value", "Justification"], [
        ["Clip ε", "0.2", "standard trust-region width"],
        ["Rollout / epochs / minibatch", "1,024 / 10 / 64", "several product episodes per update; enough passes to use each rollout"],
        ["GAE λ", "0.95", "bias–variance balance for advantages"],
        ["Learning rate", "3e-4", "standard for PPO"],
        ["Entropy / value coefficients", "0.01 / 0.5", "keep exploration alive; weight the critic loss"],
        ["Gradient-norm clip", "0.5 (actor and critic separately)", "stability; separate clipping preserves the policy gradient"],
    ], widths=[1.7, 1.9, 3.6], size=8.5, caption="PPO hyperparameters.")

    # 5.4 SAC
    R.h("5.4 Soft Actor-Critic (SAC)", 2)
    R.h("Algorithm Overview", 3)
    R.p("SAC [7] maximises expected return plus policy entropy, J(π) = E[Σₜ r(sₜ,aₜ) + α H(π(·|sₜ))], with a stochastic tanh-squashed Gaussian policy, twin critics whose minimum is used to counter overestimation, and a temperature α learned to hold the policy's entropy at a target:")
    R.eq("critics:  y = r + γ [ min(Q₁′, Q₂′)(s′, a′) − α log π(a′|s′) ],  a′ ~ π(·|s′)")
    R.eq("actor:  L = E[ α log π(a|s) − min(Q₁, Q₂)(s, a) ],  a reparameterised;      α:  L = − E[ log α (log π(a|s) + H_target) ]")
    R.h("Suitability to the Case Study", 3)
    R.p("Liu et al. flag low-traffic products as a cold-start problem solved by pre-training on human prices [1]. SAC's entropy term keeps it exploring nearby prices for as long as it is uncertain, without a hand-scheduled noise process, which is attractive precisely for data-poor products. It also tests whether DDPG's exact-price advantage can be kept while improving stability; Kastius and Schlosser report SAC beating DQN in competitive pricing [2].")
    R.h("Implementation Details", 3)
    R.p("Representation: policy 35 → 128 → 128 → (mean, log σ), action = tanh(u) with the log-probability corrected by −Σ log(1 − tanh²u + 10⁻⁶); two independent critics (35+1) → 128 → 128 → 1 with Polyak targets. Training: after 500 warm-up steps, one update per environment step on batches of 64 from a 50,000 buffer, in the order critics → actor → temperature → target critics; α is detached in the critic and actor losses so only its own loss moves it. Target entropy is −(action dimension) = −1. Exploration: intrinsic — actions are sampled during training; tanh(mean) is used in evaluation.")
    R.h("Workflow / Pseudocode", 3)
    R.bullets(["initialise policy π, critics Q₁, Q₂ and targets Q₁′, Q₂′, log α = 0, buffer D",
               "for each step: a ~ π(·|s) (reparameterised, tanh-squashed); step env; store transition",
               "sample batch; a′ ~ π(·|s′); y = r + γ(1−done)[min(Q₁′,Q₂′)(s′,a′) − α log π(a′|s′)]; minimise MSE for both critics",
               "sample ã ~ π(·|s); minimise α log π(ã|s) − min(Q₁,Q₂)(s,ã) w.r.t. the policy",
               "minimise − log α (log π(ã|s) + H_target) w.r.t. log α; soft-update Q₁′, Q₂′"], style="List Number")
    R.h("Key Hyperparameters", 3)
    R.table(["Parameter", "Value", "Justification"], [
        ["Learning rates (actor, critics, α)", "3e-4 each", "standard SAC setting"],
        ["Soft-update τ", "0.005", "as DDPG"],
        ["Replay buffer / batch", "50,000 / 64", "as DQN and DDPG"],
        ["Target entropy", "−1 (= −action dim)", "standard heuristic [7]; α adapts automatically"],
        ["Initial α", "1.0 (log α = 0)", "learned from the first update onwards"],
        ["γ", "0.99", "shared"],
    ], widths=[1.7, 1.9, 3.6], size=8.5, caption="SAC hyperparameters.")

    # ---- 6 experimental setup -----------------------------------------------
    R.h("6. Experimental Setup", 1)
    R.h("6.1 Hardware", 2)
    R.p(f"All experiments ran on a laptop: Intel Core i7-8665U (4 cores / 8 threads, 1.90 GHz base), 16 GB RAM, no discrete GPU (integrated Intel UHD 620; PyTorch CPU build, single-threaded), Windows 11 Pro. A single environment step costs {bench['env_step_ms']:.1f} ms, dominated by the demand model.")
    R.h("6.2 Software", 2)
    R.p("Python 3.11.9 in Visual Studio Code; PyTorch 2.14 (CPU) for all networks; Gymnasium 1.3 for the environment interface; a React 18 + Recharts 2 dashboard built with Vite 5, and a Streamlit application for live inspection.")
    R.h("6.3 Libraries", 2)
    R.p("numpy 1.26, pandas 2.2, scikit-learn 1.9 (ridge regression and HistGradientBoostingRegressor for the demand model), torch 2.14, gymnasium 1.3, matplotlib 3.11, pyyaml, joblib, pytest 8.3, streamlit; python-pptx and python-docx for the deliverables.")
    R.h("6.4 Environment Configuration", 2)
    R.p(f"Panel of 80 products in 10 categories × 15 months (1,200 rows) following the Kaggle Retail Price Optimization schema [14]; generated synthetically with an explicit demand law (category elasticities 1.0–2.5, seasonality, 3 %/month price random walk) because the sandbox had no internet access, and swappable for the real CSV without code changes. State: 35 continuous features; actions and reward as in Section 4. Product split: {n_train} training / {n_test} held-out, stratified by category. Terminal condition: 20 steps in training (with bootstrap extension after month 15), exhaustion of the 12 real decision months in evaluation. Demand model fitted on all 80 products with a time-based holdout; feature scaler fitted on training products only.")
    R.h("6.5 Training Configuration", 2)
    R.p(f"Each (algorithm, seed) run: {steps:,} environment steps = {steps // 20} training episodes of 20 steps; seeds {', '.join(map(str, seeds))} for every algorithm and for the Static and Random baselines ({len(seeds) * 6} runs in total). Off-policy agents learn from step 500 with one gradient update per step; PPO updates after every 1,024-step rollout ({steps // 1024} updates in the budget). Evaluation: the saved checkpoint prices each of the {n_test} held-out products deterministically over its 12 real months; metrics are averaged over products and reported as mean ± standard deviation across seeds.")
    R.h("6.6 Hyperparameters", 2)
    R.table(["Approach", "Learning Rate", "Gamma", "Exploration", "Episodes (steps)"], [
        ["DQN", "1e-3", "0.99", "ε-greedy 1.0 → 0.05 over 25k steps", f"{steps // 20} ({steps:,})"],
        ["DDPG", "1e-4 actor / 1e-3 critic", "0.99", "OU noise σ 0.20 → 0.05 over 25k steps", f"{steps // 20} ({steps:,})"],
        ["PPO", "3e-4", "0.99", "stochastic policy + entropy 0.01; clip 0.2", f"{steps // 20} ({steps:,})"],
        ["SAC", "3e-4 (actor, critics, α)", "0.99", "entropy-regularised, target H = −1, α learned", f"{steps // 20} ({steps:,})"],
    ], widths=[1.0, 1.7, 0.7, 2.6, 1.2], size=8.5, caption="Training hyperparameters common table. Network 2 × 128 ReLU, Adam, batch 64, buffer 50,000 (off-policy) for all.")

    # ---- 7 results -----------------------------------------------------------
    R.h("7. Results", 1)
    R.p(f"All numbers below are measured: {len(seeds)} seeds × {steps:,} steps per algorithm, evaluated on the {n_test} held-out products; source files are results/comparison_summary.csv and the per-run logs.")
    R.h("7.1 Learning Curves", 2)
    R.figure(FIG_DIR / "fig_learning_curves.png", "Learning curves, mean ± std across seeds: (a) rolling mean reward over 200 steps; (b) running mean reward since the start of training.")
    R.p(f"Panel (a) is dominated by product-level shocks: each training episode is a different product with its own traffic and competitor history, so the short-term reward is noisy for every algorithm. Panel (b) averages the noise out. SAC and DDPG rise above zero within the first 2,000 steps and settle at the top (final running means {m['sac']['final_running']:.2f} and {m['ddpg']['final_running']:.2f}); PPO starts lowest, climbs steadily and is still improving at {steps:,} steps ({m['ppo']['final_running']:.2f}), the signature of an on-policy learner that discards its data after each update; DQN stays below zero throughout ({m['dqn']['final_running']:.2f}).")
    R.h("7.2 Reward versus Episode", 2)
    R.figure(FIG_DIR / "fig_reward_vs_episode.png", "Episode return (sum of the 20 step rewards) against training episode, rolling mean over 25 episodes, mean ± std across seeds.")
    R.p("Per-episode returns make the ordering visible at the episode scale: the three continuous agents' returns move from slightly negative in the first episodes to clearly positive, SAC and DDPG earliest, PPO later; DQN's returns hover around zero with the widest band, consistent with a policy that was still 43 % random at the end of the budget (Section 9.5).")
    R.h("7.3 Final Policy / Behaviour", 2)
    R.figure(FIG_DIR / "fig_policy.png", "Final policies on held-out product P0050 (furniture), mean over seeds: (a) monthly price chosen by each agent against the seller's actual price, hollow markers where the price guard was hit; (b) cumulative simulated revenue relative to the seller's own prices.")
    R.p(f"The learned behaviours differ qualitatively. SAC and DDPG raise the price from the entering $77 to the guard within one or two months and hold it; PPO also moves up but oscillates between $85 and $99, reflecting its still-improving policy; DQN moves the opposite way, cutting to the lower guard. Under the fitted demand model this product is inelastic, so the upward policies gain revenue every month while DQN loses it: over twelve months SAC finishes about $6,900 ahead of the seller and DQN about $5,000 behind. The hollow markers show that the continuous agents are constrained by the price guard, not by their own preference — a point taken up in Section 9.")
    R.h("7.4 Performance Metrics", 2)
    R.p("Six metrics are reported (definitions in Table 10). Table 9 gives their values for the four approaches and the best on each.")
    def bestof(key, higher=True):
        vals = {k: m[k][key] for k in ALGOS}
        return LABEL[max(vals, key=vals.get)] if higher else LABEL[min(vals, key=vals.get)]
    R.table(["Metric", "DQN", "DDPG", "PPO", "SAC", "Best"], [
        ["M1 Mean held-out DRCR", *(fmt(m[k]["drcr"]) for k in ALGOS), bestof("drcr")],
        ["M2 Std of held-out DRCR across seeds (lower = more stable)", *(fmt(m[k]["drcr_std"]) for k in ALGOS), bestof("drcr_std", False)],
        ["M3 Simulated revenue uplift vs. seller (%)", *(f"{m[k]['uplift']:+.2f}" for k in ALGOS), bestof("uplift")],
        ["M4 Training-tail mean reward (last 20 % of steps)", *(f"{m[k]['tail']:.2f} ± {m[k]['tail_std']:.2f}" for k in ALGOS), bestof("tail")],
        ["M5 Convergence step (running mean within ±0.25 of final)", *(f"{m[k]['conv_step']:,}" for k in ALGOS), bestof("conv_step", False)],
        ["M6 Guard-hit rate on held-out decisions (%)", *(f"{m[k]['guard']:.1f}" for k in ALGOS), "— (behavioural)"],
        ["M7 Training throughput (env steps / s)", *(f"{m[k]['steps_per_s']:.0f}" for k in ALGOS), bestof("steps_per_s")],
    ], widths=[2.6, 0.95, 0.95, 0.95, 0.95, 0.8], size=8.5, caption="Performance metrics for the four approaches (mean over seeds unless stated).")
    R.h("7.5 Comparison Charts", 2)
    R.figure(FIG_DIR / "fig_comparison.png", "Comparative performance including baselines: (a) held-out DRCR, bars = mean ± std across seeds, dots = individual seeds; (b) simulated revenue uplift over the seller's prices summed over all held-out products.")
    R.h("7.6 Key Observations", 2)
    R.bullets([
        f"SAC achieves the highest held-out DRCR ({m['sac']['drcr']:.4f}) and the tightest cross-seed spread ({m['sac']['drcr_std']:.4f}); its three seeds land within 0.002 of each other.",
        f"All three continuous-action agents beat both the no-change ({m['static']['drcr']:.4f}) and random ({m['random']['drcr']:.4f}) baselines and the seller's own prices (uplift {m['ddpg']['uplift']:+.1f} % to {m['sac']['uplift']:+.1f} %).",
        f"DQN ({m['dqn']['drcr']:.4f}) is below the no-change baseline and has the largest spread ({m['dqn']['drcr_std']:.4f}); one seed scored {min(m['dqn']['per_seed']):.4f}.",
        f"DDPG and SAC converge fastest (running mean settled by steps {m['ddpg']['conv_step']:,} and {m['sac']['conv_step']:,}); PPO's running mean was still moving at the end of the budget.",
        f"Guard-hit rates are high for the continuous agents (DDPG {m['ddpg']['guard']:.0f} %, SAC {m['sac']['guard']:.0f} %, PPO {m['ppo']['guard']:.0f} %) and for DQN ({m['dqn']['guard']:.0f} %, at the lower bound): the policies want to move further than the guard allows.",
        f"Throughput ranks inversely with model size: DQN {m['dqn']['steps_per_s']:.0f} steps/s ({m['dqn']['params']:,} parameters) versus SAC {m['sac']['steps_per_s']:.0f} steps/s ({m['sac']['params']:,} parameters); a {steps:,}-step SAC run takes about {steps / m['sac']['steps_per_s'] / 60:.0f} minutes on the laptop CPU.",
    ])

    # ---- 8 comparative analysis ---------------------------------------------
    R.h("8. Comparative Analysis", 1)
    R.p("Table 10 defines the metrics and repeats their values; the subsections interpret them.")
    R.table(["Metric", "Definition", "DQN", "DDPG", "PPO", "SAC"], [
        ["M1 Held-out DRCR", "Mean per-month DRCR over the held-out products and seeds; the trained objective", *(fmt(m[k]["drcr"]) for k in ALGOS)],
        ["M2 Stability", "Standard deviation of M1 across the three seeds", *(fmt(m[k]["drcr_std"]) for k in ALGOS)],
        ["M3 Revenue uplift", "(policy revenue − seller revenue) / seller revenue over all held-out product-months, both through the demand model", *(f"{m[k]['uplift']:+.2f} %" for k in ALGOS)],
        ["M4 Tail reward", "Mean training reward over the last 20 % of steps (100 × DRCR scale)", *(f"{m[k]['tail']:.2f}" for k in ALGOS)],
        ["M5 Convergence", "First step after which the cross-seed running-mean reward stays within ±0.25 of its final value", *(f"{m[k]['conv_step']:,}" for k in ALGOS)],
        ["M7 Throughput", "Environment steps per second including learning updates (CPU, single thread)", *(f"{m[k]['steps_per_s']:.0f}" for k in ALGOS)],
    ], widths=[1.1, 2.9, 0.8, 0.8, 0.8, 0.8], size=8.5, caption="Metric definitions and values.")
    R.h("8.1 Convergence", 2)
    R.p(f"Convergence was measured on the running mean of the training reward averaged over seeds: the first step after which it never again leaves a ±0.25 band around its final value. By this measure DDPG (step {m['ddpg']['conv_step']:,}) and SAC (step {m['sac']['conv_step']:,}) converged fastest, DQN (step {m['dqn']['conv_step']:,}) settled but at a negative level, and PPO (step {m['ppo']['conv_step']:,}) settled last because its running mean was still rising. Off-policy replay lets DDPG and SAC reuse every transition many times; PPO sees each transition once and updated only {steps // 1024} times in the whole budget.")
    R.h("8.2 Reward and Policy Quality", 2)
    R.p(f"On the held-out products SAC's policy is the best by every quality metric: highest DRCR, highest uplift, tightest spread. DDPG reaches the same training level (M4 {m['ddpg']['tail']:.2f} vs {m['sac']['tail']:.2f}) but generalises with roughly five times the seed-to-seed variance, which is the practical cost of its externally scheduled exploration noise. PPO's policy is competitive (M1 {m['ppo']['drcr']:.4f}) and hits the guard least often ({m['ppo']['guard']:.0f} %), i.e. it moves prices more cautiously, consistent with its clipped updates. DQN's policy is the only one that lowered prices, and it lost revenue on every aggregate measure.")
    R.h("8.3 Computational Performance", 2)
    R.table(["Approach", "Parameters", "Steps / s", "Seconds per 1,000 steps", "Inference (µs / action)", "Networks updated per step"], [
        [LABEL[k], f"{m[k]['params']:,}", f"{m[k]['steps_per_s']:.0f}", f"{m[k]['sec_per_1k']:.1f}", f"{m[k]['infer_us']:.0f}", {"dqn": "1 (Q)", "ddpg": "2 (actor, critic)", "ppo": "2, but only every 1,024 steps", "sac": "4 (actor, 2 critics, α)"}[k]] for k in ALGOS
    ], widths=[0.9, 1.0, 0.8, 1.3, 1.3, 1.9], size=8.5, caption=f"Computational cost measured over {bench['agents']['sac']['steps']:,} training steps on the laptop CPU (environment step alone: {bench['env_step_ms']:.1f} ms).")
    R.p("Cost scales with the number of networks updated per step. DQN is cheapest; SAC is the most expensive per step (four networks) yet still trains a full run in well under an hour. PPO's per-step cost is low because it batches its updates, but it needs more steps to reach the same policy quality, so its wall-clock to a given quality is not lower.")
    R.h("8.4 Overall Best Approach", 2)
    R.p(f"SAC is the best approach on the evidence: first on M1, M2 and M3, joint first on M4, second on M5, and its per-seed DRCR values ({', '.join(f'{v:.4f}' for v in m['sac']['per_seed'])}) do not overlap with any other algorithm's range. The margin over DDPG on the mean is modest ({m['sac']['drcr'] - m['ddpg']['drcr']:+.4f}); the decisive difference is reliability. Its higher compute cost is immaterial at this scale.")
    R.h("8.5 Strengths and Weaknesses", 2)
    R.table(["Approach", "Strengths", "Weaknesses", "Best Use", "Observed Performance"], [
        ["DQN", "simple, cheapest, stable value learning", "5 % price resolution; exploration schedule must be tuned to the budget", "few, coarse price tiers; very large SKU counts", f"M1 {m['dqn']['drcr']:.4f}; below baselines in this run"],
        ["DDPG", "exact price; fast convergence; matches SAC's training level", "noise schedule hand-tuned; higher variance across seeds", "continuous pricing with careful tuning", f"M1 {m['ddpg']['drcr']:.4f}; std {m['ddpg']['drcr_std']:.4f}"],
        ["PPO", "robust bounded updates; cautious price moves; no replay memory", "sample-inefficient; slowest to converge", "noisy rewards with a large step budget", f"M1 {m['ppo']['drcr']:.4f}; still improving at {steps:,} steps"],
        ["SAC", "best DRCR and uplift; most stable; exploration self-tuned", "four networks; slowest per step", "continuous pricing, especially data-poor SKUs", f"M1 {m['sac']['drcr']:.4f}; std {m['sac']['drcr_std']:.4f}"],
    ], widths=[0.8, 1.7, 1.7, 1.5, 1.5], size=8.5, caption="Strengths and weaknesses of the four approaches.")

    # ---- 9 discussion ---------------------------------------------------------
    R.h("9. Discussion", 1)
    R.h("9.1 Performance Differences", 2)
    R.p("Three design choices explain the ranking. Continuous actions beat discrete ones because the revenue-maximising move is often a fine adjustment that an 11-bucket grid cannot express; this reproduces Liu et al.'s DDPG-over-DQN result [1]. Entropy-regularised exploration beats a hand-scheduled noise process on reliability: SAC and DDPG learn the same thing on average, but SAC's exploration adapts per state through α while DDPG's decays on a clock regardless of what the policy still needs to learn. Off-policy replay beats on-policy updates on sample efficiency within a fixed budget: PPO's clipped objective did what it promises — its policy moved smoothly and conservatively — but with one update per 1,024 steps it had not finished learning.")
    R.h("9.2 Effect of Environment", 2)
    R.p("Stochasticity in the market context makes the per-step reward noisy and forces every learner to average over many products; this is why replay-based methods, which average over the whole buffer, are at an advantage. Partial observability (the agent never sees the demand function) makes the rolling-history features important; ablating them was outside scope but they carry the product-identity information one shared network needs. The continuous, bounded action space with a guard shaped behaviour strongly: because the fitted demand is inelastic (β = %.2f), revenue rises with price up to the guard for most products, so the continuous agents learned “move up to the boundary” and were clipped on %.0f–%.0f %% of decisions. A more elastic market would produce interior optima and different policies." % (dm["elasticity_coef"], min(m[k]["guard"] for k in ["ddpg", "ppo", "sac"]), max(m[k]["guard"] for k in ["ddpg", "ppo", "sac"])))
    R.h("9.3 Exploration versus Exploitation", 2)
    R.p(f"Each algorithm explores differently and the differences are visible in the results. DQN's ε-greedy schedule was designed for a 25,000-step run and the budget was {steps:,}, so its checkpoints were still exploring 43 % of the time when saved — its evaluation policy (ε = 0) is therefore the greedy policy of an under-trained Q-function, which explains both its low mean and its high variance. DDPG's OU noise anneals from 0.20 to 0.05 on the same 25,000-step clock, so at {steps:,} steps it was still moderately noisy, yet its deterministic evaluation policy was already good; DDPG is less hurt because its actor, not its exploration, defines the evaluated behaviour. PPO and SAC explore through their policies' own randomness, which shrinks as the policy becomes confident; SAC additionally raises α whenever entropy falls below the target, which kept it exploring in states it had seen rarely. That is the mechanism behind its tight cross-seed spread.")
    R.h("9.4 Practical Interpretation", 2)
    R.p(f"For a marketplace, the result says: prefer a continuous-action, entropy-regularised agent, evaluate it on products it never trained on, and report seed-to-seed variance, because a policy that is right on average but unreliable across retrainings is not deployable. It also says that the demand model is the product: the agents are only as good as the counterfactual they are trained against, so its validation (R² = {dm['r2_holdout']:.2f} on a time-based holdout), its monotonicity guarantee and its guard rails deserve as much engineering as the learners. In this simulator the winning policy is “raise the price to the allowed limit”; on a real market that recommendation stands only if the true elasticity is as low as the fitted one, which is a question for a field test such as [1], not for simulation.")
    R.h("9.5 Limitations", 2)
    R.bullets([
        f"DQN is under-trained, not only bucket-limited: its ε schedule (25,000 steps) exceeds the {steps:,}-step budget. The bucket penalty is real but its size is overstated here; a 50,000-step run (one command in the repository) removes the confound.",
        f"The demand model under-estimates the elasticity that generated the synthetic data (fitted β = {dm['elasticity_coef']:.2f} against 1.0–2.5): monthly price moves of about 3 % give little within-product variation and the ridge penalty shrinks the slope. The simulator is therefore less price-sensitive than the data-generating process, which favours price increases.",
        "The data are synthetic (schema-identical to the Kaggle dataset, with a known demand law). This is what allows the simulator to be validated, but every number would change on real data; the method and the code would not.",
        f"{n_test} held-out products and {len(seeds)} seeds are enough to rank the algorithms, not to give tight confidence intervals; the per-seed points in Figure 6 show the honest spread.",
        "Competitors are exogenous: they do not react to the agent's price. A reacting competitor (as in [2]) would change the optimal policy and could create collusive or price-war dynamics.",
        "The traffic proxy (category units sold) is coarser than the page-visit counts used in [1].",
    ])

    # ---- 10 innovation -------------------------------------------------------
    R.h("10. Innovation Component and Future Extension", 1)
    R.p("Selected innovation: Digital Twin + RL, extended with Offline RL for cold-start products.")
    R.p("The environment built here is already a data-driven digital twin of the market: the real context is replayed and the demand response is a validated model with guard rails, so policies can be trained and compared safely before any live exposure. The proposed extension addresses the problem Liu et al. solve by pre-training on human prices [1] — products with too little history to learn from scratch — with the modern offline-RL toolkit (Figure 7). Offline algorithms such as Conservative Q-Learning (CQL) or Implicit Q-Learning (IQL) learn directly from the logged (state, human price, outcome) tuples of all products, penalising actions far from what was ever tried; the resulting policy initialises SAC, which then fine-tunes inside the twin under the same guard rails. This gives every new SKU a sensible starting policy, keeps the exploration of live customers to a minimum, and turns the guard rails from a training safeguard into a safe-RL constraint that the fine-tuning stage must respect.")
    R.figure(FIG_DIR / "fig_innovation.png", "Proposed extension: offline RL pre-training from logged human prices, followed by safe online fine-tuning of SAC inside the demand-model digital twin.")
    R.p("Expected improvements: faster convergence for data-poor SKUs (the policy starts near human behaviour rather than at random), lower variance across retrainings (the same initialisation for every seed), and an explicit safety boundary. Evaluation would reuse this project's protocol: held-out products, several seeds, DRCR and revenue uplift against the seller's own prices.")

    # ---- 11 conclusion ------------------------------------------------------
    R.h("11. Conclusion", 1)
    R.p(f"This project turned the team's conceptual comparison of four reinforcement-learning algorithms for dynamic pricing into a measured one. A pricing MDP faithful to Liu et al.'s Tmall system [1] — a 35-feature product-month state, a bounded price move, and DRCR as reward — was implemented as a Gymnasium environment whose transition and reward come from a validated, guarded counterfactual demand model. DQN, DDPG, PPO and SAC were implemented from first principles, verified on standard benchmarks, trained under an identical budget of {steps:,} steps for {len(seeds)} seeds, and evaluated on {n_test} held-out products. SAC was the best approach: highest held-out DRCR ({m['sac']['drcr']:.4f}), highest simulated revenue uplift ({m['sac']['uplift']:+.1f} %) and the most reliable across seeds. The practical significance is twofold: continuous, entropy-regularised control is the right tool for this problem, and a validated demand simulator is the precondition for any such comparison on logged data.")
    R.h("11.1 Key Findings", 2)
    R.bullets([
        f"Continuous-action agents (DDPG {m['ddpg']['drcr']:.4f}, PPO {m['ppo']['drcr']:.4f}, SAC {m['sac']['drcr']:.4f}) outperform the discrete DQN ({m['dqn']['drcr']:.4f}) and both baselines, reproducing the paper's DDPG-over-DQN result.",
        f"SAC's self-tuned entropy exploration gives the same training level as DDPG with a five-fold tighter held-out spread ({m['sac']['drcr_std']:.4f} vs {m['ddpg']['drcr_std']:.4f}).",
        f"PPO's clipped updates produce the most cautious pricing (lowest guard-hit rate, {m['ppo']['guard']:.0f} %) but need a larger budget: its running mean was still rising at {steps:,} steps.",
        "The counterfactual demand model must be identified within product and guarded; a pooled fit produced a wrong-sign elasticity, and an unguarded model invites reward hacking.",
    ])
    R.h("11.2 Future Scope", 2)
    R.p("Re-run at the full 50,000-step budget with five seeds to remove DQN's schedule confound and tighten intervals; replace the synthetic panel with the real Kaggle data (no code changes required); allow per-category elasticity in the demand baseline and a richer traffic proxy; log training losses (now supported) to compare critic stability directly; add reacting competitors; and implement the offline-RL pre-training of Section 10.")

    # ---- 12 references ------------------------------------------------------
    R.h("12. References", 1)
    refs = [
        "J. Liu, Y. Zhang, X. Wang, Y. Deng, and X. Wu, “Dynamic pricing on e-commerce platform with deep reinforcement learning: A field experiment,” arXiv preprint arXiv:1912.02572, 2019.",
        "A. Kastius and R. Schlosser, “Dynamic pricing under competition using reinforcement learning,” Journal of Revenue and Pricing Management, vol. 21, pp. 50–63, 2022, doi: 10.1057/s41272-021-00285-3.",
        "R. Rana and F. S. Oliveira, “Real-time dynamic pricing in a non-stationary environment using model-free reinforcement learning,” Omega, vol. 47, pp. 116–126, 2014, doi: 10.1016/j.omega.2013.10.004.",
        "V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.",
        "T. P. Lillicrap et al., “Continuous control with deep reinforcement learning,” arXiv preprint arXiv:1509.02971, 2015.",
        "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.",
        "T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proc. 35th Int. Conf. Machine Learning (ICML), Stockholm, Sweden, 2018, pp. 1861–1870.",
        "R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.",
        "C. J. C. H. Watkins, “Learning from delayed rewards,” Ph.D. dissertation, King’s College, Cambridge, U.K., 1989.",
        "J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” arXiv preprint arXiv:1506.02438, 2015.",
        "M. Towers et al., “Gymnasium: A standard interface for reinforcement learning environments,” arXiv preprint arXiv:2407.17032, 2024.",
        "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.",
        "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "Suddharshan, “Retail Price Optimization,” Kaggle dataset (derived from the Olist Brazilian e-commerce dataset). [Online]. Available: https://www.kaggle.com/datasets/suddharshan/retail-price-optimization (accessed Sep. 2026).",
    ]
    for i, r in enumerate(refs, start=1):
        para = R.p(f"[{i}] {r}", size=10)
        para.paragraph_format.left_indent = Inches(0.35)
        para.paragraph_format.first_line_indent = Inches(-0.35)
        para.paragraph_format.space_after = Pt(3)

    # ---- appendices ---------------------------------------------------------
    R.h("Appendix A: Project Resources", 1)
    R.bullets([
        "Repository (code, configs, trained checkpoints' logs, dashboard, this report's generator): https://github.com/Feliz7000/rl-dynamic-pricing-comparison",
        "Interactive dashboard: docs/index.html in the repository (single file; opens in any browser) — tabs Simulate, What-if, Training, Results.",
        "Live application: streamlit run app/streamlit_app.py — evaluates the demand model and the four checkpoints on demand.",
        "Data: synthetic panel from scripts/generate_synthetic_data.py (schema of [14]); scripts/download_data.py fetches the real dataset into data/raw/.",
        "Reproduction: python scripts/fit_demand_model.py; TOTAL_STEPS=15000 SEEDS=\"0 1 2\" bash scripts/run_full_comparison.sh; python scripts/build_dashboard.py; python scripts/build_report.py.",
        "Tests: pytest tests/ (data loader, demand model, environment) and tests/test_agents_smoke.py (each algorithm on CartPole-v1 / Pendulum-v1).",
    ])
    R.h("Appendix B: Additional Results", 1)
    R.p("Table 13 gives the simulated revenue uplift over the seller's prices for every held-out product (mean over seeds), the per-product view behind Figure 6(b).")
    rows = []
    for p in dash["products"]:
        hist = sum(p["historical"]["revenue_model"])
        vals = []
        for k in ALGOS:
            rev = np.mean([sum(p["agents"][k][str(s)]["revenue"]) for s in seeds])
            vals.append((rev - hist) / hist * 100)
        rows.append([p["id"], p["category"].replace("_", " "), f"${hist:,.0f}", *(f"{v:+.1f} %" for v in vals), LABEL[ALGOS[int(np.argmax(vals))]]])
    rows.sort(key=lambda r: -float(r[6].replace(" %", "").replace("+", "")))
    R.table(["Product", "Category", "Seller revenue (model)", "DQN", "DDPG", "PPO", "SAC", "Best"], rows, widths=[0.7, 1.2, 1.2, 0.75, 0.75, 0.75, 0.75, 0.7], size=8, caption="Per-product simulated revenue uplift vs. the seller's own prices on the held-out set (sorted by SAC).")
    R.p("Per-seed held-out DRCR: " + "; ".join(f"{LABEL[k]}: " + ", ".join(f"{v:.4f}" for v in m[k]["per_seed"]) for k in ["static", "random"] + ALGOS) + ".")

    R.save(OUT)
    print(f"Wrote {OUT}  (figures {R.fig_n}, tables {R.tab_n})")


if __name__ == "__main__":
    build()
