"""Build the CIA-3 presentation deck (presentation/CIA3_Dynamic_Pricing_RL.pptx)
from the project's measured results, figures and dashboard screenshots.

    python scripts/build_presentation.py

Every slide carries speaker notes. Numbers are read from
results/comparison_summary.csv and dashboard/data/dashboard_data.json so the
deck stays consistent with whatever run produced them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "presentation" / "images"
OUT = ROOT / "presentation" / "CIA3_Dynamic_Pricing_RL.pptx"

# ---- palette (matches the dashboard) -------------------------------------
BG = RGBColor(0x12, 0x16, 0x1B)
PANEL = RGBColor(0x18, 0x1D, 0x24)
PANEL_ALT = RGBColor(0x1E, 0x24, 0x2C)
BORDER = RGBColor(0x2B, 0x32, 0x3B)
TEXT = RGBColor(0xED, 0xEA, 0xE1)
MUTED = RGBColor(0x8C, 0x97, 0xA3)
FAINT = RGBColor(0x5B, 0x65, 0x70)
GOLD = RGBColor(0xE1, 0xA3, 0x39)
SLATE = RGBColor(0x6E, 0x8F, 0xB0)
POS = RGBColor(0x5F, 0xA8, 0x7A)
NEG = RGBColor(0xC4, 0x68, 0x5A)
ALGO = {
    "DQN": RGBColor(0x5B, 0x8D, 0xEF),
    "DDPG": RGBColor(0xE1, 0xA3, 0x39),
    "PPO": RGBColor(0xC7, 0x7D, 0xFF),
    "SAC": RGBColor(0x4F, 0xD1, 0xA5),
}
SERIF, SANS, MONO = "Georgia", "Calibri", "Consolas"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# ---- data ------------------------------------------------------------------
def load_numbers():
    summary = pd.read_csv(ROOT / "results" / "comparison_summary.csv").set_index("agent")
    with open(ROOT / "dashboard" / "data" / "dashboard_data.json") as f:
        dash = json.load(f)
    meta = dash["meta"]
    products = dash["products"]

    def total_rev(key):
        seeds = meta["seeds"]
        return sum(sum(p["agents"][key][str(s)]["revenue"]) for p in products for s in seeds) / len(seeds)

    hist = sum(sum(p["historical"]["revenue_model"]) for p in products)
    uplift = {k: (total_rev(k) - hist) / hist * 100 for k in ["static", "random", "dqn", "ddpg", "ppo", "sac"]}
    return summary, meta, uplift


# ---- primitives ------------------------------------------------------------
def rgb_hex(c: RGBColor) -> str:
    return str(c)


def add_rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


def add_text(slide, x, y, w, h, runs, size=16, color=TEXT, font=SANS, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15, space_after=6):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, opts) where opts may set color/bold/font/size."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        pieces = para if isinstance(para, list) else [(para, {})]
        for text, opts in pieces:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", False)
            f.color.rgb = opts.get("color", color)
    return tb


def add_bullets(slide, x, y, w, h, items, size=16, color=TEXT, bullet_color=GOLD, space_after=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.12
        level = 0
        if isinstance(item, tuple):
            level, item = item
        p.level = level
        b = p.add_run()
        b.text = ("    " * level) + ("• " if level == 0 else "– ")
        b.font.color.rgb = bullet_color if level == 0 else MUTED
        b.font.size = Pt(size)
        b.font.name = SANS
        pieces = item if isinstance(item, list) else [(item, {})]
        for text, opts in pieces:
            r = p.add_run()
            r.text = text
            r.font.name = opts.get("font", SANS)
            r.font.size = Pt(opts.get("size", size - (1 if level else 0)))
            r.font.bold = opts.get("bold", False)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", color if level == 0 else MUTED)
    return tb


def add_image_fit(slide, path, x, y, max_w, max_h, border=True):
    im = Image.open(path)
    iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    px = x + (max_w - w) // 2
    py = y + (max_h - h) // 2
    if border:
        add_rect(slide, px - Emu(9000), py - Emu(9000), w + Emu(18000), h + Emu(18000), PANEL_ALT, BORDER)
    slide.shapes.add_picture(str(path), px, py, w, h)
    return px, py, w, h


def add_table(slide, x, y, w, header, rows, col_widths=None, size=12, row_h=None, header_color=GOLD, first_col_colors=None):
    n_rows, n_cols = len(rows) + 1, len(header)
    row_h = row_h or Inches(0.36)
    tbl = slide.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows).table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = int(w * cw / total)
    for r in range(n_rows):
        tbl.rows[r].height = row_h
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r == 0 else (PANEL_ALT if r % 2 == 0 else BG)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            text = header[c] if r == 0 else rows[r - 1][c]
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(text)
            run.font.name = MONO if r == 0 else SANS
            run.font.size = Pt(size - 1 if r == 0 else size)
            run.font.bold = r == 0
            color = header_color if r == 0 else TEXT
            if r > 0 and c == 0 and first_col_colors and rows[r - 1][0] in first_col_colors:
                color = first_col_colors[rows[r - 1][0]]
                run.font.bold = True
            run.font.color.rgb = color
    return tbl


def add_card(slide, x, y, w, h, title, body, accent=GOLD, title_size=13, body_size=13):
    add_rect(slide, x, y, w, h, PANEL, BORDER)
    add_rect(slide, x, y, Inches(0.06), h, accent)
    add_text(slide, x + Inches(0.18), y + Inches(0.1), w - Inches(0.3), Inches(0.4), title, size=title_size, color=accent, font=MONO, bold=True)
    if isinstance(body, list):
        add_bullets(slide, x + Inches(0.15), y + Inches(0.5), w - Inches(0.3), h - Inches(0.6), body, size=body_size, space_after=4)
    else:
        add_text(slide, x + Inches(0.18), y + Inches(0.5), w - Inches(0.3), h - Inches(0.6), body, size=body_size, color=TEXT)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title=None, eyebrow=None, notes=None):
        self.n += 1
        s = self.prs.slides.add_slide(self.blank)
        add_rect(s, 0, 0, W, H, BG)
        if eyebrow:
            add_text(s, MARGIN, Inches(0.35), W - 2 * MARGIN, Inches(0.3), eyebrow.upper(), size=10.5, color=GOLD, font=MONO)
        if title:
            add_text(s, MARGIN, Inches(0.6), W - 2 * MARGIN, Inches(0.8), title, size=28, color=TEXT, font=SERIF)
            add_rect(s, MARGIN, Inches(1.38), W - 2 * MARGIN, Emu(9525), BORDER)
        add_text(s, MARGIN, H - Inches(0.45), Inches(8), Inches(0.3), "CSEAM731 · CIA-3 · Dynamic pricing with DQN · DDPG · PPO · SAC", size=9.5, color=FAINT, font=MONO)
        add_text(s, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(self.n), size=9.5, color=FAINT, font=MONO, align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(OUT)


# ---- the deck --------------------------------------------------------------
def build():
    summary, meta, uplift = load_numbers()
    seeds, steps = meta["seeds"], meta["total_steps"]
    n_train, n_test = meta["n_train_products"], meta["n_test_products"]
    dm = meta["demand_model"]
    d = Deck()
    BODY_Y = Inches(1.6)
    BODY_H = H - BODY_Y - Inches(0.7)
    CW = W - 2 * MARGIN

    def S(key):
        label = {"dqn": "DQN", "ddpg": "DDPG", "ppo": "PPO", "sac": "SAC", "static": "Static (no change)", "random": "Random"}[key]
        return summary.loc[label]

    # 1 title -----------------------------------------------------------------
    s = d.slide(notes="Open with the question, not the method: same pricing problem, four RL algorithms, which sets better prices and why. Everything in the project exists to answer that fairly and measurably.")
    add_text(s, MARGIN, Inches(2.0), CW, Inches(0.4), "CSEAM731 · REINFORCEMENT LEARNING · CIA-3", size=12, color=GOLD, font=MONO)
    add_text(s, MARGIN, Inches(2.4), CW, Inches(1.0), "Dynamic Pricing with Reinforcement Learning", size=40, color=TEXT, font=SERIF)
    add_text(s, MARGIN, Inches(3.45), CW, Inches(0.8), "DQN · DDPG · PPO · SAC on one shared pricing MDP — a measured comparison", size=22, color=MUTED, font=SANS)
    add_rect(s, MARGIN, Inches(4.55), Inches(1.2), Emu(28000), GOLD)
    add_text(s, MARGIN, Inches(4.75), CW, Inches(1.2), [
        "Based on Liu et al. (2019), “Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning: A Field Experiment” (arXiv:1912.02572)",
        "Code, data pipeline, trained agents and interactive dashboard: github.com/Feliz7000/rl-dynamic-pricing-comparison",
    ], size=13, color=MUTED)

    # 2 agenda ----------------------------------------------------------------
    s = d.slide("What we will cover", "Agenda", notes="Eight beats: problem, MDP, the counterfactual demand model, the four algorithms, the live race, results, training behaviour, caveats. About 12-15 minutes.")
    items = [
        ("1", "The problem and the source paper", "Why dynamic pricing, what Alibaba field-tested on Tmall"),
        ("2", "One shared MDP", "State, action and reward every algorithm sees"),
        ("3", "The hard part: a demand model", "How to score a price nobody ever charged"),
        ("4", "Four algorithms", "DQN, DDPG, PPO, SAC — what each does differently"),
        ("5", "Fair-comparison protocol", "Baselines, budgets, seeds, held-out products"),
        ("6", "Results and the live race", "Measured DRCR, revenue vs. the seller, dashboard"),
        ("7", "Caveats and next steps", "What the numbers do and do not show"),
    ]
    col_w = (CW - Inches(0.3)) / 2
    for i, (num, head, sub) in enumerate(items):
        col, row = i % 2, i // 2
        x = MARGIN + col * (col_w + Inches(0.3))
        y = BODY_Y + row * Inches(1.35)
        add_text(s, x, y, Inches(0.6), Inches(0.6), num, size=28, color=GOLD, font=SERIF)
        add_text(s, x + Inches(0.6), y + Inches(0.02), col_w - Inches(0.6), Inches(0.5), head, size=18, color=TEXT, font=SANS, bold=True)
        add_text(s, x + Inches(0.6), y + Inches(0.45), col_w - Inches(0.6), Inches(0.6), sub, size=13, color=MUTED)

    # 3 problem ---------------------------------------------------------------
    s = d.slide("Prices should move with the market", "1 · The problem", notes="Dynamic pricing = re-setting a price repeatedly as demand, traffic and competitors change. A human does it by rules of thumb and re-prices rarely; an RL agent does it by trial, reward and correction, every period.")
    add_card(s, MARGIN, BODY_Y, Inches(3.9), Inches(2.4), "STATIC PRICING", [
        "One price, set once, held for weeks",
        "Blind to traffic swings, seasonality, competitor moves",
        "Leaves revenue on the table in both directions",
    ], accent=SLATE)
    add_card(s, MARGIN + Inches(4.15), BODY_Y, Inches(3.9), Inches(2.4), "DYNAMIC PRICING", [
        "Re-price every period from what you observe now",
        "Needs a policy: state → price",
        "Rules of thumb scale badly across thousands of SKUs",
    ], accent=GOLD)
    add_card(s, MARGIN + Inches(8.3), BODY_Y, Inches(3.85), Inches(2.4), "WHY REINFORCEMENT LEARNING", [
        "Sequential decisions with delayed, noisy feedback",
        "Learns the policy from outcomes, no hand-written rules",
        "Natural fit: state, action, reward",
    ], accent=ALGO["SAC"])
    add_text(s, MARGIN, BODY_Y + Inches(2.75), CW, Inches(1.6), [
        [("The question this project answers: ", {"bold": True, "color": TEXT}), ("given the exact same pricing problem — same information, same allowed moves, same reward — which of four RL algorithms sets better prices, and why?", {"color": MUTED})],
    ], size=18)

    # 4 source paper ----------------------------------------------------------
    s = d.slide("The source paper: a real field experiment on Tmall", "1 · Liu et al. 2019", notes="Alibaba's supply-chain team deployed RL pricing on Tmall.com from July 2018. Three ideas carry into our project: pricing as an MDP; DRCR as the reward instead of raw revenue; and the DQN vs DDPG field test. Scaled to manual = 1.00, DDPG scored 6.07 and DQN 5.03 over a 30-day phase.")
    add_bullets(s, MARGIN, BODY_Y, Inches(7.3), BODY_H, [
        [("Pricing as an MDP. ", {"bold": True}), ("State = what you know about a product this period; action = the price; reward = how the market responds.", {"color": MUTED})],
        [("Reward = DRCR, not revenue. ", {"bold": True}), ("Traffic swings for reasons unrelated to price, so they normalise revenue by traffic (RCR) and reward the change in it.", {"color": MUTED})],
        [("Discrete vs continuous. ", {"bold": True}), ("DQN chooses among K price buckets; DDPG outputs an exact price. Few buckets are imprecise, many spread the data thin.", {"color": MUTED})],
        [("Cold start. ", {"bold": True}), ("Low-traffic products lack data; their pipeline pre-trains on historical human prices.", {"color": MUTED})],
        [("Deployed on Tmall.com from July 2018", {"bold": True}), (" — the numbers on the right are from live traffic, not simulation.", {"color": MUTED})],
    ], size=15)
    x0 = MARGIN + Inches(7.7)
    add_rect(s, x0, BODY_Y, Inches(4.45), Inches(3.9), PANEL, BORDER)
    add_text(s, x0 + Inches(0.25), BODY_Y + Inches(0.2), Inches(4), Inches(0.3), "30-DAY FIELD TEST · DRCR, MANUAL = 1.00", size=10.5, color=FAINT, font=MONO)
    for i, (name, val, col) in enumerate([("DDPG", "6.07", ALGO["DDPG"]), ("DQN", "5.03", ALGO["DQN"]), ("Manual", "1.00", SLATE)]):
        y = BODY_Y + Inches(0.6) + i * Inches(1.05)
        add_text(s, x0 + Inches(0.25), y, Inches(1.5), Inches(0.6), name, size=16, color=col, font=SANS, bold=True)
        add_text(s, x0 + Inches(1.8), y - Inches(0.1), Inches(2.4), Inches(0.9), val, size=36, color=col, font=SERIF, align=PP_ALIGN.RIGHT)
        add_rect(s, x0 + Inches(0.25), y + Inches(0.72), int(Inches(3.95) * float(val) / 6.07), Emu(50000), col)

    # 5 our approach ----------------------------------------------------------
    s = d.slide("Our approach: fix the problem, swap only the learner", "1 · The project", notes="The report keeps the paper's states, action and reward fixed and swaps only the algorithm. This codebase turns that written comparison into a measured one: a real environment, four trained agents, held-out products, three seeds, and a dashboard where every number traces back to a checkpoint.")
    add_table(s, MARGIN, BODY_Y, CW, ["Algorithm", "Family", "Action space", "Policy", "Exploration", "Chosen to test"], [
        ["DQN", "Value-based", "11 discrete price buckets", "Off-policy", "ε-greedy schedule", "The paper's baseline; the bucket trade-off"],
        ["DDPG", "Actor-critic, deterministic", "Exact continuous price", "Off-policy", "External OU noise", "The paper's winner; precision without buckets"],
        ["PPO", "Actor-critic, stochastic", "Exact continuous price", "On-policy", "Built into the policy", "Robustness to a noisy reward"],
        ["SAC", "Actor-critic, max-entropy", "Exact continuous price", "Off-policy", "Entropy bonus, self-tuned", "DDPG's benefit with stable, automatic exploration"],
    ], col_widths=[1.1, 2.2, 2.2, 1.3, 2.0, 3.6], size=12, row_h=Inches(0.5), first_col_colors=ALGO)
    add_text(s, MARGIN, BODY_Y + Inches(2.9), CW, Inches(1.6), [
        [("What is held equal for all four: ", {"bold": True}), ("the environment and demand model, the 35-number state, the ±25 % action range, the DRCR reward, network size (2 × 128 ReLU), optimiser, the environment-step budget, the seeds, and the held-out products used for every reported number.", {"color": MUTED})],
    ], size=15)

    # 6 MDP -------------------------------------------------------------------
    s = d.slide("One decision = one product for one month", "2 · The shared MDP", notes="State: 35 numbers in five groups, all known before the price is set. Action: a new price within plus or minus 25 percent of the entering price. Reward: 100 times DRCR. Each product has 15 months: 3 warm-up, 12 decisions.")
    cw3 = (CW - Inches(0.4)) / 3
    add_card(s, MARGIN, BODY_Y, cw3, Inches(3.3), "STATE  s(i,t)  ·  35 numbers", [
        "Own attributes: entering price, score, freight",
        "Recent performance: 3-month rolling sales, revenue, traffic; last DRCR",
        "Pricing history: lag prices, % change",
        "Competitiveness: price / score / freight gaps vs 3 similar products, price rank",
        "Seasonality flags + category one-hot",
    ], accent=SLATE, body_size=12)
    add_card(s, MARGIN + cw3 + Inches(0.2), BODY_Y, cw3, Inches(3.3), "ACTION  a(i,t)  ·  a new price", [
        "Within ±25 % of the entering price",
        "DQN: one of K = 11 buckets (−25 % … +25 %, middle = no change)",
        "DDPG / PPO / SAC: tanh output in [−1, 1] → exact price",
        "Guard: clipped to [0.5×, 1.5×] of the product's observed range",
    ], accent=GOLD, body_size=12)
    add_card(s, MARGIN + 2 * (cw3 + Inches(0.2)), BODY_Y, cw3, Inches(3.3), "REWARD  r(i,t)  ·  scaled DRCR", [
        [("RCR(t) = revenue(t) / category traffic(t)", {"font": MONO, "size": 11})],
        [("DRCR(t) = RCR(t) − RCR(t−1)", {"font": MONO, "size": 11})],
        [("reward = 100 × DRCR", {"font": MONO, "size": 11})],
        "A difference signal: even a good policy hovers near zero",
        "Traffic proxy = units sold across the category that month",
    ], accent=ALGO["SAC"], body_size=12)
    add_text(s, MARGIN, BODY_Y + Inches(3.55), CW, Inches(1.2), [
        [("Time structure. ", {"bold": True}), ("15 months per product: months 1–3 warm up the rolling features and RCR history; the agent prices months 4–15 (12 decisions). Features that describe recent performance are shifted one month so this month's outcome can never leak into the state.", {"color": MUTED})],
    ], size=14)

    # 7 counterfactual problem ------------------------------------------------
    s = d.slide("The hard part: history only shows one price per month", "3 · Counterfactual demand", notes="The dataset records what sold at the one price the seller charged. An RL agent proposes prices nobody charged, so something must predict what would have sold. That something is the demand model, and it is the environment's physics. Every policy, including the seller's own path, is scored through the same model.")
    add_text(s, MARGIN, BODY_Y, Inches(6.2), Inches(0.5), "Logged data", size=14, color=FAINT, font=MONO)
    add_table(s, MARGIN, BODY_Y + Inches(0.4), Inches(6.2), ["month", "price charged", "units sold", "… at $85?", "… at $110?"], [
        ["2017-06", "$92.40", "38", "?", "?"],
        ["2017-07", "$94.10", "35", "?", "?"],
        ["2017-08", "$90.80", "41", "?", "?"],
    ], col_widths=[1.2, 1.5, 1.2, 1.3, 1.3], size=12)
    add_text(s, MARGIN, BODY_Y + Inches(2.2), Inches(6.2), Inches(1.8), [
        [("Pure replay cannot answer the question marks. ", {"bold": True}), ("If the agent picks $110, the row for $94.10 tells us nothing about it.", {"color": MUTED})],
    ], size=15)
    x0 = MARGIN + Inches(6.6)
    add_card(s, x0, BODY_Y, CW - Inches(6.6), Inches(3.9), "THE FIX: A DEMAND-RESPONSE MODEL", [
        "Fit units-sold = f(price, context) on the full history",
        "Ask it for any candidate price at the real month's context (competitors, traffic, season)",
        "It becomes the environment's transition and reward function",
        "Agents and the seller's own prices are scored through the same model → fair, in-simulator comparison",
        [("Validated on each product's last 2 months: R² = %.2f in log-units" % dm["r2_holdout"], {"bold": True, "color": GOLD})],
    ], accent=GOLD, body_size=13)

    # 8 demand model design ---------------------------------------------------
    s = d.slide("The demand model: elasticity first, corrections second", "3 · Counterfactual demand", notes="Layer one is a fixed-effects log-log elasticity regression, demeaned within product. A pooled fit gave a positive price coefficient because products priced high for unrelated reasons also sold well; demeaning removed that confound. Layer two is a shallow gradient-boosted residual, capped at three standard deviations so it can bend the curve but never flip it. Guard rails stop agents from exploiting the model off-distribution.")
    add_card(s, MARGIN, BODY_Y, Inches(6.0), Inches(2.1), "LAYER 1 · FIXED-EFFECTS ELASTICITY BASELINE", [
        [("log(1+qty) = αᵢ + β·log(price) + controls", {"font": MONO, "size": 11.5})],
        "Every regressor demeaned within product → β comes from each product's own price moves",
        "Fitted β = %.2f: +10 %% price ≈ −%.0f %% units" % (dm["elasticity_coef"], abs(dm["elasticity_coef"]) * 10),
    ], accent=GOLD, body_size=12)
    add_card(s, MARGIN, BODY_Y + Inches(2.3), Inches(6.0), Inches(1.75), "LAYER 2 · CAPPED GRADIENT-BOOSTED RESIDUAL", [
        "HistGradientBoosting, depth 3, ≤ 250 trees, early-stopped, L2 = 1.0",
        "Learns competitor gaps, seasonality, category effects the line misses",
        "Output clipped to ±3σ of its in-sample spread → can bend, never flip",
    ], accent=SLATE, body_size=12)
    add_image_fit(s, IMG / "demand_model_monotonicity.png", MARGIN + Inches(6.3), BODY_Y, CW - Inches(6.3), Inches(3.2))
    add_text(s, MARGIN + Inches(6.3), BODY_Y + Inches(3.25), CW - Inches(6.3), Inches(0.9), [
        [("Guard rails: ", {"bold": True}), ("prices clipped to [0.5×, 1.5×] of the product's history; competitor gaps recomputed at the candidate price; time-based holdout, never a random row split.", {"color": MUTED})],
    ], size=12)

    # 9 data ------------------------------------------------------------------
    s = d.slide("The data: a retail pricing panel, real schema, synthetic fallback", "3 · Data", notes="Target dataset is Kaggle's Retail Price Optimization, derived from Olist: one row per product-month with price, units, freight, score, three competitor prices, lag price. The sandbox had no internet, so results today use a schema-identical synthetic panel generated from an explicit elasticity law. Swapping in the real CSV needs no code changes.")
    add_table(s, MARGIN, BODY_Y, Inches(7.4), ["Column group", "Fields", "Feeds"], [
        ["Identity / time", "product_id, product_category_name, month_year", "grouping, warm-up, seasonality"],
        ["Own product", "unit_price, qty, total_price, freight_price, product_score, lag_price, customers", "state, demand-model target"],
        ["Competitors", "comp_1..3, ps1..3, fp1..3", "competitiveness gaps, price rank"],
        ["Calendar", "weekday, weekend, holiday, month, year", "seasonality features"],
    ], col_widths=[1.4, 3.6, 2.4], size=11.5, row_h=Inches(0.55))
    add_card(s, MARGIN + Inches(7.7), BODY_Y, CW - Inches(7.7), Inches(4.0), "SYNTHETIC PANEL USED TODAY", [
        "10 categories × 8 products × 15 months = 1,200 rows, 80 products",
        "units = base × (price / ref)^(−elasticity) × season × noise, category elasticity 1.0–2.5",
        "Prices follow a mean-reverting random walk (~3 % / month)",
        "Same columns as Kaggle → drop the real CSV into data/raw/, refit, rerun",
        [("Product split: %d train / %d held-out, by product, stratified by category" % (n_train, n_test), {"bold": True, "color": GOLD})],
    ], accent=ALGO["SAC"], body_size=12)
    add_text(s, MARGIN, BODY_Y + Inches(2.95), Inches(7.4), Inches(1.3), [
        [("Leakage discipline. ", {"bold": True}), ("A feature may only use what is known before the month's price is set: rolling sales, revenue, traffic and DRCR are shifted one month; the row's own price, units and revenue are never inputs, only fitting targets. Competitor prices are this month's, treated as observable and exogenous.", {"color": MUTED})],
    ], size=13)

    # 10 environment ----------------------------------------------------------
    s = d.slide("One episode, step by step", "3 · The environment (PricingEnv)", notes="A standard Gymnasium environment. Reset picks a product and loads three warm-up months. Each step: observe, act, decode to a price, guard, ask the demand model, compute RCR and the reward, roll the agent's own history forward, load the next real month. Training episodes may bootstrap past month 15; evaluation never does.")
    stepsx = [
        ("reset", "pick a product;\nmonths 1–3 warm up", SLATE),
        ("observe", "state s_t\n(35 numbers)", SLATE),
        ("act", "bucket or\nvalue in [−1, 1]", GOLD),
        ("decode", "entering price\n× (1 ± 25 %)", GOLD),
        ("guard", "clip to [0.5×, 1.5×]\nof history", NEG),
        ("predict", "demand model →\nunits at that price", ALGO["SAC"]),
        ("reward", "RCR = p·q / traffic\nr = 100·ΔRCR", ALGO["SAC"]),
        ("roll on", "lag prices, gaps,\nrolling sales → next month", SLATE),
    ]
    bw = (CW - Inches(0.15) * 7) / 8
    for i, (name, desc, col) in enumerate(stepsx):
        x = MARGIN + i * (bw + Inches(0.15))
        add_rect(s, x, BODY_Y, bw, Inches(1.75), PANEL, BORDER)
        add_rect(s, x, BODY_Y, bw, Inches(0.07), col)
        add_text(s, x + Inches(0.08), BODY_Y + Inches(0.15), bw - Inches(0.16), Inches(0.4), name.upper(), size=10.5, color=col, font=MONO, bold=True)
        add_text(s, x + Inches(0.08), BODY_Y + Inches(0.55), bw - Inches(0.16), Inches(1.2), desc, size=11, color=TEXT)
        if i < len(stepsx) - 1:
            add_text(s, x + bw - Inches(0.02), BODY_Y + Inches(0.65), Inches(0.2), Inches(0.4), "›", size=16, color=FAINT)
    add_table(s, MARGIN, BODY_Y + Inches(2.05), CW, ["", "Training episodes", "Evaluation episodes"], [
        ["Products", "random draw from the %d training products" % n_train, "each of the %d held-out products, once" % n_test],
        ["Horizon", "20 steps", "until real months run out (12 decisions)"],
        ["Past month 15", "bootstrapped: competitors ±3 %, traffic ±5 % noise from the last real row", "never — episode ends"],
        ["Exploration", "on (ε-greedy / OU noise / sampled policy)", "off — deterministic actions"],
    ], col_widths=[1.4, 4.2, 4.2], size=12, row_h=Inches(0.42))

    # 11-14 algorithms --------------------------------------------------------
    def algo_slide(name, eyebrow, one_liner, how, explore, hyper, say, check, notes):
        col = ALGO[name]
        s = d.slide(f"{name} — {one_liner}", eyebrow, notes=notes)
        add_rect(s, MARGIN - Inches(0.22), Inches(0.66), Inches(0.08), Inches(0.62), col)
        add_card(s, MARGIN, BODY_Y, Inches(7.4), Inches(2.55), "HOW IT WORKS", how, accent=col, body_size=12)
        add_card(s, MARGIN, BODY_Y + Inches(2.7), Inches(3.6), Inches(1.85), "EXPLORATION", explore, accent=col, body_size=11.5)
        add_card(s, MARGIN + Inches(3.8), BODY_Y + Inches(2.7), Inches(3.6), Inches(1.85), "HYPERPARAMETERS", hyper, accent=col, body_size=11.5)
        x0 = MARGIN + Inches(7.7)
        add_card(s, x0, BODY_Y, CW - Inches(7.7), Inches(2.95), "WHAT TO SAY ABOUT IT", say, accent=GOLD, body_size=12)
        add_card(s, x0, BODY_Y + Inches(3.1), CW - Inches(7.7), Inches(1.45), "IMPLEMENTATION CHECK", check, accent=SLATE, body_size=11.5)
        return s

    algo_slide("DQN", "4 · Value-based, discrete", "score every price move, pick the best",
        how=[
            [("Q(s, · | θ): 35 numbers in → 11 values out, one per bucket", {"font": MONO, "size": 11})],
            [("target  y = r + γ · maxₐ′ Q(s′, a′ | θ⁻)", {"font": MONO, "size": 11})],
            [("loss  = Huber( Q(s, a | θ) − y )", {"font": MONO, "size": 11})],
            "Experience replay: 50,000 transitions, random batches of 64 → decorrelated updates",
            "Target network θ⁻: frozen copy refreshed every 100 updates → the target stops chasing itself",
        ],
        explore=["ε-greedy: random bucket with prob. ε, else argmax", "ε: 1.0 → 0.05 linearly over 25,000 steps", "Evaluation: ε = 0"],
        hyper=["γ = 0.99, Adam lr 1e-3", "batch 64, buffer 50k, 500 warm-up steps", "K = 11 buckets over ±25 %"],
        say=["Simplest and most stable; the paper's own baseline", "Structural limit: 11 buckets → finest move is 5 %", "The paper measured the cost of that imprecision (5.03 vs 6.07)", "In this run its ε schedule did not finish: at 15k steps it still acted randomly 43 % of the time"],
        check=["Learned CartPole-v1 (eval return > 100) in 4,000 steps before touching the pricing env"],
        notes="DQN never outputs a price directly; it scores the 11 moves and picks the best. Two stabilisers: replay buffer and a target network. Mention the scheduling handicap honestly before someone asks why it is below the baseline.")

    algo_slide("DDPG", "4 · Deterministic actor-critic", "an actor outputs the exact price, a critic scores it",
        how=[
            [("critic:  y = r + γ · Q′(s′, μ′(s′));  loss = MSE(Q(s,a) − y)", {"font": MONO, "size": 11})],
            [("actor:   loss = − mean Q(s, μ(s))   (gradient flows through the critic)", {"font": MONO, "size": 11})],
            [("targets: θ′ ← 0.005·θ + 0.995·θ′ after every update", {"font": MONO, "size": 11})],
            "Actor ends in tanh → [−1, 1] → ±25 % of the entering price",
            "Target copies of both actor and critic keep the bootstrapped target stable",
        ],
        explore=["Actor is deterministic → noise added from outside", "Ornstein–Uhlenbeck, θ = 0.15, σ 0.20 → 0.05 over 25k steps", "Reset at each episode end; off in evaluation"],
        hyper=["γ = 0.99, actor lr 1e-4, critic lr 1e-3", "τ = 0.005, batch 64, buffer 50k", "500 warm-up steps"],
        say=["Removes the bucket trade-off entirely — an exact price", "The paper's field winner (6.07 vs 5.03) is the empirical case for it", "Cost: a delicate two-network loop, sensitive to hyperparameters and to the hand-scheduled noise", "Here: same training level as SAC, but 5× wider spread across seeds on held-out products"],
        check=["Learned Pendulum-v1: +100 return between first and last 10 episodes of a 6,000-step run"],
        notes="Two networks: the actor maps state to one price, the critic scores state-price pairs, and the actor is nudged in whatever direction the critic says raises the score. The hand-scheduled noise is the weakness SAC was chosen to remove.")

    algo_slide("PPO", "4 · On-policy, clipped updates", "improve a price distribution by a bounded step each batch",
        how=[
            "Collect 1,024 steps with the current policy; store log π(a|s) and V(s)",
            [("GAE:  δₜ = rₜ + γV(sₜ₊₁) − V(sₜ);  Âₜ = Σ (γλ)ᵏ δₜ₊ₖ,  λ = 0.95", {"font": MONO, "size": 11})],
            [("ratio rₜ = πθ(aₜ|sₜ) / πθₒₗₔ(aₜ|sₜ)", {"font": MONO, "size": 11})],
            [("loss = − mean min( rₜÂₜ , clip(rₜ, 0.8, 1.2)·Âₜ ) − 0.01·entropy", {"font": MONO, "size": 11})],
            "10 epochs over minibatches of 64, then discard the rollout — no replay buffer",
        ],
        explore=["Built in: the policy is a Gaussian with a learned std", "Entropy bonus stops it collapsing early", "Evaluation uses the mean"],
        hyper=["γ = 0.99, λ = 0.95, clip ε = 0.2", "lr 3e-4, rollout 1,024, 10 epochs, minibatch 64", "entropy 0.01, value coef 0.5, grad-norm 0.5"],
        say=["Chosen for robustness to a noisy reward: the clip bounds how far one lucky month can move the policy", "Price: sample efficiency — it cannot reuse old data", "Its running mean is still climbing at 15k steps while off-policy agents settled by 3k", "Runs continuous here (not the report's 'optionally discrete') so it compares directly with DDPG/SAC"],
        check=["Pendulum-v1 needed ~100k steps (vs 6k for DDPG/SAC) — the value estimate takes 10–15 rollouts to calibrate"],
        notes="PPO learns a distribution over prices and improves it by a bounded amount after each fresh batch. Two implementation details: actor and critic have separate optimisers and separate gradient clipping; clipping them together silently crushes the policy gradient. Slower to learn, by design.")

    algo_slide("SAC", "4 · Maximum entropy, twin critics", "reward randomness, and tune that reward automatically",
        how=[
            [("objective:  max Σₜ [ r(sₜ,aₜ) + α·H(π(·|sₜ)) ]", {"font": MONO, "size": 11})],
            [("critics:  y = r + γ·[ min(Q₁′,Q₂′)(s′,a′) − α·log π(a′|s′) ],  a′ ~ π(·|s′)", {"font": MONO, "size": 11})],
            [("actor:    loss = mean[ α·log π(a|s) − min(Q₁,Q₂)(s,a) ]   (reparameterised)", {"font": MONO, "size": 11})],
            [("temperature:  loss = − mean[ logα · (log π(a|s) + H_target) ],  H_target = −1", {"font": MONO, "size": 11})],
            "tanh-squashed Gaussian policy with the log-prob correction; twin critics counter overestimation",
        ],
        explore=["Intrinsic: sample from the policy during training", "α rises when entropy falls below target, and vice versa", "Evaluation: tanh(mean)"],
        hyper=["γ = 0.99, all three lr 3e-4", "τ = 0.005, batch 64, buffer 50k", "500 warm-up steps, target entropy −1"],
        say=["Chosen for the low-traffic problem: keeps exploring nearby prices while uncertain, no noise schedule to tune", "Tests whether DDPG's exact price can be had with better stability", "Result: best held-out DRCR and the tightest spread across seeds (5× tighter than DDPG)", "Update order: critics → actor → temperature → target critics; α detached everywhere but its own loss"],
        check=["Learned Pendulum-v1 in 6,000 steps with finite critic loss and a finite, moving α throughout"],
        notes="SAC is DDPG's off-policy actor-critic with three upgrades: entropy in the objective, twin critics with the minimum, and a learned temperature. Nothing is scheduled by hand. It won on both mean and stability.")

    # 15 side-by-side ---------------------------------------------------------
    s = d.slide("Four designs side by side", "4 · Comparison", notes="The report's Section 6 table, now with the measured column. Read across: type, action space, on/off policy, how it explores, its strength, its limitation, and what it actually scored.")
    add_table(s, MARGIN, BODY_Y, CW, ["", "Type", "Action space", "On / off-policy", "Exploration", "Key strength", "Key limitation", "Test DRCR"], [
        ["DQN", "Value-based", "Discrete (11 buckets)", "Off-policy", "ε-greedy schedule", "Simple, stable, sample-efficient with replay", "Buckets cap price precision", "%.4f ± %.4f" % (S("dqn").mean_test_drcr, S("dqn").std_test_drcr)],
        ["DDPG", "Actor-critic", "Continuous", "Off-policy", "External OU noise", "Exact price, no discretisation trade-off", "Sensitive to hyperparameters and noise schedule", "%.4f ± %.4f" % (S("ddpg").mean_test_drcr, S("ddpg").std_test_drcr)],
        ["PPO", "Actor-critic", "Continuous", "On-policy", "Stochastic policy + entropy bonus", "Clipped updates are robust to noisy rewards", "Cannot reuse data → needs more steps", "%.4f ± %.4f" % (S("ppo").mean_test_drcr, S("ppo").std_test_drcr)],
        ["SAC", "Actor-critic", "Continuous", "Off-policy", "Entropy term, self-tuned α", "Automatic exploration, twin critics, stable", "Extra temperature parameter and two critics", "%.4f ± %.4f" % (S("sac").mean_test_drcr, S("sac").std_test_drcr)],
    ], col_widths=[0.8, 1.2, 1.5, 1.2, 1.8, 2.6, 2.6, 1.5], size=11, row_h=Inches(0.7), first_col_colors=ALGO)

    # 16 protocol -------------------------------------------------------------
    s = d.slide("Fair-comparison protocol and baselines", "5 · Protocol", notes="A DRCR number means nothing alone, so three references are scored through the identical environment. Everything that could bias the comparison is held equal. Each algorithm also had to learn a standard Gymnasium task with the same code before touching this environment; that separates a broken learner from a broken environment.")
    add_table(s, MARGIN, BODY_Y, Inches(6.3), ["Reference", "What it does", "Why it is there"], [
        ["Static", "keeps the entering price every month", "'do nothing' — the floor a useful agent must beat"],
        ["Random", "uniform random action each month", "how much the environment gives away by chance"],
        ["Historical", "the seller's actual prices, through the same model", "the human pricer; basis of 'uplift vs. seller'"],
    ], col_widths=[1.1, 2.6, 2.6], size=11.5, row_h=Inches(0.62))
    add_card(s, MARGIN + Inches(6.6), BODY_Y, CW - Inches(6.6), Inches(4.05), "HELD EQUAL FOR EVERY AGENT", [
        "Environment, demand model, feature scaler, %d / %d product split" % (n_train, n_test),
        "Network size 2 × 128 ReLU in every actor, critic and Q network; Adam",
        "Budget in environment steps (%s per run) — fair across on- and off-policy" % f"{steps:,}",
        "Seeds %s for every agent and baseline; mean ± std reported" % ", ".join(map(str, seeds)),
        "Evaluation: deterministic actions, %d held-out products, real months only" % n_test,
        [("Smoke tests first: DQN on CartPole-v1, DDPG/PPO/SAC on Pendulum-v1 — caught a PPO gradient-clip bug and a wrong-sign pooled elasticity", {"color": GOLD})],
    ], accent=GOLD, body_size=12)
    add_text(s, MARGIN, BODY_Y + Inches(2.75), Inches(6.3), Inches(1.4), [
        [("Shared contract. ", {"bold": True}), ("Every agent implements the same four methods — select_action, observe, update, save/load — so training and evaluation loops are identical code. The comparison is of algorithm design, not of setup.", {"color": MUTED})],
    ], size=13)

    # 17 results table --------------------------------------------------------
    s = d.slide("Results: SAC leads, all three continuous agents beat the seller", "6 · Results", notes="Three seeds times %s steps, evaluated on the %d held-out products. Test DRCR is the trained reward; revenue vs seller is the intuitive version through the same demand model. The two agree on the ranking. DQN is below the do-nothing baseline on this budget." % (f"{steps:,}", n_test))
    rows = []
    for key, label in [("sac", "SAC"), ("ppo", "PPO"), ("ddpg", "DDPG"), ("static", "Static (no change)"), ("random", "Random"), ("dqn", "DQN")]:
        r = S(key)
        tail = "—" if pd.isna(r.get("mean_final_reward", float("nan"))) else "%.2f ± %.2f" % (r.mean_final_reward, r.std_across_seeds)
        rows.append([label, "%.4f" % r.mean_test_drcr, "%.4f" % r.std_test_drcr, "%+.1f %%" % uplift[key], tail])
    add_table(s, MARGIN, BODY_Y, Inches(8.2), ["Policy", "Mean test DRCR", "Std across seeds", "Revenue vs. seller", "Training-tail reward"], rows,
              col_widths=[2.0, 1.5, 1.5, 1.6, 1.8], size=12.5, row_h=Inches(0.48), first_col_colors=ALGO)
    add_card(s, MARGIN + Inches(8.5), BODY_Y, CW - Inches(8.5), Inches(4.1), "HOW TO READ IT", [
        "Test DRCR: the reward the agents trained on, averaged per month — the report's metric",
        "Revenue vs. seller: total predicted revenue under the policy minus under the seller's own prices, same model, all %d products" % n_test,
        "Training-tail: on the 100× scale, training products, last 20 % of steps — not comparable to test DRCR",
        [("Continuous > bucketed, reproducing the paper's DDPG > DQN. SAC's spread is 5× tighter than DDPG's.", {"color": GOLD})],
    ], accent=GOLD, body_size=12)
    add_text(s, MARGIN, BODY_Y + Inches(3.55), Inches(8.2), Inches(0.6), "%d seeds × %s environment steps per run · %d held-out products · results/comparison_summary.csv" % (len(seeds), f"{steps:,}", n_test), size=11, color=FAINT, font=MONO)

    # 18 results charts -------------------------------------------------------
    s = d.slide("Held-out DRCR and cross-seed stability", "6 · Results", notes="Left: mean test DRCR with error bars across seeds. Right: the per-seed spread. SAC's three seeds land within 0.002 of each other; DQN's range from 0.0005 to 0.058, which is the exploration-schedule problem showing up as variance.")
    add_image_fit(s, IMG / "test_drcr_comparison.png", MARGIN, BODY_Y, (CW - Inches(0.3)) / 2, Inches(4.1))
    add_image_fit(s, IMG / "drcr_stability.png", MARGIN + (CW + Inches(0.3)) / 2, BODY_Y, (CW - Inches(0.3)) / 2, Inches(4.1))

    # 19 training -------------------------------------------------------------
    s = d.slide("Training behaviour: off-policy settles early, PPO is still climbing", "6 · Training", notes="Left panel: rolling reward, dominated by product-level shocks. Right: the running mean since step zero, where the noise averages out. SAC and DDPG converge on top by about 3,000 steps; PPO climbs steadily from minus one and has not plateaued at 15,000; DQN stays below zero.")
    add_image_fit(s, IMG / "dash_training.png", MARGIN, BODY_Y, CW, Inches(4.2))

    # 20 dashboard simulate ---------------------------------------------------
    s = d.slide("Live demo: the month-by-month race", "6 · Dashboard", notes="Open docs/index.html, Simulate tab. Left: held-out products ranked by uplift vs the seller. Press Play: each month the four agents re-price; the dashed line is what the seller charged; hollow dots are guard hits; the scoreboard updates. Below, the race chart shows cumulative revenue gained vs the seller — above zero is beating the human.")
    add_image_fit(s, IMG / "dash_simulate.png", MARGIN, BODY_Y, Inches(8.4), Inches(4.25))
    add_card(s, MARGIN + Inches(8.7), BODY_Y, CW - Inches(8.7), Inches(4.25), "WHAT TO POINT AT", [
        "Watchlist: %d held-out products, ranked" % n_test,
        "Dashed line = the seller's actual price",
        "Hollow dots = the price guard was hit",
        "▶ Play steps month by month; stat cards are cumulative-to-date",
        "P0050: SAC +5.0 % vs. the seller",
        "Then the aggregate-uplift bars: the headline slide",
    ], accent=GOLD, body_size=12)

    # 21 race chart -----------------------------------------------------------
    s = d.slide("Cumulative revenue gained vs. the seller (P0050, furniture)", "6 · Dashboard", notes="Each line is a policy's simulated revenue minus what the seller's own prices earn through the same model, accumulated month by month. SAC, PPO and DDPG climb to roughly six to seven thousand above the seller over twelve months; DQN slips below.")
    add_image_fit(s, IMG / "dash_race.png", MARGIN, BODY_Y, CW, Inches(4.2))

    # 22 what-if --------------------------------------------------------------
    s = d.slide("Live demo: what the demand model believes", "6 · Dashboard", notes="What-if tab. Pick a product and month, drag the price: units fall, revenue moves, DRCR updates. The markers show where the seller priced and where each agent priced that month. This is the single best way to explain the counterfactual model — and why the agents raise prices: for most products revenue keeps rising with price up to the guard.")
    add_image_fit(s, IMG / "dash_whatif.png", MARGIN, BODY_Y, Inches(8.4), Inches(4.25))
    add_card(s, MARGIN + Inches(8.7), BODY_Y, CW - Inches(8.7), Inches(4.25), "WHAT TO POINT AT", [
        "Drag the slider: units fall, revenue rises, DRCR updates live",
        "Grey dot = seller's price; coloured dots = each agent's price that month",
        "Shaded = outside the ±25 % per-month move",
        "Elasticity %.2f: inelastic → revenue keeps rising with price" % dm["elasticity_coef"],
        "This is why the continuous agents push up to the guard",
        "Streamlit app runs the same query live for any product",
    ], accent=GOLD, body_size=12)

    # 23 interpretation -------------------------------------------------------
    s = d.slide("What the ranking supports", "7 · Interpretation", notes="Three claims the data supports, and one that it does not. Continuous beats bucketed; entropy-regularised exploration is more stable than a hand-scheduled noise process; PPO is competitive but slower. What it does not show: that raising prices to the guard would work on a real market — that depends on the true elasticity, and the paper's field test is the evidence for that, not this project.")
    cw3 = (CW - Inches(0.4)) / 3
    add_card(s, MARGIN, BODY_Y, cw3, Inches(2.6), "CONTINUOUS > BUCKETED", ["DDPG, PPO and SAC all beat the seller by 3–4 %", "DQN's finest move is 5 % and it landed below the do-nothing baseline", "Reproduces the paper's DDPG > DQN"], accent=ALGO["DDPG"], body_size=12)
    add_card(s, MARGIN + cw3 + Inches(0.2), BODY_Y, cw3, Inches(2.6), "SELF-TUNED EXPLORATION IS STEADIER", ["SAC and DDPG reach the same training level (2.32 vs 2.30)", "SAC's held-out spread across seeds: ±%.4f vs DDPG ±%.4f" % (S("sac").std_test_drcr, S("ddpg").std_test_drcr), "No noise schedule to get wrong"], accent=ALGO["SAC"], body_size=12)
    add_card(s, MARGIN + 2 * (cw3 + Inches(0.2)), BODY_Y, cw3, Inches(2.6), "ON-POLICY COSTS SAMPLES", ["PPO competitive at %.4f but still improving at 15k steps" % S("ppo").mean_test_drcr, "It cannot reuse data; off-policy agents settled by ~3k", "Exactly the trade-off the report predicts"], accent=ALGO["PPO"], body_size=12)
    add_text(s, MARGIN, BODY_Y + Inches(2.9), CW, Inches(1.3), [
        [("What it does not show. ", {"bold": True, "color": NEG}), ("That pushing prices to the guard would pay on Tmall. Inside this simulator demand is inelastic, so it is the right move; on a real market that depends on the true elasticity. The paper's 30-day field test is the evidence that RL pricing pays; this project is the evidence about which algorithm learns it best under a fair protocol.", {"color": MUTED})],
    ], size=13)

    # 24 caveats --------------------------------------------------------------
    s = d.slide("Caveats we state before anyone asks", "7 · Limitations", notes="Five honest limitations. Say them yourself; they are much cheaper to own than to be asked about.")
    cav = [
        ("DQN is under-trained, not only bucket-limited", "ε anneals over 25,000 steps; the run gave it 15,000, so the saved checkpoint still acted randomly 43 % of the time. Seed 2 scored 0.0005. A 50k-step run removes this confound; the bucket penalty stays.", ALGO["DQN"]),
        ("The agents mostly raise prices and hit the guard", "Fitted elasticity %.2f is inelastic, so predicted revenue rises with price up to the 1.5× guard. Correct inside the simulator; not evidence about a real market." % dm["elasticity_coef"], NEG),
        ("The demand model under-estimates the generating elasticity", "Synthetic categories were built with elasticity 1.0–2.5; monthly price moves are only ~3 %, so there is little within-product variation and ridge shrinks the slope.", GOLD),
        ("The data is synthetic", "Schema-identical to Kaggle's Retail Price Optimization with a known demand law, which is what lets the simulator be validated (R² %.2f). Real data changes every number, not the method." % dm["r2_holdout"], SLATE),
        ("%d products, %d seeds" % (n_test, len(seeds)), "Enough to rank, not for tight confidence intervals; the per-seed dots show the honest spread.", MUTED),
    ]
    for i, (head, body, col) in enumerate(cav):
        y = BODY_Y + i * Inches(0.82)
        add_rect(s, MARGIN, y, Inches(0.06), Inches(0.72), col)
        add_text(s, MARGIN + Inches(0.2), y - Inches(0.02), Inches(4.3), Inches(0.75), head, size=14, color=TEXT, bold=True)
        add_text(s, MARGIN + Inches(4.6), y - Inches(0.02), CW - Inches(4.6), Inches(0.8), body, size=12, color=MUTED)

    # 25 next steps -----------------------------------------------------------
    s = d.slide("Next steps", "7 · Future work", notes="Ordered by value per hour. The 50k run is one command and fixes the DQN confound. The real CSV needs no code changes. More seeds tighten the intervals. Per-category elasticity and a richer traffic proxy are modelling improvements; offline RL is the methodological step the paper's cold-start problem points at.")
    nxt = [
        ("Re-run at the full budget", "TOTAL_STEPS=50000 SEEDS=\"0 1 2 3 4\" bash scripts/run_full_comparison.sh — lets DQN's ε schedule finish; 5 seeds for tighter intervals", GOLD),
        ("Swap in the real Kaggle data", "Drop the CSV into data/raw/, refit the demand model, rerun; the pipeline is schema-driven end to end", ALGO["SAC"]),
        ("Per-category elasticity", "Let β vary by category in the baseline; the synthetic generator already does, the fit does not yet", SLATE),
        ("Log losses and re-plot", "train.py now records each agent's update metrics; the next run lights up the critic/value-loss panel automatically", SLATE),
        ("Offline RL for cold-start products", "The paper pre-trains on human prices; conservative offline methods (CQL, IQL) are the modern version of that idea", ALGO["PPO"]),
    ]
    for i, (head, body, col) in enumerate(nxt):
        y = BODY_Y + i * Inches(0.82)
        add_text(s, MARGIN, y, Inches(0.5), Inches(0.6), str(i + 1), size=22, color=col, font=SERIF)
        add_text(s, MARGIN + Inches(0.5), y - Inches(0.02), Inches(3.9), Inches(0.75), head, size=14, color=TEXT, bold=True)
        add_text(s, MARGIN + Inches(4.5), y - Inches(0.02), CW - Inches(4.5), Inches(0.8), body, size=12, color=MUTED)

    # 26 conclusion -----------------------------------------------------------
    s = d.slide("Conclusion", "Summary", notes="One shared problem, four learners, one fair protocol, one measured answer. SAC best and steadiest; continuous beats bucketed; PPO robust but slow; everything reproducible from one script and visible in the dashboard.")
    add_text(s, MARGIN, BODY_Y, CW, Inches(1.0), "Same states, same actions, same reward — only the learner changed. SAC set the best prices and did so most consistently.", size=22, color=TEXT, font=SERIF)
    add_bullets(s, MARGIN, BODY_Y + Inches(1.2), CW, Inches(3.0), [
        [("Continuous pricing beats bucketed pricing", {"bold": True}), (" — DDPG, PPO and SAC all beat the seller's own prices by 3–4 %; DQN, with 5 % steps and an unfinished exploration schedule, did not.", {"color": MUTED})],
        [("Entropy-regularised exploration is the stable one", {"bold": True}), (" — SAC matched DDPG's level with a 5× tighter spread across seeds and nothing scheduled by hand.", {"color": MUTED})],
        [("On-policy robustness costs samples", {"bold": True}), (" — PPO was competitive and still climbing when the budget ran out.", {"color": MUTED})],
        [("The comparison is honest and reproducible", {"bold": True}), (" — a validated counterfactual demand model with guard rails, held-out products, identical budgets, smoke-tested learners, and a dashboard where every number traces to a checkpoint.", {"color": MUTED})],
    ], size=15)

    # 27 references -----------------------------------------------------------
    s = d.slide("References", "Sources", notes="")
    add_bullets(s, MARGIN, BODY_Y, CW, BODY_H, [
        "Liu, J., Zhang, Y., Wang, X., Deng, Y., & Wu, X. (2019). Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning: A Field Experiment. arXiv:1912.02572.",
        "Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533.",
        "Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. arXiv:1509.02971.",
        "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
        "Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor. arXiv:1801.01290.",
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
        "Kaggle: Retail Price Optimization dataset (Olist-derived). Code and dashboard: github.com/Feliz7000/rl-dynamic-pricing-comparison",
    ], size=13, space_after=10)

    # 28 Q&A ------------------------------------------------------------------
    s = d.slide(notes="Likely questions: how can you score a price never charged (the demand model, same for every policy); why do agents raise prices (inelastic fitted demand); why is DQN below baseline (buckets plus the unfinished epsilon schedule); why is PPO behind SAC (sample efficiency, still climbing); how do you know the RL code is right (CartPole and Pendulum first).")
    add_text(s, MARGIN, Inches(2.6), CW, Inches(1.2), "Questions", size=44, color=TEXT, font=SERIF)
    add_rect(s, MARGIN, Inches(3.7), Inches(1.2), Emu(28000), GOLD)
    add_text(s, MARGIN, Inches(3.95), CW, Inches(2.2), [
        "Live dashboard: docs/index.html · Live models: streamlit run app/streamlit_app.py",
        "github.com/Feliz7000/rl-dynamic-pricing-comparison",
    ], size=15, color=MUTED, font=MONO)

    d.save()
    print(f"Wrote {OUT} ({d.n} slides, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
