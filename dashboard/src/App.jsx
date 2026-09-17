import React, { useState } from "react";
import { COLORS, GLOBAL_CSS, MONO, SERIF } from "./theme.js";
import { META, SEED_OPTIONS } from "./data.js";
import { Segmented } from "./components/ui.jsx";
import SimulateView from "./views/SimulateView.jsx";
import WhatIfView from "./views/WhatIfView.jsx";
import TrainingView from "./views/TrainingView.jsx";
import ResultsView from "./views/ResultsView.jsx";

const TABS = [
  ["simulate", "Simulate"],
  ["whatif", "What-if"],
  ["training", "Training"],
  ["results", "Results"],
];

const initialTab = () => {
  const h = (typeof window !== "undefined" ? window.location.hash : "").replace("#", "");
  return TABS.some(([k]) => k === h) ? h : "simulate";
};

export default function App() {
  const [tab, setTabState] = useState(initialTab);
  const [seed, setSeed] = useState("mean");
  const setTab = (key) => {
    setTabState(key);
    if (typeof window !== "undefined") window.history.replaceState(null, "", `#${key}`);
  };

  return (
    <div style={{ background: COLORS.bg, color: COLORS.text, minHeight: "100vh" }}>
      <style>{GLOBAL_CSS}</style>

      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12, padding: "20px 28px", borderBottom: `1px solid ${COLORS.border}` }}>
        <div>
          <div style={{ fontFamily: MONO, fontSize: 11, color: COLORS.gold, letterSpacing: 0.5, marginBottom: 4 }}>CSEAM731 · CIA-3 · DQN · DDPG · PPO · SAC</div>
          <h1 style={{ fontFamily: SERIF, fontSize: 24, fontWeight: 500, margin: 0, color: COLORS.text }}>Dynamic Pricing — a reinforcement learning comparison</h1>
        </div>
        <div style={{ display: "flex", gap: 2, background: COLORS.panel, padding: 3, border: `1px solid ${COLORS.border}` }}>
          {TABS.map(([key, label]) => (
            <button key={key} className="tab-btn" onClick={() => setTab(key)} style={{ padding: "7px 16px", fontSize: 13, border: "none", cursor: "pointer", background: tab === key ? COLORS.gold : "transparent", color: tab === key ? "#161A1F" : COLORS.muted, fontWeight: 500 }}>
              {label}
            </button>
          ))}
        </div>
      </header>

      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 24, padding: "12px 28px", borderBottom: `1px solid ${COLORS.border}`, background: COLORS.panel }}>
        <Segmented label="Checkpoint seed" options={SEED_OPTIONS.map((s) => ({ value: s, label: s === "mean" ? "mean of seeds" : `seed ${s}` }))} value={seed} onChange={setSeed} />
        <span style={{ fontFamily: MONO, fontSize: 11.5, color: COLORS.faint }}>
          {META.seeds.length} seeds × {META.total_steps?.toLocaleString()} steps · {META.n_train_products} train / {META.n_test_products} held-out products · {META.data_source} data
        </span>
      </div>

      {tab === "simulate" && <SimulateView seed={seed} />}
      {tab === "whatif" && <WhatIfView seed={seed} />}
      {tab === "training" && <TrainingView seed={seed} />}
      {tab === "results" && <ResultsView />}

      <footer style={{ borderTop: `1px solid ${COLORS.border}`, padding: "16px 28px", fontSize: 12, color: COLORS.faint, lineHeight: 1.6 }}>
        Everything shown here is measured, not synthesized: prices come from the four trained checkpoints acting deterministically on held-out products, outcomes from the demand-response model fitted on the pricing data, and training curves from the logged runs ({META.seeds.length} seeds × {META.total_steps?.toLocaleString()} steps). Regenerate with <span className="mono">python scripts/export_dashboard_data.py</span> after any retraining.
      </footer>
    </div>
  );
}
