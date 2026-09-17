import React from "react";
import { ResponsiveContainer, ComposedChart, Bar, Cell, ErrorBar, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from "recharts";
import { ALGOS, BASELINES, ALL_POLICIES, META, SUMMARY } from "../data.js";
import { COLORS, MONO, SERIF, fmtNum } from "../theme.js";
import { CustomTooltip, SectionTitle, axisProps } from "../components/ui.jsx";

function Formula({ children }) {
  return <div style={{ fontFamily: MONO, fontSize: 12.5, color: COLORS.gold, background: COLORS.panelAlt, padding: "8px 10px", margin: "8px 0", borderLeft: `2px solid ${COLORS.gold}` }}>{children}</div>;
}

export default function ResultsView() {
  const rows = ALL_POLICIES.filter((p) => SUMMARY[p.key]);
  const barData = rows.map((p) => {
    const s = SUMMARY[p.key];
    const row = { name: p.label, mean: s.mean_test_drcr, err: s.std_test_drcr, fill: p.color };
    s.per_seed_test_drcr.forEach((v, i) => { row[`s${i}`] = v; });
    return row;
  });
  const best = rows.reduce((b, p) => (SUMMARY[p.key].mean_test_drcr > SUMMARY[b.key].mean_test_drcr ? p : b), rows[0]);
  const dqnNote = META.dqn_eps_decay_steps && META.total_steps && META.dqn_eps_decay_steps > META.total_steps;

  return (
    <div style={{ padding: "22px 28px 32px" }}>
      <SectionTitle
        title="Held-out test DRCR by algorithm"
        caption={`Mean Difference of Revenue Conversion Rate over ${META.n_test_products} held-out products the agents never trained on, averaged over ${META.seeds.length} seeds (bars: mean ± std across seeds; dots: individual seeds). Higher is better.`}
        right={`${META.seeds.length} seeds × ${META.total_steps?.toLocaleString()} env steps`}
      />
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={barData} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
          <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="name" {...axisProps} interval={0} />
          <YAxis {...axisProps} width={56} tickFormatter={(v) => v.toFixed(3)} />
          <Tooltip content={<CustomTooltip formatter={(v) => v.toFixed(4)} />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
          <ReferenceLine y={0} stroke={COLORS.faint} />
          <Bar dataKey="mean" name="mean test DRCR" radius={[2, 2, 0, 0]} isAnimationActive={false} maxBarSize={64}>
            {barData.map((e, i) => <Cell key={i} fill={e.fill} />)}
            <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke={COLORS.text} />
          </Bar>
          {META.seeds.map((_, i) => (
            <Scatter key={i} dataKey={`s${i}`} name={`seed ${META.seeds[i]}`} fill="#fff" fillOpacity={0.85} shape="circle" isAnimationActive={false} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      <div style={{ marginTop: 26, overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr>
              <th>Algorithm</th><th>Action space</th><th>Policy type</th><th>Mean test DRCR ± std</th><th>Per-seed</th><th>Training-tail reward</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const s = SUMMARY[p.key];
              const algo = ALGOS.find((a) => a.key === p.key);
              return (
                <tr key={p.key} style={{ background: p.key === best.key ? "rgba(225,163,57,0.06)" : "transparent" }}>
                  <td><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 9999, background: p.color, marginRight: 8 }} />{p.label}{p.key === best.key && <span className="mono" style={{ color: COLORS.gold, fontSize: 11, marginLeft: 8 }}>BEST</span>}</td>
                  <td className="muted">{algo ? algo.action_space : "—"}</td>
                  <td className="muted">{algo ? algo.policy : "Baseline"}</td>
                  <td className="mono">{fmtNum(s.mean_test_drcr, 4)} ± {fmtNum(s.std_test_drcr, 4)}</td>
                  <td className="mono muted">{s.per_seed_test_drcr.map((v) => v.toFixed(4)).join(" · ")}</td>
                  <td className="mono muted">{s.mean_final_reward == null ? "—" : `${fmtNum(s.mean_final_reward)} ± ${fmtNum(s.std_final_reward)}`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {dqnNote && (
        <p style={{ color: COLORS.muted, fontSize: 12.5, marginTop: 12, maxWidth: 780 }}>
          Note on DQN: its ε-greedy schedule anneals over {META.dqn_eps_decay_steps.toLocaleString()} steps, longer than this run's {META.total_steps.toLocaleString()}-step budget, so it never reached its low-exploration phase. Its result here understates DQN beyond the discretisation penalty the report discusses.
        </p>
      )}

      <div style={{ marginTop: 34 }}>
        <SectionTitle title="The shared MDP every algorithm was trained on" caption="Identical states, actions and reward for all four — the comparison is of algorithm design, not problem setup." />
        <div className="card-grid three-grid">
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>State s<sub>i,t</sub></div>
            <div style={{ fontSize: 13, color: COLORS.muted }}>Per product-month feature vector: own attributes (price entering the month, score, freight), recent sales/traffic (3-month rolling), pricing history (lag prices, % change), competitiveness vs. three similar products (price/score/freight gaps, rank), seasonality flags, category.</div>
          </div>
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>Action a<sub>i,t</sub></div>
            <div style={{ fontSize: 13, color: COLORS.muted }}>A new price for the month, within ±{Math.round(META.price_bound_frac * 100)}% of the entering price. DQN picks one of K={META.k_buckets} buckets; DDPG, PPO and SAC output an exact continuous price. A guard clips any price to [{META.price_clip[0]}×, {META.price_clip[1]}×] of the product's observed range.</div>
          </div>
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>Reward r<sub>i,t</sub> (DRCR)</div>
            <Formula>RCR<sub>t</sub> = revenue<sub>t</sub> / category traffic<sub>t</sub></Formula>
            <Formula>r<sub>t</sub> = {META.reward_scale} × (RCR<sub>t</sub> − RCR<sub>t−{META.tau}</sub>)</Formula>
            <div style={{ fontSize: 13, color: COLORS.muted }}>Improvement in traffic-normalised conversion vs. the previous month — a difference signal, so it hovers near zero even for a good policy.</div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 26 }}>
        <SectionTitle title="Counterfactual demand model (the simulator)" caption="The dataset is logged history — one price per product-month. To score a price the seller never charged, a demand-response model fitted on that history predicts quantity at any candidate price; every agent and the historical baseline are scored through the same model." />
        <div className="card-grid three-grid">
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>Holdout fit (log qty)</div>
            <div style={{ fontFamily: SERIF, fontSize: 22, color: COLORS.gold, lineHeight: 1 }}>R² = {fmtNum(META.demand_model.r2_holdout, 3)}</div>
            <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 6 }}>MAE {fmtNum(META.demand_model.mae_log_holdout, 3)} on each product's last 2 months (time-based holdout)</div>
          </div>
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>Price elasticity (within-product)</div>
            <div style={{ fontFamily: SERIF, fontSize: 22, color: COLORS.gold, lineHeight: 1 }}>{fmtNum(META.demand_model.elasticity_coef, 3)}</div>
            <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 6 }}>∂ log qty / ∂ log price from a fixed-effects log-log baseline; a gradient-boosted residual adds competitor/seasonality interactions, capped so it can only bend the curve, never flip its direction.</div>
          </div>
          <div className="card">
            <div className="eyebrow" style={{ marginBottom: 8 }}>Guard rails</div>
            <div style={{ fontSize: 13, color: COLORS.muted }}>Prices clipped to [{META.price_clip[0]}×, {META.price_clip[1]}×] of each product's own history so no agent can exploit the model off-distribution. Learning-agent trajectories mark months where the guard was hit. {META.n_train_products} products train the policies; {META.n_test_products} are held out for every number on this page.</div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 26 }}>
        <SectionTitle title="Baselines" />
        <div style={{ fontSize: 13, color: COLORS.muted, maxWidth: 780 }}>
          {BASELINES.map((b) => <div key={b.key} style={{ marginBottom: 6 }}><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 9999, background: b.color, marginRight: 8 }} /><b style={{ color: COLORS.text }}>{b.label}</b> — {b.key === "static" ? "keeps the entering price unchanged every month." : "uniformly random action every month."}</div>)}
          <div><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 9999, background: META.historical_color, marginRight: 8 }} /><b style={{ color: COLORS.text }}>Historical</b> — the seller's actual month-by-month prices, scored through the same demand model (used as the reference for "uplift" on the Simulate tab).</div>
        </div>
      </div>
    </div>
  );
}
