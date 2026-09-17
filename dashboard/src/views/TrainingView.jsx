import React, { useMemo, useState } from "react";
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts";
import { ALGOS, META, SUMMARY, TRAINING } from "../data.js";
import { COLORS, fmtNum } from "../theme.js";
import { CustomTooltip, PolicyToggles, StatBlock, SectionTitle, axisProps } from "../components/ui.jsx";

export default function TrainingView({ seed }) {
  const algos = ALGOS.filter((m) => TRAINING[m.key]);
  const [shown, setShown] = useState(algos.map((m) => m.key));
  const hasLoss = algos.some((m) => TRAINING[m.key].loss);

  const rewardRows = useMemo(() => {
    const ref = TRAINING[algos[0].key];
    return ref.step.map((step, i) => {
      const row = { step: Math.round(step) };
      for (const m of algos) {
        const t = TRAINING[m.key];
        if (seed === "mean") {
          row[m.key] = t.mean[i];
          row[`${m.key}_band`] = [t.mean[i] - t.std[i], t.mean[i] + t.std[i]];
        } else {
          const s = t.seeds[seed] || t.seeds[Object.keys(t.seeds)[0]];
          row[m.key] = s.reward_rolling[i];
        }
      }
      return row;
    });
  }, [algos, seed]);

  const secondRows = useMemo(() => {
    const ref = TRAINING[algos[0].key];
    return ref.step.map((step, i) => {
      const row = { step: Math.round(step) };
      for (const m of algos) {
        const t = TRAINING[m.key];
        if (hasLoss) row[m.key] = t.loss ? t.loss.mean[i] : null;
        else if (seed === "mean") {
          row[m.key] = t.running_mean[i];
          row[`${m.key}_band`] = [t.running_mean[i] - t.running_std[i], t.running_mean[i] + t.running_std[i]];
        } else row[m.key] = (t.seeds[seed] || t.seeds[Object.keys(t.seeds)[0]]).running_mean[i];
      }
      return row;
    });
  }, [algos, seed, hasLoss]);

  const visible = algos.filter((m) => shown.includes(m.key));

  // y-range from the settled part of training (skip the first 5% of steps,
  // whose transient spikes would otherwise flatten everything else)
  const yDomain = (rows) => {
    const start = Math.floor(rows.length * 0.05);
    let m = 0;
    for (const r of rows.slice(start)) for (const k of visible) if (Number.isFinite(r[k.key])) m = Math.max(m, Math.abs(r[k.key]));
    const raw = m * 1.2 || 1;
    const unit = raw > 5 ? 5 : 1;
    const lim = Math.ceil(raw / unit) * unit;
    return [-lim, lim];
  };

  return (
    <div style={{ padding: "22px 28px 32px" }}>
      <SectionTitle
        title="Training run"
        caption={`${META.seeds.length} seeds × ${META.total_steps?.toLocaleString()} environment steps per algorithm on ${META.n_train_products} training products, identical budget for every policy. Episode reward is a difference signal (DRCR × ${META.reward_scale}), so it hovers near zero even for a good policy — the level it settles at and how tightly it holds there are the comparison.`}
        right={seed === "mean" ? "mean ± std across seeds" : `seed ${seed}`}
      />

      <div className="card-grid four-grid" style={{ marginBottom: 24 }}>
        {algos.map((m) => {
          const s = SUMMARY[m.key] || {};
          return <StatBlock key={m.key} dot={m.color} label={m.label} value={s.mean_final_reward == null ? "—" : fmtNum(s.mean_final_reward)} accent={m.color} sub={s.std_final_reward == null ? "training-tail mean reward" : `± ${fmtNum(s.std_final_reward)} across seeds · mean reward, last 20% of steps`} />;
        })}
      </div>

      <PolicyToggles policies={algos} shown={shown} setShown={setShown} />

      <div className="half-grid">
        <div>
          <div className="eyebrow" style={{ marginBottom: 10 }}>Reward (rolling mean, window 200 steps)</div>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={rewardRows} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="step" {...axisProps} tickFormatter={(v) => (v / 1000).toFixed(0) + "k"} />
              <YAxis {...axisProps} width={44} domain={yDomain(rewardRows)} allowDataOverflow />
              <ReferenceLine y={0} stroke={COLORS.faint} strokeDasharray="2 3" />
              <Tooltip content={<CustomTooltip />} labelFormatter={(v) => `step ${v.toLocaleString()}`} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.muted }} />
              {seed === "mean" && visible.map((m) => (
                <Area key={`${m.key}_band`} dataKey={`${m.key}_band`} name={`${m.label} ±σ`} stroke="none" fill={m.color} fillOpacity={0.12} isAnimationActive={false} legendType="none" />
              ))}
              {visible.map((m) => (
                <Line key={m.key} type="monotone" dataKey={m.key} name={m.label} stroke={m.color} dot={false} strokeWidth={2} isAnimationActive={false} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
          <p style={{ color: COLORS.faint, fontSize: 12, margin: "8px 0 0" }}>Per-step reward smoothed over 200 steps. Shocks from individual products dominate the short-term picture — see the running mean for the trend.</p>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 10 }}>{hasLoss ? "Critic / value loss (rolling mean)" : "Running mean reward since the start of training"}</div>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={secondRows} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="step" {...axisProps} tickFormatter={(v) => (v / 1000).toFixed(0) + "k"} />
              <YAxis {...axisProps} width={44} domain={hasLoss ? ["auto", "auto"] : yDomain(secondRows)} allowDataOverflow={!hasLoss} />
              {!hasLoss && <ReferenceLine y={0} stroke={COLORS.faint} strokeDasharray="2 3" />}
              <Tooltip content={<CustomTooltip />} labelFormatter={(v) => `step ${v.toLocaleString()}`} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.muted }} />
              {!hasLoss && seed === "mean" && visible.map((m) => (
                <Area key={`${m.key}_band`} dataKey={`${m.key}_band`} name={`${m.label} ±σ`} stroke="none" fill={m.color} fillOpacity={0.12} isAnimationActive={false} legendType="none" />
              ))}
              {visible.map((m) => (
                <Line key={m.key} type="monotone" dataKey={m.key} name={m.label} stroke={m.color} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
          <p style={{ color: COLORS.faint, fontSize: 12, margin: "8px 0 0" }}>
            {hasLoss ? "Rolling mean of each agent's critic / value loss." : "Cumulative average of all rewards so far — the noise averages out and the level each algorithm converges to becomes visible. Loss curves replace this panel automatically once a training run has logged them."}
          </p>
        </div>
      </div>

      <p style={{ color: COLORS.muted, fontSize: 12.5, marginTop: 18, maxWidth: 800 }}>
        Training episodes sample a random training product and roll it forward with the demand model; rewards are noisy by construction because every product-month carries its own traffic and competitor shocks. Off-policy agents (DQN, DDPG, SAC) learn from a replay buffer every step; PPO updates only after each fresh 2048-step rollout, which is why it needs a larger budget to catch up.
      </p>
    </div>
  );
}
