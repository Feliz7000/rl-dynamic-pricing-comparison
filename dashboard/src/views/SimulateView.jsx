import React, { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Brush, BarChart, Bar, ReferenceLine, Cell } from "recharts";
import { ALGOS, ALL_POLICIES, HISTORICAL, PRODUCTS, agentSeries, cumulative, productSummary, policyMeta, sum } from "../data.js";
import { COLORS, MONO, SERIF, fmtMoney, fmtPct, fmtNum } from "../theme.js";
import { CustomTooltip, PolicyToggles, StatBlock, Watchlist, SectionTitle, axisProps } from "../components/ui.jsx";

const STATIC = ALL_POLICIES.find((p) => p.key === "static");
const CHART_POLICIES = [STATIC, ...ALGOS];
const PLAY_MS = 750;

export default function SimulateView({ seed }) {
  const summaries = useMemo(() => PRODUCTS.map((p) => productSummary(p, seed)), [seed]);
  // open on the product where RL pricing helps most (top of the ranked list)
  const [selectedId, setSelectedId] = useState(() => {
    const initial = PRODUCTS.map((p) => productSummary(p, "mean"));
    return initial.reduce((b, s) => (Math.max(...ALGOS.map((m) => s.policies[m.key].uplift)) > Math.max(...ALGOS.map((m) => b.policies[m.key].uplift)) ? s : b), initial[0]).id;
  });
  const [rankKey, setRankKey] = useState("best");
  const [shown, setShown] = useState(CHART_POLICIES.map((m) => m.key));
  const [cursor, setCursor] = useState(null); // null = show whole window
  const [playing, setPlaying] = useState(false);

  const product = PRODUCTS.find((p) => p.id === selectedId) || PRODUCTS[0];
  const nMonths = product.months.length;

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setCursor((c) => {
        const next = c == null ? 0 : c + 1;
        if (next >= nMonths - 1) {
          setPlaying(false);
          return nMonths - 1;
        }
        return next;
      });
    }, PLAY_MS);
    return () => clearInterval(id);
  }, [playing, nMonths]);

  useEffect(() => { setCursor(null); setPlaying(false); }, [selectedId]);

  const series = useMemo(() => {
    const per = {};
    for (const p of ALL_POLICIES) per[p.key] = agentSeries(product, p.key, seed);
    const histCum = cumulative(product.historical.revenue_model);
    const cums = {};
    for (const p of ALL_POLICIES) cums[p.key] = cumulative(per[p.key].revenue);
    return product.months.map((m, i) => {
      const row = { month: m, historical: product.historical.price[i], historical_cum: histCum[i] };
      for (const p of ALL_POLICIES) {
        row[p.key] = per[p.key].price[i];
        row[`${p.key}_cum`] = cums[p.key][i];
        row[`${p.key}_gain`] = cums[p.key][i] - histCum[i];
        row[`${p.key}_clipped`] = per[p.key].clipped[i];
      }
      return row;
    });
  }, [product, seed]);

  const upTo = cursor == null ? nMonths - 1 : cursor;
  const summaryNow = useMemo(() => productSummary(product, seed, upTo), [product, seed, upTo]);
  const raceSeries = cursor == null ? series : series.slice(0, cursor + 1);
  const leader = ALGOS.reduce((b, m) => (summaryNow.policies[m.key].uplift > summaryNow.policies[b.key].uplift ? m : b), ALGOS[0]);

  // aggregates across every held-out product
  const aggHist = sum(summaries.map((s) => s.histRev));
  const aggByPolicy = ALL_POLICIES.map((p) => {
    const rev = sum(summaries.map((s) => s.policies[p.key].rev));
    return { key: p.key, label: p.label, color: p.color, rev, uplift: ((rev - aggHist) / aggHist) * 100 };
  });
  const bestAgg = aggByPolicy.filter((a) => ALGOS.some((m) => m.key === a.key)).reduce((b, a) => (a.uplift > b.uplift ? a : b));
  const revenueByProduct = summaries.map((s) => {
    const row = { name: s.id.replace(/^P0*/, "P"), Historical: Math.round(s.histRev) };
    ALGOS.forEach((m) => { row[m.label] = Math.round(s.policies[m.key].rev); });
    return row;
  });
  const upliftChart = aggByPolicy.map((a) => ({ name: a.label, uplift: Number(a.uplift.toFixed(2)), fill: a.color }));

  const clippedDot = (key) => (props) => {
    const { cx, cy, payload } = props;
    if (!payload[`${key}_clipped`]) return null;
    return <circle cx={cx} cy={cy} r={3.5} fill={COLORS.bg} stroke={policyMeta(key).color} strokeWidth={1.5} />;
  };

  return (
    <div className="two-col">
      <div style={{ borderRight: `1px solid ${COLORS.border}`, paddingTop: 18 }}>
        <div style={{ padding: "0 14px 12px" }}>
          <label className="eyebrow">Rank list by</label>
          <select value={rankKey} onChange={(e) => setRankKey(e.target.value)} style={{ display: "block", marginTop: 6, width: "100%" }}>
            <option value="best">Best of 4</option>
            {ALGOS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
        <Watchlist summaries={summaries} selected={product.id} onSelect={setSelectedId} rankKey={rankKey} algos={ALGOS} />
        <div style={{ padding: "18px 14px", marginTop: 8, borderTop: `1px solid ${COLORS.border}` }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Best overall</div>
          <div style={{ fontFamily: SERIF, fontSize: 20, color: bestAgg.color }}>{bestAgg.label}</div>
          <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 4 }}>{fmtPct(bestAgg.uplift, 2)} simulated revenue vs. the seller's own prices, all {PRODUCTS.length} held-out products</div>
        </div>
      </div>

      <div style={{ padding: "18px 24px 28px", minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4, flexWrap: "wrap", gap: 8 }}>
          <h2 style={{ fontFamily: SERIF, fontSize: 22, color: COLORS.text, margin: 0, fontWeight: 500 }}>
            {product.id} · {product.category.replace(/_/g, " ")} — price trajectory
          </h2>
          <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.muted }}>{product.months[0]} → {product.months[nMonths - 1]} · {nMonths} months</span>
        </div>
        <p style={{ color: COLORS.muted, fontSize: 13, margin: "4px 0 14px", maxWidth: 720 }}>
          The dashed line is what the seller actually charged. Each agent re-prices every month from its own state — it never sees the seller's price. Hollow dots mark months where an agent hit the price guard.
        </p>

        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", marginBottom: 12 }}>
          <button onClick={() => { if (cursor != null && cursor >= nMonths - 1) setCursor(null); setPlaying((p) => !p); }} style={{ padding: "8px 18px", fontSize: 13, border: "none", cursor: "pointer", background: COLORS.gold, color: "#161A1F", fontWeight: 600 }}>
            {playing ? "❚❚ Pause" : cursor == null ? "▶ Play month by month" : cursor >= nMonths - 1 ? "↻ Replay" : "▶ Resume"}
          </button>
          <button onClick={() => { setPlaying(false); setCursor((c) => Math.min((c == null ? -1 : c) + 1, nMonths - 1)); }} style={{ padding: "8px 12px", fontSize: 13, border: `1px solid ${COLORS.border}`, cursor: "pointer", background: "transparent", color: COLORS.muted }}>Step ›</button>
          <button onClick={() => { setPlaying(false); setCursor(null); }} style={{ padding: "8px 12px", fontSize: 13, border: `1px solid ${COLORS.border}`, cursor: "pointer", background: "transparent", color: COLORS.muted }}>Show all</button>
          <span style={{ fontFamily: MONO, fontSize: 12, color: cursor == null ? COLORS.faint : COLORS.gold }}>
            {cursor == null ? "full window" : `month ${cursor + 1} / ${nMonths} · ${product.months[cursor]}`}
          </span>
        </div>

        <PolicyToggles policies={CHART_POLICIES} shown={shown} setShown={setShown} />

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={series} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
            <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="month" {...axisProps} />
            <YAxis {...axisProps} width={56} tickFormatter={(v) => "$" + v.toFixed(0)} />
            <Tooltip content={<CustomTooltip formatter={(v) => "$" + v.toFixed(2)} />} />
            {cursor != null && <ReferenceLine x={product.months[cursor]} stroke={COLORS.gold} strokeDasharray="3 3" />}
            <Line type="monotone" dataKey="historical" name={HISTORICAL.label} stroke={HISTORICAL.color} strokeDasharray="4 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            {CHART_POLICIES.filter((m) => shown.includes(m.key)).map((m) => (
              <Line key={m.key} type="monotone" dataKey={m.key} name={`${m.label} price`} stroke={m.color} dot={m.key === "static" ? false : clippedDot(m.key)} activeDot={{ r: 4 }} strokeWidth={m.key === "static" ? 1.5 : 2.25} isAnimationActive={false} />
            ))}
            <Brush dataKey="month" height={22} stroke={COLORS.border} fill={COLORS.panelAlt} travellerWidth={8} />
          </LineChart>
        </ResponsiveContainer>

        <div className="card-grid four-grid" style={{ marginTop: 20 }}>
          {ALGOS.map((m) => {
            const s = summaryNow.policies[m.key];
            return (
              <StatBlock key={m.key} dot={m.color} label={m.label} badge={m.key === leader.key ? (cursor == null ? "BEST" : "LEADING") : null} value={fmtMoney(s.rev)} accent={m.color}
                sub={<span><span style={{ color: s.uplift >= 0 ? COLORS.pos : COLORS.neg }}>{fmtPct(s.uplift)}</span> vs. seller · DRCR {fmtNum(s.drcr, 3)}{s.clippedMonths ? ` · guard ×${s.clippedMonths}` : ""}</span>} />
            );
          })}
        </div>

        <div style={{ marginTop: 30 }}>
          <SectionTitle title="Cumulative revenue gained vs. the seller — the race" caption={`Each policy's simulated revenue minus what the seller's own prices earn through the same model, accumulated month by month${cursor == null ? "" : " up to the current month"}. Above zero is beating the seller. Press Play to watch it unfold.`} />
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={raceSeries} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="month" {...axisProps} />
              <YAxis {...axisProps} width={56} tickFormatter={(v) => fmtMoney(v)} domain={["auto", "auto"]} />
              <Tooltip content={<CustomTooltip formatter={(v) => (v >= 0 ? "+" : "") + fmtMoney(v)} />} />
              <ReferenceLine y={0} stroke={HISTORICAL.color} strokeDasharray="4 4" label={{ value: "seller", fill: COLORS.faint, fontSize: 11, position: "insideTopLeft" }} />
              {CHART_POLICIES.filter((m) => shown.includes(m.key)).map((m) => (
                <Line key={m.key} type="monotone" dataKey={`${m.key}_gain`} name={m.label} stroke={m.color} dot={false} strokeWidth={m.key === "static" ? 1.5 : 2.25} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ marginTop: 34 }}>
          <SectionTitle title="Simulated revenue across all held-out products" caption="Same product, same months, same competitor prices and traffic for every policy — only the price decision differs." />
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={revenueByProduct} margin={{ top: 4, right: 8, left: -12, bottom: 4 }} barGap={2}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: COLORS.muted, fontSize: 10 }} />
              <YAxis {...axisProps} width={56} tickFormatter={(v) => fmtMoney(v)} />
              <Tooltip content={<CustomTooltip formatter={(v) => fmtMoney(v)} />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              <Legend wrapperStyle={{ fontSize: 12, color: COLORS.muted }} />
              <Bar dataKey="Historical" fill={HISTORICAL.color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
              {ALGOS.map((m) => <Bar key={m.key} dataKey={m.label} fill={m.color} radius={[2, 2, 0, 0]} isAnimationActive={false} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ marginTop: 34 }}>
          <SectionTitle title="Aggregate revenue uplift by policy" caption="The headline comparison: each policy's total simulated revenue vs. the seller's own prices, summed across all held-out products." />
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={upliftChart} layout="vertical" margin={{ top: 4, right: 24, left: 16, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" horizontal={false} />
              <XAxis type="number" {...axisProps} tickFormatter={(v) => v + "%"} />
              <YAxis type="category" dataKey="name" {...axisProps} tick={{ fill: COLORS.text, fontSize: 12.5 }} width={120} />
              <Tooltip content={<CustomTooltip formatter={(v) => fmtPct(v, 2)} />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              <ReferenceLine x={0} stroke={COLORS.faint} />
              <Bar dataKey="uplift" name="revenue uplift" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                {upliftChart.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
