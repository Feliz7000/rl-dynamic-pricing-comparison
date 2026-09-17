import React, { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ReferenceDot, ReferenceArea } from "recharts";
import { ALGOS, HISTORICAL, META, PRODUCTS, agentSeries, interpGrid } from "../data.js";
import { COLORS, MONO, SERIF, fmtMoney, fmtPct, fmtNum } from "../theme.js";
import { CustomTooltip, PolicyToggles, SliderControl, StatBlock, SectionTitle, axisProps } from "../components/ui.jsx";

export default function WhatIfView({ seed }) {
  const [productId, setProductId] = useState(PRODUCTS[0].id);
  const [monthIdx, setMonthIdx] = useState(0);
  const [shown, setShown] = useState(ALGOS.map((m) => m.key));
  const product = PRODUCTS.find((p) => p.id === productId) || PRODUCTS[0];
  const grid = product.price_grid[monthIdx];
  const histPrice = product.historical.price[monthIdx];
  const [lo, hi] = product.price_bounds;
  const [price, setPrice] = useState(histPrice);

  useEffect(() => { setMonthIdx(0); }, [productId]);
  useEffect(() => { setPrice(product.historical.price[monthIdx]); }, [product, monthIdx]);

  const rows = useMemo(() => grid.prices.map((p, i) => ({ price: p, qty: grid.qty[i], revenue: grid.revenue[i], drcr: grid.drcr[i] })), [grid]);
  const at = (p) => ({ qty: interpGrid(grid, "qty", p), revenue: interpGrid(grid, "revenue", p), drcr: interpGrid(grid, "drcr", p) });
  const now = at(price);
  const hist = at(histPrice);
  const agentPrices = ALGOS.map((m) => ({ ...m, price: agentSeries(product, m.key, seed).price[monthIdx] }));
  const revMaxIdx = grid.revenue.indexOf(Math.max(...grid.revenue));
  const bestPrice = grid.prices[revMaxIdx];
  const entering = monthIdx === 0 ? null : product.historical.price[monthIdx - 1];
  const step = (hi - lo) / 200;

  const marker = (key, p, color, y) => (
    <ReferenceDot key={key} x={p} y={y} r={5} fill={color} stroke={COLORS.bg} strokeWidth={1.5} isFront />
  );

  return (
    <div style={{ padding: "22px 28px 32px" }}>
      <div style={{ display: "flex", alignItems: "flex-end", flexWrap: "wrap", gap: 24, marginBottom: 18 }}>
        <div>
          <label className="eyebrow">Product</label>
          <select value={productId} onChange={(e) => setProductId(e.target.value)} style={{ display: "block", marginTop: 6 }}>
            {PRODUCTS.map((p) => <option key={p.id} value={p.id}>{p.id} · {p.category.replace(/_/g, " ")}</option>)}
          </select>
        </div>
        <div>
          <label className="eyebrow">Month</label>
          <select value={monthIdx} onChange={(e) => setMonthIdx(Number(e.target.value))} style={{ display: "block", marginTop: 6 }}>
            {product.months.map((m, i) => <option key={m} value={i}>{m}</option>)}
          </select>
        </div>
        <SliderControl label="Candidate price" value={price} min={lo} max={hi} step={step} onChange={setPrice} display={`$${price.toFixed(2)}`} />
        <button onClick={() => setPrice(histPrice)} style={{ padding: "7px 12px", fontSize: 12.5, border: `1px solid ${COLORS.border}`, cursor: "pointer", background: "transparent", color: COLORS.muted }}>Reset to seller's price</button>
        <button onClick={() => setPrice(bestPrice)} style={{ padding: "7px 12px", fontSize: 12.5, border: `1px solid ${COLORS.border}`, cursor: "pointer", background: "transparent", color: COLORS.muted }}>Jump to revenue peak</button>
      </div>

      <div className="card-grid four-grid" style={{ marginBottom: 22 }}>
        <StatBlock label="Predicted demand" value={`${fmtNum(now.qty, 1)} units`} sub={`seller's price → ${fmtNum(hist.qty, 1)} units`} />
        <StatBlock label="Predicted revenue" value={fmtMoney(now.revenue)} accent={COLORS.gold} sub={<span><span style={{ color: now.revenue >= hist.revenue ? COLORS.pos : COLORS.neg }}>{fmtPct(hist.revenue ? ((now.revenue - hist.revenue) / hist.revenue) * 100 : 0)}</span> vs. seller's price</span>} />
        <StatBlock label="DRCR vs. last month" value={fmtNum(now.drcr, 3)} accent={now.drcr >= 0 ? COLORS.pos : COLORS.neg} sub={`reward = ${META.reward_scale} × DRCR = ${fmtNum(META.reward_scale * now.drcr, 1)}`} />
        <StatBlock label="Context this month" value={`$${fmtNum(histPrice)}`} sub={`seller's price · competitors ${product.competitors.comp_1[monthIdx].toFixed(0)} / ${product.competitors.comp_2[monthIdx].toFixed(0)} / ${product.competitors.comp_3[monthIdx].toFixed(0)}${entering ? ` · entered at $${fmtNum(entering)}` : ""}`} />
      </div>

      <PolicyToggles policies={ALGOS} shown={shown} setShown={setShown} label="Mark agents' chosen prices" />

      <div className="half-grid">
        <div>
          <SectionTitle title="Revenue response" right={`${product.id} · ${product.months[monthIdx]}`} />
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="price" type="number" domain={[lo, hi]} {...axisProps} tickFormatter={(v) => "$" + v.toFixed(0)} />
              <YAxis {...axisProps} width={56} tickFormatter={(v) => fmtMoney(v)} domain={["auto", "auto"]} />
              <Tooltip content={<CustomTooltip formatter={(v) => fmtMoney(v)} />} labelFormatter={(v) => "$" + Number(v).toFixed(2)} />
              <ReferenceArea x1={lo} x2={Math.max(lo, histPrice * 0.75)} fill={COLORS.neg} fillOpacity={0.04} />
              <ReferenceArea x1={Math.min(hi, histPrice * 1.25)} x2={hi} fill={COLORS.neg} fillOpacity={0.04} />
              <Line type="monotone" dataKey="revenue" name="predicted revenue" stroke={COLORS.gold} dot={false} strokeWidth={2.25} isAnimationActive={false} />
              <ReferenceLine x={price} stroke={COLORS.text} strokeDasharray="3 3" />
              {marker("hist", histPrice, HISTORICAL.color, hist.revenue)}
              {agentPrices.filter((a) => shown.includes(a.key)).map((a) => marker(a.key, a.price, a.color, at(a.price).revenue))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div>
          <SectionTitle title="Demand response" right={`elasticity ${fmtNum(META.demand_model.elasticity_coef, 2)}`} />
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
              <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="price" type="number" domain={[lo, hi]} {...axisProps} tickFormatter={(v) => "$" + v.toFixed(0)} />
              <YAxis {...axisProps} width={56} tickFormatter={(v) => v.toFixed(0)} domain={["auto", "auto"]} />
              <Tooltip content={<CustomTooltip formatter={(v) => v.toFixed(1) + " units"} />} labelFormatter={(v) => "$" + Number(v).toFixed(2)} />
              <Line type="monotone" dataKey="qty" name="predicted units" stroke={COLORS.slate} dot={false} strokeWidth={2.25} isAnimationActive={false} />
              <ReferenceLine x={price} stroke={COLORS.text} strokeDasharray="3 3" />
              {marker("hist", histPrice, HISTORICAL.color, hist.qty)}
              {agentPrices.filter((a) => shown.includes(a.key)).map((a) => marker(a.key, a.price, a.color, at(a.price).qty))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 10, fontSize: 12.5, color: COLORS.muted }}>
        <span><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 9999, background: HISTORICAL.color, marginRight: 6 }} />seller's actual price</span>
        {agentPrices.filter((a) => shown.includes(a.key)).map((a) => (
          <span key={a.key}><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 9999, background: a.color, marginRight: 6 }} />{a.label} chose ${a.price.toFixed(2)}</span>
        ))}
        <span className="mono" style={{ color: COLORS.faint }}>shaded: outside the ±{Math.round(META.price_bound_frac * 100)}% per-month move an agent is allowed</span>
      </div>

      <div style={{ marginTop: 28, maxWidth: 820 }}>
        <SectionTitle title="Why a demand model is needed at all" />
        <p style={{ color: COLORS.muted, fontSize: 13, margin: 0 }}>
          The dataset only records what happened at the one price the seller actually charged each month. To score any other price, a demand-response model fitted on the full history — a within-product log-log elasticity baseline (holdout R² {fmtNum(META.demand_model.r2_holdout, 2)}) plus a capped gradient-boosted correction for competitor and seasonal effects — predicts the units that would have sold. The curves above are that model, evaluated at this month's real context (competitor prices, traffic, seasonality) for every candidate price. Agents are trained and scored through exactly this model, and so is the seller's own price path, which is what makes the "uplift vs. seller" comparison fair.
          {" "}Note the agents' markers are the prices they chose from their <em>own</em> rolled-forward state, so their achieved revenue in the Simulate tab can differ slightly from where the marker sits on this curve.
        </p>
      </div>
    </div>
  );
}
