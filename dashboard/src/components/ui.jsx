import React, { useMemo } from "react";
import { COLORS, MONO, SERIF, fmtPct } from "../theme.js";

export function CustomTooltip({ active, payload, label, formatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{ background: COLORS.panelAlt, border: `1px solid ${COLORS.border}`, padding: "10px 14px", fontFamily: MONO, fontSize: 12, color: COLORS.text, borderRadius: 4 }}>
      <div style={{ color: COLORS.muted, marginBottom: 6 }}>{label}</div>
      {payload.filter((p) => p.value != null && !(Array.isArray(p.value))).map((p) => (
        <div key={p.dataKey + p.name} style={{ color: p.color || COLORS.text, display: "flex", justifyContent: "space-between", gap: 16 }}>
          <span>{p.name}</span>
          <span>{formatter ? formatter(p.value, p.dataKey) : typeof p.value === "number" ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

export function StatBlock({ label, value, sub, accent, badge, dot }) {
  return (
    <div className="card">
      <div className="eyebrow" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        {dot && <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: 9999, background: dot }} />}
        {label} {badge && <span style={{ color: COLORS.gold }}>· {badge}</span>}
      </div>
      <div style={{ fontFamily: SERIF, fontSize: 22, color: accent || COLORS.text, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

export function SliderControl({ label, value, min, max, step, onChange, display, disabled }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 200, flex: 1 }}>
      <div className="eyebrow" style={{ display: "flex", justifyContent: "space-between" }}>
        <span>{label}</span>
        <span style={{ color: COLORS.gold, textTransform: "none" }}>{display ?? value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} style={{ cursor: disabled ? "default" : "pointer" }} />
    </div>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      {label && <span className="eyebrow">{label}</span>}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 2, background: COLORS.panel, padding: 3, border: `1px solid ${COLORS.border}` }}>
        {options.map((o) => {
          const opt = typeof o === "string" ? { value: o, label: o } : o;
          const active = opt.value === value;
          return (
            <button key={opt.value} className="seg-btn" onClick={() => onChange(opt.value)} style={{ padding: "5px 12px", fontSize: 12.5, border: "none", cursor: "pointer", background: active ? COLORS.gold : "transparent", color: active ? "#161A1F" : COLORS.muted, fontWeight: 500 }}>
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function PolicyToggles({ policies, shown, setShown, label = "Show on chart" }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", marginBottom: 14 }}>
      <span className="eyebrow">{label}</span>
      {policies.map((m) => {
        const active = shown.includes(m.key);
        return (
          <label key={m.key} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 13, color: active ? COLORS.text : COLORS.faint }}>
            <input type="checkbox" checked={active} onChange={() => setShown((prev) => (active ? prev.filter((k) => k !== m.key) : [...prev, m.key]))} style={{ accentColor: m.color }} />
            <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 9999, background: m.color }} />
            {m.label}
          </label>
        );
      })}
    </div>
  );
}

export function Watchlist({ summaries, selected, onSelect, rankKey, algos }) {
  const sorted = useMemo(() => {
    const withMetric = summaries.map((s) => ({
      ...s,
      metric: rankKey === "best" ? Math.max(...algos.map((m) => s.policies[m.key].uplift)) : s.policies[rankKey].uplift,
    }));
    return withMetric.sort((a, b) => b.metric - a.metric);
  }, [summaries, rankKey, algos]);

  return (
    <div>
      <div className="eyebrow" style={{ padding: "0 14px 10px" }}>Held-out products · uplift vs. seller</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {sorted.map((s) => {
          const active = s.id === selected;
          const positive = s.metric >= 0;
          return (
            <button key={s.id} onClick={() => onSelect(s.id)} className="watchlist-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: active ? COLORS.panelAlt : "transparent", border: "none", borderLeft: active ? `2px solid ${COLORS.gold}` : "2px solid transparent", cursor: "pointer", textAlign: "left", width: "100%" }}>
              <span style={{ display: "flex", flexDirection: "column" }}>
                <span style={{ color: active ? COLORS.text : COLORS.muted, fontSize: 13.5 }}>{s.id}</span>
                <span style={{ color: COLORS.faint, fontSize: 11 }}>{s.category.replace(/_/g, " ")}</span>
              </span>
              <span style={{ fontFamily: MONO, fontSize: 13, color: positive ? COLORS.pos : COLORS.neg }}>{fmtPct(s.metric)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function SectionTitle({ title, caption, right }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <h3 style={{ fontFamily: SERIF, fontSize: 17, color: COLORS.text, fontWeight: 500, margin: 0 }}>{title}</h3>
        {right && <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.muted }}>{right}</span>}
      </div>
      {caption && <p style={{ color: COLORS.muted, fontSize: 13, margin: "4px 0 0", maxWidth: 720 }}>{caption}</p>}
    </div>
  );
}

export const axisProps = {
  stroke: COLORS.faint,
  tick: { fill: COLORS.muted, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: COLORS.border },
};
