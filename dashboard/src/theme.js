export const COLORS = {
  bg: "#12161B",
  panel: "#181D24",
  panelAlt: "#1E242C",
  border: "#2B323B",
  text: "#EDEAE1",
  muted: "#8C97A3",
  faint: "#5B6570",
  gold: "#E1A339",
  slate: "#6E8FB0",
  pos: "#5FA87A",
  neg: "#C4685A",
};

export const MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
export const SERIF = "'Fraunces', Georgia, 'Times New Roman', serif";
export const SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

export const GLOBAL_CSS = `
  * { box-sizing: border-box; }
  html, body, #root { margin: 0; min-height: 100%; background: ${COLORS.bg}; }
  body { color: ${COLORS.text}; font-family: ${SANS}; font-size: 14px; line-height: 1.45; }
  input[type="range"] { height: 4px; accent-color: ${COLORS.gold}; width: 100%; }
  select { background: ${COLORS.panelAlt}; color: ${COLORS.text}; border: 1px solid ${COLORS.border}; padding: 6px 8px; font-size: 13px; font-family: ${SANS}; }
  button { font-family: ${SANS}; }
  .eyebrow { font-family: ${MONO}; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; color: ${COLORS.faint}; }
  .serif { font-family: ${SERIF}; font-weight: 500; }
  .mono { font-family: ${MONO}; }
  .muted { color: ${COLORS.muted}; }
  .card-grid { display: grid; border: 1px solid ${COLORS.border}; }
  .card { padding: 14px 18px; border-right: 1px solid ${COLORS.border}; }
  .card:last-child { border-right: none; }
  .watchlist-row:hover { background: ${COLORS.panelAlt} !important; }
  .tab-btn:focus-visible, .watchlist-row:focus-visible, .seg-btn:focus-visible { outline: 1px solid ${COLORS.gold}; outline-offset: -1px; }
  h1 { overflow-wrap: anywhere; }
  .two-col { display: grid; grid-template-columns: 220px 1fr; }
  .two-col > *, .half-grid > *, .card-grid > * { min-width: 0; }
  .half-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .four-grid { grid-template-columns: repeat(4, 1fr); }
  .three-grid { grid-template-columns: repeat(3, 1fr); }
  .table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .table th { text-align: left; font-family: ${MONO}; font-size: 10.5px; letter-spacing: 0.4px; text-transform: uppercase; color: ${COLORS.faint}; padding: 8px 10px; border-bottom: 1px solid ${COLORS.border}; }
  .table td { padding: 9px 10px; border-bottom: 1px solid ${COLORS.border}; vertical-align: top; }
  .table tr:last-child td { border-bottom: none; }
  @media (max-width: 760px) {
    .two-col { grid-template-columns: 1fr !important; }
    .half-grid { grid-template-columns: 1fr !important; }
    .four-grid { grid-template-columns: 1fr 1fr !important; }
    .three-grid { grid-template-columns: 1fr !important; }
    .card { border-right: none !important; border-bottom: 1px solid ${COLORS.border}; }
    .card:last-child { border-bottom: none; }
    .table { font-size: 12px; }
  }
`;

export function fmtMoney(v) {
  const sign = v < 0 ? "-" : "";
  const av = Math.abs(v);
  if (av >= 1_000_000) return sign + "$" + (av / 1_000_000).toFixed(2) + "M";
  if (av >= 1_000) return sign + "$" + (av / 1_000).toFixed(1) + "K";
  return sign + "$" + av.toFixed(0);
}

export function fmtPct(v, digits = 1) {
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(digits) + "%";
}

export function fmtNum(v, digits = 2) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "–";
}
