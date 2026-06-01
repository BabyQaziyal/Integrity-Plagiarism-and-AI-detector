// Shared theme helpers (watermelon / lemon chiffon).
export const COLORS = {
  watermelon: "#f0485f",
  lemon: "#fdf7c3",
  ink: "#2b2b2b",
  muted: "#7a7a7a",
  good: "#3aa76d",
  warn: "#e8a13a",
};

export function riskColor(integrity) {
  if (integrity == null) return COLORS.muted;
  if (integrity >= 85) return COLORS.good;
  if (integrity >= 60) return COLORS.warn;
  return COLORS.watermelon;
}

export function verdictTone(verdict) {
  if (verdict === "Low risk") return { bg: "#e9f7ef", fg: COLORS.good };
  if (verdict === "Review recommended") return { bg: "#fdf3e2", fg: COLORS.warn };
  return { bg: COLORS.watermelon + "1a", fg: COLORS.watermelon };
}

// translucent watermelon by similarity score (0..1) for text highlighting
export function highlightColor(score) {
  const a = Math.min(0.55, 0.18 + score * 0.5);
  return `rgba(240,72,95,${a.toFixed(3)})`;
}
