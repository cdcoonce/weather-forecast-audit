import type { CityStat, Season, TypicalMiss } from "../types/export";

const SEASON_WORDS: Record<Season, string> = {
  DJF: "winter",
  MAM: "spring",
  JJA: "summer",
  SON: "fall",
};

export function seasonWord(season: Season): string {
  return SEASON_WORDS[season];
}

const MINUS = "−"; // U+2212, proper minus sign -- never a hyphen (U+002D)
const NO_BIAS_DASH = "–"; // U+2013, en dash -- visually distinct from MINUS

/** A signed one-decimal number using a true minus sign, e.g. "+1.4", "−1.4". */
export function formatSignedF(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? MINUS : "";
  return `${sign}${Math.abs(value).toFixed(1)}`;
}

/**
 * A signed number for one interval endpoint, using the fewest decimals (1,
 * then 2) that keep it visibly non-zero with its sign, e.g. "+0.5", but
 * "+0.03" (which would round to "+0.0" at one decimal -- indistinguishable
 * from "no bias" despite the sign) instead shows "+0.03". Exactly zero
 * stays "0.0"; that can only happen for a not-significant interval, which
 * is never rendered as a number in the first place (see `biasCellContent`).
 */
export function formatSignedEndpoint(value: number): string {
  if (value === 0) return "0.0";
  const sign = value > 0 ? "+" : MINUS;
  const abs = Math.abs(value);
  const oneDecimal = abs.toFixed(1);
  if (oneDecimal !== "0.0") return `${sign}${oneDecimal}`;
  return `${sign}${abs.toFixed(2)}`;
}

/** The bold primary line for a significant cell, e.g. "−1.1°F". Assumes
 * `stat.bias_f` is non-null -- callers only reach this for significant cells. */
export function formatBiasPrimary(stat: CityStat): string {
  return `${formatSignedF(stat.bias_f as number)}°F`;
}

/**
 * The CI as endpoints, e.g. "−1.6 to −0.6" -- never "±", because the
 * bootstrap percentile interval need not be symmetric. Null when either
 * endpoint is missing. Each endpoint uses `formatSignedEndpoint`, not
 * `formatSignedF`, so a small-magnitude endpoint never displays as a
 * sign-contradicting "+0.0"/"−0.0".
 */
export function formatBiasEndpoints(stat: CityStat): string | null {
  if (stat.bias_lo_f == null || stat.bias_hi_f == null) return null;
  return `${formatSignedEndpoint(stat.bias_lo_f)} to ${formatSignedEndpoint(stat.bias_hi_f)}`;
}

export type BiasCellKind = "significant" | "no-detectable-bias" | "few-days";

export interface BiasCellContent {
  kind: BiasCellKind;
  /** Bold primary line text (significant cells only). */
  primary?: string;
  /** Smaller/muted secondary line with the interval endpoints (significant, when available). */
  secondary?: string;
  /** The muted glyph/word shown to sighted users for non-significant cells. */
  visible: string;
  /** The full accessible name for non-significant cells, e.g. "no detectable bias". */
  accessibleName?: string;
}

/**
 * One table cell's content for a variable x lead x season stat.
 *
 * A bias number or direction is claimed only when both `min_sample_flag`
 * and `no_detectable_bias` are false (the build spec's acceptance rule).
 * `min_sample_flag` takes precedence: a too-few-days cell reads "few days"
 * (accessible name "too few days") even if it also happens to be
 * `no_detectable_bias`. A non-significant cell shows a muted en dash with
 * the accessible name "no detectable bias" -- never a bare "–" with no
 * accessible name, so screen readers don't just hear "dash".
 */
export function biasCellContent(stat: CityStat): BiasCellContent {
  if (stat.min_sample_flag) {
    return { kind: "few-days", visible: "few days", accessibleName: "too few days" };
  }
  if (stat.no_detectable_bias || stat.bias_f == null) {
    return {
      kind: "no-detectable-bias",
      visible: NO_BIAS_DASH,
      accessibleName: "no detectable bias",
    };
  }
  const endpoints = formatBiasEndpoints(stat);
  return {
    kind: "significant",
    primary: formatBiasPrimary(stat),
    secondary: endpoints ?? undefined,
    visible: formatBiasPrimary(stat),
  };
}

/** The median of `values`, rounded to the nearest 10 (for a table caption's
 * "each cell ≈ N days"). Returns 0 for an empty list. */
export function medianCellN(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  return Math.round(median / 10) * 10;
}

/**
 * "Typical miss, day-ahead: highs 2.9°F · lows 2.2°F" -- omits a variable
 * that is null, and returns null (the line should be omitted entirely) when
 * both are null.
 */
export function formatTypicalMissLine(typicalMiss: TypicalMiss): string | null {
  const parts: string[] = [];
  if (typicalMiss.max != null) parts.push(`highs ${typicalMiss.max.toFixed(1)}°F`);
  if (typicalMiss.min != null) parts.push(`lows ${typicalMiss.min.toFixed(1)}°F`);
  if (parts.length === 0) return null;
  return `Typical miss, day-ahead: ${parts.join(" · ")}`;
}

/** `"2026-09-27T14:03:00Z"` -> `"2026-09-27 14:03 UTC"`. */
export function formatGeneratedAt(iso: string): string {
  const date = new Date(iso);
  const pad = (value: number) => String(value).padStart(2, "0");
  const y = date.getUTCFullYear();
  const m = pad(date.getUTCMonth() + 1);
  const d = pad(date.getUTCDate());
  const hh = pad(date.getUTCHours());
  const mm = pad(date.getUTCMinutes());
  return `${y}-${m}-${d} ${hh}:${mm} UTC`;
}
