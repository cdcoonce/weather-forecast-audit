import type { CityStat, Season } from "../types/export";

const SEASON_WORDS: Record<Season, string> = {
  DJF: "winter",
  MAM: "spring",
  JJA: "summer",
  SON: "fall",
};

export function seasonWord(season: Season): string {
  return SEASON_WORDS[season];
}

const MINUS = "−"; // proper minus sign, not a hyphen

function formatSignedF(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? MINUS : "";
  return `${sign}${Math.abs(value).toFixed(1)}`;
}

function halfCi(stat: CityStat): number | null {
  if (stat.bias_lo_f == null || stat.bias_hi_f == null) return null;
  return (stat.bias_hi_f - stat.bias_lo_f) / 2;
}

/**
 * One table cell's text for a variable x lead x season stat.
 *
 * A bias number or direction is claimed only when both `min_sample_flag`
 * and `no_detectable_bias` are false (the build spec's acceptance rule) --
 * the other two branches never contain a signed number.
 */
export function formatBiasCell(stat: CityStat): string {
  if (stat.min_sample_flag) {
    return `too few days (n=${stat.n})`;
  }
  if (stat.no_detectable_bias) {
    return `no detectable bias (n=${stat.n})`;
  }
  if (stat.bias_f == null) {
    return `– (n=${stat.n})`;
  }
  const signed = formatSignedF(stat.bias_f);
  const half = halfCi(stat);
  const withCi = half == null ? signed : `${signed} ± ${half.toFixed(1)}`;
  return `${withCi}°F (n=${stat.n})`;
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
