/**
 * Mirrors export_schema/v1 (weather-forecast-audit, the Python side),
 * contract version 1.1.0 (the live scorecard is additive over 1.0.0).
 * Keep in lockstep with export_schema/v1/*.schema.json; a schema bump that
 * changes shape should change these types in the same commit.
 */

export const SCHEMA_VERSION = "1.1.0";

export type Variable = "max" | "min";
export type Season = "DJF" | "MAM" | "JJA" | "SON";

export interface Manifest {
  schema_version: string;
  generated_at: string;
  data_through: string | null;
  station_count: number;
  files: string[];
}

export interface CitySummary {
  text: string;
  significant: boolean;
  variable?: Variable;
  lead_day?: number;
  season?: Season;
  bias_f?: number;
}

export interface CityIndexEntry {
  icao: string;
  label: string;
  lat: number;
  lon: number;
  climate_region: string;
  summary: CitySummary;
}

export interface CityIndex {
  schema_version: string;
  cities: CityIndexEntry[];
}

export interface CityStat {
  variable: Variable;
  lead_day: number;
  season: Season;
  n: number;
  n_dates: number;
  bias_f: number | null;
  bias_lo_f: number | null;
  bias_hi_f: number | null;
  mae_f: number | null;
  min_sample_flag: boolean;
  no_detectable_bias: boolean;
}

export interface TypicalMiss {
  max: number | null;
  min: number | null;
}

export interface City {
  schema_version: string;
  icao: string;
  label: string;
  stats: CityStat[];
  typical_miss: TypicalMiss;
}

export interface Completeness {
  schema_version: string;
  stations: unknown[];
}

export type ScorecardSource = "raw_nbm" | "baseline" | "challenger";

/** One variable x lead_day x source slice over the live window. */
export interface ScorecardStat {
  variable: Variable;
  lead_day: number;
  source: ScorecardSource;
  n: number;
  n_dates: number;
  n_blocks: number;
  mae_f: number;
  mae_lo_f: number | null;
  mae_hi_f: number | null;
  bias_f: number;
  bias_lo_f: number | null;
  bias_hi_f: number | null;
  /** 1 - MAE / MAE_raw on matched pairs; exactly 0 for `raw_nbm` itself. */
  skill: number | null;
  skill_lo: number | null;
  skill_hi: number | null;
  /** True when the window has too few bootstrap blocks for a significance call. */
  min_sample_flag: boolean;
  no_detectable_bias: boolean;
}

export interface ScorecardWindow {
  days: number;
  basis: "target_date";
  start: string;
  end: string;
}

export interface ScorecardIntervals {
  method: string;
  block_days: number;
  n_boot: number;
  level: number;
  min_sample_blocks: number;
  note: string;
}

export interface ScorecardLog {
  partitions: number;
  rows: number;
  first_run_date: string | null;
  last_run_date: string | null;
  late_rejected_total: number;
  head_chain_sha256: string;
  verified: boolean;
}

export interface Scorecard {
  schema_version: string;
  window: ScorecardWindow | null;
  intervals: ScorecardIntervals;
  log: ScorecardLog;
  national: ScorecardStat[];
}

export interface ScorecardCity {
  schema_version: string;
  icao: string;
  label: string;
  window: ScorecardWindow | null;
  stats: ScorecardStat[];
}
