/**
 * Mirrors export_schema/v1 (weather-forecast-audit, the Python side).
 * Keep in lockstep with export_schema/v1/*.schema.json; a schema bump that
 * changes shape should change these types in the same commit.
 */

export const SCHEMA_VERSION = "1.0.0";

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

export interface City {
  schema_version: string;
  icao: string;
  label: string;
  stats: CityStat[];
}

export interface Completeness {
  schema_version: string;
  stations: unknown[];
}
