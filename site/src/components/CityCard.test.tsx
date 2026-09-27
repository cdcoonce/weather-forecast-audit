import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { City, CityStat } from "../types/export";
import { CityCard } from "./CityCard";

function stat(overrides: Partial<CityStat>): CityStat {
  return {
    variable: "max",
    lead_day: 1,
    season: "JJA",
    n: 42,
    n_dates: 35,
    bias_f: null,
    bias_lo_f: null,
    bias_hi_f: null,
    mae_f: null,
    min_sample_flag: false,
    no_detectable_bias: true,
    ...overrides,
  };
}

describe("CityCard", () => {
  it("renders a significant summary with its number and direction", () => {
    const city: City = {
      schema_version: "1.0.0",
      icao: "KPHX",
      label: "PHOENIX/SKY HARBOR, AZ",
      stats: [
        stat({
          variable: "max",
          season: "JJA",
          bias_f: -1.4,
          bias_lo_f: -2.0,
          bias_hi_f: -0.8,
          no_detectable_bias: false,
        }),
      ],
    };
    const summary = {
      text: "Highs run 1.4°F cold in summer (day-ahead forecasts).",
      significant: true,
      variable: "max" as const,
      lead_day: 1,
      season: "JJA" as const,
      bias_f: -1.4,
    };

    render(<CityCard city={city} summary={summary} />);

    expect(screen.getByText(/Highs run 1\.4°F cold in summer/)).toBeInTheDocument();
  });

  it('renders "No detectable bias..." when every stat is no_detectable_bias, ' +
    "and no cell shows a signed number", () => {
    const city: City = {
      schema_version: "1.0.0",
      icao: "KBOS",
      label: "BOSTON LOGAN INTL, MA",
      stats: [
        stat({ variable: "max", season: "DJF", bias_f: 0.3, no_detectable_bias: true }),
        stat({ variable: "min", season: "JJA", bias_f: -0.2, no_detectable_bias: true }),
      ],
    };
    const summary = {
      text: "No detectable bias in day-ahead forecasts.",
      significant: false,
    };

    render(<CityCard city={city} summary={summary} />);

    expect(screen.getByText(/No detectable bias in day-ahead forecasts\./)).toBeInTheDocument();

    const cells = screen.getAllByText(/no detectable bias/i);
    expect(cells.length).toBeGreaterThan(0);
    for (const cell of cells) {
      expect(cell.textContent).not.toMatch(/[+−]\d/);
    }
  });

  it('renders "too few days" for a min_sample_flag stat', () => {
    const city: City = {
      schema_version: "1.0.0",
      icao: "KDEN",
      label: "DENVER INTL, CO",
      stats: [
        stat({
          variable: "max",
          season: "MAM",
          min_sample_flag: true,
          no_detectable_bias: true,
          bias_f: 4.0,
        }),
      ],
    };
    const summary = { text: "Not enough data yet.", significant: false };

    render(<CityCard city={city} summary={summary} />);

    const cell = screen.getByText(/too few days/i);
    expect(cell).toBeInTheDocument();
    expect(cell.textContent).not.toMatch(/[+−]\d/);
  });

  it("shows a table footnote about the missing lead-3 max column", () => {
    const city: City = {
      schema_version: "1.0.0",
      icao: "KPHX",
      label: "PHOENIX/SKY HARBOR, AZ",
      stats: [],
    };
    const summary = { text: "Not enough data yet.", significant: false };

    render(<CityCard city={city} summary={summary} />);
    expect(screen.getByText(/lead-3/i)).toBeInTheDocument();
  });

  it("shows the bias-sign explainer line", () => {
    const city: City = {
      schema_version: "1.0.0",
      icao: "KPHX",
      label: "PHOENIX/SKY HARBOR, AZ",
      stats: [],
    };
    const summary = { text: "Not enough data yet.", significant: false };

    render(<CityCard city={city} summary={summary} />);
    expect(
      screen.getByText(/Bias = forecast.*observed/i)
    ).toBeInTheDocument();
  });
});
