import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { City, CityStat, TypicalMiss } from "../types/export";
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

const NO_TYPICAL_MISS: TypicalMiss = { max: null, min: null };

function city(overrides: Partial<City>): City {
  return {
    schema_version: "1.0.0",
    icao: "KPHX",
    label: "Phoenix/Sky Harbor, AZ",
    stats: [],
    typical_miss: NO_TYPICAL_MISS,
    ...overrides,
  };
}

describe("CityCard", () => {
  it("renders a significant summary with its number and direction", () => {
    render(
      <CityCard
        city={city({
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
        })}
        summary={{
          text: "Highs run 1.4°F cold in summer (day-ahead forecasts).",
          significant: true,
          variable: "max",
          lead_day: 1,
          season: "JJA",
          bias_f: -1.4,
        }}
      />
    );

    expect(screen.getByText(/Highs run 1\.4°F cold in summer/)).toBeInTheDocument();
  });

  it("renders a significant cell as a bold '−1.1°F' primary line with the " +
    "interval endpoints beneath it, and never '±'", () => {
    render(
      <CityCard
        city={city({
          stats: [
            stat({
              variable: "max",
              season: "JJA",
              bias_f: -1.1,
              bias_lo_f: -1.6,
              bias_hi_f: -0.6,
              no_detectable_bias: false,
            }),
          ],
        })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );

    const primary = document.querySelector(".city-card__cell-primary");
    const secondary = document.querySelector(".city-card__cell-secondary");
    expect(primary?.textContent).toBe("−1.1°F");
    expect(secondary?.textContent).toBe("−1.6 to −0.6");
    expect(document.body.textContent).not.toContain("±");
  });

  it('renders "No detectable bias..." when every stat is no_detectable_bias, ' +
    "and no cell shows a signed number", () => {
    render(
      <CityCard
        city={city({
          icao: "KBOS",
          label: "Boston Logan Intl, MA",
          stats: [
            stat({ variable: "max", season: "DJF", bias_f: 0.3, no_detectable_bias: true }),
            stat({ variable: "min", season: "JJA", bias_f: -0.2, no_detectable_bias: true }),
          ],
        })}
        summary={{ text: "No detectable bias in day-ahead forecasts.", significant: false }}
      />
    );

    expect(screen.getByText(/No detectable bias in day-ahead forecasts\./)).toBeInTheDocument();

    // Scoped to table cells (<td>), not screen-wide text: the summary <p>
    // above also says "No detectable bias..." (it's a fixed prop in this
    // test), and a text-wide query would match that paragraph and never
    // actually check a single table cell's rendered content.
    const tableCells = Array.from(document.querySelectorAll("td"));
    expect(tableCells.length).toBeGreaterThan(0);
    const noBiasCells = tableCells.filter((cell) =>
      /no detectable bias/i.test(cell.textContent ?? "")
    );
    expect(noBiasCells.length).toBeGreaterThan(0);
    for (const cell of tableCells) {
      expect(cell.textContent).not.toMatch(/[+−]\d/);
    }
  });

  it("a non-significant cell shows a muted dash visibly, with 'no detectable " +
    "bias' as a visually-hidden accessible name (not visible digits)", () => {
    render(
      <CityCard
        city={city({
          stats: [stat({ variable: "max", season: "JJA", no_detectable_bias: true })],
        })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );

    const visible = document.querySelector('td[aria-hidden], td [aria-hidden="true"]');
    expect(visible?.textContent).toBe("–");
    const hiddenName = document.querySelector(".visually-hidden");
    expect(hiddenName?.textContent).toBe("no detectable bias");
  });

  it('renders "few days" visibly for a min_sample_flag stat, with "too few ' +
    'days" as the accessible name', () => {
    render(
      <CityCard
        city={city({
          icao: "KDEN",
          label: "Denver Intl, CO",
          stats: [
            stat({
              variable: "max",
              season: "MAM",
              min_sample_flag: true,
              no_detectable_bias: true,
              bias_f: 4.0,
            }),
          ],
        })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );

    const visible = screen.getByText("few days");
    expect(visible).toBeInTheDocument();
    expect(visible.textContent).not.toMatch(/\d/);
    const hiddenName = screen.getByText("too few days");
    expect(hiddenName).toBeInTheDocument();
  });

  it("shows a table footnote about the missing lead-3 max column", () => {
    render(
      <CityCard city={city({})} summary={{ text: "Not enough data yet.", significant: false }} />
    );
    expect(screen.getByText(/lead-3/i)).toBeInTheDocument();
  });

  it("shows the bias-sign explainer line", () => {
    render(
      <CityCard city={city({})} summary={{ text: "Not enough data yet.", significant: false }} />
    );
    expect(screen.getByText(/Bias = forecast.*observed/i)).toBeInTheDocument();
  });

  it("shows a caption under each table naming the dash and the median day count", () => {
    render(
      <CityCard
        city={city({
          stats: [
            stat({ variable: "max", season: "DJF", n_dates: 178, no_detectable_bias: true }),
            stat({ variable: "max", season: "JJA", n_dates: 183, no_detectable_bias: true }),
          ],
        })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );

    const captions = document.querySelectorAll(".city-card__table-caption");
    expect(captions.length).toBe(2); // one per variable table (Highs, Lows)
    expect(captions[0].textContent).toContain("no detectable bias");
    expect(captions[0].textContent).toMatch(/each cell.*180.*days/);
  });

  it("renders the typical-miss line under the summary when at least one variable is present", () => {
    render(
      <CityCard
        city={city({ typical_miss: { max: 2.9, min: 2.2 } })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );
    expect(
      screen.getByText("Typical miss, day-ahead: highs 2.9°F · lows 2.2°F")
    ).toBeInTheDocument();
  });

  it("omits a null variable from the typical-miss line", () => {
    render(
      <CityCard
        city={city({ typical_miss: { max: 2.9, min: null } })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );
    expect(screen.getByText("Typical miss, day-ahead: highs 2.9°F")).toBeInTheDocument();
  });

  it("omits the typical-miss line entirely when both variables are null", () => {
    render(
      <CityCard
        city={city({ typical_miss: NO_TYPICAL_MISS })}
        summary={{ text: "Not enough data yet.", significant: false }}
      />
    );
    expect(screen.queryByText(/Typical miss/)).not.toBeInTheDocument();
  });
});
