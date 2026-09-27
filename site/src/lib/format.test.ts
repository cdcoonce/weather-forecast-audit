import { describe, expect, it } from "vitest";
import type { CityStat } from "../types/export";
import { formatBiasCell, formatGeneratedAt, seasonWord } from "./format";

function stat(overrides: Partial<CityStat>): CityStat {
  return {
    variable: "max",
    lead_day: 1,
    season: "JJA",
    n: 42,
    n_dates: 35,
    bias_f: 1.4,
    bias_lo_f: 0.8,
    bias_hi_f: 2.0,
    mae_f: 1.6,
    min_sample_flag: false,
    no_detectable_bias: false,
    ...overrides,
  };
}

describe("seasonWord", () => {
  it("maps meteorological season codes to plain words", () => {
    expect(seasonWord("DJF")).toBe("winter");
    expect(seasonWord("MAM")).toBe("spring");
    expect(seasonWord("JJA")).toBe("summer");
    expect(seasonWord("SON")).toBe("fall");
  });
});

describe("formatBiasCell", () => {
  it("shows a signed bias and half-CI when the bias is significant", () => {
    const text = formatBiasCell(stat({ bias_f: 1.4, bias_lo_f: 0.8, bias_hi_f: 2.0 }));
    expect(text).toContain("+1.4");
    expect(text).toContain("0.6");
    expect(text).toMatch(/°F/);
  });

  it("shows a minus sign for a cold (negative) bias", () => {
    const text = formatBiasCell(
      stat({ bias_f: -1.4, bias_lo_f: -2.0, bias_hi_f: -0.8 })
    );
    expect(text).toMatch(/[−-]1\.4/);
  });

  it('reads "too few days" for a min_sample_flag cell, with no signed number', () => {
    const text = formatBiasCell(
      stat({ min_sample_flag: true, bias_f: 5.0, bias_lo_f: null, bias_hi_f: null })
    );
    expect(text).toContain("too few days");
    expect(text).not.toMatch(/[+−]\d/);
  });

  it('reads "no detectable bias" when no_detectable_bias is true, with no signed number', () => {
    const text = formatBiasCell(
      stat({ no_detectable_bias: true, bias_f: 0.3, bias_lo_f: -0.2, bias_hi_f: 0.8 })
    );
    expect(text).toContain("no detectable bias");
    expect(text).not.toMatch(/[+−]\d/);
  });

  it("min_sample_flag takes precedence over no_detectable_bias wording", () => {
    const text = formatBiasCell(stat({ min_sample_flag: true, no_detectable_bias: true }));
    expect(text).toContain("too few days");
    expect(text).not.toContain("no detectable bias");
  });
});

describe("formatGeneratedAt", () => {
  it("formats an ISO UTC timestamp as 'YYYY-MM-DD HH:MM UTC'", () => {
    expect(formatGeneratedAt("2026-09-27T14:03:00Z")).toBe("2026-09-27 14:03 UTC");
  });
});
