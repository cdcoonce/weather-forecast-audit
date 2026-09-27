import { describe, expect, it } from "vitest";
import type { CityStat, TypicalMiss } from "../types/export";
import {
  biasCellContent,
  formatBiasEndpoints,
  formatGeneratedAt,
  formatSignedEndpoint,
  formatSignedF,
  formatTypicalMissLine,
  medianCellN,
  seasonWord,
} from "./format";

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

describe("formatSignedF", () => {
  it("uses a true minus sign (U+2212), never a hyphen, for negative values", () => {
    const text = formatSignedF(-1.4);
    expect(text).toBe("−1.4");
    expect(text).not.toContain("-"); // hyphen-minus, U+002D
  });

  it("uses a plus sign for positive values", () => {
    expect(formatSignedF(1.4)).toBe("+1.4");
  });

  it("has no sign for exactly zero", () => {
    expect(formatSignedF(0)).toBe("0.0");
  });
});

describe("formatSignedEndpoint", () => {
  it("uses two decimals when one decimal would hide the sign behind '0.0'", () => {
    expect(formatSignedEndpoint(0.03)).toBe("+0.03");
    expect(formatSignedEndpoint(-0.01)).toBe("−0.01");
  });

  it("uses one decimal when it already keeps the value visibly non-zero", () => {
    expect(formatSignedEndpoint(-0.36)).toBe("−0.4");
    expect(formatSignedEndpoint(0.53)).toBe("+0.5");
  });

  it("has no sign for exactly zero", () => {
    expect(formatSignedEndpoint(0)).toBe("0.0");
  });
});

describe("formatBiasEndpoints", () => {
  it("renders both endpoints with true minus signs and 'to', never '±'", () => {
    const text = formatBiasEndpoints(
      stat({ bias_lo_f: -1.6, bias_hi_f: -0.6 })
    );
    expect(text).toBe("−1.6 to −0.6");
    expect(text).not.toContain("±");
  });

  it("returns null when either endpoint is missing", () => {
    expect(formatBiasEndpoints(stat({ bias_lo_f: null }))).toBeNull();
    expect(formatBiasEndpoints(stat({ bias_hi_f: null }))).toBeNull();
  });

  it("never renders a small-magnitude endpoint as a sign-contradicting '+0.0'/'−0.0'", () => {
    const text = formatBiasEndpoints(stat({ bias_lo_f: 0.03, bias_hi_f: 0.53 }));
    expect(text).toBe("+0.03 to +0.5");
  });
});

describe("biasCellContent", () => {
  it("a significant cell: bold primary '−1.1°F' and endpoints secondary, no '±'", () => {
    const content = biasCellContent(
      stat({ bias_f: -1.1, bias_lo_f: -1.6, bias_hi_f: -0.6 })
    );
    expect(content.kind).toBe("significant");
    expect(content.primary).toBe("−1.1°F");
    expect(content.secondary).toBe("−1.6 to −0.6");
    expect(content.primary).not.toContain("±");
    expect(content.secondary).not.toContain("±");
  });

  it("a non-significant cell: muted en dash with accessible name 'no detectable bias', no digits", () => {
    const content = biasCellContent(
      stat({ no_detectable_bias: true, bias_f: 0.3, bias_lo_f: -0.2, bias_hi_f: 0.8 })
    );
    expect(content.kind).toBe("no-detectable-bias");
    expect(content.visible).toBe("–"); // en dash, distinct from the minus sign
    expect(content.visible).not.toMatch(/\d/);
    expect(content.accessibleName).toBe("no detectable bias");
  });

  it("a too-few-days cell: muted 'few days' with accessible name 'too few days', no digits", () => {
    const content = biasCellContent(
      stat({ min_sample_flag: true, bias_f: 5.0, bias_lo_f: null, bias_hi_f: null })
    );
    expect(content.kind).toBe("few-days");
    expect(content.visible).toBe("few days");
    expect(content.visible).not.toMatch(/\d/);
    expect(content.accessibleName).toBe("too few days");
  });

  it("min_sample_flag takes precedence over no_detectable_bias", () => {
    const content = biasCellContent(
      stat({ min_sample_flag: true, no_detectable_bias: true })
    );
    expect(content.kind).toBe("few-days");
  });

  it("never returns a primary/secondary number for a non-significant kind", () => {
    for (const overrides of [{ min_sample_flag: true }, { no_detectable_bias: true }]) {
      const content = biasCellContent(stat(overrides));
      expect(content.primary).toBeUndefined();
      expect(content.secondary).toBeUndefined();
    }
  });
});

describe("medianCellN", () => {
  it("rounds the median to the nearest 10", () => {
    expect(medianCellN([181, 178, 183])).toBe(180);
    expect(medianCellN([10, 20])).toBe(20); // (10+20)/2 = 15 -> rounds to 20
    expect(medianCellN([])).toBe(0);
  });
});

describe("formatTypicalMissLine", () => {
  it("renders both variables when both are present", () => {
    const line = formatTypicalMissLine({ max: 2.9, min: 2.2 });
    expect(line).toBe("Typical miss, day-ahead: highs 2.9°F · lows 2.2°F");
  });

  it("omits a null variable", () => {
    expect(formatTypicalMissLine({ max: 2.9, min: null })).toBe(
      "Typical miss, day-ahead: highs 2.9°F"
    );
    expect(formatTypicalMissLine({ max: null, min: 2.2 })).toBe(
      "Typical miss, day-ahead: lows 2.2°F"
    );
  });

  it("returns null (line omitted) when both are null", () => {
    const typicalMiss: TypicalMiss = { max: null, min: null };
    expect(formatTypicalMissLine(typicalMiss)).toBeNull();
  });
});

describe("formatGeneratedAt", () => {
  it("formats an ISO UTC timestamp as 'YYYY-MM-DD HH:MM UTC'", () => {
    expect(formatGeneratedAt("2026-09-27T14:03:00Z")).toBe("2026-09-27 14:03 UTC");
  });
});
