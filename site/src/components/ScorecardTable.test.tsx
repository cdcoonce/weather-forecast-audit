import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ScorecardSource, ScorecardStat } from "../types/export";
import { ScorecardTable } from "./ScorecardTable";

function stat(overrides: Partial<ScorecardStat>): ScorecardStat {
  return {
    variable: "max",
    lead_day: 1,
    source: "raw_nbm",
    n: 120,
    n_dates: 30,
    n_blocks: 3,
    mae_f: 2.0,
    mae_lo_f: 1.5,
    mae_hi_f: 2.5,
    bias_f: 0.5,
    bias_lo_f: -0.2,
    bias_hi_f: 0.9,
    skill: 0.0,
    skill_lo: 0.0,
    skill_hi: 0.0,
    min_sample_flag: true,
    no_detectable_bias: true,
    ...overrides,
  };
}

const SOURCES: ScorecardSource[] = ["raw_nbm", "baseline", "challenger"];

function slice(
  variable: "max" | "min",
  lead_day: number,
  overrides: Partial<ScorecardStat> = {}
): ScorecardStat[] {
  return SOURCES.map((source) =>
    stat({
      variable,
      lead_day,
      source,
      skill: source === "raw_nbm" ? 0 : 0.1,
      skill_lo: source === "raw_nbm" ? 0 : 0.02,
      skill_hi: source === "raw_nbm" ? 0 : 0.18,
      ...overrides,
    })
  );
}

describe("ScorecardTable", () => {
  it("renders three sources per variable and lead, in a fixed order", () => {
    render(
      <ScorecardTable
        stats={[...slice("max", 1), ...slice("max", 2), ...slice("min", 1)]}
        windowDays={30}
      />
    );

    expect(screen.getByRole("table", { name: "Highs" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Lows" })).toBeInTheDocument();

    const highs = screen.getByRole("table", { name: "Highs" });
    const sourceCells = within(highs)
      .getAllByRole("rowheader")
      .map((cell) => cell.textContent)
      .filter((text) => /NBM|Baseline|Challenger/.test(text ?? ""));
    expect(sourceCells).toEqual([
      "Raw NBM",
      "Baseline",
      "Challenger",
      "Raw NBM",
      "Baseline",
      "Challenger",
    ]);
    expect(within(highs).getByText("Day 1")).toBeInTheDocument();
    expect(within(highs).getByText("Day 2")).toBeInTheDocument();
  });

  it("shows MAE and bias with their intervals, using a true minus sign", () => {
    render(<ScorecardTable stats={slice("max", 1)} windowDays={30} />);
    const rows = screen.getAllByRole("row");
    const raw = rows.find((row) => within(row).queryByText("Raw NBM"));
    expect(raw).toBeDefined();
    const text = raw?.textContent ?? "";
    expect(text).toContain("2.00");
    expect(text).toContain("1.50 to 2.50");
    expect(text).toContain("+0.50");
    expect(text).toContain("−0.20 to +0.90");
    expect(text).not.toContain("-0.20");
    expect(text).not.toContain("±");
  });

  it("shows skill versus raw for corrected sources and 'reference' for raw", () => {
    render(<ScorecardTable stats={slice("max", 1)} windowDays={30} />);
    const rows = screen.getAllByRole("row");
    const raw = rows.find((row) => within(row).queryByText("Raw NBM"));
    const baseline = rows.find((row) => within(row).queryByText("Baseline"));
    expect(raw?.textContent).toContain("reference");
    expect(raw?.textContent).not.toContain("0.0%");
    expect(baseline?.textContent).toContain("+10.0%");
    expect(baseline?.textContent).toContain("+2.0% to +18.0%");
  });

  it("carries the project's wording on a flagged group, and never claims significance", () => {
    render(
      <ScorecardTable
        stats={slice("max", 1, {
          min_sample_flag: true,
          no_detectable_bias: false,
          bias_f: 1.2,
          bias_lo_f: 0.6,
          bias_hi_f: 1.8,
        })}
        windowDays={30}
      />
    );
    expect(
      screen.getByText(/interval is approximate.*30-day window is too short for a significance call/i)
    ).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/significant|excludes zero/i);
    expect(document.querySelector(".scorecard__cell--significant")).toBeNull();
  });

  it("only an unflagged interval that excludes zero may be called out as such", () => {
    render(
      <ScorecardTable
        stats={slice("max", 1, {
          min_sample_flag: false,
          no_detectable_bias: false,
          bias_f: 1.2,
          bias_lo_f: 0.6,
          bias_hi_f: 1.8,
        })}
        windowDays={30}
      />
    );
    expect(screen.queryByText(/too short for a significance call/i)).toBeNull();
    expect(screen.getAllByText(/interval excludes zero/i).length).toBeGreaterThan(0);
  });

  it("says so when an interval cannot be estimated, rather than printing nulls", () => {
    render(
      <ScorecardTable
        stats={slice("max", 1, {
          n_blocks: 1,
          mae_lo_f: null,
          mae_hi_f: null,
          bias_lo_f: null,
          bias_hi_f: null,
        })}
        windowDays={30}
      />
    );
    expect(screen.getAllByText(/no interval/i).length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/null|NaN|undefined/);
  });

  it("draws a whisker with an accessible description when an interval exists", () => {
    render(<ScorecardTable stats={slice("max", 1)} windowDays={30} />);
    const whiskers = screen.getAllByRole("img", { name: /bias interval/i });
    expect(whiskers).toHaveLength(3);
    expect(whiskers[0]).toHaveAccessibleName(/−0.20 to \+0.90/);
  });

  it("draws no whisker at all when the interval is missing", () => {
    render(
      <ScorecardTable
        stats={slice("max", 1, { bias_lo_f: null, bias_hi_f: null })}
        windowDays={30}
      />
    );
    expect(screen.queryAllByRole("img")).toHaveLength(0);
    expect(document.querySelector(".scorecard__whisker")).toBeNull();
  });

  it("never calls out an interval that spans zero, even when the group is unflagged", () => {
    render(
      <ScorecardTable
        stats={slice("max", 1, { min_sample_flag: false, no_detectable_bias: true })}
        windowDays={30}
      />
    );
    expect(screen.queryByText(/interval excludes zero/i)).toBeNull();
    expect(document.querySelector(".scorecard__cell--significant")).toBeNull();
  });

  it("puts the approximate note on a group when any one source is flagged", () => {
    const group = slice("max", 1, { min_sample_flag: false }).map((s) =>
      s.source === "challenger" ? { ...s, min_sample_flag: true } : s
    );
    render(<ScorecardTable stats={group} windowDays={30} />);
    expect(screen.getByText(/too short for a significance call/i)).toBeInTheDocument();
  });

  it("names the window length it was given in the approximate note", () => {
    render(<ScorecardTable stats={slice("max", 1)} windowDays={21} />);
    expect(screen.getByText(/21-day window is too short/i)).toBeInTheDocument();
  });

  it("keeps the zero tick and the point estimate on the whisker's scale", () => {
    // every interval is strictly positive: zero is outside the data range, but
    // the shared scale must still include it so the tick is drawn on screen
    render(
      <ScorecardTable
        stats={slice("max", 1, {
          bias_f: 1.0,
          bias_lo_f: 0.5,
          bias_hi_f: 1.8,
          no_detectable_bias: false,
        })}
        windowDays={30}
      />
    );
    const svg = document.querySelector(".scorecard__whisker") as SVGSVGElement;
    const width = Number(svg.getAttribute("width"));
    const zero = svg.querySelector(".scorecard__whisker-zero") as SVGLineElement;
    const span = svg.querySelector(".scorecard__whisker-span") as SVGLineElement;
    const point = svg.querySelector(".scorecard__whisker-point") as SVGCircleElement;
    const x0 = Number(zero.getAttribute("x1"));
    const lo = Number(span.getAttribute("x1"));
    const hi = Number(span.getAttribute("x2"));
    const cx = Number(point.getAttribute("cx"));
    expect(x0).toBeGreaterThanOrEqual(0);
    expect(x0).toBeLessThanOrEqual(width);
    expect(x0).toBeLessThan(lo); // zero is left of a positive interval
    expect(lo).toBeLessThan(cx);
    expect(cx).toBeLessThan(hi);
  });
});
