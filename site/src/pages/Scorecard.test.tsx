import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  CityIndex,
  Scorecard as ScorecardData,
  ScorecardCity,
  ScorecardStat,
} from "../types/export";
import { Scorecard } from "./Scorecard";

const HEAD = "98f55af5bd6dc9b329f960019eee1b6d3ef0a71fe3e2b3606751b0116e78d360";

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

function nationalStats(): ScorecardStat[] {
  return (["raw_nbm", "baseline", "challenger"] as const).map((source) =>
    stat({ source, skill: source === "raw_nbm" ? 0 : 0.1 })
  );
}

function scorecard(overrides: Partial<ScorecardData> = {}): ScorecardData {
  return {
    schema_version: "1.1.0",
    window: { days: 30, basis: "target_date", start: "2026-06-01", end: "2026-06-30" },
    intervals: {
      method: "date-block bootstrap",
      block_days: 14,
      n_boot: 2000,
      level: 0.95,
      min_sample_blocks: 21,
      note: "The 30-day window spans only a few 14-day blocks.",
    },
    log: {
      partitions: 3,
      rows: 30,
      first_run_date: "2026-05-01",
      last_run_date: "2026-06-29",
      late_rejected_total: 0,
      head_chain_sha256: HEAD,
      verified: true,
    },
    national: nationalStats(),
    ...overrides,
  };
}

const CITY_INDEX: CityIndex = {
  schema_version: "1.1.0",
  cities: [
    {
      icao: "KPHX",
      label: "Phoenix/Sky Harbor, AZ",
      lat: 33.4,
      lon: -112,
      climate_region: "Southwest",
      summary: { text: "Not enough data yet.", significant: false },
    },
  ],
};

type Responder = () => { status: number; body: unknown };

function stubFetch(routes: Record<string, Responder>): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const match = String(input).match(/\/data\/(.*)$/);
    const responder = match ? routes[match[1]] : undefined;
    if (!responder) throw new Error(`unexpected fetch: ${String(input)}`);
    const { status, body } = responder();
    return { ok: status === 200, status, json: async () => body } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const ok = (body: unknown): Responder => () => ({ status: 200, body });

describe("Scorecard page", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows a loading status, then the table with window dates and the log block", async () => {
    stubFetch({ "scorecard.json": ok(scorecard()) });
    render(<Scorecard cityIndex={CITY_INDEX} />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: /live scorecard/i })).toBeInTheDocument()
    );
    expect(screen.getByText(/2026-06-01 to 2026-06-30/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Highs" })).toBeInTheDocument();

    const log = screen.getByLabelText(/prediction log/i);
    expect(within(log).getByText(/3 partitions/)).toBeInTheDocument();
    expect(within(log).getByText(/30 rows/)).toBeInTheDocument();
    expect(within(log).getByText(/98f55af5bd6d\b/)).toBeInTheDocument();
    expect(log.textContent).not.toContain(HEAD);
    expect(within(log).getByText(/verified: yes/i)).toBeInTheDocument();
  });

  it("says plainly that nothing has been logged when the window is empty", async () => {
    stubFetch({
      "scorecard.json": ok(
        scorecard({
          window: null,
          national: [],
          log: {
            partitions: 0,
            rows: 0,
            first_run_date: null,
            last_run_date: null,
            late_rejected_total: 0,
            head_chain_sha256: "0".repeat(64),
            verified: true,
          },
        })
      ),
    });
    render(<Scorecard cityIndex={CITY_INDEX} />);

    await waitFor(() =>
      expect(
        screen.getByText(/no live predictions have been logged yet/i)
      ).toBeInTheDocument()
    );
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("does not claim an empty log when rows exist but none is graded yet", async () => {
    stubFetch({ "scorecard.json": ok(scorecard({ window: null, national: [] })) });
    render(<Scorecard cityIndex={CITY_INDEX} />);

    await waitFor(() =>
      expect(screen.getByText(/none has been compared with an observation/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/no live predictions have been logged yet/i)).toBeNull();
  });

  it("shows an error with a working retry", async () => {
    let calls = 0;
    stubFetch({
      "scorecard.json": () => {
        calls += 1;
        return calls === 1
          ? { status: 500, body: {} }
          : { status: 200, body: scorecard() };
      },
    });
    render(<Scorecard cityIndex={CITY_INDEX} />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/couldn.t load/i);
    await userEvent.click(within(alert).getByRole("button", { name: /retry/i }));
    await waitFor(() =>
      expect(screen.getByRole("table", { name: "Highs" })).toBeInTheDocument()
    );
  });

  it("loads a city's file only when that city is chosen", async () => {
    const city: ScorecardCity = {
      schema_version: "1.1.0",
      icao: "KPHX",
      label: "Phoenix/Sky Harbor, AZ",
      window: scorecard().window,
      stats: nationalStats().map((s) => ({ ...s, mae_f: 3.25 })),
    };
    const fetchMock = stubFetch({
      "scorecard.json": ok(scorecard()),
      "scorecard_cities/KPHX.json": ok(city),
    });
    render(<Scorecard cityIndex={CITY_INDEX} />);
    await screen.findByRole("table", { name: "Highs" });

    const requested = () => fetchMock.mock.calls.map((c) => String(c[0]));
    expect(requested().some((url) => url.includes("scorecard_cities"))).toBe(false);

    await userEvent.type(screen.getByRole("combobox"), "Phoenix{Enter}");

    await waitFor(() =>
      expect(requested().some((url) => url.endsWith("scorecard_cities/KPHX.json"))).toBe(
        true
      )
    );
    const section = await screen.findByRole("region", { name: /Phoenix\/Sky Harbor/ });
    expect(
      within(section).getAllByText("3.25", { selector: "span" }).length
    ).toBeGreaterThan(0);
  });

  it("says so when a chosen city has no graded predictions", async () => {
    const empty: ScorecardCity = {
      schema_version: "1.1.0",
      icao: "KPHX",
      label: "Phoenix/Sky Harbor, AZ",
      window: scorecard().window,
      stats: [],
    };
    stubFetch({
      "scorecard.json": ok(scorecard()),
      "scorecard_cities/KPHX.json": ok(empty),
    });
    render(<Scorecard cityIndex={CITY_INDEX} />);
    await screen.findByRole("table", { name: "Highs" });
    await userEvent.type(screen.getByRole("combobox"), "KPHX{Enter}");

    await waitFor(() =>
      expect(screen.getByText(/no graded predictions for Phoenix/i)).toBeInTheDocument()
    );
  });
});
