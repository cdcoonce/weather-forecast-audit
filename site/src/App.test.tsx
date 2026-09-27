import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

/**
 * Renders the whole app against the committed, drift-guarded export
 * (site/src/test/fixtures/export/, produced by weather_forecast_audit.export
 * against the tracer fixture DB -- see
 * tests/integration/test_export_tracer.py::test_site_fixture_export_matches_a_fresh_export).
 * This is site test 6 from the build spec: the card renders from real
 * exported data, not just hand-built props.
 */

const dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = path.resolve(dirname, "test/fixtures/export");

function readFixture(relativePath: string): unknown {
  return JSON.parse(readFileSync(path.join(FIXTURE_DIR, relativePath), "utf-8"));
}

function stubFixtureFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const match = url.match(/\/data\/(.*)$/);
      if (!match) {
        throw new Error(`unexpected fetch: ${url}`);
      }
      const body = readFixture(match[1]);
      return {
        ok: true,
        status: 200,
        json: async () => body,
      } as Response;
    })
  );
}

describe("App (committed fixture export)", () => {
  beforeEach(() => {
    stubFixtureFetch();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.location.hash = "";
  });

  it("renders the home page with the last-updated line and a working search", async () => {
    window.location.hash = "#/";
    render(<App />);

    await waitFor(() => expect(screen.getByText(/Data through/)).toBeInTheDocument());
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("renders the KPHX card from the real exported files", async () => {
    window.location.hash = "#/city/KPHX";
    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/Phoenix\/Sky Harbor, AZ/)).toBeInTheDocument()
    );
    // The tracer fixture only ingests one issuance date per run, so every
    // stat is min_sample_flag -- exercising the "too few days" accessible
    // name against real exported data, not a hand-built fixture.
    expect(screen.getAllByText(/too few days/i).length).toBeGreaterThan(0);
  });

  it("shows not-found for an ICAO absent from the committed city index", async () => {
    window.location.hash = "#/city/ZZZZ";
    render(<App />);

    await waitFor(() => expect(screen.getByText(/city not found/i)).toBeInTheDocument());
    expect(screen.getByRole("link", { name: /back to search/i })).toBeInTheDocument();
  });
});
