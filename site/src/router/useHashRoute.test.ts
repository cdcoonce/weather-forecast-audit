import { describe, expect, it } from "vitest";
import { parseHash } from "./useHashRoute";

describe("parseHash", () => {
  it("parses the home, city and scorecard routes", () => {
    expect(parseHash("")).toEqual({ name: "home" });
    expect(parseHash("#/")).toEqual({ name: "home" });
    expect(parseHash("#/city/kphx")).toEqual({ name: "city", icao: "KPHX" });
    expect(parseHash("#/scorecard")).toEqual({ name: "scorecard" });
  });

  it("keeps unknown paths not-found, including near-misses of the scorecard", () => {
    expect(parseHash("#/nope")).toEqual({ name: "not-found" });
    expect(parseHash("#/scorecard/extra")).toEqual({ name: "not-found" });
    expect(parseHash("#/scorecards")).toEqual({ name: "not-found" });
  });

  it("treats a trailing slash the same way for scorecard and city routes", () => {
    // current contract: no trailing-slash tolerance on any route
    expect(parseHash("#/scorecard/")).toEqual({ name: "not-found" });
    expect(parseHash("#/city/KPHX/")).toEqual({ name: "not-found" });
  });
});
