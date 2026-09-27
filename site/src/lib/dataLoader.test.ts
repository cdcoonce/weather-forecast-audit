import { afterEach, describe, expect, it, vi } from "vitest";
import { SchemaVersionError, fetchJson } from "./dataLoader";

describe("fetchJson", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves data whose schema_version major matches the site's", async () => {
    const payload = { schema_version: "1.0.0", cities: [] };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => payload,
      })
    );

    const data = await fetchJson<{ schema_version: string; cities: unknown[] }>(
      "city_index.json"
    );
    expect(data).toEqual(payload);
  });

  it("rejects data from a different major schema_version", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ schema_version: "2.0.0", cities: [] }),
      })
    );

    await expect(fetchJson("city_index.json")).rejects.toThrow(SchemaVersionError);
  });

  it("accepts a newer minor/patch version within the same major", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ schema_version: "1.4.2", cities: [] }),
      })
    );

    await expect(fetchJson("city_index.json")).resolves.toMatchObject({
      schema_version: "1.4.2",
    });
  });

  it("raises a clear error on a non-ok HTTP response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })
    );

    await expect(fetchJson("missing.json")).rejects.toThrow(/missing\.json/);
  });
});
