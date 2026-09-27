import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useFetchJson } from "./useFetchJson";

describe("useFetchJson", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts loading, then reports success with the fetched data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ schema_version: "1.0.0", value: 42 }),
      })
    );

    const { result } = renderHook(() => useFetchJson<{ schema_version: string; value: number }>("thing.json"));
    expect(result.current.status).toBe("loading");

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual({ schema_version: "1.0.0", value: 42 });
  });

  it("reports an error state on failure, and retry() re-fetches", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ schema_version: "1.0.0", value: 1 }),
      });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useFetchJson<{ schema_version: string; value: number }>("thing.json"));
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.retry());
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
