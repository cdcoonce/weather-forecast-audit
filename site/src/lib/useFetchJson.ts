import { useCallback, useEffect, useState } from "react";
import { fetchJson } from "./dataLoader";

export type FetchState<T> =
  | { status: "loading"; data: null; error: null }
  | { status: "success"; data: T; error: null }
  | { status: "error"; data: null; error: Error };

interface Resolution<T> {
  path: string;
  attempt: number;
  data: T | null;
  error: Error | null;
}

/**
 * Fetches `${BASE_URL}data/{path}` (via `dataLoader.fetchJson`) and exposes
 * loading/success/error state plus a `retry()` for the "Couldn't load the
 * data." + retry-button state the site's pages show on failure.
 *
 * `loading` is derived at render time (the latest *resolved* `(path,
 * attempt)` tag vs. the current one), not set synchronously inside the
 * effect: state is only ever written from the async `.then`/`.catch`
 * callbacks, never from the effect body itself.
 */
export function useFetchJson<T extends { schema_version: string }>(
  path: string
): FetchState<T> & { retry: () => void } {
  const [attempt, setAttempt] = useState(0);
  const [resolution, setResolution] = useState<Resolution<T> | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetchJson<T>(path)
      .then((data) => {
        if (!cancelled) setResolution({ path, attempt, data, error: null });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setResolution({
            path,
            attempt,
            data: null,
            error: error instanceof Error ? error : new Error(String(error)),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [path, attempt]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);

  if (!resolution || resolution.path !== path || resolution.attempt !== attempt) {
    return { status: "loading", data: null, error: null, retry };
  }
  if (resolution.error) {
    return { status: "error", data: null, error: resolution.error, retry };
  }
  return { status: "success", data: resolution.data as T, error: null, retry };
}
