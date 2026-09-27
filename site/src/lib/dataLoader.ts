import { SCHEMA_VERSION } from "../types/export";

/** Data whose `schema_version` major component differs from the site's. */
export class SchemaVersionError extends Error {}

function majorVersion(version: string): number {
  const major = version.split(".")[0];
  const parsed = Number(major);
  return Number.isNaN(parsed) ? -1 : parsed;
}

export function assertCompatibleSchemaVersion(version: string): void {
  const expected = majorVersion(SCHEMA_VERSION);
  const actual = majorVersion(version);
  if (actual !== expected) {
    throw new SchemaVersionError(
      `This page can't read data at schema_version ${version}: it expects ` +
        `${SCHEMA_VERSION}.x. The site and its data are out of sync.`
    );
  }
}

/**
 * Fetches and parses `${BASE_URL}data/{path}`, refusing data whose
 * `schema_version` major component doesn't match the site's `SCHEMA_VERSION`
 * (see `types/export.ts`).
 */
export async function fetchJson<T extends { schema_version: string }>(
  path: string
): Promise<T> {
  const base = import.meta.env.BASE_URL;
  const url = `${base}data/${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Couldn't load ${path} (HTTP ${response.status}).`);
  }
  const data = (await response.json()) as T;
  assertCompatibleSchemaVersion(data.schema_version);
  return data;
}
