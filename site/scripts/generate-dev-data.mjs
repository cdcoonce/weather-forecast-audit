#!/usr/bin/env node
// Regenerates site/public/data/ from the real export contract, for local
// dev: `npm run generate:data` (see README's "Site" section). Never
// hand-write these files.
//
// Requires WFA_DUCKDB_PATH to already point at a built warehouse (same
// convention as every other `wfa` command -- see the repo README's "Local
// setup" and "Tracer bullet" sections for how to build one).

import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(siteDir, "..");
const outDir = path.join(siteDir, "public", "data");

if (!process.env.WFA_DUCKDB_PATH) {
  console.error(
    "generate-dev-data: WFA_DUCKDB_PATH must point at a built warehouse " +
      "(see README.md's Local setup / Tracer bullet sections)."
  );
  process.exit(1);
}

const result = spawnSync("uv", ["run", "wfa", "export", "--out", outDir], {
  cwd: repoRoot,
  stdio: "inherit",
});

if (result.error) {
  console.error(`generate-dev-data: failed to run uv: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
