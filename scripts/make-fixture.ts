#!/usr/bin/env bun
/**
 * Writes a synthetic SMOEQ8 checkpoint, config and tokenizer to a directory.
 * usage: bun scripts/make-fixture.ts DIR [--dim N] [--hidden N] [--experts N] [--top-k N] [--seed N]
 */
import { parseArgs } from "node:util";

import { writeFixture } from "../tests/support/fixture";

const { values, positionals } = parseArgs({
  args: Bun.argv.slice(2),
  allowPositionals: true,
  options: {
    dim: { type: "string" },
    hidden: { type: "string" },
    experts: { type: "string" },
    "top-k": { type: "string" },
    seed: { type: "string" },
  },
});

const dir = positionals[0];
if (!dir) {
  console.error("usage: bun scripts/make-fixture.ts DIR [--dim N] [--hidden N] [--experts N] [--top-k N] [--seed N]");
  process.exit(2);
}
const toInt = (v: string | undefined) => (v === undefined ? undefined : Number.parseInt(v, 10));
const config = Object.fromEntries(
  Object.entries({
    dim: toInt(values.dim),
    hidden_dim: toInt(values.hidden),
    experts: toInt(values.experts),
    top_k: toInt(values["top-k"]),
  }).filter(([, v]) => v !== undefined),
);
const fixture = writeFixture(dir, { config, seed: toInt(values.seed) ?? 1 });
console.log(`wrote ${fixture.tensors.length} tensors to ${fixture.model}`);
