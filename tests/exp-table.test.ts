/** The exp table in src/vecmath.c is the output of scripts/gen-exp-table.ts. */
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { exp2Table } from "../scripts/gen-exp-table";
import { ROOT } from "./support/run";

test("EXP2_TABLE in vecmath.c matches the exact integer computation", () => {
  const source = readFileSync(join(ROOT, "src", "vecmath.c"), "utf8");
  const body = /EXP2_TABLE\[32\] = \{([^}]*)\}/.exec(source)?.[1];
  expect(body).toBeDefined();
  const inC = [...body!.matchAll(/0x([0-9a-f]{16})u/g)].map((m) => BigInt(`0x${m[1]}`));
  expect(inC).toEqual(exp2Table());
});

test("the table holds 2^(j/32) for every j", () => {
  const view = new DataView(new ArrayBuffer(8));
  exp2Table().forEach((entry, j) => {
    view.setBigUint64(0, (entry + (BigInt(j) << 47n)) & ((1n << 64n) - 1n));
    expect(view.getFloat64(0)).toBe(2 ** (j / 32));
  });
});
