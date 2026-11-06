#!/usr/bin/env bun
/**
 * Prints the EXP2_TABLE of src/vecmath.c: for j = 0..31, the bits of the double
 * nearest to 2^(j/32), minus j << 47 (so that adding k << 47 for any k with
 * k mod 32 = j yields the bits of 2^(k/32)). Computed with exact integer
 * arithmetic, independent of any math library.
 *
 * usage: bun scripts/gen-exp-table.ts
 */
const N = 32n;
const GUARD = 64n; /* extra bits below the 52-bit mantissa before rounding */

/* floor(a^(1/n)) for a >= 0, by Newton's method on integers. */
function iroot(a: bigint, n: bigint): bigint {
  if (a < 2n) return a;
  let x = 1n << (BigInt(a.toString(2).length) / n + 1n); /* above the root */
  for (;;) {
    const y = ((n - 1n) * x + a / x ** (n - 1n)) / n;
    if (y >= x) return x;
    x = y;
  }
}

export function exp2Table(): bigint[] {
  return Array.from({ length: Number(N) }, (_, i) => {
    const j = BigInt(i);
    const frac = 52n + GUARD;
    /* floor(2^(j/32) * 2^frac) = floor((2^(j + 32 * frac))^(1/32)); 2^(j/32) is irrational for j > 0, so no ties. */
    const scaled = iroot(1n << (j + N * frac), N);
    const mantissa = (scaled + (1n << (GUARD - 1n))) >> GUARD; /* round to nearest: 53 bits, in [2^52, 2^53) */
    const bits = (1023n << 52n) | (mantissa - (1n << 52n));
    return (bits - (j << 47n)) & ((1n << 64n) - 1n);
  });
}

if (import.meta.main) {
  const rows = exp2Table().map((v) => `0x${v.toString(16).padStart(16, "0")}u`);
  for (let i = 0; i < rows.length; i += 4) console.log(`    ${rows.slice(i, i + 4).join(", ")},`);
}
