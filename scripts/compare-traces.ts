#!/usr/bin/env bun
/**
 * Compares two `--trace` logit dumps produced from the same prompt.
 * Reports teacher-forced NLL/perplexity for each, top-1 agreement and the
 * mean KL divergence KL(reference || candidate) of next-token distributions.
 *
 * usage: bun scripts/compare-traces.ts REF.trace CAND.trace TOKEN_IDS
 */

type Row = Float64Array;

export function parseHexFloat(text: string): number {
  const match = /^([+-]?)0x([0-9a-f]+)(?:\.([0-9a-f]*))?p([+-]?\d+)$/i.exec(text);
  if (!match) {
    if (/^[+-]?inf/i.test(text)) return text.startsWith("-") ? -Infinity : Infinity;
    if (/nan/i.test(text)) return NaN;
    throw new Error(`not a C99 hex float: ${text}`);
  }
  const [, sign, whole, fraction = "", exponent] = match;
  const mantissa = parseInt(whole + fraction, 16) / 16 ** fraction.length;
  const value = mantissa * 2 ** Number(exponent);
  return sign === "-" ? -value : value;
}

async function readTrace(path: string): Promise<Row[]> {
  const text = await Bun.file(path).text();
  return text
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => Float64Array.from(line.split(" ").slice(1), parseHexFloat));
}

function logSoftmax(row: Row): Row {
  let high = -Infinity;
  for (const v of row) high = Math.max(high, v);
  let sum = 0;
  for (const v of row) sum += Math.exp(v - high);
  const offset = high + Math.log(sum);
  return row.map((v) => v - offset);
}

function argmax(row: Row): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

async function main(): Promise<void> {
  const [refPath, candPath, idsPath] = Bun.argv.slice(2);
  if (!refPath || !candPath || !idsPath) {
    console.error("usage: bun scripts/compare-traces.ts REF.trace CAND.trace TOKEN_IDS");
    process.exit(2);
  }
  const [ref, cand] = await Promise.all([readTrace(refPath), readTrace(candPath)]);
  const ids = (await Bun.file(idsPath).text()).trim().split("\n").map(Number);
  const steps = Math.min(ref.length, cand.length, ids.length - 1);
  if (steps < 1) throw new Error("traces contain no comparable positions");

  let nllRef = 0, nllCand = 0, kl = 0, agree = 0, maxAbs = 0;
  for (let t = 0; t < steps; t++) {
    const lr = logSoftmax(ref[t]);
    const lc = logSoftmax(cand[t]);
    const next = ids[t + 1];
    nllRef -= lr[next];
    nllCand -= lc[next];
    for (let i = 0; i < lr.length; i++) {
      kl += Math.exp(lr[i]) * (lr[i] - lc[i]);
      maxAbs = Math.max(maxAbs, Math.abs(ref[t][i] - cand[t][i]));
    }
    if (argmax(ref[t]) === argmax(cand[t])) agree++;
  }
  const report = {
    positions: steps,
    reference: { nll: nllRef / steps, perplexity: Math.exp(nllRef / steps) },
    candidate: { nll: nllCand / steps, perplexity: Math.exp(nllCand / steps) },
    perplexity_change_percent: (Math.exp((nllCand - nllRef) / steps) - 1) * 100,
    top1_agreement: agree / steps,
    mean_kl_nats: kl / steps,
    max_abs_logit_diff: maxAbs,
  };
  console.log(JSON.stringify(report, null, 2));
}

if (import.meta.main) await main();
