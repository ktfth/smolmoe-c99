/** Performance pipeline: medians, run summaries and baseline comparison. */
import { describe, expect, test } from "bun:test";

import { type BenchReport, type RunStats, type ScenarioReport, compare, median, summarize } from "../scripts/bench";

function run(over: Partial<RunStats> = {}): RunStats {
  return {
    backend: "avx2", threads: 4, batch: 32, prompt_tokens: 100, generated_tokens: 10, ttft_seconds: 1,
    prefill_tokens_per_second: 50, decode_tokens_per_second: 20, rss_peak_bytes: 600 * 1048576,
    output_hash: "aa", profile: { qkv: 1, attention: 2 }, ...over,
  };
}

function report(scenario: Partial<ScenarioReport>): BenchReport {
  const s: ScenarioReport = {
    prompt_tokens: 100, generated_tokens: 10, prefill_tokens_per_second: 50, decode_tokens_per_second: 20,
    ttft_seconds: 1, rss_mib: 600, phases: {}, ...scenario,
  };
  return {
    date: "2026-11-06T00:00:00.000Z", commit: "abc1234", machine: { cpu: "cpu", logical_cores: 8, os: "test" },
    backend: "avx2", threads: 4, batch: 32, runs: 3, scenarios: { chat: s },
  };
}

describe("median", () => {
  test("odd and even lengths, input untouched", () => {
    const xs = [3, 1, 2];
    expect(median(xs)).toBe(2);
    expect(xs).toEqual([3, 1, 2]);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  test("rejects an empty list", () => {
    expect(() => median([])).toThrow();
  });
});

describe("summarize", () => {
  test("medians of throughput and phases, peak memory", () => {
    const s = summarize([
      run({ prefill_tokens_per_second: 40, profile: { qkv: 1, attention: 5 } }),
      run({ prefill_tokens_per_second: 60, profile: { qkv: 3, attention: 1 } }),
      run({ prefill_tokens_per_second: 50, rss_peak_bytes: 700 * 1048576, profile: { qkv: 2, attention: 2 } }),
    ]);
    expect(s.prefill_tokens_per_second).toBe(50);
    expect(s.phases).toEqual({ qkv: 2, attention: 2 });
    expect(s.rss_mib).toBe(700);
  });

  test("runs of one scenario must produce the same text", () => {
    expect(() => summarize([run(), run({ output_hash: "bb" })])).toThrow(/different text/);
  });
});

describe("compare", () => {
  test("higher throughput and lower latency count as better", () => {
    const deltas = compare(report({}), report({ prefill_tokens_per_second: 60, ttft_seconds: 0.5 }), 5);
    const by = Object.fromEntries(deltas.map((d) => [d.metric, d]));
    expect(by.prefill_tokens_per_second.verdict).toBe("melhor");
    expect(by.prefill_tokens_per_second.percent).toBeCloseTo(20);
    expect(by.ttft_seconds.verdict).toBe("melhor");
    expect(by.ttft_seconds.percent).toBeCloseTo(50);
    expect(by.decode_tokens_per_second.verdict).toBe("igual");
  });

  test("changes within the threshold are equal; beyond it, worse", () => {
    const deltas = compare(report({}), report({ decode_tokens_per_second: 19.2, rss_mib: 700 }), 5);
    const by = Object.fromEntries(deltas.map((d) => [d.metric, d]));
    expect(by.decode_tokens_per_second.verdict).toBe("igual");
    expect(by.rss_mib.verdict).toBe("pior");
  });

  test("metrics absent in a scenario are skipped", () => {
    const deltas = compare(report({ decode_tokens_per_second: 0 }), report({ decode_tokens_per_second: 0 }), 5);
    expect(deltas.map((d) => d.metric)).not.toContain("decode_tokens_per_second");
  });

  test("a metric that drops to zero or a missing scenario is worse, never silently equal", () => {
    const dropped = compare(report({}), report({ decode_tokens_per_second: 0 }), 5);
    expect(dropped.find((d) => d.metric === "decode_tokens_per_second")?.verdict).toBe("pior");
    const renamed: BenchReport = { ...report({}), scenarios: { outro: report({}).scenarios.chat } };
    const missing = compare(report({}), renamed, 5);
    expect(missing.length).toBeGreaterThan(0);
    expect(missing.every((d) => d.verdict === "pior")).toBe(true);
  });

  test("rejects a threshold that is not a non-negative number", () => {
    expect(() => compare(report({}), report({}), Number.NaN)).toThrow(/threshold/);
    expect(() => compare(report({}), report({}), -1)).toThrow(/threshold/);
  });
});
