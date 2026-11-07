/** Argument parsing and output encoding of scripts/ref/phi3-reference.ts. */
import { describe, expect, test } from "bun:test";

import { parseArgs, parseIds, toFloat32LE } from "../scripts/ref/phi3-reference";

describe("argument parsing", () => {
  test("ids, output, and defaults", () => {
    const args = parseArgs(["model", "--ids", "200022, 50,7", "--out", "l.f32"]);
    expect(args.modelDir).toBe("model");
    expect(args.ids).toEqual([200022, 50, 7]);
    expect(args.out).toBe("l.f32");
    expect(args.top).toBe(5);
    expect(args.generate).toBe(0);
    expect(args.selfCheck).toBe(false);
  });

  test("chat text instead of ids; self-check needs no ids", () => {
    expect(parseArgs(["m", "--system", "s", "--user", "u"]).system).toBe("s");
    expect(parseArgs(["m", "--self-check"]).ids).toBeNull();
  });

  test("rejects bad input", () => {
    expect(() => parseIds("1,-2")).toThrow();
    expect(() => parseIds("1.5")).toThrow();
    expect(() => parseIds("")).toThrow();
    expect(() => parseArgs(["m"])).toThrow(); // no ids
    expect(() => parseArgs(["m", "--ids", "1", "--system", "s", "--user", "u"])).toThrow();
    expect(() => parseArgs(["m", "--system", "s"])).toThrow();
    expect(() => parseArgs(["m", "--ids"])).toThrow();
    expect(() => parseArgs(["m", "--ids", "1", "--bogus"])).toThrow();
    expect(() => parseArgs(["a", "b", "--ids", "1"])).toThrow();
  });
});

describe("float32 little-endian output", () => {
  test("1.0 = 00 00 80 3f, -2.0 = 00 00 00 c0", () => {
    expect(Array.from(toFloat32LE(new Float64Array([1, -2])))).toEqual([0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0]);
  });
});
