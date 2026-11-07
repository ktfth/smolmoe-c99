/** Helpers shared by the model converters: options, JSON narrowing, atomic side files. */
import { copyFileSync, existsSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export type Container = "v2" | "v3";

export interface ConvertOptions {
  readonly source: string;
  readonly output: string;
  readonly container: Container;
  readonly threads: number;
  /** false only in tests with synthetic sources: skips the pinned SHA-256/revision checks */
  readonly pinned: boolean;
  readonly log: (line: string) => void;
}

export interface ConvertReport {
  readonly files: readonly { readonly path: string; readonly bytes: number }[];
  readonly seconds: number;
}

export type Json = Record<string, unknown>;

export const CONVERTER = "smolmoe scripts/convert.ts 1.0.0";

export function defaultThreads(): number {
  return Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
}

export function readJsonFile(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export function asObject(x: unknown, what: string): Json {
  if (typeof x !== "object" || x === null || Array.isArray(x)) throw new Error(`${what} must be a JSON object`);
  return x as Json;
}

export function intField(o: Json, key: string, what: string, fallback?: number): number {
  const v = o[key] ?? fallback;
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v < 1) throw new Error(`${what}: ${key} must be a positive integer`);
  return v;
}

export function numField(o: Json, key: string, what: string): number {
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`${what}: ${key} must be a finite number`);
  return v;
}

/** JSON text with object keys sorted: structural equality of parsed JSON. */
export function canonicalJson(x: unknown): string {
  if (Array.isArray(x)) return `[${x.map(canonicalJson).join(",")}]`;
  if (typeof x === "object" && x !== null) {
    const o = x as Json;
    return `{${Object.keys(o).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(o[k])}`).join(",")}}`;
  }
  return JSON.stringify(x) ?? "undefined";
}

/** temp + rename, so a reader never sees a half-written file. */
export function writeAtomic(path: string, data: Uint8Array | string): void {
  const tmp = `${path}.tmp`;
  try {
    writeFileSync(tmp, data);
    renameSync(tmp, path);
  } finally {
    rmSync(tmp, { force: true });
  }
}

export function copyLicense(source: string, output: string, name: string): string | null {
  const from = join(source, "LICENSE");
  if (!existsSync(from)) return null;
  const to = join(output, name);
  copyFileSync(from, to);
  return to;
}

export function report(paths: readonly (string | null)[], started: number): ConvertReport {
  const files = paths.filter((p): p is string => p !== null).map((path) => ({ path, bytes: statSync(path).size }));
  return { files, seconds: (performance.now() - started) / 1000 };
}
