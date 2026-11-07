#!/usr/bin/env bun
/**
 * Release files, built by .github/workflows/release.yml (and runnable by hand).
 *
 * usage: bun scripts/release.ts COMMAND ...
 *   package PLATFORM BIN OUT [--bundler PATH]
 *       OUT/smolmoe-PLATFORM[.exe]          the runtime
 *       OUT/qwen-pt-PLATFORM[.exe]          runtime + Qwen2.5-0.5B: Portuguese chat in one file
 *       OUT/tinystories-PLATFORM[.exe]      runtime + TinyStories 15M (English stories)
 *       OUT/smolmoe-tools-PLATFORM.tar.gz   smolmoe-bundle and smolmoe-quantize
 *     BIN holds a release build. --bundler is a smolmoe-bundle that runs on this machine, when BIN's
 *     cannot (an arm64 build on x64). The weights come from models/ (bun scripts/fetch-models.ts).
 *   smoke DIR PLATFORM
 *     runs each executable in DIR once; the Qwen chat must answer "Quanto é 7 mais 5?" with 12.
 *   models BIN OUT WORK
 *     single-file models (config and tokenizer embedded): Qwen Q8 and Q5 and TinyStories from
 *     models/, and Phi-4-mini converted from its pinned source in WORK, checked against
 *     models/phi-4-mini-instruct/provenance.json and split under GitHub's 2 GiB limit per file.
 *   sums DIR NAME
 *     DIR/SHA256SUMS.NAME for every file in DIR, in sha256sum format.
 *   notes VERSION
 *     the CHANGELOG section of VERSION, for the release page.
 */
import { createHash } from "node:crypto";
import { closeSync, copyFileSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, rmSync,
  statSync, writeFileSync, writeSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const MODELS = join(ROOT, "models");
const CHUNK = 8 << 20;
/** Below GitHub's 2 GiB limit per release file. */
export const PART_BYTES = 2_000_000_000;
export const PLATFORMS = ["windows-x64", "linux-x64", "linux-arm64", "macos"] as const;
export type Platform = (typeof PLATFORMS)[number];

const QWEN = "qwen2.5-0.5b-instruct";
const STORIES = "tinystories-15m";
const PHI = "phi-4-mini-instruct";

export const exeSuffix = (p: Platform): string => (p === "windows-x64" ? ".exe" : "");

function isPlatform(s: string | undefined): s is Platform {
  return (PLATFORMS as readonly string[]).includes(s ?? "");
}

/** Runs a program, failing loudly; stdout is returned, stderr goes to the log. */
function run(cmd: readonly string[], what: string, cwd?: string): string {
  const p = Bun.spawnSync([...cmd], { stdout: "pipe", stderr: "inherit", cwd });
  if (p.exitCode !== 0) throw new Error(`${what} failed (exit ${p.exitCode}): ${cmd.join(" ")}`);
  return p.stdout.toString();
}

/** Streams a file through SHA-256 (the Phi model is 4.3 GB). */
export function sha256File(path: string): string {
  const hash = createHash("sha256"), buf = Buffer.allocUnsafe(CHUNK), fd = openSync(path, "r");
  try {
    for (let n; (n = readSync(fd, buf, 0, CHUNK, null)) > 0;) hash.update(buf.subarray(0, n));
  } finally {
    closeSync(fd);
  }
  return hash.digest("hex");
}

/** "name.001", "name.002"...: `cat name.0* > name` joins them back. */
export function partNames(name: string, count: number): string[] {
  return Array.from({ length: count }, (_, i) => `${name}.${String(i + 1).padStart(3, "0")}`);
}

/** Splits `path` into OUT/<name>.001... of at most `partBytes`; returns the parts and the whole file's hash. */
export function splitFile(path: string, outDir: string, partBytes: number = PART_BYTES): { parts: string[]; sha256: string } {
  const size = statSync(path).size, names = partNames(basename(path), Math.max(1, Math.ceil(size / partBytes)));
  const hash = createHash("sha256"), buf = Buffer.allocUnsafe(Math.min(CHUNK, partBytes)), src = openSync(path, "r");
  try {
    for (const name of names) {
      const dst = openSync(join(outDir, name), "w");
      try {
        for (let left = partBytes, n; left > 0 && (n = readSync(src, buf, 0, Math.min(buf.length, left), null)) > 0;) {
          writeSync(dst, buf, 0, n);
          hash.update(buf.subarray(0, n));
          left -= n;
        }
      } finally {
        closeSync(dst);
      }
    }
  } finally {
    closeSync(src);
  }
  return { parts: names.map((n) => join(outDir, n)), sha256: hash.digest("hex") };
}

/** sha256sum format: "<hash>  <name>", sorted by name. */
export function sumsText(entries: ReadonlyMap<string, string>): string {
  return [...entries].sort(([a], [b]) => a.localeCompare(b)).map(([name, hash]) => `${hash}  ${name}\n`).join("");
}

/** The body of "## VERSION ..." in the CHANGELOG, up to the next "## " heading. A pre-release tag
 * (v0.3.0-rc1) uses the section of its version. */
export function changelogSection(text: string, version: string): string {
  const v = version.replace(/^v/, "").replace(/-.*$/, "");
  const lines = text.split("\n"), start = lines.findIndex((l) => l.startsWith(`## ${v} `) || l === `## ${v}`);
  if (start < 0) throw new Error(`CHANGELOG has no "## ${v}" section`);
  const end = lines.findIndex((l, i) => i > start && l.startsWith("## "));
  return lines.slice(start + 1, end < 0 ? undefined : end).join("\n").trim() + "\n";
}

const modelFiles = (dir: string, tokenizer: string) => {
  const base = join(MODELS, dir);
  return ["--model", join(base, "model.smoe"), "--config", join(base, "config.ini"), "--tokenizer", join(base, tokenizer)];
};

function packagePlatform(platform: Platform, binDir: string, outDir: string, bundler: string): void {
  const bin = resolve(binDir), out = resolve(outDir); /* tar below runs inside OUT */
  const x = exeSuffix(platform), runtime = join(bin, `smolmoe${x}`);
  mkdirSync(out, { recursive: true });
  copyFileSync(runtime, join(out, `smolmoe-${platform}${x}`));
  for (const [name, dir, tokenizer] of [["qwen-pt", QWEN, "tokenizer.qbp"], ["tinystories", STORIES, "tokenizer.spb"]]) {
    run([bundler, ...modelFiles(dir!, tokenizer!), "--exe", runtime, "--out", join(out, `${name}-${platform}${x}`)],
      `bundling ${name}`);
  }
  /* The archive name is relative: GNU tar on Windows would read "C:" as a host. BIN may be read-only
   * (a Docker build), so LICENSE comes from the repository root. */
  run(["tar", "-czf", `smolmoe-tools-${platform}.tar.gz`, "-C", bin, `smolmoe-bundle${x}`, `smolmoe-quantize${x}`,
    "-C", ROOT, "LICENSE"], "packing the tools", out);
}

/** Runs one program; on macOS also the x86_64 slice, through Rosetta when the machine has it. */
function runEverySlice(platform: Platform, exe: string, args: readonly string[]): string[] {
  const outs = [run([exe, ...args], basename(exe))];
  if (platform === "macos" && Bun.spawnSync(["arch", "-x86_64", "/usr/bin/true"]).exitCode === 0)
    outs.push(run(["arch", "-x86_64", exe, ...args], `${basename(exe)} (x86_64)`));
  return outs;
}

function smoke(dir: string, platform: Platform): void {
  const x = exeSuffix(platform), at = (name: string) => join(dir, `${name}-${platform}${x}`);
  for (const v of runEverySlice(platform, at("smolmoe"), ["--version"]))
    if (!/^smolmoe \d+\.\d+\.\d+/.test(v)) throw new Error(`unexpected --version: ${v}`);
  const question = ["--chat", "--prompt", "Quanto é 7 mais 5? Responda apenas com o número.", "--tokens", "16"];
  for (const answer of runEverySlice(platform, at("qwen-pt"), question))
    if (answer.trim() !== "12") throw new Error(`qwen-pt answered ${JSON.stringify(answer)}, not 12`);
  for (const story of runEverySlice(platform, at("tinystories"), ["--prompt", "Once upon a time", "--tokens", "8"]))
    if (!story.trim()) throw new Error("tinystories wrote nothing");
  console.log(`${platform}: smoke test passed`);
}

/** Phi-4-mini, converted from the pinned source and checked byte for byte against the versioned hashes. */
function convertPhi(work: string): string {
  const source = join(work, "phi-source"), out = join(work, "phi");
  const pinned = (JSON.parse(readFileSync(join(MODELS, PHI, "provenance.json"), "utf8")) as
    { output_sha256: Record<string, string> }).output_sha256;
  run(["bun", join(ROOT, "scripts", "convert.ts"), "fetch-phi3", source], "downloading Phi-4-mini");
  run(["bun", join(ROOT, "scripts", "convert.ts"), "phi3", source, out], "converting Phi-4-mini");
  rmSync(source, { recursive: true, force: true }); /* 7.7 GB: the disk is needed for the parts */
  for (const [file, want] of Object.entries(pinned)) {
    const got = sha256File(join(out, file));
    if (got !== want) throw new Error(`${PHI}/${file}: ${got}, provenance.json pins ${want}`);
  }
  return out;
}

function models(bin: string, out: string, work: string): void {
  const x = process.platform === "win32" ? ".exe" : "", bundler = join(bin, `smolmoe-bundle${x}`);
  const single = (files: readonly string[], name: string) => run([bundler, ...files, "--out", join(out, name)], name);
  mkdirSync(out, { recursive: true });
  mkdirSync(work, { recursive: true });
  single(modelFiles(QWEN, "tokenizer.qbp"), `${QWEN}-q8.smoe`);
  const q5 = join(work, "qwen-q5.smoe");
  run([join(bin, `smolmoe-quantize${x}`), "--in", join(MODELS, QWEN, "model.smoe"), "--out", q5], "quantizing to Q5");
  single(["--model", q5, ...modelFiles(QWEN, "tokenizer.qbp").slice(2)], `${QWEN}-q5.smoe`);
  single(modelFiles(STORIES, "tokenizer.spb"), `${STORIES}.smoe`);
  const phi = convertPhi(work), whole = join(work, `${PHI}.smoe`);
  run([bundler, "--model", join(phi, "model.smoe"), "--config", join(phi, "config.ini"), "--tokenizer",
    join(phi, "tokenizer.qbp"), "--out", whole], `${PHI}.smoe`);
  rmSync(phi, { recursive: true, force: true });
  const { sha256 } = splitFile(whole, out);
  rmSync(whole, { force: true });
  writeFileSync(join(work, "joined.sha256"), `${sha256}  ${PHI}.smoe\n`); /* "sums" adds it */
  for (const dir of [QWEN, PHI])
    for (const f of readdirSync(join(MODELS, dir)).filter((f) => f.startsWith("LICENSE")))
      copyFileSync(join(MODELS, dir, f), join(out, f));
  copyFileSync(join(ROOT, "THIRD_PARTY.md"), join(out, "THIRD_PARTY.md"));
}

function sums(dir: string, name: string, extra?: string): void {
  const entries = new Map<string, string>();
  for (const f of readdirSync(dir).filter((f) => !f.startsWith("SHA256SUMS"))) entries.set(f, sha256File(join(dir, f)));
  if (extra && existsSync(extra)) {
    for (const line of readFileSync(extra, "utf8").trim().split("\n")) {
      const [hash, file] = line.split(/\s+/);
      if (hash && file) entries.set(file, hash);
    }
  }
  writeFileSync(join(dir, `SHA256SUMS.${name}`), sumsText(entries));
}

function main(argv: readonly string[]): void {
  const [command, ...rest] = argv;
  const flag = (name: string) => {
    const i = rest.indexOf(name);
    return i >= 0 ? rest.splice(i, 2)[1] : undefined;
  };
  if (command === "package") {
    const bundler = flag("--bundler"), [platform, bin, out] = rest;
    if (!isPlatform(platform) || !bin || !out) throw new Error(`usage: package ${PLATFORMS.join("|")} BIN OUT`);
    packagePlatform(platform, bin, out, bundler ?? join(bin, `smolmoe-bundle${exeSuffix(platform)}`));
  } else if (command === "smoke") {
    const [dir, platform] = rest;
    if (!dir || !isPlatform(platform)) throw new Error(`usage: smoke DIR ${PLATFORMS.join("|")}`);
    smoke(dir, platform);
  } else if (command === "models") {
    const [bin, out, work] = rest;
    if (!bin || !out || !work) throw new Error("usage: models BIN OUT WORK");
    models(bin, out, work);
  } else if (command === "sums") {
    const extra = flag("--extra"), [dir, name] = rest;
    if (!dir || !name) throw new Error("usage: sums DIR NAME [--extra FILE]");
    sums(dir, name, extra);
  } else if (command === "notes") {
    if (!rest[0]) throw new Error("usage: notes VERSION");
    process.stdout.write(changelogSection(readFileSync(join(ROOT, "CHANGELOG.md"), "utf8"), rest[0]));
  } else {
    throw new Error("usage: bun scripts/release.ts package|smoke|models|sums|notes ...");
  }
}

if (import.meta.main) {
  try {
    main(Bun.argv.slice(2));
  } catch (e) {
    console.error(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}
