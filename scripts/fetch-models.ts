#!/usr/bin/env bun
/**
 * Downloads the converted weights listed in models/manifest.json from the
 * project's GitHub release, verifies size and SHA-256, and only then moves each
 * file into models/<model>/. Files already present and verified are skipped.
 *
 * usage: bun scripts/fetch-models.ts [MODEL...]        (default: every model)
 *   SMOLMOE_MODELS_REPO      owner/name holding the release (default: the origin remote)
 *   GH_TOKEN / GITHUB_TOKEN  token for a private repo (default: `gh auth token`)
 */
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, renameSync, rmSync, statSync, writeSync }
  from "node:fs";
import { dirname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const API = "https://api.github.com";
const IDLE_MS = 60_000; /* abort a download that stops sending data */
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;

export interface ModelAsset {
  readonly asset: string; /* file name in the release */
  readonly path: string; /* destination below models/, "/"-separated */
  readonly bytes: number;
  readonly sha256: string;
}

export interface Manifest {
  readonly tag: string;
  readonly files: readonly ModelAsset[];
}

export interface FetchReport {
  readonly fetched: readonly string[];
  readonly present: readonly string[];
}

/** Writes the asset's bytes to `dest`; fetchModels verifies them afterwards. */
export type Download = (asset: ModelAsset, dest: string) => Promise<void>;

/* One path segment or name: no ".", "..", hidden files or trailing dots. */
const isSafeName = (s: string) => NAME.test(s) && !s.endsWith(".");
const isRepo = (s: string) => s.split("/").length === 2 && s.split("/").every(isSafeName);
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function repoFromRemote(url: string): string | null {
  const m = url.trim().match(/^(?:https:\/\/github\.com\/|git@github\.com:)([^/]+\/[^/]+?)(?:\.git)?\/?$/);
  return m && isRepo(m[1]!) ? m[1]! : null;
}

function parseAsset(x: unknown, i: number): ModelAsset {
  const f = (x ?? {}) as Record<string, unknown>;
  const bad = (what: string) => new Error(`manifest: invalid ${what} in files[${i}]`);
  if (typeof f.asset !== "string" || !isSafeName(f.asset)) throw bad("asset");
  if (typeof f.path !== "string" || !f.path.split("/").every(isSafeName)) throw bad("path");
  if (typeof f.bytes !== "number" || !Number.isSafeInteger(f.bytes) || f.bytes <= 0) throw bad("bytes");
  if (typeof f.sha256 !== "string" || !SHA256.test(f.sha256)) throw bad("sha256");
  return { asset: f.asset, path: f.path, bytes: f.bytes, sha256: f.sha256 };
}

export function parseManifest(json: unknown): Manifest {
  const m = (json ?? {}) as Record<string, unknown>;
  if (typeof m.tag !== "string" || !isSafeName(m.tag)) throw new Error("manifest: invalid tag");
  if (!Array.isArray(m.files)) throw new Error("manifest: files must be an array");
  return { tag: m.tag, files: m.files.map(parseAsset) };
}

/* Fixed 1 MiB reads: Bun.file().stream() buffered a 556 MB file whole (929 MiB private, 8.3 s). */
export function sha256File(path: string): string {
  const hasher = new Bun.CryptoHasher("sha256");
  const buf = Buffer.allocUnsafe(1 << 20);
  const fd = openSync(path, "r");
  try {
    for (let n; (n = readSync(fd, buf, 0, buf.length, null)) > 0;) hasher.update(buf.subarray(0, n));
  } finally {
    closeSync(fd);
  }
  return hasher.digest("hex");
}

/** Why `path` is not the expected file, or null when it is. */
function mismatch(path: string, f: ModelAsset): string | null {
  const size = statSync(path).size;
  if (size !== f.bytes) return `size ${size}, expected ${f.bytes}`;
  const digest = sha256File(path);
  return digest === f.sha256 ? null : `SHA-256 ${digest}, expected ${f.sha256}`;
}

const modelOf = (f: ModelAsset) => f.path.split("/")[0]!;

function select(manifest: Manifest, only: readonly string[]): readonly ModelAsset[] {
  if (only.length === 0) return manifest.files;
  const known = new Set(manifest.files.map(modelOf));
  const unknown = only.filter((m) => !known.has(m));
  if (unknown.length) throw new Error(`unknown model ${unknown.join(", ")} (known: ${[...known].join(", ")})`);
  return manifest.files.filter((f) => only.includes(modelOf(f)));
}

/** Downloads to `${dest}.download`, verifies, then renames; the partial file never survives. */
async function fetchOne(f: ModelAsset, dest: string, download: Download): Promise<void> {
  const partial = `${dest}.download`;
  try {
    await download(f, partial);
    const problem = mismatch(partial, f);
    if (problem) throw new Error(`${f.asset}: ${problem}`);
    try {
      renameSync(partial, dest);
    } catch (e) {
      throw new Error(`cannot replace ${dest} (is smolmoe running with it?): ${message(e)}`);
    }
  } finally {
    try {
      rmSync(partial, { force: true });
    } catch (e) {
      console.error(`fetch-models: could not remove ${partial}: ${message(e)}`); /* keep the original error */
    }
  }
}

export async function fetchModels(manifest: Manifest, root: string, download: Download,
  only: readonly string[] = [], log: (line: string) => void = () => {}): Promise<FetchReport> {
  const fetched: string[] = [];
  const present: string[] = [];
  for (const f of select(manifest, only)) {
    const dest = join(root, ...f.path.split("/"));
    if (existsSync(dest) && mismatch(dest, f) === null) {
      present.push(f.path);
      continue;
    }
    mkdirSync(dirname(dest), { recursive: true });
    log(`downloading ${f.path} (${(f.bytes / 1048576).toFixed(1)} MiB)`);
    await fetchOne(f, dest, download);
    fetched.push(f.path);
  }
  return { fetched, present };
}

/* ---- GitHub release download ------------------------------------------- */

interface ReleaseAsset {
  readonly name: string;
  readonly url: string;
}

/** An AbortSignal that fires after `ms` without a reset(). */
function idleTimer(ms: number) {
  const controller = new AbortController();
  const arm = () => setTimeout(() => controller.abort(new Error(`no data for ${ms / 1000} s`)), ms);
  let timer = arm();
  return {
    signal: controller.signal,
    reset: () => { clearTimeout(timer); timer = arm(); },
    stop: () => clearTimeout(timer),
  };
}

/* Synchronous writes: the stream is pulled no faster than the disk accepts it. */
async function saveBody(body: ReadableStream<Uint8Array>, dest: string, onChunk: () => void): Promise<void> {
  const fd = openSync(dest, "w");
  try {
    for await (const chunk of body) {
      onChunk();
      for (let off = 0; off < chunk.length;) off += writeSync(fd, chunk, off);
    }
  } finally {
    closeSync(fd);
  }
}

export async function githubDownload(repo: string, tag: string, token: string | undefined,
  api: string = API): Promise<Download> {
  const headers: Record<string, string> = { "X-GitHub-Api-Version": "2022-11-28" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${api}/repos/${repo}/releases/tags/${tag}`,
    { headers: { ...headers, Accept: "application/vnd.github+json" }, signal: AbortSignal.timeout(IDLE_MS) });
  if (!res.ok) {
    const hint = token ? "" : " (private repo? set GH_TOKEN or run `gh auth login`)";
    throw new Error(`release ${tag} of ${repo}: HTTP ${res.status}${hint}`);
  }
  const assets = ((await res.json()) as { assets: ReleaseAsset[] }).assets;
  const protocol = new URL(api).protocol;
  return async (asset, dest) => {
    const found = assets.find((a) => a.name === asset.asset);
    if (!found) throw new Error(`asset ${asset.asset} is missing from release ${tag} of ${repo}`);
    const idle = idleTimer(IDLE_MS);
    try {
      /* The API redirects to signed storage, which must not receive the token. */
      const hop = await fetch(found.url,
        { headers: { ...headers, Accept: "application/octet-stream" }, redirect: "manual", signal: idle.signal });
      const location = hop.headers.get("location");
      let body = hop;
      if (location) {
        await hop.body?.cancel();
        const target = new URL(location, found.url);
        if (target.protocol !== protocol) throw new Error(`${asset.asset}: refusing redirect to ${target.protocol}`);
        body = await fetch(target, { signal: idle.signal });
      }
      if (!body.ok || !body.body) throw new Error(`${asset.asset}: HTTP ${body.status}`);
      await saveBody(body.body, dest, idle.reset);
    } finally {
      idle.stop();
    }
  };
}

function githubToken(): string | undefined {
  const env = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
  if (env) return env;
  try {
    const gh = Bun.spawnSync(["gh", "auth", "token"], { stderr: "ignore" });
    return gh.exitCode === 0 ? gh.stdout.toString().trim() || undefined : undefined;
  } catch {
    return undefined; /* gh is not installed */
  }
}

function releaseRepo(): string {
  const env = process.env.SMOLMOE_MODELS_REPO;
  if (env) {
    if (!isRepo(env)) throw new Error(`SMOLMOE_MODELS_REPO must be owner/name, got "${env}"`);
    return env;
  }
  let remote = "";
  try {
    const git = Bun.spawnSync(["git", "remote", "get-url", "origin"], { cwd: ROOT, stderr: "ignore" });
    if (git.exitCode === 0) remote = git.stdout.toString();
  } catch {
    /* git is not installed */
  }
  const repo = repoFromRemote(remote);
  if (!repo) throw new Error("cannot tell which GitHub repo holds the weights; set SMOLMOE_MODELS_REPO=owner/name");
  return repo;
}

/** Defers the release lookup until a file is actually missing. */
function lazy(make: () => Promise<Download>): Download {
  let ready: Promise<Download> | undefined;
  return async (asset, dest) => (await (ready ??= make()))(asset, dest);
}

async function main(): Promise<void> {
  const manifest = parseManifest(JSON.parse(readFileSync(join(ROOT, "models", "manifest.json"), "utf8")));
  const download = lazy(() => githubDownload(releaseRepo(), manifest.tag, githubToken()));
  const report = await fetchModels(manifest, join(ROOT, "models"), download, Bun.argv.slice(2), console.error);
  console.error(`${report.fetched.length} downloaded, ${report.present.length} already present and verified`);
}

if (import.meta.main) {
  try {
    await main();
  } catch (e) {
    console.error(`fetch-models: ${message(e)}`);
    process.exit(1);
  }
}
