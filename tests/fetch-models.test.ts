/**
 * scripts/fetch-models.ts: downloads the converted weights from a GitHub
 * release. The network is replaced by a fake downloader; what matters is that
 * nothing unverified ever lands at the final path.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { type Download, fetchModels, githubDownload, type Manifest, type ModelAsset, parseManifest, repoFromRemote }
  from "../scripts/fetch-models";
import { tempDir } from "./support/run";

const ROOT = resolve(import.meta.dir, "..");
const work = tempDir("fetch");
afterAll(() => work.dispose());

const sha256 = (data: Uint8Array) => createHash("sha256").update(data).digest("hex");

function asset(path: string, content: string): { asset: ModelAsset; bytes: Uint8Array } {
  const bytes = new TextEncoder().encode(content);
  return { asset: { asset: path.replace("/", "."), path, bytes: bytes.length, sha256: sha256(bytes) }, bytes };
}

const weights = asset("tiny/model.smoe", "pesos de teste");
const vocab = asset("tiny/tokenizer.bpe", "tokenizer");
const other = asset("other/model.smoe", "outro modelo");
const manifest: Manifest = { tag: "models-test", files: [weights.asset, vocab.asset, other.asset] };
const content = new Map([weights, vocab, other].map((x) => [x.asset.asset, x.bytes]));

/** Serves the expected bytes (or `override`) and records every call. */
function fakeDownload(override?: Uint8Array): { download: Download; calls: string[] } {
  const calls: string[] = [];
  const download: Download = async (a, dest) => {
    calls.push(a.asset);
    writeFileSync(dest, override ?? content.get(a.asset)!);
  };
  return { download, calls };
}

describe("repoFromRemote", () => {
  test("reads owner/name from HTTPS and SSH GitHub remotes", () => {
    expect(repoFromRemote("https://github.com/ktfth/smolmoe-c99.git")).toBe("ktfth/smolmoe-c99");
    expect(repoFromRemote("https://github.com/ktfth/smolmoe-c99\n")).toBe("ktfth/smolmoe-c99");
    expect(repoFromRemote("git@github.com:ktfth/smolmoe-c99.git")).toBe("ktfth/smolmoe-c99");
  });

  test("other hosts are not guessed", () => {
    expect(repoFromRemote("https://gitlab.com/a/b.git")).toBeNull();
    expect(repoFromRemote("")).toBeNull();
  });

  test("traversal in the owner or name is refused", () => {
    expect(repoFromRemote("https://github.com/../x")).toBeNull();
    expect(repoFromRemote("https://github.com/a/..")).toBeNull();
    expect(repoFromRemote("git@github.com:./b.git")).toBeNull();
  });
});

describe("parseManifest", () => {
  const good = { tag: "t", files: [weights.asset] };

  test("accepts a well-formed manifest", () => {
    expect(parseManifest(good)).toEqual(good);
  });

  test("rejects paths that escape models/, bad hashes and bad sizes", () => {
    const bad = (patch: Partial<ModelAsset>) => () => parseManifest({ tag: "t", files: [{ ...weights.asset, ...patch }] });
    expect(bad({ path: "../escape.smoe" })).toThrow(/path/);
    expect(bad({ path: "/abs/model.smoe" })).toThrow(/path/);
    expect(bad({ path: "C:\\abs\\model.smoe" })).toThrow(/path/);
    for (const path of ["a/../b", "a//b", "a/.", "", "a/...", "tiny/model.smoe."]) expect(bad({ path })).toThrow(/path/);
    expect(bad({ asset: "../x" })).toThrow(/asset/);
    expect(bad({ sha256: "abc" })).toThrow(/sha256/);
    expect(bad({ bytes: 1.5 })).toThrow(/bytes/);
    expect(bad({ bytes: 0 })).toThrow(/bytes/);
    expect(() => parseManifest({ files: [] })).toThrow(/tag/);
    expect(() => parseManifest({ tag: "../t", files: [] })).toThrow(/tag/);
  });

  test("the committed manifest matches the hashes documented in docs/MODELS.md", () => {
    const committed = parseManifest(JSON.parse(readFileSync(join(ROOT, "models", "manifest.json"), "utf8")));
    const docs = readFileSync(join(ROOT, "docs", "MODELS.md"), "utf8");
    const documented = new Map([...docs.matchAll(/^\| `([^`]+)` \| `([0-9a-f]{64})` \|$/gm)].map((m) => [m[1], m[2]]));
    expect(committed.files.length).toBeGreaterThan(0);
    expect(new Map(committed.files.map((f) => [f.path, f.sha256]))).toEqual(documented);
  });
});

describe("fetchModels", () => {
  test("downloads missing files, then finds them verified and skips the network", async () => {
    const root = join(work.path, "fresh");
    const first = fakeDownload();
    expect(await fetchModels(manifest, root, first.download)).toEqual({
      fetched: ["tiny/model.smoe", "tiny/tokenizer.bpe", "other/model.smoe"], present: [],
    });
    expect(readFileSync(join(root, "tiny", "model.smoe"))).toEqual(Buffer.from(weights.bytes));

    const second = fakeDownload();
    const report = await fetchModels(manifest, root, second.download);
    expect(second.calls).toEqual([]);
    expect(report.present).toHaveLength(3);
  });

  test("a corrupt download fails and leaves neither the file nor a partial copy", async () => {
    const root = join(work.path, "corrupt");
    const { download } = fakeDownload(new TextEncoder().encode("pesos de tesTE"));
    await expect(fetchModels(manifest, root, download)).rejects.toThrow(/SHA-256/);
    expect(existsSync(join(root, "tiny", "model.smoe"))).toBe(false);
    expect(readdirSync(join(root, "tiny"))).toEqual([]);
  });

  test("a truncated download is reported as a size mismatch", async () => {
    const root = join(work.path, "short");
    const { download } = fakeDownload(new TextEncoder().encode("pesos"));
    await expect(fetchModels(manifest, root, download)).rejects.toThrow(/size/);
  });

  test("an interrupted download keeps its own error and leaves no partial copy", async () => {
    const root = join(work.path, "interrupted");
    const download: Download = async (_a, dest) => {
      writeFileSync(dest, "pes");
      throw new Error("connection reset");
    };
    await expect(fetchModels(manifest, root, download)).rejects.toThrow(/connection reset/);
    expect(readdirSync(join(root, "tiny"))).toEqual([]);
  });

  test("a stale local file of the same size is caught by its hash and replaced", async () => {
    const root = join(work.path, "stale");
    const dest = join(root, "tiny", "model.smoe");
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, "versão antiga");
    expect(Buffer.byteLength("versão antiga")).toBe(weights.asset.bytes);
    const { download, calls } = fakeDownload();
    await fetchModels(manifest, root, download, ["tiny"]);
    expect(calls).toEqual(["tiny.model.smoe", "tiny.tokenizer.bpe"]);
    expect(readFileSync(dest)).toEqual(Buffer.from(weights.bytes));
  });

  test("only the requested models are fetched, and unknown names are refused", async () => {
    const root = join(work.path, "only");
    const { download, calls } = fakeDownload();
    await fetchModels(manifest, root, download, ["other"]);
    expect(calls).toEqual(["other.model.smoe"]);
    await expect(fetchModels(manifest, root, download, ["nope"])).rejects.toThrow(/nope/);
  });
});

describe("githubDownload", () => {
  interface Seen { readonly host: "api" | "storage"; readonly path: string; readonly auth: string | null }
  const seen: Seen[] = [];
  const record = (host: Seen["host"], req: Request) =>
    seen.push({ host, path: new URL(req.url).pathname, auth: req.headers.get("authorization") });
  const storage = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(req) {
    record("storage", req);
    return new Response(weights.bytes);
  } });
  const api = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(req) {
    record("api", req);
    const { origin, pathname } = new URL(req.url);
    if (pathname === "/repos/o/r/releases/tags/models-test") {
      return Response.json({ assets: [{ name: weights.asset.asset, url: `${origin}/assets/1` },
        { name: "ftp.smoe", url: `${origin}/assets/2` }] });
    }
    if (pathname === "/assets/1") return new Response(null, { status: 302, headers: { location: `${storage.url.origin}/blob?sig=1` } });
    if (pathname === "/assets/2") return new Response(null, { status: 302, headers: { location: "ftp://example.com/x" } });
    return new Response("not found", { status: 404 });
  } });
  afterAll(() => {
    api.stop(true);
    storage.stop(true);
  });

  test("the token goes only to the API; the signed redirect is followed without it", async () => {
    seen.length = 0;
    const dest = join(work.path, "gh.smoe");
    const download = await githubDownload("o/r", "models-test", "tok", api.url.origin);
    await download(weights.asset, dest);
    expect(readFileSync(dest)).toEqual(Buffer.from(weights.bytes));
    expect(seen.filter((s) => s.host === "api")).toEqual([
      { host: "api", path: "/repos/o/r/releases/tags/models-test", auth: "Bearer tok" },
      { host: "api", path: "/assets/1", auth: "Bearer tok" },
    ]);
    expect(seen.filter((s) => s.host === "storage")).toEqual([{ host: "storage", path: "/blob", auth: null }]);
  });

  test("a missing asset, a redirect to another protocol and a missing release are errors", async () => {
    const dest = join(work.path, "gh-bad.smoe");
    const download = await githubDownload("o/r", "models-test", "tok", api.url.origin);
    await expect(download({ ...weights.asset, asset: "absent.smoe" }, dest)).rejects.toThrow(/missing/);
    await expect(download({ ...weights.asset, asset: "ftp.smoe" }, dest)).rejects.toThrow(/refusing redirect/);
    await expect(githubDownload("o/r", "nope", undefined, api.url.origin)).rejects.toThrow(/HTTP 404.*gh auth login/);
  });
});
