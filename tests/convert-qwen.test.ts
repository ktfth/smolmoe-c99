/**
 * Golden: converting the pinned Qwen2.5-0.5B-Instruct source with the TS
 * converter reproduces the shipped files byte for byte (SHA-256 of
 * models/manifest.json, versioned config.ini / provenance.json / licence).
 * Runs only when the source is present: `bun scripts/convert.ts fetch-qwen2
 * downloads/qwen2.5-0.5b-instruct-src` (or SMOLMOE_QWEN_SOURCE=DIR).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { sha256File } from "../scripts/hf/hash";
import { QWEN2_PINS } from "../scripts/hf/qwen2";
import { BINARY, BUNDLE_BINARY, ROOT, run, runOk, tempDir } from "./support/run";

const SOURCE = resolve(ROOT, process.env.SMOLMOE_QWEN_SOURCE ?? join("downloads", "qwen2.5-0.5b-instruct-src"));
const SHIPPED = join(ROOT, "models", "qwen2.5-0.5b-instruct");
const present = existsSync(join(SOURCE, "model.safetensors")) && existsSync(join(SOURCE, "tokenizer.json"));

interface ManifestFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}
const manifest = JSON.parse(readFileSync(join(ROOT, "models", "manifest.json"), "utf8")) as { files: ManifestFile[] };
const golden = (name: string) => manifest.files.find((f) => f.path === `qwen2.5-0.5b-instruct/${name}`)!;

const convert = (...args: string[]) =>
  Bun.spawnSync(["bun", join(ROOT, "scripts", "convert.ts"), ...args], { stdout: "pipe", stderr: "pipe" });

describe.skipIf(!present)("Qwen2.5-0.5B-Instruct golden conversion", () => {
  const work = tempDir("convert-qwen");
  const out = join(work.path, "out");
  afterAll(() => work.dispose());

  /* Runs first (tests run in order); the rest compare its output. */
  test("bun scripts/convert.ts qwen2 converts the pinned source", () => {
    const r = convert("qwen2", SOURCE, out);
    expect(r.stderr.toString()).toContain("done in");
    expect(r.exitCode).toBe(0);
  }, 600_000);

  test.each(["model.smoe", "tokenizer.qbp"])("%s matches the manifest SHA-256", (name) => {
    const g = golden(name);
    expect(readFileSync(join(out, name)).length).toBe(g.bytes);
    expect(sha256File(join(out, name))).toBe(g.sha256);
  });

  test.each(["config.ini", "provenance.json", "LICENSE-Qwen.txt"])("%s equals the versioned file", (name) => {
    expect(readFileSync(join(out, name)).equals(readFileSync(join(SHIPPED, name)))).toBe(true);
  });

  test("the TS checker accepts the output", () => {
    const r = convert("check", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"));
    expect(r.exitCode).toBe(0);
    expect(JSON.parse(r.stdout.toString())).toMatchObject({ version: 2, records: 315, blobs: 0,
      tokenizer: { kind: "SMQBP1", version: 1, merges: 151387 } });
  });

  test.skipIf(!existsSync(BINARY))("the runtime loads the converted model", () => {
    const r = runOk(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"), "--prompt", "Olá", "--tokens", 2, "--context", 64, "--temperature", 0]);
    expect(r.stats.generated_tokens).toBeGreaterThan(0);
  }, 120_000);

  test.skipIf(!existsSync(BUNDLE_BINARY))("the TS single-file bundle equals smolmoe-bundle's", () => {
    const mine = join(work.path, "mine.smoe"), theirs = join(work.path, "theirs.smoe");
    const files = [join(out, "model.smoe"), join(out, "config.ini"), join(out, "tokenizer.qbp")];
    expect(convert("bundle", ...files, mine).exitCode).toBe(0);
    expect(run(["--model", files[0]!, "--config", files[1]!, "--tokenizer", files[2]!, "--out", theirs], BUNDLE_BINARY).code).toBe(0);
    expect(sha256File(mine)).toBe(sha256File(theirs));
  }, 120_000);
});

/* Qwen2.5-1.5B-Instruct: no shipped golden, so the checks are the pin (provenance), the TS checker, the runtime
 * and the NLL below. About a minute of CPU, so only on request: SMOLMOE_QWEN15_SOURCE=DIR, the source from
 * `bun scripts/convert.ts fetch-qwen2 downloads/qwen2.5-1.5b-instruct-src --size 1.5b` (3.1 GB). */
const SOURCE15 = process.env.SMOLMOE_QWEN15_SOURCE ? resolve(ROOT, process.env.SMOLMOE_QWEN15_SOURCE) : "";
const present15 = SOURCE15 !== "" && existsSync(join(SOURCE15, "model.safetensors")) &&
  existsSync(join(SOURCE15, "tokenizer.json"));

describe.skipIf(!present15)("Qwen2.5-1.5B-Instruct conversion", () => {
  const work = tempDir("convert-qwen15");
  const out = join(work.path, "out");
  afterAll(() => work.dispose());

  test("bun scripts/convert.ts qwen2 picks the 1.5B pin from config.json", () => {
    const r = convert("qwen2", SOURCE15, out);
    expect(r.stderr.toString()).toContain("done in");
    expect(r.exitCode).toBe(0);
    expect(JSON.parse(readFileSync(join(out, "provenance.json"), "utf8"))).toMatchObject({
      model: QWEN2_PINS["1.5b"].repo, revision: QWEN2_PINS["1.5b"].revision, source_sha256: QWEN2_PINS["1.5b"].modelSha256 });
    /* Same tokenizer file as the 0.5B, so the same tokenizer.qbp. */
    expect(sha256File(join(out, "tokenizer.qbp"))).toBe(golden("tokenizer.qbp").sha256);
  }, 1_200_000);

  test("the TS checker accepts the output", () => {
    const r = convert("check", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"));
    expect(r.exitCode).toBe(0);
    expect(JSON.parse(r.stdout.toString())).toMatchObject({ version: 2, blobs: 0, config: { dim: 1536, layers: 28 } });
  });

  test.skipIf(!existsSync(BINARY))("the runtime loads the converted model", () => {
    const r = runOk(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"), "--prompt", "Olá", "--tokens", 2, "--context", 64, "--temperature", 0]);
    expect(r.stats.generated_tokens).toBeGreaterThan(0);
  }, 300_000);

  /* Loading and generating prove nothing about the weights: shuffled rows or a wrong QK permutation at head_dim
   * 128 (the 0.5B golden only covers 64) load and generate too. A correct 1.5B reads a fixed pt-BR text better
   * than the shipped 0.5B (same tokenizer, so the same tokens), and greedy continues a plain sequence. (A bare
   * "A capital do Brasil é", without the chat template, is no test: the instruct model wanders.) */
  test.skipIf(!existsSync(BINARY) || !existsSync(join(SHIPPED, "model.smoe")))(
    "reads a fixed pt-BR text with a lower NLL than the 0.5B, and greedy continues the weekdays", () => {
      const score = (dir: string) => runOk(["--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"),
        "--tokenizer", join(dir, "tokenizer.qbp"), "--prompt", MAINTENANCE_TEXT, "--context", 512, "--perplexity"]).stats;
      const small = score(SHIPPED), big = score(out);
      expect(big.perplexity_tokens).toBe(small.perplexity_tokens);
      expect(small.perplexity_tokens).toBeGreaterThan(100);
      expect(big.nll).toBeLessThan(small.nll);
      const r = runOk(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
        join(out, "tokenizer.qbp"), "--prompt", "Segunda-feira, terça-feira, quarta-feira,", "--tokens", 4, "--context", 64,
        "--temperature", 0]);
      expect(r.stdout.toString()).toMatch(/^\s*quinta-feira/);
    }, 600_000);
});

const MAINTENANCE_TEXT =
  "Quando o computador fica lento, o primeiro passo é ver quanto espaço livre resta no disco. No Windows, abra o " +
  "Explorador de Arquivos e veja a unidade C:. No Linux, o comando df -h mostra o uso de cada partição. Se o disco " +
  "estiver quase cheio, apague arquivos temporários, esvazie a lixeira e remova programas que você não usa mais. " +
  "Depois, confira se há atualizações pendentes do sistema, porque elas costumam corrigir falhas de segurança e " +
  "problemas de desempenho. Se um serviço não iniciar, leia o registro de eventos e procure a primeira mensagem de " +
  "erro, que quase sempre aponta a causa. Antes de mudar qualquer configuração importante, faça uma cópia de " +
  "segurança dos seus documentos em um disco externo.";
