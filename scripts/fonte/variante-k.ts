/**
 * Variant K of the source test (plan, revision 34), on the regression only: the kit's model writes 3 to 6 English
 * keywords before the search.
 *   bun scripts/fonte/variante-k.ts --gerar --modelo models/<nome> [--modelo ...]   runs the model (heavy; CPU lock)
 *   bun scripts/fonte/variante-k.ts --medir                                         G against K, paired, per model
 * Fixed before measuring: K is G's query plus the model's keywords (K alone is shown, but does not decide). K is
 * adopted for a model if it covers, net and paired, 5 or more of the 44 more than G.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

import { consulta, Indice, tokens } from "./busca";
import { glossario, lerTrechos, regressao, sha256 } from "./indice";

const AQUI = import.meta.dir, ROOT = join(AQUI, "..", "..");
const SAIDA = join(ROOT, "build", "fonte");
const EXE = process.platform === "win32" ? ".exe" : "";
const PALAVRAS_MAX = 8, TOKENS = 32, CONTEXT = 1024, GANHO_MINIMO = 5;

type Gerado = { modelo: string; id: string; palavras: string; segundos: number; prompt_sha: string };

const arquivoDo = (modelo: string) => join(SAIDA, `k-${modelo}.jsonl`);
const lidos = (f: string): Gerado[] => existsSync(f)
  ? readFileSync(f, "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l)) : [];

async function gerar(dirs: readonly string[]): Promise<void> {
  const prompt = readFileSync(join(AQUI, "prompt-k.txt"), "utf8").trim(), promptSha = sha256(prompt);
  const binario = join(ROOT, process.env.SMOLMOE_BUILD ?? "build", `smolmoe${EXE}`);
  const itens = await regressao();
  mkdirSync(SAIDA, { recursive: true });
  for (const d of dirs) {
    const dir = resolve(d), modelo = basename(dir), f = arquivoDo(modelo);
    const feitos = lidos(f);
    if (feitos.some((g) => g.prompt_sha !== promptSha)) throw new Error(`${f} foi feito com outro prompt-k.txt`);
    const ja = new Set(feitos.map((g) => g.id));
    for (const it of itens.filter((i) => !ja.has(i.id))) {
      const t0 = performance.now();
      const r = Bun.spawnSync([binario, "--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"),
        "--tokenizer", join(dir, "tokenizer.qbp"), "--system", prompt, "--no-tools",
        "--prefix-cache", join(SAIDA, `k-${modelo}.kv`), "--context", String(CONTEXT), "--tokens", String(TOKENS),
        "--temperature", "0"], { stdin: new TextEncoder().encode(`${it.pergunta}\n/sair\n`), stdout: "pipe", stderr: "pipe" });
      if (r.exitCode !== 0) throw new Error(`${modelo} ${it.id}: o chat saiu com ${r.exitCode}: ${r.stderr.toString().slice(-300)}`);
      const g: Gerado = { modelo, id: it.id, palavras: r.stdout.toString().trim(), segundos: (performance.now() - t0) / 1000, prompt_sha: promptSha };
      appendFileSync(f, `${JSON.stringify(g)}\n`);
      console.error(`${modelo} ${it.id}: ${g.palavras.replace(/\s+/g, " ").slice(0, 80)} (${g.segundos.toFixed(1)} s)`);
    }
  }
}

/** The model's keywords as search terms: the first line that has words, at most PALAVRAS_MAX of them. */
export function palavrasDoModelo(texto: string): string[] {
  const linha = texto.split(/\r?\n/).find((l) => tokens(l).length > 0) ?? "";
  return tokens(linha).slice(0, PALAVRAS_MAX);
}

async function medir(): Promise<void> {
  const itens = await regressao(), g = await glossario(itens), indice = new Indice(await lerTrechos());
  const cobre = (it: (typeof itens)[number], termos: string[]) =>
    new RegExp(it.fato, "i").test(indice.buscar(termos).map((t) => t.texto).join("\n"));
  const G = new Map(itens.map((it) => [it.id, cobre(it, consulta(it.pergunta, g, indice))]));
  console.log(`G: ${[...G.values()].filter(Boolean).length}/${itens.length}`);
  const arquivos = [...new Bun.Glob("k-*.jsonl").scanSync({ cwd: SAIDA })].sort();
  if (arquivos.length === 0) throw new Error("nenhuma palavra gerada: rode antes --gerar");
  for (const a of arquivos) {
    const gerados = new Map(lidos(join(SAIDA, a)).map((x) => [x.id, x]));
    if (gerados.size !== itens.length) { console.log(`${a}: ${gerados.size}/${itens.length} gerados, incompleto`); continue; }
    let k = 0, so = 0, ganha = 0, perde = 0;
    for (const it of itens) {
      const extra = palavrasDoModelo(gerados.get(it.id)!.palavras);
      const comK = cobre(it, [...new Set([...consulta(it.pergunta, g, indice), ...extra])]);
      k += comK ? 1 : 0;
      so += cobre(it, extra) ? 1 : 0;
      if (comK && !G.get(it.id)) ganha++;
      if (!comK && G.get(it.id)) perde++;
    }
    const modelo = a.replace(/^k-|\.jsonl$/g, "");
    console.log(`${modelo}: K (G + palavras) ${k}/${itens.length}; K sozinha ${so}/${itens.length}; ` +
      `pareado: K cobre e G não ${ganha}, G cobre e K não ${perde}, ganho ${ganha - perde} ` +
      `-> ${ganha - perde >= GANHO_MINIMO ? "adota a K" : "fica a G"}`);
  }
}

const a = process.argv.slice(2);
if (a[0] === "--gerar") {
  const dirs = a.flatMap((x, i) => (a[i - 1] === "--modelo" ? [x] : []));
  if (dirs.length === 0) throw new Error("passe ao menos um --modelo DIR");
  await gerar(dirs);
} else if (a[0] === "--medir") await medir();
else { console.error("uso: variante-k.ts --gerar --modelo DIR [...] | --medir"); process.exit(2); }
