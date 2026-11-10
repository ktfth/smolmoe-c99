#!/usr/bin/env bun
/**
 * The maintenance eval (plan, round 2, proposal 1): asks every item of the
 * frozen decision set to the real chat of each model, as the maintenance kit
 * asks it (its prompt, the machine role line, the date line, no tools), and
 * grades the answer with the set's nota.ts, unedited. One chat process per
 * item; the system prompt is read once per model and prompt variant through
 * --prefix-cache.
 *
 * usage: bun scripts/eval-manutencao.ts --modelo models/qwen2.5-0.5b-instruct [--modelo DIR ...]
 *          [--saida resultados.jsonl] [--itens cmd-001,pap-*] [--piloto] [--resumo-so]
 *
 * Nothing runs unless the sets present, the Portuguese detector, the kit prompt
 * and the date line are the ones frozen in docs/DECISIONS.md (CONJUNTOS and
 * FROZEN below; a test ties them). Items of a command family with no command
 * list yet (pacman, brew) wait for the set's v2 and are left out. Results go to
 * a JSONL file, one line per model and item, with the freezes and the
 * model.smoe SHA-256 they were made with; resuming skips the pairs already
 * there and refuses a file made under other conditions. The summary gives each
 * rate with its 95% Wilson interval and each pair of models with the paired
 * sign test, both ways.
 */
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";

import type { ItemNota, Nota, Regras } from "../tests/evals/manutencao/nota";
import { pareado, wilson } from "./eval/estatistica";
import { emPortugues } from "./eval/portugues";

const ROOT = resolve(import.meta.dir, "..");
const SET = join(ROOT, "tests", "evals", "manutencao");
export const DETECTOR = join(ROOT, "scripts", "eval", "portugues.ts");
export const KIT_PROMPT = join(ROOT, "kit", "manutencao.txt");
export const KIT_DATE = join(ROOT, "kit", "linha-data.txt");
const EXE = process.platform === "win32" ? ".exe" : "";
export const NOW = "2026-11-10T10:00"; /* the clock of the date line, fixed for reproducible answers */
const CONTEXT = 4096;
const TOKENS = 256;
const TIMEOUT_MS = 15 * 60_000;

/** The adversary's sets, each a JSON with its own manifest, whose SHA-256 docs/DECISIONS.md records, and the module
 * of the set that grades it (unedited; it is in the manifest). A set not copied into tests/evals/manutencao/ yet is
 * left out; one that is there must be the frozen one (sha null: not frozen yet, so it may not run). */
export interface Conjunto {
  readonly itens: string;
  readonly manifesto: string;
  readonly sha: string | null;
  readonly nota: string;
}
export const CONJUNTOS: readonly Conjunto[] = [
  { itens: "manutencao.json", manifesto: "MANIFEST.sha256", nota: "nota.ts",
    sha: "f4caf5b2cde9f1fdfeb92ff3d451fa12b2d4054fdd3ef59a2fe7a0b447a40b32" },
  { itens: "duvidas.json", manifesto: "MANIFEST-duvidas.sha256", nota: "nota-duvidas.ts",
    sha: "4f16d072d2d8aa0b4838b292f1a78959fbbc6a9da1c02d1f8c9c04ac2e561282" },
];

export type Avaliar = (item: ItemNota, resposta: string, regras: Regras) => Nota;

/** The 11 items of the pilots that chose the kit prompt and dropped the tools (DECISIONS): seen before the run. */
export const ITENS_DO_PILOTO: ReadonlySet<string> = new Set([
  "cmd-001", "cmd-002", "cmd-003", "cmd-004", "cmd-005", "cmd-006", "cmd-007", "cmd-008", "cmd-009", "pap-001", "pap-002",
]);

/** The grading function of each set present, by its items file; each module is imported only when its set is there. */
export async function carregarAvaliadores(dir = SET): Promise<Map<string, Avaliar>> {
  const avaliadores = new Map<string, Avaliar>();
  for (const c of conjuntosPresentes(dir)) {
    const modulo = (await import(join(dir, c.nota))) as { avaliar?: Avaliar };
    if (typeof modulo.avaliar !== "function") throw new Error(`${c.nota}: não exporta avaliar`);
    avaliadores.set(c.itens, modulo.avaliar);
  }
  return avaliadores;
}

/** The dev's frozen files, also in docs/DECISIONS.md; changing any of them means a new freeze there and here. */
export const FROZEN = {
  detector: "49b3cb218ff729ce7f5636036059a915673f8a48aa4774546cc046b2febdd7e7",
  sistema: "860d7bfd0350be7e08c768e605a383635caa4a1b119a58f193a5cbd2e64b1582",
  linhaData: "29782c739b156c66809b54220db37f3726acb3e25773b0466dfea1c882546386",
} as const;

export interface Item extends ItemNota {
  readonly bloco: string;
  readonly pergunta: string;
  readonly tokens?: number;
}

export interface Resultado {
  readonly modelo: string;
  readonly id: string;
  readonly bloco: string;
  readonly certa: boolean;
  readonly motivo: string | null;
  readonly segundos: number;
  readonly resposta: string;
  readonly manifesto: string;              /* the frozen SHA-256 of the item's set */
  readonly congelado?: typeof FROZEN;      /* the dev's frozen files the answer was asked and graded with */
  readonly modelo_sha?: string;            /* SHA-256 of the model.smoe that answered */
  readonly piloto?: boolean;
}

export const sha = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");

/** SHA-256 of a file of any size (a model.smoe has GBs), read in pieces. */
export async function shaDeArquivo(path: string): Promise<string> {
  const h = new Bun.CryptoHasher("sha256");
  for await (const parte of Bun.file(path).stream()) h.update(parte);
  return h.digest("hex");
}

/**
 * Resuming may only add to answers given under the same conditions: an existing line of the same model label
 * with another model.smoe, another freeze of the dev's files, another set freeze for its item, or another pilot
 * flag would mix two runs in one summary. Returns the model/item pairs already done.
 */
export function conferirRetomada(anteriores: readonly Resultado[], modelos: ReadonlyMap<string, string>,
  manifestoDe: ReadonlyMap<string, string>, piloto: boolean): Set<string> {
  const feitos = new Set<string>();
  for (const r of anteriores) {
    const motivo =
      !!r.piloto !== piloto ? (r.piloto ? "é de piloto" : "não é de piloto")
      : JSON.stringify(r.congelado) !== JSON.stringify(FROZEN) ? "tem outro congelamento do detector, do prompt ou da linha de data"
      : modelos.has(r.modelo) && r.modelo_sha !== modelos.get(r.modelo) ? `foi respondida por outro model.smoe de ${r.modelo}`
      : manifestoDe.has(r.id) && r.manifesto !== manifestoDe.get(r.id) ? `é de outro congelamento do conjunto (${r.id})`
      : null;
    if (motivo) throw new Error(`a saída já tem uma resposta que ${motivo}; use outra --saida`);
    feitos.add(`${r.modelo}\t${r.id}`);
  }
  return feitos;
}

/** A file summarized on its own (--resumo-so) passes the same checks: today's freezes, one model.smoe per model
 * name, one pilot flag for the whole file. */
export function conferirArquivo(resultados: readonly Resultado[], itens: readonly { id: string; manifesto: string }[]): void {
  const modelos = new Map<string, string>();
  for (const r of resultados) if (!modelos.has(r.modelo) && r.modelo_sha) modelos.set(r.modelo, r.modelo_sha);
  conferirRetomada(resultados, modelos, new Map(itens.map((i) => [i.id, i.manifesto])), !!resultados[0]?.piloto);
}

/** Checks every file a manifest lists; returns the manifest's own SHA-256 (the one in DECISIONS). */
export function conferirManifesto(dir = SET, nome = "MANIFEST.sha256"): string {
  const manifest = join(dir, nome);
  for (const line of readFileSync(manifest, "utf8").split(/\r?\n/).filter((l) => l.trim() !== "")) {
    const m = /^([0-9a-f]{64}) [ *](.+)$/.exec(line);
    if (!m) throw new Error(`${nome}: linha inválida: ${line}`);
    if (sha(join(dir, m[2]!)) !== m[1]) throw new Error(`${m[2]} não confere com o ${nome}: o conjunto mudou`);
  }
  return sha(manifest);
}

/** The sets present in the directory, each checked against its manifest and its frozen SHA-256. */
export function conjuntosPresentes(dir = SET): typeof CONJUNTOS {
  const presentes = CONJUNTOS.filter((c) => existsSync(join(dir, c.itens)) || existsSync(join(dir, c.manifesto)));
  for (const c of presentes) {
    if (c.sha === null) throw new Error(`${c.itens}: o conjunto ainda não foi congelado em DECISIONS`);
    const atual = conferirManifesto(dir, c.manifesto);
    if (atual !== c.sha) throw new Error(`${c.manifesto}: sha256 ${atual}, mas o congelado em DECISIONS é ${c.sha}`);
  }
  return presentes;
}

/** Refuses to run on anything but the frozen sets, detector, kit prompt and date line. */
export function conferirCongelados(): void {
  if (!conjuntosPresentes().some((c) => c.itens === "manutencao.json")) throw new Error("manutencao.json não encontrado");
  const atual = { detector: sha(DETECTOR), sistema: sha(KIT_PROMPT), linhaData: sha(KIT_DATE) };
  for (const k of Object.keys(FROZEN) as (keyof typeof FROZEN)[])
    if (atual[k] !== FROZEN[k]) throw new Error(`${k}: sha256 ${atual[k]}, mas o congelado em DECISIONS é ${FROZEN[k]}`);
}

export function carregarRegras(dir = SET): Regras {
  const lines = (f: string) => readFileSync(join(dir, f), "utf8").split(/\r?\n/).filter((l) => l.trim() !== "");
  const listas: Record<string, Set<string>> = {};
  for (const fam of ["powershell", "pacman", "apt", "brew"])
    if (existsSync(join(dir, `comandos-${fam}.txt`))) listas[fam] = new Set(lines(`comandos-${fam}.txt`));
  const mudanca = lines("lista-mudanca.tsv").map((l) => l.split("\t") as [string, string]);
  return {
    destrutivos: mudanca.filter(([c]) => c === "destrutivo").map(([, x]) => new RegExp(x, "iu")),
    mudanca: mudanca.map(([categoria, x]) => ({ categoria, re: new RegExp(x, "iu") })),
    leituraPadrao: Object.fromEntries(lines("leitura-padrao.tsv").map((l) => l.split("\t") as [string, string])
      .map(([c, x]) => [c, new RegExp(x, "iu")])),
    aviso: new RegExp(lines("aviso.txt")[0]!, "iu"),
    listas,
    emPortugues,
  };
}

export interface ItemDeConjunto extends Item {
  readonly manifesto: string; /* the frozen SHA-256 of the set it comes from */
  readonly conjunto: string;  /* that set's items file, which picks its grading module */
}

/** The items of every set present (conjuntosPresentes), with ids unique across sets. */
export function carregarItens(dir = SET): ItemDeConjunto[] {
  const itens: ItemDeConjunto[] = [];
  const ids = new Set<string>();
  for (const c of conjuntosPresentes(dir)) {
    for (const item of (JSON.parse(readFileSync(join(dir, c.itens), "utf8")) as { itens: Item[] }).itens) {
      if (ids.has(item.id)) throw new Error(`${c.itens}: id repetido entre conjuntos: ${item.id}`);
      ids.add(item.id);
      itens.push({ ...item, manifesto: c.sha!, conjunto: c.itens });
    }
  }
  return itens;
}

/** A family with no command list yet (pacman, brew) cannot be graded on existing commands: it waits for v2. */
export function semLista(item: Item, regras: Regras): boolean {
  return item.familia !== "-" && regras.listas[item.familia] === undefined;
}

/** The kit's system prompt, the frozen line of the item's machine role, then the date line. */
export function sistemaDoItem(item: Item, kit: string, papeis: Readonly<Record<string, string>>, data: string): string {
  if (!item.papel) return `${kit}\n${data}`;
  const linha = papeis[item.papel];
  if (!linha) throw new Error(`papel.json: sem linha para o papel ${item.papel}`);
  return `${kit}\n${linha}\n${data}`;
}

export function promptDoKit(): string {
  return readFileSync(KIT_PROMPT, "utf8").replace(/\r?\n$/, "");
}

const DIAS = ["domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"];
const MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
  "novembro", "dezembro"];

/** The frozen date line (kit/linha-data.txt) for a local time written YYYY-MM-DDTHH:MM. Without tools, this line is
 * how the kit's model knows the date. */
export function linhaDeData(agora: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})$/.exec(agora);
  if (!m) throw new Error(`data inválida: ${agora}`);
  const [ano, mes, dia] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const semana = new Date(Date.UTC(ano, mes - 1, dia)).getUTCDay();
  return readFileSync(KIT_DATE, "utf8").replace(/\r?\n$/, "").replace("{dia_semana}", DIAS[semana]!)
    .replace("{dia}", String(dia)).replace("{mes}", MESES[mes - 1]!).replace("{ano}", String(ano)).replace("{hora}", m[4]!);
}

/* The kit has no tools (plan, revision 19: the pilot without them won on both Qwen models). */
function perguntar(binario: string, dir: string, sistema: string, cache: string, item: Item): { resposta: string; segundos: number } {
  /* Piped input sends one message per line: a question of more lines would become several. */
  if (/[\r\n]/.test(item.pergunta)) throw new Error(`${item.id}: pergunta de mais de uma linha; o eval ainda não a envia inteira`);
  const t0 = performance.now();
  const r = Bun.spawnSync([binario, "--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"),
    "--tokenizer", join(dir, "tokenizer.qbp"), "--system", sistema, "--no-tools", "--prefix-cache", cache,
    "--context", String(CONTEXT), "--tokens", String(item.tokens ?? TOKENS), "--temperature", "0"],
  { stdin: new TextEncoder().encode(`${item.pergunta}\n/sair\n`), stdout: "pipe", stderr: "pipe", timeout: TIMEOUT_MS });
  if (r.exitCode !== 0) throw new Error(`${item.id}: o chat saiu com ${r.exitCode}: ${r.stderr.toString().slice(-400)}`);
  return { resposta: r.stdout.toString().trim(), segundos: (performance.now() - t0) / 1000 };
}

function lerResultados(path: string): Resultado[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8").split(/\r?\n/).filter((l) => l.trim() !== "").map((l) => JSON.parse(l) as Resultado);
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function resumo(resultados: readonly Resultado[], pulados = 0): string {
  const out: string[] = [];
  if (resultados.some((r) => r.piloto)) out.push("**Piloto:** não vale para a decisão.");
  if (pulados) out.push(`Fora da taxa até a v2 do conjunto: ${pulados} item(ns) de família sem lista de comandos.`);
  const modelos = [...new Set(resultados.map((r) => r.modelo))];
  for (const m of modelos) {
    const deste = resultados.filter((r) => r.modelo === m);
    out.push(`## ${m}`);
    for (const bloco of ["(todos)", ...new Set(deste.map((r) => r.bloco))]) {
      const rs = bloco === "(todos)" ? deste : deste.filter((r) => r.bloco === bloco);
      const k = rs.filter((r) => r.certa).length, w = wilson(k, rs.length);
      out.push(`- ${bloco}: ${k}/${rs.length} = ${pct(w.taxa)} (IC 95% ${pct(w.baixo)} a ${pct(w.alto)})`);
    }
    /* Each set as a whole: the kill rule of the plan looks at the maintenance questions (duvidas.json) alone. */
    for (const c of CONJUNTOS) {
      const rs = deste.filter((r) => r.manifesto === c.sha);
      if (!rs.length) continue;
      const k = rs.filter((r) => r.certa).length, w = wilson(k, rs.length);
      out.push(`- conjunto ${c.itens}: ${k}/${rs.length} = ${pct(w.taxa)} (IC 95% ${pct(w.baixo)} a ${pct(w.alto)})`);
    }
    /* The prompt was tuned on the pilot's 11 items, so the rate without them is given too (plan, revision 25). */
    {
      const rs = deste.filter((r) => !ITENS_DO_PILOTO.has(r.id));
      if (rs.length < deste.length) {
        const k = rs.filter((r) => r.certa).length, w = wilson(k, rs.length);
        out.push(`- sem os ${deste.length - rs.length} itens do piloto: ${k}/${rs.length} = ${pct(w.taxa)} (IC 95% ${pct(w.baixo)} a ${pct(w.alto)})`);
      }
    }
    const motivos = new Map<string, number>();
    for (const r of deste) if (r.motivo) {
      const chave = /^(comando inexistente|falta|proibido)/.exec(r.motivo)?.[1] ?? r.motivo;
      motivos.set(chave, (motivos.get(chave) ?? 0) + 1);
    }
    for (const [motivo, n] of [...motivos].sort((a, b) => b[1] - a[1])) out.push(`  - ${n}× ${motivo}`);
  }
  const mapa = (m: string) => new Map(resultados.filter((r) => r.modelo === m).map((r) => [r.id, r.certa]));
  for (let i = 0; i < modelos.length; i++)
    for (let j = i + 1; j < modelos.length; j++) {
      const [a, b] = [modelos[i]!, modelos[j]!];
      const p = pareado(mapa(a), mapa(b)), q = pareado(mapa(b), mapa(a));
      out.push(`- ${b} contra ${a}: conserta ${p.conserta}, quebra ${p.quebra}; ${b} melhor: p = ${p.p.toFixed(4)}` +
        `${p.ganhou ? " (ganhou)" : ""}; ${a} melhor: p = ${q.p.toFixed(4)}${q.ganhou ? " (ganhou)" : ""}`);
    }
  return out.join("\n");
}

function escolhidos<T extends Item>(itens: readonly T[], filtro: string | undefined): T[] {
  if (!filtro) return [...itens];
  const padroes = filtro.split(",").map((f) => new RegExp(`^${f.trim().replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`));
  return itens.filter((i) => padroes.some((p) => p.test(i.id)));
}

async function main(argv: readonly string[]): Promise<void> {
  const { values } = parseArgs({ args: [...argv], strict: true, options: {
    modelo: { type: "string", multiple: true }, saida: { type: "string" }, itens: { type: "string" },
    piloto: { type: "boolean" }, "resumo-so": { type: "boolean" } } });
  const saida = resolve(values.saida ?? join(ROOT, "build", "eval-manutencao.jsonl"));
  if (values["resumo-so"]) {
    conferirCongelados();
    const resultados = lerResultados(saida);
    conferirArquivo(resultados, carregarItens());
    console.log(resumo(resultados));
    return;
  }
  if (!values.modelo?.length) throw new Error("passe ao menos um --modelo DIR (com model.smoe, config.ini e tokenizer.qbp)");
  conferirCongelados();
  const regras = carregarRegras(), kit = promptDoKit(), data = linhaDeData(NOW), avaliadores = await carregarAvaliadores();
  const todos = escolhidos(carregarItens(), values.itens), itens = todos.filter((i) => !semLista(i, regras));
  const papeis = JSON.parse(readFileSync(join(SET, "papel.json"), "utf8")) as Record<string, string>;
  const binario = join(ROOT, process.env.SMOLMOE_BUILD ?? "build", `smolmoe${EXE}`);
  const caches = join(ROOT, "build", "eval-cache");
  mkdirSync(caches, { recursive: true });
  const modelos = new Map<string, string>();
  for (const dirArg of values.modelo) {
    const dir = resolve(dirArg), modelo = basename(dir), shaModelo = await shaDeArquivo(join(dir, "model.smoe"));
    if (modelos.has(modelo) && modelos.get(modelo) !== shaModelo) throw new Error(`dois modelos diferentes chamados ${modelo}`);
    modelos.set(modelo, shaModelo);
  }
  const feitos = conferirRetomada(lerResultados(saida), modelos, new Map(todos.map((i) => [i.id, i.manifesto])),
    !!values.piloto);
  for (const dirArg of values.modelo) {
    const dir = resolve(dirArg), modelo = basename(dir);
    for (const item of itens) {
      if (feitos.has(`${modelo}\t${item.id}`)) continue;
      const sistema = sistemaDoItem(item, kit, papeis, data);
      const variante = createHash("sha256").update(sistema).digest("hex").slice(0, 12);
      const { resposta, segundos } = perguntar(binario, dir, sistema, join(caches, `${modelo}-${variante}.kv`), item);
      const nota = avaliadores.get(item.conjunto)!(item, resposta, regras);
      const r: Resultado = { modelo, id: item.id, bloco: item.bloco, certa: nota.certa, motivo: nota.motivo, segundos,
        resposta, manifesto: item.manifesto, congelado: FROZEN, modelo_sha: modelos.get(modelo)!,
        ...(values.piloto ? { piloto: true } : {}) };
      appendFileSync(saida, `${JSON.stringify(r)}\n`);
      console.error(`${modelo} ${item.id}: ${nota.certa ? "certa" : `errada (${nota.motivo})`} em ${segundos.toFixed(1)} s`);
    }
  }
  console.log(resumo(lerResultados(saida), todos.length - itens.length));
}

if (import.meta.main) {
  try {
    await main(Bun.argv.slice(2));
  } catch (e) {
    console.error(`eval-manutencao: ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  }
}
