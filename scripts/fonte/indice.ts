/**
 * The source test's index (plan, revisions 28 to 31).
 *   bun scripts/fonte/indice.ts --montar              cuts the sources into build/fonte/trechos.jsonl and prints its sha256
 *   bun scripts/fonte/indice.ts --regressao [--ver]   share of the regression questions whose 3 passages hold the fact
 *   bun scripts/fonte/indice.ts --consulta "texto"    the query and the 3 passages for one question
 * The search is tuned on the regression only, never on the 90 decision questions.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { consulta, Indice, lerGlossario } from "./busca";
import { montarTrechos, raizDasFontes, type Trecho } from "./fontes";

const AQUI = import.meta.dir;
const SAIDA = join(AQUI, "..", "..", "build", "fonte");
export const TRECHOS = join(SAIDA, "trechos.jsonl");

type ItemRegressao = { readonly id: string; readonly bloco: string; readonly pergunta: string; readonly fato: string };

export async function regressao(): Promise<ItemRegressao[]> {
  return JSON.parse(await Bun.file(join(AQUI, "regressao.json")).text()).itens;
}

export async function glossario(itens: readonly ItemRegressao[]) {
  return lerGlossario(await Bun.file(join(AQUI, "glossario.tsv")).text(), new Map(itens.map((i) => [i.id, i.pergunta])));
}

export async function lerTrechos(): Promise<Trecho[]> {
  const f = Bun.file(TRECHOS);
  if (!(await f.exists())) throw new Error(`falta ${TRECHOS}: rode antes --montar`);
  return (await f.text()).trim().split("\n").map((l) => JSON.parse(l));
}

export function sha256(texto: string): string {
  return new Bun.CryptoHasher("sha256").update(texto).digest("hex");
}

async function montar(): Promise<void> {
  const trechos = await montarTrechos(raizDasFontes());
  mkdirSync(SAIDA, { recursive: true });
  const texto = trechos.map((t) => JSON.stringify(t)).join("\n") + "\n";
  await Bun.write(TRECHOS, texto);
  const porFonte = new Map<string, number>();
  for (const t of trechos) porFonte.set(t.fonte, (porFonte.get(t.fonte) ?? 0) + 1);
  console.log(`${trechos.length} trechos (${[...porFonte].map(([f, n]) => `${f} ${n}`).join(", ")})`);
  console.log(`sha256 de trechos.jsonl: ${sha256(texto)}`);
}

async function medirRegressao(ver: boolean): Promise<void> {
  const itens = await regressao(), g = await glossario(itens), indice = new Indice(await lerTrechos());
  const porBloco = new Map<string, [number, number]>();
  let acertos = 0;
  for (const it of itens) {
    const achados = indice.buscar(consulta(it.pergunta, g, indice));
    const ok = new RegExp(it.fato, "i").test(achados.map((t) => t.texto).join("\n"));
    acertos += ok ? 1 : 0;
    const b = porBloco.get(it.bloco) ?? [0, 0];
    porBloco.set(it.bloco, [b[0] + (ok ? 1 : 0), b[1] + 1]);
    if (ver || !ok) console.log(`${ok ? "ok   " : "FALTA"} ${it.id}  ${achados.map((t) => t.caminho.split("/").pop()).join(" | ")}`);
  }
  console.log(`\nregressão: ${acertos}/${itens.length} (${[...porBloco].map(([b, [a, n]]) => `${b} ${a}/${n}`).join(", ")})`);
}

async function main(): Promise<void> {
  const a = process.argv.slice(2);
  if (a[0] === "--montar") return montar();
  if (a[0] === "--regressao") return medirRegressao(a.includes("--ver"));
  if (a[0] === "--consulta" && a[1]) {
    const itens = await regressao(), g = await glossario(itens), indice = new Indice(await lerTrechos());
    const q = consulta(a[1], g, indice);
    console.log(`consulta: ${q.join(" ")}\n`);
    for (const t of indice.buscar(q)) console.log(`--- ${t.caminho}\n${t.texto}\n`);
    return;
  }
  console.error("uso: indice.ts --montar | --regressao [--ver] | --consulta \"texto\"");
  process.exit(2);
}

if (import.meta.main) await main();
