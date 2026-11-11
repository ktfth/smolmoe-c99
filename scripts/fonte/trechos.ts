/**
 * The treatment of the source test (plan, revisions 28 to 31): the 3 passages found for a question, inside the frozen
 * presentation text, as one line of the user's message (the chat reads one message per line). The kit's system
 * prompt does not change. What decides the passages is frozen by SHA-256: the presentation text, the glossary, the
 * regression that checks the glossary, the search code and the passages file.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { consulta, Indice } from "./busca";
import { glossario, lerTrechos, regressao, TRECHOS } from "./indice";

const AQUI = import.meta.dir;

export const ARQUIVOS_DA_FONTE = {
  apresentacao: join(AQUI, "apresentacao.txt"),
  glossario: join(AQUI, "glossario.tsv"),
  regressao: join(AQUI, "regressao.json"),
  busca: join(AQUI, "busca.ts"),
  trechos: TRECHOS,
} as const;

export type CongeladoFonte = { readonly [k in keyof typeof ARQUIVOS_DA_FONTE]: string };

export function congelamentoDaFonte(): CongeladoFonte {
  const sha = (f: string) => createHash("sha256").update(readFileSync(f)).digest("hex");
  return Object.fromEntries(Object.entries(ARQUIVOS_DA_FONTE).map(([k, f]) => [k, sha(f)])) as CongeladoFonte;
}

/** The presentation text with the passages and the question, on one line. */
export function mensagemComTrechos(apresentacao: string, textos: readonly string[], pergunta: string): string {
  const trechos = textos.map((t, i) => `[${i + 1}] ${t.replace(/\s+/g, " ").trim()}`).join(" ");
  return apresentacao.trim().replace("{trechos}", () => trechos).replace("{pergunta}", () => pergunta);
}

export type Fonte = { readonly mensagem: (pergunta: string) => { texto: string; ids: number[] } };

export async function prepararFonte(): Promise<Fonte> {
  const apresentacao = readFileSync(ARQUIVOS_DA_FONTE.apresentacao, "utf8");
  if (!apresentacao.includes("{trechos}") || !apresentacao.includes("{pergunta}"))
    throw new Error("apresentacao.txt precisa de {trechos} e {pergunta}");
  const g = await glossario(await regressao()), indice = new Indice(await lerTrechos());
  return {
    mensagem: (pergunta) => {
      const achados = indice.buscar(consulta(pergunta, g, indice));
      return { texto: mensagemComTrechos(apresentacao, achados.map((t) => t.texto), pergunta), ids: achados.map((t) => t.id) };
    },
  };
}
