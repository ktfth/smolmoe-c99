/**
 * Word search (BM25) over the passages, with the query of variant G (plan, revision 28): the question's terms that do
 * not change language (commands, codes, paths, product names), plus a hand-made pt->en glossary. Every glossary
 * line names the regression item that motivates it, and is refused if its stem is not in that item (revision 31, V1).
 */
import type { Trecho } from "./fontes";

export const K1 = 1.2, B = 0.75, TRECHOS_POR_PERGUNTA = 3;
/* A maintenance answer lives in a command: a passage with code (backticks, in every source) weighs this much more. */
export const PESO_CODIGO = 1.3;
/* A word of the question that is not in the glossary counts only if this many passages have it: a Portuguese word
 * that happens to exist in an English text is rare, and a rare word weighs a lot in BM25. */
export const DF_MINIMO = 5;

const PARADAS = new Set((
  "a o as os um uma uns umas de do da dos das no na nos nas em por para pelo pela pelos pelas com sem que se e ou " +
  "como qual quais onde quando porque porque nao sim meu minha meus minhas seu sua isso esse essa este esta ele ela " +
  "eu me mim ja mais menos muito pouco depois antes ate entre sobre sob tem ter ser estar esta estou foi fica ficar " +
  "faco fazer vejo ver posso pode preciso usar uso usando dentro fora todo toda todos todas outro outra cada mesmo " +
  "the of and to in for on with by an is are be it this that from or as at not your you can how what which"
).split(" "));

export function dobrar(s: string): string {
  return s.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

export function tokens(s: string): string[] {
  return dobrar(s).split(/[^a-z0-9]+/).filter((t) => t.length >= 2 && !PARADAS.has(t));
}

export type LinhaGlossario = { readonly pt: string; readonly en: readonly string[]; readonly item: string };

/** Reads glossario.tsv and checks every line against the regression item it names. */
export function lerGlossario(tsv: string, perguntas: ReadonlyMap<string, string>): LinhaGlossario[] {
  return tsv.split(/\r?\n/).filter((l) => l.trim() !== "").map((l, i) => {
    const [pt, en, item, ...resto] = l.split("\t");
    if (!pt || !en || !item || resto.length > 0) throw new Error(`glossário, linha ${i + 1}: esperava "pt<TAB>en<TAB>item"`);
    const pergunta = perguntas.get(item);
    if (pergunta === undefined) throw new Error(`glossário, linha ${i + 1}: item ${item} não está na regressão`);
    if (!contemRadical(pergunta, pt)) throw new Error(`glossário, linha ${i + 1}: "${pt}" não aparece em ${item}`);
    return { pt, en: en.split(" ").filter(Boolean), item };
  });
}

/** The stem starts a word of the text (folded), and a multi-word stem matches whole words but the last. */
export function contemRadical(texto: string, radical: string): boolean {
  const t = ` ${dobrar(texto).replace(/[^a-z0-9-]+/g, " ")}`;
  return t.includes(` ${dobrar(radical)}`);
}

export class Indice {
  private readonly posting = new Map<string, Map<number, number>>();
  private readonly comprimento: number[] = [];
  private readonly media: number;

  constructor(readonly trechos: readonly Trecho[]) {
    for (const t of trechos) {
      const ts = tokens(t.texto);
      this.comprimento[t.id] = ts.length;
      for (const w of ts) {
        let p = this.posting.get(w);
        if (!p) this.posting.set(w, (p = new Map()));
        p.set(t.id, (p.get(t.id) ?? 0) + 1);
      }
    }
    this.media = this.comprimento.reduce((a, b) => a + b, 0) / Math.max(1, trechos.length);
  }

  df(w: string): number {
    return this.posting.get(w)?.size ?? 0;
  }

  buscar(termos: readonly string[], n = TRECHOS_POR_PERGUNTA): Trecho[] {
    const nota = new Map<number, number>(), N = this.trechos.length;
    for (const w of new Set(termos)) {
      const p = this.posting.get(w);
      if (!p) continue;
      const idf = Math.log(1 + (N - p.size + 0.5) / (p.size + 0.5));
      for (const [id, tf] of p) {
        const norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * this.comprimento[id]! / this.media));
        nota.set(id, (nota.get(id) ?? 0) + idf * norm);
      }
    }
    const peso = (id: number) => (this.trechos[id]!.texto.includes("`") ? PESO_CODIGO : 1);
    return [...nota].map(([id, v]) => [id, v * peso(id)] as const)
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, n).map(([id]) => this.trechos[id]!);
  }
}

/** Variant G: the question's own words that the passages use often enough, plus the glossary's English terms. A word
 * the glossary translates is not also searched as it is ("libero" would find a product called Libero). */
export function consulta(pergunta: string, glossario: readonly LinhaGlossario[], indice: Indice): string[] {
  const usadas = glossario.filter((g) => contemRadical(pergunta, g.pt));
  /* A stem shorter than 3 letters ("wi" of "wi-fi") would cut unrelated words ("windows"). */
  const radicais = usadas.flatMap((g) => tokens(g.pt)).filter((r) => r.length >= 3);
  const proprias = tokens(pergunta).filter((w) => indice.df(w) >= DF_MINIMO && !radicais.some((r) => w.startsWith(r)));
  const traduzidas = usadas.flatMap((g) => g.en.flatMap(tokens));
  return [...new Set([...proprias, ...traduzidas])];
}
