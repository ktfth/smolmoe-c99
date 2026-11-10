/**
 * The Portuguese detector of the maintenance eval (the "resposta em português"
 * rule of tests/evals/manutencao/nota.ts, which leaves this one rule to the
 * dev). It reads only the prose: code blocks, inline code and domain names are
 * commands or addresses, which look English whatever the language of the
 * answer. Frozen with its SHA-256 in docs/DECISIONS.md before the first run;
 * changing it means a new freeze.
 *
 * Each function word counts for every language it belongs to: "de", "que" and
 * "para" for Portuguese and Spanish, "do", "no" and "as" for Portuguese and
 * English. Words with ã, õ or ç count as Portuguese, words with ñ or ending in
 * -ción/-ciones as Spanish. The prose is Portuguese when its Portuguese count
 * beats both the English and the Spanish ones; a tie is not Portuguese. Prose
 * of fewer than 4 words (an answer that is mostly its code block) has no
 * language to judge and passes.
 */

const SHARED_PT_ES = ["de", "que", "para", "este", "esta", "se", "como", "por", "antes", "cada", "está", "no"];
const SHARED_PT_EN = ["do", "no", "as", "execute"];

const PT = new Set(["da", "dos", "das", "não", "uma", "um", "o", "os", "em", "na", "nos", "nas", "ao", "aos", "você",
  "é", "são", "mais", "pelo", "pela", "isso", "esse", "essa", "ou", "também", "seu", "sua", "já", "quando", "então",
  "depois", "ele", "ela", "há", "muito", "sem", "até", "aqui", "rode", "digite", "com", ...SHARED_PT_ES, ...SHARED_PT_EN]);
const EN = new Set(["the", "and", "is", "are", "of", "to", "you", "with", "this", "that", "for", "it", "on", "be",
  "your", "can", "will", "if", "from", "which", "run", "then", "before", "after", "each", "should", "would", "here",
  "there", "these", "those", "an", "in", "or", "not", "by", "them", "they", "its", "what", "how", "all", ...SHARED_PT_EN]);
const ES = new Set(["el", "los", "las", "y", "del", "en", "es", "su", "sus", "lo", "usted", "puede", "pero", "muy",
  "hay", "también", "cuando", "una", "con", "esto", "aquí", "entonces", "después", "ejecute", "utilice", "más",
  ...SHARED_PT_ES]);

/** The answer without its code (fenced blocks, inline code) and without domain names ("example.com"). */
export function prosa(resposta: string): string {
  return resposta.replace(/```[\s\S]*?(```|$)/g, " ").replace(/`[^`\n]*`/g, " ")
    .replace(/\b[\p{L}\d-]+(\.[\p{L}\d-]+)+\b/gu, " ");
}

export interface Contagem {
  readonly palavras: number;
  readonly pt: number;
  readonly en: number;
  readonly es: number;
}

export function contar(resposta: string): Contagem {
  const palavras = prosa(resposta).toLowerCase().match(/\p{L}+/gu) ?? [];
  let pt = 0, en = 0, es = 0;
  for (const p of palavras) {
    if (PT.has(p) || /[ãõç]/u.test(p)) ++pt;
    if (EN.has(p)) ++en;
    if (ES.has(p) || p.includes("ñ") || /ci[oó]n(es)?$/u.test(p) && !/ção|ções/u.test(p)) ++es;
  }
  return { palavras: palavras.length, pt, en, es };
}

export function emPortugues(resposta: string): boolean {
  const c = contar(resposta);
  if (c.palavras < 4) return true;
  return c.pt > c.en && c.pt > c.es;
}
