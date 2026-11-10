/**
 * The Portuguese detector of the maintenance eval (the "resposta em português"
 * rule of tests/evals/manutencao/nota.ts, which leaves this one rule to the
 * dev). It reads only the prose: code blocks and inline code are commands,
 * which look English whatever the language of the answer. Frozen with its
 * SHA-256 in docs/DECISIONS.md before the first run; changing it means a new
 * freeze.
 *
 * The prose is Portuguese when its Portuguese function words outnumber both
 * the English and the Spanish ones; a tie is not Portuguese. Words with ã, õ
 * or ç count as Portuguese, words with ñ as Spanish. Words spelled the same in
 * English ("use", "a") count for neither. Prose of fewer than 4 words (an
 * answer that is mostly its code block) has no language to judge and passes.
 */

const PT = new Set(["de", "do", "da", "dos", "das", "que", "não", "para", "com", "uma", "um", "o", "os", "as", "em",
  "no", "na", "nos", "nas", "ao", "aos", "você", "é", "são", "se", "mais", "como", "por", "pelo", "pela", "isso",
  "este", "esta", "esse", "essa", "ou", "também", "seu", "sua", "já", "quando", "então", "antes", "depois", "cada",
  "ele", "ela", "há", "mas", "muito", "sem", "até", "aqui", "rode", "execute", "digite"]);
const EN = new Set(["the", "and", "is", "are", "of", "to", "you", "with", "this", "that", "for", "it", "on", "be",
  "your", "can", "will", "if", "from", "which", "run", "then", "before", "after", "each", "should", "would", "here",
  "there", "these", "those", "an", "in", "or", "not", "by", "them", "they", "its", "what", "how", "all"]);
const ES = new Set(["el", "los", "las", "y", "del", "usted", "puede", "pero", "muy", "hay", "también", "cuando",
  "una", "con", "esto", "aquí", "entonces", "después", "antes", "cada", "ejecute", "utilice"]);

/** The answer without its code: fenced blocks and inline code. */
export function prosa(resposta: string): string {
  return resposta.replace(/```[\s\S]*?(```|$)/g, " ").replace(/`[^`\n]*`/g, " ");
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
    /* "antes" and "cada" are both Portuguese and Spanish: they count for both and decide nothing. */
    if (PT.has(p) || /[ãõç]/u.test(p)) ++pt;
    if (EN.has(p)) ++en;
    if (ES.has(p) || p.includes("ñ")) ++es;
  }
  return { palavras: palavras.length, pt, en, es };
}

export function emPortugues(resposta: string): boolean {
  const c = contar(resposta);
  if (c.palavras < 4) return true;
  return c.pt > c.en && c.pt > c.es;
}
