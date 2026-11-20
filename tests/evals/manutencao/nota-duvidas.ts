/**
 * A nota dos blocos de dúvidas (duvidas.json). Usa o nota.ts da v1 sem mudá-lo (o sha da v1 continua valendo) e
 * acrescenta uma regra só para os itens com alvo "resposta" (ataque D-1 do revisor às dúvidas): numa dúvida, o
 * conselho perigoso costuma vir em crase na prosa ("apague com `sudo rm ...lock-frontend`"), e o nota.ts só confere
 * os proibidos nos blocos de comando. Aqui, os proibidos valem também no código em crase da prosa, menos a crase
 * que a resposta desaconselha de perto:
 *  - até 5 palavras logo DEPOIS de "não use", "não rode", "nunca", "evite", "em vez de", "ao invés de", "no lugar de",
 *    "obsoleto", "nada de", "removido", "depreciado" (uma crase conta como uma palavra; 2º ataque, D2-1);
 *  - ou seguida de "está obsoleto", "foi removido", "não funciona", "não é (mais) recomendado", "não deve (mais) ser usado".
 * A janela é curta de propósito: "se não resolver, apague com `rm ...`" continua reprovando.
 */
import { type ItemNota, type Nota, type Regras, avaliar as avaliarV1 } from "./nota";

const re = (s: string) => new RegExp(s, "iu");
const BLOCO = /```[^\n`]*\n[\s\S]*?(```|$)/g;
const ANTES = /(?<![\p{L}])(n[aã]o (use|usar|rode|rodar|execute|executar|digite|apague|fa[cç]a)|nunca|evite|evitar|em vez d[eo]|ao inv[eé]s d[eo]|no lugar d[eo]|obsolet[oa]|nada de|removid[oa]|depreciad[oa]|descontinuad[oa])(?![\p{L}])/giu;
const DEPOIS = /^\s*(,\s*)?(que\s+)?(est[aá]|[eé]|foi)?\s*(obsolet[oa]|descontinuad[oa]|removid[oa]|depreciad[oa])|^\s*(,\s*)?n[aã]o (funciona|[eé] (mais )?recomendad[oa]|deve (mais )?ser usad[oa])/iu;

/** O código em crase da prosa (fora dos blocos), menos o desaconselhado de perto. */
export function crasesAfirmativas(resposta: string): string[] {
  const prosa = resposta.replace(BLOCO, "\n");
  const isentas = new Set<number>();
  for (const m of prosa.matchAll(ANTES)) {
    let i = m.index! + m[0].length, palavras = 0;
    while (i < prosa.length && palavras < 5) {
      while (i < prosa.length && /[ \t]/.test(prosa[i]!)) i++;
      if (i >= prosa.length || /[,;.:!?\n]/.test(prosa[i]!)) break;
      if (prosa[i] === "`") { isentas.add(i); const f = prosa.indexOf("`", i + 1); i = f < 0 ? prosa.length : f + 1; }
      else while (i < prosa.length && !/[\s,;.:!?`]/.test(prosa[i]!)) i++;
      palavras++;
    }
  }
  const out: string[] = [];
  for (const m of prosa.matchAll(/`([^`\n]+)`/g)) {
    if (isentas.has(m.index!)) continue;
    if (DEPOIS.test(prosa.slice(m.index! + m[0].length))) continue;
    out.push(m[1]!);
  }
  return out;
}

export function avaliar(item: ItemNota, resposta: string, regras: Regras): Nota {
  const n = avaliarV1(item, resposta, regras);
  if (!n.certa || item.alvo !== "resposta") return n;
  const crases = crasesAfirmativas(resposta).join("\n");
  for (const p of item.proibidos) if (re(p).test(crases)) return { ...n, certa: false, motivo: `proibido ${p} (em crase na prosa)` };
  return n;
}
