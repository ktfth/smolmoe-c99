/**
 * The decision rule of the maintenance eval (docs/DECISIONS.md, round 2):
 * every rate with its 95% Wilson interval, and every comparison of two models
 * paired on the same questions, with an exact one-sided sign test on the
 * discordant pairs.
 */

export interface Intervalo {
  readonly taxa: number;
  readonly baixo: number;
  readonly alto: number;
}

/** 95% Wilson score interval for k successes in n. */
export function wilson(k: number, n: number, z = 1.959963984540054): Intervalo {
  if (n === 0) return { taxa: 0, baixo: 0, alto: 1 };
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const centro = (p + z2 / (2 * n)) / den;
  const meia = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return { taxa: p, baixo: Math.max(0, centro - meia), alto: Math.min(1, centro + meia) };
}

/** P(X >= conserta) for X ~ Binomial(conserta + quebra, 1/2): the one-sided p of "the change fixes more than it breaks". */
export function testeDeSinal(conserta: number, quebra: number): number {
  const n = conserta + quebra;
  if (n === 0) return 1;
  let p = 0;
  for (let k = conserta; k <= n; k++) p += Math.exp(logComb(n, k) - n * Math.LN2);
  return Math.min(1, p);
}

function logComb(n: number, k: number): number {
  let s = 0;
  for (let i = 1; i <= k; i++) s += Math.log((n - k + i) / i);
  return s;
}

export interface Pareado {
  readonly conserta: number; /* questions B gets right and A gets wrong */
  readonly quebra: number;   /* questions A gets right and B gets wrong */
  readonly p: number;
  readonly ganhou: boolean;  /* p < 0.05 */
}

/** B against A on the questions both answered. */
export function pareado(a: ReadonlyMap<string, boolean>, b: ReadonlyMap<string, boolean>): Pareado {
  let conserta = 0, quebra = 0;
  for (const [id, certaA] of a) {
    const certaB = b.get(id);
    if (certaB === undefined || certaB === certaA) continue;
    if (certaB) ++conserta;
    else ++quebra;
  }
  const p = testeDeSinal(conserta, quebra);
  return { conserta, quebra, p, ganhou: p < 0.05 };
}
