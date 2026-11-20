/**
 * Summary of the source test (plan, revisions 28 to 34), on the 90 maintenance questions only, after the pass:
 *   bun scripts/fonte/resumo-fonte.ts --sem build/rodada-1-v13.jsonl --com build/fonte-com-trechos.jsonl
 * - each arm (model, with or without passages): rate with Wilson 95%, and without win-012, win-037 and ctn-008 (F-1);
 * - coverage, computed now and not before: a question is covered when every one of its required patterns matches
 *   the 3 passages it got; the rate on covered and on uncovered questions;
 * - per arm, the answers graded "proibido", and the "falta" answers that carry an English screen name (fixed list,
 *   the English pair of the Portuguese screen names the Windows rubrics ask for, revision 31);
 * - the rule of revision 28/29, paired: 1.5B with passages as good as Phi with passages (upper bound of the 95% CI of
 *   Phi - 1.5B below 10 points, Newcombe's hybrid score interval for paired proportions), or better than Phi alone
 *   (sign test); and each model with against without passages.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { CONJUNTOS, type Resultado } from "../eval-manutencao";
import { pareado, wilson } from "../eval/estatistica";
import { lerTrechos } from "./indice";

const ROOT = join(import.meta.dir, "..", "..");
export const PHI = "phi-4-mini-instruct", Q15 = "qwen2.5-1.5b-instruct";
export const FORA_F1 = new Set(["win-012", "win-037", "ctn-008"]);
export const MARGEM = 0.10;
export const NOMES_DE_TELA_EN = [
  "Disk Cleanup", "System Restore", "restore point", "Safe Mode", "Advanced startup", "Advanced options", "Resource Monitor",
  "Task Manager", "Startup apps", "Disk Management", "Change Drive Letter", "System Protection", "Computer Management",
  "User Accounts", "Sync now", "Virtual memory", "Performance Options", "Uninstall updates", "Update history", "Activation",
  "Fast startup", "Choose what the power buttons do", "Associated Handles", "Troubleshooter", "Allow an app", "Drive health",
];
const reTela = new RegExp(NOMES_DE_TELA_EN.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "i");

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const taxa = (rs: readonly Resultado[]) => {
  const k = rs.filter((r) => r.certa).length, w = wilson(k, rs.length);
  return `${k}/${rs.length} = ${pct(w.taxa)} (IC 95% ${pct(w.baixo)} a ${pct(w.alto)})`;
};

/** Newcombe (1998), method 10: 95% CI of p1 - p2 for paired proportions, from the Wilson intervals of each. */
export function diferencaPareada(a: ReadonlyMap<string, boolean>, b: ReadonlyMap<string, boolean>): { d: number; baixo: number; alto: number } {
  let n11 = 0, n10 = 0, n01 = 0, n00 = 0;
  for (const [id, x] of a) {
    const y = b.get(id);
    if (y === undefined) throw new Error(`${id}: sem par`);
    if (x && y) n11++; else if (x) n10++; else if (y) n01++; else n00++;
  }
  const n = n11 + n10 + n01 + n00, p1 = (n11 + n10) / n, p2 = (n11 + n01) / n;
  const w1 = wilson(n11 + n10, n), w2 = wilson(n11 + n01, n);
  /* Newcombe's adjusted correlation (as in R's ratesci): when positive, AD - BC loses N/2, floored at 0. */
  const den = Math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)), ad_bc = n11 * n00 - n10 * n01;
  const phi = den === 0 ? 0 : (ad_bc > 0 ? Math.max(ad_bc - n / 2, 0) : ad_bc) / den;
  const delta = Math.sqrt((p1 - w1.baixo) ** 2 - 2 * phi * (p1 - w1.baixo) * (w2.alto - p2) + (w2.alto - p2) ** 2);
  const eps = Math.sqrt((w1.alto - p1) ** 2 - 2 * phi * (w1.alto - p1) * (p2 - w2.baixo) + (p2 - w2.baixo) ** 2);
  return { d: p1 - p2, baixo: p1 - p2 - delta, alto: p1 - p2 + eps };
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { sem: { type: "string" }, com: { type: "string" } } });
  if (!values.sem || !values.com) throw new Error("uso: resumo-fonte.ts --sem ARQ --com ARQ");
  const duvidas = CONJUNTOS.find((c) => c.itens === "duvidas.json")!.sha;
  const ler = (f: string) => readFileSync(f, "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l) as Resultado)
    .filter((r) => r.manifesto === duvidas && (r.modelo === PHI || r.modelo === Q15));
  const sem = ler(values.sem), com = ler(values.com);
  if (sem.some((r) => r.fonte) || com.some((r) => !r.fonte)) throw new Error("--sem tem trechos ou --com não tem");
  const itens = (JSON.parse(readFileSync(join(ROOT, "tests", "evals", "manutencao", "duvidas.json"), "utf8")).itens as
    { id: string; obrigatorios: string[] }[]);
  const textos = new Map((await lerTrechos()).map((t) => [t.id, t.texto]));
  const idsDe = new Map(com.map((r) => [r.id, r.trechos!]));
  const coberta = new Map(itens.map((i) => {
    const texto = (idsDe.get(i.id) ?? []).map((id) => textos.get(id) ?? "").join("\n");
    return [i.id, i.obrigatorios.every((o) => new RegExp(o, "iu").test(texto))];
  }));
  const nCobertas = [...coberta.values()].filter(Boolean).length;
  const out = [`# Teste da fonte: as 90 dúvidas`, ``, `Cobertura (calculada depois da passada): ${nCobertas}/${itens.length} dúvidas têm todas as obrigatórias nos 3 trechos.`, ``];
  const bracos: [string, Resultado[]][] = [[`${PHI} sem trechos`, sem.filter((r) => r.modelo === PHI)],
    [`${PHI} com trechos`, com.filter((r) => r.modelo === PHI)], [`${Q15} sem trechos`, sem.filter((r) => r.modelo === Q15)],
    [`${Q15} com trechos`, com.filter((r) => r.modelo === Q15)]];
  for (const [nome, rs] of bracos) {
    out.push(`## ${nome}`, `- 90 dúvidas: ${taxa(rs)}`, `- sem win-012, win-037 e ctn-008: ${taxa(rs.filter((r) => !FORA_F1.has(r.id)))}`,
      `- cobertas: ${taxa(rs.filter((r) => coberta.get(r.id)))}`, `- não cobertas: ${taxa(rs.filter((r) => !coberta.get(r.id)))}`,
      `- proibido: ${rs.filter((r) => r.motivo?.startsWith("proibido")).length}`,
      `- "falta" com nome de tela em inglês: ${rs.filter((r) => r.motivo?.startsWith("falta") && reTela.test(r.resposta)).length}`,
      `- fora do português: ${rs.filter((r) => r.motivo === "fora do português").length}`);
  }
  const mapa = (rs: readonly Resultado[], sem3 = false) => new Map(rs.filter((r) => !sem3 || !FORA_F1.has(r.id)).map((r) => [r.id, r.certa]));
  out.push(``, `## Regra (pareado)`);
  for (const sem3 of [false, true]) {
    const tag = sem3 ? " (sem win-012, win-037 e ctn-008)" : "";
    const [ps, pc, qs, qc] = bracos.map(([, rs]) => mapa(rs, sem3));
    const eq = diferencaPareada(pc!, qc!);
    out.push(`- Phi com trechos − 1,5B com trechos${tag}: ${pct(eq.d)} (IC 95% ${pct(eq.baixo)} a ${pct(eq.alto)}); ` +
      `${eq.alto < MARGEM ? "tão bom quanto (limite superior abaixo de 10 pontos)" : "equivalência não demonstrada"}`);
    const s = pareado(ps!, qc!);
    out.push(`- 1,5B com trechos contra Phi sem trechos${tag}: conserta ${s.conserta}, quebra ${s.quebra}, p = ${s.p.toFixed(4)}${s.ganhou ? " (ganhou)" : ""}`);
    for (const [nome, a, b] of [["Phi", ps!, pc!], ["1,5B", qs!, qc!]] as const) {
      const m = pareado(a, b), r = pareado(b, a);
      out.push(`- ${nome} com contra sem trechos${tag}: conserta ${m.conserta}, quebra ${m.quebra}; com melhor p = ${m.p.toFixed(4)}` +
        `${m.ganhou ? " (ganhou)" : ""}; sem melhor p = ${r.p.toFixed(4)}${r.ganhou ? " (ganhou)" : ""}`);
    }
  }
  console.log(out.join("\n"));
}

if (import.meta.main) await main();
