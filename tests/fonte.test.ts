/**
 * The source test's pieces that get frozen (plan, revisions 28 to 31): the glossary rule (V1), the regression set
 * kept apart from the 90 decision questions, the source readers and the query.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { consulta, contemRadical, Indice, lerGlossario, tokens } from "../scripts/fonte/busca";
import { fatiar, secoesDocbook, secoesMarkdown } from "../scripts/fonte/fontes";
import { glossario, regressao } from "../scripts/fonte/indice";
import { diferencaPareada } from "../scripts/fonte/resumo-fonte";
import { ARQUIVOS_DA_FONTE, mensagemComTrechos } from "../scripts/fonte/trechos";
import { ROOT } from "./support/run";

describe("glossary", () => {
  const perguntas = new Map([["r-1", "A impressora de rede some quando reinicia."]]);

  test("every line names a regression item whose question holds its stem", () => {
    expect(lerGlossario("impressora\tprinter\tr-1\nreinicia\trestart\tr-1\n", perguntas).map((g) => g.en)).toEqual([["printer"], ["restart"]]);
    expect(() => lerGlossario("disco\tdisk\tr-1", perguntas)).toThrow(/não aparece em r-1/);
    expect(() => lerGlossario("impressora\tprinter\tr-9", perguntas)).toThrow(/não está na regressão/);
    expect(() => lerGlossario("impressora printer r-1", perguntas)).toThrow(/esperava/);
  });

  test("a stem starts a word, ignoring accents and case; a multi-word stem matches in sequence", () => {
    expect(contemRadical("Como desativo a Hibernação?", "hibernac")).toBe(true);
    expect(contemRadical("Como desativo a hibernação?", "nacao")).toBe(false);
    expect(contemRadical("Abra o Visualizador de Eventos", "visualizador de eventos")).toBe(true);
    expect(contemRadical("Visualizador e eventos", "visualizador de eventos")).toBe(false);
  });

  test("the frozen glossary loads, and every term is motivated by the regression", async () => {
    const itens = await regressao();
    expect((await glossario(itens)).length).toBeGreaterThan(50);
  });
});

describe("regression set", () => {
  test("unique ids, a fact that compiles, and no question repeated from the 90 decision questions", async () => {
    const itens = await regressao();
    expect(new Set(itens.map((i) => i.id)).size).toBe(itens.length);
    for (const i of itens) expect(() => new RegExp(i.fato, "i")).not.toThrow();
    const duvidas = JSON.parse(await Bun.file(join(ROOT, "tests", "evals", "manutencao", "duvidas.json")).text()).itens as { pergunta: string }[];
    const normal = (s: string) => tokens(s).join(" ");
    const decisao = new Set(duvidas.map((d) => normal(d.pergunta)));
    expect(itens.filter((i) => decisao.has(normal(i.pergunta))).map((i) => i.id)).toEqual([]);
  });
});

describe("source readers", () => {
  test("Markdown: no front matter, comments, images, shortcodes or link targets; code stays; split at headings", () => {
    const md = "---\ntitle: Fix the spooler\nms.date: 1\n---\n# Fix the spooler\n<!-- x -->\nSee [the docs](https://x).\n" +
      "![img](a.png)\n{{< note >}}\n## Steps\nRun `net stop spooler` now.\n";
    const s = secoesMarkdown(md);
    expect(s.map((x) => x.titulo)).toEqual(["Fix the spooler", "Fix the spooler > Steps"]);
    expect(s[0]!.corpo).toContain("See the docs.");
    expect(s.map((x) => x.corpo).join("")).not.toMatch(/https|img|note|ms\.date|<!--/);
    expect(s[1]!.corpo).toContain("`net stop spooler`");
  });

  test("DocBook: one section per title, under the chapter's title; table cells kept apart", () => {
    const xml = "<chapter><title>Package management</title><section><title>Hold</title>" +
      "<para>Use <literal>apt-mark hold</literal> &amp; wait.</para><table><row><entry>a</entry><entry>b</entry></row></table></section></chapter>";
    expect(secoesDocbook(xml)).toEqual([{ titulo: "Package management > Hold", corpo: "Use apt-mark hold & wait.\na b \n" }]);
  });

  test("passages carry their title and drop scraps of fewer than 8 words", () => {
    const corpo = Array.from({ length: 170 }, (_, i) => `w${i}`).join(" ");
    const t = fatiar([{ titulo: "T", corpo }, { titulo: "U", corpo: "too short" }]);
    expect(t.length).toBe(3);
    expect(t.every((x) => x.startsWith("T\n"))).toBe(true);
  });
});

describe("treatment", () => {
  test("the passages and the question go in the presentation text, on one line, numbered", () => {
    const m = mensagemComTrechos("Trechos: {trechos} Pergunta: {pergunta}\n", ["T\nrun `x`", "U\n  y $& z"], "Como?");
    expect(m).toBe("Trechos: [1] T run `x` [2] U y $& z Pergunta: Como?");
    expect(m).not.toMatch(/[\r\n]/);
  });

  test("the frozen presentation text asks for Brazilian Portuguese and Portuguese screen names, and has both slots", () => {
    const t = readFileSync(ARQUIVOS_DA_FONTE.apresentacao, "utf8");
    expect(t).toContain("{trechos}");
    expect(t).toContain("{pergunta}");
    expect(t).toMatch(/português do Brasil/);
    expect(t).toMatch(/nomes de tela do Windows em português/);
  });
});

describe("summary", () => {
  const pares = (n11: number, n10: number, n01: number, n00: number) => {
    const a = new Map<string, boolean>(), b = new Map<string, boolean>();
    const por = (n: number, x: boolean, y: boolean) => { for (let i = 0; i < n; i++) { const id = `${x}${y}${i}`; a.set(id, x); b.set(id, y); } };
    por(n11, true, true); por(n10, true, false); por(n01, false, true); por(n00, false, false);
    return diferencaPareada(a, b);
  };

  test("Newcombe's paired interval matches R's ratesci (MOVER-Wilson, Newcombe's correlation): 1, 1, 7, 12", () => {
    const r = pares(1, 1, 7, 12);
    expect(r.d).toBeCloseTo(-6 / 21, 6);
    expect(r.baixo).toBeCloseTo(-0.507, 3);
    expect(r.alto).toBeCloseTo(-0.026, 3);
  });

  test("a positive correlation narrows the interval, and Newcombe's adjustment narrows it less", () => {
    const largura = (r: { baixo: number; alto: number }) => r.alto - r.baixo;
    const corr = pares(12, 5, 2, 21), indep = pares(4, 13, 10, 13);
    expect(corr.d).toBeCloseTo(indep.d, 9);
    expect(largura(corr)).toBeLessThan(largura(indep));
    expect(corr.baixo).toBeCloseTo(-0.0594, 3); /* with the unadjusted phi it would be -0.0506 */
  });
});

describe("query (variant G)", () => {
  const trechos = Array.from({ length: 6 }, (_, id) => ({ id, fonte: "f", caminho: `p${id}`, texto: `windows printer restart libero ${id}` }));
  const indice = new Indice(trechos);
  const g = [{ pt: "wi-fi", en: ["wireless"], item: "r" }, { pt: "libero", en: ["allow"], item: "r" }];

  test("a word the glossary translates is not searched as it is, but a 2-letter stem cuts nothing", () => {
    const q = consulta("Como libero o Wi-Fi no Windows?", g, indice);
    expect(q).toContain("windows");
    expect(q).toContain("wireless");
    expect(q).toContain("allow");
    expect(q).not.toContain("libero");
  });
});
