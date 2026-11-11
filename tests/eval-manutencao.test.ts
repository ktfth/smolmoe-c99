/**
 * The maintenance eval: it runs only on what DECISIONS froze (set, detector,
 * kit prompt), the decision rule's numbers, the Portuguese detector (the one
 * rule nota.ts leaves to the dev) and one real item through the chat.
 */
import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { testeDeSinal, wilson } from "../scripts/eval/estatistica";
import { emPortugues } from "../scripts/eval/portugues";
import {
  CONJUNTOS, DETECTOR, FONTE, FROZEN, ITENS_DO_PILOTO, KIT_DATE, KIT_PROMPT, NOW, type Resultado, carregarAvaliadores, carregarItens, carregarRegras,
  conferirArquivo, conferirCongelados, conferirManifesto, conferirRetomada, conjuntosPresentes, linhaDeData, promptDoKit, renotar, resumo,
  semLista, sha, sistemaDoItem, viradas,
} from "../scripts/eval-manutencao";
import { avaliar } from "./evals/manutencao/nota";
import { ROOT, bundledModel, tempDir } from "./support/run";

const SET = join(ROOT, "tests", "evals", "manutencao");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);
const decisions = () => readFileSync(join(ROOT, "docs", "DECISIONS.md"), "utf8");

describe("what was frozen", () => {
  test("the sets, the detector, the kit prompt and the date line are the ones in DECISIONS, checked before running", () => {
    expect(conferirManifesto()).toBe(CONJUNTOS[0]!.sha);
    expect(sha(DETECTOR)).toBe(FROZEN.detector);
    expect(sha(KIT_PROMPT)).toBe(FROZEN.sistema);
    expect(sha(KIT_DATE)).toBe(FROZEN.linhaData);
    for (const v of [...Object.values(FROZEN), ...CONJUNTOS.map((c) => c.sha).filter((s) => s !== null),
      ...Object.values(FONTE ?? {})])
      expect(decisions()).toContain(v);
    expect(() => conferirCongelados()).not.toThrow();
  });

  test("every set present is frozen, and every item carries the frozen SHA-256 of its set", () => {
    const presentes = conjuntosPresentes();
    expect(presentes.every((c) => c.sha !== null)).toBe(true);
    const itens = carregarItens();
    expect(new Set(itens.map((i) => i.id)).size).toBe(itens.length);
    expect(itens.filter((i) => i.id.startsWith("cmd-")).every((i) => i.manifesto === CONJUNTOS[0]!.sha)).toBe(true);
  });

  test("each set present is graded by its own module, which its manifest freezes", async () => {
    const avaliadores = await carregarAvaliadores();
    for (const c of conjuntosPresentes()) {
      expect(readFileSync(join(SET, c.manifesto), "utf8")).toMatch(new RegExp(`[ *]${c.nota.replace(".", "\\.")}\\r?$`, "m"));
      expect(avaliadores.get(c.itens)).toBe((await import(join(SET, c.nota))).avaliar);
    }
    expect(avaliadores.get("manutencao.json")).toBe(avaliar);
    expect(CONJUNTOS.find((c) => c.itens === "duvidas.json")?.nota).toBe("nota-duvidas.ts");
  });

  test("resuming refuses a file made under other conditions instead of mixing two runs", () => {
    const linha = (extra: Partial<Resultado>): Resultado => ({ modelo: "m", id: "cmd-001", bloco: "comando", certa: true,
      motivo: null, segundos: 1, resposta: "", manifesto: CONJUNTOS[0]!.sha!, congelado: FROZEN, modelo_sha: "aa", ...extra });
    const modelos = new Map([["m", "aa"]]), manifestos = new Map([["cmd-001", CONJUNTOS[0]!.sha!]]);
    expect(conferirRetomada([linha({})], modelos, manifestos, false)).toEqual(new Set(["m\tcmd-001"]));
    expect(() => conferirRetomada([linha({ piloto: true })], modelos, manifestos, false)).toThrow(/piloto/);
    expect(() => conferirRetomada([linha({})], modelos, manifestos, true)).toThrow(/não é de piloto/);
    expect(() => conferirRetomada([linha({ congelado: { ...FROZEN, detector: "0f3f9366" } as typeof FROZEN })], modelos,
      manifestos, false)).toThrow(/outro congelamento do detector/);
    expect(() => conferirRetomada([linha({ congelado: undefined })], modelos, manifestos, false)).toThrow(/congelamento/);
    expect(() => conferirRetomada([linha({ modelo_sha: "bb" })], modelos, manifestos, false)).toThrow(/outro model\.smoe/);
    expect(() => conferirRetomada([linha({ manifesto: "f00" })], modelos, manifestos, false)).toThrow(/conjunto/);
    /* --resumo-so checks the file the same way: one pilot flag, one model.smoe per name, today's freezes. */
    const itens = [{ id: "cmd-001", manifesto: CONJUNTOS[0]!.sha! }, { id: "cmd-002", manifesto: CONJUNTOS[0]!.sha! }];
    expect(() => conferirArquivo([linha({}), linha({ id: "cmd-002" })], itens)).not.toThrow();
    expect(() => conferirArquivo([linha({}), linha({ id: "cmd-002", piloto: true })], itens)).toThrow(/piloto/);
    expect(() => conferirArquivo([linha({}), linha({ id: "cmd-002", modelo_sha: "bb" })], itens)).toThrow(/model\.smoe/);
    /* The source test: an answer with passages never mixes with one without, nor with another freeze of them. */
    const f = { apresentacao: "1", glossario: "2", regressao: "3", busca: "4", trechos: "5" };
    expect(conferirRetomada([linha({ fonte: f })], modelos, manifestos, false, f)).toEqual(new Set(["m\tcmd-001"]));
    expect(() => conferirRetomada([linha({ fonte: f })], modelos, manifestos, false)).toThrow(/tem trechos da fonte/);
    expect(() => conferirRetomada([linha({})], modelos, manifestos, false, f)).toThrow(/não tem trechos da fonte/);
    expect(() => conferirRetomada([linha({ fonte: { ...f, glossario: "9" } })], modelos, manifestos, false, f))
      .toThrow(/outro congelamento da fonte/);
    expect(() => conferirArquivo([linha({ fonte: f }), linha({ id: "cmd-002" })], itens)).toThrow(/não tem trechos/);
  });

  test("a regrade grades the stored answers again with today's sets, and refuses a changed question", async () => {
    const regras = carregarRegras(), avaliadores = await carregarAvaliadores();
    const item = carregarItens().find((i) => i.id === "cmd-001")!;
    const linha = (extra: Partial<Resultado>): Resultado => ({ modelo: "m", id: "cmd-001", bloco: "comando", certa: false,
      motivo: "x", segundos: 1, resposta: "Use:\n```powershell\nGet-Volume\n```", manifesto: "antigo", congelado: FROZEN,
      modelo_sha: "aa", ...extra });
    const [novo] = renotar([linha({})], [item], avaliadores, regras);
    expect(novo).toMatchObject({ certa: true, motivo: null, manifesto: item.manifesto, modelo_sha: "aa" });
    expect(viradas([linha({})], [novo!])).toBe("- m: 0 de certa para errada, 1 de errada para certa");
    expect(() => renotar([linha({ pergunta_sha: "0".repeat(64) })], [item], avaliadores, regras)).toThrow(/a pergunta mudou/);
    expect(() => renotar([linha({ id: "nao-existe" })], [item], avaliadores, regras)).toThrow(/não existe/);
    expect(() => renotar([linha({ congelado: { ...FROZEN, sistema: "x" } as typeof FROZEN })], [item], avaliadores, regras))
      .toThrow(/outro congelamento/);
  });

  test("the kit prompt asks for Brazilian Portuguese and a code block; a role item adds its frozen line, then the date", () => {
    const papeis = JSON.parse(readFileSync(join(SET, "papel.json"), "utf8")) as Record<string, string>;
    const itens = carregarItens(), kit = promptDoKit(), data = linhaDeData(NOW);
    expect(kit).toContain("português do Brasil");
    expect(kit).toContain("bloco de código");
    expect(kit.endsWith("\n")).toBe(false);
    const semPapel = itens.find((i) => !i.papel)!, dev = itens.find((i) => i.papel === "dev")!;
    const prod = itens.find((i) => i.papel === "producao" && (i as { par?: string }).par === (dev as { par?: string }).par)!;
    expect(sistemaDoItem(semPapel, kit, papeis, data)).toBe(`${kit}\n${data}`);
    expect(sistemaDoItem(dev, kit, papeis, data)).toBe(`${kit}\n${papeis.dev}\n${data}`);
    expect(sistemaDoItem(prod, kit, papeis, data)).toBe(`${kit}\n${papeis.producao}\n${data}`);
    expect(prod.pergunta).toBe(dev.pergunta);
  });

  test("the date line names the weekday, the day, the month and the time in Portuguese", () => {
    expect(linhaDeData("2026-11-10T10:00")).toBe("Agora é terça-feira, 10 de novembro de 2026, 10:00.");
    expect(linhaDeData("2028-02-29T23:59")).toBe("Agora é terça-feira, 29 de fevereiro de 2028, 23:59.");
    expect(linhaDeData("2026-03-01T07:05")).toBe("Agora é domingo, 1 de março de 2026, 07:05.");
    expect(() => linhaDeData("10/11/2026")).toThrow(/data inválida/);
  });

  test("items of a family with no command list (pacman, brew) wait for v2; the others all run", () => {
    const regras = carregarRegras(), itens = carregarItens();
    const fora = itens.filter((i) => semLista(i, regras));
    expect(fora.length).toBeGreaterThan(0);
    expect(new Set(fora.map((i) => i.familia))).toEqual(new Set(["pacman", "brew"]));
    expect(itens.filter((i) => i.familia === "powershell" || i.familia === "apt" || i.familia === "-")
      .every((i) => !semLista(i, regras))).toBe(true);
  });
});

describe("decision rule", () => {
  test("Wilson 95%: 24 of 48 is 36% to 64%, as in the plan", () => {
    const w = wilson(24, 48);
    expect(w.baixo).toBeCloseTo(0.3637, 3);
    expect(w.alto).toBeCloseTo(0.6363, 3);
    expect(wilson(0, 60).alto).toBeCloseTo(0.0602, 3);
    expect(wilson(2, 60).alto).toBeCloseTo(0.1136, 3);
  });

  test("the sign test wins at 9 of 10, 12 of 15 and 15 of 20, not at 8 of 10", () => {
    expect(testeDeSinal(9, 1)).toBeCloseTo(0.01074, 4);
    expect(testeDeSinal(12, 3)).toBeCloseTo(0.01758, 4);
    expect(testeDeSinal(15, 5)).toBeCloseTo(0.02069, 4);
    expect(testeDeSinal(8, 2)).toBeGreaterThan(0.05);
    expect(testeDeSinal(0, 0)).toBe(1);
  });

  test("the summary pairs two models on the same items, both ways, and marks a pilot", () => {
    const r = (modelo: string, id: string, certa: boolean) => ({ modelo, id, bloco: "comando", certa,
      motivo: certa ? null : "sem bloco de comando", segundos: 1, resposta: "", manifesto: "", piloto: true });
    const ids = Array.from({ length: 10 }, (_, i) => `c${i}`);
    const texto = resumo([...ids.map((id) => r("a", id, false)), ...ids.map((id, i) => r("b", id, i < 9))], 3);
    expect(texto).toContain("**Piloto:** não vale para a decisão.");
    expect(texto).toContain("Fora da taxa até a v2 do conjunto: 3 item(ns)");
    expect(texto).toContain("- comando: 9/10");
    expect(texto).toContain("b contra a: conserta 9, quebra 0; b melhor: p = 0.0020 (ganhou); a melhor: p = 1.0000");
    /* Each set as a whole too: the kill rule looks at the maintenance questions alone. */
    const duvidas = CONJUNTOS.find((c) => c.itens === "duvidas.json")!.sha!;
    const porConjunto = resumo([{ ...r("q", "win-001", true), manifesto: duvidas }, { ...r("q", "win-002", false), manifesto: duvidas },
      { ...r("q", "cmd-001", true), manifesto: CONJUNTOS[0]!.sha! }]);
    expect(porConjunto).toContain("- conjunto duvidas.json: 1/2 = 50.0%");
    expect(porConjunto).toContain("- conjunto manutencao.json: 1/1 = 100.0%");
    /* And without the pilot's 11 items, which the prompt was tuned on. */
    expect(ITENS_DO_PILOTO.size).toBe(11);
    expect(porConjunto).toContain("- sem os 1 itens do piloto: 1/2 = 50.0%");
  });
});

describe("Portuguese detector", () => {
  test.each([["referencias.json", 70], ["referencias-duvidas.json", 85]])(
    "passes every right reference answer of %s", (arquivo, minimo) => {
      const refs = JSON.parse(readFileSync(join(SET, arquivo), "utf8")) as
        Record<string, { certa?: string; certas?: string[] }>;
      const textos = Object.values(refs).flatMap((r) => [r.certa, ...(r.certas ?? [])]).filter((t): t is string => !!t);
      expect(textos.length).toBeGreaterThan(minimo);
      for (const t of textos) expect([t, emPortugues(t)]).toEqual([t, true]);
    });

  test("rejects English and Spanish prose, whatever the code says, and a tie is not Portuguese", () => {
    expect(emPortugues("Use `Get-Service` to list them.")).toBe(false); /* the adversary's case: "use" is English too */
    expect(emPortugues("Use the following command to check the free space on each disk, then run it again.")).toBe(false);
    expect(emPortugues("Para ver el espacio libre, ejecute el comando y luego revise los discos con cuidado.")).toBe(false);
    expect(emPortugues("Para ver o espaço livre de cada disco, rode o comando abaixo:\n```powershell\nGet-Volume | Format-Table -AutoSize\n```")).toBe(true);
    expect(emPortugues("```bash\nsudo apt update && sudo apt upgrade -y\n```")).toBe(true);
  });

  test("words spelled the same in Portuguese and Spanish count for both (the reviewer's three Spanish sentences)", () => {
    expect(emPortugues("Para ver el espacio libre del disco, use este comando que muestra el uso de cada partición:")).toBe(false);
    expect(emPortugues("Ejecute este comando en PowerShell para ver los procesos que más memoria usan:")).toBe(false);
    expect(emPortugues("Puede usar el siguiente comando para reiniciar el servicio de impresión en Windows:")).toBe(false);
    expect(emPortugues("Para ver os processos que mais usam memória, rode este comando no PowerShell:")).toBe(true);
    /* A domain is not prose: example.com does not make "com" Portuguese. */
    expect(emPortugues("Download the installer from example.com and run it as administrator.")).toBe(false);
  });

  test("a tie goes to Portuguese only when the other language has no word of its own (the reviewer's low)", () => {
    expect(emPortugues("Use este comando para ver se está cheio:")).toBe(true);
    expect(emPortugues("Use este comando para ver si está lleno:")).toBe(false);
    expect(emPortugues("Use do as no execute here.")).toBe(false);
  });

  test("works inside nota.ts: an English answer with the right command is wrong for the language only", () => {
    const item = carregarItens().find((i) => i.id === "cmd-001")!;
    const nota = avaliar(item, "To see the free space of each disk, use this command and check the output:\n```powershell\nGet-Volume\n```",
      carregarRegras());
    expect(nota).toMatchObject({ certa: false, motivo: "fora do português" });
  });
});

const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");

describe.skipIf(!qwen)("one item through the real chat", () => {
  test("cmd-001 on the Qwen 0.5B with the kit prompt: one JSONL line, graded, and the prompt cache written", () => {
    const work = tempDir("eval-manutencao");
    const eval_ = (saida: string) => Bun.spawnSync(["bun", join(ROOT, "scripts", "eval-manutencao.ts"), "--modelo",
      join(ROOT, "models", "qwen2.5-0.5b-instruct"), "--itens", "cmd-001", "--saida", saida, "--piloto"],
    { stdout: "pipe", stderr: "pipe" });
    try {
      const saida = join(work.path, "r.jsonl");
      const r = eval_(saida);
      expect(r.exitCode).toBe(0);
      const linhas = readFileSync(saida, "utf8").trim().split("\n");
      expect(linhas).toHaveLength(1);
      const res = JSON.parse(linhas[0]!) as { id: string; resposta: string; manifesto: string; piloto: boolean };
      expect(res).toMatchObject({ id: "cmd-001", manifesto: CONJUNTOS[0]!.sha, congelado: FROZEN, piloto: true });
      expect(res.resposta.length).toBeGreaterThan(0);
      expect(r.stdout.toString()).toContain("## qwen2.5-0.5b-instruct");
      expect(readdirSync(join(ROOT, "build", "eval-cache")).some((f) => /^qwen2\.5-0\.5b-instruct-[0-9a-f]{12}\.kv$/.test(f)))
        .toBe(true);
      /* A second run finds the item done and asks nothing. */
      expect(eval_(saida).exitCode).toBe(0);
      expect(readFileSync(saida, "utf8").trim().split("\n")).toHaveLength(1);
    } finally {
      work.dispose();
    }
  }, SLOW);
});
