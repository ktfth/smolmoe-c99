#!/usr/bin/env bun
/**
 * Gera ambiente-casos.json e MANIFEST-ambiente.sha256: a tabela congelada de como o runtime decide o ambiente da
 * máquina a partir de SMOLMOE_ENV (requisito fechado pelo dono: dev|development|prod|production; pela revisão 22 do
 * plano, é a única fonte do papel). É uma especificação para um teste
 * determinístico do runtime, não um item de modelo: o modelo só vê a linha de papel que o runtime decidiu.
 *
 * Regras usadas na tabela (as duas primeiras batem com o plano, revisão 22, e estão perguntadas ao dono):
 *  1. o valor é comparado depois de tirar os brancos das pontas (espaço, tab, CR, LF) e sem diferenciar maiúsculas;
 *  2. vazio (SMOLMOE_ENV=) vale como ausente: "não declarado";
 *  3. dev|development → dev; prod|production → produção (dono);
 *  4. qualquer outro valor → produção, com aviso "valor inválido" (plano, 4a: inválido vale como produção);
 *  5. sem a variável → produção, com aviso "não declarado" (plano, 4a);
 *  6. a comparação é ASCII, sem locale (no tr_TR, o tolower de "I" daria "ı"); só os brancos ASCII são aparados (um
 *     NBSP não é branco: o valor fica inválido, e vale produção);
 *  7. o aviso de valor inválido mostra o valor com os caracteres de controle (C0, DEL e C1, onde U+009B é CSI) como
 *     \xNN, cortado em 64 caracteres (+ "…"); o corte conta caracteres, nunca parte um UTF-8 ao meio.
 *
 * uso: bun gerar-ambiente.ts
 */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Papel = "dev" | "producao";
type Aviso = "nenhum" | "nao_declarado" | "invalido";
interface Caso {
  id: string;
  /** null = a variável não existe no ambiente; "" = existe e está vazia. Brancos aparecem como estão (JSON). */
  SMOLMOE_ENV: string | null;
  esperado: Papel;
  aviso: Aviso;
  /** Só no aviso "invalido": o valor como o aviso deve mostrá-lo (bytes de controle como \\xNN, no máximo 64
   *  caracteres e "…" se for maior). Evita sequência de terminal (ESC) e linha enorme no aviso. */
  mostra?: string;
  nota: string;
}

/** A regra, escrita uma vez, para conferir a tabela à mão escrita abaixo (e para o dev comparar com o C). */
function normaliza(v: string): Papel | "invalido" {
  const t = v.replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "").toLowerCase();
  if (t === "dev" || t === "development") return "dev";
  if (t === "prod" || t === "production") return "producao";
  return "invalido";
}
function mostra(v: string): string {
  const esc = Array.from(v, (ch) => {
    const c = ch.codePointAt(0)!;
    return c < 0x20 || (c >= 0x7f && c <= 0x9f) ? "\\x" + c.toString(16).padStart(2, "0") : ch;
  }).join("");
  const cps = Array.from(esc);
  return cps.length > 64 ? cps.slice(0, 64).join("") + "\u2026" : esc;
}
function decide(c: Pick<Caso, "SMOLMOE_ENV">): { esperado: Papel; aviso: Aviso } {
  if (c.SMOLMOE_ENV === null || c.SMOLMOE_ENV.trim() === "") return { esperado: "producao", aviso: "nao_declarado" };
  const l = normaliza(c.SMOLMOE_ENV);
  return l === "invalido" ? { esperado: "producao", aviso: "invalido" } : { esperado: l, aviso: "nenhum" };
}

const casos: Omit<Caso, "id">[] = [
  /* só a variável */
  { SMOLMOE_ENV: null, esperado: "producao", aviso: "nao_declarado", nota: "ausente" },
  { SMOLMOE_ENV: "", esperado: "producao", aviso: "nao_declarado", nota: "vazia (SMOLMOE_ENV=)" },
  { SMOLMOE_ENV: "   ", esperado: "producao", aviso: "nao_declarado", nota: "só espaços" },
  { SMOLMOE_ENV: "dev", esperado: "dev", aviso: "nenhum", nota: "dev" },
  { SMOLMOE_ENV: "development", esperado: "dev", aviso: "nenhum", nota: "development" },
  { SMOLMOE_ENV: "prod", esperado: "producao", aviso: "nenhum", nota: "prod" },
  { SMOLMOE_ENV: "production", esperado: "producao", aviso: "nenhum", nota: "production" },
  { SMOLMOE_ENV: "PROD", esperado: "producao", aviso: "nenhum", nota: "maiúsculas" },
  { SMOLMOE_ENV: "Production", esperado: "producao", aviso: "nenhum", nota: "inicial maiúscula" },
  { SMOLMOE_ENV: "DEV", esperado: "dev", aviso: "nenhum", nota: "maiúsculas em dev (regra 1: escolha a confirmar)" },
  { SMOLMOE_ENV: "Development", esperado: "dev", aviso: "nenhum", nota: "inicial maiúscula em dev" },
  { SMOLMOE_ENV: " dev", esperado: "dev", aviso: "nenhum", nota: "espaço antes" },
  { SMOLMOE_ENV: "prod ", esperado: "producao", aviso: "nenhum", nota: "espaço depois" },
  { SMOLMOE_ENV: "\tdev\r\n", esperado: "dev", aviso: "nenhum", nota: "tab e CRLF (valor copiado de um arquivo do Windows)" },
  { SMOLMOE_ENV: "d ev", esperado: "producao", aviso: "invalido", nota: "espaço no meio" },
  { SMOLMOE_ENV: "producao", esperado: "producao", aviso: "invalido", nota: "em português: não é um dos valores" },
  { SMOLMOE_ENV: "produção", esperado: "producao", aviso: "invalido", nota: "em português com acento" },
  { SMOLMOE_ENV: "desenvolvimento", esperado: "producao", aviso: "invalido", nota: "dev em português: inválido vale produção" },
  { SMOLMOE_ENV: "staging", esperado: "producao", aviso: "invalido", nota: "outro nome comum" },
  { SMOLMOE_ENV: "test", esperado: "producao", aviso: "invalido", nota: "outro nome comum" },
  { SMOLMOE_ENV: "devel", esperado: "producao", aviso: "invalido", nota: "abreviação fora da lista" },
  { SMOLMOE_ENV: "dev,prod", esperado: "producao", aviso: "invalido", nota: "dois valores" },
  { SMOLMOE_ENV: "1", esperado: "producao", aviso: "invalido", nota: "número" },
  { SMOLMOE_ENV: "\"dev\"", esperado: "producao", aviso: "invalido", nota: "aspas dentro do valor (erro de .env)" },
  { SMOLMOE_ENV: "\u001b[31mprod", esperado: "producao", aviso: "invalido", mostra: "\\x1b[31mprod", nota: "ESC: o aviso não pode mandar a sequência ao terminal (AM-2)" },
  { SMOLMOE_ENV: "dev\u0007", esperado: "producao", aviso: "invalido", mostra: "dev\\x07", nota: "BEL no fim: não é branco, fica inválido" },
  { SMOLMOE_ENV: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", esperado: "producao", aviso: "invalido", mostra: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx…", nota: "valor de 300 caracteres: o aviso corta em 64 (AM-2)" },
  { SMOLMOE_ENV: "\u00a0dev", esperado: "producao", aviso: "invalido", mostra: "\u00a0dev", nota: "NBSP antes: só brancos ASCII são aparados" },
  { SMOLMOE_ENV: "PRODUCTION", esperado: "producao", aviso: "nenhum", nota: "comparação ASCII sem locale (tr_TR: I → ı)" },
  { SMOLMOE_ENV: "\u009b31mprod", esperado: "producao", aviso: "invalido", mostra: "\\x9b31mprod", nota: "CSI de 8 bits (C1): escapado como \\x9b" },
  { SMOLMOE_ENV: "çççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççç", esperado: "producao", aviso: "invalido", mostra: "çççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççç…", nota: "300 × ç: o corte conta caracteres, não bytes" },
  { SMOLMOE_ENV: "prod\u007f", esperado: "producao", aviso: "invalido", mostra: "prod\\x7f", nota: "DEL no fim: escapado como \\x7f" },
];

let falhas = 0;
const tabela: Caso[] = casos.map((c, i) => {
  const id = `amb-${String(i + 1).padStart(2, "0")}`;
  const d = decide(c);
  const m = c.aviso === "invalido" ? mostra(c.SMOLMOE_ENV!) : undefined;
  if (c.aviso === "invalido" && c.mostra !== undefined && c.mostra !== m) { console.error(`${id}: mostra ${JSON.stringify(c.mostra)}, a regra dá ${JSON.stringify(m)}`); falhas++; }
  if (d.esperado !== c.esperado || d.aviso !== c.aviso) {
    console.error(`${id} (${c.nota}): a tabela diz ${c.esperado}/${c.aviso}, a regra dá ${d.esperado}/${d.aviso}`);
    falhas++;
  }
  return { id, ...c, ...(c.aviso === "invalido" ? { mostra: m } : {}) };
});
if (falhas) { console.error(`${falhas} falha(s): nada foi escrito`); process.exit(1); }

const DIR = import.meta.dir;
writeFileSync(join(DIR, "ambiente-casos.json"), `${JSON.stringify({ versao: 3, variavel: "SMOLMOE_ENV", casos: tabela }, null, 2)}\n`);
const sha = (f: string) => createHash("sha256").update(readFileSync(join(DIR, f))).digest("hex");
writeFileSync(join(DIR, "MANIFEST-ambiente.sha256"), `${sha("ambiente-casos.json")}  ambiente-casos.json\n`);
console.log(`${tabela.length} casos; a tabela bate com a regra`);
console.log(readFileSync(join(DIR, "MANIFEST-ambiente.sha256"), "utf8").trim());
