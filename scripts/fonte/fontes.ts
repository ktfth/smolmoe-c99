/**
 * Turns the offline sources into small passages for the source test (plan, revisions 28 to 31). Only for the
 * experiment: nothing here goes into the C runtime before the result.
 *
 * Sources, under SMOLMOE_FONTES (default ../fontes-offline, next to the repository):
 * - sa/support/windows-client: MicrosoftDocs/SupportArticles-docs, CC BY 4.0;
 * - psdocs/docset/winserver2022-ps: MicrosoftDocs/windows-powershell-docs, CC BY 4.0;
 * - docker/content/{manuals,reference/cli}: docker/docs, Apache 2.0;
 * - archwiki/usr/share/doc/arch-wiki/html/en: arch-wiki-docs, GFDL 1.3 or later.
 */
import { join, relative } from "node:path";

export type Trecho = { readonly id: number; readonly fonte: string; readonly caminho: string; readonly texto: string };

export const PALAVRAS_POR_TRECHO = 80;

type Fonte = { readonly nome: string; readonly dir: string; readonly padrao: string; readonly ler: (bruto: string) => Secao[] };
type Secao = { readonly titulo: string; readonly corpo: string };

export function raizDasFontes(): string {
  return process.env.SMOLMOE_FONTES ?? join(import.meta.dir, "..", "..", "..", "fontes-offline");
}

export function fontes(raiz: string): readonly Fonte[] {
  return [
    { nome: "windows-client", dir: join(raiz, "sa", "support", "windows-client"), padrao: "**/*.md", ler: secoesMarkdown },
    { nome: "powershell", dir: join(raiz, "psdocs", "docset", "winserver2022-ps"), padrao: "**/*.md", ler: secoesMarkdown },
    { nome: "docker", dir: join(raiz, "docker", "content", "manuals"), padrao: "**/*.md", ler: secoesMarkdown },
    { nome: "docker", dir: join(raiz, "docker", "content", "reference", "cli"), padrao: "**/*.md", ler: secoesMarkdown },
    { nome: "archwiki", dir: join(raiz, "archwiki", "usr", "share", "doc", "arch-wiki", "html", "en"), padrao: "*.html", ler: secoesHtml },
    { nome: "wsl", dir: join(raiz, "wsl", "WSL"), padrao: "**/*.md", ler: secoesMarkdown },
    { nome: "debian-reference", dir: join(raiz, "debref", "rawxml"), padrao: "*.rawxml", ler: secoesDocbook },
  ];
}

/** Debian Reference (DocBook): one section per <title>; table rows become lines. */
export function secoesDocbook(bruto: string): Secao[] {
  const s = bruto.replace(/<!--[\s\S]*?-->/g, "").replace(/@-@codename-stable@-@/g, "stable").replace(/@-@[\w-]+@-@/g, "")
    .replace(/<\/(row|para|listitem|screen|title)>/gi, (m) => (m.toLowerCase() === "</title>" ? "\u0002" : "\n"))
    .replace(/<title>/gi, "\u0001").replace(/<\/entry>/gi, " ").replace(/<[^>]+>/g, "");
  const secoes: Secao[] = [];
  let livro = "";
  for (const parte of decodificar(s).split("\u0001")) {
    const fim = parte.indexOf("\u0002");
    if (fim < 0) continue;
    const titulo = parte.slice(0, fim).trim(), corpo = parte.slice(fim + 1);
    if (!livro) livro = titulo;
    if (corpo.trim()) secoes.push({ titulo: titulo === livro ? livro : `${livro} > ${titulo}`, corpo });
  }
  return secoes;
}

const ENTIDADES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: " ", "#39": "'" };

function decodificar(s: string): string {
  return s.replace(/&(#x[0-9a-f]+|#\d+|\w+);/gi, (m, e: string) => {
    if (e[0] === "#") {
      const n = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return Number.isFinite(n) && n > 0 && n < 0x110000 ? String.fromCodePoint(n) : m;
    }
    return ENTIDADES[e.toLowerCase()] ?? m;
  });
}

/** Markdown: drops the front matter, comments, images, Hugo shortcodes and link targets; keeps code, which is where
 * the commands are. Splits at headings. */
export function secoesMarkdown(bruto: string): Secao[] {
  let s = bruto.replace(/^﻿/, "").replace(/\r\n?/g, "\n");
  const titulo = /^---\n[\s\S]*?^title:\s*["']?(.+?)["']?\s*$/m.exec(s)?.[1] ?? "";
  s = s.replace(/^---\n[\s\S]*?\n---\n/, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/\{\{[<%][\s\S]*?[>%]\}\}/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^\s*:::.*$/gm, "")
    .replace(/<[^>\n]+>/g, "");
  const secoes: Secao[] = [];
  let atual = titulo, corpo: string[] = [];
  for (const linha of s.split("\n")) {
    const h = /^#{1,6}\s+(.*)$/.exec(linha);
    if (h) {
      if (corpo.join("").trim()) secoes.push({ titulo: atual, corpo: corpo.join("\n") });
      atual = titulo && h[1] !== titulo ? `${titulo} > ${h[1]}` : h[1]!;
      corpo = [];
    } else corpo.push(linha);
  }
  if (corpo.join("").trim()) secoes.push({ titulo: atual, corpo: corpo.join("\n") });
  return secoes.map((x) => ({ titulo: decodificar(x.titulo), corpo: decodificar(x.corpo) }));
}

/** ArchWiki page: the article body only, split at h2/h3. */
export function secoesHtml(bruto: string): Secao[] {
  const titulo = decodificar(/<h1[^>]*>([\s\S]*?)<\/h1>/i.exec(bruto)?.[1]?.replace(/<[^>]+>/g, "").trim() ?? "");
  const inicio = bruto.search(/<div[^>]+class="mw-parser-output"/i);
  let s = inicio >= 0 ? bruto.slice(inicio) : bruto;
  s = s.replace(/<(script|style)[\s\S]*?<\/\1>/gi, "")
    .replace(/<div[^>]+(id="toc"|class="[^"]*(navbox|printfooter|catlinks)[^"]*")[\s\S]*?<\/div>/gi, "")
    .replace(/<h[23][^>]*>([\s\S]*?)<\/h[23]>/gi, (_, t: string) => `\n\u0001${t.replace(/<[^>]+>/g, "").trim()}\n`)
    .replace(/<\/?(code|pre)\b[^>]*>/gi, "`")
    .replace(/<(br|\/p|\/li|\/tr|\/h\d)[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, "");
  /* The first part is the introduction; every other part starts with its heading line. */
  const [intro = "", ...partes] = decodificar(s).split("\u0001");
  const secoes: Secao[] = intro.trim() ? [{ titulo, corpo: intro }] : [];
  for (const parte of partes) {
    const quebra = parte.indexOf("\n");
    const cab = (quebra < 0 ? parte : parte.slice(0, quebra)).trim(), corpo = quebra < 0 ? "" : parte.slice(quebra + 1);
    if (corpo.trim()) secoes.push({ titulo: `${titulo} > ${cab}`, corpo });
  }
  return secoes;
}

/** Cuts each section into passages of about PALAVRAS_POR_TRECHO words, each one led by its title. */
export function fatiar(secoes: readonly Secao[]): string[] {
  const out: string[] = [];
  for (const { titulo, corpo } of secoes) {
    const palavras = corpo.replace(/[ \t]+/g, " ").replace(/\n\s*\n+/g, "\n").trim().split(/ +/);
    for (let i = 0; i < palavras.length; i += PALAVRAS_POR_TRECHO) {
      const pedaco = palavras.slice(i, i + PALAVRAS_POR_TRECHO).join(" ").trim();
      if (pedaco.split(/\s+/).length >= 8) out.push(`${titulo}\n${pedaco}`);
    }
  }
  return out;
}

const categorias = (html: string) => new Set([...html.matchAll(/title="Category:([^"]+)"/g)].map((m) => decodificar(m[1]!)));

/** ArchWiki pages about one laptop model (the subcategories of Category:Laptops, by vendor) are a list of hardware
 * quirks that match almost any question; category listing pages carry no text. Both stay out, by topic. */
async function paginaFora(dir: string, caminhos: readonly string[]): Promise<(caminho: string, html: string) => boolean> {
  const lista = caminhos.find((c) => nomeDaPagina(c) === "Category:Laptops.html");
  if (lista === undefined) throw new Error(`falta Category:Laptops.html em ${dir}`);
  const laptops = categorias(await Bun.file(join(dir, lista)).text());
  laptops.delete("Hardware");
  laptops.add("Laptops");
  return (caminho, html) => nomeDaPagina(caminho).startsWith("Category:") || [...categorias(html)].some((c) => laptops.has(c));
}

/* MSYS tar on Windows writes the ":" of a page name as U+F03A, since Windows file names cannot hold it. */
const nomeDaPagina = (caminho: string) => caminho.replaceAll("", ":");

export async function montarTrechos(raiz: string): Promise<Trecho[]> {
  const trechos: Trecho[] = [];
  for (const f of fontes(raiz)) {
    const caminhos = [...new Bun.Glob(f.padrao).scanSync({ cwd: f.dir })].sort();
    if (caminhos.length === 0) throw new Error(`fonte vazia ou ausente: ${f.dir}`);
    const fora = f.nome === "archwiki" ? await paginaFora(f.dir, caminhos) : () => false;
    for (const c of caminhos) {
      const bruto = await Bun.file(join(f.dir, c)).text();
      if (fora(c, bruto)) continue;
      for (const texto of fatiar(f.ler(bruto))) {
        trechos.push({ id: trechos.length, fonte: f.nome, caminho: relative(raiz, join(f.dir, c)).replaceAll("\\", "/"), texto });
      }
    }
  }
  return trechos;
}
