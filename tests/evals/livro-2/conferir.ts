/*
 * Confere o conjunto liv2 (perguntas.json + referencias.json) com a mesma nota de scripts/cmdcheck/eval-docs.ts
 * (funções blocos e util, copiadas sem mudar a regra).
 *
 *   bun conferir.ts [perguntas.json] [referencias.json]     estrutura + nota das referências
 *   bun conferir.ts --comandos [...]                        também confere que os comandos das respostas certas existem
 *
 * Sai com código 1 se alguma resposta certa reprovar, alguma errada passar ou a estrutura estiver fora do combinado.
 * --comandos usa pwsh, powershell.exe (5.1) e o WSL (distro em LIVRO2_WSL_DISTRO, padrão warp-sandbox).
 */
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";

interface Pergunta {
  readonly id: string;
  readonly shell: "powershell" | "posix";
  readonly alvo?: "comando" | "resposta";
  readonly pergunta: string;
  readonly esperado?: readonly string[];
  readonly obrigatorios?: readonly string[];
  readonly proibidos?: readonly string[];
}
interface Referencia {
  readonly id: string;
  readonly estilo: "dia a dia" | "técnico";
  readonly certa: string;
  readonly errada: string;
  readonly verificar: readonly string[];
}

const SUFIXO = "Ponha o comando recomendado num bloco de código (só ele em bloco).";
const args = process.argv.slice(2);
const comandos = args.includes("--comandos");
const posicionais = args.filter((a) => !a.startsWith("--"));
const arqPerguntas = posicionais[0] ?? join(import.meta.dir, "perguntas.json");
const arqReferencias = posicionais[1] ?? join(import.meta.dir, "referencias.json");

/* --- a nota, igual à de scripts/cmdcheck/eval-docs.ts --- */
function blocos(resposta: string): string {
  return [...resposta.matchAll(/```[^\n]*\n([\s\S]*?)(```|$)/g)].map((m) => m[1]).join("\n");
}
function util(p: Pergunta, resposta: string): boolean {
  if (p.esperado) return p.esperado.some((e) => new RegExp(e, "i").test(resposta));
  const alvo = p.alvo === "resposta" ? resposta : blocos(resposta);
  if (!alvo.trim()) return false;
  return (p.obrigatorios ?? []).every((e) => new RegExp(e, "iu").test(alvo)) &&
    !(p.proibidos ?? []).some((e) => new RegExp(e, "iu").test(alvo));
}
/* --- fim da nota --- */

/* Por que um texto reprova: o primeiro obrigatório que não casa ou o primeiro proibido que casa. */
function motivo(p: Pergunta, resposta: string): string {
  const alvo = blocos(resposta);
  if (!alvo.trim()) return "sem bloco de código";
  const falta = (p.obrigatorios ?? []).find((e) => !new RegExp(e, "iu").test(alvo));
  if (falta) return `falta /${falta}/`;
  const proibido = (p.proibidos ?? []).find((e) => new RegExp(e, "iu").test(alvo));
  return proibido ? `proibido /${proibido}/` : "passa";
}

const perguntas = (JSON.parse(readFileSync(arqPerguntas, "utf8")) as { perguntas: Pergunta[] }).perguntas;
const referencias = (JSON.parse(readFileSync(arqReferencias, "utf8")) as { referencias: Referencia[] }).referencias;
const erros: string[] = [];
const erro = (m: string): void => void erros.push(m);

/* Estrutura */
if (perguntas.length !== 120) erro(`são ${perguntas.length} perguntas, não 120`);
perguntas.forEach((p, i) => {
  const esperado = `liv2-${String(i + 1).padStart(3, "0")}`;
  if (p.id !== esperado) erro(`posição ${i + 1}: id ${p.id}, esperado ${esperado}`);
  if (p.shell !== "powershell" && p.shell !== "posix") erro(`${p.id}: shell ${p.shell}`);
  if (p.alvo !== "comando") erro(`${p.id}: alvo ${p.alvo}`);
  if (!p.pergunta.endsWith(SUFIXO)) erro(`${p.id}: a pergunta não termina com o pedido de bloco`);
  if (!p.obrigatorios?.length) erro(`${p.id}: sem obrigatórios`);
  for (const e of [...(p.obrigatorios ?? []), ...(p.proibidos ?? [])]) {
    try { new RegExp(e, "iu"); } catch (x) { erro(`${p.id}: regex inválida /${e}/: ${(x as Error).message}`); }
  }
});
const ids = new Set(perguntas.map((p) => p.id));
if (ids.size !== perguntas.length) erro("ids repetidos");
const textos = new Set(perguntas.map((p) => p.pergunta));
if (textos.size !== perguntas.length) erro("perguntas repetidas");
if (referencias.length !== perguntas.length) erro(`são ${referencias.length} referências para ${perguntas.length} perguntas`);

/* Contagem por shell e estilo */
const contagem = new Map<string, number>();
for (const p of perguntas) {
  const ref = referencias.find((r) => r.id === p.id);
  if (!ref) { erro(`${p.id}: sem referência`); continue; }
  const chave = `${p.shell} / ${ref.estilo}`;
  contagem.set(chave, (contagem.get(chave) ?? 0) + 1);
}
for (const shell of ["powershell", "posix"]) {
  for (const estilo of ["dia a dia", "técnico"]) {
    const n = contagem.get(`${shell} / ${estilo}`) ?? 0;
    console.log(`${shell} / ${estilo}: ${n}`);
    if (n !== 30) erro(`${shell} / ${estilo}: ${n}, esperado 30`);
  }
}

/* Nota das referências */
let certasOk = 0;
let erradasOk = 0;
for (const p of perguntas) {
  const ref = referencias.find((r) => r.id === p.id);
  if (!ref) continue;
  const nBlocos = [...ref.certa.matchAll(/```[^\n]*\n[\s\S]*?```/g)].length;
  if (nBlocos !== 1) erro(`${p.id}: a resposta certa tem ${nBlocos} blocos de código, não 1`);
  if (util(p, ref.certa)) certasOk++; else erro(`${p.id}: a certa reprova (${motivo(p, ref.certa)})`);
  if (!util(p, ref.errada)) erradasOk++; else erro(`${p.id}: a errada passa`);
}
console.log(`certas que passam: ${certasOk}/${perguntas.length}`);
console.log(`erradas que reprovam: ${erradasOk}/${perguntas.length}`);

/* --- conferência dos comandos das respostas certas --- */
function rodar(cmd: string, argv: readonly string[], input?: string): { code: number; out: string } {
  /* O powershell.exe 5.1 roda sem o PSModulePath herdado: o do pwsh 7 faz ele carregar módulos errados e "perder" cmdlets. */
  const env = cmd !== "powershell.exe" ? process.env
    : Object.fromEntries(Object.entries(process.env).filter(([k]) => k.toLowerCase() !== "psmodulepath"));
  const r = spawnSync(cmd, argv, { input, env, encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.error) return { code: 127, out: String(r.error) };
  return { code: r.status ?? 1, out: `${r.stdout ?? ""}${r.stderr ?? ""}` };
}

/* Template comum, sem String.raw: o Bun 1.2 estraga o texto não ASCII dentro de String.raw. */
const SCRIPT_PS = `
$ErrorActionPreference = 'Stop'
$itens = Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json
$e51 = $PSVersionTable.PSVersion.Major -lt 6
foreach ($it in $itens) {
  $s = [string]$it.spec
  try {
    if ($s.StartsWith('51:')) { if (-not $e51) { continue }; $s = $s.Substring(3) }
    if ($s.StartsWith('ajuda:')) {
      $i = $s.LastIndexOf(':'); $linha = $s.Substring(6, $i - 6); $txt = $s.Substring($i + 1)
      $o = cmd.exe /c "$linha 2>&1" | Out-String
      if ($o -notmatch [regex]::Escape($txt)) { throw "a ajuda de '$linha' não cita $txt" }
      continue
    }
    if ($s.StartsWith('exe:')) { $null = Get-Command $s.Substring(4) -CommandType Application; continue }
    if ($s.StartsWith('svc:')) { $null = Get-Service $s.Substring(4); continue }
    if ($s.StartsWith('cim:')) {
      $p = $s.Split(':'); $c = Get-CimClass -ClassName $p[1]
      if (-not ($c.CimClassProperties.Name -contains $p[2])) { throw "a classe $($p[1]) não tem $($p[2])" }
      continue
    }
    $p = $s.Split(':'); $cmd = Get-Command $p[0]
    if ($p.Length -gt 1) {
      foreach ($par in $p[1].Split(',')) { if (-not $cmd.Parameters.ContainsKey($par)) { throw "$($p[0]) não tem -$par" } }
    }
  } catch { $t = [char]9; "FALHA$t$($it.id)$t$($it.spec)$t$($_.Exception.Message)" }
}
"FIM"
`;

const SCRIPT_SH = String.raw`
CACHE=$(mktemp -d /tmp/livro2-XXXXXX)
manq() { LC_ALL=C MANWIDTH=1000 man "$1" 2>/dev/null | grep -qF -- "$2"; }
helpq() { LC_ALL=C sh -c "$1 --help" 2>&1 | grep -qF -- "$2"; }
pkgcmd() {
  for f in /var/lib/apt/lists/*_cnf_Commands-amd64; do /usr/lib/apt/apt-helper cat-file "$f"; done 2>/dev/null |
    awk -v p="$1" -v c="$2" '/^name: /{n=$2} /^commands: /{ if (n==p) { m=split(substr($0,11),a,","); for (i=1;i<=m;i++) if (a[i]==c) f=1 } } END{exit !f}'
}
debman() {
  d="$CACHE/$1"
  if [ ! -d "$d" ]; then
    mkdir -p "$d" && (cd "$d" && apt-get download "$1" >/dev/null 2>&1) && dpkg-deb -x "$d"/*.deb "$d/x"
  fi
  sec=$(echo "$2" | sed 's/.*\.//')
  f=$(ls "$d/x/usr/share/man/man$sec/$2"* 2>/dev/null | head -n 1)
  [ -n "$f" ] && LC_ALL=C MANWIDTH=1000 man -l "$f" 2>/dev/null | grep -qF -- "$3"
}
while IFS='	' read -r id tipo a b c; do
  case "$tipo" in
    cmd) command -v "$a" >/dev/null && man -w "$a" >/dev/null 2>&1 ;;
    man) manq "$a" "$b" ;;
    help) helpq "$a" "$b" ;;
    sh) sh -c "$a" >/dev/null 2>&1 ;;
    pkg) pkgcmd "$a" "$b" ;;
    deb) debman "$a" "$b" "$c" ;;
    *) false ;;
  esac || printf 'FALHA\t%s\t%s %s %s %s\n' "$id" "$tipo" "$a" "$b" "$c"
done
rm -rf "$CACHE"
echo FIM
`;

function partir(texto: string, n: number): string[] {
  const partes = texto.split(":");
  return [...partes.slice(0, n - 1), partes.slice(n - 1).join(":")];
}

function conferirComandos(): void {
  const ps: { id: string; spec: string }[] = [];
  const sh: string[] = [];
  for (const p of perguntas) {
    const ref = referencias.find((r) => r.id === p.id);
    if (!ref) continue;
    if (!ref.verificar.length) erro(`${p.id}: nada para verificar`);
    for (const v of ref.verificar) {
      if (p.shell === "powershell") { ps.push({ id: p.id, spec: v }); continue; }
      const i = v.indexOf(":");
      const tipo = v.slice(0, i);
      const resto = v.slice(i + 1);
      /* sh: leva o resto inteiro; deb: tem 3 campos; os outros, 2. O último campo fica com o resto do texto. */
      const campos = tipo === "sh" ? [resto] : partir(resto, tipo === "deb" ? 3 : 2);
      sh.push([p.id, tipo, ...campos].join("\t"));
    }
  }
  const tmp = mkdtempSync(join(tmpdir(), "livro2-"));
  try {
    const arqPs = join(tmp, "itens.json");
    const arqScript = join(tmp, "conferir.ps1");
    writeFileSync(arqPs, JSON.stringify(ps));
    writeFileSync(arqScript, `﻿${SCRIPT_PS}`);
    for (const [nome, exe] of [["pwsh", "pwsh"], ["powershell 5.1", "powershell.exe"]] as const) {
      const r = rodar(exe, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", arqScript, arqPs]);
      const linhas = r.out.split(/\r?\n/);
      if (!linhas.some((l) => l.trim() === "FIM")) erro(`${nome}: não terminou: ${r.out.slice(0, 500)}`);
      const falhas = linhas.filter((l) => l.startsWith("FALHA"));
      falhas.forEach((l) => erro(`${nome}: ${l.replace(/^FALHA\t/, "").replace(/\t/g, " | ")}`));
      console.log(`${nome}: ${ps.length} verificações, ${falhas.length} falhas`);
    }
    const distro = process.env.LIVRO2_WSL_DISTRO ?? "warp-sandbox";
    const r = rodar("wsl.exe", ["-d", distro, "-e", "sh", "-c", SCRIPT_SH], `${sh.join("\n")}\n`);
    const linhas = r.out.split(/\r?\n/);
    if (!linhas.some((l) => l.trim() === "FIM")) erro(`WSL (${distro}): não terminou: ${r.out.slice(0, 500)}`);
    const falhas = linhas.filter((l) => l.startsWith("FALHA"));
    falhas.forEach((l) => erro(`WSL: ${l.replace(/^FALHA\t/, "").replace(/\t/g, " | ")}`));
    console.log(`WSL (${distro}): ${sh.length} verificações, ${falhas.length} falhas`);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

if (comandos) conferirComandos();

if (erros.length) {
  console.log(`\n${erros.length} problema(s):`);
  for (const e of erros) console.log(`- ${e}`);
  process.exit(1);
}
console.log("\nconjunto conferido: tudo certo");
