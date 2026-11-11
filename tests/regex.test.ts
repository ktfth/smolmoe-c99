/**
 * The C regex matcher (src/regex.c) against JavaScript's RegExp with "iu", on
 * every line of the frozen change list (plan, proposal 4a, revision 23) and of the runtime's additions to it
 * (kit/lista-mudanca-acrescimos.tsv, revision 33):
 * - the subset is closed: whatever the list does not use is refused;
 * - no input where JavaScript matches and C does not; the other way round only
 *   with the two non-ASCII code points "iu" folds into ASCII (long s, Kelvin sign);
 * - matching stays linear on long and hostile inputs.
 */
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { BUILD, ROOT } from "./support/run";

const EXE = process.platform === "win32" ? ".exe" : "";
const REGEX = join(BUILD, `test-regex${EXE}`);
const LISTAS = [join(ROOT, "tests", "evals", "manutencao", "lista-mudanca.tsv"), join(ROOT, "kit", "lista-mudanca-acrescimos.tsv")];

/** Runs every (pattern, input) pair through test-regex: "1", "0" or "E <error>" per pair. In batches: Bun 1.2.15
 * on Windows crashes on a stdin of tens of MB. */
function emC(pares: readonly (readonly [string, string])[]): string[] {
  const enc = new TextEncoder(), out: string[] = [];
  for (let inicio = 0; inicio < pares.length; inicio += 20_000) {
    const lote = pares.slice(inicio, inicio + 20_000), partes: Uint8Array[] = [];
    for (const [p, s] of lote) {
      const pb = enc.encode(p), sb = enc.encode(s);
      partes.push(enc.encode(`${pb.length} ${sb.length}\n`), pb, sb);
    }
    const r = Bun.spawnSync([REGEX], { stdin: Buffer.concat(partes), stdout: "pipe", stderr: "pipe" });
    expect(r.exitCode).toBe(0);
    const linhas = r.stdout.toString().trim().split(/\r?\n/);
    expect(linhas.length).toBe(lote.length);
    out.push(...linhas);
  }
  return out;
}

/* Real commands, at least one for each line of the list, so the comparison sees matches and not only misses. */
const SEMENTES = [
  "rm -rf /tmp/x", "rm --recursive pasta", "Remove-Item C:\\temp -Recurse -Force", "Format-Volume -DriveLetter D",
  "format D: /q", "diskpart", "Clear-Disk -Number 1", "Initialize-Disk 2", "dd if=/dev/zero of=/dev/sdb bs=1M",
  "mkfs.ext4 /dev/sdb1", "wipefs -a /dev/sdb", "shred -u segredo", "fdisk /dev/sda", "parted /dev/nvme0n1",
  "pacman -Rdd pacote", "apt purge nginx", "apt-get remove --purge nginx", "Clear-RecycleBin -Force",
  "del /f /s *.tmp", "rd /s /q C:\\velho", "rmdir /S pasta", "cipher /w:C", "reg delete HKCU\\Software\\X",
  "Remove-ItemProperty -Path HKCU:\\X -Name Y", "bcdedit /set {current} safeboot minimal", "docker system prune -a",
  "docker rm web", "docker rmi imagem", "docker volume rm dados", "chmod -R 777 /", "chown -R ana /",
  "docker compose down -v", "docker-compose down --volumes", "Restart-Computer", "Stop-Computer -Force",
  "shutdown /r /t 0", "shutdown -h now", "Restart-Service spooler", "Stop-Service wuauserv",
  "Set-Service wuauserv -StartupType Disabled", "sc stop spooler", "sc.exe delete servico", "net stop spooler",
  "iisreset", "sudo reboot", "systemctl restart nginx", "service nginx restart", "docker stop web",
  "docker compose stop", "kubectl drain no1", "kubectl scale deploy/web --replicas=0",
  "kubectl rollout restart deploy/web", "winget upgrade --all", "choco upgrade all", "apt upgrade",
  "apt-get dist-upgrade", "pacman -Syu", "pacman -Su", "pacman -R pacote", "dnf upgrade", "brew upgrade",
  "npx prisma migrate deploy", "prisma db push", "python manage.py migrate", "rails db:migrate", "rake db:drop",
  "alembic upgrade head", "flyway migrate", "psql -c 'DROP TABLE t'", "mysql -e 'delete from t where 1'",
  "vssadmin delete shadows /all /quiet", "vssadmin.exe Delete Shadows /For=C:", "wmic shadowcopy delete",
  "timeshift --delete --snapshot X", "timeshift --delete-all", "sudo snapper -c root delete 42", "snapper rm 3",
  "vssadmin list shadows", "timeshift --list", "snapper list",
  "ls -la", "Get-Service", "docker ps", "systemctl status nginx", "apt list --upgradable", "git status",
];

const ESPACOS = [" ", "  ", "\t", "\n", "\u00a0", "\u2003", "\u3000", "\ufeff", "\u202f", "\u2028", "\u1680"];
const KELVIN = String.fromCodePoint(0x212a);
const LIXO = ["x", "_", "9", "é", "É", "ç", "ſ", "\u212a", "😀", "-", ".", "/", "\\", ";", "&&", "|", "'", "\"", "=", ":",
  "sudo ", "echo ", "#", "\r"];

/** Deterministic pseudo-random numbers (mulberry32), so a failure can be reproduced. */
function aleatorio(semente: number): () => number {
  let a = semente >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Edits cut by code point, never inside a surrogate pair: a lone surrogate is not text a terminal can send. */
function mutar(s: string, r: () => number): string {
  const um = <T>(xs: readonly T[]) => xs[Math.floor(r() * xs.length)]!;
  let out = [...s];
  for (let n = 1 + Math.floor(r() * 3); n > 0; n--) {
    const k = Math.floor(r() * 8), i = Math.floor(r() * (out.length + 1));
    if (k === 0) out = out.map((c) => (r() < 0.5 ? c.toUpperCase() : c.toLowerCase()));
    else if (k === 1) out = out.map((c) => (c === "s" && r() < 0.3 ? "ſ" : /k/i.test(c) && r() < 0.3 ? KELVIN : c));
    else if (k === 2) out = out.map((c) => (c === " " ? um(ESPACOS) : c));
    else if (k === 3) out = [...out.slice(0, i), ...um(LIXO), ...out.slice(i)];
    else if (k === 4) out = [...out.slice(0, i), ...out.slice(i + 1 + Math.floor(r() * 3))];
    else if (k === 5) out = [...um(LIXO), ...out];
    else if (k === 6) out = [...out, ...um(LIXO)];
    else out = out.slice(0, i);
  }
  return out.join("");
}

const lista = () => LISTAS.flatMap((f) => readFileSync(f, "utf8").split(/\r?\n/).filter((l) => l.trim() !== "")
  .map((l) => l.split("\t")[1]!));

describe.skipIf(!existsSync(REGEX))("the C matcher", () => {
  test("refuses everything outside the subset the list uses, and accepts every line of the list", () => {
    const fora = ["a{2}", "a{1,3}", "a*?", "a+?", "a??", "(?=a)", "(?!a)", "(?<=a)b", "(?<!a)b", "(?<n>a)", "(a)\\1",
      "\\p{L}", "\\P{L}", "\\e", "\\-", "\\:", "é", "a{", "}", "]", "\\b*", "^+", "[\\w-z]", "[a-\\d]", "[z-a]", "(a",
      "a)", "\\cA", "\\x41", "\\u0041", "\\k<n>",
      /* JavaScriptCore does not fold long s and the Kelvin sign into a class's letters: negated, that would miss. */
      "[^a-z]", "[^Rr]", "[^A-Z0-9]"];
    const respostas = emC(fora.map((p) => [p, "a"] as const));
    fora.forEach((p, i) => expect([p, respostas[i]!.startsWith("E ")]).toEqual([p, true]));
    /* A negated class without letters is in the subset, and decides as JavaScript does. */
    const semLetras = emC([["[^\\n]", "x"], ["[^\\n]", "\n"], ["[^0-9]", "ſ"], ["[^ ]", "K"]]);
    expect(semLetras).toEqual(["1", "0", "1", "1"]);
    const linhas = lista();
    expect(linhas.length).toBeGreaterThan(40);
    const aceitas = emC(linhas.map((p) => [p, "rm -rf /"] as const));
    linhas.forEach((p, i) => expect([p, aceitas[i]]).not.toEqual([p, expect.stringMatching(/^E /)]));
  });

  test("decides every line of the list as JavaScript's RegExp with \"iu\" does, on thousands of inputs", () => {
    const r = aleatorio(20261110);
    const entradas = new Set<string>(SEMENTES);
    for (const s of SEMENTES) for (let i = 0; i < 80; i++) entradas.add(mutar(s, r));
    for (let i = 0; i < 400; i++) {
      const pedacos = Array.from({ length: 1 + Math.floor(r() * 6) }, () =>
        r() < 0.5 ? SEMENTES[Math.floor(r() * SEMENTES.length)]!.split(" ")[Math.floor(r() * 3)] ?? "" : LIXO[Math.floor(r() * LIXO.length)]!);
      entradas.add(pedacos.join(ESPACOS[Math.floor(r() * ESPACOS.length)]));
    }
    const linhas = lista(), todas = [...entradas];
    const pares = linhas.flatMap((p) => todas.map((s) => [p, s] as const));
    const c = emC(pares);
    let positivos = 0;
    const jsSimCNao: string[] = [], cSimJsNao: string[] = [];
    pares.forEach(([p, s], i) => {
      const js = new RegExp(p, "iu").test(s);
      if (js) ++positivos;
      if (js && c[i] !== "1") jsSimCNao.push(`${p} :: ${JSON.stringify(s)} -> C ${c[i]}`);
      if (!js && c[i] === "1") cSimJsNao.push(`${p} :: ${JSON.stringify(s)}`);
    });
    expect(jsSimCNao).toEqual([]);
    /* The other way round is allowed only with long s or the Kelvin sign, and listed. */
    expect(cSimJsNao.filter((x) => !/ſ|\\u212a|\u212a/.test(x))).toEqual([]);
    expect(pares.length).toBeGreaterThan(100_000);
    expect(positivos).toBeGreaterThan(2_000);
    for (const p of linhas) expect([p, pares.some(([q, s], i) => q === p && c[i] === "1" && s.length > 0)]).toEqual([p, true]);
  }, 120_000);

  test("matching is linear: long and hostile inputs take milliseconds", () => {
    const longo = "a".repeat(8000), xs = "x".repeat(5000);
    const t0 = performance.now();
    const c = emC([["(a|aa)*b", longo], ["(x+x+)+y", xs], ["\\b(psql|mysql)\\b[^\\n]*\\b(drop|alter)\\b", `psql ${longo}`],
      ["(a*)*$", longo]]);
    expect(c).toEqual(["0", "0", "0", "1"]);
    expect(performance.now() - t0).toBeLessThan(5_000);
  });
});
