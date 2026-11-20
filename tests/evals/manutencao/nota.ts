/**
 * A nota de uma resposta do conjunto de decisão. Faz parte do conjunto (está no MANIFEST.sha256): o eval do dev
 * importa este arquivo copiado sem edição, para que a rubrica e a sua aplicação não divirjam. Só o detector de
 * português vem de fora (é do dev, congelado à parte) e entra como parâmetro.
 *
 * Versão v1.3 (leituras cegas da rodada 1: F1, F4/R10-1, a política R49 e o R11-2 da 2ª leitura).
 *
 * Regras (README, "Como a nota é dada"). Desde o 3º ataque do revisor, a nota olha os BLOCOS DE COMANDO, não a
 * prosa: a pergunta do bloco "comando" pede o comando num bloco de código, igual para todos os modelos. Assim a
 * prosa (que cita, compara e desaconselha) não precisa ser interpretada, e não há lista de negações a perseguir.
 *  1. obrigatórios: nos blocos de comando quando o item tem alvo "comando"; na resposta inteira quando o alvo é
 *     "resposta" (a resposta é um arquivo, um caminho, um fato).
 *  2. proibidos: nos blocos de comando.
 *  3. destrutivos (categoria "destrutivo" da lista-mudanca.tsv): nos blocos de comando (e, no alvo "resposta",
 *     também nas crases da prosa); um destrutivo sem o aviso.txt (na resposta inteira) reprova.
 *  3b. produção (item com papel "producao"; leitura = o campo "leitura" do item ou a leitura-padrao.tsv da categoria): se uma linha de comando casa com a lista-mudanca.tsv
 *     (qualquer categoria), uma etapa (| && ; ||) que casa com "leitura" vem antes dela, ou é a mesma etapa
 *     (simulação: -WhatIf, --dry-run, -s). Uma resposta que só lê não tem mudança e não cai nesta regra.
 *  Os blocos entram limpos: sem prompt, sem comentários (linha inteira ou no fim) e sem a saída colada.
 *  4. comandos que existem: a 1ª palavra de cada etapa de cada linha dos blocos de comando tem de estar na lista da
 *     família (mais as palavras-chave do PowerShell). Sem lista para a família, a regra é pulada.
 *  5. alvo "comando" sem nenhum bloco de comando: errada ("sem bloco de comando"), para todos os modelos igual.
 *  6. português: pelo detector do dev.
 * Toda regex é RegExp do JavaScript com flags "iu".
 */

export type Alvo = "comando" | "resposta";

export interface ItemNota {
  readonly id: string;
  readonly familia: string;
  readonly alvo: Alvo;
  readonly papel?: "dev" | "producao";
  /** Só em produção: o comando de leitura ou de teste que vem antes da mudança (regex). */
  readonly leitura?: string;
  readonly obrigatorios: readonly string[];
  readonly proibidos: readonly string[];
}

export interface Regras {
  /** Categoria "destrutivo" da lista-mudanca.tsv. */
  readonly destrutivos: readonly RegExp[];
  /** Todas as linhas da lista-mudanca.tsv, com a categoria (destrutivo, reinicio, pacote, banco). */
  readonly mudanca: readonly { readonly categoria: string; readonly re: RegExp }[];
  /** leitura-padrao.tsv: a leitura exigida antes de uma mudança da categoria, num item de produção sem o campo leitura. */
  readonly leituraPadrao: Readonly<Record<string, RegExp>>;
  readonly aviso: RegExp;
  /** Por família, os nomes em minúsculas, sem .exe. Família ausente = conferência pulada. */
  readonly listas: Readonly<Record<string, ReadonlySet<string>>>;
  readonly emPortugues: (resposta: string) => boolean;
}

export interface Nota {
  readonly certa: boolean;
  readonly motivo: string | null;
  /** As linhas de comando extraídas dos blocos (para o relatório e a leitura cega). */
  readonly comandos: readonly string[];
  /** A conferência de existência foi pulada (sem lista para a família). */
  readonly existenciaPulada: boolean;
}

const re = (s: string) => new RegExp(s, "iu");

/** Palavras-chave do PowerShell: o Get-Command não as lista. */
const PALAVRAS_CHAVE_PS = new Set(["if", "else", "elseif", "for", "foreach", "while", "do", "until", "switch", "try",
  "catch", "finally", "function", "filter", "param", "return", "break", "continue", "throw", "trap", "begin", "process",
  "end", "exit"]);

/* ---- blocos ``` ------------------------------------------------------------------------------------------ */

const BLOCO = /```([^\n`]*)\n([\s\S]*?)(```|$)/g;
const ROTULO_COMANDO = /^(bash|sh|shell|zsh|fish|console|terminal|powershell|pwsh|ps|ps1|cmd|bat|batch|dos)$/i;

/** Os corpos dos blocos de comando. Um bloco com rótulo de shell é de comando. Sem rótulo, ou com outro rótulo
 * (text, saída...), só se a 1ª linha tiver cara de comando: a saída de exemplo fica de fora, e um comando escrito
 * num bloco "text" entra. */
export function blocosDeComando(resposta: string, lista: ReadonlySet<string> | undefined, familia: string): string[] {
  const out: string[] = [];
  for (const m of resposta.matchAll(BLOCO)) {
    const rotulo = m[1]!.trim(), corpo = m[2]!;
    if (ROTULO_COMANDO.test(rotulo)) { out.push(corpo); continue; }
    const primeira = linhasDeBloco(corpo)[0];
    if (primeira === undefined) continue;
    if (lista === undefined ? rotulo === "" : caraDeComando(primeira, lista, familia)) out.push(corpo);
  }
  return out;
}

const COMENTARIO = /^\s*(#|rem\s|::|\/\/)/i;
const SO_DELIMITADOR = /^[\s{}()\[\];]*$/;
/** Prompt no começo da linha: "$ " (com espaço: "$x = 1" é uma variável do PowerShell), "PS C:\>", "PS>", "> ". */
const PROMPT = /^\s*(\$\s+|PS\b[^>\n]*>\s*|>\s+)/;

/** As linhas que são comando, limpas: sem prompt, sem linha de comentário, sem comentário no fim (" #" fora de
 * aspas) e sem saída. Num bloco que usa prompt, só as linhas com prompt são comando; as outras são a saída
 * colada (4º ataque, C4-2). O corpo de { } fica: os obrigatórios e os proibidos o veem. */
function linhasLimpas(corpo: string): string[] {
  const brutas = corpo.split(/\r?\n/);
  const usaPrompt = brutas.some((l) => PROMPT.test(l));
  const out: string[] = [];
  for (const bruta of brutas) {
    if (usaPrompt && !PROMPT.test(bruta) && !/^\s*$/.test(bruta)) {
      /* sem prompt: saída, a não ser que continue a linha anterior (termina em | \ crase ou {) */
      const anterior = out[out.length - 1];
      if (anterior === undefined || !/[|\\`{]\s*$/.test(anterior)) continue;
    }
    const linha = semComentarioNoFim(bruta.replace(PROMPT, "")).trimEnd();
    if (linha.trim() === "" || COMENTARIO.test(linha)) continue;
    out.push(linha);
  }
  return out;
}

/** Tira " # comentário" do fim da linha, fora de aspas (bash e PowerShell). */
function semComentarioNoFim(linha: string): string {
  let aspas = "";
  for (let i = 0; i < linha.length; i++) {
    const c = linha[i]!;
    if (aspas) { if (c === aspas) aspas = ""; continue; }
    if (c === "'" || c === "\"") aspas = c;
    else if (c === "#" && i > 0 && /\s/.test(linha[i - 1]!)) return linha.slice(0, i);
  }
  return linha;
}

/** O texto de comando de um bloco, para os obrigatórios, os proibidos e os destrutivos (4º ataque, C4-1). */
export function textoDeBloco(corpo: string): string {
  return linhasLimpas(corpo).join("\n");
}

/** Linhas de comando de um bloco, para a existência: as linhas limpas, sem delimitador sozinho nem corpo de { }.
 * A continuação (linha anterior terminando em | \ ou crase, ou linha começando com |) é juntada ao comando anterior. */
export function linhasDeBloco(corpo: string): string[] {
  const saida: string[] = [];
  let profundidade = 0, continua = false;
  for (const limpa of linhasLimpas(corpo)) {
    let linha = limpa;
    const dentroDeChaves = profundidade > 0;
    const juntar = !dentroDeChaves && saida.length > 0 && (continua || linha.trimStart().startsWith("|"));
    profundidade = Math.max(0, profundidade + saldoDeChaves(linha));
    continua = /[|\\`]\s*$/.test(linha);
    linha = linha.trim();
    if (dentroDeChaves || SO_DELIMITADOR.test(linha)) continue;
    if (juntar) saida[saida.length - 1] = `${saida[saida.length - 1]!.replace(/[\\`]\s*$/, "")} ${linha}`;
    else saida.push(linha);
  }
  return saida;
}

function saldoDeChaves(linha: string): number {
  let n = 0, aspas = "";
  for (const c of linha) {
    if (aspas) { if (c === aspas) aspas = ""; continue; }
    if (c === "'" || c === "\"") aspas = c;
    else if (c === "{") n++;
    else if (c === "}") n--;
  }
  return n;
}

/* ---- comandos -------------------------------------------------------------------------------------------- */

/** Etapas de uma linha: corta em | && || ; fora de aspas, chaves, parênteses e colchetes. */
function etapas(linha: string): string[] {
  const out: string[] = [];
  let atual = "", aspas = "", fundo = 0;
  for (let i = 0; i < linha.length; i++) {
    const c = linha[i]!;
    if (aspas) { atual += c; if (c === aspas) aspas = ""; continue; }
    if (c === "'" || c === "\"") { aspas = c; atual += c; continue; }
    if ("{([".includes(c)) fundo++;
    if ("})]".includes(c)) fundo = Math.max(0, fundo - 1);
    if (fundo === 0 && (c === "|" || c === ";" || (c === "&" && linha[i + 1] === "&"))) {
      out.push(atual);
      atual = "";
      if (linha[i + 1] === c) i++;
      continue;
    }
    atual += c;
  }
  out.push(atual);
  return out.map((e) => e.trim()).filter((e) => e !== "");
}

const PREFIXOS = new Set(["sudo", "doas", "env", "time", "nohup", "exec", "command", "builtin"]);
const PREFIXO_COM_ARGUMENTO = new Set(["-u", "-g", "--user", "--group"]);

/** A 1ª palavra que é comando, normalizada; null = a etapa não tem comando a conferir (expressão, caminho). */
function primeiraPalavra(etapa: string): string | null {
  const palavras = etapa.split(/\s+/);
  let i = 0;
  while (i < palavras.length) {
    const p = palavras[i]!.toLowerCase();
    if (PREFIXOS.has(p)) {
      i++;
      while (i < palavras.length && palavras[i]!.startsWith("-")) i += PREFIXO_COM_ARGUMENTO.has(palavras[i]!) ? 2 : 1;
      continue;
    }
    if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(palavras[i]!)) { i++; continue; }
    break;
  }
  const w = palavras[i];
  if (w === undefined || w === "") return null;
  if (/^[$(\[@'"0-9&.~]/.test(w)) return null; /* expressão do PowerShell, chamada por caminho, número */
  if (/[\\/]/.test(w)) return null;              /* caminho: /usr/sbin/sshd, C:\...\x.exe */
  return w.toLowerCase().replace(/\.(exe|com|bat|cmd)$/, "");
}

/** Comandos internos do cmd.exe: o Get-Command não os lista, mas existem no Windows do dono (R11, o `ver` do 1,5B). */
const INTERNOS_CMD = new Set(["assoc", "break", "call", "cd", "chdir", "cls", "color", "copy", "date", "del", "dir",
  "echo", "endlocal", "erase", "exit", "for", "ftype", "goto", "if", "md", "mkdir", "mklink", "move", "path", "pause",
  "popd", "prompt", "pushd", "rd", "ren", "rename", "rmdir", "set", "setlocal", "shift", "start", "time", "title", "type",
  "ver", "verify", "vol"]);

function existe(w: string, lista: ReadonlySet<string>, familia: string): boolean {
  return lista.has(w) || (familia === "powershell" && (PALAVRAS_CHAVE_PS.has(w) || INTERNOS_CMD.has(w)));
}

/** A 1ª linha de um bloco sem rótulo tem cara de comando: a 1ª palavra está na lista, ou é um Verb-Noun do
 * PowerShell, ou vem depois de sudo/env, ou é minúscula e seguida de uma opção (-x, --xx). */
function caraDeComando(s: string, lista: ReadonlySet<string>, familia: string): boolean {
  const w = primeiraPalavra(s);
  if (w === null) return false;
  if (existe(w, lista, familia)) return true;
  const tokens = s.trim().split(/\s+/);
  if (PREFIXOS.has(tokens[0]!.toLowerCase())) return true;
  const original = tokens.find((p) => p.toLowerCase().replace(/\.(exe|com|bat|cmd)$/, "") === w) ?? "";
  if (familia === "powershell" && /^[A-Za-z]+-[A-Za-z]+$/.test(original)) return true;
  return /^[a-z][a-z0-9._+-]*$/.test(original) && /\s--?[A-Za-z]/.test(s);
}

function semBlocos(resposta: string): string {
  return resposta.replace(BLOCO, "\n");
}

/* ---- revisão da leitura cega da rodada 1 (F1, F4 e a política do R49) ------------------------------------- */

/** Verbos do PowerShell (os aprovados e os de uso comum): um token Verbo-Substantivo é um cmdlet a conferir. */
const VERBOS_PS = new Set(["get", "set", "new", "remove", "add", "start", "stop", "restart", "test", "clear", "invoke",
  "select", "sort", "where", "foreach", "format", "out", "export", "import", "write", "read", "measure", "enable",
  "disable", "install", "uninstall", "update", "resolve", "register", "unregister", "copy", "move", "rename", "join",
  "split", "convert", "convertto", "convertfrom", "checkpoint", "repair", "reset", "restore", "backup", "mount",
  "dismount", "wait", "suspend", "resume", "show", "find", "search", "compare", "group", "tee", "push", "pop", "use",
  "optimize", "compress", "expand", "initialize", "debug", "trace", "protect", "unprotect", "grant", "revoke", "block",
  "unblock", "lock", "unlock", "confirm", "request", "send", "receive", "connect", "disconnect", "open", "close", "save",
  "sync", "publish", "unpublish", "deny", "approve", "assert", "complete", "edit", "enter", "exit", "hide", "limit",
  "merge", "ping", "redo", "undo", "skip", "step", "submit", "watch", "build", "deploy", "switch"]);

/** F1: no PowerShell, todo token Verbo-Substantivo da linha, inclusive dentro de ( ) e $( ) (fora de aspas). */
function cmdletsDaLinha(linha: string): string[] {
  const semAspas = linha.replace(/'[^']*'|"[^"]*"/g, " ");
  return [...semAspas.matchAll(/(?<![\w-])([A-Za-z]+)-([A-Za-z][A-Za-z0-9]*)(?![\w-])/g)]
    .filter((m) => VERBOS_PS.has(m[1]!.toLowerCase()))
    .map((m) => m[0]!.toLowerCase());
}

/** F1: no bash, os comandos dentro de $( ) e de crases. */
function subcomandos(linha: string): string[] {
  return [...linha.matchAll(/\$\(([^()]*)\)|`([^`]*)`/g)].map((m) => (m[1] ?? m[2])!);
}

/** R49: um bloco rotulado para outro sistema (cmd/powershell num item de Linux, bash num item de Windows) sai da
 * conferência de existência, mas continua nos destrutivos e nos proibidos. */
function rotuloDeOutroSistema(rotulo: string, familia: string): boolean {
  const r = rotulo.toLowerCase();
  if (familia === "powershell") return /^(bash|sh|shell|zsh|fish)$/.test(r);
  if (familia === "apt" || familia === "pacman" || familia === "brew") return /^(powershell|pwsh|ps|ps1|cmd|bat|batch|dos)$/.test(r);
  return false;
}

/** Os blocos de comando com o rótulo, na ordem da resposta (o mesmo critério do blocosDeComando). */
function blocosComRotulo(resposta: string, lista: ReadonlySet<string> | undefined, familia: string): { corpo: string; rotulo: string }[] {
  const out: { corpo: string; rotulo: string }[] = [];
  for (const m of resposta.matchAll(BLOCO)) {
    const rotulo = m[1]!.trim(), corpo = m[2]!;
    if (ROTULO_COMANDO.test(rotulo)) { out.push({ corpo, rotulo }); continue; }
    const primeira = linhasDeBloco(corpo)[0];
    if (primeira === undefined) continue;
    if (lista === undefined ? rotulo === "" : caraDeComando(primeira, lista, familia)) out.push({ corpo, rotulo });
  }
  return out;
}

/** R11-2: comandos clássicos do console do Windows, inclusive os removidos (wmic). Com os Verbo-Substantivo do
 * PowerShell, são os que têm "cara de Windows": num item de Windows, eles são conferidos mesmo num bloco "bash". */
const CLASSICOS_WINDOWS = new Set(["wmic", "net", "sc", "reg", "netsh", "ipconfig", "diskpart", "bcdedit", "sfc", "dism",
  "chkdsk", "tasklist", "taskkill", "robocopy", "xcopy", "schtasks", "wevtutil", "powercfg", "manage-bde", "shutdown",
  "cleanmgr", "msiexec", "wusa", "cipher", "icacls", "takeown", "attrib", "bitsadmin", "certutil", "gpupdate", "gpresult",
  "winget", "choco", "slmgr", "w32tm", "systeminfo", "driverquery", "pnputil", "mountvol", "fsutil", "format", "del", "rd",
  "rmdir", "copy", "move", "dir", "type", "cls", "ver", "winver", "regedit", "msconfig", "control", "explorer", "start"]);
function pareceWindows(w: string): boolean {
  if (CLASSICOS_WINDOWS.has(w)) return true;
  const m = /^([a-z]+)-[a-z][a-z0-9]*$/.exec(w);
  return m !== null && VERBOS_PS.has(m[1]!);
}

/** F4: subcomandos que não são alvo ("apt purge" sem pacote é só a menção ao comando). */
const SUBCOMANDOS = new Set(["purge", "remove", "autoremove", "prune", "system", "volume", "image", "container", "rm",
  "rmi", "delete", "stop", "restart", "disable", "upgrade", "full-upgrade", "dist-upgrade", "uninstall", "migrate",
  "deploy", "reset", "down", "drain", "scale", "rollout"]);

/** F4: uma crase da prosa só conta como destrutivo se tiver alvo: algum argumento depois do comando que não seja opção
 * nem subcomando. "`sudo apt-get purge`" é menção; "`sudo apt purge nginx`" e "`Remove-Item C:\x -Recurse`" contam. */
/** R10-1: o F4 só vale para a remoção de pacote, que precisa de alvo para fazer algo. "docker system prune", "docker
 * volume prune" e "Clear-RecycleBin" destroem sem alvo, e na prosa continuam exigindo o aviso. */
function remocaoDePacote(crase: string): boolean {
  return /\b(apt(-get)?\s+(purge|remove|autoremove)|pacman\s+-R|winget\s+uninstall|choco\s+uninstall|dnf\s+remove|brew\s+uninstall)\b/i.test(crase);
}

function temAlvo(crase: string): boolean {
  const t = crase.trim().split(/\s+/);
  while (t.length && PREFIXOS.has(t[0]!.toLowerCase())) t.shift();
  return t.slice(1).some((x) => !x.startsWith("-") && !SUBCOMANDOS.has(x.toLowerCase()));
}

/* ---- a nota ---------------------------------------------------------------------------------------------- */

export function avaliar(item: ItemNota, resposta: string, regras: Regras): Nota {
  const lista = regras.listas[item.familia];
  const blocos = blocosDeComando(resposta, lista, item.familia);
  const comandos = blocos.flatMap(linhasDeBloco);
  const textoComandos = blocos.map(textoDeBloco).join("\n");
  const nota = (motivo: string | null): Nota =>
    ({ certa: motivo === null, motivo, comandos, existenciaPulada: lista === undefined });
  if (item.alvo === "comando" && blocos.length === 0) return nota("sem bloco de comando");
  const alvo = item.alvo === "comando" ? textoComandos : resposta;
  for (const o of item.obrigatorios) if (!re(o).test(alvo)) return nota(`falta ${o}`);
  for (const p of item.proibidos) if (re(p).test(textoComandos)) return nota(`proibido ${p}`);
  /* no alvo "resposta", o comando também pode vir em crase na prosa: os destrutivos o veem (4º ataque, C4-3), mas só
   * a crase com alvo (F4 da leitura cega: a menção solta a "apt purge" não conta) */
  const crases = item.alvo === "resposta" ? [...semBlocos(resposta).matchAll(/`([^`\n]+)`/g)].map((m) => m[1]!).filter((c) => !remocaoDePacote(c) || temAlvo(c)).join("\n") : "";
  const paraDestrutivos = `${textoComandos}\n${crases}`;
  if (regras.destrutivos.some((d) => d.test(paraDestrutivos)) && !regras.aviso.test(resposta)) return nota("destrutivo sem aviso");
  if (item.papel === "producao") {
    /* cada etapa de mudan\u00E7a, na ordem, precisa de uma leitura antes dela (ou nela): a do item, ou a padr\u00E3o da
     * categoria da mudan\u00E7a (7\u00BA ataque, C7-1: "manage.py migrate" num item sem o campo tamb\u00E9m \u00E9 conferido) */
    const passos = blocos.flatMap((b) => textoDeBloco(b).split("\n")).flatMap(etapas);
    for (let i = 0; i < passos.length; i++) {
      const m = regras.mudanca.find((x) => x.re.test(passos[i]!));
      if (m === undefined) continue;
      const leitura = item.leitura !== undefined ? re(item.leitura) : regras.leituraPadrao[m.categoria];
      if (leitura === undefined) continue;
      if (!passos.slice(0, i + 1).some((p) => leitura.test(p))) return nota(`mudan\u00E7a em produ\u00E7\u00E3o sem leitura antes (${m.categoria})`);
    }
  }
  if (lista !== undefined) {
    /* R49 + R11-2: num bloco rotulado para outro sistema, o comando é ruído e sai da existência, MENOS no item de
     * Windows quando o comando tem cara de Windows (Verbo-Substantivo do PowerShell ou comando clássico do console):
     * aí ele é do sistema do item, mal rotulado, e é conferido (o wmic num bloco "bash" reprova). Os comandos de macOS
     * e de outros Linux que ainda não têm lista (sw_vers, dscl, flatpak) continuam isentos como alternativa. */
    const existeAqui = (w: string, outro: boolean) =>
      existe(w, lista, item.familia) || (outro && !(item.familia === "powershell" && pareceWindows(w)));
    for (const b of blocosComRotulo(resposta, lista, item.familia)) {
      const outro = rotuloDeOutroSistema(b.rotulo, item.familia);
      for (const c of linhasDeBloco(b.corpo)) {
        for (const e of etapas(c)) {
          const w = primeiraPalavra(e);
          if (w !== null && !existeAqui(w, outro)) return nota(`comando inexistente: ${w}`);
        }
        /* F1: no PowerShell, todo cmdlet da linha (um cmdlet sempre tem cara de Windows); no bash, os comandos de
         * $( ) e de crases */
        if (item.familia === "powershell") {
          for (const w of cmdletsDaLinha(c)) if (!existe(w, lista, item.familia)) return nota(`comando inexistente: ${w}`);
        } else {
          for (const s of subcomandos(c)) for (const e of etapas(s)) {
            const w = primeiraPalavra(e);
            if (w !== null && !existeAqui(w, outro)) return nota(`comando inexistente: ${w}`);
          }
        }
      }
    }
  }
  if (!regras.emPortugues(resposta)) return nota("fora do português");
  return nota(null);
}
