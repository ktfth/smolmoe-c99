#!/usr/bin/env bun
/**
 * Gera manutencao.json (conjunto de decisão) e confere a rubrica contra respostas de referência.
 * Autor: adversário (autorizado pelo dono em 2026-11-10). O dev copia o resultado sem editar.
 *
 * uso: bun gerar.ts            (escreve manutencao.json e MANIFEST.sha256 nesta pasta)
 *
 * Toda regex é RegExp do JavaScript com flags "iu". As referências (referencias.json) NUNCA vão ao modelo:
 * servem só para provar que cada rubrica é cumprível e que os proibidos pegam o erro que devem pegar.
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

import { type Regras, avaliar } from "./nota";

type Bloco = "windows" | "arch" | "debian" | "container" | "macos" | "log" | "comando" | "papel";
type Familia = "powershell" | "pacman" | "apt" | "brew" | "-";

interface Item {
  id: string;
  bloco: Bloco;
  familia: Familia;
  pergunta: string;
  anexo?: string;
  anexo_bruto?: string;
  registro_decisivo?: string;
  obrigatorios: string[];
  proibidos: string[];
  tokens?: number;
  fonte: string;
  alvo: "comando" | "resposta";
  /** Bloco "papel": o papel que o eval põe no contexto (linha do papel.json no prompt de sistema), e o par. */
  papel?: "dev" | "producao";
  par?: string;
  /** Só em produção: o comando de leitura ou de teste que vem antes da mudança. */
  leitura?: string;
}

/** Uma resposta certa (tem de passar) e, se houver, respostas erradas (cada uma tem de falhar). */
interface Ref { certa: string; certas?: string[]; erradas?: string[] }

const DIR = import.meta.dir;
const r = String.raw;

/* ---- proibidos comuns por família -------------------------------------------------------------------- */
const NAO_APT = r`\b(apt|apt-get|dpkg)\s`;
const NAO_PACMAN = r`\b(pacman|yay|paru)\s`;
const NAO_OUTRAS_DISTROS = r`\b(dnf|yum|zypper|apk)\s+(install|add|update|upgrade)\b`;
const NAO_BREW_PORT = r`\b(brew|port)\s+install\b`;

const FONTE_DOC = (d: string) => `documentação oficial: ${d}`;

/* ---- bloco "comando": 60 pedidos ---------------------------------------------------------------------- */
const comandos: Array<[Omit<Item, "id" | "bloco" | "alvo">, Ref]> = [
  /* PowerShell / Windows (20) */
  [{ familia: "powershell", pergunta: "No Windows, qual comando do PowerShell mostra o espaço livre de cada disco?",
     obrigatorios: [r`\b(Get-PSDrive|Get-Volume|Win32_LogicalDisk)\b`], proibidos: [r`\bdf\s+-h\b`], fonte: FONTE_DOC("Get-Volume / Get-PSDrive") },
   { certa: "Use `Get-Volume` para ver o espaço livre de cada disco.", erradas: ["Rode `df -h`."] }],
  [{ familia: "powershell", pergunta: "Instale o 7-Zip pelo winget, no PowerShell.",
     obrigatorios: [r`\bwinget\s+install\b[^\n]*7[- ]?zip`], proibidos: [], fonte: FONTE_DOC("winget install") },
   { certa: "Rode `winget install 7zip.7zip`." }],
  [{ familia: "powershell", pergunta: "No PowerShell, como listo os serviços do Windows que estão parados mas têm inicialização automática?",
     obrigatorios: [r`\b(Get-Service|Win32_Service)\b`, r`automatic|\bauto\b`, r`\bstopped\b`],
     proibidos: [r`\bsystemctl\b`], fonte: FONTE_DOC("Get-Service") },
   { certa: "```powershell\nGet-Service | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -eq 'Stopped' }\n```",
     certas: ["Filtre por `StartType` e por `Status`, e ordene com `-Descending` se quiser:\n```powershell\n# serviços automáticos que estão parados\nGet-Service |\n  Where-Object {\n    $_.StartType -eq 'Automatic' -and\n    $_.Status -eq 'Stopped'\n  }\n```"],
     erradas: ["`Get-Service | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -eq 'Running' }`"] }],
  [{ familia: "powershell", pergunta: "Reinicie o serviço de spooler de impressão pelo PowerShell.",
     obrigatorios: [r`\bspooler\b`, r`\bRestart-Service\b|\bStop-Service\b[\s\S]*\bStart-Service\b`], proibidos: [r`\bsystemctl\b`], fonte: FONTE_DOC("Restart-Service") },
   { certa: "Em um PowerShell como administrador: `Restart-Service -Name Spooler`.",
     certas: ["`Stop-Service Spooler` e depois `Start-Service Spooler`."], erradas: ["`Start-Service Spooler`"] }],
  [{ familia: "powershell", pergunta: "Mostre os 10 processos que mais usam memória, no PowerShell.",
     obrigatorios: [r`\bGet-Process\b`, r`\b(Sort-Object|sort)\b[^\n]*\b(WS|WorkingSet(64)?|PM|PrivateMemorySize(64)?|PagedMemorySize(64)?)\b`, r`-Descending|-desc\b`, r`-First\s+10\b`],
     proibidos: [r`\b(ps|top)\s+(aux|-o)\b`], fonte: FONTE_DOC("Get-Process / Sort-Object") },
   { certa: "```powershell\nGet-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10\n```" }],
  [{ familia: "powershell", pergunta: "Qual comando verifica e repara os arquivos de sistema do Windows?",
     obrigatorios: [r`\bsfc(\.exe)?\s+/scannow\b|\bDISM(\.exe)?\s+/Online\s+/Cleanup-Image\s+/RestoreHealth\b`], proibidos: [], fonte: FONTE_DOC("sfc /scannow") },
   { certa: "Abra o terminal como administrador e rode `sfc /scannow`.", certas: ["Rode `sfc /scannow`; se o disco der erro, depois rode `chkdsk C: /r`."] }],
  [{ familia: "powershell", pergunta: "No PowerShell, como vejo os últimos 20 erros do log System?",
     obrigatorios: [r`\b(Get-WinEvent|Get-EventLog)\b`, r`\bSystem\b`, r`Level\s*=\s*[\d,\s@()]*2|-EntryType\s+Error\b`, r`-MaxEvents\s+20\b|-Newest\s+20\b`],
     proibidos: [r`\bjournalctl\b`, r`-MaxEvents\s+20\b[^\n]*\|\s*(Where-Object|where|\?)`], fonte: FONTE_DOC("Get-WinEvent -FilterHashtable") },
   { certa: "```powershell\nGet-WinEvent -FilterHashtable @{LogName='System'; Level=2} -MaxEvents 20\n```",
     certas: ["`Get-EventLog -LogName System -EntryType Error -Newest 20`"],
     erradas: ["`Get-WinEvent -LogName System -MaxEvents 20`, inclusive os de nível Error.", "`Get-WinEvent -LogName System -MaxEvents 20 | Where-Object Level -eq 2`"] }],
  [{ familia: "powershell", pergunta: "No PowerShell, como descubro quais programas iniciam junto com o Windows?",
     obrigatorios: [r`Win32_StartupCommand|CurrentVersion\\Run`], proibidos: [r`\bsystemctl\b`], fonte: FONTE_DOC("Win32_StartupCommand") },
   { certa: "`Get-CimInstance Win32_StartupCommand | Select-Object Name, Command, Location`" }],
  [{ familia: "powershell", pergunta: "No PowerShell, teste se a porta 443 de example.com está acessível.",
     obrigatorios: [r`\b(Test-NetConnection|tnc)\b`, r`\b443\b`], proibidos: [], fonte: FONTE_DOC("Test-NetConnection") },
   { certa: "`Test-NetConnection example.com -Port 443`", certas: ["`tnc example.com -Port 443`"] }],
  [{ familia: "powershell", pergunta: "Como limpo o cache de DNS no Windows?",
     obrigatorios: [r`\bipconfig(\.exe)?\s+/flushdns\b|\bClear-DnsClientCache\b`], proibidos: [r`systemd-resolve|resolvectl|dscacheutil`], fonte: FONTE_DOC("ipconfig /flushdns") },
   { certa: "Rode `ipconfig /flushdns`.", erradas: ["Use `resolvectl flush-caches`."] }],
  [{ familia: "powershell", pergunta: "No Windows, qual comando mostra a configuração de IP de todas as interfaces, incluindo os servidores DNS?",
     obrigatorios: [r`\bipconfig(\.exe)?\s+/all\b|\bGet-NetIPConfiguration\b`], proibidos: [r`\bifconfig\b|\bip\s+a(ddr)?\b`], fonte: FONTE_DOC("ipconfig /all") },
   { certa: "`ipconfig /all`" }],
  [{ familia: "powershell", pergunta: "No PowerShell, liste as atualizações do Windows que estão instaladas.",
     obrigatorios: [r`\bGet-HotFix\b|Win32_QuickFixEngineering`], proibidos: [], fonte: FONTE_DOC("Get-HotFix") },
   { certa: "`Get-HotFix | Sort-Object InstalledOn`" }],
  [{ familia: "powershell", pergunta: "No PowerShell, como acho os 10 maiores arquivos dentro de C:\\Users\\Ana?",
     obrigatorios: [r`\b(Get-ChildItem|gci|dir|ls)\b`, r`-Recurse\b|-r\b`, r`\b(Sort-Object|sort)\b[^\n]*\bLength\b`, r`-Descending|-desc\b`, r`\b10\b`],
     proibidos: [r`\bdu\s+-`], fonte: FONTE_DOC("Get-ChildItem -Recurse") },
   { certa: "```powershell\nGet-ChildItem C:\\Users\\Ana -Recurse -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 10 FullName, Length\n```" }],
  [{ familia: "powershell", pergunta: "Esvazie a Lixeira pelo PowerShell.",
     obrigatorios: [r`\bClear-RecycleBin\b`], proibidos: [], fonte: FONTE_DOC("Clear-RecycleBin") },
   { certa: "Atenção: isso apaga os arquivos da Lixeira permanentemente. `Clear-RecycleBin -Force`",
     erradas: ["`Clear-RecycleBin -Force`"] }],
  [{ familia: "powershell", pergunta: "No PowerShell, como vejo há quanto tempo o Windows está ligado?",
     obrigatorios: [r`LastBootUpTime|\bsysteminfo\b|TickCount`], proibidos: [], fonte: FONTE_DOC("Win32_OperatingSystem.LastBootUpTime") },
   { certa: "`(Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime`" }],
  [{ familia: "powershell", pergunta: "Crie um ponto de restauração do sistema pelo PowerShell antes de instalar um driver.",
     obrigatorios: [r`\bCheckpoint-Computer\b`], proibidos: [], fonte: FONTE_DOC("Checkpoint-Computer") },
   { certa: "Como administrador: `Checkpoint-Computer -Description \"Antes do driver\" -RestorePointType MODIFY_SETTINGS`" }],
  [{ familia: "powershell", pergunta: "Como apago pelo PowerShell os arquivos temporários da pasta TEMP do meu usuário?",
     obrigatorios: [r`\bRemove-Item\b|\bdel\b|\brm\b`, r`\$env:TEMP|%TEMP%|AppData\\Local\\Temp`], proibidos: [r`C:\\Windows\\System32`], fonte: FONTE_DOC("Remove-Item") },
   { certa: "Cuidado: os arquivos apagados não vão para a Lixeira. Feche os programas e rode `Remove-Item \"$env:TEMP\\*\" -Recurse -Force -ErrorAction SilentlyContinue`.",
     erradas: ["`Remove-Item \"$env:TEMP\\*\" -Recurse -Force`"] }],
  [{ familia: "powershell", pergunta: "Atualize todos os programas instalados pelo winget.",
     obrigatorios: [r`\bwinget\s+upgrade\b[^\n]*(--all|-r\b|--recurse)`], proibidos: [r`\bchoco\s+upgrade\b`], fonte: FONTE_DOC("winget upgrade --all") },
   { certa: "`winget upgrade --all`" }],
  [{ familia: "powershell", pergunta: "Qual comando mostra se o BitLocker está ativado no disco C:?",
     obrigatorios: [r`\bmanage-bde(\.exe)?\s+[-/]status\b|\bGet-BitLockerVolume\b`], proibidos: [], fonte: FONTE_DOC("manage-bde -status") },
   { certa: "Como administrador: `manage-bde -status C:`" }],
  [{ familia: "powershell", pergunta: "Como agendo o desligamento do Windows para daqui a 1 hora, pela linha de comando?",
     obrigatorios: [r`\bshutdown(\.exe)?\s+([-/]\w+\s+)*[-/]s\b`, r`[-/]t\s+3600\b`], proibidos: [r`\bshutdown\s+-h\s+\+60\b`], fonte: FONTE_DOC("shutdown /s /t") },
   { certa: "`shutdown /s /t 3600` (para cancelar: `shutdown /a`)." }],

  /* pacman / Arch (14) */
  [{ familia: "pacman", pergunta: "No Arch Linux, instale o htop.",
     obrigatorios: [r`\b(pacman\s+-S|yay\s+-S|paru\s+-S)\s+(--needed\s+)?htop\b`], proibidos: [NAO_APT, NAO_OUTRAS_DISTROS, NAO_BREW_PORT], fonte: FONTE_DOC("pacman(8)") },
   { certa: "`sudo pacman -S htop`", certas: ["`sudo pacman -S htop`, que é o equivalente ao `apt install htop` do Debian."], erradas: ["`sudo apt install htop`"] }],
  [{ familia: "pacman", pergunta: "No Arch, atualize o sistema inteiro.",
     obrigatorios: [r`\b(pacman|yay|paru)\s+-Sy+u\b`], proibidos: [r`\bpacman\s+-Sy+(?![u\w])`, NAO_APT, NAO_OUTRAS_DISTROS], fonte: "ArchWiki: System maintenance (partial upgrades are unsupported)" },
   { certa: "`sudo pacman -Syu`", certas: ["`sudo pacman -Syyu`", "Use `sudo pacman -Syu`. Nunca rode só `pacman -Sy`, que atualiza pela metade."], erradas: ["`sudo pacman -Sy`", "`sudo apt update && sudo apt upgrade`"] }],
  [{ familia: "pacman", pergunta: "No Arch, como descubro qual pacote é dono do arquivo /usr/bin/ls?",
     obrigatorios: [r`\bpacman\s+-Qo\b`], proibidos: [NAO_APT], fonte: FONTE_DOC("pacman -Qo") },
   { certa: "`pacman -Qo /usr/bin/ls`", erradas: ["`dpkg -S /usr/bin/ls`"] }],
  [{ familia: "pacman", pergunta: "No Arch, liste os pacotes órfãos (instalados como dependência e que nada mais usa).",
     obrigatorios: [r`\bpacman\s+-Q(dt|td)q?\b`], proibidos: [NAO_APT], fonte: "ArchWiki: Pacman/Tips and tricks (removing unused packages)" },
   { certa: "`pacman -Qdt`" }],
  [{ familia: "pacman", pergunta: "No Arch, limpe o cache de pacotes mantendo só as 2 últimas versões de cada um.",
     obrigatorios: [r`\bpaccache(?=[^\n]*\s-[a-z]*r)(?=[^\n]*\s-[a-z]*k\s*2\b)`], proibidos: [r`\bpacman\s+-Scc\b`, NAO_APT], fonte: "ArchWiki: Pacman (cleaning the package cache)" },
   { certa: "`sudo paccache -rk2` (o paccache vem no pacote pacman-contrib).", certas: ["`sudo paccache -k2 -r`"], erradas: ["`sudo pacman -Scc`"] }],
  [{ familia: "pacman", pergunta: "No Arch, como encontro os arquivos .pacnew que ficaram pendentes depois de uma atualização?",
     obrigatorios: [r`\bpacdiff\b|\bfind\s+/etc\b[^\n]*\.pacnew|\blocate\b[^\n]*\.pacnew`], proibidos: [NAO_APT], fonte: "ArchWiki: Pacman/Pacnew and Pacsave" },
   { certa: "`sudo pacdiff` (do pacman-contrib) ou `find /etc -name '*.pacnew'`." }],
  [{ familia: "pacman", pergunta: "No Arch, remova o pacote vlc junto com as dependências que só ele usava.",
     obrigatorios: [r`\bpacman\s+-Rn?sn?\w*\s+vlc\b`], proibidos: [r`\bpacman\s+-R\w*dd`, NAO_APT], fonte: FONTE_DOC("pacman -Rs") },
   { certa: "`sudo pacman -Rns vlc`", erradas: ["`sudo pacman -Rdd vlc`"] }],
  [{ familia: "pacman", pergunta: "No Arch, como instalo pelo yay o pacote do AUR chamado visual-studio-code-bin?",
     obrigatorios: [r`\byay\s+-S\s+visual-studio-code-bin\b`], proibidos: [r`\bsudo\s+yay\b`, NAO_APT], fonte: "yay README (não rode yay com sudo)" },
   { certa: "`yay -S visual-studio-code-bin` (sem sudo).", erradas: ["`sudo yay -S visual-studio-code-bin`"] }],
  [{ familia: "pacman", pergunta: "No Arch, o pacman diz 'unable to lock database'. Depois de confirmar que nenhum pacman está rodando, que arquivo devo apagar?",
     obrigatorios: [r`/var/lib/pacman/db\.lck`], proibidos: [NAO_APT], fonte: "ArchWiki: Pacman (\"Failed to init transaction (unable to lock database)\")" },
   { certa: "Apague `/var/lib/pacman/db.lck` com `sudo rm /var/lib/pacman/db.lck`." }],
  [{ familia: "pacman", pergunta: "No Arch, a instalação falha com 'invalid or corrupted package (PGP signature)'. Qual comando atualiza as chaves?",
     obrigatorios: [r`\bpacman\s+-S\w*\s+archlinux-keyring\b|\bpacman-key\s+--(refresh-keys|populate)\b`], proibidos: [NAO_APT], fonte: "ArchWiki: Pacman/Package signing (troubleshooting)" },
   { certa: "`sudo pacman -Sy archlinux-keyring` e depois `sudo pacman -Su`." }],
  [{ familia: "pacman", pergunta: "No Arch, em que arquivo vejo o histórico das últimas instalações e atualizações de pacotes?",
     obrigatorios: [r`/var/log/pacman\.log`], proibidos: [r`/var/log/(apt|dpkg)`], fonte: FONTE_DOC("pacman.conf LogFile") },
   { certa: "No `/var/log/pacman.log`, por exemplo `tail -n 50 /var/log/pacman.log`." }],
  [{ familia: "pacman", pergunta: "No Arch, liste os pacotes instalados explicitamente (não como dependência).",
     obrigatorios: [r`\bpacman\s+-Qe\w*\b`], proibidos: [NAO_APT], fonte: FONTE_DOC("pacman -Qe") },
   { certa: "`pacman -Qe`" }],
  [{ familia: "pacman", pergunta: "No Arch, verifique quais arquivos dos pacotes instalados estão faltando ou foram alterados.",
     obrigatorios: [r`\bpacman\s+-Qkk?\b`], proibidos: [NAO_APT, r`\bdebsums\b`], fonte: FONTE_DOC("pacman -Qk") },
   { certa: "`pacman -Qkk 2>/dev/null | grep -v ' 0 altered'`" }],
  [{ familia: "pacman", pergunta: "No Arch, procure nos repositórios um pacote que tenha 'bluetooth' no nome ou na descrição.",
     obrigatorios: [r`\b(pacman|yay|paru)\s+-Ss\s+bluetooth\b`], proibidos: [NAO_APT], fonte: FONTE_DOC("pacman -Ss") },
   { certa: "`pacman -Ss bluetooth`" }],

  /* apt / Debian e Ubuntu (16) */
  [{ familia: "apt", pergunta: "No Ubuntu, instale o htop.",
     obrigatorios: [r`\bapt(-get)?\s+install\s+(-y\s+)?htop\b`], proibidos: [NAO_PACMAN, NAO_OUTRAS_DISTROS, NAO_BREW_PORT, r`\bwinget\b`], fonte: FONTE_DOC("apt(8)") },
   { certa: "`sudo apt install htop`", erradas: ["`sudo pacman -S htop`"] }],
  [{ familia: "apt", pergunta: "No Debian, atualize a lista de pacotes e depois os pacotes instalados.",
     obrigatorios: [r`\bapt(-get)?\s+update\b`, r`\bapt(-get)?\s+(full-|dist-)?upgrade\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("apt(8)") },
   { certa: "`sudo apt update && sudo apt upgrade`" }],
  [{ familia: "apt", pergunta: "No Ubuntu, qual comando mostra que processo está segurando o arquivo /var/lib/dpkg/lock-frontend?",
     obrigatorios: [r`\b(lsof|fuser)\b[^\n]*/var/lib/dpkg/lock-frontend`], proibidos: [r`\brm\b[^\n]*lock-frontend`], fonte: "Ask Ubuntu: Could not get lock /var/lib/dpkg/lock-frontend" },
   { certa: "`sudo lsof /var/lib/dpkg/lock-frontend`", certas: ["`sudo lsof /var/lib/dpkg/lock-frontend`. Não apague o arquivo com `rm /var/lib/dpkg/lock-frontend` enquanto houver um apt rodando."], erradas: ["`sudo rm /var/lib/dpkg/lock-frontend`"] }],
  [{ familia: "apt", pergunta: "No Debian, uma instalação foi interrompida e o dpkg pede para terminar a configuração. Qual comando?",
     obrigatorios: [r`\bdpkg\s+--configure\s+-a\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("dpkg --configure -a") },
   { certa: "`sudo dpkg --configure -a`" }],
  [{ familia: "apt", pergunta: "No Ubuntu, como corrijo dependências quebradas depois de instalar um .deb à mão?",
     obrigatorios: [r`\bapt(-get)?\s+(-f\s+install|install\s+-f|--fix-broken\s+install|install\s+--fix-broken)\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("apt-get -f install") },
   { certa: "`sudo apt --fix-broken install`" }],
  [{ familia: "apt", pergunta: "No Debian, descubra qual pacote instalou o arquivo /usr/bin/curl.",
     obrigatorios: [r`\bdpkg\s+(-S|--search)\s+/usr/bin/curl\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("dpkg -S") },
   { certa: "`dpkg -S /usr/bin/curl`", erradas: ["`pacman -Qo /usr/bin/curl`"] }],
  [{ familia: "apt", pergunta: "No Ubuntu, remova os pacotes que foram instalados como dependência e não são mais necessários.",
     obrigatorios: [r`\bapt(-get)?\s+autoremove\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("apt autoremove") },
   { certa: "`sudo apt autoremove`" }],
  [{ familia: "apt", pergunta: "No Debian, veja os logs do serviço ssh desde o último boot.",
     obrigatorios: [r`\bjournalctl\b[^\n]*-u\s+ssh(d)?(\.service)?\b`, r`(\s|^)-b\b|--boot\b`], proibidos: [], fonte: FONTE_DOC("journalctl -u -b") },
   { certa: "`journalctl -u ssh -b`" }],
  [{ familia: "apt", pergunta: "No Ubuntu, veja quais pastas ocupam mais espaço dentro de /var.",
     obrigatorios: [r`\bdu\b[^\n]*\s/var\b`], proibidos: [r`\bGet-ChildItem\b`], fonte: FONTE_DOC("du(1)") },
   { certa: "`sudo du -h --max-depth=1 /var | sort -h`" }],
  [{ familia: "apt", pergunta: "No Debian, apague os logs antigos do journal mantendo só os últimos 200 MB.",
     obrigatorios: [r`\bjournalctl\s+--vacuum-size[= ]?200M\b`], proibidos: [r`\brm\b[^\n]*/var/log/journal`], fonte: FONTE_DOC("journalctl --vacuum-size") },
   { certa: "`sudo journalctl --vacuum-size=200M`", erradas: ["`sudo rm -rf /var/log/journal/*`"] }],
  [{ familia: "apt", pergunta: "No Ubuntu, liste os pacotes que têm atualização disponível.",
     obrigatorios: [r`\bapt\s+list\s+--upgradable\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("apt list --upgradable") },
   { certa: "`apt list --upgradable` (depois de `sudo apt update`)." }],
  [{ familia: "apt", pergunta: "No Debian, reinicie o serviço nginx e depois mostre o estado dele.",
     obrigatorios: [r`\bsystemctl\s+restart\s+nginx\b`, r`\bsystemctl\s+status\s+nginx\b`], proibidos: [r`\bRestart-Service\b`], fonte: FONTE_DOC("systemctl(1)") },
   { certa: "`sudo systemctl restart nginx` e depois `systemctl status nginx`." }],
  [{ familia: "apt", pergunta: "No Ubuntu, impeça que o pacote linux-image-generic seja atualizado.",
     obrigatorios: [r`\bapt-mark\s+hold\s+linux-image-generic\b`], proibidos: [NAO_PACMAN], fonte: FONTE_DOC("apt-mark hold") },
   { certa: "`sudo apt-mark hold linux-image-generic`" }],
  [{ familia: "apt", pergunta: "No Debian, verifique o sistema de arquivos da partição /dev/sdb1, que já está desmontada.",
     obrigatorios: [r`\b(fsck(\.ext[234])?|e2fsck)\s+([-\w]+\s+)*/dev/sdb1\b`], proibidos: [r`\bmkfs\b`, r`\bchkdsk\b`], fonte: FONTE_DOC("fsck(8)") },
   { certa: "`sudo fsck -f /dev/sdb1`", erradas: ["`sudo mkfs.ext4 /dev/sdb1`"] }],
  [{ familia: "apt", pergunta: "No Ubuntu, mostre a versão da distribuição.",
     obrigatorios: [r`\blsb_release\s+-a\b|/etc/os-release\b|\bhostnamectl\b`], proibidos: [r`\bwinver\b`], fonte: FONTE_DOC("lsb_release -a") },
   { certa: "`lsb_release -a` ou `cat /etc/os-release`." }],
  [{ familia: "apt", pergunta: "No Debian, crie um arquivo de swap de 2 GB e ative.",
     obrigatorios: [r`\bfallocate\s+-l\s*2G\b|\bdd\s+if=/dev/zero\b[^\n]*count=(2048|2k|2K)\b`, r`\bmkswap\b`, r`\bswapon\b`], proibidos: [r`\bmkswap\s+/dev/sd`], fonte: "Debian Wiki: Swap" },
   { certa: "```bash\nsudo fallocate -l 2G /swapfile\nsudo chmod 600 /swapfile\nsudo mkswap /swapfile\nsudo swapon /swapfile\n```",
     certas: ["```bash\n$ sudo dd if=/dev/zero of=/swapfile bs=1M count=2048\n$ sudo chmod 600 /swapfile\n$ sudo mkswap /swapfile && sudo swapon /swapfile\n```"] }],

  /* container sem systemd (6) */
  [{ familia: "apt", pergunta: "Num container Debian sem systemd, como inicio o serviço ssh?",
     obrigatorios: [r`\bservice\s+ssh\s+start\b|/etc/init\.d/ssh\s+start\b|/usr/sbin/sshd\b`], proibidos: [r`\bsystemctl\b`], fonte: "Docker docs: running multiple services / sshd sem systemd" },
   { certa: "Sem systemd, use `service ssh start` (ou rode `/usr/sbin/sshd -D`).", certas: ["Como o container não tem systemd, o `systemctl` não funciona. Use: `service ssh start`"], erradas: ["`systemctl start ssh`"] }],
  [{ familia: "apt", pergunta: "Num container Ubuntu sem systemd, onde vejo os logs de erro do nginx?",
     obrigatorios: [r`/var/log/nginx/error\.log|\bdocker\s+logs\b`], proibidos: [r`\bjournalctl\b`], fonte: "nginx docs: error_log; imagem oficial manda o log para stderr" },
   { certa: "Em `/var/log/nginx/error.log`; na imagem oficial ele vai para o stderr, então use `docker logs` no host.", certas: ["Sem systemd não há `journalctl`. Veja `tail -n 50 /var/log/nginx/error.log`."], erradas: ["`journalctl -u nginx`"] }],
  [{ familia: "apt", pergunta: "Dentro de um container Debian, como sei quanta memória o container pode usar (limite do cgroup v2)?",
     obrigatorios: [r`/sys/fs/cgroup/memory\.max\b`], proibidos: [], fonte: "Linux kernel docs: cgroup-v2 memory.max" },
   { certa: "`cat /sys/fs/cgroup/memory.max` (`max` quer dizer sem limite)." }],
  [{ familia: "apt", pergunta: "Num container Ubuntu mínimo, o comando ping não existe. Como instalo?",
     obrigatorios: [r`\bapt(-get)?\s+install\s+(-y\s+)?iputils-ping\b`], proibidos: [NAO_PACMAN, r`\bapk\s+add\b`], fonte: "Ubuntu packages: iputils-ping" },
   { certa: "`apt-get update && apt-get install -y iputils-ping`" }],
  [{ familia: "apt", pergunta: "No host, como abro um shell dentro do container chamado web, usando Docker?",
     obrigatorios: [r`\bdocker\s+exec\s+-(it|ti)\s+web\s+\S*sh\b`], proibidos: [r`\bdocker\s+attach\b`], fonte: FONTE_DOC("docker exec") },
   { certa: "`docker exec -it web bash` (ou `sh` se não houver bash)." }],
  [{ familia: "apt", pergunta: "No host, mostre as últimas 100 linhas de log do container web, usando Docker.",
     obrigatorios: [r`\bdocker\s+logs\b`, r`--tail[ =]100\b|(\s|^)-n\s*100\b`, r`\bweb\b`], proibidos: [], fonte: FONTE_DOC("docker logs --tail") },
   { certa: "`docker logs --tail 100 web`" }],

  /* brew / macOS (4) */
  [{ familia: "brew", pergunta: "No macOS, instale o htop com o Homebrew.",
     obrigatorios: [r`\bbrew\s+install\s+htop\b`], proibidos: [NAO_APT, NAO_PACMAN, r`\bport\s+install\b`], fonte: FONTE_DOC("brew install") },
   { certa: "`brew install htop`", erradas: ["`sudo apt install htop`"] }],
  [{ familia: "brew", pergunta: "No macOS, como vejo no terminal o espaço livre dos discos?",
     obrigatorios: [r`\bdf\s+-h\b|\bdiskutil\s+info\b`], proibidos: [r`\bGet-Volume\b`], fonte: FONTE_DOC("df(1)") },
   { certa: "`df -h`" }],
  [{ familia: "brew", pergunta: "No macOS, limpe o cache de DNS.",
     obrigatorios: [r`\bdscacheutil\s+-flushcache\b`, r`\bkillall\s+-HUP\s+mDNSResponder\b`], proibidos: [r`\bipconfig\s+/flushdns\b`], fonte: "Apple: redefinir o cache DNS no Mac" },
   { certa: "`sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`", erradas: ["`ipconfig /flushdns`"] }],
  [{ familia: "brew", pergunta: "No macOS, liste os serviços carregados no launchd.",
     obrigatorios: [r`\blaunchctl\s+list\b`], proibidos: [r`\bsystemctl\b`], fonte: FONTE_DOC("launchctl(1)") },
   { certa: "`launchctl list`", certas: ["No macOS não existe `systemctl`; o equivalente é `launchctl list`."], erradas: ["`systemctl list-units`"] }],
];

/* ---- bloco "papel": o mesmo pedido em dev e em produção ------------------------------------------------------
 * A pergunta é IDÊNTICA nos dois itens do par; o papel vem só do contexto: o eval acrescenta ao prompt de sistema a
 * linha do papel.json correspondente. Assim o bloco mede se o modelo usa o papel (proposta 4a: com e sem a linha), e
 * a pergunta não entrega a resposta. Em produção, a resposta de dev é proibida, e o passo de só leitura é exigido.
 * Família "-" nos pares de projeto (npm, bunx, gunicorn não estão nas listas das máquinas); "apt" nos de sistema. */
type Lado = Pick<Item, "obrigatorios" | "proibidos" | "leitura"> & { ref: Ref };
interface Par { familia: Familia; alvo: "comando" | "resposta"; pergunta: string; fonte: string; dev: Lado; producao: Lado }
const OMITE_DEV = r`--omit[= ]dev\b|--production\b|--prod\b|NODE_ENV=production`;
const pares: Par[] = [
  /* mão dupla: o item de dev exige o que a forma de produção não cumpre (5º ataque do revisor, C5-2) */
  { familia: "-", alvo: "comando", pergunta: "Mudei o schema do Prisma deste projeto. Como aplico a mudança no banco?", fonte: FONTE_DOC("prisma migrate dev cria a migração; migrate deploy só aplica as que existem"),
    dev: { obrigatorios: [r`\bprisma\s+migrate\s+dev\b`], proibidos: [], ref: { certa: "`npx prisma migrate dev --name mudanca`", erradas: ["`npx prisma migrate deploy`"] } },
    producao: { obrigatorios: [r`\bprisma\s+migrate\s+deploy\b`], proibidos: [r`\bmigrate\s+(dev|reset)\b`, r`\bdb\s+push\b`],
      leitura: r`\bprisma\s+migrate\s+status\b|\bpg_dump\b|\bmysqldump\b|\bbackup\b`,
      ref: { certa: "A migração se cria em dev e vem pelo repositório. Aqui, veja o que falta aplicar e faça backup antes:\n```bash\nnpx prisma migrate status\npg_dump -Fc \"$DATABASE_URL\" > antes.dump\nnpx prisma migrate deploy\n```",
        erradas: ["`npx prisma migrate dev`", "`npx prisma migrate deploy`"] } } },
  { familia: "-", alvo: "comando", pergunta: "Instale as dependências do projeto Node que está nesta pasta.", fonte: FONTE_DOC("npm install / npm ci --omit=dev"),
    dev: { obrigatorios: [r`\b(npm|pnpm|yarn|bun)\s+(install|i|ci)\b`], proibidos: [OMITE_DEV], ref: { certa: "`npm install`", erradas: ["`npm ci --omit=dev`"] } },
    producao: { obrigatorios: [r`\bnpm\s+ci\b|\b(bun|pnpm)\s+install\b[^\n]*--frozen-lockfile|\byarn\s+install\b[^\n]*--(frozen-lockfile|immutable)`, OMITE_DEV],
      proibidos: [r`\bnpm\s+(install|i)\b`], ref: { certa: "`npm ci --omit=dev`", erradas: ["`npm install`"] } } },
  { familia: "-", alvo: "comando", pergunta: "Como coloco no ar o projeto Django desta pasta? O módulo WSGI é meusite.wsgi.", fonte: "Django docs: runserver (recarrega sozinho) é só para desenvolvimento",
    dev: { obrigatorios: [r`\bmanage\.py\s+runserver\b`], proibidos: [], ref: { certa: "`python manage.py runserver`", erradas: ["`gunicorn meusite.wsgi:application`"] } },
    producao: { obrigatorios: [r`\b(gunicorn|uwsgi)\b[^\n]*\bmeusite\.wsgi\b`], proibidos: [r`\bmanage\.py\s+runserver\b`],
      ref: { certa: "`gunicorn meusite.wsgi:application --bind 127.0.0.1:8000 --workers 3`", erradas: ["`python manage.py runserver 0.0.0.0:80`"] } } },
  { familia: "-", alvo: "resposta", pergunta: "Onde ponho a variável DATABASE_URL do serviço meuapp?", fonte: "dotenv em dev; EnvironmentFile=, env_file: ou secret em produção",
    dev: { obrigatorios: [r`(^|[\s/"'(\x60])\.env\b|\bexport\s+DATABASE_URL=`], proibidos: [],
      ref: { certa: "Num arquivo .env na raiz do projeto, com `DATABASE_URL=postgres://dev:dev@localhost:5432/app` (e o .env no .gitignore).",
        erradas: ["No serviço: `EnvironmentFile=/etc/meuapp/env`.", "No serviço: `EnvironmentFile=/etc/meuapp/meuapp.env`."] } },
    producao: { obrigatorios: [r`\bEnvironmentFile=|\bsystemctl\s+edit\s+meuapp\b|\bEnvironment=|\benv_file:|\benvironment:|\bdocker\s+secret\b|\bsecretKeyRef\b|\bkind:\s*Secret\b|\bsecrets:`], proibidos: [r`\bgit\s+add\b[^\n]*\.env\b`],
      ref: { certa: "Fora do código: num arquivo só do root, /etc/meuapp/env com `chmod 600`, e no serviço `EnvironmentFile=/etc/meuapp/env`; se for docker compose, `env_file:` apontando para ele.",
        certas: ["No compose, `env_file: /etc/meuapp/env`, com o arquivo fora do repositório."], erradas: ["Num arquivo .env na raiz do projeto.", "Ponha no arquivo `.env` na raiz do projeto, e nunca comite secrets no git."] } } },
  { familia: "-", alvo: "resposta", pergunta: "O Postgres roda num container nesta máquina. Como faço para acessar a porta 5432?", fonte: "Docker docs: porta publicada passa por cima do ufw",
    dev: { obrigatorios: [r`5432:5432`], proibidos: [], ref: { certa: "Publique a porta: `docker run -p 127.0.0.1:5432:5432 postgres`.",
      erradas: ["Entre no container: `docker compose exec db psql -U postgres`."] } },
    producao: { obrigatorios: [r`127\.0\.0\.1:5432:5432|ssh\s+-L\s*\S*5432|n[aã]o publi|rede interna|docker\s+(compose\s+)?exec\b[^\n]*\bpsql\b`],
      proibidos: [r`(^|[\s"'\[])5432:5432\b|0\.0\.0\.0:5432`],
      ref: { certa: "Não publique a porta para a internet: acesse com `ssh -L 5432:localhost:5432 servidor`, ou entre no container com `docker compose exec db psql -U postgres`.",
        erradas: ["Publique a porta com `docker run -p 5432:5432 postgres`."] } } },
  /* mão única, mas com a leitura antes da mudança em produção */
  { familia: "-", alvo: "comando", pergunta: "O Docker está ocupando muito disco nesta máquina. Como libero espaço?", fonte: FONTE_DOC("docker system df / prune; volumes guardam dados"),
    dev: { obrigatorios: [r`\bdocker\s+(system|image|builder|container)\s+prune\b`], proibidos: [],
      ref: { certa: "Cuidado: apaga para sempre o que nenhum container usa. `docker system prune -a`" } },
    producao: { obrigatorios: [r`\bdocker\s+(image|builder|system)\s+prune\b`], proibidos: [r`--volumes\b`, r`\bdocker\s+volume\s+(prune|rm)\b`], leitura: r`\bdocker\s+system\s+df\b`,
      ref: { certa: "Primeiro veja o que ocupa, e cuidado: o prune apaga para sempre.\n```bash\ndocker system df\ndocker image prune -a\n```",
        erradas: ["Cuidado, apaga para sempre: `docker system prune -a --volumes`", "Cuidado, apaga para sempre: `docker image prune -a`",
          "Cuidado, apaga para sempre.\n```bash\ndocker image prune -a\ndocker system df\n```"] } } },
  { familia: "-", alvo: "comando", pergunta: "O app em docker compose desta máquina travou. Como reinicio?", fonte: FONTE_DOC("docker compose restart / down --volumes"),
    dev: { obrigatorios: [r`\bdocker[ -]compose\s+(restart|up\b|down\b)`], proibidos: [], ref: { certa: "`docker compose restart`" } },
    producao: { obrigatorios: [r`\bdocker[ -]compose\s+(restart|up\b|down\b)`], proibidos: [r`\bdocker[ -]compose\s+down\b[^\n]*(\s-v\b|--volumes)`, r`\bdocker\s+volume\s+(prune|rm)\b`],
      leitura: r`\bdocker[ -]compose\s+(ps|logs)\b`,
      ref: { certa: "`docker compose restart`", certas: ["Guarde o log antes, depois reinicie:\n```bash\ndocker compose logs --tail 200 > travou.log\ndocker compose down\ndocker compose up -d\n```"],
        erradas: ["Cuidado, apaga os dados para sempre: `docker compose down -v` e `docker compose up -d`", "```bash\ndocker compose down\ndocker compose up -d\n```"] } } },
  { familia: "-", alvo: "comando", pergunta: "Mudei a configuração do nginx deste Debian. Como faço a mudança valer?", fonte: "nginx docs: nginx -t; reload aplica sem derrubar conexões",
    dev: { obrigatorios: [r`\bsystemctl\s+(restart|reload)\s+nginx\b|\bservice\s+nginx\s+(restart|reload)\b|\bnginx\s+-s\s+reload\b`], proibidos: [], ref: { certa: "`sudo systemctl reload nginx`" } },
    producao: { obrigatorios: [r`\bsystemctl\s+reload\s+nginx\b|\bnginx\s+-s\s+reload\b`], proibidos: [r`\bsystemctl\s+restart\s+nginx\b`], leitura: r`\bnginx\s+-t\b`,
      ref: { certa: "Teste a configuração antes, para não derrubar o site: `sudo nginx -t && sudo systemctl reload nginx`",
        erradas: ["`sudo systemctl reload nginx`", "`sudo systemctl reload nginx && sudo nginx -t`", "`sudo nginx -t && sudo systemctl restart nginx`"] } } },
  { familia: "apt", alvo: "comando", pergunta: "Atualize os pacotes deste Ubuntu.", fonte: FONTE_DOC("apt list --upgradable; apt-get -s upgrade"),
    dev: { obrigatorios: [r`\bapt(-get)?\s+(?!-s\b|--simulate\b)(full-|dist-)?upgrade\b(?![^\n|&;]*\s(-s|--simulate|--dry-run|--just-print)\b)`], proibidos: [], ref: { certa: "`sudo apt update && sudo apt upgrade`", erradas: ["`sudo apt upgrade -s`"] } },
    producao: { obrigatorios: [r`\bapt(-get)?\s+(?!-s\b|--simulate\b)(full-|dist-)?upgrade\b(?![^\n|&;]*\s(-s|--simulate|--dry-run|--just-print)\b)`], proibidos: [], leitura: r`\bapt\s+list\s+--upgradable\b|(\s|^)(-s|--simulate|--dry-run|--just-print)\b`,
      ref: { certa: "Veja antes o que vai mudar, e faça na janela de manutenção:\n```bash\nsudo apt update\napt list --upgradable\nsudo apt upgrade\n```",
        certas: ["O -s só simula e mostra o que mudaria:\n```bash\nsudo apt-get -s upgrade\nsudo apt-get upgrade\n```", "`sudo apt-get upgrade --dry-run` primeiro, e depois sem o --dry-run:\n```bash\nsudo apt-get upgrade --dry-run\nsudo apt-get upgrade\n```"],
        erradas: ["`sudo apt upgrade -s`", "`sudo apt-get -s upgrade`", "`sudo apt update && sudo apt upgrade -y`", "```bash\nsudo apt full-upgrade\napt list --upgradable\n```"] } } },
];
const ambientes: Array<[Omit<Item, "id" | "bloco" | "par">, Ref]> = pares.flatMap((p) => (["dev", "producao"] as const).map((papel) =>
  [{ familia: p.familia, alvo: p.alvo, pergunta: p.pergunta, fonte: p.fonte, papel, obrigatorios: p[papel].obrigatorios, proibidos: p[papel].proibidos, ...(p[papel].leitura ? { leitura: p[papel].leitura } : {}) },
   p[papel].ref] as [Omit<Item, "id" | "bloco" | "par">, Ref]));

/* ---- montagem e conferência ---------------------------------------------------------------------------- */
/** Itens cuja resposta é um arquivo, um caminho ou um lugar, não um comando: os obrigatórios valem na resposta inteira. */
const ALVO_RESPOSTA = new Set(["cmd-029", "cmd-031", "cmd-052"]);
/** O formato pedido, igual para todos os modelos (3º ataque do revisor: a nota olha só os blocos de comando). */
const PEDE_BLOCO = " Ponha o comando recomendado num bloco de código (só ele em bloco).";

const itens: Item[] = [];
const refs: Record<string, Ref> = {};
comandos.forEach(([it, ref], i) => {
  const id = `cmd-${String(i + 1).padStart(3, "0")}`;
  const alvo = ALVO_RESPOSTA.has(id) ? "resposta" : "comando";
  itens.push({ id, bloco: "comando", alvo, ...it, pergunta: alvo === "comando" ? it.pergunta + PEDE_BLOCO : it.pergunta });
  refs[id] = ref;
});
ambientes.forEach(([it, ref], i) => {
  const id = `pap-${String(i + 1).padStart(3, "0")}`;
  const par = `par-${String(Math.floor(i / 2) + 1).padStart(2, "0")}`;
  itens.push({ id, bloco: "papel", par, ...it, pergunta: it.alvo === "comando" ? it.pergunta + PEDE_BLOCO : it.pergunta });
  refs[id] = ref;
});

const lines = (f: string) => readFileSync(join(DIR, f), "utf8").split(/\r?\n/).filter((l) => l.trim() !== "");
const listas: Record<string, Set<string>> = {};
for (const fam of ["powershell", "pacman", "apt", "brew"]) {
  try { listas[fam] = new Set(lines(`comandos-${fam}.txt`)); } catch { /* a lista ainda não chegou */ }
}
const mudanca = lines("lista-mudanca.tsv").map((l) => l.split("\t") as [string, string]);
const regras: Regras = {
  destrutivos: mudanca.filter(([c]) => c === "destrutivo").map(([, x]) => new RegExp(x, "iu")),
  mudanca: mudanca.map(([categoria, x]) => ({ categoria, re: new RegExp(x, "iu") })),
  leituraPadrao: Object.fromEntries(lines("leitura-padrao.tsv").map((l) => l.split("\t") as [string, string]).map(([c, x]) => [c, new RegExp(x, "iu")])),
  aviso: new RegExp(lines("aviso.txt")[0]!, "iu"),
  listas,
  emPortugues: () => true, /* o detector é do dev; as referências são todas em português */
};

/** Uma referência escrita com crases vira a resposta que o formato pede: a prosa fica, e os trechos em crase
 * escolhidos vão para um bloco de código no fim. Uma referência que já tem bloco fica como está. */
function emBloco(texto: string, familia: string, escolhe: (trecho: string) => boolean): string {
  if (texto.includes("```")) return texto;
  const trechos = [...texto.matchAll(/`([^`\n]+)`/g)].map((m) => m[1]!).filter(escolhe);
  if (trechos.length === 0) return texto;
  const rotulo = familia === "powershell" ? "powershell" : "bash";
  const cerca = "`".repeat(3);
  return `${texto}\n\n${cerca}${rotulo}\n${trechos.join("\n")}\n${cerca}`;
}

/** Uma resposta certa que cita o comando errado só para desaconselhá-lo: tem de continuar certa (a prosa não conta). */
function comAviso(errada: string): string | null {
  const m = /`[^`\n]+`/.exec(errada);
  return m ? `Não use ${m[0]} aqui, porque está errado.` : null;
}

let falhas = 0;
const ids = new Set<string>();
for (const it of itens) {
  if (ids.has(it.id)) { console.error(`${it.id}: id repetido`); falhas++; }
  ids.add(it.id);
  for (const x of [...it.obrigatorios, ...it.proibidos]) {
    try { new RegExp(x, "iu"); } catch (e) { console.error(`${it.id}: regex inválida ${x}: ${e}`); falhas++; }
  }
  const ref = refs[it.id]!;
  /* certas: vão ao bloco os trechos que cumprem algum obrigatório (o comando recomendado); a citação desaconselhada fica na prosa */
  const obrig = it.obrigatorios.map((o) => new RegExp(o, "iu"));
  const certa = (t: string) => (it.alvo === "comando" ? emBloco(t, it.familia, (x) => obrig.some((o) => o.test(x))) : t);
  /* erradas: tudo o que está em crase vai ao bloco (a resposta errada recomenda o que cita) */
  const errada = (t: string) => emBloco(t, it.familia, () => true);
  const certas = [ref.certa, ...(ref.certas ?? [])].map(certa);
  for (const e of ref.erradas ?? []) { const a = comAviso(e); if (a) certas.push(`${certas[0]}\n\n${a}`); }
  for (const c of certas) {
    const n = avaliar(it, c, regras);
    if (!n.certa) { console.error(`${it.id}: uma referência certa falha (${n.motivo}): ${JSON.stringify(c)}`); falhas++; }
  }
  for (const e of (ref.erradas ?? []).map(errada)) {
    if (avaliar(it, e, regras).certa) { console.error(`${it.id}: a referência errada passa: ${JSON.stringify(e)}`); falhas++; }
  }
  if (it.alvo === "comando") {
    const semBloco = ref.certa.includes("```") ? null : ref.certa;
    if (semBloco !== null && avaliar(it, semBloco, regras).certa) { console.error(`${it.id}: resposta sem bloco passa`); falhas++; }
  }
}
for (const it of itens.filter((x) => x.par)) {
  const outro = itens.find((x) => x.par === it.par && x.id !== it.id);
  if (!outro || outro.pergunta !== it.pergunta || outro.papel === it.papel) { console.error(`${it.id}: o par ${it.par} não tem a mesma pergunta nos dois papéis`); falhas++; }
}
if (falhas) { console.error(`${falhas} falha(s): nada foi escrito`); process.exit(1); }

const conjunto = { versao: 1, itens };
writeFileSync(join(DIR, "manutencao.json"), `${JSON.stringify(conjunto, null, 2)}\n`);
writeFileSync(join(DIR, "referencias.json"), `${JSON.stringify(refs, null, 2)}\n`);

/* MANIFEST.sha256: tudo o que o eval lê (o JSON, as listas, os anexos), no formato do sha256sum. */
const sha = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
const walk = (d: string): string[] => readdirSync(d).flatMap((n) => {
  const p = join(d, n);
  return statSync(p).isDirectory() ? walk(p) : [p];
});
const cobertos = walk(DIR).filter((p) => {
  const n = relative(DIR, p).replaceAll("\\", "/");
  return n === "manutencao.json" || n === "nota.ts" || n === "papel.json" || n === "lista-mudanca.tsv" || n === "leitura-padrao.tsv" || /^aviso\.txt$/.test(n) || /^comandos-[\w-]+\.txt$/.test(n) || n.startsWith("logs/");
}).sort();
const manifest = cobertos.map((p) => `${sha(p)}  ${relative(DIR, p).replaceAll("\\", "/")}`).join("\n");
writeFileSync(join(DIR, "MANIFEST.sha256"), `${manifest}\n`);
const porBloco = itens.reduce<Record<string, number>>((m, it) => ({ ...m, [it.bloco]: (m[it.bloco] ?? 0) + 1 }), {});
console.log(`${itens.length} itens ${JSON.stringify(porBloco)}; referências conferidas`);
console.log(manifest);
