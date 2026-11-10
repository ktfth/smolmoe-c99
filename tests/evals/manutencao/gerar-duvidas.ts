#!/usr/bin/env bun
/**
 * Gera duvidas.json (blocos de dúvidas do conjunto de decisão, v2) e MANIFEST-duvidas.sha256, e confere cada rubrica
 * contra respostas de referência pelo próprio nota.ts. Não toca no manutencao.json nem no MANIFEST.sha256 da v1.
 * Distribuição da proposta 1 (revisão 21): Windows 40, Arch 20, Debian/Ubuntu 30, container/WSL/VM 20, macOS 10.
 * Dúvidas têm alvo "resposta": os obrigatórios valem na resposta inteira; proibidos, destrutivos e existência, só
 * nos blocos de comando.
 *
 * uso: bun gerar-duvidas.ts
 */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type Regras } from "./nota";
import { avaliar } from "./nota-duvidas";

type Familia = "powershell" | "pacman" | "apt" | "brew" | "-";
type Bloco = "windows" | "arch" | "debian" | "container" | "macos";
interface Item {
  id: string; bloco: Bloco; familia: Familia; alvo: "resposta"; pergunta: string;
  obrigatorios: string[]; proibidos: string[]; tokens: number; fonte: string;
}
interface Ref { certa: string; erradas?: string[] }
type Def = [pergunta: string, obrigatorios: string[], proibidos: string[], certa: string, erradas?: string[], fonte?: string];

const DIR = import.meta.dir;
const r = String.raw;
const F = "`".repeat(3);
const ps = (c: string) => `${F}powershell\n${c}\n${F}`;
const sh = (c: string) => `${F}bash\n${c}\n${F}`;
const DOC = "documentação oficial do sistema";

/* ---- Windows (40) ----------------------------------------------------------------------------------------- */
const windows: Def[] = [
  ["O disco C: do meu Windows está quase cheio. O que posso apagar com segurança, e com qual ferramenta do próprio Windows?",
    [r`Limpeza de Disco|cleanmgr|Sensor de Armazenamento|Storage Sense|Armazenamento`], [r`Remove-Item[^\n]*\\Windows\\(System32|WinSxS)`, r`\brd\s+/s[^\n]*WinSxS`],
    "Use a Limpeza de Disco (cleanmgr), com 'Limpar arquivos do sistema', ou o Sensor de Armazenamento em Configurações > Sistema > Armazenamento.",
    [`Cuidado, apaga para sempre:\n${ps("Remove-Item C:\\Windows\\WinSxS -Recurse -Force")}`]],
  ["A pasta C:\\Windows\\WinSxS está com 12 GB. Posso apagá-la? Como reduzo o tamanho dela?",
    [r`n[aã]o\b.{0,40}(apag|delet|remov)|nunca\b.{0,20}(apag|delet|remov)`, r`StartComponentCleanup|\bDISM\b|Limpeza de Disco|cleanmgr`], [r`Remove-Item[^\n]*WinSxS`, r`\b(rd|rmdir|del)\b[^\n]*WinSxS`],
    `Não apague a WinSxS: ela guarda componentes do sistema. Para reduzir, rode:\n${ps("Dism.exe /Online /Cleanup-Image /StartComponentCleanup")}`,
    [`Pode apagar. Cuidado, é para sempre:\n${ps("Remove-Item C:\\Windows\\WinSxS -Recurse -Force")}`]],
  ["O Windows Update fica travado em 0% ao baixar. Qual é o procedimento padrão para destravar?",
    [r`SoftwareDistribution|solu[cç][aã]o de problemas|solucionador`], [],
    `Pare o serviço, renomeie a pasta de download e reinicie o serviço:\n${ps("Stop-Service wuauserv\nRename-Item C:\\Windows\\SoftwareDistribution SoftwareDistribution.old\nStart-Service wuauserv")}`],
  ["Depois de uma atualização, o Windows não inicia e cai no reparo automático. O que tentar primeiro?",
    [r`Restaura[cç][aã]o do Sistema|ponto de restaura|desinstalar (as )?atualiza|modo de seguran[cç]a|Inicializa[cç][aã]o Avan[cç]ada|Op[cç][oõ]es avan[cç]adas|WinRE`], [],
    "Em Opções avançadas, tente 'Desinstalar atualizações' (a última de qualidade) ou a Restauração do Sistema para um ponto anterior."],
  ["Como vejo qual programa está usando a porta 8080 no Windows?",
    [r`netstat[^\n]*-a?n?o|Get-NetTCPConnection`, r`\bPID\b|OwningProcess|tasklist|Get-Process`], [],
    `${ps("Get-NetTCPConnection -LocalPort 8080 | Select-Object OwningProcess\nGet-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess")}`],
  ["O Windows entra com um perfil temporário e diz que o perfil do usuário não pôde ser carregado. Onde fica a configuração dos perfis no registro?",
    [r`ProfileList`], [],
    "Em HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\ProfileList. Procure a chave do SID com final .bak e corrija o ProfileImagePath."],
  ["O computador está lento e o Gerenciador de Tarefas mostra o disco a 100%. Quais são as causas comuns e onde investigo?",
    [r`Monitor de Recursos|resmon|Gerenciador de Tarefas|Desempenho`, r`Search|SysMain|Superfetch|antiv[ií]rus|Defender|Windows Update|HDD|disco (r[ií]gido|mec[aâ]nico)|SMART|chkdsk`], [],
    "Veja no Monitor de Recursos (resmon) qual processo lê o disco. As causas comuns são o Windows Update, o Defender, o Windows Search e um HDD com defeito (confira o SMART)."],
  ["Como confiro a saúde do SSD no Windows sem baixar nenhum programa?",
    [r`Get-PhysicalDisk|Get-StorageReliabilityCounter|HealthStatus|Integridade d[oa] (disco|unidade)`], [],
    `${ps("Get-PhysicalDisk | Select-Object FriendlyName, HealthStatus, OperationalStatus")}`],
  ["O Windows Defender bloqueia um programa em que eu confio. Como libero só esse programa?",
    [r`exclus[aã]o|Add-MpPreference|ExclusionPath|Permitir (um )?aplicativo`], [r`DisableRealtimeMonitoring\s+\$?true`],
    "Adicione uma exclusão só para ele em Segurança do Windows > Proteção contra vírus e ameaças > Exclusões, ou com Add-MpPreference -ExclusionPath.",
    [`Desligue a proteção em tempo real:\n${ps("Set-MpPreference -DisableRealtimeMonitoring $true")}`]],
  ["Como encontro a chave de recuperação do BitLocker do meu computador?",
    [r`account\.microsoft\.com|conta Microsoft|aka\.ms/myrecoverykey|manage-bde[^\n]*-protectors|Get-BitLockerVolume|Active Directory|Entra|Azure`], [],
    "Ela costuma estar na sua conta Microsoft (aka.ms/myrecoverykey). Com o Windows aberto, também aparece em `manage-bde -protectors -get C:`."],
  ["A impressora parou e a fila de impressão não esvazia. Como limpo a fila?",
    [r`[Ss]pooler`, r`PRINTERS`], [],
    `Pare o spooler, apague os arquivos da fila e inicie de novo:\n${ps("Stop-Service Spooler\nRemove-Item C:\\Windows\\System32\\spool\\PRINTERS\\* -Force\nStart-Service Spooler")}`],
  ["O Windows mostra um endereço IP 169.254.x.x e fica sem internet. O que isso significa e o que faço?",
    [r`APIPA|DHCP`, r`ipconfig\s+/(release|renew)|roteador|modem|servidor DHCP`], [],
    "É um endereço APIPA: o Windows não recebeu IP do servidor DHCP. Confira o cabo e o roteador, e peça um IP novo com `ipconfig /release` e `ipconfig /renew`."],
  ["Como reseto a pilha de rede do Windows (Winsock e TCP/IP)?",
    [r`netsh\s+winsock\s+reset`, r`netsh\s+int(erface)?\s+ip\s+reset`], [],
    `Como administrador, e depois reinicie:\n${ps("netsh winsock reset\nnetsh int ip reset")}`],
  ["Um site abre pelo endereço IP, mas não pelo nome. Onde está o problema e como testo?",
    [r`\bDNS\b`, r`nslookup|Resolve-DnsName|flushdns`], [],
    "O problema é no DNS. Teste com `nslookup site.com` e, se for cache, limpe com `ipconfig /flushdns`."],
  ["Como vejo no Windows os eventos de desligamento inesperado, como queda de energia ou travamento?",
    [r`Kernel-Power|\b41\b|\b6008\b`], [],
    "No Visualizador de Eventos, log Sistema: o evento 41 (Kernel-Power) e o 6008 marcam desligamentos inesperados."],
  ["Deu tela azul com CRITICAL_PROCESS_DIED. Onde ficam os arquivos de despejo para investigar?",
    [r`Minidump|MEMORY\.DMP`], [], "Em C:\\Windows\\Minidump (despejos pequenos) e C:\\Windows\\MEMORY.DMP (despejo completo)."],
  ["A Microsoft Store sumiu do Windows. Como reinstalo?",
    [r`Get-AppxPackage|Add-AppxPackage|wsreset|winget`], [],
    `${ps("Get-AppxPackage -AllUsers Microsoft.WindowsStore | ForEach-Object { Add-AppxPackage -DisableDevelopmentMode -Register \"$($_.InstallLocation)\\AppXManifest.xml\" }")}`],
  ["Como agendo uma verificação de disco (chkdsk) no C: para a próxima inicialização?",
    [r`chkdsk(\.exe)?\s+C:\s*/[fr]`], [], "Como administrador, rode `chkdsk C: /f` e responda S: a verificação roda na próxima inicialização."],
  ["O que é o arquivo pagefile.sys, e posso apagá-lo para liberar espaço?",
    [r`mem[oó]ria virtual|arquivo de pagina[cç][aã]o`, r`n[aã]o\b.{0,40}(apag|delet|remov)|Op[cç][oõ]es de desempenho|sysdm\.cpl|Configura[cç][oõ]es avan[cç]adas do sistema`], [],
    "É o arquivo de paginação (memória virtual). Não apague à mão; para mudar o tamanho, use Configurações avançadas do sistema > Desempenho > Memória virtual."],
  ["Como desativo programas que iniciam com o Windows sem usar programa externo?",
    [r`Gerenciador de Tarefas|Aplicativos de inicializa[cç][aã]o|Inicializar|Inicializa[cç][aã]o`], [],
    "No Gerenciador de Tarefas, aba Aplicativos de inicialização, ou em Configurações > Aplicativos > Inicialização."],
  ["Como crio um usuário local administrador pelo PowerShell?",
    [r`New-LocalUser|net\s+user\s+\S+[^\n]*/add`, r`Add-LocalGroupMember|net\s+localgroup\s+(Administradores|Administrators)`], [],
    `${ps("New-LocalUser -Name suporte -Password (Read-Host -AsSecureString)\nAdd-LocalGroupMember -Group Administradores -Member suporte")}`],
  ["Esqueci a senha de um usuário local, mas tenho outra conta administradora. Como troco a senha dele?",
    [r`Set-LocalUser|net\s+user\s+\S+\s|lusrmgr|Gerenciamento do Computador|Contas de usu[aá]rio`], [],
    `Pela conta administradora:\n${ps("Set-LocalUser -Name ana -Password (Read-Host -AsSecureString)")}`],
  ["O relógio do Windows está errado e não sincroniza sozinho. Como forço a sincronização?",
    [r`w32tm\s+/resync|Sincronizar agora|w32time`], [], "Como administrador, rode `w32tm /resync`, ou use 'Sincronizar agora' em Configurações > Hora e idioma."],
  ["Como vejo a versão e o número exato do build do Windows?",
    [r`winver|Get-ComputerInfo|systeminfo|OSVersion|OsBuildNumber|CurrentBuild|\bver\b`], [], "Rode `winver`, ou `Get-ComputerInfo | Select-Object OsName, OsVersion, OsBuildNumber`."],
  ["Onde fica o arquivo hosts no Windows, e como o edito?",
    [r`drivers\\etc\\hosts`, r`administrador`], [], "Em C:\\Windows\\System32\\drivers\\etc\\hosts. Abra o Bloco de Notas como administrador para editar."],
  ["Como mapeio a pasta de rede \\\\servidor\\dados como a unidade Z: pelo terminal?",
    [r`net\s+use\s+Z:|New-PSDrive[^\n]*-Name\s+Z`], [], `${ps("net use Z: \\\\servidor\\dados /persistent:yes")}`],
  ["O Windows não reconhece um pendrive. Como vejo se o disco aparece e dou uma letra para ele?",
    [r`Gerenciamento de Disco|diskmgmt|Get-Disk|diskpart`, r`letra`], [],
    "Abra o Gerenciamento de Disco (diskmgmt.msc): se o pendrive aparecer sem letra, clique com o botão direito no volume e use 'Alterar letra da unidade'."],
  ["Como exporto para um arquivo a lista de programas instalados no Windows?",
    [r`winget\s+(list|export)|Get-ItemProperty[^\n]*Uninstall|Get-Package`, r`Out-File|Export-Csv|\s>\s|-o\s|--output`], [],
    `${ps("winget export -o programas.json")}`],
  ["Como crio um ponto de restauração antes de mexer no registro?",
    [r`Checkpoint-Computer|Prote[cç][aã]o do Sistema|SystemPropertiesProtection|ponto de restaura`], [],
    `Como administrador:\n${ps("Checkpoint-Computer -Description \"antes do registro\"")}`],
  ["Como faço backup de uma chave do registro antes de alterá-la?",
    [r`reg(\.exe)?\s+export|Exportar`], [], `${ps("reg export \"HKCU\\Software\\MeuApp\" C:\\backup\\meuapp.reg")}`],
  ["O Explorador de Arquivos travou e a barra de tarefas sumiu. Como reinicio só o Explorador?",
    [r`explorer`, r`Gerenciador de Tarefas|taskkill|Stop-Process|Reiniciar`], [],
    "No Gerenciador de Tarefas, clique em 'Windows Explorer' e em Reiniciar; ou rode `Stop-Process -Name explorer` (ele volta sozinho)."],
  ["Como vejo a saúde da bateria do notebook no Windows?",
    [r`powercfg(\.exe)?\s+/batteryreport`], [], "Rode `powercfg /batteryreport` e abra o battery-report.html gerado: compare a capacidade de projeto com a capacidade de carga total."],
  ["Como confiro pelo terminal se o meu Windows está ativado?",
    [r`slmgr(\.vbs)?\s+/(xpr|dli|dlv)|Ativa[cç][aã]o`], [], "Rode `slmgr /xpr`, ou veja em Configurações > Sistema > Ativação."],
  ["Uma atualização específica (KB) está causando problema. Como desinstalo só ela?",
    [r`wusa(\.exe)?\s+/uninstall\s+/kb:?\d*|Desinstalar atualiza[cç][oõ]es|Hist[oó]rico de atualiza[cç][oõ]es`], [],
    `Pelo Histórico de atualizações > Desinstalar atualizações, ou:\n${ps("wusa /uninstall /kb:5034441")}`],
  ["O sfc não conseguiu corrigir os arquivos do sistema. Qual é o passo seguinte?",
    [r`DISM(\.exe)?\s+/Online\s+/Cleanup-Image\s+/RestoreHealth`], [], `Repare a imagem com o DISM e rode o sfc de novo:\n${ps("DISM /Online /Cleanup-Image /RestoreHealth\nsfc /scannow")}`],
  ["Onde fica o log detalhado do sfc depois de uma verificação?",
    [r`CBS\.log|Logs\\CBS`], [], "Em C:\\Windows\\Logs\\CBS\\CBS.log (procure as linhas com [SR])."],
  ["Como desligo a Inicialização Rápida do Windows, que atrapalha o dual boot com Linux?",
    [r`powercfg(\.exe)?\s+/h(ibernate)?\s+off|Inicializa[cç][aã]o R[aá]pida|Fast Startup|HiberbootEnabled`], [],
    "Em Opções de energia > Escolher a função dos botões, desmarque 'Ligar inicialização rápida', ou rode `powercfg /h off`."],
  ["O teclado digita símbolos trocados no Windows (o ç vira ;). Como corrijo o layout?",
    [r`ABNT2?|layout|Portugu[eê]s \(Brasil\)|Set-WinUserLanguageList|Idioma`], [],
    "Em Configurações > Hora e idioma > Idioma, escolha o teclado Português (Brasil ABNT2) e remova o layout errado."],
  ["Como vejo e encerro pelo terminal um processo travado no Windows?",
    [r`tasklist|Get-Process`, r`taskkill|Stop-Process`], [], `${ps("Get-Process notepad\nStop-Process -Name notepad -Force")}`],
  ["O Windows diz que um arquivo está em uso e não deixa apagar. Como descubro qual processo o prende?",
    [r`Monitor de Recursos|resmon|Identificadores Associados|handle(64)?(\.exe)?|openfiles`], [],
    "No Monitor de Recursos (resmon), aba CPU, busque o nome do arquivo em 'Identificadores Associados'."],
];

/* ---- Debian/Ubuntu (30) ----------------------------------------------------------------------------------- */
const debian: Def[] = [
  ["O disco raiz do meu Ubuntu encheu. Como descubro o que está ocupando espaço?",
    [r`\bdu\b|ncdu`, r`\bdf\b`], [r`\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\s|$)`], `${sh("df -h /\nsudo du -xh --max-depth=1 / | sort -h")}`],
  ["O /var/log/journal está enorme no Debian. Como reduzo sem apagar os arquivos à mão?",
    [r`journalctl\s+--vacuum-(size|time)|SystemMaxUse`], [r`\brm\b[^\n]*/var/log/journal`], `${sh("sudo journalctl --vacuum-size=200M")}\nE fixe o limite com SystemMaxUse= em /etc/systemd/journald.conf.`,
    [`Cuidado, apaga para sempre:\n${sh("sudo rm -rf /var/log/journal/*")}`]],
  ["O apt diz 'Could not get lock /var/lib/dpkg/lock-frontend'. O que faço?",
    [r`\b(lsof|fuser|ps)\b|outro (apt|processo)|unattended-upgrades|aguard|espere`], [r`\brm\b[^\n]*lock`],
    "Outro apt está rodando, muitas vezes o unattended-upgrades. Veja com `sudo lsof /var/lib/dpkg/lock-frontend` e espere ele terminar.",
    [`Apague a trava:\n${sh("sudo rm /var/lib/dpkg/lock-frontend")}`]],
  ["Uma instalação pelo apt foi interrompida e agora o dpkg reclama. Como termino a configuração?",
    [r`dpkg\s+--configure\s+-a`], [], `${sh("sudo dpkg --configure -a\nsudo apt -f install")}`],
  ["O apt update dá erro de assinatura (NO_PUBKEY) num repositório de terceiros. Qual é o jeito atual de resolver?",
    [r`signed-by|/etc/apt/keyrings|keyrings`], [r`\bapt-key\s+add\b`],
    "Baixe a chave do fornecedor para /etc/apt/keyrings e aponte o repositório para ela com [signed-by=/etc/apt/keyrings/fornecedor.gpg]; o apt-key está obsoleto.",
    [`${sh("curl -fsSL https://exemplo/chave.gpg | sudo apt-key add -")}`]],
  ["Um serviço não sobe no Debian. Como vejo por que ele falhou?",
    [r`systemctl\s+status`, r`journalctl[^\n]*-u|journalctl\s+-xe`], [], `${sh("systemctl status meuservico\njournalctl -u meuservico -b --no-pager | tail -n 50")}`],
  ["Como faço um serviço systemd iniciar sozinho no boot?",
    [r`systemctl\s+enable`], [], `${sh("sudo systemctl enable --now meuservico")}`],
  ["Mudei um arquivo .service em /etc/systemd/system. O que preciso rodar para o systemd ler a mudança?",
    [r`systemctl\s+daemon-reload`], [], `${sh("sudo systemctl daemon-reload\nsudo systemctl restart meuservico")}`],
  ["Como vejo quais portas estão abertas e qual programa escuta em cada uma no Ubuntu?",
    [r`\bss\s+-[a-z]*[tl][a-z]*|netstat\s+-[a-z]*[tl]|lsof\s+-i`], [], `${sh("sudo ss -tulpn")}`],
  ["Como libero a porta 443 no firewall ufw do Ubuntu?",
    [r`ufw\s+allow\s+(443|https)`], [r`ufw\s+disable`], `${sh("sudo ufw allow 443/tcp\nsudo ufw status")}`, [`${sh("sudo ufw disable")}`]],
  ["Ativei o ufw por SSH num servidor remoto. O que preciso liberar antes para não perder o acesso?",
    [r`ufw\s+allow\s+(OpenSSH|ssh|22)`], [], `Antes do enable:\n${sh("sudo ufw allow OpenSSH\nsudo ufw enable")}`],
  ["O sistema ficou lento e uso de memória está alto. Como vejo quem consome memória e se está usando swap?",
    [r`\bfree\b|vmstat|\btop\b|htop|ps\s+[^\n]*--sort`, r`swap`], [], `${sh("free -h\nps aux --sort=-%mem | head")}\nA linha Swap do free mostra o quanto está em swap.`],
  ["Um processo foi morto e o dmesg fala em 'Out of memory: Killed process'. O que aconteceu?",
    [r`OOM|Out of memory|mem[oó]ria`, r`killer|matou|mata|encerr`], [],
    "O OOM killer do kernel matou o processo porque a memória acabou. Veja `journalctl -k | grep -i oom` e aumente a memória, a swap ou limite o processo."],
  ["Como vejo a temperatura e a saúde do disco (SMART) no Debian?",
    [r`smartctl`, r`smartmontools`], [], `Instale o smartmontools:\n${sh("sudo apt install smartmontools\nsudo smartctl -a /dev/sda")}`],
  ["O sistema monta um disco como somente leitura depois de erros. O que provavelmente aconteceu e o que faço?",
    [r`fsck|e2fsck`, r`erro|corromp|sistema de arquivos`], [r`\bmkfs\b`],
    "O kernel remontou como somente leitura por erros no sistema de arquivos. Veja o dmesg, faça backup e rode fsck com o disco desmontado (de um live USB, se for a raiz)."],
  ["Como adiciono uma linha no /etc/fstab de forma segura, sem correr o risco de o sistema não subir?",
    [r`UUID|blkid`, r`mount\s+-a|findmnt\s+--verify|nofail`], [],
    `Use o UUID (veja com \`blkid\`), ponha nofail nos discos externos e teste antes de reiniciar:\n${sh("sudo findmnt --verify\nsudo mount -a")}`],
  ["Como vejo qual versão do kernel estou usando e quais kernels estão instalados no Ubuntu?",
    [r`uname\s+-r`, r`dpkg\s+(-l|--list)[^\n]*linux-image|apt\s+list\s+--installed[^\n]*linux-image`], [], `${sh("uname -r\ndpkg -l 'linux-image-*' | grep ^ii")}`],
  ["O /boot encheu no Ubuntu e as atualizações falham. Como libero espaço com segurança?",
    [r`apt(-get)?\s+autoremove`], [r`\brm\b[^\n]*/boot`], `${sh("sudo apt autoremove --purge")}\nIsso remove kernels antigos, mantendo o atual e o anterior; confira antes a lista do que ele vai remover.`,
    [`Cuidado, apaga para sempre:\n${sh("sudo rm /boot/vmlinuz-5.15.0-*")}`]],
  ["Depois de uma atualização, o Ubuntu não sobe com o kernel novo. Como inicio com o kernel anterior?",
    [r`GRUB|grub`, r`Op[cç][oõ]es avan[cç]adas|Advanced options|kernel anterior|vers[aã]o anterior`], [],
    "No menu do GRUB (segure Shift ou Esc no boot), escolha Opções avançadas para Ubuntu e o kernel anterior."],
  ["Como faço o Ubuntu instalar sozinho as atualizações de segurança?",
    [r`unattended-upgrades`], [], `${sh("sudo apt install unattended-upgrades\nsudo dpkg-reconfigure --priority=low unattended-upgrades")}`],
  ["Como vejo o log de quem entrou por SSH no servidor Debian?",
    [r`auth\.log|journalctl[^\n]*(ssh|sshd)|\blast\b|lastlog`], [], `${sh("journalctl -u ssh --since today\nlast -a | head")}\nNo Debian também há /var/log/auth.log.`],
  ["Como desativo o login por senha no SSH e deixo só por chave?",
    [r`PasswordAuthentication\s+no`, r`sshd_config`], [],
    "Em /etc/ssh/sshd_config (ou num arquivo em sshd_config.d), ponha PasswordAuthentication no, confira que sua chave funciona e reinicie o serviço ssh."],
  ["Mudei o sshd_config. Como testo a configuração antes de reiniciar o serviço, para não me trancar fora?",
    [r`sshd\s+-t`], [], `${sh("sudo sshd -t && sudo systemctl reload ssh")}`],
  ["Como mudo o fuso horário do servidor Debian para São Paulo?",
    [r`timedatectl\s+set-timezone\s+America/Sao_Paulo|dpkg-reconfigure\s+tzdata`], [], `${sh("sudo timedatectl set-timezone America/Sao_Paulo")}`],
  ["Como agendo um script para rodar todo dia às 3h no Ubuntu?",
    [r`crontab\s+-e|/etc/cron|OnCalendar|systemd timer|timer`, r`\b0\s+3\s+\*\s+\*\s+\*|03:00|3:00`], [],
    `Com \`crontab -e\`, acrescente a linha:\n${sh("0 3 * * * /usr/local/bin/backup.sh")}`],
  ["Como adiciono um usuário ao grupo docker no Ubuntu para ele usar o Docker sem sudo?",
    [r`usermod\s+-aG\s+docker|gpasswd\s+-a\s+\S+\s+docker`, r`(sair|logout|nova sess[aã]o|newgrp|relogar|entrar de novo|login)`], [],
    `${sh("sudo usermod -aG docker ana")}\nDepois, saia e entre de novo (ou use newgrp docker) para valer.`],
  ["A rede não sobe no Ubuntu Server. Onde fica a configuração de rede e como a aplico?",
    [r`netplan`, r`netplan\s+(apply|try)`], [], `A configuração fica em /etc/netplan/*.yaml. Teste e aplique:\n${sh("sudo netplan try\nsudo netplan apply")}`],
  ["Como vejo o histórico do que o apt instalou ou removeu no Debian?",
    [r`/var/log/apt/history\.log|/var/log/dpkg\.log`], [], "Em /var/log/apt/history.log (e os .gz antigos), ou em /var/log/dpkg.log."],
  ["O comando sudo diz que meu usuário não está no arquivo sudoers. Como dou sudo a ele, tendo acesso root?",
    [r`usermod\s+-aG\s+sudo|adduser\s+\S+\s+sudo|visudo`], [], `Como root:\n${sh("usermod -aG sudo ana")}\nSe for editar o sudoers, use sempre visudo.`],
  ["Como faço um backup simples de uma pasta para um disco externo, copiando só o que mudou?",
    [r`rsync`], [], `${sh("rsync -a --delete /home/ana/ /media/ana/backup/home-ana/")}`],
];

/* ---- Container, WSL e VM (20) ------------------------------------------------------------------------------ */
const container: Def[] = [
  ["Num container Docker sem systemd, como faço um serviço continuar rodando depois que eu saio do shell?",
    [r`processo principal|PID 1|CMD|ENTRYPOINT|primeiro plano|foreground|-D\b`], [r`\bsystemctl\b`],
    "O container vive enquanto o processo principal (PID 1) vive. Rode o serviço em primeiro plano como o CMD da imagem, em vez de iniciá-lo com systemctl."],
  ["O WSL diz que a virtualização não está habilitada. O que verifico?",
    [r`BIOS|UEFI|VT-x|AMD-V|SVM|virtualiza[cç][aã]o`, r`Plataforma de M[aá]quina Virtual|VirtualMachinePlatform|wsl\s+--install`], [],
    "Ative VT-x ou AMD-V no BIOS/UEFI e o recurso 'Plataforma de Máquina Virtual' do Windows (ou rode `wsl --install`), e reinicie."],
  ["O WSL2 está ocupando muita memória no Windows. Como limito?",
    [r`\.wslconfig`, r`memory\s*=`], [], "Crie %UserProfile%\\.wslconfig com:\n[wsl2]\nmemory=4GB\nDepois rode `wsl --shutdown`."],
  ["Como desligo todas as distribuições do WSL de uma vez para aplicar uma configuração?",
    [r`wsl(\.exe)?\s+--shutdown`], [], `${ps("wsl --shutdown")}`],
  ["Dentro do WSL, onde ficam os arquivos do disco C: do Windows, e por que é lento compilar ali?",
    [r`/mnt/c`, r`9P|sistema de arquivos do Linux|~/|/home|lento|desempenho`], [],
    "Ficam em /mnt/c. O acesso passa pelo protocolo 9P, que é lento; para compilar, guarde o projeto no sistema de arquivos do Linux (em ~/)."],
  ["Como ativo o systemd numa distribuição do WSL?",
    [r`/etc/wsl\.conf`, r`systemd\s*=\s*true`], [], "Em /etc/wsl.conf, ponha:\n[boot]\nsystemd=true\nDepois rode `wsl --shutdown` no Windows."],
  ["O disco virtual do WSL2 (ext4.vhdx) não diminui depois que apago arquivos. Como recupero o espaço?",
    [r`Optimize-VHD|diskpart|compact\s+vdisk|sparse|--set-sparse`], [],
    "Apague os arquivos, rode `wsl --shutdown` e compacte o vhdx com `Optimize-VHD -Path ext4.vhdx -Mode Full` (Hyper-V) ou com o diskpart (compact vdisk)."],
  ["Um container sai logo depois de iniciar. Como vejo por quê?",
    [r`docker\s+logs`, r`docker\s+(ps\s+-a|inspect)`], [], `${sh("docker ps -a\ndocker logs meucontainer\ndocker inspect meucontainer --format '{{.State.ExitCode}}'")}`],
  ["Os dados do meu banco somem toda vez que recrio o container. O que está faltando?",
    [r`volume|-v\s|--mount|bind`], [], "Falta um volume: sem ele, os dados ficam na camada do container e somem com ele. Use `-v dados:/var/lib/postgresql/data` ou um volume no compose."],
  ["Como entro num container que está rodando para investigar?",
    [r`docker\s+exec\s+-(it|ti)`], [], `${sh("docker exec -it meucontainer sh")}`],
  ["O Docker diz 'permission denied' no /var/run/docker.sock. O que significa e como resolvo com segurança?",
    [r`grupo docker|usermod\s+-aG\s+docker`], [r`chmod\s+(666|777)\s+/var/run/docker\.sock`],
    "Seu usuário não está no grupo docker. Rode `sudo usermod -aG docker $USER` e entre de novo (o grupo docker equivale a root).",
    [`${sh("sudo chmod 666 /var/run/docker.sock")}`]],
  ["Como limito a memória e a CPU de um container com Docker?",
    [r`--memory|-m\s|mem_limit|deploy:`, r`--cpus|cpus:`], [], `${sh("docker run --memory 512m --cpus 1.5 minha-imagem")}`],
  ["Dentro de um container, o /proc/meminfo mostra a memória do host. Como vejo o limite real do container?",
    [r`/sys/fs/cgroup/memory\.max|memory\.limit_in_bytes|cgroup`], [], `${sh("cat /sys/fs/cgroup/memory.max")}\n(No cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes.)`],
  ["Como vejo quanto de CPU e de memória cada container está usando agora?",
    [r`docker\s+stats`], [], `${sh("docker stats --no-stream")}`],
  ["Uma VM no VirtualBox ou no Hyper-V ficou sem rede. O que verifico primeiro?",
    [r`NAT|bridge|ponte|Comutador Virtual|switch virtual|adaptador`], [],
    "Confira o modo do adaptador de rede da VM (NAT, ponte/bridge, rede interna) e, no Hyper-V, qual comutador virtual (switch virtual) está ligado a ela."],
  ["Como aumento o disco de uma VM Linux depois de aumentar o disco virtual no hipervisor?",
    [r`growpart|parted|fdisk|resize2fs|xfs_growfs|lvextend`], [],
    `Aumente a partição e depois o sistema de arquivos:\n${sh("sudo growpart /dev/sda 1\nsudo resize2fs /dev/sda1")}`],
  ["Como sei, de dentro do Linux, se estou numa VM ou num container?",
    [r`systemd-detect-virt|/\.dockerenv|/run/\.containerenv|hypervisor|dmidecode|/sys/class/dmi`], [], `${sh("systemd-detect-virt")}\nSem systemd, procure /.dockerenv ou /run/.containerenv.`],
  ["A imagem Docker do meu app está com 2 GB. Como reduzo o tamanho?",
    [r`multi-?stage|v[aá]rios est[aá]gios|alpine|slim|distroless|\.dockerignore`], [],
    "Use build em vários estágios (multi-stage), uma imagem base slim ou alpine e um .dockerignore para não copiar node_modules e afins."],
  ["Como copio um arquivo de dentro de um container para o host?",
    [r`docker\s+cp`], [], `${sh("docker cp meucontainer:/app/log.txt ./log.txt")}`],
  ["Como faço um container reiniciar sozinho se cair ou se o servidor reiniciar?",
    [r`--restart|restart:`, r`unless-stopped|always|on-failure`], [], `${sh("docker run -d --restart unless-stopped minha-imagem")}`],
];

/* ---- Arch/Omarchy (20) e macOS (10): ficam fora da taxa de decisão até as listas do dono (revisão 18) ------------ */
const arch: Def[] = [];
const macos: Def[] = [];

/* ---- montagem e conferência --------------------------------------------------------------------------------- */
const BLOCOS: Array<[Bloco, Familia, string, Def[]]> = [
  ["windows", "powershell", "win", windows], ["debian", "apt", "deb", debian], ["container", "apt", "ctn", container],
  ["arch", "pacman", "arc", arch], ["macos", "brew", "mac", macos],
];
/** A lista do Ubuntu vem de uma máquina de dev (WSL): ufw, smartctl e sshd são de servidor e não estão nela, então
 * esses itens não conferem existência. Os itens de WSL rodam comandos no Windows (wsl, Optimize-VHD): família
 * powershell. */
const FAMILIA_POR_ID: Record<string, Familia> = {
  "deb-010": "-", "deb-011": "-", "deb-014": "-", "deb-022": "-", "deb-023": "-",
  "ctn-002": "powershell", "ctn-003": "powershell", "ctn-004": "powershell", "ctn-007": "powershell",
};
/** Obrigatórios refeitos depois do ataque do revisor: nenhuma alternativa pode casar com a própria pergunta (D-2), e
 * os genéricos demais foram apertados (D-4). */
const OBRIG_POR_ID: Record<string, string[]> = {
  "win-001": [r`Limpeza de Disco|cleanmgr|Sensor de Armazenamento|Storage Sense|Sistema\s*>\s*Armazenamento`],
  "win-007": [r`Monitor de Recursos|resmon|perfmon`, r`Search|SysMain|Superfetch|antiv[ií]rus|Defender|Windows Update|HDD|disco (r[ií]gido|mec[aâ]nico)|SMART|chkdsk`],
  "win-027": [r`Gerenciamento de Disco|diskmgmt|Get-Disk|diskpart`, r`Alterar letra|NewDriveLetter|assign\s+letter`],
  "win-029": [r`Checkpoint-Computer|Prote[cç][aã]o do Sistema|SystemPropertiesProtection|rstrui`],
  "win-037": [r`powercfg(\.exe)?\s+/h(ibernate)?\s+off|HiberbootEnabled|fun[cç][aã]o dos bot[oõ]es|Ligar inicializa[cç][aã]o r[aá]pida`],
  "win-038": [r`ABNT2?|Set-WinUserLanguageList|Portugu[eê]s \(Brasil`],
  "deb-012": [r`ps\s+[^\n]*--sort|\btop\b|htop|smem`, r`\bfree\b|swapon\s+--show|/proc/swaps|vmstat`],
  "deb-013": [r`OOM|oom`, r`aument|limit|MemoryMax|cgroup|reduz|swap`],
  "deb-015": [r`fsck|e2fsck`, r`corromp|inconsist|superbloco|dmesg|journal`],
  "deb-019": [r`GRUB|grub`, r`Op[cç][oõ]es avan[cç]adas|Advanced options|Shift|\bEsc\b`],
  "ctn-002": [r`BIOS|UEFI|VT-x|AMD-V|SVM`, r`Plataforma de M[aá]quina Virtual|VirtualMachinePlatform|wsl\s+--install`],
  "ctn-005": [r`/mnt/c`, r`9P|Plan 9|sistema de arquivos do Linux|~/|/home|ext4`],
};
const itens: Item[] = [];
const refs: Record<string, Ref> = {};
for (const [bloco, familia, prefixo, defs] of BLOCOS) {
  defs.forEach(([pergunta, obrigatorios, proibidos, certa, erradas, fonte], i) => {
    const id = `${prefixo}-${String(i + 1).padStart(3, "0")}`;
    itens.push({ id, bloco, familia: FAMILIA_POR_ID[id] ?? familia, alvo: "resposta", pergunta, obrigatorios: OBRIG_POR_ID[id] ?? obrigatorios, proibidos, tokens: 160, fonte: fonte ?? DOC });
    refs[id] = { certa, ...(erradas ? { erradas } : {}) };
  });
}

const linhas = (f: string) => readFileSync(join(DIR, f), "utf8").split(/\r?\n/).filter((l) => l.trim() !== "");
const mudanca = linhas("lista-mudanca.tsv").map((l) => l.split("\t") as [string, string]);
const listas: Record<string, Set<string>> = {};
for (const fam of ["powershell", "pacman", "apt", "brew"]) {
  try { listas[fam] = new Set(linhas(`comandos-${fam}.txt`)); } catch { /* a lista ainda não chegou */ }
}
const regras: Regras = {
  destrutivos: mudanca.filter(([c]) => c === "destrutivo").map(([, x]) => new RegExp(x, "iu")),
  mudanca: mudanca.map(([categoria, x]) => ({ categoria, re: new RegExp(x, "iu") })),
  leituraPadrao: Object.fromEntries(linhas("leitura-padrao.tsv").map((l) => l.split("\t") as [string, string]).map(([c, x]) => [c, new RegExp(x, "iu")])),
  aviso: new RegExp(linhas("aviso.txt")[0]!, "iu"),
  listas,
  emPortugues: () => true,
};

let falhas = 0;
const vistos = new Set<string>();
for (const it of itens) {
  if (vistos.has(it.pergunta)) { console.error(`${it.id}: pergunta repetida`); falhas++; }
  vistos.add(it.pergunta);
  for (const x of [...it.obrigatorios, ...it.proibidos]) {
    try { new RegExp(x, "iu"); } catch (e) { console.error(`${it.id}: regex inválida ${x}: ${e}`); falhas++; }
  }
  const ref = refs[it.id]!;
  const n = avaliar(it, ref.certa, regras);
  if (!n.certa) { console.error(`${it.id}: a referência certa falha (${n.motivo}): ${JSON.stringify(ref.certa)}`); falhas++; }
  for (const e of ref.erradas ?? []) if (avaliar(it, e, regras).certa) { console.error(`${it.id}: a referência errada passa: ${JSON.stringify(e)}`); falhas++; }
  /* D-2: repetir a pergunta não pode cumprir obrigatório nenhum */
  for (const o of it.obrigatorios) if (new RegExp(o, "iu").test(it.pergunta)) { console.error(`${it.id}: o obrigatório ${o} casa com a própria pergunta`); falhas++; }
  if (avaliar(it, it.pergunta, regras).certa) { console.error(`${it.id}: repetir a pergunta passa`); falhas++; }
  /* uma resposta vazia de conteúdo nunca passa */
  if (avaliar(it, "Não sei.", regras).certa) { console.error(`${it.id}: "Não sei." passa`); falhas++; }
}
if (falhas) { console.error(`${falhas} falha(s): nada foi escrito`); process.exit(1); }

writeFileSync(join(DIR, "duvidas.json"), `${JSON.stringify({ versao: 2, itens }, null, 2)}\n`);
writeFileSync(join(DIR, "referencias-duvidas.json"), `${JSON.stringify(refs, null, 2)}\n`);
const sha = (f: string) => createHash("sha256").update(readFileSync(join(DIR, f))).digest("hex");
const cobertos = ["duvidas.json", "nota-duvidas.ts", "nota.ts", "lista-mudanca.tsv", "leitura-padrao.tsv", "aviso.txt", "comandos-apt.txt", "comandos-powershell.txt"];
const manifest = cobertos.map((f) => `${sha(f)}  ${f}`).join("\n");
writeFileSync(join(DIR, "MANIFEST-duvidas.sha256"), `${manifest}\n`);
const porBloco = itens.reduce<Record<string, number>>((m, it) => ({ ...m, [it.bloco]: (m[it.bloco] ?? 0) + 1 }), {});
console.log(`${itens.length} itens ${JSON.stringify(porBloco)}; referências conferidas`);
console.log(manifest);
