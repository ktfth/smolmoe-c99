# smolmoe-inventario

Fotografa, só lendo arquivos, o que importa para a defesa de uma máquina Linux, e mostra o que mudou desde a última
fotografia. Não usa modelo, não executa outros programas, não acessa a rede e não altera nada.

```sh
make inventario                                        # o programa e os testes dele (fora de `make`)
sudo build/smolmoe-inventario --saida base.txt         # fotografia de referência
sudo build/smolmoe-inventario --saida hoje.txt
build/smolmoe-inventario --comparar base.txt hoje.txt  # 0: nada mudou, 1: mudou, 2: erro
```

O relatório é gravado com permissão `0600` (é a visão de root sobre a máquina), por um arquivo temporário exclusivo
renomeado no fim.

Saída de `--comparar`:

```text
+	porta	tcp 0.0.0.0:4444	processo=nc uid=1000 exposta=sim
+	inicio	/etc/cron.d/miner	sha256=3f1c...
~	config	/etc/ssh/sshd_config	modo=0644 dono=0 sha256=aaaa... -> modo=0644 dono=0 sha256=bbbb...
+	nota	sem acesso: /etc/sudoers.d	
-	usuario	bob	uid=1001 shell=/bin/bash
```

## O que entra no relatório

| Categoria | Fonte | Por quê |
|---|---|---|
| `porta` | `/proc/net/{tcp,tcp6,udp,udp6}` + `/proc/*/fd` | Serviço novo escutando; `exposta=sim` quando não é só loopback |
| `usuario` | `/etc/passwd` | Contas com shell de login ou uid 0 |
| `admin` | `/etc/group` + gid de `/etc/passwd` | Membros de `root`, `sudo`, `wheel`, `admin`, `docker`, `lxd`, `libvirt`, `disk` (todos equivalem a root ou leem discos), listados no grupo ou com ele como grupo primário |
| `suid` | `/usr`, `/bin`, `/sbin`, `/opt`, `/etc` | Binários SUID/SGID com modo, dono e SHA-256 completo: um novo, ou um trocado, é sinal forte |
| `gravavel` | as mesmas pastas | Arquivos e pastas graváveis por todos (sem sticky bit) |
| `inicio` | cron, systemd (`/etc` e `~/.config`, `~/.local/share`), `rc.local`, `init.d`, `rc?.d`, `profile.d`, autostart, `.bashrc`/`.bash_profile`/`.bash_login`/`.profile`/`.zshrc` dos usuários, `ld.so.preload` | Onde persistência costuma ser plantada; arquivos com modo, dono e hash; links com o destino e o hash do arquivo de destino (um serviço habilitado por link muda no destino) |
| `config` | `sshd_config`, `sudoers`, `pam.d`, `hosts`, `ld.so.conf`, `nsswitch.conf` | Mudança em configuração de segurança |
| `chave_ssh` | `~/.ssh/authorized_keys` de cada usuário | Quem entra sem senha e com que opções (`command=`, `from=` por extenso); chave nova é o primeiro lugar a olhar |
| `alerta` | — | Outro usuário com uid 0, conta sem senha (`/etc/shadow`), `ld.so.preload` presente; `conta_anormal` (conta que a glibc aceita mas que está escrita fora do padrão ou não cabe no relatório), `grupo_anormal` (grupo de admin com outro nome ou outro gid, ou cuja linha não pôde ser lida), `shadow_anormal` e `chave_anormal` (linha do shadow ou de `authorized_keys` que não pôde ser lida), `nao_regular` (FIFO, socket, dispositivo ou pasta no lugar de um arquivo esperado) |

Linhas começando com `#` (host, data) não entram na comparação, com uma exceção: as **notas** (`# nota:`, caminhos
sem acesso, sistemas de arquivos montados dentro das pastas verificadas, erros de leitura) entram, como itens
`nota`. Perder cobertura é uma mudança, e o `--comparar` também avisa quando o relatório novo tem notas.

## Uso diário com cron

```sh
# /etc/cron.d/inventario  (como root)
15 6 * * * root /usr/local/bin/smolmoe-inventario --saida /var/lib/inventario/hoje.txt && /usr/local/bin/smolmoe-inventario --comparar /var/lib/inventario/base.txt /var/lib/inventario/hoje.txt > /var/lib/inventario/mudancas.txt
```

Quando uma mudança for sua (atualização de pacote, porta nova de propósito), confira e promova: `cp hoje.txt base.txt`.

## Disco de outra máquina

`--raiz /mnt/disco` lê um sistema montado, por exemplo o disco de uma máquina suspeita, sem iniciá-la. Portas
ficam de fora (não há `/proc`), e o relatório registra isso como nota. Links simbólicos da imagem que levam para
fora dela (`/home -> /home`, `/etc/passwd -> /etc/passwd`) não são seguidos: viram nota `fora da raiz`, para o
relatório não descrever a máquina que faz a análise. Uma raiz inexistente ou sem `/etc/passwd` é erro (código 2) e
nenhum relatório é gravado.

## O que o relatório garante

- **Nenhuma conta que a glibc aceita some** (nos arquivos locais; veja os limites). `/etc/passwd`, `/etc/shadow` e
  `/etc/group` são lidos como o `nss_files` da glibc lê:
  - os brancos antes da linha são pulados, quantos forem, e comentários também;
  - campos que faltam ficam vazios, e a linha termina no primeiro byte nulo;
  - o `\r` final fica no último campo;
  - o uid e o gid passam pelo `strtou32` (`-0`, `+0`, ` 0` e `000` são 0; `-1` é 4294967295);
  - o shadow segue o parser da glibc (de 5 a 9 campos, conferido campo a campo com a `fgetspent`);
  - até um defeito da glibc 2.30+ é reproduzido: uma linha recuada que termina sem `\n` (a última) ou que tem byte
    nulo é lida com os últimos bytes repetidos, como o PAM a lê.

  Conta escrita fora do padrão, com campo grande demais, em linha de mais de 8 KB ou sem nome vira alerta
  `conta_anormal` com o nome (cortado e marcado se preciso), o uid e o número da linha. Linha que a glibc recusa vira
  nota com o número da linha, uma por linha, para duas nunca virarem uma só no `--comparar`.
- **Grupo de admin é pelo gid, não só pelo nome.** O `initgroups` junta todo gid em que o usuário aparece, e o sudo
  casa `%sudo` pelo gid do `getgrnam` e pelo nome do `getgrgid`. Por isso conta como grupo de admin toda linha com o
  nome de um, ou com o gid de uma linha que tem esse nome. Outro nome com esse gid, ou outro gid com esse nome, vira
  alerta `grupo_anormal`, e os membros aparecem como `admin` do grupo canônico. Os membros são lidos como o
  `initgroups` lê, porque é ele que dá o privilégio ao login, ao sshd e ao sudo:
  - linha inteira, sem a cauda repetida;
  - brancos antes de cada membro pulados;
  - linhas comentadas também: `#x:x:27:ana` dá o sudo à ana, embora o `getent` não mostre a linha, e vira alerta.
- **Nada inventado.** Uma linha longa nunca é partida: o resto dela não vira outra conta. Linha de `authorized_keys`
  ou de grupo de admin que não dá para ler inteira vira alerta; opções entre aspas em `authorized_keys` são lidas
  como o sshd lê; portas em `/proc/net` precisam ter exatamente 4 dígitos hex.
- **Não trava e não abre o que não é arquivo.** Abrir já é um ato: o `/dev/watchdog` arma na abertura, uma fita
  rebobina. Por isso o tipo é conferido antes de abrir e de novo no descritor aberto (o mesmo arquivo).
  - FIFO, socket, dispositivo (um link para `/dev/zero`) ou pasta no lugar de um arquivo esperado vira alerta
    `nao_regular`, sem ser aberto. O sshd também recusa esses tipos.
  - O hash abre com `O_NOFOLLOW`.
  - Na máquina viva, rodando como root, os arquivos das homes (`authorized_keys`, `.bashrc` etc.) são abertos como o
    dono (grupos, egid e euid), como o sshd faz. Um link trocado depois da conferência não faz o root abrir o que só
    o root pode. Se a troca não pega, o arquivo não é aberto.
  - Cada arquivo de texto é lido até 64 MiB e cada hash cobre até 256 MiB: um arquivo esparso de 1 TB no `$HOME` de
    um usuário vira nota, em vez de prender a execução do root.
  - Com `--raiz`, o arquivo aberto é o caminho que o `realpath` confirmou dentro da imagem.
  - O `--comparar` aceita arquivo comum ou pipe (`--comparar base.txt <(ssh host smolmoe-inventario)`).
- **Linhas limitadas.** Nenhum item passa de 8191 bytes: campos enormes são cortados sem partir caractere UTF-8 e
  marcados com ` [cortado]`; muitos detalhes na mesma chave terminam em `, ...`. Todo relatório que o programa
  escreve, o `--comparar` lê.
- **Relatório danificado é recusado**, nunca lido pela metade: linha sem dois tabs, com byte de controle ou longa
  demais, ou um número de itens diferente do declarado no cabeçalho (arquivo cortado).

Como é testado (`tests/test_inventory.c`, roda em `make inventario` e sob ASan/UBSan no CI, pelo `make sanitize`):
sistemas aleatórios com o resultado esperado calculado à parte no teste (oráculo: nem item a mais nem a menos);
fuzz dos arquivos do sistema (passwd, group, shadow, `/proc/net`, `authorized_keys`) com checagem de que cada item
aponta para algo que está na entrada e, onde há glibc, de que toda conta com uid 0 que a própria `fgetpwent` lê e
toda senha vazia que a `fgetspent` lê aparecem no relatório; FIFOs (sem escritor e com um escritor de bytes nulos sem fim) e arquivos esparsos de 4 GiB,
coletados num processo filho com 5 s de limite; fuzz do leitor de relatórios e do diff (aplicar o diff ao relatório
antigo precisa dar o novo). `SM_FUZZ_ITERS=20000 build/test-inventory` roda uma bateria maior.

## Limites

- **Como usuário comum** não lê `/etc/shadow`, crontabs de outros usuários nem o dono de sockets de outros processos;
  esses caminhos aparecem como notas. Rode como root.
- **Não detecta rootkit de kernel:** um kernel comprometido pode mentir em `/proc` e no sistema de arquivos. Para isso,
  rode `--raiz` a partir de um sistema limpo (pendrive) contra o disco montado.
- **Binário comum trocado passa batido.** Só os SUID/SGID têm hash: um `sshd`, `pam_unix.so` ou libc substituído
  (o trojan de userland clássico) não aparece no `--comparar`. Também ficam de fora `modprobe.d`, `udev/rules.d`,
  `update-motd.d`, `apt.conf.d`, `NetworkManager/dispatcher.d` e `~/.ssh/rc`. Para verificar pacotes, use o
  gerenciador da distribuição (`debsums`, `rpm -Va`) a partir de um sistema limpo.
- **A leitura reproduz a glibc 2.30 em diante** (conferida na 2.39). Em musl (Alpine) ou numa glibc mais antiga, a
  própria máquina pode ler uma linha fora do padrão de outro jeito: a cauda repetida de uma linha recuada, por
  exemplo, só existe na glibc 2.30+. Uma conta assim pode sair como nota em vez de alerta, ou o contrário.
- **Só os arquivos locais.** Contas vindas de outro módulo NSS (LDAP, SSSD, um `libnss_*.so` plantado) não aparecem;
  o `/etc/nsswitch.conf` é acompanhado em `config`, então uma mudança nele aparece no `--comparar`. Linhas `+`/`-`
  do nss_compat em `/etc/passwd` são lidas como contas comuns; o que elas incluem do NIS/LDAP não é consultado.
- **Atualizações de pacote mudam hashes** (`~` em `suid`, `inicio`, `config`). Isso é esperado: confira e promova a base.
- **Portas UDP efêmeras** de alguns programas (clientes mDNS, navegadores) podem aparecer e sumir entre execuções.
- O **número** de itens não tem teto: uma máquina com milhares de binários SUID gera um relatório grande. Cada linha é
  limitada; o arquivo inteiro, não.
- **Guarde a base fora da máquina** (ou assinada): quem tem root pode reescrever `base.txt`.
- Coleta só em Linux. Em Windows o binário só faz `--comparar`; em macOS lê o que existe e deixa o resto de fora.
