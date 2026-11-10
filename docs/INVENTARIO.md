# smolmoe-inventario

Fotografa, só lendo arquivos, o que importa para a defesa de uma máquina Linux, e mostra o que mudou desde a última
fotografia. Não usa modelo, não executa outros programas, não acessa a rede e não altera nada.

```sh
make build/smolmoe-inventario                          # ou só `make`
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
| `config` | `sshd_config`, `sudoers`, `pam.d`, `hosts`, `ld.so.conf` | Mudança em configuração de segurança |
| `chave_ssh` | `~/.ssh/authorized_keys` de cada usuário | Quem entra sem senha e com que opções (`command=`, `from=` por extenso); chave nova é o primeiro lugar a olhar |
| `alerta` | — | Outro usuário com uid 0, conta sem senha (`/etc/shadow`), `ld.so.preload` presente |

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

- **Nada inventado.** Linhas de arquivo do sistema longas demais (mais de 8 KB) ou com byte nulo são descartadas
  inteiras, nunca partidas; contas com uid vazio ou não numérico, ou campos maiores que o suportado, são ignoradas
  com nota, em vez de truncadas ou lidas como uid 0; opções entre aspas em `authorized_keys` são lidas como o sshd
  lê; portas em `/proc/net` precisam ter exatamente 4 dígitos hex.
- **Linhas limitadas.** Nenhum item passa de 8191 bytes: campos enormes são cortados sem partir caractere UTF-8 e
  marcados com ` [cortado]`; muitos detalhes na mesma chave terminam em `, ...`. Todo relatório que o programa
  escreve, o `--comparar` lê.
- **Relatório danificado é recusado**, nunca lido pela metade: linha sem dois tabs, com byte de controle ou longa
  demais, ou um número de itens diferente do declarado no cabeçalho (arquivo cortado).

Como é testado (`tests/test_inventory.c`, roda em `make test-unit` e sob ASan/UBSan no CI):
sistemas aleatórios com o resultado esperado calculado à parte no teste (oráculo: nem item a mais nem a menos);
fuzz dos arquivos do sistema (passwd, group, shadow, `/proc/net`, `authorized_keys`) com checagem de que cada item
aponta para algo que está na entrada; fuzz do leitor de relatórios e do diff (aplicar o diff ao relatório antigo
precisa dar o novo). `SM_FUZZ_ITERS=20000 build/test-inventory` roda uma bateria maior.

## Limites

- **Como usuário comum** não lê `/etc/shadow`, crontabs de outros usuários nem o dono de sockets de outros processos;
  esses caminhos aparecem como notas. Rode como root.
- **Não detecta rootkit de kernel:** um kernel comprometido pode mentir em `/proc` e no sistema de arquivos. Para isso,
  rode `--raiz` a partir de um sistema limpo (pendrive) contra o disco montado.
- **Atualizações de pacote mudam hashes** (`~` em `suid`, `inicio`, `config`). Isso é esperado: confira e promova a base.
- **Portas UDP efêmeras** de alguns programas (clientes mDNS, navegadores) podem aparecer e sumir entre execuções.
- O **número** de itens não tem teto: uma máquina com milhares de binários SUID gera um relatório grande. Cada linha é
  limitada; o arquivo inteiro, não.
- **Guarde a base fora da máquina** (ou assinada): quem tem root pode reescrever `base.txt`.
- Coleta só em Linux. Em Windows o binário só faz `--comparar`; em macOS lê o que existe e deixa o resto de fora.
