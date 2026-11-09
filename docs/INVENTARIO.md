# smolmoe-inventario

Fotografa, só lendo arquivos, o que importa para a defesa de uma máquina Linux, e mostra o que mudou desde a última
fotografia. Não usa modelo, não executa outros programas, não acessa a rede e não altera nada.

```sh
make build/smolmoe-inventario                          # ou só `make`
sudo build/smolmoe-inventario --saida base.txt         # fotografia de referência
sudo build/smolmoe-inventario --saida hoje.txt
build/smolmoe-inventario --comparar base.txt hoje.txt  # 0: nada mudou, 1: mudou, 2: erro
```

Saída de `--comparar`:

```text
+	porta	tcp 0.0.0.0:4444	processo=nc uid=1000 exposta=sim
+	inicio	/etc/cron.d/miner	sha256=3f1c...
~	config	/etc/ssh/sshd_config	sha256=aaaa... -> sha256=bbbb...
-	usuario	bob	uid=1001 shell=/bin/bash
```

## O que entra no relatório

| Categoria | Fonte | Por quê |
|---|---|---|
| `porta` | `/proc/net/{tcp,tcp6,udp,udp6}` + `/proc/*/fd` | Serviço novo escutando; `exposta=sim` quando não é só loopback |
| `usuario` | `/etc/passwd` | Contas com shell de login ou uid 0 |
| `admin` | `/etc/group` | Membros de `root`, `sudo`, `wheel`, `admin`, `docker`, `lxd`, `libvirt`, `disk` (todos equivalem a root ou leem discos) |
| `suid` | `/usr`, `/bin`, `/sbin`, `/opt`, `/etc` | Binários SUID/SGID com hash: um novo, ou um trocado, é sinal forte |
| `gravavel` | as mesmas pastas | Arquivos e pastas graváveis por todos (sem sticky bit) |
| `inicio` | cron, systemd (`/etc`), `rc.local`, `init.d`, `profile.d`, autostart, `.bashrc`/`.profile`/`.zshrc` dos usuários, `ld.so.preload` | Onde persistência costuma ser plantada; arquivos com hash, links com destino |
| `config` | `sshd_config`, `sudoers`, `pam.d`, `hosts`, `ld.so.conf` | Mudança em configuração de segurança |
| `chave_ssh` | `~/.ssh/authorized_keys` de cada usuário | Quem entra sem senha; chave nova é o primeiro lugar a olhar |
| `alerta` | — | Outro usuário com uid 0, conta sem senha (`/etc/shadow`), `ld.so.preload` presente |

Linhas começando com `#` (host, data, caminhos sem acesso) não entram na comparação.

## Uso diário com cron

```sh
# /etc/cron.d/inventario  (como root)
15 6 * * * root /usr/local/bin/smolmoe-inventario --saida /var/lib/inventario/hoje.txt && /usr/local/bin/smolmoe-inventario --comparar /var/lib/inventario/base.txt /var/lib/inventario/hoje.txt > /var/lib/inventario/mudancas.txt
```

Quando uma mudança for sua (atualização de pacote, porta nova de propósito), confira e promova: `cp hoje.txt base.txt`.

## Disco de outra máquina

`--raiz /mnt/disco` lê um sistema montado, por exemplo o disco de uma máquina suspeita, sem iniciá-la. Portas
ficam de fora (não há `/proc`), e o relatório registra isso como nota.

## Limites

- **Como usuário comum** não lê `/etc/shadow`, crontabs de outros usuários nem o dono de sockets de outros processos;
  esses caminhos aparecem como notas. Rode como root.
- **Não detecta rootkit de kernel:** um kernel comprometido pode mentir em `/proc` e no sistema de arquivos. Para isso,
  rode `--raiz` a partir de um sistema limpo (pendrive) contra o disco montado.
- **Atualizações de pacote mudam hashes** (`~` em `suid`, `inicio`, `config`). Isso é esperado: confira e promova a base.
- **Portas UDP efêmeras** de alguns programas (clientes mDNS, navegadores) podem aparecer e sumir entre execuções.
- **Guarde a base fora da máquina** (ou assinada): quem tem root pode reescrever `base.txt`.
- Coleta só em Linux. Em Windows o binário só faz `--comparar`; em macOS lê o que existe e deixa o resto de fora.
