# Plano: modelo local para inventário defensivo dos próprios equipamentos

Objetivo: um executável `smolmoe` que, numa máquina sua, lê o estado dela (usuários, portas, serviços,
inicialização, permissões, atualizações), explica o que encontrou e grava um relatório comparável com o anterior.
Uso pessoal, offline, só leitura.

## Estado e revisão (franca)

- **Feito:** o inventário e a comparação (fases 2 e 3) existem como `smolmoe-inventario`, um binário separado,
  sem modelo. Ver [INVENTARIO.md](INVENTARIO.md). Ele lê arquivos em vez de chamar `systemctl`/`ss`, então não
  executa nenhum programa.
- **Rebaixado:** o modelo (fases 1 e 4). Um modelo de 0,5B a 1,5B não acrescenta detecção: o que acha intrusão é
  o diff determinístico (porta nova, chave SSH nova, cron novo, SUID trocado). O modelo só reformularia em prosa o
  que o diff já diz, e erraria às vezes. Só vale ligá-lo ao chat quando houver uma pergunta que o diff não responde
  (ex.: "o que este script de cron novo faz?"), e aí o valor está em ler o arquivo, não em listar o sistema.
- O restante deste documento é o plano original, mantido como referência.

## 1. Princípio de segurança

A proteção não fica no modelo, fica no runtime:

1. **O modelo nunca escreve comando que é executado.** Ele só escolhe o nome de uma sonda de um catálogo fixo
   (um `enum`). O runtime em C roda o comando correspondente com argumentos fixos, sem shell no meio.
2. **Só leitura.** Nenhuma sonda altera arquivo, serviço, firewall ou registro. Corrigir algo continua sendo ação
   sua, digitada com `!comando` (que já existe e nunca passa pelo modelo).
3. **Só a máquina local.** Nada de varredura ativa de rede. A única visão da rede é a tabela ARP/vizinhos que o
   sistema já tem.
4. **Saída do modelo é dado não confiável**, como já é hoje em `src/tools.c`: chamada mal formada vira
   `{"erro": ...}`, nunca execução.

Consequência sobre "sem censura": para esse uso, a recusa do modelo não é a barreira de segurança, e um modelo
instruct comum não recusa ler o próprio sistema. Derivados de Qwen2.5 com a mesma arquitetura (inclusive versões com
o alinhamento removido) passariam pelo mesmo conversor, mas o plano não depende disso: a escolha recomendada abaixo é
o checkpoint oficial, com revisão e SHA-256 fixos, porque pesos de origem desconhecida são um risco de cadeia de
suprimento maior do que qualquer recusa.

## 2. Escolha dos pesos

| Opção | Tamanho Q8 | RAM | Papel |
|---|---:|---:|---|
| **Qwen2.5-Coder-1.5B-Instruct** (recomendado) | ~1,7 GB | ~1,8 GB | Lê saída de comando, explica scripts e configs |
| Qwen2.5-0.5B-Instruct (já suportado) | 556 MB | ~0,6 GB | Máquinas fracas; respostas mais curtas |
| Phi-4-mini-instruct (já suportado) | 4,3 GB | ~4,4 GB | Mais qualidade, ~4 tok/s |

Por que o Coder 1.5B: mesma arquitetura `qwen2` que `scripts/hf/qwen2.ts` já converte (sem janela deslizante, sem
`rope_scaling`, `silu`, embeddings amarrados), mesmo template ChatML com `<tool_call>`, e treino focado em código e
terminal. Licença Apache-2.0.

## 3. Encaixe dos pesos no runtime

1. `scripts/convert.ts`: comando `fetch-qwen2-coder` com revisão fixa do Hugging Face e SHA-256 de cada arquivo,
   seguindo `fetch-qwen2`. Conversão: `bun scripts/convert.ts qwen2 downloads/qwen2.5-coder-1.5b models/qwen2.5-coder-1.5b-instruct`.
2. `bun scripts/convert.ts check` no resultado; gravar `provenance.json` e `LICENSE-Qwen.txt` no diretório.
3. `models/manifest.json` e `docs/MODELS.md`: entrada nova com hashes de saída (a conversão é determinística).
4. Makefile: apelido `defesa` (CLI) e `defesa-gui`, como `qwen`/`gemma`. Apelido no runtime: `smolmoe defesa`.
5. Medir com `bun scripts/bench.ts` e perplexidade no texto PT-BR padrão; registrar em `DECISIONS.md`.

Critério de aceite: o modelo abre, responde em PT-BR, e a suíte de paridade numérica passa como nos outros.

## 4. Sondas (ferramenta `inspecionar`)

Uma ferramenta nova, `inspecionar`, com um único argumento `sonda` restrito a esta lista:

| Sonda | Linux | Windows | macOS |
|---|---|---|---|
| `sistema` | `/etc/os-release`, `uname -r` | `Get-CimInstance Win32_OperatingSystem` | `sw_vers` |
| `usuarios` | `/etc/passwd` (shell de login), grupo `sudo`/`wheel` | `Get-LocalUser`, admins locais | `dscl . list /Users` |
| `portas` | `/proc/net/tcp*`, `/proc/net/udp*` | `Get-NetTCPConnection -State Listen` | `lsof -nP -iTCP -sTCP:LISTEN` |
| `servicos` | `systemctl list-units --type=service --state=running` | `Get-Service` (rodando) | `launchctl list` |
| `inicializacao` | units habilitadas, `cron`, `~/.config/autostart` | `Run`/`RunOnce`, tarefas agendadas | `LaunchAgents`, `LaunchDaemons` |
| `permissoes` | SUID/SGID em `/usr`, graváveis por todos em `/etc` | ACLs de `Program Files` com escrita para `Users` | SUID em `/usr` |
| `atualizacoes` | gerenciador de pacotes, modo só listagem | `Get-HotFix` | `softwareupdate -l` |
| `firewall` | `nft list ruleset` ou `ufw status` (leitura) | `Get-NetFirewallProfile` | `socketfilterfw --getglobalstate` |
| `vizinhos` | `/proc/net/arp` | `Get-NetNeighbor` | `arp -an` |

Implementação:

- `src/defense.c` / `defense.h`: tabela `{sonda, sistema, argv fixo}`. Onde o dado está em arquivo (`/proc`,
  `/etc`), o C lê direto, sem processo filho. Onde precisa de processo, reaproveita `src/process.c`, com `argv`
  montado pelo runtime, sem interpolação de texto do modelo.
- **Resumo em C antes do modelo.** `TOOL_RESULT_MAX` é 384 bytes e o contexto padrão é 2048 tokens; a saída bruta
  não cabe. O C transforma cada sonda em fatos compactos (ex.: `{"portas":[{"porta":22,"processo":"sshd","escuta":"0.0.0.0"}],"total":3}`)
  e guarda a saída completa só no relatório.
- `tools_relevant()` / `tools_forced_choice()`: reconhecer pedidos como "quais portas estão abertas" e abrir a chamada
  quando o modelo não abrir, como já acontece com `calcular`.
- Sondas que exigem privilégio retornam `{"erro":"precisa de administrador"}`; o runtime nunca eleva sozinho.
- Limite de tempo e de bytes por sonda; processo filho morto ao estourar.

## 5. Artefatos

- `smolmoe defesa --relatorio DIR`: roda todas as sondas sem o modelo e grava `DIR/AAAA-MM-DD-host.json`
  (fatos estruturados) e `.md` (legível), cada um com SHA-256 no fim. Funciona sem pesos carregados.
- `--comparar ANTERIOR.json`: diferença entre dois relatórios (porta nova, usuário novo, item de inicialização novo,
  binário SUID novo). Isso é o núcleo defensivo: mudança inesperada é o sinal que importa.
- No chat, o modelo recebe o relatório resumido ou a diferença e explica cada item, com a ressalva de que é leitura
  automática e deve ser conferida.

## 6. Fases

| Fase | Entrega | Aceite |
|---|---|---|
| 1 | Pesos do Coder 1.5B convertidos e apelido `defesa` | `convert.ts check` ok; bench e perplexidade registrados |
| 2 | `defense.c` com `sistema`, `usuarios`, `portas` no Linux | Testes em C com `/proc` falso (fixtures em `tests/`); nenhuma sonda aceita argumento livre |
| 3 | `--relatorio` e `--comparar` | Relatório idêntico em duas execuções seguidas sem mudança no sistema |
| 4 | Ferramenta `inspecionar` no chat | `scripts/eval-tools.ts` com perguntas novas, escritas antes da implementação |
| 5 | Demais sondas e Windows/macOS | Mesmos testes por plataforma no CI |

## 7. Fora do escopo

- Varredura de outros hosts, exploração, coleta de credenciais ou qualquer ação ofensiva.
- Correção automática: o modelo sugere, você executa com `!comando`.
- Acesso à internet em tempo de execução.
