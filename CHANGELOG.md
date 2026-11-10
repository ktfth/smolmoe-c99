# Changelog

## Não lançado

### Adicionado
- **`smolmoe-inventario`:** inventário defensivo só leitura de máquinas Linux (portas em escuta com processo,
  usuários com login, grupos administrativos, binários SUID/SGID com hash, caminhos graváveis por todos, pontos de
  persistência, chaves SSH autorizadas, configurações de segurança e alertas) e `--comparar` entre dois relatórios,
  com código de saída para uso em cron. Lê só arquivos; não executa programas nem usa a rede. `--raiz` lê um disco
  montado. Detalhes em `docs/INVENTARIO.md`.
- Testes por fuzzing e por oráculo do inventário, e fuzzing das chamadas de ferramenta escritas pelo modelo no chat
  (resultado sempre limitado e em JSON; nenhuma chamada aceita para perguntas sem número, data, conversão ou contagem).
  `SM_FUZZ_ITERS` aumenta as baterias.

### Corrigido
- Inventário: o teste unitário não compilava no Windows (funções só usadas no trecho POSIX).
- Inventário: itens que não existem não aparecem mais. Linhas longas de `/etc/passwd`, uid vazio ou não numérico,
  opções entre aspas em `authorized_keys` e portas fora do formato geravam contas, alertas de uid 0, chaves ou portas
  falsas.
- Inventário, a partir do review do PR: relatório gravado com `0600` por arquivo temporário exclusivo; `--raiz` não
  segue links que saem da imagem; raiz inexistente ou sem `/etc/passwd` é erro; membros de grupo administrativo por
  grupo primário; SHA-256 completo; modo e dono dos arquivos acompanhados; hash do destino dos links (serviços
  habilitados); opções das chaves SSH por extenso; dono de socket com todos os processos, em ordem fixa; portas
  lidas antes dos donos; endereços corretos em máquinas big-endian; mais pontos de persistência (`rc?.d`,
  `.bash_login`, `~/.local/share/systemd/user`); notas de cobertura comparadas pelo `--comparar`; montagens internas
  e erros de leitura registrados como nota; arquivo ilegível vira nota em vez de valor `?`; relatório cortado ou
  erro de escrita na comparação dão erro.
- Inventário: itens grandes demais (link com destino longo, centenas de sockets na mesma porta) tornavam o relatório
  ilegível para o próprio `--comparar`; agora são cortados e marcados. Detalhes repetidos na mesma chave aparecem uma
  vez só.
- Inventário: contas que a glibc trata como root sumiam do relatório. Uid `-0`, `+0`, ` 0` ou `00000000000`, linha
  sem os últimos campos, espaço antes do nome, byte nulo depois de uma conta válida, nome, home ou shell grandes e
  linha de mais de 8 KB descartavam a conta. O `/etc/passwd` agora é lido como o `nss_files` da glibc lê, e a conta
  estranha vira alerta `conta_anormal` com nome, uid e linha. Um teste antigo exigia esconder `neg:x:-0:0`; ele foi
  invertido. Um comentário depois de espaços (`  #x:x:0:0...`) não vira mais uma conta.
- Inventário: o `--comparar` não acusava uma segunda linha ruim no `/etc/passwd`, porque todas recebiam a mesma nota e
  notas iguais viram uma só. Cada nota agora leva o número da linha.
- Inventário: um usuário comum travava a execução do root. Um FIFO no lugar de `authorized_keys` bloqueava na
  abertura, um FIFO com bytes nulos ou um link para `/dev/zero` faziam a leitura nunca acabar, e um arquivo esparso de
  terabytes era lido ou passado no hash inteiro. Só arquivo comum é lido, conferido no descritor aberto, com limite de
  bytes por arquivo (64 MiB de texto, 256 MiB de hash). O resto vira alerta `nao_regular` ou nota. Corrigidos por
  revisão, sem teste que falhe antes (reproduzir exigiria um ponto de injeção no código): a troca do arquivo por um
  FIFO entre o `lstat` e a abertura no hash, o link para `/dev/zero` na máquina viva e a leitura com o euid do dono
  (os testes não rodam como root).
- Inventário: linha longa de `authorized_keys` ou de grupo de admin escondia a chave ou o membro com uma nota
  genérica; agora é alerta. Falha de abertura que não seja "não existe" vira nota com o motivo.
- Inventário: milhares de brancos antes de uma linha escondiam uma conta root no `/etc/passwd` e uma senha vazia no
  `/etc/shadow`. Os brancos agora são pulados como a glibc pula, sem ocupar a linha. O shadow passa a seguir o
  parser da glibc: campos até `sp_max` e nome maior que a linha como alerta `shadow_anormal`. O `\r` final fica no
  último campo, e a última linha recuada e sem `\n` é lida como a glibc 2.30+ a lê (com os últimos bytes
  repetidos).
- Inventário: pertencer a um grupo de admin por gid passava batido (`sudo2:x:27:ana`, ` sudo:x:27:bia`) e já existia
  antes deste trabalho. Agora vale o gid, como para o `initgroups` e o sudo, e um nome ou gid diferente do canônico
  vira alerta `grupo_anormal`.
- Inventário: o tipo do arquivo é conferido antes de abrir, e não só depois, porque abrir um dispositivo já tem
  efeito (`/dev/watchdog` arma na abertura). O hash abre com `O_NOFOLLOW`. Na máquina viva, os arquivos das homes
  são abertos como o dono (grupos, egid e euid), como o sshd faz, e não são abertos se a troca não pegar.
- Inventário: os membros de grupo de admin agora são lidos como o `initgroups` lê, que é quem dá o privilégio.
  Uma linha comentada (`#x:x:27:ana`) dá o sudo e não aparecia; um membro depois de `, ` aparecia com o espaço; e a
  última linha recuada aparecia com o membro trocado (`anana`). O shadow segue o parser da glibc campo a campo, e uma
  linha com 10 campos não inventa mais `senha_vazia`. A cauda repetida da glibc vale também para linhas recuadas com
  byte nulo. Um membro com branco ou `\r` no fim (`ana `), que não é a conta `ana`, vem marcado no detalhe.
- Inventário: o `--comparar` aceita um pipe como relatório (`<(ssh host ...)`). O `/etc/nsswitch.conf` entrou na
  lista de `config`.
- Inventário: na musl, uma linha `#evil:x:0:0::/root:/bin/sh` do `/etc/passwd` é conta uid 0, que o `getpwnam`
  acha, e o inventário a descartava em silêncio como comentário. Agora ela deixa uma nota com o número da linha e o
  uid que a musl lê (módulo 2^32), também quando a linha passa de 8 KB. Uma linha com uid vazio
  (`vazio:x::0:...`), que a glibc recusa e a musl lê como uid 0, também diz isso na nota.
- Inventário: um uid, gid ou número do shadow acima de 4294967295 (ou `-1`) era lido como 4294967295, e a linha
  virava conta. A glibc 2.39 recusa essas linhas (conferido com `fgetpwent`, `fgetgrent` e `fgetspent`), e a musl lê
  `4294967296` como uid 0: agora é nota, com o uid da musl. O fuzz confere também que todo `usuario` listado é uma
  conta que a `fgetpwent` lê, com o mesmo uid.

### Mudado
- `smolmoe-inventario` saiu de `make` e de `make test-unit`: `make inventario` compila e roda os testes dele. O CI
  roda esse alvo no Linux, ARM64, macOS e Windows, e o `make sanitize` também.

## 0.5.0 — 2026-11-08

### Adicionado
- **Gemma 3 1B Instruct:** suporte ao runtime C, tokenizer e conversor do Hugging Face. O checkpoint usado pelo
  atalho `make gemma` tem revisão e hashes SHA-256 fixados; o modelo também pode ser aberto pelo chat, pela janela e
  pela interface web.
- **Cinco ferramentas locais no chat:** `data_atual`, `dias_ate`, `calcular`, `converter` e `contar`. Para pedidos
  reconhecidos, o runtime recupera uma chamada de ferramenta quando o Gemma responde sem iniciá-la.
- Atalhos `make chat`, `make run-gui` e `make run-web` para abrir as interfaces diretamente com um modelo.

### Alterado
- README, guia de modelos e roadmap atualizados para documentar o Gemma, as ferramentas e os comandos disponíveis.
- Dependências dos atalhos de modelos alinhadas aos arquivos que cada modo de chat precisa.

## 0.4.0 — 2026-11-07

### Adicionado
- **Chat numa janela** (`smolmoe-gui`, `make gui`), em C++17 com Dear ImGui: Win32 + Direct3D 11 no Windows,
  GLFW + OpenGL 3 no Linux e no macOS.
  - A resposta chega aos poucos, sem travar a janela.
  - O botão "Parar" (ou Esc) interrompe uma resposta, e Ctrl+N começa outra conversa.
  - Os avisos das ferramentas aparecem resumidos, e dá para copiar qualquer mensagem.
  - O modelo vem de um `.smoe` (botão no Windows, caminho digitado no Linux e no macOS, arrastar ou linha de comando)
    ou de dentro do próprio executável.
  - O Release do Windows passa a trazer `qwen-pt-gui-windows-x64.exe` (janela + Qwen, dois cliques) e
    `smolmoe-gui-windows-x64.exe`.
  - Guia em [docs/DESKTOP.md](docs/DESKTOP.md).
- **Parar uma resposta no meio**, no motor (`EngineSink.stop`). O que já foi escrito fica no histórico. Uma parada
  dentro de uma chamada de ferramenta tira a mensagem do histórico. O teste prova isso: depois da parada, a mesma
  pergunta recebe exatamente a resposta de uma conversa nova.
- **Um atalho por modelo:** `make qwen`, `qwen-gui`, `qwen-q5`, `qwen-q5-gui`, `phi`, `phi-gui`, `stories` e
  `all-models` geram em `dist/` o executável com o modelo dentro. Eles baixam os pesos que faltam, e o Q5 é gerado do
  Q8. `make help` lista tudo; `make qwen-pt` continua valendo.
- **`make` no PowerShell e no cmd**, sem outro gerenciador de build. O Makefile usa o shell que vem com o compilador
  e põe a pasta dele na frente do PATH durante o build.
- [docs/CAMINHOS.md](docs/CAMINHOS.md): todos os jeitos de rodar (janela, terminal, navegador, scripts, outro
  modelo, da fonte), em cada sistema, e qual modelo escolher.
- **Chat no navegador** (`web/`, `make web`): o modelo roda inteiro no navegador, em WebAssembly com SIMD128 e
  threads. A página aceita um arquivo `.smoe` do computador ou baixa os pesos uma vez para o cache do navegador. A
  página inteira tem 254 KB, e no Chrome do notebook de testes o Qwen escreve a 11 tokens/s. Guia em
  [docs/WEB.md](docs/WEB.md).
- Build WebAssembly com WASI (`make wasi`, zig 0.13). O hash dos logits é o do binário nativo com 1, 2 e 4 threads, e
  um chat com ferramenta sai idêntico, byte a byte. Detalhes em [DECISIONS.md](docs/DECISIONS.md), iteração 10.

### Alterado
- A lógica da conversa saiu do chat do terminal para `src/engine.c`, que o terminal, o navegador e a janela usam. A
  saída do terminal é a mesma, byte a byte.
- `die()` aceita um gancho (`die_set_handler`), que a janela usa para mostrar o erro em vez de sumir.
- A página do navegador fica para uso local; a publicação foi adiada.

## 0.3.1 — 2026-11-07

### Alterado
- **ARM64 de 2,6× a 3,7× mais rápido** (Apple Silicon, Linux ARM, Termux): kernels NEON para os produtos
  escalares Q8 e Q5, a quantização das ativações e a SwiGLU, com resultados idênticos bit a bit aos do código
  escalar. No Mac do CI, a escrita no chat foi de 17,3 para 64,3 tokens/s. Detalhes em
  [DECISIONS.md](docs/DECISIONS.md), iteração 9.
- O CI roda os testes com o modelo real também no Windows e no macOS.

## 0.3.0 — 2026-11-07

### Adicionado
- **Release com binários prontos** para Windows x64, Linux x64, Linux ARM64 (também Termux, no Android) e macOS
  (Intel e Apple Silicon num binário só). Os binários de Linux são estáticos (musl).
  - Executáveis de arquivo único para o Qwen (chat em português) e o TinyStories.
  - Modelos num arquivo só: Qwen Q8 e Q5, TinyStories e Phi-4-mini. O Phi vai em partes, por causa do limite de
    2 GB do GitHub, e é convertido no CI e conferido byte a byte contra o `provenance.json`.
  - Cada binário passa por um teste rápido no próprio sistema antes de publicar (no ARM64, emulado), e o Release
    traz `SHA256SUMS`. Instruções em [docs/INSTALL.md](docs/INSTALL.md).
- **`!comando` no chat:** a linha roda no shell que abriu o smolmoe (PowerShell, pwsh, cmd, Git Bash, bash, zsh) e a
  saída vai direto para o terminal, sem passar pelo modelo.
  - `!cd` muda a pasta do chat, e Ctrl+C interrompe só o comando.
  - `--shell` escolhe o shell; `--no-shell-escape` desliga a função.
  - Com entrada por pipe, a função fica desligada, a menos que se use `--shell-escape`.
  - Texto colado com várias linhas vira uma mensagem só, e nenhuma linha dele roda como comando.
- Texto colado com várias linhas agora vai ao modelo como **uma** mensagem; antes, cada linha virava uma pergunta.
- **Chamada de funções no chat** (template do Qwen2.5). Há duas ferramentas embutidas: `data_atual` e `dias_ate`, que
  aceita data em texto e feriados brasileiros, inclusive os móveis. O modelo escolhe a ferramenta, e as datas são
  calculadas em C.
  - As ferramentas só são oferecidas quando a mensagem fala de data ou de tempo.
  - A data escrita pelo usuário vale mais que o argumento do modelo.
  - Uma chamada que não combina com a mensagem é descartada.
  - Medição: 12 de 15 certas em perguntas novas, que não foram usadas no desenho. Os limites conhecidos estão em
    [DECISIONS.md](docs/DECISIONS.md), iteração 4.
- **Ferramenta `calcular`:** o modelo escreve a conta e o programa a calcula, com símbolos ou palavras ("15% de
  230", "dois mil e quinhentos") e números no formato brasileiro.
  - Todo número da conta precisa vir da mensagem.
  - Quando a mensagem é só uma conta e o modelo escolhe outra ferramenta, o runtime abre a chamada de `calcular`.
  - **Medição às cegas:** 12 de 40 contas certas com ela e 11 de 40 sem ela. Ela também faz errar contas que o modelo
    acertava sozinho, quando ele monta a fórmula errada. Entrou por decisão do usuário; detalhes em
    [DECISIONS.md](docs/DECISIONS.md), iteração 7.
- `scripts/eval-tools.ts`: conjuntos da calculadora (`--set calcdev` e `--set calc`, às cegas), linha de base sem
  ferramentas (`--no-tools`), intervalo de confiança de 95% e leitura de números só no formato brasileiro.
- `--no-tools` desliga as ferramentas. `--now AAAA-MM-DDTHH:MM` fixa o relógio, para execuções reproduzíveis.
- Campo `tool_calls` nas estatísticas do chat.
- `scripts/eval-tools.ts`: mede a escolha de ferramenta e a resposta.
  - Há três conjuntos: o de desenvolvimento e dois de perguntas novas.
  - `--session` mede a conversa longa.
- `scripts/fetch-models.ts` (`make models`): baixa os pesos do Release `models-v1` e confere o SHA-256.
- `--profile`: tempo de cada passada dividido por fase, nas estatísticas.
- `scripts/bench.ts` refeito como esteira de desempenho:
  - cenários fixos, medianas e tempo por fase;
  - `--save` e `--compare` contra uma base, e `--ab` para comparar dois binários alternados na mesma sessão;
  - sai com erro se alguma métrica piorar.

- **Phi-4-mini-instruct (Microsoft, 3,8B)**, o primeiro modelo fora da família Qwen: 17% melhor em português (bits
  por byte) e certo em contas e comandos em que o Qwen 0,5B errava; cerca de 4 tokens/s neste notebook. Detalhes
  e validação em [DECISIONS.md](docs/DECISIONS.md), iteração 6.
  - Chaves opcionais `rotary_dim` e `rope_attention_factor` na config (RoPE parcial e escalado).
  - Tokenizer `o200k` (o do GPT-4o), `.qbp` v2 e template de chat do Phi; o v1 do Qwen segue idêntico.
- `scripts/convert.ts`: conversor em TypeScript (Qwen2 e Phi-3), no lugar do Python. Reproduz os arquivos do Qwen
  byte a byte. `fetch-qwen2` e `fetch-phi3` baixam os originais fixados e conferem o SHA-256.
- `scripts/ref/`: implementação de referência do Phi-4-mini em TypeScript (double), para validar o runtime.
- `scripts/gen-unicode.ts`: gerador das tabelas Unicode em TypeScript (classes no Unicode 16.0, NFC no 15.0).

### Alterado
- O limite de memória padrão passou a ser 512 MiB mais o cache de atenção de até 2048 posições, então modelos
  maiores abrem sem `--memory-mib`. Com a opção, o limite é exatamente o pedido.
- **Leitura do prompt de 21% a 39% mais rápida**, e primeiro token de 17% a 26% mais cedo, em duas medições A/B.
  Quantização, ativação e normalização passaram a usar AVX2 e todas as threads; antes eram 23% do tempo numa thread só.
  Detalhes em [DECISIONS.md](docs/DECISIONS.md), iteração 5.
- Exponencial própria (corretamente arredondada em mais de 99,99% das entradas), idêntica no escalar e no AVX2 e em
  qualquer sistema operacional. Os números mudam no último bit em casos raros. A perplexidade foi de 11,3885 para
  11,3681.
- Com as ferramentas ligadas, o chat leva cerca de 5,5 s a mais para abrir: o prompt de sistema vai de 31 para 397
  tokens (279 com as de data e mais 118 da `calcular`). `--no-tools` volta ao comportamento anterior.
- `.env` e `.env.*` ficam fora do Git (o token do Hugging Face mora ali).

## 0.2.0 — 2026-11-06

Reescrita do runtime para eficiência e para uso em produção. Números e justificativas em
[docs/DECISIONS.md](docs/DECISIONS.md).

### Adicionado
- **Executável único estilo llamafile**: `smolmoe-bundle` (contêiner v3 com config e tokenizer embutidos; `--exe` anexa ao
  runtime) e `make qwen-pt`. Sem `--model`, o runtime usa o modelo anexado a si mesmo.
- **Chat multi-turno** (`-i`, padrão sem `--prompt`) com cache KV reaproveitado, `/limpar` e `/sair`; entrada UTF-8 correta
  no console do Windows.
- Mensagens de erro do Windows em UTF-8 (antes saíam na codepage local).
- Formato Q5 (dtype 2, 22 bytes por 32 pesos) e a ferramenta `smolmoe-quantize`: 39% menos memória por +1,7% de
  perplexidade. Opcional; não acelera no hardware medido.
- `--perplexity`: NLL e perplexidade do prompt com teacher forcing; texto de avaliação em `tests/goldens/eval-pt.txt`.
- Kernels AVX2 agora exigem F16C (presente em todo CPU com AVX2 do mercado).
- Suporte nativo a Windows (MinGW-w64 GCC/Clang): argv em UTF-8, console em UTF-8 e build estático em `make release`.
- Pool de threads determinístico (`--threads`, padrão: núcleos físicos).
- Prefill em lote (`--batch`, padrão 32) com kernel AVX2 de 4 tokens por linha de pesos.
- Amostragem com `--top-k` (padrão 40) e `--top-p` (padrão 0,95).
- `--version`, e os campos `version`, `threads`, `batch` e `mapped_bytes` nas estatísticas.
- Suíte de testes: unitários em C (kernels, f16, pool) e integração em TypeScript/Bun (55 testes).
- Scripts `bench.ts`, `compare-traces.ts` e `make-fixture.ts`. CI para Linux, Windows e macOS.

### Alterado (incompatível)
- Os pesos agora são mapeados em memória. `--memory-mib` limita só o heap (KV, ativações, tokenizer); antes incluía um
  cache de pesos.
- As ativações são quantizadas em Q8 antes de cada produto matricial. Os logits diferem da v0.1 em +0,05% de
  perplexidade, e hashes antigos de `--hash-logits` não valem mais.
- O limite do prompt passou de `context-1` bytes para `4 × context` bytes; a checagem real é sobre os tokens.
- Os arquivos de saída (`--trace`, `--token-ids`, `--stats`) são escritos em modo binário (LF em todas as plataformas).
- Os modelos foram movidos de `examples/` para `models/` e saíram do controle de versão.

### Removido
- `--drop-cache` e os campos `cache_hits`, `cache_misses`, `read_bytes`, `expert_read_bytes` e `cache_bytes`, que só
  faziam sentido com o cache de tiles.
- O binário Linux pré-compilado, as evidências de validação da v0.1 e os testes em Python.

## 0.1.0 — 2026-11-05

Versão inicial: runtime de uma thread com cache de tiles sobre `pread`, só Linux.
