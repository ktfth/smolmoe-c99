# App para Android

O chat do smolmoe num aplicativo de Android, com o modelo dentro: instala, toca no ícone e conversa. Não precisa de
Termux nem de internet, e o app não pede nenhuma permissão. É a mesma tela da [janela do desktop](DESKTOP.md) (Dear
ImGui) com o mesmo motor de conversa e as cinco ferramentas (`data_atual`, `dias_ate`, `calcular`, `converter`,
`contar`).

## Instalar

1. Gere o APK (próxima seção) ou pegue `smolmoe-qwen-pt.apk` pronto.
2. Passe para o celular e abra o arquivo. O Android pede para permitir a instalação "de fontes desconhecidas" para o
   app que abriu o arquivo (Arquivos, Chrome...): permita só essa vez.
3. O app aparece como **smolmoe**. Na primeira abertura ele confere os pesos e processa as instruções da conversa,
   o que leva alguns segundos.

Pelo cabo, com a depuração USB ligada: `adb install dist/smolmoe-qwen-pt.apk`.

Requisitos: Android 10 ou mais novo, processador ARM64 (`arm64-v8a`, praticamente todo celular desde 2018), OpenGL ES 3
e ~1,2 GB livres (o APK tem 560 MB e o Android guarda uma cópia na instalação).

## Usar

- **Escrever:** toque no campo de baixo. Abre a caixa de texto do próprio Android, com o seu teclado, acentos,
  correção e colar. **Enviar** manda a mensagem; **Fechar** guarda o rascunho no campo.
- **Parar:** o botão **Parar** interrompe a resposta.
- **Copiar:** toque e segure uma mensagem; aparece **Copiar**.
- **Rolar:** arraste a conversa com o dedo.
- **Nova conversa:** o botão no alto.

O app segue o tema claro ou escuro do celular. Ao sair do app a conversa continua na memória enquanto o Android não
fechar o processo.

O que não existe no app: o `!comando` do terminal (o Android não deixa um app rodar comandos do sistema) e a escolha de
outro modelo (o modelo é o que vem no APK).

## Gerar o APK

Precisa do Android SDK com um NDK, build-tools e uma plataforma (o Android Studio instala os três; `sdkmanager` também
serve), de um JDK (`javac`) e da variável `ANDROID_HOME` apontando para o SDK. Não usa Gradle.

```sh
make apk                     # dist/smolmoe-qwen-pt.apk: Qwen2.5-0.5B Q8, arm64-v8a
make apk APK_MODEL=qwen-q5   # dist/smolmoe-qwen-pt-q5.apk: o Q5 (340 MB)
make apk APK_ABI=x86_64      # dist/smolmoe-qwen-pt-x86_64.apk: para o emulador num PC
```

O APK sai assinado com a chave de depuração (`~/.android/debug.keystore`). Para assinar com a sua:
`make apk APK_KEYSTORE=minha.jks APK_KS_PASS=... APK_KEY_ALIAS=...`. Um celular só atualiza o app por cima com um
APK da mesma chave; trocar de chave pede desinstalar antes.

### Como é montado

`android/android.mk` faz tudo com as ferramentas do SDK:

1. O clang do NDK compila o runtime em C (com os mesmos `-Werror` e as mesmas flags de determinismo do desktop), a
   tela (`gui/view.cpp`, `gui/session.cpp`), a camada do Android (`gui/platform_android.cpp`), o Dear ImGui com o
   backend Android e OpenGL ES 3, e a `native_app_glue` do NDK, numa só `libsmolmoe.so`.
2. `javac` e `d8` compilam `MainActivity.java`: uma `NativeActivity` que só faz o que o lado nativo não faz bem, a
   caixa de texto do sistema e a área de transferência.
3. `smolmoe-bundle` junta pesos, configuração e tokenizador num `.smoe` autocontido, que vira o asset `model.smoe`.
4. `aapt` empacota **sem compressão** (`-0 ''`), `zipalign -P 16` põe a biblioteca numa página de 16 KiB (Android 15)
   e `apksigner` assina.

O modelo não é extraído na instalação nem copiado ao abrir: o runtime mapeia o trecho do APK onde o asset está
(`AAsset_openFileDescriptor64` dá o descritor, o início e o tamanho; `desktop_open` aceita `offset` e `length`). A
biblioteca também é lida direto do APK (`extractNativeLibs="false"`).

### Detalhes da camada do Android

- **Threads:** celulares misturam núcleos rápidos e lentos (o Dimensity 7300 do moto g86 tem 4 Cortex-A78 e 4 A55).
  O trabalho é dividido em partes iguais, então os lentos atrasariam cada passo; o app usa só os núcleos que chegam à
  maior frequência máxima (`cpufreq`).
- **Prioridade:** a thread do app roda com a prioridade da tela (nice −10) e as threads criadas por ela herdam isso.
  A thread do modelo volta para a prioridade normal antes de abrir o pool. Sem isso, no emulador, o modelo tomava todos
  os núcleos e o Android mostrava "não está respondendo" para o app e para a própria interface do sistema.
- **Barras do sistema:** a `NativeActivity` desenha atrás da barra de status e da de navegação; a tela usa
  `contentRect`, a parte que elas deixam livre. O alvo é a API 34, que não força o modo de ponta a ponta da API 35.

## O que foi testado

| | Resultado |
|---|---|
| Build `arm64-v8a` e `x86_64` (NDK r28, build-tools 36.1) | compila sem aviso com `-Werror` |
| Alinhamento | `zipalign -c -P 16` passa; modelo e `.so` gravados sem compressão |
| Emulador Android 16, x86_64 | abre, conversa, chama `dias_ate`, copia, segue as barras do sistema |
| Emulador Android 16, `arm64-v8a` traduzido | abre e chega à conversa (lento: o emulador traduz as instruções) |
| Celular de verdade (moto g86 5G) | **ainda não testado** |

A velocidade no celular ainda não foi medida. No emulador x86_64 deu 2,4 tokens/s, mas o emulador divide a CPU com o
PC e não diz nada sobre um celular.
