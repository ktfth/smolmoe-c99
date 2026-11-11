# Segurança

O smolmoe abre arquivos de modelo que não acompanham o programa: `.smoe`, tokenizador (`.qbp`, `.spb` ou o texto `SMBPE1`) e `config.ini`, também quando viajam dentro de um executável. Um arquivo malformado não deve corromper memória. Se corromper, ou se fizer o processo executar conteúdo do arquivo, isso é uma falha de segurança.

O runtime não isola o modelo. Um arquivo válido pode ocupar CPU e RAM à vontade, e o chat com `!comando` roda no shell o que foi digitado. Isso é o funcionamento do programa. Não abra um modelo de origem desconhecida quando a integridade do processo importa, e não deixe o escape de shell disponível para quem você não confia.

## Como reportar

Use o reporte privado do GitHub neste repositório: **Security → Advisories → Report a vulnerability**. Não abra uma issue pública com o arquivo que dispara a falha, com um dump de memória ou com um passo a passo de exploração.

No reporte, diga:

- a versão (`smolmoe --version`) e o sistema operacional;
- o que você esperava e o que aconteceu;
- um arquivo pequeno que reproduz o problema, se puder enviá-lo só por esse canal;
- a linha que o AddressSanitizer ou o UndefinedBehaviorSanitizer apontou, se você rodou `make sanitize` ou `make fuzz`.

O recebimento é confirmado, e a correção entra descrita no ramo principal. Não há um prazo prometido: o projeto é mantido por uma pessoa.

## Fora deste canal

- um modelo que erra a resposta, inventa um fato ou só consome muita memória, sem corromper o processo;
- pesos que você não tem licença para redistribuir;
- o texto que o modelo produz.
