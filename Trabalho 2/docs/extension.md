# Extensao Obrigatoria: Pontuacao Ponderada

## Modificacao escolhida

Foi adicionada uma extensao de regra de pontuacao mantendo as regras de captura,
pass e termino do Othello padrao.

Agora existem dois modos configuraveis:

- standard: vencedor por contagem simples de pecas
- weighted: vencedor por pontuacao posicional

No modo weighted:

- canto vale 4 pontos
- borda vale 2 pontos
- casa interna vale 1 ponto

## Por que e uma extensao nao trivial

A mudanca nao altera apenas exibicao de resultado. Ela altera o objetivo do jogo:

- no modo padrao, maximizar numero de pecas pode bastar
- no modo weighted, controlar cantos e bordas passa a ter valor estrategico maior

Isso afeta diretamente tomada de decisao de agentes de busca, especialmente em
finais de jogo e estados terminais.

## Como foi implementada sem duplicar logica

A regra foi isolada em modulo proprio:

- src/game/scoring.py

Esse modulo concentra:

- validacao de modo de pontuacao
- pesos posicionais
- calculo de score por jogador
- determinacao de vencedor a partir dos scores

A logica de jogo principal continua no modulo de regras, que apenas delega o
calculo de score/vencedor para o modulo de scoring.

## Impacto esperado na estrategia

### Minimax

- no modo weighted, estados terminais agora refletem pontuacao ponderada
- cantos e bordas tendem a ser mais valorizados no objetivo final
- linhas de jogo que perdem pecas, mas ganham posicoes fortes, podem ser melhores

### MCTS

- rollouts continuam usando mesmo motor de regras
- recompensa final usa vencedor definido pelo modo ativo
- em weighted, rollouts tendem a favorecer trajetorias com controle posicional

## Compatibilidade e configuracao

A configuracao de jogo agora aceita scoring_mode:

- scoring_mode="standard" (padrao)
- scoring_mode="weighted"

As partidas, Minimax e MCTS continuam funcionando sem alteracoes externas.

## Preparacao para futuras extensoes

A arquitetura ficou pronta para novas variacoes porque:

- pesos estao encapsulados em ScoreWeights
- modo de pontuacao e validado centralmente
- regras de captura/termino nao foram duplicadas
- runner e metricas expõem score final e modo utilizado
