# Othello/Reversi IA - Entrega Final

Projeto da disciplina de IA para comparar agentes no Othello/Reversi, com foco em:

- Minimax com poda Alfa-Beta
- Monte Carlo Tree Search (MCTS) com UCT
- Heuristicas de avaliacao e de rollout
- Experimentos reproduziveis com exportacao em CSV
- Extensao de pontuacao ponderada (modo alternativo de score)

## 1) Descricao do trabalho

O projeto implementa:

- motor completo do jogo (tabuleiro, jogadas legais, capturas em 8 direcoes, pass, terminal e vencedor)
- agente Minimax com poda Alfa-Beta e profundidade configuravel
- agente MCTS com selecao UCT, expansao, rollout e backpropagation
- conjunto de heuristicas de estado e politicas de rollout reutilizaveis
- modulo de experimentos para comparacao sistematica entre configuracoes

Objetivo central: comparar taxa de vitoria, custo computacional e qualidade de decisao entre Minimax e MCTS em diferentes cenarios.

## 2) Estrutura do projeto

- src/game: dominio do jogo (board, state, rules, config, scoring)
- src/agents: agentes Minimax e MCTS
- src/evaluation: heuristicas de estado e politicas de rollout
- src/experiments: runner de partidas, suite de experimentos, trace Alfa-Beta
- src/utils: constantes e formatacao
- tests: testes unitarios
- docs: documentacao de arquitetura, heuristicas, extensao, experimentos e exemplo de trace

## 3) Requisitos

- Python 3.11+

Nao ha dependencia obrigatoria externa para executar jogo e testes.
Para graficos dos experimentos: matplotlib (opcional).

## 4) Como executar

### 4.1 Partida simples

```bash
python main.py
```

### 4.2 Partida com parametros

```bash
python main.py --board-size 8 --seed 42 --verbose
```

### 4.3 Ajuda completa da CLI principal

```bash
python main.py --help
```

## 5) Como rodar testes

```bash
python -m unittest discover -s tests
```

## 6) Como rodar experimentos

Runner principal de experimentos:

```bash
python -m src.experiments.runner --matches-per-config 8 --base-seed 42 --board-sizes 4,6,8 --scoring-modes standard,weighted --output-dir results/experiments --generate-plots
```

Execucao rapida (sanity check):

```bash
python -m src.experiments.runner --matches-per-config 2 --board-sizes 4,6 --scoring-modes standard --output-dir results/experiments_quick
```

Saidas esperadas:

- CSV por partida: experiment_matches_<timestamp>.csv
- CSV agregado: experiment_summary_<timestamp>.csv
- graficos opcionais: plot_win_draw_rates.png, plot_computational_cost.png, plot_quality_margin.png

## 7) Como parametrizar o projeto

### 7.1 Alterar tamanho do tabuleiro

Na partida simples:

```bash
python main.py --board-size 4
python main.py --board-size 6
python main.py --board-size 8
```

Nos experimentos:

```bash
python -m src.experiments.runner --board-sizes 4,6,8
```

### 7.2 Alterar profundidade do Minimax

```bash
python main.py --minimax-depth 5
```

### 7.3 Escolher heuristica de avaliacao do Minimax

Heuristicas disponiveis por nome:

- balanced
- aggressive
- lightweight

Exemplo:

```bash
python main.py --minimax-eval aggressive
```

### 7.4 Alterar orcamento do MCTS

Por numero de simulacoes:

```bash
python main.py --mcts-simulations 1000
```

Por limite de tempo (segundos):

```bash
python main.py --mcts-time-limit 0.5
```

Observacao: quando `mcts-time-limit` e definido, o MCTS roda em modo limitado por tempo.

### 7.5 Escolher politica de rollout do MCTS

Politicas disponiveis:

- random
- rollout_simple
- rollout_successor_eval
- rollout_topk

Exemplos:

```bash
python main.py --mcts-rollout-policy random
python main.py --mcts-rollout-policy rollout_simple
python main.py --mcts-rollout-policy rollout_successor_eval --mcts-rollout-heuristic lightweight
python main.py --mcts-rollout-policy rollout_topk --mcts-rollout-heuristic balanced --mcts-rollout-topk 3
```

## 8) Modo padrao e extensao

O projeto possui dois modos de pontuacao:

- standard: contagem simples de discos
- weighted: ponderacao por posicao (canto=4, borda=2, interno=1)

Exemplos:

```bash
python main.py --scoring-mode standard
python main.py --scoring-mode weighted
```

## 9) Resumo tecnico (entrega)

### Minimax com Alfa-Beta

- busca adversarial em profundidade limitada
- poda de ramos quando `alpha >= beta`
- suporte a ordenacao de jogadas para melhorar eficiencia
- metricas: nos expandidos, podas e profundidade atingida

### Conjunto de heuristicas implementadas

Heuristicas de avaliacao de estado (para Minimax e uso auxiliar):

- balanced
- aggressive
- lightweight

Features usadas nas composicoes:

- diferenca de pecas
- mobilidade
- controle de cantos
- controle de bordas
- proximidade de canto

### MCTS

- selecao: usa UCT
- expansao: cria filho para uma jogada ainda nao tentada
- rollout: simula ate terminal (random ou heuristico)
- backpropagation: atualiza visitas e valor medio dos nos

### Implementacao do UCT (breve)

No passo de selecao, cada filho e pontuado por:

`UCT = Q + c * sqrt(ln(N_pai) / N_filho)`

Onde:

- `Q`: valor medio de recompensa do no
- `c`: constante de exploracao (`--mcts-exploration-constant`)
- `N_pai`: visitas do no pai
- `N_filho`: visitas do no filho

Isso equilibra exploracao (testar ramos pouco visitados) e explotacao (preferir ramos promissores).

### Rollout aleatorio vs rollout heuristico

- random: mais simples e barato por decisao, porem com variancia maior
- heuristico: tende a estabilizar estimativas do MCTS e melhorar qualidade media, com maior custo por simulacao

### Diferentes rollout heuristics disponiveis

- rollout_simple: regra leve de priorizacao (canto/mobilidade/borda)
- rollout_successor_eval: escolhe jogada pelo valor do estado sucessor com heuristica de avaliacao
- rollout_topk: sorteia entre as top-k jogadas ranqueadas por heuristica

### Diferenca entre heuristicas de estado (Minimax) e heuristicas de rollout (MCTS)

- heuristicas de estado (Minimax): avaliam o valor de um estado na fronteira da busca para aproximar a utilidade real
- heuristicas de rollout (MCTS): priorizam escolhas durante simulacoes para reduzir ruido e guiar playouts

Em resumo: Minimax usa heuristica como funcao de valor; MCTS usa heuristica como guia de simulacao (quando politica nao e random).

### Quando cada politica de rollout tende a ser vantajosa

- random: baseline rapido e simples; util para comparar efeito de guias heuristicas
- rollout_simple: bom custo-beneficio quando se quer ganho moderado sem avaliar sucessor completo
- rollout_successor_eval: tende a ser melhor quando a heuristica de estado captura bem o jogo, mas custa mais
- rollout_topk: equilibrio entre diversidade e qualidade, reduzindo determinismo excessivo do greedy

### Extensao implementada

- modo `weighted` com pontuacao por posicao
- integracao com regras, Minimax, MCTS, runner e experimentos
- comparacao entre `standard` e `weighted` na suite experimental

### Analise experimental

O modulo de experimentos calcula, por cenario:

- taxa de vitoria e empate
- tempo medio por jogada e por partida
- nos expandidos/podas medias (Minimax)
- simulacoes e nos criados medios (MCTS)
- margem media de score como proxy de qualidade de decisao

Detalhes operacionais: ver docs/experiments.md

## 10) Limitacoes atuais (resumo curto)

- heuristicas de avaliacao ainda simples em relacao a avaliadores de alto nivel
- rollout heuristico ainda raso (nao usa modelos aprendidos)
- rollout aleatorio possui alta variancia
- custo do Minimax cresce rapidamente com profundidade
- custo computacional cresce com tamanho do tabuleiro

## 11) Documentacao complementar

- docs/ARCHITECTURE.md
- docs/heuristic.md
- docs/extension.md
- docs/experiments.md
- docs/alpha_beta_example.md
