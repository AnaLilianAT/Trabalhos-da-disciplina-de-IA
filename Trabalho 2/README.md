# Othello/Reversi IA (Projeto Base)

Projeto inicial em Python 3.11 para comparar agentes de IA em Othello/Reversi:

- Min-Max com poda Alfa-Beta
- Monte Carlo Tree Search (MCTS)

Este repositório contem apenas a arquitetura, interfaces principais e implementacoes baseline para facilitar evolucao incremental e testes.

Status atual:

- Motor do jogo Othello/Reversi completo e deterministico (tabuleiro, jogadas, captura, pass, fim de jogo, vencedor)
- Agente Minimax com poda Alfa-Beta e profundidade limitada
- Agente MCTS com selecao UCT, expansao, rollout e retropropagacao
- Extensao de pontuacao ponderada opcional (standard ou weighted)

## Objetivos desta base

- Codigo modular e testavel
- Separacao clara entre regras do jogo, agentes e avaliacao
- Suporte a tabuleiros configuraveis (padrao 6x6, com suporte para 4x4, 6x6, 8x8)
- Suporte a qualquer tamanho par valido >= 4 (exemplos comuns: 4x4, 6x6, 8x8)
- Facilidade para extensoes futuras (novas heuristicas, variacoes de regras, novos agentes)

## Estrutura

- src/game: representacao do dominio do jogo (tabuleiro, estado, regras, configuracao)
- src/agents: classe base de agente e esqueletos de Minimax/MCTS
- src/evaluation: interfaces e heuristicas
- src/experiments: execucao de partidas e metricas
- src/utils: constantes e formatacao util
- tests: testes iniciais de configuracao, tabuleiro e regras basicas
- docs: notas de arquitetura

## Como executar

Executar partida baseline:

python main.py

Com parametros:

python main.py --board-size 8 --verbose --seed 42

Selecionar modo de pontuacao:

python main.py --scoring-mode weighted

Executar testes:

python -m unittest discover -s tests

## Interfaces principais

- Agent.choose_move(state) -> Move: contrato unificado para todos os agentes
- rules.get_legal_moves(state) / rules.apply_move(state, move): motor de regras desacoplado de agentes
- Heuristic.evaluate(state, perspective_player) -> float: avaliacao plugavel para busca
- experiments.runner.play_game(...): orquestracao completa de confronto entre agentes com seed, verbose e estatisticas
