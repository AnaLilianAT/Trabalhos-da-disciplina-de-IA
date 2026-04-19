# Arquitetura Inicial

Este documento descreve as interfaces centrais do projeto base.

## Camadas

- game: regra de negocio do Othello/Reversi (sem logica de agente)
- agents: estrategias de decisao (sem implementar regras de tabuleiro)
- evaluation: funcoes de avaliacao reutilizaveis
- experiments: orquestracao de partidas e metricas
- utils: constantes e formatacao

## Contratos principais

- Agent.choose_move(state) -> Move
- rules.get_legal_moves(state) -> list[Move]
- rules.apply_move(state, move) -> GameState
- Heuristic.evaluate(state, perspective_player) -> float
- play_match(black_agent, white_agent, ...) -> MatchMetrics

## Pontos de extensao futuros

- Variacoes de regras via novos parametros em GameConfig ou novos modulos em game
- Novas heuristicas em evaluation/heuristic.py
- Agentes adicionais em src/agents
- Suites de comparacao e torneios em src/experiments
