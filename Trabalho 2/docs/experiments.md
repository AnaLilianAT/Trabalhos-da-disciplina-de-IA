# Experimentos: Minimax vs MCTS

Este documento descreve como executar e interpretar os experimentos de comparacao entre agentes no Othello/Reversi.

O modulo principal esta em `src/experiments/runner.py` e cobre:

- taxa de vitoria e empates
- custo computacional
- proxy de qualidade de decisao (margem de score)
- suporte a modo de pontuacao `standard` e `weighted`
- exportacao de resultados em CSV
- tabelas resumo no terminal
- graficos opcionais com matplotlib

## Estrutura de cenarios da suite padrao

A suite padrao gera automaticamente os seguintes blocos de experimentos:

1. Minimax(balanced) vs MCTS com politicas de rollout diferentes:
   - random
   - rollout_simple
   - rollout_successor_eval
   - rollout_topk

2. Comparacao de profundidades do Minimax:
   - d3 vs d4
   - d4 vs d5
   - d3 vs d5

3. Comparacao de orcamento de simulacoes no MCTS:
   - 100 vs 500
   - 500 vs 1000
   - 100 vs 1000

4. Comparacao de heuristica do Minimax:
   - balanced vs aggressive

5. Comparacao entre politicas de rollout do MCTS:
   - random vs rollout_simple
   - rollout_simple vs rollout_successor_eval
   - rollout_successor_eval vs rollout_topk

6. Comparacao entre heuristicas usadas no rollout_successor_eval:
   - balanced vs aggressive

7. Experimento adicional variando tamanho de tabuleiro:
   - board_size_sweep_*_minimax_balanced_vs_mcts_rollout_simple

Todos os blocos acima sao executados para os modos de pontuacao selecionados (`standard`, `weighted`) e com alternancia de cor entre partidas para reduzir vies de BLACK/WHITE.

## Como reproduzir

### Executar suite completa

```bash
python -m src.experiments.runner \
  --matches-per-config 8 \
  --base-seed 42 \
  --board-sizes 4,6,8 \
  --scoring-modes standard,weighted \
  --output-dir results/experiments \
  --generate-plots
```

### Execucao mais rapida (sanity check)

```bash
python -m src.experiments.runner \
  --matches-per-config 2 \
  --board-sizes 4,6 \
  --scoring-modes standard \
  --output-dir results/experiments_quick
```

### Sem alternancia de cor (nao recomendado para comparacao final)

```bash
python -m src.experiments.runner --no-alternate-colors
```

## Arquivos gerados

No diretorio de saida sao gerados:

- `experiment_matches_<timestamp>.csv`: resultados por partida
- `experiment_summary_<timestamp>.csv`: metricas agregadas por cenario
- `plot_win_draw_rates.png` (opcional)
- `plot_computational_cost.png` (opcional)
- `plot_quality_margin.png` (opcional)

## Metricas agregadas

As tabelas e o CSV de resumo incluem, entre outras, as metricas:

- taxa de vitoria de Agent A
- taxa de vitoria de Agent B
- taxa de empate
- tempo medio por jogada
- tempo medio por partida
- nos expandidos medios por jogada (Minimax)
- podas medias por jogada (Minimax)
- simulacoes medias por jogada (MCTS)
- nos criados medios por jogada (MCTS)
- margem media de score para Agent A
- margem normalizada media de score para Agent A

## Interpretacao sugerida

### Influencia da heuristica do Minimax

Ao comparar `minimax_balanced_vs_aggressive`, observe:

- mudanca na taxa de vitoria
- variacao de tempo medio por jogada
- variacao de podas medias por jogada

Se a heuristica for mais informativa para o cenario, voce pode ver melhor margem de score com custo similar (ou ligeiramente maior) de busca.

### Rollout aleatorio vs rollout heuristico no MCTS

Nos cenarios `mcts_random_vs_rollout_simple` e `minimax_balanced_vs_mcts_*`:

- rollout aleatorio tende a ter maior variancia
- rollout heuristico tende a estabilizar decisoes
- a diferenca aparece em taxa de vitoria e margem de score

### Diferencas entre rollout heuristics

Compare os cenarios:

- `mcts_rollout_simple_vs_rollout_successor_eval`
- `mcts_rollout_successor_eval_vs_rollout_topk`
- `mcts_successor_eval_balanced_vs_aggressive`

Esses cenarios ajudam a responder:

- quanto vale usar avaliacao de sucessor durante rollout
- se o top-k melhora robustez em relacao ao greedy
- quanto a heuristica usada dentro do rollout_successor_eval impacta desempenho

## Observacoes de reproducibilidade

- Cada partida usa seed derivada de `base_seed` e do indice da partida.
- Os agentes alternam BLACK/WHITE (quando `alternate_colors=True`).
- Todas as configuracoes dos agentes sao serializadas no CSV, incluindo:
  - Minimax: profundidade e heuristica
  - MCTS: numero de simulacoes, constante de exploracao, rollout_policy, rollout_heuristic_name, top-k e time limit

## Uso por API (sem CLI)

```python
from src.experiments.runner import build_and_run_default_experiment_suite

build_and_run_default_experiment_suite(
    matches_per_configuration=4,
    base_seed=123,
    board_sizes=(4, 6, 8),
    scoring_modes=("standard", "weighted"),
    output_dir="results/experiments_api",
    generate_plots=False,
)
```
