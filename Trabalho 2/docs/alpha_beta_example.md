# Exemplo Minimax com Poda Alfa-Beta

## Resumo
- Profundidade limite: 2
- Tamanho do tabuleiro: 6x6
- Heuristica de avaliacao: balanced
- Ordenacao de jogadas: True
- Jogada escolhida no no raiz: (1, 2)
- Valor minimax da jogada escolhida: 0.0000
- Nos expandidos: 10
- Eventos de poda: 3
- Ramos podados: 6

## Tabuleiro inicial usado
```text
     0  1  2  3  4  5
   +-------------+
 0 | . . . . . . |
 1 | . . . . . . |
 2 | . . W B . . |
 3 | . . B W . . |
 4 | . . . . . . |
 5 | . . . . . . |
   +-------------+
Legend: B=BLACK, W=WHITE, .=EMPTY
```

## Jogadas consideradas no no raiz
- (1, 2), (2, 1), (3, 4), (4, 3)

## Jogadas consideradas por no expandido
| No | Tipo | Profundidade | Jogadas consideradas |
| --- | --- | --- | --- |
| N1 | MAX | 0 | (1, 2), (2, 1), (3, 4), (4, 3) |
| N2 | MIN | 1 | (1, 3), (1, 1), (3, 1) |
| N6 | MIN | 1 | (3, 1), (1, 1), (1, 3) |
| N10 | MIN | 1 | (2, 4), (4, 4), (4, 2) |
| N14 | MIN | 1 | (4, 2), (4, 4), (2, 4) |

## Folhas e valores heuristicas/terminais
| No | Tipo | Profundidade | Jogada pai | Valor folha |
| --- | --- | --- | --- | --- |
| N3 | corte_profundidade | 2 | (1, 3) | 0.0000 |
| N4 | corte_profundidade | 2 | (1, 1) | 0.0222 |
| N5 | corte_profundidade | 2 | (3, 1) | 0.0278 |
| N7 | corte_profundidade | 2 | (3, 1) | 0.0000 |
| N11 | corte_profundidade | 2 | (2, 4) | 0.0000 |
| N15 | corte_profundidade | 2 | (4, 2) | 0.0000 |

## Valores propagados
| No | Tipo | Profundidade | Jogada pai | Valor propagado | alfa_in -> alfa_out | beta_in -> beta_out |
| --- | --- | --- | --- | --- | --- | --- |
| N1 | MAX | 0 | ROOT | 0.0000 | -inf -> 0.0000 | +inf -> +inf |
| N2 | MIN | 1 | (1, 2) | 0.0000 | -inf -> -inf | +inf -> 0.0000 |
| N6 | MIN | 1 | (2, 1) | 0.0000 | 0.0000 -> 0.0000 | +inf -> 0.0000 |
| N10 | MIN | 1 | (3, 4) | 0.0000 | 0.0000 -> 0.0000 | +inf -> 0.0000 |
| N14 | MIN | 1 | (4, 3) | 0.0000 | 0.0000 -> 0.0000 | +inf -> 0.0000 |
| N3 | MAX | 2 | (1, 3) | 0.0000 | -inf -> -inf | +inf -> +inf |
| N4 | MAX | 2 | (1, 1) | 0.0222 | -inf -> -inf | 0.0000 -> 0.0000 |
| N5 | MAX | 2 | (3, 1) | 0.0278 | -inf -> -inf | 0.0000 -> 0.0000 |
| N7 | MAX | 2 | (3, 1) | 0.0000 | 0.0000 -> 0.0000 | +inf -> +inf |
| N11 | MAX | 2 | (2, 4) | 0.0000 | 0.0000 -> 0.0000 | +inf -> +inf |
| N15 | MAX | 2 | (4, 2) | 0.0000 | 0.0000 -> 0.0000 | +inf -> +inf |

## Podas Alfa-Beta
- N6 (MIN, profundidade 1): poda de [(1, 1), (1, 3)] com condicao alpha (0.0000) >= beta (0.0000).
- N10 (MIN, profundidade 1): poda de [(4, 4), (4, 2)] com condicao alpha (0.0000) >= beta (0.0000).
- N14 (MIN, profundidade 1): poda de [(4, 4), (2, 4)] com condicao alpha (0.0000) >= beta (0.0000).

## Arvore textual da busca
```text
+- N1 MAX d=0 move=ROOT a=-inf->0.0000 b=+inf->+inf v=0.0000 [INTERNAL]
   |- N2 MIN d=1 move=(1, 2) a=-inf->-inf b=+inf->0.0000 v=0.0000 [INTERNAL]
   |  |- N3 MAX d=2 move=(1, 3) a=-inf->-inf b=+inf->+inf v=0.0000 [DEPTH-CUTOFF leaf=0.0000]
   |  |- N4 MAX d=2 move=(1, 1) a=-inf->-inf b=0.0000->0.0000 v=0.0222 [DEPTH-CUTOFF leaf=0.0222]
   |  +- N5 MAX d=2 move=(3, 1) a=-inf->-inf b=0.0000->0.0000 v=0.0278 [DEPTH-CUTOFF leaf=0.0278]
   |- N6 MIN d=1 move=(2, 1) a=0.0000->0.0000 b=+inf->0.0000 v=0.0000 [INTERNAL]
   |  !!! PRUNED at N6: (1, 1), (1, 3) (alpha (0.0000) >= beta (0.0000))
   |  |- N7 MAX d=2 move=(3, 1) a=0.0000->0.0000 b=+inf->+inf v=0.0000 [DEPTH-CUTOFF leaf=0.0000]
   |  |- N8 MAX d=2 move=(1, 1) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
   |  +- N9 MAX d=2 move=(1, 3) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
   |- N10 MIN d=1 move=(3, 4) a=0.0000->0.0000 b=+inf->0.0000 v=0.0000 [INTERNAL]
   |  !!! PRUNED at N10: (4, 4), (4, 2) (alpha (0.0000) >= beta (0.0000))
   |  |- N11 MAX d=2 move=(2, 4) a=0.0000->0.0000 b=+inf->+inf v=0.0000 [DEPTH-CUTOFF leaf=0.0000]
   |  |- N12 MAX d=2 move=(4, 4) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
   |  +- N13 MAX d=2 move=(4, 2) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
   +- N14 MIN d=1 move=(4, 3) a=0.0000->0.0000 b=+inf->0.0000 v=0.0000 [INTERNAL]
      !!! PRUNED at N14: (4, 4), (2, 4) (alpha (0.0000) >= beta (0.0000))
      |- N15 MAX d=2 move=(4, 2) a=0.0000->0.0000 b=+inf->+inf v=0.0000 [DEPTH-CUTOFF leaf=0.0000]
      |- N16 MAX d=2 move=(4, 4) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
      +- N17 MAX d=2 move=(2, 4) a=0.0000->0.0000 b=0.0000->0.0000 v=- [PRUNED BRANCH]
```

## Arquivo DOT opcional
- Arquivo gerado: docs/alpha_beta_example.dot