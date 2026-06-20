# Etapa 3 — Seleção de Features (Mutual Information)

**Técnica:** Mutual Information (`mutual_info_classif` / `mutual_info_regression`), calculada sobre o conjunto de treino.

**Justificativa:** captura dependências não lineares (coerente com a MLP), lida com variáveis binárias/ordinais/contínuas e tem variante para classificação e regressão.

**Critério de corte:** manter as top-k features cuja MI acumulada atinge 90% da MI total.

## Quantidade de features por tarefa

| Tarefa | Inicial | Selecionadas | Descartadas |
| --- | --- | --- | --- |
| binary | 21 | 11 | 10 |
| multiclass | 21 | 11 | 10 |
| regression | 21 | 12 | 9 |

## Comparação todas × selecionadas

| tarefa | conjunto | n_features | metrica_treino | metrica_val | metrica_teste | gap_treino_val | tempo_treino_s | epocas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | todas | 21 | 0.8358 | 0.8277 | 0.8262 | 0.0081 | 29.59 | 31 |
| binary | selecionadas | 11 | 0.829 | 0.8229 | 0.8215 | 0.0061 | 24.43 | 26 |
| multiclass | todas | 21 | 0.428 | 0.4236 | 0.4235 | 0.0044 | 22.84 | 25 |
| multiclass | selecionadas | 11 | 0.4244 | 0.4194 | 0.4181 | 0.005 | 37.84 | 34 |
| regression | todas | 21 | 0.1811 | 0.1801 | 0.1736 | 0.0009 | 95.26 | 76 |
| regression | selecionadas | 12 | 0.1745 | 0.1734 | 0.1663 | 0.0011 | 125.81 | 100 |

> Métrica principal: ROC-AUC (binária), F1 macro (multiclasse), R² (regressão).

## Discussão (orientada pelos resultados)

- **binary** (21→11 features): métrica no teste 0.8262→0.8215 (Δ=-0.0047); gap treino-val 0.0081→0.0061 (Δ=-0.0020); tempo 29.59s→24.43s (Δ=-5.16s, em 31→26 épocas).
- **multiclass** (21→11 features): métrica no teste 0.4235→0.4181 (Δ=-0.0054); gap treino-val 0.0044→0.005 (Δ=+0.0006); tempo 22.84s→37.84s (Δ=+15.00s, em 25→34 épocas).
- **regression** (21→12 features): métrica no teste 0.1736→0.1663 (Δ=-0.0073); gap treino-val 0.0009→0.0011 (Δ=+0.0002); tempo 95.26s→125.81s (Δ=+30.55s, em 76→100 épocas).

**Leitura geral:**

- **Desempenho:** a seleção mantém a métrica praticamente inalterada (quedas pequenas, da ordem de 0,5–1 ponto), usando ~metade das features.
- **Overfitting:** o `gap_treino_val` é pequeno em todas as tarefas (forte regularização do early stopping + class_weight); a seleção não o aumenta de forma relevante.
- **Tempo de treino:** menos features reduzem o nº de parâmetros e o custo *por época*, mas o tempo de parede total depende de quantas épocas o early stopping permite. Na binária o tempo caiu; na multiclasse e na regressão o modelo selecionado convergiu em **mais épocas**, elevando o tempo total — um efeito do early stopping, não do tamanho do modelo.
- **Interpretabilidade:** menos atributos tornam o modelo mais fácil de explicar; o ranking de MI já evidencia os fatores mais associados ao alvo (`GenHlth`, `HighBP`, `BMI`, `Age` para diabetes).
- **Regressão:** o R² baixo (~0,17) é esperado — os indicadores de saúde explicam apenas parte da variância do BMI.
