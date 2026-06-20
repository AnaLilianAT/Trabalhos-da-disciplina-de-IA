# Etapa 1 — Discussão da Análise Exploratória

## Visão geral do dataset

- **Fonte:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015.
- **Amostras:** 253,680 (arquivos de 253k); 21 atributos preditores + 1 alvo.
- Três tarefas sobre a mesma família de dados: classificação binária (`Diabetes_binary`), multiclasse (`Diabetes_012`) e regressão (`BMI`).

## Valores ausentes

- Total de células ausentes no dataset binário: **0**.
- O dataset publicado já vem limpo; ainda assim, o pipeline de pré-processamento implementa imputação (mediana/moda) para cumprir o requisito e ser robusto a novas amostras.

## Desbalanceamento de classes

**Binária (`Diabetes_binary`):**

| classe | contagem | percentual |
| --- | --- | --- |
| 0.0 | 218334 | 86.07 |
| 1.0 | 35346 | 13.93 |

**Multiclasse (`Diabetes_012`):**

| classe | contagem | percentual |
| --- | --- | --- |
| 0.0 | 213703 | 84.24 |
| 1.0 | 4631 | 1.83 |
| 2.0 | 35346 | 13.93 |

- O alvo binário é desbalanceado (~14% positivos) e o multiclasse é severamente desbalanceado (a classe de pré-diabetes representa <2%). Isso motiva o uso de `class_weight='balanced'` e de métricas robustas (ROC-AUC, PR-AUC, F1 macro) nas etapas de classificação.

## Outliers

- `BMI`: faixa observada 12–98; ~**9,820** valores acima do limite superior do IQR (41.5), concentrados em BMIs muito altos.
- `MentHlth` e `PhysHlth` (dias) apresentam forte concentração em zero (zero-inflation), com cauda até 30.

## Correlações elevadas

Pares com |correlação de Pearson| ≥ 0.5:

| feature_a | feature_b | correlacao |
| --- | --- | --- |
| GenHlth | PhysHlth | 0.524 |

- Apenas `GenHlth`↔`PhysHlth` cruza o limiar de 0,5. Logo abaixo dele (faixa ~0,4–0,5) aparecem pares correlacionados de forma moderada, como o eixo de saúde geral (`GenHlth`↔`DiffWalk`, `PhysHlth`↔`DiffWalk`) e o eixo socioeconômico (`Education`↔`Income`). Não há multicolinearidade extrema, mas essa redundância parcial reforça a utilidade da seleção de features na Etapa 3.
