# Plano de Implementação — Trabalho de Aprendizado Supervisionado (Diabetes)

> **Para o Claude Code:** este arquivo é a especificação completa do projeto. Implemente o pipeline conforme descrito, respeitando a estrutura de pastas, as decisões justificadas e a lista de artefatos (figuras, tabelas, métricas) que o relatório final precisa. Quando uma decisão estiver marcada como **[ESCOLHA RECOMENDADA]**, siga-a, mas mantenha o código parametrizável para troca fácil.

---

## 0. Visão geral

Construir um pipeline de Machine Learning sobre o dataset **CDC Diabetes Health Indicators (BRFSS 2015)**, cobrindo **três tarefas**:

1. **Classificação binária** — prever se a pessoa é diabética (`Diabetes_binary`).
2. **Classificação multiclasse** — prever `Diabetes_012` (0 = sem diabetes, 1 = pré-diabetes, 2 = diabetes).
3. **Regressão** — prever o **BMI** (índice de massa corporal) a partir dos demais indicadores de saúde.

O mesmo dataset (mesma fonte/famílias de arquivos) atende às três tarefas — isso deve ser explicitado e justificado no relatório.

**Stack:**
- Python 3.11
- TensorFlow/Keras para as MLPs **[ESCOLHA RECOMENDADA]** — facilita curva de aprendizado (`history`), `Dropout`, regularização L2 (`kernel_regularizer`), `EarlyStopping` e integração com Optuna. (PyTorch é alternativa válida; o plano é portável.)
- scikit-learn para split, scaling, seleção de features e modelos clássicos
- Optuna para otimização de hiperparâmetros
- SHAP (extra opcional), XGBoost (extra opcional)
- matplotlib/seaborn para gráficos

**Cobertura das etapas do enunciado:** este plano cobre Etapas 1–7 + extras (SHAP, comparação com modelos clássicos, HPO dos clássicos).

---

## 1. Dataset — descrição e obtenção

O dataset vem do BRFSS 2015 (CDC) e está disponível no Kaggle em três arquivos. **Use todos os três**, cada um para a tarefa mais adequada:

| Arquivo | Linhas | Alvo | Uso neste trabalho |
|---|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_binary` (0/1) — **desbalanceado** (~13,9% positivos) | **Classificação binária** (cenário realista, demonstra tratamento de desbalanceamento) |
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_012` (0/1/2) — fortemente desbalanceado (classe 1 ≈ 1,8%) | **Classificação multiclasse** |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70.692 | `Diabetes_binary` (0/1) — balanceado 50/50 | Comparação/sanidade opcional; mencionar no relatório |

> **Decisão:** usar a versão **desbalanceada** (253k) para a classificação binária dá mais conteúdo para discutir (desbalanceamento, ROC-AUC, PR-AUC, `class_weight`). A versão 50/50 pode ser citada como baseline simplificado.

### 1.1 Atributos (21 features) e tipos

Todos já estão numéricos. Tratar os tipos assim:

**Binárias (0/1):** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.

**Ordinais (inteiro com ordem):** `GenHlth` (1–5), `Age` (1–13, faixas etárias), `Education` (1–6), `Income` (1–8).

**Contínuas / contagem:** `BMI` (≈12–98, única verdadeiramente contínua), `MentHlth` (0–30 dias), `PhysHlth` (0–30 dias).

### 1.2 Alvos
- Binário: `Diabetes_binary`
- Multiclasse: `Diabetes_012`
- Regressão: `BMI` (remover `BMI` das features quando ele for o alvo)

### 1.3 Problemas conhecidos a documentar (Etapa 1)
- **Valores ausentes:** o dataset publicado já vem limpo (normalmente **sem NaN**). Implementar a verificação e a lógica de imputação mesmo assim (para cumprir o requisito), reportar que nenhum/poucos ausentes foram encontrados.
- **Classes desbalanceadas:** sim, tanto no binário quanto (severamente) no multiclasse — gerar gráfico de distribuição e discutir.
- **Outliers:** principalmente em `BMI` (valores muito altos) e zero-inflation em `MentHlth`/`PhysHlth`.
- **Correlação elevada:** `GenHlth`↔`PhysHlth`↔`DiffWalk`; `Education`↔`Income`; `HighBP`/`HighChol`/`HeartDiseaseorAttack`. Gerar heatmap de correlação.

### 1.4 Como obter os dados
1. Baixar manualmente do Kaggle: `https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset`
2. Colocar os 3 CSVs em `data/raw/`.
3. (Opcional) suportar `kagglehub` para download automático, com fallback claro caso não haja credenciais/rede.

> No README, documentar que os CSVs **não** são versionados no repositório (adicionar `data/raw/*.csv` ao `.gitignore`) e como obtê-los.

---

## 2. Estrutura do projeto

```
trabalho-diabetes-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                  # caminhos, seeds, nomes de colunas, constantes
├── data/
│   ├── raw/                    # CSVs originais do Kaggle (não versionar)
│   └── processed/              # splits salvos (opcional, .parquet)
├── src/
│   ├── __init__.py
│   ├── data_loading.py         # carregar CSVs, validar schema
│   ├── eda.py                  # análise exploratória + figuras Etapa 1
│   ├── preprocessing.py        # split, imputação, encoding, scaling (sem leakage)
│   ├── feature_selection.py    # técnica única de seleção (Etapa 3)
│   ├── models.py               # construção das MLPs (binária/multi/regressão)
│   ├── train.py                # rotinas de treino + histórico
│   ├── evaluation.py           # métricas + matriz de confusão + curvas
│   ├── optuna_tuning.py        # HPO (Etapa 6)
│   ├── regularization.py       # experimentos da Etapa 7
│   ├── classical_models.py     # extras: RF, SVM, XGBoost, LogReg/Linear/Ridge/Lasso
│   ├── shap_analysis.py        # extra opcional
│   └── plotting.py             # helpers de gráficos com estilo consistente
├── notebooks/                  # um notebook por etapa, importando de src/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_classificacao_mlp.ipynb
│   ├── 05_regressao_mlp.ipynb
│   ├── 06_optuna.ipynb
│   └── 07_regularizacao.ipynb
├── scripts/
│   └── run_all.py              # executa o pipeline ponta a ponta e salva artefatos
├── outputs/
│   ├── figures/                # todos os .png para o relatório
│   ├── metrics/                # .json e .csv com métricas e tabelas
│   └── models/                 # modelos salvos (.keras) e best_params (.json)
└── report/
    └── relatorio.md            # esqueleto do relatório técnico
```

> **Organização recomendada:** lógica reutilizável em `src/`; notebooks por etapa só orquestram e renderizam (bons para o relatório); `scripts/run_all.py` reproduz tudo de forma não interativa. Toda figura é salva em `outputs/figures/` com nome descritivo (`etapaX_<descricao>.png`).

### 2.1 requirements.txt (fixar versões)
```
numpy
pandas
scikit-learn
tensorflow            # ou tensorflow-cpu
optuna
matplotlib
seaborn
shap                  # extra
xgboost               # extra
imbalanced-learn      # opcional (SMOTE)
kagglehub             # opcional (download)
jupyter
```

### 2.2 Reprodutibilidade (`config.py`)
- `SEED = 42` aplicado a `numpy`, `random`, `tensorflow`.
- Centralizar caminhos, listas de colunas (binárias/ordinais/contínuas), proporções de split.
- Função `set_seeds()` chamada no início de cada script/notebook.

---

## 3. Decisões globais (justificar no relatório)

1. **Split estratificado train/val/test = 60/20/20** com `stratify` no alvo (para classificação). Para regressão, split simples (sem stratify) ou stratify por faixas de `BMI`.
   - `train`: ajustar scaler, seleção de features e treinar.
   - `val`: early stopping, curvas de aprendizado e objetivo do Optuna.
   - `test`: **somente** avaliação final, uma vez.
2. **Prevenção de vazamento (leakage):** `StandardScaler`, imputador e seletor de features são `fit` **apenas no treino** e aplicados (`transform`) em val/test. Usar `Pipeline`/`ColumnTransformer` do sklearn sempre que possível.
3. **Scaling [ESCOLHA RECOMENDADA]:** `StandardScaler` nas features ordinais e contínuas; binárias podem passar direto (`passthrough`) via `ColumnTransformer`. Justificar `RobustScaler` como alternativa para `BMI`/`MentHlth`/`PhysHlth` por causa de outliers — implementar como opção configurável.
4. **Encoding:** as features já estão codificadas (binárias 0/1; ordinais com ordem preservada → manter como inteiro, **não** aplicar One-Hot para não perder a ordem). Documentar essa decisão. Para o alvo multiclasse no Keras, usar `sparse_categorical_crossentropy` (rótulos inteiros 0/1/2 — não precisa one-hot).
5. **Desbalanceamento:** usar `class_weight='balanced'` no treino das classificações; reportar métricas robustas a desbalanceamento (ROC-AUC, PR-AUC, F1, macro-F1). Mencionar SMOTE como alternativa (opcional).

---

## 4. Etapa 1 — EDA (`src/eda.py`, `notebooks/01_eda.ipynb`)

Gerar e salvar:
- Tabela-resumo: nº de amostras, nº de atributos, tipos, alvo de cada tarefa.
- `df.describe()` e contagem de nulos por coluna → `outputs/metrics/eda_summary.csv`.
- **Distribuição das classes** (binária e multiclasse) → `etapa1_dist_classes_binaria.png`, `etapa1_dist_classes_multiclasse.png`.
- **Histograma do BMI** + boxplot (mostra outliers) → `etapa1_bmi_dist.png`.
- **Heatmap de correlação** (Pearson para numéricas; pode usar todas) → `etapa1_correlacao.png`.
- Boxplots de algumas features vs alvo (ex.: `GenHlth`, `BMI`, `Age` por `Diabetes_binary`) → `etapa1_features_vs_target.png`.
- Texto/markdown discutindo: ausentes, desbalanceamento, outliers, correlações altas (listar pares com |corr| > 0,5).

---

## 5. Etapa 2 — Pré-processamento (`src/preprocessing.py`)

Implementar função `build_preprocessor()` retornando um `ColumnTransformer` e uma função `split_data(df, target, task)`:
- Verificar e tratar ausentes: `SimpleImputer(strategy='median')` para contínuas/ordinais e `most_frequent` para binárias (mesmo que não disparem — cumprir requisito e reportar).
- Encoding: conforme decisão da seção 3 (manter ordinais; binárias 0/1).
- Scaling: `ColumnTransformer` com `StandardScaler` em ordinais+contínuas e `passthrough` em binárias.
- Tratamento de outliers (opcional): clipping de `BMI` no percentil 99 — implementar como flag; documentar impacto.

Salvar, para o relatório, um quadro com **cada escolha + justificativa** (`outputs/metrics/preprocessing_choices.md`).

---

## 6. Etapa 3 — Seleção de features (`src/feature_selection.py`)

**Técnica única [ESCOLHA RECOMENDADA]: Mutual Information** (`mutual_info_classif` para classificação, `mutual_info_regression` para regressão).

**Justificativa:** lida bem com a mistura de variáveis binárias/ordinais/contínuas, captura relações **não lineares** (coerente com o uso de MLP), não assume linearidade nem normalidade, e tem variante para classificação e regressão (consistência entre as três tarefas). (Alternativa defensável: Random Forest Feature Importance — deixar implementada como opção.)

Implementar:
- Calcular MI no **conjunto de treino** (após scaling), produzir **ranking** de importância → salvar `outputs/metrics/feature_ranking_<tarefa>.csv` e gráfico de barras `etapa3_mi_ranking_<tarefa>.png`.
- **Critério de corte:** manter as top-k features cuja MI acumulada represente ~90% da MI total **ou** descartar features com MI ≈ 0; reportar quantidade inicial (21 ou 20 para regressão) e final.
- Salvar a lista de features selecionadas por tarefa.

**Comparação obrigatória (todas vs selecionadas):**
- Treinar a MLP baseline (mesma arquitetura/seed) duas vezes: (a) todas as features, (b) só as selecionadas.
- Comparar: métrica principal, sinais de overfitting (gap treino-val) e **tempo de treinamento** (cronometrar) → tabela `outputs/metrics/feature_selection_comparison.csv` + gráfico.
- Discutir: desempenho, overfitting, tempo, interpretabilidade.

(Fazer essa comparação ao menos para a tarefa binária; repetir para as demais é desejável.)

---

## 7. Etapa 4 — Classificação com MLP (`src/models.py`, `src/train.py`, `src/evaluation.py`)

Implementar **duas** MLPs: binária e multiclasse.

### 7.1 Arquitetura baseline (justificar cada escolha)
**Binária:**
- `Input(n_features)` → `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` → `Dense(1, sigmoid)`
- Loss: `binary_crossentropy`; Otimizador: `Adam(learning_rate=1e-3)`
- `epochs=100` com `EarlyStopping(patience=10, restore_best_weights=True)`; `batch_size=256`; `class_weight='balanced'`.

**Multiclasse:**
- igual, mas saída `Dense(3, softmax)`; loss `sparse_categorical_crossentropy`; `class_weight` por classe.

Justificar no relatório: nº de camadas, neurônios, ativações (ReLU nas ocultas, sigmoid/softmax na saída), otimizador, learning rate, épocas, batch size.

### 7.2 Avaliação
**Binária:** Accuracy, Precision, Recall, F1, **ROC-AUC**, matriz de confusão. (Adicionar PR-AUC por causa do desbalanceamento.)
**Multiclasse:** Accuracy, Precision macro, Recall macro, **F1 macro**, matriz de confusão.

Gráficos:
- **Curva de aprendizado** (loss treino × val por época) → `etapa4_loss_<tarefa>.png`.
- Evolução da métrica principal por época → `etapa4_metric_<tarefa>.png`.
- Matriz de confusão (normalizada) → `etapa4_confmat_<tarefa>.png`.
- Curva ROC (binária) → `etapa4_roc_binaria.png`.

Salvar métricas em `outputs/metrics/classification_metrics.json`.

### 7.3 Extra (recomendado): comparação com modelo clássico
Treinar `RandomForestClassifier`, `LogisticRegression` e (extra) `XGBoost` nas mesmas features; comparar métricas e **custo computacional** (tempo de treino/inferência) → tabela + discussão.

---

## 8. Etapa 5 — Regressão com MLP (`src/models.py`)

**Alvo:** `BMI` (remover `BMI` das features; demais 20 features + opcionalmente o status de diabetes como feature).

### 8.1 Arquitetura baseline
- `Input(n_features)` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(32, relu)` → `Dense(1, linear)`
- Loss: `mse`; métrica `mae`; `Adam(1e-3)`; `EarlyStopping`; `batch_size=256`.

> **Expectativa realista:** R² de BMI a partir desses indicadores tende a ser **modesto** (≈0,10–0,30). Isso é esperado e deve ser discutido honestamente — os indicadores explicam só parte da variância do BMI.

### 8.2 Avaliação
Métricas: **MAE, MSE, RMSE, R²** → `outputs/metrics/regression_metrics.json`.
Gráficos:
- Valores reais × preditos (scatter + linha y=x) → `etapa5_real_vs_pred.png`.
- Resíduos (resíduo × predito e histograma de resíduos) → `etapa5_residuos.png`.
- Curva de aprendizado (loss por época) → `etapa5_loss.png`.

### 8.3 Extra (recomendado): comparação
`LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, (extra) `XGBRegressor`; comparar MAE/RMSE/R² e discutir vantagens/desvantagens.

---

## 9. Etapa 6 — Otimização com Optuna (`src/optuna_tuning.py`)

Otimizar a MLP de **cada** tarefa (priorizar a binária; aplicar às demais se houver tempo).

### 9.1 Espaço de busca sugerido (justificar)
- `n_layers`: 1–4
- `units_l{i}`: {32, 64, 128, 256}
- `activation`: {relu, tanh, elu}
- `dropout`: 0.0–0.5
- `learning_rate`: log-uniforme 1e-4 … 1e-2
- `optimizer`: {adam, rmsprop, sgd} (+ `momentum` 0.0–0.9 se sgd)
- `batch_size`: {64, 128, 256, 512}
- `l2` (weight decay): log-uniforme 1e-6 … 1e-2
- épocas fixas (ex.: 100) com EarlyStopping; usar **pruning** (`MedianPruner` + callback de pruning do Keras por época)

### 9.2 Função objetivo
- Binária: **maximizar ROC-AUC** na validação.
- Multiclasse: **maximizar F1 macro** na validação.
- Regressão: **minimizar RMSE** na validação.
- `n_trials = 40` (configurável). Salvar `study` e histórico.

### 9.3 Entregáveis
- Espaço de busca documentado.
- Nº de trials executados.
- **Melhor conjunto de hiperparâmetros** → `outputs/models/best_params_<tarefa>.json`.
- Valor da função objetivo.
- Gráficos do Optuna: histórico de otimização e importância dos hiperparâmetros → `etapa6_optuna_history_<tarefa>.png`, `etapa6_optuna_param_importance_<tarefa>.png`.
- **Comparação modelo original × otimizado** (mesma métrica, no conjunto de teste) → tabela `outputs/metrics/optuna_comparison.csv`.
- Discutir: ganho de desempenho, impacto no tempo de treino, hiperparâmetros mais influentes.

> Após o estudo, **retreinar** o modelo final com os melhores params (em train, validando em val) e avaliar **uma vez** no test.

---

## 10. Etapa 7 — Regularização e overfitting (`src/regularization.py`)

Comparar, na tarefa de classificação:
- **Rede SEM regularização** (sem dropout, sem L2, sem early stopping — deixar treinar muitas épocas para induzir overfitting).
- **Rede COM regularização** — aplicar **pelo menos uma** técnica; recomendado combinar **Dropout + L2 (weight decay) + Early Stopping**. Documentar cada uma.

Entregáveis:
- Curvas de treino × validação das duas redes (lado a lado) → `etapa7_curvas_comparacao.png`.
- Tabela comparando métricas finais (treino vs teste) das duas → `outputs/metrics/regularization_comparison.csv`.
- Discussão respondendo explicitamente:
  1. A rede apresentou sinais de overfitting? (mostrar gap treino-val)
  2. Qual técnica de regularização foi usada?
  3. Houve melhoria em dados não vistos (teste)?
  4. Qual estratégia teve melhor equilíbrio desempenho × generalização?

---

## 11. Extras opcionais (pontuação adicional)

### 11.1 SHAP (`src/shap_analysis.py`)
- `SHAP Summary Plot` (importância global) → `extra_shap_summary.png`.
- Explicação **local** de ≥1 predição (`force`/`waterfall plot`).
- Comparar features mais relevantes do SHAP × selecionadas na Etapa 3; discutir se o SHAP corrobora a seleção.
- Para MLP, usar `KernelExplainer`/`DeepExplainer` em uma amostra (ex.: 1.000 linhas) por custo computacional.

### 11.2 HPO dos modelos clássicos
- Otimizar (Optuna ou GridSearch) RF/XGBoost e comparar com a MLP otimizada.

---

## 12. Artefatos esperados (checklist para o relatório)

**Figuras** (`outputs/figures/`): distribuição de classes (bin/multi), distribuição/boxplot BMI, heatmap de correlação, ranking de MI por tarefa, curvas de loss (4/5/7), métrica por época, matrizes de confusão, curva ROC, real×predito, resíduos, histórico Optuna, importância de hiperparâmetros, comparação de regularização, (extra) SHAP.

**Tabelas/JSON** (`outputs/metrics/`): resumo EDA, escolhas de pré-processamento, ranking de features, comparação todas×selecionadas, métricas de classificação/regressão, comparação original×otimizado, comparação de regularização, comparação com modelos clássicos.

**Modelos** (`outputs/models/`): `.keras` finais + `best_params_*.json`.

---

## 13. Esqueleto do relatório técnico (`report/relatorio.md`)

1. Introdução e descrição do problema
2. Etapa 1 — Dataset (procedência/justificativa: BRFSS/CDC, uso em literatura; amostras, atributos, alvos, distribuição, problemas)
3. Etapa 2 — Pré-processamento (com justificativas)
4. Etapa 3 — Seleção de features (técnica, justificativa, ranking, critério, comparação todas×selecionadas)
5. Etapa 4 — Classificação com MLP (arquitetura justificada, métricas, gráficos)
6. Etapa 5 — Regressão com MLP
7. Etapa 6 — Optuna (espaço, trials, melhores params, comparação)
8. Etapa 7 — Regularização e overfitting (4 perguntas respondidas)
9. Extras (SHAP, modelos clássicos)
10. Conclusões
11. Como reproduzir (README resumido)

---

## 14. Ordem de execução recomendada

1. `config.py` + `requirements.txt` + `README.md` + `.gitignore`
2. `src/data_loading.py` → carregar e validar
3. `src/eda.py` + notebook 01
4. `src/preprocessing.py` + notebook 02
5. `src/feature_selection.py` + notebook 03 (inclui comparação todas×selecionadas)
6. `src/models.py` + `src/train.py` + `src/evaluation.py` → notebooks 04 e 05
7. `src/optuna_tuning.py` → notebook 06
8. `src/regularization.py` → notebook 07
9. Extras (`classical_models.py`, `shap_analysis.py`)
10. `scripts/run_all.py` (reprodução ponta a ponta) + relatório

## 15. Critérios de qualidade
- Sem vazamento de dados (fit só no treino).
- Seeds fixas; resultados reprodutíveis.
- Toda decisão acompanhada de justificativa no relatório.
- Código modular e comentado; figuras com títulos/eixos legíveis e salvas em alta resolução (`dpi=150`).
- Avaliação final no teste feita **uma única vez** por tarefa.
