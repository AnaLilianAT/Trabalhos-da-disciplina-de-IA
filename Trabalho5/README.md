# Trabalho de Aprendizado Supervisionado — CDC Diabetes Health Indicators (BRFSS 2015)

Pipeline completo de Machine Learning sobre o dataset **CDC Diabetes Health Indicators
(BRFSS 2015)**, cobrindo três tarefas de aprendizado supervisionado:

1. **Classificação binária** — prever se a pessoa é diabética (`Diabetes_binary`).
2. **Classificação multiclasse** — prever `Diabetes_012` (0 = sem diabetes, 1 = pré-diabetes, 2 = diabetes).
3. **Regressão** — prever o **BMI** (índice de massa corporal) a partir dos demais indicadores.

O projeto cobre as Etapas 1–7 do enunciado (EDA, pré-processamento, seleção de features,
classificação e regressão com MLP, otimização com Optuna e análise de regularização/overfitting),
além de extras opcionais (SHAP e comparação com modelos clássicos).

## Estrutura do projeto

```
.
├── config.py                # caminhos, seeds, nomes de colunas, constantes
├── requirements.txt
├── data/
│   ├── raw/                 # CSVs originais do Kaggle (NÃO versionados)
│   └── processed/           # splits salvos (opcional)
├── src/                     # lógica reutilizável
│   ├── data_loading.py      # carregar CSVs e validar schema
│   ├── eda.py               # análise exploratória (Etapa 1)
│   └── plotting.py          # helpers de gráficos com estilo consistente
├── notebooks/               # um notebook por etapa, importando de src/
│   └── 01_eda.ipynb
├── scripts/                 # execução ponta a ponta
├── outputs/
│   ├── figures/             # .png para o relatório
│   ├── metrics/             # .json e .csv com métricas e tabelas
│   └── models/              # modelos salvos e best_params
└── report/                  # relatório técnico
```

## Dados

O dataset vem do **BRFSS 2015 (CDC)** e está disponível no Kaggle em três arquivos
(todos com 21 atributos + 1 alvo, já numéricos):

| Arquivo | Linhas | Alvo | Uso |
|---|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_binary` (0/1), desbalanceado | Classificação binária |
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_012` (0/1/2), desbalanceado | Classificação multiclasse |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70.692 | `Diabetes_binary` (0/1), balanceado 50/50 | Baseline/sanidade opcional |

A regressão usa qualquer um dos arquivos de 253k removendo `BMI` das features e usando-o como alvo.

### Como obter os dados

Os CSVs **não são versionados** (ver `.gitignore`). Para reproduzir:

1. Baixe do Kaggle: <https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset>
2. Coloque os três `.csv` em `data/raw/`.

## Como executar

```bash
pip install -r requirements.txt

# Etapa 1 — EDA (gera figuras em outputs/figures/ e métricas em outputs/metrics/)
python -m src.eda
```

Cada etapa também possui um notebook correspondente em `notebooks/` que importa a lógica de `src/`.

## Reprodutibilidade

- `SEED = 42` aplicado a `numpy`, `random` e `tensorflow` via `config.set_seeds()`.
- Caminhos, listas de colunas e proporções de split centralizados em `config.py`.
