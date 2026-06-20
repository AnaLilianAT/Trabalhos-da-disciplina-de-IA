"""Configuração central do projeto: caminhos, seeds, nomes de colunas e constantes.

Importado por todos os módulos de `src/` e pelos notebooks para garantir
reprodutibilidade e evitar a duplicação de constantes pelo código.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fixa as seeds de `random`, `numpy` e (se disponível) `tensorflow`.

    Deve ser chamada no início de cada script/notebook. O TensorFlow é
    importado de forma preguiçosa para que a EDA não dependa dele.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow só é necessário a partir da Etapa 4.
        pass


# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

REPORT_DIR = ROOT / "report"

# Garante que as pastas de saída existam ao importar a config.
for _d in (PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Arquivos do dataset (BRFSS 2015 / CDC)
# --------------------------------------------------------------------------- #
CSV_BINARY = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
CSV_MULTICLASS = RAW_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
CSV_BINARY_5050 = RAW_DIR / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# Mapa tarefa -> (arquivo, coluna alvo). A regressão reutiliza o arquivo binário.
DATASETS = {
    "binary": {"path": CSV_BINARY, "target": "Diabetes_binary"},
    "multiclass": {"path": CSV_MULTICLASS, "target": "Diabetes_012"},
    "binary_5050": {"path": CSV_BINARY_5050, "target": "Diabetes_binary"},
    "regression": {"path": CSV_BINARY, "target": "BMI"},
}

# --------------------------------------------------------------------------- #
# Colunas e seus tipos (todas já numéricas no dataset publicado)
# --------------------------------------------------------------------------- #
TARGET_BINARY = "Diabetes_binary"
TARGET_MULTICLASS = "Diabetes_012"
TARGET_REGRESSION = "BMI"

# Binárias (0/1)
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

# Ordinais (inteiro com ordem)
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]

# Contínuas / contagem
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

# Todas as 21 features (ordem das colunas do CSV, menos o alvo)
ALL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Rótulos legíveis para os alvos (usados em gráficos)
CLASS_LABELS_BINARY = {0: "Não diabético", 1: "Diabético"}
CLASS_LABELS_MULTICLASS = {0: "Sem diabetes", 1: "Pré-diabetes", 2: "Diabetes"}

# --------------------------------------------------------------------------- #
# Split e treino
# --------------------------------------------------------------------------- #
# Split 60/20/20 (train/val/test). TEST_SIZE separado primeiro; VAL_SIZE é a
# fração do restante (0.25 * 0.8 = 0.2 do total).
TEST_SIZE = 0.20
VAL_SIZE = 0.25

# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
FIG_DPI = 150
