"""Etapa 2 — Pré-processamento.

Implementa o split estratificado 60/20/20, a imputação de ausentes, o
encoding (mantendo a codificação numérica existente) e o scaling, tudo
encapsulado em um ``ColumnTransformer`` do scikit-learn para evitar
vazamento de dados (o ``fit`` ocorre apenas no conjunto de treino).

Funções principais:
- ``split_data``       : separa train/val/test (estratificado).
- ``build_preprocessor``: monta o ``ColumnTransformer`` para uma tarefa.
- ``prepare_data``     : orquestra load -> split -> fit no treino -> transform.

Execução de sanidade e geração do quadro de justificativas:
    python -m src.preprocessing
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import config
from src import data_loading


# --------------------------------------------------------------------------- #
# Transformador de outliers (clipping por percentil, sem leakage)
# --------------------------------------------------------------------------- #
class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clipa os valores no percentil superior aprendido **no treino**.

    Usado opcionalmente para atenuar os outliers de `BMI`. O limite é
    estimado em ``fit`` (somente treino) e reaplicado em val/test, evitando
    vazamento de informação.
    """

    def __init__(self, upper_percentile: float = 99.0):
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.minimum(X, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        """Clipping não altera as colunas — repassa os nomes de entrada.

        Necessário para que ``Pipeline``/``ColumnTransformer`` consigam
        montar ``get_feature_names_out`` quando o clipper está no pipeline.
        """
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(
            [f"x{i}" for i in range(getattr(self, "n_features_in_", 1))],
            dtype=object,
        )


# --------------------------------------------------------------------------- #
# Classificação das colunas por tipo
# --------------------------------------------------------------------------- #
def classify_columns(columns: Sequence[str]):
    """Separa as colunas em (binárias, ordinais, contínuas).

    Trata `Diabetes_binary`/`Diabetes_012` como binárias/ordinais quando
    aparecem como **feature** (caso da regressão, que usa o arquivo binário
    e mantém `Diabetes_binary` como preditor).
    """
    binary, ordinal, continuous = [], [], []
    for c in columns:
        if c in config.BINARY_FEATURES or c == config.TARGET_BINARY:
            binary.append(c)
        elif c in config.ORDINAL_FEATURES or c == config.TARGET_MULTICLASS:
            ordinal.append(c)
        elif c in config.CONTINUOUS_FEATURES:
            continuous.append(c)
        else:
            raise ValueError(f"Coluna sem tipo definido: {c!r}")
    return binary, ordinal, continuous


# --------------------------------------------------------------------------- #
# Split estratificado 60/20/20
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, target: str, task: str,
               test_size: float = config.TEST_SIZE,
               val_size: float = config.VAL_SIZE,
               seed: int = config.SEED):
    """Separa X/y em train/val/test (60/20/20).

    - Classificação: estratifica pelo próprio alvo.
    - Regressão: estratifica por faixas (decis) de `BMI`, para que treino,
      validação e teste tenham distribuições de alvo semelhantes.
    """
    X = df.drop(columns=[target])
    y = df[target]

    if task == "regression":
        # Estratificação por decis do alvo contínuo.
        strat = pd.qcut(y, q=10, labels=False, duplicates="drop")
    else:
        strat = y

    # 1) separa o teste (20%)
    X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(
        X, y, strat, test_size=test_size, random_state=seed, stratify=strat)

    # 2) separa a validação a partir do restante (val_size do restante = 20% do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=strat_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
# Pré-processador (ColumnTransformer)
# --------------------------------------------------------------------------- #
def build_preprocessor(feature_columns: Sequence[str],
                       scaler: str = "standard",
                       clip_bmi: bool = False) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` de pré-processamento para as features dadas.

    Parameters
    ----------
    feature_columns:
        Colunas preditoras (sem o alvo).
    scaler:
        ``"standard"`` (padrão) ou ``"robust"`` (mais robusto a outliers,
        sugerido como alternativa para BMI/MentHlth/PhysHlth).
    clip_bmi:
        Se ``True`` e ``BMI`` estiver entre as features, aplica clipping no
        percentil 99 antes do scaling.

    Estratégia (justificativas em ``preprocessing_choices.md``):
    - Binárias: imputação por moda; **passthrough** (sem scaling).
    - Ordinais + contínuas: imputação por mediana; scaling.
    - A codificação numérica original é mantida (ordinais preservam a ordem;
      não se aplica One-Hot).
    """
    binary, ordinal, continuous = classify_columns(feature_columns)
    scaler_cls = {"standard": StandardScaler, "robust": RobustScaler}[scaler]

    transformers = []

    # Binárias: só imputação (moda); sem scaling.
    if binary:
        transformers.append((
            "bin",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
            binary,
        ))

    # BMI com clipping opcional (pipeline próprio para isolar o clipper).
    num_cols = ordinal + continuous
    if clip_bmi and "BMI" in continuous:
        num_cols = [c for c in num_cols if c != "BMI"]
        transformers.append((
            "bmi",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clip", PercentileClipper(99.0)),
                ("scaler", scaler_cls()),
            ]),
            ["BMI"],
        ))

    # Demais numéricas (ordinais + contínuas): mediana + scaling.
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler_cls()),
            ]),
            num_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- #
# Orquestração: load -> split -> fit(treino) -> transform
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Conjuntos já pré-processados e metadados associados."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    task: str
    target: str


def prepare_data(task: str, scaler: str = "standard", clip_bmi: bool = False,
                 features: Sequence[str] | None = None) -> PreparedData:
    """Pipeline completo de pré-processamento para uma tarefa.

    O ``fit`` do pré-processador ocorre **somente no treino**; val e test são
    apenas transformados (prevenção de vazamento).

    Parameters
    ----------
    task:
        ``"binary"``, ``"multiclass"`` ou ``"regression"``.
    features:
        Subconjunto opcional de features (usado na Etapa 3 para comparar
        "todas" vs "selecionadas"). Se ``None``, usa todas as features.
    """
    config.set_seeds(config.SEED)
    df = data_loading.load_dataset(task)
    target = config.DATASETS[task]["target"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target, task)

    if features is not None:
        features = list(features)
        X_train, X_val, X_test = X_train[features], X_val[features], X_test[features]

    feature_columns = list(X_train.columns)
    pre = build_preprocessor(feature_columns, scaler=scaler, clip_bmi=clip_bmi)

    X_train_t = pre.fit_transform(X_train)          # fit SÓ no treino
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    return PreparedData(
        X_train=np.asarray(X_train_t, dtype=np.float32),
        X_val=np.asarray(X_val_t, dtype=np.float32),
        X_test=np.asarray(X_test_t, dtype=np.float32),
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=list(pre.get_feature_names_out()),
        preprocessor=pre,
        task=task,
        target=target,
    )


def compute_class_weights(y) -> dict:
    """Pesos balanceados por classe (para `class_weight` no treino, Etapa 4)."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# --------------------------------------------------------------------------- #
# Quadro de justificativas (Etapa 2) -> outputs/metrics/preprocessing_choices.md
# --------------------------------------------------------------------------- #
def write_preprocessing_choices(filename: str = "preprocessing_choices.md"):
    """Gera o quadro 'escolha + justificativa' exigido pelo enunciado."""
    lines = [
        "# Etapa 2 — Escolhas de Pré-processamento e Justificativas",
        "",
        "| Aspecto | Escolha | Justificativa |",
        "| --- | --- | --- |",
        "| Split | Estratificado 60/20/20 (train/val/test) | Treino para ajuste; "
        "validação para early stopping/Optuna; teste avaliado **uma única vez**. "
        "Estratificação preserva a proporção das classes (e dos decis de BMI na "
        "regressão) nos três conjuntos, importante dado o desbalanceamento. |",
        "| Prevenção de vazamento | `fit` apenas no treino | Imputador, scaler e "
        "(opcional) clipper aprendem estatísticas só do treino e são aplicados via "
        "`transform` em val/test, encapsulados em `ColumnTransformer`/`Pipeline`. |",
        "| Valores ausentes (contínuas/ordinais) | `SimpleImputer(strategy='median')` "
        "| A mediana é robusta a outliers (relevantes em BMI/MentHlth/PhysHlth). O "
        "dataset não tem ausentes, mas a etapa cumpre o requisito e dá robustez a "
        "novas amostras. |",
        "| Valores ausentes (binárias) | `SimpleImputer(strategy='most_frequent')` | "
        "Moda é a estatística adequada para variáveis 0/1. |",
        "| Encoding | Manter codificação numérica original | Binárias já são 0/1; "
        "ordinais (`GenHlth`, `Age`, `Education`, `Income`) têm ordem natural — "
        "**não** se aplica One-Hot para não perder a ordem nem inflar a dimensão. |",
        "| Scaling | `StandardScaler` em ordinais+contínuas; binárias `passthrough` "
        "| Padroniza escalas distintas (ex.: BMI ~12–98 vs. dias 0–30), ajudando a "
        "convergência da MLP. Padronizar variáveis 0/1 é desnecessário e prejudica a "
        "interpretabilidade, por isso passam direto. |",
        "| Alternativa de scaling | `RobustScaler` (configurável) | Usa mediana/IQR; "
        "alternativa menos sensível aos outliers de BMI/MentHlth/PhysHlth. |",
        "| Outliers (opcional) | Clipping de BMI no percentil 99 (flag `clip_bmi`) | "
        "Atenua a cauda extrema do BMI sem descartar amostras; limite aprendido no "
        "treino. Desativado por padrão; impacto documentado. |",
        "| Desbalanceamento | `class_weight='balanced'` no treino (Etapa 4) | "
        "Compensa as classes minoritárias sem alterar os dados; complementado por "
        "métricas robustas (ROC-AUC, PR-AUC, F1 macro). |",
        "",
    ]
    path = config.METRICS_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    write_preprocessing_choices()
    print("Quadro de justificativas salvo em outputs/metrics/preprocessing_choices.md\n")

    for _task in ("binary", "multiclass", "regression"):
        data = prepare_data(_task)
        print(
            f"{_task:11s} -> X_train={data.X_train.shape}, "
            f"X_val={data.X_val.shape}, X_test={data.X_test.shape}, "
            f"n_features={len(data.feature_names)}"
        )
        if _task != "regression":
            print(f"              class_weights={compute_class_weights(data.y_train)}")
