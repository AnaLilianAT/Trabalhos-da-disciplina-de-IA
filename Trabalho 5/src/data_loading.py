"""Carregamento e validação dos CSVs do dataset BRFSS 2015.

Centraliza a leitura para que as demais etapas recebam DataFrames já
validados (schema correto, colunas esperadas presentes).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


# Número esperado de colunas: 21 features + 1 alvo.
EXPECTED_N_COLUMNS = len(config.ALL_FEATURES) + 1


def load_dataset(task: str) -> pd.DataFrame:
    """Carrega o CSV associado a uma tarefa e valida o schema.

    Parameters
    ----------
    task:
        Uma das chaves de ``config.DATASETS``: ``"binary"``,
        ``"multiclass"``, ``"binary_5050"`` ou ``"regression"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame com as colunas originais do CSV.
    """
    if task not in config.DATASETS:
        raise ValueError(
            f"Tarefa desconhecida: {task!r}. "
            f"Opções: {sorted(config.DATASETS)}"
        )

    path = config.DATASETS[task]["path"]
    target = config.DATASETS[task]["target"]
    df = load_csv(path)
    validate_schema(df, target)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lê um CSV bruto, com mensagem clara caso o arquivo não exista."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {path}\n"
            "Baixe os CSVs do Kaggle e coloque-os em data/raw/ "
            "(ver instruções no README)."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, target: str) -> None:
    """Valida que o DataFrame tem o número e os nomes de colunas esperados.

    Levanta ``ValueError`` em caso de divergência; isso evita que erros de
    schema se propaguem silenciosamente para as etapas seguintes.
    """
    if df.shape[1] != EXPECTED_N_COLUMNS:
        raise ValueError(
            f"Número de colunas inesperado: {df.shape[1]} "
            f"(esperado {EXPECTED_N_COLUMNS})."
        )

    if target not in df.columns:
        raise ValueError(f"Coluna alvo ausente: {target!r}.")

    # Toda coluna do CSV deve ser uma feature conhecida, o alvo da tarefa ou
    # uma das colunas-alvo nativas do dataset. Na regressão (alvo = BMI), o
    # arquivo binário ainda traz a coluna `Diabetes_binary`, que o plano
    # permite usar como feature extra — portanto não é "inesperada".
    native_targets = {config.TARGET_BINARY, config.TARGET_MULTICLASS}
    expected = set(config.ALL_FEATURES) | {target} | native_targets
    unexpected = set(df.columns) - expected
    if unexpected:
        raise ValueError(f"Colunas inesperadas no CSV: {sorted(unexpected)}.")


def get_features_target(df: pd.DataFrame, target: str):
    """Separa X (features) e y (alvo).

    Para a regressão (``target == 'BMI'``), o BMI é removido das features.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


if __name__ == "__main__":
    # Sanidade: carregar cada dataset e reportar dimensões.
    for _task in ("binary", "multiclass", "binary_5050"):
        _df = load_dataset(_task)
        _tgt = config.DATASETS[_task]["target"]
        print(f"{_task:14s} -> shape={_df.shape}, alvo={_tgt!r}")
