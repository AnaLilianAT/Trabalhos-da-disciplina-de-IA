"""Etapa 1 — Análise Exploratória de Dados (EDA).

Gera as figuras e tabelas exigidas pelo enunciado para descrever o dataset e
os problemas conhecidos (valores ausentes, desbalanceamento, outliers e
correlações elevadas). Os artefatos são salvos em ``outputs/figures/`` e
``outputs/metrics/``.

Execução:
    python -m src.eda
"""
from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src import data_loading, plotting

# Limiar para considerar uma correlação "elevada" (em valor absoluto).
HIGH_CORR_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Tabelas / métricas
# --------------------------------------------------------------------------- #
def dataset_overview() -> pd.DataFrame:
    """Tabela-resumo das três tarefas: amostras, atributos e alvo."""
    rows = []
    for task in ("binary", "multiclass", "regression"):
        info = config.DATASETS[task]
        df = data_loading.load_dataset(task)
        n_features = df.shape[1] - 1  # exclui o alvo
        rows.append(
            {
                "tarefa": task,
                "arquivo": info["path"].name,
                "n_amostras": df.shape[0],
                "n_atributos": n_features,
                "alvo": info["target"],
                "tipo_problema": {
                    "binary": "classificação binária",
                    "multiclass": "classificação multiclasse",
                    "regression": "regressão",
                }[task],
            }
        )
    return pd.DataFrame(rows)


def feature_types_table() -> pd.DataFrame:
    """Tabela com o tipo semântico de cada uma das 21 features."""
    rows = []
    for col in config.ALL_FEATURES:
        if col in config.BINARY_FEATURES:
            tipo = "binária (0/1)"
        elif col in config.ORDINAL_FEATURES:
            tipo = "ordinal (inteiro com ordem)"
        else:
            tipo = "contínua / contagem"
        rows.append({"feature": col, "tipo": tipo})
    return pd.DataFrame(rows)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """describe() transposto + contagem de nulos por coluna."""
    desc = df.describe().T
    desc["n_nulos"] = df.isna().sum()
    desc["pct_nulos"] = (df.isna().mean() * 100).round(4)
    return desc


def class_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Contagem e proporção de cada classe do alvo."""
    counts = df[target].value_counts().sort_index()
    pct = (df[target].value_counts(normalize=True).sort_index() * 100).round(2)
    return pd.DataFrame({"classe": counts.index, "contagem": counts.values, "percentual": pct.values})


def high_correlations(df: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> pd.DataFrame:
    """Pares de colunas com |correlação de Pearson| acima do limiar."""
    corr = df.corr(numeric_only=True)
    rows = []
    for a, b in combinations(corr.columns, 2):
        r = corr.loc[a, b]
        if abs(r) >= threshold:
            rows.append({"feature_a": a, "feature_b": b, "correlacao": round(r, 3)})
    return (
        pd.DataFrame(rows)
        .sort_values("correlacao", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def plot_class_distribution(df: pd.DataFrame, target: str, labels: dict, filename: str, titulo: str):
    """Gráfico de barras da distribuição das classes (com % anotada)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target].value_counts().sort_index()
    total = counts.sum()
    xs = [labels.get(int(c), str(c)) for c in counts.index]
    bars = ax.bar(xs, counts.values, color=sns.color_palette("muted")[: len(counts)])
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,}\n({v / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title(titulo)
    ax.set_ylabel("Nº de amostras")
    ax.set_ylim(0, counts.max() * 1.15)
    return plotting.save_fig(fig, filename)


def plot_bmi_distribution(df: pd.DataFrame, filename: str = "etapa1_bmi_dist.png"):
    """Histograma + boxplot do BMI (evidencia outliers)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["BMI"], bins=60, kde=True, ax=axes[0], color=sns.color_palette("muted")[0])
    axes[0].set_title("Distribuição do BMI")
    axes[0].set_xlabel("BMI")
    axes[0].set_ylabel("Frequência")

    sns.boxplot(x=df["BMI"], ax=axes[1], color=sns.color_palette("muted")[2])
    axes[1].set_title("Boxplot do BMI (outliers em valores altos)")
    axes[1].set_xlabel("BMI")
    fig.suptitle("BMI — distribuição e outliers", fontweight="bold")
    return plotting.save_fig(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "etapa1_correlacao.png"):
    """Heatmap da matriz de correlação de Pearson (todas as colunas)."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Matriz de correlação de Pearson", fontsize=14)
    return plotting.save_fig(fig, filename)


def plot_features_vs_target(df: pd.DataFrame, target: str, labels: dict,
                            features=("GenHlth", "BMI", "Age"),
                            filename: str = "etapa1_features_vs_target.png"):
    """Boxplots de features selecionadas estratificadas pelo alvo binário."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    df = df.copy()
    df["_alvo"] = df[target].map(lambda c: labels.get(int(c), str(c)))
    for ax, feat in zip(np.atleast_1d(axes), features):
        sns.boxplot(data=df, x="_alvo", y=feat, hue="_alvo", legend=False, ax=ax, palette="muted")
        ax.set_title(f"{feat} por {target}")
        ax.set_xlabel("")
    fig.suptitle("Features vs. alvo (classificação binária)", fontweight="bold")
    return plotting.save_fig(fig, filename)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def run() -> None:
    """Executa toda a EDA e salva figuras e métricas."""
    config.set_seeds()
    plotting.set_style()

    print("Carregando datasets...")
    df_bin = data_loading.load_dataset("binary")
    df_multi = data_loading.load_dataset("multiclass")

    # ---- Tabelas / métricas ----
    print("Gerando tabelas-resumo...")
    overview = dataset_overview()
    overview.to_csv(config.METRICS_DIR / "eda_dataset_overview.csv", index=False)

    feature_types_table().to_csv(config.METRICS_DIR / "eda_feature_types.csv", index=False)

    summary = summary_statistics(df_bin)
    summary.to_csv(config.METRICS_DIR / "eda_summary.csv")

    class_distribution(df_bin, config.TARGET_BINARY).to_csv(
        config.METRICS_DIR / "eda_class_dist_binaria.csv", index=False)
    class_distribution(df_multi, config.TARGET_MULTICLASS).to_csv(
        config.METRICS_DIR / "eda_class_dist_multiclasse.csv", index=False)

    high_corr = high_correlations(df_bin)
    high_corr.to_csv(config.METRICS_DIR / "eda_high_correlations.csv", index=False)

    # ---- Figuras ----
    print("Gerando figuras...")
    plot_class_distribution(
        df_bin, config.TARGET_BINARY, config.CLASS_LABELS_BINARY,
        "etapa1_dist_classes_binaria.png", "Distribuição das classes — binária")
    plot_class_distribution(
        df_multi, config.TARGET_MULTICLASS, config.CLASS_LABELS_MULTICLASS,
        "etapa1_dist_classes_multiclasse.png", "Distribuição das classes — multiclasse")
    plot_bmi_distribution(df_bin)
    plot_correlation_heatmap(df_bin)
    plot_features_vs_target(df_bin, config.TARGET_BINARY, config.CLASS_LABELS_BINARY)

    print("\nEDA concluída.")
    print(f"  Figuras salvas em: {config.FIGURES_DIR}")
    print(f"  Métricas salvas em: {config.METRICS_DIR}")


if __name__ == "__main__":
    run()
