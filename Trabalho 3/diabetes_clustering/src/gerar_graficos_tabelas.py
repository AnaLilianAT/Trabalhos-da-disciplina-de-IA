"""Generate additional report figures directly from CSV tables.

This script is intentionally standalone and does not modify the existing
pipeline. It reads tables already produced in `outputs/tables` and writes
the requested figures to `outputs/figures`.

Usage:
    python src/gerar_graficos_tabelas.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")


# Academic-style palette
C = {
    "azul": "#1a5276",
    "azul_cl": "#2e86c1",
    "verde": "#1e8449",
    "verde_cl": "#27ae60",
    "laranja": "#ca6f1e",
    "laranja_cl": "#e67e22",
    "vermelho": "#922b21",
    "cinza_bg": "#f8f9fa",
    "cinza_bd": "#dee2e6",
    "texto": "#212529",
    "roxo": "#6c3483",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "#fdfdfd",
        "font.size": 11,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "outputs" / "tables"
OUT = PROJECT_ROOT / "outputs" / "figures"


def _save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(BASE / name, **kwargs)


def fig1_kmeans_metricas() -> None:
    km = (
        _read_csv("kmeans_results.csv")
        .dropna(subset=["k"])
        .sort_values("k")
        .drop_duplicates(subset=["k"], keep="last")
    )

    if km.empty:
        raise ValueError("A tabela kmeans_results.csv nao possui valores de k para plotagem.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "K-Means: Metricas por Numero de Clusters",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    metrics = [
        ("silhouette_score", "Silhouette Score\n(↑ melhor)", 0.01, "{:.4f}"),
        ("davies_bouldin_index", "Davies-Bouldin Index\n(↓ melhor)", 0.005, "{:.4f}"),
        ("calinski_harabasz_score", "Calinski-Harabasz Score\n(↑ melhor)", 20, "{:.0f}"),
    ]

    colors = sns.color_palette("Blues", n_colors=max(len(km), 3))

    for ax, (column, title, offset, formatter) in zip(axes, metrics):
        ax.bar(
            km["k"].astype(int).astype(str),
            km[column],
            color=colors[: len(km)],
            edgecolor="white",
            width=0.6,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel("Score")
        ax.set_xticks(np.arange(len(km)))
        ax.set_xticklabels(km["k"].astype(int).tolist())

        for i, value in enumerate(km[column]):
            ax.text(
                i,
                value + offset,
                formatter.format(value),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=C["texto"],
            )

    _save(fig, "fig1_kmeans_metricas.png")
    print("Fig 1 OK")


def fig2_agglomerative_metricas() -> None:
    agg = _read_csv("agglomerative_results.csv")
    linkages = ["ward", "complete", "average", "single"]
    colors_lnk = [C["azul_cl"], C["vermelho"], C["verde_cl"], C["laranja_cl"]]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Agglomerative Clustering: Metricas por Linkage e Numero de Clusters",
        fontsize=13,
        fontweight="bold",
    )

    metrics = [
        ("silhouette_score", "Silhouette Score (↑ melhor)"),
        ("davies_bouldin_index", "Davies-Bouldin Index (↓ melhor)"),
        ("calinski_harabasz_score", "Calinski-Harabasz Score (↑ melhor)"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        for lnk, clr, mk in zip(linkages, colors_lnk, markers):
            sub = agg[agg["linkage"] == lnk].sort_values("n_clusters")
            if sub.empty:
                continue
            ax.plot(
                sub["n_clusters"],
                sub[col],
                marker=mk,
                color=clr,
                label=lnk.capitalize(),
                linewidth=2,
                markersize=6,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Numero de Clusters (k)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)

    _save(fig, "fig2_agglomerative_metricas.png")
    print("Fig 2 OK")


def fig3_dbscan_metricas() -> None:
    dbscan = _read_csv("dbscan_results.csv")
    eps_vals = sorted(dbscan["eps"].dropna().unique().tolist())[:4]
    colors_eps = [C["azul_cl"], C["verde_cl"], C["laranja_cl"], C["vermelho"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "DBSCAN: Efeito dos Hiperparametros nas Metricas",
        fontsize=13,
        fontweight="bold",
    )

    for eps, clr in zip(eps_vals, colors_eps):
        sub = dbscan[(dbscan["eps"] == eps) & dbscan["silhouette_score"].notna()].sort_values(
            "min_samples"
        )
        if sub.empty:
            continue
        ax1.plot(
            sub["min_samples"],
            sub["silhouette_score"],
            marker="o",
            color=clr,
            label=f"eps={eps}",
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            sub["min_samples"],
            sub["noise_percentage"],
            marker="o",
            color=clr,
            label=f"eps={eps}",
            linewidth=2,
            markersize=6,
        )

    ax1.set_title("Silhouette Score (↑ melhor)", fontsize=11)
    ax1.set_xlabel("min_samples")
    ax1.set_ylabel("Silhouette Score")
    ax1.legend(fontsize=9)

    ax2.set_title("Percentual de Pontos de Ruido (↓ melhor)", fontsize=11)
    ax2.set_xlabel("min_samples")
    ax2.set_ylabel("% Ruido")
    ax2.legend(fontsize=9)
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.5)

    _save(fig, "fig3_dbscan_metricas.png")
    print("Fig 3 OK")


def fig4_comparacao_algoritmos() -> None:
    best = _read_csv("best_models_summary.csv")
    algos = best["algorithm"].str.upper().tolist()
    sil = best["silhouette_score"].tolist()
    dbi = best["davies_bouldin_index"].tolist()
    ch_raw = best["calinski_harabasz_score"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Comparacao dos Melhores Modelos por Algoritmo", fontsize=13, fontweight="bold")

    algo_colors = [C["azul_cl"], C["verde_cl"], C["laranja_cl"]][: len(algos)]

    metric_specs = [
        (axes[0], sil, "Silhouette Score (↑ melhor)", 0.005, "{:.3f}"),
        (axes[1], dbi, "Davies-Bouldin Index (↓ melhor)", 0.01, "{:.3f}"),
        (axes[2], ch_raw, "Calinski-Harabasz Score (↑ melhor)", 10, "{:.1f}"),
    ]

    for ax, values, title, offset, formatter in metric_specs:
        bars = ax.bar(algos, values, color=algo_colors, width=0.5, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Score")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                formatter.format(value),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    _save(fig, "fig4_comparacao_algoritmos.png")
    print("Fig 4 OK")


def fig5_kmeans_perfil_clusters() -> None:
    km_mean = _read_csv("kmeans_best_cluster_profile_mean.csv").set_index("cluster")

    key_vars = [
        "HighBP",
        "HighChol",
        "BMI",
        "Smoker",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "GenHlth",
        "Age",
        "Education",
        "Income",
    ]
    key_vars = [column for column in key_vars if column in km_mean.columns]
    sub_mean = km_mean[key_vars]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(key_vars))
    width = 0.8 / max(len(sub_mean), 1)
    cluster_colors = sns.color_palette("Set2", n_colors=max(len(sub_mean), 3))

    for i, (idx, row) in enumerate(sub_mean.iterrows()):
        offset = (i - len(sub_mean) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            row.values,
            width,
            label=f"Cluster {idx}",
            color=cluster_colors[i],
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(key_vars, rotation=30, ha="right", fontsize=10)
    ax.set_title(
        "Perfil dos Clusters K-Means: Medias das Principais Variaveis",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Valor Medio")
    ax.legend(title="Cluster", fontsize=10)

    _save(fig, "fig5_kmeans_perfil_clusters.png")
    print("Fig 5 OK")


def _plot_stacked_distribution(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    clusters = df["cluster"].tolist()
    no_diab = df["0.0"].tolist()
    diab = df["1.0"].tolist()
    x = np.arange(len(clusters))

    ax.bar(x, no_diab, 0.5, label="Sem Diabetes (0)", color=C["azul_cl"], edgecolor="white")
    ax.bar(
        x,
        diab,
        0.5,
        bottom=no_diab,
        label="Com Diabetes (1)",
        color=C["laranja_cl"],
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters], fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("% de Pacientes")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)

    for xi, (nd, d) in enumerate(zip(no_diab, diab)):
        ax.text(
            xi,
            nd / 2,
            f"{nd:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        ax.text(
            xi,
            nd + d / 2,
            f"{d:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )


def fig6_distribuicao_diabetes_clusters() -> None:
    km_pct = _read_csv("kmeans_best_vs_original_labels_percentage.csv")
    agg_pct = _read_csv("agglomerative_best_vs_original_labels_percentage.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Distribuicao da Classe Diabetes por Cluster (Melhores Modelos)",
        fontsize=13,
        fontweight="bold",
    )

    _plot_stacked_distribution(axes[0], km_pct, "K-Means")
    _plot_stacked_distribution(axes[1], agg_pct, "Agglomerative - Melhor Configuracao")

    _save(fig, "fig6_distribuicao_diabetes_clusters.png")
    print("Fig 6 OK")


def fig7_pca_variancia() -> None:
    pca = _read_csv("pca_explained_variance.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(1, len(pca) + 1)
    bars = ax.bar(
        x,
        pca["explained_variance_ratio"] * 100,
        color=C["azul_cl"],
        alpha=0.8,
        edgecolor="white",
        label="Variancia por componente",
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        pca["cumulative_explained_variance_ratio"] * 100,
        color=C["laranja_cl"],
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="Variancia acumulada",
    )
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.6)
    ax2.text(len(x) - 0.5, 81, "80%", color="gray", fontsize=9)

    ax.set_title(
        "PCA – Variancia Explicada por Componente Principal",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variancia Explicada (%)")
    ax2.set_ylabel("Variancia Acumulada (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    _save(fig, "fig7_pca_variancia.png")
    print("Fig 7 OK")


def fig8_correlacao_pearson() -> None:
    pearson = _read_csv("pearson_correlation_matrix.csv", index_col=0)

    top_vars = [
        "HighBP",
        "HighChol",
        "BMI",
        "Smoker",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Age",
        "Education",
        "Income",
    ]
    top_vars = [column for column in top_vars if column in pearson.index]
    sub_pearson = pearson.loc[top_vars, top_vars]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.zeros_like(sub_pearson, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        sub_pearson,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Correlacao de Pearson"},
    )

    ax.set_title("Matriz de Correlacao de Pearson – Variaveis de Saude", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    _save(fig, "fig8_correlacao_pearson.png")
    print("Fig 8 OK")


def main() -> None:
    fig1_kmeans_metricas()
    fig2_agglomerative_metricas()
    fig3_dbscan_metricas()
    fig4_comparacao_algoritmos()
    fig5_kmeans_perfil_clusters()
    fig6_distribuicao_diabetes_clusters()
    fig7_pca_variancia()
    fig8_correlacao_pearson()
    print("\nTodos os 8 graficos foram gerados com sucesso.")


if __name__ == "__main__":
    main()
