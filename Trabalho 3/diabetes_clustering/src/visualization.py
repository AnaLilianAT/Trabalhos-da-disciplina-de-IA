"""Final visualization utilities for report-ready clustering figures."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COLUMN
from src.preprocessing import apply_pca


sns.set_theme(style="whitegrid")


def _sanitize_name(name: str) -> str:
    """Convert labels into file-system friendly names."""
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _save_figure(figure: plt.Figure, output_name: str) -> None:
    """Save a figure in the project figure directory and release memory."""
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _format_cluster_labels(labels: pd.Series, highlight_noise: bool = False) -> pd.Series:
    """Format cluster labels for cleaner legends in report figures."""
    if highlight_noise:
        return labels.apply(lambda value: "Noise" if value == -1 else f"Cluster {value}")
    return labels.apply(lambda value: f"Cluster {value}")


def _select_overall_best_algorithm(best_models_df: pd.DataFrame) -> pd.Series:
    """Select the overall best algorithm using the same academic tie-break logic."""
    valid_df = best_models_df.dropna(
        subset=[
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_score",
        ]
    ).copy()
    if valid_df.empty:
        raise ValueError("No valid best-model summary is available for final visualizations.")

    sorted_df = valid_df.sort_values(
        by=[
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_score",
        ],
        ascending=[False, True, False],
    )
    return sorted_df.iloc[0]


def plot_best_metric_barplots(best_models_df: pd.DataFrame) -> None:
    """Generate final barplots comparing the best metrics of each algorithm."""
    metric_specs = [
        (
            "silhouette_score",
            "Best Silhouette Score by Algorithm",
            "final_best_silhouette_score_by_algorithm.png",
        ),
        (
            "davies_bouldin_index",
            "Best Davies-Bouldin Index by Algorithm",
            "final_best_davies_bouldin_index_by_algorithm.png",
        ),
        (
            "calinski_harabasz_score",
            "Best Calinski-Harabasz Score by Algorithm",
            "final_best_calinski_harabasz_score_by_algorithm.png",
        ),
    ]

    for metric_column, title, output_name in metric_specs:
        plot_df = best_models_df.dropna(subset=[metric_column]).copy()
        if plot_df.empty:
            continue

        figure, axis = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=plot_df,
            x="algorithm",
            y=metric_column,
            hue="algorithm",
            dodge=False,
            legend=False,
            palette="Set2",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Algorithm")
        axis.set_ylabel(metric_column.replace("_", " ").title())

        _save_figure(figure, output_name)


def plot_pca_clusters(
    X_scaled: pd.DataFrame,
    labels: pd.Series,
    algorithm_name: str,
    output_name: str,
    highlight_noise: bool = False,
) -> None:
    """Generate a PCA 2D plot colored by cluster assignment."""
    aligned_X = X_scaled.loc[labels.index].copy()
    pca_df, _ = apply_pca(aligned_X, n_components=2)
    plot_df = pca_df.copy()
    plot_df["cluster"] = _format_cluster_labels(labels, highlight_noise=highlight_noise).values

    if highlight_noise:
        unique_labels = list(dict.fromkeys(plot_df["cluster"].tolist()))
        palette = {"Noise": "#6c757d"}
        cluster_labels = [label for label in unique_labels if label != "Noise"]
        color_cycle = sns.color_palette("tab10", n_colors=max(len(cluster_labels), 1))
        for color, cluster_label in zip(color_cycle, cluster_labels):
            palette[cluster_label] = color
    else:
        palette = "tab10"

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=palette,
        alpha=0.75,
        s=32,
        edgecolor=None,
        ax=axis,
    )
    axis.set_title(f"PCA 2D - Best {algorithm_name}")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_figure(figure, output_name)


def plot_label_distribution_by_cluster(
    labels: pd.Series,
    y: pd.Series,
    algorithm_name: str,
    output_name: str,
) -> pd.DataFrame:
    """Plot the distribution of Diabetes_binary by cluster for interpretation only.

    Important:
    - `Diabetes_binary` is used here only after clustering.
    - It must not be used for training or primary model selection.
    """
    aligned_y = y.loc[labels.index].copy()
    plot_df = pd.DataFrame(
        {
            "cluster": _format_cluster_labels(labels, highlight_noise=True),
            TARGET_COLUMN: aligned_y.values,
        },
        index=labels.index,
    )

    percentage_df = (
        pd.crosstab(plot_df["cluster"], plot_df[TARGET_COLUMN], normalize="index") * 100
    ).sort_index()
    percentage_df = percentage_df.rename(columns={0: "Non-Diabetic", 1: "Diabetic"})

    figure, axis = plt.subplots(figsize=(9, 6))
    percentage_df.plot(
        kind="bar",
        stacked=True,
        color=["#8ecae6", "#d62828"],
        ax=axis,
        width=0.8,
    )
    axis.set_title(f"Distribution of {TARGET_COLUMN} by Cluster - {algorithm_name}")
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Percentage")
    axis.legend(title=TARGET_COLUMN, bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_figure(figure, output_name)
    return percentage_df


def plot_cluster_profile_heatmap(
    X_original: pd.DataFrame,
    labels: pd.Series,
    algorithm_name: str,
    output_name: str,
) -> pd.DataFrame:
    """Generate a heatmap of standardized mean cluster profiles."""
    aligned_X = X_original.loc[labels.index].copy()
    profile_df = aligned_X.copy()
    profile_df["cluster"] = labels.values

    mean_df = profile_df.groupby("cluster").mean(numeric_only=True).sort_index()
    standardized_mean_df = mean_df.copy()

    for column in standardized_mean_df.columns:
        column_std = standardized_mean_df[column].std()
        if pd.isna(column_std) or column_std == 0:
            standardized_mean_df[column] = 0.0
        else:
            standardized_mean_df[column] = (
                standardized_mean_df[column] - standardized_mean_df[column].mean()
            ) / column_std

    standardized_mean_df.index = [f"Cluster {index}" for index in standardized_mean_df.index]

    figure_width = max(10, len(standardized_mean_df.columns) * 0.55)
    figure, axis = plt.subplots(figsize=(figure_width, 6))
    sns.heatmap(
        standardized_mean_df,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Standardized Mean"},
        ax=axis,
    )
    axis.set_title(f"Standardized Cluster Profile Heatmap - {algorithm_name}")
    axis.set_xlabel("Variables")
    axis.set_ylabel("Cluster")

    _save_figure(figure, output_name)
    return standardized_mean_df


def generate_final_report_figures(
    X_original: pd.DataFrame,
    X_scaled: pd.DataFrame,
    y: pd.Series,
    best_models_df: pd.DataFrame,
    best_model_outputs: dict[str, dict],
) -> dict:
    """Generate final report-ready visualizations for the best clustering models."""
    print("\n[Visualization] Generating final report figures...")

    plot_best_metric_barplots(best_models_df)

    algorithm_specs = [
        ("kmeans", "K-Means", "final_kmeans_best_pca.png", False),
        ("agglomerative", "Agglomerative Clustering", "final_agglomerative_best_pca.png", False),
        ("dbscan", "DBSCAN", "final_dbscan_best_pca.png", True),
    ]

    for algorithm_key, algorithm_title, output_name, highlight_noise in algorithm_specs:
        if algorithm_key not in best_model_outputs:
            continue

        labels = best_model_outputs[algorithm_key]["labels"]
        plot_pca_clusters(
            X_scaled,
            labels,
            algorithm_name=algorithm_title,
            output_name=output_name,
            highlight_noise=highlight_noise,
        )
        plot_label_distribution_by_cluster(
            labels,
            y,
            algorithm_name=algorithm_title,
            output_name=f"final_{_sanitize_name(algorithm_key)}_label_distribution_by_cluster.png",
        )

    overall_best_row = _select_overall_best_algorithm(best_models_df)
    overall_algorithm_key = str(overall_best_row["algorithm"])
    overall_best_output = best_model_outputs[overall_algorithm_key]

    standardized_profile_df = plot_cluster_profile_heatmap(
        X_original,
        overall_best_output["labels"],
        algorithm_name=f"Best Overall - {overall_algorithm_key}",
        output_name=f"final_{_sanitize_name(overall_algorithm_key)}_best_overall_cluster_profile_heatmap.png",
    )

    print("[Visualization] Final report figures saved to outputs/figures.")
    return {
        "overall_best_algorithm": overall_algorithm_key,
        "profile_shape": standardized_profile_df.shape,
    }


def generate_figures(*args, **kwargs) -> dict:
    """Backward-compatible wrapper for final figure generation."""
    return generate_final_report_figures(*args, **kwargs)


def generate_tables(metrics_table: pd.DataFrame) -> None:
    """Keep a minimal compatibility hook for older pipeline stages."""
    _ = metrics_table
