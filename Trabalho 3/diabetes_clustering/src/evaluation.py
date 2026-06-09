"""Evaluation and interpretation helpers for clustering experiments."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TABLES_DIR, TARGET_COLUMN


sns.set_theme(style="whitegrid")

ALL_RESULTS_PATH = TABLES_DIR / "all_clustering_results.csv"
BEST_MODELS_SUMMARY_PATH = TABLES_DIR / "best_models_summary.csv"


def _ensure_algorithm_column(results_df: pd.DataFrame, algorithm_name: str) -> pd.DataFrame:
    """Return a copy of the results dataframe with a consistent algorithm column."""
    normalized_df = results_df.copy()
    normalized_df["algorithm"] = algorithm_name
    return normalized_df


def _sanitize_name(name: str) -> str:
    """Convert labels into file-system friendly names."""
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _plot_best_metric_by_algorithm(
    best_models_df: pd.DataFrame,
    metric_column: str,
    title: str,
    output_name: str,
) -> None:
    """Plot a bar chart comparing the best metric value for each algorithm."""
    plot_df = best_models_df.dropna(subset=[metric_column]).copy()
    if plot_df.empty:
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="algorithm",
        y=metric_column,
        hue="algorithm",
        palette="Set2",
        dodge=False,
        legend=False,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Algorithm")
    axis.set_ylabel(metric_column.replace("_", " ").title())

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def combine_results(
    kmeans_results: pd.DataFrame,
    agg_results: pd.DataFrame,
    dbscan_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the result tables of all clustering algorithms into one dataframe."""
    combined_df = pd.concat(
        [
            _ensure_algorithm_column(kmeans_results, "kmeans"),
            _ensure_algorithm_column(agg_results, "agglomerative"),
            _ensure_algorithm_column(dbscan_results, "dbscan"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_df.to_csv(ALL_RESULTS_PATH, index=False)
    print(f"[Evaluation] Combined clustering results saved to: {ALL_RESULTS_PATH}")
    return combined_df


def select_best_models(all_results: pd.DataFrame) -> pd.DataFrame:
    """Select the best configuration for each algorithm using metric-based tie-breaking.

    Selection rule:
    1. higher Silhouette Score
    2. lower Davies-Bouldin Index
    3. higher Calinski-Harabasz Score
    """
    print("\n[Evaluation] Selecting the best configuration for each algorithm...")

    best_rows: list[pd.Series] = []

    for algorithm_name, algorithm_df in all_results.groupby("algorithm", sort=False):
        valid_df = algorithm_df.dropna(
            subset=[
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
            ]
        ).copy()

        if valid_df.empty:
            print(
                "[Evaluation] No valid metric-based configuration found for "
                f"{algorithm_name}. Skipping."
            )
            continue

        sorted_df = valid_df.sort_values(
            by=[
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
            ],
            ascending=[False, True, False],
        )
        best_rows.append(sorted_df.iloc[0])

    best_models_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_models_df.to_csv(BEST_MODELS_SUMMARY_PATH, index=False)
    print(f"[Evaluation] Best models summary saved to: {BEST_MODELS_SUMMARY_PATH}")

    _plot_best_metric_by_algorithm(
        best_models_df,
        metric_column="silhouette_score",
        title="Best Silhouette Score by Algorithm",
        output_name="best_silhouette_score_by_algorithm.png",
    )
    _plot_best_metric_by_algorithm(
        best_models_df,
        metric_column="davies_bouldin_index",
        title="Best Davies-Bouldin Index by Algorithm",
        output_name="best_davies_bouldin_index_by_algorithm.png",
    )
    _plot_best_metric_by_algorithm(
        best_models_df,
        metric_column="calinski_harabasz_score",
        title="Best Calinski-Harabasz Score by Algorithm",
        output_name="best_calinski_harabasz_score_by_algorithm.png",
    )

    return best_models_df


def compare_with_original_labels(
    cluster_labels: pd.Series,
    y: pd.Series,
    algorithm_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cluster assignments with the original diabetes label for interpretation only.

    Important:
    - `Diabetes_binary` is used here strictly for post-hoc interpretation.
    - It must never be used to train or select the clustering models.
    """
    print(f"\n[Evaluation] Comparing {algorithm_name} clusters with {TARGET_COLUMN}...")

    labels_series = pd.Series(cluster_labels, name="cluster").copy()
    aligned_y = y.loc[labels_series.index].copy()

    contingency_df = pd.crosstab(labels_series, aligned_y, rownames=["cluster"], colnames=[TARGET_COLUMN])
    percentage_df = pd.crosstab(
        labels_series,
        aligned_y,
        rownames=["cluster"],
        colnames=[TARGET_COLUMN],
        normalize="index",
    ) * 100

    safe_algorithm_name = _sanitize_name(algorithm_name)
    contingency_path = TABLES_DIR / f"{safe_algorithm_name}_vs_original_labels_contingency.csv"
    percentage_path = TABLES_DIR / f"{safe_algorithm_name}_vs_original_labels_percentage.csv"

    contingency_df.to_csv(contingency_path)
    percentage_df.to_csv(percentage_path)

    print(f"[Evaluation] Contingency table saved to: {contingency_path}")
    print(f"[Evaluation] Percentage table saved to: {percentage_path}")

    return contingency_df, percentage_df


def cluster_profiles(
    X_original: pd.DataFrame,
    labels: pd.Series,
    algorithm_name: str,
) -> dict[str, pd.DataFrame]:
    """Create descriptive cluster profiles from the original input variables.

    This function is purely interpretative and should be used only after
    clustering is complete.
    """
    print(f"\n[Evaluation] Building cluster profiles for {algorithm_name}...")

    labels_series = pd.Series(labels, name="cluster").copy()
    profiled_df = X_original.loc[labels_series.index].copy()
    profiled_df["cluster"] = labels_series.values

    mean_df = profiled_df.groupby("cluster").mean(numeric_only=True)
    median_df = profiled_df.groupby("cluster").median(numeric_only=True)
    std_df = profiled_df.groupby("cluster").std(numeric_only=True)

    safe_algorithm_name = _sanitize_name(algorithm_name)
    mean_path = TABLES_DIR / f"{safe_algorithm_name}_cluster_profile_mean.csv"
    median_path = TABLES_DIR / f"{safe_algorithm_name}_cluster_profile_median.csv"
    std_path = TABLES_DIR / f"{safe_algorithm_name}_cluster_profile_std.csv"

    mean_df.to_csv(mean_path)
    median_df.to_csv(median_path)
    std_df.to_csv(std_path)

    print(f"[Evaluation] Cluster means saved to: {mean_path}")
    print(f"[Evaluation] Cluster medians saved to: {median_path}")
    print(f"[Evaluation] Cluster standard deviations saved to: {std_path}")

    standardized_mean_df = mean_df.copy()
    for column in standardized_mean_df.columns:
        column_std = standardized_mean_df[column].std()
        if pd.isna(column_std) or column_std == 0:
            standardized_mean_df[column] = 0.0
        else:
            standardized_mean_df[column] = (
                standardized_mean_df[column] - standardized_mean_df[column].mean()
            ) / column_std

    if not standardized_mean_df.empty:
        figure_width = max(10, len(standardized_mean_df.columns) * 0.5)
        figure, axis = plt.subplots(figsize=(figure_width, 6))
        sns.heatmap(
            standardized_mean_df,
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=axis,
            cbar_kws={"label": "Standardized Mean"},
        )
        axis.set_title(f"Standardized Cluster Profile Heatmap - {algorithm_name}")
        axis.set_xlabel("Variables")
        axis.set_ylabel("Cluster")

        heatmap_path = FIGURES_DIR / f"{safe_algorithm_name}_cluster_profile_heatmap.png"
        figure.tight_layout()
        figure.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"[Evaluation] Cluster profile heatmap saved to: {heatmap_path}")

    return {
        "mean": mean_df,
        "median": median_df,
        "std": std_df,
    }


def evaluate_clustering_results(
    kmeans_results: pd.DataFrame,
    agg_results: pd.DataFrame,
    dbscan_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the comparative evaluation stage across all clustering algorithms."""
    all_results = combine_results(kmeans_results, agg_results, dbscan_results)
    best_models = select_best_models(all_results)
    return all_results, best_models
