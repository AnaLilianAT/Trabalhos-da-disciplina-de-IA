"""Clustering algorithm entry points for the project."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    AGGLOMERATIVE_MAX_SAMPLES,
    DBSCAN_MAX_SAMPLES,
    FIGURES_DIR,
    KMEANS_MAX_SAMPLES,
    RANDOM_STATE,
    TABLES_DIR,
)
from src.preprocessing import apply_pca


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

sns.set_theme(style="whitegrid")

KMEANS_RESULTS_PATH = TABLES_DIR / "kmeans_results.csv"
AGGLOMERATIVE_RESULTS_PATH = TABLES_DIR / "agglomerative_results.csv"
DBSCAN_RESULTS_PATH = TABLES_DIR / "dbscan_results.csv"


def _calculate_clustering_metrics(X_scaled: pd.DataFrame, labels: pd.Series) -> dict:
    """Calculate internal clustering metrics when more than one cluster exists."""
    n_clusters = labels.nunique()

    if n_clusters <= 1:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "n_clusters_found": n_clusters,
        }

    try:
        return {
            "silhouette_score": silhouette_score(X_scaled, labels),
            "davies_bouldin_index": davies_bouldin_score(X_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
            "n_clusters_found": n_clusters,
        }
    except ValueError:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "n_clusters_found": n_clusters,
        }


def _plot_metric_lines(
    results_df: pd.DataFrame,
    x_column: str,
    metric_column: str,
    title: str,
    output_name: str,
    hue_column: str | None = None,
) -> None:
    """Plot a clustering metric as a function of a configuration variable."""
    plot_df = results_df.dropna(subset=[metric_column]).copy()
    if plot_df.empty:
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=plot_df,
        x=x_column,
        y=metric_column,
        hue=hue_column,
        style=hue_column,
        marker="o",
        linewidth=2.0,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(x_column.replace("_", " ").title())
    axis.set_ylabel(metric_column.replace("_", " ").title())
    axis.set_xticks(sorted(plot_df[x_column].unique().tolist()))
    if hue_column is not None:
        axis.legend(title=hue_column.replace("_", " ").title())

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_metric_heatmap(
    results_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_name: str,
    fill_value: float | None = None,
) -> None:
    """Plot a heatmap for DBSCAN metrics across eps and min_samples."""
    heatmap_df = results_df.pivot(
        index="min_samples",
        columns="eps",
        values=value_column,
    ).sort_index()

    if fill_value is not None:
        heatmap_df = heatmap_df.fillna(fill_value)

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("eps")
    axis.set_ylabel("min_samples")

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_cluster_pca(
    X_scaled: pd.DataFrame,
    labels: pd.Series,
    title: str,
    output_name: str,
    highlight_noise: bool = False,
) -> None:
    """Create a 2D PCA plot colored by cluster assignments."""
    pca_df, _ = apply_pca(X_scaled, n_components=2)
    plot_df = pca_df.copy()

    if highlight_noise:
        plot_df["cluster"] = labels.apply(lambda value: "noise" if value == -1 else f"cluster_{value}")
        unique_labels = sorted(plot_df["cluster"].unique().tolist())
        palette = {"noise": "#6c757d"}
        color_cycle = sns.color_palette("tab10", n_colors=max(len(unique_labels) - 1, 1))
        cluster_labels = [label for label in unique_labels if label != "noise"]
        for color, cluster_label in zip(color_cycle, cluster_labels):
            palette[cluster_label] = color
    else:
        plot_df["cluster"] = labels.astype(str).values
        palette = "tab10"

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=palette,
        alpha=0.7,
        s=30,
        ax=axis,
        edgecolor=None,
    )
    axis.set_title(title)
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.legend(title="Cluster")

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _fit_kmeans_for_k(X_scaled: pd.DataFrame, k: int) -> tuple[KMeans, pd.Series, dict]:
    """Fit a K-Means model for a single value of k and evaluate it."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = pd.Series(model.fit_predict(X_scaled), index=X_scaled.index, name=f"kmeans_k_{k}")
    metrics = _calculate_clustering_metrics(X_scaled, labels)
    return model, labels, metrics


def _sample_for_kmeans(
    X_scaled: pd.DataFrame,
    max_samples: int = KMEANS_MAX_SAMPLES,
) -> tuple[pd.DataFrame, dict]:
    """Sample large datasets to keep K-Means experiments practical."""
    original_size = len(X_scaled)
    if original_size <= max_samples:
        print(
            "[Clustering] K-Means will use the full dataset: "
            f"{original_size} instances."
        )
        return X_scaled, {
            "used_sampling": False,
            "sampling_strategy": "full_dataset",
            "original_size": original_size,
            "sample_size": original_size,
        }

    print(
        "[Clustering] K-Means input is large. "
        f"Using a reproducible random sample of {max_samples} out of {original_size} instances."
    )
    X_sampled = X_scaled.sample(n=max_samples, random_state=RANDOM_STATE)
    return X_sampled, {
        "used_sampling": True,
        "sampling_strategy": "random",
        "original_size": original_size,
        "sample_size": len(X_sampled),
    }


def run_kmeans_experiments(
    X_scaled: pd.DataFrame,
    k_values: range = range(2, 11),
    max_samples: int = KMEANS_MAX_SAMPLES,
) -> tuple[pd.DataFrame, dict[int, pd.Series], dict]:
    """Run K-Means for multiple k values and save report-ready outputs."""
    print("\n[Clustering] Running K-Means experiments...")

    X_used, sample_info = _sample_for_kmeans(X_scaled, max_samples=max_samples)
    print(
        "[Clustering] K-Means experiments will use "
        f"{sample_info['sample_size']} instances "
        f"(original dataset size: {sample_info['original_size']})."
    )

    results: list[dict] = []
    labels_by_k: dict[int, pd.Series] = {}
    models_by_k: dict[int, KMeans] = {}

    for k in k_values:
        print(f"[Clustering] Training K-Means with k={k}...")
        model, labels, metrics = _fit_kmeans_for_k(X_used, k)

        labels_by_k[k] = labels
        models_by_k[k] = model
        results.append(
            {
                "algorithm": "kmeans",
                "k": k,
                "inertia": model.inertia_,
                "n_clusters_found": metrics["n_clusters_found"],
                "silhouette_score": metrics["silhouette_score"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "used_sampling": sample_info["used_sampling"],
                "sampling_strategy": sample_info["sampling_strategy"],
                "original_size": sample_info["original_size"],
                "sample_size": sample_info["sample_size"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(KMEANS_RESULTS_PATH, index=False)
    print(f"[Clustering] K-Means results saved to: {KMEANS_RESULTS_PATH}")

    _plot_metric_lines(
        results_df,
        x_column="k",
        metric_column="silhouette_score",
        title="Silhouette Score by K for K-Means",
        output_name="kmeans_silhouette_by_k.png",
    )
    _plot_metric_lines(
        results_df,
        x_column="k",
        metric_column="davies_bouldin_index",
        title="Davies-Bouldin Index by K for K-Means",
        output_name="kmeans_davies_bouldin_by_k.png",
    )
    _plot_metric_lines(
        results_df,
        x_column="k",
        metric_column="calinski_harabasz_score",
        title="Calinski-Harabasz Score by K for K-Means",
        output_name="kmeans_calinski_harabasz_by_k.png",
    )

    valid_results_df = results_df.dropna(subset=["silhouette_score"]).copy()
    if valid_results_df.empty:
        raise ValueError("K-Means experiments did not produce a valid solution with more than one cluster.")

    best_row = valid_results_df.sort_values(by="silhouette_score", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_labels = labels_by_k[best_k]
    best_model = models_by_k[best_k]

    _plot_cluster_pca(
        X_used,
        best_labels,
        title=f"K-Means PCA Projection (Best k = {best_k})",
        output_name="kmeans_best_pca_clusters.png",
    )

    best_model_info = {
        "k": best_k,
        "model": best_model,
        "labels": best_labels,
        "silhouette_score": float(best_row["silhouette_score"]),
        "davies_bouldin_index": float(best_row["davies_bouldin_index"]),
        "calinski_harabasz_score": float(best_row["calinski_harabasz_score"]),
        "used_sampling": bool(best_row["used_sampling"]),
        "sampling_strategy": str(best_row["sampling_strategy"]),
        "original_size": int(best_row["original_size"]),
        "sample_size": int(best_row["sample_size"]),
    }

    print(
        "[Clustering] Best K-Means model selected by silhouette score: "
        f"k={best_k}, silhouette={best_model_info['silhouette_score']:.4f}"
    )

    return results_df, labels_by_k, best_model_info


def _sample_for_agglomerative(
    X_scaled: pd.DataFrame,
    y: pd.Series | None = None,
    max_samples: int = AGGLOMERATIVE_MAX_SAMPLES,
) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Sample large datasets to make Agglomerative Clustering computationally feasible.

    The full dataset is preserved for the rest of the pipeline. Sampling is only
    applied inside Agglomerative experiments because hierarchical clustering can
    become prohibitively expensive on very large datasets.
    """
    original_size = len(X_scaled)
    if original_size <= max_samples:
        print(
            "[Clustering] Agglomerative will use the full dataset: "
            f"{original_size} instances."
        )
        return X_scaled, y, {
            "used_sampling": False,
            "sampling_strategy": "full_dataset",
            "original_size": original_size,
            "sample_size": original_size,
        }

    print(
        "[Clustering] Agglomerative input is large. "
        f"Using a reproducible sample of {max_samples} out of {original_size} instances."
    )

    if y is not None:
        try:
            X_sampled, _, y_sampled, _ = train_test_split(
                X_scaled,
                y,
                train_size=max_samples,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            sampling_strategy = "stratified"
            print("[Clustering] Stratified sampling was used based on the target distribution.")
            return X_sampled, y_sampled, {
                "used_sampling": True,
                "sampling_strategy": sampling_strategy,
                "original_size": original_size,
                "sample_size": len(X_sampled),
            }
        except ValueError:
            print(
                "[Clustering] Stratified sampling was not possible for this y distribution. "
                "Falling back to simple random sampling."
            )

    X_sampled = X_scaled.sample(n=max_samples, random_state=RANDOM_STATE)
    y_sampled = y.loc[X_sampled.index] if y is not None else None
    print("[Clustering] Simple random sampling was used.")
    return X_sampled, y_sampled, {
        "used_sampling": True,
        "sampling_strategy": "random",
        "original_size": original_size,
        "sample_size": len(X_sampled),
    }


def _fit_agglomerative_for_configuration(
    X_scaled: pd.DataFrame,
    linkage: str,
    n_clusters: int,
) -> tuple[AgglomerativeClustering, pd.Series, dict]:
    """Fit AgglomerativeClustering for one linkage/cluster-count configuration."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",
    )
    labels = pd.Series(
        model.fit_predict(X_scaled),
        index=X_scaled.index,
        name=f"agglomerative_{linkage}_k_{n_clusters}",
    )
    metrics = _calculate_clustering_metrics(X_scaled, labels)
    return model, labels, metrics


def run_agglomerative_experiments(
    X_scaled: pd.DataFrame,
    y: pd.Series | None = None,
    n_cluster_values: range = range(2, 11),
    linkages: tuple[str, ...] = ("ward", "complete", "average", "single"),
    max_samples: int = AGGLOMERATIVE_MAX_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """Run Agglomerative Clustering experiments across linkages and cluster counts."""
    print("\n[Clustering] Running Agglomerative Clustering experiments...")

    X_used, _, sample_info = _sample_for_agglomerative(X_scaled, y, max_samples=max_samples)
    print(
        "[Clustering] Agglomerative experiments will use "
        f"{sample_info['sample_size']} instances "
        f"(original dataset size: {sample_info['original_size']})."
    )

    results: list[dict] = []
    labels_by_configuration: dict[str, pd.Series] = {}
    models_by_configuration: dict[str, AgglomerativeClustering] = {}

    for linkage in linkages:
        for n_clusters in n_cluster_values:
            print(
                "[Clustering] Training AgglomerativeClustering with "
                f"linkage={linkage}, n_clusters={n_clusters}..."
            )
            model, labels, metrics = _fit_agglomerative_for_configuration(
                X_used,
                linkage=linkage,
                n_clusters=n_clusters,
            )

            configuration_name = f"{linkage}_k_{n_clusters}"
            labels_by_configuration[configuration_name] = labels
            models_by_configuration[configuration_name] = model

            results.append(
                {
                    "algorithm": "agglomerative",
                    "linkage": linkage,
                    "n_clusters": n_clusters,
                    "configuration": configuration_name,
                    "n_clusters_found": metrics["n_clusters_found"],
                    "silhouette_score": metrics["silhouette_score"],
                    "davies_bouldin_index": metrics["davies_bouldin_index"],
                    "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                    "used_sampling": sample_info["used_sampling"],
                    "sampling_strategy": sample_info["sampling_strategy"],
                    "original_size": sample_info["original_size"],
                    "sample_size": sample_info["sample_size"],
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(AGGLOMERATIVE_RESULTS_PATH, index=False)
    print(f"[Clustering] Agglomerative results saved to: {AGGLOMERATIVE_RESULTS_PATH}")

    _plot_metric_lines(
        results_df,
        x_column="n_clusters",
        metric_column="silhouette_score",
        title="Agglomerative Silhouette Score by Number of Clusters and Linkage",
        output_name="agglomerative_silhouette_by_n_clusters_and_linkage.png",
        hue_column="linkage",
    )
    _plot_metric_lines(
        results_df,
        x_column="n_clusters",
        metric_column="davies_bouldin_index",
        title="Agglomerative Davies-Bouldin by Number of Clusters and Linkage",
        output_name="agglomerative_davies_bouldin_by_n_clusters_and_linkage.png",
        hue_column="linkage",
    )
    _plot_metric_lines(
        results_df,
        x_column="n_clusters",
        metric_column="calinski_harabasz_score",
        title="Agglomerative Calinski-Harabasz by Number of Clusters and Linkage",
        output_name="agglomerative_calinski_harabasz_by_n_clusters_and_linkage.png",
        hue_column="linkage",
    )

    valid_results_df = results_df.dropna(subset=["silhouette_score"]).copy()
    if valid_results_df.empty:
        raise ValueError(
            "Agglomerative experiments did not produce a valid solution with more than one cluster."
        )

    best_row = valid_results_df.sort_values(by="silhouette_score", ascending=False).iloc[0]
    best_configuration = str(best_row["configuration"])
    best_labels = labels_by_configuration[best_configuration]
    best_model = models_by_configuration[best_configuration]

    _plot_cluster_pca(
        X_used,
        best_labels,
        title=(
            "Agglomerative PCA Projection "
            f"(Best linkage = {best_row['linkage']}, n_clusters = {int(best_row['n_clusters'])})"
        ),
        output_name="agglomerative_best_pca_clusters.png",
    )

    best_configuration_info = {
        "configuration": best_configuration,
        "linkage": str(best_row["linkage"]),
        "n_clusters": int(best_row["n_clusters"]),
        "model": best_model,
        "labels": best_labels,
        "silhouette_score": float(best_row["silhouette_score"]),
        "davies_bouldin_index": float(best_row["davies_bouldin_index"]),
        "calinski_harabasz_score": float(best_row["calinski_harabasz_score"]),
        "used_sampling": bool(best_row["used_sampling"]),
        "sampling_strategy": str(best_row["sampling_strategy"]),
        "original_size": int(best_row["original_size"]),
        "sample_size": int(best_row["sample_size"]),
    }

    print(
        "[Clustering] Best Agglomerative configuration selected by silhouette score: "
        f"{best_configuration} "
        f"(silhouette={best_configuration_info['silhouette_score']:.4f})."
    )

    return results_df, labels_by_configuration, best_configuration_info


def _sample_for_dbscan(
    X_scaled: pd.DataFrame,
    max_samples: int = DBSCAN_MAX_SAMPLES,
) -> tuple[pd.DataFrame, dict]:
    """Sample large datasets to keep DBSCAN computationally feasible."""
    original_size = len(X_scaled)
    if original_size <= max_samples:
        print(
            "[Clustering] DBSCAN will use the full dataset: "
            f"{original_size} instances."
        )
        return X_scaled, {
            "used_sampling": False,
            "sampling_strategy": "full_dataset",
            "original_size": original_size,
            "sample_size": original_size,
        }

    print(
        "[Clustering] DBSCAN input is large. "
        f"Using a reproducible random sample of {max_samples} out of {original_size} instances."
    )
    X_sampled = X_scaled.sample(n=max_samples, random_state=RANDOM_STATE)
    return X_sampled, {
        "used_sampling": True,
        "sampling_strategy": "random",
        "original_size": original_size,
        "sample_size": len(X_sampled),
    }


def _calculate_dbscan_metrics(X_scaled: pd.DataFrame, labels: pd.Series) -> dict:
    """Calculate DBSCAN-specific metrics while ignoring noise points in the scores."""
    noise_mask = labels == -1
    n_noise_points = int(noise_mask.sum())
    n_total_points = len(labels)
    noise_percentage = (n_noise_points / n_total_points) * 100 if n_total_points > 0 else 0.0

    valid_mask = ~noise_mask
    valid_labels = labels.loc[valid_mask]
    valid_X = X_scaled.loc[valid_mask]
    n_clusters_found = int(valid_labels.nunique())

    if n_clusters_found < 2:
        return {
            "n_clusters_found": n_clusters_found,
            "n_noise_points": n_noise_points,
            "noise_percentage": noise_percentage,
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
        }

    metrics = _calculate_clustering_metrics(valid_X, valid_labels)
    metrics.update(
        {
            "n_clusters_found": n_clusters_found,
            "n_noise_points": n_noise_points,
            "noise_percentage": noise_percentage,
        }
    )
    return metrics


def _fit_dbscan_for_configuration(
    X_scaled: pd.DataFrame,
    eps: float,
    min_samples: int,
) -> tuple[DBSCAN, pd.Series, dict]:
    """Fit DBSCAN for one eps/min_samples configuration."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = pd.Series(
        model.fit_predict(X_scaled),
        index=X_scaled.index,
        name=f"dbscan_eps_{eps}_min_samples_{min_samples}",
    )
    metrics = _calculate_dbscan_metrics(X_scaled, labels)
    return model, labels, metrics


def run_dbscan_experiments(
    X_scaled: pd.DataFrame,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    max_samples: int = DBSCAN_MAX_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """Run DBSCAN experiments across eps and min_samples values."""
    print("\n[Clustering] Running DBSCAN experiments...")

    if eps_values is None:
        eps_values = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    if min_samples_values is None:
        min_samples_values = [5, 10, 20, 50]

    X_used, sample_info = _sample_for_dbscan(X_scaled, max_samples=max_samples)
    print(
        "[Clustering] DBSCAN experiments will use "
        f"{sample_info['sample_size']} instances "
        f"(original dataset size: {sample_info['original_size']})."
    )

    results: list[dict] = []
    labels_by_configuration: dict[str, pd.Series] = {}
    models_by_configuration: dict[str, DBSCAN] = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            print(
                "[Clustering] Training DBSCAN with "
                f"eps={eps}, min_samples={min_samples}..."
            )
            model, labels, metrics = _fit_dbscan_for_configuration(
                X_used,
                eps=eps,
                min_samples=min_samples,
            )

            configuration_name = f"eps_{eps}_min_samples_{min_samples}"
            labels_by_configuration[configuration_name] = labels
            models_by_configuration[configuration_name] = model

            results.append(
                {
                    "algorithm": "dbscan",
                    "eps": eps,
                    "min_samples": min_samples,
                    "configuration": configuration_name,
                    "n_clusters_found": metrics["n_clusters_found"],
                    "n_noise_points": metrics["n_noise_points"],
                    "noise_percentage": metrics["noise_percentage"],
                    "silhouette_score": metrics["silhouette_score"],
                    "davies_bouldin_index": metrics["davies_bouldin_index"],
                    "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                    "used_sampling": sample_info["used_sampling"],
                    "sampling_strategy": sample_info["sampling_strategy"],
                    "original_size": sample_info["original_size"],
                    "sample_size": sample_info["sample_size"],
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(DBSCAN_RESULTS_PATH, index=False)
    print(f"[Clustering] DBSCAN results saved to: {DBSCAN_RESULTS_PATH}")

    _plot_metric_heatmap(
        results_df,
        value_column="silhouette_score",
        title="DBSCAN Silhouette Score by eps and min_samples",
        output_name="dbscan_silhouette_by_eps_and_min_samples.png",
    )
    _plot_metric_heatmap(
        results_df,
        value_column="noise_percentage",
        title="DBSCAN Noise Percentage by eps and min_samples",
        output_name="dbscan_noise_percentage_by_eps_and_min_samples.png",
    )

    valid_results_df = results_df.dropna(subset=["silhouette_score"]).copy()
    if valid_results_df.empty:
        raise ValueError(
            "DBSCAN experiments did not produce a valid solution with at least two non-noise clusters."
        )

    best_row = valid_results_df.sort_values(by="silhouette_score", ascending=False).iloc[0]
    best_configuration = str(best_row["configuration"])
    best_labels = labels_by_configuration[best_configuration]
    best_model = models_by_configuration[best_configuration]

    _plot_cluster_pca(
        X_used,
        best_labels,
        title=(
            "DBSCAN PCA Projection "
            f"(Best eps = {best_row['eps']}, min_samples = {int(best_row['min_samples'])})"
        ),
        output_name="dbscan_best_pca_clusters.png",
        highlight_noise=True,
    )

    best_configuration_info = {
        "configuration": best_configuration,
        "eps": float(best_row["eps"]),
        "min_samples": int(best_row["min_samples"]),
        "model": best_model,
        "labels": best_labels,
        "n_clusters_found": int(best_row["n_clusters_found"]),
        "n_noise_points": int(best_row["n_noise_points"]),
        "noise_percentage": float(best_row["noise_percentage"]),
        "silhouette_score": float(best_row["silhouette_score"]),
        "davies_bouldin_index": float(best_row["davies_bouldin_index"]),
        "calinski_harabasz_score": float(best_row["calinski_harabasz_score"]),
        "used_sampling": bool(best_row["used_sampling"]),
        "sampling_strategy": str(best_row["sampling_strategy"]),
        "original_size": int(best_row["original_size"]),
        "sample_size": int(best_row["sample_size"]),
    }

    print(
        "[Clustering] Best DBSCAN configuration selected by silhouette score: "
        f"{best_configuration} "
        f"(silhouette={best_configuration_info['silhouette_score']:.4f})."
    )

    return results_df, labels_by_configuration, best_configuration_info


def run_kmeans(features: pd.DataFrame) -> dict:
    """Compatibility wrapper around the K-Means experiment pipeline."""
    results_df, labels_by_k, best_model = run_kmeans_experiments(features)
    return {
        "algorithm": "kmeans",
        "results": results_df,
        "labels_by_configuration": labels_by_k,
        "best_model": best_model,
    }


def run_agglomerative(features: pd.DataFrame, y: pd.Series | None = None) -> dict:
    """Compatibility wrapper around the Agglomerative experiment pipeline."""
    results_df, labels_by_configuration, best_configuration = run_agglomerative_experiments(
        features,
        y=y,
    )
    return {
        "algorithm": "agglomerative",
        "results": results_df,
        "labels_by_configuration": labels_by_configuration,
        "best_model": best_configuration,
    }


def run_dbscan(features: pd.DataFrame) -> dict:
    """Compatibility wrapper around the DBSCAN experiment pipeline."""
    results_df, labels_by_configuration, best_configuration = run_dbscan_experiments(features)
    return {
        "algorithm": "dbscan",
        "results": results_df,
        "labels_by_configuration": labels_by_configuration,
        "best_model": best_configuration,
    }


def run_all_clustering_algorithms(features: pd.DataFrame) -> list[dict]:
    """Orchestrate all clustering experiments required by the assignment."""
    return [
        run_kmeans(features),
        run_agglomerative(features),
        run_dbscan(features),
    ]
