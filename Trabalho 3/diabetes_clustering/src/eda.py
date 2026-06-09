"""Exploratory data analysis helpers for the diabetes dataset."""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, RANDOM_STATE, TABLES_DIR, TARGET_COLUMN


sns.set_theme(style="whitegrid")

MAX_SCATTER_SAMPLES = 5000


def _sanitize_filename(name: str) -> str:
    """Convert arbitrary labels into file-system friendly names."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _get_numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns used in descriptive analysis."""
    numeric_df = dataframe.select_dtypes(include="number").copy()
    if numeric_df.empty:
        raise ValueError("EDA requires at least one numeric column.")
    return numeric_df


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Persist a matplotlib figure and release memory."""
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def calculate_univariate_statistics(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for every numeric feature."""
    numeric_df = _get_numeric_features(features)

    stats_df = pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "std": numeric_df.std(),
            "min": numeric_df.min(),
            "q1": numeric_df.quantile(0.25),
            "q2": numeric_df.quantile(0.50),
            "q3": numeric_df.quantile(0.75),
            "max": numeric_df.max(),
        }
    )
    stats_df.index.name = "variable"
    output_path = TABLES_DIR / "univariate_statistics.csv"
    stats_df.to_csv(output_path)
    return stats_df


def plot_histograms(features: pd.DataFrame) -> list[Path]:
    """Generate one histogram per numeric feature."""
    numeric_df = _get_numeric_features(features)
    output_paths: list[Path] = []

    for column in numeric_df.columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(data=numeric_df, x=column, bins=30, kde=True, ax=axis, color="#2a6f97")
        axis.set_title(f"Histogram of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")

        output_path = FIGURES_DIR / f"histogram_{_sanitize_filename(column)}.png"
        _save_figure(figure, output_path)
        output_paths.append(output_path)

    return output_paths


def plot_boxplots(features: pd.DataFrame) -> list[Path]:
    """Generate one boxplot per numeric feature."""
    numeric_df = _get_numeric_features(features)
    output_paths: list[Path] = []

    for column in numeric_df.columns:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=numeric_df[column], ax=axis, color="#90be6d")
        axis.set_title(f"Boxplot of {column}")
        axis.set_xlabel(column)

        output_path = FIGURES_DIR / f"boxplot_{_sanitize_filename(column)}.png"
        _save_figure(figure, output_path)
        output_paths.append(output_path)

    return output_paths


def calculate_correlation_matrices(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices."""
    numeric_df = _get_numeric_features(features)

    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")

    pearson_corr.to_csv(TABLES_DIR / "pearson_correlation_matrix.csv")
    spearman_corr.to_csv(TABLES_DIR / "spearman_correlation_matrix.csv")

    return pearson_corr, spearman_corr


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, method_name: str) -> Path:
    """Create a correlation heatmap for a given matrix."""
    figure_width = max(10, ceil(len(correlation_matrix.columns) * 0.6))
    figure, axis = plt.subplots(figsize=(figure_width, figure_width))

    sns.heatmap(
        correlation_matrix,
        ax=axis,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    axis.set_title(f"{method_name} Correlation Heatmap")
    axis.set_xlabel("Variables")
    axis.set_ylabel("Variables")

    output_path = FIGURES_DIR / f"{_sanitize_filename(method_name)}_correlation_heatmap.png"
    _save_figure(figure, output_path)
    return output_path


def _sample_rows_for_plotting(
    features: pd.DataFrame,
    y: pd.Series | None = None,
    max_samples: int = MAX_SCATTER_SAMPLES,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Downsample very large dataframes for more readable scatter plots."""
    if len(features) <= max_samples:
        return features, y

    sampled_features = features.sample(n=max_samples, random_state=RANDOM_STATE)
    sampled_y = None
    if y is not None:
        sampled_y = y.loc[sampled_features.index]

    return sampled_features, sampled_y


def generate_scatter_plots(features: pd.DataFrame) -> list[Path]:
    """Generate scatter plots for selected feature pairs."""
    numeric_df = _get_numeric_features(features)
    sampled_df, _ = _sample_rows_for_plotting(numeric_df)

    selected_pairs = [
        ("BMI", "GenHlth"),
        ("BMI", "Age"),
        ("PhysHlth", "GenHlth"),
        ("MentHlth", "GenHlth"),
        ("Age", "Income"),
    ]

    output_paths: list[Path] = []

    for x_column, y_column in selected_pairs:
        if x_column not in sampled_df.columns or y_column not in sampled_df.columns:
            continue

        figure, axis = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=sampled_df,
            x=x_column,
            y=y_column,
            ax=axis,
            alpha=0.35,
            s=20,
            color="#bc4749",
            edgecolor=None,
        )
        axis.set_title(f"{y_column} vs {x_column}")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)

        output_path = FIGURES_DIR / (
            f"scatter_{_sanitize_filename(x_column)}_vs_{_sanitize_filename(y_column)}.png"
        )
        _save_figure(figure, output_path)
        output_paths.append(output_path)

    return output_paths


def run_pca_for_visualization(
    features: pd.DataFrame,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Apply PCA with two components for exploratory visualization only."""
    numeric_df = _get_numeric_features(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(
        principal_components,
        columns=["PC1", "PC2"],
        index=numeric_df.index,
    )

    explained_variance_df = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": pca.explained_variance_ratio_.cumsum(),
        }
    )
    explained_variance_df.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    output_paths: list[Path] = []

    sampled_pca_df, sampled_y = _sample_rows_for_plotting(pca_df, y)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=sampled_pca_df,
        x="PC1",
        y="PC2",
        ax=axis,
        alpha=0.5,
        s=25,
        color="#1d3557",
        edgecolor=None,
    )
    axis.set_title("PCA Scatter Plot (2 Components)")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")

    output_path = FIGURES_DIR / "pca_scatter_2_components.png"
    _save_figure(figure, output_path)
    output_paths.append(output_path)

    if sampled_y is not None:
        labeled_plot_df = sampled_pca_df.copy()
        labeled_plot_df[TARGET_COLUMN] = sampled_y.astype(str).values

        figure, axis = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=labeled_plot_df,
            x="PC1",
            y="PC2",
            hue=TARGET_COLUMN,
            ax=axis,
            alpha=0.6,
            s=25,
            palette="Set1",
            edgecolor=None,
        )
        axis.set_title(
            "PCA Scatter Plot Colored by Diabetes_binary\n"
            "(Exploratory Only, Not Used in Clustering)"
        )
        axis.set_xlabel("Principal Component 1")
        axis.set_ylabel("Principal Component 2")
        axis.legend(title=TARGET_COLUMN)

        output_path = FIGURES_DIR / "pca_scatter_label_not_used_for_clustering.png"
        _save_figure(figure, output_path)
        output_paths.append(output_path)

    return pca_df, explained_variance_df, output_paths


def run_eda(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> dict:
    """Execute all descriptive analyses and save report-ready artifacts."""
    if y is None and TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()

    univariate_stats = calculate_univariate_statistics(X)
    histogram_paths = plot_histograms(X)
    boxplot_paths = plot_boxplots(X)

    pearson_corr, spearman_corr = calculate_correlation_matrices(X)
    pearson_heatmap_path = plot_correlation_heatmap(pearson_corr, "Pearson")
    spearman_heatmap_path = plot_correlation_heatmap(spearman_corr, "Spearman")

    scatter_paths = generate_scatter_plots(X)
    _, explained_variance_df, pca_plot_paths = run_pca_for_visualization(X, y)

    return {
        "dataset_shape": df.shape,
        "feature_shape": X.shape,
        "target_available": y is not None,
        "univariate_statistics_path": str(TABLES_DIR / "univariate_statistics.csv"),
        "pearson_correlation_path": str(TABLES_DIR / "pearson_correlation_matrix.csv"),
        "spearman_correlation_path": str(TABLES_DIR / "spearman_correlation_matrix.csv"),
        "pca_explained_variance_path": str(TABLES_DIR / "pca_explained_variance.csv"),
        "n_univariate_variables": len(univariate_stats),
        "n_histograms": len(histogram_paths),
        "n_boxplots": len(boxplot_paths),
        "n_scatter_plots": len(scatter_paths),
        "n_pca_plots": len(pca_plot_paths),
        "pearson_heatmap_path": str(pearson_heatmap_path),
        "spearman_heatmap_path": str(spearman_heatmap_path),
        "pca_explained_variance_sum": float(
            explained_variance_df["cumulative_explained_variance_ratio"].iloc[-1]
        ),
    }
