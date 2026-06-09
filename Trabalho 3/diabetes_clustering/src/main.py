"""Main entry point for the complete diabetes clustering pipeline.

This file is designed to work with:
python src/main.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.clustering import (  # noqa: E402
    run_agglomerative_experiments,
    run_dbscan_experiments,
    run_kmeans_experiments,
)
from src.config import (  # noqa: E402
    AGGLOMERATIVE_MAX_SAMPLES,
    CLUSTERED_DATA_FILE,
    DBSCAN_MAX_SAMPLES,
    KMEANS_MAX_SAMPLES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TABLES_DIR,
    ensure_directories,
)
from src.data_loader import basic_dataset_info, load_dataset, save_processed_data  # noqa: E402
from src.eda import run_eda  # noqa: E402
from src.evaluation import (  # noqa: E402
    cluster_profiles,
    compare_with_original_labels,
    evaluate_clustering_results,
)
from src.preprocessing import preprocess_data, reattach_target  # noqa: E402
from src.report_builder import build_report  # noqa: E402
from src.visualization import generate_final_report_figures  # noqa: E402


def _print_stage(title: str) -> None:
    """Print a clear section header for the current pipeline stage."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def _save_interpretation_dataset(
    X_original,
    y,
    labels,
    algorithm_name: str,
) -> Path:
    """Save a dataset with clusters and the original label for interpretation only."""
    interpretation_df = reattach_target(X_original.loc[labels.index].copy(), y.loc[labels.index].copy())
    interpretation_df["cluster"] = labels.values

    output_path = TABLES_DIR / f"{algorithm_name}_cluster_interpretation_dataset.csv"
    save_processed_data(interpretation_df, output_path)
    print(f"[Main] Interpretation dataset saved to: {output_path}")
    return output_path


def main() -> None:
    """Execute the full clustering pipeline end-to-end."""
    _print_stage("Initializing Project Structure")
    ensure_directories()
    print("[Main] Required directories are ready.")
    print(
        "[Main] Configured sampling limits: "
        f"K-Means={KMEANS_MAX_SAMPLES}, "
        f"Agglomerative={AGGLOMERATIVE_MAX_SAMPLES}, "
        f"DBSCAN={DBSCAN_MAX_SAMPLES}"
    )

    _print_stage("1. Loading Dataset")
    try:
        df, X, y = load_dataset(RAW_DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Main] Dataset loading failed: {exc}")
        return

    basic_dataset_info(df)

    print("\n[Main] Dataset successfully separated for unsupervised learning.")
    print(f"[Main] Original dataframe shape: {df.shape}")
    print(f"[Main] Feature matrix X shape: {X.shape}")
    print(f"[Main] Target vector y shape: {y.shape}")
    print(f"[Main] Target column kept out of clustering inputs: '{TARGET_COLUMN}'")
    print(f"[Main] Target present in X: {TARGET_COLUMN in X.columns}")

    _print_stage("2. Running Descriptive Analysis")
    eda_summary = run_eda(df, X, y)
    print("[Main] EDA completed successfully.")
    print(f"[Main] Univariate statistics: {eda_summary['univariate_statistics_path']}")
    print(f"[Main] Pearson correlation matrix: {eda_summary['pearson_correlation_path']}")
    print(f"[Main] Spearman correlation matrix: {eda_summary['spearman_correlation_path']}")
    print(f"[Main] PCA explained variance: {eda_summary['pca_explained_variance_path']}")
    print(
        "[Main] PCA cumulative explained variance (2 components): "
        f"{eda_summary['pca_explained_variance_sum']:.4f}"
    )

    _print_stage("3. Preprocessing Input Features")
    X_processed = preprocess_data(X)
    print("[Main] Preprocessing completed successfully.")
    print(f"[Main] Processed feature matrix shape: {X_processed.shape}")

    _print_stage("4. Running K-Means Experiments")
    kmeans_results_df, _, best_kmeans_model = run_kmeans_experiments(
        X_processed,
        max_samples=KMEANS_MAX_SAMPLES,
    )
    print("[Main] K-Means completed successfully.")
    print(f"[Main] Tested k values: {kmeans_results_df['k'].tolist()}")
    print(f"[Main] Best K-Means k: {best_kmeans_model['k']}")
    print(f"[Main] Best K-Means silhouette score: {best_kmeans_model['silhouette_score']:.4f}")
    print(
        "[Main] K-Means instances used: "
        f"{best_kmeans_model['sample_size']} "
        f"(original size: {best_kmeans_model['original_size']})"
    )

    _print_stage("5. Running Agglomerative Clustering Experiments")
    agglomerative_results_df, _, best_agglomerative_model = run_agglomerative_experiments(
        X_processed,
        y=y,
        max_samples=AGGLOMERATIVE_MAX_SAMPLES,
    )
    print("[Main] Agglomerative Clustering completed successfully.")
    print(
        "[Main] Tested linkages: "
        f"{sorted(agglomerative_results_df['linkage'].unique().tolist())}"
    )
    print(
        "[Main] Best Agglomerative configuration: "
        f"{best_agglomerative_model['configuration']}"
    )
    print(
        "[Main] Agglomerative instances used: "
        f"{best_agglomerative_model['sample_size']} "
        f"(original size: {best_agglomerative_model['original_size']})"
    )

    _print_stage("6. Running DBSCAN Experiments")
    dbscan_results_df, _, best_dbscan_model = run_dbscan_experiments(
        X_processed,
        max_samples=DBSCAN_MAX_SAMPLES,
    )
    print("[Main] DBSCAN completed successfully.")
    print(f"[Main] Tested eps values: {sorted(dbscan_results_df['eps'].unique().tolist())}")
    print(
        "[Main] Tested min_samples values: "
        f"{sorted(dbscan_results_df['min_samples'].unique().tolist())}"
    )
    print(f"[Main] Best DBSCAN configuration: {best_dbscan_model['configuration']}")
    print(f"[Main] Best DBSCAN noise percentage: {best_dbscan_model['noise_percentage']:.2f}%")

    _print_stage("7. Comparing Clustering Results")
    all_results_df, best_models_df = evaluate_clustering_results(
        kmeans_results_df,
        agglomerative_results_df,
        dbscan_results_df,
    )
    print("[Main] Comparative evaluation completed successfully.")
    print(
        "[Main] Compared algorithms: "
        f"{sorted(all_results_df['algorithm'].dropna().unique().tolist())}"
    )
    print(f"[Main] Best models identified: {best_models_df['algorithm'].tolist()}")

    _print_stage("8. Interpreting Best Clusters with Diabetes_binary")
    best_model_outputs = {
        "kmeans": best_kmeans_model,
        "agglomerative": best_agglomerative_model,
        "dbscan": best_dbscan_model,
    }

    for algorithm_name, best_model in best_model_outputs.items():
        print(f"\n[Main] Interpreting best {algorithm_name} model...")
        labels = best_model["labels"]

        _save_interpretation_dataset(X, y, labels, f"{algorithm_name}_best")
        compare_with_original_labels(labels, y, f"{algorithm_name}_best")
        cluster_profiles(X, labels, f"{algorithm_name}_best")

    _print_stage("9. Generating Final Report Figures")
    visualization_summary = generate_final_report_figures(
        X_original=X,
        X_scaled=X_processed,
        y=y,
        best_models_df=best_models_df,
        best_model_outputs=best_model_outputs,
    )
    print("[Main] Final visualizations completed successfully.")
    print(
        "[Main] Overall best algorithm for final profile heatmap: "
        f"{visualization_summary['overall_best_algorithm']}"
    )

    _print_stage("10. Saving Final Clustered Dataset for Interpretation")
    overall_best_algorithm = visualization_summary["overall_best_algorithm"]
    overall_best_labels = best_model_outputs[overall_best_algorithm]["labels"]
    clustered_df = reattach_target(
        X.loc[overall_best_labels.index].copy(),
        y.loc[overall_best_labels.index].copy(),
    )
    clustered_df["cluster"] = overall_best_labels.values
    clustered_df["algorithm"] = overall_best_algorithm
    save_processed_data(clustered_df, CLUSTERED_DATA_FILE)
    print(f"[Main] Final clustered interpretation dataset saved to: {CLUSTERED_DATA_FILE}")

    _print_stage("11. Building Final Report")
    report_summary = build_report(
        dataset_df=df,
        feature_df=X,
        best_models_df=best_models_df,
    )
    print(f"[Main] Markdown report saved to: {report_summary['markdown_path']}")
    print(f"[Main] HTML report saved to: {report_summary['html_path']}")
    print(f"[Main] PDF conversion status: {report_summary['pdf_message']}")

    _print_stage("Pipeline Finished")
    print("[Main] All requested pipeline stages completed.")
    print("[Main] Outputs were saved under outputs/, reports/ and data/processed/.")


if __name__ == "__main__":
    main()
