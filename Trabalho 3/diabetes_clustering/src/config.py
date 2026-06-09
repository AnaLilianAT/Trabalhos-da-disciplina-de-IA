"""Central configuration for paths and global project settings."""

from pathlib import Path


RANDOM_STATE = 42
KMEANS_MAX_SAMPLES = 20000
AGGLOMERATIVE_MAX_SAMPLES = 10000
DBSCAN_MAX_SAMPLES = 10000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Diabetes_binary"
RAW_DATA_PATH = RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "diabetes_processed.csv"
CLUSTERED_DATA_FILE = PROCESSED_DATA_DIR / "diabetes_clustered.csv"
METRICS_TABLE_FILE = TABLES_DIR / "clustering_metrics.csv"
CLUSTER_SUMMARY_FILE = TABLES_DIR / "cluster_summary.csv"
FINAL_REPORT_FILE = REPORTS_DIR / "diabetes_clustering_report.pdf"
REPORT_MARKDOWN_FILE = REPORTS_DIR / "report.md"
REPORT_HTML_FILE = REPORTS_DIR / "report.html"
REPORT_PDF_INSTRUCTIONS_FILE = REPORTS_DIR / "pdf_conversion_instructions.txt"


def ensure_directories() -> None:
    """Create the expected project directories if they do not exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
