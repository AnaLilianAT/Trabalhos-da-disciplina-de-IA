"""Utilities responsible for reading raw and processed datasets."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN


def load_dataset(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the dataset and separate features from the diabetes label.

    This function keeps the original dataframe intact and removes the target
    only from X, which will be used later in clustering algorithms.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Place 'diabetes_binary_health_indicators_BRFSS2015.csv' inside data/raw."
        )

    df = pd.read_csv(dataset_path)
    n_rows, n_columns = df.shape
    print(f"Dataset loaded successfully: {n_rows} rows, {n_columns} columns.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    return df, X, y


def basic_dataset_info(df: pd.DataFrame) -> None:
    """Display basic structural information about the dataset."""
    print("\nBasic dataset info")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

    if TARGET_COLUMN in df.columns:
        print(f"\nTarget distribution ({TARGET_COLUMN}):")
        print(df[TARGET_COLUMN].value_counts(dropna=False).sort_index())


def save_processed_data(dataframe: pd.DataFrame, file_path: Path) -> None:
    """Persist processed data generated in later pipeline stages.

    Responsibility:
    - save intermediate datasets in a consistent place;
    - keep output writing logic centralized.
    """
    dataframe.to_csv(file_path, index=False)
