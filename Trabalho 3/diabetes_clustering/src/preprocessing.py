"""Data cleaning and feature preparation for clustering."""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TABLES_DIR


OUTLIER_SUMMARY_PATH = TABLES_DIR / "outlier_summary_iqr.csv"

ID_KEYWORDS = ("id", "identifier", "record")
NAME_KEYWORDS = ("name", "nome")
DATE_KEYWORDS = ("date", "data", "time", "timestamp")


def remove_unnecessary_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious identifier, name, or date columns when they exist."""
    print("\n[Preprocessing] Removing unnecessary columns...")

    cleaned_X = X.copy()
    columns_to_remove: list[str] = []

    for column in cleaned_X.columns:
        column_lower = column.lower()

        if any(keyword in column_lower for keyword in ID_KEYWORDS):
            columns_to_remove.append(column)
            continue

        if any(keyword in column_lower for keyword in NAME_KEYWORDS):
            columns_to_remove.append(column)
            continue

        if any(keyword in column_lower for keyword in DATE_KEYWORDS):
            columns_to_remove.append(column)
            continue

        if pd.api.types.is_datetime64_any_dtype(cleaned_X[column]):
            columns_to_remove.append(column)

    if columns_to_remove:
        cleaned_X = cleaned_X.drop(columns=columns_to_remove)
        print(f"[Preprocessing] Removed columns: {columns_to_remove}")
    else:
        print(
            "[Preprocessing] No ID, name, or date columns were found in the current dataset."
        )

    return cleaned_X


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using median for numeric and mode for categorical."""
    print("\n[Preprocessing] Handling missing values...")

    filled_X = X.copy()
    missing_counts = filled_X.isna().sum()
    total_missing = int(missing_counts.sum())

    if total_missing == 0:
        print("[Preprocessing] No missing values found.")
        return filled_X

    print(f"[Preprocessing] Total missing values before treatment: {total_missing}")
    print(missing_counts[missing_counts > 0])

    numeric_columns = filled_X.select_dtypes(include="number").columns
    categorical_columns = filled_X.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        if filled_X[column].isna().any():
            median_value = filled_X[column].median()
            filled_X[column] = filled_X[column].fillna(median_value)

    for column in categorical_columns:
        if filled_X[column].isna().any():
            mode_series = filled_X[column].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
            filled_X[column] = filled_X[column].fillna(fill_value)

    print(
        "[Preprocessing] Missing values after treatment: "
        f"{int(filled_X.isna().sum().sum())}"
    )
    return filled_X


def handle_outliers_iqr(X: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric outliers using IQR-based lower and upper bounds."""
    print("\n[Preprocessing] Handling outliers with IQR clipping...")

    clipped_X = X.copy()
    numeric_columns = clipped_X.select_dtypes(include="number").columns
    outlier_summary: list[dict] = []

    if len(numeric_columns) == 0:
        print("[Preprocessing] No numeric columns available for outlier treatment.")
        pd.DataFrame(columns=["variable", "q1", "q3", "iqr", "lower_bound", "upper_bound", "n_outliers"]).to_csv(
            OUTLIER_SUMMARY_PATH,
            index=False,
        )
        return clipped_X

    for column in numeric_columns:
        q1 = clipped_X[column].quantile(0.25)
        q3 = clipped_X[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (clipped_X[column] < lower_bound) | (clipped_X[column] > upper_bound)
        n_outliers = int(outlier_mask.sum())

        clipped_X[column] = clipped_X[column].clip(lower=lower_bound, upper=upper_bound)

        outlier_summary.append(
            {
                "variable": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_outliers": n_outliers,
            }
        )

    outlier_summary_df = pd.DataFrame(outlier_summary)
    outlier_summary_df.to_csv(OUTLIER_SUMMARY_PATH, index=False)

    total_outliers = int(outlier_summary_df["n_outliers"].sum())
    print(f"[Preprocessing] Total detected outliers before clipping: {total_outliers}")
    print(f"[Preprocessing] Outlier summary saved to: {OUTLIER_SUMMARY_PATH}")

    return clipped_X


def _encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode non-numeric columns if any remain before scaling."""
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    if not categorical_columns:
        return X

    print(
        "[Preprocessing] Encoding categorical columns before scaling: "
        f"{categorical_columns}"
    )
    return pd.get_dummies(X, columns=categorical_columns, drop_first=False)


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler and preserve dataframe structure."""
    print("\n[Preprocessing] Scaling features with StandardScaler...")

    features_to_scale = _encode_categorical_features(X.copy())
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features_to_scale)
    X_scaled = pd.DataFrame(
        scaled_array,
        columns=features_to_scale.columns,
        index=features_to_scale.index,
    )

    print(f"[Preprocessing] Scaled feature matrix shape: {X_scaled.shape}")
    return X_scaled, scaler


def apply_pca(X_scaled: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, PCA]:
    """Apply PCA for low-dimensional visualization."""
    print(f"\n[Preprocessing] Applying PCA with {n_components} components...")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    transformed_array = pca.fit_transform(X_scaled)
    component_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(transformed_array, columns=component_names, index=X_scaled.index)

    print(
        "[Preprocessing] PCA completed. Explained variance ratio: "
        f"{pca.explained_variance_ratio_}"
    )
    return pca_df, pca


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """Execute the full preprocessing pipeline and save the processed dataset."""
    print("\n[Preprocessing] Starting preprocessing pipeline...")
    print(f"[Preprocessing] Input shape: {X.shape}")

    X_cleaned = remove_unnecessary_columns(X)
    X_filled = handle_missing_values(X_cleaned)
    X_clipped = handle_outliers_iqr(X_filled)
    X_processed, _ = scale_features(X_clipped)

    X_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"[Preprocessing] Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"[Preprocessing] Output shape: {X_processed.shape}")
    return X_processed


def reattach_target(features_dataframe: pd.DataFrame, target_series: pd.Series) -> pd.DataFrame:
    """Reattach the target after clustering for interpretation only."""
    output = features_dataframe.copy()
    output[target_series.name] = target_series.values
    return output
