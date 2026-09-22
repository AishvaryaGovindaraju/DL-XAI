"""
data_loading.py
===============
Phase 1 - load the CDC BRFSS2015 diabetes indicators CSV and describe it.

Nothing here modifies the data. It only reads the raw file and reports what
is in it, which is the evidence the paper's dataset section is written from.
"""

import pandas as pd

import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw CSV and fail loudly if it is missing or malformed."""
    path = config.RAW_CSV if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    df = pd.read_csv(path)

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' missing from {path}")
    expected_columns = config.N_FEATURES + 1
    if df.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, found {df.shape[1]}")

    return df


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Class counts and percentages for the binary diabetes target."""
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    return pd.DataFrame(
        {
            "count": counts,
            "percent": (counts / len(df) * 100).round(2),
        }
    )


def data_quality_report(df: pd.DataFrame) -> dict:
    """The integrity facts the paper has to state: size, gaps, duplicates."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "unique_values_per_column": df.nunique().to_dict(),
    }


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column mean / std / quartiles, transposed for readability."""
    return df.describe().T


def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each feature with the target, high to low."""
    correlations = df.corr(numeric_only=True)[config.TARGET_COLUMN]
    return correlations.drop(config.TARGET_COLUMN).sort_values(ascending=False)


def run_eda(verbose: bool = True) -> pd.DataFrame:
    """Load the data and print the Phase 1 summary. Returns the DataFrame."""
    df = load_raw_data()
    if not verbose:
        return df

    report = data_quality_report(df)
    print(f"Rows: {report['n_rows']:,}   Columns: {report['n_columns']}")
    print(f"Missing values: {report['n_missing']}   "
          f"Duplicate rows: {report['n_duplicate_rows']:,}")
    print("\nTarget distribution (0 = no diabetes, 1 = diabetes):")
    print(target_distribution(df))
    print("\nCorrelation with target (top 10):")
    print(correlation_with_target(df).head(10))
    return df


if __name__ == "__main__":
    run_eda()


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Loads the raw BRFSS2015 CSV from the path defined in config.py.
# [x] Validates the file exists, the target column is present and the
#     column count is 21 features + 1 target.
# [x] Reports class balance for Diabetes_binary (counts and percentages).
# [x] Reports data-quality facts: row/column counts, missing values,
#     duplicate rows, dtypes, unique values per column.
# [x] Produces descriptive statistics and feature-target correlations.
# [x] run_eda() prints the Phase 1 summary in one call; running this file
#     directly does exactly that.
# [x] Does NOT split, scale or otherwise alter the data - that is
#     preprocessing.py.
# =====================================================================
