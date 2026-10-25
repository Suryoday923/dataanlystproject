"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: Titanic passenger dataset (Seaborn-data GitHub repository)

Install dependencies:
    python -m pip install pandas numpy matplotlib requests

Run:
    python data_cleaning.py
"""

from pathlib import Path
import requests
import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "titanic_raw.csv"
CLEAN_PATH = DATA_DIR / "titanic_cleaned.csv"
PLOT_PATH = DATA_DIR / "data_quality_plots.png"


def download_dataset() -> pd.DataFrame:
    """Download and preserve the original CSV, then load it with Pandas."""
    print("Downloading Titanic dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    RAW_PATH.write_bytes(response.content)
    data = pd.read_csv(RAW_PATH)
    print(f"Saved raw dataset to: {RAW_PATH.resolve()}")
    return data


def show_initial_exploration(data: pd.DataFrame) -> None:
    """Print structure, descriptive statistics, missingness, and duplicates."""
    print("\n--- INITIAL EXPLORATION ---")
    print("Dataset shape:", data.shape)
    print("\nColumn names:", data.columns.tolist())
    print("\nData types and non-null counts:")
    data.info()
    print("\nFirst five rows:")
    print(data.head())
    print("\nNumeric descriptive statistics:")
    print(data.describe(include="number").T)

    missing = pd.DataFrame({
        "missing_count": data.isna().sum(),
        "missing_percent": (data.isna().mean() * 100).round(2),
    }).sort_values("missing_count", ascending=False)
    print("\nMissing-value profile:")
    print(missing)

    print("\nExact duplicate rows:", int(data.duplicated().sum()))
    for column in data.select_dtypes(include=["object", "category"]).columns:
        print(f"\nCategory counts for {column}:")
        print(data[column].value_counts(dropna=False).head(20))


def report_iqr_outliers(series: pd.Series, column_name: str) -> dict:
    """Flag potential outliers using IQR fences; do not automatically delete them."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"column": column_name, "message": "No numeric values to assess."}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = (values < lower) | (values > upper)

    result = {
        "column": column_name,
        "q1": round(float(q1), 3),
        "q3": round(float(q3), 3),
        "iqr": round(float(iqr), 3),
        "lower_fence": round(float(lower), 3),
        "upper_fence": round(float(upper), 3),
        "outlier_count": int(flagged.sum()),
        "outlier_percent": round(float(flagged.mean() * 100), 2),
    }
    print(f"\nIQR outlier report for {column_name}:")
    print(result)
    return result


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply documented, conservative preprocessing decisions."""
    clean = raw.copy()

    # Trim accidental whitespace in string values.
    text_columns = clean.select_dtypes(include=["object", "category"]).columns
    for column in text_columns:
        clean[column] = clean[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Standardize known lower-case embarkation codes without guessing unknown values.
    if "embarked" in clean.columns:
        clean["embarked"] = clean["embarked"].replace(
            {"s": "S", "c": "C", "q": "Q"}
        )

    # Impute age with its median; median is less sensitive to skew than mean.
    if "age" in clean.columns and clean["age"].notna().any():
        clean["age"] = clean["age"].fillna(clean["age"].median())

    # Impute missing embarkation categories with the most frequent category.
    if "embarked" in clean.columns and clean["embarked"].notna().any():
        clean["embarked"] = clean["embarked"].fillna(
            clean["embarked"].mode().iloc[0]
        )

    # Deck is highly incomplete and is excluded from this example's analysis-ready
    # copy. The raw CSV still preserves it for any future project.
    if "deck" in clean.columns:
        clean = clean.drop(columns=["deck"])

    # Report remaining missingness rather than silently filling every field.
    print("\n--- REMAINING MISSING VALUES AFTER CLEANING ---")
    remaining = clean.isna().sum()
    print(remaining[remaining > 0] if (remaining > 0).any() else "No missing values.")
    return clean


def validate_values(data: pd.DataFrame) -> None:
    """Check selected categorical domains and basic numeric sanity."""
    print("\n--- VALIDATION CHECKS ---")
    expected = {
        "survived": {0, 1},
        "pclass": {1, 2, 3},
        "sex": {"male", "female"},
        "embarked": {"C", "Q", "S"},
    }

    for column, allowed_values in expected.items():
        if column in data.columns:
            observed = set(data[column].dropna().unique())
            print(f"{column}: unexpected values = {observed - allowed_values}")

    if "age" in data.columns:
        print("Age values below zero:", int((data["age"] < 0).sum()))
    if "fare" in data.columns:
        print("Fare values below zero:", int((data["fare"] < 0).sum()))


def save_plots(data: pd.DataFrame) -> None:
    """Save a histogram and boxplot to support visual data-quality inspection."""
    if "age" not in data.columns or "fare" not in data.columns:
        return

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    data["age"].dropna().plot(
        kind="hist", bins=25, ax=axes[0], title="Age distribution"
    )
    data["fare"].dropna().plot(
        kind="box", ax=axes[1], title="Fare boxplot"
    )
    axes[0].set_xlabel("Age")
    axes[1].set_ylabel("Fare")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)
    print(f"\nSaved plots to: {PLOT_PATH.resolve()}")


def main() -> None:
    raw = download_dataset()
    show_initial_exploration(raw)

    # Detect outliers before changing the dataset; retain flagged values pending review.
    for column in ["age", "fare"]:
        if column in raw.columns:
            report_iqr_outliers(raw[column], column)

    cleaned = clean_dataset(raw)
    validate_values(cleaned)
    save_plots(cleaned)

    print("\n--- FINAL SUMMARY ---")
    print("Raw shape:", raw.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Exact duplicate rows in cleaned data:", int(cleaned.duplicated().sum()))

    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_PATH.resolve()}")


if __name__ == "__main__":
    main()
