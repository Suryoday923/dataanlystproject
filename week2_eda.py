"""Week 2: Exploratory Data Analysis and Visualization.
Install: python -m pip install pandas matplotlib seaborn
Run: python week2_eda.py
Place titanic.csv beside this script, or update DATA_URL to download it.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
BASE = Path(__file__).parent
DATA_PATH = BASE / "titanic_eda_dataset.csv"
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.read_csv(DATA_URL)

print("Shape:", df.shape)
print(df.head())
print("\nNumeric summary:\n", df.describe(include="number").T)
print("\nMissing values:\n", df.isna().sum())
print("\nSurvival counts:\n", df["Survived"].value_counts())
print("\nSurvival by sex:\n", df.groupby("Sex")["Survived"].agg(["mean", "count"]))
print("\nSurvival by class:\n", df.groupby("Pclass")["Survived"].agg(["mean", "count"]))

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.countplot(data=df, x="Survived", color="steelblue", ax=ax)
ax.set(title="Passenger Survival Counts",
       xlabel="Survival status (0 = did not survive, 1 = survived)",
       ylabel="Passenger count")
fig.tight_layout(); fig.savefig(OUT / "01_survival_counts.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=df, x="Sex", y="Survived", errorbar=None, color="steelblue", ax=ax)
ax.set(title="Survival Rate by Sex", xlabel="Sex", ylabel="Survival rate", ylim=(0, 1))
fig.tight_layout(); fig.savefig(OUT / "02_survival_by_sex.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None, order=[1,2,3], color="steelblue", ax=ax)
ax.set(title="Survival Rate by Passenger Class", xlabel="Passenger class (1 = first, 3 = third)",
       ylabel="Survival rate", ylim=(0, 1))
fig.tight_layout(); fig.savefig(OUT / "03_survival_by_class.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.histplot(data=df, x="Age", hue="Survived", bins=25, kde=True, element="step",
             stat="count", common_norm=False, ax=ax)
ax.set(title="Age Distribution by Survival Status", xlabel="Age (years)", ylabel="Passenger count")
fig.tight_layout(); fig.savefig(OUT / "04_age_distribution.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Survived", ax=ax)
ax.set(title="Fare by Passenger Class and Survival", xlabel="Passenger class", ylabel="Fare")
ax.legend(title="Survived", labels=["No", "Yes"])
fig.tight_layout(); fig.savefig(OUT / "05_fare_by_class.png", dpi=160); plt.close(fig)

numeric = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Matrix of Numeric Features")
fig.tight_layout(); fig.savefig(OUT / "06_correlation_heatmap.png", dpi=160); plt.close(fig)

print(f"\nSaved six annotated visualizations to: {OUT.resolve()}")
