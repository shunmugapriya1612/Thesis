from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

OUTPUT_DIR = "processed_datasets"
RESULTS_DIR = "results/phase1_solar_flare"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\nSTARTING SOLAR FLARE DATASET ANALYSIS")
print("=" * 80)


# =====================================================
# LOAD SOLAR FLARE DATASET
# =====================================================

print("\nProcessing Solar Flare Dataset...")

solar = fetch_ucirepo(id=89)

X = solar.data.features
y = solar.data.targets

df = pd.concat([X, y], axis=1)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nVariables:")
print(solar.variables)


# =====================================================
# SAVE ORIGINAL DATASET
# =====================================================

df.to_csv(
    f"{OUTPUT_DIR}/solar_flare_original.csv",
    index=False
)

print("\nSaved: solar_flare_original.csv")


# =====================================================
# BASIC DATASET ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("BASIC DATASET ANALYSIS")
print("=" * 80)

print("\nNumber of Observations:")
print(len(df))

print("\nNumber of Variables:")
print(len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())


# =====================================================
# DATA TYPES
# =====================================================

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(df.dtypes)


# =====================================================
# MISSING VALUE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_summary)

print(
    f"\nTotal Missing Values: {df.isnull().sum().sum()}"
)


# =====================================================
# DUPLICATE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(
    f"Number of Duplicate Rows: {duplicate_count}"
)


# =====================================================
# IDENTIFY FEATURE TYPES
# =====================================================

categorical_columns = [
    "modified Zurich class",
    "largest spot size",
    "spot distribution"
]

target_columns = [
    "common flares",
    "moderate flares",
    "severe flares"
]

feature_columns = [
    column
    for column in df.columns
    if column not in target_columns
]

numerical_columns = [
    column
    for column in feature_columns
    if column not in categorical_columns
]

print("\n" + "=" * 80)
print("FEATURE TYPES")
print("=" * 80)

print("\nCategorical Features:")
print(categorical_columns)

print("\nNumerical Features:")
print(numerical_columns)


# =====================================================
# CATEGORICAL FEATURE DISTRIBUTIONS
# =====================================================

print("\n" + "=" * 80)
print("CATEGORICAL FEATURE DISTRIBUTIONS")
print("=" * 80)

for column in categorical_columns:

    print(f"\n{column}:")

    print(
        df[column]
        .value_counts(dropna=False)
    )


# =====================================================
# NUMERICAL FEATURE SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 80)

numerical_summary = df[numerical_columns].describe()

print(numerical_summary)


# =====================================================
# TARGET PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("TARGET PREPARATION")
print("=" * 80)

# Create a single categorical target based on
# the highest severity of the observed flare.

df["Target"] = "No Flare"

df.loc[
    df["common flares"] == 1,
    "Target"
] = "Common Flare"

df.loc[
    df["moderate flares"] == 1,
    "Target"
] = "Moderate Flare"

df.loc[
    df["severe flares"] == 1,
    "Target"
] = "Severe Flare"


# =====================================================
# TARGET DISTRIBUTION
# =====================================================

target_counts = (
    df["Target"]
    .value_counts()
)

print("\nTarget Class Distribution:")
print(target_counts)


# =====================================================
# TARGET CLASS PERCENTAGES
# =====================================================

target_percentages = (
    df["Target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

target_summary = pd.DataFrame({
    "Count": target_counts,
    "Percentage": target_percentages
})

print("\nTarget Distribution with Percentages:")
print(target_summary)


# =====================================================
# CLASS IMBALANCE RATIO
# =====================================================

majority_class_count = target_counts.max()
minority_class_count = target_counts.min()

imbalance_ratio = (
    majority_class_count /
    minority_class_count
)

print(
    f"\nMajority Class Count: "
    f"{majority_class_count}"
)

print(
    f"Minority Class Count: "
    f"{minority_class_count}"
)

print(
    f"Majority-to-Minority Ratio: "
    f"{imbalance_ratio:.2f}:1"
)


# =====================================================
# SAVE TARGET DISTRIBUTION
# =====================================================

target_summary.to_csv(
    f"{RESULTS_DIR}/target_distribution.csv"
)


# =====================================================
# CATEGORICAL FEATURE vs TARGET ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("CATEGORICAL FEATURE VS TARGET ANALYSIS")
print("=" * 80)

for column in categorical_columns:

    print(
        f"\n--- {column} vs Target ---"
    )

    cross_tab = pd.crosstab(
        df[column],
        df["Target"]
    )

    print(cross_tab)

    cross_tab.to_csv(
        f"{RESULTS_DIR}/"
        f"{column.replace(' ', '_')}_vs_target.csv"
    )


# =====================================================
# NUMERICAL FEATURE VS TARGET ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("NUMERICAL FEATURE VS TARGET ANALYSIS")
print("=" * 80)

for column in numerical_columns:

    print(
        f"\n--- {column} by Target ---"
    )

    grouped_summary = (
        df.groupby("Target")[column]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max"
            ]
        )
        .round(3)
    )

    print(grouped_summary)

    grouped_summary.to_csv(
        f"{RESULTS_DIR}/"
        f"{column.replace(' ', '_')}_by_target.csv"
    )


# =====================================================
# NUMERICAL FEATURE CORRELATION
# =====================================================

print("\n" + "=" * 80)
print("NUMERICAL FEATURE CORRELATION")
print("=" * 80)

correlation_matrix = (
    df[numerical_columns]
    .corr()
    .round(3)
)

print(correlation_matrix)

correlation_matrix.to_csv(
    f"{RESULTS_DIR}/numerical_correlation_matrix.csv"
)


# =====================================================
# TARGET VISUALIZATION
# =====================================================

plt.figure(figsize=(8, 5))

target_counts.plot(
    kind="bar"
)

plt.title(
    "Solar Flare Target Class Distribution"
)

plt.xlabel("Flare Class")
plt.ylabel("Number of Observations")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/target_distribution.png",
    dpi=300
)

plt.close()


# =====================================================
# CATEGORICAL FEATURE VISUALIZATIONS
# =====================================================

for column in categorical_columns:

    distribution = (
        df[column]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    distribution.plot(
        kind="bar"
    )

    plt.title(
        f"{column} Distribution"
    )

    plt.xlabel(column)
    plt.ylabel("Number of Observations")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/"
        f"{column.replace(' ', '_')}_distribution.png",
        dpi=300
    )

    plt.close()


# =====================================================
# REMOVE ORIGINAL TARGET VARIABLES
# =====================================================

df = df.drop(
    columns=[
        "common flares",
        "moderate flares",
        "severe flares"
    ]
)


# =====================================================
# FINAL PROCESSED DATASET INFORMATION
# =====================================================

print("\n" + "=" * 80)
print("FINAL PROCESSED DATASET")
print("=" * 80)

print("\nProcessed Dataset Shape:")
print(df.shape)

print("\nProcessed Dataset Columns:")
print(df.columns.tolist())

print("\nFinal Target Distribution:")
print(
    df["Target"].value_counts()
)


# =====================================================
# SAVE PROCESSED DATASET
# =====================================================

df.to_csv(
    f"{OUTPUT_DIR}/solar_flare_processed.csv",
    index=False
)

print(
    "\nSaved: "
    "solar_flare_processed.csv"
)


# =====================================================
# COMPLETION MESSAGE
# =====================================================

print("\n" + "=" * 80)
print("SOLAR FLARE PHASE 1 COMPLETED")
print("=" * 80)

print(
    "\nResults saved in:"
    f"\n{RESULTS_DIR}"
)

print(
    "\nProcessed dataset saved in:"
    f"\n{OUTPUT_DIR}/solar_flare_processed.csv"
)