from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# =====================================================
# Create output folders
# =====================================================

OUTPUT_DIR = "processed_datasets"
VISUALIZATION_DIR = os.path.join(
    "results",
    "phase1_smart_grid_stability",
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


print("\nSTARTING DATASET PREPARATION")
print("=" * 80)


# =====================================================
# DATASET - SMART GRID STABILITY
# =====================================================

print("\nProcessing Smart Grid Stability...")

grid = fetch_ucirepo(id=471)

X = grid.data.features
y = grid.data.targets

df = pd.concat([X, y], axis=1)


# =====================================================
# DATASET INFORMATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nVariables:")
print(grid.variables)


# =====================================================
# SAVE ORIGINAL DATASET
# =====================================================

df.to_csv(
    f"{OUTPUT_DIR}/smart_grid_stability_original.csv",
    index=False
)

print("Saved: smart_grid_stability_original.csv")


# =====================================================
# TARGET PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("TARGET PREPARATION")
print("=" * 80)


# Rename the original target column to maintain a
# consistent target variable across all datasets.

df = df.rename(
    columns={
        "stabf": "Target"
    }
)


# =====================================================
# TARGET CLASS DISTRIBUTION
# =====================================================

target_distribution = (
    df["Target"]
    .value_counts()
)

print("\nTarget Class Distribution:")
print(target_distribution)


# =====================================================
# TARGET PERCENTAGES
# =====================================================

target_percentages = (
    df["Target"]
    .value_counts(normalize=True)
    * 100
)

print("\nTarget Class Percentages:")
print(target_percentages.round(2))


# =====================================================
# SAVE PROCESSED DATASET
# =====================================================

df.to_csv(
    f"{OUTPUT_DIR}/smart_grid_stability_processed.csv",
    index=False
)

print("\nSaved: smart_grid_stability_processed.csv")


# =====================================================
# VISUALIZATION PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("GENERATING PHASE 1 VISUALIZATIONS")
print("=" * 80)


# =====================================================
# DEFINE MODEL FEATURES
# =====================================================

FEATURE_COLUMNS = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4"
]


# =====================================================
# CREATE NUMERIC TARGET FOR CORRELATION ANALYSIS
# =====================================================

# Stable = 0
# Unstable = 1

target_numeric = (
    df["Target"]
    .map({
        "stable": 0,
        "unstable": 1
    })
)


# =====================================================
# VISUALIZATION 1
# TARGET CLASS DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=target_distribution.index,
    y=target_distribution.values
)

plt.title(
    "Smart Grid Stability Target Class Distribution"
)

plt.xlabel(
    "Grid Stability Class"
)

plt.ylabel(
    "Number of Samples"
)


# Add counts and percentages above bars

for i, value in enumerate(target_distribution.values):

    percentage = (
        value / len(df) * 100
    )

    ax.text(
        i,
        value + 100,
        f"{value:,}\n({percentage:.2f}%)",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "01_target_class_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# =====================================================
# VISUALIZATION 2
# FEATURE DISTRIBUTIONS BY TARGET CLASS
# =====================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)

axes = axes.flatten()

for i, feature in enumerate(FEATURE_COLUMNS):

    sns.boxplot(
        data=df,
        x="Target",
        y=feature,
        ax=axes[i]
    )

    axes[i].set_title(
        f"{feature} by Stability Class"
    )

    axes[i].set_xlabel(
        "Stability Class"
    )

    axes[i].set_ylabel(
        feature
    )


plt.suptitle(
    "Smart Grid Stability Feature Distributions by Target Class",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "02_feature_distributions_by_target.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# =====================================================
# VISUALIZATION 3
# FEATURE CORRELATION HEATMAP
# =====================================================

correlation_matrix = df[
    FEATURE_COLUMNS
].corr()


plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    "Smart Grid Stability Feature Correlation Heatmap"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "03_feature_correlation_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# =====================================================
# VISUALIZATION 4
# FEATURE-TARGET RELATIONSHIPS
# =====================================================

feature_target_correlations = {}

for feature in FEATURE_COLUMNS:

    feature_target_correlations[feature] = (
        df[feature]
        .corr(target_numeric)
    )


feature_target_series = (
    pd.Series(
        feature_target_correlations
    )
    .sort_values()
)


plt.figure(
    figsize=(10, 7)
)

feature_target_series.plot(
    kind="barh"
)

plt.axvline(
    0,
    linewidth=1
)

plt.title(
    "Smart Grid Stability Feature-Target Relationships"
)

plt.xlabel(
    "Correlation with Stability Target"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "04_feature_target_relationships.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# =====================================================
# OPTIONAL DATASET INFORMATION
# =====================================================

print("\n" + "=" * 80)
print("SMART GRID STABILITY DATASET SUMMARY")
print("=" * 80)

print("\nTotal Samples:")
print(len(df))

print("\nTotal Columns:")
print(len(df.columns))

print("\nModel Features:")
print(len(FEATURE_COLUMNS))

print("\nFeature List:")
print(FEATURE_COLUMNS)

print("\nTarget Distribution:")
print(target_distribution)

print("\nTarget Percentages:")
print(target_percentages.round(2))

print("\nMissing Values:")
print(
    df.isnull().sum().sum()
)

print("\nDuplicate Rows:")
print(
    df.duplicated().sum()
)

print("\nVisualizations saved to:")
print(VISUALIZATION_DIR)


print("\n" + "=" * 80)
print("PHASE 1 SMART GRID STABILITY DATASET ANALYSIS COMPLETED")
print("=" * 80)