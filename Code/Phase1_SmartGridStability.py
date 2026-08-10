from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# =====================================================
# Create output folder
# =====================================================

OUTPUT_DIR = "processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSTARTING DATASET PREPARATION")
print("=" * 80)

# =====================================================
# DATASET 2 - Smart Grid Stability
# =====================================================

print("\nProcessing Smart Grid Stability...")

grid = fetch_ucirepo(id=471)

X = grid.data.features
y = grid.data.targets

df = pd.concat([X, y], axis=1)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


print("\nVariables:")
print(grid.variables)

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
df = df.rename(columns={"stabf": "Target"})

print("\nTarget Class Distribution:")
print(df["Target"].value_counts())

df.to_csv(
    f"{OUTPUT_DIR}/smart_grid_stability_processed.csv",
    index=False
)

print("Saved: smart_grid_stability_processed.csv")