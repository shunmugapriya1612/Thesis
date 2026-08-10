from sklearn.datasets import fetch_openml
import pandas as pd
import os

# =====================================================
# Create output folder
# =====================================================

OUTPUT_DIR = "processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSTARTING DATASET PREPARATION")
print("=" * 80)

# =====================================================
# DATASET 5 - Electricity
# =====================================================

print("\nProcessing Electricity...")

X, y = fetch_openml(
    data_id=151,
    return_X_y=True,
    as_frame=True
)

df = pd.concat([X, y], axis=1)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFeature Columns:")
print(list(X.columns))

print("\nTarget Column:")
print(y.name)

df.to_csv(
    f"{OUTPUT_DIR}/electricity_original.csv",
    index=False
)

print("Saved: electricity_original.csv")
# =====================================================
# TARGET PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("TARGET PREPARATION")
print("=" * 80)

# Rename the original target column to maintain a
# consistent target variable across all datasets.
df = df.rename(columns={"class": "Target"})

print("\nTarget Class Distribution:")
print(df["Target"].value_counts())

df.to_csv(
    f"{OUTPUT_DIR}/electricity_processed.csv",
    index=False
)

print("Saved: electricity_processed.csv")