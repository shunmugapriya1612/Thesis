from ucimlrepo import fetch_ucirepo
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
# DATASET 4 - Solar Flare
# =====================================================

print("\nProcessing Solar Flare...")

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

df.to_csv(
    f"{OUTPUT_DIR}/solar_flare_original.csv",
    index=False
)

print("Saved: solar_flare_original.csv")

# =====================================================
# TARGET PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("TARGET PREPARATION")
print("=" * 80)

# Create a single categorical target based on
# the highest severity of the observed flare.

df["Target"] = "No Flare"

df.loc[df["common flares"] == 1, "Target"] = "Common Flare"
df.loc[df["moderate flares"] == 1, "Target"] = "Moderate Flare"
df.loc[df["severe flares"] == 1, "Target"] = "Severe Flare"

print("\nTarget Class Distribution:")
print(df["Target"].value_counts())

df = df.drop(
    columns=[
        "common flares",
        "moderate flares",
        "severe flares"
    ]
)

df.to_csv(
    f"{OUTPUT_DIR}/solar_flare_processed.csv",
    index=False
)

print("Saved: solar_flare_processed.csv")

