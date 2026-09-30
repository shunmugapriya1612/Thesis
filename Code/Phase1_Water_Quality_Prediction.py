# ================================================================
# PHASE 1 - WATER QUALITY PREDICTION
# Dataset: Water Potability
#
# Purpose:
# 1. Basic dataset analysis
# 2. Data quality analysis
# 3. Exploratory data analysis and visualization
# 4. Data preprocessing
# 5. Train / validation / test split
# 6. Baseline MLP classifier
# 7. Baseline performance evaluation
#
# IMPORTANT:
# - No tables, plots, metrics, or model information are saved.
# - Visualizations are displayed only.
# - The test set remains untouched until final evaluation.
# ================================================================

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
# ================================================================
# RESULTS DIRECTORY
# ================================================================

RESULTS_DIR = Path("results") / "phase1_water_quality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ================================================================
# 2. PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Data" / "water_potability.csv"


print("\n" + "=" * 70)
print("PHASE 1 - WATER QUALITY PREDICTION")
print("=" * 70)

print("\nDataset path:")
print(DATA_PATH)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n" + "=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Please place water_potability.csv inside the Data folder."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())


# ================================================================
# 4. CLEAN COLUMN NAMES
# ================================================================

print("\n" + "=" * 70)
print("2. CLEANING COLUMN NAMES")
print("=" * 70)

df.columns = df.columns.str.strip()

print("\nCleaned columns:")
print(df.columns.tolist())


# ================================================================
# 5. IDENTIFY TARGET
# ================================================================

target_candidates = [
    col for col in df.columns
    if col.lower() == "potability"
]

if not target_candidates:
    raise ValueError(
        "Target column 'Potability' was not found."
    )

TARGET_COLUMN = target_candidates[0]

print("\nTarget column:")
print(TARGET_COLUMN)


# ================================================================
# 6. FEATURES AND TARGET
# ================================================================

X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# Convert target to numeric if required
y = pd.to_numeric(y, errors="coerce")

if y.isna().any():
    raise ValueError("Target contains invalid or missing values.")

y = y.astype(int)

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget values:")
print(np.sort(y.unique()))


# ================================================================
# 7. BASIC DATASET INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("3. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeature data types:")
print(X.dtypes)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(X.describe().T)


# ================================================================
# 8. TARGET ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("4. TARGET ANALYSIS")
print("=" * 70)

class_counts = y.value_counts().sort_index()

print("\nNumber of classes:")
print(y.nunique())

print("\nClass distribution:")
print(class_counts)

print("\nClass percentages:")
print((class_counts / len(y) * 100).round(2))


# ================================================================
# 9. TARGET DISTRIBUTION VISUALIZATION
# ================================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y,
    hue=y,
    legend=False
)

plt.title("Water Potability Class Distribution")
plt.xlabel("Potability Class")
plt.ylabel("Number of Samples")

plt.xticks(
    ticks=[0, 1],
    labels=["Non-Potable (0)", "Potable (1)"]
)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "01_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 10. MISSING VALUE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("5. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_counts = X.isnull().sum()

missing_percentages = (
    missing_counts / len(X) * 100
).round(2)

missing_table = pd.DataFrame({
    "Missing Values": missing_counts,
    "Percentage": missing_percentages
})

print("\nMissing values:")
print(missing_table)

print("\nTotal missing values:")
print(int(missing_counts.sum()))


# ================================================================
# 11. MISSING VALUE VISUALIZATION
# ================================================================

missing_nonzero = missing_counts[missing_counts > 0]

if len(missing_nonzero) > 0:

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=missing_nonzero.index,
        y=missing_nonzero.values
    )

    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Number of Missing Values")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "02_missing_values.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

else:
    print("\nNo missing values detected.")


# ================================================================
# 12. DUPLICATE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("6. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_features = X.duplicated().sum()
duplicate_complete = df.duplicated().sum()

print(f"\nDuplicate feature rows: {duplicate_features}")
print(f"Duplicate complete rows: {duplicate_complete}")


# ================================================================
# 13. FEATURE DESCRIPTIVE VISUALIZATION
# ================================================================

print("\n" + "=" * 70)
print("7. FEATURE DISTRIBUTIONS")
print("=" * 70)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

n_features = len(numeric_features)

n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 4 * n_rows)
)

axes = np.array(axes).reshape(-1)

for i, feature in enumerate(numeric_features):

    axes[i].hist(
        X[feature].dropna(),
        bins=30,
        edgecolor="black"
    )

    axes[i].set_title(feature)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Feature Distributions",
    fontsize=16
)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "03_feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 14. BOXPLOTS / OUTLIER INSPECTION
# ================================================================

print("\n" + "=" * 70)
print("8. FEATURE RANGE AND OUTLIER INSPECTION")
print("=" * 70)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 4 * n_rows)
)

axes = np.array(axes).reshape(-1)

for i, feature in enumerate(numeric_features):

    axes[i].boxplot(
        X[feature].dropna()
    )

    axes[i].set_title(feature)
    axes[i].set_ylabel("Value")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Feature Boxplots",
    fontsize=16
)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "04_feature_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 15. CORRELATION ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("9. CORRELATION ANALYSIS")
print("=" * 70)

correlation_matrix = X.corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(3))


plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")


plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "05_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 16. FEATURE CORRELATION WITH TARGET
# ================================================================

correlation_with_target = (
    df.corr(numeric_only=True)[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .sort_values()
)

print("\n" + "=" * 70)
print("10. FEATURE CORRELATION WITH TARGET")
print("=" * 70)

print(
    correlation_with_target.round(4)
)


plt.figure(figsize=(9, 5))

sns.barplot(
    x=correlation_with_target.values,
    y=correlation_with_target.index,
    orient="h"
)

plt.title("Feature Correlation with Potability")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "06_feature_target_correlation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ================================================================
# 17. PREPARE FEATURES
# ================================================================

print("\n" + "=" * 70)
print("11. FEATURE PREPARATION")
print("=" * 70)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nFeature data types after numeric conversion:")
print(X.dtypes)


# ================================================================
# 18. THREE-WAY DATA SPLIT
# ================================================================
#
# 80% development data
# 20% final test data
#
# Development data:
#   80% train
#   20% validation
#
# Overall:
#   64% training
#   16% validation
#   20% test
#
# The test set is kept untouched until the final evaluation.
# ================================================================

print("\n" + "=" * 70)
print("12. TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_STATE
)

print("\nTraining samples:")
print(len(X_train))

print("\nValidation samples:")
print(len(X_val))

print("\nTest samples:")
print(len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_val.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())


# ================================================================
# 19. MISSING VALUE IMPUTATION
# ================================================================

print("\n" + "=" * 70)
print("13. MISSING VALUE IMPUTATION")
print("=" * 70)

# IMPORTANT:
# The imputer is fitted ONLY on training data.

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

print("\nImputation completed.")

print(f"Training matrix: {X_train_imputed.shape}")
print(f"Validation matrix: {X_val_imputed.shape}")
print(f"Test matrix: {X_test_imputed.shape}")


# ================================================================
# 20. FEATURE SCALING
# ================================================================

print("\n" + "=" * 70)
print("14. FEATURE SCALING")
print("=" * 70)

# IMPORTANT:
# The scaler is fitted ONLY on training data.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("\nFeature scaling completed.")


# ================================================================
# 21. BASELINE MLP
# ================================================================

print("\n" + "=" * 70)
print("15. BASELINE MLP CLASSIFIER")
print("=" * 70)

BASELINE_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "batch_size": "auto",
    "max_iter": 500,
    "random_state": RANDOM_STATE
}

print("\nBaseline MLP configuration:")

for parameter, value in BASELINE_PARAMS.items():
    print(f"{parameter}: {value}")


# ================================================================
# 22. TRAIN BASELINE MODEL
# ================================================================

model = MLPClassifier(
    **BASELINE_PARAMS
)

start_time = time.perf_counter()

model.fit(
    X_train_scaled,
    y_train
)

training_time = time.perf_counter() - start_time


# ================================================================
# 23. VALIDATION PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("16. VALIDATION PERFORMANCE")
print("=" * 70)

start_time = time.perf_counter()

y_val_pred = model.predict(X_val_scaled)

validation_prediction_time = (
    time.perf_counter() - start_time
)

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_precision = precision_score(
    y_val,
    y_val_pred,
    average="macro",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    y_val_pred,
    average="macro",
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro",
    zero_division=0
)

val_weighted_f1 = f1_score(
    y_val,
    y_val_pred,
    average="weighted",
    zero_division=0
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_val_pred
)

print(f"\nValidation Accuracy:           {val_accuracy:.4f}")
print(f"Validation Macro Precision:   {val_precision:.4f}")
print(f"Validation Macro Recall:      {val_recall:.4f}")
print(f"Validation Macro F1:          {val_f1:.4f}")
print(f"Validation Weighted F1:       {val_weighted_f1:.4f}")
print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")


# ================================================================
# 24. FINAL TEST PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("17. FINAL TEST PERFORMANCE")
print("=" * 70)

print(
    "\nThe final test set is evaluated only after the baseline model "
    "has been trained."
)

start_time = time.perf_counter()

y_test_pred = model.predict(X_test_scaled)

test_prediction_time = (
    time.perf_counter() - start_time
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_test_pred
)

print(f"\nTest Accuracy:           {test_accuracy:.4f}")
print(f"Test Macro Precision:   {test_precision:.4f}")
print(f"Test Macro Recall:      {test_recall:.4f}")
print(f"Test Macro F1:          {test_f1:.4f}")
print(f"Test Weighted F1:       {test_weighted_f1:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

print(f"\nTraining Time:           {training_time:.4f} seconds")
print(
    f"Validation Prediction:  "
    f"{validation_prediction_time:.6f} seconds"
)
print(
    f"Test Prediction Time:   "
    f"{test_prediction_time:.6f} seconds"
)

print(f"\nIterations:              {model.n_iter_}")
print(f"Final Training Loss:     {model.loss_:.6f}")


# ================================================================
# 25. CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("18. CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Non-Potable (0)",
            "Potable (1)"
        ],
        zero_division=0
    )
)


# ================================================================
# 26. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("19. CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion matrix:")
print(cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Non-Potable (0)",
        "Potable (1)"
    ],
    yticklabels=[
        "Non-Potable (0)",
        "Potable (1)"]
)

plt.title("Baseline MLP - Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "07_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ================================================================
# 27. TRAINING LOSS CURVE
# ================================================================

print("\n" + "=" * 70)
print("20. TRAINING LOSS CURVE")
print("=" * 70)

print(
    f"\nNumber of training iterations: {len(model.loss_curve_)}"
)

print(
    f"Initial loss: {model.loss_curve_[0]:.6f}"
)

print(
    f"Final loss: {model.loss_curve_[-1]:.6f}"
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(model.loss_curve_) + 1),
    model.loss_curve_
)

plt.title("Baseline MLP Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "08_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 28. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)

print("\nDataset:")
print("Water Potability")

print(f"\nTotal samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")

print("\nClass distribution:")

for class_value, count in class_counts.items():
    percentage = count / len(y) * 100

    print(
        f"Class {class_value}: "
        f"{count} samples ({percentage:.2f}%)"
    )

print("\nMissing values:")
print(f"Total missing values: {missing_counts.sum()}")

print("\nBaseline MLP:")
print(f"Test Accuracy:          {test_accuracy:.4f}")
print(f"Test Macro F1:          {test_f1:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
print(f"Training Time:          {training_time:.4f} sec")
print(f"Iterations:             {model.n_iter_}")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETED")
print("=" * 70)