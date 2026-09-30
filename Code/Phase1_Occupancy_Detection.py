# ============================================================
# PHASE 1: WATER QUALITY PREDICTION
# EXPLORATORY DATA ANALYSIS AND BASELINE MLP CLASSIFIER
# ============================================================

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this filename if your dataset has another name.
DATA_PATH = PROJECT_ROOT / "Data" / "water_potability.csv"

OUTPUT_DIR = PROJECT_ROOT / "water_quality_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("WATER QUALITY PREDICTION DATASET")
print("=" * 70)

print("\nDataset path:")
print(DATA_PATH)

print("\nOutput directory:")
print(OUTPUT_DIR)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset was not found:\n{DATA_PATH}\n\n"
        "Please check the filename inside the Data folder."
    )

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(data.shape)

print("\nDataset columns:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COLUMN NAMES")
print("=" * 70)

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
    .str.replace(" ", "_")
)

print("\nCleaned columns:")
print(data.columns.tolist())

# ============================================================
# 4. DEFINE TARGET AND FEATURES
# ============================================================

print("\n" + "=" * 70)
print("DEFINING FEATURES AND TARGET")
print("=" * 70)

# The common Water Potability dataset uses "Potability".
TARGET_COLUMN = "Potability"

# Handle different capitalization automatically.
matching_target_columns = [
    column
    for column in data.columns
    if column.lower() == TARGET_COLUMN.lower()
]

if len(matching_target_columns) == 0:
    raise ValueError(
        f"\nTarget column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns:\n{data.columns.tolist()}\n\n"
        "Please change TARGET_COLUMN to the correct target name."
    )

TARGET_COLUMN = matching_target_columns[0]

X_raw = data.drop(columns=[TARGET_COLUMN]).copy()
y_raw = data[TARGET_COLUMN].copy()

print("\nTarget column:")
print(TARGET_COLUMN)

print("\nFeature shape:")
print(X_raw.shape)

print("\nTarget shape:")
print(y_raw.shape)

print("\nTarget values:")
print(y_raw.unique())

# ============================================================
# 5. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print("\nFeature columns:")
print(X_raw.columns.tolist())

print("\nFeature data types:")
print(X_raw.dtypes)

print("\nDataset information:")
data.info()

print("\nDescriptive statistics:")
descriptive_statistics = X_raw.describe().T
print(descriptive_statistics)

descriptive_statistics.to_csv(
    OUTPUT_DIR / "water_quality_descriptive_statistics.csv"
)

# ============================================================
# 6. TARGET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET INFORMATION")
print("=" * 70)

y_raw = y_raw.astype(str).str.strip()

print("\nUnique target values:")
print(y_raw.unique())

print("\nNumber of classes:")
print(y_raw.nunique())

class_distribution = y_raw.value_counts().sort_index()

print("\nTarget class distribution:")
print(class_distribution)

class_distribution.to_csv(
    OUTPUT_DIR / "water_quality_class_distribution.csv",
    header=["count"]
)

# ============================================================
# 7. MISSING VALUES AND DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AND DUPLICATES")
print("=" * 70)

missing_values = X_raw.isnull().sum()

print("\nMissing values by feature:")
print(missing_values)

print("\nTotal missing values:")
print(int(missing_values.sum()))

print("\nDuplicate feature rows:")
print(int(X_raw.duplicated().sum()))

print("\nDuplicate complete rows:")
print(int(data.duplicated().sum()))

# ============================================================
# 8. TARGET DISTRIBUTION PLOT
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

plt.figure(figsize=(8, 5))

sns.countplot(
    x=y_raw,
    order=sorted(y_raw.unique())
)

plt.title("Water Quality Target Distribution")
plt.xlabel("Water Quality Class")
plt.ylabel("Number of Samples")
plt.tight_layout()

target_distribution_plot_path = (
    OUTPUT_DIR / "water_quality_class_distribution.png"
)

plt.savefig(
    target_distribution_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Target distribution plot saved to:",
    target_distribution_plot_path
)

# ============================================================
# 9. ENSURE NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)

# Convert feature columns to numeric where possible.
# Invalid values are converted to NaN and handled by imputation.
X_numeric = X_raw.apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nNumeric feature shape:")
print(X_numeric.shape)

print("\nFeatures after numeric conversion:")
print(X_numeric.dtypes)

# ============================================================
# 10. FEATURE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTIONS")
print("=" * 70)

numeric_features = X_numeric.columns.tolist()

number_of_features = len(numeric_features)
number_of_columns = 3
number_of_rows = int(
    np.ceil(number_of_features / number_of_columns)
)

fig, axes = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(15, number_of_rows * 3)
)

axes = np.array(axes).reshape(-1)

for index, feature in enumerate(numeric_features):
    axes[index].hist(
        X_numeric[feature].dropna(),
        bins=25,
        edgecolor="black"
    )

    axes[index].set_title(feature, fontsize=10)
    axes[index].set_xlabel("Value")
    axes[index].set_ylabel("Frequency")

for index in range(number_of_features, len(axes)):
    axes[index].axis("off")

plt.suptitle(
    "Water Quality Feature Distributions",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

feature_distribution_plot_path = (
    OUTPUT_DIR / "water_quality_feature_distributions.png"
)

plt.savefig(
    feature_distribution_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Feature distributions plot saved to:",
    feature_distribution_plot_path
)

# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION HEATMAP")
print("=" * 70)

correlation_matrix = X_numeric.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)

plt.title("Water Quality Feature Correlation Heatmap")
plt.tight_layout()

correlation_heatmap_path = (
    OUTPUT_DIR / "water_quality_correlation_heatmap.png"
)

plt.savefig(
    correlation_heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Correlation heatmap saved to:",
    correlation_heatmap_path
)

# ============================================================
# 12. TRAINING AND TESTING SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AND TESTING SPLIT")
print("=" * 70)

# The common Water Potability dataset does not normally provide
# separate training and testing files.
# We therefore create a reproducible stratified split.

from sklearn.model_selection import train_test_split

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_numeric,
    y_raw,
    test_size=0.20,
    random_state=42,
    stratify=y_raw
)

print("\nTraining feature shape:")
print(X_train_raw.shape)

print("\nTesting feature shape:")
print(X_test_raw.shape)

print("\nTraining target shape:")
print(y_train_raw.shape)

print("\nTesting target shape:")
print(y_test_raw.shape)

# ============================================================
# 13. TARGET ENCODING
# ============================================================

print("\n" + "=" * 70)
print("TARGET ENCODING")
print("=" * 70)

# Convert target labels to integer values.
# This supports targets such as 0/1 or string labels.
class_names = sorted(y_train_raw.unique())

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

y_train = y_train_raw.map(class_to_index).to_numpy()
y_test = y_test_raw.map(class_to_index).to_numpy()

print("\nClass names:")
print(class_names)

print("\nClass-to-index mapping:")
print(class_to_index)

# ============================================================
# 14. IMPUTATION
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE IMPUTATION")
print("=" * 70)

# Fit the imputer only on training data.
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

print("\nTraining matrix after imputation:")
print(X_train_imputed.shape)

print("\nTesting matrix after imputation:")
print(X_test_imputed.shape)

# ============================================================
# 15. FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

# Fit the scaler only on training data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("Feature scaling completed.")

# ============================================================
# 16. BASELINE MLP CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MLP CLASSIFIER")
print("=" * 70)

baseline_mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=False
)

training_start_time = time.perf_counter()

baseline_mlp.fit(
    X_train_scaled,
    y_train
)

training_end_time = time.perf_counter()

training_time = (
    training_end_time - training_start_time
)

prediction_start_time = time.perf_counter()

y_pred = baseline_mlp.predict(X_test_scaled)

prediction_end_time = time.perf_counter()

prediction_time = (
    prediction_end_time - prediction_start_time
)

# ============================================================
# 17. BASELINE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE PERFORMANCE")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy:             {accuracy:.4f}")
print(f"Macro Precision:      {macro_precision:.4f}")
print(f"Macro Recall:         {macro_recall:.4f}")
print(f"Macro F1:             {macro_f1:.4f}")
print(f"Weighted F1:          {weighted_f1:.4f}")
print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
print(f"Training Time:        {training_time:.4f} seconds")
print(f"Prediction Time:      {prediction_time:.6f} seconds")
print(f"Iterations:           {baseline_mlp.n_iter_}")

baseline_results = pd.DataFrame([{
    "Model": "Baseline MLP",
    "Accuracy": accuracy,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1": macro_f1,
    "Weighted F1": weighted_f1,
    "Balanced Accuracy": balanced_accuracy,
    "Training Time Seconds": training_time,
    "Prediction Time Seconds": prediction_time,
    "Iterations": baseline_mlp.n_iter_
}])

baseline_results_path = (
    OUTPUT_DIR / "water_quality_baseline_results.csv"
)

baseline_results.to_csv(
    baseline_results_path,
    index=False
)

print(
    "\nBaseline results saved to:",
    baseline_results_path
)

# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

classification_report_text = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=[str(name) for name in class_names],
    zero_division=0
)

print("\nActual classification report:")
print(classification_report_text)

classification_report_path = (
    OUTPUT_DIR / "water_quality_classification_report.txt"
)

with open(
    classification_report_path,
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write(classification_report_text)

print(
    "Classification report saved to:",
    classification_report_path
)

# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

confusion_matrix_values = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nActual confusion matrix values:")
print(confusion_matrix_values)

confusion_matrix_dataframe = pd.DataFrame(
    confusion_matrix_values,
    index=class_names,
    columns=class_names
)

print("\nConfusion matrix with class labels:")
print(confusion_matrix_dataframe)

confusion_matrix_csv_path = (
    OUTPUT_DIR / "water_quality_confusion_matrix.csv"
)

confusion_matrix_dataframe.to_csv(
    confusion_matrix_csv_path
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    confusion_matrix_values,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Water Quality Baseline MLP Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

confusion_matrix_plot_path = (
    OUTPUT_DIR / "water_quality_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix CSV saved to:",
    confusion_matrix_csv_path
)

print(
    "Confusion matrix image saved to:",
    confusion_matrix_plot_path
)

# ============================================================
# 20. TRAINING LOSS CURVE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOSS CURVE")
print("=" * 70)

loss_values = baseline_mlp.loss_curve_

print("\nNumber of loss values:")
print(len(loss_values))

print("\nTraining loss values:")
print(loss_values)

loss_dataframe = pd.DataFrame({
    "Iteration": range(1, len(loss_values) + 1),
    "Loss": loss_values
})

loss_csv_path = (
    OUTPUT_DIR / "water_quality_training_loss.csv"
)

loss_dataframe.to_csv(
    loss_csv_path,
    index=False
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(loss_values) + 1),
    loss_values,
    marker="o",
    markersize=3
)

plt.title("Water Quality Baseline MLP Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()

loss_plot_path = (
    OUTPUT_DIR / "water_quality_training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTraining loss CSV saved to:", loss_csv_path)
print("Training loss image saved to:", loss_plot_path)

# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAVING TEST PREDICTIONS")
print("=" * 70)

prediction_results = pd.DataFrame({
    "Actual Class": [
        class_names[value]
        for value in y_test
    ],
    "Predicted Class": [
        class_names[value]
        for value in y_pred
    ],
    "Actual Encoded": y_test,
    "Predicted Encoded": y_pred
})

predictions_path = (
    OUTPUT_DIR / "water_quality_test_predictions.csv"
)

prediction_results.to_csv(
    predictions_path,
    index=False
)

print(
    "Test predictions saved to:",
    predictions_path
)

# ============================================================
# 22. VERIFY ALL OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED RESULTS")
print("=" * 70)

expected_files = [
    "water_quality_descriptive_statistics.csv",
    "water_quality_class_distribution.csv",
    "water_quality_class_distribution.png",
    "water_quality_feature_distributions.png",
    "water_quality_correlation_heatmap.png",
    "water_quality_baseline_results.csv",
    "water_quality_classification_report.txt",
    "water_quality_confusion_matrix.csv",
    "water_quality_confusion_matrix.png",
    "water_quality_training_loss.csv",
    "water_quality_training_loss.png",
    "water_quality_test_predictions.csv"
]

created_count = 0
missing_count = 0

for filename in expected_files:
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        file_size = file_path.stat().st_size
        print(
            f"[CREATED] {filename} "
            f"({file_size} bytes)"
        )
        created_count += 1
    else:
        print(f"[MISSING] {filename}")
        missing_count += 1

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nTotal files expected:", len(expected_files))
print("Files created:", created_count)
print("Files missing:", missing_count)

# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

if missing_count == 0:
    print("ALL WATER QUALITY RESULTS WERE CREATED SUCCESSFULLY")
else:
    print("WARNING: SOME WATER QUALITY RESULT FILES ARE MISSING")

print("=" * 70)