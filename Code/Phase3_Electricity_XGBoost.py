# ============================================================
# PHASE 3 - ELECTRICITY DATASET
# XGBOOST BASELINE CLASSIFIER
# ============================================================
#
# PURPOSE:
# This phase tests whether a nonlinear tree-based model
# (XGBoost) can capture the relationships in the Electricity
# dataset better than the MLP baseline.
#
# IMPORTANT:
# - NO SMOTE
# - NO SMOTENC
# - NO class weighting
# - NO hyperparameter optimization
#
# This is intentionally a BASELINE XGBoost experiment.
#
# Later phases can use:
# Random Search
# Bayesian Optimization
# Differential Evolution
#
# to optimize the XGBoost hyperparameters.
#
# ============================================================


import time
import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix 
)

from xgboost import XGBClassifier


# ============================================================
# WARNING CONTROL
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "processed_datasets/electricity_processed.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

RESULTS_DIR = "results"

VISUALIZATION_DIR = os.path.join(
    RESULTS_DIR,
    "phase3_electricity_xgboost_visualizations"
)

os.makedirs(VISUALIZATION_DIR, exist_ok=True)
# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Electricity Dataset...")
print("=" * 70)

data = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(data.shape)

print("\nFirst Five Rows:")
print(data.head())


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["Target"])
y = data["Target"]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nOriginal Target Distribution:")
print(y.value_counts())


print("\nOriginal Target Percentages:")
print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
)
# ============================================================
# VISUALIZATION 1 - TARGET CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y,
    order=["DOWN", "UP"]
)

plt.title("Electricity Target Class Distribution")
plt.xlabel("Electricity Class")
plt.ylabel("Number of Samples")

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


# ============================================================
# TARGET ENCODING
# ============================================================
#
# XGBoost expects numerical target labels.
#
# DOWN = 0
# UP   = 1
#
# ============================================================

target_mapping = {
    "DOWN": 0,
    "UP": 1
}

y_encoded = y.map(target_mapping)


print("\nEncoded Target Distribution:")
print(y_encoded.value_counts())

print("\nEncoded Target Classes:")
print("0: DOWN")
print("1: UP")


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCategorical Features:")
for feature in categorical_features:
    print("-", feature)


print("\nNumerical Features:")
for feature in numerical_features:
    print("-", feature)

# ============================================================
# VISUALIZATION 4 - FEATURE DISTRIBUTIONS BY TARGET
# ============================================================

important_features = [
    "nswprice",
    "nswdemand",
    "vicdemand",
    "period"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

for ax, feature in zip(axes.flatten(), important_features):

    sns.boxplot(
        data=data,
        x="Target",
        y=feature,
        order=["DOWN", "UP"],
        ax=ax
    )

    ax.set_title(f"{feature} by Electricity Class")
    ax.set_xlabel("Electricity Class")
    ax.set_ylabel(feature)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "04_feature_distributions_by_target.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
#
# Stratification preserves the DOWN / UP distribution
# between training and testing datasets.
#
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)


print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# ============================================================
# ONE-HOT ENCODING
# ============================================================
#
# The Electricity dataset contains one categorical feature:
#
# day
#
# XGBoost can work with categorical variables in some
# configurations, but one-hot encoding provides a simple,
# consistent representation for this experiment.
#
# ============================================================

if len(categorical_features) > 0:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = encoder.fit_transform(
        X_train[categorical_features]
    )

    X_test_cat = encoder.transform(
        X_test[categorical_features]
    )

else:

    X_train_cat = np.empty(
        (len(X_train), 0)
    )

    X_test_cat = np.empty(
        (len(X_test), 0)
    )


# ============================================================
# NUMERICAL FEATURES
# ============================================================

X_train_num = X_train[
    numerical_features
].to_numpy()

X_test_num = X_test[
    numerical_features
].to_numpy()


# ============================================================
# COMBINE NUMERICAL + ENCODED CATEGORICAL FEATURES
# ============================================================

X_train_processed = np.hstack(
    [
        X_train_num,
        X_train_cat
    ]
)

X_test_processed = np.hstack(
    [
        X_test_num,
        X_test_cat
    ]
)


# ============================================================
# FEATURE NAMES
# ============================================================

encoded_feature_names = []

encoded_feature_names.extend(
    numerical_features
)

if len(categorical_features) > 0:

    encoded_feature_names.extend(
        encoder.get_feature_names_out(
            categorical_features
        )
    )


print("\nFeature Matrix After Encoding:")
print(
    "Training:",
    X_train_processed.shape
)

print(
    "Testing :",
    X_test_processed.shape
)


# ============================================================
# IMPORTANT:
# NO STANDARDIZATION
# ============================================================
#
# Unlike MLP/SVM models, tree-based XGBoost does not require
# feature standardization.
#
# Tree splits are based on feature thresholds, so scaling
# the numerical variables is unnecessary.
#
# ============================================================

print("\nFeature Standardization:")
print("Not required for XGBoost.")


# ============================================================
# PHASE 3 - XGBOOST BASELINE
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 - XGBOOST BASELINE")
print("=" * 70)


print("\nTraining Data Distribution:")
print(y_train.value_counts())


print("\nSMOTE / SMOTENC:")
print("Not applied.")


print("\nClass Weighting:")
print("Not applied.")


# ============================================================
# BUILD XGBOOST MODEL
# ============================================================
#
# These are deliberately fixed baseline parameters.
#
# We are NOT optimizing them yet.
#
# Later optimization experiments can search these parameters.
#
# ============================================================

print("\nBuilding XGBoost Baseline Model...")


model = XGBClassifier(

    n_estimators=100,

    max_depth=6,

    learning_rate=0.1,

    subsample=1.0,

    colsample_bytree=1.0,

    min_child_weight=1,

    gamma=0,

    reg_alpha=0,

    reg_lambda=1,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining XGBoost Baseline...")

training_start = time.time()

model.fit(
    X_train_processed,
    y_train
)

training_time = time.time() - training_start


print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nMaking XGBoost Predictions...")

y_pred = model.predict(
    X_test_processed
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

weighted_precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 XGBOOST BASELINE RESULTS")
print("=" * 70)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Weighted Precision : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall    : {weighted_recall:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "DOWN",
            "UP"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix\n")

print(cm)
# ============================================================
# VISUALIZATION 3 - CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["DOWN", "UP"],
    yticklabels=["DOWN", "UP"]
)

plt.title("XGBoost Baseline Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "03_xgboost_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)


feature_importance = pd.DataFrame({

    "Feature":
        encoded_feature_names,

    "Importance":
        model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop Features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)
# ============================================================
# VISUALIZATION 2 - XGBOOST FEATURE IMPORTANCE
# ============================================================

top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "02_xgboost_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# BASELINE HYPERPARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 XGBOOST BASELINE HYPERPARAMETERS")
print("=" * 70)

print(
    "n_estimators      : 100"
)

print(
    "max_depth         : 6"
)

print(
    "learning_rate     : 0.1"
)

print(
    "subsample         : 1.0"
)

print(
    "colsample_bytree  : 1.0"
)

print(
    "min_child_weight  : 1"
)

print(
    "gamma             : 0"
)

print(
    "reg_alpha         : 0"
)

print(
    "reg_lambda        : 1"
)

print(
    "random_state      : 42"
)


print(
    f"\nTraining Time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 INTERPRETATION")
print("=" * 70)

print(
    """
This experiment evaluates whether a nonlinear
tree-based model can capture the relationships
in the Electricity dataset more effectively
than the MLP baseline.

No SMOTE or SMOTENC was applied.

No class weighting was applied.

No hyperparameter optimization was applied.

The purpose is to establish a clean XGBoost
baseline before introducing optimization algorithms.
"""
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 XGBOOST SUMMARY")
print("=" * 70)

print(
    f"Accuracy        : {accuracy:.4f}"
)

print(
    f"Weighted F1     : {weighted_f1:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print(
    f"Training Time   : {training_time:.2f} seconds"
)

print("\n")
print("=" * 70)
print("PHASE 3 XGBOOST BASELINE COMPLETED")
print("=" * 70)

