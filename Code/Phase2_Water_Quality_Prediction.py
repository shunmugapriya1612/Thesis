# ================================================================
# PHASE 2 - WATER QUALITY PREDICTION
#
# Baseline Refinement + Class Imbalance + Feature Relationship
# Analysis
#
# Dataset:
# Water Potability
#
# ================================================================
#
# PHASE 1 -> PHASE 2 -> PHASE 3
# ================================================================
#
# PHASE 1:
#   Establish baseline MLP and understand the dataset.
#
# PHASE 2:
#   Diagnose the main limitations of the baseline and determine
#   which preprocessing/model choices should be fixed before
#   optimization.
#
# PHASE 3:
#   Apply Random Search, Bayesian Optimization, Hyperband,
#   LEA, SOCIAL, LEA + Bayesian Optimization, and
#   SOCIAL + Bayesian Optimization.
#
# ================================================================


# ================================================================
# PHASE 1 FINDINGS
# ================================================================
#
# Dataset:
#
#   Total samples       = 3276
#   Features            = 9
#   Classes             = 2
#
# Class distribution:
#
#   Class 0 = 1998 (60.99%)
#   Class 1 = 1278 (39.01%)
#
# Missing values:
#
#   ph                  = 491  (14.99%)
#   Sulfate             = 781  (23.84%)
#   Trihalomethanes     = 162  (4.95%)
#
#   Total missing       = 1434
#
# Phase 1 baseline:
#
#   hidden_layer_sizes  = (64, 32)
#   activation          = relu
#   solver              = adam
#   alpha               = 0.0001
#   learning_rate_init  = 0.001
#   max_iter            = 500
#   random_state        = 42
#
# Phase 1 test result:
#
#   Accuracy            = 0.5899
#   Macro F1            = 0.5600
#   Balanced Accuracy   = 0.5597
#   Potable Recall      = 0.4219
#
# The confusion matrix showed that the model was weaker at
# identifying potable water (Class 1).
#
#
# ================================================================
# PHASE 2A - CONVERGENCE ANALYSIS
# ================================================================
#
# Question:
#
#   "Is the low performance caused by insufficient training?"
#
# Tested:
#
#   max_iter = 500
#   max_iter = 1000
#   max_iter = 1500
#
# Results:
#
#   max_iter     Validation Macro F1     Test Macro F1
#   --------------------------------------------------
#      500              0.6127              0.5600
#     1000              0.6057              0.5552
#     1500              0.6083              0.5417
#
# Although training loss continued to decrease as the number
# of iterations increased, validation and test generalization
# did not improve.
#
# The model at 1500 maximum iterations actually stopped at
# 1094 iterations because the optimizer converged, but its
# validation/test performance was still worse than the 500
# iteration configuration.
#
# DECISION:
#
#   Keep max_iter = 500
#
# Reason:
#
#   Highest validation Macro F1.
#
# Conclusion:
#
#   Simply training longer is NOT the main solution.
#
#
# ================================================================
# PHASE 2B - MISSING-VALUE / IMPUTATION ANALYSIS
# ================================================================
#
# Question:
#
#   "Is median imputation causing the low performance?"
#
# Tested:
#
#   1. Median imputation
#   2. KNN imputation
#   3. K-Means-based cluster imputation
#
# Results:
#
#   Method       Validation Macro F1     Test Macro F1
#   ---------------------------------------------------
#   Median              0.6127              0.5600
#   KNN                 0.5841              0.5604
#   K-Means             0.6093              0.5515
#
# DECISION:
#
#   Keep median imputation.
#
# Reason:
#
#   Median produced the highest validation Macro F1.
#
# KNN did not improve validation performance.
#
# K-Means-based imputation was close to median but did not
# produce an improvement and had lower test performance.
#
# Conclusion:
#
#   More complicated imputation is NOT justified by the
#   current evidence.
#
#
# ================================================================
# FIXED CONFIGURATION AFTER PHASE 2A + 2B
# ================================================================
#
# Imputation:
#   Median
#
# Scaling:
#   StandardScaler
#
# MLP:
#   hidden_layer_sizes = (64, 32)
#   activation         = relu
#   solver             = adam
#   alpha              = 0.0001
#   learning_rate_init = 0.001
#   max_iter           = 500
#   random_state       = 42
#
#
# ================================================================
# PHASE 2C - CLASS IMBALANCE ANALYSIS
# ================================================================
#
# Why are we testing class balancing?
#
# Training data:
#
#   Class 0 = 1278
#   Class 1 = 818
#
# The imbalance is MODERATE rather than extreme.
#
# However, Class 1 (Potable) had considerably weaker recall
# than Class 0.
#
# Therefore we asked:
#
#   "Is the model struggling because it sees fewer
#    potable-water training examples?"
#
#
# ---------------------------------------------------------------
# RANDOM OVERSAMPLING
# ---------------------------------------------------------------
#
# Random Oversampling duplicates existing minority-class
# observations until the classes have equal representation.
#
# IMPORTANT:
#
# Random Oversampling does NOT create synthetic observations.
#
# Example:
#
#   Before:
#       Class 0 = 1278
#       Class 1 = 818
#
#   After:
#       Class 0 = 1278
#       Class 1 = 1278
#
# We deliberately tested this simple approach before considering
# synthetic-data methods such as SMOTE.
#
#
# PHASE 2C RESULTS:
#
#   Metric                    Baseline     Oversampling
#   ---------------------------------------------------
#   Validation Accuracy       0.6298          0.6393
#   Validation Macro F1       0.6127          0.6168
#   Validation Balanced Acc   0.6134          0.6158
#   Validation Potable Recall 0.5392          0.5098
#
#   Test Accuracy             0.5899          0.5716
#   Test Macro F1             0.5600          0.5412
#   Test Balanced Accuracy    0.5597          0.5412
#   Test Potable Recall       0.4219          0.4023
#
# Training loss:
#
#   Baseline          = 0.126606
#   Oversampling      = 0.086164
#
# Interpretation:
#
#   Oversampling balanced the TRAINING CLASS COUNTS successfully.
#
#   However, it did NOT solve the predictive problem.
#
#   The validation improvement in Macro F1 was only:
#
#       +0.0041
#
#   while the important potable-class recall decreased.
#
#   More importantly, all principal TEST metrics became worse.
#
# DECISION:
#
#   Do NOT retain Random Oversampling as the fixed Phase 2
#   configuration.
#
# Therefore:
#
#   Median imputation + max_iter=500 + original class
#   distribution remains the current fixed baseline.
#
# We should NOT immediately move to SMOTE because the current
# evidence does not show that simply increasing the number of
# Class-1 observations solves the underlying problem.
#
#
# ================================================================
# PHASE 2D - FEATURE RELATIONSHIP AND CLASS SEPARABILITY
# ================================================================
#
# We now investigate the remaining important question:
#
#   "What information is actually present in the nine features,
#    and how clearly can the two classes be separated?"
#
# This is important before Phase 3 optimization.
#
# The Phase 1 Pearson correlations with Potability were extremely
# small:
#
#   Organic_carbon      -0.0300
#   Sulfate             -0.0236
#   Hardness            -0.0138
#   Conductivity        -0.0081
#   ph                  -0.0036
#   Turbidity            +0.0016
#   Trihalomethanes     +0.0071
#   Chloramines         +0.0238
#   Solids              +0.0337
#
# This means there are no strong SIMPLE LINEAR relationships
# between an individual feature and Potability.
#
# However:
#
#   Pearson correlation ≈ 0
#
# does NOT necessarily mean:
#
#   feature contains no useful information.
#
# There may still be:
#
#   - nonlinear relationships
#   - monotonic but nonlinear relationships
#   - thresholds
#   - interactions between features
#   - class overlap
#
#
# Therefore Phase 2D investigates:
#
#   1. Spearman correlation
#      -> monotonic relationships
#
#   2. Mutual information
#      -> possible nonlinear dependence
#
#   3. Class-wise feature distributions
#      -> whether feature values of the two classes overlap
#
#   4. PCA visualization
#      -> whether the classes form separable regions in the
#         multivariate feature space
#
#
# ================================================================
# HOW THIS HELPS PHASE 3 OPTIMIZATION
# ================================================================
#
# We are NOT optimizing the model during Phase 2D.
#
# We are learning what kind of optimization problem we have.
#
# Example 1:
#
#   Strong nonlinear information
#           +
#   reasonable class separation
#
# could indicate that the MLP contains useful signal but its
# architecture/hyperparameters may not be extracting it well.
#
# This would justify searching:
#
#   hidden-layer architecture
#   activation
#   learning rate
#   regularization
#   batch size
#
#
# Example 2:
#
#   Useful nonlinear information
#           +
#   heavy class overlap
#
# would suggest that optimization may help only to a limited
# extent because the feature space itself is difficult.
#
#
# Example 3:
#
#   Very little information
#           +
#   strong class overlap
#
# would indicate that optimization alone probably has limited
# headroom and feature engineering/representation may become
# more important.
#
# Therefore the feature-relationship analysis helps us design
# a sensible Phase 3 search space instead of blindly trying
# hyperparameters.
#
#
# IMPORTANT:
#
# Phase 2D uses TRAINING DATA for relationship analysis.
#
# The final TEST labels are not used to design the optimization
# strategy. This prevents test-set leakage into Phase 3.
#
# ================================================================


# ================================================================
# IMPORTS
# ================================================================

from pathlib import Path
import time
import random
import warnings

import numpy as np
import pandas as pd

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
    confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler


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

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "water_potability.csv"
)


# ================================================================
# 3. FIXED CONFIGURATION
# ================================================================

TARGET_COLUMN = "Potability"

# From Phase 2A:
MAX_ITER = 500

# From Phase 2B:
IMPUTATION_STRATEGY = "median"

# Fixed MLP configuration from Phase 1:
HIDDEN_LAYERS = (64, 32)
ACTIVATION = "relu"
SOLVER = "adam"
ALPHA = 0.0001
LEARNING_RATE_INIT = 0.001
BATCH_SIZE = "auto"


# ================================================================
# 4. HEADER
# ================================================================

print("\n" + "=" * 70)
print("PHASE 2 - WATER QUALITY PREDICTION")
print("BASELINE REFINEMENT + FEATURE RELATIONSHIP ANALYSIS")
print("=" * 70)


# ================================================================
# 5. LOAD DATASET
# ================================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()


print("\nDataset shape:")
print(df.shape)


# ================================================================
# 6. FEATURES AND TARGET
# ================================================================

X = df.drop(
    columns=[TARGET_COLUMN]
).copy()

y = df[TARGET_COLUMN].astype(int).copy()


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ================================================================
# 7. TARGET DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

class_counts = (
    y.value_counts()
    .sort_index()
)

print("\nClass counts:")
print(class_counts)

print("\nClass percentages:")

for class_value, count in class_counts.items():

    percentage = (
        count / len(y) * 100
    )

    print(
        f"Class {class_value}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ================================================================
# 8. SAME TRAIN / VALIDATION / TEST SPLIT
# ================================================================
#
# 64% Training
# 16% Validation
# 20% Test
#
# Exactly the same split used throughout Phase 1 and Phase 2.
# ================================================================

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


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"\nTraining samples:   {len(X_train)}"
)

print(
    f"Validation samples: {len(X_val)}"
)

print(
    f"Test samples:       {len(X_test)}"
)


# ================================================================
# 9. MEDIAN IMPUTATION
# ================================================================
#
# FIXED FROM PHASE 2B.
#
# The imputer is fitted ONLY on training data.
# ================================================================

print("\n" + "=" * 70)
print("MEDIAN IMPUTATION")
print("=" * 70)


imputer = SimpleImputer(
    strategy=IMPUTATION_STRATEGY
)


X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)


X_val_imputed = pd.DataFrame(
    imputer.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)


X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


print(
    "\nTraining missing values after imputation:",
    X_train_imputed.isna().sum().sum()
)

print(
    "Validation missing values after imputation:",
    X_val_imputed.isna().sum().sum()
)

print(
    "Test missing values after imputation:",
    X_test_imputed.isna().sum().sum()
)


# ================================================================
# 10. STANDARD SCALING
# ================================================================
#
# Scaler is fitted ONLY on the training data.
# ================================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_imputed
)


X_val_scaled = scaler.transform(
    X_val_imputed
)


X_test_scaled = scaler.transform(
    X_test_imputed
)


# ================================================================
# 11. METRIC FUNCTION
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            )
    }


# ================================================================
# 12. TRAIN FIXED BASELINE
# ================================================================
#
# This is now our fixed Phase 2 baseline:
#
#   Median imputation
#   StandardScaler
#   MLP (64,32)
#   max_iter = 500
#
# ================================================================

print("\n" + "=" * 70)
print("FIXED BASELINE")
print("=" * 70)


baseline_model = MLPClassifier(

    hidden_layer_sizes=HIDDEN_LAYERS,

    activation=ACTIVATION,

    solver=SOLVER,

    alpha=ALPHA,

    learning_rate_init=LEARNING_RATE_INIT,

    batch_size=BATCH_SIZE,

    max_iter=MAX_ITER,

    random_state=RANDOM_STATE
)


baseline_start = time.perf_counter()


with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning
    )

    baseline_model.fit(
        X_train_scaled,
        y_train
    )


baseline_training_time = (
    time.perf_counter()
    - baseline_start
)


baseline_val_pred = (
    baseline_model.predict(
        X_val_scaled
    )
)


baseline_test_pred = (
    baseline_model.predict(
        X_test_scaled
    )
)


baseline_val_metrics = calculate_metrics(
    y_val,
    baseline_val_pred
)


baseline_test_metrics = calculate_metrics(
    y_test,
    baseline_test_pred
)


baseline_val_potable_recall = recall_score(
    y_val,
    baseline_val_pred,
    pos_label=1,
    zero_division=0
)


baseline_test_potable_recall = recall_score(
    y_test,
    baseline_test_pred,
    pos_label=1,
    zero_division=0
)


print("\nValidation:")
print(
    f"Accuracy:           "
    f"{baseline_val_metrics['accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{baseline_val_metrics['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{baseline_val_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Potable Recall:     "
    f"{baseline_val_potable_recall:.4f}"
)


print("\nTest:")
print(
    f"Accuracy:           "
    f"{baseline_test_metrics['accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{baseline_test_metrics['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{baseline_test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Potable Recall:     "
    f"{baseline_test_potable_recall:.4f}"
)


print(
    f"\nTraining time: "
    f"{baseline_training_time:.4f} sec"
)


# ================================================================
# 13. PHASE 2C - RANDOM OVERSAMPLING
# ================================================================
#
# We are NOT retaining oversampling automatically.
#
# We reproduce the controlled experiment here so that the final
# Phase 2 script contains the complete reasoning chain.
#
# Only TRAINING DATA is oversampled.
# Validation and test remain untouched.
# ================================================================

print("\n" + "=" * 70)
print("PHASE 2C - RANDOM OVERSAMPLING")
print("=" * 70)


print("\nTraining distribution BEFORE oversampling:")

print(
    y_train.value_counts()
    .sort_index()
)


oversampler = RandomOverSampler(
    sampling_strategy="auto",
    random_state=RANDOM_STATE
)


X_train_over, y_train_over = oversampler.fit_resample(
    X_train_scaled,
    y_train
)


print("\nTraining distribution AFTER oversampling:")

print(
    pd.Series(y_train_over)
    .value_counts()
    .sort_index()
)


# ================================================================
# 14. TRAIN OVERSAMPLED MODEL
# ================================================================

oversampled_model = MLPClassifier(

    hidden_layer_sizes=HIDDEN_LAYERS,

    activation=ACTIVATION,

    solver=SOLVER,

    alpha=ALPHA,

    learning_rate_init=LEARNING_RATE_INIT,

    batch_size=BATCH_SIZE,

    max_iter=MAX_ITER,

    random_state=RANDOM_STATE
)


oversampled_start = time.perf_counter()


with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning
    )

    oversampled_model.fit(
        X_train_over, y_train_over
    )


oversampled_training_time = (
    time.perf_counter()
    - oversampled_start
)


oversampled_val_pred = (
    oversampled_model.predict(
        X_val_scaled
    )
)


oversampled_test_pred = (
    oversampled_model.predict(
        X_test_scaled
    )
)


oversampled_val_metrics = calculate_metrics(
    y_val,
    oversampled_val_pred
)


oversampled_test_metrics = calculate_metrics(
    y_test,
    oversampled_test_pred
)


oversampled_val_potable_recall = recall_score(
    y_val,
    oversampled_val_pred,
    pos_label=1,
    zero_division=0
)


oversampled_test_potable_recall = recall_score(
    y_test,
    oversampled_test_pred,
    pos_label=1,
    zero_division=0
)


print("\nValidation after oversampling:")

print(
    f"Accuracy:           "
    f"{oversampled_val_metrics['accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{oversampled_val_metrics['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{oversampled_val_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Potable Recall:     "
    f"{oversampled_val_potable_recall:.4f}"
)


print("\nTest after oversampling:")

print(
    f"Accuracy:           "
    f"{oversampled_test_metrics['accuracy']:.4f}"
)

print(
    f"Macro F1:           "
    f"{oversampled_test_metrics['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{oversampled_test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Potable Recall:     "
    f"{oversampled_test_potable_recall:.4f}"
)


# ================================================================
# 15. CLASS-BALANCING COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("PHASE 2C COMPARISON")
print("=" * 70)


balance_results = pd.DataFrame({

    "Experiment": [
        "Fixed Baseline",
        "Random Oversampling"
    ],

    "Validation Accuracy": [
        baseline_val_metrics["accuracy"],
        oversampled_val_metrics["accuracy"]
    ],

    "Validation Macro F1": [
        baseline_val_metrics["macro_f1"],
        oversampled_val_metrics["macro_f1"]
    ],

    "Validation Balanced Accuracy": [
        baseline_val_metrics["balanced_accuracy"],
        oversampled_val_metrics["balanced_accuracy"]
    ],

    "Validation Potable Recall": [
        baseline_val_potable_recall,
        oversampled_val_potable_recall
    ],

    "Test Accuracy": [
        baseline_test_metrics["accuracy"],
        oversampled_test_metrics["accuracy"]
    ],

    "Test Macro F1": [
        baseline_test_metrics["macro_f1"],
        oversampled_test_metrics["macro_f1"]
    ],

    "Test Balanced Accuracy": [
        baseline_test_metrics["balanced_accuracy"],
        oversampled_test_metrics["balanced_accuracy"]
    ],

    "Test Potable Recall": [
        baseline_test_potable_recall,
        oversampled_test_potable_recall
    ]
})


print(
    balance_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# 16. OVERSAMPLING INTERPRETATION
# ================================================================

validation_f1_change = (
    oversampled_val_metrics["macro_f1"]
    - baseline_val_metrics["macro_f1"]
)


validation_recall_change = (
    oversampled_val_potable_recall
    - baseline_val_potable_recall
)


test_f1_change = (
    oversampled_test_metrics["macro_f1"]
    - baseline_test_metrics["macro_f1"]
)


test_recall_change = (
    oversampled_test_potable_recall
    - baseline_test_potable_recall
)


print("\n" + "=" * 70)
print("PHASE 2C INTERPRETATION")
print("=" * 70)


print(
    f"\nValidation Macro F1 change: "
    f"{validation_f1_change:+.4f}"
)


print(
    f"Validation Potable Recall change: "
    f"{validation_recall_change:+.4f}"
)


print(
    f"Test Macro F1 change: "
    f"{test_f1_change:+.4f}"
)


print(
    f"Test Potable Recall change: "
    f"{test_recall_change:+.4f}"
)


print(
    "\nConclusion:"
)


print(
    "Random Oversampling balances the training class counts, "
    "but it does not improve the final generalization "
    "performance for this dataset."
)


print(
    "\nTherefore the fixed Phase 2 baseline remains:"
)


print(
    "Median imputation + StandardScaler + MLP (64,32) "
    "+ max_iter=500 + original class distribution."
)


# ================================================================
# 17. PHASE 2C VALIDATION VISUALIZATION
# ================================================================

x = np.arange(2)

width = 0.22


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - width,
    balance_results[
        "Validation Accuracy"
    ],
    width,
    label="Accuracy"
)


plt.bar(
    x,
    balance_results[
        "Validation Macro F1"
    ],
    width,
    label="Macro F1"
)


plt.bar(
    x + width,
    balance_results[
        "Validation Balanced Accuracy"
    ],
    width,
    label="Balanced Accuracy"
)


plt.xticks(
    x,
    balance_results[
        "Experiment"
    ]
)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Score"
)


plt.xlabel(
    "Experiment"
)


plt.title(
    "Phase 2C - Validation Performance"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ================================================================
# 18. PHASE 2C TEST VISUALIZATION
# ================================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - width,
    balance_results[
        "Test Accuracy"
    ],
    width,
    label="Accuracy"
)


plt.bar(
    x,
    balance_results[
        "Test Macro F1"
    ],
    width,
    label="Macro F1"
)


plt.bar(
    x + width,
    balance_results[
        "Test Balanced Accuracy"
    ],
    width,
    label="Balanced Accuracy"
)


plt.xticks(
    x,
    balance_results[
        "Experiment"
    ]
)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Score"
)


plt.xlabel(
    "Experiment"
)


plt.title(
    "Phase 2C - Test Performance"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ================================================================
# 19. PHASE 2D - FEATURE RELATIONSHIP ANALYSIS
# ================================================================
#
# From Phase 1, Pearson feature-target correlations were very
# close to zero.
#
# We now investigate:
#
#   A. Spearman correlation
#   B. Mutual information
#   C. Class-wise feature distributions
#   D. PCA class separability
#
# Only TRAINING data is used for this diagnostic analysis.
#
# ================================================================

print("\n" + "=" * 70)
print("PHASE 2D - FEATURE RELATIONSHIP ANALYSIS")
print("=" * 70)


# ================================================================
# 20. SPEARMAN CORRELATION
# ================================================================
#
# Pearson:
#   primarily measures linear association.
#
# Spearman:
#   measures monotonic association based on ranks.
#
# A feature can have:
#
#   Pearson ≈ 0
#   Spearman != 0
#
# which would suggest a non-linear but monotonic relationship.
# ================================================================

spearman_correlations = (
    X_train_imputed
    .corrwith(
        y_train,
        method="spearman"
    )
    .sort_values()
)


print("\n" + "=" * 70)
print("SPEARMAN CORRELATION WITH POTABILITY")
print("=" * 70)


print(
    spearman_correlations.round(4)
)


plt.figure(
    figsize=(9, 5)
)


spearman_correlations.plot(
    kind="barh"
)


plt.xlabel(
    "Spearman Correlation"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Spearman Feature-Target Relationships"
)


plt.grid(
    axis="x",
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ================================================================
# 21. MUTUAL INFORMATION
# ================================================================
#
# Mutual Information can identify statistical dependence that
# is not necessarily linear or monotonic.
#
# Higher MI:
#
#   potentially more information about the target.
#
# Lower MI:
#
#   weaker detectable dependence.
#
# MI does NOT tell us causality and should not be interpreted
# as a direct model feature-importance measure.
# ================================================================

mi_values = mutual_info_classif(

    X_train_imputed,

    y_train,

    random_state=RANDOM_STATE
)


mi_results = pd.Series(

    mi_values,

    index=X_train_imputed.columns

).sort_values(
    ascending=False
)


print("\n" + "=" * 70)
print("MUTUAL INFORMATION WITH POTABILITY")
print("=" * 70)


print(
    mi_results.round(4)
)


plt.figure(
    figsize=(9, 5)
)


mi_results.sort_values().plot(
    kind="barh"
)


plt.xlabel(
    "Mutual Information"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Feature Mutual Information with Potability"
)


plt.grid(
    axis="x",
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ================================================================
# 22. CLASS-WISE FEATURE DISTRIBUTIONS
# ================================================================
#
# We now look at each feature separately for:
#
#   Class 0 = Non-Potable
#   Class 1 = Potable
#
# The objective is to determine how much the two classes overlap.
#
# Heavy overlap:
#
#   difficult classification boundary.
#
# Clear separation:
#
#   useful signal potentially available for the classifier.
# ================================================================

plot_df = X_train_imputed.copy()

plot_df["Potability"] = y_train.values


fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 13)
)


axes = axes.flatten()


for index, feature in enumerate(
    X_train_imputed.columns
):

    sns.boxplot(

        data=plot_df,

        x="Potability",

        y=feature,

        hue="Potability",

        legend=False,

        ax=axes[index]
    )


    axes[index].set_title(
        feature
    )


    axes[index].set_xlabel(
        "Potability Class"
    )


    axes[index].set_ylabel(
        "Value"
    )


plt.suptitle(
    "Feature Distributions by Potability Class",
    fontsize=16
)


plt.tight_layout()


plt.show()


# ================================================================
# 23. PCA CLASS-SEPARABILITY ANALYSIS
# ================================================================
#
# PCA is used ONLY for visualization here.
#
# It is NOT replacing the original features.
#
# The goal:
#
#   Project 9-dimensional training data into 2 dimensions
#   and visually inspect whether class 0 and class 1 occupy
#   different or heavily overlapping regions.
#
# ================================================================

pca = PCA(
    n_components=2
)


X_train_pca = pca.fit_transform(
    X_train_scaled
)


explained_variance = (
    pca.explained_variance_ratio_
)


print("\n" + "=" * 70)
print("PCA CLASS-SEPARABILITY ANALYSIS")
print("=" * 70)


print(
    f"\nPC1 explained variance: "
    f"{explained_variance[0]:.4f}"
)


print(
    f"PC2 explained variance: "
    f"{explained_variance[1]:.4f}"
)


print(
    f"Total explained variance by PC1 + PC2: "
    f"{explained_variance.sum():.4f}"
)


plt.figure(
    figsize=(10, 7)
)


class_0_mask = (
    y_train.to_numpy() == 0
)


class_1_mask = (
    y_train.to_numpy() == 1
)


plt.scatter(

    X_train_pca[
        class_0_mask,
        0
    ],

    X_train_pca[
        class_0_mask,
        1
    ],

    alpha=0.5,

    label="Non-Potable (0)"
)


plt.scatter(

    X_train_pca[
        class_1_mask,
        0
    ],

    X_train_pca[
        class_1_mask,
        1
    ],

    alpha=0.5,

    label="Potable (1)"
)


plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
)


plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
)


plt.title(
    "PCA Visualization of Potability Classes"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ================================================================
# 24. IDENTIFY TOP FEATURES
# ================================================================

top_mi_features = (
    mi_results.head(5)
)


top_spearman_features = (
    spearman_correlations
    .abs()
    .sort_values(
        ascending=False
    )
    .head(5)
)


print("\n" + "=" * 70)
print("TOP FEATURES BY MUTUAL INFORMATION")
print("=" * 70)


print(
    top_mi_features.round(4)
)


print("\n" + "=" * 70)
print("TOP FEATURES BY ABSOLUTE SPEARMAN CORRELATION")
print("=" * 70)


print(
    top_spearman_features.round(4)
)


# ================================================================
# 25. PHASE 2D INTERPRETATION GUIDANCE
# ================================================================
#
# We intentionally do not automatically change the features or
# MLP based on these diagnostic results.
#
# Instead, use the following logic:
#
# ---------------------------------------------------------------
#
# CASE A:
#
#   Mutual information is meaningful
#   AND
#   PCA/classes show reasonable separation
#
# Interpretation:
#
#   The feature space contains useful nonlinear information.
#
# Phase 3 implication:
#
#   Hyperparameter optimization could have meaningful headroom.
#
#   The search space should include:
#
#       - architecture
#       - learning rate
#       - activation
#       - regularization
#       - batch size
#
#
# ---------------------------------------------------------------
#
# CASE B:
#
#   Mutual information is meaningful
#   BUT
#   PCA/classes heavily overlap
#
# Interpretation:
#
#   Useful information exists, but the class boundary is difficult.
#
# Phase 3 implication:
#
#   Optimization may help, but gains may be limited.
#
#   Feature engineering/interactions may become important later.
#
#
# ---------------------------------------------------------------
#
# CASE C:
#
#   Mutual information is generally very low
#   AND
#   PCA shows strong class overlap
#
# Interpretation:
#
#   The available nine features may provide limited predictive
#   information for this target.
#
# Phase 3 implication:
#
#   Very aggressive model-complexity optimization may simply
#   increase overfitting rather than produce strong generalization.
#
#
# ---------------------------------------------------------------
#
# CASE D:
#
#   Some features show substantially higher MI than others.
#
# Interpretation:
#
#   Feature selection or feature-weighting experiments may be
#   useful later.
#
# ================================================================


# ================================================================
# 26. CURRENT PHASE 2 CONFIGURATION
# ================================================================

print("\n" + "=" * 70)
print("CURRENT FIXED PHASE 2 CONFIGURATION")
print("=" * 70)


print(
    "\nImputation:"
)

print(
    "Median"
)


print(
    "\nScaling:"
)

print(
    "StandardScaler"
)


print(
    "\nClass balancing:"
)

print(
    "None"
)


print(
    "\nMLP architecture:"
)

print(
    HIDDEN_LAYERS
)


print(
    "\nActivation:"
)

print(
    ACTIVATION
)


print(
    "\nSolver:"
)

print(
    SOLVER
)


print(
    "\nLearning rate:"
)

print(
    LEARNING_RATE_INIT
)


print(
    "\nAlpha:"
)

print(
    ALPHA
)


print(
    "\nMax iterations:"
)

print(
    MAX_ITER
)


# ================================================================
# 27. CURRENT BASELINE SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("CURRENT BASELINE SUMMARY")
print("=" * 70)


print(
    f"\nValidation Accuracy: "
    f"{baseline_val_metrics['accuracy']:.4f}"
)


print(
    f"Validation Macro F1: "
    f"{baseline_val_metrics['macro_f1']:.4f}"
)


print(
    f"Validation Balanced Accuracy: "
    f"{baseline_val_metrics['balanced_accuracy']:.4f}"
)


print(
    f"Validation Potable Recall: "
    f"{baseline_val_potable_recall:.4f}"
)


print(
    f"\nTest Accuracy: "
    f"{baseline_test_metrics['accuracy']:.4f}"
)


print(
    f"Test Macro F1: "
    f"{baseline_test_metrics['macro_f1']:.4f}"
)


print(
    f"Test Balanced Accuracy: "
    f"{baseline_test_metrics['balanced_accuracy']:.4f}"
)


print(
    f"Test Potable Recall: "
    f"{baseline_test_potable_recall:.4f}"
)


# ================================================================
# 28. PHASE 2 COMPLETED
# ================================================================

print("\n" + "=" * 70)
print("PHASE 2 COMPLETED")
print("=" * 70)