# ============================================================
# PHASE 5 - DIFFERENTIAL EVOLUTION
# SOLAR FLARE CLASSIFICATION USING IMBALANCED TRAINING DATA
# ============================================================
#
# Purpose:
# --------
# This phase investigates whether Differential Evolution (DE)
# can find a better MLP hyperparameter configuration for the
# original highly imbalanced Solar Flare dataset.
#
# IMPORTANT:
# ----------
# This experiment intentionally DOES NOT use:
#
#       - Full SMOTENC
#       - Controlled SMOTENC
#       - Artificial oversampling
#       - Class-weight based resampling
#
# The original imbalanced training distribution is retained.
#
#
# Why Differential Evolution?
# ---------------------------
# LEA and TBLO use teacher/learner-based mechanisms to update
# candidate solutions.
#
# Differential Evolution uses a fundamentally different
# population-based mechanism.
#
# DE generates new candidate solutions using differences
# between existing population members.
#
# In simplified form:
#
#       Mutant = X1 + F * (X2 - X3)
#
# where:
#
#       X1, X2, X3 = different candidate solutions
#       F          = differential weight
#
# This allows DE to explore the hyperparameter search space
# using relationships between existing candidate solutions.
#
# Therefore, DE provides a different optimization strategy
# from LEA and TBLO rather than another teacher/learner method.
#
#
# Why Macro F1?
# -------------
# The dataset is highly imbalanced:
#
#       Common Flare       : 125
#       Moderate Flare     : 50
#       No Flare            : 1203
#       Severe Flare        : 11
#
# Accuracy alone can be misleading because the No Flare class
# dominates the dataset.
#
# Therefore Macro F1 is used as the primary optimization
# objective so that all four classes contribute equally.
#
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import time
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from scipy.optimize import differential_evolution

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.exceptions import ConvergenceWarning


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

POPULATION_SIZE = 10

MAX_GENERATIONS = 5

# Differential Evolution parameters
DE_MUTATION = (0.5, 1.0)

DE_RECOMBINATION = 0.7

# ------------------------------------------------------------
# Suppress only MLP convergence warnings during optimization.
#
# The optimizer evaluates many MLP configurations. Some
# configurations may reach max_iter before convergence.
#
# These warnings are expected during hyperparameter search and
# do not mean that Differential Evolution has failed.
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Solar Flare Dataset...")


DATA_PATH = "processed_datasets/solar_flare_processed.csv"


data = pd.read_csv(DATA_PATH)


print("\nDataset Shape:")
print(data.shape)


print("\nFirst Five Rows:")
print(data.head())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

TARGET_COLUMN = "Target"


X = data.drop(
    columns=[TARGET_COLUMN]
)


y = data[TARGET_COLUMN]


print("\nFeature Matrix Shape:")
print(X.shape)


print("\nTarget Shape:")
print(y.shape)


# ============================================================
# DISPLAY ORIGINAL TARGET DISTRIBUTION
# ============================================================

print("\nOriginal Target Distribution:")
print(y.value_counts())


# ============================================================
# ENCODE TARGET
# ============================================================

target_mapping = {

    "Common Flare": 0,

    "Moderate Flare": 1,

    "No Flare": 2,

    "Severe Flare": 3

}


y = y.map(target_mapping)


if y.isna().any():

    raise ValueError(
        "Unknown target class detected."
    )


y = y.astype(int)


print("\nEncoded Target Distribution:")
print(y.value_counts().sort_index())


print("\nEncoded Target Classes:")

for label, value in target_mapping.items():

    print(
        f"{value}: {label}"
    )


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================
#
# We explicitly include both object and string dtypes.
#
# This avoids the Pandas warning associated with using only
# include=["object"] in newer Pandas versions.
#
# ============================================================

categorical_features = X.select_dtypes(

    include=["object", "str"]

).columns.tolist()


print("\nCategorical Features:")

for feature in categorical_features:

    print(feature)


categorical_indices = [

    X.columns.get_loc(feature)

    for feature in categorical_features

]


print("\nCategorical Feature Indices:")
print(categorical_indices)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
#
# The test set remains completely untouched.
#
# It is NOT:
#
#       - used for optimization
#       - used for cross-validation
#       - used for training
#
# ============================================================

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=RANDOM_STATE

)


print("\nTraining Feature Shape:")
print(X_train.shape)


print("\nTesting Feature Shape:")
print(X_test.shape)


print("\nTraining Target Distribution:")
print(y_train.value_counts().sort_index())


print("\nTesting Target Distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# FEATURE ENCODING
# ============================================================
#
# IMPORTANT:
# ----------
# No SMOTE or SMOTENC is performed in Phase 5.
#
# The original imbalanced training data is retained.
#
# Categorical features are one-hot encoded.
# Numerical features are passed through unchanged.
#
# ============================================================

numerical_features = [

    feature

    for feature in X.columns

    if feature not in categorical_features

]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_features

        ),

        (
            "numerical",

            "passthrough",

            numerical_features

        )

    ]

)


print("\nEncoding Features...")


X_train_encoded = preprocessor.fit_transform(
    X_train
)


X_test_encoded = preprocessor.transform(
    X_test
)


print("\nTraining Feature Shape After Encoding:")
print(X_train_encoded.shape)


print("\nTesting Feature Shape After Encoding:")
print(X_test_encoded.shape)


# ============================================================
# FEATURE STANDARDIZATION
# ============================================================

print("\nStandardizing Features...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_encoded
)


X_test_scaled = scaler.transform(
    X_test_encoded
)


print("\nTraining Feature Shape After Standardization:")
print(X_train_scaled.shape)


print("\nTesting Feature Shape After Standardization:")
print(X_test_scaled.shape)


print("\nFeature Standardization Completed.")


# ============================================================
# VERIFY IMBALANCE IS RETAINED
# ============================================================

print("\n" + "=" * 70)

print("PHASE 5 - IMBALANCED TRAINING DATA")

print("=" * 70)


print("\nNo SMOTE or SMOTENC is applied.")


print("\nTraining Distribution Used by DE:")

print(

    pd.Series(y_train).value_counts().sort_index()

)


# ============================================================
# STRATIFIED CROSS-VALIDATION
# ============================================================
#
# Stratification attempts to preserve the class distribution
# in every validation fold.
#
# ============================================================

cv = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# DIFFERENTIAL EVOLUTION SEARCH SPACE
# ============================================================
#
# Differential Evolution operates on numerical vectors.
#
# Therefore categorical MLP hyperparameters are represented
# using integer IDs.
#
# Architecture ID:
#
#       0 -> (16,)
#       1 -> (32,)
#       2 -> (64,)
#       3 -> (128,)
#       4 -> (32, 16)
#       5 -> (64, 32)
#       6 -> (128, 64)
#
# Activation ID:
#
#       0 -> relu
#       1 -> tanh
#
# Solver:
#
#       Only adam is used in this experiment.
#
# Continuous parameters:
#
#       learning_rate_init
#       alpha
#
# Integer parameters:
#
#       batch_size
#       max_iter
#
# ============================================================

ARCHITECTURES = [

    (16,),

    (32,),

    (64,),

    (128,),

    (32, 16),

    (64, 32),

    (128, 64)

]


ACTIVATIONS = [

    "relu",

    "tanh"

]


# ============================================================
# DE SEARCH BOUNDS
# ============================================================
#
# Each tuple corresponds to one dimension of the DE solution.
#
# Vector:
#
#       [architecture,
#        activation,
#        learning_rate_init,
#        alpha,
#        batch_size,
#        max_iter]
#
# ============================================================

DE_BOUNDS = [

    (0, len(ARCHITECTURES) - 1),

    (0, len(ACTIVATIONS) - 1),

    (0.0001, 0.005),

    (0.00001, 0.01),

    (16, 64),

    (500, 1000)

]


# ============================================================
# CONVERT DE VECTOR TO MLP PARAMETERS
# ============================================================

def decode_solution(solution):

    """
    Convert a numerical Differential Evolution solution
    into valid MLPClassifier hyperparameters.

    DE works with numerical vectors, while MLPClassifier
    requires actual Python values such as tuples and strings.
    """

    architecture_index = int(
        np.clip(
            round(solution[0]),
            0,
            len(ARCHITECTURES) - 1
        )
    )


    activation_index = int(
        np.clip(
            round(solution[1]),
            0,
            len(ACTIVATIONS) - 1
        )
    )


    learning_rate_init = float(
        solution[2]
    )


    alpha = float(
        solution[3]
    )


    batch_size = int(
        round(solution[4])
    )


    max_iter = int(
        round(solution[5])
    )


    return {

        "hidden_layer_sizes":
            ARCHITECTURES[architecture_index],

        "activation":
            ACTIVATIONS[activation_index],

        "solver":
            "adam",

        "learning_rate_init":
            learning_rate_init,

        "alpha":
            alpha,

        "batch_size":
            batch_size,

        "max_iter":
            max_iter

    }


# ============================================================
# DIFFERENTIAL EVOLUTION OBJECTIVE FUNCTION
# ============================================================
#
# The objective function is evaluated for every candidate
# generated by Differential Evolution.
#
# Each candidate:
#
#       1. Is converted into valid MLP parameters.
#       2. Is evaluated using 5-fold Stratified CV.
#       3. Macro F1 is calculated for every validation fold.
#       4. Mean Macro F1 is returned.
#
# scipy minimizes the objective function.
#
# Therefore:
#
#       objective = -Macro F1
#
# A larger Macro F1 produces a smaller objective value.
#
# ============================================================

evaluation_counter = 0


def objective_function(solution):

    global evaluation_counter

    evaluation_counter += 1


    params = decode_solution(
        solution
    )


    fold_scores = []


    for train_indices, validation_indices in cv.split(

        X_train_scaled,
        y_train

    ):

        X_cv_train = X_train_scaled[
            train_indices
        ]

        X_cv_validation = X_train_scaled[
            validation_indices
        ]


        y_cv_train = y_train.iloc[
            train_indices
        ]

        y_cv_validation = y_train.iloc[
            validation_indices
        ]


        model = MLPClassifier(

            hidden_layer_sizes=
                params["hidden_layer_sizes"],

            activation=
                params["activation"],

            solver=
                params["solver"],

            learning_rate_init=
                params["learning_rate_init"],

            alpha=
                params["alpha"],

            batch_size=
                params["batch_size"],

            max_iter=
                params["max_iter"],

            random_state=RANDOM_STATE

        )


        model.fit(

            X_cv_train,

            y_cv_train

        )


        predictions = model.predict(

            X_cv_validation

        )


        score = f1_score(

            y_cv_validation,

            predictions,

            average="macro",

            zero_division=0

        )


        fold_scores.append(
            score
        )


    mean_macro_f1 = np.mean(
        fold_scores
    )


    return -mean_macro_f1


# ============================================================
# START DIFFERENTIAL EVOLUTION
# ============================================================

print("\n" + "=" * 70)

print("PHASE 5 - DIFFERENTIAL EVOLUTION")

print("=" * 70)


print("\nOptimization Algorithm:")
print("Differential Evolution")


print("\nPopulation Size:")
print(POPULATION_SIZE)


print("\nMaximum Generations:")
print(MAX_GENERATIONS)


print("\nCross-validation folds:")
print(N_SPLITS)


print("\nOptimization Metric:")
print("Macro F1")


print("\nStarting Differential Evolution...")


optimization_start = time.time()


de_result = differential_evolution(

    objective_function,

    bounds=DE_BOUNDS,

    strategy="best1bin",

    maxiter=MAX_GENERATIONS,

    popsize=POPULATION_SIZE,

    mutation=DE_MUTATION,

    recombination=DE_RECOMBINATION,

    seed=RANDOM_STATE,

    polish=True,

    updating="immediate",

    workers=1,

    disp=True

)


optimization_end = time.time()


de_runtime = (

    optimization_end

    - optimization_start

)


# ============================================================
# DE COMPLETION
# ============================================================

print("\nDifferential Evolution completed.")


print(

    f"Optimization Time : "
    f"{de_runtime:.2f} seconds"

)


print(

    f"Objective Evaluations : "
    f"{evaluation_counter}"

)


# ============================================================
# DE BEST PARAMETERS
# ============================================================

best_de_params = decode_solution(

    de_result.x

)


best_de_cv_macro_f1 = -de_result.fun


print("\n" + "=" * 70)

print("BEST HYPERPARAMETERS FOUND BY DIFFERENTIAL EVOLUTION")

print("=" * 70)


for key, value in best_de_params.items():

    print(

        f"{key}: {value}"

    )


print(

    f"\nBest Cross Validation Macro F1: "
    f"{best_de_cv_macro_f1:.4f}"

)


# ============================================================
# TRAIN BEST DE MODEL
# ============================================================
#
# After DE identifies the best hyperparameter configuration,
# the model is trained using the COMPLETE original imbalanced
# training dataset.
#
# No SMOTE is applied.
#
# ============================================================

print("\n" + "=" * 70)

print("TRAINING BEST DIFFERENTIAL EVOLUTION MODEL")

print("=" * 70)


best_de_model = MLPClassifier(

    hidden_layer_sizes=
        best_de_params["hidden_layer_sizes"],

    activation=
        best_de_params["activation"],

    solver=
        best_de_params["solver"],

    learning_rate_init=
        best_de_params["learning_rate_init"],

    alpha=
        best_de_params["alpha"],

    batch_size=
        best_de_params["batch_size"],

    max_iter=
        best_de_params["max_iter"],

    random_state=RANDOM_STATE

)


training_start = time.time()


best_de_model.fit(

    X_train_scaled,

    y_train

)


training_end = time.time()


de_training_runtime = (

    training_end

    - training_start

)


print(

    f"\nTraining Time : "
    f"{de_training_runtime:.2f} seconds"

)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print("\nMaking Differential Evolution Test Predictions...")


de_predictions = best_de_model.predict(

    X_test_scaled

)


# ============================================================
# CALCULATE TEST METRICS
# ============================================================

de_accuracy = accuracy_score(

    y_test,

    de_predictions

)


de_precision = precision_score(

    y_test,

    de_predictions,

    average="weighted",

    zero_division=0

)


de_recall = recall_score(

    y_test,

    de_predictions,

    average="weighted",

    zero_division=0

)


de_weighted_f1 = f1_score(

    y_test,

    de_predictions,

    average="weighted",

    zero_division=0

)


de_macro_f1 = f1_score(

    y_test,

    de_predictions,

    average="macro",

    zero_division=0

)


# ============================================================
# DISPLAY DE RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "IMBALANCED DATA + DIFFERENTIAL EVOLUTION RESULTS"
)

print("=" * 70)


print(

    f"Accuracy           : "
    f"{de_accuracy:.4f}"

)


print(

    f"Weighted Precision : "
    f"{de_precision:.4f}"

)


print(

    f"Weighted Recall    : "
    f"{de_recall:.4f}"

)


print(

    f"Weighted F1        : "
    f"{de_weighted_f1:.4f}"

)


print(

    f"Macro F1           : "
    f"{de_macro_f1:.4f}"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        de_predictions,

        labels=[0, 1, 2, 3],

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        de_predictions,

        labels=[0, 1, 2, 3]

    )

)

# ============================================================
# VISUALIZATION 1 - DIFFERENTIAL EVOLUTION PERFORMANCE
# ============================================================

import matplotlib.pyplot as plt


metrics = [
    "Accuracy",
    "Weighted F1",
    "Macro F1"
]


values = [
    de_accuracy,
    de_weighted_f1,
    de_macro_f1
]


plt.figure(figsize=(8, 5))


bars = plt.bar(
    metrics,
    values
)


plt.xlabel("Performance Metric")

plt.ylabel("Score")

plt.title(
    "Differential Evolution - Test Performance"
)


plt.ylim(0, 1)


# Display values above bars
for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 2 - DIFFERENTIAL EVOLUTION COMPUTATIONAL TIME
# ============================================================

time_labels = [
    "Optimization Time",
    "Training Time"
]


time_values = [
    de_runtime,
    de_training_runtime
]


plt.figure(figsize=(8, 5))


bars = plt.bar(
    time_labels,
    time_values
)


plt.xlabel("Processing Stage")

plt.ylabel("Time (seconds)")

plt.title(
    "Differential Evolution - Computational Time"
)


# Display values above bars
for bar, value in zip(bars, time_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + (max(time_values) * 0.02),
        f"{value:.2f} s",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.show()
# ============================================================
# FINAL DE SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("DIFFERENTIAL EVOLUTION SUMMARY")

print("=" * 70)


print(

    f"Best CV Macro F1 : "
    f"{best_de_cv_macro_f1:.4f}"

)


print(

    f"Optimization Time : "
    f"{de_runtime:.2f} seconds"

)


print(

    f"Training Time     : "
    f"{de_training_runtime:.2f} seconds"

)


print(

    f"Test Accuracy     : "
    f"{de_accuracy:.4f}"

)


print(

    f"Test Weighted F1  : "
    f"{de_weighted_f1:.4f}"

)


print(

    f"Test Macro F1     : "
    f"{de_macro_f1:.4f}"

)


# ============================================================
# EXPERIMENT COMPLETED
# ============================================================

print("\n" + "=" * 70)

print("PHASE 5 DIFFERENTIAL EVOLUTION COMPLETED")

print("=" * 70)