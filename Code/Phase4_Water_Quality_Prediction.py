# =====================================================
# WATER QUALITY PREDICTION DATASET
# =====================================================
#
# Dataset Name:
# Water Potability
#
# Source:
# Kaggle / UCI-style Water Potability dataset used in this thesis
#
# Machine Learning Task:
# Binary Classification
#
# =====================================================
# DATASET DESCRIPTION
# =====================================================
#
# The Water Potability dataset contains physicochemical
# measurements of water samples together with a binary
# potability label. The dataset includes pH, hardness,
# dissolved solids, chloramines, sulfate, conductivity,
# organic carbon, trihalomethanes, and turbidity.
#
# The target variable "Potability" indicates whether the
# water sample is considered potable (1) or non-potable (0).
#
# =====================================================
# PHASE 3 OBJECTIVE
# =====================================================
#
# Compare multiple MLP hyperparameter optimization methods
# under an equal computational resource budget.
#
# Optimizers:
#   1. Random Search
#   2. Bayesian Optimization
#   3. Hyperband-style successive halving
#   4. Learning Evolutionary Algorithm (LEA)
#   5. Teaching-Learning-Based Optimization (TLBO)
#   6. Simulated Annealing (SA)
#
# Primary optimization metric:
#   Validation Macro F1
#
# Higher Macro F1 indicates better validation performance.
#
# IMPORTANT EXPERIMENTAL RULES
# ----------------------------
# 1. The Phase 2 preprocessing pipeline is frozen.
# 2. The validation set is used for optimizer selection.
# 3. The test set remains untouched until final evaluation.
# 4. Every optimizer receives the same total computational
#    resource budget of 3,000 training-iteration units.
# 5. Final models are retrained on Train + Validation data.
# 6. The final test set is evaluated only after optimization.
#
# =====================================================

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import math
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning


# =====================================================
# WARNING HANDLING
# =====================================================
#
# Convergence warnings are suppressed during the long Phase 3
# search because the fixed max_iter value is intentionally used
# as a computational resource cap.
#
# Final training iteration counts are still recorded through
# model.n_iter_.
# =====================================================

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# =====================================================
# GLOBAL RANDOM SEED
# =====================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = "Data/water_potability.csv"

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "../Data/water_potability.csv"


# =====================================================
# TRAIN / VALIDATION / TEST SPLIT
# =====================================================
#
# Water Quality is an i.i.d. tabular classification problem,
# so stratified random splitting is retained from the frozen
# Phase 2 pipeline.
#
# Final proportions:
#   Training   = 64%
#   Validation = 16%
#   Testing    = 20%
# =====================================================

TRAIN_RATIO = 0.64
VALIDATION_RATIO = 0.16
TEST_RATIO = 0.20


# =====================================================
# FROZEN PHASE 2 PREPROCESSING
# =====================================================
#
# Phase 2 selected median imputation.
#
# StandardScaler is retained for MLP training.
#
# The scaler and imputer used during optimization are fitted
# only on the training set.
#
# For final model training, a new imputer and scaler are fitted
# on the complete development set (Training + Validation) and
# then applied to the untouched test set.
# =====================================================

FROZEN_IMPUTATION = "median"


# =====================================================
# EQUAL COMPUTATIONAL BUDGET
# =====================================================
#
# Common maximum MLP training resource:
#     500 iterations
#
# Common total budget per optimizer:
#     6 full-equivalent evaluations × 500
#     = 3,000 training-iteration resource units
#
# Hyperband receives the same total resource through staged
# successive halving:
#     8 candidates × 200
#     3 survivors  × 300
#     1 survivor   × 500
#
# Total:
#     1600 + 900 + 500 = 3000
# =====================================================

N_OPTIMIZER_EVALUATIONS = 6

MAX_ITER = 500

TOTAL_RESOURCE_BUDGET = (
    N_OPTIMIZER_EVALUATIONS * MAX_ITER
)


# -----------------------------------------------------
# Hyperband budget
# -----------------------------------------------------

HYPERBAND_INITIAL_CANDIDATES = 8
HYPERBAND_SURVIVORS_STAGE_1 = 3
HYPERBAND_SURVIVORS_STAGE_2 = 1

HYPERBAND_RESOURCES = [
    200,
    300,
    500,
]


# -----------------------------------------------------
# LEA budget
# -----------------------------------------------------
#
# Initial population:
#     3 evaluations
#
# One generation:
#     3 offspring evaluations
#
# Total:
#     6 evaluations × 500 = 3000
# -----------------------------------------------------

LEA_POPULATION_SIZE = 3
LEA_GENERATIONS = 1
LEA_SURVIVOR_COUNT = 2


# -----------------------------------------------------
# TLBO budget
# -----------------------------------------------------
#
# Initial population:
#     2 evaluations
#
# One generation:
#     2 teacher-phase evaluations
#     2 learner-phase evaluations
#
# Total:
#     6 evaluations × 500 = 3000
# -----------------------------------------------------

TLBO_POPULATION_SIZE = 2
TLBO_GENERATIONS = 1


# -----------------------------------------------------
# Simulated Annealing budget
# -----------------------------------------------------
#
# Initial configuration:
#     1 evaluation
#
# Neighbour evaluations:
#     5 evaluations
#
# Total:
#     6 evaluations × 500 = 3000
# -----------------------------------------------------

SA_EVALUATIONS = 6
SA_INITIAL_TEMPERATURE = 0.03
SA_COOLING_RATE = 0.90


# =====================================================
# SEARCH SPACE
# =====================================================
#
# The underlying MLP search space is identical for all
# optimizers. Only the search strategy changes.
#
# Adam is fixed as the MLP solver and is therefore not part
# of the optimization search space.
# =====================================================

SEARCH_SPACE = {

    "hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],

    "activation": [
        "relu",
        "tanh",
    ],

    "learning_rate_init": [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.005,
    ],

    "alpha": [
        0.00001,
        0.0001,
        0.001,
        0.01,
    ],

    "batch_size": [
        "auto",
        16,
        32,
        64,
        128,
    ],
}


# =====================================================
# HELPER: CONFIGURATION SIGNATURE
# =====================================================


def configuration_signature(params):
    """Create a hashable signature for a hyperparameter configuration."""

    return (
        params["hidden_layer_sizes"],
        params["activation"],
        params["learning_rate_init"],
        params["alpha"],
        params["batch_size"],
    )


# =====================================================
# HELPER: RANDOM CONFIGURATION SAMPLING
# =====================================================


def sample_random_configuration(rng):
    """Sample one configuration from the common discrete search space."""

    return {
        parameter: rng.choice(values)
        for parameter, values in SEARCH_SPACE.items()
    }


# =====================================================
# LOAD DATASET
# =====================================================

print("\n" + "=" * 80)
print("PHASE 3 - WATER QUALITY PREDICTION")
print("=" * 80)

print("\nDataset Path:")
print(os.path.abspath(DATASET_PATH))


dataset = pd.read_csv(DATASET_PATH)

print("\nDataset Shape:")
print(dataset.shape)

print("\nFirst Five Rows:")
print(dataset.head())


# =====================================================
# FEATURE MATRIX AND TARGET VARIABLE
# =====================================================

X = dataset.drop(
    columns=["Potability"]
)

y = dataset["Potability"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nPotability Distribution:")
print(y.value_counts().sort_index())


# =====================================================
# ONE-HOT ENCODING
# =====================================================
#
# No categorical predictor variables are present in this
# dataset. All predictors are numerical, so no one-hot
# encoding is necessary.
# =====================================================


# =====================================================
# TRAIN / VALIDATION / TEST SPLIT
# =====================================================

print("\n" + "=" * 80)
print("FIXED STRATIFIED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_RATIO,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO),
    stratify=y_train_temp,
    random_state=RANDOM_STATE,
)

# Preserve raw copies for final development-set preprocessing.
X_train_raw = X_train.copy()
X_validation_raw = X_validation.copy()
X_test_raw = X_test.copy()

y_train_raw = y_train.copy()
y_validation_raw = y_validation.copy()
y_test_raw = y_test.copy()

print("\nTraining Feature Shape:")
print(X_train.shape)

print("Validation Feature Shape:")
print(X_validation.shape)

print("Testing Feature Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("Validation Target Shape:")
print(y_validation.shape)

print("Testing Target Shape:")
print(y_test.shape)


# =====================================================
# OPTIMIZATION-STAGE IMPUTATION
# =====================================================
#
# The optimizer sees only the training-fitted preprocessing.
# This preserves the frozen Phase 2 protocol.
# =====================================================

print("\nApplying frozen Phase 2 imputation:")
print(FROZEN_IMPUTATION)

if FROZEN_IMPUTATION != "median":
    raise ValueError(
        "Unexpected frozen imputation method."
    )


optimization_imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = optimization_imputer.fit_transform(
    X_train
)

X_validation_imputed = optimization_imputer.transform(
    X_validation
)

X_test_imputed = optimization_imputer.transform(
    X_test
)


# =====================================================
# OPTIMIZATION-STAGE STANDARDIZATION
# =====================================================

print("\nApplying StandardScaler...")

optimization_scaler = StandardScaler()

X_train_scaled = optimization_scaler.fit_transform(
    X_train_imputed
)

X_validation_scaled = optimization_scaler.transform(
    X_validation_imputed
)

X_test_scaled = optimization_scaler.transform(
    X_test_imputed
)

print("Feature Standardization Completed.")


# =====================================================
# HELPER: BUILD MLP
# =====================================================
#
# The optimizer resource is represented by max_iter.
# Early stopping is disabled and tol is fixed so that the
# allocated iteration budget remains explicit and comparable.
# =====================================================


def build_mlp(params, max_iter=MAX_ITER):
    """Build a deterministic MLP classifier."""

    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver="adam",
        learning_rate_init=params["learning_rate_init"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=max_iter,
    )


# =====================================================
# HELPER: VALIDATION METRICS
# =====================================================


def calculate_classification_metrics(
    y_true,
    predictions,
):
    """Calculate all classification metrics used by the thesis."""

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "potable_recall": recall_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
    }


# =====================================================
# HELPER: VALIDATION MODEL EVALUATION
# =====================================================
#
# All optimizers use exactly the same validation set and
# exactly the same primary metric: Macro F1.
# =====================================================


def evaluate_validation_model(model):
    """Evaluate a trained model on the fixed validation set."""

    predictions = model.predict(
        X_validation_scaled
    )

    metrics = calculate_classification_metrics(
        y_validation,
        predictions,
    )

    return metrics


# =====================================================
# HELPER: COMMON OPTIMIZER CONFIGURATION EVALUATION
# =====================================================
#
# This function is the central evaluation mechanism for every
# optimizer. It ensures that Random Search, Bayesian, Hyperband,
# LEA, TLBO and SA all use:
#
#   X_train_scaled -> training
#   X_validation_scaled -> validation
#   Macro F1 -> primary fitness
#   max_iter -> resource allocation
# =====================================================


def evaluate_configuration(
    params,
    max_iter=MAX_ITER,
):
    """Train and evaluate one MLP configuration."""

    start = time.time()

    model = build_mlp(
        params,
        max_iter=max_iter,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    validation_metrics = evaluate_validation_model(
        model
    )

    runtime = time.time() - start

    return {
        "params": params.copy(),
        "model": model,
        "validation_metrics": validation_metrics,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "runtime": runtime,
        "iterations": model.n_iter_,
    }


# =====================================================
# HELPER: FINAL TEST EVALUATION
# =====================================================


def evaluate_test_model(
    model,
    method_name,
):
    """Evaluate one final development-trained model on the untouched test set."""

    predictions = model.predict(
        X_test_final
    )

    metrics = calculate_classification_metrics(
        y_test_raw,
        predictions,
    )

    print(
        f"\n========== {method_name.upper()} RESULTS =========="
    )

    print(
        f"Accuracy             : {metrics['accuracy']:.4f}"
    )
    print(
        f"Macro Precision      : {metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall         : {metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1 Score       : {metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1 Score    : {metrics['weighted_f1']:.4f}"
    )
    print(
        f"Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Potable Recall       : {metrics['potable_recall']:.4f}"
    )

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test_raw,
            predictions,
            target_names=[
                "Non-Potable",
                "Potable",
            ],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix\n")
    print(
        confusion_matrix(
            y_test_raw,
            predictions,
        )
    )

    return {
        **metrics,
        "predictions": predictions,
    }


# =====================================================
# HELPER: TRAINING TIME
# =====================================================


def train_final_model(
    params,
    method_name,
):
    """Train one selected configuration on the complete development set."""

    print(
        f"\nTraining Final {method_name} Model on Development Data..."
    )

    start = time.time()

    model = build_mlp(
        params,
        max_iter=MAX_ITER,
    )

    model.fit(
        X_development_final,
        y_development,
    )

    training_time = time.time() - start

    print(
        f"Training Time: {training_time:.2f} seconds"
    )

    return model, training_time


# =====================================================
# HELPER: SAVE TRAINING LOSS PLOT
# =====================================================


def save_training_loss_plot(
    model,
    method_name,
):
    """Save the MLP training-loss curve."""

    if not hasattr(model, "loss_curve_"):
        return

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        model.loss_curve_
    )

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(
        f"{method_name} Training Loss"
    )
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(
        RESULTS_DIR,
        f"water_quality_phase3_{method_name.lower().replace(' ', '_')}_training_loss.png"
    )

    plt.savefig(
        file_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =====================================================
# BASELINE MLP
# =====================================================
#
# Baseline configuration frozen from Phase 2.
# =====================================================

baseline_params = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": "auto",
}

print("\n" + "=" * 80)
print("BASELINE MLP")
print("=" * 80)

baseline_validation_start = time.time()

baseline_validation_model = build_mlp(
    baseline_params,
    max_iter=MAX_ITER,
)

baseline_validation_model.fit(
    X_train_scaled,
    y_train,
)

baseline_validation_training_time = (
    time.time() - baseline_validation_start
)

baseline_validation_metrics = evaluate_validation_model(
    baseline_validation_model
)

print(
    f"\nBaseline Validation Accuracy          : {baseline_validation_metrics['accuracy']:.4f}"
)
print(
    f"Baseline Validation Macro Precision   : {baseline_validation_metrics['macro_precision']:.4f}"
)
print(
    f"Baseline Validation Macro Recall      : {baseline_validation_metrics['macro_recall']:.4f}"
)
print(
    f"Baseline Validation Macro F1          : {baseline_validation_metrics['macro_f1']:.4f}"
)
print(
    f"Baseline Validation Balanced Accuracy : {baseline_validation_metrics['balanced_accuracy']:.4f}"
)
print(
    f"Baseline Validation Potable Recall    : {baseline_validation_metrics['potable_recall']:.4f}"
)


# =====================================================
# RANDOM SEARCH
# =====================================================

print("\n" + "=" * 80)
print("RANDOM SEARCH")
print("=" * 80)

random_rng = random.Random(
    RANDOM_STATE
)

random_search_runs = []
random_seen = set()
random_convergence = []

random_start = time.time()

for evaluation in range(
    1,
    N_OPTIMIZER_EVALUATIONS + 1,
):

    while True:

        params = sample_random_configuration(
            random_rng
        )

        signature = configuration_signature(
            params
        )

        if signature not in random_seen:
            random_seen.add(signature)
            break

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    random_search_runs.append(
        run
    )

    best_so_far = max(
        result["validation_macro_f1"]
        for result in random_search_runs
    )

    random_convergence.append(
        {
            "Method": "Random Search",
            "Evaluation": evaluation,
            "Cumulative Resource Units": evaluation * MAX_ITER,
            "Best Validation Macro F1": best_so_far,
        }
    )

    print(
        f"Evaluation {evaluation:02d}/{N_OPTIMIZER_EVALUATIONS} "
        f"- Macro F1 = {run['validation_macro_f1']:.4f}"
    )

random_search_runtime = (
    time.time() - random_start
)

random_best_run = max(
    random_search_runs,
    key=lambda x: x["validation_macro_f1"],
)

random_best_params = (
    random_best_run["params"]
)

random_best_score = (
    random_best_run["validation_macro_f1"]
)

random_evaluations = len(
    random_search_runs
)

print("\nRandom Search completed.")
print(
    f"Optimization Time: {random_search_runtime:.2f} seconds"
)
print(
    f"Best Validation Macro F1: {random_best_score:.4f}"
)

print("\nBest Hyperparameters Found By Random Search")
for key, value in random_best_params.items():
    print(f"{key}: {value}")


# =====================================================
# BAYESIAN OPTIMIZATION
# =====================================================

print("\n" + "=" * 80)
print("BAYESIAN OPTIMIZATION")
print("=" * 80)

HIDDEN_LAYER_MAP = {
    "32": (32,),
    "64": (64,),
    "128": (128,),
    "64_32": (64, 32),
    "128_64": (128, 64),
    "128_64_32": (128, 64, 32),
}

bayesian_runs = {}
bayesian_convergence = []

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    ),
    study_name="Water_Quality_MLP_Bayesian",
)


def bayesian_objective(trial):

    hidden_layer_name = trial.suggest_categorical(
        "hidden_layer_sizes",
        list(HIDDEN_LAYER_MAP.keys()),
    )

    params = {
        "hidden_layer_sizes": HIDDEN_LAYER_MAP[
            hidden_layer_name
        ],
        "activation": trial.suggest_categorical(
            "activation",
            SEARCH_SPACE["activation"],
        ),
        "learning_rate_init": trial.suggest_categorical(
            "learning_rate_init",
            SEARCH_SPACE["learning_rate_init"],
        ),
        "alpha": trial.suggest_categorical(
            "alpha",
            SEARCH_SPACE["alpha"],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            SEARCH_SPACE["batch_size"],
        ),
    }

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    bayesian_runs[
        trial.number
    ] = run

    best_so_far = max(
        result["validation_macro_f1"]
        for result in bayesian_runs.values()
    )

    bayesian_convergence.append(
        {
            "Method": "Bayesian Optimization",
            "Evaluation": trial.number + 1,
            "Cumulative Resource Units": (
                trial.number + 1
            ) * MAX_ITER,
            "Best Validation Macro F1": best_so_far,
        }
    )

    return run["validation_macro_f1"]


bayesian_start = time.time()

study.optimize(
    bayesian_objective,
    n_trials=N_OPTIMIZER_EVALUATIONS,
)

bayesian_runtime = (
    time.time() - bayesian_start
)

bayesian_best_run = bayesian_runs[
    study.best_trial.number
]

bayesian_best_params = (
    bayesian_best_run["params"]
)

bayesian_best_score = (
    bayesian_best_run["validation_macro_f1"]
)

bayesian_evaluations = len(
    bayesian_runs
)

print("\nBayesian Optimization completed.")
print(
    f"Optimization Time: {bayesian_runtime:.2f} seconds"
)
print(
    f"Best Validation Macro F1: {bayesian_best_score:.4f}"
)

print("\nBest Hyperparameters Found By Bayesian Optimization")
for key, value in bayesian_best_params.items():
    print(f"{key}: {value}")


# =====================================================
# HYPERBAND-STYLE SUCCESSIVE HALVING
# =====================================================
#
# This implementation performs explicit staged resource
# allocation so that the total resource consumption is known:
#
# Stage 1: 8 candidates × 200
# Stage 2: 3 candidates × 300
# Stage 3: 1 candidate  × 500
#
# The same configuration is retrained from scratch at each
# stage. Therefore, this is a custom Hyperband-style
# successive-halving implementation rather than a warm-start
# continuation implementation.
# =====================================================

print("\n" + "=" * 80)
print("HYPERBAND")
print("=" * 80)

hyperband_rng = random.Random(
    RANDOM_STATE + 1
)

hyperband_candidates = []
hyperband_seen = set()

while len(hyperband_candidates) < HYPERBAND_INITIAL_CANDIDATES:

    params = sample_random_configuration(
        hyperband_rng
    )

    signature = configuration_signature(
        params
    )

    if signature in hyperband_seen:
        continue

    hyperband_seen.add(signature)
    hyperband_candidates.append(params)

hyperband_evaluations = 0
hyperband_resource_units = 0
hyperband_convergence = []
hyperband_best_score = float("-inf")
hyperband_best_params = None

hyperband_start = time.time()

stage_definitions = [
    (
        "Stage 1",
        HYPERBAND_RESOURCES[0],
        HYPERBAND_SURVIVORS_STAGE_1,
    ),
    (
        "Stage 2",
        HYPERBAND_RESOURCES[1],
        HYPERBAND_SURVIVORS_STAGE_2,
    ),
    (
        "Stage 3",
        HYPERBAND_RESOURCES[2],
        0,
    ),
]

final_stage_results = []

for stage_name, resource, survivor_count in stage_definitions:

    print("\n" + "-" * 70)
    print(stage_name)
    print(
        f"Candidates: {len(hyperband_candidates)}"
    )
    print(
        f"Max Iterations: {resource}"
    )
    print("-" * 70)

    stage_results = []

    for params in hyperband_candidates:

        run = evaluate_configuration(
            params,
            max_iter=resource,
        )

        stage_results.append(run)

        hyperband_evaluations += 1
        hyperband_resource_units += resource

        if (
            run["validation_macro_f1"]
            > hyperband_best_score
        ):
            hyperband_best_score = (
                run["validation_macro_f1"]
            )
            hyperband_best_params = (
                run["params"].copy()
            )

        hyperband_convergence.append(
            {
                "Method": "Hyperband",
                "Evaluation": hyperband_evaluations,
                "Cumulative Resource Units": hyperband_resource_units,
                "Best Validation Macro F1": hyperband_best_score,
                "Stage": stage_name,
            }
        )

        print(
            f"Evaluation {hyperband_evaluations:02d} "
            f"- Macro F1 = "
            f"{run['validation_macro_f1']:.4f}"
        )

    final_stage_results = stage_results

    if survivor_count > 0:

        stage_results = sorted(
            stage_results,
            key=lambda x: x["validation_macro_f1"],
            reverse=True,
        )

        hyperband_candidates = [
            result["params"]
            for result in stage_results[:survivor_count]
        ]

hyperband_runtime = (
    time.time() - hyperband_start
)

# The final-stage candidate is the Hyperband-selected model.
hyperband_final_run = max(
    final_stage_results,
    key=lambda x: x["validation_macro_f1"],
)

hyperband_best_params = (
    hyperband_final_run["params"]
)

hyperband_best_score = (
    hyperband_final_run["validation_macro_f1"]
)

print("\nHyperband completed.")
print(
    f"Optimization Time: {hyperband_runtime:.2f} seconds"
)
print(
    f"Best Final-Stage Validation Macro F1: {hyperband_best_score:.4f}"
)

print("\nBest Hyperparameters Found By Hyperband")
for key, value in hyperband_best_params.items():
    print(f"{key}: {value}")


# =====================================================
# LEARNING EVOLUTIONARY ALGORITHM (LEA)
# =====================================================
#
# Compact discrete LEA:
#   - evaluate 3 initial candidates
#   - retain the top 2
#   - generate and evaluate 3 mutated offspring
#   - retain the best 3 for the next population
#
# Total = 6 evaluations.
# =====================================================

print("\n" + "=" * 80)
print("LEARNING EVOLUTIONARY ALGORITHM (LEA)")
print("=" * 80)

lea_rng = random.Random(
    RANDOM_STATE + 2
)

lea_population = []
lea_seen = set()

while len(lea_population) < LEA_POPULATION_SIZE:

    params = sample_random_configuration(
        lea_rng
    )

    signature = configuration_signature(
        params
    )

    if signature in lea_seen:
        continue

    lea_seen.add(signature)
    lea_population.append(params)

lea_evaluations = 0
lea_convergence = []
lea_best_score = float("-inf")
lea_best_params = None

lea_start = time.time()

# -----------------------------------------------------
# Initial population evaluation
# -----------------------------------------------------

lea_population_results = []

for params in lea_population:

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    lea_population_results.append(run)
    lea_evaluations += 1

    if run["validation_macro_f1"] > lea_best_score:
        lea_best_score = run["validation_macro_f1"]
        lea_best_params = run["params"].copy()

lea_convergence.append(
    {
        "Method": "LEA",
        "Evaluation": lea_evaluations,
        "Cumulative Resource Units": lea_evaluations * MAX_ITER,
        "Best Validation Macro F1": lea_best_score,
        "Generation": 0,
    }
)

# -----------------------------------------------------
# LEA generation(s)
# -----------------------------------------------------

for generation in range(
    1,
    LEA_GENERATIONS + 1,
):

    print(
        f"\nGeneration {generation}/{LEA_GENERATIONS}"
    )

    ranked_population = sorted(
        lea_population_results,
        key=lambda x: x["validation_macro_f1"],
        reverse=True,
    )

    survivors = ranked_population[
        :LEA_SURVIVOR_COUNT
    ]

    offspring_results = []

    while len(offspring_results) < LEA_POPULATION_SIZE:

        parent = lea_rng.choice(
            survivors
        )["params"].copy()

        parameter = lea_rng.choice(
            list(SEARCH_SPACE.keys())
        )

        current_value = parent[parameter]

        alternatives = [
            value
            for value in SEARCH_SPACE[parameter]
            if value != current_value
        ]

        if alternatives:
            parent[parameter] = lea_rng.choice(
                alternatives
            )

        child_run = evaluate_configuration(
            parent,
            MAX_ITER,
        )

        offspring_results.append(child_run)
        lea_evaluations += 1

        if (
            child_run["validation_macro_f1"]
            > lea_best_score
        ):
            lea_best_score = (
                child_run["validation_macro_f1"]
            )
            lea_best_params = (
                child_run["params"].copy()
            )

        lea_convergence.append(
            {
                "Method": "LEA",
                "Evaluation": lea_evaluations,
                "Cumulative Resource Units": lea_evaluations * MAX_ITER,
                "Best Validation Macro F1": lea_best_score,
                "Generation": generation,
            }
        )

    # Keep the best population-sized set from survivors + offspring.
    lea_population_results = sorted(
        survivors + offspring_results,
        key=lambda x: x["validation_macro_f1"],
        reverse=True,
    )[:LEA_POPULATION_SIZE]

lea_runtime = time.time() - lea_start

print("\nLEA Optimization completed.")
print(
    f"Optimization Time: {lea_runtime:.2f} seconds"
)
print(
    f"Evaluations: {lea_evaluations}"
)
print(
    f"Best Validation Macro F1: {lea_best_score:.4f}"
)

print("\nBest Hyperparameters Found By LEA")
for key, value in lea_best_params.items():
    print(f"{key}: {value}")


# =====================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# =====================================================
#
# Discrete TLBO representation:
#   - each hyperparameter is represented by an integer index
#     in its allowed option list
#   - teacher phase moves candidates toward the best learner
#   - learner phase compares candidates with peers
#
# Budget:
#   2 initial + 2 teacher + 2 learner = 6 evaluations
# =====================================================

print("\n" + "=" * 80)
print("TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)")
print("=" * 80)

tlbo_rng = np.random.default_rng(
    RANDOM_STATE + 3
)

TLBO_KEYS = [
    "hidden_layer_sizes",
    "activation",
    "learning_rate_init",
    "alpha",
    "batch_size",
]


def params_to_vector(params):
    """Map categorical/discrete hyperparameters to index positions."""

    vector = []

    for key in TLBO_KEYS:

        options = SEARCH_SPACE[key]

        current_position = 0

        for position, option in enumerate(options):

            if option == params[key]:
                current_position = position
                break

        vector.append(current_position)

    return np.array(
        vector,
        dtype=float,
    )


def vector_to_params(vector):
    """Map continuous TLBO positions back to valid discrete values."""

    params = {}

    for index, key in enumerate(TLBO_KEYS):

        options = SEARCH_SPACE[key]

        rounded_index = int(
            round(vector[index])
        )

        rounded_index = max(
            0,
            min(
                rounded_index,
                len(options) - 1,
            ),
        )

        params[key] = options[
            rounded_index
        ]

    return params


# -----------------------------------------------------
# Create initial population
# -----------------------------------------------------

tlbo_population = []
tlbo_seen = set()
tlbo_initial_rng = random.Random(
    RANDOM_STATE + 100
)

while len(tlbo_population) < TLBO_POPULATION_SIZE:

    params = sample_random_configuration(
        tlbo_initial_rng
    )

    signature = configuration_signature(
        params
    )

    if signature in tlbo_seen:
        continue

    tlbo_seen.add(signature)
    tlbo_population.append(params)

# -----------------------------------------------------
# Initial evaluations
# -----------------------------------------------------

tlbo_population_results = []
tlbo_convergence = []
tlbo_evaluations = 0
tlbo_best_score = float("-inf")
tlbo_best_params = None

tlbo_start = time.time()

for params in tlbo_population:

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    tlbo_population_results.append(run)
    tlbo_evaluations += 1

    if run["validation_macro_f1"] > tlbo_best_score:
        tlbo_best_score = run["validation_macro_f1"]
        tlbo_best_params = run["params"].copy()

tlbo_convergence.append(
    {
        "Method": "TLBO",
        "Evaluation": tlbo_evaluations,
        "Cumulative Resource Units": tlbo_evaluations * MAX_ITER,
        "Best Validation Macro F1": tlbo_best_score,
        "Generation": 0,
        "Phase": "Initial",
    }
)

# -----------------------------------------------------
# TLBO generation(s)
# -----------------------------------------------------

for generation in range(
    1,
    TLBO_GENERATIONS + 1,
):

    print(
        f"\nGeneration {generation}/{TLBO_GENERATIONS}"
    )

    # =============================================
    # TEACHER PHASE
    # =============================================

    teacher_result = max(
        tlbo_population_results,
        key=lambda x: x["validation_macro_f1"],
    )

    teacher_vector = params_to_vector(
        teacher_result["params"]
    )

    population_vectors = np.vstack(
        [
            params_to_vector(
                result["params"]
            )
            for result in tlbo_population_results
        ]
    )

    mean_vector = population_vectors.mean(
        axis=0
    )

    teacher_phase_results = []

    for current_result in list(
        tlbo_population_results
    ):

        current_vector = params_to_vector(
            current_result["params"]
        )

        teaching_factor = int(
            tlbo_rng.integers(
                1,
                3,
            )
        )

        random_vector = tlbo_rng.random(
            len(TLBO_KEYS)
        )

        new_vector = (
            current_vector
            + random_vector
            * (
                teacher_vector
                - teaching_factor * mean_vector
            )
        )

        new_params = vector_to_params(
            new_vector
        )

        run = evaluate_configuration(
            new_params,
            MAX_ITER,
        )

        teacher_phase_results.append(run)
        tlbo_evaluations += 1

        if run["validation_macro_f1"] > tlbo_best_score:
            tlbo_best_score = run["validation_macro_f1"]
            tlbo_best_params = run["params"].copy()

        tlbo_convergence.append(
            {
                "Method": "TLBO",
                "Evaluation": tlbo_evaluations,
                "Cumulative Resource Units": tlbo_evaluations * MAX_ITER,
                "Best Validation Macro F1": tlbo_best_score,
                "Generation": generation,
                "Phase": "Teacher",
            }
        )

    # Replacement occurs only if the teacher-phase candidate
    # improves the corresponding current learner.
    for index, candidate_run in enumerate(
        teacher_phase_results
    ):

        if (
            candidate_run["validation_macro_f1"]
            > tlbo_population_results[index]["validation_macro_f1"]
        ):
            tlbo_population_results[index] = (
                candidate_run
            )

    # =============================================
    # LEARNER PHASE
    # =============================================

    learner_phase_results = []

    for learner_index in range(
        TLBO_POPULATION_SIZE
    ):

        current_result = (
            tlbo_population_results[
                learner_index
            ]
        )

        current_vector = params_to_vector(
            current_result["params"]
        )

        partner_index = int(
            tlbo_rng.integers(
                0,
                TLBO_POPULATION_SIZE,
            )
        )

        while partner_index == learner_index:
            partner_index = int(
                tlbo_rng.integers(
                    0,
                    TLBO_POPULATION_SIZE,
                )
            )

        partner_result = (
            tlbo_population_results[
                partner_index
            ]
        )

        partner_vector = params_to_vector(
            partner_result["params"]
        )

        random_vector = tlbo_rng.random(
            len(TLBO_KEYS)
        )

        if (
            partner_result["validation_macro_f1"]
            > current_result["validation_macro_f1"]
        ):

            new_vector = (
                current_vector
                + random_vector
                * (
                    partner_vector
                    - current_vector
                )
            )

        else:

            new_vector = (
                current_vector
                + random_vector
                * (
                    current_vector
                    - partner_vector
                )
            )

        new_params = vector_to_params(
            new_vector
        )

        run = evaluate_configuration(
            new_params,
            MAX_ITER,
        )

        learner_phase_results.append(run)
        tlbo_evaluations += 1

        if run["validation_macro_f1"] > tlbo_best_score:
            tlbo_best_score = run["validation_macro_f1"]
            tlbo_best_params = run["params"].copy()

        tlbo_convergence.append(
            {
                "Method": "TLBO",
                "Evaluation": tlbo_evaluations,
                "Cumulative Resource Units": tlbo_evaluations * MAX_ITER,
                "Best Validation Macro F1": tlbo_best_score,
                "Generation": generation,
                "Phase": "Learner",
            }
        )

        if (
            run["validation_macro_f1"]
            > tlbo_population_results[learner_index]["validation_macro_f1"]
        ):
            tlbo_population_results[learner_index] = run

print("\nTLBO Optimization completed.")
tlbo_runtime = time.time() - tlbo_start

print(
    f"Optimization Time: {tlbo_runtime:.2f} seconds"
)
print(
    f"Evaluations: {tlbo_evaluations}"
)
print(
    f"Best Validation Macro F1: {tlbo_best_score:.4f}"
)

print("\nBest Hyperparameters Found By TLBO")
for key, value in tlbo_best_params.items():
    print(f"{key}: {value}")


# =====================================================
# SIMULATED ANNEALING (SA)
# =====================================================
#
# SA explores a single current hyperparameter configuration.
# Each new neighbour changes exactly one hyperparameter.
# Better configurations are always accepted.
# Worse configurations may be accepted according to the
# temperature-based acceptance probability.
# =====================================================

print("\n" + "=" * 80)
print("SIMULATED ANNEALING")
print("=" * 80)


def sa_random_choice(
    options,
    rng,
):
    """Select one option using an index to safely handle tuple choices."""

    index = int(
        rng.integers(
            0,
            len(options),
        )
    )

    return options[index]


def create_sa_solution(rng):
    """Create a random starting solution."""

    return {
        parameter: sa_random_choice(
            values,
            rng,
        )
        for parameter, values in SEARCH_SPACE.items()
    }


def create_sa_neighbour(
    current_solution,
    rng,
):
    """Change one hyperparameter to create a neighbouring solution."""

    neighbour = current_solution.copy()

    parameter = sa_random_choice(
        list(SEARCH_SPACE.keys()),
        rng,
    )

    current_value = neighbour[
        parameter
    ]

    alternatives = [
        value
        for value in SEARCH_SPACE[parameter]
        if value != current_value
    ]

    if alternatives:
        neighbour[parameter] = sa_random_choice(
            alternatives,
            rng,
        )

    return neighbour


def evaluate_sa_candidate(params):
    """Evaluate SA candidates using the SAME fixed validation protocol."""

    return evaluate_configuration(
        params,
        MAX_ITER,
    )


sa_rng = np.random.default_rng(
    RANDOM_STATE + 4
)

sa_start = time.time()
sa_convergence = []

sa_current_solution = create_sa_solution(
    sa_rng
)

sa_current_run = evaluate_sa_candidate(
    sa_current_solution
)

sa_current_score = (
    sa_current_run["validation_macro_f1"]
)

sa_best_solution = sa_current_solution.copy()
sa_best_score = sa_current_score

sa_convergence.append(
    {
        "Method": "Simulated Annealing",
        "Evaluation": 1,
        "Cumulative Resource Units": MAX_ITER,
        "Best Validation Macro F1": sa_best_score,
        "Current Macro F1": sa_current_score,
        "Temperature": SA_INITIAL_TEMPERATURE,
    }
)

print(
    f"Evaluation 01/{SA_EVALUATIONS} "
    f"- Macro F1 = {sa_current_score:.4f}"
)

for evaluation in range(
    2,
    SA_EVALUATIONS + 1,
):

    temperature = (
        SA_INITIAL_TEMPERATURE
        * (
            SA_COOLING_RATE
            ** (evaluation - 2)
        )
    )

    temperature = max(
        temperature,
        1e-12,
    )

    neighbour_solution = create_sa_neighbour(
        sa_current_solution,
        sa_rng,
    )

    neighbour_run = evaluate_sa_candidate(
        neighbour_solution
    )

    neighbour_score = (
        neighbour_run["validation_macro_f1"]
    )

    delta = (
        neighbour_score
        - sa_current_score
    )

    if delta >= 0:
        accept_solution = True
    else:
        acceptance_probability = math.exp(
            delta / temperature
        )
        accept_solution = (
            sa_rng.random()
            < acceptance_probability
        )

    if accept_solution:
        sa_current_solution = (
            neighbour_solution.copy()
        )
        sa_current_score = neighbour_score

    if neighbour_score > sa_best_score:
        sa_best_score = neighbour_score
        sa_best_solution = (
            neighbour_solution.copy()
        )

    sa_convergence.append(
        {
            "Method": "Simulated Annealing",
            "Evaluation": evaluation,
            "Cumulative Resource Units": evaluation * MAX_ITER,
            "Best Validation Macro F1": sa_best_score,
            "Current Macro F1": sa_current_score,
            "Temperature": temperature,
        }
    )

    print(
        f"Evaluation {evaluation:02d}/{SA_EVALUATIONS} "
        f"- Current Macro F1 = {sa_current_score:.4f}, "
        f"Best Macro F1 = {sa_best_score:.4f}"
    )

sa_optimization_time = (
    time.time() - sa_start
)

sa_evaluations = SA_EVALUATIONS

print("\nSimulated Annealing completed.")
print(
    f"Optimization Time: {sa_optimization_time:.2f} seconds"
)
print(
    f"Best Validation Macro F1: {sa_best_score:.4f}"
)

print("\nBest Hyperparameters Found By Simulated Annealing")
for key, value in sa_best_solution.items():
    print(f"{key}: {value}")


# =====================================================
# COMPUTATIONAL BUDGET CHECK
# =====================================================
#
# This check MUST pass before any final test evaluation.
# =====================================================

print("\n" + "=" * 80)
print("COMPUTATIONAL BUDGET CHECK")
print("=" * 80)

budget_results = pd.DataFrame(
    {
        "Method": [
            "Random Search",
            "Bayesian Optimization",
            "Hyperband",
            "LEA",
            "TLBO",
            "Simulated Annealing",
        ],
        "Evaluations": [
            random_evaluations,
            bayesian_evaluations,
            hyperband_evaluations,
            lea_evaluations,
            tlbo_evaluations,
            sa_evaluations,
        ],
        "Resource Units": [
            random_evaluations * MAX_ITER,
            bayesian_evaluations * MAX_ITER,
            hyperband_resource_units,
            lea_evaluations * MAX_ITER,
            tlbo_evaluations * MAX_ITER,
            sa_evaluations * MAX_ITER,
        ],
    }
)

print(budget_results)

assert all(
    budget_results["Resource Units"]
    == TOTAL_RESOURCE_BUDGET
), (
    "ERROR: Optimizers do not have equal computational budgets."
)

print(
    f"\nEqual computational budget confirmed: "
    f"{TOTAL_RESOURCE_BUDGET} resource units per optimizer."
)


# =====================================================
# FINAL DEVELOPMENT PREPROCESSING
# =====================================================
#
# The final models are trained on Training + Validation.
# A NEW imputer and scaler are fitted on the complete
# development data only. The test set is only transformed.
# =====================================================

print("\n" + "=" * 80)
print("FINAL DEVELOPMENT PREPROCESSING")
print("=" * 80)

X_development_raw = pd.concat(
    [
        X_train_raw,
        X_validation_raw,
    ],
    ignore_index=True,
)

y_development = pd.concat(
    [
        y_train_raw.reset_index(drop=True),
        y_validation_raw.reset_index(drop=True),
    ],
    ignore_index=True,
)

final_imputer = SimpleImputer(
    strategy="median"
)

X_development_imputed = final_imputer.fit_transform(
    X_development_raw
)

X_test_imputed_final = final_imputer.transform(
    X_test_raw
)

final_scaler = StandardScaler()

X_development_final = final_scaler.fit_transform(
    X_development_imputed
)

X_test_final = final_scaler.transform(
    X_test_imputed_final
)

print("Development Feature Shape:")
print(X_development_final.shape)

print("Final Test Feature Shape:")
print(X_test_final.shape)


# =====================================================
# PREPARE RESULTS DIRECTORY
# =====================================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# =====================================================
# FINAL MODEL TRAINING
# =====================================================
#
# All selected optimizer configurations now receive the same
# final training protocol:
#
#     Development data = Train + Validation
#     Test data = untouched until this stage
# =====================================================

final_models = {}
final_training_times = {}
final_test_metrics = {}


# -----------------------------------------------------
# Baseline
# -----------------------------------------------------

baseline_final_model, baseline_final_training_time = train_final_model(
    baseline_params,
    "Baseline MLP",
)

baseline_test_metrics = evaluate_test_model(
    baseline_final_model,
    "Baseline MLP",
)

final_models["Baseline MLP"] = baseline_final_model
final_training_times["Baseline MLP"] = baseline_final_training_time
final_test_metrics["Baseline MLP"] = baseline_test_metrics

save_training_loss_plot(
    baseline_final_model,
    "Baseline MLP",
)


# -----------------------------------------------------
# Random Search
# -----------------------------------------------------

random_final_model, random_final_training_time = train_final_model(
    random_best_params,
    "Random Search",
)

random_test_metrics = evaluate_test_model(
    random_final_model,
    "Random Search",
)

final_models["Random Search"] = random_final_model
final_training_times["Random Search"] = random_final_training_time
final_test_metrics["Random Search"] = random_test_metrics

save_training_loss_plot(
    random_final_model,
    "Random Search",
)


# -----------------------------------------------------
# Bayesian Optimization
# -----------------------------------------------------

bayesian_final_model, bayesian_final_training_time = train_final_model(
    bayesian_best_params,
    "Bayesian Optimization",
)

bayesian_test_metrics = evaluate_test_model(
    bayesian_final_model,
    "Bayesian Optimization",
)

final_models["Bayesian Optimization"] = bayesian_final_model
final_training_times["Bayesian Optimization"] = bayesian_final_training_time
final_test_metrics["Bayesian Optimization"] = bayesian_test_metrics

save_training_loss_plot(
    bayesian_final_model,
    "Bayesian Optimization",
)


# -----------------------------------------------------
# Hyperband
# -----------------------------------------------------

hyperband_final_model, hyperband_final_training_time = train_final_model(
    hyperband_best_params,
    "Hyperband",
)

hyperband_test_metrics = evaluate_test_model(
    hyperband_final_model,
    "Hyperband",
)

final_models["Hyperband"] = hyperband_final_model
final_training_times["Hyperband"] = hyperband_final_training_time
final_test_metrics["Hyperband"] = hyperband_test_metrics

save_training_loss_plot(
    hyperband_final_model,
    "Hyperband",
)


# -----------------------------------------------------
# LEA
# -----------------------------------------------------

lea_final_model, lea_final_training_time = train_final_model(
    lea_best_params,
    "LEA",
)

lea_test_metrics = evaluate_test_model(
    lea_final_model,
    "LEA",
)

final_models["LEA"] = lea_final_model
final_training_times["LEA"] = lea_final_training_time
final_test_metrics["LEA"] = lea_test_metrics

save_training_loss_plot(
    lea_final_model,
    "LEA",
)


# -----------------------------------------------------
# TLBO
# -----------------------------------------------------

tlbo_final_model, tlbo_final_training_time = train_final_model(
    tlbo_best_params,
    "TLBO",
)

tlbo_test_metrics = evaluate_test_model(
    tlbo_final_model,
    "TLBO",
)

final_models["TLBO"] = tlbo_final_model
final_training_times["TLBO"] = tlbo_final_training_time
final_test_metrics["TLBO"] = tlbo_test_metrics

save_training_loss_plot(
    tlbo_final_model,
    "TLBO",
)


# -----------------------------------------------------
# Simulated Annealing
# -----------------------------------------------------

sa_final_model, sa_final_training_time = train_final_model(
    sa_best_solution,
    "Simulated Annealing",
)

sa_test_metrics = evaluate_test_model(
    sa_final_model,
    "Simulated Annealing",
)

final_models["Simulated Annealing"] = sa_final_model
final_training_times["Simulated Annealing"] = sa_final_training_time
final_test_metrics["Simulated Annealing"] = sa_test_metrics

save_training_loss_plot(
    sa_final_model,
    "Simulated Annealing",
)


# =====================================================
# BUILD FINAL COMPARISON TABLE
# =====================================================

methods = [
    "Baseline MLP",
    "Random Search",
    "Bayesian Optimization",
    "Hyperband",
    "LEA",
    "TLBO",
    "Simulated Annealing",
]

validation_scores = {
    "Baseline MLP": baseline_validation_metrics["macro_f1"],
    "Random Search": random_best_score,
    "Bayesian Optimization": bayesian_best_score,
    "Hyperband": hyperband_best_score,
    "LEA": lea_best_score,
    "TLBO": tlbo_best_score,
    "Simulated Annealing": sa_best_score,
}

best_params_by_method = {
    "Baseline MLP": baseline_params,
    "Random Search": random_best_params,
    "Bayesian Optimization": bayesian_best_params,
    "Hyperband": hyperband_best_params,
    "LEA": lea_best_params,
    "TLBO": tlbo_best_params,
    "Simulated Annealing": sa_best_solution,
}

optimizer_evaluations = {
    "Baseline MLP": 0,
    "Random Search": random_evaluations,
    "Bayesian Optimization": bayesian_evaluations,
    "Hyperband": hyperband_evaluations,
    "LEA": lea_evaluations,
    "TLBO": tlbo_evaluations,
    "Simulated Annealing": sa_evaluations,
}

optimizer_resource_units = {
    "Baseline MLP": 0,
    "Random Search": random_evaluations * MAX_ITER,
    "Bayesian Optimization": bayesian_evaluations * MAX_ITER,
    "Hyperband": hyperband_resource_units,
    "LEA": lea_evaluations * MAX_ITER,
    "TLBO": tlbo_evaluations * MAX_ITER,
    "Simulated Annealing": sa_evaluations * MAX_ITER,
}

optimization_times = {
    "Baseline MLP": 0.0,
    "Random Search": random_search_runtime,
    "Bayesian Optimization": bayesian_runtime,
    "Hyperband": hyperband_runtime,
    "LEA": lea_runtime,
    "TLBO": tlbo_runtime,
    "Simulated Annealing": sa_optimization_time,
}

comparison_rows = []

baseline_test_macro_f1 = (
    final_test_metrics["Baseline MLP"]["macro_f1"]
)

for method in methods:

    params = best_params_by_method[method]
    test_metrics = final_test_metrics[method]
    test_macro_f1 = test_metrics["macro_f1"]

    if baseline_test_macro_f1 != 0:
        test_macro_f1_improvement = (
            (test_macro_f1 - baseline_test_macro_f1)
            / baseline_test_macro_f1
        ) * 100
    else:
        test_macro_f1_improvement = np.nan

    comparison_rows.append(
        {
            "Method": method,
            "Validation Macro F1": validation_scores[method],
            "Test Accuracy": test_metrics["accuracy"],
            "Test Macro Precision": test_metrics["macro_precision"],
            "Test Macro Recall": test_metrics["macro_recall"],
            "Test Macro F1": test_macro_f1,
            "Test Weighted F1": test_metrics["weighted_f1"],
            "Test Balanced Accuracy": test_metrics["balanced_accuracy"],
            "Test Potable Recall": test_metrics["potable_recall"],
            "Test Macro F1 Improvement (%)": test_macro_f1_improvement,
            "Optimization Time (s)": optimization_times[method],
            "Final Training Time (s)": final_training_times[method],
            "Optimization Evaluations": optimizer_evaluations[method],
            "Optimization Resource Units": optimizer_resource_units[method],
            "Final Training Iterations": final_models[method].n_iter_,
            "Final Training Loss": final_models[method].loss_curve_[-1],
            "Hidden Layers": params["hidden_layer_sizes"],
            "Activation": params["activation"],
            "Learning Rate": params["learning_rate_init"],
            "Alpha": params["alpha"],
            "Batch Size": params["batch_size"],
        }
    )

comparison_results = pd.DataFrame(
    comparison_rows
)


# =====================================================
# PRINT COMPARISON RESULTS
# =====================================================

print("\n" + "=" * 80)
print("COMPARISON OF OPTIMIZATION METHODS")
print("=" * 80)

print("\nComparison Results:")
print(comparison_results)


# =====================================================
# CONVERGENCE DATASET
# =====================================================

convergence_frames = []

convergence_frames.extend(
    random_convergence
)

convergence_frames.extend(
    bayesian_convergence
)

convergence_frames.extend(
    hyperband_convergence
)

convergence_frames.extend(
    lea_convergence
)

convergence_frames.extend(
    tlbo_convergence
)

convergence_frames.extend(
    sa_convergence
)

convergence_df = pd.DataFrame(
    convergence_frames
)


# =====================================================
# SAVE RESULTS
# =====================================================

results_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_results.csv",
)

comparison_results.to_csv(
    results_file,
    index=False,
)

convergence_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_convergence.csv",
)

convergence_df.to_csv(
    convergence_file,
    index=False,
)


# =====================================================
# SAVE SA CONVERGENCE SEPARATELY
# =====================================================

sa_convergence_df = pd.DataFrame(
    sa_convergence
)

sa_convergence_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_sa_convergence.csv",
)

sa_convergence_df.to_csv(
    sa_convergence_file,
    index=False,
)


# =====================================================
# SAVE COMPUTATIONAL BUDGET TABLE
# =====================================================

budget_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_check.csv",
)

budget_results.to_csv(
    budget_file,
    index=False,
)


# =====================================================
# PLOT CONVERGENCE VS RESOURCE UNITS
# =====================================================
#
# Cumulative resource units are used rather than raw
# evaluation number because Hyperband allocates different
# iteration budgets to different stages.
# =====================================================

plt.figure(
    figsize=(10, 6)
)

for method in [
    "Random Search",
    "Bayesian Optimization",
    "Hyperband",
    "LEA",
    "TLBO",
    "Simulated Annealing",
]:

    method_data = convergence_df[
        convergence_df["Method"] == method
    ]

    if method_data.empty:
        continue

    method_data = method_data.sort_values(
        "Cumulative Resource Units"
    )

    plt.plot(
        method_data["Cumulative Resource Units"],
        method_data["Best Validation Macro F1"],
        marker="o",
        label=method,
    )

plt.xlabel("Cumulative Resource Units")
plt.ylabel("Best Validation Macro F1")
plt.title(
    "Water Quality Optimizer Convergence Under Equal Resource Budget"
)
plt.grid(True)
plt.legend()
plt.tight_layout()

convergence_plot_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_convergence.png",
)

plt.savefig(
    convergence_plot_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================
# PLOT SA TEMPERATURE
# =====================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    sa_convergence_df["Evaluation"],
    sa_convergence_df["Temperature"],
    marker="o",
)

plt.xlabel("Evaluation")
plt.ylabel("Temperature")
plt.title(
    "Simulated Annealing Cooling Schedule"
)
plt.grid(True)
plt.tight_layout()

sa_temperature_plot_file = os.path.join(
    RESULTS_DIR,
    "water_quality_phase3_equal_budget_sa_temperature.png",
)

plt.savefig(
    sa_temperature_plot_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# =====================================================
# SELECT METHOD USING VALIDATION DATA
# =====================================================
#
# The selected method is determined ONLY by the predefined
# validation criterion: highest Validation Macro F1.
#
# The test set is NOT used to choose the method.
# =====================================================

best_validation_row = comparison_results.loc[
    comparison_results["Validation Macro F1"].idxmax()
]

best_validation_method = (
    best_validation_row["Method"]
)

best_validation_score = (
    best_validation_row["Validation Macro F1"]
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("WATER QUALITY PHASE 3 OPTIMIZATION COMPLETED")
print("=" * 80)

print("\nPrimary Optimization Metric:")
print("Validation Macro F1")
print("Higher values indicate better classification performance.")

print("\nOptimizer Budget:")
print(
    f"{TOTAL_RESOURCE_BUDGET} training-iteration resource units per optimizer."
)

print("\nValidation-selected Method:")
print(best_validation_method)

print("Validation Macro F1:")
print(
    f"{best_validation_score:.4f}"
)

print("\nUntouched Test Set:")
print(
    "Test metrics are reported descriptively and were not used "
    "for optimizer selection."
)

print("\nOptimization Methods:")
for method in methods[1:]:
    print(method)

print("\nResults saved to:")
print(results_file)

print("\nConvergence history saved to:")
print(convergence_file)

print("\nSA convergence history saved to:")
print(sa_convergence_file)

print("\nBudget check saved to:")
print(budget_file)

print("\nConvergence plot saved to:")
print(convergence_plot_file)

print("\nSA temperature plot saved to:")
print(sa_temperature_plot_file)

print("\n" + "=" * 80)
