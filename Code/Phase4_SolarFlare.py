# ============================================================
# PHASE 4 - SOLAR FLARE OPTIMIZER COMPARISON
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# Final controlled comparison of hyperparameter optimization
# algorithms for Solar Flare classification.
#
# The experimental design combines the useful methodological
# elements established in the earlier Solar Flare experiments:
#
# 1. Four-class Solar Flare classification.
# 2. Stratified 80/20 train-test split.
# 3. Five-fold Stratified CV on the training data.
# 4. Controlled SMOTENC performed INSIDE each CV training fold.
# 5. One common MLP hyperparameter search space.
# 6. Macro F1 as the primary optimization metric.
# 7. Equal computational resource budget for every optimizer.
# 8. Untouched test set used only after optimization.
#
# OPTIMIZERS COMPARED
# ------------------------------------------------------------
# 1. Random Search
# 2. Bayesian Optimization (Optuna/TPE)
# 3. Hyperband-style Successive Halving
# 4. LEA
# 5. TLBO
# 6. Simulated Annealing
# 7. Budget-limited Differential Evolution
#
# ============================================================

import math
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from imblearn.over_sampling import SMOTENC

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "processed_datasets" / "solar_flare_processed.csv"
TARGET_COLUMN = "Target"

TEST_SIZE = 0.20
N_SPLITS = 5

# ------------------------------------------------------------
# Equal computational budget
# ------------------------------------------------------------
#
# One full candidate evaluation at MAX_ITER costs:
#
#     5 CV folds x 500 MLP iterations
#     = 2,500 resource units
#
# Every optimizer receives 15,000 resource units.
#
# Six full evaluations therefore use:
#
#     6 x 5 x 500 = 15,000
#
# Hyperband uses:
#
#     8 x 5 x 200
#     3 x 5 x 300
#     1 x 5 x 500
#     = 15,000
#
# ------------------------------------------------------------

MAX_ITER = 500

N_OPTIMIZER_EVALUATIONS = 6

TOTAL_RESOURCE_BUDGET = (
    N_OPTIMIZER_EVALUATIONS
    * N_SPLITS
    * MAX_ITER
)

# ------------------------------------------------------------
# Hyperband budget
# ------------------------------------------------------------

HYPERBAND_INITIAL_CANDIDATES = 8
HYPERBAND_SURVIVORS_STAGE_1 = 3
HYPERBAND_SURVIVORS_STAGE_2 = 1
HYPERBAND_RESOURCES = [200, 300, 500]

# ------------------------------------------------------------
# LEA budget
# ------------------------------------------------------------

LEA_POPULATION_SIZE = 3
LEA_SURVIVOR_COUNT = 2
LEA_GENERATIONS = 1

# ------------------------------------------------------------
# TLBO budget
# ------------------------------------------------------------

TLBO_POPULATION_SIZE = 2
TLBO_GENERATIONS = 1

# ------------------------------------------------------------
# Simulated Annealing budget
# ------------------------------------------------------------

SA_EVALUATIONS = 6
SA_INITIAL_TEMPERATURE = 0.03
SA_COOLING_RATE = 0.90

# ------------------------------------------------------------
# Differential Evolution budget
# ------------------------------------------------------------
#
# A compact best/1/bin-style DE is used because a conventional
# large DE population would consume far more resources than the
# other optimizers. With population 3 and one generation:
#
#     3 initial candidates
#     3 trial candidates
#     = 6 evaluations
#
# This is explicitly a budget-limited DE implementation.
# ------------------------------------------------------------

DE_POPULATION_SIZE = 3
DE_GENERATIONS = 1
DE_MUTATION_FACTOR = 0.80
DE_CROSSOVER_RATE = 0.70

# ------------------------------------------------------------
# Controlled SMOTENC strategy from the earlier Solar Flare
# imbalance experiments.
# ------------------------------------------------------------

CONTROLLED_TARGETS = {
    0: 400,  # Common Flare
    1: 250,  # Moderate Flare
    3: 100,  # Severe Flare
}

CONTROLLED_SMOTENC_K_NEIGHBORS = 3

# ------------------------------------------------------------
# Solar Flare categorical features
# ------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "modified Zurich class",
    "largest spot size",
    "spot distribution",
]

# ------------------------------------------------------------
# Label mapping used by earlier Solar Flare scripts.
# ------------------------------------------------------------

TARGET_MAPPING = {
    "Common Flare": 0,
    "Moderate Flare": 1,
    "No Flare": 2,
    "Severe Flare": 3,
}

CLASS_NAMES = [
    "Common Flare",
    "Moderate Flare",
    "No Flare",
    "Severe Flare",
]

# ------------------------------------------------------------
# One common MLP hyperparameter search space.
# max_iter is NOT a searchable hyperparameter. It is the
# controlled computational resource.
# ------------------------------------------------------------

SEARCH_SPACE = {
    "hidden_layer_sizes": [
        "16",
        "32",
        "64",
        "128",
        "32_16",
        "64_32",
        "128_64",
    ],
    "activation": [
        "relu",
        "tanh",
    ],
    "learning_rate_init": [
        0.0001,
        0.0005,
        0.001,
        0.005,
    ],
    "alpha": [
        0.00001,
        0.0001,
        0.001,
        0.01,
    ],
    "batch_size": [
        16,
        32,
        64,
    ],
}

HIDDEN_LAYER_MAP = {
    "16": (16,),
    "32": (32,),
    "64": (64,),
    "128": (128,),
    "32_16": (32, 16),
    "64_32": (64, 32),
    "128_64": (128, 64),
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# WARNINGS / OPTUNA LOGGING
# ============================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# HELPER: RANDOM CONFIGURATION
# ============================================================

def sample_random_configuration(rng):
    """Sample one valid configuration from the common search space."""
    return {
        key: rng.choice(values)
        for key, values in SEARCH_SPACE.items()
    }


def configuration_signature(params):
    """Create a hashable signature for a configuration."""
    return (
        params["hidden_layer_sizes"],
        params["activation"],
        params["learning_rate_init"],
        params["alpha"],
        params["batch_size"],
    )


def mutate_configuration(params, rng, n_changes=1):
    """Change one or more hyperparameters using the common search space."""
    new_params = params.copy()

    keys = list(SEARCH_SPACE.keys())
    n_changes = max(1, min(n_changes, len(keys)))

    changed_keys = rng.sample(keys, n_changes)

    for key in changed_keys:
        alternatives = [
            value
            for value in SEARCH_SPACE[key]
            if value != new_params[key]
        ]

        if alternatives:
            new_params[key] = rng.choice(alternatives)

    return new_params


def ensure_unique_configuration(params, seen, rng, max_attempts=100):
    """Return a configuration not already present in seen when possible."""
    candidate = params.copy()

    for _ in range(max_attempts):
        signature = configuration_signature(candidate)

        if signature not in seen:
            seen.add(signature)
            return candidate

        candidate = mutate_configuration(
            candidate,
            rng,
            n_changes=1,
        )

    # As a final fallback, generate a fresh random configuration.
    for _ in range(max_attempts):
        candidate = sample_random_configuration(rng)
        signature = configuration_signature(candidate)
        if signature not in seen:
            seen.add(signature)
            return candidate

    # The search space is large enough that reaching here is extremely unlikely.
    return candidate


# ============================================================
# HELPER: MODEL CREATION
# ============================================================

def build_mlp(params, max_iter=MAX_ITER):
    """Build the MLP with the controlled resource limit."""
    hidden_layers = HIDDEN_LAYER_MAP[params["hidden_layer_sizes"]]

    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
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


# ============================================================
# HELPER: FOLD PREPROCESSING + CONTROLLED SMOTENC
# ============================================================

def prepare_cv_fold(
    X_train_fold,
    y_train_fold,
    X_validation_fold,
):
    """
    Fit all preprocessing using only the CV training fold.

    Sequence:
        categorical imputation
        numerical imputation
        ordinal encoding
        controlled SMOTENC
        one-hot encoding
        standardization

    The validation fold is never oversampled.
    """

    X_train_fold = X_train_fold.copy()
    X_validation_fold = X_validation_fold.copy()

    numerical_features = [
        column
        for column in X_train_fold.columns
        if column not in CATEGORICAL_FEATURES
    ]

    # --------------------------------------------------------
    # Categorical preprocessing before SMOTENC
    # --------------------------------------------------------

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_cat_train = categorical_imputer.fit_transform(
        X_train_fold[CATEGORICAL_FEATURES]
    )

    X_cat_validation = categorical_imputer.transform(
        X_validation_fold[CATEGORICAL_FEATURES]
    )

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_cat_train = ordinal_encoder.fit_transform(
        X_cat_train
    )

    X_cat_validation = ordinal_encoder.transform(
        X_cat_validation
    )

    # --------------------------------------------------------
    # Numerical preprocessing before SMOTENC
    # --------------------------------------------------------

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    X_num_train = numerical_imputer.fit_transform(
        X_train_fold[numerical_features]
    )

    X_num_validation = numerical_imputer.transform(
        X_validation_fold[numerical_features]
    )

    # --------------------------------------------------------
    # Put categorical variables first so SMOTENC indices are
    # exactly [0, 1, 2].
    # --------------------------------------------------------

    X_prepared_train = np.column_stack(
        [
            X_cat_train,
            X_num_train,
        ]
    )

    X_prepared_validation = np.column_stack(
        [
            X_cat_validation,
            X_num_validation,
        ]
    )

    # --------------------------------------------------------
    # Controlled SMOTENC
    # --------------------------------------------------------

    current_counts = (
        pd.Series(y_train_fold)
        .value_counts()
        .to_dict()
    )

    sampling_strategy = {}

    for class_id, desired_count in CONTROLLED_TARGETS.items():
        current_count = current_counts.get(class_id, 0)

        if (
            current_count > 0
            and current_count < desired_count
        ):
            sampling_strategy[class_id] = desired_count

    if sampling_strategy:
        eligible_counts = [
            current_counts[class_id]
            for class_id in sampling_strategy
            if current_counts.get(class_id, 0) > 1
        ]

        if eligible_counts:
            safe_k = min(
                CONTROLLED_SMOTENC_K_NEIGHBORS,
                min(eligible_counts) - 1,
            )
        else:
            safe_k = 0

        if safe_k >= 1:
            smote = SMOTENC(
                categorical_features=[0, 1, 2],
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE,
                k_neighbors=safe_k,
            )

            X_resampled, y_resampled = smote.fit_resample(
                X_prepared_train,
                y_train_fold,
            )
        else:
            X_resampled = X_prepared_train
            y_resampled = np.asarray(y_train_fold)
    else:
        X_resampled = X_prepared_train
        y_resampled = np.asarray(y_train_fold)

    # --------------------------------------------------------
    # One-hot encoding AFTER SMOTENC
    # --------------------------------------------------------

    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    X_cat_resampled = X_resampled[:, :3]
    X_num_resampled = X_resampled[:, 3:]

    X_cat_validation_after = X_prepared_validation[:, :3]
    X_num_validation_after = X_prepared_validation[:, 3:]

    X_cat_resampled = onehot_encoder.fit_transform(
        X_cat_resampled
    )

    X_cat_validation_after = onehot_encoder.transform(
        X_cat_validation_after
    )

    X_final_train = np.column_stack(
        [
            X_cat_resampled,
            X_num_resampled,
        ]
    )

    X_final_validation = np.column_stack(
        [
            X_cat_validation_after,
            X_num_validation_after,
        ]
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_final_train = scaler.fit_transform(
        X_final_train
    )

    X_final_validation = scaler.transform(
        X_final_validation
    )

    return (
        X_final_train,
        X_final_validation,
        np.asarray(y_resampled),
    )


# ============================================================
# HELPER: CROSS-VALIDATED CONFIGURATION EVALUATION
# ============================================================

def evaluate_configuration(
    params,
    max_iter=MAX_ITER,
):
    """
    Evaluate one hyperparameter configuration using the fixed
    five-fold stratified CV protocol.
    """

    evaluation_start = time.time()

    fold_scores = []
    fold_iterations = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_cv_train = X_train.iloc[train_indices].copy()
        X_cv_validation = X_train.iloc[validation_indices].copy()

        y_cv_train = y_train.iloc[train_indices].copy()
        y_cv_validation = y_train.iloc[validation_indices].copy()

        (
            X_cv_train_processed,
            X_cv_validation_processed,
            y_cv_train_processed,
        ) = prepare_cv_fold(
            X_cv_train,
            y_cv_train,
            X_cv_validation,
        )

        model = build_mlp(
            params,
            max_iter=max_iter,
        )

        model.fit(
            X_cv_train_processed,
            y_cv_train_processed,
        )

        predictions = model.predict(
            X_cv_validation_processed
        )

        score = f1_score(
            y_cv_validation,
            predictions,
            average="macro",
            zero_division=0,
        )

        fold_scores.append(float(score))
        fold_iterations.append(int(model.n_iter_))

    mean_macro_f1 = float(
        np.mean(fold_scores)
    )

    runtime = time.time() - evaluation_start

    resource_units = (
        N_SPLITS * max_iter
    )

    return {
        "params": params.copy(),
        "validation_macro_f1": mean_macro_f1,
        "fold_scores": fold_scores,
        "mean_iterations": float(np.mean(fold_iterations)),
        "max_iterations_observed": int(max(fold_iterations)),
        "runtime": float(runtime),
        "resource_units": int(resource_units),
        "fold_count": N_SPLITS,
    }


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("PHASE 4 - SOLAR FLARE EQUAL-BUDGET OPTIMIZER COMPARISON")
print("=" * 80)

print("\nDataset Path:")
print(DATA_PATH)

data = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(data.shape)

print("\nFirst Five Rows:")
print(data.head())


# ============================================================
# TARGET AND FEATURES
# ============================================================

X = data.drop(columns=[TARGET_COLUMN]).copy()

# Keep the original Solar Flare target as categorical labels.
# Numeric IDs are created only as an internal representation for
# SMOTENC and metric calculations; the task remains 4-class
# classification throughout the experiment.
raw_y = data[TARGET_COLUMN].astype(str).str.strip()

unknown_labels = sorted(
    set(raw_y.unique()) - set(TARGET_MAPPING.keys())
)

if unknown_labels:
    raise ValueError(
        "Unknown target class label(s) found: "
        f"{unknown_labels}. Expected labels: "
        f"{list(TARGET_MAPPING.keys())}"
    )

y = raw_y.map(TARGET_MAPPING).astype("int64")

print("\nSolar Flare Target Label Mapping:")
for label, class_id in TARGET_MAPPING.items():
    print(f"{label} -> {class_id}")

print("\nOriginal Target Distribution:")
print(raw_y.value_counts())

print("\nEncoded Target Distribution:")
print(y.value_counts().sort_index())

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

missing_categorical = [
    feature
    for feature in CATEGORICAL_FEATURES
    if feature not in X.columns
]

if missing_categorical:
    raise ValueError(
        "Missing categorical Solar Flare features: "
        + str(missing_categorical)
    )

print("\nCategorical Features:")
for feature in CATEGORICAL_FEATURES:
    print(feature)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
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
# STRATIFIED CV
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# PRINT EXPERIMENT DESIGN
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT DESIGN")
print("=" * 80)

print("\nPrimary Optimization Metric:")
print("5-Fold CV Macro F1")

print("\nControlled SMOTENC Targets:")
for class_id, target_count in CONTROLLED_TARGETS.items():
    print(f"Class {class_id}: {target_count}")

print("\nSMOTENC k-neighbors:")
print(CONTROLLED_SMOTENC_K_NEIGHBORS)

print("\nMLP Full Resource per Candidate:")
print(MAX_ITER)

print("\nCV Folds:")
print(N_SPLITS)

print("\nTotal Resource Budget per Optimizer:")
print(TOTAL_RESOURCE_BUDGET)


# ============================================================
# PHASE 4 BASELINE
# ============================================================
#
# The baseline retains the MLP architecture used in the earlier
# Solar Flare experiments, while using the Phase 4 common 500-
# iteration resource cap.
#
# Baseline is not counted against any optimizer budget.
# ============================================================

baseline_params = {
    "hidden_layer_sizes": "64",
    "activation": "relu",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": 32,
}

print("\n" + "=" * 80)
print("PHASE 4 CONTROLLED-SMOTENC BASELINE")
print("=" * 80)

baseline_cv_run = evaluate_configuration(
    baseline_params,
    MAX_ITER,
)

print(
    f"\nBaseline CV Macro F1: "
    f"{baseline_cv_run['validation_macro_f1']:.4f}"
)


# ============================================================
# RANDOM SEARCH
# ============================================================

print("\n" + "=" * 80)
print("RANDOM SEARCH")
print("=" * 80)

random_rng = random.Random(RANDOM_STATE)
random_search_runs = []
random_seen = set()
random_start = time.time()

for evaluation in range(
    1,
    N_OPTIMIZER_EVALUATIONS + 1,
):
    params = ensure_unique_configuration(
        sample_random_configuration(random_rng),
        random_seen,
        random_rng,
    )

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    random_search_runs.append(run)

    print(
        f"Evaluation {evaluation:02d}/"
        f"{N_OPTIMIZER_EVALUATIONS} "
        f"- CV Macro F1 = "
        f"{run['validation_macro_f1']:.4f}"
    )

random_optimization_time = time.time() - random_start

random_best_run = max(
    random_search_runs,
    key=lambda item: item["validation_macro_f1"],
)

random_best_params = random_best_run["params"]
random_best_cv_macro_f1 = random_best_run["validation_macro_f1"]
random_evaluations = len(random_search_runs)


print("\nRandom Search completed.")
print(
    f"Optimization Time: "
    f"{random_optimization_time:.2f} seconds"
)
print(
    f"Best CV Macro F1: "
    f"{random_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in random_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

print("\n" + "=" * 80)
print("BAYESIAN OPTIMIZATION")
print("=" * 80)

bayesian_runs = {}
bayesian_convergence = []

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    ),
    study_name="Solar_Flare_Phase4_Bayesian",
)


def bayesian_objective(trial):
    params = {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            SEARCH_SPACE["hidden_layer_sizes"],
        ),
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

    bayesian_runs[trial.number] = run

    best_so_far = max(
        item["validation_macro_f1"]
        for item in bayesian_runs.values()
    )

    bayesian_convergence.append(
        {
            "Evaluation": trial.number + 1,
            "Cumulative Resource Units": (
                (trial.number + 1) * N_SPLITS * MAX_ITER
            ),
            "Best Observed CV Macro F1": best_so_far,
        }
    )

    return run["validation_macro_f1"]


bayesian_start = time.time()

study.optimize(
    bayesian_objective,
    n_trials=N_OPTIMIZER_EVALUATIONS,
)

bayesian_optimization_time = time.time() - bayesian_start

best_trial = study.best_trial
bayesian_best_run = bayesian_runs[best_trial.number]
bayesian_best_params = bayesian_best_run["params"]
bayesian_best_cv_macro_f1 = bayesian_best_run["validation_macro_f1"]
bayesian_evaluations = len(bayesian_runs)

print("\nBayesian Optimization completed.")
print(
    f"Optimization Time: "
    f"{bayesian_optimization_time:.2f} seconds"
)
print(
    f"Best CV Macro F1: "
    f"{bayesian_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in bayesian_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# HYPERBAND-STYLE SUCCESSIVE HALVING
# ============================================================

print("\n" + "=" * 80)
print("HYPERBAND-STYLE SUCCESSIVE HALVING")
print("=" * 80)

hyperband_rng = random.Random(RANDOM_STATE + 1)
hyperband_seen = set()
hyperband_convergence = []
hyperband_evaluations = 0
hyperband_resource_units = 0
hyperband_start = time.time()

hyperband_candidates = []

while len(hyperband_candidates) < HYPERBAND_INITIAL_CANDIDATES:
    params = ensure_unique_configuration(
        sample_random_configuration(hyperband_rng),
        hyperband_seen,
        hyperband_rng,
    )
    hyperband_candidates.append(params)

hyperband_final_stage_runs = []

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

for stage_name, stage_resource, survivor_count in stage_definitions:
    print("\n" + "-" * 70)
    print(stage_name)
    print(f"Candidates: {len(hyperband_candidates)}")
    print(f"MLP Iterations per Candidate: {stage_resource}")
    print("-" * 70)

    stage_runs = []

    for params in hyperband_candidates:
        run = evaluate_configuration(
            params,
            stage_resource,
        )

        stage_runs.append(run)
        hyperband_evaluations += 1
        hyperband_resource_units += run["resource_units"]

        best_observed = max(
            item["validation_macro_f1"]
            for item in stage_runs
        )

        hyperband_convergence.append(
            {
                "Evaluation": hyperband_evaluations,
                "Cumulative Resource Units": hyperband_resource_units,
                "Best Observed CV Macro F1": best_observed,
            }
        )

        print(
            f"Evaluation {hyperband_evaluations:02d} "
            f"- CV Macro F1 = "
            f"{run['validation_macro_f1']:.4f}"
        )

    stage_runs = sorted(
        stage_runs,
        key=lambda item: item["validation_macro_f1"],
        reverse=True,
    )

    hyperband_final_stage_runs = stage_runs.copy()

    if survivor_count > 0:
        hyperband_candidates = [
            item["params"]
            for item in stage_runs[:survivor_count]
        ]

hyperband_optimization_time = time.time() - hyperband_start

# Final-stage selection only: the configuration that survived to
# the full resource stage is the Hyperband-selected configuration.
hyperband_best_run = max(
    hyperband_final_stage_runs,
    key=lambda item: item["validation_macro_f1"],
)

hyperband_best_params = hyperband_best_run["params"]
hyperband_best_cv_macro_f1 = hyperband_best_run["validation_macro_f1"]

print("\nHyperband completed.")
print(
    f"Optimization Time: "
    f"{hyperband_optimization_time:.2f} seconds"
)
print(
    f"Resource Units Used: "
    f"{hyperband_resource_units}"
)
print(
    f"Final-Stage CV Macro F1: "
    f"{hyperband_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in hyperband_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# LEARNING EVOLUTIONARY ALGORITHM (LEA)
# ============================================================

print("\n" + "=" * 80)
print("LEARNING EVOLUTIONARY ALGORITHM (LEA)")
print("=" * 80)

lea_rng = random.Random(RANDOM_STATE + 2)
lea_seen = set()
lea_convergence = []
lea_evaluations = 0
lea_resource_units = 0
lea_start = time.time()

lea_population = []

while len(lea_population) < LEA_POPULATION_SIZE:
    params = ensure_unique_configuration(
        sample_random_configuration(lea_rng),
        lea_seen,
        lea_rng,
    )
    lea_population.append(params)

lea_population_results = []

# Initial population: 3 evaluations.
for params in lea_population:
    run = evaluate_configuration(params, MAX_ITER)
    lea_population_results.append(run)
    lea_evaluations += 1
    lea_resource_units += run["resource_units"]

    current_best = max(
        item["validation_macro_f1"]
        for item in lea_population_results
    )

    lea_convergence.append(
        {
            "Evaluation": lea_evaluations,
            "Cumulative Resource Units": lea_resource_units,
            "Best Observed CV Macro F1": current_best,
        }
    )

# One generation: keep the top two, generate three offspring.
for generation in range(LEA_GENERATIONS):
    ranked = sorted(
        lea_population_results,
        key=lambda item: item["validation_macro_f1"],
        reverse=True,
    )

    survivors = ranked[:LEA_SURVIVOR_COUNT]

    offspring_results = []

    for _ in range(
        LEA_POPULATION_SIZE
    ):
        parent = lea_rng.choice(survivors)["params"]

        offspring = mutate_configuration(
            parent,
            lea_rng,
            n_changes=1,
        )

        offspring = ensure_unique_configuration(
            offspring,
            lea_seen,
            lea_rng,
        )

        run = evaluate_configuration(
            offspring,
            MAX_ITER,
        )

        offspring_results.append(run)
        lea_evaluations += 1
        lea_resource_units += run["resource_units"]

        current_best = max(
            item["validation_macro_f1"]
            for item in survivors + offspring_results
        )

        lea_convergence.append(
            {
                "Evaluation": lea_evaluations,
                "Cumulative Resource Units": lea_resource_units,
                "Best Observed CV Macro F1": current_best,
            }
        )

    lea_population_results = sorted(
        survivors + offspring_results,
        key=lambda item: item["validation_macro_f1"],
        reverse=True,
    )[:LEA_POPULATION_SIZE]

lea_optimization_time = time.time() - lea_start

lea_best_run = max(
    lea_population_results,
    key=lambda item: item["validation_macro_f1"],
)

lea_best_params = lea_best_run["params"]
lea_best_cv_macro_f1 = lea_best_run["validation_macro_f1"]

print("\nLEA completed.")
print(
    f"Optimization Time: "
    f"{lea_optimization_time:.2f} seconds"
)
print(f"Evaluations: {lea_evaluations}")
print(
    f"Best CV Macro F1: "
    f"{lea_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in lea_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# ============================================================

print("\n" + "=" * 80)
print("TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)")
print("=" * 80)

TLBO_KEYS = [
    "hidden_layer_sizes",
    "activation",
    "learning_rate_init",
    "alpha",
    "batch_size",
]


def params_to_vector(params):
    vector = []

    for key in TLBO_KEYS:
        options = SEARCH_SPACE[key]
        index = options.index(params[key])
        vector.append(float(index))

    return np.asarray(vector, dtype=float)


def vector_to_params(vector):
    params = {}

    for index, key in enumerate(TLBO_KEYS):
        options = SEARCH_SPACE[key]
        rounded_index = int(round(vector[index]))
        rounded_index = max(
            0,
            min(rounded_index, len(options) - 1),
        )
        params[key] = options[rounded_index]

    return params


tlbo_rng = np.random.default_rng(RANDOM_STATE + 3)
tlbo_seen = set()
tlbo_convergence = []
tlbo_evaluations = 0
tlbo_resource_units = 0
tlbo_start = time.time()

tlbo_population = []

while len(tlbo_population) < TLBO_POPULATION_SIZE:
    params = ensure_unique_configuration(
        sample_random_configuration(
            random.Random(
                int(tlbo_rng.integers(0, 2**31 - 1))
            )
        ),
        tlbo_seen,
        random.Random(
            int(tlbo_rng.integers(0, 2**31 - 1))
        ),
    )
    tlbo_population.append(params)

# Initial evaluations.
tlbo_population_results = []

for params in tlbo_population:
    run = evaluate_configuration(params, MAX_ITER)
    tlbo_population_results.append(run)
    tlbo_evaluations += 1
    tlbo_resource_units += run["resource_units"]

current_best = max(
    item["validation_macro_f1"]
    for item in tlbo_population_results
)

tlbo_convergence.append(
    {
        "Evaluation": tlbo_evaluations,
        "Cumulative Resource Units": tlbo_resource_units,
        "Best Observed CV Macro F1": current_best,
    }
)

for generation in range(TLBO_GENERATIONS):
    # --------------------------------------------------------
    # Teacher phase
    # --------------------------------------------------------
    teacher_result = max(
        tlbo_population_results,
        key=lambda item: item["validation_macro_f1"],
    )

    teacher_vector = params_to_vector(
        teacher_result["params"]
    )

    population_vectors = np.vstack(
        [
            params_to_vector(result["params"])
            for result in tlbo_population_results
        ]
    )

    mean_vector = population_vectors.mean(axis=0)

    teacher_phase_results = []

    for current_result in tlbo_population_results:
        current_vector = params_to_vector(
            current_result["params"]
        )

        teaching_factor = int(
            tlbo_rng.integers(1, 3)
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
        tlbo_resource_units += run["resource_units"]

    # Greedy teacher-phase replacement.
    updated_population = []

    for current_result, teacher_result_new in zip(
        tlbo_population_results,
        teacher_phase_results,
    ):
        if (
            teacher_result_new["validation_macro_f1"]
            >
            current_result["validation_macro_f1"]
        ):
            updated_population.append(teacher_result_new)
        else:
            updated_population.append(current_result)

    tlbo_population_results = updated_population

    current_best = max(
        item["validation_macro_f1"]
        for item in tlbo_population_results + teacher_phase_results
    )

    tlbo_convergence.append(
        {
            "Evaluation": tlbo_evaluations,
            "Cumulative Resource Units": tlbo_resource_units,
            "Best Observed CV Macro F1": current_best,
        }
    )

    # --------------------------------------------------------
    # Learner phase
    # --------------------------------------------------------
    learner_phase_results = []

    for learner_index in range(TLBO_POPULATION_SIZE):
        current_result = tlbo_population_results[learner_index]

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

        partner_result = tlbo_population_results[partner_index]

        current_vector = params_to_vector(
            current_result["params"]
        )

        partner_vector = params_to_vector(
            partner_result["params"]
        )

        random_vector = tlbo_rng.random(
            len(TLBO_KEYS)
        )

        if (
            partner_result["validation_macro_f1"]
            >
            current_result["validation_macro_f1"]
        ):
            new_vector = (
                current_vector
                + random_vector
                * (partner_vector - current_vector)
            )
        else:
            new_vector = (
                current_vector
                + random_vector
                * (current_vector - partner_vector)
            )

        learner_params = vector_to_params(
            new_vector
        )

        run = evaluate_configuration(
            learner_params,
            MAX_ITER,
        )

        learner_phase_results.append(run)
        tlbo_evaluations += 1
        tlbo_resource_units += run["resource_units"]

    updated_population = []

    for current_result, learner_result in zip(
        tlbo_population_results,
        learner_phase_results,
    ):
        if (
            learner_result["validation_macro_f1"]
            >
            current_result["validation_macro_f1"]
        ):
            updated_population.append(learner_result)
        else:
            updated_population.append(current_result)

    tlbo_population_results = updated_population

    current_best = max(
        item["validation_macro_f1"]
        for item in (
            tlbo_population_results
            + learner_phase_results
        )
    )

    tlbo_convergence.append(
        {
            "Evaluation": tlbo_evaluations,
            "Cumulative Resource Units": tlbo_resource_units,
            "Best Observed CV Macro F1": current_best,
        }
    )

tlbo_optimization_time = time.time() - tlbo_start

tlbo_best_run = max(
    tlbo_population_results,
    key=lambda item: item["validation_macro_f1"],
)

tlbo_best_params = tlbo_best_run["params"]
tlbo_best_cv_macro_f1 = tlbo_best_run["validation_macro_f1"]

print("\nTLBO completed.")
print(
    f"Optimization Time: "
    f"{tlbo_optimization_time:.2f} seconds"
)
print(f"Evaluations: {tlbo_evaluations}")
print(
    f"Best CV Macro F1: "
    f"{tlbo_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in tlbo_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# SIMULATED ANNEALING
# ============================================================

print("\n" + "=" * 80)
print("SIMULATED ANNEALING")
print("=" * 80)

sa_rng = random.Random(RANDOM_STATE + 4)
sa_seen = set()
sa_convergence = []
sa_evaluations = 0
sa_resource_units = 0
sa_start = time.time()

sa_current_params = ensure_unique_configuration(
    sample_random_configuration(sa_rng),
    sa_seen,
    sa_rng,
)

sa_current_run = evaluate_configuration(
    sa_current_params,
    MAX_ITER,
)

sa_evaluations = 1
sa_resource_units = sa_current_run["resource_units"]

sa_current_score = sa_current_run["validation_macro_f1"]
sa_best_score = sa_current_score
sa_best_params = sa_current_params.copy()

sa_convergence.append(
    {
        "Evaluation": 1,
        "Cumulative Resource Units": sa_resource_units,
        "Current CV Macro F1": sa_current_score,
        "Best Observed CV Macro F1": sa_best_score,
        "Temperature": SA_INITIAL_TEMPERATURE,
    }
)

print(
    f"Evaluation 01/{SA_EVALUATIONS} "
    f"- CV Macro F1 = {sa_current_score:.4f}"
)

for evaluation in range(
    2,
    SA_EVALUATIONS + 1,
):
    temperature = max(
        SA_INITIAL_TEMPERATURE
        * (SA_COOLING_RATE ** (evaluation - 2)),
        1e-12,
    )

    neighbour_params = ensure_unique_configuration(
        mutate_configuration(
            sa_current_params,
            sa_rng,
            n_changes=1,
        ),
        sa_seen,
        sa_rng,
    )

    neighbour_run = evaluate_configuration(
        neighbour_params,
        MAX_ITER,
    )

    neighbour_score = neighbour_run["validation_macro_f1"]

    sa_evaluations += 1
    sa_resource_units += neighbour_run["resource_units"]

    delta = neighbour_score - sa_current_score

    if delta >= 0:
        accept = True
    else:
        acceptance_probability = math.exp(
            delta / temperature
        )
        accept = (
            sa_rng.random()
            < acceptance_probability
        )

    if accept:
        sa_current_params = neighbour_params.copy()
        sa_current_score = neighbour_score

    if neighbour_score > sa_best_score:
        sa_best_score = neighbour_score
        sa_best_params = neighbour_params.copy()

    sa_convergence.append(
        {
            "Evaluation": evaluation,
            "Cumulative Resource Units": sa_resource_units,
            "Current CV Macro F1": sa_current_score,
            "Best Observed CV Macro F1": sa_best_score,
            "Temperature": temperature,
        }
    )

    print(
        f"Evaluation {evaluation:02d}/{SA_EVALUATIONS} "
        f"- Current CV Macro F1 = {sa_current_score:.4f}, "
        f"Best CV Macro F1 = {sa_best_score:.4f}"
    )

sa_optimization_time = time.time() - sa_start

print("\nSimulated Annealing completed.")
print(
    f"Optimization Time: "
    f"{sa_optimization_time:.2f} seconds"
)
print(
    f"Best CV Macro F1: "
    f"{sa_best_score:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in sa_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# DIFFERENTIAL EVOLUTION - BUDGET-LIMITED BEST/1/BIN
# ============================================================

print("\n" + "=" * 80)
print("DIFFERENTIAL EVOLUTION - BUDGET-LIMITED")
print("=" * 80)

DE_KEYS = [
    "hidden_layer_sizes",
    "activation",
    "learning_rate_init",
    "alpha",
    "batch_size",
]


def de_params_to_vector(params):
    return np.array(
        [
            SEARCH_SPACE["hidden_layer_sizes"].index(
                params["hidden_layer_sizes"]
            ),
            SEARCH_SPACE["activation"].index(
                params["activation"]
            ),
            SEARCH_SPACE["learning_rate_init"].index(
                params["learning_rate_init"]
            ),
            SEARCH_SPACE["alpha"].index(
                params["alpha"]
            ),
            SEARCH_SPACE["batch_size"].index(
                params["batch_size"]
            ),
        ],
        dtype=float,
    )


def de_vector_to_params(vector):
    params = {}

    for dimension, key in enumerate(DE_KEYS):
        options = SEARCH_SPACE[key]
        index = int(round(vector[dimension]))
        index = max(0, min(index, len(options) - 1))
        params[key] = options[index]

    return params


def decode_de_population_vector(vector):
    return de_vector_to_params(vector)


de_rng = np.random.default_rng(RANDOM_STATE + 5)
de_seen = set()
de_convergence = []
de_evaluations = 0
de_resource_units = 0
de_start = time.time()

# Initial population of three candidates.
de_population_params = []
de_population_results = []

while len(de_population_params) < DE_POPULATION_SIZE:
    random_seed = int(
        de_rng.integers(0, 2**31 - 1)
    )
    local_rng = random.Random(random_seed)

    params = ensure_unique_configuration(
        sample_random_configuration(local_rng),
        de_seen,
        local_rng,
    )

    run = evaluate_configuration(
        params,
        MAX_ITER,
    )

    de_population_params.append(params)
    de_population_results.append(run)

    de_evaluations += 1
    de_resource_units += run["resource_units"]

    current_best = max(
        item["validation_macro_f1"]
        for item in de_population_results
    )

    de_convergence.append(
        {
            "Evaluation": de_evaluations,
            "Cumulative Resource Units": de_resource_units,
            "Best Observed CV Macro F1": current_best,
        }
    )

# One best/1/bin generation.
for generation in range(DE_GENERATIONS):
    current_vectors = np.vstack(
        [
            de_params_to_vector(params)
            for params in de_population_params
        ]
    )

    current_scores = np.array(
        [
            run["validation_macro_f1"]
            for run in de_population_results
        ],
        dtype=float,
    )

    best_index = int(
        np.argmax(current_scores)
    )

    best_vector = current_vectors[best_index]

    new_population_params = []
    new_population_results = []

    for target_index in range(
        DE_POPULATION_SIZE
    ):
        candidate_indices = [
            index
            for index in range(DE_POPULATION_SIZE)
            if index != target_index
        ]

        # With population 3, the two non-target members provide
        # the difference vector required by best/1-style mutation.
        r1, r2 = candidate_indices

        mutant = (
            best_vector
            + DE_MUTATION_FACTOR
            * (
                current_vectors[r1]
                - current_vectors[r2]
            )
        )

        trial = current_vectors[target_index].copy()

        forced_dimension = int(
            de_rng.integers(
                0,
                len(DE_KEYS),
            )
        )

        for dimension in range(len(DE_KEYS)):
            if (
                de_rng.random() < DE_CROSSOVER_RATE
                or dimension == forced_dimension
            ):
                trial[dimension] = mutant[dimension]

        trial_params = decode_de_population_vector(trial)

        # The discrete projection can occasionally reproduce an
        # existing member. In that case, mutate one categorical
        # dimension to keep the trial informative.
        trial_signature = configuration_signature(
            trial_params
        )

        if trial_signature in de_seen:
            forced_key = de_rng.choice(DE_KEYS)
            options = SEARCH_SPACE[forced_key]
            alternatives = [
                value
                for value in options
                if value != trial_params[forced_key]
            ]

            if alternatives:
                trial_params[forced_key] = de_rng.choice(
                    alternatives
                )

        run = evaluate_configuration(
            trial_params,
            MAX_ITER,
        )

        de_evaluations += 1
        de_resource_units += run["resource_units"]

        target_result = de_population_results[target_index]

        if (
            run["validation_macro_f1"]
            >
            target_result["validation_macro_f1"]
        ):
            selected_params = trial_params
            selected_run = run
        else:
            selected_params = de_population_params[target_index]
            selected_run = target_result

        de_seen.add(
            configuration_signature(selected_params)
        )

        new_population_params.append(selected_params)
        new_population_results.append(selected_run)

        current_best = max(
            item["validation_macro_f1"]
            for item in (
                new_population_results
                + de_population_results
            )
        )

        de_convergence.append(
            {
                "Evaluation": de_evaluations,
                "Cumulative Resource Units": de_resource_units,
                "Best Observed CV Macro F1": current_best,
            }
        )

    de_population_params = new_population_params
    de_population_results = new_population_results


de_optimization_time = time.time() - de_start

de_best_run = max(
    de_population_results,
    key=lambda item: item["validation_macro_f1"],
)

de_best_params = de_best_run["params"]
de_best_cv_macro_f1 = de_best_run["validation_macro_f1"]

print("\nDifferential Evolution completed.")
print(
    f"Optimization Time: "
    f"{de_optimization_time:.2f} seconds"
)
print(f"Evaluations: {de_evaluations}")
print(
    f"Best CV Macro F1: "
    f"{de_best_cv_macro_f1:.4f}"
)

print("\nBest Hyperparameters:")
for key, value in de_best_params.items():
    print(f"{key}: {value}")


# ============================================================
# COMMON BUDGET CHECK
# ============================================================

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
            "Differential Evolution",
        ],
        "Evaluations": [
            random_evaluations,
            bayesian_evaluations,
            hyperband_evaluations,
            lea_evaluations,
            tlbo_evaluations,
            sa_evaluations,
            de_evaluations,
        ],
        "CV Folds per Evaluation": [
            N_SPLITS,
            N_SPLITS,
            N_SPLITS,
            N_SPLITS,
            N_SPLITS,
            N_SPLITS,
            N_SPLITS,
        ],
        "Resource Units": [
            random_evaluations * N_SPLITS * MAX_ITER,
            bayesian_evaluations * N_SPLITS * MAX_ITER,
            hyperband_resource_units,
            lea_evaluations * N_SPLITS * MAX_ITER,
            tlbo_evaluations * N_SPLITS * MAX_ITER,
            sa_evaluations * N_SPLITS * MAX_ITER,
            de_evaluations * N_SPLITS * MAX_ITER,
        ],
    }
)

budget_results["CV Model Fits"] = (
    budget_results["Evaluations"]
    * budget_results["CV Folds per Evaluation"]
)

print(budget_results)

if not all(
    budget_results["Resource Units"]
    == TOTAL_RESOURCE_BUDGET
):
    raise AssertionError(
        "ERROR: Optimizers do not have equal computational budgets."
    )

print(
    "\nEqual computational budget confirmed: "
    f"{TOTAL_RESOURCE_BUDGET} resource units per optimizer."
)

budget_results.to_csv(
    RESULTS_DIR / "solar_flare_phase4_budget_check.csv",
    index=False,
)


# ============================================================
# FINAL TRAINING PREPROCESSING
# ============================================================
#
# Optimization used only training data through CV.
# The test set has not been used for selection.
#
# For the final model, the selected hyperparameters are retrained
# on the COMPLETE training partition (80% of the data) using the
# same Controlled SMOTENC pipeline. The untouched test set is then
# evaluated exactly once per final model.
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL TRAINING AND TEST EVALUATION")
print("=" * 80)


def prepare_final_training_data():
    """Fit preprocessing and Controlled SMOTENC on all training data."""

    numerical_features = [
        column
        for column in X_train.columns
        if column not in CATEGORICAL_FEATURES
    ]

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_cat_train = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )

    X_cat_test = categorical_imputer.transform(
        X_test[CATEGORICAL_FEATURES]
    )

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_cat_train = ordinal_encoder.fit_transform(
        X_cat_train
    )

    X_cat_test = ordinal_encoder.transform(
        X_cat_test
    )

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    X_num_train = numerical_imputer.fit_transform(
        X_train[numerical_features]
    )

    X_num_test = numerical_imputer.transform(
        X_test[numerical_features]
    )

    X_prepared_train = np.column_stack(
        [
            X_cat_train,
            X_num_train,
        ]
    )

    X_prepared_test = np.column_stack(
        [
            X_cat_test,
            X_num_test,
        ]
    )

    current_counts = (
        y_train.value_counts()
        .to_dict()
    )

    sampling_strategy = {}

    for class_id, desired_count in CONTROLLED_TARGETS.items():
        current_count = current_counts.get(class_id, 0)

        if (
            current_count > 0
            and current_count < desired_count
        ):
            sampling_strategy[class_id] = desired_count

    if sampling_strategy:
        eligible_counts = [
            current_counts[class_id]
            for class_id in sampling_strategy
            if current_counts[class_id] > 1
        ]

        if eligible_counts:
            safe_k = min(
                CONTROLLED_SMOTENC_K_NEIGHBORS,
                min(eligible_counts) - 1,
            )
        else:
            safe_k = 0

        if safe_k >= 1:
            smote = SMOTENC(
                categorical_features=[0, 1, 2],
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE,
                k_neighbors=safe_k,
            )

            X_resampled, y_resampled = smote.fit_resample(
                X_prepared_train,
                y_train,
            )
        else:
            X_resampled = X_prepared_train
            y_resampled = np.asarray(y_train)
    else:
        X_resampled = X_prepared_train
        y_resampled = np.asarray(y_train)

    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    X_cat_resampled = onehot_encoder.fit_transform(
        X_resampled[:, :3]
    )

    X_cat_test = onehot_encoder.transform(
        X_prepared_test[:, :3]
    )

    X_final_train = np.column_stack(
        [
            X_cat_resampled,
            X_resampled[:, 3:],
        ]
    )

    X_final_test = np.column_stack(
        [
            X_cat_test,
            X_prepared_test[:, 3:],
        ]
    )

    scaler = StandardScaler()

    X_final_train = scaler.fit_transform(
        X_final_train
    )

    X_final_test = scaler.transform(
        X_final_test
    )

    return (
        X_final_train,
        y_resampled,
        X_final_test,
    )


X_final_train, y_final_train, X_final_test = (
    prepare_final_training_data()
)

print("\nFinal Training Feature Shape:")
print(X_final_train.shape)

print("\nFinal Test Feature Shape:")
print(X_final_test.shape)

print("\nFinal Training Target Distribution:")
print(pd.Series(y_final_train).value_counts().sort_index())


# ============================================================
# FINAL MODEL TRAINING HELPER
# ============================================================

def train_final_model(params):
    start = time.time()

    model = build_mlp(
        params,
        MAX_ITER,
    )

    model.fit(
        X_final_train,
        y_final_train,
    )

    training_time = time.time() - start

    predictions = model.predict(
        X_final_test
    )

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Weighted Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Weighted Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Weighted F1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Macro F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "training_time": training_time,
    }


# ============================================================
# SELECT BEST OPTIMIZER BY CV MACRO F1
# ============================================================

optimizer_summary = {
    "Random Search": {
        "best_params": random_best_params,
        "cv_macro_f1": random_best_cv_macro_f1,
        "optimization_time": random_optimization_time,
        "evaluations": random_evaluations,
        "resource_units": random_evaluations * N_SPLITS * MAX_ITER,
    },
    "Bayesian Optimization": {
        "best_params": bayesian_best_params,
        "cv_macro_f1": bayesian_best_cv_macro_f1,
        "optimization_time": bayesian_optimization_time,
        "evaluations": bayesian_evaluations,
        "resource_units": bayesian_evaluations * N_SPLITS * MAX_ITER,
    },
    "Hyperband": {
        "best_params": hyperband_best_params,
        "cv_macro_f1": hyperband_best_cv_macro_f1,
        "optimization_time": hyperband_optimization_time,
        "evaluations": hyperband_evaluations,
        "resource_units": hyperband_resource_units,
    },
    "LEA": {
        "best_params": lea_best_params,
        "cv_macro_f1": lea_best_cv_macro_f1,
        "optimization_time": lea_optimization_time,
        "evaluations": lea_evaluations,
        "resource_units": lea_resource_units,
    },
    "TLBO": {
        "best_params": tlbo_best_params,
        "cv_macro_f1": tlbo_best_cv_macro_f1,
        "optimization_time": tlbo_optimization_time,
        "evaluations": tlbo_evaluations,
        "resource_units": tlbo_resource_units,
    },
    "Simulated Annealing": {
        "best_params": sa_best_params,
        "cv_macro_f1": sa_best_score,
        "optimization_time": sa_optimization_time,
        "evaluations": sa_evaluations,
        "resource_units": sa_resource_units,
    },
    "Differential Evolution": {
        "best_params": de_best_params,
        "cv_macro_f1": de_best_cv_macro_f1,
        "optimization_time": de_optimization_time,
        "evaluations": de_evaluations,
        "resource_units": de_resource_units,
    },
}

best_optimizer = max(
    optimizer_summary,
    key=lambda method: optimizer_summary[method]["cv_macro_f1"],
)

print("\n" + "=" * 80)
print("VALIDATION-BASED OPTIMIZER SELECTION")
print("=" * 80)
print(
    f"Selected optimizer: {best_optimizer}"
)
print(
    f"CV Macro F1: "
    f"{optimizer_summary[best_optimizer]['cv_macro_f1']:.4f}"
)
print(
    "\nTest set has not been used for optimizer selection."
)


# ============================================================
# FINAL BASELINE + FINAL OPTIMIZER MODELS
# ============================================================

final_results = []
final_models = {}
final_predictions = {}

# Baseline first.
baseline_final = train_final_model(
    baseline_params
)

final_models["Baseline MLP"] = baseline_final
final_predictions["Baseline MLP"] = baseline_final["predictions"]

final_results.append(
    {
        "Method": "Baseline MLP",
        "Validation Macro F1": baseline_cv_run["validation_macro_f1"],
        "Test Accuracy": baseline_final["metrics"]["Accuracy"],
        "Test Weighted F1": baseline_final["metrics"]["Weighted F1"],
        "Test Macro F1": baseline_final["metrics"]["Macro F1"],
        "Optimization Time (s)": 0.0,
        "Final Training Time (s)": baseline_final["training_time"],
        "Evaluations": 0,
        "CV Model Fits": 0,
        "Optimization Resource Units": 0,
        "Final Training Iterations": baseline_final["model"].n_iter_,
        "Final Training Loss": baseline_final["model"].loss_curve_[-1],
        "Hidden Layers": baseline_params["hidden_layer_sizes"],
        "Activation": baseline_params["activation"],
        "Learning Rate": baseline_params["learning_rate_init"],
        "Alpha": baseline_params["alpha"],
        "Batch Size": baseline_params["batch_size"],
    }
)


for method, info in optimizer_summary.items():
    print("\n" + "-" * 70)
    print(f"Training final {method} model...")

    final_run = train_final_model(
        info["best_params"]
    )

    final_models[method] = final_run
    final_predictions[method] = final_run["predictions"]

    print(
        f"Final Training Time: "
        f"{final_run['training_time']:.2f} seconds"
    )

    print(
        f"Test Macro F1: "
        f"{final_run['metrics']['Macro F1']:.4f}"
    )

    final_results.append(
        {
            "Method": method,
            "Validation Macro F1": info["cv_macro_f1"],
            "Test Accuracy": final_run["metrics"]["Accuracy"],
            "Test Weighted F1": final_run["metrics"]["Weighted F1"],
            "Test Macro F1": final_run["metrics"]["Macro F1"],
            "Optimization Time (s)": info["optimization_time"],
            "Final Training Time (s)": final_run["training_time"],
            "Evaluations": info["evaluations"],
            "CV Model Fits": info["evaluations"] * N_SPLITS,
            "Optimization Resource Units": info["resource_units"],
            "Final Training Iterations": final_run["model"].n_iter_,
            "Final Training Loss": final_run["model"].loss_curve_[-1],
            "Hidden Layers": info["best_params"]["hidden_layer_sizes"],
            "Activation": info["best_params"]["activation"],
            "Learning Rate": info["best_params"]["learning_rate_init"],
            "Alpha": info["best_params"]["alpha"],
            "Batch Size": info["best_params"]["batch_size"],
        }
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

comparison_results = pd.DataFrame(
    final_results
)

# Test-set improvement relative to the final Phase 4 baseline.
baseline_test_macro_f1 = float(
    comparison_results.loc[
        comparison_results["Method"] == "Baseline MLP",
        "Test Macro F1",
    ].iloc[0]
)

comparison_results["Test Macro F1 Improvement (%)"] = (
    (
        comparison_results["Test Macro F1"]
        - baseline_test_macro_f1
    )
    / baseline_test_macro_f1
    * 100.0
)

print("\n" + "=" * 80)
print("SOLAR FLARE PHASE 4 COMPARISON")
print("=" * 80)
print(comparison_results.to_string(index=False))

comparison_results.to_csv(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_results.csv",
    index=False,
)


# ============================================================
# FINAL TEST DETAILS FOR EACH OPTIMIZER
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST METRICS")
print("=" * 80)

for method, run in final_models.items():
    print("\n" + "-" * 70)
    print(method)
    print("-" * 70)

    metrics = run["metrics"]

    print(
        f"Accuracy          : {metrics['Accuracy']:.4f}"
    )
    print(
        f"Weighted Precision: "
        f"{metrics['Weighted Precision']:.4f}"
    )
    print(
        f"Weighted Recall   : "
        f"{metrics['Weighted Recall']:.4f}"
    )
    print(
        f"Weighted F1       : "
        f"{metrics['Weighted F1']:.4f}"
    )
    print(
        f"Macro F1          : "
        f"{metrics['Macro F1']:.4f}"
    )


# ============================================================
# SELECTED OPTIMIZER DETAILED TEST REPORT
# ============================================================

selected_run = final_models[best_optimizer]
selected_predictions = selected_run["predictions"]

print("\n" + "=" * 80)
print("VALIDATION-SELECTED OPTIMIZER - TEST REPORT")
print("=" * 80)
print(
    f"Selected Optimizer: {best_optimizer}"
)
print(
    f"CV Macro F1: "
    f"{optimizer_summary[best_optimizer]['cv_macro_f1']:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        selected_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        selected_predictions,
        labels=[0, 1, 2, 3],
    )
)


# ============================================================
# CONVERGENCE HISTORY
# ============================================================

convergence_frames = []


def append_convergence(method, rows):
    for row in rows:
        convergence_frames.append(
            {
                "Method": method,
                "Evaluation": row["Evaluation"],
                "Cumulative Resource Units": row[
                    "Cumulative Resource Units"
                ],
                "Best Observed CV Macro F1": row[
                    "Best Observed CV Macro F1"
                ],
            }
        )


# Random Search convergence is reconstructed from the completed runs.
random_best_so_far = -np.inf
for index, run in enumerate(random_search_runs, start=1):
    random_best_so_far = max(
        random_best_so_far,
        run["validation_macro_f1"],
    )
    convergence_frames.append(
        {
            "Method": "Random Search",
            "Evaluation": index,
            "Cumulative Resource Units": index * N_SPLITS * MAX_ITER,
            "Best Observed CV Macro F1": random_best_so_far,
        }
    )

append_convergence(
    "Bayesian Optimization",
    bayesian_convergence,
)

append_convergence(
    "Hyperband",
    hyperband_convergence,
)

append_convergence(
    "LEA",
    lea_convergence,
)

append_convergence(
    "TLBO",
    tlbo_convergence,
)

append_convergence(
    "Differential Evolution",
    de_convergence,
)

for row in sa_convergence:
    convergence_frames.append(
        {
            "Method": "Simulated Annealing",
            "Evaluation": row["Evaluation"],
            "Cumulative Resource Units": row[
                "Cumulative Resource Units"
            ],
            "Best Observed CV Macro F1": row[
                "Best Observed CV Macro F1"
            ],
        }
    )

convergence_df = pd.DataFrame(convergence_frames)

convergence_df.to_csv(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_convergence.csv",
    index=False,
)


# ============================================================
# CONVERGENCE PLOT
# ============================================================

plt.figure(figsize=(12, 7))

for method in convergence_df["Method"].unique():
    method_df = convergence_df[
        convergence_df["Method"] == method
    ].sort_values(
        "Cumulative Resource Units"
    )

    plt.plot(
        method_df["Cumulative Resource Units"],
        method_df["Best Observed CV Macro F1"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=method,
    )

plt.xlabel("Cumulative Resource Units")
plt.ylabel("Best Observed CV Macro F1")
plt.title(
    "Solar Flare Phase 4 - Optimizer Convergence Under Equal Budget"
)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_convergence.png",
    dpi=200,
)
plt.close()


# ============================================================
# SA TEMPERATURE PLOT
# ============================================================

sa_convergence_df = pd.DataFrame(sa_convergence)

sa_convergence_df.to_csv(
    RESULTS_DIR / "solar_flare_phase4_sa_convergence.csv",
    index=False,
)

plt.figure(figsize=(10, 6))
plt.plot(
    sa_convergence_df["Evaluation"],
    sa_convergence_df["Temperature"],
    marker="o",
)
plt.xlabel("Evaluation")
plt.ylabel("Temperature")
plt.title(
    "Solar Flare Phase 4 - Simulated Annealing Temperature"
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "solar_flare_phase4_sa_temperature.png",
    dpi=200,
)
plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SOLAR FLARE PHASE 4 COMPLETED")
print("=" * 80)

print(
    f"\nPrimary Optimization Metric: "
    f"5-Fold CV Macro F1"
)

print(
    f"Equal Resource Budget per Optimizer: "
    f"{TOTAL_RESOURCE_BUDGET}"
)

print(
    f"\nValidation-Selected Optimizer: "
    f"{best_optimizer}"
)

print(
    f"Validation CV Macro F1: "
    f"{optimizer_summary[best_optimizer]['cv_macro_f1']:.4f}"
)

print(
    "\nImportant: the test set was used only for final evaluation "
    "and not for optimizer selection."
)

print(
    "\nResults saved to:"
)
print(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_results.csv"
)
print(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_convergence.csv"
)
print(
    RESULTS_DIR / "solar_flare_phase4_equal_budget_convergence.png"
)
print(
    RESULTS_DIR / "solar_flare_phase4_budget_check.csv"
)

