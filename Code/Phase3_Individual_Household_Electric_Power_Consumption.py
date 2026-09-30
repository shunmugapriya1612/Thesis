# =====================================================
# PHASE 3 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER
# CONSUMPTION (UCI 235)
# =====================================================
#
# Objective:
# Compare multiple hyperparameter optimization algorithms
# for MLP-based next-hour electricity consumption forecasting.
#
# FINAL FROZEN PHASE 2 PIPELINE
# -----------------------------
# Hourly aggregation
# Current + historical lags:
# [0, 1, 2, 3, 6, 12, 24]
# Forward-fill imputation
# Raw calendar features
# StandardScaler
#
# Forecast target:
# Global_active_power(t+1)
#
# PRIMARY OPTIMIZATION METRIC
# ---------------------------
# Validation RMSE
#
# Lower RMSE = better forecasting performance
#
# Optimizers:
# 1. Random Search
# 2. Bayesian Optimization
# 3. Hyperband
# 4. LEA
# 5. TLBO
# 6. Simulated Annealing
#
# BASELINE
# --------
# Fixed MLP configuration from Phase 1
#
# COMPUTATIONAL BUDGET
# --------------------
# All optimizers receive the same maximum resource budget:
#
#     20 evaluations x 500 iterations
#     = 10,000 training-iteration resource units
#
# Hyperband uses:
#
#     20 candidates x 200 iterations
#     10 survivors  x 350 iterations
#      5 survivors  x 500 iterations
#
# Total:
#
#     20*200 + 10*350 + 5*500
#     = 10,000 resource units
#
# This gives all methods a comparable optimization budget.
#
# IMPORTANT:
# -----------
# The test set is untouched during optimization.
#
# Hyperparameters are selected using the chronological
# training and validation periods only.
#
# After selection, the best configuration for each method
# is retrained on the full development set
# (training + validation) and evaluated once on the
# untouched test set.
#
# WHY SIMULATED ANNEALING IS SUITABLE
# -----------------------------------
# This dataset uses a mixed hyperparameter search space:
#
# - hidden-layer architecture      -> categorical
# - activation                    -> categorical
# - learning rate                 -> discrete numeric choices
# - alpha / regularization        -> discrete numeric choices
# - batch size                    -> categorical / mixed
#
# Simulated Annealing is suitable because it can directly
# explore this mixed discrete search space through
# neighbourhood moves.
#
# Each new configuration changes one or more hyperparameters.
#
# Better configurations are always accepted.
#
# Worse configurations may still be accepted early in the
# search according to a temperature-based probability.
#
# This allows SA to escape local configurations and explore
# different combinations of architecture, learning rate,
# regularization and batch size.
#
# PHASE 2 identified a substantial generalization gap:
#
# Training R²  = 0.8338
# Validation R² = 0.4544
#
# Therefore, the hypothesis is that a different combination
# of network capacity, learning rate, regularization and
# batch size may generalize better to unseen time periods.
#
# SA MAY improve forecasting performance if it finds such
# a configuration.
#
# IMPORTANT:
# SA is NOT assumed to be better than TLBO, LEA, Bayesian
# Optimization or any other method. Its effectiveness must
# be determined empirically under the equal budget.
# =====================================================


# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = (
    "/Users/priya/Documents/Thesis/Data/"
    "household_power_consumption.txt"
)

TARGET_COLUMN = "Global_active_power"

RANDOM_STATE = 42


# -----------------------------------------------------
# Chronological split
# -----------------------------------------------------

TRAIN_RATIO = 0.64
VALIDATION_RATIO = 0.16
TEST_RATIO = 0.20


# -----------------------------------------------------
# Forecast horizon
# -----------------------------------------------------

FORECAST_HORIZON = 1


# -----------------------------------------------------
# Frozen Phase 2 lag structure
# -----------------------------------------------------

LAG_HOURS = [
    0,
    1,
    2,
    3,
    6,
    12,
    24
]


# -----------------------------------------------------
# Frozen Phase 2 imputation method
# -----------------------------------------------------
#
# Phase 2 selected forward fill based on validation RMSE.
# This is kept fixed for all Phase 3 optimizers.
# -----------------------------------------------------

FROZEN_IMPUTATION = "forward_fill"


# =====================================================
# EQUAL COMPUTATIONAL BUDGET
# =====================================================

N_OPTIMIZER_EVALUATIONS = 6

MAX_ITER = 500

TOTAL_RESOURCE_BUDGET = (
    N_OPTIMIZER_EVALUATIONS
    * MAX_ITER
)


# -----------------------------------------------------
# Hyperband resource schedule
#
# 20 candidates -> 10 -> 5
# -----------------------------------------------------

HYPERBAND_INITIAL_CANDIDATES = 8

HYPERBAND_SURVIVORS_STAGE_1 = 3

HYPERBAND_SURVIVORS_STAGE_2 = 1

HYPERBAND_RESOURCES = [
    200,
    300,
    500
]


# -----------------------------------------------------
# LEA budget
#
# Initial population = 5
# 3 generations x 5 evaluations
#
# Total = 20 evaluations
# -----------------------------------------------------

LEA_POPULATION_SIZE = 3

LEA_GENERATIONS = 1


# -----------------------------------------------------
# TLBO budget
#
# Initial population = 4
#
# 2 generations
# Each generation:
#   4 teacher evaluations
#   4 learner evaluations
#
# Total = 4 + 8 + 8 = 20
# -----------------------------------------------------

TLBO_POPULATION_SIZE = 2

TLBO_GENERATIONS = 1


# -----------------------------------------------------
# SA budget
#
# Initial solution = 1
# Neighbour evaluations = 19
#
# Total = 20
# -----------------------------------------------------

SA_EVALUATIONS = 6

SA_INITIAL_TEMPERATURE = 0.05

SA_COOLING_RATE = 0.90


# =====================================================
# SEARCH SPACE
# =====================================================

SEARCH_SPACE = {

    "hidden_layer_sizes": [

        (32,),

        (64,),

        (128,),

        (64, 32),

        (128, 64),

        (128, 64, 32)

    ],

    "activation": [

        "relu",

        "tanh"

    ],

    "learning_rate_init": [

        0.0001,

        0.0003,

        0.001,

        0.003,

        0.005

    ],

    "alpha": [

        0.00001,

        0.0001,

        0.001,

        0.01

    ],

    "batch_size": [

        "auto",

        16,

        32,

        64,

        128

    ],

    # -------------------------------------------------
    # Adam remains fixed.
    # -------------------------------------------------

    "solver": [

        "adam"

    ]

}


# =====================================================
# HELPER:
# CREATE MODEL DATA
# =====================================================

def create_model_data(
    hourly_data
):
    """
    Create frozen Phase 2 feature representation.

    Predict:

        Global_active_power(t+1)

    Using:

        current and historical information available
        at time t and earlier.
    """

    feature_data = pd.DataFrame(
        index=hourly_data.index
    )


    lag_source_columns = [

        "Global_active_power",

        "Global_reactive_power",

        "Voltage",

        "Global_intensity",

        "Sub_metering_1",

        "Sub_metering_2",

        "Sub_metering_3"

    ]


    # -------------------------------------------------
    # Create frozen Phase 2 lag structure
    # -------------------------------------------------

    for column in lag_source_columns:

        for lag in LAG_HOURS:

            feature_name = (
                f"{column}_lag_{lag}h"
            )

            feature_data[
                feature_name
            ] = (
                hourly_data[
                    column
                ].shift(lag)
            )


    # -------------------------------------------------
    # Raw calendar features
    #
    # Phase 2 found no RMSE advantage from replacing
    # these with cyclical features.
    # -------------------------------------------------

    feature_data["hour"] = (
        hourly_data.index.hour
    )

    feature_data["day_of_week"] = (
        hourly_data.index.dayofweek
    )

    feature_data["month"] = (
        hourly_data.index.month
    )

    feature_data["day_of_month"] = (
        hourly_data.index.day
    )


    # -------------------------------------------------
    # One-step-ahead target
    # -------------------------------------------------

    target = (
        hourly_data[
            TARGET_COLUMN
        ]
        .shift(
            -FORECAST_HORIZON
        )
    )


    model_data = feature_data.copy()

    model_data["Target"] = target


    # -------------------------------------------------
    # Remove observations where the future target does
    # not exist.
    # -------------------------------------------------

    model_data = (
        model_data[
            model_data["Target"].notna()
        ]
        .copy()
    )


    return model_data


# =====================================================
# HELPER:
# APPLY CAUSAL FORWARD-FILL IMPUTATION
# =====================================================

def apply_causal_forward_fill(
    model_data,
    train_end_time
):
    """
    Apply forward-fill only.

    Forward fill is causal because each missing value is
    replaced using the most recent previously observed value.

    No future observation is used.

    Leading missing values are replaced with medians
    calculated only from the training period.
    """

    feature_columns = [

        column

        for column in model_data.columns

        if column != "Target"

    ]


    features = (
        model_data[
            feature_columns
        ]
        .copy()
    )


    # -------------------------------------------------
    # Causal forward fill
    # -------------------------------------------------

    features = (
        features
        .ffill()
    )


    # -------------------------------------------------
    # Training-only fallback medians
    #
    # These are used only for values that are still
    # missing after forward filling, mainly at the
    # beginning of the time series.
    # -------------------------------------------------

    training_features = (
        features.loc[
            features.index <= train_end_time
        ]
    )


    training_medians = (
        training_features
        .median()
    )


    features = (
        features
        .fillna(
            training_medians
        )
    )


    result = model_data.copy()

    result[
        feature_columns
    ] = features


    return result


# =====================================================
# LOAD DATASET
# =====================================================

print("\n" + "=" * 80)
print(
    "PHASE 3 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER"
)
print(
    "CONSUMPTION"
)
print("=" * 80)


print("\nDataset Path:")
print(DATA_PATH)


data = pd.read_csv(

    DATA_PATH,

    sep=";",

    low_memory=False,

    na_values=["?"]

)


print("\nDataset Shape:")
print(data.shape)


print("\nFirst Five Rows:")
print(data.head())


# =====================================================
# CREATE TIMESTAMP
# =====================================================

print("\nCreating Timestamp...")


data["Timestamp"] = pd.to_datetime(

    data["Date"]
    + " "
    + data["Time"],

    format="%d/%m/%Y %H:%M:%S",

    errors="coerce"

)


data = (
    data
    .drop(
        columns=[
            "Date",
            "Time"
        ]
    )
)


data = (
    data
    .sort_values(
        "Timestamp"
    )
    .reset_index(
        drop=True
    )
)


print("\nTimestamp Range:")
print(
    data["Timestamp"].min(),
    "to",
    data["Timestamp"].max()
)


# =====================================================
# NUMERIC CONVERSION
# =====================================================

numeric_columns = [

    "Global_active_power",

    "Global_reactive_power",

    "Voltage",

    "Global_intensity",

    "Sub_metering_1",

    "Sub_metering_2",

    "Sub_metering_3"

]


for column in numeric_columns:

    data[column] = pd.to_numeric(

        data[column],

        errors="coerce"

    )


# =====================================================
# HOURLY AGGREGATION
# =====================================================

data = (
    data
    .set_index(
        "Timestamp"
    )
)


print(
    "\nAggregating data to hourly frequency..."
)


hourly_data = (
    data[
        numeric_columns
    ]
    .resample(
        "1h"
    )
    .mean()
)


print("\nHourly Dataset Shape:")
print(
    hourly_data.shape
)


# =====================================================
# CREATE FROZEN PHASE 2 FEATURE DATA
# =====================================================

print(
    "\nCreating frozen Phase 2 feature representation..."
)


raw_model_data = create_model_data(
    hourly_data
)


print(
    "\nModeling Dataset Shape Before Imputation:"
)

print(
    raw_model_data.shape
)


# =====================================================
# FIXED CHRONOLOGICAL SPLIT
# =====================================================
#
# The split boundaries are determined once and then
# reused by every optimizer.
# =====================================================

n_samples = len(
    raw_model_data
)


train_end_position = int(
    n_samples
    * TRAIN_RATIO
)


validation_end_position = int(
    n_samples
    * (
        TRAIN_RATIO
        + VALIDATION_RATIO
    )
)


TRAIN_END_TIME = (
    raw_model_data.index[
        train_end_position - 1
    ]
)


VALIDATION_END_TIME = (
    raw_model_data.index[
        validation_end_position - 1
    ]
)


print(
    "\n" + "=" * 80
)

print(
    "FIXED CHRONOLOGICAL SPLIT"
)

print(
    "=" * 80
)


print(
    "\nTraining ends:"
)

print(
    TRAIN_END_TIME
)


print(
    "\nValidation ends:"
)

print(
    VALIDATION_END_TIME
)


# =====================================================
# APPLY FROZEN PHASE 2 IMPUTATION
# =====================================================

print(
    "\nApplying frozen Phase 2 imputation:"
)

print(
    FROZEN_IMPUTATION
)


if FROZEN_IMPUTATION == "forward_fill":

    model_data = apply_causal_forward_fill(

        raw_model_data,

        TRAIN_END_TIME

    )

else:

    raise ValueError(
        "Unexpected frozen imputation method."
    )


# =====================================================
# CHRONOLOGICAL SPLIT
# =====================================================

train_mask = (
    model_data.index
    <= TRAIN_END_TIME
)


validation_mask = (
    (model_data.index > TRAIN_END_TIME)
    &
    (model_data.index <= VALIDATION_END_TIME)
)


test_mask = (
    model_data.index
    > VALIDATION_END_TIME
)


train_data = (
    model_data.loc[
        train_mask
    ]
    .copy()
)


validation_data = (
    model_data.loc[
        validation_mask
    ]
    .copy()
)


test_data = (
    model_data.loc[
        test_mask
    ]
    .copy()
)


feature_columns = [

    column

    for column in model_data.columns

    if column != "Target"

]


X_train = train_data[
    feature_columns
]

y_train = train_data[
    "Target"
]


X_validation = validation_data[
    feature_columns
]

y_validation = validation_data[
    "Target"
]


X_test = test_data[
    feature_columns
]

y_test = test_data[
    "Target"
]


print(
    "\nTraining Feature Shape:"
)

print(
    X_train.shape
)


print(
    "\nValidation Feature Shape:"
)

print(
    X_validation.shape
)


print(
    "\nTesting Feature Shape:"
)

print(
    X_test.shape
)


# =====================================================
# STANDARDIZATION
# =====================================================
#
# The scaler is fitted ONLY on training data for the
# optimization stage.
# =====================================================

print(
    "\nApplying StandardScaler..."
)


scaler = StandardScaler()


X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)


X_validation_scaled = (
    scaler.transform(
        X_validation
    )
)


X_test_scaled = (
    scaler.transform(
        X_test
    )
)


print(
    "Feature Standardization Completed."
)


# =====================================================
# HELPER:
# BUILD MLP
# =====================================================

def build_mlp(
    params,
    max_iter=MAX_ITER
):
    """
    Build a deterministic MLP configuration.
    """

    return MLPRegressor(

        hidden_layer_sizes=(
            params[
                "hidden_layer_sizes"
            ]
        ),

        activation=(
            params[
                "activation"
            ]
        ),

        solver="adam",

        learning_rate_init=(
            params[
                "learning_rate_init"
            ]
        ),

        alpha=(
            params[
                "alpha"
            ]
        ),

        batch_size=(
            params[
                "batch_size"
            ]
        ),

        max_iter=max_iter,

        random_state=RANDOM_STATE,

        early_stopping=False,

        tol=0.0,

        n_iter_no_change=max_iter

    )


# =====================================================
# HELPER:
# CALCULATE METRICS
# =====================================================

def calculate_metrics(
    y_true,
    predictions
):

    mse = mean_squared_error(
        y_true,
        predictions
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        y_true,
        predictions
    )


    r2 = r2_score(
        y_true,
        predictions
    )


    return {

        "MAE": mae,

        "RMSE": rmse,

        "MSE": mse,

        "R2": r2

    }


# =====================================================
# HELPER:
# EVALUATE CONFIGURATION
# =====================================================

def evaluate_configuration(
    params,
    max_iter=MAX_ITER
):
    """
    Train one MLP configuration on the chronological
    training period and evaluate it on validation data.

    Validation RMSE is the optimization objective.
    """

    model = build_mlp(
        params,
        max_iter=max_iter
    )


    start_time = time.time()


    model.fit(

        X_train_scaled,

        y_train

    )


    runtime = (
        time.time()
        - start_time
    )


    validation_predictions = (
        model.predict(
            X_validation_scaled
        )
    )


    validation_metrics = (
        calculate_metrics(

            y_validation,

            validation_predictions

        )
    )


    return {

        "params": params,

        "model": model,

        "validation_metrics": (
            validation_metrics
        ),

        "validation_rmse": (
            validation_metrics[
                "RMSE"
            ]
        ),

        "runtime": runtime,

        "iterations": (
            model.n_iter_
        )

    }


# =====================================================
# BASELINE MLP
# =====================================================

baseline_params = {

    "hidden_layer_sizes": (
        64,
        32
    ),

    "activation": "relu",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": "auto"

}


print(
    "\n" + "=" * 80
)

print(
    "BASELINE MLP"
)

print(
    "=" * 80
)


baseline_validation_run = (
    evaluate_configuration(
        baseline_params
    )
)


baseline_validation_metrics = (
    baseline_validation_run[
        "validation_metrics"
    ]
)


print(
    "\nBaseline Validation Performance"
)


print(
    f"MAE  : "
    f"{baseline_validation_metrics['MAE']:.4f}"
)


print(
    f"RMSE : "
    f"{baseline_validation_metrics['RMSE']:.4f}"
)


print(
    f"MSE  : "
    f"{baseline_validation_metrics['MSE']:.4f}"
)


print(
    f"R²   : "
    f"{baseline_validation_metrics['R2']:.4f}"
)


# =====================================================
# HELPER:
# RANDOM PARAMETER SAMPLE
# =====================================================

def sample_random_configuration(
    rng
):

    hidden_index = int(
        rng.integers(
            0,
            len(
                SEARCH_SPACE[
                    "hidden_layer_sizes"
                ]
            )
        )
    )


    activation_index = int(
        rng.integers(
            0,
            len(
                SEARCH_SPACE[
                    "activation"
                ]
            )
        )
    )


    learning_rate_index = int(
        rng.integers(
            0,
            len(
                SEARCH_SPACE[
                    "learning_rate_init"
                ]
            )
        )
    )


    alpha_index = int(
        rng.integers(
            0,
            len(
                SEARCH_SPACE[
                    "alpha"
                ]
            )
        )
    )


    batch_index = int(
        rng.integers(
            0,
            len(
                SEARCH_SPACE[
                    "batch_size"
                ]
            )
        )
    )


    return {

        "hidden_layer_sizes": (
            SEARCH_SPACE[
                "hidden_layer_sizes"
            ][
                hidden_index
            ]
        ),

        "activation": (
            SEARCH_SPACE[
                "activation"
            ][
                activation_index
            ]
        ),

        "learning_rate_init": (
            SEARCH_SPACE[
                "learning_rate_init"
            ][
                learning_rate_index
            ]
        ),

        "alpha": (
            SEARCH_SPACE[
                "alpha"
            ][
                alpha_index
            ]
        ),

        "batch_size": (
            SEARCH_SPACE[
                "batch_size"
            ][
                batch_index
            ]
        )

    }


# =====================================================
# HELPER:
# CONFIGURATION SIGNATURE
# =====================================================

def configuration_signature(
    params
):

    return (

        str(
            params[
                "hidden_layer_sizes"
            ]
        ),

        params[
            "activation"
        ],

        params[
            "learning_rate_init"
        ],

        params[
            "alpha"
        ],

        str(
            params[
                "batch_size"
            ]
        )

    )


# =====================================================
# RANDOM SEARCH
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "RANDOM SEARCH"
)

print(
    "=" * 80
)


random_start = time.time()


random_rng = np.random.default_rng(
    RANDOM_STATE
)


random_results = []

random_convergence = []


random_seen = set()


while len(random_results) < N_OPTIMIZER_EVALUATIONS:

    params = sample_random_configuration(
        random_rng
    )


    signature = (
        configuration_signature(
            params
        )
    )


    if signature in random_seen:

        continue


    random_seen.add(
        signature
    )


    run = evaluate_configuration(
        params
    )


    random_results.append(
        run
    )


    current_best = min(

        result[
            "validation_rmse"
        ]

        for result in random_results

    )


    random_convergence.append({

        "Evaluation": len(
            random_results
        ),

        "Best Validation RMSE": (
            current_best
        )

    })


    print(

        f"Evaluation "
        f"{len(random_results):02d}/"
        f"{N_OPTIMIZER_EVALUATIONS}"
        f" - RMSE = "
        f"{run['validation_rmse']:.4f}"

    )


random_optimization_time = (
    time.time()
    - random_start
)


random_best_run = min(

    random_results,

    key=lambda x:
        x["validation_rmse"]

)


random_best_params = (
    random_best_run[
        "params"
    ]
)


random_best_score = (
    random_best_run[
        "validation_rmse"
    ]
)


random_model = (
    random_best_run[
        "model"
    ]
)


random_training_runtime = (
    random_best_run[
        "runtime"
    ]
)


random_evaluations = (
    len(random_results)
)


print(
    "\nRandom Search completed."
)


print(
    f"Optimization Time: "
    f"{random_optimization_time:.2f} seconds"
)


print(
    f"Best Validation RMSE: "
    f"{random_best_score:.4f}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    random_best_params
)


# =====================================================
# BAYESIAN OPTIMIZATION
# =====================================================
#
# Bayesian Optimization uses previous validation results
# to guide the next configuration.
#
# The objective is validation RMSE minimization.
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "BAYESIAN OPTIMIZATION"
)

print(
    "=" * 80
)


optuna.logging.set_verbosity(
    optuna.logging.WARNING
)


study = optuna.create_study(

    direction="minimize",

    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    ),

    study_name=(
        "Household_Power_MLP_Bayesian"
    )

)


bayesian_convergence = []

bayesian_runs = {}

bayesian_start = time.time()

def bayesian_objective(
    trial
):

    hidden_options = [

        "32",

        "64",

        "128",

        "64_32",

        "128_64",

        "128_64_32"

    ]


    batch_options = [

        "auto",

        "16",

        "32",

        "64",

        "128"

    ]


    hidden_name = (
        trial.suggest_categorical(
            "hidden_layer_sizes",
            hidden_options
        )
    )


    activation = (
        trial.suggest_categorical(
            "activation",
            ["relu", "tanh"]
        )
    )


    learning_rate = (
        trial.suggest_categorical(
            "learning_rate_init",
            SEARCH_SPACE[
                "learning_rate_init"
            ]
        )
    )


    alpha = (
        trial.suggest_categorical(
            "alpha",
            SEARCH_SPACE[
                "alpha"
            ]
        )
    )


    batch_name = (
        trial.suggest_categorical(
            "batch_size",
            batch_options
        )
    )


    hidden_mapping = {

        "32": (32,),

        "64": (64,),

        "128": (128,),

        "64_32": (64, 32),

        "128_64": (128, 64),

        "128_64_32": (
            128,
            64,
            32
        )

    }


    batch_mapping = {

        "auto": "auto",

        "16": 16,

        "32": 32,

        "64": 64,

        "128": 128

    }


    params = {

        "hidden_layer_sizes": (
            hidden_mapping[
                hidden_name
            ]
        ),

        "activation": activation,

        "learning_rate_init": (
            learning_rate
        ),

        "alpha": alpha,

        "batch_size": (
            batch_mapping[
                batch_name
            ]
        )

    }


    run = evaluate_configuration(
        params
    )

    bayesian_runs[
        trial.number
    ] = run


    # -------------------------------------------------
    # Track best validation RMSE safely
    #
    # Trial 0 has not been completed yet, so
    # study.best_value cannot be accessed.
    # -------------------------------------------------

    if trial.number == 0:

        current_best = run[
            "validation_rmse"
        ]

    else:

        current_best = min(

            study.best_value,

            run[
                "validation_rmse"
            ]

        )
        


    bayesian_convergence.append({

        "Trial": (
            trial.number + 1
        ),

        "Best Validation RMSE": (
            current_best
        )

    })


    trial.set_user_attr(
        "runtime",
        run["runtime"]
    )


    return (
        run[
            "validation_rmse"
        ]
    )


study.optimize(

    bayesian_objective,

    n_trials=N_OPTIMIZER_EVALUATIONS,

    show_progress_bar=False

)


bayesian_optimization_time = (
    time.time()
    - bayesian_start
)


best_trial = study.best_trial


best_params = {

    "hidden_layer_sizes": (
        {
            "32": (32,),
            "64": (64,),
            "128": (128,),
            "64_32": (64, 32),
            "128_64": (128, 64),
            "128_64_32": (
                128,
                64,
                32
            )
        }[
            best_trial.params[
                "hidden_layer_sizes"
            ]
        ]
    ),

    "activation": (
        best_trial.params[
            "activation"
        ]
    ),

    "learning_rate_init": (
        best_trial.params[
            "learning_rate_init"
        ]
    ),

    "alpha": (
        best_trial.params[
            "alpha"
        ]
    ),

    "batch_size": (
        {
            "auto": "auto",
            "16": 16,
            "32": 32,
            "64": 64,
            "128": 128
        }[
            best_trial.params[
                "batch_size"
            ]
        ]
    )

}


best_value = (
    study.best_value
)


# Train the selected Bayesian configuration once
# on the optimization training period.

best_trial = study.best_trial

bayesian_best_validation_run = (
    bayesian_runs[
        best_trial.number
    ]
)

bayesian_model = (
    bayesian_best_validation_run[
        "model"
    ]
)

bayesian_training_runtime = (
    bayesian_best_validation_run[
        "runtime"
    ]
)

print(
    "\nBayesian Optimization completed."
)


print(
    f"Optimization Time: "
    f"{bayesian_optimization_time:.2f} seconds"
)


print(
    f"Best Validation RMSE: "
    f"{best_value:.4f}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    best_params
)


# =====================================================
# HYPERBAND
# =====================================================
#
# Custom successive-halving implementation.
#
# Resource schedule:
#
# Stage 1:
# 20 candidates x 200 iterations
#
# Stage 2:
# 10 survivors x 350 iterations
#
# Stage 3:
# 5 survivors x 500 iterations
#
# Total resource:
#
# 4000 + 3500 + 2500
# = 10000
#
# This matches the optimizer budget used by the other
# methods.
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "HYPERBAND"
)

print(
    "=" * 80
)


hyperband_start = time.time()


hyperband_rng = np.random.default_rng(
    RANDOM_STATE + 1
)


hyperband_candidates = []

hyperband_seen = set()


while len(
    hyperband_candidates
) < HYPERBAND_INITIAL_CANDIDATES:

    params = sample_random_configuration(
        hyperband_rng
    )


    signature = (
        configuration_signature(
            params
        )
    )


    if signature in hyperband_seen:

        continue


    hyperband_seen.add(
        signature
    )


    hyperband_candidates.append(
        params
    )


hyperband_records = []

hyperband_convergence = []

hyperband_global_best = np.inf

hyperband_evaluations = 0


for stage_index, resource in enumerate(
    HYPERBAND_RESOURCES
):

    print(
        "\n" + "-" * 70
    )


    print(
        f"Hyperband Stage "
        f"{stage_index + 1}"
    )


    print(
        f"Candidates: "
        f"{len(hyperband_candidates)}"
    )


    print(
        f"Max Iterations: "
        f"{resource}"
    )


    print(
        "-" * 70
    )


    stage_results = []


    for params in hyperband_candidates:

        run = evaluate_configuration(

            params,

            max_iter=resource

        )


        hyperband_evaluations += 1


        score = run[
            "validation_rmse"
        ]


        if score < hyperband_global_best:

            hyperband_global_best = score


        stage_results.append({

            "params": params,

            "score": score,

            "run": run,

            "resource": resource,

            "stage": (
                stage_index + 1
            )

        })


        hyperband_convergence.append({

            "Evaluation": (
                hyperband_evaluations
            ),

            "Stage": (
                stage_index + 1
            ),

            "Resource": resource,

            "Best Validation RMSE": (
                hyperband_global_best
            )

        })


        hyperband_records.append(
            stage_results[-1]
        )


        print(

            f"Evaluation "
            f"{hyperband_evaluations:02d}"
            f" - RMSE = "
            f"{score:.4f}"

        )


    stage_results.sort(
        key=lambda x:
            x["score"]
    )


    if stage_index == 0:

        survivors = (
            HYPERBAND_SURVIVORS_STAGE_1
        )

    elif stage_index == 1:

        survivors = (
            HYPERBAND_SURVIVORS_STAGE_2
        )

    else:

        survivors = 0


    if survivors > 0:

        hyperband_candidates = [

            result[
                "params"
            ]

            for result in
            stage_results[
                :survivors
            ]

        ]


hyperband_optimization_time = (
    time.time()
    - hyperband_start
)


hyperband_best_record = min(

    stage_results,

    key=lambda x:
        x["score"]

)


hyperband_best_score = (
    hyperband_best_record[
        "score"
    ]
)


hyperband_best_params = (
    hyperband_best_record[
        "params"
    ]
)


hyperband_model = (
    hyperband_best_record[
        "run"
    ]["model"]
)


hyperband_training_runtime = (
    hyperband_best_record[
        "run"
    ]["runtime"]
)


hyperband_resource_units = (

    HYPERBAND_INITIAL_CANDIDATES
    * HYPERBAND_RESOURCES[0]

    +

    HYPERBAND_SURVIVORS_STAGE_1
    * HYPERBAND_RESOURCES[1]

    +

    HYPERBAND_SURVIVORS_STAGE_2
    * HYPERBAND_RESOURCES[2]

)


print(
    "\nHyperband completed."
)


print(
    f"Optimization Time: "
    f"{hyperband_optimization_time:.2f} seconds"
)


print(
    f"Best Validation RMSE: "
    f"{hyperband_best_score:.4f}"
)


print(
    f"Resource Units Used: "
    f"{hyperband_resource_units}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    hyperband_best_params
)


# =====================================================
# HELPER:
# MUTATE CONFIGURATION
# =====================================================

def mutate_configuration(
    params,
    rng,
    n_changes=1
):
    """
    Create a neighbouring configuration by changing
    one or more hyperparameters.

    This is used by LEA and Simulated Annealing.
    """

    new_params = dict(
        params
    )


    mutable_keys = [

        "hidden_layer_sizes",

        "activation",

        "learning_rate_init",

        "alpha",

        "batch_size"

    ]


    n_changes = min(

        n_changes,

        len(
            mutable_keys
        )

    )


    selected_indices = (
        rng.choice(
            len(
                mutable_keys
            ),
            size=n_changes,
            replace=False
        )
    )


    for index in selected_indices:

        key = mutable_keys[
            int(index)
        ]


        current_value = (
            new_params[
                key
            ]
        )


        options = (
            SEARCH_SPACE[
                key
            ]
        )


        current_position = None


        for position, value in enumerate(
            options
        ):

            if value == current_value:

                current_position = position

                break


        possible_positions = [

            position

            for position in
            range(
                len(options)
            )

            if position != current_position

        ]


        new_position = (
            int(
                rng.choice(
                    possible_positions
                )
            )
        )


        new_params[
            key
        ] = options[
            new_position
        ]


    return new_params


# =====================================================
# LEA
# =====================================================
#
# Learning-based Evolutionary Algorithm
#
# Population based search:
#
# 1. Create a population
# 2. Evaluate the population
# 3. Keep the best survivors
# 4. Mutate survivors
# 5. Evaluate offspring
# 6. Repeat
#
# Budget:
# Initial population = 5 evaluations
# 3 generations x 5 offspring = 15
#
# Total = 20 evaluations
#
# This provides a population-based evolutionary method
# distinct from SA's single-solution neighbourhood search.
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "LEA - LEARNING-BASED EVOLUTIONARY ALGORITHM"
)

print(
    "=" * 80
)


lea_start = time.time()


lea_rng = np.random.default_rng(
    RANDOM_STATE + 2
)


lea_population = []

lea_seen = set()


while len(
    lea_population
) < LEA_POPULATION_SIZE:

    params = sample_random_configuration(
        lea_rng
    )


    signature = (
        configuration_signature(
            params
        )
    )


    if signature in lea_seen:

        continue


    lea_seen.add(
        signature
    )


    lea_population.append(
        params
    )


lea_results = []


# -----------------------------------------------------
# Initial population evaluation
# -----------------------------------------------------

print(
    "\nInitial LEA Population"
)


for params in lea_population:

    run = evaluate_configuration(
        params
    )


    lea_results.append(
        run
    )


lea_convergence = []


lea_evaluations = len(
    lea_results
)


lea_best_score = min(

    result[
        "validation_rmse"
    ]

    for result in
    lea_results

)


lea_best_params = min(

    lea_results,

    key=lambda x:
        x["validation_rmse"]

)["params"]


lea_convergence.append({

    "Evaluation": lea_evaluations,

    "Generation": 0,

    "Best Validation RMSE": (
        lea_best_score
    )

})


# -----------------------------------------------------
# Evolutionary generations
# -----------------------------------------------------

for generation in range(
    1,
    LEA_GENERATIONS + 1
):

    print(
        f"\nGeneration "
        f"{generation}/"
        f"{LEA_GENERATIONS}"
    )


    # ---------------------------------------------
    # Select survivors
    # ---------------------------------------------

    sorted_population = sorted(

        lea_results,

        key=lambda x:
            x["validation_rmse"]

    )


    survivor_count = 2


    survivors = (
        sorted_population[
            :survivor_count
        ]
    )


    survivor_params = [

        survivor[
            "params"
        ]

        for survivor in survivors

    ]


    # ---------------------------------------------
    # Generate offspring
    # ---------------------------------------------

    offspring_params = []

    offspring_signatures = set()


    while len(
        offspring_params
    ) < LEA_POPULATION_SIZE:

        parent_index = int(
            lea_rng.integers(
                0,
                len(
                    survivor_params
                )
            )
        )


        parent = (
            survivor_params[
                parent_index
            ]
        )


        child = mutate_configuration(

            parent,

            lea_rng,

            n_changes=1
            if lea_rng.random() < 0.8
            else 2

        )


        signature = (
            configuration_signature(
                child
            )
        )


        if (
            signature
            in offspring_signatures
        ):

            continue


        offspring_signatures.add(
            signature
        )


        offspring_params.append(
            child
        )


    # ---------------------------------------------
    # Evaluate offspring
    # ---------------------------------------------

    offspring_results = []


    for params in offspring_params:

        run = evaluate_configuration(
            params
        )


        offspring_results.append(
            run
        )


        lea_results.append(
            run
        )


        lea_evaluations += 1


        lea_best_score = min(

            lea_best_score,

            run[
                "validation_rmse"
            ]

        )


        if (
            run[
                "validation_rmse"
            ]
            == lea_best_score
        ):

            lea_best_params = (
                run["params"]
            )


        lea_convergence.append({

            "Evaluation": (
                lea_evaluations
            ),

            "Generation": generation,

            "Best Validation RMSE": (
                lea_best_score
            )

        })


        print(

            f"Evaluation "
            f"{lea_evaluations:02d}/"
            f"{N_OPTIMIZER_EVALUATIONS}"
            f" - RMSE = "
            f"{run['validation_rmse']:.4f}"

        )


    # ---------------------------------------------
    # Create next population
    # ---------------------------------------------

    combined_results = (
        survivors
        + offspring_results
    )


    combined_results = sorted(

        combined_results,

        key=lambda x:
            x["validation_rmse"]

    )


    lea_results = (
        combined_results[
            :LEA_POPULATION_SIZE
        ]
    )


lea_optimization_time = (
    time.time()
    - lea_start
)


lea_best_run = min(

    lea_results,

    key=lambda x:
        x["validation_rmse"]

)


lea_best_solution = (
    lea_best_run[
        "params"
    ]
)


lea_best_score = (
    lea_best_run[
        "validation_rmse"
    ]
)


lea_model = (
    lea_best_run[
        "model"
    ]
)


lea_training_runtime = (
    lea_best_run[
        "runtime"
    ]
)


print(
    "\nLEA Optimization completed."
)


print(
    f"Optimization Time: "
    f"{lea_optimization_time:.2f} seconds"
)


print(
    f"Evaluations: "
    f"{lea_evaluations}"
)


print(
    f"Best Validation RMSE: "
    f"{lea_best_score:.4f}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    lea_best_solution
)


# =====================================================
# TLBO
# =====================================================
#
# Teaching-Learning-Based Optimization
#
# The method works in two phases:
#
# 1. Teacher Phase
#    Solutions learn from the current best teacher.
#
# 2. Learner Phase
#    Solutions interact with another learner.
#
# Because the search space contains categorical values,
# each option is represented by a discrete index.
#
# Generated values are rounded and clipped back to valid
# hyperparameter options.
#
# Budget:
#
# Initial population = 4
# 2 generations:
#   teacher = 4
#   learner = 4
#
# Total = 20 evaluations
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "TLBO - TEACHING-LEARNING-BASED OPTIMIZATION"
)

print(
    "=" * 80
)


tlbo_start = time.time()


tlbo_rng = np.random.default_rng(
    RANDOM_STATE + 3
)


TLBO_KEYS = [

    "hidden_layer_sizes",

    "activation",

    "learning_rate_init",

    "alpha",

    "batch_size"

]


def params_to_vector(
    params
):

    vector = []


    for key in TLBO_KEYS:

        options = (
            SEARCH_SPACE[
                key
            ]
        )


        current_position = 0


        for position, option in enumerate(
            options
        ):

            if option == params[
                key
            ]:

                current_position = position

                break


        vector.append(
            current_position
        )


    return np.array(
        vector,
        dtype=float
    )


def vector_to_params(
    vector
):

    params = {}


    for index, key in enumerate(
        TLBO_KEYS
    ):

        options = (
            SEARCH_SPACE[
                key
            ]
        )


        rounded_index = int(
            round(
                vector[index]
            )
        )


        rounded_index = max(

            0,

            min(
                rounded_index,
                len(options) - 1
            )

        )


        params[key] = (
            options[
                rounded_index
            ]
        )


    return params


# -----------------------------------------------------
# Create initial TLBO population
# -----------------------------------------------------

tlbo_population = []

tlbo_seen = set()


while len(
    tlbo_population
) < TLBO_POPULATION_SIZE:

    params = sample_random_configuration(
        tlbo_rng
    )


    signature = (
        configuration_signature(
            params
        )
    )


    if signature in tlbo_seen:

        continue


    tlbo_seen.add(
        signature
    )


    tlbo_population.append(
        params
    )


# -----------------------------------------------------
# Initial evaluations
# -----------------------------------------------------

tlbo_population_results = []


for params in tlbo_population:

    run = evaluate_configuration(
        params
    )


    tlbo_population_results.append(
        run
    )


tlbo_evaluations = (
    len(
        tlbo_population_results
    )
)


tlbo_best_score = min(

    result[
        "validation_rmse"
    ]

    for result in
    tlbo_population_results

)


tlbo_best_solution = min(

    tlbo_population_results,

    key=lambda x:
        x["validation_rmse"]

)["params"]


tlbo_convergence = []


tlbo_convergence.append({

    "Evaluation": tlbo_evaluations,

    "Generation": 0,

    "Phase": "Initial",

    "Best Validation RMSE": (
        tlbo_best_score
    )

})


# -----------------------------------------------------
# TLBO generations
# -----------------------------------------------------

for generation in range(
    1,
    TLBO_GENERATIONS + 1
):

    print(
        f"\nGeneration "
        f"{generation}/"
        f"{TLBO_GENERATIONS}"
    )


    # =============================================
    # TEACHER PHASE
    # =============================================

    print(
        "Teacher Phase"
    )


    teacher_result = min(

        tlbo_population_results,

        key=lambda x:
            x["validation_rmse"]

    )


    teacher_vector = params_to_vector(

        teacher_result[
            "params"
        ]

    )


    population_vectors = np.vstack([

        params_to_vector(
            result[
                "params"
            ]
        )

        for result in
        tlbo_population_results

    ])


    mean_vector = (
        population_vectors.mean(
            axis=0
        )
    )


    teacher_phase_results = []


    for current_result in (
        tlbo_population_results
    ):

        current_vector = params_to_vector(

            current_result[
                "params"
            ]

        )


        teaching_factor = int(
            tlbo_rng.integers(
                1,
                3
            )
        )


        random_vector = (
            tlbo_rng.random(
                len(
                    TLBO_KEYS
                )
            )
        )


        new_vector = (

            current_vector

            +

            random_vector
            * (
                teacher_vector
                -
                teaching_factor
                * mean_vector
            )

        )


        new_params = vector_to_params(
            new_vector
        )


        run = evaluate_configuration(
            new_params
        )


        tlbo_evaluations += 1


        teacher_phase_results.append(
            run
        )


        if (
            run[
                "validation_rmse"
            ]
            <
            current_result[
                "validation_rmse"
            ]
        ):

            replacement = run

        else:

            replacement = current_result


        # -------------------------------------------------
        # Replace current individual
        # -------------------------------------------------

        current_index = (
            tlbo_population_results.index(
                current_result
            )
        )


        tlbo_population_results[
            current_index
        ] = replacement


        tlbo_best_score = min(

            tlbo_best_score,

            run[
                "validation_rmse"
            ]

        )


        if (
            run[
                "validation_rmse"
            ]
            <=
            tlbo_best_score
        ):

            tlbo_best_solution = (
                run[
                    "params"
                ]
            )


        tlbo_convergence.append({

            "Evaluation": (
                tlbo_evaluations
            ),

            "Generation": generation,

            "Phase": "Teacher",

            "Best Validation RMSE": (
                tlbo_best_score
            )

        })


        print(

            f"Evaluation "
            f"{tlbo_evaluations:02d}/"
            f"{N_OPTIMIZER_EVALUATIONS}"
            f" - RMSE = "
            f"{run['validation_rmse']:.4f}"

        )


    # =============================================
    # LEARNER PHASE
    # =============================================

    print(
        "Learner Phase"
    )


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

            current_result[
                "params"
            ]

        )


        partner_index = int(

            tlbo_rng.integers(

                0,

                TLBO_POPULATION_SIZE

            )

        )


        while partner_index == learner_index:

            partner_index = int(

                tlbo_rng.integers(

                    0,

                    TLBO_POPULATION_SIZE

                )

            )


        partner_result = (
            tlbo_population_results[
                partner_index
            ]
        )


        partner_vector = params_to_vector(

            partner_result[
                "params"
            ]

        )


        random_vector = (
            tlbo_rng.random(
                len(
                    TLBO_KEYS
                )
            )
        )


        if (

            current_result[
                "validation_rmse"
            ]

            <

            partner_result[
                "validation_rmse"
            ]

        ):

            new_vector = (

                current_vector

                +

                random_vector
                * (
                    current_vector
                    - partner_vector
                )

            )

        else:

            new_vector = (

                current_vector

                +

                random_vector
                * (
                    partner_vector
                    - current_vector
                )

            )


        new_params = vector_to_params(
            new_vector
        )


        run = evaluate_configuration(
            new_params
        )


        tlbo_evaluations += 1


        learner_phase_results.append(
            run
        )


        if (
            run[
                "validation_rmse"
            ]
            <
            current_result[
                "validation_rmse"
            ]
        ):

            tlbo_population_results[
                learner_index
            ] = run


        tlbo_best_score = min(

            tlbo_best_score,

            run[
                "validation_rmse"
            ]

        )


        if (
            run[
                "validation_rmse"
            ]
            <=
            tlbo_best_score
        ):

            tlbo_best_solution = (
                run[
                    "params"
                ]
            )


        tlbo_convergence.append({

            "Evaluation": (
                tlbo_evaluations
            ),

            "Generation": generation,

            "Phase": "Learner",

            "Best Validation RMSE": (
                tlbo_best_score
            )

        })


        print(

            f"Evaluation "
            f"{tlbo_evaluations:02d}/"
            f"{N_OPTIMIZER_EVALUATIONS}"
            f" - RMSE = "
            f"{run['validation_rmse']:.4f}"

        )


tlbo_optimization_time = (
    time.time()
    - tlbo_start
)


tlbo_best_result = min(

    tlbo_population_results,

    key=lambda x:
        x["validation_rmse"]

)


tlbo_best_solution = (
    tlbo_best_result[
        "params"
    ]
)


tlbo_best_score = (
    tlbo_best_result[
        "validation_rmse"
    ]
)


tlbo_model = (
    tlbo_best_result[
        "model"
    ]
)


tlbo_training_runtime = (
    tlbo_best_result[
        "runtime"
    ]
)


print(
    "\nTLBO Optimization completed."
)


print(
    f"Optimization Time: "
    f"{tlbo_optimization_time:.2f} seconds"
)


print(
    f"Evaluations: "
    f"{tlbo_evaluations}"
)


print(
    f"Best Validation RMSE: "
    f"{tlbo_best_score:.4f}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    tlbo_best_solution
)


# =====================================================
# SIMULATED ANNEALING
# =====================================================
#
# SA is included as a methodologically distinct optimizer.
#
# WHY SA IS SUITABLE:
#
# The search space contains categorical and discrete
# hyperparameters. SA does not require a differentiable
# or continuous objective space.
#
# The optimizer keeps one current solution and explores
# neighbouring configurations.
#
# Better configurations are always accepted.
#
# Worse configurations can be accepted according to:
#
#       exp(-increase / temperature)
#
# Early in the optimization, the temperature is higher,
# allowing more exploration and occasional acceptance
# of worse configurations.
#
# As temperature decreases, the algorithm becomes more
# selective and focuses on exploitation.
#
# WHY SA MIGHT IMPROVE FORECASTING PERFORMANCE:
#
# Phase 2 showed:
#
# Training R²  = 0.8338
# Validation R² = 0.4544
#
# This indicates a sizeable generalization gap.
#
# SA can potentially locate a different combination of:
#
# - network size
# - learning rate
# - regularization strength
# - batch size
# - activation
#
# that gives a lower validation RMSE.
#
# For example, a smaller network with stronger
# regularization may generalize better than the Phase 1
# baseline.
#
# HOWEVER:
#
# SA is NOT guaranteed to improve the result.
#
# The thesis should determine its effectiveness empirically
# under the same computational budget as the other methods.
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "SIMULATED ANNEALING"
)

print(
    "=" * 80
)


sa_start = time.time()


sa_rng = np.random.default_rng(
    RANDOM_STATE + 4
)


# -----------------------------------------------------
# Initial solution
# -----------------------------------------------------

sa_current_params = (
    sample_random_configuration(
        sa_rng
    )
)


sa_current_run = (
    evaluate_configuration(
        sa_current_params
    )
)

sa_best_run = sa_current_run

sa_current_score = (
    sa_current_run[
        "validation_rmse"
    ]
)


sa_best_solution = (
    sa_current_params.copy()
)


sa_best_score = (
    sa_current_score
)


sa_evaluations = 1


sa_convergence = [

    {

        "Evaluation": 1,

        "Current Validation RMSE": (
            sa_current_score
        ),

        "Best Validation RMSE": (
            sa_best_score
        ),

        "Temperature": (
            SA_INITIAL_TEMPERATURE
        )

    }

]


print(

    f"Evaluation 01/"
    f"{SA_EVALUATIONS}"
    f" - RMSE = "
    f"{sa_current_score:.4f}"

)


# -----------------------------------------------------
# Annealing loop
# -----------------------------------------------------

for evaluation in range(
    2,
    SA_EVALUATIONS + 1
):

    temperature = (

        SA_INITIAL_TEMPERATURE

        *

        (
            SA_COOLING_RATE
            ** (
                evaluation - 2
            )
        )

    )


    # ---------------------------------------------
    # Generate neighbour
    # ---------------------------------------------

    neighbour_params = (
        mutate_configuration(

            sa_current_params,

            sa_rng,

            n_changes=1

        )
    )


    neighbour_run = (
        evaluate_configuration(
            neighbour_params
        )
    )


    neighbour_score = (
        neighbour_run[
            "validation_rmse"
        ]
    )


    sa_evaluations += 1


    # ---------------------------------------------
    # Decide whether to accept the neighbour
    #
    # Since this is a minimization problem:
    #
    # delta = neighbour - current
    #
    # delta <= 0:
    #     better solution -> accept
    #
    # delta > 0:
    #     worse solution -> probabilistic acceptance
    # ---------------------------------------------

    delta = (
        neighbour_score
        -
        sa_current_score
    )


    if delta <= 0:

        accept = True

    else:

        if temperature > 0:

            acceptance_probability = math.exp(

                -delta
                /
                temperature

            )

        else:

            acceptance_probability = 0.0


        accept = (
            sa_rng.random()
            <
            acceptance_probability
        )


    if accept:

        sa_current_params = (
            neighbour_params
        )

        sa_current_score = (
            neighbour_score
        )


    # ---------------------------------------------
    # Update global best
    # ---------------------------------------------

    if (
        neighbour_score
        <
        sa_best_score
    ):

        sa_best_score = (
            neighbour_score
        )

        sa_best_solution = (
            neighbour_params.copy()
        )

        sa_best_run = (
            neighbour_run
        )


    sa_convergence.append({

        "Evaluation": (
            sa_evaluations
        ),

        "Current Validation RMSE": (
            sa_current_score
        ),

        "Best Validation RMSE": (
            sa_best_score
        ),

        "Temperature": temperature

    })


    print(

        f"Evaluation "
        f"{evaluation:02d}/"
        f"{SA_EVALUATIONS}"
        f" - Current RMSE = "
        f"{sa_current_score:.4f}"
        f", Best RMSE = "
        f"{sa_best_score:.4f}"

    )


sa_optimization_time = (
    time.time()
    - sa_start
)


# -----------------------------------------------------
# Train best SA configuration
# -----------------------------------------------------



sa_model = (
    sa_best_run[
        "model"
    ]
)


sa_training_runtime = (
    sa_best_run[
        "runtime"
    ]
)


print(
    "\nSimulated Annealing completed."
)


print(
    f"Optimization Time: "
    f"{sa_optimization_time:.2f} seconds"
)


print(
    f"Evaluations: "
    f"{sa_evaluations}"
)


print(
    f"Best Validation RMSE: "
    f"{sa_best_score:.4f}"
)


print(
    "\nBest Hyperparameters:"
)


print(
    sa_best_solution
)


# =====================================================
# VERIFY EQUAL COMPUTATIONAL BUDGET
# =====================================================

hyperband_resource_units = (
    HYPERBAND_INITIAL_CANDIDATES
    * HYPERBAND_RESOURCES[0]
    +
    HYPERBAND_SURVIVORS_STAGE_1
    * HYPERBAND_RESOURCES[1]
    +
    HYPERBAND_SURVIVORS_STAGE_2
    * HYPERBAND_RESOURCES[2]
)


budget_check = pd.DataFrame({

    "Method": [

        "Random Search",

        "Bayesian Optimization",

        "Hyperband",

        "LEA",

        "TLBO",

        "Simulated Annealing"

    ],

    "Evaluations": [

        random_evaluations,

        N_OPTIMIZER_EVALUATIONS,

        (
            HYPERBAND_INITIAL_CANDIDATES
            +
            HYPERBAND_SURVIVORS_STAGE_1
            +
            HYPERBAND_SURVIVORS_STAGE_2
        ),

        lea_evaluations,

        tlbo_evaluations,

        sa_evaluations

    ],

    "Resource Units": [

        random_evaluations
        * MAX_ITER,

        N_OPTIMIZER_EVALUATIONS
        * MAX_ITER,

        hyperband_resource_units,

        lea_evaluations
        * MAX_ITER,

        tlbo_evaluations
        * MAX_ITER,

        sa_evaluations
        * MAX_ITER

    ]

})


print(
    "\n" + "=" * 80
)

print(
    "COMPUTATIONAL BUDGET CHECK"
)

print(
    "=" * 80
)

print(
    budget_check
)


assert all(

    budget_check[
        "Resource Units"
    ]
    ==
    TOTAL_RESOURCE_BUDGET

), (
    "ERROR: Optimizers do not have "
    "equal computational budgets."
)
# =====================================================
# FINAL DEVELOPMENT TRAINING FUNCTION
# =====================================================
#
# After each optimizer selects its hyperparameters, the
# model is retrained on:
#
#     Training + Validation
#
# The test set is evaluated only once.
#
# The scaler is fitted only on development data.
# =====================================================

def train_final_model_on_development(
    params
):

    X_development = pd.concat(

        [
            X_train,
            X_validation
        ]

    )


    y_development = pd.concat(

        [
            y_train,
            y_validation
        ]

    )


    # -------------------------------------------------
    # Fit scaler on development data only
    # -------------------------------------------------

    development_scaler = (
        StandardScaler()
    )


    X_development_scaled = (
        development_scaler.fit_transform(
            X_development
        )
    )


    X_test_final_scaled = (
        development_scaler.transform(
            X_test
        )
    )


    # -------------------------------------------------
    # Build final model
    # -------------------------------------------------

    model = build_mlp(
        params,
        max_iter=MAX_ITER
    )


    start_time = time.time()


    model.fit(

        X_development_scaled,

        y_development

    )


    training_runtime = (
        time.time()
        - start_time
    )


    # -------------------------------------------------
    # Test predictions
    # -------------------------------------------------

    test_predictions = (
        model.predict(
            X_test_final_scaled
        )
    )


    test_metrics = (
        calculate_metrics(

            y_test,

            test_predictions

        )
    )


    return {

        "model": model,

        "test_metrics": test_metrics,

        "test_predictions": (
            test_predictions
        ),

        "training_runtime": (
            training_runtime
        ),

        "iterations": (
            model.n_iter_
        ),

        "loss": (
            model.loss_
        )

    }


# =====================================================
# FINAL BASELINE TEST
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "FINAL BASELINE TEST EVALUATION"
)

print(
    "=" * 80
)


baseline_final_run = (
    train_final_model_on_development(
        baseline_params
    )
)


baseline_model = (
    baseline_final_run[
        "model"
    ]
)


baseline_test_metrics = (
    baseline_final_run[
        "test_metrics"
    ]
)


baseline_training_runtime = (
    baseline_final_run[
        "training_runtime"
    ]
)


print(
    "\nBaseline Test Performance"
)


print(
    f"MAE  : "
    f"{baseline_test_metrics['MAE']:.4f}"
)


print(
    f"RMSE : "
    f"{baseline_test_metrics['RMSE']:.4f}"
)


print(
    f"MSE  : "
    f"{baseline_test_metrics['MSE']:.4f}"
)


print(
    f"R²   : "
    f"{baseline_test_metrics['R2']:.4f}"
)


# =====================================================
# FINAL RANDOM SEARCH TEST
# =====================================================

print(
    "\nTraining final Random Search model..."
)


random_final_run = (
    train_final_model_on_development(
        random_best_params
    )
)


random_model = (
    random_final_run[
        "model"
    ]
)


random_test_metrics = (
    random_final_run[
        "test_metrics"
    ]
)


random_final_training_runtime = (
    random_final_run[
        "training_runtime"
    ]
)


# =====================================================
# FINAL BAYESIAN TEST
# =====================================================

print(
    "Training final Bayesian Optimization model..."
)


bayesian_final_run = (
    train_final_model_on_development(
        best_params
    )
)


bayesian_model = (
    bayesian_final_run[
        "model"
    ]
)


bayesian_test_metrics = (
    bayesian_final_run[
        "test_metrics"
    ]
)


bayesian_final_training_runtime = (
    bayesian_final_run[
        "training_runtime"
    ]
)


# =====================================================
# FINAL HYPERBAND TEST
# =====================================================

print(
    "Training final Hyperband model..."
)


hyperband_final_run = (
    train_final_model_on_development(
        hyperband_best_params
    )
)


hyperband_model = (
    hyperband_final_run[
        "model"
    ]
)


hyperband_test_metrics = (
    hyperband_final_run[
        "test_metrics"
    ]
)


hyperband_final_training_runtime = (
    hyperband_final_run[
        "training_runtime"
    ]
)


# =====================================================
# FINAL LEA TEST
# =====================================================

print(
    "Training final LEA model..."
)


lea_final_run = (
    train_final_model_on_development(
        lea_best_solution
    )
)


lea_model = (
    lea_final_run[
        "model"
    ]
)


lea_test_metrics = (
    lea_final_run[
        "test_metrics"
    ]
)


lea_final_training_runtime = (
    lea_final_run[
        "training_runtime"
    ]
)


# =====================================================
# FINAL TLBO TEST
# =====================================================

print(
    "Training final TLBO model..."
)


tlbo_final_run = (
    train_final_model_on_development(
        tlbo_best_solution
    )
)


tlbo_model = (
    tlbo_final_run[
        "model"
    ]
)


tlbo_test_metrics = (
    tlbo_final_run[
        "test_metrics"
    ]
)


tlbo_final_training_runtime = (
    tlbo_final_run[
        "training_runtime"
    ]
)


# =====================================================
# FINAL SA TEST
# =====================================================

print(
    "Training final Simulated Annealing model..."
)


sa_final_run = (
    train_final_model_on_development(
        sa_best_solution
    )
)


sa_model = (
    sa_final_run[
        "model"
    ]
)


sa_test_metrics = (
    sa_final_run[
        "test_metrics"
    ]
)


sa_final_training_runtime = (
    sa_final_run[
        "training_runtime"
    ]
)

# =====================================================
# SAVE EXPERIMENT RESULTS
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "COMPARISON OF OPTIMIZATION METHODS"
)

print(
    "=" * 80
)


RESULTS_DIR = "results"


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


baseline_test_rmse = (
    baseline_test_metrics[
        "RMSE"
    ]
)


comparison_results = pd.DataFrame({

    "Method": [

        "Baseline MLP",

        "Random Search",

        "Bayesian Optimization",

        "Hyperband",

        "LEA",

        "TLBO",

        "Simulated Annealing"

    ],


    "Validation RMSE": [

        baseline_validation_metrics[
            "RMSE"
        ],

        random_best_score,

        best_value,

        hyperband_best_score,

        lea_best_score,

        tlbo_best_score,

        sa_best_score

    ],


    "Test MAE": [

        baseline_test_metrics[
            "MAE"
        ],

        random_test_metrics[
            "MAE"
        ],

        bayesian_test_metrics[
            "MAE"
        ],

        hyperband_test_metrics[
            "MAE"
        ],

        lea_test_metrics[
            "MAE"
        ],

        tlbo_test_metrics[
            "MAE"
        ],

        sa_test_metrics[
            "MAE"
        ]

    ],


    "Test RMSE": [

        baseline_test_metrics[
            "RMSE"
        ],

        random_test_metrics[
            "RMSE"
        ],

        bayesian_test_metrics[
            "RMSE"
        ],

        hyperband_test_metrics[
            "RMSE"
        ],

        lea_test_metrics[
            "RMSE"
        ],

        tlbo_test_metrics[
            "RMSE"
        ],

        sa_test_metrics[
            "RMSE"
        ]

    ],


    "Test MSE": [

        baseline_test_metrics[
            "MSE"
        ],

        random_test_metrics[
            "MSE"
        ],

        bayesian_test_metrics[
            "MSE"
        ],

        hyperband_test_metrics[
            "MSE"
        ],

        lea_test_metrics[
            "MSE"
        ],

        tlbo_test_metrics[
            "MSE"
        ],

        sa_test_metrics[
            "MSE"
        ]

    ],


    "Test R2": [

        baseline_test_metrics[
            "R2"
        ],

        random_test_metrics[
            "R2"
        ],

        bayesian_test_metrics[
            "R2"
        ],

        hyperband_test_metrics[
            "R2"
        ],

        lea_test_metrics[
            "R2"
        ],

        tlbo_test_metrics[
            "R2"
        ],

        sa_test_metrics[
            "R2"
        ]

    ],


    "Test RMSE Improvement (%)": [

        0,

        (
            (
                baseline_test_rmse
                -
                random_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        ),

        (
            (
                baseline_test_rmse
                -
                bayesian_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        ),

        (
            (
                baseline_test_rmse
                -
                hyperband_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        ),

        (
            (
                baseline_test_rmse
                -
                lea_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        ),

        (
            (
                baseline_test_rmse
                -
                tlbo_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        ),

        (
            (
                baseline_test_rmse
                -
                sa_test_metrics[
                    "RMSE"
                ]
            )
            /
            baseline_test_rmse
            * 100
        )

    ],


    "Optimization Time (s)": [

        0,

        random_optimization_time,

        bayesian_optimization_time,

        hyperband_optimization_time,

        lea_optimization_time,

        tlbo_optimization_time,

        sa_optimization_time

    ],


    "Final Training Time (s)": [

        baseline_training_runtime,

        random_final_training_runtime,

        bayesian_final_training_runtime,

        hyperband_final_training_runtime,

        lea_final_training_runtime,

        tlbo_final_training_runtime,

        sa_final_training_runtime

    ],


    "Optimization Evaluations": [

        1,

        random_evaluations,

        N_OPTIMIZER_EVALUATIONS,

        hyperband_evaluations,

        lea_evaluations,

        tlbo_evaluations,

        sa_evaluations

    ],


    "Optimization Resource Units": [

        0,

        TOTAL_RESOURCE_BUDGET,

        TOTAL_RESOURCE_BUDGET,

        hyperband_resource_units,

        TOTAL_RESOURCE_BUDGET,

        TOTAL_RESOURCE_BUDGET,

        TOTAL_RESOURCE_BUDGET

    ],


    "Final Training Iterations": [

        baseline_final_run[
            "iterations"
        ],

        random_final_run[
            "iterations"
        ],

        bayesian_final_run[
            "iterations"
        ],

        hyperband_final_run[
            "iterations"
        ],

        lea_final_run[
            "iterations"
        ],

        tlbo_final_run[
            "iterations"
        ],

        sa_final_run[
            "iterations"
        ]

    ],


    "Final Training Loss": [

        baseline_final_run[
            "loss"
        ],

        random_final_run[
            "loss"
        ],

        bayesian_final_run[
            "loss"
        ],

        hyperband_final_run[
            "loss"
        ],

        lea_final_run[
            "loss"
        ],

        tlbo_final_run[
            "loss"
        ],

        sa_final_run[
            "loss"
        ]

    ],


    "Hidden Layers": [

        baseline_params[
            "hidden_layer_sizes"
        ],

        random_best_params[
            "hidden_layer_sizes"
        ],

        best_params[
            "hidden_layer_sizes"
        ],

        hyperband_best_params[
            "hidden_layer_sizes"
        ],

        lea_best_solution[
            "hidden_layer_sizes"
        ],

        tlbo_best_solution[
            "hidden_layer_sizes"
        ],

        sa_best_solution[
            "hidden_layer_sizes"
        ]

    ],


    "Activation": [

        baseline_params[
            "activation"
        ],

        random_best_params[
            "activation"
        ],

        best_params[
            "activation"
        ],

        hyperband_best_params[
            "activation"
        ],

        lea_best_solution[
            "activation"
        ],

        tlbo_best_solution[
            "activation"
        ],

        sa_best_solution[
            "activation"
        ]

    ],


    "Learning Rate": [

        baseline_params[
            "learning_rate_init"
        ],

        random_best_params[
            "learning_rate_init"
        ],

        best_params[
            "learning_rate_init"
        ],

        hyperband_best_params[
            "learning_rate_init"
        ],

        lea_best_solution[
            "learning_rate_init"
        ],

        tlbo_best_solution[
            "learning_rate_init"
        ],

        sa_best_solution[
            "learning_rate_init"
        ]

    ],


    "Alpha": [

        baseline_params[
            "alpha"
        ],

        random_best_params[
            "alpha"
        ],

        best_params[
            "alpha"
        ],

        hyperband_best_params[
            "alpha"
        ],

        lea_best_solution[
            "alpha"
        ],

        tlbo_best_solution[
            "alpha"
        ],

        sa_best_solution[
            "alpha"
        ]

    ],


    "Batch Size": [

        baseline_params[
            "batch_size"
        ],

        random_best_params[
            "batch_size"
        ],

        best_params[
            "batch_size"
        ],

        hyperband_best_params[
            "batch_size"
        ],

        lea_best_solution[
            "batch_size"
        ],

        tlbo_best_solution[
            "batch_size"
        ],

        sa_best_solution[
            "batch_size"
        ]

    ]

})


print(
    "\nComparison Results:"
)

print(
    comparison_results
)


comparison_results.to_csv(

    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_results.csv",

    index=False

)


# =====================================================
# SAVE CONVERGENCE HISTORY
# =====================================================

convergence_frames = []


# -----------------------------------------------------
# Random Search
# -----------------------------------------------------

for row in random_convergence:

    convergence_frames.append({

        "Method": "Random Search",

        "Evaluation": (
            row[
                "Evaluation"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]
        )

    })


# -----------------------------------------------------
# Bayesian Optimization
# -----------------------------------------------------

for row in bayesian_convergence:

    convergence_frames.append({

        "Method": "Bayesian Optimization",

        "Evaluation": (
            row[
                "Trial"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]
        )

    })


# -----------------------------------------------------
# Hyperband
# -----------------------------------------------------

for row in hyperband_convergence:

    convergence_frames.append({

        "Method": "Hyperband",

        "Evaluation": (
            row[
                "Evaluation"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]
        )

    })


# -----------------------------------------------------
# LEA
# -----------------------------------------------------

for row in lea_convergence:

    convergence_frames.append({

        "Method": "LEA",

        "Evaluation": (
            row[
                "Evaluation"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]

        )

    })


# -----------------------------------------------------
# TLBO
# -----------------------------------------------------

for row in tlbo_convergence:

    convergence_frames.append({

        "Method": "TLBO",

        "Evaluation": (
            row[
                "Evaluation"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]

        )

    })


# -----------------------------------------------------
# Simulated Annealing
# -----------------------------------------------------

for row in sa_convergence:

    convergence_frames.append({

        "Method": "Simulated Annealing",

        "Evaluation": (
            row[
                "Evaluation"
            ]
        ),

        "Best Validation RMSE": (
            row[
                "Best Validation RMSE"
            ]

        )

    })


convergence_df = pd.DataFrame(
    convergence_frames
)


convergence_df.to_csv(

    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_convergence.csv",

    index=False

)


# -----------------------------------------------------
# Save SA-specific convergence information
# -----------------------------------------------------

sa_convergence_df = pd.DataFrame(
    sa_convergence
)


sa_convergence_df.to_csv(

    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_sa_convergence.csv",

    index=False

)


# =====================================================
# VALIDATION RMSE COMPARISON PLOT
# =====================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison_results[
        "Method"
    ],

    comparison_results[
        "Validation RMSE"
    ]

)


plt.ylabel(
    "Validation RMSE"
)


plt.title(
    "Phase 3 - Validation RMSE Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.show()


# =====================================================
# TEST RMSE COMPARISON PLOT
# =====================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison_results[
        "Method"
    ],

    comparison_results[
        "Test RMSE"
    ]

)


plt.ylabel(
    "Test RMSE"
)


plt.title(
    "Phase 3 - Test RMSE Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.show()


# =====================================================
# TEST R² COMPARISON PLOT
# =====================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison_results[
        "Method"
    ],

    comparison_results[
        "Test R2"
    ]

)


plt.ylabel(
    "Test R²"
)


plt.title(
    "Phase 3 - Test R² Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.show()


# =====================================================
# CONVERGENCE PLOT
# =====================================================

plt.figure(
    figsize=(12, 7)
)


for method in convergence_df[
    "Method"
].unique():

    method_data = (
        convergence_df[
            convergence_df[
                "Method"
            ]
            == method
        ]
        .sort_values(
            "Evaluation"
        )
    )


    plt.plot(

        method_data[
            "Evaluation"
        ],

        method_data[
            "Best Validation RMSE"
        ],

        marker="o",

        markersize=3,

        label=method

    )


plt.xlabel(
    "Evaluation"
)


plt.ylabel(
    "Best Validation RMSE"
)


plt.title(
    "Phase 3 Optimizer Convergence"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()


# =====================================================
# FINAL BEST VALIDATION MODEL
# =====================================================

best_validation_method = (
    comparison_results.loc[
        comparison_results[
            "Validation RMSE"
        ].idxmin(),
        "Method"
    ]
)


best_validation_rmse = (
    comparison_results[
        "Validation RMSE"
    ].min()
)


# =====================================================
# BEST TEST MODEL
# =====================================================

best_test_method = (
    comparison_results.loc[
        comparison_results[
            "Test RMSE"
        ].idxmin(),
        "Method"
    ]
)


best_test_rmse = (
    comparison_results[
        "Test RMSE"
    ].min()
)


best_test_r2 = (
    comparison_results.loc[
        comparison_results[
            "Test RMSE"
        ].idxmin(),
        "Test R2"
    ]
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print(
    "\n" + "=" * 80
)

print(
    "PHASE 3 - HOUSEHOLD ELECTRIC POWER"
)

print(
    "OPTIMIZATION COMPLETED"
)

print(
    "=" * 80
)


print(
    "\nPrimary Optimization Metric:"
)

print(
    "Validation RMSE"
)

print(
    "Lower values indicate better forecasting performance."
)


print(
    "\nFrozen Phase 2 Pipeline:"
)


print(
    "Imputation:",
    FROZEN_IMPUTATION
)


print(
    "Lag Set:",
    LAG_HOURS
)


print(
    "Calendar Features:",
    "Raw"
)


print(
    "\nOptimizer Budget:"
)


print(
    f"{TOTAL_RESOURCE_BUDGET} "
    "training-iteration resource units per optimizer."
)


print(
    "\nValidation RMSE:"
)


print(
    f"Best Validation Method: "
    f"{best_validation_method}"
)


print(
    f"Best Validation RMSE: "
    f"{best_validation_rmse:.4f}"
)


print(
    "\nUntouched Test Set:"
)


print(
    f"Best Test RMSE: "
    f"{best_test_rmse:.4f}"
)


print(
    f"Corresponding Test R²: "
    f"{best_test_r2:.4f}"
)


print(
    "\nOptimization Methods:"
)


print(
    "Random Search"
)


print(
    "Bayesian Optimization"
)


print(
    "Hyperband"
)


print(
    "LEA"
)


print(
    "TLBO"
)


print(
    "Simulated Annealing"
)


print(
    "\nResults saved to:"
)


print(
    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_results.csv"
)


print(
    "\nConvergence history saved to:"
)


print(
    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_convergence.csv"
)


print(
    "\nSA convergence history saved to:"
)


print(
    f"{RESULTS_DIR}/"
    "individual_household_power_phase3_sa_convergence.csv"
)


print(
    "\n" + "=" * 80
)