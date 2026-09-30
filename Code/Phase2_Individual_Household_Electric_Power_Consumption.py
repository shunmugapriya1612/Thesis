# =====================================================
# PHASE 2 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER
# CONSUMPTION (UCI 235)
# =====================================================
#
# Objective:
# Diagnose the Phase 1 baseline and make justified
# improvements before hyperparameter optimization.
#
# Task:
# Next-hour Global Active Power forecasting
#
# Model:
# MLPRegressor
#
# Primary Phase 2 selection metric:
# Validation RMSE
#
# Secondary metrics:
# MAE
# MSE
# R²
#
# Important:
# - Chronological train / validation / test split
# - No random shuffling
# - Test set is NOT used for Phase 2 selection
# - All imputation methods are causal
# - No future information is used for feature construction
# - Phase 3 will use the frozen Phase 2 pipeline
#
# Phase 2 experiments:
#
# A. Imputation
#    - Median
#    - Forward Fill
#    - Causal Rolling Median
#
# B. Lag structure
#    - Phase 1 lag set
#    - Extended lag set
#
# C. Calendar representation
#    - Raw calendar features
#    - Cyclical time features
#
# D. Naive forecasting benchmarks
#    - Persistence
#    - 24-hour seasonal
#    - 168-hour seasonal
#
# E. Diagnostics
#    - Target distribution
#    - Autocorrelation
#    - Feature correlation
#    - Temporal patterns
#    - Residual diagnostics
#
# F. Final Phase 2 pipeline selection
# =====================================================


# =====================================================
# IMPORT LIBRARIES
# =====================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

TRAIN_RATIO = 0.64
VALIDATION_RATIO = 0.16
TEST_RATIO = 0.20

FORECAST_HORIZON = 1

# -----------------------------------------------------
# Phase 1 lag structure
# -----------------------------------------------------

BASE_LAGS = [
    0,
    1,
    2,
    3,
    6,
    12,
    24
]

# -----------------------------------------------------
# Extended lag structure
#
# 48h  = 2 days
# 72h  = 3 days
# 168h = 1 week
# -----------------------------------------------------

EXTENDED_LAGS = [
    0,
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168
]

# -----------------------------------------------------
# Causal rolling window
# -----------------------------------------------------

ROLLING_WINDOW = 24

# -----------------------------------------------------
# Fixed MLP configuration
#
# Phase 2 is NOT hyperparameter optimization.
# We keep the Phase 1 MLP configuration fixed while
# changing preprocessing / feature representation.
# -----------------------------------------------------

BASELINE_PARAMS = {

    "hidden_layer_sizes": (64, 32),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": "auto",

    "max_iter": 500,

    "random_state": RANDOM_STATE

}


# =====================================================
# LOAD DATASET
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER")
print("CONSUMPTION")
print("=" * 80)

print("\nDataset Path:")
print(DATA_PATH)


data = pd.read_csv(
    DATA_PATH,
    sep=";",
    low_memory=False,
    na_values=["?"]
)


# =====================================================
# INITIAL DATASET ANALYSIS
# =====================================================

print("\nDataset Shape:")
print(data.shape)

print("\nFirst Five Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())


# =====================================================
# CREATE TIMESTAMP
# =====================================================

print("\nCreating Timestamp...")

data["Timestamp"] = pd.to_datetime(
    data["Date"] + " " + data["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)


data = data.drop(
    columns=["Date", "Time"]
)


data = data.sort_values(
    "Timestamp"
).reset_index(drop=True)


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

data = data.set_index(
    "Timestamp"
)


print("\nAggregating data to hourly frequency...")

hourly_data = data[
    numeric_columns
].resample("1h").mean()


print("\nHourly Dataset Shape:")
print(hourly_data.shape)


# =====================================================
# MISSING VALUE DIAGNOSTICS
# =====================================================

print("\nHourly Missing Values:")
print(
    hourly_data.isnull().sum()
)


missing_hour_mask = (
    hourly_data[numeric_columns]
    .isnull()
    .all(axis=1)
)


missing_complete_hours = (
    missing_hour_mask.sum()
)


total_hours = len(hourly_data)


missing_percentage = (
    missing_complete_hours
    / total_hours
    * 100
)


print(
    "\nComplete Hourly Missing Records:"
)

print(
    missing_complete_hours
)


print(
    f"Percentage of Hourly Records: "
    f"{missing_percentage:.2f}%"
)


# =====================================================
# HELPER FUNCTION:
# CREATE FORECAST FEATURES
# =====================================================

def create_model_data(
    feature_source,
    raw_target_source,
    lag_hours,
    use_cyclical_features=False
):
    """
    Create forecasting features and one-step-ahead target.

    Predict:
        Global_active_power(t+1)

    Using:
        information available at time t and earlier.
    """

    feature_data = pd.DataFrame(
        index=feature_source.index
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
    # Lag features
    # -------------------------------------------------

    for column in lag_source_columns:

        for lag in lag_hours:

            feature_name = (
                f"{column}_lag_{lag}h"
            )

            feature_data[feature_name] = (
                feature_source[column].shift(lag)
            )


    # -------------------------------------------------
    # Calendar features
    # -------------------------------------------------

    hour = feature_source.index.hour

    day_of_week = (
        feature_source.index.dayofweek
    )

    month = (
        feature_source.index.month
    )

    day_of_month = (
        feature_source.index.day
    )


    if use_cyclical_features:

        # ---------------------------------------------
        # Hour-of-day cyclical encoding
        # ---------------------------------------------

        feature_data["hour_sin"] = np.sin(
            2 * np.pi * hour / 24
        )

        feature_data["hour_cos"] = np.cos(
            2 * np.pi * hour / 24
        )


        # ---------------------------------------------
        # Day-of-week cyclical encoding
        # ---------------------------------------------

        feature_data["day_of_week_sin"] = np.sin(
            2 * np.pi * day_of_week / 7
        )

        feature_data["day_of_week_cos"] = np.cos(
            2 * np.pi * day_of_week / 7
        )


        # ---------------------------------------------
        # Month cyclical encoding
        # ---------------------------------------------

        feature_data["month_sin"] = np.sin(
            2 * np.pi * (month - 1) / 12
        )

        feature_data["month_cos"] = np.cos(
            2 * np.pi * (month - 1) / 12
        )

    else:

        # ---------------------------------------------
        # Original Phase 1 calendar features
        # ---------------------------------------------

        feature_data["hour"] = hour

        feature_data["day_of_week"] = (
            day_of_week
        )

        feature_data["month"] = month

        feature_data["day_of_month"] = (
            day_of_month
        )


    # -------------------------------------------------
    # One-step-ahead target
    #
    # Target at t = actual consumption at t+1
    # -------------------------------------------------

    target = raw_target_source[
        TARGET_COLUMN
    ].shift(
        -FORECAST_HORIZON
    )


    model_data = feature_data.copy()

    model_data["Target"] = target


    # -------------------------------------------------
    # Target must NOT be imputed.
    #
    # Rows without a real future target are removed.
    # -------------------------------------------------

    model_data = model_data[
        model_data["Target"].notna()
    ].copy()


    return model_data


# =====================================================
# CREATE REFERENCE DATASET
# =====================================================
#
# This reference dataset is used ONLY to determine
# chronological split timestamps.
#
# The split dates remain fixed across Phase 2 experiments.
# This makes the experiments comparable.
# =====================================================

reference_model_data = create_model_data(
    feature_source=hourly_data,
    raw_target_source=hourly_data,
    lag_hours=BASE_LAGS,
    use_cyclical_features=False
)


reference_n = len(
    reference_model_data
)


reference_train_end_position = int(
    reference_n * TRAIN_RATIO
)


reference_validation_end_position = int(
    reference_n
    * (TRAIN_RATIO + VALIDATION_RATIO)
)


TRAIN_END_TIME = (
    reference_model_data.index[
        reference_train_end_position - 1
    ]
)


VALIDATION_END_TIME = (
    reference_model_data.index[
        reference_validation_end_position - 1
    ]
)


print("\n" + "=" * 80)
print("FIXED CHRONOLOGICAL SPLIT")
print("=" * 80)

print("\nTraining ends:")
print(TRAIN_END_TIME)

print("\nValidation ends:")
print(VALIDATION_END_TIME)


# =====================================================
# HELPER FUNCTION:
# SPLIT CHRONOLOGICALLY
# =====================================================

def chronological_split(
    model_data
):
    """
    Split using fixed timestamps.

    This keeps the validation and test periods
    consistent across Phase 2 experiments.
    """

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
        model_data.index > VALIDATION_END_TIME
    )


    train_data = model_data.loc[
        train_mask
    ].copy()

    validation_data = model_data.loc[
        validation_mask
    ].copy()

    test_data = model_data.loc[
        test_mask
    ].copy()


    return (
        train_data,
        validation_data,
        test_data
    )


# =====================================================
# HELPER FUNCTION:
# CAUSAL FEATURE IMPUTATION
# =====================================================

def impute_features(
    model_data,
    method,
    reference_end_time,
    rolling_window=24
):
    """
    Apply missing-value treatment to model input features.

    Methods:
        median
        forward_fill
        causal_rolling_median

    Important:
    - Target is never imputed.
    - Median fallback values are calculated only from
      the reference period.
    - Forward fill only uses previous observations.
    - Rolling median uses only previous observations.
    """

    feature_columns = [
        column
        for column in model_data.columns
        if column != "Target"
    ]


    features = model_data[
        feature_columns
    ].copy()


    # -------------------------------------------------
    # Reference data for fallback medians
    # -------------------------------------------------

    reference_features = features.loc[
        features.index <= reference_end_time
    ]


    fallback_medians = (
        reference_features.median()
    )


    # =================================================
    # MEDIAN
    # =================================================

    if method == "median":

        imputer = SimpleImputer(
            strategy="median"
        )


        imputed_array = (
            imputer.fit(
                reference_features
            )
            .transform(features)
        )


        imputed_features = pd.DataFrame(
            imputed_array,
            index=features.index,
            columns=feature_columns
        )


    # =================================================
    # FORWARD FILL
    # =================================================

    elif method == "forward_fill":

        imputed_features = (
            features.ffill()
        )


        # -------------------------------------------------
        # Any values still missing at the beginning are
        # replaced with reference-period medians.
        #
        # This is only a fallback and does not look into
        # the future.
        # -------------------------------------------------

        imputed_features = (
            imputed_features.fillna(
                fallback_medians
            )
        )


    # =================================================
    # CAUSAL ROLLING MEDIAN
    # =================================================

    elif method == "causal_rolling_median":

        # ---------------------------------------------
        # Shift by one first.
        #
        # Therefore, the current observation is NEVER
        # used to calculate its own replacement.
        # ---------------------------------------------

        rolling_values = (
            features
            .shift(1)
            .rolling(
                window=rolling_window,
                min_periods=1
            )
            .median()
        )


        imputed_features = (
            features.copy()
        )


        missing_mask = (
            imputed_features.isna()
        )


        imputed_features[
            missing_mask
        ] = rolling_values[
            missing_mask
        ]


        # -------------------------------------------------
        # Remaining leading missing values use medians
        # calculated from the reference period.
        # -------------------------------------------------

        imputed_features = (
            imputed_features.fillna(
                fallback_medians
            )
        )


    else:

        raise ValueError(
            f"Unknown imputation method: {method}"
        )


    result = model_data.copy()

    result[
        feature_columns
    ] = imputed_features


    return result


# =====================================================
# HELPER FUNCTION:
# CALCULATE REGRESSION METRICS
# =====================================================

def calculate_metrics(
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    mse = mean_squared_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mse
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
# HELPER FUNCTION:
# TRAIN PHASE 2 MLP
# =====================================================

def train_phase2_model(
    model_data,
    reference_end_time=TRAIN_END_TIME
):
    """
    Train the fixed Phase 1 MLP on the training period
    and evaluate on validation data.
    """

    train_data, validation_data, _ = (
        chronological_split(
            model_data
        )
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


    # -------------------------------------------------
    # Safety imputation
    #
    # This handles any remaining values caused by the
    # beginning of lag sequences.
    #
    # Fitted only on training data.
    # -------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )


    X_train_imputed = (
        imputer.fit_transform(
            X_train
        )
    )


    X_validation_imputed = (
        imputer.transform(
            X_validation
        )
    )


    # -------------------------------------------------
    # Standardization
    # -------------------------------------------------

    scaler = StandardScaler()


    X_train_scaled = (
        scaler.fit_transform(
            X_train_imputed
        )
    )


    X_validation_scaled = (
        scaler.transform(
            X_validation_imputed
        )
    )


    # -------------------------------------------------
    # Build fixed MLP
    # -------------------------------------------------

    model = MLPRegressor(
        **BASELINE_PARAMS
    )


    training_start = time.time()


    model.fit(
        X_train_scaled,
        y_train
    )


    training_time = (
        time.time()
        - training_start
    )


    # -------------------------------------------------
    # Training predictions
    # -------------------------------------------------

    train_predictions = (
        model.predict(
            X_train_scaled
        )
    )


    # -------------------------------------------------
    # Validation predictions
    # -------------------------------------------------

    validation_predictions = (
        model.predict(
            X_validation_scaled
        )
    )


    train_metrics = calculate_metrics(
        y_train,
        train_predictions
    )


    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )


    return {

        "model": model,

        "training_time": training_time,

        "train_metrics": train_metrics,

        "validation_metrics": validation_metrics,

        "validation_predictions": (
            validation_predictions
        ),

        "y_validation": y_validation,

        "feature_count": len(
            feature_columns
        )

    }


# =====================================================
# PHASE 2-A
# IMPUTATION COMPARISON
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-A - IMPUTATION COMPARISON")
print("=" * 80)

print(
    "\nPrimary metric: Validation RMSE"
)

imputation_methods = [

    "median",

    "forward_fill",

    "causal_rolling_median"

]


imputation_results = []

imputation_runs = {}


for method in imputation_methods:

    print("\n" + "-" * 70)

    print(
        f"Testing Imputation: {method}"
    )

    print("-" * 70)


    # -------------------------------------------------
    # Create Phase 1-style feature representation
    # -------------------------------------------------

    raw_feature_data = create_model_data(
        feature_source=hourly_data,
        raw_target_source=hourly_data,
        lag_hours=BASE_LAGS,
        use_cyclical_features=False
    )


    # -------------------------------------------------
    # Apply selected imputation method
    # -------------------------------------------------

    imputed_model_data = impute_features(
        model_data=raw_feature_data,
        method=method,
        reference_end_time=TRAIN_END_TIME,
        rolling_window=ROLLING_WINDOW
    )


    # -------------------------------------------------
    # Train fixed MLP
    # -------------------------------------------------

    run = train_phase2_model(
        imputed_model_data
    )


    imputation_runs[
        method
    ] = {

        "model_data": imputed_model_data,

        "run": run

    }


    metrics = run[
        "validation_metrics"
    ]


    imputation_results.append({

        "Imputation": method,

        "Validation MAE": metrics["MAE"],

        "Validation RMSE": metrics["RMSE"],

        "Validation MSE": metrics["MSE"],

        "Validation R2": metrics["R2"],

        "Training Time (s)": (
            run["training_time"]
        )

    })


    print(
        f"Validation MAE  : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"Validation RMSE : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"Validation R²   : "
        f"{metrics['R2']:.4f}"
    )


imputation_results_df = pd.DataFrame(
    imputation_results
)


print(
    "\nImputation Comparison Results"
)

print(
    imputation_results_df
    .sort_values("Validation RMSE")
)


# =====================================================
# SELECT BEST IMPUTATION
# =====================================================

best_imputation_row = (
    imputation_results_df
    .sort_values(
        "Validation RMSE"
    )
    .iloc[0]
)


BEST_IMPUTATION = (
    best_imputation_row[
        "Imputation"
    ]
)


print(
    "\nSelected Imputation:"
)

print(
    BEST_IMPUTATION
)


best_imputed_model_data = (
    imputation_runs[
        BEST_IMPUTATION
    ]["model_data"]
)


# =====================================================
# PHASE 2-B
# TEMPORAL LAG COMPARISON
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-B - TEMPORAL LAG ANALYSIS")
print("=" * 80)

print(
    "\nComparing Phase 1 lag structure against"
    " extended temporal lags."
)


lag_configurations = {

    "Phase 1 Lags": BASE_LAGS,

    "Extended Lags": EXTENDED_LAGS

}


lag_results = []

lag_runs = {}


for configuration_name, lag_set in (
    lag_configurations.items()
):

    print("\n" + "-" * 70)

    print(
        f"Testing: {configuration_name}"
    )

    print(
        f"Lags: {lag_set}"
    )

    print("-" * 70)


    # -------------------------------------------------
    # Create model data
    # -------------------------------------------------

    raw_lag_model_data = create_model_data(

        feature_source=hourly_data,

        raw_target_source=hourly_data,

        lag_hours=lag_set,

        use_cyclical_features=False

    )


    # -------------------------------------------------
    # Apply selected imputation
    # -------------------------------------------------

    lag_model_data = impute_features(

        model_data=raw_lag_model_data,

        method=BEST_IMPUTATION,

        reference_end_time=TRAIN_END_TIME,

        rolling_window=ROLLING_WINDOW

    )


    # -------------------------------------------------
    # Train fixed MLP
    # -------------------------------------------------

    run = train_phase2_model(
        lag_model_data
    )


    lag_runs[
        configuration_name
    ] = {

        "model_data": lag_model_data,

        "run": run,

        "lags": lag_set

    }


    metrics = run[
        "validation_metrics"
    ]


    lag_results.append({

        "Lag Configuration": configuration_name,

        "Number of Lags": len(
            lag_set
        ),

        "Validation MAE": metrics["MAE"],

        "Validation RMSE": metrics["RMSE"],

        "Validation MSE": metrics["MSE"],

        "Validation R2": metrics["R2"],

        "Training Time (s)": (
            run["training_time"]
        )

    })


    print(
        f"Validation MAE  : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"Validation RMSE : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"Validation R²   : "
        f"{metrics['R2']:.4f}"
    )


lag_results_df = pd.DataFrame(
    lag_results
)


print(
    "\nLag Configuration Comparison"
)

print(
    lag_results_df
    .sort_values("Validation RMSE")
)


# =====================================================
# SELECT BEST LAG STRUCTURE
# =====================================================

best_lag_row = (
    lag_results_df
    .sort_values(
        "Validation RMSE"
    )
    .iloc[0]
)


BEST_LAG_CONFIGURATION = (
    best_lag_row[
        "Lag Configuration"
    ]
)


BEST_LAGS = lag_runs[
    BEST_LAG_CONFIGURATION
]["lags"]


print(
    "\nSelected Lag Configuration:"
)

print(
    BEST_LAG_CONFIGURATION
)

print(
    "Selected Lags:",
    BEST_LAGS
)


# =====================================================
# PHASE 2-C
# CYCLICAL TIME FEATURE COMPARISON
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-C - TIME FEATURE REPRESENTATION")
print("=" * 80)


calendar_configurations = {

    "Raw Calendar Features": False,

    "Cyclical Time Features": True

}


calendar_results = []

calendar_runs = {}


for configuration_name, use_cyclical in (
    calendar_configurations.items()
):

    print("\n" + "-" * 70)

    print(
        f"Testing: {configuration_name}"
    )

    print("-" * 70)


    # -------------------------------------------------
    # Build model data using selected lag structure
    # -------------------------------------------------

    raw_calendar_model_data = (
        create_model_data(

            feature_source=hourly_data,

            raw_target_source=hourly_data,

            lag_hours=BEST_LAGS,

            use_cyclical_features=(
                use_cyclical
            )

        )
    )


    # -------------------------------------------------
    # Apply selected imputation
    # -------------------------------------------------

    calendar_model_data = impute_features(

        model_data=raw_calendar_model_data,

        method=BEST_IMPUTATION,

        reference_end_time=TRAIN_END_TIME,

        rolling_window=ROLLING_WINDOW

    )


    # -------------------------------------------------
    # Train fixed MLP
    # -------------------------------------------------

    run = train_phase2_model(
        calendar_model_data
    )


    calendar_runs[
        configuration_name
    ] = {

        "model_data": calendar_model_data,

        "run": run,

        "cyclical": use_cyclical

    }


    metrics = run[
        "validation_metrics"
    ]


    calendar_results.append({

        "Time Representation": (
            configuration_name
        ),

        "Feature Count": (
            run["feature_count"]
        ),

        "Validation MAE": metrics["MAE"],

        "Validation RMSE": metrics["RMSE"],

        "Validation MSE": metrics["MSE"],

        "Validation R2": metrics["R2"],

        "Training Time (s)": (
            run["training_time"]
        )

    })


    print(
        f"Validation MAE  : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"Validation RMSE : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"Validation R²   : "
        f"{metrics['R2']:.4f}"
    )


calendar_results_df = pd.DataFrame(
    calendar_results
)


print(
    "\nTime Feature Comparison"
)

print(
    calendar_results_df
    .sort_values("Validation RMSE")
)


# =====================================================
# SELECT BEST TIME REPRESENTATION
# =====================================================

best_calendar_row = (
    calendar_results_df
    .sort_values(
        "Validation RMSE"
    )
    .iloc[0]
)


BEST_TIME_CONFIGURATION = (
    best_calendar_row[
        "Time Representation"
    ]
)


best_calendar_model_data = (
    calendar_runs[
        BEST_TIME_CONFIGURATION
    ]["model_data"]
)


print(
    "\nSelected Time Representation:"
)

print(
    BEST_TIME_CONFIGURATION
)


# =====================================================
# PHASE 2-D
# NAIVE FORECASTING BENCHMARKS
# =====================================================
#
# These are not competitors in Phase 3.
#
# They tell us whether the MLP is learning more than
# simple temporal persistence / seasonality.
#
# Evaluation here is performed on VALIDATION ONLY.
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-D - NAIVE FORECASTING BENCHMARKS")
print("=" * 80)


selected_model_data = (
    best_calendar_model_data
)


train_data, validation_data, test_data = (
    chronological_split(
        selected_model_data
    )
)


# -----------------------------------------------------
# Validation target
# -----------------------------------------------------

validation_target = (
    validation_data["Target"]
)


validation_indices = (
    validation_data.index
)


# -----------------------------------------------------
# Selected feature source after imputation
# -----------------------------------------------------
#
# We use the selected imputed hourly data to construct
# causal naive predictions.
# -----------------------------------------------------

selected_feature_source = (
    best_calendar_model_data
)


# -----------------------------------------------------
# Persistence benchmark
#
# Predict:
# y(t+1) = y(t)
#
# -----------------------------------------------------

persistence_predictions = (
    hourly_data[
        TARGET_COLUMN
    ]
    .reindex(validation_indices)
)


# -----------------------------------------------------
# 24-hour seasonal benchmark
#
# Target is t+1.
#
# Same hour on the previous day:
# t+1-24 = t-23
#
# Therefore shift(23).
# -----------------------------------------------------

seasonal_24_predictions = (
    hourly_data[
        TARGET_COLUMN
    ]
    .shift(23)
    .reindex(validation_indices)
)


# -----------------------------------------------------
# 168-hour weekly seasonal benchmark
#
# Same hour one week earlier:
# t+1-168 = t-167
#
# Therefore shift(167).
# -----------------------------------------------------

seasonal_168_predictions = (
    hourly_data[
        TARGET_COLUMN
    ]
    .shift(167)
    .reindex(validation_indices)
)


benchmark_predictions = {

    "Persistence": persistence_predictions,

    "24-hour Seasonal Naive": (
        seasonal_24_predictions
    ),

    "168-hour Seasonal Naive": (
        seasonal_168_predictions
    )

}


benchmark_results = []


for method_name, predictions in (
    benchmark_predictions.items()
):

    comparison = pd.concat(
        [
            validation_target,
            predictions.rename("Prediction")
        ],
        axis=1
    ).dropna()


    metrics = calculate_metrics(

        comparison[
            "Target"
        ],

        comparison[
            "Prediction"
        ]

    )


    benchmark_results.append({

        "Benchmark": method_name,

        "Validation MAE": (
            metrics["MAE"]
        ),

        "Validation RMSE": (
            metrics["RMSE"]
        ),

        "Validation MSE": (
            metrics["MSE"]
        ),

        "Validation R2": (
            metrics["R2"]
        )

    })


benchmark_results_df = pd.DataFrame(
    benchmark_results
)


print(
    "\nNaive Forecasting Results"
)

print(
    benchmark_results_df
    .sort_values("Validation RMSE")
)


# -----------------------------------------------------
# MLP validation performance for comparison
# -----------------------------------------------------

selected_phase2_validation_metrics = (
    calendar_runs[
        BEST_TIME_CONFIGURATION
    ]["run"]["validation_metrics"]
)


print(
    "\nSelected MLP Validation Performance:"
)

print(
    f"MAE  : "
    f"{selected_phase2_validation_metrics['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{selected_phase2_validation_metrics['RMSE']:.4f}"
)

print(
    f"R²   : "
    f"{selected_phase2_validation_metrics['R2']:.4f}"
)


# =====================================================
# PHASE 2-E
# TARGET AUTOCORRELATION DIAGNOSTICS
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-E - TEMPORAL DEPENDENCY DIAGNOSTICS")
print("=" * 80)


train_target_series = (
    hourly_data[
        TARGET_COLUMN
    ]
    .loc[
        :TRAIN_END_TIME
    ]
    .dropna()
)


autocorrelation_results = []


for lag in [1, 6, 12, 24, 48, 72, 168]:

    correlation = (
        train_target_series
        .autocorr(
            lag=lag
        )
    )


    autocorrelation_results.append({

        "Lag (hours)": lag,

        "Autocorrelation": correlation

    })


autocorrelation_df = pd.DataFrame(
    autocorrelation_results
)


print(
    "\nTarget Autocorrelation"
)

print(
    autocorrelation_df
)


# =====================================================
# PLOT AUTOCORRELATION VALUES
# =====================================================

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    autocorrelation_df[
        "Lag (hours)"
    ].astype(str),
    autocorrelation_df[
        "Autocorrelation"
    ]
)

plt.xlabel(
    "Lag (hours)"
)

plt.ylabel(
    "Autocorrelation"
)

plt.title(
    "Global Active Power Autocorrelation"
)

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2-F
# TEMPORAL TARGET PATTERNS
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-F - TEMPORAL PATTERN ANALYSIS")
print("=" * 80)


training_period = hourly_data.loc[
    :TRAIN_END_TIME
].copy()


# -----------------------------------------------------
# Add temporal labels
# -----------------------------------------------------

training_period["Hour"] = (
    training_period.index.hour
)

training_period["DayOfWeek"] = (
    training_period.index.dayofweek
)


# =====================================================
# AVERAGE POWER BY HOUR
# =====================================================

hourly_profile = (
    training_period
    .groupby("Hour")[
        TARGET_COLUMN
    ]
    .mean()
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    hourly_profile.index,
    hourly_profile.values
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Mean Global Active Power"
)

plt.title(
    "Average Global Active Power by Hour"
)

plt.xticks(
    range(24)
)

plt.tight_layout()

plt.show()


# =====================================================
# AVERAGE POWER BY DAY OF WEEK
# =====================================================

weekday_profile = (
    training_period
    .groupby("DayOfWeek")[
        TARGET_COLUMN
    ]
    .mean()
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    weekday_profile.index,
    weekday_profile.values,
    marker="o"
)

plt.xlabel(
    "Day of Week"
)

plt.ylabel(
    "Mean Global Active Power"
)

plt.title(
    "Average Global Active Power by Day of Week"
)

plt.xticks(
    range(7),
    [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ]
)

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2-G
# TRAIN / VALIDATION DISTRIBUTION DIAGNOSTICS
# =====================================================
#
# Test distribution is deliberately not inspected here.
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-G - TRAIN / VALIDATION DISTRIBUTION")
print("=" * 80)


train_target = (
    train_data["Target"]
    .dropna()
)


validation_target_series = (
    validation_data["Target"]
    .dropna()
)


distribution_summary = pd.DataFrame({

    "Training": [

        train_target.mean(),

        train_target.std(),

        train_target.min(),

        train_target.quantile(0.25),

        train_target.median(),

        train_target.quantile(0.75),

        train_target.max()

    ],

    "Validation": [

        validation_target_series.mean(),

        validation_target_series.std(),

        validation_target_series.min(),

        validation_target_series.quantile(0.25),

        validation_target_series.median(),

        validation_target_series.quantile(0.75),

        validation_target_series.max()

    ]

}, index=[

    "Mean",

    "Std",

    "Min",

    "25%",

    "Median",

    "75%",

    "Max"

])


print(
    distribution_summary
)


# =====================================================
# TARGET DISTRIBUTION PLOT
# =====================================================

plt.figure(
    figsize=(10, 5)
)

plt.hist(
    train_target,
    bins=50,
    alpha=0.6,
    label="Training"
)

plt.hist(
    validation_target_series,
    bins=50,
    alpha=0.6,
    label="Validation"
)

plt.xlabel(
    "Global Active Power"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Training vs Validation Target Distribution"
)

plt.legend()

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2-H
# ORIGINAL ELECTRICAL FEATURE CORRELATION
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-H - FEATURE CORRELATION DIAGNOSTICS")
print("=" * 80)


original_training = hourly_data.loc[
    :TRAIN_END_TIME,
    numeric_columns
].copy()


correlation_matrix = (
    original_training
    .corr(
        method="spearman"
    )
)


print(
    "\nSpearman Correlation Matrix:"
)

print(
    correlation_matrix
)


# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar(
    label="Spearman Correlation"
)

plt.xticks(
    range(
        len(numeric_columns)
    ),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(
        len(numeric_columns)
    ),
    numeric_columns
)

plt.title(
    "Spearman Correlation of Electrical Variables"
)

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2-I
# BASELINE RESIDUAL DIAGNOSTICS
# =====================================================
#
# Use the original Phase 1-style baseline:
#
# Median imputation
# Base lags
# Raw calendar features
# Fixed MLP
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-I - BASELINE RESIDUAL DIAGNOSTICS")
print("=" * 80)


baseline_phase2_run = (
    imputation_runs[
        "median"
    ]["run"]
)


baseline_validation_predictions = (
    baseline_phase2_run[
        "validation_predictions"
    ]
)


baseline_validation_target = (
    baseline_phase2_run[
        "y_validation"
    ]
)


baseline_residuals = (
    baseline_validation_target.values
    -
    baseline_validation_predictions
)


print(
    "\nBaseline Validation Residual Statistics"
)


print(
    f"Mean Residual : "
    f"{np.mean(baseline_residuals):.4f}"
)

print(
    f"Std Residual  : "
    f"{np.std(baseline_residuals):.4f}"
)

print(
    f"Min Residual  : "
    f"{np.min(baseline_residuals):.4f}"
)

print(
    f"Max Residual  : "
    f"{np.max(baseline_residuals):.4f}"
)


# =====================================================
# RESIDUALS OVER TIME
# =====================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    baseline_validation_target.index,
    baseline_residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Baseline MLP Validation Residuals Over Time"
)

plt.tight_layout()

plt.show()


# =====================================================
# PREDICTED VS ACTUAL
# =====================================================

plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    baseline_validation_predictions,
    baseline_validation_target,
    alpha=0.3
)


min_value = min(
    baseline_validation_predictions.min(),
    baseline_validation_target.min()
)


max_value = max(
    baseline_validation_predictions.max(),
    baseline_validation_target.max()
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


plt.xlabel(
    "Predicted Global Active Power"
)

plt.ylabel(
    "Actual Global Active Power"
)

plt.title(
    "Baseline MLP - Predicted vs Actual"
)

plt.tight_layout()

plt.show()


# =====================================================
# TRAINING LOSS CURVE
# =====================================================

baseline_model = (
    baseline_phase2_run[
        "model"
    ]
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    baseline_model.loss_curve_
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Training Loss"
)

plt.title(
    "Baseline MLP Training Loss"
)

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2-J
# TRAINING VS VALIDATION PERFORMANCE
# =====================================================

baseline_train_metrics = (
    baseline_phase2_run[
        "train_metrics"
    ]
)


baseline_validation_metrics = (
    baseline_phase2_run[
        "validation_metrics"
    ]
)


comparison_train_validation = pd.DataFrame({

    "Metric": [

        "MAE",

        "RMSE",

        "MSE",

        "R²"

    ],

    "Training": [

        baseline_train_metrics["MAE"],

        baseline_train_metrics["RMSE"],

        baseline_train_metrics["MSE"],

        baseline_train_metrics["R2"]

    ],

    "Validation": [

        baseline_validation_metrics["MAE"],

        baseline_validation_metrics["RMSE"],

        baseline_validation_metrics["MSE"],

        baseline_validation_metrics["R2"]

    ]

})


print(
    "\nBaseline Training vs Validation"
)

print(
    comparison_train_validation
)


# =====================================================
# HELPER FUNCTION:
# FINAL DEVELOPMENT TRAINING
# =====================================================

def train_final_phase2_model(
    model_data,
    development_end_time
):
    """
    Train the selected Phase 2 configuration on
    training + validation data and evaluate once
    on the untouched test period.
    """

    train_data, validation_data, test_data = (
        chronological_split(
            model_data
        )
    )


    development_data = pd.concat(
        [
            train_data,
            validation_data
        ]
    )


    feature_columns = [
        column
        for column in model_data.columns
        if column != "Target"
    ]


    X_development = development_data[
        feature_columns
    ]

    y_development = development_data[
        "Target"
    ]


    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        "Target"
    ]


    # -------------------------------------------------
    # Fit imputer only on development data
    # -------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )


    X_development_imputed = (
        imputer.fit_transform(
            X_development
        )
    )


    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )


    # -------------------------------------------------
    # Fit scaler only on development data
    # -------------------------------------------------

    scaler = StandardScaler()


    X_development_scaled = (
        scaler.fit_transform(
            X_development_imputed
        )
    )


    X_test_scaled = (
        scaler.transform(
            X_test_imputed
        )
    )


    # -------------------------------------------------
    # Final model
    # -------------------------------------------------

    model = MLPRegressor(
        **BASELINE_PARAMS
    )


    training_start = time.time()


    model.fit(
        X_development_scaled,
        y_development
    )


    training_time = (
        time.time()
        - training_start
    )


    # -------------------------------------------------
    # Test predictions
    # -------------------------------------------------

    test_predictions = (
        model.predict(
            X_test_scaled
        )
    )


    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )


    return {

        "model": model,

        "training_time": (
            training_time
        ),

        "test_metrics": test_metrics,

        "test_predictions": (
            test_predictions
        ),

        "y_test": y_test,

        "feature_count": len(
            feature_columns
        )

    }


# =====================================================
# PHASE 2-K
# FINAL SELECTED PHASE 2 CONFIGURATION
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2-K - FINAL PHASE 2 CONFIGURATION")
print("=" * 80)


print(
    "\nSelected Imputation:"
)

print(
    BEST_IMPUTATION
)


print(
    "\nSelected Lag Configuration:"
)

print(
    BEST_LAG_CONFIGURATION
)


print(
    "\nSelected Lags:"
)

print(
    BEST_LAGS
)


print(
    "\nSelected Time Representation:"
)

print(
    BEST_TIME_CONFIGURATION
)


# =====================================================
# FINAL TEST EVALUATION
# =====================================================
#
# The test set is accessed ONLY now.
# =====================================================

print(
    "\nTraining final Phase 2 model "
    "on development data..."
)


final_phase2_run = (
    train_final_phase2_model(
        model_data=selected_model_data,
        development_end_time=VALIDATION_END_TIME
    )
)


final_phase2_test_metrics = (
    final_phase2_run[
        "test_metrics"
    ]
)


print(
    "\n" + "=" * 80
)

print(
    "FINAL PHASE 2 TEST RESULTS"
)

print(
    "=" * 80
)


print(
    f"MAE  : "
    f"{final_phase2_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{final_phase2_test_metrics['RMSE']:.4f}"
)

print(
    f"MSE  : "
    f"{final_phase2_test_metrics['MSE']:.4f}"
)

print(
    f"R²   : "
    f"{final_phase2_test_metrics['R2']:.4f}"
)

print(
    f"Training Time : "
    f"{final_phase2_run['training_time']:.2f} seconds"
)


# =====================================================
# TEST ACTUAL VS PREDICTED
# =====================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    final_phase2_run[
        "y_test"
    ].index,

    final_phase2_run[
        "y_test"
    ].values,

    label="Actual"
)


plt.plot(
    final_phase2_run[
        "y_test"
    ].index,

    final_phase2_run[
        "test_predictions"
    ],

    label="Predicted"
)


plt.xlabel(
    "Time"
)

plt.ylabel(
    "Global Active Power"
)

plt.title(
    "Final Phase 2 Model - Actual vs Predicted"
)

plt.legend()

plt.tight_layout()

plt.show()


# =====================================================
# PHASE 2 FINAL SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("PHASE 2 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER")
print("CONSUMPTION COMPLETED")
print("=" * 80)


print(
    "\nPRIMARY SELECTION METRIC:"
)

print(
    "Validation RMSE"
)


print(
    "\nSelected Imputation:"
)

print(
    BEST_IMPUTATION
)


print(
    "\nSelected Lag Configuration:"
)

print(
    BEST_LAG_CONFIGURATION
)


print(
    "\nSelected Lag Set:"
)

print(
    BEST_LAGS
)


print(
    "\nSelected Time Representation:"
)

print(
    BEST_TIME_CONFIGURATION
)


print(
    "\nFinal Phase 2 Test Performance:"
)


print(
    f"MAE  = "
    f"{final_phase2_test_metrics['MAE']:.4f}"
)


print(
    f"RMSE = "
    f"{final_phase2_test_metrics['RMSE']:.4f}"
)


print(
    f"MSE  = "
    f"{final_phase2_test_metrics['MSE']:.4f}"
)


print(
    f"R²   = "
    f"{final_phase2_test_metrics['R2']:.4f}"
)


print(
    "\nPhase 2 pipeline is now frozen."
)


print(
    "\nThis frozen pipeline should be used unchanged "
    "for Phase 3 optimizer comparison."
)


print(
    "\nPhase 3 methods:"
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
    "\n" + "=" * 80
)