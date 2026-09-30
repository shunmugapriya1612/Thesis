# =====================================================
# PHASE 1 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER
# CONSUMPTION (UCI 235)
# =====================================================
#
# Objective:
# Establish a reproducible baseline for one-step-ahead
# electricity consumption forecasting using an MLP.
#
# Dataset:
# Individual Household Electric Power Consumption
# UCI Machine Learning Repository - Dataset 235
#
# Task:
# Time-series regression / next-hour forecasting
#
# Target:
# Global_active_power
#
# Important:
# - Chronological train / validation / test split
# - No random shuffling
# - Test set remains untouched until final evaluation
# - Preprocessing is fitted only on training data
# =====================================================


# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
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

# Change this path if your filename is different.
DATA_PATH = (
    "/Users/priya/Documents/Thesis/Data/"
    "household_power_consumption.txt"
)

TARGET_COLUMN = "Global_active_power"

RANDOM_STATE = 42

# Chronological split:
# 64% training
# 16% validation
# 20% testing
TRAIN_RATIO = 0.64
VALIDATION_RATIO = 0.16
TEST_RATIO = 0.20

# Forecasting horizon:
# Predict electricity consumption for the next hour.
FORECAST_HORIZON = 1

# Lagged observations used as predictors.
#
# Since the data will be aggregated to hourly frequency,
# these represent:
#
# 1  hour
# 2  hours
# 3  hours
# 6  hours
# 12 hours
# 24 hours
LAG_HOURS = [0, 1, 2, 3, 6, 12, 24]


# =====================================================
# LOAD DATASET
# =====================================================

print("\n" + "=" * 80)
print("PHASE 1 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER")
print("CONSUMPTION")
print("=" * 80)

print("\nDataset Path:")
print(DATA_PATH)


# UCI household power dataset is semicolon-separated.
#
# The original dataset uses '?' for missing values,
# therefore '?' is explicitly treated as NaN.

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

print("\nData Types:")
print(data.dtypes)

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

# Remove the original Date and Time columns.
data = data.drop(
    columns=["Date", "Time"]
)

# Sort chronologically.
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
# CONVERT NUMERIC COLUMNS
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


print("\nNumeric Conversion Completed.")


# =====================================================
# SET TIMESTAMP AS INDEX
# =====================================================

data = data.set_index(
    "Timestamp"
)


# =====================================================
# HOURLY AGGREGATION
# =====================================================
#
# The original dataset contains minute-level observations.
#
# For a practical thesis experiment, we aggregate the
# observations to hourly frequency.
#
# This preserves the temporal structure while greatly
# reducing computational cost for MLP experiments.
# =====================================================

print("\nAggregating data to hourly frequency...")

hourly_data = data[
    numeric_columns
].resample("1h").mean()


print("\nHourly Dataset Shape:")
print(hourly_data.shape)


# =====================================================
# BASIC HOURLY DATA ANALYSIS
# =====================================================

print("\nHourly Missing Values:")
print(hourly_data.isnull().sum())

print("\nHourly Descriptive Statistics:")
print(hourly_data.describe())


# =====================================================
# CREATE LAG FEATURES
# =====================================================
#
# Forecasting must only use information that would have
# been available before the prediction time.
#
# Therefore, all electrical measurements are shifted into
# the past.
#
# This avoids using current/future electricity measurements
# as predictors for the target.
# =====================================================

print("\nCreating Lag Features...")

lag_source_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

feature_data = pd.DataFrame(
    index=hourly_data.index
)

for column in lag_source_columns:

    for lag in LAG_HOURS:

        feature_name = (
            f"{column}_lag_{lag}h"
        )

        feature_data[feature_name] = (
            hourly_data[column].shift(lag)
        )


# =====================================================
# CALENDAR FEATURES
# =====================================================
#
# These features describe the temporal position of each
# observation.
#
# They are known at prediction time and therefore do not
# introduce target leakage.
# =====================================================

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


# =====================================================
# CREATE FORECAST TARGET
# =====================================================
#
# Forecast horizon = 1 hour.
#
# At time t, the model receives historical information.
# The target is electricity consumption at time t+1.
# =====================================================

target = hourly_data[
    TARGET_COLUMN
].shift(-FORECAST_HORIZON)


# =====================================================
# COMBINE FEATURES AND TARGET
# =====================================================

model_data = feature_data.copy()

model_data["Target"] = target


# =====================================================
# REMOVE INVALID FORECAST ROWS
# =====================================================
#
# Rows near the beginning contain missing lag values.
# The final row contains a missing target because there is
# no future observation available for forecasting.
#
# We remove rows where the target is unavailable.
# Predictor missing values are handled later using training-
# fitted median imputation.
# =====================================================

model_data = model_data[
    model_data["Target"].notna()
].copy()


print("\nFinal Modeling Dataset Shape:")
print(model_data.shape)


# =====================================================
# FEATURE / TARGET SEPARATION
# =====================================================

X = model_data.drop(
    columns=["Target"]
)

y = model_data["Target"]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# =====================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# =====================================================
#
# IMPORTANT:
#
# We DO NOT use random train_test_split here.
#
# The observations are kept in chronological order.
#
# 64% -> training
# 16% -> validation
# 20% -> testing
# =====================================================

n_samples = len(X)

train_end = int(
    n_samples * TRAIN_RATIO
)

validation_end = int(
    n_samples * (TRAIN_RATIO + VALIDATION_RATIO)
)


X_train = X.iloc[
    :train_end
].copy()

y_train = y.iloc[
    :train_end
].copy()


X_validation = X.iloc[
    train_end:validation_end
].copy()

y_validation = y.iloc[
    train_end:validation_end
].copy()


X_test = X.iloc[
    validation_end:
].copy()

y_test = y.iloc[
    validation_end:
].copy()


print("\nChronological Data Split")

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nValidation Feature Shape:")
print(X_validation.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Time Range:")
print(
    X_train.index.min(),
    "to",
    X_train.index.max()
)

print("\nValidation Time Range:")
print(
    X_validation.index.min(),
    "to",
    X_validation.index.max()
)

print("\nTesting Time Range:")
print(
    X_test.index.min(),
    "to",
    X_test.index.max()
)


# =====================================================
# MEDIAN IMPUTATION
# =====================================================
#
# The imputer is fitted ONLY on the training set.
#
# This prevents information from the validation/test
# periods from influencing preprocessing.
# =====================================================

print("\nPerforming Median Imputation...")

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_validation_imputed = imputer.transform(
    X_validation
)

X_test_imputed = imputer.transform(
    X_test
)


print("Missing Value Imputation Completed.")


# =====================================================
# FEATURE STANDARDIZATION
# =====================================================
#
# StandardScaler is fitted ONLY on training data.
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imputed
)

X_validation_scaled = scaler.transform(
    X_validation_imputed
)

X_test_scaled = scaler.transform(
    X_test_imputed
)


print("Feature Standardization Completed.")


# =====================================================
# BASELINE MLP HYPERPARAMETERS
# =====================================================
#
# This is the same baseline philosophy used for the other
# thesis datasets.
# =====================================================

baseline_params = {

    "hidden_layer_sizes": (64, 32),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": "auto",

    "max_iter": 500,

    "random_state": RANDOM_STATE

}


print("\nBaseline Hyperparameters")

for parameter, value in baseline_params.items():

    print(
        f"{parameter}: {value}"
    )


# =====================================================
# BUILD BASELINE MLP
# =====================================================

print("\nBuilding Baseline MLP Model...")

baseline_model = MLPRegressor(
    **baseline_params
)


# =====================================================
# TRAIN BASELINE MODEL
# =====================================================

print("\nTraining Baseline MLP...")

baseline_training_start = time.time()


baseline_model.fit(
    X_train_scaled,
    y_train
)


baseline_training_runtime = (
    time.time()
    - baseline_training_start
)


print(
    f"\nBaseline Training Time: "
    f"{baseline_training_runtime:.2f} seconds"
)


# =====================================================
# VALIDATION PREDICTIONS
# =====================================================

print("\nMaking Validation Predictions...")

baseline_validation_predictions = (
    baseline_model.predict(
        X_validation_scaled
    )
)


# =====================================================
# BASELINE VALIDATION METRICS
# =====================================================

baseline_validation_mae = (
    mean_absolute_error(
        y_validation,
        baseline_validation_predictions
    )
)

baseline_validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        baseline_validation_predictions
    )
)

baseline_validation_mse = (
    mean_squared_error(
        y_validation,
        baseline_validation_predictions
    )
)

baseline_validation_r2 = (
    r2_score(
        y_validation,
        baseline_validation_predictions
    )
)


print("\n" + "=" * 60)
print("BASELINE VALIDATION PERFORMANCE")
print("=" * 60)

print(
    f"Validation MAE  : "
    f"{baseline_validation_mae:.4f}"
)

print(
    f"Validation RMSE : "
    f"{baseline_validation_rmse:.4f}"
)

print(
    f"Validation MSE  : "
    f"{baseline_validation_mse:.4f}"
)

print(
    f"Validation R²   : "
    f"{baseline_validation_r2:.4f}"
)


# =====================================================
# FINAL BASELINE TRAINING ON DEVELOPMENT DATA
# =====================================================
#
# After the baseline configuration has been established,
# train a fresh baseline model on:
#
#     Training + Validation
#
# The test set remains completely untouched.
#
# This gives the final baseline model access to all
# development data before testing.
# =====================================================

print(
    "\nTraining Final Baseline MLP "
    "on Development Data..."
)


# -----------------------------------------------------
# COMBINE TRAINING AND VALIDATION DATA
# -----------------------------------------------------

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


# -----------------------------------------------------
# FIT DEVELOPMENT PREPROCESSING
# -----------------------------------------------------
#
# Because the final baseline now uses train + validation,
# preprocessing is refitted on the development data.
#
# The test set is still completely isolated.
# -----------------------------------------------------

development_imputer = SimpleImputer(
    strategy="median"
)

X_development_imputed = (
    development_imputer.fit_transform(
        X_development
    )
)

X_test_final_imputed = (
    development_imputer.transform(
        X_test
    )
)


development_scaler = StandardScaler()

X_development_scaled = (
    development_scaler.fit_transform(
        X_development_imputed
    )
)

X_test_final_scaled = (
    development_scaler.transform(
        X_test_final_imputed
    )
)


# =====================================================
# TRAIN FINAL BASELINE MODEL
# =====================================================

final_baseline_model = MLPRegressor(
    **baseline_params
)


final_baseline_training_start = time.time()


final_baseline_model.fit(
    X_development_scaled,
    y_development
)


final_baseline_training_runtime = (
    time.time()
    - final_baseline_training_start
)


# =====================================================
# FINAL TEST PREDICTIONS
# =====================================================

print("\nMaking Final Test Predictions...")

baseline_test_predictions = (
    final_baseline_model.predict(
        X_test_final_scaled
    )
)


# =====================================================
# FINAL TEST METRICS
# =====================================================

baseline_test_mae = (
    mean_absolute_error(
        y_test,
        baseline_test_predictions
    )
)

baseline_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_test_predictions
    )
)

baseline_test_mse = (
    mean_squared_error(
        y_test,
        baseline_test_predictions
    )
)

baseline_test_r2 = (
    r2_score(
        y_test,
        baseline_test_predictions
    )
)


# =====================================================
# BASELINE TEST RESULTS
# =====================================================

print("\n" + "=" * 80)
print("BASELINE MLP RESULTS")
print("=" * 80)

print(
    f"MAE              : "
    f"{baseline_test_mae:.4f}"
)

print(
    f"RMSE             : "
    f"{baseline_test_rmse:.4f}"
)

print(
    f"MSE              : "
    f"{baseline_test_mse:.4f}"
)

print(
    f"R²               : "
    f"{baseline_test_r2:.4f}"
)

print(
    f"Training Time    : "
    f"{final_baseline_training_runtime:.2f} seconds"
)

print(
    f"Training Iterations: "
    f"{final_baseline_model.n_iter_}"
)

print(
    f"Final Training Loss: "
    f"{final_baseline_model.loss_:.6f}"
)


# =====================================================
# BASELINE MODEL INFORMATION
# =====================================================

print("\nBaseline Hyperparameters")

for parameter, value in baseline_params.items():

    print(
        f"{parameter}: {value}"
    )


# =====================================================
# ACTUAL VS PREDICTED - TEST SET
# =====================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual"
)

plt.plot(
    y_test.index,
    baseline_test_predictions,
    label="Predicted"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Global Active Power"
)

plt.title(
    "Baseline MLP - Actual vs Predicted "
    "Global Active Power"
)

plt.legend()

plt.tight_layout()

plt.show()


# =====================================================
# TRAINING LOSS CURVE
# =====================================================

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
# FINAL PHASE 1 SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("PHASE 1 - INDIVIDUAL HOUSEHOLD ELECTRIC POWER")
print("CONSUMPTION COMPLETED")
print("=" * 80)

print("\nTask:")
print("Next-Hour Global Active Power Forecasting")

print("\nModel:")
print("MLPRegressor")

print("\nPrimary Validation Metrics:")
print("MAE")
print("RMSE")
print("R²")

print("\nFinal Evaluation:")
print("Untouched Chronological Test Set")

print("\nBaseline Test Performance:")

print(
    f"MAE  = {baseline_test_mae:.4f}"
)

print(
    f"RMSE = {baseline_test_rmse:.4f}"
)

print(
    f"R²   = {baseline_test_r2:.4f}"
)

print("\n" + "=" * 80)