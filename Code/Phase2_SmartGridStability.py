# =====================================================
# SMART GRID STABILITY DATASET
# =====================================================
#
# Dataset Name:
# Smart Grid Stability
#
# Source:
# UCI Machine Learning Repository
# Dataset ID: 471
#
# Original Machine Learning Task:
# Classification
#
# Current Machine Learning Task:
# Binary Classification
#
# =====================================================
# DATASET DESCRIPTION
# =====================================================
#
# The Smart Grid Stability dataset contains observations
# describing the stability behaviour of an electrical
# power grid.
#
# The dataset contains electrical grid parameters related
# to the operating conditions and stability behaviour of
# the system.
#
# The dataset contains 10,000 observations and 13 original
# variables.
#
# Twelve variables are continuous predictor variables,
# while the original dataset contains two target variables:
#
# • stab  - continuous stability measure
# • stabf - binary stability classification
#
# During Phase 1, the original "stabf" target variable was
# renamed to "Target" to maintain a consistent target
# variable name across the datasets used in this study.
#
# =====================================================
# INPUT FEATURES
# =====================================================
#
# The predictor variables used for classification are:
#
# • tau1
# • tau2
# • tau3
# • tau4
# • p1
# • p2
# • p3
# • p4
# • g1
# • g2
# • g3
# • g4
#
# All twelve predictor variables are continuous numerical
# variables.
#
# =====================================================
# TARGET VARIABLE
# =====================================================
#
# Target:
# Target
#
# Number of Classes:
# 2
#
# The Target variable represents the binary stability
# condition of the electrical grid:
#
# • stable
# • unstable
#
# The original "stabf" variable was used as the
# classification target during Phase 1 and renamed to
# "Target" in the processed dataset.
#
# The continuous "stab" variable is not used as an input
# feature because it represents the continuous stability
# target from the original dataset.
#
# Including "stab" as an input feature could introduce
# target leakage because it contains information directly
# associated with the stability target.
#
# =====================================================
# TARGET DISTRIBUTION
# =====================================================
#
# The processed dataset contains the following class
# distribution:
#
# • unstable: 6,380 observations
# • stable:   3,620 observations
#
# The classes are therefore not equally distributed.
#
# Weighted F1-score is consequently used as the primary
# optimization metric so that the contribution of both
# classes is considered according to their frequency.
#
# =====================================================
# WHY THIS DATASET WAS SELECTED
# =====================================================
#
# The Smart Grid Stability dataset provides a binary
# classification problem related to electrical grid
# stability.
#
# Its continuous numerical feature space provides a
# suitable environment for evaluating the MLP classifier
# and different hyperparameter optimization algorithms.
#
# The dataset also provides a different classification
# problem from the Electricity dataset, allowing the
# optimization methods to be evaluated across different
# electrical-grid-related datasets.
# =====================================================


import random
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)

import time
import os

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import warnings

from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)


# =====================================================
# LOAD PROCESSED DATASET
# =====================================================
#
# Load the Smart Grid Stability dataset prepared during
# Phase 1.
#
# The processed dataset contains the twelve predictor
# variables, the original continuous "stab" variable,
# and the renamed binary classification target "Target".
# =====================================================

dataset = pd.read_csv(
    "processed_datasets/smart_grid_stability_processed.csv"
)

print("\nDataset Shape:")
print(dataset.shape)

print("\nFirst Five Rows:")
print(dataset.head())


RESULTS_DIR = "results"
VISUALIZATION_DIR = os.path.join(
    RESULTS_DIR,
    "phase2_smart_grid_stability",
    "visualizations"
)

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

os.makedirs(

    RESULTS_DIR,

    exist_ok=True

)


os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# =====================================================
# FEATURE MATRIX AND TARGET VARIABLE
# =====================================================
#
# Separate the predictor variables (X) from the target
# variable (y).
#
# The "Target" column represents the binary stability
# class that the MLP classifier will learn to predict.
#
# The original "stab" variable is also excluded from the
# feature matrix because it represents the continuous
# stability target from the original dataset.
#
# Excluding "stab" prevents target leakage and ensures that
# the MLP uses only the intended predictor variables.
# =====================================================

X = dataset.drop(
    columns=[
        "Target",
        "stab"
    ]
)

y = dataset["Target"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(
    y.value_counts()
)


# =====================================================
# FEATURE DATA TYPES
# =====================================================
#
# Inspect the data types of the predictor variables.
#
# The Smart Grid Stability dataset contains continuous
# numerical predictor variables, so categorical encoding
# is not required.
# =====================================================

print("\nFeature Data Types:")
print(X.dtypes)


# =====================================================
# ONE-HOT ENCODING
# =====================================================
#
# The Smart Grid Stability dataset contains only
# continuous numerical predictor variables.
#
# Therefore, no one-hot encoding is required.
#
# The Target variable contains the class labels "stable"
# and "unstable". It is retained as the classification
# target because MLPClassifier can directly handle
# categorical class labels.
# =====================================================


# =====================================================
# TRAIN-TEST SPLIT
# =====================================================
#
# Split the dataset into training and testing subsets.
#
# Twenty percent of the observations are reserved for
# final model evaluation.
#
# Stratified sampling is used to preserve the original
# distribution of the stable and unstable classes in both
# the training and testing datasets.
#
# A fixed random state is used to ensure that the same
# train-test split can be reproduced across experiments.
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    stratify=y,

    random_state=42

)

print("\nTraining Feature Shape:")
print(X_train.shape)

print("Testing Feature Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("Testing Target Shape:")
print(y_test.shape)


# =====================================================
# FEATURE STANDARDIZATION
# =====================================================
#
# Feature standardization scales the numerical predictor
# variables so that they have approximately zero mean and
# unit variance.
#
# Standardization is important for MLP classifiers because
# neural network optimization can be affected when input
# features have substantially different scales.
#
# The StandardScaler is fitted only on the training data.
#
# The same transformation is then applied to the testing
# data to prevent information from the testing dataset from
# influencing the preprocessing process.
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

print("\nFeature Standardization Completed.")

# =====================================================
# MLP HYPERPARAMETER SEARCH SPACE
# =====================================================
#
# Define the hyperparameter search space used by all
# optimization algorithms in this study. Each optimization
# method explores different combinations from this common
# search space, ensuring a fair comparison under identical
# experimental conditions.
#
# Hyperparameters included in the search space:
#
# • Hidden layer architecture
# • Activation function
# • Learning rate
# • L2 regularization strength (alpha)
# • Batch size
#
# The Adam optimizer is fixed throughout the study to
# ensure that differences in model performance are due to
# the optimization algorithms rather than changes in the
# optimization solver.
#
# The same hyperparameter search space is used for the
# Electricity dataset to ensure consistency across all
# optimization algorithms and facilitate fair performance
# comparisons.
# =====================================================

SEARCH_SPACE = {

    "hidden_layer_sizes": [

        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32)

    ],

    "activation": [

        "relu",
        "tanh"

    ],

    "solver": [

        "adam"

    ],

    "learning_rate_init": [

        0.0001,
        0.001

    ],

    "alpha": [

        0.0001,
        0.001,
        0.01

    ],

    "batch_size": [

        32,
        64,
        128

    ]

}

# =====================================================
# BUILD MLP MODEL
# =====================================================
#
# Construct an MLP classifier using a specified set of
# hyperparameters. This function is shared by the baseline
# model and all optimization algorithms to ensure that
# every experiment uses the same network architecture and
# training procedure.
#
# By using a single model-building function, only the
# hyperparameter values change between experiments while
# the underlying implementation remains identical.
# =====================================================



def build_mlp(params):

    return MLPClassifier(

        hidden_layer_sizes=params["hidden_layer_sizes"],

        activation=params["activation"],

        solver=params.get("solver", "adam"),

        learning_rate_init=params["learning_rate_init"],

        alpha=params["alpha"],

        batch_size=params["batch_size"],

        max_iter=500,

        random_state=42

    )

# =====================================================
# EVALUATE MODEL
# =====================================================
#
# Evaluate the trained MLP classifier on the testing
# dataset using multiple classification metrics.
#
# The following performance measures are reported:
#
# • Accuracy
# • Precision (Weighted)
# • Recall (Weighted)
# • F1-Score (Weighted)
#
# A confusion matrix and classification report are also
# generated to provide a detailed assessment of the
# classifier's predictive performance.
# =====================================================



def evaluate_model(model, X_test, y_test, method_name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )
    

    print(f"\n========== {method_name.upper()} RESULTS ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    print(

        classification_report(
            y_test,
            predictions
        )

    )
    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return accuracy, precision, recall, f1
# =====================================================
# TRAIN MLP MODEL
# =====================================================
#
# Train an MLP classifier using the supplied training
# dataset and measure the model training time.
#
# Returns:
# • Trained model
# • Training time (seconds)
# =====================================================

def train_mlp_model(model, X_train, y_train):

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    end = time.time()

    training_time = end - start

    return model, training_time

# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

def plot_training_loss(model, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.grid(True)

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

def print_optimization_summary(
    method_name,
    best_params,
    best_cv_score,        
    optimization_time,
    training_time
):

    print(f"\nBest Hyperparameters Found By {method_name}\n")

    for key, value in best_params.items():

            print(f"{key}: {value}")

    if best_cv_score is not None:

        print(
                f"\nBest Cross Validation F1 Score: "
                f"{best_cv_score:.4f}"
            )

    print(
            f"\nOptimization Time : "
            f"{optimization_time:.2f} seconds"
        )

    print(
            f"Training Time     : "
            f"{training_time:.2f} seconds"
        )
    
# =====================================================
# BASELINE MLP MODEL
# =====================================================
#
# The baseline MLP provides a reference performance
# against which all hyperparameter optimization methods
# are compared.
#
# The baseline uses a fixed set of MLP hyperparameters
# rather than performing hyperparameter optimization.
#
# The same baseline configuration is used across the
# datasets to maintain consistency in the experimental
# evaluation.
# =====================================================

print("\nBuilding Baseline MLP Model...")

baseline_model = MLPClassifier(

    hidden_layer_sizes=(64,),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    alpha=0.0001,

    batch_size=32,

    max_iter=500,

    random_state=42

)
# =====================================================
# TRAIN BASELINE MLP
# =====================================================
#
# Train the baseline MLP using the complete training
# dataset and record the training time.
# =====================================================

print("\nTraining Baseline MLP...")

baseline_model, baseline_training_runtime = train_mlp_model(

    baseline_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{baseline_training_runtime:.2f} seconds"
)
# =====================================================
# BASELINE MODEL EVALUATION
# =====================================================
#
# Evaluate the baseline MLP on the independent testing
# dataset using the same metrics that will be used for
# all optimization algorithms.
# =====================================================

print("\nMaking Predictions...")

baseline_accuracy, \
baseline_precision, \
baseline_recall, \
baseline_f1 = evaluate_model(

    baseline_model,

    X_test,

    y_test,

    "Baseline MLP"

)
# =====================================================
# VISUALIZE BASELINE TRAINING LOSS
# =====================================================

plot_training_loss(
    baseline_model,
    "Baseline MLP Training Loss",
    "01_baseline_mlp_training_loss.png"
)
# =====================================================
# BASELINE HYPERPARAMETERS
# =====================================================

print("\nBaseline Hyperparameters")

print(
    "hidden_layer_sizes: (64,)"
)

print(
    "activation: relu"
)

print(
    "solver: adam"
)

print(
    "learning_rate_init: 0.001"
)

print(
    "alpha: 0.0001"
)

print(
    "batch_size: 32"
)
# =====================================================
# RANDOM SEARCH OPTIMIZATION
# =====================================================
#
# Random Search is a hyperparameter optimization method
# that randomly samples candidate combinations from the
# predefined MLP hyperparameter search space.
#
# Each sampled configuration is evaluated using
# cross-validation, and the configuration achieving the
# highest weighted F1-score is selected as the best
# solution.
#
# Unlike the Baseline MLP, which uses one fixed
# hyperparameter configuration, Random Search explores
# multiple configurations to identify a potentially
# better-performing MLP model.
#
# -----------------------------------------------------
# WHY RANDOM SEARCH IS USED
# -----------------------------------------------------
#
# Random Search provides a simple and computationally
# efficient method for exploring the MLP hyperparameter
# search space.
#
# It is used as a reference optimization method against
# which the more advanced optimization algorithms can be
# compared.
#
# Unlike Grid Search, which evaluates every possible
# combination, Random Search evaluates only a predefined
# number of randomly selected configurations.
#
# Compared with Bayesian Optimization, Random Search does
# not use information from previous evaluations to guide
# subsequent trials.
#
# Compared with Hyperband, Random Search does not
# progressively allocate additional computational
# resources to promising configurations.
#
# Compared with LEA and TLBO, Random Search does not
# iteratively improve a population of candidate solutions.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Simple to implement and understand.
#
# • Can explore a large search space without evaluating
#   every possible combination.
#
# • Provides a useful baseline for comparison with more
#   advanced optimization algorithms.
#
# • Suitable for discrete MLP hyperparameter spaces.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Candidate configurations are selected independently.
#
# • Previous trial results are not used to guide later
#   trials.
#
# • Some sampled configurations may provide poor
#   performance.
#
# • Optimization time increases as the number of trials
#   increases.
# =====================================================
# =====================================================
# RANDOM SEARCH CONFIGURATION
# =====================================================
#
# Twenty candidate configurations are sampled from the
# common MLP hyperparameter search space.
#
# The same search space and evaluation budget are used
# across the optimization algorithms to maintain a fair
# experimental comparison.
#
# Five-fold cross-validation is used to evaluate each
# candidate configuration.
#
# Weighted F1-score is used as the optimization metric
# because the Smart Grid Stability dataset contains an
# unequal distribution of stable and unstable classes.
# =====================================================

from sklearn.model_selection import RandomizedSearchCV
# =====================================================
# BUILD RANDOM SEARCH OPTIMIZER
# =====================================================

print("\nBuilding Random Search Optimizer...")

random_search = RandomizedSearchCV(

    estimator=MLPClassifier(
        max_iter=500,
        random_state=42
    ),

    param_distributions=SEARCH_SPACE,

    n_iter=5,

    scoring="f1_weighted",

    cv=5,

    random_state=42,

    n_jobs=-1

)
# =====================================================
# START RANDOM SEARCH
# =====================================================
#
# Random Search evaluates 20 randomly selected
# hyperparameter configurations.
#
# The weighted F1-score obtained through 5-fold
# cross-validation is used to determine the best
# configuration.
# =====================================================

print("\nStarting Random Search...")

start = time.time()

random_search.fit(
    X_train,
    y_train
)

end = time.time()

random_search_runtime = end - start

print("\nRandom Search completed.")

print(
    f"Optimization Time: "
    f"{random_search_runtime:.2f} seconds"
)
# =====================================================
# RETRIEVE BEST HYPERPARAMETERS
# =====================================================
#
# Retrieve the hyperparameter configuration that achieved
# the highest weighted F1-score during Random Search.
# =====================================================

best_random_params = random_search.best_params_

print("\nBest Hyperparameters Found By Random Search\n")

for key, value in best_random_params.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{random_search.best_score_:.4f}"
)
# =====================================================
# BUILD RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the final MLP classifier using the best
# hyperparameters identified by Random Search.
#
# The common build_mlp() function is used to maintain
# consistency with the Baseline MLP and other optimization
# algorithms.
# =====================================================

print("\nBuilding Random Search Optimized MLP Model...")

random_search_model = build_mlp(
    best_random_params
)
# =====================================================
# TRAIN RANDOM SEARCH OPTIMIZED MLP
# =====================================================
#
# Train the optimized MLP using the complete training
# dataset.
#
# The final model training time is recorded separately
# from the Random Search optimization time.
# =====================================================

print("\nTraining Random Search Optimized MLP...")

random_search_model, random_search_training_runtime = train_mlp_model(

    random_search_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{random_search_training_runtime:.2f} seconds"
)
# =====================================================
# RANDOM SEARCH MODEL EVALUATION
# =====================================================
#
# Evaluate the optimized MLP on the independent testing
# dataset using the same metrics as the Baseline MLP.
# =====================================================

print("\nMaking Predictions...")

random_search_accuracy, \
random_search_precision, \
random_search_recall, \
random_search_f1 = evaluate_model(

    random_search_model,

    X_test,

    y_test,

    "Random Search"

)
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

plot_training_loss(
    random_search_model,
    "Random Search Optimized MLP Training Loss",
    "02_random_search_mlp_training_loss.png"
)
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Random Search",

    best_params=best_random_params,

    best_cv_score=random_search.best_score_,

    optimization_time=random_search_runtime,

    training_time=random_search_training_runtime

)
# =====================================================
# BAYESIAN OPTIMIZATION USING OPTUNA
# =====================================================
#
# Optuna is used as a Bayesian hyperparameter
# optimization framework to identify promising MLP
# hyperparameter configurations.
#
# Unlike Random Search, which samples configurations
# independently, Optuna uses information obtained from
# previously evaluated trials to guide subsequent trials
# towards promising regions of the search space.
#
# Each candidate configuration is evaluated using
# weighted F1-score obtained from 5-fold cross-validation.
#
# -----------------------------------------------------
# WHY OPTUNA IS USED
# -----------------------------------------------------
#
# The Smart Grid Stability dataset provides a discrete
# hyperparameter search space for the MLP classifier.
#
# Optuna is used to investigate whether a guided
# hyperparameter search can identify a better-performing
# configuration than independent random sampling.
#
# Compared with Random Search, Optuna uses information
# from previous trials to guide subsequent evaluations.
#
# Compared with Hyperband, Optuna focuses on selecting
# promising hyperparameter configurations rather than
# primarily allocating computational resources.
#
# Compared with LEA and TLBO, Optuna does not use a
# population-based evolutionary or teacher-learner
# mechanism. Instead, it uses a probabilistic optimization
# strategy to guide the search.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Uses information from previous trials to guide the
#   hyperparameter search.
#
# • Efficient for discrete hyperparameter search spaces.
#
# • Can identify promising configurations with a limited
#   number of trials.
#
# • Provides reproducible optimization when a fixed random
#   seed is used.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Each trial still requires model training and
#   cross-validation.
#
# • Optimization time increases with the number of trials.
#
# • Performance depends on the selected search space and
#   optimization strategy.
# =====================================================
import optuna

# =====================================================
# OPTUNA OBJECTIVE FUNCTION
# =====================================================
#
# The objective function defines one Bayesian
# optimization trial.
#
# During each trial, Optuna selects one combination of
# MLP hyperparameters from the predefined search space.
#
# Each candidate configuration is evaluated using
# weighted F1-score obtained from 5-fold cross-validation.
#
# The hidden-layer architectures are represented as
# strings during Optuna optimization because Optuna's
# categorical distribution does not directly support
# tuple values for persistent categorical choices.
#
# The selected architecture is converted back to a tuple
# before the MLP model is constructed.
# =====================================================

def objective(trial):

    hidden_layer_choice = trial.suggest_categorical(

        "hidden_layer_sizes",

        [
            "16",
            "32",
            "64",
            "32_16",
            "64_32"
        ]

    )

    architecture_map = {

        "16": (16,),

        "32": (32,),

        "64": (64,),

        "32_16": (32, 16),

        "64_32": (64, 32)

    }

    params = {

        "hidden_layer_sizes":
            architecture_map[hidden_layer_choice],

        "activation": trial.suggest_categorical(

            "activation",

            [
                "relu",
                "tanh"
            ]

        ),

        "solver": "adam",

        "learning_rate_init":
            trial.suggest_categorical(

                "learning_rate_init",

                [
                    0.0001,
                    0.001
                ]

            ),

        "alpha": trial.suggest_categorical(

            "alpha",

            [
                0.0001,
                0.001,
                0.01
            ]

        ),

        "batch_size": trial.suggest_categorical(

            "batch_size",

            [
                32,
                64,
                128
            ]

        )

    }

    model = build_mlp(params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1

    ).mean()

    return score

# =====================================================
# CREATE OPTUNA STUDY
# =====================================================
#
# Create a Bayesian optimization study that attempts to
# maximize the weighted F1-score returned by the objective
# function.
#
# Twenty trials are used to maintain the same optimization
# budget as Random Search and the other optimization
# algorithms.
#
# A fixed random seed is used to ensure reproducibility.
# =====================================================

print("\nBuilding Bayesian Optimizer...")

study = optuna.create_study(

    direction="maximize",

    study_name="MLP_Bayesian_Optimization",

    sampler=optuna.samplers.TPESampler(
        seed=42
    )

)
# =====================================================
# START BAYESIAN OPTIMIZATION
# =====================================================
#
# Optuna sequentially evaluates candidate
# hyperparameter configurations.
#
# The TPE sampler uses information from previous trials
# to guide subsequent trials toward more promising
# regions of the search space.
#
# Twenty trials are performed to maintain consistency
# with the optimization budget used by the other methods.
# =====================================================

print("\nStarting Bayesian Optimization...")

start = time.time()

study.optimize(

    objective,

    n_trials=20

)

end = time.time()

bayesian_runtime = end - start

print("\nBayesian Optimization completed.")

print(
    f"Optimization Time: "
    f"{bayesian_runtime:.2f} seconds"
)
# =====================================================
# RETRIEVE BEST HYPERPARAMETERS
# =====================================================
#
# Retrieve the hyperparameter configuration that achieved
# the highest weighted F1-score during Bayesian
# Optimization.
#
# The hidden_layer_sizes value is converted from the
# string representation used by Optuna back into the tuple
# format required by MLPClassifier.
# =====================================================

best_params = study.best_params.copy()

architecture_map = {

    "16": (16,),

    "32": (32,),

    "64": (64,),

    "32_16": (32, 16),

    "64_32": (64, 32)

}

best_params["hidden_layer_sizes"] = architecture_map[
    best_params["hidden_layer_sizes"]
]

print("\nBest Hyperparameters Found\n")

for key, value in best_params.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{study.best_value:.4f}"
)
# =====================================================
# BUILD BAYESIAN OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the final MLP classifier using the best
# hyperparameter configuration identified by Optuna.
#
# The common build_mlp() function is reused to ensure
# consistency with the Baseline MLP and Random Search
# experiments.
# =====================================================

print("\nBuilding Bayesian Optimized MLP Model...")

bayesian_model = build_mlp(
    best_params
)
# =====================================================
# TRAIN BAYESIAN OPTIMIZED MLP
# =====================================================
#
# Train the Bayesian-optimized MLP using the complete
# training dataset.
#
# Training time is measured separately from the Bayesian
# optimization time so that the computational cost of
# hyperparameter optimization can be reported separately.
# =====================================================

print("\nTraining Bayesian Optimized MLP...")

bayesian_model, bayesian_training_runtime = train_mlp_model(

    bayesian_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{bayesian_training_runtime:.2f} seconds"
)
# =====================================================
# BAYESIAN OPTIMIZED MODEL EVALUATION
# =====================================================
#
# Evaluate the optimized MLP on the independent testing
# dataset using the same evaluation metrics as the
# Baseline MLP and Random Search.
# =====================================================

print("\nMaking Predictions...")

bayesian_accuracy, \
bayesian_precision, \
bayesian_recall, \
bayesian_f1 = evaluate_model(

    bayesian_model,

    X_test,

    y_test,

    "Bayesian Optimization"

)
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

plot_training_loss(
    bayesian_model,
    "Bayesian Optimization MLP Training Loss",
    "03_bayesian_optimization_mlp_training_loss.png"
)
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Bayesian Optimization",

    best_params=best_params,

    best_cv_score=study.best_value,

    optimization_time=bayesian_runtime,

    training_time=bayesian_training_runtime

)
# =====================================================
# HYPERBAND OPTIMIZATION
# =====================================================
#
# Hyperband is a resource-aware hyperparameter
# optimization method that evaluates multiple candidate
# MLP configurations while progressively allocating more
# training resources to promising configurations.
#
# Candidate configurations that perform poorly during
# earlier evaluation stages can be eliminated, allowing
# computational resources to be concentrated on more
# promising configurations.
#
# -----------------------------------------------------
# WHY HYPERBAND IS USED
# -----------------------------------------------------
#
# The Smart Grid Stability dataset provides a
# multi-dimensional MLP hyperparameter search space.
#
# Hyperband is used to investigate whether resource-aware
# hyperparameter optimization can identify an effective
# MLP configuration while reducing unnecessary computation
# on poorly performing candidates.
#
# Unlike Random Search, Hyperband does not necessarily
# allocate the same computational budget to every
# candidate configuration.
#
# Compared with Bayesian Optimization, Hyperband does not
# construct a probabilistic model to guide the search.
# Instead, it focuses on efficient allocation of training
# resources and early elimination of weak configurations.
#
# Compared with LEA and TLBO, Hyperband does not evolve
# candidate solutions through population-based updates.
# Instead, it progressively reduces the number of
# candidates according to their performance.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Efficient allocation of computational resources.
#
# • Poor-performing configurations can be eliminated early.
#
# • Allows more candidate configurations to be explored
#   within a limited computational budget.
#
# • Suitable for computationally expensive MLP training.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Poor early performance may cause a potentially useful
#   configuration to be eliminated.
#
# • Results depend on the selected resource budget and
#   reduction factor.
#
# • Hyperband does not use information from previous
#   evaluations through a probabilistic surrogate model.
# =====================================================
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
# =====================================================
# HYPERBAND CONFIGURATION
# =====================================================
#
# factor = 3
#
# The reduction factor controls how aggressively
# underperforming configurations are eliminated between
# successive resource levels.
#
# resource = "max_iter"
#
# The maximum number of MLP training iterations is used
# as the computational resource.
#
# max_resources = 500
#
# The maximum training budget is set to 500 iterations,
# matching the maximum number of iterations used by the
# MLP models in the other experiments.
#
# Five-fold cross-validation and weighted F1-score are
# used to maintain consistency with the other optimization
# methods.
# =====================================================

print("\nBuilding Hyperband Optimizer...")

hyperband = HalvingRandomSearchCV(

    estimator=MLPClassifier(

        max_iter=500,

        random_state=42

    ),

    param_distributions=SEARCH_SPACE,

    factor=3,

    resource="max_iter",

    max_resources=500,
    n_candidates=5,

    scoring="f1_weighted",

    cv=5,

    random_state=42,

    n_jobs=-1

)
# =====================================================
# START HYPERBAND OPTIMIZATION
# =====================================================

print("\nStarting Hyperband Optimization...")

start = time.time()

hyperband.fit(

    X_train,

    y_train

)

end = time.time()

hyperband_runtime = end - start

print("\nHyperband Optimization completed.")

print(
    f"Optimization Time: "
    f"{hyperband_runtime:.2f} seconds"
)
# =====================================================
# RETRIEVE BEST HYPERPARAMETERS
# =====================================================
#
# Retrieve the hyperparameter configuration that achieved
# the highest weighted F1-score during Hyperband
# optimization.
# =====================================================

best_hyperband_params = hyperband.best_params_

print("\nBest Hyperparameters Found By Hyperband\n")

for key, value in best_hyperband_params.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{hyperband.best_score_:.4f}"
)
# =====================================================
# BUILD HYPERBAND OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the final MLP classifier using the best
# hyperparameters identified by Hyperband.
#
# The common build_mlp() function is reused to maintain
# consistency with the Baseline, Random Search and
# Bayesian Optimization experiments.
# =====================================================

print("\nBuilding Hyperband Optimized MLP Model...")

hyperband_model = build_mlp(
    best_hyperband_params
)
# =====================================================
# TRAIN HYPERBAND OPTIMIZED MLP
# =====================================================
#
# Train the optimized MLP using the complete training
# dataset.
#
# Training time is recorded separately from the
# Hyperband optimization time.
# =====================================================

print("\nTraining Hyperband Optimized MLP...")

hyperband_model, hyperband_training_runtime = train_mlp_model(

    hyperband_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{hyperband_training_runtime:.2f} seconds"
)
# =====================================================
# HYPERBAND MODEL EVALUATION
# =====================================================
#
# Evaluate the trained Hyperband-optimized MLP on the
# independent testing dataset.
#
# The same evaluation metrics are used as for all
# previous models to maintain a consistent comparison.
# =====================================================

print("\nMaking Predictions...")

hyperband_accuracy, \
hyperband_precision, \
hyperband_recall, \
hyperband_f1 = evaluate_model(

    hyperband_model,

    X_test,

    y_test,

    "Hyperband"

)
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

plot_training_loss(
    hyperband_model,
    "Hyperband MLP Training Loss",
    "04_hyperband_mlp_training_loss.png"
)
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Hyperband",

    best_params=best_hyperband_params,

    best_cv_score=hyperband.best_score_,

    optimization_time=hyperband_runtime,

    training_time=hyperband_training_runtime

)
# =====================================================
# LEARNING EVOLUTIONARY ALGORITHM (LEA)
# =====================================================
#
# LEA is a population-based evolutionary optimization
# algorithm that searches for high-performing MLP
# hyperparameter configurations through iterative
# population improvement.
#
# The algorithm starts with a population of candidate
# solutions, where each candidate represents one possible
# combination of MLP hyperparameters.
#
# During each generation, candidate solutions are
# evaluated according to their weighted F1-score.
# High-performing configurations are retained, while
# new candidate solutions are generated by modifying
# hyperparameters of selected surviving configurations.
#
# -----------------------------------------------------
# WHY LEA IS USED
# -----------------------------------------------------
#
# LEA is used to investigate whether an evolutionary
# population-based search strategy can identify an
# effective MLP hyperparameter configuration for the
# Smart Grid Stability classification problem.
#
# Unlike Random Search, which evaluates independently
# sampled configurations, LEA uses the performance of
# previously evaluated configurations to guide the
# generation of subsequent candidates.
#
# Compared with Bayesian Optimization, LEA does not
# construct a probabilistic surrogate model. Instead,
# it improves candidate configurations through
# population-based evolutionary search.
#
# Compared with Hyperband, which focuses on allocating
# computational resources efficiently, LEA focuses on
# progressively improving the population of candidate
# hyperparameter configurations.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Population-based exploration of the hyperparameter
#   search space.
#
# • Retains high-performing candidate configurations.
#
# • Maintains exploration through random mutation.
#
# • Suitable for discrete MLP hyperparameter spaces.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires repeated evaluation of candidate solutions.
#
# • Computational cost increases with population size
#   and number of generations.
#
# • Random mutation may produce inferior configurations.
#
# • The algorithm may converge toward a local optimum.
# =====================================================
# =====================================================
# LEA POPULATION INITIALIZATION
# =====================================================
#
# LEA starts with a population of candidate solutions.
#
# Each candidate represents one possible combination of
# MLP hyperparameters selected from the common
# SEARCH_SPACE.
#
# A fixed random seed is used to ensure that the same
# initial population can be reproduced across experimental
# runs.
# =====================================================

random.seed(42)

LEA_POPULATION_SIZE = 20

lea_population = []

for _ in range(LEA_POPULATION_SIZE):

    candidate = {

        parameter: random.choice(values)

        for parameter, values in SEARCH_SPACE.items()

    }

    lea_population.append(candidate)


print("\nInitial LEA Population Created.")

print(
    f"Population Size: "
    f"{len(lea_population)}"
)
# =====================================================
# LEA FITNESS EVALUATION
# =====================================================
#
# Each candidate hyperparameter configuration is
# evaluated using 5-fold cross-validation.
#
# Weighted F1-score is used as the fitness measure,
# maintaining consistency with Random Search, Bayesian
# Optimization and Hyperband.
#
# A higher weighted F1-score indicates a better-performing
# candidate configuration.
# =====================================================

def evaluate_lea_candidate(params):

    model = build_mlp(params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1

    ).mean()

    return score
# =====================================================
# EVALUATE INITIAL LEA POPULATION
# =====================================================
#
# Evaluate every candidate in the initial population
# before beginning the evolutionary generations.
#
# This establishes the initial fitness values that will
# be used to identify the strongest candidate solutions.
# =====================================================

print("\nEvaluating Initial LEA Population...")

lea_fitness = []

for candidate in lea_population:

    score = evaluate_lea_candidate(
        candidate
    )

    lea_fitness.append(score)


print("\nInitial Population Evaluation Completed.")

print(
    f"Best Initial F1 Score: "
    f"{max(lea_fitness):.4f}"
)
# =====================================================
# LEA OPTIMIZATION PARAMETERS
# =====================================================
#
# POPULATION_SIZE = 20
#
# Twenty candidate configurations are maintained in each
# generation. This provides sufficient diversity while
# keeping the computational cost manageable.
#
# SURVIVAL_RATE = 0.50
#
# The top 50% of candidates are retained after each
# generation. These candidates act as parents for the
# next generation.
#
# MAX_ITERATIONS = 20
#
# Twenty generations are used to provide multiple
# opportunities for the population to improve while
# maintaining a comparable optimization budget with the
# other methods.
# =====================================================

POPULATION_SIZE = 20

SURVIVAL_RATE = 0.50

MAX_ITERATIONS = 5


# =====================================================
# INITIALIZE BEST LEA SOLUTION
# =====================================================
#
# Keep track of the best hyperparameter configuration
# discovered throughout the complete evolutionary search.
#
# The best solution is not limited to the final
# generation because an earlier generation may contain
# the overall best configuration.
# =====================================================

best_lea_score = float("-inf")

best_lea_solution = None


# =====================================================
# START LEA OPTIMIZATION
# =====================================================
#
# Each generation performs the following operations:
#
# 1. Evaluate the current population.
# 2. Identify the fitness of each candidate.
# 3. Rank candidates according to weighted F1-score.
# 4. Retain the highest-performing candidates.
# 5. Select surviving candidates as parents.
# 6. Mutate one hyperparameter of each new candidate.
# 7. Create the next generation.
#
# The best solution discovered across all generations is
# retained for the final MLP model.
# =====================================================

print("\nStarting LEA Optimization...")

start = time.time()

for iteration in range(MAX_ITERATIONS):

    print(
        f"\nGeneration "
        f"{iteration + 1}/{MAX_ITERATIONS}"
    )

    fitness_scores = []


    # =================================================
    # EVALUATE CURRENT POPULATION
    # =================================================
    #
    # Every candidate in the current population is
    # evaluated using 5-fold cross-validation.
    # =================================================

    for solution in lea_population:

        score = evaluate_lea_candidate(
            solution
        )

        fitness_scores.append(score)


        # ---------------------------------------------
        # UPDATE GLOBAL BEST SOLUTION
        # ---------------------------------------------
        #
        # Store the best candidate discovered across
        # all generations.
        # ---------------------------------------------

        if score > best_lea_score:

            best_lea_score = score

            best_lea_solution = solution.copy()


    print(
        f"Best F1 Score: "
        f"{best_lea_score:.4f}"
    )


    # =================================================
    # RANK CANDIDATE SOLUTIONS
    # =================================================
    #
    # Candidates are sorted according to their fitness
    # scores from highest to lowest.
    #
    # The best-performing candidate is therefore placed
    # at the beginning of the ranked population.
    # =================================================

    ranked_population = [

        solution

        for _, solution in sorted(

            zip(
                fitness_scores,
                lea_population
            ),

            key=lambda x: x[0],

            reverse=True

        )

    ]


    # =================================================
    # SELECT SURVIVING CANDIDATES
    # =================================================
    #
    # Retain the highest-performing 50% of candidates.
    #
    # These candidates become the parents for generating
    # the next population.
    # =================================================

    survivor_count = int(

        POPULATION_SIZE *
        SURVIVAL_RATE

    )

    survivors = ranked_population[
        :survivor_count
    ]


    # =================================================
    # GENERATE NEXT POPULATION
    # =================================================
    #
    # The surviving candidates are retained.
    #
    # Additional candidates are generated by:
    #
    # 1. Randomly selecting a surviving parent.
    # 2. Copying the parent's hyperparameters.
    # 3. Randomly selecting one hyperparameter.
    # 4. Replacing its value with another value from
    #    the predefined SEARCH_SPACE.
    #
    # This mutation introduces exploration while
    # preserving information from high-performing
    # configurations.
    # =================================================

    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(
            survivors
        ).copy()

        parameter = random.choice(
            list(parent.keys())
        )


        # ---------------------------------------------
        # KEEP SOLVER FIXED
        # ---------------------------------------------
        #
        # Adam is fixed throughout the experiment.
        # Therefore, the solver parameter is not mutated.
        # ---------------------------------------------

        if parameter != "solver":

            parent[parameter] = random.choice(

                SEARCH_SPACE[parameter]

            )

        new_population.append(
            parent
        )


    # =================================================
    # REPLACE CURRENT POPULATION
    # =================================================
    #
    # The newly generated population becomes the
    # population evaluated during the next generation.
    # =================================================

    lea_population = new_population


end = time.time()

lea_runtime = end - start


# =====================================================
# LEA OPTIMIZATION COMPLETED
# =====================================================

print("\nLEA Optimization completed.")

print(
    f"Optimization Time: "
    f"{lea_runtime:.2f} seconds"
)

print(
    "\nBest Hyperparameters Found By LEA\n"
)

for key, value in best_lea_solution.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{best_lea_score:.4f}"
)
# =====================================================
# BUILD LEA OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the final MLP classifier using the best
# hyperparameter configuration identified during the
# LEA evolutionary search.
#
# The best solution is taken from all generations rather
# than only the final generation.
# =====================================================

print("\nBuilding LEA Optimized MLP Model...")

lea_model = build_mlp(
    best_lea_solution
)
# =====================================================
# TRAIN LEA OPTIMIZED MLP
# =====================================================
#
# Train the LEA-optimized MLP using the complete training
# dataset.
#
# The training time is recorded separately from the LEA
# optimization time so that the computational cost of
# hyperparameter optimization can be distinguished from
# the cost of training the final model.
# =====================================================

print("\nTraining LEA Optimized MLP...")

lea_model, lea_training_runtime = train_mlp_model(

    lea_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{lea_training_runtime:.2f} seconds"
)
# =====================================================
# LEA MODEL EVALUATION
# =====================================================
#
# Evaluate the trained LEA-optimized MLP on the
# independent testing dataset.
#
# The same evaluation metrics used for the Baseline,
# Random Search, Bayesian Optimization and Hyperband
# models are used here to ensure a consistent comparison.
# =====================================================

print("\nMaking Predictions...")

lea_accuracy, \
lea_precision, \
lea_recall, \
lea_f1 = evaluate_model(

    lea_model,

    X_test,

    y_test,

    "LEA"

)
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================
#
# Plot the training loss curve of the final
# LEA-optimized MLP.
#
# The curve provides information about the convergence
# behaviour of the optimized model during training.
# =====================================================

plot_training_loss(
    lea_model,
    "LEA MLP Training Loss",
    "05_lea_mlp_training_loss.png"
)
# =====================================================
# PRINT LEA OPTIMIZATION SUMMARY
# =====================================================
#
# Display the best hyperparameter configuration,
# cross-validation F1-score, optimization runtime and
# final model training runtime.
# =====================================================

print_optimization_summary(

    method_name="LEA",

    best_params=best_lea_solution,

    best_cv_score=best_lea_score,

    optimization_time=lea_runtime,

    training_time=lea_training_runtime

)

# =====================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# =====================================================
#
# TLBO is a population-based optimization algorithm that
# improves candidate solutions through two learning phases:
#
# 1. Teacher Phase
# 2. Learner Phase
#
# Each candidate solution represents one possible
# combination of MLP hyperparameters.
#
# During the Teacher Phase, the best-performing candidate
# acts as the teacher and influences the population toward
# improved hyperparameter configurations.
#
# During the Learner Phase, candidate solutions learn by
# interacting with other randomly selected candidates.
#
# Candidate configurations are evaluated using weighted
# F1-score obtained through 5-fold cross-validation.
#
# -----------------------------------------------------
# WHY TLBO IS USED
# -----------------------------------------------------
#
# TLBO is included to investigate whether a
# teacher-learner population-based optimization strategy
# can identify effective MLP hyperparameters for the
# Smart Grid Stability classification problem.
#
# Unlike Random Search, TLBO does not evaluate completely
# independent configurations. Candidate solutions are
# iteratively modified based on information obtained from
# the current population.
#
# Compared with Bayesian Optimization, TLBO does not
# construct a probabilistic surrogate model. Instead,
# candidate solutions are improved through teacher and
# learner interactions.
#
# Compared with Hyperband, which focuses on computational
# resource allocation, TLBO focuses on progressively
# improving candidate hyperparameter configurations.
#
# Compared with LEA, which uses evolutionary selection and
# mutation, TLBO updates candidate solutions through
# teacher and learner learning mechanisms.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Population-based optimization.
#
# • Uses information from the best candidate solution.
#
# • Does not require a separate surrogate model.
#
# • Uses teacher and learner interactions to explore the
#   hyperparameter search space.
#
# • Suitable for discrete hyperparameter optimization.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires repeated evaluation of candidate solutions.
#
# • Performance depends on the population size and number
#   of optimization iterations.
#
# • Random learner selection can produce inferior
#   candidate configurations.
#
# • May converge toward a local optimum.
# =====================================================
# =====================================================
# TLBO PARAMETER CONFIGURATION
# =====================================================
#
# POPULATION_SIZE = 20
#
# Twenty candidate hyperparameter configurations are
# maintained during each iteration.
#
# MAX_ITERATIONS = 20
#
# Twenty optimization iterations are performed to provide
# multiple opportunities for the population to improve.
#
# The same population size and iteration budget used by
# LEA are maintained to support a fair comparison between
# the population-based optimization algorithms.
# =====================================================

TLBO_POPULATION_SIZE = 20

TLBO_MAX_ITERATIONS = 5
# =====================================================
# TLBO POPULATION INITIALIZATION
# =====================================================
#
# Generate the initial population by randomly selecting
# hyperparameter values from the common SEARCH_SPACE.
#
# A fixed random seed is used to make the initial
# population reproducible.
# =====================================================

random.seed(42)

tlbo_population = []

for _ in range(TLBO_POPULATION_SIZE):

    candidate = {

        parameter: random.choice(values)

        for parameter, values in SEARCH_SPACE.items()

    }

    tlbo_population.append(candidate)


print("\nInitial TLBO Population Created.")

print(
    f"Population Size: "
    f"{len(tlbo_population)}"
)
# =====================================================
# TLBO FITNESS EVALUATION
# =====================================================
#
# Each TLBO candidate is evaluated using the same
# 5-fold cross-validation procedure used by the other
# optimization algorithms.
#
# Weighted F1-score is used as the fitness measure.
# =====================================================

def evaluate_tlbo_candidate(params):

    model = build_mlp(params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1

    ).mean()

    return score
# =====================================================
# INITIALIZE BEST TLBO SOLUTION
# =====================================================

best_tlbo_score = float("-inf")

best_tlbo_solution = None
# =====================================================
# START TLBO OPTIMIZATION
# =====================================================
#
# TLBO optimization is performed for a fixed number of
# iterations.
#
# Each iteration consists of two learning phases:
#
# 1. Teacher Phase
# 2. Learner Phase
#
# The best-performing candidate in the current population
# is selected as the teacher.
# =====================================================

print("\nStarting TLBO Optimization...")

start = time.time()

for iteration in range(TLBO_MAX_ITERATIONS):

    print(
        f"\nTLBO Iteration "
        f"{iteration + 1}/{TLBO_MAX_ITERATIONS}"
    )

    # =================================================
    # EVALUATE CURRENT POPULATION
    # =================================================
    #
    # Evaluate every candidate using 5-fold
    # cross-validation and weighted F1-score.
    # =================================================

    fitness_scores = []

    for candidate in tlbo_population:

        score = evaluate_tlbo_candidate(
            candidate
        )

        fitness_scores.append(score)

        if score > best_tlbo_score:

            best_tlbo_score = score

            best_tlbo_solution = candidate.copy()


    # =================================================
    # IDENTIFY TEACHER
    # =================================================
    #
    # The candidate with the highest fitness becomes
    # the teacher for the current iteration.
    # =================================================

    teacher_index = fitness_scores.index(
        max(fitness_scores)
    )

    teacher = tlbo_population[
        teacher_index
    ].copy()

    teacher_score = fitness_scores[
        teacher_index
    ]

    print(
        f"Teacher F1 Score: "
        f"{teacher_score:.4f}"
    )
        # =================================================
    # TEACHER PHASE
    # =================================================
    #
    # The teacher represents the best-performing
    # hyperparameter configuration in the current
    # population.
    #
    # Candidate solutions are modified toward the
    # teacher configuration.
    #
    # Because the MLP hyperparameters are discrete,
    # the continuous TLBO update is implemented as a
    # probabilistic selection between the candidate's
    # current value and the teacher's corresponding
    # value.
    # =================================================

    teacher_population = []

    for candidate in tlbo_population:

        new_candidate = candidate.copy()

        for parameter in SEARCH_SPACE.keys():

            # Adam remains fixed throughout the study.

            if parameter == "solver":

                continue

            # With probability 0.5, move the candidate
            # toward the teacher's value.

            if random.random() < 0.5:

                new_candidate[parameter] = (
                    teacher[parameter]
                )

        teacher_population.append(
            new_candidate
        )

        # =================================================
    # EVALUATE TEACHER-PHASE CANDIDATES
    # =================================================
    #
    # Evaluate the modified candidates and retain the
    # teacher-phase solution when it performs better
    # than the corresponding original candidate.
    # =================================================

    for index in range(
        TLBO_POPULATION_SIZE
    ):

        original_candidate = (
            tlbo_population[index]
        )

        original_score = (
            fitness_scores[index]
        )

        new_candidate = (
            teacher_population[index]
        )

        new_score = evaluate_tlbo_candidate(
            new_candidate
        )

        if new_score > original_score:

            tlbo_population[index] = (
                new_candidate
            )

            fitness_scores[index] = (
                new_score
            )

        if new_score > best_tlbo_score:

            best_tlbo_score = new_score

            best_tlbo_solution = (
                new_candidate.copy()
            )
        # =================================================
    # LEARNER PHASE
    # =================================================
    #
    # In the Learner Phase, each candidate interacts with
    # another randomly selected candidate from the current
    # population.
    #
    # If the selected learner performs better than the
    # current candidate, the candidate learns from it.
    # Otherwise, the candidate is modified in the opposite
    # direction.
    #
    # Since the MLP hyperparameters are discrete values,
    # the update is implemented by selecting parameter
    # values from the current candidate and the selected
    # learner.
    # =================================================

    learner_population = []

    for index in range(
        TLBO_POPULATION_SIZE
    ):

        current_candidate = (
            tlbo_population[index].copy()
        )

        # Select a different candidate randomly.

        learner_indices = [

            i

            for i in range(
                TLBO_POPULATION_SIZE
            )

            if i != index

        ]

        partner_index = random.choice(
            learner_indices
        )

        partner = tlbo_population[
            partner_index
        ]

        current_score = fitness_scores[
            index
        ]

        partner_score = fitness_scores[
            partner_index
        ]

        new_candidate = (
            current_candidate.copy()
        )

        # -------------------------------------------------
        # LEARN FROM A BETTER PARTNER
        # -------------------------------------------------

        if partner_score > current_score:

            for parameter in SEARCH_SPACE.keys():

                if parameter == "solver":

                    continue

                if random.random() < 0.5:

                    new_candidate[parameter] = (
                        partner[parameter]
                    )

        # -------------------------------------------------
        # LEARN FROM A WEAKER PARTNER
        # -------------------------------------------------

        else:

            mutable_parameters = [

                parameter

                for parameter in SEARCH_SPACE.keys()

                if parameter != "solver"

            ]

            parameter = random.choice(
                mutable_parameters
            )

            new_candidate[parameter] = random.choice(
                SEARCH_SPACE[parameter]
            )

        learner_population.append(
            new_candidate
        )


    # =================================================
    # EVALUATE LEARNER-PHASE CANDIDATES
    # =================================================
    #
    # Evaluate the newly generated learner candidates.
    #
    # A learner-phase candidate replaces the current
    # candidate only when it achieves a higher weighted
    # F1-score.
    # =================================================

    for index in range(
        TLBO_POPULATION_SIZE
    ):

        new_candidate = (
            learner_population[index]
        )

        new_score = evaluate_tlbo_candidate(
            new_candidate
        )

        current_score = fitness_scores[
            index
        ]

        if new_score > current_score:

            tlbo_population[index] = (
                new_candidate
            )

            fitness_scores[index] = (
                new_score
            )

        if new_score > best_tlbo_score:

            best_tlbo_score = new_score

            best_tlbo_solution = (
                new_candidate.copy()
            )


    # =================================================
    # ITERATION SUMMARY
    # =================================================

    print(
        f"Best TLBO F1 Score: "
        f"{best_tlbo_score:.4f}"
    )
# =====================================================
# TLBO OPTIMIZATION COMPLETED
# =====================================================

end = time.time()

tlbo_runtime = end - start

print("\nTLBO Optimization completed.")

print(
    f"Optimization Time: "
    f"{tlbo_runtime:.2f} seconds"
)

print(
    "\nBest Hyperparameters Found By TLBO\n"
)

for key, value in best_tlbo_solution.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{best_tlbo_score:.4f}"
)
# =====================================================
# BUILD TLBO OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the final MLP classifier using the best
# hyperparameter configuration identified during the
# TLBO optimization process.
#
# The common build_mlp() function is reused to ensure
# consistency with all previous optimization methods.
# =====================================================

print("\nBuilding TLBO Optimized MLP Model...")

tlbo_model = build_mlp(
    best_tlbo_solution
)
# =====================================================
# TRAIN TLBO OPTIMIZED MLP
# =====================================================
#
# Train the TLBO-optimized MLP using the complete
# training dataset.
#
# Training time is measured separately from the TLBO
# optimization time.
# =====================================================

print("\nTraining TLBO Optimized MLP...")

tlbo_model, tlbo_training_runtime = train_mlp_model(

    tlbo_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{tlbo_training_runtime:.2f} seconds"
)
# =====================================================
# TLBO MODEL EVALUATION
# =====================================================
#
# Evaluate the trained TLBO-optimized MLP on the
# independent testing dataset.
#
# The same metrics used for all previous methods are
# reported to ensure a consistent comparison.
# =====================================================

print("\nMaking Predictions...")

tlbo_accuracy, \
tlbo_precision, \
tlbo_recall, \
tlbo_f1 = evaluate_model(

    tlbo_model,

    X_test,

    y_test,

    "TLBO"

)
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

plot_training_loss(
    tlbo_model,
    "TLBO MLP Training Loss",
    "06_tlbo_mlp_training_loss.png"
)
# =====================================================
# PRINT TLBO OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="TLBO",

    best_params=best_tlbo_solution,

    best_cv_score=best_tlbo_score,

    optimization_time=tlbo_runtime,

    training_time=tlbo_training_runtime

)

# =====================================================
# SAVE EXPERIMENT RESULTS
# =====================================================
#
# Store the performance metrics and execution times of
# all optimization methods.
#
# The results are saved as a CSV file so that the
# performance of all methods can be compared consistently
# across the Smart Grid Stability dataset.
# =====================================================



comparison_results = pd.DataFrame({

    "Method": [

        "Baseline MLP",
        "Random Search",
        "Bayesian Optimization",
        "Hyperband",
        "LEA",
        "TLBO"

    ],

    "Accuracy": [

        baseline_accuracy,
        random_search_accuracy,
        bayesian_accuracy,
        hyperband_accuracy,
        lea_accuracy,
        tlbo_accuracy

    ],

    "Precision": [

        baseline_precision,
        random_search_precision,
        bayesian_precision,
        hyperband_precision,
        lea_precision,
        tlbo_precision

    ],

    "Recall": [

        baseline_recall,
        random_search_recall,
        bayesian_recall,
        hyperband_recall,
        lea_recall,
        tlbo_recall

    ],

    "F1 Score": [

        baseline_f1,
        random_search_f1,
        bayesian_f1,
        hyperband_f1,
        lea_f1,
        tlbo_f1

    ],

    "Optimization Time (s)": [

        0,
        random_search_runtime,
        bayesian_runtime,
        hyperband_runtime,
        lea_runtime,
        tlbo_runtime

    ],

    "Training Time (s)": [

        baseline_training_runtime,
        random_search_training_runtime,
        bayesian_training_runtime,
        hyperband_training_runtime,
        lea_training_runtime,
        tlbo_training_runtime

    ]

})


print("\nComparison Results")

print(
    comparison_results
)


comparison_results.to_csv(

    f"{RESULTS_DIR}/smart_grid_stability_results.csv",

    index=False

)

# ============================================================
# 1. OPTIMIZATION PERFORMANCE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

comparison_results.set_index("Method")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Smart Grid Stability Phase 2 - Optimization Performance Comparison"
)
plt.ylabel("Score")
plt.xlabel("Optimization Method")
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "07_optimization_performance_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 2. OPTIMIZATION TIME COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_results["Method"],
    comparison_results["Optimization Time (s)"]
)

plt.title(
    "Smart Grid Stability Phase 2 - Optimization Time Comparison"
)
plt.xlabel("Optimization Method")
plt.ylabel("Optimization Time (seconds)")
plt.xticks(rotation=45)
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "08_optimization_time_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# EXISTING CODE - KEEP THIS
# ============================================================

print("\nComparison Results")
print(comparison_results)

comparison_results.to_csv(
    f"{RESULTS_DIR}/smart_grid_stability_results.csv",
    index=False
)

print("\nResults saved successfully.")


