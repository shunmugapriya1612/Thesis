 # =====================================================
# SOLAR FLARE DATASET
# =====================================================
#
# Dataset Name:
# Solar Flare
#
# Source:
# UCI Machine Learning Repository
# Dataset ID: 89
#
# Original Machine Learning Task:
# Classification
#
# Current Machine Learning Task:
# Multi-Class Classification
#
# =====================================================
# DATASET DESCRIPTION
# =====================================================
#
# The Solar Flare dataset contains observations describing
# solar active regions and their associated solar flare
# activity.
#
# The dataset contains 1,389 observations and 13 original
# variables.
#
# The original dataset contains ten predictor variables
# and three target variables representing different levels
# of solar flare activity:
#
# • common flares
# • moderate flares
# • severe flares
#
# During Phase 1, these three target variables were combined
# into a single categorical target variable named "Target".
#
# The resulting target represents four possible flare
# severity categories:
#
# • No Flare
# • Common Flare
# • Moderate Flare
# • Severe Flare
#
# =====================================================
# INPUT FEATURES
# =====================================================
#
# The predictor variables used for classification are:
#
# • modified Zurich class
# • largest spot size
# • spot distribution
# • activity
# • evolution
# • previous 24 hour flare activity
# • historically-complex
# • became complex on this pass
# • area
# • area of largest spot
#
# The dataset contains both categorical and integer-valued
# predictor variables.
#
# The three categorical predictor variables are:
#
# • modified Zurich class
# • largest spot size
# • spot distribution
#
# These categorical variables require encoding before they
# can be used by the MLP classifier.
#
# =====================================================
# TARGET VARIABLE
# =====================================================
#
# Target:
# Target
#
# Number of Classes:
# 4
#
# The Target variable represents the observed solar flare
# activity associated with each observation.
#
# The four target classes are:
#
# • No Flare
# • Common Flare
# • Moderate Flare
# • Severe Flare
#
# The original target variables:
#
# • common flares
# • moderate flares
# • severe flares
#
# were removed after the combined categorical Target
# variable was created during Phase 1.
#
# =====================================================
# TARGET DISTRIBUTION
# =====================================================
#
# The processed dataset contains the following class
# distribution:
#
# • No Flare:          1,203 observations
# • Common Flare:        125 observations
# • Moderate Flare:       50 observations
# • Severe Flare:        11 observations
#
# The classes are therefore highly imbalanced.
#
# Weighted F1-score is consequently used as the primary
# optimization metric so that the contribution of each
# class is considered according to its frequency.
#
# =====================================================
# WHY THIS DATASET WAS SELECTED
# =====================================================
#
# The Solar Flare dataset provides a multi-class
# classification problem involving solar flare activity.
#
# Its combination of categorical and numerical predictor
# variables provides an additional environment for
# evaluating the MLP classifier and the different
# hyperparameter optimization algorithms.
#
# The dataset also contains four classes with substantially
# different frequencies, allowing the optimization methods
# to be evaluated under a highly imbalanced multi-class
# classification setting.
#
# =====================================================
# =====================================================
# IMPORT REQUIRED LIBRARIES
# =====================================================

import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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

print("\nLoading Solar Flare Dataset...")

df = pd.read_csv(
    "processed_datasets/solar_flare_processed.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())
# =====================================================
# FEATURE AND TARGET SEPARATION
# =====================================================
#
# The Target column contains the four Solar Flare
# classification classes.
#
# The Target column is separated from the predictor
# variables before preprocessing.
# =====================================================

X = df.drop(
    columns=["Target"]
)

y = df["Target"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())
# =====================================================
# CATEGORICAL FEATURE ENCODING
# =====================================================
#
# The Solar Flare dataset contains three categorical
# predictor variables:
#
# • modified Zurich class
# • largest spot size
# • spot distribution
#
# One-hot encoding is applied to convert these
# categorical variables into numerical features suitable
# for the MLP classifier.
# =====================================================

categorical_columns = [

    "modified Zurich class",
    "largest spot size",
    "spot distribution"

]

X = pd.get_dummies(

    X,

    columns=categorical_columns,

    dtype=int

)

print("\nFeature Matrix After Encoding:")
print(X.shape)

print("\nFeature Data Types:")
print(X.dtypes)

# =====================================================
# TARGET ENCODING
# =====================================================
#
# The categorical Target variable is converted into
# numerical class labels for MLP classification.
# =====================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nEncoded Target Classes:")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{index}: {class_name}"
    )

# =====================================================
# TRAIN-TEST SPLIT
# =====================================================
#
# The dataset is divided into training and testing
# subsets using an 80:20 split.
#
# Stratification is used to preserve the class
# distribution between the training and testing sets.
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# =====================================================
# FEATURE STANDARDIZATION
# =====================================================
#
# Standardization is applied because MLP classifiers
# are sensitive to differences in feature scale.
#
# The scaler is fitted only on the training data and
# subsequently applied to the testing data.
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

print(
    "\nFeature Standardization Completed."
)
# =====================================================
# MLP HYPERPARAMETER SEARCH SPACE
# =====================================================
#
# The same hyperparameter search space is used by all
# optimization algorithms to ensure a consistent
# experimental setup.
#
# Hyperparameters include:
#
# • Hidden layer architecture
# • Activation function
# • Learning rate
# • L2 regularization strength
# • Batch size
#
# The Adam solver is fixed throughout the experiment.
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
# Construct an MLP classifier using the supplied
# hyperparameter configuration.
#
# The same function is reused by the baseline model and
# all optimization algorithms.
# =====================================================

def build_mlp(params):

    return MLPClassifier(

        hidden_layer_sizes=params["hidden_layer_sizes"],

        activation=params["activation"],

        solver=params.get(
            "solver",
            "adam"
        ),

        learning_rate_init=params[
            "learning_rate_init"
        ],

        alpha=params["alpha"],

        batch_size=params["batch_size"],

        max_iter=1000,

        random_state=42

    )
# =====================================================
# EVALUATE MODEL
# =====================================================
#
# Evaluate the trained MLP classifier on the independent
# testing dataset.
#
# The reported metrics are:
#
# • Accuracy
# • Weighted Precision
# • Weighted Recall
# • Weighted F1-score
#
# A classification report and confusion matrix are also
# provided.
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    method_name
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"\n========== "
        f"{method_name.upper()} RESULTS =========="
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )
# =====================================================
# TRAIN MLP MODEL
# =====================================================
#
# Train an MLP classifier and record the training time.
# =====================================================

def train_mlp_model(
    model,
    X_train,
    y_train
):

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    end = time.time()

    training_time = end - start

    return (
        model,
        training_time
    )
# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

def plot_training_loss(
    model,
    title
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        model.loss_curve_
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        title
    )

    plt.grid(
        True
    )

    plt.show()
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

    print(
        f"\nBest Hyperparameters Found By "
        f"{method_name}\n"
    )

    for key, value in best_params.items():

        print(
            f"{key}: {value}"
        )

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
# BASELINE MULTI-LAYER PERCEPTRON (MLP)
# =====================================================
#
# The Baseline Multi-Layer Perceptron (MLP) is a
# feedforward artificial neural network that serves as
# the reference model for evaluating all hyperparameter
# optimization algorithms in this study.
#
# Unlike the optimization methods, the baseline model
# uses manually selected hyperparameters based on
# commonly adopted neural network settings rather than
# searching for an optimal configuration.
#
# -----------------------------------------------------
# WHY A BASELINE MODEL IS REQUIRED
# -----------------------------------------------------
#
# The baseline establishes a performance benchmark
# against which all optimization algorithms
# (Random Search, Bayesian Optimization, Hyperband,
# LEA and TLBO) are compared.
#
# Using identical preprocessing steps, training data,
# testing data and evaluation metrics ensures that any
# improvement in predictive performance can be
# attributed to hyperparameter optimization rather
# than differences in the experimental setup.
#
# -----------------------------------------------------
# BASELINE HYPERPARAMETERS
# -----------------------------------------------------
#
# The baseline MLP uses the following manually selected
# hyperparameters:
#
# • Hidden Layer Architecture : (64,)
# • Activation Function       : ReLU
# • Optimizer                 : Adam
# • Learning Rate             : 0.001
# • L2 Regularization (alpha) : 0.0001
# • Batch Size                : 32
#
# These values provide a balanced starting point and
# are used consistently across the datasets evaluated
# in this study.
#
# -----------------------------------------------------
# CHALLENGES AND LIMITATIONS
# -----------------------------------------------------
#
# Since the hyperparameters are manually selected, they
# may not represent the optimal configuration for the
# Solar Flare dataset.
#
# The baseline model does not explore alternative
# network architectures or learning parameters and
# therefore cannot automatically adapt to the highly
# imbalanced multi-class classification problem.
#
# Consequently, the baseline serves only as a reference
# model, while the optimization algorithms attempt to
# identify hyperparameter combinations that achieve
# better generalization and classification performance.
# =====================================================


# =====================================================
# BASELINE HYPERPARAMETERS
# =====================================================

baseline_params = {

    "hidden_layer_sizes": (64,),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": 32

}


# =====================================================
# BUILD BASELINE MLP MODEL
# =====================================================
#
# Construct the baseline MLP classifier using the
# predefined hyperparameters. The same model-building
# function will later be reused by all optimization
# algorithms to ensure a fair comparison.
# =====================================================

print("\nBuilding Baseline MLP Model...")

baseline_model = build_mlp(

    baseline_params

)


# =====================================================
# TRAIN BASELINE MLP MODEL
# =====================================================
#
# Train the baseline MLP using the standardized training
# dataset. During training, the network performs forward
# propagation, computes the prediction error, updates
# the network weights using backpropagation and repeats
# this process until convergence or the maximum number
# of training iterations is reached.
# =====================================================

print("\nTraining Baseline MLP...")

baseline_model, baseline_runtime = train_mlp_model(

    baseline_model,

    X_train,

    y_train

)

print(

    f"Training completed in "

    f"{baseline_runtime:.2f} seconds"

)


# =====================================================
# EVALUATE BASELINE MLP MODEL
# =====================================================
#
# Evaluate the trained baseline model on the testing
# dataset and calculate the classification performance
# metrics. These results serve as the benchmark for
# comparing all subsequent hyperparameter optimization
# algorithms.
# =====================================================

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
# VISUALIZE TRAINING LOSS
# =====================================================
#
# Plot the training loss recorded during each iteration
# of network training. The loss curve illustrates the
# convergence behaviour of the baseline MLP and
# provides insight into the effectiveness of the
# learning process.
# =====================================================

plot_training_loss(

    baseline_model,

    "Baseline MLP Training Loss"

)


print("\nBaseline Hyperparameters")

for key, value in baseline_params.items():

    print(

        f"{key}: {value}"

    )


# =====================================================
# RANDOM SEARCH
# =====================================================
#
# Random Search is a hyperparameter optimization
# technique that evaluates randomly selected
# hyperparameter combinations from a predefined search
# space.
#
# Unlike Grid Search, which evaluates every possible
# hyperparameter combination, Random Search randomly
# samples a fixed number of candidate solutions,
# providing a computationally efficient alternative for
# identifying high-performing models.
#
# -----------------------------------------------------
# WHY RANDOM SEARCH IS USED
# -----------------------------------------------------
#
# The Solar Flare dataset contains a multi-dimensional
# hyperparameter search space for the MLP classifier.
# Random Search is therefore used to efficiently explore
# different MLP hyperparameter combinations while
# limiting the total number of model evaluations.
#
# Compared with the manually configured baseline model,
# Random Search explores multiple candidate
# hyperparameter combinations, increasing the likelihood
# of identifying a better-performing MLP configuration.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Simple to implement.
#
# • Computationally more efficient than Grid Search.
#
# • Can discover high-performing hyperparameter
#   combinations without evaluating the entire search
#   space.
#
# • Easily parallelized across multiple processors.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Candidate hyperparameter combinations are sampled
#   independently.
#
# • Previously evaluated solutions are not used to guide
#   future searches.
#
# • Good solutions may be missed due to random sampling.
# =====================================================


# -----------------------------------------------------
# RANDOM SEARCH PARAMETER CONFIGURATION
# -----------------------------------------------------
#
# n_iter = 20
# Twenty hyperparameter combinations are randomly
# sampled from the predefined search space. This value
# maintains consistency with the evaluation budget used
# by the other optimization algorithms.
#
# scoring = "f1_weighted"
# Weighted F1-score is used as the optimization metric
# because the Solar Flare dataset contains a highly
# imbalanced distribution across the four target classes.
# Unlike accuracy, weighted F1-score considers both
# precision and recall while accounting for class
# imbalance.
#
# cv = 5
# Five-fold cross-validation provides a consistent
# estimate of model performance across different
# training-validation splits.
#
# random_state = 42
# A fixed random seed ensures that the randomly sampled
# hyperparameter combinations are reproducible.
#
# n_jobs = -1
# All available processor cores are utilized to evaluate
# the cross-validation folds in parallel.
#
# max_iter = 500
# The maximum number of training iterations for each MLP
# model is fixed at 500 to maintain consistency with the
# baseline and other optimization algorithms.
# -----------------------------------------------------

print("\nBuilding Random Search Optimizer...")

random_search = RandomizedSearchCV(

    estimator=MLPClassifier(

        max_iter=1000,

        random_state=42

    ),

    param_distributions=SEARCH_SPACE,

    n_iter=5,

    scoring="f1_weighted",

    cv=5,

    random_state=42,

    n_jobs=1

)


# =====================================================
# START RANDOM SEARCH
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
# BUILD RANDOM SEARCH OPTIMIZED MLP
# =====================================================
#
# Construct the MLP classifier using the best
# hyperparameter combination identified by Random Search.
# =====================================================

print(

    "\nBuilding Random Search Optimized MLP Model..."

)

random_model = build_mlp(

    random_search.best_params_

)


# =====================================================
# TRAIN RANDOM SEARCH OPTIMIZED MLP
# =====================================================
#
# Train the optimized MLP classifier using the complete
# training dataset and record the model training time.
# =====================================================

print(

    "\nTraining Random Search Optimized MLP..."

)

random_model, random_training_runtime = train_mlp_model(

    random_model,

    X_train,

    y_train

)

print(

    f"Training completed in "

    f"{random_training_runtime:.2f} seconds"

)


# =====================================================
# EVALUATE RANDOM SEARCH OPTIMIZED MLP
# =====================================================
#
# Evaluate the optimized MLP classifier using the testing
# dataset and calculate the classification performance
# metrics.
# =====================================================

print("\nMaking Predictions...")

random_accuracy, \
random_precision, \
random_recall, \
random_f1 = evaluate_model(

    random_model,

    X_test,

    y_test,

    "Random Search"

)


# =====================================================
# VISUALIZE TRAINING LOSS
# =====================================================

plot_training_loss(

    random_model,

    "Random Search Optimized MLP Training Loss"

)


# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Random Search",

    best_params=random_search.best_params_,

    best_cv_score=random_search.best_score_,

    optimization_time=random_search_runtime,

    training_time=random_training_runtime

)
# =====================================================
# BAYESIAN OPTIMIZATION
# =====================================================
#
# Bayesian Optimization is a sequential hyperparameter
# optimization method that uses information from previous
# evaluations to guide the selection of subsequent
# hyperparameter configurations.
#
# The optimization process is implemented using Optuna.
# At each trial, Optuna selects a candidate configuration
# from the predefined MLP hyperparameter search space.
#
# The candidate configuration is evaluated using
# 5-fold cross-validation and weighted F1-score.
#
# -----------------------------------------------------
# WHY BAYESIAN OPTIMIZATION IS USED
# -----------------------------------------------------
#
# The Solar Flare dataset contains a multi-dimensional
# hyperparameter search space and a highly imbalanced
# four-class target distribution.
#
# Bayesian Optimization is used to investigate whether
# a sequential model-based search strategy can identify
# high-performing MLP configurations more efficiently
# than randomly sampling configurations.
#
# Unlike Random Search, which evaluates independently
# sampled configurations, Bayesian Optimization uses
# information from previously evaluated configurations
# to guide subsequent trials.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Uses information from previous evaluations to guide
#   subsequent hyperparameter trials.
#
# • Can focus the search on promising regions of the
#   hyperparameter space.
#
# • Can achieve good configurations with a limited number
#   of objective evaluations.
#
# • Suitable for computationally expensive model
#   optimization.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires additional computation to maintain the
#   optimization model.
#
# • Performance depends on the search space and objective
#   function.
#
# • The optimization process can be more complex than
#   Random Search.
# =====================================================

import optuna
# =====================================================
# BAYESIAN OPTIMIZATION OBJECTIVE
# =====================================================
#
# The objective function defines how Optuna evaluates
# each candidate hyperparameter configuration.
#
# Each trial selects one combination of MLP
# hyperparameters and evaluates the resulting model
# using 5-fold cross-validation.
#
# Weighted F1-score is returned as the objective value
# because the Solar Flare target contains highly
# imbalanced classes.
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

    activation = trial.suggest_categorical(

        "activation",

        [
            "relu",
            "tanh"
        ]

    )

    learning_rate_init = trial.suggest_categorical(

        "learning_rate_init",

        [
            0.0001,
            0.001
        ]

    )

    alpha = trial.suggest_categorical(

        "alpha",

        [
            0.0001,
            0.001,
            0.01
        ]

    )

    batch_size = trial.suggest_categorical(

        "batch_size",

        [
            32,
            64,
            128
        ]

    )

    # -------------------------------------------------
    # Convert architecture string to tuple
    # -------------------------------------------------

    hidden_layer_mapping = {

        "16": (16,),

        "32": (32,),

        "64": (64,),

        "32_16": (32, 16),

        "64_32": (64, 32)

    }

    params = {

        "hidden_layer_sizes":
            hidden_layer_mapping[
                hidden_layer_choice
            ],

        "activation":
            activation,

        "solver":
            "adam",

        "learning_rate_init":
            learning_rate_init,

        "alpha":
            alpha,

        "batch_size":
            batch_size

    }

    model = build_mlp(
        params
    )

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=1

    ).mean()

    return score

# =====================================================
# BUILD BAYESIAN OPTIMIZER
# =====================================================

print("\nBuilding Bayesian Optimizer...")

study = optuna.create_study(

    direction="maximize",

    study_name="MLP_Bayesian_Optimization"

)

# =====================================================
# START BAYESIAN OPTIMIZATION
# =====================================================

print("\nStarting Bayesian Optimization...")

start = time.time()

study.optimize(

    objective,

    n_trials=20

)

end = time.time()

bayesian_runtime = end - start

print(
    "\nBayesian Optimization completed."
)

print(
    f"Optimization Time: "
    f"{bayesian_runtime:.2f} seconds"
)
# =====================================================
# RETRIEVE BEST HYPERPARAMETERS
# =====================================================

best_params = study.best_params

hidden_layer_mapping = {

    "16": (16,),

    "32": (32,),

    "64": (64,),

    "32_16": (32, 16),

    "64_32": (64, 32)

}

bayesian_best_params = {

    "hidden_layer_sizes":
        hidden_layer_mapping[
            best_params[
                "hidden_layer_sizes"
            ]
        ],

    "activation":
        best_params["activation"],

    "solver":
        "adam",

    "learning_rate_init":
        best_params[
            "learning_rate_init"
        ],

    "alpha":
        best_params["alpha"],

    "batch_size":
        best_params["batch_size"]

}

print(
    "\nBest Hyperparameters Found\n"
)

for key, value in bayesian_best_params.items():

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

print(
    "\nBuilding Bayesian Optimized MLP Model..."
)

bayesian_model = build_mlp(

    bayesian_best_params

)


# =====================================================
# TRAIN BAYESIAN OPTIMIZED MLP
# =====================================================

print(
    "\nTraining Bayesian Optimized MLP..."
)

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
# EVALUATE BAYESIAN OPTIMIZED MLP
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

    "Bayesian Optimized MLP Training Loss"

)


# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Bayesian Optimization",

    best_params=bayesian_best_params,

    best_cv_score=study.best_value,

    optimization_time=bayesian_runtime,

    training_time=bayesian_training_runtime

)
# =====================================================
# HYPERBAND OPTIMIZATION
# =====================================================
#
# Hyperband is a resource-efficient hyperparameter
# optimization method that evaluates many candidate
# configurations while allocating more computational
# resources to promising configurations.
#
# For the MLP model, the computational resource is
# represented by the maximum number of training iterations.
#
# Hyperband initially evaluates candidate configurations
# with a smaller training budget and progressively allocates
# more iterations to configurations that perform well.
#
# The objective is evaluated using 5-fold cross-validation
# and weighted F1-score.
#
# -----------------------------------------------------
# WHY HYPERBAND IS USED
# -----------------------------------------------------
#
# The Solar Flare dataset contains a multi-dimensional
# MLP hyperparameter search space. Hyperband is included
# to investigate whether resource-aware optimization can
# identify effective configurations while reducing the
# computational cost of evaluating weak configurations.
#
# Unlike Random Search, which evaluates configurations
# independently, Hyperband allocates different amounts of
# training resources to candidate configurations.
#
# Compared with Bayesian Optimization, Hyperband does not
# rely primarily on a probabilistic surrogate model.
# Instead, it uses successive resource allocation to
# identify promising configurations.
#
# Compared with LEA and TLBO, Hyperband focuses on
# computational resource allocation rather than
# evolutionary population updates or teacher-learner
# interactions.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Efficient allocation of computational resources.
#
# • Can evaluate a large number of candidate
#   configurations.
#
# • Poor-performing configurations can receive fewer
#   training resources.
#
# • Suitable for computationally expensive neural
#   network hyperparameter optimization.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Performance depends on the resource allocation
#   strategy.
#
# • Candidate configurations may be eliminated before
#   reaching their full training budget.
#
# • Results can depend on the selected training-resource
#   range.
# =====================================================
# =====================================================
# HYPERBAND PARAMETER CONFIGURATION
# =====================================================
#
# The Hyperband implementation uses randomized candidate
# configurations together with different maximum training
# iteration budgets.
#
# The maximum iteration values represent the computational
# resources allocated to the MLP configurations.
# =====================================================

HYPERBAND_ITERATIONS = [
    27,
    54,
    81
]
# =====================================================
# HYPERBAND OPTIMIZATION FUNCTION
# =====================================================

def run_hyperband():

    print("\nStarting Hyperband Optimization...")

    start = time.time()

    candidates = []

    # -------------------------------------------------
    # Generate candidate configurations
    # -------------------------------------------------

    for parameter_values in range(20):

        candidate = {
            parameter: random.choice(values)
            for parameter, values in SEARCH_SPACE.items()
        }

        candidates.append(candidate)

    best_score = float("-inf")
    best_params = None

    # -------------------------------------------------
    # Successive resource allocation
    # -------------------------------------------------

    for max_iter in HYPERBAND_ITERATIONS:

        print(
            f"\nHyperband Resource Level: "
            f"{max_iter} iterations"
        )

        scores = []

        for params in candidates:

            hyperband_params = params.copy()

            hyperband_params["max_iter"] = max_iter

            model = MLPClassifier(

                hidden_layer_sizes=
                    hyperband_params[
                        "hidden_layer_sizes"
                    ],

                activation=
                    hyperband_params[
                        "activation"
                    ],

                solver=
                    hyperband_params[
                        "solver"
                    ],

                learning_rate_init=
                    hyperband_params[
                        "learning_rate_init"
                    ],

                alpha=
                    hyperband_params[
                        "alpha"
                    ],

                batch_size=
                    hyperband_params[
                        "batch_size"
                    ],

                max_iter=max_iter,

                random_state=42

            )

            score = cross_val_score(

                model,

                X_train,

                y_train,

                cv=5,

                scoring="f1_weighted",

                n_jobs=1

            ).mean()

            scores.append(score)

            if score > best_score:

                best_score = score

                best_params = hyperband_params.copy()

        # -------------------------------------------------
        # Rank candidates
        # -------------------------------------------------

        ranked = sorted(

            zip(scores, candidates),

            key=lambda x: x[0],

            reverse=True

        )

        # -------------------------------------------------
        # Retain top-performing configurations
        # -------------------------------------------------

        survivor_count = max(

            1,

            len(ranked) // 2

        )

        candidates = [

            candidate.copy()

            for _, candidate in
            ranked[:survivor_count]

        ]

        print(
            f"Best F1 Score: "
            f"{best_score:.4f}"
        )

    end = time.time()

    runtime = end - start

    return (

        best_params,

        best_score,

        runtime

    )
# =====================================================
# RUN HYPERBAND
# =====================================================

print("\nBuilding Hyperband Optimizer...")

random.seed(42)

hyperband_best_params, \
hyperband_best_score, \
hyperband_runtime = run_hyperband()

print(
    "\nHyperband Optimization completed."
)

print(
    f"Optimization Time: "
    f"{hyperband_runtime:.2f} seconds"
)
# =====================================================
# RETRIEVE BEST HYPERPARAMETERS
# =====================================================

print(
    "\nBest Hyperparameters Found By Hyperband\n"
)

for key, value in hyperband_best_params.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{hyperband_best_score:.4f}"
)
# =====================================================
# BUILD HYPERBAND OPTIMIZED MLP MODEL
# =====================================================

print(
    "\nBuilding Hyperband Optimized MLP Model..."
)

hyperband_model = build_mlp(

    hyperband_best_params

)


# =====================================================
# TRAIN HYPERBAND OPTIMIZED MLP
# =====================================================

print(
    "\nTraining Hyperband Optimized MLP..."
)

hyperband_model, \
hyperband_training_runtime = train_mlp_model(

    hyperband_model,

    X_train,

    y_train

)

print(
    f"Training completed in "
    f"{hyperband_training_runtime:.2f} seconds"
)


# =====================================================
# EVALUATE HYPERBAND MODEL
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

    "Hyperband Optimized MLP Training Loss"


)


# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Hyperband",

    best_params=hyperband_best_params,

    best_cv_score=hyperband_best_score,

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
# Information obtained from high-performing solutions is
# then used to generate improved candidate solutions for
# subsequent generations.
#
# -----------------------------------------------------
# WHY LEA IS USED
# -----------------------------------------------------
#
# The Solar Flare dataset represents a multiclass
# classification problem with a highly imbalanced
# distribution across the four target classes.
#
# LEA is used to investigate whether a population-based
# evolutionary search strategy can identify effective MLP
# hyperparameter configurations for Solar Flare
# classification.
#
# Unlike Random Search, which evaluates independently
# sampled configurations, LEA iteratively improves a
# population based on the performance of previously
# evaluated candidate solutions.
#
# Compared with Bayesian Optimization, which builds a
# probabilistic model of the objective function, LEA
# improves candidate solutions through evolutionary
# population-based search.
#
# Compared with Hyperband, which focuses on allocating
# computational resources efficiently and eliminating
# weak configurations early, LEA focuses on improving
# candidate hyperparameter configurations across
# successive generations.
#
# Compared with TLBO, which uses teacher and learner
# interactions to update candidate solutions, LEA uses
# evolutionary learning mechanisms to improve the
# population.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Population-based exploration of the hyperparameter
#   search space.
#
# • Uses information from high-performing solutions to
#   guide subsequent generations.
#
# • Can explore different regions of the search space
#   through evolutionary updates.
#
# • Suitable for optimizing neural network
#   hyperparameters.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires multiple candidate evaluations across
#   successive generations.
#
# • Computational cost can increase with population size
#   and number of generations.
#
# • Performance depends on the design of the evolutionary
#   update mechanism.
# =====================================================


# =====================================================
# LEA POPULATION INITIALIZATION
# =====================================================
#
# LEA starts with a population of candidate solutions.
# Each candidate represents one possible combination of
# MLP hyperparameters selected from the common
# SEARCH_SPACE.
#
# The initial population provides a diverse set of
# hyperparameter configurations that can be evaluated and
# subsequently improved during the evolutionary search.
#
# A fixed random seed is used to ensure that the initial
# population can be reproduced across experimental runs.
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
# Each candidate solution is evaluated using the same
# 5-fold cross-validation procedure used by the other
# optimization algorithms.
#
# Weighted F1-score is used as the fitness measure.
# This maintains consistency with the Random Search,
# Bayesian Optimization and Hyperband experiments.
#
# A candidate configuration with a higher weighted
# F1-score represents a better-performing MLP
# configuration.
# =====================================================

def evaluate_lea_candidate(params):

    model = build_mlp(params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=1

    ).mean()

    return score


# =====================================================
# LEA PARAMETER CONFIGURATION
# =====================================================
#
# POPULATION_SIZE = 20
#
# The population size determines the number of candidate
# MLP hyperparameter configurations evaluated during each
# generation. A population of 20 provides sufficient
# diversity while keeping the computational cost of
# repeated 5-fold cross-validation manageable.
#
# SURVIVAL_RATE = 0.50
#
# The survival rate determines the proportion of the
# highest-performing candidate solutions retained after
# each generation. The top 50% are retained and used as
# parents for generating the next population.
#
# MAX_ITERATIONS = 20
#
# The optimization is performed for 20 generations.
# This provides multiple opportunities for candidate
# solutions to improve while keeping the optimization
# budget consistent with the other population-based
# optimization methods.
# =====================================================

POPULATION_SIZE = 20

SURVIVAL_RATE = 0.50

MAX_ITERATIONS = 5

best_score = float("-inf")

best_solution = None


# =====================================================
# START LEA OPTIMIZATION
# =====================================================
#
# Each generation performs the following steps:
#
# 1. Evaluate the current population.
# 2. Rank candidate solutions according to fitness.
# 3. Retain the highest-performing candidates.
# 4. Randomly select surviving candidates as parents.
# 5. Mutate one hyperparameter of each new candidate.
# 6. Create a new population.
#
# The best solution found across all generations is
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

    # -------------------------------------------------
    # EVALUATE CURRENT POPULATION
    # -------------------------------------------------

    for solution in lea_population:

        score = evaluate_lea_candidate(

            solution

        )

        fitness_scores.append(score)

        if score > best_score:

            best_score = score

            best_solution = solution.copy()


    print(
        f"Best F1 Score: "
        f"{best_score:.4f}"
    )


    # -------------------------------------------------
    # RANK CANDIDATE SOLUTIONS
    # -------------------------------------------------
    #
    # Candidates are sorted from highest to lowest
    # weighted F1-score.
    # -------------------------------------------------

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


    # -------------------------------------------------
    # SELECT SURVIVING CANDIDATES
    # -------------------------------------------------
    #
    # The highest-performing candidates are retained
    # and used as parents for the next generation.
    # -------------------------------------------------

    survivor_count = int(

        POPULATION_SIZE *
        SURVIVAL_RATE

    )

    survivors = ranked_population[

        :survivor_count

    ]


    # -------------------------------------------------
    # GENERATE NEXT POPULATION
    # -------------------------------------------------
    #
    # A surviving candidate is randomly selected and
    # one of its hyperparameters is randomly mutated.
    #
    # The solver remains fixed as Adam throughout the
    # experiment.
    # -------------------------------------------------

    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(

            survivors

        ).copy()

        parameter = random.choice(

            list(parent.keys())

        )

        if parameter != "solver":

            parent[parameter] = random.choice(

                SEARCH_SPACE[parameter]

            )

        new_population.append(

            parent

        )


    lea_population = new_population


# =====================================================
# LEA OPTIMIZATION COMPLETED
# =====================================================

end = time.time()

lea_runtime = end - start

print(
    "\nLEA Optimization completed."
)

print(
    f"Optimization Time: "
    f"{lea_runtime:.2f} seconds"
)


# =====================================================
# DISPLAY BEST LEA HYPERPARAMETERS
# =====================================================

print(
    "\nBest Hyperparameters Found By LEA\n"
)

for key, value in best_solution.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{best_score:.4f}"
)


# =====================================================
# BUILD LEA OPTIMIZED MLP MODEL
# =====================================================

print(
    "\nBuilding LEA Optimized MLP Model..."
)

lea_model = build_mlp(

    best_solution

)


# =====================================================
# TRAIN LEA OPTIMIZED MLP
# =====================================================

print(
    "\nTraining LEA Optimized MLP..."
)

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
# EVALUATE LEA OPTIMIZED MLP
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

plot_training_loss(

    lea_model,

    "LEA Optimized MLP Training Loss"

)


# =====================================================
# PRINT LEA OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="LEA",

    best_params=best_solution,

    best_cv_score=best_score,

    optimization_time=lea_runtime,

    training_time=lea_training_runtime

)
# =====================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# =====================================================
#
# TLBO is a population-based optimization algorithm
# inspired by the teaching and learning process in a
# classroom.
#
# Each candidate solution represents one possible
# combination of MLP hyperparameters.
#
# The optimization process consists of two phases:
#
# 1. Teacher Phase
# 2. Learner Phase
#
# In the Teacher Phase, the best-performing candidate
# acts as the teacher and influences the remaining
# candidate solutions.
#
# In the Learner Phase, candidate solutions interact
# with randomly selected learners to further improve
# their hyperparameter configurations.
#
# Candidate solutions are evaluated using weighted
# F1-score obtained through 5-fold cross-validation.
#
# -----------------------------------------------------
# WHY TLBO IS USED
# -----------------------------------------------------
#
# The Solar Flare dataset represents a multiclass
# classification problem with a highly imbalanced
# distribution across the four target classes.
#
# TLBO is included to investigate whether a
# teacher-learner based population optimization strategy
# can identify effective MLP hyperparameter
# configurations.
#
# Unlike Random Search, TLBO does not independently
# sample every candidate configuration. Instead,
# information from high-performing candidates is used
# to guide the improvement of other candidates.
#
# Compared with Bayesian Optimization, TLBO does not
# construct a probabilistic surrogate model.
# Instead, candidate solutions are improved through
# teacher and learner interactions.
#
# Compared with Hyperband, which focuses on efficient
# allocation of computational resources, TLBO focuses
# on improving candidate solutions through population
# interactions.
#
# Compared with LEA, which uses evolutionary survival
# and mutation mechanisms, TLBO improves candidate
# solutions through teacher and learner phases.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Population-based exploration of the hyperparameter
#   search space.
#
# • Does not require a separate probabilistic model.
#
# • Uses information from the best-performing candidate
#   to guide population improvement.
#
# • Contains both teacher-based and learner-based
#   learning mechanisms.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires repeated evaluation of candidate solutions.
#
# • Computational cost increases with population size
#   and number of iterations.
#
# • Performance depends on the initial population.
#
# • Random learner selection may generate inferior
#   candidate solutions.
# =====================================================


# =====================================================
# TLBO POPULATION INITIALIZATION
# =====================================================
#
# TLBO begins with a population of candidate solutions.
# Each candidate represents one possible combination of
# MLP hyperparameters selected from the common
# SEARCH_SPACE.
#
# A fixed random seed is used to ensure that the initial
# population can be reproduced across experimental runs.
# =====================================================

random.seed(42)

TLBO_POPULATION_SIZE = 20

TLBO_MAX_ITERATIONS = 5

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
# Each candidate solution is evaluated using the same
# 5-fold cross-validation procedure used by the other
# optimization algorithms.
#
# Weighted F1-score is used as the fitness measure to
# maintain consistency with the other optimization
# methods.
# =====================================================

def evaluate_tlbo_candidate(params):

    model = build_mlp(params)

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_weighted",

        n_jobs=1

    ).mean()

    return score


# =====================================================
# INITIALIZE BEST TLBO SOLUTION
# =====================================================

tlbo_best_score = float("-inf")

tlbo_best_solution = None


# =====================================================
# START TLBO OPTIMIZATION
# =====================================================
#
# Each TLBO iteration consists of two learning phases:
#
# 1. Teacher Phase
# 2. Learner Phase
#
# The best-performing candidate is identified and used
# as the teacher.
#
# Candidate solutions are then updated through teacher
# and learner interactions.
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

    tlbo_fitness = []

    for candidate in tlbo_population:

        score = evaluate_tlbo_candidate(

            candidate

        )

        tlbo_fitness.append(score)


    # =================================================
    # IDENTIFY TEACHER
    # =================================================
    #
    # The candidate with the highest fitness value is
    # selected as the teacher for the current iteration.
    # =================================================

    best_index = tlbo_fitness.index(

        max(tlbo_fitness)

    )

    teacher = tlbo_population[
        best_index
    ].copy()

    teacher_score = tlbo_fitness[
        best_index
    ]


    # Update global best solution

    if teacher_score > tlbo_best_score:

        tlbo_best_score = teacher_score

        tlbo_best_solution = teacher.copy()


    print(
        f"Best F1 Score: "
        f"{tlbo_best_score:.4f}"
    )


    # =================================================
    # TEACHER PHASE
    # =================================================
    #
    # The teacher attempts to move the population toward
    # better-performing hyperparameter configurations.
    #
    # A candidate is randomly selected and one of its
    # hyperparameters is updated using information from
    # the teacher.
    #
    # Since MLP hyperparameters are discrete values,
    # the update is implemented through selection from
    # the predefined SEARCH_SPACE.
    # =================================================

    for i in range(

        TLBO_POPULATION_SIZE

    ):

        current_solution = tlbo_population[
            i
        ].copy()

        parameter = random.choice(

            list(current_solution.keys())

        )

        if parameter != "solver":

            teacher_value = teacher[
                parameter
            ]

            current_solution[
                parameter
            ] = teacher_value


        new_score = evaluate_tlbo_candidate(

            current_solution

        )

        if new_score > tlbo_fitness[i]:

            tlbo_population[i] = (
                current_solution
            )

            tlbo_fitness[i] = new_score


    # =================================================
    # LEARNER PHASE
    # =================================================
    #
    # Each candidate interacts with another randomly
    # selected candidate from the population.
    #
    # If the randomly selected learner has a better
    # fitness value, information from that learner is
    # used to update the current candidate.
    #
    # Otherwise, the current candidate influences the
    # randomly selected learner.
    # =================================================

    for i in range(

        TLBO_POPULATION_SIZE

    ):

        learner_index = random.choice(

            [

                j

                for j in range(
                    TLBO_POPULATION_SIZE
                )

                if j != i

            ]

        )

        current_solution = (
            tlbo_population[i].copy()
        )

        learner_solution = (
            tlbo_population[
                learner_index
            ].copy()
        )


        parameter = random.choice(

            list(current_solution.keys())

        )


        if parameter != "solver":

            if (
                tlbo_fitness[
                    learner_index
                ]
                >
                tlbo_fitness[i]
            ):

                current_solution[
                    parameter
                ] = learner_solution[
                    parameter
                ]

            else:

                current_solution[
                    parameter
                ] = random.choice(

                    SEARCH_SPACE[
                        parameter
                    ]

                )


        new_score = evaluate_tlbo_candidate(

            current_solution

        )

        if new_score > tlbo_fitness[i]:

            tlbo_population[i] = (
                current_solution
            )

            tlbo_fitness[i] = new_score


# =====================================================
# TLBO OPTIMIZATION COMPLETED
# =====================================================

end = time.time()

tlbo_runtime = end - start

print(
    "\nTLBO Optimization completed."
)

print(
    f"Optimization Time: "
    f"{tlbo_runtime:.2f} seconds"
)


# =====================================================
# DISPLAY BEST TLBO HYPERPARAMETERS
# =====================================================

print(
    "\nBest Hyperparameters Found By TLBO\n"
)

for key, value in tlbo_best_solution.items():

    print(
        f"{key}: {value}"
    )

print(
    f"\nBest Cross Validation F1 Score: "
    f"{tlbo_best_score:.4f}"
)


# =====================================================
# BUILD TLBO OPTIMIZED MLP MODEL
# =====================================================

print(
    "\nBuilding TLBO Optimized MLP Model..."
)

tlbo_model = build_mlp(

    tlbo_best_solution

)


# =====================================================
# TRAIN TLBO OPTIMIZED MLP
# =====================================================

print(
    "\nTraining TLBO Optimized MLP..."
)

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
# EVALUATE TLBO OPTIMIZED MLP
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

    "TLBO Optimized MLP Training Loss"

)


# =====================================================
# PRINT TLBO OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="TLBO",

    best_params=tlbo_best_solution,

    best_cv_score=tlbo_best_score,

    optimization_time=tlbo_runtime,

    training_time=tlbo_training_runtime

)
# =====================================================
# CREATE RESULTS DIRECTORY
# =====================================================
#
# Create a dedicated directory for storing the final
# experimental results of the Solar Flare dataset.
#
# The results will contain the performance metrics and
# execution times of the baseline model and all
# hyperparameter optimization methods.
# =====================================================

RESULTS_DIR = "results"

os.makedirs(

    RESULTS_DIR,

    exist_ok=True

)


# =====================================================
# SAVE EXPERIMENTAL RESULTS
# =====================================================
#
# Store the performance metrics and execution times of
# all optimization methods.
#
# The results are saved as a CSV file so that they can
# later be used for comparative analysis, tables and
# visualizations in the thesis.
#
# The following methods are compared:
#
# • Baseline MLP
# • Random Search
# • Bayesian Optimization
# • Hyperband
# • LEA
# • TLBO
#
# Accuracy, Precision, Recall and F1-score represent
# performance on the independent test dataset.
#
# Optimization Time represents the time required to
# search for the best hyperparameter configuration.
#
# Training Time represents the time required to train
# the final MLP using the selected configuration.
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
        random_accuracy,
        bayesian_accuracy,
        hyperband_accuracy,
        lea_accuracy,
        tlbo_accuracy

    ],

    "Precision": [

        baseline_precision,
        random_precision,
        bayesian_precision,
        hyperband_precision,
        lea_precision,
        tlbo_precision

    ],

    "Recall": [

        baseline_recall,
        random_recall,
        bayesian_recall,
        hyperband_recall,
        lea_recall,
        tlbo_recall

    ],

    "F1 Score": [

        baseline_f1,
        random_f1,
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

        baseline_runtime,
        random_training_runtime,
        bayesian_training_runtime,
        hyperband_training_runtime,
        lea_training_runtime,
        tlbo_training_runtime

    ]

})


# =====================================================
# DISPLAY COMPARISON RESULTS
# =====================================================

print("\n" + "=" * 80)

print("SOLAR FLARE EXPERIMENTAL COMPARISON")

print("=" * 80)

print(

    comparison_results.to_string(
        index=False
    )

)


# =====================================================
# SAVE RESULTS TO CSV
# =====================================================

comparison_results.to_csv(

    f"{RESULTS_DIR}/solar_flare_results.csv",

    index=False

)

print(

    "\nResults saved successfully to: "

    f"{RESULTS_DIR}/solar_flare_results.csv"

)