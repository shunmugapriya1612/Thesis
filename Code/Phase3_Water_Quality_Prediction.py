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
# Original Machine Learning Task:
# Binary Classification
#
# Current Machine Learning Task:
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
# =====================================================
# INPUT FEATURES
# =====================================================
#
# The dataset contains numerical water-quality measurements,
# including:
#
# • pH
# • Hardness
# • Solids
# • Chloramines
# • Sulfate
# • Conductivity
# • Organic carbon
# • Trihalomethanes
# • Turbidity
#
# All input variables are numerical.
#
# =====================================================
# TARGET VARIABLE
# =====================================================
#
# Target:
# Potability
#
# Number of Classes:
# 2
#
# The Potability variable represents two classes:
# 0 = Non-Potable
# 1 = Potable
#
# =====================================================
# WHY THIS DATASET WAS SELECTED
# =====================================================
#
# The Water Potability dataset contains missing values,
# moderate class imbalance, and substantial overlap between
# the classes. These characteristics make it useful for
# evaluating whether MLP hyperparameter optimization can
# improve classification performance without changing the
# underlying feature set.
# =====================================================

import random
import time
import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    PredefinedSplit,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "Data/water_potability.csv"

# Allow execution from the project root as well as from Code/.
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "../Data/water_potability.csv"

print("\nDataset Path:")
print(os.path.abspath(DATASET_PATH))

dataset = pd.read_csv(
    DATASET_PATH
)

print("\nDataset Shape:")
print(dataset.shape)

print("\nFirst Five Rows:")
print(dataset.head())

# =====================================================
# FEATURE MATRIX AND TARGET VARIABLE
# =====================================================
#
# Separate the predictor variables (X) from the target
# variable (y). The Potability column represents the class
# labels to be predicted by the MLP classifier, while all
# remaining variables are used as input features.
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
# One-Hot Encoding converts categorical predictor variables
# into numerical representations that can be processed by
# machine learning algorithms.
#
# The Water Potability dataset contains only continuous
# numerical predictor variables. Therefore, no categorical
# feature encoding is required and this preprocessing step
# is omitted.
# =====================================================

# =====================================================
# TRAIN-VALIDATION-TEST SPLIT
# =====================================================
#
# The Water Quality experiment uses a fixed three-way split:
#
# • 64% training data
# • 16% validation data
# • 20% testing data
#
# Stratified sampling preserves the class distribution.
# The validation set is used during hyperparameter
# optimization, while the test set remains untouched until
# final evaluation.
# =====================================================

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=0.20,
    stratify=y_train_temp,
    random_state=42
)

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
# MISSING VALUE IMPUTATION
# =====================================================
#
# The Water Potability dataset contains missing values in
# several predictor variables. Median imputation is retained
# from Phase 2 because it provided the strongest validation
# Macro F1 among the tested imputation strategies.
#
# The imputer is fitted only on the training data. The same
# transformation is then applied to validation and test data
# to prevent information leakage.
# =====================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_validation = imputer.transform(
    X_validation
)

X_test = imputer.transform(
    X_test
)

# =====================================================
# FEATURE STANDARDIZATION
# =====================================================
#
# Feature standardization scales the predictor variables to
# have zero mean and unit variance. This is important for MLP
# classifiers because neural-network optimization can be
# affected when input features have substantially different
# scales.
#
# The StandardScaler is fitted only on the training data.
# The same transformation is applied to validation and test
# data to prevent information leakage.
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_validation = scaler.transform(
    X_validation
)

X_test = scaler.transform(
    X_test
)

print("\nMissing Value Imputation Completed.")
print("Feature Standardization Completed.")

# =====================================================
# FINAL DEVELOPMENT SET FOR MODEL RETRAINING
# =====================================================
#
# After hyperparameter selection, the best configuration is
# retrained using the complete development data consisting
# of the training and validation sets. The independent test
# set is used only once for final evaluation.
# =====================================================

X_development = np.vstack(
    [X_train, X_validation]
)

y_development = pd.concat(
    [y_train.reset_index(drop=True), y_validation.reset_index(drop=True)],
    ignore_index=True
)

# =====================================================
# VALIDATION SPLIT OBJECT
# =====================================================
#
# PredefinedSplit allows sklearn search procedures to use the
# fixed validation set rather than creating new random folds.
# Training rows receive -1, while validation rows receive 0.
# =====================================================

validation_fold = np.concatenate(
    [
        np.full(
            len(X_train),
            -1,
            dtype=int
        ),
        np.zeros(
            len(X_validation),
            dtype=int
        )
    ]
)

X_optimization = np.vstack(
    [X_train, X_validation]
)

y_optimization = pd.concat(
    [y_train.reset_index(drop=True), y_validation.reset_index(drop=True)],
    ignore_index=True
)

validation_split = PredefinedSplit(
    test_fold=validation_fold
)

# =====================================================
# MLP HYPERPARAMETER SEARCH SPACE
# =====================================================
#
# Define the hyperparameter search space used by all
# optimization algorithms in this study. Each optimization
# method explores different combinations from this common
# search space, ensuring that the underlying model choices
# remain consistent across the comparison.
#
# Hyperparameters included in the search space:
#
# • Hidden layer architecture
# • Activation function
# • Learning rate
# • L2 regularization strength (alpha)
# • Batch size
#
# The Adam optimizer is fixed throughout the study so that
# observed differences are attributable to the hyperparameter
# search methods rather than changes in the MLP solver.
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

    ]

}

# =====================================================
# BUILD MLP MODEL
# =====================================================
#
# Construct an MLP classifier using a specified set of
# hyperparameters. This function is shared by the baseline
# model and all optimization algorithms so that only the
# hyperparameter values change between experiments.
# =====================================================

def build_mlp(params, max_iter=500):

    return MLPClassifier(

        hidden_layer_sizes=params["hidden_layer_sizes"],

        activation=params["activation"],

        solver=params.get(
            "solver",
            "adam"
        ),

        learning_rate_init=params["learning_rate_init"],

        alpha=params["alpha"],

        batch_size=params["batch_size"],

        max_iter=max_iter,

        random_state=42

    )

# =====================================================
# VALIDATION MODEL EVALUATION
# =====================================================
#
# Hyperparameter optimization is performed on the fixed
# validation set using Macro F1 as the primary fitness metric.
# Macro F1 gives equal importance to both Potability classes,
# which is appropriate because the dataset is moderately
# imbalanced and Phase 1 showed weak recall for the potable
# class.
# =====================================================

def evaluate_validation_model(model):

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    macro_precision = precision_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        predictions
    )

    potable_recall = recall_score(
        y_validation,
        predictions,
        pos_label=1,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "potable_recall": potable_recall
    }

# =====================================================
# FINAL TEST MODEL EVALUATION
# =====================================================
#
# The final model is trained on the complete development set
# (training + validation) and evaluated on the untouched test
# set. No hyperparameter decisions are made using test data.
# =====================================================

def evaluate_test_model(model, method_name):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    potable_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        average="binary",
        zero_division=0
    )

    print(f"\n========== {method_name.upper()} RESULTS ==========")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Macro Precision      : {macro_precision:.4f}")
    print(f"Macro Recall         : {macro_recall:.4f}")
    print(f"Macro F1 Score       : {macro_f1:.4f}")
    print(f"Weighted F1 Score    : {weighted_f1:.4f}")
    print(f"Balanced Accuracy    : {balanced_accuracy:.4f}")
    print(f"Potable Recall       : {potable_recall:.4f}")

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-Potable", "Potable"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix\n")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "potable_recall": potable_recall,
        "predictions": predictions
    }

# =====================================================
# TRAIN MLP MODEL
# =====================================================
#
# Train an MLP classifier and measure training time.
# =====================================================

def train_mlp_model(model, X_data, y_data):

    start = time.time()

    model.fit(
        X_data,
        y_data
    )

    end = time.time()

    training_time = end - start

    return model, training_time

# =====================================================
# PLOT TRAINING LOSS
# =====================================================

def plot_training_loss(model, title):

    if hasattr(model, "loss_curve_"):

        plt.figure(figsize=(8, 5))

        plt.plot(
            model.loss_curve_
        )

        plt.xlabel("Iteration")

        plt.ylabel("Training Loss")

        plt.title(title)

        plt.grid(True)

        plt.show()

# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

def print_optimization_summary(
    method_name,
    best_params,
    best_score,
    optimization_time,
    training_time
):

    print(
        f"\nBest Hyperparameters Found By {method_name}\n"
    )

    for key, value in best_params.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nBest Validation Macro F1: {best_score:.4f}"
    )

    print(
        f"\nOptimization Time : {optimization_time:.2f} seconds"
    )

    print(
        f"Training Time     : {training_time:.2f} seconds"
    )

# =====================================================
# BASELINE MULTI-LAYER PERCEPTRON (MLP)
# =====================================================
#
# The baseline MLP is the reference model for evaluating all
# hyperparameter optimization algorithms.
#
# The Water Quality baseline retains the configuration used
# in Phase 2:
#
# • Hidden Layer Architecture : (64, 32)
# • Activation Function       : ReLU
# • Optimizer                 : Adam
# • Learning Rate             : 0.001
# • L2 Regularization (alpha) : 0.0001
# • Batch Size                : auto
#
# The baseline provides the reference against which all
# optimized configurations are compared.
# =====================================================

baseline_params = {

    "hidden_layer_sizes": (64, 32),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": "auto"

}

# =====================================================
# BUILD BASELINE MLP MODEL
# =====================================================

print("\nBuilding Baseline MLP Model...")

baseline_validation_model = build_mlp(
    baseline_params
)

# =====================================================
# TRAIN BASELINE MLP MODEL
# =====================================================

print("\nTraining Baseline MLP...")

baseline_validation_model, baseline_validation_training_time = train_mlp_model(
    baseline_validation_model,
    X_train,
    y_train
)

baseline_validation_metrics = evaluate_validation_model(
    baseline_validation_model
)

print("\nBaseline Validation Performance")
print(f"Validation Accuracy          : {baseline_validation_metrics['accuracy']:.4f}")
print(f"Validation Macro Precision   : {baseline_validation_metrics['macro_precision']:.4f}")
print(f"Validation Macro Recall      : {baseline_validation_metrics['macro_recall']:.4f}")
print(f"Validation Macro F1          : {baseline_validation_metrics['macro_f1']:.4f}")
print(f"Validation Balanced Accuracy : {baseline_validation_metrics['balanced_accuracy']:.4f}")
print(f"Validation Potable Recall    : {baseline_validation_metrics['potable_recall']:.4f}")

# =====================================================
# FINAL BASELINE MODEL TRAINING
# =====================================================

print("\nTraining Final Baseline MLP on Development Data...")

baseline_model = build_mlp(
    baseline_params
)

baseline_model, baseline_training_runtime = train_mlp_model(
    baseline_model,
    X_development,
    y_development
)

baseline_test_metrics = evaluate_test_model(
    baseline_model,
    "Baseline MLP"
)

plot_training_loss(
    baseline_model,
    "Baseline MLP Training Loss"
)

print("\nBaseline Hyperparameters")
for key, value in baseline_params.items():
    print(f"{key}: {value}")

# =====================================================
# RANDOM SEARCH
# =====================================================
#
# Random Search samples hyperparameter combinations from the
# predefined search space. The fixed validation split is used
# as the optimization objective so that the Water Quality
# experiment follows the Phase 2 validation protocol.
# =====================================================

N_RANDOM_TRIALS = 20

print("\nBuilding Random Search Optimizer...")

random_search = RandomizedSearchCV(

    estimator=MLPClassifier(

        max_iter=500,

        random_state=42

    ),

    param_distributions=SEARCH_SPACE,

    n_iter=N_RANDOM_TRIALS,

    scoring="f1_macro",

    cv=validation_split,

    random_state=42,

    n_jobs=-1

)

print("\nStarting Random Search Optimization...")

start = time.time()

random_search.fit(
    X_optimization,
    y_optimization
)

end = time.time()

random_search_runtime = end - start

print("\nRandom Search completed.")
print(f"Optimization Time: {random_search_runtime:.2f} seconds")
print(f"Best Validation Macro F1: {random_search.best_score_:.4f}")

print("\nBest Hyperparameters Found By Random Search")
for key, value in random_search.best_params_.items():
    print(f"{key}: {value}")

# =====================================================
# BUILD RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding Random Search Optimized MLP Model...")

random_model = build_mlp(
    random_search.best_params_
)

# =====================================================
# TRAIN RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================

print("\nTraining Random Search Optimized MLP...")

random_model, random_training_runtime = train_mlp_model(
    random_model,
    X_development,
    y_development
)

random_test_metrics = evaluate_test_model(
    random_model,
    "Random Search"
)

plot_training_loss(
    random_model,
    "Random Search Optimized MLP Training Loss"
)

print_optimization_summary(
    method_name="Random Search",
    best_params=random_search.best_params_,
    best_score=random_search.best_score_,
    optimization_time=random_search_runtime,
    training_time=random_training_runtime
)

# =====================================================
# BAYESIAN OPTIMIZATION USING OPTUNA
# =====================================================
#
# Optuna performs sequential Bayesian optimization using the
# validation Macro F1 as the objective. String labels are used
# for hidden-layer architectures so that categorical choices
# are handled cleanly.
# =====================================================

HIDDEN_LAYER_MAP = {
    "32": (32,),
    "64": (64,),
    "128": (128,),
    "64_32": (64, 32),
    "128_64": (128, 64),
    "128_64_32": (128, 64, 32)
}

print("\nBuilding Bayesian Optimizer...")

bayesian_convergence = []


def bayesian_objective(trial):

    hidden_layer_name = trial.suggest_categorical(
        "hidden_layer_sizes",
        list(HIDDEN_LAYER_MAP.keys())
    )

    params = {

        "hidden_layer_sizes": HIDDEN_LAYER_MAP[
            hidden_layer_name
        ],

        "activation": trial.suggest_categorical(
            "activation",
            SEARCH_SPACE["activation"]
        ),

        "solver": "adam",

        "learning_rate_init": trial.suggest_categorical(
            "learning_rate_init",
            SEARCH_SPACE["learning_rate_init"]
        ),

        "alpha": trial.suggest_categorical(
            "alpha",
            SEARCH_SPACE["alpha"]
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            SEARCH_SPACE["batch_size"]
        )

    }

    model = build_mlp(
        params
    )

    model.fit(
        X_train,
        y_train
    )

    score = evaluate_validation_model(
        model
    )["macro_f1"]

    bayesian_convergence.append(
        {
            "Trial": trial.number + 1,
            "Best Validation Macro F1": score
        }
    )

    return score

study = optuna.create_study(
    direction="maximize",
    study_name="Water_Quality_MLP_Bayesian"
)

print("\nStarting Bayesian Optimization...")

start = time.time()

study.optimize(
    bayesian_objective,
    n_trials=20
)

end = time.time()

bayesian_runtime = end - start

best_params = study.best_params.copy()

best_params["hidden_layer_sizes"] = HIDDEN_LAYER_MAP[
    best_params["hidden_layer_sizes"]
]

best_params["solver"] = "adam"

print("\nBayesian Optimization completed.")
print(f"Optimization Time: {bayesian_runtime:.2f} seconds")
print(f"Best Validation Macro F1: {study.best_value:.4f}")

print("\nBest Hyperparameters Found By Bayesian Optimization")
for key, value in best_params.items():
    print(f"{key}: {value}")

# =====================================================
# BUILD BAYESIAN OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding Bayesian Optimized MLP Model...")

bayesian_model = build_mlp(
    best_params
)

# =====================================================
# TRAIN BAYESIAN OPTIMIZED MLP MODEL
# =====================================================

print("\nTraining Bayesian Optimized MLP...")

bayesian_model, bayesian_training_runtime = train_mlp_model(
    bayesian_model,
    X_development,
    y_development
)

bayesian_test_metrics = evaluate_test_model(
    bayesian_model,
    "Bayesian Optimization"
)

plot_training_loss(
    bayesian_model,
    "Bayesian Optimized MLP Training Loss"
)

print_optimization_summary(
    method_name="Bayesian Optimization",
    best_params=best_params,
    best_score=study.best_value,
    optimization_time=bayesian_runtime,
    training_time=bayesian_training_runtime
)

# =====================================================
# HYPERBAND OPTIMIZATION
# =====================================================
#
# Hyperband is a resource-aware hyperparameter optimization
# method. It begins with many candidate configurations at a
# smaller max_iter value and gradually retains only promising
# configurations for larger training budgets.
#
# The maximum resource remains 500 iterations, matching the
# maximum training budget used by the baseline and other
# optimizers.
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

    scoring="f1_macro",

    cv=validation_split,

    random_state=42,

    n_jobs=-1

)

print("\nStarting Hyperband Optimization...")

start = time.time()

hyperband.fit(
    X_optimization,
    y_optimization
)

end = time.time()

hyperband_runtime = end - start

print("\nHyperband Optimization completed.")
print(f"Optimization Time: {hyperband_runtime:.2f} seconds")
print(f"Best Validation Macro F1: {hyperband.best_score_:.4f}")

print("\nBest Hyperparameters Found By Hyperband")
for key, value in hyperband.best_params_.items():
    print(f"{key}: {value}")

# =====================================================
# BUILD HYPERBAND OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding Hyperband Optimized MLP Model...")

hyperband_model = build_mlp(
    hyperband.best_params_
)

# =====================================================
# TRAIN HYPERBAND OPTIMIZED MLP MODEL
# =====================================================

print("\nTraining Hyperband Optimized MLP...")

hyperband_model, hyperband_training_runtime = train_mlp_model(
    hyperband_model,
    X_development,
    y_development
)

hyperband_test_metrics = evaluate_test_model(
    hyperband_model,
    "Hyperband"
)

plot_training_loss(
    hyperband_model,
    "Hyperband Optimized MLP Training Loss"
)

print_optimization_summary(
    method_name="Hyperband",
    best_params=hyperband.best_params_,
    best_score=hyperband.best_score_,
    optimization_time=hyperband_runtime,
    training_time=hyperband_training_runtime
)

# =====================================================
# LEARNING EVOLUTIONARY ALGORITHM (LEA)
# =====================================================
#
# LEA is a population-based hyperparameter optimization
# algorithm. Candidate solutions are evaluated, the strongest
# candidates survive, and new candidates are generated through
# mutation of selected hyperparameters.
#
# The implementation follows the same compact evolutionary
# structure used in the Appliance Energy code rather than a
# more elaborate continuous-domain formulation.
# =====================================================

POPULATION_SIZE = 20
SURVIVAL_RATE = 0.50
MAX_ITERATIONS = 5

print("\nGenerating LEA Initial Population...")

lea_population = []

for _ in range(POPULATION_SIZE):

    candidate = {

        parameter: random.choice(
            values
        )

        for parameter, values in SEARCH_SPACE.items()

    }

    lea_population.append(
        candidate
    )

print("Initial Population Created.")
print(f"Population Size: {len(lea_population)}")

lea_best_score = float("-inf")
lea_best_solution = None
lea_runtime_start = time.time()
lea_convergence = []
lea_evaluations = 0

for iteration in range(MAX_ITERATIONS):

    print(
        f"\nGeneration {iteration + 1}/{MAX_ITERATIONS}"
    )

    lea_fitness = []

    for solution in lea_population:

        model = build_mlp(
            solution
        )

        model.fit(
            X_train,
            y_train
        )

        score = evaluate_validation_model(
            model
        )["macro_f1"]

        lea_evaluations += 1
        lea_fitness.append(
            score
        )

        if score > lea_best_score:

            lea_best_score = score

            lea_best_solution = solution.copy()

    ranked_population = [

        solution

        for _, solution in sorted(

            zip(
                lea_fitness,
                lea_population
            ),

            key=lambda x: x[0],

            reverse=True

        )

    ]

    print(
        f"Best Macro F1: {lea_best_score:.4f}"
    )

    lea_convergence.append(
        {
            "Generation": iteration + 1,
            "Best Validation Macro F1": lea_best_score
        }
    )

    survivor_count = int(
        POPULATION_SIZE * SURVIVAL_RATE
    )

    survivors = ranked_population[
        :survivor_count
    ]

    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(
            survivors
        ).copy()

        parameter = random.choice(
            list(SEARCH_SPACE.keys())
        )

        parent[parameter] = random.choice(
            SEARCH_SPACE[parameter]
        )

        new_population.append(
            parent
        )

    lea_population = new_population

lea_runtime = time.time() - lea_runtime_start

print("\nLEA Optimization completed.")
print(f"Optimization Time: {lea_runtime:.2f} seconds")
print(f"Evaluations: {lea_evaluations}")

print("\nBest Hyperparameters Found By LEA")
for key, value in lea_best_solution.items():
    print(f"{key}: {value}")

print(
    f"\nBest Validation Macro F1: {lea_best_score:.4f}"
)

# =====================================================
# BUILD LEA OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding LEA Optimized MLP Model...")

lea_model = build_mlp(
    lea_best_solution
)

# =====================================================
# TRAIN LEA OPTIMIZED MLP MODEL
# =====================================================

print("\nTraining LEA Optimized MLP...")

lea_model, lea_training_runtime = train_mlp_model(
    lea_model,
    X_development,
    y_development
)

lea_test_metrics = evaluate_test_model(
    lea_model,
    "LEA"
)

plot_training_loss(
    lea_model,
    "LEA Optimized MLP Training Loss"
)

print_optimization_summary(
    method_name="LEA",
    best_params=lea_best_solution,
    best_score=lea_best_score,
    optimization_time=lea_runtime,
    training_time=lea_training_runtime
)

# =====================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# =====================================================
#
# TLBO is a population-based hyperparameter optimization
# algorithm inspired by teaching and learning. The best
# learner acts as the teacher, while learners can also learn
# from better-performing peers.
# =====================================================

TLBO_POPULATION_SIZE = 20
TLBO_MAX_ITERATIONS = 5

print("\nGenerating TLBO Initial Population...")

tlbo_population = []

for _ in range(TLBO_POPULATION_SIZE):

    candidate = {

        parameter: random.choice(
            values
        )

        for parameter, values in SEARCH_SPACE.items()

    }

    tlbo_population.append(
        candidate
    )

print("Initial Population Created.")
print(f"Population Size: {len(tlbo_population)}")

tlbo_best_score = float("-inf")
tlbo_best_solution = None
tlbo_runtime_start = time.time()
tlbo_convergence = []
tlbo_evaluations = 0

def evaluate_tlbo_solution(solution):

    model = build_mlp(
        solution
    )

    model.fit(
        X_train,
        y_train
    )

    score = evaluate_validation_model(
        model
    )["macro_f1"]

    return score

for iteration in range(TLBO_MAX_ITERATIONS):

    print(
        f"\nGeneration {iteration + 1}/{TLBO_MAX_ITERATIONS}"
    )

    fitness_scores = []

    # -------------------------------------------------
    # Evaluate Current Population
    # -------------------------------------------------

    for solution in tlbo_population:

        score = evaluate_tlbo_solution(
            solution
        )

        tlbo_evaluations += 1
        fitness_scores.append(
            score
        )

        if score > tlbo_best_score:

            tlbo_best_score = score

            tlbo_best_solution = solution.copy()

    teacher_index = fitness_scores.index(
        max(fitness_scores)
    )

    teacher = tlbo_population[
        teacher_index
    ]

    teacher_score = fitness_scores[
        teacher_index
    ]

    print(
        f"Teacher Macro F1: {teacher_score:.4f}"
    )

    # -------------------------------------------------
    # Teacher Phase
    # -------------------------------------------------

    for i in range(
        TLBO_POPULATION_SIZE
    ):

        if i == teacher_index:
            continue

        new_solution = tlbo_population[
            i
        ].copy()

        parameter = random.choice(
            [
                key
                for key in SEARCH_SPACE.keys()
                if key != "solver"
            ]
        )

        new_solution[parameter] = teacher[
            parameter
        ]

        new_score = evaluate_tlbo_solution(
            new_solution
        )

        tlbo_evaluations += 1

        if new_score > fitness_scores[i]:

            tlbo_population[i] = new_solution

            fitness_scores[i] = new_score

            if new_score > tlbo_best_score:

                tlbo_best_score = new_score

                tlbo_best_solution = new_solution.copy()

    # -------------------------------------------------
    # Learner Phase
    # -------------------------------------------------

    for i in range(
        TLBO_POPULATION_SIZE
    ):

        partner = random.randrange(
            TLBO_POPULATION_SIZE
        )

        while partner == i:

            partner = random.randrange(
                TLBO_POPULATION_SIZE
            )

        learner = tlbo_population[
            i
        ].copy()

        parameter = random.choice(
            [
                key
                for key in SEARCH_SPACE.keys()
                if key != "solver"
            ]
        )

        if fitness_scores[partner] > fitness_scores[i]:

            learner[parameter] = tlbo_population[
                partner
            ][parameter]

        else:

            learner[parameter] = random.choice(
                SEARCH_SPACE[parameter]
            )

        learner_score = evaluate_tlbo_solution(
            learner
        )

        tlbo_evaluations += 1

        if learner_score > fitness_scores[i]:

            tlbo_population[i] = learner

            fitness_scores[i] = learner_score

            if learner_score > tlbo_best_score:

                tlbo_best_score = learner_score

                tlbo_best_solution = learner.copy()

    print(
        f"Best Macro F1: {tlbo_best_score:.4f}"
    )

    tlbo_convergence.append(
        {
            "Generation": iteration + 1,
            "Best Validation Macro F1": tlbo_best_score
        }
    )

tlbo_runtime = time.time() - tlbo_runtime_start

print("\nTLBO Optimization completed.")
print(f"Optimization Time: {tlbo_runtime:.2f} seconds")
print(f"Evaluations: {tlbo_evaluations}")

print("\nBest Hyperparameters Found By TLBO")
for key, value in tlbo_best_solution.items():
    print(f"{key}: {value}")

print(
    f"\nBest Validation Macro F1: {tlbo_best_score:.4f}"
)

# =====================================================
# BUILD TLBO OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding TLBO Optimized MLP Model...")

tlbo_model = build_mlp(
    tlbo_best_solution
)

# =====================================================
# TRAIN TLBO OPTIMIZED MLP MODEL
# =====================================================

print("\nTraining TLBO Optimized MLP...")

tlbo_model, tlbo_training_runtime = train_mlp_model(
    tlbo_model,
    X_development,
    y_development
)

tlbo_test_metrics = evaluate_test_model(
    tlbo_model,
    "TLBO"
)

plot_training_loss(
    tlbo_model,
    "TLBO Optimized MLP Training Loss"
)

print_optimization_summary(
    method_name="TLBO",
    best_params=tlbo_best_solution,
    best_score=tlbo_best_score,
    optimization_time=tlbo_runtime,
    training_time=tlbo_training_runtime
)

# =====================================================
# SIMULATED ANNEALING (SA)
# =====================================================
#
# Simulated Annealing (SA) is added after TLBO so that
# both TLBO and SA can be compared in the final Phase 3
# results.
#
# -----------------------------------------------------
# HOW SIMULATED ANNEALING WORKS
# -----------------------------------------------------
#
# SA treats one MLP hyperparameter configuration as the
# current solution.
#
# Example:
#
# hidden layers : (64,32)
# activation    : relu
# learning rate : 0.001
# alpha         : 0.0001
# batch size    : 32
#
# A neighbouring solution is then created by changing
# one hyperparameter.
#
# Example:
#
# hidden layers : (64,32)
# activation    : tanh
# learning rate : 0.001
# alpha         : 0.0001
# batch size    : 32
#
# The new configuration is evaluated using Validation
# Macro F1.
#
# If the new solution is better, it is accepted.
#
# If the new solution is worse, it may still be accepted
# with a probability controlled by the temperature:
#
#     P(accept) = exp(delta / temperature)
#
# where:
#
#     delta = new_score - current_score
#
# The temperature gradually decreases during the search.
#
# Early stage:
#     high exploration
#     worse solutions may be accepted
#
# Later stage:
#     stronger exploitation
#     worse solutions are rarely accepted
#
# -----------------------------------------------------
# SA VS LEA
# -----------------------------------------------------
#
# LEA maintains a population of candidate solutions and
# improves the population through survivor selection and
# mutation.
#
# SA does not maintain a population.
# Instead, it follows one current solution and explores
# neighbouring solutions.
#
# Therefore, SA provides a different search mechanism from
# the population-based evolutionary strategy of LEA.
#
# -----------------------------------------------------
# SA VS TLBO
# -----------------------------------------------------
#
# TLBO maintains a population of learners and improves
# them through Teacher and Learner phases.
#
# SA does not require a teacher or learner population.
# Exploration is controlled through temperature and the
# probability of accepting a worse neighbouring solution.
#
# SA can therefore escape local optima using a mechanism
# that is different from the population interactions used
# by TLBO.
#
# SA is not assumed to be better than LEA or TLBO.
# The purpose is to provide another optimization strategy
# for empirical comparison.
# =====================================================


import math


# =====================================================
# SA PARAMETERS
# =====================================================
#
# Twenty evaluations are used so that the SA experiment
# is consistent with the 20-trial Bayesian Optimization
# setting already used in Phase 3.
#
# The final thesis comparison should still consider the
# complete computational budget, not only the number of
# evaluations.
# =====================================================

SA_EVALUATIONS = 20

SA_INITIAL_TEMPERATURE = 0.03

SA_COOLING_RATE = 0.90


# =====================================================
# SA RANDOM CHOICE HELPER
# =====================================================
#
# numpy.choice can produce an error when the search space
# contains tuples of different sizes such as:
#
#     (32,)
#     (64,32)
#     (128,64,32)
#
# Selecting an index first avoids that problem.
# =====================================================

def sa_random_choice(
    options,
    rng
):

    index = int(
        rng.integers(
            0,
            len(options)
        )
    )

    return options[index]


# =====================================================
# CREATE INITIAL SA CONFIGURATION
# =====================================================

def create_sa_solution(
    rng
):

    solution = {}

    for parameter, values in SEARCH_SPACE.items():

        # Adam remains fixed throughout the experiment.

        if parameter == "solver":

            solution[
                parameter
            ] = "adam"

        else:

            solution[
                parameter
            ] = sa_random_choice(
                values,
                rng
            )

    return solution


# =====================================================
# CREATE SA NEIGHBOUR
# =====================================================
#
# A neighbour is generated by changing exactly one
# hyperparameter.
#
# This creates a local movement through the discrete
# hyperparameter search space.
# =====================================================

def create_sa_neighbour(
    current_solution,
    rng
):

    neighbour = (
        current_solution.copy()
    )

    parameters = [

        parameter

        for parameter in SEARCH_SPACE.keys()

        if parameter != "solver"
    ]

    parameter = sa_random_choice(
        parameters,
        rng
    )

    current_value = (
        neighbour[
            parameter
        ]
    )

    alternatives = [

        value

        for value
        in SEARCH_SPACE[
            parameter
        ]

        if value != current_value
    ]

    if alternatives:

        neighbour[
            parameter
        ] = sa_random_choice(
            alternatives,
            rng
        )

    return neighbour


# =====================================================
# SA FITNESS FUNCTION
# =====================================================
#
# Validation Macro F1 is used because the Water Quality
# problem has moderate class imbalance and the potable
# class showed weaker recall during the earlier phases.
#
# The same metric is therefore used as the optimization
# objective for SA.
# =====================================================

def evaluate_sa_candidate(
    params
):

    model = build_mlp(
        params
    )

    score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="f1_macro",

        n_jobs=-1

    ).mean()

    return float(
        score
    )


# =====================================================
# INITIALIZE SA
# =====================================================

print("\n" + "=" * 80)

print(
    "SIMULATED ANNEALING "
    "HYPERPARAMETER OPTIMIZATION"
)

print("=" * 80)


sa_rng = np.random.default_rng(
    42
)


sa_start = time.time()


# =====================================================
# INITIAL CURRENT SOLUTION
# =====================================================

sa_current_solution = (
    create_sa_solution(
        sa_rng
    )
)


sa_current_score = (
    evaluate_sa_candidate(
        sa_current_solution
    )
)


# =====================================================
# INITIAL BEST SOLUTION
# =====================================================

sa_best_solution = (
    sa_current_solution.copy()
)


sa_best_score = (
    sa_current_score
)


# =====================================================
# SA CONVERGENCE HISTORY
# =====================================================

sa_convergence = []


sa_convergence.append(

    {

        "Evaluation":
            1,

        "Current Macro F1":
            sa_current_score,

        "Best Macro F1":
            sa_best_score,

        "Temperature":
            SA_INITIAL_TEMPERATURE
    }

)


print(
    f"Evaluation 01/"
    f"{SA_EVALUATIONS} "
    f"- Macro F1 = "
    f"{sa_current_score:.4f}"
)


# =====================================================
# START SA SEARCH
# =====================================================

for evaluation in range(

    2,

    SA_EVALUATIONS + 1

):

    # -------------------------------------------------
    # CALCULATE CURRENT TEMPERATURE
    # -------------------------------------------------

    temperature = (

        SA_INITIAL_TEMPERATURE

        *

        (
            SA_COOLING_RATE
            **
            (evaluation - 2)
        )

    )


    # Prevent numerical zero.

    temperature = max(

        temperature,

        1e-12

    )


    # -------------------------------------------------
    # CREATE NEIGHBOUR
    # -------------------------------------------------

    neighbour_solution = (
        create_sa_neighbour(

            sa_current_solution,

            sa_rng

        )
    )


    # -------------------------------------------------
    # EVALUATE NEIGHBOUR
    # -------------------------------------------------

    neighbour_score = (
        evaluate_sa_candidate(

            neighbour_solution

        )
    )


    # -------------------------------------------------
    # CALCULATE SCORE DIFFERENCE
    # -------------------------------------------------

    delta = (

        neighbour_score

        -

        sa_current_score

    )


    # -------------------------------------------------
    # ACCEPTANCE DECISION
    # -------------------------------------------------
    #
    # Better solutions are always accepted.
    #
    # Worse solutions may still be accepted according
    # to the simulated annealing acceptance probability.
    # -------------------------------------------------

    if delta >= 0:

        accept_solution = True

    else:

        acceptance_probability = math.exp(

            delta
            /
            temperature

        )

        accept_solution = (

            sa_rng.random()

            <

            acceptance_probability

        )


    # -------------------------------------------------
    # UPDATE CURRENT SOLUTION
    # -------------------------------------------------

    if accept_solution:

        sa_current_solution = (
            neighbour_solution.copy()
        )

        sa_current_score = (
            neighbour_score
        )


    # -------------------------------------------------
    # UPDATE GLOBAL BEST
    # -------------------------------------------------

    if neighbour_score > sa_best_score:

        sa_best_score = (
            neighbour_score
        )

        sa_best_solution = (
            neighbour_solution.copy()
        )


    # -------------------------------------------------
    # SAVE CONVERGENCE INFORMATION
    # -------------------------------------------------

    sa_convergence.append(

        {

            "Evaluation":
                evaluation,

            "Current Macro F1":
                sa_current_score,

            "Best Macro F1":
                sa_best_score,

            "Temperature":
                temperature
        }

    )


    # -------------------------------------------------
    # PRINT PROGRESS
    # -------------------------------------------------

    print(

        f"Evaluation "
        f"{evaluation:02d}/"
        f"{SA_EVALUATIONS} "
        f"- Current Macro F1 = "
        f"{sa_current_score:.4f}, "
        f"Best Macro F1 = "
        f"{sa_best_score:.4f}"

    )


# =====================================================
# SA OPTIMIZATION COMPLETED
# =====================================================

sa_optimization_time = (

    time.time()

    -

    sa_start

)


sa_evaluations = (
    SA_EVALUATIONS
)


print(
    "\nSimulated Annealing "
    "Optimization completed."
)


print(
    f"Optimization Time: "
    f"{sa_optimization_time:.2f} seconds"
)


print(
    "\nBest Hyperparameters "
    "Found By Simulated Annealing\n"
)


for key, value in (
    sa_best_solution.items()
):

    print(
        f"{key}: {value}"
    )


print(
    f"\nBest Validation Macro F1: "
    f"{sa_best_score:.4f}"
)


# =====================================================
# BUILD SA OPTIMIZED MLP
# =====================================================

print(
    "\nBuilding Simulated Annealing "
    "Optimized MLP Model..."
)


sa_model = build_mlp(
    sa_best_solution
)


# =====================================================
# TRAIN SA OPTIMIZED MLP
# =====================================================

print(
    "\nTraining Simulated Annealing "
    "Optimized MLP..."
)


sa_model, sa_training_runtime = (
    train_mlp_model(

        sa_model,

        X_train,

        y_train

    )
)


print(

    f"Training completed in "
    f"{sa_training_runtime:.2f} seconds"

)


# =====================================================
# SA TEST PREDICTIONS
# =====================================================

print(
    "\nMaking Predictions..."
)


sa_predictions = (
    sa_model.predict(
        X_test
    )
)


# =====================================================
# SA TEST METRICS
# =====================================================

sa_test_metrics = {

    "accuracy":
        accuracy_score(

            y_test,

            sa_predictions

        ),

    "macro_precision":
        precision_score(

            y_test,

            sa_predictions,

            average="macro",

            zero_division=0

        ),

    "macro_recall":
        recall_score(

            y_test,

            sa_predictions,

            average="macro",

            zero_division=0

        ),

    "macro_f1":
        f1_score(

            y_test,

            sa_predictions,

            average="macro",

            zero_division=0

        ),

    "weighted_f1":
        f1_score(

            y_test,

            sa_predictions,

            average="weighted",

            zero_division=0

        ),

    "balanced_accuracy":
        balanced_accuracy_score(

            y_test,

            sa_predictions

        ),

    "potable_recall":
        recall_score(

            y_test,

            sa_predictions,

            pos_label=1,

            zero_division=0

        )
}


# =====================================================
# SA TEST RESULTS
# =====================================================

print(
    "\n========== "
    "SIMULATED ANNEALING RESULTS "
    "=========="
)


print(

    f"Accuracy : "
    f"{sa_test_metrics['accuracy']:.4f}"

)


print(

    f"Macro Precision : "
    f"{sa_test_metrics['macro_precision']:.4f}"

)


print(

    f"Macro Recall : "
    f"{sa_test_metrics['macro_recall']:.4f}"

)


print(

    f"Macro F1 : "
    f"{sa_test_metrics['macro_f1']:.4f}"

)


print(

    f"Weighted F1 : "
    f"{sa_test_metrics['weighted_f1']:.4f}"

)


print(

    f"Balanced Accuracy : "
    f"{sa_test_metrics['balanced_accuracy']:.4f}"

)


print(

    f"Potable Recall : "
    f"{sa_test_metrics['potable_recall']:.4f}"

)


print(
    f"Optimization Time : "
    f"{sa_optimization_time:.2f} seconds"
)


print(
    f"Training Time : "
    f"{sa_training_runtime:.2f} seconds"
)


# =====================================================
# SA CLASSIFICATION REPORT
# =====================================================

print(
    "\nClassification Report\n"
)


print(

    classification_report(

        y_test,

        sa_predictions,

        zero_division=0

    )

)


# =====================================================
# SA CONFUSION MATRIX
# =====================================================

sa_confusion_matrix = (
    confusion_matrix(

        y_test,

        sa_predictions

    )
)


print(
    "\nConfusion Matrix\n"
)


print(
    sa_confusion_matrix
)


# =====================================================
# VISUALIZE SA TRAINING LOSS
# =====================================================

plot_training_loss(

    sa_model,

    "Simulated Annealing "
    "Optimized MLP Training Loss"

)


# =====================================================
# VISUALIZE SA CONVERGENCE
# =====================================================

sa_convergence_df = (
    pd.DataFrame(
        sa_convergence
    )
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    sa_convergence_df[
        "Evaluation"
    ],

    sa_convergence_df[
        "Best Macro F1"
    ],

    marker="o"

)


plt.xlabel(
    "Evaluation"
)


plt.ylabel(
    "Best Validation Macro F1"
)


plt.title(
    "Simulated Annealing Convergence"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# =====================================================
# VISUALIZE SA TEMPERATURE
# =====================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    sa_convergence_df[
        "Evaluation"
    ],

    sa_convergence_df[
        "Temperature"
    ],

    marker="o"

)


plt.xlabel(
    "Evaluation"
)


plt.ylabel(
    "Temperature"
)


plt.title(
    "Simulated Annealing Cooling Schedule"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# =====================================================
# VISUALIZE SA CONFUSION MATRIX
# =====================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    sa_confusion_matrix
)


plt.colorbar()


plt.xticks(

    [0, 1],

    [
        "Non-Potable",
        "Potable"
    ]

)


plt.yticks(

    [0, 1],

    [
        "Non-Potable",
        "Potable"
    ]

)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Simulated Annealing "
    "Confusion Matrix"
)


for i in range(

    sa_confusion_matrix.shape[0]

):

    for j in range(

        sa_confusion_matrix.shape[1]

    ):

        plt.text(

            j,

            i,

            sa_confusion_matrix[
                i,
                j
            ],

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.show()


# =====================================================
# SAVE EXPERIMENT RESULTS
# =====================================================
#
# Keep TLBO and add SA beside it.
# =====================================================

RESULTS_DIR = "results"


os.makedirs(

    RESULTS_DIR,

    exist_ok=True

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


    "Validation Macro F1": [

        baseline_validation_metrics[
            "macro_f1"
        ],

        random_search.best_score_,

        study.best_value,

        hyperband.best_score_,

        lea_best_score,

        tlbo_best_score,

        sa_best_score

    ],


    "Test Accuracy": [

        baseline_test_metrics[
            "accuracy"
        ],

        random_test_metrics[
            "accuracy"
        ],

        bayesian_test_metrics[
            "accuracy"
        ],

        hyperband_test_metrics[
            "accuracy"
        ],

        lea_test_metrics[
            "accuracy"
        ],

        tlbo_test_metrics[
            "accuracy"
        ],

        sa_test_metrics[
            "accuracy"
        ]

    ],


    "Test Macro Precision": [

        baseline_test_metrics[
            "macro_precision"
        ],

        random_test_metrics[
            "macro_precision"
        ],

        bayesian_test_metrics[
            "macro_precision"
        ],

        hyperband_test_metrics[
            "macro_precision"
        ],

        lea_test_metrics[
            "macro_precision"
        ],

        tlbo_test_metrics[
            "macro_precision"
        ],

        sa_test_metrics[
            "macro_precision"
        ]

    ],


    "Test Macro Recall": [

        baseline_test_metrics[
            "macro_recall"
        ],

        random_test_metrics[
            "macro_recall"
        ],

        bayesian_test_metrics[
            "macro_recall"
        ],

        hyperband_test_metrics[
            "macro_recall"
        ],

        lea_test_metrics[
            "macro_recall"
        ],

        tlbo_test_metrics[
            "macro_recall"
        ],

        sa_test_metrics[
            "macro_recall"
        ]

    ],


    "Test Macro F1": [

        baseline_test_metrics[
            "macro_f1"
        ],

        random_test_metrics[
            "macro_f1"
        ],

        bayesian_test_metrics[
            "macro_f1"
        ],

        hyperband_test_metrics[
            "macro_f1"
        ],

        lea_test_metrics[
            "macro_f1"
        ],

        tlbo_test_metrics[
            "macro_f1"
        ],

        sa_test_metrics[
            "macro_f1"
        ]

    ],


    "Test Weighted F1": [

        baseline_test_metrics[
            "weighted_f1"
        ],

        random_test_metrics[
            "weighted_f1"
        ],

        bayesian_test_metrics[
            "weighted_f1"
        ],

        hyperband_test_metrics[
            "weighted_f1"
        ],

        lea_test_metrics[
            "weighted_f1"
        ],

        tlbo_test_metrics[
            "weighted_f1"
        ],

        sa_test_metrics[
            "weighted_f1"
        ]

    ],


    "Test Balanced Accuracy": [

        baseline_test_metrics[
            "balanced_accuracy"
        ],

        random_test_metrics[
            "balanced_accuracy"
        ],

        bayesian_test_metrics[
            "balanced_accuracy"
        ],

        hyperband_test_metrics[
            "balanced_accuracy"
        ],

        lea_test_metrics[
            "balanced_accuracy"
        ],

        tlbo_test_metrics[
            "balanced_accuracy"
        ],

        sa_test_metrics[
            "balanced_accuracy"
        ]

    ],


    "Test Potable Recall": [

        baseline_test_metrics[
            "potable_recall"
        ],

        random_test_metrics[
            "potable_recall"
        ],

        bayesian_test_metrics[
            "potable_recall"
        ],

        hyperband_test_metrics[
            "potable_recall"
        ],

        lea_test_metrics[
            "potable_recall"
        ],

        tlbo_test_metrics[
            "potable_recall"
        ],

        sa_test_metrics[
            "potable_recall"
        ]

    ],


    "Optimization Time (s)": [

        0,

        random_search_runtime,

        bayesian_runtime,

        hyperband_runtime,

        lea_runtime,

        tlbo_runtime,

        sa_optimization_time

    ],


    "Training Time (s)": [

        baseline_training_runtime,

        random_training_runtime,

        bayesian_training_runtime,

        hyperband_training_runtime,

        lea_training_runtime,

        tlbo_training_runtime,

        sa_training_runtime

    ],


    "Evaluations": [

        1,

        N_RANDOM_TRIALS,

        20,

        int(

            np.sum(

                hyperband.n_candidates_

            )

        )

        if hasattr(

            hyperband,

            "n_candidates_"

        )

        else np.nan,

        lea_evaluations,

        tlbo_evaluations,

        sa_evaluations

    ],


    "Final Training Iterations": [

        baseline_model.n_iter_,

        random_model.n_iter_,

        bayesian_model.n_iter_,

        hyperband_model.n_iter_,

        lea_model.n_iter_,

        tlbo_model.n_iter_,

        sa_model.n_iter_

    ],


    "Final Training Loss": [

        baseline_model.loss_curve_[-1],

        random_model.loss_curve_[-1],

        bayesian_model.loss_curve_[-1],

        hyperband_model.loss_curve_[-1],

        lea_model.loss_curve_[-1],

        tlbo_model.loss_curve_[-1],

        sa_model.loss_curve_[-1]

    ],


    "Hidden Layers": [

        baseline_params[
            "hidden_layer_sizes"
        ],

        random_search.best_params_[
            "hidden_layer_sizes"
        ],

        best_params[
            "hidden_layer_sizes"
        ],

        hyperband.best_params_[
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

        random_search.best_params_[
            "activation"
        ],

        best_params[
            "activation"
        ],

        hyperband.best_params_[
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

        random_search.best_params_[
            "learning_rate_init"
        ],

        best_params[
            "learning_rate_init"
        ],

        hyperband.best_params_[
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

        random_search.best_params_[
            "alpha"
        ],

        best_params[
            "alpha"
        ],

        hyperband.best_params_[
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

        random_search.best_params_[
            "batch_size"
        ],

        best_params[
            "batch_size"
        ],

        hyperband.best_params_[
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


# =====================================================
# PRINT COMPARISON RESULTS
# =====================================================

print(
    "\nComparison Results"
)


print(
    comparison_results
)


# =====================================================
# SAVE COMPARISON RESULTS
# =====================================================

comparison_results.to_csv(

    f"{RESULTS_DIR}/"
    "water_quality_phase3_results.csv",

    index=False

)


print(
    "\nResults saved to:"
)


print(

    f"{RESULTS_DIR}/"
    "water_quality_phase3_results.csv"

)


# =====================================================
# SAVE CONVERGENCE HISTORY
# =====================================================
#
# Retain the existing Bayesian, LEA and TLBO convergence
# history and add SA convergence.
# =====================================================

convergence_frames = []


# -----------------------------------------------------
# BAYESIAN OPTIMIZATION
# -----------------------------------------------------

for row in bayesian_convergence:

    convergence_frames.append(

        {

            "Method":
                "Bayesian Optimization",

            "Iteration":
                row[
                    "Trial"
                ],

            "Best Validation Macro F1":
                row[
                    "Best Validation Macro F1"
                ]

        }

    )


# -----------------------------------------------------
# LEA
# -----------------------------------------------------

for row in lea_convergence:

    convergence_frames.append(

        {

            "Method":
                "LEA",

            "Iteration":
                row[
                    "Generation"
                ],

            "Best Validation Macro F1":
                row[
                    "Best Validation Macro F1"
                ]

        }

    )


# -----------------------------------------------------
# TLBO
# -----------------------------------------------------

for row in tlbo_convergence:

    convergence_frames.append(

        {

            "Method":
                "TLBO",

            "Iteration":
                row[
                    "Generation"
                ],

            "Best Validation Macro F1":
                row[
                    "Best Validation Macro F1"
                ]

        }

    )


# -----------------------------------------------------
# SIMULATED ANNEALING
# -----------------------------------------------------

for row in sa_convergence:

    convergence_frames.append(

        {

            "Method":
                "Simulated Annealing",

            "Iteration":
                row[
                    "Evaluation"
                ],

            "Best Validation Macro F1":
                row[
                    "Best Macro F1"
                ]

        }

    )


# =====================================================
# SAVE CONVERGENCE CSV
# =====================================================

if convergence_frames:

    convergence_df = pd.DataFrame(
        convergence_frames
    )


    convergence_df.to_csv(

        f"{RESULTS_DIR}/"
        "water_quality_phase3_convergence.csv",

        index=False

    )


    print(
        "\nConvergence history saved to:"
    )


    print(

        f"{RESULTS_DIR}/"
        "water_quality_phase3_convergence.csv"

    )


# =====================================================
# SAVE SA-SPECIFIC CONVERGENCE
# =====================================================

sa_convergence_file = (

    f"{RESULTS_DIR}/"
    "water_quality_phase3_sa_convergence.csv"

)


sa_convergence_df.to_csv(

    sa_convergence_file,

    index=False

)


print(
    "\nSA convergence history saved to:"
)


print(
    sa_convergence_file
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print(
    "\n" + "=" * 80
)


print(
    "WATER QUALITY PHASE 3 "
    "OPTIMIZATION COMPLETED"
)


print(
    "=" * 80
)


# -----------------------------------------------------
# FIND BEST VALIDATION METHOD
# -----------------------------------------------------

best_method = comparison_results.loc[

    comparison_results[
        "Validation Macro F1"
    ].idxmax(),

    "Method"

]


best_validation_score = (

    comparison_results[
        "Validation Macro F1"
    ].max()

)


print(

    f"Best Validation Method : "
    f"{best_method}"

)


print(

    f"Best Validation Macro F1 : "
    f"{best_validation_score:.4f}"

)


# =====================================================
# OPTIMIZATION METHODS
# =====================================================

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


# =====================================================
# PRIMARY OPTIMIZATION METRIC
# =====================================================

print(
    "\nPrimary Optimization Metric:"
)


print(
    "Validation Macro F1"
)


# =====================================================
# FINAL EVALUATION
# =====================================================

print(
    "\nFinal Evaluation:"
)


print(
    "Untouched Test Set"
)


# =====================================================
# RESULTS FILE
# =====================================================

print(
    "\nResults saved to:"
)


print(

    f"{RESULTS_DIR}/"
    "water_quality_phase3_results.csv"

)


# =====================================================
# CONVERGENCE FILE
# =====================================================

print(
    "\nConvergence history saved to:"
)


print(

    f"{RESULTS_DIR}/"
    "water_quality_phase3_convergence.csv"

)


print(
    "\nSA convergence history saved to:"
)


print(
    sa_convergence_file
)


print(
    "=" * 80
)