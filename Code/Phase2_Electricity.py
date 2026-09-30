# =====================================================
# ELECTRICITY DATASET
# =====================================================
#
# Dataset Name:
# Electricity
#
# Source:
# UCI Machine Learning Repository
# Dataset ID: 151
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
# The Electricity dataset contains observations collected
# from the Australian New South Wales electricity market.
# Each observation represents the operating conditions of
# the electricity market at a particular point in time,
# including information related to electricity demand,
# scheduled supply, market transfers and electricity
# pricing.
#
# In the original dataset, the target variable "class"
# indicates whether the electricity price increased (UP)
# or decreased (DOWN), making it a binary classification
# problem.
#
# During Phase 1, the original target variable was
# renamed from "class" to "Target" to maintain a
# consistent target variable naming convention across all
# datasets used in this study. No modifications were made
# to the class labels.
#
# =====================================================
# INPUT FEATURES
# =====================================================
#
# The dataset contains electricity market-related
# measurements, including:
#
# • Day of the week
# • Time period
# • Electricity demand
# • Scheduled electricity supply
# • Market transfer information
#
# The predictor variables consist of both numerical and
# categorical market attributes describing the operating
# conditions of the electricity market.
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
# The Target variable contains two electricity price
# movement categories:
#
# • UP
# • DOWN
#
# These classes indicate whether the electricity price
# increased or decreased during the corresponding market
# period.
#
# =====================================================
# WHY THIS DATASET WAS SELECTED
# =====================================================
#
# Compared with the Energy Efficiency and Appliance
# Energy Prediction datasets, the Electricity dataset
# represents a real-world electricity market
# classification problem containing a substantially
# larger number of observations.
#
# The dataset provides an opportunity to evaluate the
# performance of different MLP hyperparameter
# optimization algorithms on binary classification using
# electricity market data collected under realistic
# operating conditions.
# =====================================================
import random
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
import time
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
# VISUALIZATION OUTPUT DIRECTORY
# =====================================================

VISUALIZATION_DIR = "results/phase2_electricity/visualizations"

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)
dataset = pd.read_csv(
    "processed_datasets/electricity_processed.csv"
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
# variable (y). The Target column represents the binary
# class labels to be predicted by the MLP classifier,
# while all remaining variables are used as predictor
# variables.
# =====================================================

X = dataset.drop(
    columns=["Target"]
)

y = dataset["Target"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# =====================================================
# ONE-HOT ENCODING
# =====================================================
#
# One-Hot Encoding converts categorical predictor
# variables into numerical representations that can be
# processed by machine learning algorithms.
#
# The Electricity dataset contains only numerical
# predictor variables. Therefore, no categorical feature
# encoding is required and this preprocessing step is
# omitted.
# =====================================================

# =====================================================
# TRAIN-TEST SPLIT
# =====================================================
#
# Split the dataset into training and testing subsets.
# Stratified sampling preserves the original class
# distribution in both subsets.
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
# Feature standardization scales the predictor variables
# to have zero mean and unit variance, ensuring that all
# input features contribute equally during neural network
# training.
#
# The Electricity dataset contains numerical predictor
# variables measured on different scales. Therefore,
# feature standardization is performed before training
# the MLP classifier.
#
# The StandardScaler is fitted only on the training data,
# and the same transformation is applied to the testing
# data to prevent data leakage.
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

# =====================================================
# VISUALIZE AND SAVE TRAINING LOSS
# =====================================================

def plot_training_loss(model, title, filename):

    plt.figure(figsize=(8, 5))

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

    plt.tight_layout()

    output_path = os.path.join(
        VISUALIZATION_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Visualization saved to:\n{output_path}"
    )
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
# attributed to hyperparameter optimization rather than
# differences in the experimental setup.
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
# are widely adopted for training feedforward neural
# networks.
#
# -----------------------------------------------------
# CHALLENGES AND LIMITATIONS
# -----------------------------------------------------
#
# Since the hyperparameters are manually selected, they
# may not represent the optimal configuration for the
# Electricity dataset.
#
# The baseline model does not explore alternative
# network architectures or learning parameters and
# therefore cannot automatically adapt to improve
# predictive performance.
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
    "Baseline MLP Training Loss",
    "baseline_mlp_training_loss.png"
)


print("\nBaseline Hyperparameters")

for key, value in baseline_params.items():

    print(f"{key}: {value}")
 
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
# The Electricity dataset contains a relatively large
# number of observations, making exhaustive
# hyperparameter optimization computationally
# expensive.
#
# Random Search is therefore used to efficiently explore
# different MLP hyperparameter combinations while
# limiting the total number of model evaluations.
#
# Compared with the manually configured baseline model,
# Random Search explores multiple candidate
# hyperparameter combinations, increasing the likelihood
# of identifying a better-performing MLP
# configuration.
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
#   completely at random.
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
# provides a balance between search quality and
# computational cost while maintaining consistency with
# the number of evaluations used by Bayesian
# Optimization, LEA and TLBO.
#
# scoring = "f1_weighted"
# Weighted F1-score is used as the optimization metric
# because the Electricity dataset contains an unequal
# distribution of the UP and DOWN classes. Unlike
# accuracy, the weighted F1-score considers both
# precision and recall while accounting for class
# imbalance.
#
# cv = 5
# Five-fold cross-validation provides a reliable estimate
# of model generalization by evaluating each candidate
# hyperparameter configuration across five different
# training-validation splits. A value of five is widely
# adopted in machine learning studies as it provides a
# good balance between evaluation stability and
# computational efficiency.
#
# random_state = 42
# A fixed random seed ensures that the randomly sampled
# hyperparameter combinations are reproducible, allowing
# the experimental results to be replicated.
#
# n_jobs = -1
# All available processor cores are utilized to evaluate
# multiple cross-validation folds in parallel, reducing
# the overall optimization time.
#
# max_iter = 500
# The maximum number of training iterations for each MLP
# model is fixed at 500. This value provides sufficient
# opportunity for convergence while ensuring consistency
# across the baseline model and all optimization
# algorithms evaluated in this study.
# -----------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV

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

print("\nBuilding Random Search Optimized MLP Model...")

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

print("\nTraining Random Search Optimized MLP...")

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
    "Random Search Optimized MLP Training Loss",
    "random_search_mlp_training_loss.png"
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
# BAYESIAN OPTIMIZATION USING OPTUNA
# =====================================================
#
# Optuna is a Bayesian hyperparameter optimization
# framework that sequentially identifies promising
# hyperparameter combinations based on the performance of
# previously evaluated trials.
#
# Unlike Random Search, which samples hyperparameters
# independently, Optuna constructs a probabilistic model
# of the objective function and selects new
# hyperparameter configurations that are expected to
# improve the optimization objective.
#
# -----------------------------------------------------
# WHY OPTUNA IS USED
# -----------------------------------------------------
#
# The Electricity dataset contains a relatively large
# number of observations together with a discrete
# hyperparameter search space for the MLP classifier.
# Optuna is used because it efficiently handles
# categorical hyperparameters while requiring fewer model
# evaluations than exhaustive or purely random search
# methods.
#
# Compared with Random Search, Optuna uses information
# obtained from previous trials to guide the search
# towards more promising hyperparameter configurations.
#
# Compared with LEA and TLBO, Optuna does not rely on
# population-based evolutionary operators such as
# mutation or teacher-learner interactions. Instead, it
# models the search process probabilistically to identify
# promising regions of the hyperparameter search space.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Efficient Bayesian hyperparameter optimization.
#
# • Naturally supports categorical hyperparameters.
#
# • Requires fewer evaluations than Random Search.
#
# • Well suited for computationally expensive machine
#   learning models such as Multi-Layer Perceptrons.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Each optimization step incurs additional modelling
#   overhead.
#
# • Performance depends on the quality of the surrogate
#   model.
#
# • Optimization may still converge to local optima.
# =====================================================
# -----------------------------------------------------
# OPTUNA PARAMETER CONFIGURATION
# -----------------------------------------------------
#
# n_trials = 20
# Twenty optimization trials are performed to maintain
# the same evaluation budget used by Random Search,
# Hyperband, LEA and TLBO. Using an identical number of
# evaluations enables a fair comparison between all
# optimization algorithms.
#
# scoring = "f1_weighted"
# Weighted F1-score is used as the optimization metric
# because the Electricity dataset contains an unequal
# distribution of the UP and DOWN classes. This metric
# considers both precision and recall while accounting
# for class imbalance.
#
# cv = 5
# Five-fold cross-validation provides a reliable estimate
# of model generalization while maintaining a reasonable
# computational cost. This value is widely adopted in
# machine learning studies and is used consistently
# throughout this research.
#
# solver = "adam"
# The Adam optimization algorithm is fixed for all
# experiments because it provides efficient gradient-
# based optimization for neural networks and ensures
# consistency across all optimization algorithms.
#
# max_iter = 500
# Each candidate MLP model is trained for a maximum of
# 500 iterations. Using the same maximum number of
# training iterations across all optimization algorithms
# ensures a fair comparison of optimization performance.
#
# random_state = 42
# A fixed random seed is used during model construction
# to ensure that the optimization process is
# reproducible.
# -----------------------------------------------------
import optuna
from sklearn.model_selection import cross_val_score
# =====================================================
# OPTUNA OBJECTIVE FUNCTION
# =====================================================
#
# The objective function defines one optimization trial.
# During each trial, Optuna selects one combination of
# MLP hyperparameters from the predefined search space.
#
# Each candidate model is evaluated using the weighted
# F1-score obtained from 5-fold cross-validation.
#
# The weighted F1-score is returned to Optuna, which uses
# the result to guide subsequent trials towards more
# promising hyperparameter configurations.
# =====================================================

def objective(trial):

    params = {

        "hidden_layer_sizes": trial.suggest_categorical(

            "hidden_layer_sizes",

            [
                "16",
                "32",
                "64",
                "32_16",
                "64_32"
            ]

        ),

        "activation": trial.suggest_categorical(

            "activation",

            [
                "relu",
                "tanh"
            ]

        ),

        "solver": "adam",

        "learning_rate_init": trial.suggest_categorical(

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

    # -------------------------------------------------
    # CONVERT HIDDEN LAYER ARCHITECTURE
    # -------------------------------------------------

    architecture_map = {

        "16": (16,),
        "32": (32,),
        "64": (64,),
        "32_16": (32, 16),
        "64_32": (64, 32)

    }

    params["hidden_layer_sizes"] = architecture_map[
        params["hidden_layer_sizes"]
    ]

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
# The study evaluates 20 different hyperparameter
# configurations. Each trial represents one candidate MLP
# configuration evaluated using 5-fold cross-validation.
#
# A fixed random seed is used to ensure reproducibility
# of the optimization process.
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
# hyperparameter configurations. The results of previous
# trials are used by the TPE sampler to guide subsequent
# trials towards more promising regions of the search
# space.
#
# Twenty trials are used to maintain consistency with the
# evaluation budget of the other optimization algorithms
# in this study.
# =====================================================

print("\nStarting Bayesian Optimization...")

start = time.time()

study.optimize(

    objective,

    n_trials=5

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
# optimization.
#
# The best configuration is stored in Optuna's
# best_params attribute and will be used to construct the
# final Bayesian-optimized MLP model.
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

    print(f"{key}: {value}")


# =====================================================
# BUILD BAYESIAN OPTIMIZED MLP MODEL
# =====================================================
#
# Construct the MLP classifier using the best
# hyperparameter configuration identified by Optuna.
#
# The build_mlp() function is reused to ensure that the
# optimized model is constructed using the same model
# configuration and maximum training iterations as the
# baseline and other optimization methods.
# =====================================================

print("\nBuilding Bayesian Optimized MLP Model...")

bayesian_model = build_mlp(

    best_params

)
# =====================================================
# TRAIN BAYESIAN OPTIMIZED MLP MODEL
# =====================================================
#
# Train the Bayesian-optimized MLP classifier using the
# complete training dataset.
#
# The optimized model is trained using the same training
# procedure as the baseline and other optimization
# methods to ensure a fair comparison.
#
# The training runtime is recorded separately from the
# Bayesian optimization runtime. This allows the
# computational cost of finding the hyperparameters to be
# distinguished from the cost of training the final
# optimized model.
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
# MODEL EVALUATION
# =====================================================
#
# Evaluate the trained Bayesian-optimized MLP classifier
# on the independent testing dataset.
#
# The same evaluation metrics used for the Baseline MLP
# and Random Search are used here to ensure a consistent
# comparison across all optimization methods.
#
# The evaluation metrics include:
#
# • Accuracy
# • Precision
# • Recall
# • Weighted F1-score
#
# The testing dataset is not used during hyperparameter
# optimization or model training. It therefore provides
# an independent estimate of the generalization
# performance of the optimized model.
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
#
# Plot the training loss recorded during training of the
# Bayesian-optimized MLP.
#
# The loss curve provides information about the learning
# and convergence behaviour of the final optimized model.
# =====================================================

plot_training_loss(
    bayesian_model,
    "Bayesian Optimized MLP Training Loss",
    "bayesian_optimization_mlp_training_loss.png"
)
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================
#
# Display the best hyperparameter configuration, best
# cross-validation F1-score, Bayesian optimization time
# and final model training time.
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
# optimization algorithm that efficiently searches for
# high-performing hyperparameter combinations by
# progressively allocating more computational resources
# to promising candidate solutions while eliminating
# poorly performing candidates at an early stage.
#
# Unlike traditional search methods that allocate equal
# computational resources to every candidate, Hyperband
# evaluates many candidate solutions initially using
# limited resources and gradually focuses the search on
# the best-performing hyperparameter configurations.
#
# -----------------------------------------------------
# WHY HYPERBAND IS USED
# -----------------------------------------------------
#
# The Electricity dataset contains a relatively large
# number of observations and a defined hyperparameter
# search space for the MLP classifier. Training multiple
# candidate neural network configurations can therefore
# require considerable computational resources.
#
# Hyperband is used to improve optimization efficiency by
# allocating additional training resources to promising
# configurations while terminating weaker configurations
# at earlier stages.
#
# Unlike Random Search, which evaluates each sampled
# configuration using the same computational budget,
# Hyperband progressively allocates more resources to
# promising configurations and eliminates poorly
# performing configurations early.
#
# Compared with Bayesian Optimization, Hyperband does
# not construct a probabilistic surrogate model to guide
# the search. Instead, it improves optimization efficiency
# through adaptive allocation of computational resources.
#
# Compared with LEA and TLBO, which improve candidate
# solutions using population-based search mechanisms,
# Hyperband focuses on efficiently allocating training
# resources among candidate configurations.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Efficiently allocates computational resources by
#   terminating poorly performing configurations early.
#
# • Evaluates multiple hyperparameter configurations
#   within a controlled computational budget.
#
# • Reduces unnecessary training compared with giving
#   every configuration the maximum training budget.
#
# • Well suited for computationally expensive models
#   such as neural networks.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Performance depends on the initial configurations
#   evaluated during the search.
#
# • A promising configuration may be eliminated if its
#   performance is relatively poor at an early stage.
#
# • Hyperband does not use a probabilistic surrogate model
#   to learn from previous evaluations.
# =====================================================

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
# =====================================================
# HYPERBAND CONFIGURATION
# =====================================================
#
# factor = 3
#
# The reduction factor determines how aggressively
# Hyperband eliminates poorly performing configurations
# between successive resource levels.
#
# A value of 3 provides a balance between maintaining
# candidate diversity and allocating additional resources
# to promising configurations.
#
# resource = "max_iter"
#
# The maximum number of MLP training iterations is used
# as the computational resource. Configurations that
# perform well at lower resource levels can receive
# additional training iterations.
#
# max_resources = 500
#
# The maximum computational budget is set to 500 training
# iterations, matching the maximum number of iterations
# used by the baseline MLP and the other optimization
# experiments.
#
# Using the same maximum training budget helps maintain
# consistency across the optimization methods.
#
# scoring = "f1_weighted"
#
# Weighted F1-score is used as the optimization metric,
# consistent with the other hyperparameter optimization
# methods in this study.
#
# cv = 5
#
# Five-fold cross-validation is used to obtain a robust
# estimate of model performance during optimization.
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

    scoring="f1_weighted",

    cv=5,

    random_state=42,

    n_jobs=-1

)
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
# BUILD HYPERBAND OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding Hyperband Optimized MLP Model...")

hyperband_model = build_mlp(

    hyperband.best_params_

)
# =====================================================
# TRAIN HYPERBAND OPTIMIZED MLP
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
# MODEL EVALUATION
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
    "Hyperband Optimized MLP Training Loss",
    "hyperband_mlp_training_loss.png"
)
# =====================================================
# PRINT OPTIMIZATION SUMMARY
# =====================================================

print_optimization_summary(

    method_name="Hyperband",

    best_params=hyperband.best_params_,

    best_cv_score=hyperband.best_score_,

    optimization_time=hyperband_runtime,

    training_time=hyperband_training_runtime

)
# =====================================================
# LEARNING EVOLUTIONARY ALGORITHM (LEA)
# =====================================================
#
# LEA is a population-based evolutionary optimization
# algorithm that searches for high-performing
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
# The Electricity dataset contains a relatively large
# number of observations, making the selection of suitable
# MLP hyperparameters important for achieving good
# classification performance.
#
# LEA is used to investigate whether a population-based
# evolutionary search strategy can identify effective MLP
# hyperparameter configurations compared with the
# conventional search and model-based optimization
# methods.
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
# MLP hyperparameters selected from the common SEARCH_SPACE.
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
# Weighted F1-score is used as the fitness measure because
# it accounts for the unequal distribution of the DOWN and
# UP classes in the Electricity dataset.
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
# This provides multiple opportunities for the candidate
# solutions to improve while keeping the optimization
# budget comparable with the other optimization methods.
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
    # Surviving candidates are retained as parents.
    # New candidates are generated by randomly changing
    # one hyperparameter of a selected parent.
    # -------------------------------------------------

    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(
            survivors
        ).copy()

        parameter = random.choice(
            list(parent.keys())
        )

        # The solver remains fixed as Adam throughout
        # the experiment.

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

print("\nLEA Optimization completed.")

print(
    f"Optimization Time: "
    f"{lea_runtime:.2f} seconds"
)

print("\nBest Hyperparameters Found By LEA\n")

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
#
# Construct the final MLP classifier using the best
# hyperparameter configuration identified during the
# LEA optimization process.
#
# The same build_mlp() function used by the baseline
# and other optimization methods is used to maintain
# consistency across all experiments.
# =====================================================

print("\nBuilding LEA Optimized MLP Model...")

lea_model = build_mlp(
    best_solution
)
# =====================================================
# TRAIN LEA OPTIMIZED MLP MODEL
# =====================================================
#
# Train the final LEA-optimized MLP using the complete
# training dataset.
#
# The training time is measured separately from the LEA
# optimization time. This allows the computational cost
# of hyperparameter optimization to be distinguished
# from the cost of training the final model.
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
# MODEL EVALUATION
# =====================================================
#
# Evaluate the final LEA-optimized MLP on the independent
# testing dataset.
#
# The same evaluation metrics used for the Baseline,
# Random Search, Bayesian Optimization, and Hyperband
# models are used here:
#
# • Accuracy
# • Weighted Precision
# • Weighted Recall
# • Weighted F1-score
#
# The testing dataset remains independent from the LEA
# hyperparameter optimization process.
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
# MODEL EVALUATION
# =====================================================
#
# Evaluate the final LEA-optimized MLP on the independent
# testing dataset.
#
# The same evaluation metrics used for the Baseline,
# Random Search, Bayesian Optimization, and Hyperband
# models are used here:
#
# • Accuracy
# • Weighted Precision
# • Weighted Recall
# • Weighted F1-score
#
# The testing dataset remains independent from the LEA
# hyperparameter optimization process.
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
# LEA-optimized MLP to visualize its training and
# convergence behaviour.
# =====================================================

plot_training_loss(
    lea_model,
    "LEA Optimized MLP Training Loss",
    "lea_mlp_training_loss.png"
)
# =====================================================
# PRINT LEA OPTIMIZATION SUMMARY
# =====================================================
#
# Display the best hyperparameter configuration,
# cross-validation F1-score, LEA optimization time,
# and final model training time.
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
# TLBO is a population-based optimization algorithm that
# improves candidate solutions through two learning phases:
#
# • Teacher Phase
# • Learner Phase
#
# Each candidate solution represents one possible
# combination of MLP hyperparameters.
#
# During the Teacher Phase, candidate solutions are updated
# by learning from the best-performing solution in the
# current population, which acts as the teacher.
#
# During the Learner Phase, candidate solutions learn by
# interacting with another randomly selected candidate
# solution.
#
# Each candidate configuration is evaluated using weighted
# F1-score obtained from 5-fold cross-validation.
#
# -----------------------------------------------------
# WHY TLBO IS USED
# -----------------------------------------------------
#
# The Electricity dataset contains a relatively large
# number of observations and a multi-dimensional MLP
# hyperparameter search space.
#
# TLBO is used to investigate whether a population-based
# teaching and learning strategy can identify effective
# MLP hyperparameter configurations.
#
# Unlike Random Search, which evaluates independently
# sampled configurations, TLBO iteratively improves the
# population using information from better-performing
# candidate solutions.
#
# Compared with Bayesian Optimization, TLBO does not
# construct a probabilistic surrogate model. Instead,
# candidate solutions are improved through teacher and
# learner interactions.
#
# Compared with Hyperband, which focuses on efficient
# allocation of computational resources, TLBO focuses on
# improving candidate hyperparameter configurations through
# population-based learning.
#
# Compared with LEA, which uses selection and mutation,
# TLBO improves candidate solutions through teacher and
# learner phases.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Uses information from the best-performing solution.
#
# • Does not require a separate mutation probability.
#
# • Provides both exploitation through the teacher phase
#   and exploration through the learner phase.
#
# • Suitable for population-based hyperparameter
#   optimization.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Requires repeated evaluation of candidate solutions.
#
# • Computational cost increases with population size and
#   number of iterations.
#
# • Performance depends on the initial population.
#
# • Random learner interactions may generate inferior
#   candidate solutions.
# =====================================================
# =====================================================
# TLBO PARAMETER CONFIGURATION
# =====================================================
#
# POPULATION_SIZE = 20
#
# The population size determines the number of candidate
# MLP hyperparameter configurations evaluated during each
# TLBO iteration. A population of 20 provides sufficient
# diversity while keeping the computational cost of
# repeated 5-fold cross-validation manageable.
#
# MAX_ITERATIONS = 20
#
# The TLBO optimization is performed for 20 iterations.
# This provides multiple opportunities for candidate
# hyperparameter configurations to improve while keeping
# the optimization budget comparable with the other
# optimization algorithms.
# =====================================================

TLBO_POPULATION_SIZE = 20

TLBO_MAX_ITERATIONS = 5
# =====================================================
# TLBO POPULATION INITIALIZATION
# =====================================================
#
# TLBO starts with a population of candidate solutions.
# Each candidate represents one possible combination of
# MLP hyperparameters selected from the common SEARCH_SPACE.
#
# A fixed random seed is used to ensure that population
# initialization and subsequent random learner selections
# are reproducible across experimental runs.
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
# Each candidate hyperparameter configuration is
# evaluated using 5-fold cross-validation.
#
# Weighted F1-score is used as the fitness measure,
# maintaining consistency with the Baseline, Random
# Search, Bayesian Optimization, Hyperband, and LEA
# experiments.
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
#
# The best solution discovered during the TLBO search
# is stored globally so that improvements obtained during
# either the Teacher Phase or Learner Phase are retained.
# =====================================================

tlbo_best_score = float("-inf")

tlbo_best_solution = None
# =====================================================
# START TLBO OPTIMIZATION
# =====================================================
#
# TLBO improves the population through two consecutive
# learning phases:
#
# 1. Teacher Phase
# 2. Learner Phase
#
# In the Teacher Phase, the best-performing candidate in
# the current population is selected as the teacher.
#
# The remaining candidates attempt to move towards the
# teacher's hyperparameter configuration.
#
# Candidate solutions are evaluated using weighted
# F1-score obtained from 5-fold cross-validation.
# =====================================================

print("\nStarting TLBO Optimization...")

start = time.time()

for iteration in range(TLBO_MAX_ITERATIONS):

    print(
        f"\nIteration "
        f"{iteration + 1}/{TLBO_MAX_ITERATIONS}"
    )

    # =================================================
    # EVALUATE CURRENT POPULATION
    # =================================================
    #
    # Evaluate every candidate in the current population
    # and determine its weighted F1-score.
    # =================================================

    tlbo_fitness = []

    for candidate in tlbo_population:

        score = evaluate_tlbo_candidate(
            candidate
        )

        tlbo_fitness.append(score)

        # Keep track of the best solution found across
        # all TLBO iterations.

        if score > tlbo_best_score:

            tlbo_best_score = score

            tlbo_best_solution = candidate.copy()

    # =================================================
    # SELECT TEACHER
    # =================================================
    #
    # The candidate with the highest fitness in the
    # current population is selected as the teacher.
    #
    # The teacher represents the best hyperparameter
    # configuration currently available to the population.
    # =================================================

    teacher_index = tlbo_fitness.index(
        max(tlbo_fitness)
    )

    teacher = tlbo_population[
        teacher_index
    ].copy()

    teacher_score = tlbo_fitness[
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
    # Each candidate learns from the current teacher.
    #
    # For each candidate, hyperparameters are considered
    # individually. With a probability of 0.5, the
    # candidate adopts the corresponding hyperparameter
    # value from the teacher.
    #
    # Otherwise, the candidate retains its current value.
    #
    # The solver remains fixed as Adam throughout the
    # experiment.
    # =================================================

    teacher_population = []

    for candidate in tlbo_population:

        new_candidate = candidate.copy()

        for parameter in SEARCH_SPACE:

            if parameter == "solver":

                continue

            if random.random() < 0.5:

                new_candidate[parameter] = (
                    teacher[parameter]
                )

        teacher_population.append(
            new_candidate
        )
    # =================================================
    # EVALUATE TEACHER PHASE
    # =================================================
    #
    # Evaluate the newly generated candidates.
    #
    # If a Teacher Phase candidate performs better than
    # the corresponding current candidate, it replaces
    # the current candidate in the population.
    # =================================================

    teacher_fitness = []

    for candidate in teacher_population:

        score = evaluate_tlbo_candidate(
            candidate
        )

        teacher_fitness.append(score)

    # =================================================
    # ACCEPT IMPROVED TEACHER-PHASE SOLUTIONS
    # =================================================

    for index in range(
        TLBO_POPULATION_SIZE
    ):

        if (
            teacher_fitness[index]
            >
            tlbo_fitness[index]
        ):

            tlbo_population[index] = (
                teacher_population[index]
            )

            tlbo_fitness[index] = (
                teacher_fitness[index]
            )

            # Update global best solution.

            if (
                teacher_fitness[index]
                >
                tlbo_best_score
            ):

                tlbo_best_score = (
                    teacher_fitness[index]
                )

                tlbo_best_solution = (
                    teacher_population[index].copy()
                )
    # =================================================
    # LEARNER PHASE
    # =================================================
    #
    # During the Learner Phase, each candidate interacts
    # with another randomly selected candidate from the
    # current population.
    #
    # If the randomly selected learner has a better fitness
    # score, the current candidate attempts to move towards
    # the better candidate.
    #
    # If the current candidate has a better fitness score,
    # the randomly selected candidate is used as the weaker
    # learner and the current candidate is retained as the
    # reference solution.
    #
    # This phase introduces additional exploration and
    # allows candidates to learn from one another rather
    # than relying only on the best teacher.
    # =================================================

    learner_population = []

    learner_fitness = []

    for index, candidate in enumerate(
        tlbo_population
    ):

        # -------------------------------------------------
        # SELECT RANDOM LEARNER
        # -------------------------------------------------
        #
        # Select another candidate from the population.
        # The selected candidate must be different from
        # the current candidate.
        # -------------------------------------------------

        learner_index = random.choice(
            [
                i
                for i in range(TLBO_POPULATION_SIZE)
                if i != index
            ]
        )

        other_candidate = tlbo_population[
            learner_index
        ]

        current_score = tlbo_fitness[
            index
        ]

        other_score = tlbo_fitness[
            learner_index
        ]

        # -------------------------------------------------
        # DETERMINE BETTER LEARNER
        # -------------------------------------------------
        #
        # The candidate with the higher fitness score is
        # considered the better learner.
        # -------------------------------------------------

        if other_score > current_score:

            better_candidate = other_candidate

            weaker_candidate = candidate

        else:

            better_candidate = candidate

            weaker_candidate = other_candidate

        # -------------------------------------------------
        # GENERATE LEARNER-PHASE CANDIDATE
        # -------------------------------------------------

        new_candidate = weaker_candidate.copy()

        for parameter in SEARCH_SPACE:

            # Keep Adam fixed throughout the experiment.

            if parameter == "solver":

                continue

            # With probability 0.5, learn the corresponding
            # hyperparameter from the better-performing
            # candidate.

            if random.random() < 0.5:

                new_candidate[parameter] = (
                    better_candidate[parameter]
                )

        learner_population.append(
            new_candidate
        )

    # =================================================
    # EVALUATE LEARNER PHASE
    # =================================================
    #
    # Evaluate the newly generated learner candidates
    # using 5-fold cross-validation and weighted F1-score.
    # =================================================

    for candidate in learner_population:

        score = evaluate_tlbo_candidate(
            candidate
        )

        learner_fitness.append(
            score
        )

    # =================================================
    # ACCEPT IMPROVED LEARNER SOLUTIONS
    # =================================================
    #
    # A learner-phase candidate replaces the corresponding
    # population candidate only when it achieves a higher
    # weighted F1-score.
    #
    # The global best solution is updated whenever a new
    # candidate achieves the highest score observed during
    # the TLBO optimization.
    # =================================================

    for index in range(
        TLBO_POPULATION_SIZE
    ):

        if (
            learner_fitness[index]
            >
            tlbo_fitness[index]
        ):

            tlbo_population[index] = (
                learner_population[index]
            )

            tlbo_fitness[index] = (
                learner_fitness[index]
            )

            if (
                learner_fitness[index]
                >
                tlbo_best_score
            ):

                tlbo_best_score = (
                    learner_fitness[index]
                )

                tlbo_best_solution = (
                    learner_population[index].copy()
                )

    print(
        f"Best F1 Score: "
        f"{tlbo_best_score:.4f}"
    )
# =====================================================
# TLBO OPTIMIZATION COMPLETED
# =====================================================
#
# After all TLBO iterations have been completed, record
# the total optimization time and display the best
# hyperparameter configuration found during the complete
# optimization process.
# =====================================================

end = time.time()

tlbo_runtime = end - start

print("\nTLBO Optimization completed.")

print(
    f"Optimization Time: "
    f"{tlbo_runtime:.2f} seconds"
)

print("\nBest Hyperparameters Found By TLBO\n")

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
#
# Construct the final MLP classifier using the best
# hyperparameter configuration identified by TLBO.
# =====================================================

print("\nBuilding TLBO Optimized MLP Model...")

tlbo_model = build_mlp(
    tlbo_best_solution
)
# =====================================================
# TRAIN TLBO OPTIMIZED MLP MODEL
# =====================================================
#
# Train the final TLBO-optimized MLP using the complete
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
# MODEL EVALUATION
# =====================================================
#
# Evaluate the final TLBO-optimized MLP on the independent
# testing dataset using the same metrics as all previous
# optimization methods.
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
    "TLBO Optimized MLP Training Loss",
    "tlbo_mlp_training_loss.png"
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
# SAVE EXPERIMENT RESULTS
# =====================================================
#
# Store the performance metrics and execution times of
# all optimization methods.
#
# The results are saved as a CSV file for subsequent
# comparative analysis across optimization methods.
# =====================================================
import os
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


print("\n" + "=" * 80)
print("FINAL OPTIMIZATION METHOD COMPARISON")
print("=" * 80)

print(
    comparison_results.to_string(
        index=False
    )
)


comparison_results.to_csv(

    f"{RESULTS_DIR}/electricity_results.csv",

    index=False

)

print(
    "\nResults saved successfully."
)

print(
    f"File: "
    f"{RESULTS_DIR}/electricity_results.csv"
)