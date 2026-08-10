# =====================================================
# APPLIANCE ENERGY PREDICTION DATASET
# =====================================================
#
# Dataset Name:
# Appliance Energy Prediction
#
# Source:
# UCI Machine Learning Repository
# Dataset ID: 374
#
# Original Machine Learning Task:
# Regression
#
# Current Machine Learning Task:
# Multi-class Classification
#
# =====================================================
# DATASET DESCRIPTION
# =====================================================
#
# The Appliance Energy Prediction dataset contains indoor
# environmental measurements collected from a residential
# building together with the corresponding appliance energy
# consumption. The dataset includes temperature, humidity,
# weather observations, and other environmental variables
# recorded over time.
#
# In the original dataset, the target variable
# "Appliances" represents appliance energy consumption
# measured in Watt-hours (Wh), making it a regression
# problem.
# =====================================================
# INPUT FEATURES
# =====================================================
# The dataset contains environmental and weather-related
# measurements, including:
#
# • Indoor temperatures
# • Indoor humidity levels
# • Outdoor weather conditions
# • Atmospheric pressure
# • Wind speed
# • Visibility
# • Other building sensor measurements
#
# All input variables are numerical.
#
# =====================================================
# TARGET VARIABLE
# =====================================================
#
# Target:
# Cluster
#
# Number of Classes:
# 5
#
# The Cluster variable represents five appliance energy
# consumption categories generated using K-Means clustering
# during Phase 1.
#
# =====================================================
# WHY THIS DATASET WAS SELECTED
# =====================================================
#
# Compared with the Energy Efficiency dataset, this dataset
# contains substantially more observations and a larger
# feature space, providing a more challenging classification
# problem for evaluating hyperparameter optimization
# algorithms.
#

import random
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

dataset = pd.read_csv(
    "processed_datasets/appliance_energy_clustered.csv"
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
# variable (y). The Cluster column represents the class
# labels to be predicted by the MLP classifier, while all
# remaining variables are used as input features.
# =====================================================

X = dataset.drop(
    columns=["Cluster",
    "Consumption_Class",
        "date"]
)

y = dataset["Cluster"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nCluster Distribution:")
print(y.value_counts().sort_index())

# =====================================================
# ONE-HOT ENCODING
# =====================================================
#
# One-Hot Encoding converts categorical predictor variables
# into numerical representations that can be processed by
# machine learning algorithms.
#
# The Appliance Energy Prediction dataset contains only
# continuous numerical predictor variables. Therefore,
# no categorical feature encoding is required and this
# preprocessing step is omitted.
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
# Although the target variable (Appliances) was
# standardized during Phase 1, that transformation was
# performed solely to enable K-Means clustering for
# generating the Cluster labels. The predictor variables
# were not standardized during Phase 1.
#
# Therefore, the predictor variables are standardized in
# Phase 2 before training the MLP classifier. The
# StandardScaler is fitted only on the training data, and
# the same transformation is applied to the testing data
# to prevent data leakage.
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

def plot_training_loss(model, title):

    plt.figure(figsize=(8,5))

    plt.plot(model.loss_curve_)

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
# The Baseline Multi-Layer Perceptron (MLP) is a feedforward
# artificial neural network that serves as the reference
# model for evaluating all hyperparameter optimization
# algorithms in this study.
#
# Unlike the optimization methods, the baseline model uses
# manually selected hyperparameters based on commonly used
# default settings and prior knowledge rather than searching
# for an optimal configuration.
#
# -----------------------------------------------------
# WHY A BASELINE MODEL IS REQUIRED
# -----------------------------------------------------
#
# The baseline establishes a performance benchmark against
# which all optimization algorithms (Random Search,
# Bayesian Optimization, Hyperband, LEA, and TLBO) are
# compared.
#
# Using identical preprocessing steps, training data,
# testing data, and evaluation metrics ensures that any
# improvement in predictive performance can be attributed
# to hyperparameter optimization rather than differences
# in the experimental setup.
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
# These values provide a balanced starting point and are
# widely adopted for training feedforward neural networks.
#
# -----------------------------------------------------
# CHALLENGES AND LIMITATIONS
# -----------------------------------------------------
#
# Since the hyperparameters are manually selected, they may
# not represent the optimal configuration for the Appliance
# Energy Prediction dataset.
#
# The baseline model does not explore alternative network
# architectures or learning parameters and therefore cannot
# automatically adapt to improve predictive performance.
#
# Consequently, the baseline serves only as a reference
# model, while the optimization algorithms attempt to
# identify hyperparameter combinations that achieve better
# generalization and classification performance.
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
# propagation, computes the prediction error, updates the
# network weights using backpropagation, and repeats this
# process until convergence or the maximum number of
# training iterations is reached.
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
# metrics. These results will serve as the benchmark for
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
# convergence behaviour of the baseline MLP and provides
# insight into the effectiveness of the learning process.
# =====================================================

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
# Random Search is a hyperparameter optimization technique
# that randomly samples combinations of hyperparameters
# from a predefined search space and evaluates each
# combination using cross-validation.
#
# Unlike Grid Search, which evaluates every possible
# hyperparameter combination, Random Search explores only
# a randomly selected subset of the search space. This
# significantly reduces computational cost while often
# identifying high-performing configurations.
#
# -----------------------------------------------------
# WHY RANDOM SEARCH IS USED
# -----------------------------------------------------
#
# Random Search is included in this study to provide a
# simple and computationally efficient benchmark for
# hyperparameter optimization on the Appliance Energy
# Prediction dataset.
#
# Unlike the baseline MLP, which uses manually selected
# hyperparameters, Random Search automatically explores
# different hyperparameter combinations from the
# predefined search space to identify improved model
# configurations.
#
# Compared with Bayesian Optimization, Random Search does
# not learn from previously evaluated hyperparameter
# combinations. Instead, each candidate solution is
# sampled independently at random.
#
# Compared with LEA and TLBO, which iteratively improve
# candidate solutions through evolutionary or
# teacher-learner interactions, Random Search performs no
# adaptive learning during the optimization process.
#
# Despite its simplicity, Random Search is widely used as
# a benchmark because it provides a straightforward and
# computationally efficient approach for evaluating the
# effectiveness of more advanced optimization algorithms.
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Simple to implement.
# • Computationally more efficient than Grid Search.
# • Explores diverse regions of the search space.
# • Frequently identifies near-optimal solutions with
#   fewer model evaluations.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Samples hyperparameter combinations completely at
#   random without learning from previously evaluated
#   solutions.
#
# • High-performing regions discovered during the search
#   are not exploited in later iterations.
#
# • The quality of the final solution depends on the
#   number of random samples evaluated.
# =====================================================

from sklearn.model_selection import RandomizedSearchCV

# =====================================================
# RANDOM SEARCH OPTIMIZATION
# =====================================================
#
# Configure the RandomizedSearchCV optimizer using the
# predefined MLP hyperparameter search space.
#
# Twenty randomly selected hyperparameter combinations
# are evaluated using 5-fold cross-validation, and the
# configuration achieving the highest weighted F1-score
# is selected as the optimal solution.
# =====================================================
print("\nBuilding Random Search Optimizer...")

random_search = RandomizedSearchCV(

    estimator=MLPClassifier(

        max_iter=500,

        random_state=42

    ),

    param_distributions=SEARCH_SPACE,

    n_iter=20,

    scoring="f1_weighted",

    cv=5,

    random_state=42,

    n_jobs=-1

)

print("\nStarting Random Search Optimization...")

start = time.time()

random_search.fit(

    X_train,

    y_train

)

end = time.time()

random_search_runtime = end - start

print("\nRandom Search completed.")
print(f"Optimization Time: {random_search_runtime:.2f} seconds")

# =====================================================
# BUILD RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================
#
# Construct an MLP classifier using the optimal
# hyperparameter configuration identified by Random
# Search. The optimized model is then trained on the
# complete training dataset before being evaluated on the
# unseen testing dataset.
# =====================================================

print("\nBuilding Random Search Optimized MLP Model...")

random_model = build_mlp(
    random_search.best_params_
)

# =====================================================
# TRAIN RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================
#
# Train the optimized MLP classifier using the
# standardized training dataset. The network performs
# forward propagation, computes the prediction error,
# updates the model weights through backpropagation, and
# repeats this process until convergence or the maximum
# number of training iterations is reached.
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
# EVALUATE RANDOM SEARCH OPTIMIZED MLP MODEL
# =====================================================
#
# Evaluate the trained Random Search optimized MLP on the
# testing dataset using the same evaluation metrics as the
# baseline model. The resulting performance metrics are
# later compared with those obtained from the baseline and
# other optimization algorithms.
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
#
# Plot the training loss recorded during each iteration
# of network training. The loss curve illustrates the
# convergence behaviour of the Random Search optimized
# MLP and provides insight into the effectiveness of the
# learning process.
# =====================================================

plot_training_loss(

    random_model,

    "Random Search Optimized MLP Training Loss"

)

# =====================================================
# RANDOM SEARCH SUMMARY
# =====================================================
#
# Display the optimal hyperparameter configuration
# identified by Random Search together with the best
# cross-validation weighted F1-score obtained during the
# optimization process.
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
# The Appliance Energy Prediction dataset contains a
# discrete hyperparameter search space for the MLP
# classifier. Optuna is used because it efficiently
# handles categorical hyperparameters while requiring
# fewer model evaluations than exhaustive or random
# search methods.
#
# Compared with Random Search, Optuna uses information
# obtained from previous trials to guide the search
# towards more promising hyperparameter configurations.
#
# Compared with LEA and TLBO, Optuna does not rely on
# evolutionary operators such as mutation or teacher-
# learner interactions. Instead, it models the search
# process probabilistically.
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
# • Well suited for expensive machine learning models.
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
# Each candidate model is evaluated using weighted
# F1-score obtained from 5-fold cross-validation.
# =====================================================

def objective(trial):

    params = {

        "hidden_layer_sizes": trial.suggest_categorical(

            "hidden_layer_sizes",

            [
                (16,),
                (32,),
                (64,),
                (32, 16),
                (64, 32)
            ]

        ),

        "activation": trial.suggest_categorical(

            "activation",

            ["relu", "tanh"]

        ),

        "solver": "adam",

        "learning_rate_init": trial.suggest_categorical(

            "learning_rate_init",

            [0.0001, 0.001]

        ),

        "alpha": trial.suggest_categorical(

            "alpha",

            [0.0001, 0.001, 0.01]

        ),

        "batch_size": trial.suggest_categorical(

            "batch_size",

            [32, 64, 128]

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
# maximize the weighted F1-score obtained from the
# objective function.
#
# Twenty optimization trials are performed to maintain
# consistency with the Random Search, Hyperband, LEA,
# and TLBO optimization algorithms.
# =====================================================

print("\nBuilding Bayesian Optimizer...")

study = optuna.create_study(

    direction="maximize",

    study_name="MLP_Bayesian_Optimization"

)

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
# BUILD OPTIMIZED MLP MODEL
# =====================================================
#
# Retrieve the best hyperparameter combination identified
# by Bayesian Optimization and construct the final MLP
# classifier using these optimized hyperparameters.
#
# The optimized model will subsequently be trained on the
# complete training dataset before evaluation on the
# independent test dataset.
# =====================================================

best_params = study.best_params

print("\nBest Hyperparameters Found\n")

for key, value in best_params.items():

    print(f"{key}: {value}")

bayesian_model = build_mlp(

    best_params

)
# =====================================================
# TRAIN OPTIMIZED MODEL
# =====================================================
#
# Train the optimized MLP classifier using the complete
# training dataset and record the total training time.
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
# Evaluate the optimized MLP classifier on the unseen
# testing dataset using the same evaluation metrics as
# the remaining optimization algorithms.
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
# The Appliance Energy Prediction dataset contains a
# relatively large hyperparameter search space for the
# MLP classifier, making exhaustive evaluation
# computationally expensive. Hyperband is therefore used
# to efficiently identify high-performing
# hyperparameter configurations while reducing the
# overall optimization time.
#
# Unlike Random Search, which evaluates every candidate
# hyperparameter combination using the same amount of
# computational resources, Hyperband progressively
# allocates additional resources only to promising
# candidates and eliminates poorly performing
# configurations early in the optimization process.
#
# Compared with Bayesian Optimization, Hyperband does
# not construct a probabilistic surrogate model to guide
# the search. Instead, it improves optimization
# efficiency through adaptive resource allocation.
#
# Compared with LEA and TLBO, which iteratively improve
# candidate solutions using evolutionary and
# teacher-learner mechanisms, Hyperband focuses on
# efficiently managing computational resources rather
# than modifying candidate solutions during the
# optimization process.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Efficiently allocates computational resources by
#   terminating poorly performing candidate solutions
#   early.
#
# • Evaluates a larger number of hyperparameter
#   combinations within a fixed computational budget.
#
# • Reduces unnecessary training time compared with
#   evaluating every candidate using the same resources.
#
# • Well suited for computationally expensive models
#   such as neural networks.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Performance depends on the initial hyperparameter
#   candidates selected for evaluation.
#
# • Promising hyperparameter configurations may be
#   discarded prematurely if their early performance is
#   poor.
#
# • Does not learn from previous evaluations through a
#   probabilistic model like Bayesian Optimization.
# =====================================================

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV


# -----------------------------------------------------
# HYPERBAND CONFIGURATION
# -----------------------------------------------------
#
# factor = 3
# The reduction factor determines how aggressively
# Hyperband eliminates poorly performing hyperparameter
# configurations after each evaluation stage. A value of
# 3 provides a good balance between exploring many
# candidate solutions and allocating additional
# computational resources to the most promising
# configurations. This is also the standard value
# recommended for Hyperband.
#
# resource = "max_iter"
# The number of training iterations (max_iter) is used as
# the computational resource allocated to each candidate
# MLP model. Promising hyperparameter configurations are
# progressively trained for more iterations, while poorly
# performing configurations are discarded early.
#
# max_resources = 500
# The maximum computational budget is set to 500 training
# iterations, matching the maximum number of iterations
# used by the baseline MLP and all other optimization
# algorithms. Using the same upper limit ensures a fair
# comparison across all optimization methods.
#
# min_resources
# The minimum computational budget is determined
# automatically by HalvingRandomSearchCV based on the
# dataset and search configuration. Therefore, the
# default value is retained.

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
print("\nBuilding Hyperband Optimized MLP Model...")

hyperband_model = build_mlp(
    hyperband.best_params_
)

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

plot_training_loss(

    hyperband_model,

    "Hyperband Optimized MLP Training Loss"

)

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
# LEA is a population-based hyperparameter optimization
# algorithm that iteratively improves candidate MLP
# hyperparameter configurations through evaluation,
# selection, and mutation.
#
# During each iteration, candidate solutions are
# evaluated using weighted F1-score obtained from
# 5-fold cross-validation. The best-performing
# hyperparameter configurations are retained, while new
# candidate solutions are generated by randomly mutating
# selected hyperparameters from the surviving
# configurations.
#
# -----------------------------------------------------
# WHY LEA IS USED
# -----------------------------------------------------
#
# The Appliance Energy Prediction dataset contains a
# multi-dimensional hyperparameter search space in which
# different combinations of MLP hyperparameters may
# produce substantially different classification
# performance.
#
# LEA is included in this study because it progressively
# improves candidate hyperparameter configurations by
# exploiting information obtained from previously
# evaluated solutions while maintaining diversity through
# mutation.
#
# Unlike Random Search, which samples candidate
# hyperparameter combinations independently, LEA uses the
# performance of previous generations to guide the search
# toward promising regions of the search space.
#
# Compared with Bayesian Optimization, LEA does not
# construct a probabilistic surrogate model. Instead, it
# relies on evolutionary principles to improve candidate
# solutions over successive generations.
#
# Compared with Hyperband, which focuses on efficient
# allocation of computational resources, LEA searches for
# improved hyperparameter configurations by evolving the
# population across multiple generations.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Exploits high-performing candidate solutions while
#   maintaining diversity through mutation.
#
# • Effective for discrete hyperparameter search spaces.
#
# • Simple to implement and computationally efficient.
#
# • Can progressively improve solutions over successive
#   generations.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Performance depends on the initial population.
#
# • Random mutation may occasionally generate inferior
#   candidate solutions.
#
# • Does not explicitly model the search space like
#   Bayesian Optimization.
#
# • May converge to a local optimum if population
#   diversity decreases significantly.
# =====================================================
# -----------------------------------------------------
# LEA PARAMETER CONFIGURATION
# -----------------------------------------------------
#
# POPULATION_SIZE = 20
# The population size determines the number of candidate
# hyperparameter combinations evaluated during each
# generation. A population of 20 provides sufficient
# diversity to explore the predefined search space while
# maintaining a reasonable computational cost for
# repeated 5-fold cross-validation.
#
# SURVIVAL_RATE = 0.50
# The survival rate specifies the proportion of the
# highest-performing candidate solutions retained after
# each generation. Retaining the top 50% balances
# exploitation of promising solutions with exploration
# through mutation of the surviving candidates.
#
# MAX_ITERATIONS = 20
# The optimization process is performed for 20
# generations, providing sufficient opportunities for
# candidate solutions to improve while keeping the total
# optimization time comparable to the other optimization
# algorithms evaluated in this study.
# -----------------------------------------------------

POPULATION_SIZE = 20

SURVIVAL_RATE = 0.50

MAX_ITERATIONS = 20

# =====================================================
# GENERATE INITIAL POPULATION
# =====================================================
#
# Create the initial population by randomly generating
# candidate MLP hyperparameter combinations from the
# predefined search space.
#
# Each candidate solution represents one possible set of
# MLP hyperparameters that will be evaluated using
# 5-fold cross-validation. The initial population serves
# as the starting point for the evolutionary
# optimization process.
# =====================================================

population = []

for _ in range(POPULATION_SIZE):

    candidate = {

        parameter: random.choice(values)

        for parameter, values in SEARCH_SPACE.items()

    }

    population.append(candidate)

print("\nInitial Population Created.")

print(f"Population Size: {len(population)}")

# =====================================================
# EVALUATE POPULATION FITNESS
# =====================================================
#
# Evaluate every candidate hyperparameter combination in
# the current population using 5-fold cross-validation.
#
# The weighted F1-score obtained from cross-validation is
# used as the fitness value. Candidate solutions with
# higher fitness scores represent better-performing MLP
# hyperparameter configurations and are more likely to
# survive to the next generation.
# =====================================================
fitness_scores = []

best_score = float("-inf")

best_solution = None

start = time.time()

for iteration in range(MAX_ITERATIONS):

    print(f"\nGeneration {iteration + 1}/{MAX_ITERATIONS}")

    fitness_scores = []

    for solution in population:

        model = build_mlp(solution)

        score = cross_val_score(

            model,

            X_train,

            y_train,

            cv=5,

            scoring="f1_weighted",

            n_jobs=-1

        ).mean()

        fitness_scores.append(score)

        if score > best_score:

            best_score = score

            best_solution = solution.copy()

# =====================================================
# RANK CANDIDATE SOLUTIONS
# =====================================================
#
# Pair each candidate hyperparameter combination with its
# corresponding weighted F1-score and sort the population
# in descending order of fitness.
#
# The highest-performing candidate solutions are placed
# at the beginning of the population, enabling the best
# candidates to be selected for the next generation.
# =====================================================
    ranked_population = [

        solution

        for _, solution in sorted(

            zip(fitness_scores, population),

            key=lambda x: x[0],

            reverse=True

        )

    ]
# =====================================================
# SELECT SURVIVING CANDIDATES
# =====================================================
#
# Retain the top-performing candidate solutions based on
# the specified survival rate.
#
# These surviving candidates represent the best
# hyperparameter configurations from the current
# generation and serve as the parents for generating the
# next population.
# =====================================================
    survivor_count = int(

        POPULATION_SIZE * SURVIVAL_RATE

    )
    print(
    f"Best F1 Score: {max(fitness_scores):.4f}"
    )
    survivors = ranked_population[:survivor_count]
# =====================================================
# GENERATE NEXT POPULATION
# =====================================================
#
# Randomly select one of the surviving candidate
# solutions and randomly modify one of its
# hyperparameters to generate a new candidate solution.
#
# This mutation process introduces diversity into the
# population while preserving information from the
# highest-performing candidates until the population
# size is restored.
# =====================================================
    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(survivors).copy()

        parameter = random.choice(

            list(parent.keys())

        )

        if parameter != "solver":

            parent[parameter] = random.choice(

                SEARCH_SPACE[parameter]

            )

        new_population.append(parent)

    population = new_population
end = time.time()

lea_runtime = end - start
print("\nBuilding LEA Optimized MLP Model...")

lea_model = build_mlp(best_solution)

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

plot_training_loss(

    lea_model,

    "LEA Optimized MLP Training Loss"

)

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
# TLBO is a population-based hyperparameter optimization
# algorithm inspired by the teaching and learning process
# in a classroom. Candidate hyperparameter combinations
# are considered as learners, while the best-performing
# candidate in the population acts as the teacher.
#
# During each generation, the optimization process is
# divided into two phases. In the Teacher Phase, the
# teacher attempts to improve the overall population by
# guiding learners towards better solutions. In the
# Learner Phase, candidate solutions further improve by
# interacting and learning from one another.
#
# -----------------------------------------------------
# WHY TLBO IS USED
# -----------------------------------------------------
#
# The Appliance Energy Prediction dataset contains a
# multi-dimensional MLP hyperparameter search space.
# TLBO is included in this study because it improves
# candidate hyperparameter configurations through
# collaborative learning rather than random mutation or
# probabilistic modelling.
#
# Unlike Random Search, which samples hyperparameter
# combinations independently, TLBO continuously refines
# candidate solutions over successive generations.
#
# Compared with Bayesian Optimization, TLBO does not
# construct a probabilistic surrogate model. Instead,
# knowledge is transferred directly between candidate
# solutions through teacher and learner interactions.
#
# Compared with Hyperband, which focuses on allocating
# computational resources efficiently, TLBO focuses on
# improving candidate hyperparameter configurations
# through iterative learning.
#
# Compared with LEA, which generates new solutions using
# mutation, TLBO improves existing solutions through
# teacher guidance and learner interactions without
# mutation.
#
# -----------------------------------------------------
# ADVANTAGES
# -----------------------------------------------------
#
# • Simple population-based optimization algorithm.
#
# • Does not require algorithm-specific parameters such
#   as crossover or mutation probabilities.
#
# • Balances exploration and exploitation through teacher
#   and learner phases.
#
# • Well suited for discrete hyperparameter optimization.
#
# -----------------------------------------------------
# LIMITATIONS
# -----------------------------------------------------
#
# • Performance depends on the quality of the teacher.
#
# • May converge prematurely if population diversity
#   decreases significantly.
#
# • Can require several generations to identify highly
#   competitive hyperparameter configurations.
# =====================================================
# -----------------------------------------------------
# TLBO PARAMETER CONFIGURATION
# -----------------------------------------------------
#
# POPULATION_SIZE = 20
# The population size determines the number of learner
# solutions participating in the optimization process.
# A population of 20 provides sufficient diversity for
# exploring the predefined MLP hyperparameter search
# space while maintaining a reasonable computational cost
# for repeated 5-fold cross-validation.
#
# MAX_ITERATIONS = 20
# The optimization process is performed for 20
# generations. This provides sufficient opportunities
# for the learners to improve through the Teacher and
# Learner phases while maintaining computational
# consistency with the other optimization algorithms
# evaluated in this study.
# -----------------------------------------------------

POPULATION_SIZE = 20

MAX_ITERATIONS = 20
population = []

for _ in range(POPULATION_SIZE):

    candidate = {

        parameter: random.choice(values)

        for parameter, values in SEARCH_SPACE.items()

    }

    population.append(candidate)

print("\nInitial Population Created.")
print(f"Population Size: {len(population)}")
# =====================================================
# EVALUATE POPULATION FITNESS
# =====================================================
#
# Evaluate every learner in the current population using
# 5-fold cross-validation.
#
# The weighted F1-score obtained from cross-validation is
# used as the fitness value. Learners with higher fitness
# scores represent better-performing MLP hyperparameter
# configurations and are more likely to influence other
# learners during the Teacher and Learner phases.
# =====================================================
fitness_scores = []

best_score = float("-inf")

best_solution = None

start = time.time()

for iteration in range(MAX_ITERATIONS):

    print(f"\nGeneration {iteration + 1}/{MAX_ITERATIONS}")

    fitness_scores = []

    for solution in population:

        model = build_mlp(solution)

        score = cross_val_score(

            model,

            X_train,

            y_train,

            cv=5,

            scoring="f1_weighted",

            n_jobs=-1

        ).mean()

        fitness_scores.append(score)

        if score > best_score:

            best_score = score

            best_solution = solution.copy()

# =====================================================
# IDENTIFY THE TEACHER
# =====================================================
#
# Select the learner with the highest weighted F1-score
# in the current generation. This learner acts as the
# teacher and represents the best hyperparameter
# configuration discovered in the current population.
# =====================================================
    teacher_index = fitness_scores.index(

        max(fitness_scores)

    )

    teacher = population[teacher_index]
# =====================================================
# UPDATE LEARNERS USING THE TEACHER
# =====================================================
#
# Each learner is updated by randomly selecting one
# hyperparameter and replacing it with the corresponding
# value from the teacher. The updated learner is
# evaluated and retained only if it improves the
# weighted F1-score.
# =====================================================

# =====================================================
# TEACHER PHASE
# =====================================================
#
# Each learner attempts to improve by adopting knowledge
# from the teacher. A randomly selected hyperparameter is
# replaced with the corresponding value from the teacher.
# The updated learner is evaluated using 5-fold cross-
# validation and retained only if its weighted F1-score
# improves.
# =====================================================
    for i in range(POPULATION_SIZE):

        if i == teacher_index:

            continue

        new_solution = population[i].copy()

        parameter = random.choice(

            list(SEARCH_SPACE.keys())

        )

        new_solution[parameter] = teacher[parameter]

        model = build_mlp(new_solution)

        new_score = cross_val_score(

            model,

            X_train,

            y_train,

            cv=5,

            scoring="f1_weighted",

            n_jobs=-1

        ).mean()

        if new_score > fitness_scores[i]:

            population[i] = new_solution

            fitness_scores[i] = new_score

            if new_score > best_score:

                best_score = new_score

                best_solution = new_solution.copy()

# =====================================================
# LEARNER PHASE
# =====================================================
#
# During the Learner Phase, each learner interacts with
# another randomly selected learner. Knowledge is
# exchanged based on their relative fitness values.
#
# The updated learner is evaluated using 5-fold
# cross-validation and retained only if it achieves a
# higher weighted F1-score than its previous
# configuration.
# =====================================================

    for i in range(POPULATION_SIZE):

        partner = random.randrange(POPULATION_SIZE)

        while partner == i:

            partner = random.randrange(POPULATION_SIZE)

        new_solution = population[i].copy()

        parameter = random.choice(

            list(SEARCH_SPACE.keys())

        )

        if fitness_scores[partner] > fitness_scores[i]:

            new_solution[parameter] = population[partner][parameter]

        else:

            continue

        model = build_mlp(new_solution)

        new_score = cross_val_score(

            model,

            X_train,

            y_train,

            cv=5,

            scoring="f1_weighted",

            n_jobs=-1

        ).mean()

        if new_score > fitness_scores[i]:

            population[i] = new_solution

            fitness_scores[i] = new_score

            if new_score > best_score:

                best_score = new_score

                best_solution = new_solution.copy()


end = time.time()

tlbo_runtime = end - start
print("\nTLBO Optimization completed.")

print(
        f"Optimization Time: "
        f"{tlbo_runtime:.2f} seconds")
    
print("\nBuilding TLBO Optimized MLP Model...")

tlbo_model = build_mlp(
        best_solution
    )

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
plot_training_loss(

    tlbo_model,

    "TLBO Optimized MLP Training Loss"

)
print_optimization_summary(

    method_name="TLBO",

    best_params=best_solution,

    best_cv_score=best_score,

    optimization_time=tlbo_runtime,

    training_time=tlbo_training_runtime

)
import os

RESULTS_DIR = "results"

os.makedirs(

    RESULTS_DIR,

    exist_ok=True

)
# =====================================================
# SAVE EXPERIMENT RESULTS
# =====================================================
#
# Store the performance metrics and execution times of
# all optimization methods. The results are saved as a
# CSV file for subsequent comparative analysis and
# visualization.
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

print("\nComparison Results")

print(comparison_results)

comparison_results.to_csv(

    f"{RESULTS_DIR}/appliance_energy_results.csv",

    index=False

)

print("\nResults saved successfully.")