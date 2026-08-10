import pandas as pd
import random
import numpy as np
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)
import os
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
# =====================================================
# LOAD CLUSTERED DATASET
# =====================================================

df = pd.read_csv(
    "processed_datasets/energy_efficiency_clustered.csv"
)
# =====================================================
# DEFINE FEATURES AND TARGET
# =====================================================

X = df[[
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8"
]]
y = df["Cluster"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nCluster Distribution:")
print(y.value_counts().sort_index())

# =====================================================
# TRAIN-TEST SPLIT
# =====================================================
# Split the dataset into training (80%) and testing (20%)
# subsets. Stratified sampling preserves the proportion of
# each cluster in both subsets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
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
# FEATURE SCALING
# =====================================================
# Standardize the training features so that each feature
# has a mean of 0 and a standard deviation of 1.
# The same transformation is then applied to the test set
# using the scaler fitted on the training data.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =====================================================
# COMMON HYPERPARAMETER SEARCH SPACE
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
# Creates an MLP classifier using a given set of
# hyperparameters. All optimization methods use this
# function so that the same MLP configuration is
# constructed consistently throughout the study.

def build_mlp(params):

    return MLPClassifier(

        # One hidden layer with 64 neurons.
        # A moderate network size that is large enough to learn
        # nonlinear patterns without being unnecessarily complex.
        hidden_layer_sizes=params["hidden_layer_sizes"],

        # ReLU is widely used because it trains faster and reduces
        # the vanishing gradient problem compared to sigmoid or tanh.
        activation=params["activation"],

        # Adam is chosen because it generally converges faster and
        # performs well on a wide range of problems.
        solver=params.get("solver", "adam"),

        # The default learning rate in scikit-learn.
        # Provides stable weight updates during training.
        learning_rate_init=params["learning_rate_init"],

        # Default L2 regularization strength to help reduce overfitting.
        alpha=params["alpha"],

        # A small batch size provides stable learning while keeping
        # computation efficient.
        batch_size=params["batch_size"],

        # Allows sufficient iterations for the network to converge.
        max_iter=500,

        # Ensures reproducible experimental results.
        random_state=42

    )
# =====================================================
# EVALUATE MODEL
# =====================================================
# Evaluates a trained MLP model on the testing dataset
# and prints the performance metrics.

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

    print(f"\n{method_name.upper()} PERFORMANCE")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    return accuracy, precision, recall, f1
# =====================================================
# BASELINE MLP MODEL
# =====================================================
# Construct the baseline MLP classifier to provide a
# reference performance for this benchmark dataset.
# Subsequent hyperparameter optimization methods(Random Search, 
# Bayesian Optimization, Hyperband, LEA, and SOCIAL) will
# be evaluated against this baseline using identical
# training and evaluation settings to assess
# the effectiveness of each optimization method.

baseline_params = {

    "hidden_layer_sizes": (64,),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": 32
}

baseline_model = build_mlp(baseline_params)

# =====================================================
# TRAIN BASELINE MLP MODEL
# =====================================================
# Train the MLP using forward propagation, error computation,
# backpropagation, and iterative weight updates until convergence.



start = time.time()

baseline_model.fit(X_train, y_train)

end = time.time()

baseline_runtime = end - start

print("\nBaseline MLP model training completed.")
print(f"Training Time: {baseline_runtime:.2f} seconds")

# =====================================================
# PREDICT TEST DATA
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

plt.figure(figsize=(8,5))
plt.plot(baseline_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Baseline MLP Training Loss")
plt.grid(True)
plt.show()


# =====================================================
# RANDOM SEARCH HYPERPARAMETER OPTIMIZATION
# =====================================================
# Random Search is used as the first hyperparameter
# optimization method to identify improved MLP
# configurations. Its performance will serve as a
# benchmark for comparison with Bayesian Optimization,
# Hyperband, LEA, and SOCIAL.

random_search = RandomizedSearchCV(

    # Base MLP model whose hyperparameters will be optimized.
    estimator=MLPClassifier(
        max_iter=500,
        random_state=42
    ),

    # Search space containing candidate hyperparameter values.
    # Random Search samples different combinations from this
    # space to identify improved MLP configurations.
    param_distributions=SEARCH_SPACE,

    # Evaluate 20 randomly selected hyperparameter combinations
    # from a search space of 180 possible combinations. Sampling
    # approximately 11% of the search space provides a reasonable
    # balance between search diversity and computational cost.
    n_iter=20,

    # Evaluate each sampled hyperparameter combination using
    # 5-fold cross-validation(done to see how well model 
    # generalize the unseen data). Five folds provide a reliable
    # estimate of model performance while maintaining a good
    # balance between computational efficiency and evaluation
    # stability.
    cv=5,

    # Select the hyperparameter combination with the highest
    # weighted F1-score. Accuracy alone may be biased towards
    # the majority cluster, Precision ignores missed samples,
    # and Recall ignores false positives. The weighted F1-score
    # balances Precision and Recall while accounting for the
    # imbalanced cluster distribution.
    scoring="f1_weighted",

    # Ensures reproducible random sampling.
    random_state=42,

    # Use all available CPU cores for parallel processing.
    n_jobs=-1
)

# =====================================================
# TRAIN RANDOM SEARCH
# =====================================================

start = time.time()

random_search.fit(X_train, y_train)

end = time.time()

random_search_runtime = end - start

print("\nRandom Search completed.")
print(f"Training Time: {random_search_runtime:.2f} seconds")

# =====================================================
# BEST HYPERPARAMETERS
# =====================================================

print("\nBest Hyperparameters")
print(random_search.best_params_)

print("\nBest Cross Validation F1 Score")
print(random_search.best_score_)

# =====================================================
# BEST OPTIMIZED MODEL
# =====================================================

best_model = random_search.best_estimator_

# =====================================================
# PREDICT USING BEST MODEL
# =====================================================

random_accuracy, \
random_precision, \
random_recall, \
random_f1 = evaluate_model(

    best_model,
    X_test,
    y_test,
    "Random Search"
)

plt.figure(figsize=(8,5))
plt.plot(best_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Random Search Optimized MLP Training Loss")
plt.grid(True)
plt.show()


# =====================================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# =====================================================
# Bayesian Optimization uses the results from previous
# trials to intelligently select the next set of
# hyperparameters. Unlike Random Search, it focuses on
# promising regions of the search space to identify
# better hyperparameter combinations with fewer
# evaluations.
#
# Challenge:
# BayesSearchCV was initially considered for Bayesian
# Optimization. However, it expects each hyperparameter
# to follow a fixed structure throughout the search
# process. In this study, the hidden layer architecture
# contains both one-hidden-layer models (e.g., (64,))
# and two-hidden-layer models (e.g., (32,16)), causing
# search space representation errors before model
# training could begin.
#
# Solution:
# Optuna is used because it builds the neural network
# architecture separately for each trial. It first
# selects the number of hidden layers and then chooses
# the number of neurons for each layer. This allows both
# one-hidden-layer and two-hidden-layer MLP models to be
# optimized while keeping the same search space used in
# Random Search for a fair comparison.


# =====================================================
# OBJECTIVE FUNCTION
# =====================================================
# Define the objective function used by Optuna.
# Each trial creates a new MLP configuration,
# evaluates it using 5-fold cross-validation,
# and returns the weighted F1-score.

def objective(trial):

    # ---------------------------------------------
    # Number of Hidden Layers
    # ---------------------------------------------

    n_layers = trial.suggest_int(
        "n_layers",
        1,
        2
    )

    hidden_layers = []

    # ---------------------------------------------
    # Number of Neurons in Each Layer
    # ---------------------------------------------

    for i in range(n_layers):

        neurons = trial.suggest_categorical(
            f"layer_{i}",
            [16, 32, 64]
        )

        hidden_layers.append(neurons)

    hidden_layers = tuple(hidden_layers)

    # ---------------------------------------------
    # Remaining Hyperparameters
    # ---------------------------------------------

    activation = trial.suggest_categorical(
        "activation",
        ["relu", "tanh"]
    )

    alpha = trial.suggest_categorical(
        "alpha",
        [0.0001, 0.001, 0.01]
    )

    learning_rate = trial.suggest_categorical(
        "learning_rate_init",
        [0.0001, 0.001]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    # ---------------------------------------------
    # Build MLP
    # ---------------------------------------------

    model = MLPClassifier(

        hidden_layer_sizes=hidden_layers,

        activation=activation,

        solver="adam",

        alpha=alpha,

        learning_rate_init=learning_rate,

        batch_size=batch_size,

        max_iter=500,

        random_state=42

    )

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

study = optuna.create_study(
    direction="maximize"
)
# =====================================================
# RUN BAYESIAN OPTIMIZATION
# =====================================================

start = time.time()

study.optimize(

    objective,

    n_trials=20

)

end = time.time()

optuna_runtime = end - start

print("\nBayesian Optimization completed.")
print(f"Training Time: {optuna_runtime:.2f} seconds")

# =====================================================
# BEST HYPERPARAMETERS
# =====================================================
# Display the best hyperparameter combination identified
# by Bayesian Optimization.

print("\nBest Hyperparameters")
print(study.best_params)

print("\nBest Cross Validation F1 Score")
print(f"{study.best_value:.4f}")
# =====================================================
# BUILD OPTIMIZED MLP MODEL
# =====================================================
# Build the final MLP model using the best
# hyperparameters identified by Optuna.
best_params = study.best_params

hidden_layers = [best_params["layer_0"]]

if best_params["n_layers"] == 2:
    hidden_layers.append(best_params["layer_1"])

hidden_layers = tuple(hidden_layers)

converted_params = {

    "hidden_layer_sizes": hidden_layers,

    "activation": best_params["activation"],

    "solver": "adam",

    "learning_rate_init": best_params["learning_rate_init"],

    "alpha": best_params["alpha"],

    "batch_size": best_params["batch_size"]
}

bayes_model = build_mlp(converted_params)
# =====================================================
# TRAIN OPTIMIZED MODEL
# =====================================================
# Train the optimized MLP using the complete
# training dataset.

bayes_model.fit(X_train, y_train)

print("\nBayesian Optimized MLP training completed.")

plt.figure(figsize=(8,5))
plt.plot(bayes_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Bayesian Optimization MLP Training Loss")
plt.grid(True)
plt.show()
# =====================================================
# PREDICT TEST DATA
# =====================================================

bayes_accuracy, \
bayes_precision, \
bayes_recall, \
bayes_f1 = evaluate_model(

    bayes_model,
    X_test,
    y_test,
    "Bayesian Optimization"
)


# =====================================================
# HYPERBAND HYPERPARAMETER OPTIMIZATION
# =====================================================
# Hyperband begins by randomly generating multiple
# hyperparameter combinations. Instead of fully training
# every model, it allocates only a small amount of
# training resources to each candidate. Poor-performing
# models are eliminated early, while the remaining
# promising models continue training with additional
# resources.
#
# Difference from Baseline MLP:
# The Baseline MLP trains a single model using manually
# selected hyperparameters without performing any
# hyperparameter optimization.
#
# Difference from Random Search:
# Random Search evaluates randomly selected
# hyperparameter combinations by fully training every
# candidate. Hyperband also begins with random
# combinations but reduces computation by stopping
# weak-performing models early.
#
# Difference from Bayesian Optimization:
# Bayesian Optimization uses the results from previous
# trials to intelligently select the next
# hyperparameter combination. Hyperband does not learn
# from previous trials; instead, it improves efficiency
# by allocating more training resources only to the most
# promising randomly generated models.

# =====================================================
# CREATE HYPERBAND SEARCH OBJECT
# =====================================================
# Configure the Hyperband optimizer. Randomly generated
# hyperparameter combinations are initially trained
# using limited resources. After each round, poorly
# performing models are eliminated while the remaining
# models continue training with additional resources.

hyperband_search = HalvingRandomSearchCV(

    # Base MLP model whose hyperparameters
    # will be optimized.
    estimator=MLPClassifier(
        max_iter=500,
        random_state=42
    ),

    # Search space explored by Hyperband.
    param_distributions=SEARCH_SPACE,

    # Approximately one-third of the best-performing
    # hyperparameter combinations continue to the
    # next round, while the remaining combinations
    # are eliminated.
    factor=3,

    # The maximum number of training iterations allocated
    # to each hyperparameter combination/models before its
    # performance is evaluated.
    # Poor-performing combination are discarded before using
    # the maximum number of iterations.
    resource="max_iter",

    # Promising models may train for up to 500 iterations,
    # matching the Baseline MLP, Random Search and
    # Bayesian Optimization for a fair comparison.
    max_resources=500,

    # Five-fold cross-validation provides reliable
    # model evaluation while balancing computational cost.
    cv=5,

    # Optimize the weighted F1-score because
    # the clusters are imbalanced.
    scoring="f1_weighted",

    # Ensures reproducible results.
    random_state=42,

    # Use all available CPU cores.
    n_jobs=-1

)
# =====================================================
# RUN HYPERBAND OPTIMIZATION
# =====================================================

start = time.time()

hyperband_search.fit(
    X_train,
    y_train
)

end = time.time()

hyperband_runtime = end - start

print("\nHyperband completed.")
print(f"Training Time: {hyperband_runtime:.2f} seconds")

# =====================================================
# BEST HYPERPARAMETERS
# =====================================================

print("\nBest Hyperparameters")
print(hyperband_search.best_params_)

print("\nBest Cross Validation F1 Score")
print(f"{hyperband_search.best_score_:.4f}")

# =====================================================
# BUILD OPTIMIZED MLP MODEL
# =====================================================
# Build the final MLP model using the best
# hyperparameters identified by Hyperband.

hyperband_model = build_mlp(
    hyperband_search.best_params_
)

# =====================================================
# TRAIN OPTIMIZED MLP MODEL
# =====================================================
# Train the optimized MLP using the complete
# training dataset.

hyperband_model.fit(X_train, y_train)

print("\nHyperband Optimized MLP training completed.")

plt.figure(figsize=(8,5))
plt.plot(hyperband_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Hyperband MLP Training Loss")
plt.grid(True)
plt.show()

# =====================================================
# PREDICT TEST DATA
# =====================================================

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
# LEA HYPERPARAMETER OPTIMIZATION
# =====================================================
#
# LEA searches the same hyperparameter space as the
# previous optimization methods. Each individual in the
# population represents one candidate MLP
# hyperparameter configuration.
#
# The best solution found by LEA is used to train the
# final MLP model.

# =====================================================
# LEA PARAMETERS
# =====================================================

# Number of candidate hyperparameter combinations maintained
# in each generation(iteration). Unlike Random Search, these candidates
# evolve over successive iterations by retaining high-performing
# solutions and generating new ones through mutation.
POPULATION_SIZE = 20

# Number of optimization generations(iterations). In each iteration, the
# population is evaluated, the best-performing solutions are
# selected, and new candidate hyperparameter combinations are
# generated. This iterative refinement distinguishes LEA from
# Random Search, while differing from Bayesian Optimization
# (surrogate model-based search) and Hyperband (resource-based
# early stopping).
MAX_ITERATIONS = 20


# All optimization methods use 5-fold cross-validation to
# evaluate candidate hyperparameter combinations. Therefore,
# increasing the search budget (e.g., number of iterations,
# trials, generations, or population size) increases the
# overall computational cost.

# =====================================================
# CREATE RANDOM LEA SOLUTION
# =====================================================

def create_solution():

    return {

        "hidden_layer_sizes": random.choice(
            SEARCH_SPACE["hidden_layer_sizes"]
        ),

        "activation": random.choice(
            SEARCH_SPACE["activation"]
        ),

        "solver": "adam",    #optimizer algorithm

        "learning_rate_init": random.choice(
            SEARCH_SPACE["learning_rate_init"] 
        ),

        "alpha": random.choice(         
            SEARCH_SPACE["alpha"]       #penalize the weight(so The model is not relying too heavily on certain features) 
                                        #to reduce overfitting
        ),

        "batch_size": random.choice(
            SEARCH_SPACE["batch_size"]   
        )

    }

# =====================================================
# INITIAL POPULATION
# =====================================================

population = [

    create_solution()

    for _ in range(POPULATION_SIZE)

]
# =====================================================
# FITNESS FUNCTION
# =====================================================
# Evaluate one LEA solution using 5-fold cross-validation.
# The weighted F1-score is used as the fitness value.

def evaluate_solution(solution):

    model = build_mlp(solution)

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
# RUN LEA OPTIMIZATION
# =====================================================

start = time.time()

best_solution = None
best_score = -np.inf

for iteration in range(MAX_ITERATIONS):

    print(f"\nLEA Iteration {iteration + 1}/{MAX_ITERATIONS}")

    fitness_scores = []

    for solution in population:

        score = evaluate_solution(solution)

        fitness_scores.append(score)

        if score > best_score:

            best_score = score
            best_solution = solution.copy()

    # -------------------------------------------------
    # Sort population by fitness
    # -------------------------------------------------

    # Pair each candidate solution (hyperparameter combination) with its fitness score
    # (mean weighted F1-score obtained from 5-fold cross-validation) and sort them in descending order.

    ranked_population = [

        solution

        for _, solution in sorted(

            zip(fitness_scores, population),

            key=lambda x: x[0],

            reverse=True

        )

    ]

    # -------------------------------------------------
    # Keep the best half
    # -------------------------------------------------

    survivors = ranked_population[:POPULATION_SIZE // 2]

    # -------------------------------------------------
    # Generate new offspring
    # -------------------------------------------------

    # Randomly select one of the surviving candidate solutions (hyperparameter combinations)
    # and randomly modify one of its hyperparameters to create a new candidate 
    # until the population size is restored.

    new_population = survivors.copy()

    while len(new_population) < POPULATION_SIZE:

        parent = random.choice(survivors).copy()

        parameter = random.choice(list(parent.keys()))

        if parameter != "solver":

            parent[parameter] = random.choice(
                SEARCH_SPACE[parameter]
            )

        new_population.append(parent)

    population = new_population

end = time.time()

lea_runtime = end - start

print("\nLEA completed.")
print(f"Training Time: {lea_runtime:.2f} seconds")

print("\nBest Hyperparameters")
print(best_solution)

print("\nBest Cross Validation F1 Score")
print(best_score)
# =====================================================
# BUILD OPTIMIZED MLP MODEL
# =====================================================

lea_model = build_mlp(best_solution)

# =====================================================
# TRAIN OPTIMIZED MODEL
# =====================================================

lea_model.fit(X_train, y_train)

print("\nLEA Optimized MLP training completed.")
# =====================================================
# PREDICT TEST DATA
# =====================================================

lea_accuracy, \
lea_precision, \
lea_recall, \
lea_f1 = evaluate_model(

    lea_model,

    X_test,

    y_test,

    "LEA"

)
plt.figure(figsize=(8,5))

plt.plot(lea_model.loss_curve_)

plt.xlabel("Iteration")

plt.ylabel("Training Loss")

plt.title("LEA Optimized MLP Training Loss")

plt.grid(True)

plt.show()

# =====================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# =====================================================
#
# TLBO is a population-based metaheuristic optimization
# algorithm inspired by the learning process in a classroom.
# Each learner represents one candidate MLP hyperparameter
# configuration, and its fitness is measured using the mean
# weighted F1-score obtained from 5-fold cross-validation.
#
# TLBO improves candidate solutions through two phases:
#
# 1. Teacher Phase:
#    The candidate with the highest fitness score is identified
#    as the teacher. Other candidate solutions are updated toward
#    the teacher by adopting selected hyperparameter values from
#    the best-performing configuration while retaining parts of
#    their existing configuration.
#
# 2. Learner Phase:
#    Candidate solutions interact with other randomly selected
#    learners. Their fitness scores are compared, and the weaker
#    learner is guided toward the better-performing learner by
#    adopting selected hyperparameter values from that solution.
#
# The Teacher and Learner phases are repeated for every iteration,
# allowing the population to progressively move toward promising
# regions of the hyperparameter search space.
#
# -----------------------------------------------------
# LIMITATIONS OF TLBO
# -----------------------------------------------------
#
# Since learners are repeatedly guided toward better-performing
# solutions, multiple candidates may gradually become similar or
# identical. This can reduce population diversity and may cause
# repeated evaluation of similar hyperparameter configurations.
#
# Strong convergence toward the current teacher can also result
# in premature convergence if the teacher represents a locally
# good solution rather than the globally best solution.
#
# -----------------------------------------------------
# WHY TLBO WAS SELECTED
# -----------------------------------------------------
#
# TLBO was selected as the social/learning-based optimization
# method because it provides guided population-based search with
# relatively few algorithm-specific control parameters.
#
# Compared with alternatives such as PSO, ACO, or GWO, TLBO does
# not require additional parameters such as inertia weights,
# velocity coefficients, pheromone settings, or multiple social
# hierarchy coefficients. This makes the method easier to adapt
# and compare under a common experimental setup.
#
# TLBO can also be adapted to the discrete and categorical
# hyperparameters used by the MLP, such as hidden-layer
# structures, activation functions, learning rates, regularization
# strengths, and batch sizes.
#
# -----------------------------------------------------
# DIFFERENCE FROM OTHER OPTIMIZATION METHODS
# -----------------------------------------------------
#
# Random Search independently samples hyperparameter combinations
# without using information obtained from previous evaluations.
#
# Bayesian Optimization uses previous evaluation results to build
# a probabilistic model and predict promising hyperparameter
# configurations for subsequent evaluations.
#
# LEA uses the fitness score to retain high-performing candidate
# solutions, discards weaker candidates, and generates new
# solutions through random single-hyperparameter mutation.
#
# TLBO instead uses the fitness information to guide candidate
# solutions toward the best-performing teacher and toward better
# peer solutions through Teacher and Learner phases.
#
# Hybrid optimization methods combine complementary search
# strategies (for example, Bayesian guidance with evolutionary
# refinement) to balance exploration and exploitation.
#
# Therefore, TLBO is not assumed to be universally better than
# Bayesian Optimization, LEA, or hybrid approaches. Its advantage
# is that it provides a distinct guided social-learning search
# strategy with few control parameters. Its effectiveness is
# determined experimentally by comparing predictive performance,
# computational time, convergence behavior, and robustness under
# the same dataset and evaluation conditions.
# =====================================================

# =====================================================
# TLBO PARAMETERS
# =====================================================

# Number of candidate hyperparameter configurations
POPULATION_SIZE = 20

# Number of optimization iterations
MAX_ITERATIONS = 20

# =====================================================
# CREATE RANDOM TLBO SOLUTION
# =====================================================

def create_tlbo_solution():

    return {

        "hidden_layer_sizes": random.choice(
            SEARCH_SPACE["hidden_layer_sizes"]
        ),

        "activation": random.choice(
            SEARCH_SPACE["activation"]
        ),

        "solver": "adam",

        "learning_rate_init": random.choice(
            SEARCH_SPACE["learning_rate_init"]
        ),

        "alpha": random.choice(
            SEARCH_SPACE["alpha"]
        ),

        "batch_size": random.choice(
            SEARCH_SPACE["batch_size"]
        )

    }


# =====================================================
# INITIAL POPULATION
# =====================================================

population = [

    create_tlbo_solution()

    for _ in range(POPULATION_SIZE)

]

# =====================================================
# FITNESS FUNCTION
# =====================================================
# Evaluate one TLBO solution using 5-fold
# cross-validation.
#
# The weighted F1-score is used as the fitness value.
# =====================================================

def evaluate_tlbo_solution(solution):

    model = build_mlp(solution)

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
# RUN TLBO OPTIMIZATION
# =====================================================

start = time.time()

best_solution = None
best_score = -np.inf

for iteration in range(MAX_ITERATIONS):

    print(f"\nTLBO Iteration {iteration + 1}/{MAX_ITERATIONS}")

    # -------------------------------------------------
    # Evaluate Population
    # -------------------------------------------------

    fitness_scores = []

    for solution in population:

        score = evaluate_tlbo_solution(solution)

        fitness_scores.append(score)

        if score > best_score:

            best_score = score

            best_solution = solution.copy()

    # -------------------------------------------------
    # Identify Teacher
    # -------------------------------------------------

    teacher_index = np.argmax(fitness_scores)

    teacher = population[teacher_index].copy()

    teacher_score = fitness_scores[teacher_index]

    # -------------------------------------------------
    # Teacher Phase
    # -------------------------------------------------

    for i in range(POPULATION_SIZE):

        if i == teacher_index:
            continue

        learner = population[i].copy()

        parameter = random.choice(

            [k for k in learner.keys() if k != "solver"]

        )

        # Teacher attempts to improve learner

        learner[parameter] = teacher[parameter]

        learner_score = evaluate_tlbo_solution(learner)

        if learner_score > fitness_scores[i]:

            population[i] = learner

            fitness_scores[i] = learner_score

            if learner_score > best_score:

                best_score = learner_score

                best_solution = learner.copy()

    # -------------------------------------------------
    # Learner Phase
    # -------------------------------------------------

    for i in range(POPULATION_SIZE):

        partner_index = random.choice(

            [j for j in range(POPULATION_SIZE) if j != i]

        )

        learner = population[i].copy()

        partner = population[partner_index]

        parameter = random.choice(

            [k for k in learner.keys() if k != "solver"]

        )

        # -------------------------------------------------
        # Learn from better learner
        # -------------------------------------------------

        if fitness_scores[partner_index] > fitness_scores[i]:

            learner[parameter] = partner[parameter]

        else:

            learner[parameter] = random.choice(

                SEARCH_SPACE[parameter]

            )

        learner_score = evaluate_tlbo_solution(learner)

        if learner_score > fitness_scores[i]:

            population[i] = learner

            fitness_scores[i] = learner_score

            if learner_score > best_score:

                best_score = learner_score

                best_solution = learner.copy()

end = time.time()

tlbo_runtime = end - start

print("\nTLBO completed.")

print(f"Training Time: {tlbo_runtime:.2f} seconds")

print("\nBest Hyperparameters")

print(best_solution)

print("\nBest Cross Validation F1 Score")

print(best_score)

# =====================================================
# BUILD OPTIMIZED MLP MODEL
# =====================================================

print("\nBuilding TLBO Optimized MLP Model...")

tlbo_model = build_mlp(best_solution)

# =====================================================
# TRAIN FINAL MODEL
# =====================================================

print("\nTraining TLBO Optimized MLP...")

train_start = time.time()

tlbo_model.fit(X_train, y_train)

train_end = time.time()

tlbo_training_time = train_end - train_start

print(f"Training completed in {tlbo_training_time:.2f} seconds")

# =====================================================
# MAKE PREDICTIONS
# =====================================================

print("\nMaking Predictions...")

tlbo_predictions = tlbo_model.predict(X_test)

# =====================================================
# EVALUATE MODEL
# =====================================================

print("\nEvaluating TLBO Optimized Model...")

tlbo_accuracy = accuracy_score(
    y_test,
    tlbo_predictions
)

tlbo_precision = precision_score(
    y_test,
    tlbo_predictions,
    average="weighted",
    zero_division=0
)

tlbo_recall = recall_score(
    y_test,
    tlbo_predictions,
    average="weighted",
    zero_division=0
)

tlbo_f1 = f1_score(
    y_test,
    tlbo_predictions,
    average="weighted"
)

print("\n========== TLBO RESULTS ==========")

print(f"Accuracy : {tlbo_accuracy:.4f}")

print(f"Precision: {tlbo_precision:.4f}")

print(f"Recall   : {tlbo_recall:.4f}")

print(f"F1 Score : {tlbo_f1:.4f}")

print(f"Optimization Time : {tlbo_runtime:.2f} seconds")

print(f"Training Time     : {tlbo_training_time:.2f} seconds")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        tlbo_predictions,

        zero_division=0

    )

)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(

    y_test,

    tlbo_predictions

)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("TLBO Confusion Matrix")

plt.show()

# =====================================================
# TRAINING LOSS CURVE
# =====================================================

if hasattr(tlbo_model, "loss_curve_"):

    plt.figure(figsize=(8,5))

    plt.plot(

        tlbo_model.loss_curve_,

        linewidth=2

    )

    plt.xlabel("Iterations")

    plt.ylabel("Loss")

    plt.title("Training Loss Curve - TLBO Optimized MLP")

    plt.grid(True)

    plt.show()

# =====================================================
# DISPLAY BEST HYPERPARAMETERS
# =====================================================

print("\nBest Hyperparameters Found By TLBO\n")

for key, value in best_solution.items():

    print(f"{key}: {value}")


# =====================================================
# PERFORMANCE COMPARISON
# =====================================================
print("\nPERFORMANCE COMPARISON")
print("-" * 110)

print(f"{'Method':<24}"
      f"{'Accuracy':<12}"
      f"{'Precision':<12}"
      f"{'Recall':<12}"
      f"{'F1 Score':<12}"
      f"{'Time (s)'}")

print(f"{'Baseline MLP':<24}"
      f"{baseline_accuracy:<12.4f}"
      f"{baseline_precision:<12.4f}"
      f"{baseline_recall:<12.4f}"
      f"{baseline_f1:<12.4f}"
      f"{baseline_runtime:<10.2f}")

print(f"{'Random Search':<24}"
      f"{random_accuracy:<12.4f}"
      f"{random_precision:<12.4f}"
      f"{random_recall:<12.4f}"
      f"{random_f1:<12.4f}"
      f"{random_search_runtime:<10.2f}")

print(f"{'Bayesian Optimization':<24}"
      f"{bayes_accuracy:<12.4f}"
      f"{bayes_precision:<12.4f}"
      f"{bayes_recall:<12.4f}"
      f"{bayes_f1:<12.4f}"
      f"{optuna_runtime:<10.2f}")

print(f"{'Hyperband':<24}"
      f"{hyperband_accuracy:<12.4f}"
      f"{hyperband_precision:<12.4f}"
      f"{hyperband_recall:<12.4f}"
      f"{hyperband_f1:<12.4f}"
      f"{hyperband_runtime:<10.2f}")

print(f"{'LEA':<24}"
      f"{lea_accuracy:<12.4f}"
      f"{lea_precision:<12.4f}"
      f"{lea_recall:<12.4f}"
      f"{lea_f1:<12.4f}"
      f"{lea_runtime:<10.2f}")

print(f"{'TLBO':<24}"
      f"{tlbo_accuracy:<12.4f}"
      f"{tlbo_precision:<12.4f}"
      f"{tlbo_recall:<12.4f}"
      f"{tlbo_f1:<12.4f}"
      f"{tlbo_runtime:<10.2f}")
# =====================================================
# SAVE EXPERIMENT RESULTS
# =====================================================
# Store the performance of each optimization method.
# The results will be used later to compare all
# optimization techniques under identical settings.

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
        bayes_accuracy,
        hyperband_accuracy,
        lea_accuracy,
        tlbo_accuracy
    ],

    "Precision": [
        baseline_precision,
        random_precision,
        bayes_precision,
        hyperband_precision,
        lea_precision,
        tlbo_precision
    ],

    "Recall": [
        baseline_recall,
        random_recall,
        bayes_recall,
        hyperband_recall,
        lea_recall,
        tlbo_recall
    ],

    "F1 Score": [
        baseline_f1,
        random_f1,
        bayes_f1,
        hyperband_f1,
        lea_f1,
        tlbo_f1
    ],

    "Training Time (s)": [
        baseline_runtime,
        random_search_runtime,
        optuna_runtime,
        hyperband_runtime,
        lea_runtime,
        tlbo_runtime
    ]

})

comparison_results.to_csv(
    f"{RESULTS_DIR}/energy_efficiency_results.csv",
    index=False
)

print("\nResults saved successfully.")