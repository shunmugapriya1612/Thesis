# ============================================================
# PHASE 4 - ELECTRICITY DATASET
# XGBOOST HYPERPARAMETER OPTIMIZATION
# ============================================================
#
# OPTIMIZATION METHODS:
#
# 1. Random Search
# 2. Bayesian Optimization
# 3. Hyperband
# 4. Learner Evolution Algorithm (LEA)
# 5. Teaching-Learning-Based Optimization (TLBO)
#
# IMPORTANT:
# ------------------------------------------------------------
# Phase 3 already established the XGBoost baseline.
#
# Therefore, this file DOES NOT retrain the Phase 3 baseline.
#
# Phase 4 only:
#   - prepares the same data
#   - searches XGBoost hyperparameters
#   - trains the best configuration from each optimizer
#   - evaluates each final model on the same test set
#   - stores the results in Excel
#
# NO SMOTE
# NO SMOTENC
# NO CLASS WEIGHTING
#
# The Electricity dataset is reasonably balanced:
# DOWN = 57.55%
# UP   = 42.45%
#
# Therefore, this experiment focuses on
# HYPERPARAMETER OPTIMIZATION rather than imbalance handling.
# ============================================================


import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

RESULTS_DIR = "/Users/priya/Documents/Thesis/results"
VISUALIZATION_DIR = os.path.join(
    RESULTS_DIR,
    "phase4_electricity_xgboost_visualizations"
)

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ============================================================
# WARNING CONTROL
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

# ------------------------------------------------------------
# All optimization algorithms are intentionally limited to
# 5 iterations for this experimental phase.
# ------------------------------------------------------------

RANDOM_SEARCH_ITERATIONS = 5
BAYESIAN_ITERATIONS = 5
HYPERBAND_ITERATIONS = 5
LEA_ITERATIONS = 5
TLBO_ITERATIONS = 5


# ============================================================
# DATA / OUTPUT PATHS
# ============================================================

DATA_PATH = (
    "/Users/priya/Documents/Thesis/"
    "processed_datasets/electricity_processed.csv"
)

RESULTS_DIR = (
    "/Users/priya/Documents/Thesis/results"
)

EXCEL_PATH = (
    "/Users/priya/Documents/Thesis/results/"
    "electricity_xgboost_optimization_results.xlsx"
)


os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

print("\n" + "=" * 80)

print("PHASE 4 - ELECTRICITY XGBOOST OPTIMIZATION")

print("=" * 80)

print("\nConfiguration:")

print(
    f"Random State              : "
    f"{RANDOM_STATE}"
)

print(
    f"Cross-validation folds    : "
    f"{CV_FOLDS}"
)

print(
    f"Optimization Metric       : "
    f"Macro F1"
)

print(
    f"Random Search Iterations  : "
    f"{RANDOM_SEARCH_ITERATIONS}"
)

print(
    f"Bayesian Iterations       : "
    f"{BAYESIAN_ITERATIONS}"
)

print(
    f"Hyperband Iterations      : "
    f"{HYPERBAND_ITERATIONS}"
)

print(
    f"LEA Iterations            : "
    f"{LEA_ITERATIONS}"
)

print(
    f"TLBO Iterations           : "
    f"{TLBO_ITERATIONS}"
)

print(
    f"\nData File:\n{DATA_PATH}"
)

print(
    f"\nExcel Output:\n{EXCEL_PATH}"
)
# ============================================================
# LOAD ELECTRICITY DATASET
# ============================================================

print("\n" + "=" * 80)

print("LOADING ELECTRICITY DATASET")

print("=" * 80)


data = pd.read_csv(DATA_PATH)


print("\nDataset Shape:")
print(data.shape)


print("\nFirst Five Rows:")
print(data.head())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(
    columns=["Target"]
)

y = data["Target"]


print("\nFeature Matrix Shape:")
print(X.shape)


print("\nTarget Shape:")
print(y.shape)


# ============================================================
# TARGET ENCODING
# ============================================================
#
# Same encoding used in Phase 3:
#
# DOWN = 0
# UP   = 1
# ============================================================

target_mapping = {
    "DOWN": 0,
    "UP": 1
}


y_encoded = y.map(
    target_mapping
)


print("\nEncoded Target Distribution:")
print(
    y_encoded.value_counts()
)


print("\nEncoded Target Classes:")

print("0: DOWN")
print("1: UP")


# ============================================================
# FEATURE TYPE ANALYSIS
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCategorical Features:")

for feature in categorical_features:
    print("-", feature)


print("\nNumerical Features:")

for feature in numerical_features:
    print("-", feature)


# ============================================================
# ELECTRICITY DATASET SPECIFIC NOTE
# ============================================================
#
# In the processed Electricity dataset used in Phase 3,
# 'day' is already represented numerically.
#
# Therefore, no additional one-hot encoding is necessary.
#
# XGBoost can directly process the numerical feature matrix.
# ============================================================


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# We reproduce the SAME 80/20 stratified split used in Phase 3.
#
# This is critical for a fair comparison.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y_encoded

)


print("\nTraining Feature Shape:")
print(X_train.shape)


print("\nTesting Feature Shape:")
print(X_test.shape)


print("\nTraining Target Distribution:")
print(
    y_train.value_counts()
)


print("\nTesting Target Distribution:")
print(
    y_test.value_counts()
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================
#
# This simplifies the optimization routines.
# ============================================================

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values


print("\nData preparation completed.")

# ============================================================
# XGBOOST HYPERPARAMETER SEARCH SPACE
# ============================================================
#
# Every optimization algorithm uses this SAME search space.
#
# This ensures a fair comparison between:
#
# Random Search
# Bayesian Optimization
# Hyperband
# LEA
# TLBO
#
# ============================================================

PARAMETER_SPACE = {

    "n_estimators": (
        50,
        300
    ),

    "max_depth": (
        2,
        10
    ),

    "learning_rate": (
        0.01,
        0.30
    ),

    "subsample": (
        0.60,
        1.00
    ),

    "colsample_bytree": (
        0.60,
        1.00
    ),

    "min_child_weight": (
        1,
        10
    ),

    "gamma": (
        0.0,
        5.0
    ),

    "reg_alpha": (
        0.0,
        2.0
    ),

    "reg_lambda": (
        0.1,
        5.0
    )
}


# ============================================================
# PARAMETER NAMES
# ============================================================

PARAMETER_NAMES = list(
    PARAMETER_SPACE.keys()
)


print("\n" + "=" * 80)

print("XGBOOST HYPERPARAMETER SEARCH SPACE")

print("=" * 80)


for parameter, bounds in PARAMETER_SPACE.items():

    print(
        f"{parameter:20s}: "
        f"{bounds}"
    )

# ============================================================
# RANDOM PARAMETER GENERATION
# ============================================================

def generate_random_parameters(
    rng
):

    params = {

        "n_estimators": rng.randint(
            PARAMETER_SPACE["n_estimators"][0],
            PARAMETER_SPACE["n_estimators"][1] + 1
        ),

        "max_depth": rng.randint(
            PARAMETER_SPACE["max_depth"][0],
            PARAMETER_SPACE["max_depth"][1] + 1
        ),

        "learning_rate": rng.uniform(
            PARAMETER_SPACE["learning_rate"][0],
            PARAMETER_SPACE["learning_rate"][1]
        ),

        "subsample": rng.uniform(
            PARAMETER_SPACE["subsample"][0],
            PARAMETER_SPACE["subsample"][1]
        ),

        "colsample_bytree": rng.uniform(
            PARAMETER_SPACE["colsample_bytree"][0],
            PARAMETER_SPACE["colsample_bytree"][1]
        ),

        "min_child_weight": rng.randint(
            PARAMETER_SPACE["min_child_weight"][0],
            PARAMETER_SPACE["min_child_weight"][1] + 1
        ),

        "gamma": rng.uniform(
            PARAMETER_SPACE["gamma"][0],
            PARAMETER_SPACE["gamma"][1]
        ),

        "reg_alpha": rng.uniform(
            PARAMETER_SPACE["reg_alpha"][0],
            PARAMETER_SPACE["reg_alpha"][1]
        ),

        "reg_lambda": rng.uniform(
            PARAMETER_SPACE["reg_lambda"][0],
            PARAMETER_SPACE["reg_lambda"][1]
        )
    }

    return params


# ============================================================
# CLIP PARAMETERS TO VALID SEARCH SPACE
# ============================================================

def clip_parameters(
    params
):

    clipped = {}

    for parameter, value in params.items():

        lower, upper = PARAMETER_SPACE[
            parameter
        ]

        value = np.clip(
            value,
            lower,
            upper
        )

        # Integer parameters
        if parameter in [
            "n_estimators",
            "max_depth",
            "min_child_weight"
        ]:

            value = int(
                round(value)
            )

        clipped[parameter] = value

    return clipped


# ============================================================
# BUILD XGBOOST MODEL
# ============================================================

def build_xgboost(
    params
):

    model = XGBClassifier(

        n_estimators=int(
            params["n_estimators"]
        ),

        max_depth=int(
            params["max_depth"]
        ),

        learning_rate=float(
            params["learning_rate"]
        ),

        subsample=float(
            params["subsample"]
        ),

        colsample_bytree=float(
            params["colsample_bytree"]
        ),

        min_child_weight=int(
            params["min_child_weight"]
        ),

        gamma=float(
            params["gamma"]
        ),

        reg_alpha=float(
            params["reg_alpha"]
        ),

        reg_lambda=float(
            params["reg_lambda"]
        ),

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    return model


# ============================================================
# CROSS-VALIDATION EVALUATION
# ============================================================

def evaluate_cv(
    params
):

    params = clip_parameters(
        params
    )

    model = build_xgboost(
        params
    )

    cv = StratifiedKFold(

        n_splits=CV_FOLDS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring="f1_macro",

        n_jobs=-1
    )

    return float(
        np.mean(scores)
    )


# ============================================================
# FINAL TEST SET EVALUATION
# ============================================================

def evaluate_final_model(
    params,
    method_name
):

    params = clip_parameters(
        params
    )

    model = build_xgboost(
        params
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

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

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 80)

    print(
        f"{method_name.upper()} FINAL TEST RESULTS"
    )

    print("=" * 80)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Training Time      : "
        f"{training_time:.2f} seconds"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "DOWN",
                "UP"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    result = {

        "Method": method_name,

        "Accuracy": accuracy,

        "Weighted Precision": precision,

        "Weighted Recall": recall,

        "Weighted F1": weighted_f1,

        "Macro F1": macro_f1,

        "Training Time": training_time

    }

    return (
        result,
        model,
        predictions
    )

# ============================================================
# RANDOM SEARCH
# ============================================================
#
# RANDOM SEARCH:
#
# Randomly samples hyperparameter configurations from the
# predefined search space.
#
# Unlike Bayesian optimization, Random Search does not learn
# from previous trials.
#
# Each configuration is independently sampled.
# ============================================================

def random_search():

    print("\n" + "=" * 80)

    print("RANDOM SEARCH")

    print("=" * 80)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    best_score = -np.inf

    best_params = None

    history = []

    start_time = time.time()


    for iteration in range(
        1,
        RANDOM_SEARCH_ITERATIONS + 1
    ):

        params = generate_random_parameters(
            rng
        )

        score = evaluate_cv(
            params
        )

        history.append({

            "Iteration": iteration,

            "Macro F1": score,

            **params

        })

        print(
            f"Iteration {iteration}/"
            f"{RANDOM_SEARCH_ITERATIONS}"
            f" | Macro F1 = {score:.4f}"
        )


        if score > best_score:

            best_score = score

            best_params = params.copy()


    optimization_time = (
        time.time() - start_time
    )


    return (
        best_params,
        best_score,
        optimization_time,
        pd.DataFrame(history)
    )

# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================
#
# Bayesian Optimization differs from Random Search because
# it uses information from previous evaluations to decide
# where to search next.
#
# The optimizer builds a surrogate model of the relationship
# between hyperparameters and Macro F1.
#
# It then selects promising configurations using an
# acquisition strategy.
# ============================================================

def bayesian_optimization():

    print("\n" + "=" * 80)

    print("BAYESIAN OPTIMIZATION")

    print("=" * 80)


    try:

        from skopt import Optimizer

        from skopt.space import (
            Integer,
            Real
        )

    except ImportError:

        raise ImportError(

            "scikit-optimize is required for "
            "Bayesian Optimization.\n"
            "Install using:\n"
            "pip install scikit-optimize"

        )


    dimensions = [

        Integer(
            PARAMETER_SPACE["n_estimators"][0],
            PARAMETER_SPACE["n_estimators"][1],
            name="n_estimators"
        ),

        Integer(
            PARAMETER_SPACE["max_depth"][0],
            PARAMETER_SPACE["max_depth"][1],
            name="max_depth"
        ),

        Real(
            PARAMETER_SPACE["learning_rate"][0],
            PARAMETER_SPACE["learning_rate"][1],
            name="learning_rate"
        ),

        Real(
            PARAMETER_SPACE["subsample"][0],
            PARAMETER_SPACE["subsample"][1],
            name="subsample"
        ),

        Real(
            PARAMETER_SPACE["colsample_bytree"][0],
            PARAMETER_SPACE["colsample_bytree"][1],
            name="colsample_bytree"
        ),

        Integer(
            PARAMETER_SPACE["min_child_weight"][0],
            PARAMETER_SPACE["min_child_weight"][1],
            name="min_child_weight"
        ),

        Real(
            PARAMETER_SPACE["gamma"][0],
            PARAMETER_SPACE["gamma"][1],
            name="gamma"
        ),

        Real(
            PARAMETER_SPACE["reg_alpha"][0],
            PARAMETER_SPACE["reg_alpha"][1],
            name="reg_alpha"
        ),

        Real(
            PARAMETER_SPACE["reg_lambda"][0],
            PARAMETER_SPACE["reg_lambda"][1],
            name="reg_lambda"
        )
    ]


    optimizer = Optimizer(

        dimensions,

        random_state=RANDOM_STATE,

        base_estimator="GP",

        acq_func="EI"
    )


    best_score = -np.inf

    best_params = None

    history = []

    start_time = time.time()


    for iteration in range(
        1,
        BAYESIAN_ITERATIONS + 1
    ):

        suggested = optimizer.ask()


        params = dict(
            zip(
                PARAMETER_NAMES,
                suggested
            )
        )


        score = evaluate_cv(
            params
        )


        # skopt minimizes the objective.
        # Therefore Macro F1 is negated.
        optimizer.tell(
            suggested,
            -score
        )


        history.append({

            "Iteration": iteration,

            "Macro F1": score,

            **params

        })


        print(
            f"Iteration {iteration}/"
            f"{BAYESIAN_ITERATIONS}"
            f" | Macro F1 = {score:.4f}"
        )


        if score > best_score:

            best_score = score

            best_params = params.copy()


    optimization_time = (
        time.time() - start_time
    )


    return (
        best_params,
        best_score,
        optimization_time,
        pd.DataFrame(history)
    )

# ============================================================
# HYPERBAND
# ============================================================
#
# Hyperband is different from Random Search and Bayesian
# Optimization because it explicitly allocates computational
# resources to configurations.
#
# For XGBoost:
#
#     n_estimators = boosting rounds
#
# is used as the resource.
#
# Configurations with poor Macro F1 are eliminated early,
# while promising configurations receive more boosting rounds.
#
# This is particularly useful when model training becomes
# expensive.
# ============================================================

def hyperband():

    print("\n" + "=" * 80)

    print("HYPERBAND")

    print("=" * 80)


    rng = np.random.RandomState(
        RANDOM_STATE
    )


    # --------------------------------------------------------
    # Generate initial candidate configurations
    # --------------------------------------------------------

    candidates = []


    for i in range(
        HYPERBAND_ITERATIONS
    ):

        params = generate_random_parameters(
            rng
        )

        # Start with a relatively small number
        # of boosting rounds.

        params["n_estimators"] = 50

        candidates.append(
            params
        )


    history = []

    start_time = time.time()


    # --------------------------------------------------------
    # Resource levels
    # --------------------------------------------------------

    resource_levels = [
        50,
        100,
        200
    ]


    for round_number, resource in enumerate(
        resource_levels,
        start=1
    ):

        print(
            f"\nHyperband Round "
            f"{round_number}"
        )

        print(
            f"Boosting Rounds: "
            f"{resource}"
        )


        scores = []


        for candidate_number, params in enumerate(
            candidates,
            start=1
        ):

            current_params = params.copy()

            current_params[
                "n_estimators"
            ] = resource


            score = evaluate_cv(
                current_params
            )


            scores.append(
                (
                    score,
                    current_params
                )
            )


            history.append({

                "Round": round_number,

                "Candidate": candidate_number,

                "n_estimators": resource,

                "Macro F1": score,

                **{
                    key: value
                    for key, value
                    in current_params.items()
                    if key != "n_estimators"
                }

            })


            print(
                f"Candidate "
                f"{candidate_number}"
                f" | Macro F1 = "
                f"{score:.4f}"
            )


        # ----------------------------------------------------
        # Rank configurations
        # ----------------------------------------------------

        scores.sort(
            key=lambda x: x[0],
            reverse=True
        )


        # ----------------------------------------------------
        # Keep the strongest half.
        # ----------------------------------------------------

        keep_count = max(
            1,
            int(
                np.ceil(
                    len(scores) / 2
                )
            )
        )


        candidates = [

            params

            for score, params

            in scores[:keep_count]

        ]


        print(
            f"Candidates retained: "
            f"{len(candidates)}"
        )


    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    final_scores = []


    for params in candidates:

        score = evaluate_cv(
            params
        )

        final_scores.append(
            (
                score,
                params
            )
        )


    final_scores.sort(
        key=lambda x: x[0],
        reverse=True
    )


    best_score, best_params = (
        final_scores[0]
    )


    optimization_time = (
        time.time() - start_time
    )


    return (
        best_params,
        best_score,
        optimization_time,
        pd.DataFrame(history)
    )

# ============================================================
# LEARNER EVOLUTION ALGORITHM (LEA)
# ============================================================
#
# LEA treats each hyperparameter configuration as a learner.
#
# Better-performing learners influence the evolution of
# weaker learners.
#
# Therefore, unlike Random Search, LEA does not simply
# generate completely independent configurations.
#
# The population gradually moves toward better regions
# of the search space.
# ============================================================

def lea_optimization():

    print("\n" + "=" * 80)

    print("LEARNER EVOLUTION ALGORITHM (LEA)")

    print("=" * 80)


    rng = np.random.RandomState(
        RANDOM_STATE
    )


    population_size = 5


    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = [

        generate_random_parameters(
            rng
        )

        for _ in range(
            population_size
        )

    ]


    best_score = -np.inf

    best_params = None

    history = []

    start_time = time.time()


    # --------------------------------------------------------
    # Evolution iterations
    # --------------------------------------------------------

    for iteration in range(
        1,
        LEA_ITERATIONS + 1
    ):

        scored_population = []


        for learner_id, learner in enumerate(
            population,
            start=1
        ):

            score = evaluate_cv(
                learner
            )


            scored_population.append(
                (
                    score,
                    learner.copy()
                )
            )


            history.append({

                "Iteration": iteration,

                "Learner": learner_id,

                "Macro F1": score,

                **learner

            })


        # ----------------------------------------------------
        # Sort learners
        # ----------------------------------------------------

        scored_population.sort(
            key=lambda x: x[0],
            reverse=True
        )


        # Best learner
        current_best_score = (
            scored_population[0][0]
        )

        current_best_params = (
            scored_population[0][1]
        )


        if current_best_score > best_score:

            best_score = (
                current_best_score
            )

            best_params = (
                current_best_params.copy()
            )


        print(
            f"Iteration {iteration}/"
            f"{LEA_ITERATIONS}"
            f" | Best Macro F1 = "
            f"{current_best_score:.4f}"
        )


        # ----------------------------------------------------
        # Generate next population
        # ----------------------------------------------------
        #
        # The strongest learner acts as the reference.
        #
        # New learners are generated by moving current
        # learners toward the stronger learner.
        # ----------------------------------------------------

        new_population = [

            current_best_params.copy()

        ]


        while len(new_population) < population_size:

            parent = scored_population[
                rng.randint(
                    len(scored_population)
                )
            ][1]


            child = {}


            for parameter in PARAMETER_NAMES:

                lower, upper = (
                    PARAMETER_SPACE[
                        parameter
                    ]
                )


                # Differential-style learner
                # movement toward the best learner.

                best_value = (
                    current_best_params[
                        parameter
                    ]
                )

                parent_value = (
                    parent[
                        parameter
                    ]
                )


                value = (
                    parent_value
                    +
                    rng.uniform(
                        0,
                        1
                    )
                    *
                    (
                        best_value
                        -
                        parent_value
                    )
                )


                # Small mutation component

                if rng.rand() < 0.30:

                    value += (
                        rng.normal(
                            0,
                            0.10
                            *
                            (
                                upper
                                -
                                lower
                            )
                        )
                    )


                child[
                    parameter
                ] = value


            child = clip_parameters(
                child
            )


            new_population.append(
                child
            )


        population = new_population


    optimization_time = (
        time.time() - start_time
    )


    return (
        best_params,
        best_score,
        optimization_time,
        pd.DataFrame(history)
    )

# ============================================================
# TEACHING-LEARNING-BASED OPTIMIZATION (TLBO)
# ============================================================
#
# TLBO consists of two major phases:
#
# 1. TEACHER PHASE
#
# The best solution acts as the teacher and attempts to
# move the population toward better values.
#
# 2. LEARNER PHASE
#
# Learners interact with one another and learn from a
# better-performing learner.
#
# This makes TLBO fundamentally different from:
#
# Random Search:
#     independent random sampling
#
# Bayesian Optimization:
#     surrogate-model-guided search
#
# LEA:
#     evolutionary movement toward better learners
#
# ============================================================

def tlbo_optimization():

    print("\n" + "=" * 80)

    print(
        "TEACHING-LEARNING-BASED "
        "OPTIMIZATION (TLBO)"
    )

    print("=" * 80)


    rng = np.random.RandomState(
        RANDOM_STATE
    )


    population_size = 5


    population = [

        generate_random_parameters(
            rng
        )

        for _ in range(
            population_size
        )

    ]


    best_score = -np.inf

    best_params = None

    history = []

    start_time = time.time()


    for iteration in range(
        1,
        TLBO_ITERATIONS + 1
    ):


        # ====================================================
        # EVALUATE CURRENT POPULATION
        # ====================================================

        scores = []


        for learner_id, learner in enumerate(
            population,
            start=1
        ):

            score = evaluate_cv(
                learner
            )


            scores.append(
                score
            )


            history.append({

                "Iteration": iteration,

                "Learner": learner_id,

                "Macro F1": score,

                **learner

            })


        # ====================================================
        # IDENTIFY TEACHER
        # ====================================================

        teacher_index = int(
            np.argmax(scores)
        )


        teacher = population[
            teacher_index
        ].copy()


        teacher_score = scores[
            teacher_index
        ]


        if teacher_score > best_score:

            best_score = (
                teacher_score
            )

            best_params = (
                teacher.copy()
            )


        print(
            f"Iteration {iteration}/"
            f"{TLBO_ITERATIONS}"
            f" | Teacher Macro F1 = "
            f"{teacher_score:.4f}"
        )


        # ====================================================
        # TEACHER PHASE
        # ====================================================

        new_population = []


        for i, learner in enumerate(
            population
        ):

            new_learner = learner.copy()


            mean_values = {}


            for parameter in PARAMETER_NAMES:

                values = [

                    p[parameter]

                    for p in population

                ]


                mean_values[
                    parameter
                ] = np.mean(
                    values
                )


            # Teaching factor
            # TLBO commonly uses TF ∈ {1, 2}

            teaching_factor = rng.randint(
                1,
                3
            )


            for parameter in PARAMETER_NAMES:

                lower, upper = (
                    PARAMETER_SPACE[
                        parameter
                    ]
                )


                current_value = (
                    learner[
                        parameter
                    ]
                )


                teacher_value = (
                    teacher[
                        parameter
                    ]
                )


                mean_value = (
                    mean_values[
                        parameter
                    ]
                )


                new_value = (

                    current_value

                    +

                    rng.rand()

                    *

                    (
                        teacher_value
                        -
                        teaching_factor
                        *
                        mean_value
                    )

                )


                new_learner[
                    parameter
                ] = new_value


            new_learner = clip_parameters(
                new_learner
            )


            new_score = evaluate_cv(
                new_learner
            )


            if new_score > scores[i]:

                new_population.append(
                    new_learner
                )

            else:

                new_population.append(
                    learner
                )


        population = new_population


        # ====================================================
        # LEARNER PHASE
        # ====================================================

        for i in range(
            population_size
        ):

            j = rng.randint(
                population_size
            )


            while j == i:

                j = rng.randint(
                    population_size
                )


            learner_i = population[i]

            learner_j = population[j]


            score_i = evaluate_cv(
                learner_i
            )

            score_j = evaluate_cv(
                learner_j
            )


            new_learner = (
                learner_i.copy()
            )


            for parameter in PARAMETER_NAMES:

                lower, upper = (
                    PARAMETER_SPACE[
                        parameter
                    ]
                )


                value_i = (
                    learner_i[
                        parameter
                    ]
                )

                value_j = (
                    learner_j[
                        parameter
                    ]
                )


                if score_i < score_j:

                    new_value = (

                        value_i

                        +

                        rng.rand()

                        *

                        (
                            value_j
                            -
                            value_i
                        )

                    )

                else:

                    new_value = (

                        value_i

                        +

                        rng.rand()

                        *

                        (
                            value_i
                            -
                            value_j
                        )

                    )


                new_learner[
                    parameter
                ] = new_value


            new_learner = clip_parameters(
                new_learner
            )


            new_score = evaluate_cv(
                new_learner
            )


            if new_score > score_i:

                population[i] = (
                    new_learner
                )


    optimization_time = (
        time.time() - start_time
    )


    return (
        best_params,
        best_score,
        optimization_time,
        pd.DataFrame(history)
    )

# ============================================================
# RUN OPTIMIZATION ALGORITHMS
# ============================================================

optimization_results = {}

optimization_histories = {}


# ============================================================
# RANDOM SEARCH
# ============================================================

random_best_params, \
random_best_score, \
random_time, \
random_history = random_search()


optimization_results[
    "Random Search"
] = {

    "best_params": random_best_params,

    "best_cv_score": random_best_score,

    "optimization_time": random_time

}


optimization_histories[
    "Random Search"
] = random_history


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

bayes_best_params, \
bayes_best_score, \
bayes_time, \
bayes_history = bayesian_optimization()


optimization_results[
    "Bayesian Optimization"
] = {

    "best_params": bayes_best_params,

    "best_cv_score": bayes_best_score,

    "optimization_time": bayes_time

}


optimization_histories[
    "Bayesian Optimization"
] = bayes_history


# ============================================================
# HYPERBAND
# ============================================================

hyperband_best_params, \
hyperband_best_score, \
hyperband_time, \
hyperband_history = hyperband()


optimization_results[
    "Hyperband"
] = {

    "best_params": hyperband_best_params,

    "best_cv_score": hyperband_best_score,

    "optimization_time": hyperband_time

}


optimization_histories[
    "Hyperband"
] = hyperband_history


# ============================================================
# LEA
# ============================================================

lea_best_params, \
lea_best_score, \
lea_time, \
lea_history = lea_optimization()


optimization_results[
    "LEA"
] = {

    "best_params": lea_best_params,

    "best_cv_score": lea_best_score,

    "optimization_time": lea_time

}


optimization_histories[
    "LEA"
] = lea_history


# ============================================================
# TLBO
# ============================================================

tlbo_best_params, \
tlbo_best_score, \
tlbo_time, \
tlbo_history = tlbo_optimization()


optimization_results[
    "TLBO"
] = {

    "best_params": tlbo_best_params,

    "best_cv_score": tlbo_best_score,

    "optimization_time": tlbo_time

}


optimization_histories[
    "TLBO"
] = tlbo_history


print("\n" + "=" * 80)

print("ALL OPTIMIZATION ALGORITHMS COMPLETED")

print("=" * 80)
# ============================================================
# FINAL MODEL TRAINING AND TEST EVALUATION
# ============================================================
#
# IMPORTANT:
#
# The test set has NOT been used during optimization.
#
# Each optimizer selected its best configuration using
# 5-fold CV on the training data.
#
# Only now do we evaluate the selected configuration
# against the untouched test set.
# ============================================================

final_results = []

final_models = {}

final_predictions = {}


for method_name, result in optimization_results.items():

    print("\n" + "=" * 80)

    print(
        f"TRAINING BEST {method_name.upper()} MODEL"
    )

    print("=" * 80)


    best_params = result[
        "best_params"
    ]


    best_cv_score = result[
        "best_cv_score"
    ]


    optimization_time = result[
        "optimization_time"
    ]


    print("\nBest Hyperparameters:")


    for parameter, value in best_params.items():

        print(
            f"{parameter:20s}: "
            f"{value}"
        )


    print(
        f"\nBest CV Macro F1: "
        f"{best_cv_score:.4f}"
    )


    print(
        f"Optimization Time: "
        f"{optimization_time:.2f} seconds"
    )


    evaluation, model, predictions = (
        evaluate_final_model(
            best_params,
            method_name
        )
    )


    evaluation[
        "Best CV Macro F1"
    ] = best_cv_score


    evaluation[
        "Optimization Time"
    ] = optimization_time


    # Add best parameters

    for parameter, value in best_params.items():

        evaluation[
            parameter
        ] = value


    final_results.append(
        evaluation
    )


    final_models[
        method_name
    ] = model


    final_predictions[
        method_name
    ] = predictions


final_results_df = pd.DataFrame(
    final_results
)
# ============================================================
# PHASE 3 XGBOOST BASELINE REFERENCE
# ============================================================
#
# The Phase 3 baseline was already trained and evaluated.
#
# We DO NOT retrain it here.
#
# These values are copied from the completed Phase 3
# Electricity XGBoost experiment.
# ============================================================

phase3_baseline = {

    "Method": "Phase 3 XGBoost Baseline",

    "Accuracy": 0.8740,

    "Weighted Precision": 0.8738,

    "Weighted Recall": 0.8740,

    "Weighted F1": 0.8738,

    "Macro F1": 0.8706,

    "Training Time": 0.21,

    "Best CV Macro F1": np.nan,

    "Optimization Time": 0.0,

    "n_estimators": 100,

    "max_depth": 6,

    "learning_rate": 0.1,

    "subsample": 1.0,

    "colsample_bytree": 1.0,

    "min_child_weight": 1,

    "gamma": 0.0,

    "reg_alpha": 0.0,

    "reg_lambda": 1.0
}


baseline_df = pd.DataFrame(
    [phase3_baseline]
)


# ============================================================
# COMBINED COMPARISON
# ============================================================

comparison_df = pd.concat(

    [
        baseline_df,
        final_results_df
    ],

    ignore_index=True

)


print("\n" + "=" * 80)

print("FINAL XGBOOST OPTIMIZATION COMPARISON")

print("=" * 80)


print(
    comparison_df[
        [
            "Method",
            "Accuracy",
            "Weighted F1",
            "Macro F1",
            "Training Time",
            "Optimization Time"
        ]
    ].to_string(
        index=False
    )
)
# ============================================================
# SAVE RESULTS TO EXCEL
# ============================================================
#
# Excel sheets:
#
# 1. Baseline
# 2. Random_Search
# 3. Bayesian
# 4. Hyperband
# 5. LEA
# 6. TLBO
# 7. Comparison
#
# Each optimizer sheet contains its trial history.
# ============================================================

print("\n" + "=" * 80)

print("SAVING RESULTS TO EXCEL")

print("=" * 80)


with pd.ExcelWriter(
    EXCEL_PATH,
    engine="openpyxl"
) as writer:


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_df.to_excel(

        writer,

        sheet_name="Baseline",

        index=False

    )


    # --------------------------------------------------------
    # RANDOM SEARCH
    # --------------------------------------------------------

    optimization_histories[
        "Random Search"
    ].to_excel(

        writer,

        sheet_name="Random_Search",

        index=False

    )


    # --------------------------------------------------------
    # BAYESIAN
    # --------------------------------------------------------

    optimization_histories[
        "Bayesian Optimization"
    ].to_excel(

        writer,

        sheet_name="Bayesian",

        index=False

    )


    # --------------------------------------------------------
    # HYPERBAND
    # --------------------------------------------------------

    optimization_histories[
        "Hyperband"
    ].to_excel(

        writer,

        sheet_name="Hyperband",

        index=False

    )


    # --------------------------------------------------------
    # LEA
    # --------------------------------------------------------

    optimization_histories[
        "LEA"
    ].to_excel(

        writer,

        sheet_name="LEA",

        index=False

    )


    # --------------------------------------------------------
    # TLBO
    # --------------------------------------------------------

    optimization_histories[
        "TLBO"
    ].to_excel(

        writer,

        sheet_name="TLBO",

        index=False

    )


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison_df.to_excel(

        writer,

        sheet_name="Comparison",

        index=False

    )


print(
    f"\nExcel file saved successfully:\n"
    f"{EXCEL_PATH}"
)


# ============================================================
# VISUALIZATION 1
# TARGET CLASS DISTRIBUTION
# ============================================================

target_counts = data["Target"].value_counts()

target_percentages = (
    data["Target"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    target_counts.index,
    target_counts.values
)

plt.xlabel(
    "Target Class"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Electricity Dataset - Target Class Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add count and percentage labels
for bar, count, percentage in zip(
    bars,
    target_counts.values,
    target_percentages.values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "01_target_class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# VISUALIZATION 2
# FEATURE-TARGET RELATIONSHIPS
# ============================================================

# Encode target temporarily for correlation analysis
target_encoded = (
    data["Target"]
    .map({
        "DOWN": 0,
        "UP": 1
    })
)

numerical_features = [
    "date",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer"
]

feature_target_data = data[
    numerical_features
].copy()

feature_target_data["Target"] = target_encoded

feature_target_corr = (
    feature_target_data
    .corr()["Target"]
    .drop("Target")
    .sort_values()
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_target_corr.index,
    feature_target_corr.values
)

plt.xlabel(
    "Correlation with Target"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Electricity Dataset - Feature-Target Relationships"
)

plt.axvline(
    0,
    linewidth=1
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "02_feature_target_relationships.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# VISUALIZATION 3
# FEATURE-FEATURE CORRELATION HEATMAP
# ============================================================

correlation_features = [
    "date",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer"
]

correlation_matrix = (
    data[correlation_features]
    .corr()
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title(
    "Electricity Dataset - Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "03_feature_correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# VISUALIZATION 4
# IMPORTANT FEATURE DISTRIBUTIONS BY TARGET
# ============================================================

important_features = [
    "nswprice",
    "nswdemand",
    "vicdemand",
    "period"
]

for feature in important_features:

    plt.figure(
        figsize=(8, 6)
    )

    sns.boxplot(
        data=data,
        x="Target",
        y=feature
    )

    plt.xlabel(
        "Target"
    )

    plt.ylabel(
        feature
    )

    plt.title(
        f"Electricity Dataset - {feature} Distribution by Target"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "04_feature_distributions_by_target.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# VISUALIZATIONS - ELECTRICITY XGBOOST OPTIMIZATION
# ============================================================
#
# Visualizations:
#
# 1. Performance Comparison
#    - Accuracy
#    - Weighted F1
#    - Macro F1
#
# 2. Optimization Time Comparison
#
# 3. Optimization Convergence
#
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# VISUALIZATION 1 - PERFORMANCE COMPARISON
# ============================================================

print("\nGenerating Performance Comparison...")

methods = comparison_df["Method"].tolist()

accuracy_values = comparison_df["Accuracy"].tolist()

weighted_f1_values = comparison_df["Weighted F1"].tolist()

macro_f1_values = comparison_df["Macro F1"].tolist()


x = np.arange(len(methods))

width = 0.25


plt.figure(figsize=(12, 6))


plt.bar(
    x - width,
    accuracy_values,
    width,
    label="Accuracy"
)


plt.bar(
    x,
    weighted_f1_values,
    width,
    label="Weighted F1"
)


plt.bar(
    x + width,
    macro_f1_values,
    width,
    label="Macro F1"
)


plt.xlabel("Method")

plt.ylabel("Score")

plt.title(
    "Electricity XGBoost - Optimization Performance Comparison"
)


plt.xticks(
    x,
    methods,
    rotation=30,
    ha="right"
)


plt.ylim(0, 1)


plt.legend()


plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "05_optimization_performance_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# VISUALIZATION 2 - OPTIMIZATION TIME COMPARISON
# ============================================================

print("\nGenerating Optimization Time Comparison...")


optimization_times = (
    comparison_df["Optimization Time"]
    .tolist()
)


plt.figure(figsize=(10, 6))


bars = plt.bar(
    methods,
    optimization_times
)


plt.xlabel("Method")

plt.ylabel("Optimization Time (seconds)")

plt.title(
    "Electricity XGBoost - Optimization Time Comparison"
)


plt.xticks(
    rotation=30,
    ha="right"
)


# Display time above each bar
for bar, value in zip(
    bars,
    optimization_times
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + (max(optimization_times) * 0.02),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "06_optimization_time_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# VISUALIZATION 3 - OPTIMIZATION CONVERGENCE
# ============================================================
#
# This visualization shows the Macro F1 obtained during
# the optimization process for each optimization algorithm.
#
# The histories use:
# - Random Search       -> Iteration + Macro F1
# - Bayesian            -> Iteration + Macro F1
# - Hyperband           -> Round + Macro F1
# - LEA                 -> Iteration + Macro F1
# - TLBO                -> Iteration + Macro F1
#
# ============================================================

print("\nGenerating Optimization Convergence Plot...")


plt.figure(figsize=(10, 6))


# ============================================================
# RANDOM SEARCH
# ============================================================

random_history = optimization_histories[
    "Random Search"
]


plt.plot(
    random_history["Iteration"],
    random_history["Macro F1"],
    marker="o",
    label="Random Search"
)


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

bayesian_history = optimization_histories[
    "Bayesian Optimization"
]


plt.plot(
    bayesian_history["Iteration"],
    bayesian_history["Macro F1"],
    marker="o",
    label="Bayesian Optimization"
)


# ============================================================
# HYPERBAND
# ============================================================

hyperband_history = optimization_histories[
    "Hyperband"
]


plt.plot(
    hyperband_history["Round"],
    hyperband_history["Macro F1"],
    marker="o",
    label="Hyperband"
)


# ============================================================
# LEA
# ============================================================

lea_history = optimization_histories[
    "LEA"
]


plt.plot(
    lea_history["Iteration"],
    lea_history["Macro F1"],
    marker="o",
    label="LEA"
)


# ============================================================
# TLBO
# ============================================================

tlbo_history = optimization_histories[
    "TLBO"
]


plt.plot(
    tlbo_history["Iteration"],
    tlbo_history["Macro F1"],
    marker="o",
    label="TLBO"
)


# ============================================================
# GRAPH FORMATTING
# ============================================================

plt.xlabel(
    "Iteration / Round"
)

plt.ylabel(
    "CV Macro F1"
)

plt.title(
    "Electricity XGBoost - Optimization Convergence"
)


plt.ylim(0, 1)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    os.path.join(VISUALIZATION_DIR, "07_optimization_convergence.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
print("\n")
print("=" * 80)
print("ELECTRICITY DATASET VISUALIZATIONS COMPLETED")
print("=" * 80)
# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)

print("PHASE 4 ELECTRICITY XGBOOST OPTIMIZATION SUMMARY")

print("=" * 80)


print("\n")

print(
    comparison_df[
        [
            "Method",
            "Accuracy",
            "Weighted F1",
            "Macro F1",
            "Training Time",
            "Optimization Time"
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)

print(
    "PHASE 4 ELECTRICITY XGBOOST "
    "OPTIMIZATION COMPLETED"
)

print("=" * 80)


print(
    f"\nResults saved to:\n"
    f"{EXCEL_PATH}"
)