# ============================================================
# PHASE 3 - CONTROLLED SMOTENC + RANDOM SEARCH
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This phase investigates whether hyperparameter optimization
# can improve the MLP model when a controlled SMOTENC strategy
# is used to address the severe class imbalance in the
# Solar Flare dataset.
#
#
# IMPORTANT METHODOLOGICAL CORRECTION
# ------------------------------------------------------------
# In the previous Phase 3 experiments, SMOTENC was applied
# before cross-validation.
#
# Although this allowed us to study the effect of SMOTENC,
# applying oversampling before cross-validation can introduce
# information leakage because synthetic samples may be related
# to observations that subsequently appear in validation folds.
#
# Therefore, in this phase:
#
#       Original Training Data
#                ↓
#          5-Fold CV
#                ↓
#       ┌────────────────┐
#       │ Training Fold  │
#       │       ↓        │
#       │ Controlled     │
#       │ SMOTENC        │
#       │       ↓        │
#       │ Encoding       │
#       │       ↓        │
#       │ Scaling        │
#       │       ↓        │
#       │ MLP            │
#       └───────┬────────┘
#               ↓
#        Validation Fold
#
# The validation fold is NEVER oversampled.
#
# The final test set is also NEVER used during
# hyperparameter optimization.
#
#
# OPTIMIZATION METHOD
# ------------------------------------------------------------
# Random Search
#
# Primary optimization metric:
#       Macro F1
#
# Macro F1 is selected because the Solar Flare dataset contains
# four highly imbalanced classes. It gives equal importance to
# each class instead of allowing the majority "No Flare" class
# to dominate the optimization objective.
#
# Additional evaluation metrics:
#
#       Accuracy
#       Weighted Precision
#       Weighted Recall
#       Weighted F1
#       Macro F1
#       Classification Report
#       Confusion Matrix
#
# ============================================================
# SOLAR FLARE CLASSIFICATION
#
# CONTROLLED SMOTENC
# RANDOM SEARCH
# BAYESIAN OPTIMIZATION
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTENC

from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer


warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

N_ITER = 20


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = "processed_datasets/solar_flare_processed.csv"

# ============================================================
# FEATURE INFORMATION
# ============================================================

CATEGORICAL_FEATURES = [
    "modified Zurich class",
    "largest spot size",
    "spot distribution"
]


TARGET_COLUMN = "Target"


# ============================================================
# CONTROLLED SMOTENC TARGET DISTRIBUTION
# ============================================================
#
# Original training distribution:
#
# Common Flare    : 100
# Moderate Flare  : 40
# No Flare        : 962
# Severe Flare    : 9
#
# Controlled SMOTENC:
#
# Common Flare    : 400
# Moderate Flare  : 250
# No Flare        : unchanged
# Severe Flare    : 100
#
# During cross-validation the majority class count changes
# slightly from fold to fold. Therefore the estimator applies
# these targets dynamically.
#
# ============================================================

CONTROLLED_TARGETS = {
    0: 400,       # Common Flare
    1: 250,       # Moderate Flare
    3: 100        # Severe Flare
}


# ============================================================
# HIDDEN LAYER ARCHITECTURE MAPPING
# ============================================================
#
# We use strings in Random Search and Bayesian Optimization.
#
# This avoids the tuple-related problems encountered with
# scikit-optimize.
#
# The strings are converted to actual MLP architectures
# inside ControlledSMOTENCMLP.
#
# ============================================================

HIDDEN_LAYER_MAP = {

    "16": (16,),

    "32": (32,),

    "64": (64,),

    "128": (128,),

    "32_16": (32, 16),

    "64_32": (64, 32),

    "128_64": (128, 64)

}


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Solar Flare Dataset...")

data = pd.read_csv(DATA_PATH)


print("\nDataset Shape:")
print(data.shape)


print("\nFirst Five Rows:")
print(data.head())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(
    columns=[TARGET_COLUMN]
)

y = data[TARGET_COLUMN]


print("\nFeature Matrix Shape:")
print(X.shape)


print("\nTarget Shape:")
print(y.shape)


# ============================================================
# ORIGINAL TARGET DISTRIBUTION
# ============================================================

print("\nOriginal Target Distribution:")
print(y.value_counts())


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


print("\nEncoded Target Distribution:")
print(
    pd.Series(y).value_counts().sort_index()
)


print("\nEncoded Target Classes:")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
#
# The test set is completely untouched.
#
# It is NOT used during:
#
# - SMOTENC
# - Cross-validation
# - Random Search
# - Bayesian Optimization
#
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    stratify=y,

    random_state=RANDOM_STATE

)


print("\nTraining Feature Shape:")
print(X_train.shape)


print("\nTesting Feature Shape:")
print(X_test.shape)


print("\nTraining Target Distribution:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)


print("\nTesting Target Distribution:")

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)


# ============================================================
# CUSTOM CONTROLLED SMOTENC + MLP CLASSIFIER
# ============================================================


class ControlledSMOTENCMLP(
    BaseEstimator,
    ClassifierMixin
):


    def __init__(

        self,

        hidden_layer_sizes="64",

        activation="relu",

        solver="adam",

        learning_rate_init=0.001,

        alpha=0.0001,

        batch_size=32,

        max_iter=1000,

        random_state=42

    ):

        self.hidden_layer_sizes = (
            hidden_layer_sizes
        )

        self.activation = activation

        self.solver = solver

        self.learning_rate_init = (
            learning_rate_init
        )

        self.alpha = alpha

        self.batch_size = batch_size

        self.max_iter = max_iter

        self.random_state = random_state


    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X, y):


        # ----------------------------------------------------
        # Convert X to DataFrame
        # ----------------------------------------------------

        X = X.copy()

        if not isinstance(X, pd.DataFrame):

            X = pd.DataFrame(X)


        # ----------------------------------------------------
        # Identify categorical and numerical columns
        # ----------------------------------------------------

        categorical_indices = [
            X.columns.get_loc(
                column
            )
            for column in CATEGORICAL_FEATURES
        ]


        numerical_features = [
            column
            for column in X.columns
            if column not in CATEGORICAL_FEATURES
        ]


        # ----------------------------------------------------
        # CATEGORICAL PREPROCESSING
        # ----------------------------------------------------

        self.categorical_imputer_ = (
            SimpleImputer(
                strategy="most_frequent"
            )
        )


        X_cat = (
            self.categorical_imputer_
            .fit_transform(
                X[CATEGORICAL_FEATURES]
            )
        )


        self.ordinal_encoder_ = (
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )


        X_cat = (
            self.ordinal_encoder_
            .fit_transform(X_cat)
        )


        # ----------------------------------------------------
        # NUMERICAL PREPROCESSING
        # ----------------------------------------------------

        self.numerical_imputer_ = (
            SimpleImputer(
                strategy="median"
            )
        )


        X_num = (
            self.numerical_imputer_
            .fit_transform(
                X[numerical_features]
            )
        )


        # ----------------------------------------------------
        # COMBINE DATA
        #
        # Categorical columns are deliberately placed first.
        #
        # Therefore SMOTENC categorical indices become:
        #
        # [0, 1, 2]
        #
        # ----------------------------------------------------

        X_prepared = np.column_stack(
            [
                X_cat,
                X_num
            ]
        )


        categorical_indices_smote = [
            0,
            1,
            2
        ]


        # ----------------------------------------------------
        # DETERMINE CURRENT CLASS COUNTS
        # ----------------------------------------------------

        current_counts = (
            pd.Series(y)
            .value_counts()
            .to_dict()
        )


        # ----------------------------------------------------
        # BUILD DYNAMIC CONTROLLED TARGET
        #
        # Only increase a class if it is currently below
        # the desired target.
        # ----------------------------------------------------

        sampling_strategy = {}


        for class_id, desired_count in (
            CONTROLLED_TARGETS.items()
        ):

            current_count = current_counts.get(
                class_id,
                0
            )


            if (
                current_count > 0
                and current_count < desired_count
            ):

                sampling_strategy[
                    class_id
                ] = desired_count


        # ----------------------------------------------------
        # APPLY CONTROLLED SMOTENC
        # ----------------------------------------------------

        if len(sampling_strategy) > 0:

            smote = SMOTENC(

                categorical_features=(
                    categorical_indices_smote
                ),

                sampling_strategy=(
                    sampling_strategy
                ),

                random_state=self.random_state,

                k_neighbors=5

            )


            X_resampled, y_resampled = (
                smote.fit_resample(
                    X_prepared,
                    y
                )
            )

        else:

            X_resampled = X_prepared

            y_resampled = y


        # ----------------------------------------------------
        # ONE-HOT ENCODE CATEGORICAL FEATURES
        # ----------------------------------------------------

        self.onehot_encoder_ = (
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )


        X_cat_resampled = (
            X_resampled[:, :3]
        )


        X_num_resampled = (
            X_resampled[:, 3:]
        )


        X_cat_encoded = (
            self.onehot_encoder_
            .fit_transform(
                X_cat_resampled
            )
        )


        # ----------------------------------------------------
        # COMBINE ENCODED CATEGORICAL + NUMERICAL
        # ----------------------------------------------------

        X_final = np.column_stack(

            [
                X_cat_encoded,
                X_num_resampled
            ]

        )


        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        self.scaler_ = StandardScaler()


        X_final = (
            self.scaler_
            .fit_transform(X_final)
        )


        # ----------------------------------------------------
        # SAVE FEATURE COUNT
        # ----------------------------------------------------

        self.n_features_in_ = (
            X_final.shape[1]
        )


        # ----------------------------------------------------
        # CONVERT HIDDEN LAYER STRING TO TUPLE
        # ----------------------------------------------------

        if isinstance(
            self.hidden_layer_sizes,
            str
        ):

            hidden_layers = (
                HIDDEN_LAYER_MAP[
                    self.hidden_layer_sizes
                ]
            )

        else:

            hidden_layers = (
                self.hidden_layer_sizes
            )


        # ----------------------------------------------------
        # BUILD MLP
        # ----------------------------------------------------

        self.model_ = MLPClassifier(

            hidden_layer_sizes=hidden_layers,

            activation=self.activation,

            solver=self.solver,

            learning_rate_init=(
                self.learning_rate_init
            ),

            alpha=self.alpha,

            batch_size=self.batch_size,

            max_iter=self.max_iter,

            random_state=self.random_state

        )


        # ----------------------------------------------------
        # TRAIN MLP
        # ----------------------------------------------------

        self.model_.fit(

            X_final,

            y_resampled

        )


        # ----------------------------------------------------
        # SAVE CLASSES
        # ----------------------------------------------------

        self.classes_ = (
            self.model_.classes_
        )


        return self


    # ========================================================
    # TRANSFORM TEST DATA
    # ========================================================

    def _transform(self, X):


        X = X.copy()


        if not isinstance(X, pd.DataFrame):

            X = pd.DataFrame(X)


        # ----------------------------------------------------
        # CATEGORICAL
        # ----------------------------------------------------

        X_cat = (
            self.categorical_imputer_
            .transform(
                X[CATEGORICAL_FEATURES]
            )
        )


        X_cat = (
            self.ordinal_encoder_
            .transform(X_cat)
        )


        # ----------------------------------------------------
        # NUMERICAL
        # ----------------------------------------------------

        X_num = (
            self.numerical_imputer_
            .transform(
                X[
                    [
                        column
                        for column in X.columns
                        if column
                        not in CATEGORICAL_FEATURES
                    ]
                ]
            )
        )


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        X_cat_encoded = (
            self.onehot_encoder_
            .transform(X_cat)
        )


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        X_final = np.column_stack(

            [
                X_cat_encoded,
                X_num
            ]

        )


        # ----------------------------------------------------
        # STANDARDIZE
        # ----------------------------------------------------

        X_final = (
            self.scaler_
            .transform(X_final)
        )


        return X_final


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):

        X_transformed = (
            self._transform(X)
        )


        return self.model_.predict(
            X_transformed
        )


    # ========================================================
    # PREDICT PROBA
    # ========================================================

    def predict_proba(self, X):

        X_transformed = (
            self._transform(X)
        )


        return self.model_.predict_proba(
            X_transformed
        )


# ============================================================
# STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# RANDOM SEARCH
# ============================================================

print("\n" + "=" * 70)

print("PHASE 4 - RANDOM SEARCH")

print("=" * 70)


# ============================================================
# RANDOM SEARCH SPACE
# ============================================================

RANDOM_SEARCH_SPACE = {

    "hidden_layer_sizes": [

        "16",

        "32",

        "64",

        "128",

        "32_16",

        "64_32",

        "128_64"

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

        0.0005,

        0.001,

        0.005

    ],

    "alpha": [

        0.00001,

        0.0001,

        0.001,

        0.01

    ],

    "batch_size": [

        16,

        32,

        64

    ],

    "max_iter": [

        500,

        750,

        1000

    ]

}


print("\nSearch configurations :")

print(N_ITER)


print("\nCross-validation folds:")

print(N_SPLITS)


print("\nOptimization metric:")

print("Macro F1")


# ============================================================
# BUILD RANDOM SEARCH
# ============================================================

random_search = RandomizedSearchCV(

    estimator=ControlledSMOTENCMLP(

        random_state=RANDOM_STATE

    ),

    param_distributions=RANDOM_SEARCH_SPACE,

    n_iter=N_ITER,

    scoring="f1_macro",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1,

    return_train_score=False

)


# ============================================================
# START RANDOM SEARCH
# ============================================================

print("\nStarting Random Search...")


random_optimization_start = (
    time.time()
)


random_search.fit(

    X_train,

    y_train

)


random_optimization_end = (
    time.time()
)


random_search_runtime = (

    random_optimization_end
    -
    random_optimization_start

)


print("\nRandom Search completed.")


print(

    f"Optimization Time : "
    f"{random_search_runtime:.2f} seconds"

)


# ============================================================
# BEST RANDOM SEARCH PARAMETERS
# ============================================================

print("\n" + "=" * 70)

print(
    "BEST HYPERPARAMETERS FOUND BY RANDOM SEARCH"
)

print("=" * 70)


for key, value in (
    random_search.best_params_.items()
):

    print(
        f"{key}: {value}"
    )


print(

    f"\nBest Cross Validation Macro F1: "
    f"{random_search.best_score_:.4f}"

)


# ============================================================
# TRAIN BEST RANDOM MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING BEST RANDOM SEARCH MODEL")

print("=" * 70)


best_random_model = (
    ControlledSMOTENCMLP(

        **random_search.best_params_,

        random_state=RANDOM_STATE

    )
)


random_training_start = (
    time.time()
)


best_random_model.fit(

    X_train,

    y_train

)


random_training_end = (
    time.time()
)


random_training_runtime = (

    random_training_end
    -
    random_training_start

)


print(

    f"\nTraining Time : "
    f"{random_training_runtime:.2f} seconds"

)


# ============================================================
# RANDOM SEARCH TEST PREDICTIONS
# ============================================================

print("\nMaking Random Search Test Predictions...")


random_predictions = (
    best_random_model.predict(
        X_test
    )
)


# ============================================================
# RANDOM SEARCH METRICS
# ============================================================

random_accuracy = accuracy_score(

    y_test,

    random_predictions

)


random_precision = precision_score(

    y_test,

    random_predictions,

    average="weighted",

    zero_division=0

)


random_recall = recall_score(

    y_test,

    random_predictions,

    average="weighted",

    zero_division=0

)


random_weighted_f1 = f1_score(

    y_test,

    random_predictions,

    average="weighted",

    zero_division=0

)


random_macro_f1 = f1_score(

    y_test,

    random_predictions,

    average="macro",

    zero_division=0

)


# ============================================================
# RANDOM SEARCH RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "CONTROLLED SMOTENC + RANDOM SEARCH RESULTS"
)

print("=" * 70)


print(
    f"Accuracy           : "
    f"{random_accuracy:.4f}"
)


print(
    f"Weighted Precision : "
    f"{random_precision:.4f}"
)


print(
    f"Weighted Recall    : "
    f"{random_recall:.4f}"
)


print(
    f"Weighted F1        : "
    f"{random_weighted_f1:.4f}"
)


print(
    f"Macro F1           : "
    f"{random_macro_f1:.4f}"
)


print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        random_predictions,

        labels=[0, 1, 2, 3],

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        random_predictions,

        labels=[0, 1, 2, 3]

    )

)


# ============================================================
# RANDOM SEARCH SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("RANDOM SEARCH SUMMARY")

print("=" * 70)


print(

    f"Best CV Macro F1 : "
    f"{random_search.best_score_:.4f}"

)


print(

    f"Optimization Time : "
    f"{random_search_runtime:.2f} seconds"

)


print(

    f"Training Time     : "
    f"{random_training_runtime:.2f} seconds"

)


print(

    f"Test Accuracy     : "
    f"{random_accuracy:.4f}"

)


print(

    f"Test Weighted F1  : "
    f"{random_weighted_f1:.4f}"

)


print(

    f"Test Macro F1     : "
    f"{random_macro_f1:.4f}"

)


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================


print("\n\n" + "=" * 70)

print("BAYESIAN OPTIMIZATION")

print("=" * 70)


# ============================================================
# BAYESIAN SEARCH SPACE
# ============================================================
#
# IMPORTANT:
#
# hidden_layer_sizes uses STRINGS here.
#
# ControlledSMOTENCMLP converts the string into the actual
# MLP architecture.
#
# This avoids the previous skopt tuple/bounds problem.
#
# ============================================================

BAYESIAN_SEARCH_SPACE = {

    "hidden_layer_sizes": Categorical([

        "16",

        "32",

        "64",

        "128",

        "32_16",

        "64_32",

        "128_64"

    ]),

    "activation": Categorical([

        "relu",

        "tanh"

    ]),

    "solver": Categorical([

        "adam"

    ]),

    "learning_rate_init": Real(

        0.0001,

        0.005,

        prior="log-uniform"

    ),

    "alpha": Real(

        0.00001,

        0.01,

        prior="log-uniform"

    ),

    "batch_size": Integer(

        16,

        64

    ),

    "max_iter": Integer(

        500,

        1000

    )

}


print("\nSearch configurations :")

print(N_ITER)


print("\nCross-validation folds:")

print(N_SPLITS)


print("\nOptimization metric:")

print("Macro F1")


# ============================================================
# BUILD BAYESIAN OPTIMIZATION
# ============================================================

bayesian_search = BayesSearchCV(

    estimator=ControlledSMOTENCMLP(

        random_state=RANDOM_STATE

    ),

    search_spaces=BAYESIAN_SEARCH_SPACE,

    n_iter=N_ITER,

    scoring="f1_macro",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1,

    return_train_score=False

)


# ============================================================
# START BAYESIAN OPTIMIZATION
# ============================================================

print("\nStarting Bayesian Optimization...")


bayesian_optimization_start = (
    time.time()
)


bayesian_search.fit(

    X_train,

    y_train

)


bayesian_optimization_end = (
    time.time()
)


bayesian_optimization_runtime = (

    bayesian_optimization_end
    -
    bayesian_optimization_start

)


print("\nBayesian Optimization completed.")


print(

    f"Optimization Time : "
    f"{bayesian_optimization_runtime:.2f} seconds"

)


# ============================================================
# BEST BAYESIAN PARAMETERS
# ============================================================

print("\n" + "=" * 70)

print(
    "BEST HYPERPARAMETERS FOUND BY BAYESIAN OPTIMIZATION"
)

print("=" * 70)


for key, value in (
    bayesian_search.best_params_.items()
):

    print(
        f"{key}: {value}"
    )


print(

    f"\nBest Cross Validation Macro F1: "
    f"{bayesian_search.best_score_:.4f}"

)


# ============================================================
# TRAIN BEST BAYESIAN MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING BEST BAYESIAN MODEL")

print("=" * 70)


best_bayesian_model = (
    ControlledSMOTENCMLP(

        **bayesian_search.best_params_,

        random_state=RANDOM_STATE

    )
)


bayesian_training_start = (
    time.time()
)


best_bayesian_model.fit(

    X_train,

    y_train

)


bayesian_training_end = (
    time.time()
)


bayesian_training_runtime = (

    bayesian_training_end
    -
    bayesian_training_start

)


print(

    f"\nTraining Time : "
    f"{bayesian_training_runtime:.2f} seconds"

)


# ============================================================
# BAYESIAN TEST PREDICTIONS
# ============================================================

print("\nMaking Bayesian Test Predictions...")


bayesian_predictions = (
    best_bayesian_model.predict(
        X_test
    )
)


# ============================================================
# BAYESIAN METRICS
# ============================================================

bayesian_accuracy = accuracy_score(

    y_test,

    bayesian_predictions

)


bayesian_precision = precision_score(

    y_test,

    bayesian_predictions,

    average="weighted",

    zero_division=0

)


bayesian_recall = recall_score(

    y_test,

    bayesian_predictions,

    average="weighted",

    zero_division=0

)


bayesian_weighted_f1 = f1_score(

    y_test,

    bayesian_predictions,

    average="weighted",

    zero_division=0

)


bayesian_macro_f1 = f1_score(

    y_test,

    bayesian_predictions,

    average="macro",

    zero_division=0

)


# ============================================================
# BAYESIAN RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "CONTROLLED SMOTENC + BAYESIAN OPTIMIZATION RESULTS"
)

print("=" * 70)


print(
    f"Accuracy           : "
    f"{bayesian_accuracy:.4f}"
)


print(
    f"Weighted Precision : "
    f"{bayesian_precision:.4f}"
)


print(
    f"Weighted Recall    : "
    f"{bayesian_recall:.4f}"
)


print(
    f"Weighted F1        : "
    f"{bayesian_weighted_f1:.4f}"
)


print(
    f"Macro F1           : "
    f"{bayesian_macro_f1:.4f}"
)


print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        bayesian_predictions,

        labels=[0, 1, 2, 3],

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        bayesian_predictions,

        labels=[0, 1, 2, 3]

    )

)


# ============================================================
# BAYESIAN SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("BAYESIAN OPTIMIZATION SUMMARY")

print("=" * 70)


print(

    f"Best CV Macro F1 : "
    f"{bayesian_search.best_score_:.4f}"

)


print(

    f"Optimization Time : "
    f"{bayesian_optimization_runtime:.2f} seconds"

)


print(

    f"Training Time     : "
    f"{bayesian_training_runtime:.2f} seconds"

)


print(

    f"Test Accuracy     : "
    f"{bayesian_accuracy:.4f}"

)


print(

    f"Test Weighted F1  : "
    f"{bayesian_weighted_f1:.4f}"

)


print(

    f"Test Macro F1     : "
    f"{bayesian_macro_f1:.4f}"

)


# ============================================================
# ============================================================
# FINAL RandomSearch vs Bayesian COMPARISON
# ============================================================
# ============================================================


print("\n" + "=" * 70)

print("RandomSearch vs Bayesian COMPARISON")

print("=" * 70)


# ============================================================
# RANDOM SEARCH
# ============================================================

print("\nRandom Search:")


print(

    f"Accuracy        : "
    f"{random_accuracy:.4f}"

)


print(

    f"Weighted F1     : "
    f"{random_weighted_f1:.4f}"

)


print(

    f"Macro F1        : "
    f"{random_macro_f1:.4f}"

)


print(

    f"CV Macro F1     : "
    f"{random_search.best_score_:.4f}"

)


print(

    f"Optimization Time: "
    f"{random_search_runtime:.2f} seconds"

)


# ============================================================
# BAYESIAN
# ============================================================

print("\nBayesian Optimization:")


print(

    f"Accuracy        : "
    f"{bayesian_accuracy:.4f}"

)


print(

    f"Weighted F1     : "
    f"{bayesian_weighted_f1:.4f}"

)


print(

    f"Macro F1        : "
    f"{bayesian_macro_f1:.4f}"

)


print(

    f"CV Macro F1     : "
    f"{bayesian_search.best_score_:.4f}"

)


print(

    f"Optimization Time: "
    f"{bayesian_optimization_runtime:.2f} seconds"

)


# ============================================================
# DETERMINE BEST TEST MACRO F1
# ============================================================

if random_macro_f1 > bayesian_macro_f1:

    best_method = "Random Search"

    improvement = (
        random_macro_f1
        -
        bayesian_macro_f1
    )

else:

    best_method = "Bayesian Optimization"

    improvement = (
        bayesian_macro_f1
        -
        random_macro_f1
    )


print("\n" + "=" * 70)

print("BEST OPTIMIZATION METHOD")

print("=" * 70)


print(

    f"Best Method : "
    f"{best_method}"

)


print(

    f"Macro F1 Difference : "
    f"{improvement:.4f}"

)



# ============================================================
# VISUALIZATION 1 - RANDOM SEARCH VS BAYESIAN PERFORMANCE
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


methods = [
    "Random Search",
    "Bayesian Optimization"
]


accuracy_values = [
    random_accuracy,
    bayesian_accuracy
]


weighted_f1_values = [
    random_weighted_f1,
    bayesian_weighted_f1
]


macro_f1_values = [
    random_macro_f1,
    bayesian_macro_f1
]


x = np.arange(len(methods))

width = 0.25


plt.figure(figsize=(10, 6))


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


plt.xlabel("Optimization Method")

plt.ylabel("Score")

plt.title(
    "Random Search vs Bayesian Optimization - Performance Comparison"
)


plt.xticks(
    x,
    methods
)


plt.ylim(0, 1)


plt.legend()


plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 2 - OPTIMIZATION TIME
# ============================================================


optimization_times = [
    random_search_runtime,
    bayesian_optimization_runtime
]


plt.figure(figsize=(8, 5))


plt.bar(
    methods,
    optimization_times
)


plt.xlabel("Optimization Method")

plt.ylabel("Optimization Time (seconds)")

plt.title(
    "Random Search vs Bayesian Optimization - Optimization Time"
)


plt.tight_layout()

plt.show()
# ============================================================
# FINAL COMPLETION
# ============================================================

print("\n" + "=" * 70)

print("RandomSearch and Bayesian COMPLETED")

print("=" * 70)