from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


# =====================================================
# PHASE 3 - SOLAR FLARE
# IMBALANCE-AWARE MODEL OPTIMIZATION
# =====================================================
#
# Phase 3 investigates whether handling the severe class
# imbalance in the Solar Flare dataset can improve the
# recognition of minority flare classes.
#
# The original Phase 2 experiment is NOT modified.
#
# Target classes:
#
# 0 - Common Flare
# 1 - Moderate Flare
# 2 - No Flare
# 3 - Severe Flare
#
# SMOTENC is used because the Solar Flare dataset contains
# both categorical and numerical features.
#
# IMPORTANT:
# SMOTENC is applied ONLY to the training data.
# The test data remains completely untouched.
# =====================================================


# =====================================================
# LOAD SOLAR FLARE DATASET
# =====================================================

print("\nLoading Solar Flare Dataset...")

DATA_PATH = "processed_datasets/solar_flare_processed.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())


# =====================================================
# SEPARATE FEATURES AND TARGET
# =====================================================

X = df.drop(columns=["Target"])

y = df["Target"]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# =====================================================
# TARGET DISTRIBUTION
# =====================================================

print("\nOriginal Target Distribution:")
print(y.value_counts())


# =====================================================
# ENCODE TARGET
# =====================================================
#
# The target is converted into numerical labels so that
# the machine learning model can process the classes.
# =====================================================

target_mapping = {

    "Common Flare": 0,

    "Moderate Flare": 1,

    "No Flare": 2,

    "Severe Flare": 3

}

y = y.map(target_mapping)


print("\nEncoded Target Distribution:")
print(y.value_counts().sort_index())


print("\nEncoded Target Classes:")

for label, class_name in enumerate(target_mapping):

    print(
        f"{label}: {class_name}"
    )


# =====================================================
# IDENTIFY CATEGORICAL FEATURES
# =====================================================
#
# SMOTENC requires us to explicitly identify categorical
# feature columns.
#
# These are the original categorical Solar Flare variables:
#
# • modified Zurich class
# • largest spot size
# • spot distribution
# =====================================================

categorical_columns = [

    "modified Zurich class",

    "largest spot size",

    "spot distribution"

]


categorical_indices = [

    X.columns.get_loc(column)

    for column in categorical_columns

]


print("\nCategorical Features:")

for column in categorical_columns:

    print(column)


print("\nCategorical Feature Indices:")

print(categorical_indices)


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
#
# Stratification is used so that the class distribution
# is represented in both training and testing datasets.
#
# random_state=42 ensures reproducibility.
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

print("\nTraining Target Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Target Distribution:")
print(y_test.value_counts().sort_index())


# =====================================================
# APPLY SMOTENC TO TRAINING DATA
# =====================================================
#
# SMOTENC generates synthetic samples for minority
# classes while respecting the categorical features.
#
# IMPORTANT:
#
# Only X_train and y_train are passed to SMOTENC.
#
# X_test and y_test are NOT modified.
# =====================================================

print("\nApplying SMOTENC to Training Data...")

smote_nc = SMOTENC(

    categorical_features=categorical_indices,

    random_state=42,

    k_neighbors=3

)


X_train_resampled, y_train_resampled = smote_nc.fit_resample(

    X_train,

    y_train

)


# =====================================================
# DISPLAY BALANCED TRAINING DISTRIBUTION
# =====================================================

print("\nTraining Distribution Before SMOTENC:")

print(
    y_train.value_counts().sort_index()
)


print("\nTraining Distribution After SMOTENC:")

print(
    pd.Series(y_train_resampled).value_counts().sort_index()
)


# =====================================================
# DISPLAY FINAL SHAPES
# =====================================================

print("\nTraining Feature Shape After SMOTENC:")

print(X_train_resampled.shape)


print("\nTraining Target Shape After SMOTENC:")

print(y_train_resampled.shape)


print("\nTesting Feature Shape:")

print(X_test.shape)


print("\nTesting Target Shape:")

print(y_test.shape)

print("\nSMOTENC processing completed successfully.")

# =====================================================
# FEATURE ENCODING AFTER SMOTENC
# =====================================================
#
# SMOTENC is performed before one-hot encoding because
# the Solar Flare dataset contains categorical features.
#
# After SMOTENC, the categorical variables are converted
# into numerical one-hot encoded features.
#
# The encoder is fitted using the resampled training data
# and then applied to the untouched test data.
# =====================================================

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


print("\nEncoding Features After SMOTENC...")


encoder = OneHotEncoder(

    handle_unknown="ignore",

    sparse_output=False

)


X_train_encoded = encoder.fit_transform(

    X_train_resampled

)


X_test_encoded = encoder.transform(

    X_test

)


print("\nFeature Matrix After Encoding:")

print(
    X_train_encoded.shape
)

print(
    X_test_encoded.shape
)


# =====================================================
# FEATURE STANDARDIZATION
# =====================================================
#
# Standardization places the input features on a similar
# numerical scale.
#
# The scaler is fitted ONLY on the training data.
# The same transformation is then applied to the test data.
# =====================================================

print("\nStandardizing Features...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(

    X_train_encoded

)


X_test_scaled = scaler.transform(

    X_test_encoded

)


print("\nTraining Feature Shape After Standardization:")

print(
    X_train_scaled.shape
)


print("\nTesting Feature Shape After Standardization:")

print(
    X_test_scaled.shape
)


print("\nFeature Standardization Completed.")


# =====================================================
# BASELINE MLP MODEL
# =====================================================
#
# The Phase 3 baseline uses the same MLP configuration
# as the original Phase 2 experiment.
#
# The purpose is to determine whether handling class
# imbalance alone improves Solar Flare classification.
#
# Baseline configuration:
#
# • Hidden layers: (64,)
# • Activation: ReLU
# • Solver: Adam
# • Learning rate: 0.001
# • Alpha: 0.0001
# • Batch size: 32
# • Maximum iterations: 1000
#
# The original Phase 2 baseline is NOT modified.
# =====================================================

from sklearn.neural_network import MLPClassifier

import time


print("\nBuilding Phase 3 Baseline MLP Model...")


baseline_params = {

    "hidden_layer_sizes": (64,),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": 32,

    "max_iter": 1000,

    "random_state": 42

}


baseline_model = MLPClassifier(

    **baseline_params

)


# =====================================================
# TRAIN PHASE 3 BASELINE MLP
# =====================================================

print("\nTraining Phase 3 Baseline MLP...")


start = time.time()


baseline_model.fit(

    X_train_scaled,

    y_train_resampled

)


end = time.time()


baseline_training_runtime = end - start


print(
    f"\nTraining completed in "
    f"{baseline_training_runtime:.2f} seconds"
)


# =====================================================
# MAKE PREDICTIONS
# =====================================================

print("\nMaking Predictions...")


y_pred = baseline_model.predict(

    X_test_scaled

)


# =====================================================
# EVALUATE PHASE 3 BASELINE
# =====================================================
#
# Accuracy is retained for comparison with Phase 2.
#
# Weighted F1 is also retained for consistency.
#
# Macro F1 is added because Solar Flare contains severe
# class imbalance and macro F1 gives equal importance to
# all four classes.
# =====================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix

)


baseline_accuracy = accuracy_score(

    y_test,

    y_pred

)


baseline_precision_weighted = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


baseline_recall_weighted = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


baseline_f1_weighted = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


baseline_f1_macro = f1_score(

    y_test,

    y_pred,

    average="macro",

    zero_division=0

)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 60)

print("PHASE 3 BASELINE MLP RESULTS")

print("=" * 60)


print(
    f"Accuracy          : "
    f"{baseline_accuracy:.4f}"
)


print(
    f"Weighted Precision : "
    f"{baseline_precision_weighted:.4f}"
)


print(
    f"Weighted Recall    : "
    f"{baseline_recall_weighted:.4f}"
)


print(
    f"Weighted F1        : "
    f"{baseline_f1_weighted:.4f}"
)


print(
    f"Macro F1           : "
    f"{baseline_f1_macro:.4f}"
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        y_pred,

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# =====================================================
# DISPLAY BASELINE HYPERPARAMETERS
# =====================================================

print("\nPhase 3 Baseline Hyperparameters")


for key, value in baseline_params.items():

    print(

        f"{key}: {value}"

    )


print(

    f"\nTraining Time: "
    f"{baseline_training_runtime:.2f} seconds"

)


print("\nPhase 3 Baseline MLP Completed.")
# =====================================================
# PHASE 3C - CONTROLLED SMOTENC
# =====================================================
#
# The previous Phase 3 experiment used full SMOTENC
# balancing, where every class was increased to match
# the majority class.
#
# Original training distribution:
#
# Common Flare       : 100
# Moderate Flare     : 40
# No Flare           : 962
# Severe Flare       : 9
#
# Full SMOTENC changed the distribution to:
#
# Common Flare       : 962
# Moderate Flare     : 962
# No Flare            : 962
# Severe Flare        : 962
#
# Although full SMOTENC increased the representation of
# minority classes, it required a very large number of
# synthetic samples to be generated from very small
# original classes.
#
# In particular:
#
# Severe Flare:
#
# 9 real samples -> 962 training samples
#
# This means that 953 additional synthetic Severe Flare
# samples were generated from only 9 real observations.
#
# The Phase 3 full-SMOTENC experiment showed:
#
# Accuracy          : 0.7230
# Weighted F1       : 0.7456
# Macro F1          : 0.2659
#
# Severe Flare recall improved from 0.00 to 0.50,
# showing that the model became more willing to identify
# the minority class.
#
# However, overall accuracy and weighted F1 decreased,
# while Moderate Flare recall remained 0.00.
#
# Therefore, full balancing may be too aggressive for
# this highly imbalanced dataset, particularly for the
# extremely rare Severe Flare class.
#
# =====================================================
# CONTROLLED SMOTENC APPROACH
# =====================================================
#
# Instead of forcing every class to contain the same
# number of samples, controlled SMOTENC increases the
# minority classes to selected target levels.
#
# Proposed training distribution:
#
# No Flare          : 962  (unchanged)
# Common Flare      : 400
# Moderate Flare    : 250
# Severe Flare      : 100
#
# The purpose is NOT to guarantee better performance.
#
# The purpose is to investigate whether moderate
# oversampling can:
#
# • improve minority-class recognition
# • improve Macro F1
# • improve recall for Moderate and Severe Flare
# • reduce the dominance of the No Flare class
# • avoid excessive synthetic generation
#
# The test dataset remains completely untouched.
#
# This experiment therefore provides a controlled
# comparison between:
#
# 1. Original training distribution
# 2. Full SMOTENC
# 3. Controlled SMOTENC
#
# =====================================================

print("\n" + "=" * 60)

print("PHASE 3C - CONTROLLED SMOTENC")

print("=" * 60)


# =====================================================
# DEFINE CONTROLLED SAMPLING STRATEGY
# =====================================================
#
# The dictionary specifies the desired number of
# training samples for each minority class.
#
# Class labels:
#
# 0 = Common Flare
# 1 = Moderate Flare
# 2 = No Flare
# 3 = Severe Flare
#
# The majority class (No Flare) remains unchanged.
# =====================================================

controlled_sampling_strategy = {

    0: 400,

    1: 250,

    3: 100

}


print("\nControlled SMOTENC Target Distribution:")

print(
    "Common Flare   : 400"
)

print(
    "Moderate Flare : 250"
)

print(
    "No Flare       : 962 (unchanged)"
)

print(
    "Severe Flare   : 100"
)


# =====================================================
# CREATE CONTROLLED SMOTENC
# =====================================================

controlled_smote_nc = SMOTENC(

    categorical_features=categorical_indices,

    sampling_strategy=controlled_sampling_strategy,

    random_state=42,

    k_neighbors=3

)


# =====================================================
# APPLY CONTROLLED SMOTENC
# =====================================================
#
# SMOTENC is applied ONLY to the original training data.
#
# The test data is NOT passed to SMOTENC.
# =====================================================

print("\nApplying Controlled SMOTENC...")


X_train_controlled, y_train_controlled = (

    controlled_smote_nc.fit_resample(

        X_train,

        y_train

    )

)


# =====================================================
# DISPLAY DISTRIBUTION BEFORE CONTROLLED SMOTENC
# =====================================================

print("\nTraining Distribution Before Controlled SMOTENC:")

print(

    y_train.value_counts().sort_index()

)


# =====================================================
# DISPLAY DISTRIBUTION AFTER CONTROLLED SMOTENC
# =====================================================

print("\nTraining Distribution After Controlled SMOTENC:")

print(

    pd.Series(

        y_train_controlled

    ).value_counts().sort_index()

)


# =====================================================
# DISPLAY DATASET SIZES
# =====================================================

print("\nTraining Feature Shape Before SMOTENC:")

print(

    X_train.shape

)


print("\nTraining Feature Shape After Controlled SMOTENC:")

print(

    X_train_controlled.shape

)


print("\nTesting Feature Shape:")

print(

    X_test.shape

)


# =====================================================
# ENCODE CONTROLLED TRAINING DATA
# =====================================================
#
# One-hot encoding is performed AFTER SMOTENC because
# SMOTENC was applied while the categorical variables
# were still explicitly identified.
#
# The encoder is fitted only on the controlled training
# data and then applied to the untouched test data.
# =====================================================

print("\nEncoding Controlled Training Data...")


controlled_encoder = OneHotEncoder(

    handle_unknown="ignore",

    sparse_output=False

)


X_train_controlled_encoded = (

    controlled_encoder.fit_transform(

        X_train_controlled

    )

)


X_test_controlled_encoded = (

    controlled_encoder.transform(

        X_test

    )

)


print("\nControlled Training Feature Shape After Encoding:")

print(

    X_train_controlled_encoded.shape

)


print("\nTesting Feature Shape After Encoding:")

print(

    X_test_controlled_encoded.shape

)


# =====================================================
# STANDARDIZE FEATURES
# =====================================================
#
# The scaler is fitted only on the controlled training
# data.
#
# The same transformation is applied to the untouched
# test data.
# =====================================================

print("\nStandardizing Controlled Training Data...")


controlled_scaler = StandardScaler()


X_train_controlled_scaled = (

    controlled_scaler.fit_transform(

        X_train_controlled_encoded

    )

)


X_test_controlled_scaled = (

    controlled_scaler.transform(

        X_test_controlled_encoded

    )

)


print("\nControlled Training Feature Shape After Scaling:")

print(

    X_train_controlled_scaled.shape

)


print("\nTesting Feature Shape After Scaling:")

print(

    X_test_controlled_scaled.shape

)


# =====================================================
# BUILD CONTROLLED-SMOTENC BASELINE MLP
# =====================================================
#
# The same MLP configuration used in the original
# Phase 2 and Phase 3 baseline experiments is retained.
#
# This ensures that the main experimental change is
# the sampling strategy rather than the neural-network
# architecture.
# =====================================================

print("\nBuilding Controlled SMOTENC Baseline MLP...")


controlled_baseline_params = {

    "hidden_layer_sizes": (64,),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": 32,

    "max_iter": 1000,

    "random_state": 42

}


controlled_baseline_model = MLPClassifier(

    **controlled_baseline_params

)


# =====================================================
# TRAIN CONTROLLED-SMOTENC BASELINE
# =====================================================

print("\nTraining Controlled SMOTENC Baseline MLP...")


controlled_start = time.time()


controlled_baseline_model.fit(

    X_train_controlled_scaled,

    y_train_controlled

)


controlled_end = time.time()


controlled_training_runtime = (

    controlled_end - controlled_start

)


print(

    f"\nTraining completed in "

    f"{controlled_training_runtime:.2f} seconds"

)


# =====================================================
# MAKE PREDICTIONS
# =====================================================
#
# Predictions are made on the ORIGINAL untouched test
# dataset.
# =====================================================

print("\nMaking Predictions...")


controlled_y_pred = (

    controlled_baseline_model.predict(

        X_test_controlled_scaled

    )

)


# =====================================================
# CALCULATE PERFORMANCE METRICS
# =====================================================

controlled_accuracy = accuracy_score(

    y_test,

    controlled_y_pred

)


controlled_precision_weighted = precision_score(

    y_test,

    controlled_y_pred,

    average="weighted",

    zero_division=0

)


controlled_recall_weighted = recall_score(

    y_test,

    controlled_y_pred,

    average="weighted",

    zero_division=0

)


controlled_f1_weighted = f1_score(

    y_test,

    controlled_y_pred,

    average="weighted",

    zero_division=0

)


controlled_f1_macro = f1_score(

    y_test,

    controlled_y_pred,

    average="macro",

    zero_division=0

)


# =====================================================
# DISPLAY CONTROLLED SMOTENC RESULTS
# =====================================================

print("\n" + "=" * 60)

print("CONTROLLED SMOTENC BASELINE MLP RESULTS")

print("=" * 60)


print(

    f"Accuracy          : "

    f"{controlled_accuracy:.4f}"

)


print(

    f"Weighted Precision : "

    f"{controlled_precision_weighted:.4f}"

)


print(

    f"Weighted Recall    : "

    f"{controlled_recall_weighted:.4f}"

)


print(

    f"Weighted F1        : "

    f"{controlled_f1_weighted:.4f}"

)


print(

    f"Macro F1           : "

    f"{controlled_f1_macro:.4f}"

)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        controlled_y_pred,

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        controlled_y_pred

    )

)


# =====================================================
# DISPLAY HYPERPARAMETERS
# =====================================================

print("\nControlled SMOTENC Baseline Hyperparameters")


for key, value in controlled_baseline_params.items():

    print(

        f"{key}: {value}"

    )


print(

    f"\nTraining Time: "

    f"{controlled_training_runtime:.2f} seconds"

)


# =====================================================
# PHASE 3 SMOTENC COMPARISON
# =====================================================
#
# This comparison allows us to determine whether
# controlled oversampling provides a better balance
# between overall predictive performance and minority
# class recognition than full SMOTENC.
# =====================================================

print("\n" + "=" * 60)

print("SMOTENC STRATEGY COMPARISON")

print("=" * 60)


print(

    f"\nFull SMOTENC Accuracy : "

    f"{baseline_accuracy:.4f}"

)


print(

    f"Full SMOTENC Weighted F1 : "

    f"{baseline_f1_weighted:.4f}"

)


print(

    f"Full SMOTENC Macro F1 : "

    f"{baseline_f1_macro:.4f}"

)


print(

    f"\nControlled SMOTENC Accuracy : "

    f"{controlled_accuracy:.4f}"

)


print(

    f"Controlled SMOTENC Weighted F1 : "

    f"{controlled_f1_weighted:.4f}"

)


print(

    f"Controlled SMOTENC Macro F1 : "

    f"{controlled_f1_macro:.4f}"

)


print("\nControlled SMOTENC experiment completed.")

# =====================================================
# PHASE 3D - IMBALANCE-AWARE WEIGHTED BASELINE
# =====================================================
#
# Phase 2 established the original MLP baseline without
# any treatment for class imbalance.
#
# Phase 3 tested Full SMOTENC:
#
# Common Flare       : 100  -> 962
# Moderate Flare     : 40   -> 962
# No Flare           : 962  -> 962
# Severe Flare       : 9    -> 962
#
# Full SMOTENC improved recognition of the Severe Flare
# class, but overall accuracy decreased substantially.
#
# Phase 3C tested Controlled SMOTENC:
#
# Common Flare       : 100  -> 400
# Moderate Flare     : 40   -> 250
# No Flare           : 962  -> 962
# Severe Flare       : 9    -> 100
#
# Controlled SMOTENC recovered some overall accuracy,
# but minority-class recognition remained weak.
#
# =====================================================
# WHY TEST A WEIGHTED APPROACH?
# =====================================================
#
# SMOTENC changes the training dataset by generating
# synthetic observations.
#
# A weighting approach takes a different approach:
#
# • The original training observations are retained.
# • No synthetic observations are generated.
# • Minority-class observations are given greater
#   importance during training.
#
# Therefore, this experiment allows us to determine
# whether the poor minority-class performance is mainly
# caused by the class imbalance itself rather than the
# lack of synthetic observations.
#
# NOTE:
# MLPClassifier does not provide a class_weight parameter
# in all scikit-learn versions.
#
# Therefore, this implementation uses controlled
# oversampling by repeating ORIGINAL observations.
#
# This is fundamentally different from SMOTENC:
#
# SMOTENC:
#     Generates synthetic observations.
#
# Weighted/Repeated training:
#     Reuses the original observations with greater
#     frequency according to their class importance.
#
# No new feature combinations are generated.
#
# =====================================================

print("\n" + "=" * 60)

print("PHASE 3D - IMBALANCE-AWARE WEIGHTED BASELINE")

print("=" * 60)


# =====================================================
# DISPLAY ORIGINAL TRAINING DISTRIBUTION
# =====================================================

print("\nOriginal Training Distribution:")

print(

    pd.Series(

        y_train

    ).value_counts().sort_index()

)


# =====================================================
# DEFINE CLASS IMPORTANCE
# =====================================================
#
# The majority class receives weight 1.
#
# Minority classes receive greater representation.
#
# We use moderate weights rather than attempting to make
# all classes equal.
#
# This avoids the extremely aggressive transformation
# observed with full SMOTENC.
# =====================================================

class_weights = {

    0: 3.0,    # Common Flare

    1: 5.0,    # Moderate Flare

    2: 1.0,    # No Flare

    3: 8.0     # Severe Flare

}


print("\nClass Weights:")

print("Common Flare   :", class_weights[0])

print("Moderate Flare :", class_weights[1])

print("No Flare       :", class_weights[2])

print("Severe Flare   :", class_weights[3])


# =====================================================
# CREATE WEIGHTED TRAINING DATA
# =====================================================
#
# Each ORIGINAL training observation is repeated according
# to the importance assigned to its class.
#
# No synthetic observations are generated.
# =====================================================

print("\nCreating Imbalance-Aware Training Data...")


weighted_indices = []


for class_label, weight in class_weights.items():

    class_indices = (

        y_train[y_train == class_label].index

    )

    repetition_count = int(weight)


    for index in class_indices:

        weighted_indices.extend(

            [index] * repetition_count

        )


# =====================================================
# CONVERT INDICES TO NUMPY POSITIONS
# =====================================================
#
# X_train may use a pandas index after train_test_split.
# Therefore, we reset the index before selecting the
# repeated observations.
# =====================================================

X_train_reset = X_train.reset_index(drop=True)

y_train_reset = y_train.reset_index(drop=True)


weighted_indices = []


for position, class_label in enumerate(y_train_reset):

    repetition_count = int(

        class_weights[class_label]

    )

    weighted_indices.extend(

        [position] * repetition_count

    )


# =====================================================
# CREATE WEIGHTED DATASET
# =====================================================

X_train_weighted = (

    X_train_reset.iloc[weighted_indices]

    .reset_index(drop=True)

)


y_train_weighted = (

    y_train_reset.iloc[weighted_indices]

    .reset_index(drop=True)

)


# =====================================================
# DISPLAY NEW TRAINING DISTRIBUTION
# =====================================================

print("\nTraining Distribution After Weighting:")

print(

    y_train_weighted.value_counts().sort_index()

)


print("\nWeighted Training Feature Shape:")

print(

    X_train_weighted.shape

)


print("\nWeighted Training Target Shape:")

print(

    y_train_weighted.shape

)


# =====================================================
# ENCODE WEIGHTED TRAINING DATA
# =====================================================
#
# The encoder is fitted ONLY on the weighted training
# data.
#
# The original test data remains untouched.
# =====================================================

print("\nEncoding Weighted Training Data...")


weighted_encoder = OneHotEncoder(

    handle_unknown="ignore",

    sparse_output=False

)


X_train_weighted_encoded = (

    weighted_encoder.fit_transform(

        X_train_weighted

    )

)


X_test_weighted_encoded = (

    weighted_encoder.transform(

        X_test

    )

)


print("\nWeighted Training Feature Shape After Encoding:")

print(

    X_train_weighted_encoded.shape

)


print("\nTesting Feature Shape After Encoding:")

print(

    X_test_weighted_encoded.shape

)


# =====================================================
# STANDARDIZE FEATURES
# =====================================================

print("\nStandardizing Weighted Training Data...")


weighted_scaler = StandardScaler()


X_train_weighted_scaled = (

    weighted_scaler.fit_transform(

        X_train_weighted_encoded

    )

)


X_test_weighted_scaled = (

    weighted_scaler.transform(

        X_test_weighted_encoded

    )

)


print("\nWeighted Training Feature Shape After Scaling:")

print(

    X_train_weighted_scaled.shape

)


print("\nTesting Feature Shape After Scaling:")

print(

    X_test_weighted_scaled.shape

)


print("\nFeature Standardization Completed.")


# =====================================================
# BUILD WEIGHTED BASELINE MLP
# =====================================================
#
# The MLP architecture remains identical to Phase 2.
#
# This ensures that the primary experimental change is
# the imbalance-handling strategy.
# =====================================================

print("\nBuilding Imbalance-Aware Baseline MLP...")


weighted_baseline_params = {

    "hidden_layer_sizes": (64,),

    "activation": "relu",

    "solver": "adam",

    "learning_rate_init": 0.001,

    "alpha": 0.0001,

    "batch_size": 32,

    "max_iter": 1000,

    "random_state": 42

}


weighted_baseline_model = MLPClassifier(

    **weighted_baseline_params

)


# =====================================================
# TRAIN WEIGHTED BASELINE
# =====================================================

print("\nTraining Imbalance-Aware Baseline MLP...")


weighted_start = time.time()


weighted_baseline_model.fit(

    X_train_weighted_scaled,

    y_train_weighted

)


weighted_end = time.time()


weighted_training_runtime = (

    weighted_end - weighted_start

)


print(

    f"\nTraining completed in "

    f"{weighted_training_runtime:.2f} seconds"

)


# =====================================================
# MAKE PREDICTIONS
# =====================================================
#
# Predictions are made on the ORIGINAL untouched test
# dataset.
# =====================================================

print("\nMaking Predictions...")


weighted_y_pred = (

    weighted_baseline_model.predict(

        X_test_weighted_scaled

    )

)


# =====================================================
# CALCULATE PERFORMANCE METRICS
# =====================================================

weighted_accuracy = accuracy_score(

    y_test,

    weighted_y_pred

)


weighted_precision = precision_score(

    y_test,

    weighted_y_pred,

    average="weighted",

    zero_division=0

)


weighted_recall = recall_score(

    y_test,

    weighted_y_pred,

    average="weighted",

    zero_division=0

)


weighted_f1 = f1_score(

    y_test,

    weighted_y_pred,

    average="weighted",

    zero_division=0

)


weighted_macro_f1 = f1_score(

    y_test,

    weighted_y_pred,

    average="macro",

    zero_division=0

)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 60)

print("IMBALANCE-AWARE BASELINE MLP RESULTS")

print("=" * 60)


print(

    f"Accuracy          : "

    f"{weighted_accuracy:.4f}"

)


print(

    f"Weighted Precision : "

    f"{weighted_precision:.4f}"

)


print(

    f"Weighted Recall    : "

    f"{weighted_recall:.4f}"

)


print(

    f"Weighted F1        : "

    f"{weighted_f1:.4f}"

)


print(

    f"Macro F1           : "

    f"{weighted_macro_f1:.4f}"

)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")


print(

    classification_report(

        y_test,

        weighted_y_pred,

        target_names=[

            "Common Flare",

            "Moderate Flare",

            "No Flare",

            "Severe Flare"

        ],

        zero_division=0

    )

)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix\n")


print(

    confusion_matrix(

        y_test,

        weighted_y_pred

    )

)


# =====================================================
# DISPLAY HYPERPARAMETERS
# =====================================================

print("\nImbalance-Aware Baseline Hyperparameters")


for key, value in weighted_baseline_params.items():

    print(

        f"{key}: {value}"

    )


print("\nClass Weights Used:")


for key, value in class_weights.items():

    class_name = {

        0: "Common Flare",

        1: "Moderate Flare",

        2: "No Flare",

        3: "Severe Flare"

    }[key]


    print(

        f"{class_name}: {value}"

    )


print(

    f"\nTraining Time: "

    f"{weighted_training_runtime:.2f} seconds"

)


# =====================================================
# FINAL IMBALANCE STRATEGY COMPARISON
# =====================================================
#
# This comparison summarizes the three approaches
# investigated after the original Phase 2 baseline.
# =====================================================

print("\n" + "=" * 60)

print("IMBALANCE STRATEGY COMPARISON")

print("=" * 60)


print("\nOriginal Phase 2:")

print(

    f"Accuracy : {0.8165:.4f}"

)

print(

    f"Weighted F1 : {0.7883:.4f}"

)


print("\nFull SMOTENC:")

print(

    f"Accuracy : {baseline_accuracy:.4f}"

)

print(

    f"Weighted F1 : {baseline_f1_weighted:.4f}"

)

print(

    f"Macro F1 : {baseline_f1_macro:.4f}"

)


print("\nControlled SMOTENC:")

print(

    f"Accuracy : {controlled_accuracy:.4f}"

)

print(

    f"Weighted F1 : {controlled_f1_weighted:.4f}"

)

print(

    f"Macro F1 : {controlled_f1_macro:.4f}"

)


print("\nImbalance-Aware Weighted:")

print(

    f"Accuracy : {weighted_accuracy:.4f}"

)

print(

    f"Weighted F1 : {weighted_f1:.4f}"

)

print(

    f"Macro F1 : {weighted_macro_f1:.4f}"

)



# ============================================================
# VISUALIZATION 
# IMBALANCE STRATEGY PERFORMANCE COMPARISON
# ============================================================

strategies = [
    "Full SMOTENC",
    "Controlled SMOTENC",
    "Imbalance-Aware Weighted"
]

accuracy_values = [
    baseline_accuracy,
    controlled_accuracy,
    weighted_accuracy
]

weighted_f1_values = [
    baseline_f1_weighted,
    controlled_f1_weighted,
    weighted_f1
]

macro_f1_values = [
    baseline_f1_macro,
    controlled_f1_macro,
    weighted_macro_f1
]


x = np.arange(
    len(strategies)
)

width = 0.25


plt.figure(
    figsize=(11, 6)
)

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


plt.xlabel(
    "Imbalance Handling Strategy"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Solar Flare Dataset - Imbalance Strategy Performance Comparison"
)

plt.xticks(
    x,
    strategies,
    rotation=20,
    ha="right"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()

print("\nPhase 3D completed.")