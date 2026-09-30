from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

OUTPUT_DIR = "processed_datasets"

VISUALIZATION_DIR = (
    "results/phase1_electricity/visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    VISUALIZATION_DIR,
    exist_ok=True
)


print("\nSTARTING DATASET PREPARATION")
print("=" * 80)


# =====================================================
# DATASET 5 - ELECTRICITY
# =====================================================

print("\nProcessing Electricity...")


X, y = fetch_openml(
    data_id=151,
    return_X_y=True,
    as_frame=True
)


df = pd.concat(
    [X, y],
    axis=1
)


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Information:")
print(df.info())


print("\nFeature Columns:")
print(list(X.columns))


print("\nTarget Column:")
print(y.name)


df.to_csv(
    f"{OUTPUT_DIR}/electricity_original.csv",
    index=False
)


print(
    "Saved: electricity_original.csv"
)


# =====================================================
# TARGET PREPARATION
# =====================================================

print("\n" + "=" * 80)
print("TARGET PREPARATION")
print("=" * 80)


df = df.rename(
    columns={
        "class": "Target"
    }
)


print("\nTarget Class Distribution:")


target_distribution = (
    df["Target"].value_counts()
)


print(
    target_distribution
)


df.to_csv(
    f"{OUTPUT_DIR}/electricity_processed.csv",
    index=False
)


print(
    "Saved: electricity_processed.csv"
)


# =====================================================
# DATA QUALITY ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("DATA QUALITY ANALYSIS")
print("=" * 80)


# -----------------------------------------------------
# Missing value analysis
# -----------------------------------------------------

print("\nMissing Values:")


missing_values = (
    df.isnull().sum()
)


print(
    missing_values
)


total_missing = (
    missing_values.sum()
)


print(
    f"\nTotal Missing Values: "
    f"{total_missing}"
)


# -----------------------------------------------------
# Duplicate row analysis
# -----------------------------------------------------

duplicate_count = (
    df.duplicated().sum()
)


print(
    f"\nDuplicate Rows: "
    f"{duplicate_count}"
)


# -----------------------------------------------------
# Infinite value analysis
# -----------------------------------------------------

numeric_columns = (
    df.select_dtypes(
        include=[np.number]
    ).columns
)


infinite_counts = (
    np.isinf(
        df[numeric_columns]
    ).sum()
)


print(
    "\nInfinite Values in Numerical Features:"
)


print(
    infinite_counts
)


total_infinite = (
    infinite_counts.sum()
)


print(
    f"\nTotal Infinite Values: "
    f"{total_infinite}"
)


# =====================================================
# TARGET BALANCE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("TARGET BALANCE ANALYSIS")
print("=" * 80)


total_samples = len(df)


print(
    "\nTarget Distribution with Percentages:"
)


for class_name, count in (
    target_distribution.items()
):

    percentage = (
        count / total_samples
    ) * 100


    print(
        f"{class_name:>5} : "
        f"{count:>6} samples "
        f"({percentage:>6.2f}%)"
    )


# -----------------------------------------------------
# Majority-class baseline
# -----------------------------------------------------

majority_class = (
    target_distribution.idxmax()
)


majority_count = (
    target_distribution.max()
)


majority_accuracy = (
    majority_count /
    total_samples
)


print(
    "\nMajority Class:"
)


print(
    majority_class
)


print(
    f"Majority-Class Baseline Accuracy: "
    f"{majority_accuracy:.4f}"
)


print(
    f"Majority-Class Baseline Accuracy: "
    f"{majority_accuracy * 100:.2f}%"
)


# =====================================================
# FEATURE TYPE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("FEATURE TYPE ANALYSIS")
print("=" * 80)


feature_columns = [
    column
    for column in df.columns
    if column != "Target"
]


categorical_features = (
    df[
        feature_columns
    ]
    .select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    )
    .columns
    .tolist()
)


numerical_features = (
    df[
        feature_columns
    ]
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)


print(
    "\nNumerical Features:"
)


for feature in numerical_features:

    print(
        f"- {feature}"
    )


print(
    "\nCategorical Features:"
)


for feature in categorical_features:

    print(
        f"- {feature}"
    )


# =====================================================
# UNIQUE VALUE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("UNIQUE VALUE ANALYSIS")
print("=" * 80)


for feature in feature_columns:

    print(
        f"{feature:<15} : "
        f"{df[feature].nunique()} unique values"
    )


# =====================================================
# NUMERICAL FEATURE STATISTICS
# =====================================================

print("\n" + "=" * 80)
print("NUMERICAL FEATURE STATISTICS")
print("=" * 80)


if len(numerical_features) > 0:

    numerical_statistics = (
        df[
            numerical_features
        ]
        .describe()
        .T
    )


    numerical_statistics[
        "median"
    ] = (
        df[
            numerical_features
        ].median()
    )


    numerical_statistics[
        "skewness"
    ] = (
        df[
            numerical_features
        ].skew()
    )


    print(
        numerical_statistics[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "median",
                "75%",
                "max",
                "skewness"
            ]
        ].round(4)
    )


# =====================================================
# FEATURE STATISTICS BY TARGET
# =====================================================

print("\n" + "=" * 80)
print("FEATURE STATISTICS BY TARGET")
print("=" * 80)


if len(numerical_features) > 0:

    print(
        "\nMean of Numerical Features "
        "for Each Target Class:"
    )


    target_feature_means = (
        df.groupby(
            "Target",
            observed=True
        )[numerical_features]
        .mean()
    )


    print(
        target_feature_means.round(4)
    )


# -----------------------------------------------------
# Median values by target
# -----------------------------------------------------

print(
    "\nMedian of Numerical Features "
    "for Each Target Class:"
)


target_feature_medians = (
    df.groupby(
        "Target",
        observed=True
    )[numerical_features]
    .median()
)


print(
    target_feature_medians.round(4)
)


# =====================================================
# CORRELATION ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)


# Convert binary target to numerical form
# only for correlation analysis.

target_numeric = (
    df["Target"]
    .map(
        {
            "DOWN": 0,
            "UP": 1
        }
    )
)


correlation_data = (
    df[
        numerical_features
    ].copy()
)


correlation_data[
    "Target"
] = target_numeric


correlation_matrix = (
    correlation_data.corr()
)


print(
    "\nCorrelation Matrix:"
)


print(
    correlation_matrix.round(4)
)


print(
    "\nFeature-Target Correlations:"
)


target_correlations = (
    correlation_matrix[
        "Target"
    ]
    .drop("Target")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)


print(
    target_correlations.round(4)
)


# =====================================================
# DATE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("DATE / TEMPORAL ANALYSIS")
print("=" * 80)


if "date" in df.columns:

    print(
        "\nDate Data Type:"
    )


    print(
        df["date"].dtype
    )


    print(
        "\nFirst 10 Date Values:"
    )


    print(
        df["date"]
        .head(10)
        .to_list()
    )


    print(
        "\nLast 10 Date Values:"
    )


    print(
        df["date"]
        .tail(10)
        .to_list()
    )


    print(
        "\nDate Minimum:"
    )


    print(
        df["date"].min()
    )


    print(
        "\nDate Maximum:"
    )


    print(
        df["date"].max()
    )


    print(
        "\nNumber of Unique Date Values:"
    )


    print(
        df["date"].nunique()
    )


# =====================================================
# PERIOD ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("PERIOD ANALYSIS")
print("=" * 80)


if "period" in df.columns:

    print(
        "\nPeriod Statistics:"
    )


    print(
        df["period"]
        .describe()
        .round(4)
    )


    print(
        "\nNumber of Unique Period Values:"
    )


    print(
        df["period"].nunique()
    )


    print(
        "\nMinimum Period:"
    )


    print(
        df["period"].min()
    )


    print(
        "\nMaximum Period:"
    )


    print(
        df["period"].max()
    )


# =====================================================
# TARGET DISTRIBUTION BY DAY
# =====================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION BY DAY")
print("=" * 80)


if "day" in df.columns:

    day_target_counts = (
        pd.crosstab(
            df["day"],
            df["Target"]
        )
    )


    print(
        "\nTarget Counts by Day:"
    )


    print(
        day_target_counts
    )


    day_target_percentages = (
        pd.crosstab(
            df["day"],
            df["Target"],
            normalize="index"
        ) * 100
    )


    print(
        "\nTarget Percentages by Day:"
    )


    print(
        day_target_percentages.round(2)
    )


# =====================================================
# TARGET DISTRIBUTION BY PERIOD
# =====================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION BY PERIOD")
print("=" * 80)


if "period" in df.columns:

    period_target_percentages = (
        pd.crosstab(
            df["period"],
            df["Target"],
            normalize="index"
        ) * 100
    )


    print(
        "\nTarget Percentages by Period:"
    )


    print(
        period_target_percentages.round(2)
    )


# =====================================================
# TARGET DISTRIBUTION BY DATE
# =====================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION BY DATE")
print("=" * 80)


if "date" in df.columns:

    date_target_percentages = (
        pd.crosstab(
            df["date"],
            df["Target"],
            normalize="index"
        ) * 100
    )


    print(
        "\nFirst 20 Dates:"
    )


    print(
        date_target_percentages
        .head(20)
        .round(2)
    )


    print(
        "\nLast 20 Dates:"
    )


    print(
        date_target_percentages
        .tail(20)
        .round(2)
    )


# =====================================================
# TARGET TRANSITION ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("TARGET TRANSITION ANALYSIS")
print("=" * 80)


if (
    "date" in df.columns
    and
    "period" in df.columns
):

    temporal_df = (
        df.copy()
    )


    # Sort observations chronologically.

    temporal_df = (
        temporal_df
        .sort_values(
            by=[
                "date",
                "period"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # Previous target state.

    temporal_df[
        "Previous_Target"
    ] = (
        temporal_df[
            "Target"
        ].shift(1)
    )


    transition_data = (
        temporal_df
        .dropna(
            subset=[
                "Previous_Target"
            ]
        )
    )


    # Transition counts.

    transition_counts = (
        pd.crosstab(
            transition_data[
                "Previous_Target"
            ],
            transition_data[
                "Target"
            ]
        )
    )


    print(
        "\nTarget Transition Counts:"
    )


    print(
        transition_counts
    )


    # Transition percentages.

    transition_percentages = (
        pd.crosstab(
            transition_data[
                "Previous_Target"
            ],
            transition_data[
                "Target"
            ],
            normalize="index"
        ) * 100
    )


    print(
        "\nTarget Transition Percentages:"
    )


    print(
        transition_percentages.round(2)
    )


# =====================================================
# PHASE 1 VISUALIZATIONS
# =====================================================

print("\n" + "=" * 80)
print("GENERATING PHASE 1 VISUALIZATIONS")
print("=" * 80)


# =====================================================
# VISUALIZATION 1
# TARGET CLASS DISTRIBUTION
# =====================================================

target_counts = (
    df["Target"]
    .value_counts()
)


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    target_counts.index.astype(str),
    target_counts.values
)


plt.xlabel(
    "Electricity Class"
)


plt.ylabel(
    "Number of Samples"
)


plt.title(
    "Electricity Target Class Distribution"
)


for i, value in enumerate(
    target_counts.values
):

    plt.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


plt.savefig(
    f"{VISUALIZATION_DIR}/target_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


plt.close()


# =====================================================
# VISUALIZATION 2
# CORRELATION MATRIX
# =====================================================

plt.figure(
    figsize=(9, 7)
)


correlation_values = (
    correlation_matrix.values
)


plt.imshow(
    correlation_values,
    aspect="auto"
)


plt.colorbar(
    label="Correlation"
)


plt.xticks(
    range(
        len(
            correlation_matrix.columns
        )
    ),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(
        len(
            correlation_matrix.index
        )
    ),
    correlation_matrix.index
)


plt.title(
    "Correlation Matrix of Electricity Features and Target"
)


plt.tight_layout()


plt.savefig(
    f"{VISUALIZATION_DIR}/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


plt.close()


# =====================================================
# VISUALIZATION 3
# TARGET DISTRIBUTION BY PERIOD
# =====================================================

if "period" in df.columns:

    period_percentages = (
        pd.crosstab(
            df["period"],
            df["Target"],
            normalize="index"
        ) * 100
    )


    plt.figure(
        figsize=(10, 5)
    )


    if "DOWN" in (
        period_percentages.columns
    ):

        plt.plot(
            period_percentages.index,
            period_percentages[
                "DOWN"
            ],
            marker="o",
            label="DOWN"
        )


    if "UP" in (
        period_percentages.columns
    ):

        plt.plot(
            period_percentages.index,
            period_percentages[
                "UP"
            ],
            marker="o",
            label="UP"
        )


    plt.xlabel(
        "Normalized Period"
    )


    plt.ylabel(
        "Percentage of Samples"
    )


    plt.title(
        "Electricity Class Distribution Across Periods"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        f"{VISUALIZATION_DIR}/target_distribution_by_period.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


    plt.close()


# =====================================================
# VISUALIZATION 4
# TARGET TRANSITION MATRIX
# =====================================================

if "transition_percentages" in locals():

    plt.figure(
        figsize=(6, 5)
    )


    transition_values = (
        transition_percentages.values
    )


    plt.imshow(
        transition_values,
        aspect="auto"
    )


    plt.colorbar(
        label="Percentage (%)"
    )


    plt.xticks(
        range(
            len(
                transition_percentages.columns
            )
        ),
        transition_percentages.columns
    )


    plt.yticks(
        range(
            len(
                transition_percentages.index
            )
        ),
        transition_percentages.index
    )


    plt.xlabel(
        "Current Target"
    )


    plt.ylabel(
        "Previous Target"
    )


    plt.title(
        "Electricity Target Transition Percentages"
    )


    # Add percentages inside the matrix.

    for i in range(
        transition_values.shape[0]
    ):

        for j in range(
            transition_values.shape[1]
        ):

            plt.text(
                j,
                i,
                f"{transition_values[i, j]:.2f}%",
                ha="center",
                va="center"
            )


    plt.tight_layout()


    plt.savefig(
        f"{VISUALIZATION_DIR}/target_transition_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


    plt.close()


# =====================================================
# VISUALIZATION SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETED")
print("=" * 80)


print(
    "\nVisualizations saved to:"
)


print(
    VISUALIZATION_DIR
)


# =====================================================
# FINAL PHASE 1 SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("PHASE 1 ELECTRICITY DATASET SUMMARY")
print("=" * 80)


print(
    f"\nTotal Samples       : "
    f"{len(df)}"
)


print(
    f"Total Features      : "
    f"{len(feature_columns)}"
)


print(
    f"Numerical Features  : "
    f"{len(numerical_features)}"
)


print(
    f"Categorical Features: "
    f"{len(categorical_features)}"
)


print(
    "\nTarget Distribution:"
)


print(
    target_distribution
)


print(
    f"\nMajority-Class Baseline: "
    f"{majority_accuracy:.4f}"
)


print(
    f"\nMissing Values : "
    f"{total_missing}"
)


print(
    f"Duplicate Rows : "
    f"{duplicate_count}"
)


print(
    f"Infinite Values: "
    f"{total_infinite}"
)


print(
    "\nPhase 1 Electricity Dataset "
    "Analysis Completed."
)