from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# =====================================================
# Create output folder
# =====================================================

OUTPUT_DIR = "processed_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSTARTING DATASET PREPARATION")
print("=" * 80)

# =====================================================
# DATASET 1 - Energy Efficiency
# =====================================================

print("\nProcessing Energy Efficiency...")

energy = fetch_ucirepo(id=242)

X = energy.data.features
y = energy.data.targets

df = pd.concat([X, y], axis=1)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nVariables:")
print(energy.variables)

df.to_csv(
    f"{OUTPUT_DIR}/energy_efficiency_original.csv",
    index=False
)

print("Saved: energy_efficiency_original.csv")

# =====================================================
# EXPLORATORY DATA ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("TARGET VARIABLE SUMMARY")
print("=" * 80)

print("\nHeating Load (Y1) Statistics")
print(y["Y1"].describe())

print("\nCooling Load (Y2) Statistics")
print(y["Y2"].describe())

# =====================================================
# CORRELATION ANALYSIS
# =====================================================
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Calculate the correlation between the target variables.
correlation = y.corr()

# The target variables (Heating Load(Y1) and Cooling Load(Y2)) are highly correlated.
# However, a high correlation does not necessarily indicate the presence of
# natural clusters. Therefore, exploratory visualization is performed before
# applying K-Means clustering.
print(correlation)

# Visualize the target variables to assess whether they
# exhibit natural groupings suitable for K-Means clustering.
# =====================================================
# TARGET VISUALIZATION
# =====================================================
# Visualize the target variables to understand their distributions
# and examine whether they exhibit natural groupings suitable for
# K-Means clustering.
plt.figure(figsize=(6,4))
plt.hist(y["Y1"], bins=20)
plt.title("Heating Load (Y1)")
plt.xlabel("Heating Load")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(y["Y2"], bins=20)
plt.title("Cooling Load (Y2)")
plt.xlabel("Cooling Load")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y["Y1"], y["Y2"])
plt.title("Heating Load vs Cooling Load")
plt.xlabel("Heating Load (Y1)")
plt.ylabel("Cooling Load (Y2)")
plt.grid(True)
plt.show()
# Y1 and Y2 are not normally distributed and exhibit multiple peaks.
# The scatter plot indicates potential cluster structures.
# Therefore, the Elbow Method will be used to determine the optimal number of clusters for K-Means.

# =====================================================
# PREPARE TARGET VARIABLES FOR CLUSTERING
# =====================================================
# Standardize the target variables so that Y1 and Y2 have the same scale.
# Since K-Means is a distance-based algorithm, standardization prevents one
# variable from dominating the clustering process. The standardized data will
# then be used to determine the optimal number of clusters using the Elbow
# Method and Silhouette Score.
print("\n" + "=" * 80)
print("STANDARDIZING TARGET VARIABLES")
print("=" * 80)

# Select only the two target variables
target_data = df[["Y1", "Y2"]]


scaler = StandardScaler()

target_scaled = scaler.fit_transform(target_data)

# =====================================================
# ELBOW METHOD
# =====================================================
# The Elbow Method is used to identify the optimal number
# of clusters by analyzing the Within-Cluster Sum of Squares (WCSS).
#
# From the Elbow Method, it was observed that the reduction
# in WCSS begins to slow down after k = 4. This indicates
# that adding more clusters beyond this point provides only
# marginal improvement in cluster compactness.
#
# Therefore, k = 4 is considered as a candidate for the
# optimal number of clusters.

print("\n" + "=" * 80)
print("ELBOW METHOD")
print("=" * 80)



wcss = []

# Evaluate different values of k
k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(target_scaled)

    wcss.append(kmeans.inertia_)

    print(f"k = {k}, WCSS = {kmeans.inertia_:.2f}")


plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# =====================================================
# SILHOUETTE SCORE
# =====================================================
# The Silhouette Score evaluates the quality of the clusters generated by K-Means.
# It ranges from -1 to 1, where values closer to 1 indicate well-separated and compact clusters.
# Although k = 2 achieved the highest score (0.7165), k = 4 also produced a reasonably good score (0.5967).
# Therefore, k = 4 was selected as it provides a good balance between cluster quality and the Elbow Method results.

print("\n" + "=" * 80)
print("SILHOUETTE SCORE")
print("=" * 80)

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(target_scaled)

    score = silhouette_score(target_scaled, labels)

    silhouette_scores.append(score)

    print(f"k = {k}, Silhouette Score = {score:.4f}")

# =====================================================
# K-MEANS CLUSTERING
# =====================================================

# Based on the Elbow Method and Silhouette Score,
# k = 4 was selected as the optimal number of clusters.

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(target_scaled)
df["Cluster"] = clusters

cluster_mapping = {
    0: "Low",
    2: "Medium",
    1: "High",
    3: "Very High"
}

df["Energy_Class"] = df["Cluster"].map(cluster_mapping)

print("\n" + "=" * 80)
print("ENERGY CLASS DISTRIBUTION")
print("=" * 80)

print(df["Energy_Class"].value_counts())

cluster_summary = (
    df.groupby("Energy_Class")[["Y1", "Y2"]]
      .mean()
      .reindex(["Low", "Medium", "High", "Very High"])
)

print("\nAverage Heating and Cooling Load for Each Energy Class")
print(cluster_summary.round(2))

plt.figure(figsize=(7,6))

plt.scatter(
    df["Y1"],
    df["Y2"],
    c=df["Cluster"],
    cmap="viridis"
)

plt.title("Energy Classes Identified Using K-Means Clustering")
plt.xlabel("Heating Load (Y1)")
plt.ylabel("Cooling Load (Y2)")
plt.colorbar(label="Cluster ID")
plt.grid(True)
plt.show()

df.to_csv(
    f"{OUTPUT_DIR}/energy_efficiency_clustered.csv",
    index=False
)

print("Saved: energy_efficiency_clustered.csv")