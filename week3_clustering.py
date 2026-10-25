"""
Week 3: Unsupervised Learning and Clustering Analysis
Dataset: Iris dataset bundled with scikit-learn (originally a public UCI dataset).
Run: python week3_clustering.py
Dependencies: pandas, numpy, matplotlib, scikit-learn, scipy
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

iris = load_iris(as_frame=True)
data = iris.frame.copy()
features = iris.feature_names
X = data[features]
X_scaled = StandardScaler().fit_transform(X)

# Compare k values using inertia and silhouette score.
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    print(f"k={k}: inertia={model.inertia_:.3f}, silhouette={silhouette_score(X_scaled, labels):.3f}")

# Final K-Means model: k=3, selected for interpretable three-group segmentation.
model = KMeans(n_clusters=3, random_state=42, n_init=10)
data["cluster"] = model.fit_predict(X_scaled)
print("\\nCluster sizes:")
print(data["cluster"].value_counts().sort_index())
print("\\nCluster feature means (original units):")
print(data.groupby("cluster")[features].mean().round(2))
print("\\nOverall silhouette:", round(silhouette_score(X_scaled, data["cluster"]), 3))

# 2-D visualization using PCA; PCA is only for visualization, not clustering.
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
for cluster_id in sorted(data["cluster"].unique()):
    mask = data["cluster"].to_numpy() == cluster_id
    plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {cluster_id}", alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means clusters shown in PCA space")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_pca_clusters.png", dpi=160)
plt.show()

# Independent hierarchical comparison using Ward linkage.
hier = AgglomerativeClustering(n_clusters=3, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)
print("Hierarchical silhouette:", round(silhouette_score(X_scaled, hier_labels), 3))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="lastp", p=25, show_leaf_counts=True)
plt.title("Hierarchical clustering dendrogram")
plt.tight_layout()
plt.savefig("hierarchical_dendrogram.png", dpi=160)
plt.show()
