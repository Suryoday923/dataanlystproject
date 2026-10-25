"""
Week 6 Capstone: End-to-end data science pipeline.
Dataset: Breast Cancer Wisconsin (Diagnostic), loaded from scikit-learn.
Educational project only; not a medical diagnostic tool.
Install: pip install pandas numpy matplotlib scikit-learn
Run: python week6_capstone.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score, ConfusionMatrixDisplay, RocCurveDisplay

data = load_breast_cancer(as_frame=True).frame
X, y = data.drop(columns="target"), data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
classifier = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, random_state=42))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    classifier, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
prob = classifier.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("5-fold CV accuracy:", cv_scores["test_accuracy"].mean())

# Unsupervised analysis: fit scaler and K-Means on training data only.
cluster_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_scaled = cluster_prep.fit_transform(X_train)
X_test_scaled = cluster_prep.transform(X_test)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)
print("Training silhouette:", silhouette_score(X_train_scaled, cluster_labels))
