"""
Week 4: Supervised Learning Model Implementation
Classification task on the public Breast Cancer Wisconsin (Diagnostic) dataset.
Run: python week4_supervised_learning.py
Dependencies: pandas, numpy, matplotlib, scikit-learn
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay)

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target  # 0=malignant, 1=benign in this scikit-learn dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Test accuracy:", accuracy_score(y_test, pred))
print("Test precision:", precision_score(y_test, pred))
print("Test recall:", recall_score(y_test, pred))
print("Test F1:", f1_score(y_test, pred))
print("Test ROC-AUC:", roc_auc_score(y_test, prob))
print("5-fold CV mean accuracy:", scores["test_accuracy"].mean())
print("5-fold CV mean ROC-AUC:", scores["test_roc_auc"].mean())

ConfusionMatrixDisplay.from_predictions(y_test, pred,
    display_labels=["Malignant", "Benign"], values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=160)
plt.show()

RocCurveDisplay.from_predictions(y_test, prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=160)
plt.show()
