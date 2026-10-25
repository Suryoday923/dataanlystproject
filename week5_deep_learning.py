"""
Week 5: Deep Learning Fundamentals — handwritten digit classification.
Uses scikit-learn's MLPClassifier for a lightweight CPU neural-network baseline.
Install: pip install scikit-learn matplotlib numpy
Run: python week5_deep_learning.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

digits = load_digits()
X = digits.data.astype("float32") / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
    alpha=0.001, batch_size=32, learning_rate_init=0.001,
    max_iter=300, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=160)
plt.show()
plt.plot(model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.tight_layout()
plt.savefig("training_loss_curve.png", dpi=160)
plt.show()
