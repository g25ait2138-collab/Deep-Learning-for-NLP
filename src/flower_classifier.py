"""Reproducible educational binary flower classifier (single sigmoid neuron).

The eight labels are synthetic red/blue classes, not botanical species.
This is logistic regression, not a multi-hidden-layer neural network.
"""
from dataclasses import dataclass
import numpy as np

DATA = np.array([
    [3.0, 1.5, 1], [2.0, 1.0, 0], [4.0, 1.5, 1], [3.0, 1.0, 0],
    [3.5, 0.5, 1], [2.0, 0.5, 0], [5.5, 1.0, 1], [1.0, 1.0, 0],
], dtype=float)
MYSTERY_FLOWER = np.array([4.5, 1.0], dtype=float)


def sigmoid(z):
    """Numerically stable sigmoid, works on scalars and arrays."""
    x = np.asarray(z, dtype=float)
    ans = np.empty_like(x)
    positive = x >= 0
    ans[positive] = 1 / (1 + np.exp(-x[positive]))
    ez = np.exp(x[~positive])
    ans[~positive] = ez / (1 + ez)
    return float(ans) if ans.ndim == 0 else ans


def binary_cross_entropy_from_logits(logits, targets):
    """Mean binary cross-entropy computed stably in logit space."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)
    return float(np.mean(np.logaddexp(0, z) - y * z))


@dataclass
class FlowerModel:
    weights: np.ndarray
    bias: float

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (np.asarray(self.predict_proba(X)) >= threshold).astype(int)


def train_model(data=DATA, *, seed=42, learning_rate=0.1, epochs=6000):
    """Full-batch gradient descent, a fixed RNG seed, and stable BCE loss.

    Returns (model, loss_history). This is demonstration-only training on 8
    labeled points; there is no independent generalization estimate.
    """
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError('data must have columns [length, width, label]')
    if len(matrix) == 0 or not np.all(np.isin(matrix[:, 2], [0, 1])):
        raise ValueError('data must have at least one row with binary labels')
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError('learning_rate and epochs must be positive')
    X, y = matrix[:, :2], matrix[:, 2]
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=2)
    b = 0.0
    history = []
    for _ in range(epochs):
        logits = X @ w + b
        probabilities = sigmoid(logits)
        history.append(binary_cross_entropy_from_logits(logits, y))
        residuals = probabilities - y
        w -= learning_rate * (X.T @ residuals) / len(X)
        b -= learning_rate * float(np.mean(residuals))
    return FlowerModel(w, b), np.asarray(history)
