"""
train_models.py
===============
Three ML classifiers implemented from scratch:
  - Multinomial Naive Bayes
  - Binary Logistic Regression (gradient descent)
  - K-Nearest Neighbours (Euclidean distance)
"""

import math
from collections import Counter


# ───────────────────────────────────────────────
# Naive Bayes
# ───────────────────────────────────────────────

class NaiveBayesClassifier:
    """Multinomial Naive Bayes for pre-vectorised features."""

    name = "Naive Bayes"

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_log_prob = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n_features = len(X[0]) if X else 0
        class_counts = Counter(y)
        total = len(y)

        for c in self.classes:
            self.class_priors[c] = math.log(class_counts[c] / total)
            feature_sums = [0.0] * n_features
            for xi, yi in zip(X, y):
                if yi == c:
                    for j in range(n_features):
                        feature_sums[j] += xi[j]
            denom = sum(feature_sums) + self.alpha * n_features
            self.feature_log_prob[c] = [
                math.log((feature_sums[j] + self.alpha) / denom)
                for j in range(n_features)
            ]

    def predict_one(self, x):
        best, best_score = None, float("-inf")
        for c in self.classes:
            score = self.class_priors[c]
            for j, xj in enumerate(x):
                if xj > 0:
                    score += xj * self.feature_log_prob[c][j]
            if score > best_score:
                best_score, best = score, c
        return best

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba_one(self, x):
        scores = {}
        for c in self.classes:
            s = self.class_priors[c]
            for j, xj in enumerate(x):
                if xj > 0:
                    s += xj * self.feature_log_prob[c][j]
            scores[c] = s
        mx = max(scores.values())
        exp_s = {c: math.exp(scores[c] - mx) for c in self.classes}
        tot = sum(exp_s.values())
        return {c: exp_s[c] / tot for c in self.classes}


# ───────────────────────────────────────────────
# Logistic Regression
# ───────────────────────────────────────────────

class LogisticRegressionClassifier:
    """Binary logistic regression trained with gradient descent."""

    name = "Logistic Regression"

    def __init__(self, lr=0.05, epochs=400, reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights = []
        self.bias = 0.0
        self.classes = []

    @staticmethod
    def _sigmoid(z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        self.classes = sorted(set(y))
        label_map = {self.classes[0]: 0.0, self.classes[1]: 1.0}
        y_bin = [label_map[yi] for yi in y]
        n_features = len(X[0]) if X else 0

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0
            n = len(X)
            for xi, yi in zip(X, y_bin):
                z = sum(xi[j] * self.weights[j] for j in range(n_features)) + self.bias
                error = self._sigmoid(z) - yi
                for j in range(n_features):
                    dw[j] += error * xi[j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n + self.reg * self.weights[j])
            self.bias -= self.lr * (db / n)

    def predict_one(self, x):
        z = sum(xj * wj for xj, wj in zip(x, self.weights)) + self.bias
        return self.classes[1] if self._sigmoid(z) >= 0.5 else self.classes[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba_one(self, x):
        z = sum(xj * wj for xj, wj in zip(x, self.weights)) + self.bias
        p1 = self._sigmoid(z)
        return {self.classes[0]: 1.0 - p1, self.classes[1]: p1}


# ───────────────────────────────────────────────
# K-Nearest Neighbours
# ───────────────────────────────────────────────

class KNNClassifier:
    """K-Nearest Neighbours with Euclidean distance."""

    name = "K-Nearest Neighbours"

    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []
        self.classes = []

    @staticmethod
    def _euclidean(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = sorted(set(y))

    def predict_one(self, x):
        dists = sorted(
            ((self._euclidean(x, xi), yi) for xi, yi in zip(self.X_train, self.y_train)),
            key=lambda d: d[0],
        )
        votes = Counter(d[1] for d in dists[: self.k])
        return votes.most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba_one(self, x):
        dists = sorted(
            ((self._euclidean(x, xi), yi) for xi, yi in zip(self.X_train, self.y_train)),
            key=lambda d: d[0],
        )
        votes = Counter(d[1] for d in dists[: self.k])
        total = sum(votes.values())
        return {c: votes.get(c, 0) / total for c in self.classes}


# --------------- registry ---------------

CLASSIFIERS = {
    "NaiveBayes": lambda: NaiveBayesClassifier(alpha=1.0),
    "LogisticRegression": lambda: LogisticRegressionClassifier(lr=0.05, epochs=400, reg=0.01),
    "KNN": lambda: KNNClassifier(k=5),
}
