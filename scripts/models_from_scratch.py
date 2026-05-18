from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(
        self,
        lr: float = 0.05,
        max_iter: int = 200,
        l2: float = 1e-4,
        pos_weight: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        n_samples, n_features = x.shape
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(0.0, 0.01, size=n_features)
        self.b = 0.0

        y = y.astype(float)
        sample_weight = np.where(y == 1.0, self.pos_weight, 1.0)

        for _ in range(self.max_iter):
            logits = x @ self.w + self.b
            p = sigmoid(logits)
            err = (p - y) * sample_weight

            grad_w = (x.T @ err) / n_samples + self.l2 * self.w
            grad_b = float(np.mean(err))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not fitted.")
        logits = x @ self.w + self.b
        p1 = sigmoid(logits)
        return np.column_stack([1.0 - p1, p1])


class GaussianNBScratch:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: dict[int, float] = {}
        self.theta_: dict[int, np.ndarray] = {}
        self.var_: dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNBScratch":
        self.classes_ = np.unique(y)
        n_samples = len(y)
        eps = self.var_smoothing * np.var(x, axis=0).max()

        for c in self.classes_:
            x_c = x[y == c]
            self.class_log_prior_[int(c)] = np.log(len(x_c) / n_samples)
            self.theta_[int(c)] = np.mean(x_c, axis=0)
            self.var_[int(c)] = np.var(x_c, axis=0) + eps
        return self

    def _joint_log_likelihood(self, x: np.ndarray, c: int) -> np.ndarray:
        mean = self.theta_[c]
        var = self.var_[c]
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        log_prob -= 0.5 * np.sum(((x - mean) ** 2) / var, axis=1)
        return self.class_log_prior_[c] + log_prob

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
        jll = np.column_stack([self._joint_log_likelihood(x, int(c)) for c in self.classes_])
        m = np.max(jll, axis=1, keepdims=True)
        p = np.exp(jll - m)
        p = p / np.sum(p, axis=1, keepdims=True)
        return p


class KNNScratch:
    def __init__(
        self,
        n_neighbors: int = 5,
        distance: str = "euclidean",
        weights: str = "uniform",
        max_train_samples: int | None = None,
        random_state: int = 42,
        chunk_size: int = 256,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.weights = weights
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def _subsample_train(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.max_train_samples is None or len(y) <= self.max_train_samples:
            return x, y

        rng = np.random.default_rng(self.random_state)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)

        n_pos_target = max(1, int(self.max_train_samples * (len(idx_pos) / len(y))))
        n_neg_target = max(1, self.max_train_samples - n_pos_target)

        picked = np.concatenate([idx_pos[:n_pos_target], idx_neg[:n_neg_target]])
        rng.shuffle(picked)
        return x[picked], y[picked]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNScratch":
        x_sub, y_sub = self._subsample_train(x, y)
        self.x_train = x_sub.astype(float)
        self.y_train = y_sub.astype(int)
        return self

    def _pairwise_distance(self, x_chunk: np.ndarray) -> np.ndarray:
        if self.x_train is None:
            raise RuntimeError("Model is not fitted.")
        if self.distance != "euclidean":
            raise ValueError("Only euclidean distance is supported currently.")

        # Efficient squared euclidean distance:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        a2 = np.sum(x_chunk**2, axis=1, keepdims=True)
        b2 = np.sum(self.x_train**2, axis=1, keepdims=True).T
        d2 = np.maximum(a2 + b2 - 2.0 * (x_chunk @ self.x_train.T), 0.0)
        return np.sqrt(d2)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted.")

        n = len(x)
        out = np.zeros((n, 2), dtype=float)
        k = min(self.n_neighbors, len(self.y_train))

        for i in range(0, n, self.chunk_size):
            j = min(i + self.chunk_size, n)
            d = self._pairwise_distance(x[i:j])
            nn_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            nn_y = self.y_train[nn_idx]
            if self.weights == "uniform":
                p1 = np.mean(nn_y == 1, axis=1)
            elif self.weights == "distance":
                eps = 1e-12
                nn_d = np.take_along_axis(d, nn_idx, axis=1)
                w = 1.0 / (nn_d + eps)
                p1 = np.sum(w * (nn_y == 1), axis=1) / np.sum(w, axis=1)
            else:
                raise ValueError("weights must be 'uniform' or 'distance'.")
            out[i:j, 1] = p1
            out[i:j, 0] = 1.0 - p1
        return out
