from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.arange(0.05, 1.00, 0.05)
    best_thr = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    for thr in candidates:
        m = binary_metrics(y_true, y_score, thr)
        if m["recall"] >= 0.85:
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thr = float(thr)
        elif best_f1 < 0 and m["recall"] > best_recall:
            best_recall = m["recall"]
            best_thr = float(thr)
    return best_thr


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    p = max(1, int(np.sum(y == 1)))
    n = max(1, int(np.sum(y == 0)))
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tp / p, [1.0]])
    fpr = np.concatenate([[0.0], fp / n, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def pr_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    p = max(1, int(np.sum(y == 1)))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / p
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    pos_folds = np.array_split(idx_pos, n_splits)
    neg_folds = np.array_split(idx_neg, n_splits)
    for i in range(n_splits):
        test_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
        yield train_idx, test_idx


def stratified_subsample(
    x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_target = max(1, int(max_samples * (len(idx_pos) / len(y))))
    n_neg_target = max(1, max_samples - n_pos_target)
    picked = np.concatenate([idx_pos[:n_pos_target], idx_neg[:n_neg_target]])
    rng.shuffle(picked)
    return x[picked], y[picked]


def standardize_by_train(x_train: np.ndarray, x_valid: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_valid - mu) / sigma
