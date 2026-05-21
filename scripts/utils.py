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


def f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_lift_curve(y_true: np.ndarray, y_score: np.ndarray, n_buckets: int = 10) -> dict:
    order = np.argsort(-y_score)
    y = y_true[order]
    total_positives = max(1, int(np.sum(y == 1)))
    baseline_rate = total_positives / max(1, len(y))
    bucket_size = len(y) // n_buckets
    buckets = []
    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < n_buckets - 1 else len(y)
        bucket_positives = int(np.sum(y[start:end] == 1))
        bucket_rate = bucket_positives / (end - start) if end > start else 0.0
        lift = bucket_rate / baseline_rate if baseline_rate > 0 else 0.0
        buckets.append({
            "bucket": i + 1,
            "samples": end - start,
            "positives": bucket_positives,
            "positive_rate": float(bucket_rate),
            "lift": float(lift),
        })
    return {"buckets": buckets, "baseline_rate": float(baseline_rate)}


def compute_cumulative_gain(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    order = np.argsort(-y_score)
    y = y_true[order]
    total_positives = max(1, int(np.sum(y == 1)))
    cumulative_tp = np.cumsum(y == 1)
    alerts = np.arange(1, len(y) + 1)
    recall = cumulative_tp / total_positives
    steps = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    result = []
    for frac in steps:
        k = max(1, int(frac * len(y)))
        result.append({
            "alert_fraction": float(frac),
            "alerts": k,
            "cumulative_recall": float(recall[k - 1]),
        })
    return result


def compute_amount_weighted_metrics(
    y_true: np.ndarray, y_score: np.ndarray, amount: np.ndarray, thr: float
) -> dict:
    y_pred = (y_score >= thr).astype(int)
    return compute_amount_weighted_metrics_from_predictions(y_true, y_pred, amount)


def compute_amount_weighted_metrics_from_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, amount: np.ndarray
) -> dict:
    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_amount = float(np.sum(amount * tp_mask.astype(float)))
    fp_amount = float(np.sum(amount * fp_mask.astype(float)))
    fn_amount = float(np.sum(amount * fn_mask.astype(float)))
    total_fraud_amount = tp_amount + fn_amount
    weighted_precision = tp_amount / (tp_amount + fp_amount) if (tp_amount + fp_amount) > 0 else 0.0
    weighted_recall = tp_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
    weighted_f1 = (
        (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall))
        if (weighted_precision + weighted_recall) > 0
        else 0.0
    )
    return {
        "amount_weighted_precision": float(weighted_precision),
        "amount_weighted_recall": float(weighted_recall),
        "amount_weighted_f1": float(weighted_f1),
        "tp_amount": tp_amount,
        "fp_amount": fp_amount,
        "fn_amount": fn_amount,
    }


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> dict:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_results = []
    total = len(y_true)
    ece_sum = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            bin_results.append({"bin": i + 1, "range": [float(lo), float(hi)], "count": 0, "accuracy": 0.0, "confidence": 0.0, "ece_contrib": 0.0})
            continue
        bin_accuracy = float(np.mean(y_true[mask].astype(float)))
        bin_confidence = float(np.mean(y_score[mask]))
        ece_contrib = n_bin / total * abs(bin_accuracy - bin_confidence)
        ece_sum += ece_contrib
        bin_results.append({
            "bin": i + 1,
            "range": [float(lo), float(hi)],
            "count": n_bin,
            "accuracy": float(bin_accuracy),
            "confidence": float(bin_confidence),
            "ece_contrib": float(ece_contrib),
        })
    return {"ece": float(ece_sum), "bins": bin_results}


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    eps = 1e-10
    lo, hi = float(np.min(expected)), float(np.max(expected))
    bins = np.linspace(lo, hi, n_bins + 1)
    expected_counts = np.zeros(n_bins, dtype=float)
    actual_counts = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        if i == n_bins - 1:
            expected_counts[i] = float(np.sum((expected >= bins[i]) & (expected <= bins[i + 1])))
            actual_counts[i] = float(np.sum((actual >= bins[i]) & (actual <= bins[i + 1])))
        else:
            expected_counts[i] = float(np.sum((expected >= bins[i]) & (expected < bins[i + 1])))
            actual_counts[i] = float(np.sum((actual >= bins[i]) & (actual < bins[i + 1])))
    expected_rate = expected_counts / max(1, len(expected))
    actual_rate = actual_counts / max(1, len(actual))
    expected_rate = np.maximum(expected_rate, eps)
    actual_rate = np.maximum(actual_rate, eps)
    psi = float(np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)))
    return psi
