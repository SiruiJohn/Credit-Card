from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_from_scratch import GaussianNBScratch, KNNScratch, LogisticRegressionScratch


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


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


def standardize_by_train(x_train: np.ndarray, x_valid: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_valid - mu) / sigma


def stratified_subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate_model_cv(name: str, x: np.ndarray, y: np.ndarray, quick: bool) -> dict:
    n_splits = 2 if quick else 5
    fold_results = []
    start_all = time.perf_counter()

    for fold_id, (tr_idx, va_idx) in enumerate(stratified_kfold_indices(y, n_splits=n_splits), start=1):
        x_tr, x_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        x_tr, x_va = standardize_by_train(x_tr, x_va)

        if name == "logistic_regression":
            pos_weight = (len(y_tr) - np.sum(y_tr)) / max(np.sum(y_tr), 1)
            model = LogisticRegressionScratch(
                lr=0.05, max_iter=(60 if quick else 180), l2=1e-4, pos_weight=float(pos_weight)
            )
        elif name == "gaussian_nb":
            model = GaussianNBScratch(var_smoothing=1e-9)
        elif name == "knn":
            # kNN is O(N_train * N_eval); use stratified subsampling for practical runtime.
            x_tr, y_tr = stratified_subsample(x_tr, y_tr, max_samples=(4000 if quick else 8000), seed=42 + fold_id)
            x_va, y_va = stratified_subsample(x_va, y_va, max_samples=(5000 if quick else 12000), seed=84 + fold_id)
            model = KNNScratch(
                n_neighbors=(9 if quick else 15),
                max_train_samples=None,
                chunk_size=256,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        t0 = time.perf_counter()
        model.fit(x_tr, y_tr)
        train_seconds = time.perf_counter() - t0
        score_va = model.predict_proba(x_va)[:, 1]

        thr = choose_threshold(y_va, score_va)
        m = binary_metrics(y_va, score_va, thr)
        m["roc_auc"] = roc_auc_score_manual(y_va, score_va)
        m["pr_auc"] = pr_auc_score_manual(y_va, score_va)
        m["train_seconds"] = float(train_seconds)
        m["fold"] = fold_id
        fold_results.append(m)

    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_seconds"]
    summary = {}
    for k in keys:
        vals = np.array([fr[k] for fr in fold_results], dtype=float)
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    summary["n_splits"] = n_splits
    summary["total_seconds"] = float(time.perf_counter() - start_all)

    return {"model": name, "fold_results": fold_results, "summary": summary}


def train_on_full(name: str, x: np.ndarray, y: np.ndarray, quick: bool):
    x_train, x_apply = standardize_by_train(x, x)
    if name == "logistic_regression":
        pos_weight = (len(y) - np.sum(y)) / max(np.sum(y), 1)
        model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=float(pos_weight))
    elif name == "gaussian_nb":
        model = GaussianNBScratch(var_smoothing=1e-9)
    elif name == "knn":
        model = KNNScratch(
            n_neighbors=(9 if quick else 15),
            max_train_samples=(3000 if quick else 5000),
            chunk_size=256,
            random_state=42,
        )
    else:
        raise ValueError(name)
    model.fit(x_train, y)
    return model, x_apply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 2-fold quick baseline for faster run.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)

    common_features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    x_main = df_main[common_features].to_numpy(dtype=float)
    y_main = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[common_features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)

    all_results = {
        "config": {
            "quick_mode": bool(args.quick),
            "common_features_count": len(common_features),
            "common_features_sample": common_features[:10],
        },
        "cv": {},
        "external_eval": {},
    }

    for model_name in ["logistic_regression", "gaussian_nb", "knn"]:
        cv_res = evaluate_model_cv(model_name, x_main, y_main, quick=args.quick)
        all_results["cv"][model_name] = cv_res

        model, x_main_scaled = train_on_full(model_name, x_main, y_main, quick=args.quick)
        x_ext_scaled = (x_ext - np.mean(x_main, axis=0)) / np.where(np.std(x_main, axis=0) == 0, 1.0, np.std(x_main, axis=0))
        score_main = model.predict_proba(x_main_scaled)[:, 1]
        if model_name == "knn":
            # External dataset is very large; evaluate kNN on a stratified subset to keep runtime feasible.
            x_ext_eval, y_ext_eval = stratified_subsample(
                x_ext_scaled,
                y_ext,
                max_samples=(5000 if args.quick else 10000),
                seed=123,
            )
            score_ext = model.predict_proba(x_ext_eval)[:, 1]
            y_ext_used = y_ext_eval
        else:
            score_ext = model.predict_proba(x_ext_scaled)[:, 1]
            y_ext_used = y_ext
        thr = choose_threshold(y_main, score_main)
        ext_metrics = binary_metrics(y_ext_used, score_ext, thr)
        ext_metrics["roc_auc"] = roc_auc_score_manual(y_ext_used, score_ext)
        ext_metrics["pr_auc"] = pr_auc_score_manual(y_ext_used, score_ext)
        if model_name == "knn":
            ext_metrics["note"] = f"Evaluated on stratified external subset (n={len(y_ext_used)}) for runtime feasibility."
        all_results["external_eval"][model_name] = ext_metrics

    out_json = OUT_DIR / "phase2_baseline_results.json"
    out_md = OUT_DIR / "phase2_baseline_summary.md"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Phase2 Baseline Summary", "", f"- quick_mode: {args.quick}", f"- common_features: {len(common_features)}", ""]
    for m in ["logistic_regression", "gaussian_nb", "knn"]:
        s = all_results["cv"][m]["summary"]
        ext = all_results["external_eval"][m]
        md += [
            f"## {m}",
            f"- CV Recall(mean): {s['recall_mean']:.4f}",
            f"- CV F1(mean): {s['f1_mean']:.4f}",
            f"- CV PR-AUC(mean): {s['pr_auc_mean']:.4f}",
            f"- External Recall: {ext['recall']:.4f}",
            f"- External F1: {ext['f1']:.4f}",
            f"- External PR-AUC: {ext['pr_auc']:.4f}",
            f"- Note: {ext.get('note', 'Full external set evaluation.')}",
            "",
        ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
