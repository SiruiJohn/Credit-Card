from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_from_scratch import GaussianNBScratch, KNNScratch, LogisticRegressionScratch
from utils import (
    binary_metrics,
    choose_threshold,
    pr_auc_score_manual,
    roc_auc_score_manual,
    standardize_by_train,
    stratified_kfold_indices,
    stratified_subsample,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


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
