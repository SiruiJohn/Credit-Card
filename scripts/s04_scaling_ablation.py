from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import KNNScratch, LogisticRegressionScratch
from utils import binary_metrics, choose_threshold, pr_auc_score_manual, roc_auc_score_manual, stratified_subsample


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage4"


def fit_scaler(x_train: np.ndarray, method: str) -> dict:
    if method == "standard":
        center = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
    elif method == "minmax":
        center = np.min(x_train, axis=0)
        scale = np.max(x_train, axis=0) - center
    elif method == "robust":
        center = np.median(x_train, axis=0)
        q75 = np.percentile(x_train, 75, axis=0)
        q25 = np.percentile(x_train, 25, axis=0)
        scale = q75 - q25
    else:
        raise ValueError(f"Unknown scaler: {method}")
    scale = np.where(scale == 0, 1.0, scale)
    return {"method": method, "center": center, "scale": scale}


def transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    return (x - scaler["center"]) / scaler["scale"]


def eval_lr(x_main: np.ndarray, y_main: np.ndarray, x_ext: np.ndarray, y_ext: np.ndarray, method: str) -> dict:
    scaler = fit_scaler(x_main, method)
    x_main_sc = transform(x_main, scaler)
    x_ext_sc = transform(x_ext, scaler)

    pos_weight = float((len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1))
    model = LogisticRegressionScratch(lr=0.05, max_iter=160, l2=1e-4, pos_weight=pos_weight)
    model.fit(x_main_sc, y_main)
    score_main = model.predict_proba(x_main_sc)[:, 1]
    score_ext = model.predict_proba(x_ext_sc)[:, 1]
    thr = choose_threshold(y_main, score_main)
    metrics = binary_metrics(y_ext, score_ext, thr)
    metrics["roc_auc"] = roc_auc_score_manual(y_ext, score_ext)
    metrics["pr_auc"] = pr_auc_score_manual(y_ext, score_ext)
    return metrics


def eval_knn(x_main: np.ndarray, y_main: np.ndarray, x_ext: np.ndarray, y_ext: np.ndarray, method: str) -> dict:
    scaler = fit_scaler(x_main, method)
    x_main_sc = transform(x_main, scaler)
    x_ext_sc = transform(x_ext, scaler)

    x_tr, y_tr = stratified_subsample(x_main_sc, y_main, max_samples=5000, seed=211)
    x_ev, y_ev = stratified_subsample(x_ext_sc, y_ext, max_samples=10000, seed=311)
    model = KNNScratch(n_neighbors=9, weights="distance", chunk_size=256, random_state=42)
    model.fit(x_tr, y_tr)
    score_train = model.predict_proba(x_tr)[:, 1]
    score_ext = model.predict_proba(x_ev)[:, 1]
    thr = choose_threshold(y_tr, score_train)
    metrics = binary_metrics(y_ev, score_ext, thr)
    metrics["roc_auc"] = roc_auc_score_manual(y_ev, score_ext)
    metrics["pr_auc"] = pr_auc_score_manual(y_ev, score_ext)
    metrics["note"] = f"kNN evaluated on stratified external subset n={len(y_ev)}"
    return metrics


def plot_results(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model in zip(axes, ["logistic_regression", "knn_distance"]):
        sub = df[df["model"] == model]
        x = np.arange(len(sub))
        width = 0.25
        ax.bar(x - width, sub["precision"], width=width, label="Precision", color="#4C78A8")
        ax.bar(x, sub["recall"], width=width, label="Recall", color="#F58518")
        ax.bar(x + width, sub["f1"], width=width, label="F1", color="#54A24B")
        ax.set_xticks(x, sub["scaler"])
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    fig.suptitle("Scaling Ablation Inspired by Notebook Preprocessing")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scaling_ablation_compare.png", dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)
    features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    x_main = df_main[features].to_numpy(dtype=float)
    y_main = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)

    rows = []
    for method in ["standard", "minmax", "robust"]:
        for model_name, evaluator in [
            ("logistic_regression", eval_lr),
            ("knn_distance", eval_knn),
        ]:
            metrics = evaluator(x_main, y_main, x_ext, y_ext, method)
            rows.append({"scaler": method, "model": model_name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "scaling_ablation_results.csv", index=False)
    (OUT_DIR / "scaling_ablation_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(df)

    md = ["# Scaling Ablation", "", "Inspired by the reference notebooks' scaler comparisons.", ""]
    for _, row in df.sort_values(["model", "f1"], ascending=[True, False]).iterrows():
        md.append(
            f"- {row['model']} + {row['scaler']}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, pr_auc={row['pr_auc']:.4f}, thr={row['threshold']:.2f}"
        )
    (OUT_DIR / "scaling_ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved scaling ablation outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
