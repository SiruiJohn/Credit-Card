from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import KNNScratch


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


def stratified_subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42):
    if len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos = max(1, int(max_samples * (len(idx_pos) / len(y))))
    n_neg = max(1, max_samples - n_pos)
    idx = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
    rng.shuffle(idx)
    return x[idx], y[idx]


def train_valid_split(x: np.ndarray, y: np.ndarray, valid_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_va = max(1, int(len(idx_pos) * valid_ratio))
    n_neg_va = max(1, int(len(idx_neg) * valid_ratio))
    va_idx = np.concatenate([idx_pos[:n_pos_va], idx_neg[:n_neg_va]])
    tr_idx = np.setdiff1d(np.arange(len(y)), va_idx, assume_unique=False)
    return x[tr_idx], y[tr_idx], x[va_idx], y[va_idx]


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray):
    best = {"thr": 0.5, "f1": -1.0, "recall": -1.0}
    for thr in np.arange(0.05, 1.0, 0.05):
        m = binary_metrics(y_true, y_score, float(thr))
        if m["recall"] >= 0.85 and m["f1"] > best["f1"]:
            best = {"thr": float(thr), "f1": m["f1"], "recall": m["recall"]}
        elif best["f1"] < 0 and m["recall"] > best["recall"]:
            best = {"thr": float(thr), "f1": m["f1"], "recall": m["recall"]}
    return float(best["thr"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)
    features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))

    x = df_main[features].to_numpy(dtype=float)
    y = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)

    x, y = stratified_subsample(x, y, max_samples=8000, seed=7)
    x_tr, y_tr, x_va, y_va = train_valid_split(x, y, valid_ratio=0.2, seed=42)
    x_tr, x_va = standardize_by_train(x_tr, x_va)
    _, x_ext_sc = standardize_by_train(x_tr, x_ext)
    x_ext_sc, y_ext_sc = stratified_subsample(x_ext_sc, y_ext, max_samples=5000, seed=99)

    k_values = [3, 5, 9, 15, 21]
    rows = []
    for k in k_values:
        model = KNNScratch(n_neighbors=k, max_train_samples=2000, chunk_size=256, random_state=42)
        model.fit(x_tr, y_tr)
        va_score = model.predict_proba(x_va)[:, 1]
        thr = choose_threshold(y_va, va_score)

        ext_score = model.predict_proba(x_ext_sc)[:, 1]
        m = binary_metrics(y_ext_sc, ext_score, thr)
        rows.append(
            {
                "k": k,
                "threshold": thr,
                "precision_ext": m["precision"],
                "recall_ext": m["recall"],
                "f1_ext": m["f1"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_ext", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "knn_tuning_results.csv", index=False)
    (OUT_DIR / "knn_tuning_results.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(df["k"], df["f1_ext"], marker="o", label="F1")
    plt.plot(df["k"], df["recall_ext"], marker="o", label="Recall")
    plt.plot(df["k"], df["precision_ext"], marker="o", label="Precision")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title("kNN Tuning on External Subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "knn_tuning_curve.png", dpi=180)
    plt.close()

    best = df.iloc[0].to_dict()
    summary = [
        "# kNN Tuning Summary",
        "",
        f"- Candidate k: {k_values}",
        f"- Best k by external F1: {int(best['k'])}",
        f"- Best threshold: {best['threshold']:.2f}",
        f"- External Precision: {best['precision_ext']:.4f}",
        f"- External Recall: {best['recall_ext']:.4f}",
        f"- External F1: {best['f1_ext']:.4f}",
    ]
    (OUT_DIR / "knn_tuning_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("Saved kNN tuning outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
