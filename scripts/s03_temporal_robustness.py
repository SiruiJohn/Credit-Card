from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import LogisticRegressionScratch


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage3"


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.85) -> float:
    best_thr = 0.5
    best_key = (-1.0, -1.0)  # (f1, precision)
    for thr in np.arange(0.01, 1.0, 0.01):
        m = binary_metrics(y_true, y_score, float(thr))
        if m["recall"] >= target_recall:
            key = (m["f1"], m["precision"])
            if key > best_key:
                best_key = key
                best_thr = float(thr)
    if best_key[0] < 0:
        # fallback: maximize recall first, then f1
        best_thr = 0.5
        best_rec, best_f1 = -1.0, -1.0
        for thr in np.arange(0.01, 1.0, 0.01):
            m = binary_metrics(y_true, y_score, float(thr))
            if m["recall"] > best_rec or (m["recall"] == best_rec and m["f1"] > best_f1):
                best_rec = m["recall"]
                best_f1 = m["f1"]
                best_thr = float(thr)
    return best_thr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_MAIN)
    if "Time" not in df.columns:
        raise ValueError("Time column is required for temporal robustness analysis.")

    df = df.sort_values("Time").reset_index(drop=True)
    # 5 chronological windows with similar sample sizes
    df["time_window"] = pd.qcut(df["Time"], q=5, labels=False, duplicates="drop") + 1

    feature_cols = [c for c in df.columns if c not in {"Class", "time_window"}]
    rows = []

    for window_id in sorted(df["time_window"].unique()):
        if int(window_id) == 1:
            continue
        train_df = df[df["time_window"] < window_id]
        test_df = df[df["time_window"] == window_id]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["Class"].to_numpy(dtype=int)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["Class"].to_numpy(dtype=int)

        x_train_sc, x_test_sc = standardize(x_train, x_test)
        pos = np.sum(y_train)
        pos_weight = (len(y_train) - pos) / max(pos, 1)
        model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=float(pos_weight))
        model.fit(x_train_sc, y_train)

        score_train = model.predict_proba(x_train_sc)[:, 1]
        score_test = model.predict_proba(x_test_sc)[:, 1]
        thr = choose_threshold(y_train, score_train, target_recall=0.85)
        met = binary_metrics(y_test, score_test, thr)
        rows.append(
            {
                "window": int(window_id),
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "fraud_rate_test": float(np.mean(y_test)),
                "threshold": float(thr),
                "precision": met["precision"],
                "recall": met["recall"],
                "f1": met["f1"],
                "tp": met["tp"],
                "tn": met["tn"],
                "fp": met["fp"],
                "fn": met["fn"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "temporal_robustness_windows.csv", index=False)

    # Plot temporal metrics
    plt.figure(figsize=(8, 4.5))
    plt.plot(result_df["window"], result_df["precision"], marker="o", label="Precision")
    plt.plot(result_df["window"], result_df["recall"], marker="o", label="Recall")
    plt.plot(result_df["window"], result_df["f1"], marker="o", label="F1")
    plt.xlabel("Time Window (chronological)")
    plt.ylabel("Score")
    plt.title("Temporal Robustness (Expanding-Window Evaluation)")
    plt.xticks(result_df["window"])
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "temporal_robustness_metrics.png", dpi=180)
    plt.close()

    # Plot threshold drift
    plt.figure(figsize=(7, 4))
    plt.plot(result_df["window"], result_df["threshold"], marker="o")
    plt.xlabel("Time Window (chronological)")
    plt.ylabel("Selected Threshold")
    plt.title("Threshold Drift Across Time Windows")
    plt.xticks(result_df["window"])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "temporal_threshold_drift.png", dpi=180)
    plt.close()

    summary = [
        "# Temporal Robustness Summary",
        "",
        f"- Evaluated windows: {len(result_df)}",
        f"- Mean Precision: {result_df['precision'].mean():.4f}",
        f"- Mean Recall: {result_df['recall'].mean():.4f}",
        f"- Mean F1: {result_df['f1'].mean():.4f}",
        f"- Recall std: {result_df['recall'].std(ddof=0):.4f}",
        f"- F1 std: {result_df['f1'].std(ddof=0):.4f}",
        "",
        "## Window-wise metrics",
    ]
    for _, r in result_df.iterrows():
        summary.append(
            f"- W{int(r['window'])}: thr={r['threshold']:.2f}, precision={r['precision']:.4f}, recall={r['recall']:.4f}, f1={r['f1']:.4f}, fraud_rate={r['fraud_rate_test']:.5f}"
        )

    (OUT_DIR / "temporal_robustness_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "temporal_robustness_windows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved temporal robustness outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
