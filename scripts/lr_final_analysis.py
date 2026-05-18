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
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma, mu, sigma


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float):
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


def choose_threshold_fine(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.85):
    candidates = np.arange(0.01, 1.00, 0.01)
    rows = []
    for thr in candidates:
        m = binary_metrics(y_true, y_score, float(thr))
        rows.append(m)
    df = pd.DataFrame(rows)

    feasible = df[df["recall"] >= target_recall]
    if len(feasible) > 0:
        best = feasible.sort_values(["f1", "precision"], ascending=False).iloc[0]
    else:
        best = df.sort_values(["recall", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), df


def plot_threshold(df: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion_matrix(cm: dict, out_path: Path, title: str) -> None:
    values = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    plt.figure(figsize=(5, 4))
    plt.imshow(values, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels[i, j]}\n{values[i, j]:,}", ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)

    features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    x_main = df_main[features].to_numpy(dtype=float)
    y_main = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)

    x_main_sc, x_ext_sc, _, _ = standardize_by_train(x_main, x_ext)

    pos_weight = (len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1)
    model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=float(pos_weight))
    model.fit(x_main_sc, y_main)

    # Feature importance from absolute LR weights
    if model.w is None:
        raise RuntimeError("LR model weights not available.")
    fi = pd.DataFrame({"feature": features, "coef": model.w, "abs_coef": np.abs(model.w)})
    fi = fi.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    fi.to_csv(OUT_DIR / "lr_feature_importance.csv", index=False)

    topn = 15 if len(fi) >= 15 else len(fi)
    top = fi.head(topn).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["abs_coef"])
    plt.xlabel("|Coefficient|")
    plt.title("LR Feature Importance (Top by |coef|)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lr_feature_importance_top15.png", dpi=180)
    plt.close()

    # Fine-grained threshold selection
    score_main = model.predict_proba(x_main_sc)[:, 1]
    score_ext = model.predict_proba(x_ext_sc)[:, 1]
    best_thr, df_scan_main = choose_threshold_fine(y_main, score_main, target_recall=0.85)
    df_scan_ext = pd.DataFrame([binary_metrics(y_ext, score_ext, float(t)) for t in np.arange(0.01, 1.00, 0.01)])

    df_scan_main.to_csv(OUT_DIR / "lr_threshold_scan_main_001.csv", index=False)
    df_scan_ext.to_csv(OUT_DIR / "lr_threshold_scan_ext_001.csv", index=False)
    plot_threshold(df_scan_main, OUT_DIR / "lr_threshold_scan_main_001.png", "LR Threshold Scan (Main, step=0.01)")
    plot_threshold(df_scan_ext, OUT_DIR / "lr_threshold_scan_ext_001.png", "LR Threshold Scan (External, step=0.01)")

    final_ext = binary_metrics(y_ext, score_ext, best_thr)
    cm_json = {
        "selected_threshold": best_thr,
        "confusion_matrix_ext2023": {
            "tp": final_ext["tp"],
            "tn": final_ext["tn"],
            "fp": final_ext["fp"],
            "fn": final_ext["fn"],
        },
        "precision": final_ext["precision"],
        "recall": final_ext["recall"],
        "f1": final_ext["f1"],
        "accuracy": final_ext["accuracy"],
    }
    plot_confusion_matrix(
        cm_json["confusion_matrix_ext2023"],
        OUT_DIR / "lr_confusion_matrix_ext2023_thr087.png",
        "LR Confusion Matrix on External 2023 (thr=0.87)",
    )
    (OUT_DIR / "lr_final_confusion_matrix_ext2023.json").write_text(
        json.dumps(cm_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = [
        "# LR Final Analysis Summary",
        "",
        f"- Selected threshold (step=0.01): {best_thr:.2f}",
        f"- External Precision: {final_ext['precision']:.4f}",
        f"- External Recall: {final_ext['recall']:.4f}",
        f"- External F1: {final_ext['f1']:.4f}",
        f"- External Accuracy: {final_ext['accuracy']:.4f}",
        f"- Top 5 important features: {', '.join(fi.head(5)['feature'].tolist())}",
    ]
    (OUT_DIR / "lr_final_analysis_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("Saved LR final analysis outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
