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
OUT_DIR = ROOT / "outputs" / "stage3"


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma


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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def add_cost_columns(df: pd.DataFrame, cost_pairs: list[tuple[int, int]]) -> pd.DataFrame:
    out = df.copy()
    for c_fp, c_fn in cost_pairs:
        col = f"cost_fp{c_fp}_fn{c_fn}"
        out[col] = c_fp * out["fp"] + c_fn * out["fn"]
    return out


def choose_best_threshold(df_main: pd.DataFrame, c_fp: int, c_fn: int) -> float:
    col = f"cost_fp{c_fp}_fn{c_fn}"
    best = df_main.sort_values([col, "f1"], ascending=[True, False]).iloc[0]
    return float(best["threshold"])


def plot_cost_curves(df: pd.DataFrame, cost_pairs: list[tuple[int, int]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for c_fp, c_fn in cost_pairs:
        col = f"cost_fp{c_fp}_fn{c_fn}"
        plt.plot(df["threshold"], df[col], label=f"FP:FN = {c_fp}:{c_fn}")
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
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

    x_main_sc, x_ext_sc = standardize_by_train(x_main, x_ext)

    pos_weight = (len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1)
    model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=float(pos_weight))
    model.fit(x_main_sc, y_main)

    score_main = model.predict_proba(x_main_sc)[:, 1]
    score_ext = model.predict_proba(x_ext_sc)[:, 1]

    thresholds = np.arange(0.01, 1.0, 0.01)
    rows_main = [binary_metrics(y_main, score_main, float(t)) for t in thresholds]
    rows_ext = [binary_metrics(y_ext, score_ext, float(t)) for t in thresholds]
    scan_main = pd.DataFrame(rows_main)
    scan_ext = pd.DataFrame(rows_ext)

    cost_pairs = [(1, 5), (1, 10), (1, 20)]
    scan_main = add_cost_columns(scan_main, cost_pairs)
    scan_ext = add_cost_columns(scan_ext, cost_pairs)

    scan_main.to_csv(OUT_DIR / "cost_scan_main.csv", index=False)
    scan_ext.to_csv(OUT_DIR / "cost_scan_ext2023.csv", index=False)

    plot_cost_curves(
        scan_main,
        cost_pairs,
        OUT_DIR / "cost_curve_main.png",
        "Cost-Sensitive Threshold Curve (Main Dataset)",
    )
    plot_cost_curves(
        scan_ext,
        cost_pairs,
        OUT_DIR / "cost_curve_ext2023.png",
        "Cost-Sensitive Threshold Curve (External 2023 Dataset)",
    )

    selected_rows = []
    summary_lines = ["# Cost-Sensitive Threshold Optimization Summary", ""]
    for c_fp, c_fn in cost_pairs:
        thr = choose_best_threshold(scan_main, c_fp, c_fn)
        m_main = binary_metrics(y_main, score_main, thr)
        m_ext = binary_metrics(y_ext, score_ext, thr)
        cost_col = f"cost_fp{c_fp}_fn{c_fn}"
        cost_main = int(c_fp * m_main["fp"] + c_fn * m_main["fn"])
        cost_ext = int(c_fp * m_ext["fp"] + c_fn * m_ext["fn"])
        selected_rows.append(
            {
                "cost_ratio_fp_fn": f"{c_fp}:{c_fn}",
                "selected_threshold_from_main": thr,
                "main_precision": m_main["precision"],
                "main_recall": m_main["recall"],
                "main_f1": m_main["f1"],
                "main_cost": cost_main,
                "ext_precision": m_ext["precision"],
                "ext_recall": m_ext["recall"],
                "ext_f1": m_ext["f1"],
                "ext_cost": cost_ext,
            }
        )
        summary_lines.extend(
            [
                f"## Cost ratio FP:FN = {c_fp}:{c_fn}",
                f"- Selected threshold (from main): {thr:.2f}",
                f"- Main: precision={m_main['precision']:.4f}, recall={m_main['recall']:.4f}, f1={m_main['f1']:.4f}, cost={cost_main}",
                f"- External: precision={m_ext['precision']:.4f}, recall={m_ext['recall']:.4f}, f1={m_ext['f1']:.4f}, cost={cost_ext}",
                "",
            ]
        )

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT_DIR / "cost_sensitive_selection.csv", index=False)
    (OUT_DIR / "cost_sensitive_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (OUT_DIR / "cost_sensitive_selection.json").write_text(
        json.dumps(selected_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved stage3 cost-sensitive outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
