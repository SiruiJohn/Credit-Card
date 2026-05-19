from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import LogisticRegressionScratch
from utils import binary_metrics, pr_auc_score_manual, roc_auc_score_manual, stratified_kfold_indices


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage5"

SCALERS = ["standard", "minmax"]
COST_PAIRS = [(1, 5), (1, 10), (1, 20)]


def fit_scaler(x_train: np.ndarray, method: str) -> dict:
    if method == "standard":
        center = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
    elif method == "minmax":
        center = np.min(x_train, axis=0)
        scale = np.max(x_train, axis=0) - center
    else:
        raise ValueError(f"Unknown scaler: {method}")
    scale = np.where(scale == 0, 1.0, scale)
    return {"center": center, "scale": scale}


def transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    return (x - scaler["center"]) / scaler["scale"]


def choose_threshold_from_scan(scan: pd.DataFrame, target_recall: float = 0.85) -> float:
    feasible = scan[scan["recall"] >= target_recall]
    if len(feasible) > 0:
        best = feasible.sort_values(["f1", "precision"], ascending=False).iloc[0]
    else:
        best = scan.sort_values(["recall", "f1"], ascending=False).iloc[0]
    return float(best["threshold"])


def threshold_scan(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    rows = [binary_metrics(y_true, y_score, float(t)) for t in np.arange(0.01, 1.00, 0.01)]
    return pd.DataFrame(rows)


def add_auc(metrics: dict, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = dict(metrics)
    out["roc_auc"] = roc_auc_score_manual(y_true, y_score)
    out["pr_auc"] = pr_auc_score_manual(y_true, y_score)
    return out


def fit_lr(x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 180) -> LogisticRegressionScratch:
    pos_weight = float((len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1))
    model = LogisticRegressionScratch(lr=0.05, max_iter=max_iter, l2=1e-4, pos_weight=pos_weight)
    model.fit(x_train, y_train)
    return model


def make_oof_scores(x: np.ndarray, y: np.ndarray, scaler_name: str) -> tuple[np.ndarray, list[dict]]:
    oof_score = np.zeros(len(y), dtype=float)
    fold_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(stratified_kfold_indices(y, n_splits=5, seed=42), start=1):
        x_tr_raw, x_va_raw = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        scaler = fit_scaler(x_tr_raw, scaler_name)
        x_tr = transform(x_tr_raw, scaler)
        x_va = transform(x_va_raw, scaler)

        t0 = time.perf_counter()
        model = fit_lr(x_tr, y_tr)
        train_seconds = time.perf_counter() - t0
        score_va = model.predict_proba(x_va)[:, 1]
        oof_score[va_idx] = score_va

        scan = threshold_scan(y_va, score_va)
        fold_thr = choose_threshold_from_scan(scan)
        m = add_auc(binary_metrics(y_va, score_va, fold_thr), y_va, score_va)
        m.update({"fold": fold_id, "fold_threshold": fold_thr, "train_seconds": train_seconds})
        fold_rows.append(m)

    return oof_score, fold_rows


def train_full_and_score_ext(
    x_main: np.ndarray,
    y_main: np.ndarray,
    x_ext: np.ndarray,
    scaler_name: str,
) -> np.ndarray:
    scaler = fit_scaler(x_main, scaler_name)
    x_main_sc = transform(x_main, scaler)
    x_ext_sc = transform(x_ext, scaler)
    model = fit_lr(x_main_sc, y_main)
    return model.predict_proba(x_ext_sc)[:, 1]


def cost_from_metrics(metrics: dict, c_fp: int, c_fn: int) -> int:
    return int(c_fp * metrics["fp"] + c_fn * metrics["fn"])


def cost_selection(scan: pd.DataFrame, y_ext: np.ndarray, score_ext: np.ndarray, scaler_name: str) -> list[dict]:
    rows = []
    for c_fp, c_fn in COST_PAIRS:
        tmp = scan.copy()
        tmp["cost"] = c_fp * tmp["fp"] + c_fn * tmp["fn"]
        best = tmp.sort_values(["cost", "f1"], ascending=[True, False]).iloc[0]
        thr = float(best["threshold"])
        m_ext = binary_metrics(y_ext, score_ext, thr)
        rows.append(
            {
                "scaler": scaler_name,
                "cost_ratio_fp_fn": f"{c_fp}:{c_fn}",
                "selected_threshold_from_oof": thr,
                "oof_cost": int(best["cost"]),
                "external_precision": m_ext["precision"],
                "external_recall": m_ext["recall"],
                "external_f1": m_ext["f1"],
                "external_cost": cost_from_metrics(m_ext, c_fp, c_fn),
            }
        )
    return rows


def plot_metric_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.22
    ax.bar(x - width, df["external_precision"], width=width, label="Precision", color="#4C78A8")
    ax.bar(x, df["external_recall"], width=width, label="Recall", color="#F58518")
    ax.bar(x + width, df["external_f1"], width=width, label="F1", color="#54A24B")
    ax.set_xticks(x, df["scaler"])
    ax.set_ylim(0, 1.05)
    ax.set_title("OOF-Selected Threshold External Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "oof_minmax_external_compare.png", dpi=180)
    plt.close()


def plot_threshold_curves(scan: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(scan["threshold"], scan["precision"], label="Precision")
    plt.plot(scan["threshold"], scan["recall"], label="Recall")
    plt.plot(scan["threshold"], scan["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(metrics: dict, out_path: Path, title: str) -> None:
    values = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    plt.figure(figsize=(5, 4))
    plt.imshow(values, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels[i, j]}\n{values[i, j]:,}", ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
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

    comparison_rows = []
    all_fold_rows = []
    all_cost_rows = []

    for scaler_name in SCALERS:
        oof_score, fold_rows = make_oof_scores(x_main, y_main, scaler_name)
        for row in fold_rows:
            all_fold_rows.append({"scaler": scaler_name, **row})

        score_ext = train_full_and_score_ext(x_main, y_main, x_ext, scaler_name)
        scan_oof = threshold_scan(y_main, oof_score)
        scan_ext = threshold_scan(y_ext, score_ext)
        selected_thr = choose_threshold_from_scan(scan_oof)

        oof_metrics = add_auc(binary_metrics(y_main, oof_score, selected_thr), y_main, oof_score)
        ext_metrics = add_auc(binary_metrics(y_ext, score_ext, selected_thr), y_ext, score_ext)

        scan_oof.to_csv(OUT_DIR / f"oof_threshold_scan_{scaler_name}.csv", index=False)
        scan_ext.to_csv(OUT_DIR / f"external_threshold_scan_{scaler_name}.csv", index=False)
        plot_threshold_curves(scan_oof, OUT_DIR / f"oof_threshold_scan_{scaler_name}.png", f"OOF Threshold Scan ({scaler_name})")
        plot_threshold_curves(
            scan_ext,
            OUT_DIR / f"external_threshold_scan_{scaler_name}.png",
            f"External Threshold Scan ({scaler_name})",
        )
        plot_confusion(
            ext_metrics,
            OUT_DIR / f"external_confusion_{scaler_name}_oof_threshold.png",
            f"External Confusion Matrix ({scaler_name}, thr={selected_thr:.2f})",
        )

        comparison_rows.append(
            {
                "scaler": scaler_name,
                "selected_threshold_from_oof": selected_thr,
                "oof_precision": oof_metrics["precision"],
                "oof_recall": oof_metrics["recall"],
                "oof_f1": oof_metrics["f1"],
                "oof_pr_auc": oof_metrics["pr_auc"],
                "external_precision": ext_metrics["precision"],
                "external_recall": ext_metrics["recall"],
                "external_f1": ext_metrics["f1"],
                "external_pr_auc": ext_metrics["pr_auc"],
                "external_tp": ext_metrics["tp"],
                "external_tn": ext_metrics["tn"],
                "external_fp": ext_metrics["fp"],
                "external_fn": ext_metrics["fn"],
            }
        )
        all_cost_rows.extend(cost_selection(scan_oof, y_ext, score_ext, scaler_name))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "oof_minmax_comparison.csv", index=False)
    (OUT_DIR / "oof_minmax_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(all_fold_rows).to_csv(OUT_DIR / "oof_fold_metrics.csv", index=False)
    pd.DataFrame(all_cost_rows).to_csv(OUT_DIR / "oof_cost_sensitive_selection.csv", index=False)
    plot_metric_comparison(comparison_df)

    best = comparison_df.sort_values("external_f1", ascending=False).iloc[0]
    md = [
        "# OOF Threshold and MinMax Mainline Review",
        "",
        "Thresholds are selected from 5-fold out-of-fold predictions on the main dataset, then transferred to the full external 2023 dataset.",
        "",
        "## External performance",
    ]
    for _, row in comparison_df.iterrows():
        md.append(
            f"- {row['scaler']}: thr={row['selected_threshold_from_oof']:.2f}, "
            f"precision={row['external_precision']:.4f}, recall={row['external_recall']:.4f}, "
            f"f1={row['external_f1']:.4f}, pr_auc={row['external_pr_auc']:.4f}, "
            f"TP={int(row['external_tp'])}, FP={int(row['external_fp'])}, FN={int(row['external_fn'])}, TN={int(row['external_tn'])}"
        )
    md += [
        "",
        f"## Best external F1",
        f"- {best['scaler']} scaling with OOF-selected threshold {best['selected_threshold_from_oof']:.2f}.",
        f"- External F1 = {best['external_f1']:.4f}.",
        "",
        "## Interpretation",
        "MinMax scaling is promoted from an exploratory Stage4 finding to a strict OOF-threshold review. It improves external precision and F1, but external recall falls below the high-recall operating target. It is therefore a strong low-false-alarm candidate, while the standard-scaling LR remains the safer high-recall baseline.",
    ]
    (OUT_DIR / "oof_minmax_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved OOF threshold and MinMax review outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
