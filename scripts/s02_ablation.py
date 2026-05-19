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
RAW_MAIN = ROOT / "Credit Card Fraud Detection" / "creditcard.csv"
EXT_DATA = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


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
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.05, 1.0, 0.05):
        m = binary_metrics(y_true, y_score, float(thr))
        if m["recall"] >= 0.85 and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
    if best_f1 < 0:
        # Fallback: maximize recall if target cannot be met
        recalls = [binary_metrics(y_true, y_score, float(t)) for t in np.arange(0.05, 1.0, 0.05)]
        best_thr = max(recalls, key=lambda x: x["recall"])["threshold"]
    return best_thr


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma, mu, sigma


def prepare_main(dedup: bool) -> pd.DataFrame:
    df = pd.read_csv(RAW_MAIN)
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    amt_mean = float(df["Amount"].mean())
    amt_std = float(df["Amount"].std(ddof=0))
    if amt_std == 0:
        amt_std = 1.0
    df["Amount_z"] = (df["Amount"] - amt_mean) / amt_std
    return df


def split_features(df: pd.DataFrame, include_amount_raw: bool):
    feature_cols = [c for c in df.columns if c != "Class"]
    if not include_amount_raw and "Amount" in feature_cols:
        feature_cols.remove("Amount")
    if include_amount_raw:
        # Keep both Amount and Amount_z when raw amount is enabled.
        pass
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["Class"].to_numpy(dtype=int)
    return x, y, feature_cols


def run_ablation() -> dict:
    variants = [
        {"name": "baseline_dedup_auto_weight", "dedup": True, "include_amount_raw": False, "weight_factor": 1.0},
        {"name": "no_dedup_auto_weight", "dedup": False, "include_amount_raw": False, "weight_factor": 1.0},
        {"name": "dedup_with_amount_raw", "dedup": True, "include_amount_raw": True, "weight_factor": 1.0},
        {"name": "dedup_low_pos_weight", "dedup": True, "include_amount_raw": False, "weight_factor": 0.5},
        {"name": "dedup_high_pos_weight", "dedup": True, "include_amount_raw": False, "weight_factor": 1.5},
    ]
    ext_df = pd.read_csv(EXT_DATA)
    ext_y = ext_df["Class"].to_numpy(dtype=int)
    rows = []

    for v in variants:
        df = prepare_main(v["dedup"])
        x, y, feature_cols = split_features(df, include_amount_raw=v["include_amount_raw"])

        # Align external features
        ext_feature_cols = [c for c in feature_cols if c in ext_df.columns]
        x = df[ext_feature_cols].to_numpy(dtype=float)
        x_ext = ext_df[ext_feature_cols].to_numpy(dtype=float)

        x_train, x_train_scaled, mu, sigma = standardize_by_train(x, x)
        x_ext_scaled = (x_ext - mu) / sigma
        pos_weight_auto = (len(y) - np.sum(y)) / max(np.sum(y), 1)
        pos_weight = float(pos_weight_auto * v["weight_factor"])

        model = LogisticRegressionScratch(lr=0.05, max_iter=160, l2=1e-4, pos_weight=pos_weight)
        model.fit(x_train, y)
        y_score_train = model.predict_proba(x_train_scaled)[:, 1]
        y_score_ext = model.predict_proba(x_ext_scaled)[:, 1]

        thr = choose_threshold(y, y_score_train)
        m = binary_metrics(ext_y, y_score_ext, thr)
        rows.append(
            {
                "variant": v["name"],
                "dedup": v["dedup"],
                "include_amount_raw": v["include_amount_raw"],
                "weight_factor": v["weight_factor"],
                "threshold": thr,
                "precision_ext": m["precision"],
                "recall_ext": m["recall"],
                "f1_ext": m["f1"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "n_features": len(ext_feature_cols),
            }
        )
    return {"ablation_rows": rows}


def plot_threshold_scan(df_scan: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df_scan["threshold"], df_scan["precision"], label="Precision")
    plt.plot(df_scan["threshold"], df_scan["recall"], label="Recall")
    plt.plot(df_scan["threshold"], df_scan["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline setting for threshold scan and confusion matrix export
    main_df = prepare_main(dedup=True)
    ext_df = pd.read_csv(EXT_DATA)

    feat_cols = [c for c in main_df.columns if c not in ("Class", "Amount")]
    feat_cols = [c for c in feat_cols if c in ext_df.columns]
    x_main = main_df[feat_cols].to_numpy(dtype=float)
    y_main = main_df["Class"].to_numpy(dtype=int)
    x_ext = ext_df[feat_cols].to_numpy(dtype=float)
    y_ext = ext_df["Class"].to_numpy(dtype=int)

    x_train_scaled, x_main_scaled, mu, sigma = standardize_by_train(x_main, x_main)
    x_ext_scaled = (x_ext - mu) / sigma

    pos_weight = float((len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1))
    model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=pos_weight)
    model.fit(x_train_scaled, y_main)

    score_main = model.predict_proba(x_main_scaled)[:, 1]
    score_ext = model.predict_proba(x_ext_scaled)[:, 1]
    selected_thr = choose_threshold(y_main, score_main)

    thresholds = np.arange(0.05, 1.0, 0.05)
    rows_main = [binary_metrics(y_main, score_main, float(t)) for t in thresholds]
    rows_ext = [binary_metrics(y_ext, score_ext, float(t)) for t in thresholds]
    df_scan_main = pd.DataFrame(rows_main)
    df_scan_ext = pd.DataFrame(rows_ext)
    df_scan_main.to_csv(OUT_DIR / "threshold_scan_main.csv", index=False)
    df_scan_ext.to_csv(OUT_DIR / "threshold_scan_ext2023.csv", index=False)
    plot_threshold_scan(df_scan_main, OUT_DIR / "threshold_scan_main.png", "Threshold Scan (Main)")
    plot_threshold_scan(df_scan_ext, OUT_DIR / "threshold_scan_ext2023.png", "Threshold Scan (External 2023)")

    cm = binary_metrics(y_ext, score_ext, selected_thr)
    (OUT_DIR / "confusion_matrix_ext2023.json").write_text(
        json.dumps(
            {
                "selected_threshold": selected_thr,
                "confusion_matrix_ext2023": {
                    "tp": cm["tp"],
                    "tn": cm["tn"],
                    "fp": cm["fp"],
                    "fn": cm["fn"],
                },
                "precision": cm["precision"],
                "recall": cm["recall"],
                "f1": cm["f1"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ablation = run_ablation()
    ablation_df = pd.DataFrame(ablation["ablation_rows"]).sort_values("f1_ext", ascending=False)
    ablation_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    (OUT_DIR / "ablation_results.json").write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = [
        "# Phase2 Advanced Analysis",
        "",
        f"- Selected threshold (from main): {selected_thr:.2f}",
        f"- External metrics at selected threshold: precision={cm['precision']:.4f}, recall={cm['recall']:.4f}, f1={cm['f1']:.4f}",
        "",
        "## Best Ablation Variants (Top 3 by external F1)",
    ]
    for _, r in ablation_df.head(3).iterrows():
        summary.append(
            f"- {r['variant']}: F1={r['f1_ext']:.4f}, Recall={r['recall_ext']:.4f}, Precision={r['precision_ext']:.4f}, thr={r['threshold']:.2f}"
        )
    (OUT_DIR / "phase2_advanced_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved advanced outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
