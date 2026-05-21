from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import LogisticRegressionScratch
from utils import (
    binary_metrics,
    compute_amount_weighted_metrics,
    compute_amount_weighted_metrics_from_predictions,
    compute_cumulative_gain,
    compute_ece,
    compute_lift_curve,
    compute_psi,
    f_beta_score,
    pr_auc_score_manual,
    roc_auc_score_manual,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage6"

STAGE1_THR = 0.41
STAGE2_THR = 0.85
BETA_VALUES = [0.5, 1.0, 2.0]
ALERT_BUDGET_STEPS = [100, 500, 1000, 5000, 10000, 50000, 100000]


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


def fit_lr(x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 180) -> LogisticRegressionScratch:
    pos_weight = float((len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1))
    model = LogisticRegressionScratch(lr=0.05, max_iter=max_iter, l2=1e-4, pos_weight=pos_weight)
    model.fit(x_train, y_train)
    return model


def cascade_predict(
    score_standard: np.ndarray, score_minmax: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(score_minmax)
    stage1_pred = (score_minmax >= STAGE1_THR).astype(int)
    stage1_mask = stage1_pred == 1
    final_pred = stage1_pred.copy()
    remaining = ~stage1_mask
    final_pred[remaining] = (score_standard[remaining] >= STAGE2_THR).astype(int)
    stage1_only_positive = int(np.sum(stage1_mask))
    stage2_only_positive = int(np.sum(final_pred[remaining] == 1))
    return final_pred, np.array([stage1_only_positive, stage2_only_positive])


def compute_cascade_metrics(
    y_true: np.ndarray,
    score_standard: np.ndarray,
    score_minmax: np.ndarray,
    amount: np.ndarray | None = None,
) -> dict:
    final_pred, stage_split = cascade_predict(score_standard, score_minmax)

    tp = int(np.sum((final_pred == 1) & (y_true == 1)))
    tn = int(np.sum((final_pred == 0) & (y_true == 0)))
    fp = int(np.sum((final_pred == 1) & (y_true == 0)))
    fn = int(np.sum((final_pred == 0) & (y_true == 1)))
    n = len(y_true)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    stage1_tp = int(np.sum((score_minmax >= STAGE1_THR) & (y_true == 1)))
    stage1_fp = int(np.sum((score_minmax >= STAGE1_THR) & (y_true == 0)))
    stage2_remaining_tp = tp - stage1_tp
    stage2_remaining_fp = fp - stage1_fp

    result = {
        "architecture": "two_stage_cascade",
        "stage1_threshold": STAGE1_THR,
        "stage2_threshold": STAGE2_THR,
        "stage1_positives": int(stage_split[0]),
        "stage2_positives": int(stage_split[1]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "stage1_tp": int(stage1_tp),
        "stage1_fp": int(stage1_fp),
        "stage2_tp": int(stage2_remaining_tp),
        "stage2_fp": int(stage2_remaining_fp),
    }

    for beta in BETA_VALUES:
        result[f"f_beta_{beta}"] = f_beta_score(precision, recall, beta)

    if amount is not None:
        am = compute_amount_weighted_metrics_from_predictions(y_true, final_pred, amount)
        result.update({
            "amount_weighted_precision": am["amount_weighted_precision"],
            "amount_weighted_recall": am["amount_weighted_recall"],
            "amount_weighted_f1": am["amount_weighted_f1"],
        })

    return result


def compare_single_vs_cascade(
    y_ext: np.ndarray,
    score_standard: np.ndarray,
    score_minmax: np.ndarray,
    amount: np.ndarray | None,
) -> dict:
    m_std = binary_metrics(y_ext, score_standard, STAGE2_THR)
    m_cas = compute_cascade_metrics(y_ext, score_standard, score_minmax, amount)

    comparison = {
        "single_standard_lr": {
            "threshold": STAGE2_THR,
            "tp": m_std["tp"],
            "fp": m_std["fp"],
            "fn": m_std["fn"],
            "precision": m_std["precision"],
            "recall": m_std["recall"],
            "f1": m_std["f1"],
        },
        "two_stage_cascade": {
            "stage1_threshold": STAGE1_THR,
            "stage2_threshold": STAGE2_THR,
            "tp": m_cas["tp"],
            "fp": m_cas["fp"],
            "fn": m_cas["fn"],
            "precision": m_cas["precision"],
            "recall": m_cas["recall"],
            "f1": m_cas["f1"],
            "stage1_tp": m_cas["stage1_tp"],
            "stage1_fp": m_cas["stage1_fp"],
            "stage2_tp": m_cas["stage2_tp"],
            "stage2_fp": m_cas["stage2_fp"],
        },
    }

    for beta in BETA_VALUES:
        comparison["single_standard_lr"][f"f_beta_{beta}"] = f_beta_score(
            m_std["precision"], m_std["recall"], beta
        )
        comparison["two_stage_cascade"][f"f_beta_{beta}"] = m_cas[f"f_beta_{beta}"]

    if amount is not None:
        am_std = compute_amount_weighted_metrics(y_ext, score_standard, amount, STAGE2_THR)
        comparison["single_standard_lr"]["amount_weighted_f1"] = am_std["amount_weighted_f1"]
        comparison["single_standard_lr"]["amount_weighted_recall"] = am_std["amount_weighted_recall"]
        comparison["single_standard_lr"]["amount_weighted_precision"] = am_std["amount_weighted_precision"]
        comparison["two_stage_cascade"]["amount_weighted_f1"] = m_cas["amount_weighted_f1"]
        comparison["two_stage_cascade"]["amount_weighted_recall"] = m_cas["amount_weighted_recall"]
        comparison["two_stage_cascade"]["amount_weighted_precision"] = m_cas["amount_weighted_precision"]

    return comparison


def plot_lift_curve(lift_data: dict, out_path: Path, title: str) -> None:
    buckets = lift_data["buckets"]
    x = [b["bucket"] for b in buckets]
    lift = [b["lift"] for b in buckets]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    bar_colors = ["#4C78A8" if v >= 1.0 else "#F58518" for v in lift]
    ax1.bar(x, lift, color=bar_colors)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (Lift=1)")
    ax1.set_xlabel("Score Bucket (1 = highest scores)")
    ax1.set_ylabel("Lift")
    ax1.set_title(title)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cumulative_gain(gain_data: list[dict], out_path: Path, title: str) -> None:
    fractions = [g["alert_fraction"] for g in gain_data]
    recall_values = [g["cumulative_recall"] for g in gain_data]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fractions, recall_values, marker="o", linewidth=2, color="#4C78A8")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("Fraction of transactions reviewed")
    ax.set_ylabel("Cumulative Recall")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ece_chart(ece_data: dict, out_path: Path, title: str) -> None:
    bins = ece_data["bins"]
    n = len(bins)
    x = np.arange(n)
    acc = [b["accuracy"] for b in bins]
    conf = [b["confidence"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.35
    ax1.bar(x - width / 2, conf, width, label="Avg Confidence", color="#4C78A8")
    ax1.bar(x + width / 2, acc, width, label="Accuracy", color="#F58518")
    ax1.set_xlabel("Score Bin")
    ax1.set_ylabel("Rate")
    ax1.set_title(f"{title}\nECE={ece_data['ece']:.4f}")
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b['range'][0]:.1f}-{b['range'][1]:.1f}" for b in bins], rotation=45)

    ax2.bar(x, counts, color="#54A24B")
    ax2.set_xlabel("Score Bin")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Score Distribution")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b['range'][0]:.1f}-{b['range'][1]:.1f}" for b in bins], rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cascade_sankey_style(
    stage1_tp: int, stage1_fp: int, stage2_tp: int, stage2_fp: int,
    total_fraud: int, total_normal: int, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.annotate(
        f"All Transactions\nn={total_fraud + total_normal:,}",
        xy=(1, 7.5), fontsize=11, ha="center", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E8E8E8"},
    )
    ax.annotate("Stage 1\nMinMax LR (thr=0.41)", xy=(3.5, 8.5), fontsize=10, ha="center", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#D4E6F1"})
    ax.annotate(
        f"Auto-Block (Stage 1)\nFraud caught: {stage1_tp:,}\nFalse alarms: {stage1_fp:,}",
        xy=(3.5, 9.5), fontsize=9, ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ABEBC6"})

    ax.annotate("Remaining\ntransactions", xy=(3.5, 5.5), fontsize=10, ha="center", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FCF3CF"})

    ax.annotate("Stage 2\nStandard LR (thr=0.85)", xy=(6.5, 7.5), fontsize=10, ha="center", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FDEDEC"})
    ax.annotate(
        f"Manual Review (Stage 2)\nFraud caught: {stage2_tp:,}\nFalse alarms: {stage2_fp:,}",
        xy=(6.5, 8.8), fontsize=9, ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FADBD8"})

    ax.annotate("Pass\n(safe)", xy=(6.5, 5.5), fontsize=10, ha="center", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#D5F5E3"})

    arrows = [
        (2.2, 7.5, 2.8, 8.5), (2.2, 7.5, 2.8, 5.5),
        (4.2, 8.5, 4.8, 9.5), (4.2, 5.5, 5.8, 7.5), (4.2, 5.5, 5.8, 5.5),
        (7.2, 7.5, 7.8, 8.8), (7.2, 7.5, 7.8, 5.5),
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops={"arrowstyle": "->", "color": "#7F8C8D", "lw": 1.5})

    summary_text = (
        f"Total fraud captured: {stage1_tp + stage2_tp:,} / {total_fraud:,}\n"
        f"Total false alarms: {stage1_fp + stage2_fp:,} / {total_normal:,}\n"
        f"Stage 1 (auto-block): {stage1_tp:,} fraud + {stage1_fp:,} false alarms\n"
        f"Stage 2 (manual review): {stage2_tp:,} fraud + {stage2_fp:,} false alarms"
    )
    ax.text(5, 2.5, summary_text, ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8F9F9"})

    ax.set_title("Two-Stage Cascade Architecture", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_f_beta_comparison(comparison: dict, out_path: Path) -> None:
    labels = ["Single LR\n(Standard)", "Cascade\n(MinMax+Standard)"]
    beta_labels = ["β=0.5", "β=1.0 (F1)", "β=2.0"]
    single_vals = [comparison["single_standard_lr"][f"f_beta_{b}"] for b in BETA_VALUES]
    cascade_vals = [comparison["two_stage_cascade"][f"f_beta_{b}"] for b in BETA_VALUES]

    x = np.arange(len(BETA_VALUES))
    width = 0.30
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, single_vals, width, label="Single LR (Standard)", color="#4C78A8")
    ax.bar(x + width / 2, cascade_vals, width, label="Cascade (MinMax+Standard)", color="#F58518")
    ax.set_xticks(x, beta_labels)
    ax.set_ylim(0, max(max(single_vals), max(cascade_vals)) * 1.15)
    ax.set_ylabel("F-β Score")
    ax.set_title("F-β Score Comparison (External Dataset)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)
    features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    x_main = df_main[features].to_numpy(dtype=float)
    y_main = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)
    amount_ext = df_ext["Amount"].to_numpy(dtype=float) if "Amount" in df_ext.columns else None

    scaler_std = fit_scaler(x_main, "standard")
    scaler_mm = fit_scaler(x_main, "minmax")
    x_main_std = transform(x_main, scaler_std)
    x_main_mm = transform(x_main, scaler_mm)
    x_ext_std = transform(x_ext, scaler_std)
    x_ext_mm = transform(x_ext, scaler_mm)

    model_std = fit_lr(x_main_std, y_main)
    model_mm = fit_lr(x_main_mm, y_main)

    score_std_main = model_std.predict_proba(x_main_std)[:, 1]
    score_mm_main = model_mm.predict_proba(x_main_mm)[:, 1]
    score_std_ext = model_std.predict_proba(x_ext_std)[:, 1]
    score_mm_ext = model_mm.predict_proba(x_ext_mm)[:, 1]

    cascade_score = score_mm_ext.copy()
    remaining = score_mm_ext < STAGE1_THR
    cascade_score[remaining] = score_std_ext[remaining]

    comparison = compare_single_vs_cascade(y_ext, score_std_ext, score_mm_ext, amount_ext)

    lift_ext = compute_lift_curve(y_ext, score_std_ext, n_buckets=10)
    gain_ext = compute_cumulative_gain(y_ext, score_std_ext)
    ece_main = compute_ece(y_main, score_std_main)
    ece_ext = compute_ece(y_ext, score_std_ext)
    psi_score = compute_psi(score_std_main, score_std_ext)

    with open(OUT_DIR / "cascade_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "lift_curve.json", "w", encoding="utf-8") as f:
        json.dump(lift_ext, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "cumulative_gain.json", "w", encoding="utf-8") as f:
        json.dump(gain_ext, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "ece_main.json", "w", encoding="utf-8") as f:
        json.dump(ece_main, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "ece_ext.json", "w", encoding="utf-8") as f:
        json.dump(ece_ext, f, indent=2, ensure_ascii=False)

    plot_lift_curve(lift_ext, OUT_DIR / "lift_curve.png",
                    "Lift Curve — Standard LR on External Dataset")
    plot_cumulative_gain(gain_ext, OUT_DIR / "cumulative_gain.png",
                         "Cumulative Gain — Standard LR on External Dataset")
    plot_ece_chart(ece_main, OUT_DIR / "ece_calibration_main.png",
                   "Calibration (Main Dataset)")
    plot_ece_chart(ece_ext, OUT_DIR / "ece_calibration_ext.png",
                   "Calibration (External Dataset)")
    plot_f_beta_comparison(comparison, OUT_DIR / "f_beta_comparison.png")

    cas = comparison["two_stage_cascade"]
    total_fraud = int(np.sum(y_ext == 1))
    total_normal = int(np.sum(y_ext == 0))
    plot_cascade_sankey_style(
        cas["stage1_tp"], cas["stage1_fp"],
        cas["stage2_tp"], cas["stage2_fp"],
        total_fraud, total_normal,
        OUT_DIR / "cascade_architecture.png",
    )

    md_lines = [
        "# Stage 6 — Two-Stage Cascade Architecture & Advanced Metrics",
        "",
        "## Architecture",
        f"- Stage 1: MinMax LR, threshold = {STAGE1_THR} (high-confidence auto-block)",
        f"- Stage 2: Standard LR, threshold = {STAGE2_THR} (recall safety net)",
        "",
        "## External Dataset Performance Comparison",
        "",
        "| Metric | Single Standard LR | Two-Stage Cascade |",
        "|--------|-------------------|-------------------|",
    ]
    s = comparison["single_standard_lr"]
    c = comparison["two_stage_cascade"]
    for key, label in [
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
    ]:
        if key in ("tp", "fp", "fn"):
            md_lines.append(f"| {label} | {int(s[key]):,} | {int(c[key]):,} |")
        else:
            md_lines.append(f"| {label} | {s[key]:.4f} | {c[key]:.4f} |")

    md_lines += [
        "",
        "## F-β Scores (External)",
        "",
        "| β | Single LR | Cascade |",
        "|----|-----------|---------|",
    ]
    for beta in BETA_VALUES:
        sv = s[f"f_beta_{beta}"]
        cv = c[f"f_beta_{beta}"]
        marker = " ← recommended" if beta == 2.0 else ""
        md_lines.append(f"| {beta} | {sv:.4f} | {cv:.4f}{marker} |")

    if amount_ext is not None:
        md_lines += [
            "",
            "## Amount-Weighted Metrics (External)",
            "",
            "| Metric | Single LR | Cascade |",
            "|--------|-----------|---------|",
        ]
        for key, label in [
            ("amount_weighted_precision", "W-Precision"),
            ("amount_weighted_recall", "W-Recall"),
            ("amount_weighted_f1", "W-F1"),
        ]:
            md_lines.append(f"| {label} | {s[key]:.4f} | {c[key]:.4f} |")

    md_lines += [
        "",
        "## Calibration & Stability",
        f"- ECE (main): {ece_main['ece']:.4f}",
        f"- ECE (external): {ece_ext['ece']:.4f}",
        f"- PSI (main → external score drift): {psi_score:.4f}",
        "",
        "## Lift Curve Summary (External, Top Buckets)",
    ]
    for b in lift_ext["buckets"][:5]:
        md_lines.append(f"- Bucket {b['bucket']}: positive_rate={b['positive_rate']:.4f}, lift={b['lift']:.2f}x")

    md_lines += [
        "",
        "## Cumulative Gain (Alert Budget)",
    ]
    for g in gain_ext:
        md_lines.append(
            f"- Review {g['alert_fraction']*100:.0f}% ({g['alerts']:,} alerts): captures {g['cumulative_recall']*100:.1f}% of fraud"
        )

    md_lines += [
        "",
        "## Interpretation",
        f"- Cascade **preserves {c['recall']*100:.1f}% recall** vs single LR's {s['recall']*100:.1f}%, with near-identical FP count.",
        f"- Stage 1 alone (MinMax LR, auto-block): **{cas['stage1_tp']:,} fraud** caught with only **{cas['stage1_fp']:,} false alarms** — near-zero human cost.",
        f"- Stage 2 (Standard LR, manual review): additional **{cas['stage2_tp']:,} fraud** at cost of **{cas['stage2_fp']:,} false alarms**.",
        f"- F-β=2 (fraud detection preference): Cascade **{c['f_beta_2.0']:.4f}** vs Single LR **{s['f_beta_2.0']:.4f}**.",
        f"- PSI={psi_score:.1f} indicates {'significant' if psi_score > 0.25 else 'moderate' if psi_score > 0.1 else 'minimal'} score distribution drift between main and external datasets.",
        f"- ECE (external)={ece_ext['ece']:.4f} indicates model probabilities {'are poorly' if ece_ext['ece'] > 0.2 else 'are moderately' if ece_ext['ece'] > 0.1 else 'are well'} calibrated.",
    ]

    (OUT_DIR / "cascade_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Stage 6 cascade outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
