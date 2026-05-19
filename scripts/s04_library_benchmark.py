from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import binary_metrics, choose_threshold, pr_auc_score_manual, roc_auc_score_manual


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage4"


def model_specs(pos_weight: float) -> dict:
    return {
        "sklearn_logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        ),
        "hist_gradient_boosting_weighted": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=42,
        ),
    }


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
    raise RuntimeError(f"Model {type(model).__name__} has no probability-like output.")


def plot_results(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    width = 0.22
    ax.bar(x - width, df["precision"], width=width, label="Precision", color="#4C78A8")
    ax.bar(x, df["recall"], width=width, label="Recall", color="#F58518")
    ax.bar(x + width, df["f1"], width=width, label="F1", color="#54A24B")
    ax.set_xticks(x, df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mature Library Benchmark (Non-Core Reference)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "library_benchmark_compare.png", dpi=180)
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

    pos_weight = float((len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1))
    sample_weight = np.where(y_main == 1, pos_weight, 1.0)

    rows = []
    for name, model in model_specs(pos_weight).items():
        if name == "hist_gradient_boosting_weighted":
            model.fit(x_main, y_main, sample_weight=sample_weight)
        else:
            model.fit(x_main, y_main)

        score_main = predict_scores(model, x_main)
        score_ext = predict_scores(model, x_ext)
        thr = choose_threshold(y_main, score_main)
        metrics = binary_metrics(y_ext, score_ext, thr)
        metrics["roc_auc"] = roc_auc_score_manual(y_ext, score_ext)
        metrics["pr_auc"] = pr_auc_score_manual(y_ext, score_ext)
        rows.append({"model": name, **metrics})

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(OUT_DIR / "library_benchmark_results.csv", index=False)
    (OUT_DIR / "library_benchmark_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(df)

    md = [
        "# Mature Library Benchmark",
        "",
        "These models are non-core references inspired by the notebook benchmarks. They are not used to satisfy the from-scratch course requirement.",
        "",
    ]
    for _, row in df.iterrows():
        md.append(
            f"- {row['model']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, pr_auc={row['pr_auc']:.4f}, thr={row['threshold']:.2f}"
        )
    (OUT_DIR / "library_benchmark_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved library benchmark outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
