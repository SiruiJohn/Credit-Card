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
SELECTED_THR = 0.87
TOP_FEATURES = ["V4", "V14", "V12", "V10", "V11", "Amount_z"]


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma


def fit_lr_on_main(df_main: pd.DataFrame, feature_cols: list[str]) -> LogisticRegressionScratch:
    x_main = df_main[feature_cols].to_numpy(dtype=float)
    y_main = df_main["Class"].to_numpy(dtype=int)
    x_main_sc, _ = standardize_by_train(x_main, x_main)
    pos_weight = (len(y_main) - np.sum(y_main)) / max(np.sum(y_main), 1)
    model = LogisticRegressionScratch(lr=0.05, max_iter=180, l2=1e-4, pos_weight=float(pos_weight))
    model.fit(x_main_sc, y_main)
    return model


def assign_case_group(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    group = np.empty(len(y_true), dtype=object)
    group[(y_true == 1) & (y_pred == 1)] = "TP"
    group[(y_true == 0) & (y_pred == 1)] = "FP"
    group[(y_true == 1) & (y_pred == 0)] = "FN"
    group[(y_true == 0) & (y_pred == 0)] = "TN"
    return group


def save_boxplot(
    df: pd.DataFrame,
    feature_cols: list[str],
    groups: list[str],
    out_path: Path,
    title: str,
) -> None:
    n_features = len(feature_cols)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.ravel()
    for i, feat in enumerate(feature_cols):
        data = [df.loc[df["case_group"] == g, feat].to_numpy() for g in groups]
        axes[i].boxplot(data, tick_labels=groups, showfliers=False)
        axes[i].set_title(feat)
        axes[i].tick_params(axis="x", labelrotation=0)
    for j in range(n_features, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)
    feature_cols = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    analysis_features = [f for f in TOP_FEATURES if f in df_ext.columns]

    model = fit_lr_on_main(df_main, feature_cols)
    x_main = df_main[feature_cols].to_numpy(dtype=float)
    x_ext = df_ext[feature_cols].to_numpy(dtype=float)
    _, x_ext_sc = standardize_by_train(x_main, x_ext)

    y_ext = df_ext["Class"].to_numpy(dtype=int)
    score_ext = model.predict_proba(x_ext_sc)[:, 1]
    y_pred = (score_ext >= SELECTED_THR).astype(int)
    df_ext = df_ext.copy()
    df_ext["score"] = score_ext
    df_ext["pred"] = y_pred
    df_ext["case_group"] = assign_case_group(y_ext, y_pred)

    # Summary counts
    counts = df_ext["case_group"].value_counts().to_dict()

    # Feature profile summary by case group
    rows = []
    for feat in analysis_features:
        grp = (
            df_ext.groupby("case_group")[feat]
            .agg(["mean", "median", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_value", "median": "median_value", "std": "std_value"})
        )
        grp["feature"] = feat
        rows.append(grp)
    stats_df = pd.concat(rows, ignore_index=True)
    stats_df.to_csv(OUT_DIR / "error_case_feature_stats.csv", index=False)

    # Targeted comparison metrics
    targeted_rows = []
    for feat in analysis_features:
        means = (
            df_ext.groupby("case_group")[feat]
            .mean()
            .reindex(["TP", "FP", "FN", "TN"])
            .to_dict()
        )
        targeted_rows.append(
            {
                "feature": feat,
                "fp_minus_tn_mean": float(means.get("FP", np.nan) - means.get("TN", np.nan)),
                "fn_minus_tp_mean": float(means.get("FN", np.nan) - means.get("TP", np.nan)),
                "tp_mean": float(means.get("TP", np.nan)),
                "fp_mean": float(means.get("FP", np.nan)),
                "fn_mean": float(means.get("FN", np.nan)),
                "tn_mean": float(means.get("TN", np.nan)),
            }
        )
    targeted_df = pd.DataFrame(targeted_rows)
    targeted_df.to_csv(OUT_DIR / "error_case_delta_summary.csv", index=False)

    save_boxplot(
        df_ext,
        analysis_features,
        ["TN", "FP"],
        OUT_DIR / "error_profile_fp_vs_tn.png",
        "False Positive Profile vs True Negative",
    )
    save_boxplot(
        df_ext,
        analysis_features,
        ["TP", "FN"],
        OUT_DIR / "error_profile_fn_vs_tp.png",
        "False Negative Profile vs True Positive",
    )

    # Human-readable summary
    fp_rank = targeted_df.reindex(targeted_df["fp_minus_tn_mean"].abs().sort_values(ascending=False).index)
    fn_rank = targeted_df.reindex(targeted_df["fn_minus_tp_mean"].abs().sort_values(ascending=False).index)
    summary = [
        "# Error Case Analysis Summary",
        "",
        f"- Threshold: {SELECTED_THR:.2f}",
        f"- Case counts: TP={counts.get('TP', 0)}, FP={counts.get('FP', 0)}, FN={counts.get('FN', 0)}, TN={counts.get('TN', 0)}",
        "",
        "## Largest FP-TN mean shifts",
    ]
    for _, r in fp_rank.head(3).iterrows():
        summary.append(
            f"- {r['feature']}: FP-TN mean shift = {r['fp_minus_tn_mean']:.4f}"
        )
    summary.extend(["", "## Largest FN-TP mean shifts"])
    for _, r in fn_rank.head(3).iterrows():
        summary.append(
            f"- {r['feature']}: FN-TP mean shift = {r['fn_minus_tp_mean']:.4f}"
        )

    (OUT_DIR / "error_case_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT_DIR / "error_case_counts.json").write_text(
        json.dumps({"threshold": SELECTED_THR, "counts": counts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved error analysis outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
