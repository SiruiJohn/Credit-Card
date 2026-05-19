from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage4"

FOCUS_FEATURES = ["V4", "V14", "V12", "V10", "V11", "Amount_z"]


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["Class"].value_counts().sort_index().items()}


def save_class_distribution(df_ext: pd.DataFrame) -> None:
    counts = df_ext["Class"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Normal", "Fraud"], counts.values, color=["#4C78A8", "#F58518"])
    axes[0].set_title("External 2023 Class Counts")
    axes[0].set_ylabel("Rows")
    for idx, value in enumerate(counts.values):
        axes[0].text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    axes[1].pie(counts.values, labels=["Normal", "Fraud"], autopct="%1.1f%%", colors=["#4C78A8", "#F58518"])
    axes[1].set_title("External 2023 Class Share")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_class_distribution.png", dpi=180)
    plt.close()


def save_histograms_by_class(df_ext: pd.DataFrame) -> None:
    features = [f for f in FOCUS_FEATURES if f in df_ext.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        values_0 = df_ext.loc[df_ext["Class"] == 0, feature].dropna()
        values_1 = df_ext.loc[df_ext["Class"] == 1, feature].dropna()
        lo, hi = np.nanpercentile(df_ext[feature], [1, 99])
        bins = np.linspace(lo, hi, 45)
        ax.hist(values_0.clip(lo, hi), bins=bins, alpha=0.55, density=True, label="Normal", color="#4C78A8")
        ax.hist(values_1.clip(lo, hi), bins=bins, alpha=0.55, density=True, label="Fraud", color="#F58518")
        ax.axvline(values_0.mean(), color="#4C78A8", linestyle="--", linewidth=1)
        ax.axvline(values_1.mean(), color="#F58518", linestyle="--", linewidth=1)
        ax.set_title(feature)
    for ax in axes[len(features) :]:
        ax.axis("off")
    axes[0].legend(loc="best")
    fig.suptitle("External Feature Distributions by Class", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_feature_histograms_by_class.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_boxplots_by_class(df_ext: pd.DataFrame) -> None:
    features = [f for f in FOCUS_FEATURES if f in df_ext.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        grouped = [
            df_ext.loc[df_ext["Class"] == 0, feature].dropna().clip(*np.nanpercentile(df_ext[feature], [1, 99])),
            df_ext.loc[df_ext["Class"] == 1, feature].dropna().clip(*np.nanpercentile(df_ext[feature], [1, 99])),
        ]
        ax.boxplot(grouped, tick_labels=["Normal", "Fraud"], showfliers=False)
        ax.set_title(feature)
    for ax in axes[len(features) :]:
        ax.axis("off")
    fig.suptitle("External Feature Boxplots by Class (1%-99% clipped)", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_feature_boxplots_by_class.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_correlation_heatmap(df_ext: pd.DataFrame) -> list[dict[str, float]]:
    numeric = df_ext.select_dtypes(include="number")
    corr_to_class = numeric.corr(numeric_only=True)["Class"].drop("Class").sort_values(key=lambda s: s.abs(), ascending=False)
    top_features = corr_to_class.head(12).index.tolist()
    corr = numeric[top_features + ["Class"]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("External Correlation Heatmap (Top Class-Correlated Features)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_correlation_heatmap_top.png", dpi=180)
    plt.close()

    return [{"feature": k, "corr_with_class": float(v)} for k, v in corr_to_class.head(12).items()]


def save_pairplot_sample(df_ext: pd.DataFrame) -> None:
    features = ["V1", "V2", "V3", "V4"]
    sample = pd.concat(
        [
            df_ext[df_ext["Class"] == 0].sample(n=1500, random_state=42),
            df_ext[df_ext["Class"] == 1].sample(n=1500, random_state=42),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, row_feature in enumerate(features):
        for j, col_feature in enumerate(features):
            ax = axes[i, j]
            if i == j:
                for cls, color, label in [(0, "#4C78A8", "Normal"), (1, "#F58518", "Fraud")]:
                    vals = sample.loc[sample["Class"] == cls, row_feature]
                    lo, hi = np.nanpercentile(vals, [1, 99])
                    ax.hist(vals.clip(lo, hi), bins=25, alpha=0.6, color=color, label=label)
            else:
                ax.scatter(
                    sample[col_feature],
                    sample[row_feature],
                    c=np.where(sample["Class"] == 1, "#F58518", "#4C78A8"),
                    alpha=0.35,
                    s=5,
                    linewidths=0,
                )
            if i == 3:
                ax.set_xlabel(col_feature)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(row_feature)
            else:
                ax.set_yticklabels([])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("External Sample Pair Plot Inspired by Notebook EDA", y=0.93)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_pairplot_v1_v4_sample.png", dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)

    save_class_distribution(df_ext)
    save_histograms_by_class(df_ext)
    save_boxplots_by_class(df_ext)
    top_corr = save_correlation_heatmap(df_ext)
    save_pairplot_sample(df_ext)

    summary = {
        "main_rows": int(len(df_main)),
        "main_class_counts": class_counts(df_main),
        "external_rows": int(len(df_ext)),
        "external_class_counts": class_counts(df_ext),
        "external_duplicate_rows": int(df_ext.duplicated().sum()),
        "top_external_correlations_with_class": top_corr,
    }
    (OUT_DIR / "external_eda_notebook_inspired.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# External EDA Inspired by Reference Notebooks",
        "",
        f"- External rows: {len(df_ext):,}",
        f"- External class counts: {class_counts(df_ext)}",
        f"- Duplicate rows after dropping id: {int(df_ext.duplicated().sum())}",
        "",
        "## Top external correlations with Class",
    ]
    for item in top_corr[:8]:
        md.append(f"- {item['feature']}: {item['corr_with_class']:.4f}")
    md += [
        "",
        "## Generated figures",
        "- `external_class_distribution.png`",
        "- `external_feature_histograms_by_class.png`",
        "- `external_feature_boxplots_by_class.png`",
        "- `external_correlation_heatmap_top.png`",
        "- `external_pairplot_v1_v4_sample.png`",
    ]
    (OUT_DIR / "external_eda_notebook_inspired_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved notebook-inspired external EDA outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()

