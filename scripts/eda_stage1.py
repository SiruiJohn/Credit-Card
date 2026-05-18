import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "Credit Card Fraud Detection" / "creditcard.csv"
DATA_2023 = ROOT / "Credit Card Fraud Detection_2023" / "creditcard_2023.csv"
OUT_DIR = ROOT / "outputs" / "stage1"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def basic_summary(df: pd.DataFrame, label_col: str) -> dict:
    class_counts = df[label_col].value_counts().to_dict()
    total = len(df)
    fraud = int(class_counts.get(1, 0))
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "fraud_ratio": fraud / total if total else 0.0,
    }


def numeric_stats(df: pd.DataFrame) -> dict:
    amount = df["Amount"]
    time = df["Time"] if "Time" in df.columns else None

    stats = {
        "amount_describe": {k: float(v) for k, v in amount.describe().to_dict().items()},
        "amount_quantiles": {
            str(k): float(v)
            for k, v in amount.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict().items()
        },
        "amount_skew": float(amount.skew()),
        "amount_kurtosis": float(amount.kurt()),
    }

    if time is not None:
        stats["time_quantiles"] = {
            str(k): float(v)
            for k, v in time.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict().items()
        }
        stats["time_skew"] = float(time.skew())
        stats["time_kurtosis"] = float(time.kurt())
    return stats


def correlation_with_label(df: pd.DataFrame, label_col: str = "Class") -> dict:
    corr = df.corr(numeric_only=True)[label_col].drop(label_col).sort_values()
    return {
        "top_negative": {k: float(v) for k, v in corr.head(8).to_dict().items()},
        "top_positive": {k: float(v) for k, v in corr.tail(8).to_dict().items()},
    }


def save_figures(df: pd.DataFrame, label_col: str = "Class") -> None:
    class_counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Class 0", "Class 1"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    plt.title("Class Distribution (Main Dataset)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution_main.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df["Amount"], bins=100)
    plt.title("Amount Distribution (Raw)")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "amount_distribution_main.png", dpi=160)
    plt.close()


def write_markdown_report(main_report: dict, ext_report: dict) -> None:
    content = f"""# Stage1 EDA Summary

## Main Dataset
- Shape: {tuple(main_report["basic"]["shape"])}
- Missing total: {main_report["basic"]["missing_total"]}
- Duplicate rows: {main_report["basic"]["duplicate_rows"]}
- Class counts: {main_report["basic"]["class_counts"]}
- Fraud ratio: {main_report["basic"]["fraud_ratio"]:.6f}

## Main Dataset Key Findings
- Amount skewness: {main_report["numeric"]["amount_skew"]:.4f}
- Amount kurtosis: {main_report["numeric"]["amount_kurtosis"]:.4f}
- Strong negative correlation features: {list(main_report["correlation"]["top_negative"].keys())[:5]}
- Strong positive correlation features: {list(main_report["correlation"]["top_positive"].keys())[-5:]}

## External Dataset (2023)
- Shape: {tuple(ext_report["basic"]["shape"])}
- Missing total: {ext_report["basic"]["missing_total"]}
- Duplicate rows: {ext_report["basic"]["duplicate_rows"]}
- Class counts: {ext_report["basic"]["class_counts"]}
- Fraud ratio: {ext_report["basic"]["fraud_ratio"]:.6f}
"""
    (OUT_DIR / "eda_summary.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    df_main = pd.read_csv(DATA_MAIN)
    df_2023 = pd.read_csv(DATA_2023)

    main_report = {
        "basic": basic_summary(df_main, "Class"),
        "numeric": numeric_stats(df_main),
        "correlation": correlation_with_label(df_main, "Class"),
    }

    ext_report = {
        "basic": basic_summary(df_2023, "Class"),
        "numeric": numeric_stats(df_2023),
        "correlation": correlation_with_label(df_2023, "Class"),
    }

    (OUT_DIR / "eda_main.json").write_text(
        json.dumps(main_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "eda_2023.json").write_text(
        json.dumps(ext_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_figures(df_main, "Class")
    write_markdown_report(main_report, ext_report)
    print(f"EDA outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
