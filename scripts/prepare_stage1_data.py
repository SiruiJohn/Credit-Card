import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "Credit Card Fraud Detection" / "creditcard.csv"
DATA_2023 = ROOT / "Credit Card Fraud Detection_2023" / "creditcard_2023.csv"
OUT_DIR = ROOT / "outputs" / "stage1"


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["Class"].value_counts().sort_index().items()}


def write_csv_checked(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    written_rows = sum(1 for _ in tmp_path.open("r", encoding="utf-8")) - 1
    if written_rows != len(df):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"CSV write check failed for {path}: expected {len(df)} rows, got {written_rows}.")
    tmp_path.replace(path)


def zscore_fit(series: pd.Series) -> tuple[float, float]:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0:
        std = 1.0
    return mean, std


def zscore_apply(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (series - mean) / std


def prep_main() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_MAIN)
    before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    amt_mean, amt_std = zscore_fit(df["Amount"])
    df["Amount_z"] = zscore_apply(df["Amount"], amt_mean, amt_std)

    meta = {
        "source": str(DATA_MAIN),
        "rows_before_dedup": before,
        "rows_after_dedup": after,
        "removed_duplicates": before - after,
        "duplicate_rows_before_dedup": duplicate_rows,
        "class_counts_after_dedup": class_counts(df),
        "amount_zscore": {"mean": amt_mean, "std": amt_std},
    }
    return df, meta


def prep_2023(main_amount_stats: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_2023)
    rows_before_drop = int(len(df))
    duplicate_rows_before_drop = int(df.duplicated().sum())
    dropped_columns = []
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        dropped_columns.append("id")
    duplicate_rows_after_drop = int(df.duplicated().sum())

    amt_mean = float(main_amount_stats["mean"])
    amt_std = float(main_amount_stats["std"])
    if amt_std == 0:
        amt_std = 1.0
    df["Amount_z"] = zscore_apply(df["Amount"], amt_mean, amt_std)

    meta = {
        "source": str(DATA_2023),
        "rows_before_drop_columns": rows_before_drop,
        "rows_after_drop_columns": int(len(df)),
        "duplicate_rows_before_drop_columns": duplicate_rows_before_drop,
        "duplicate_rows_after_drop_columns": duplicate_rows_after_drop,
        "class_counts": class_counts(df),
        "dropped_columns": dropped_columns,
        "amount_zscore_reference": {"mean": amt_mean, "std": amt_std},
    }
    return df, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main, meta_main = prep_main()
    df_2023, meta_2023 = prep_2023(meta_main["amount_zscore"])

    out_main = OUT_DIR / "main_prepared.csv"
    out_2023 = OUT_DIR / "ext2023_prepared.csv"
    out_meta = OUT_DIR / "prepare_metadata.json"

    write_csv_checked(df_main, out_main)
    write_csv_checked(df_2023, out_2023)
    out_meta.write_text(
        json.dumps({"main": meta_main, "ext2023": meta_2023}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Prepared main data: {out_main}")
    print(f"Prepared ext data:  {out_2023}")
    print(f"Metadata:           {out_meta}")


if __name__ == "__main__":
    main()
