from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_from_scratch import KNNScratch


ROOT = Path(__file__).resolve().parents[1]
DATA_MAIN = ROOT / "outputs" / "stage1" / "main_prepared.csv"
DATA_EXT = ROOT / "outputs" / "stage1" / "ext2023_prepared.csv"
OUT_DIR = ROOT / "outputs" / "stage2"


def stratified_subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42):
    if len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos = max(1, int(max_samples * (len(idx_pos) / len(y))))
    n_neg = max(1, max_samples - n_pos)
    idx = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
    rng.shuffle(idx)
    return x[idx], y[idx]


def train_valid_split(x: np.ndarray, y: np.ndarray, valid_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_va = max(1, int(len(idx_pos) * valid_ratio))
    n_neg_va = max(1, int(len(idx_neg) * valid_ratio))
    va_idx = np.concatenate([idx_pos[:n_pos_va], idx_neg[:n_neg_va]])
    tr_idx = np.setdiff1d(np.arange(len(y)), va_idx, assume_unique=False)
    return x[tr_idx], y[tr_idx], x[va_idx], y[va_idx]


def standardize_by_train(x_train: np.ndarray, x_apply: np.ndarray):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    return (x_train - mu) / sigma, (x_apply - mu) / sigma


def fit_pca(x_train: np.ndarray, n_components: int):
    mu = np.mean(x_train, axis=0)
    xc = x_train - mu
    # SVD-based PCA
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:n_components].T
    return mu, comp


def transform_pca(x: np.ndarray, mu: np.ndarray, comp: np.ndarray):
    return (x - mu) @ comp


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray):
    best_thr = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    for thr in np.arange(0.05, 1.0, 0.05):
        m = binary_metrics(y_true, y_score, float(thr))
        if m["recall"] >= 0.85 and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
        elif best_f1 < 0 and m["recall"] > best_recall:
            best_recall = m["recall"]
            best_thr = float(thr)
    return best_thr


def eval_setting(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_ext: np.ndarray,
    y_ext: np.ndarray,
    k: int,
    weights: str,
):
    model = KNNScratch(
        n_neighbors=k,
        weights=weights,
        max_train_samples=2000,
        chunk_size=256,
        random_state=42,
    )
    model.fit(x_tr, y_tr)
    thr = choose_threshold(y_va, model.predict_proba(x_va)[:, 1])
    ext_score = model.predict_proba(x_ext)[:, 1]
    m = binary_metrics(y_ext, ext_score, thr)
    return thr, m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(DATA_MAIN)
    df_ext = pd.read_csv(DATA_EXT)
    features = sorted(list((set(df_main.columns) & set(df_ext.columns)) - {"Class", "Amount"}))
    x = df_main[features].to_numpy(dtype=float)
    y = df_main["Class"].to_numpy(dtype=int)
    x_ext = df_ext[features].to_numpy(dtype=float)
    y_ext = df_ext["Class"].to_numpy(dtype=int)

    x, y = stratified_subsample(x, y, max_samples=8000, seed=7)
    x_ext, y_ext = stratified_subsample(x_ext, y_ext, max_samples=5000, seed=99)

    x_tr, y_tr, x_va, y_va = train_valid_split(x, y, valid_ratio=0.2, seed=42)
    x_tr, x_va = standardize_by_train(x_tr, x_va)
    _, x_ext = standardize_by_train(x_tr, x_ext)

    settings = []
    for weights in ["uniform", "distance"]:
        for k in [3, 5, 9, 15]:
            thr, m = eval_setting(x_tr, y_tr, x_va, y_va, x_ext, y_ext, k=k, weights=weights)
            settings.append(
                {
                    "model": f"knn_{weights}",
                    "k": k,
                    "threshold": thr,
                    "precision_ext": m["precision"],
                    "recall_ext": m["recall"],
                    "f1_ext": m["f1"],
                }
            )

    # PCA + distance-weighted kNN
    for n_comp in [8, 12, 16]:
        mu, comp = fit_pca(x_tr, n_components=n_comp)
        x_tr_p = transform_pca(x_tr, mu, comp)
        x_va_p = transform_pca(x_va, mu, comp)
        x_ext_p = transform_pca(x_ext, mu, comp)
        thr, m = eval_setting(x_tr_p, y_tr, x_va_p, y_va, x_ext_p, y_ext, k=9, weights="distance")
        settings.append(
            {
                "model": f"pca{n_comp}_knn_distance",
                "k": 9,
                "threshold": thr,
                "precision_ext": m["precision"],
                "recall_ext": m["recall"],
                "f1_ext": m["f1"],
            }
        )

    df = pd.DataFrame(settings).sort_values("f1_ext", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "knn_advanced_results.csv", index=False)

    plt.figure(figsize=(8, 4))
    xs = np.arange(len(df))
    plt.bar(xs, df["f1_ext"].to_numpy())
    plt.xticks(xs, [f"{m}|k={k}" for m, k in zip(df["model"], df["k"])], rotation=45, ha="right")
    plt.ylabel("External F1")
    plt.title("kNN Advanced Settings Comparison")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "knn_advanced_f1_compare.png", dpi=180)
    plt.close()

    best = df.iloc[0]
    summary = [
        "# kNN Advanced Tuning Summary",
        "",
        f"- Best setting: {best['model']} (k={int(best['k'])})",
        f"- Threshold: {best['threshold']:.2f}",
        f"- External Precision: {best['precision_ext']:.4f}",
        f"- External Recall: {best['recall_ext']:.4f}",
        f"- External F1: {best['f1_ext']:.4f}",
    ]
    (OUT_DIR / "knn_advanced_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("Saved advanced kNN tuning outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
