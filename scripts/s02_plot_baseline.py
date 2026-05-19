from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = ROOT / "outputs" / "stage2" / "phase2_baseline_results.json"
OUT_DIR = ROOT / "outputs" / "stage2"


def bar_plot(labels, values, title, ylabel, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    models = ["logistic_regression", "gaussian_nb"]

    cv_recall = [data["cv"][m]["summary"]["recall_mean"] for m in models]
    cv_f1 = [data["cv"][m]["summary"]["f1_mean"] for m in models]
    cv_prauc = [data["cv"][m]["summary"]["pr_auc_mean"] for m in models]
    ext_recall = [data["external_eval"][m]["recall"] for m in models]
    ext_f1 = [data["external_eval"][m]["f1"] for m in models]
    ext_prauc = [data["external_eval"][m]["pr_auc"] for m in models]

    bar_plot(models, cv_recall, "CV Recall Mean", "Recall", OUT_DIR / "cv_recall_mean.png")
    bar_plot(models, cv_f1, "CV F1 Mean", "F1", OUT_DIR / "cv_f1_mean.png")
    bar_plot(models, cv_prauc, "CV PR-AUC Mean", "PR-AUC", OUT_DIR / "cv_prauc_mean.png")
    bar_plot(models, ext_recall, "External Recall", "Recall", OUT_DIR / "ext_recall.png")
    bar_plot(models, ext_f1, "External F1", "F1", OUT_DIR / "ext_f1.png")
    bar_plot(models, ext_prauc, "External PR-AUC", "PR-AUC", OUT_DIR / "ext_prauc.png")

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
