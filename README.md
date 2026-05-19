# Credit Card Fraud Detection Project

AI3013 course project for credit card fraud detection, from-scratch machine learning models, and cross-dataset generalization analysis.

## Project Structure

```
Credit Card/
├── main.py                         ← Pipeline orchestrator (one-click run all stages)
├── scripts/
│   ├── utils.py                    ← Shared utility functions (metrics, sampling, scaling)
│   ├── models_from_scratch.py      ← From-scratch implementations: LR, GNB, kNN
│   ├── s01_prepare_data.py         ← Stage 1: Data cleaning & preparation
│   ├── s01_eda.py                  ← Stage 1: Exploratory data analysis
│   ├── s02_train_baseline.py       ← Stage 2: 5-fold CV + external evaluation
│   ├── s02_plot_baseline.py        ← Stage 2: Visualize CV & external metrics
│   ├── s02_ablation.py             ← Stage 2: Ablation study (dedup, features, weights)
│   ├── s02_lr_analysis.py          ← Stage 2: LR fine threshold scan + feature importance
│   ├── s02_tune_knn.py             ← Stage 2: kNN basic hyperparameter tuning
│   ├── s02_tune_knn_advanced.py    ← Stage 2: kNN advanced tuning (PCA + distance weights)
│   ├── s03_cost_threshold.py       ← Stage 3: Cost-sensitive threshold optimization
│   ├── s03_temporal_robustness.py  ← Stage 3: Temporal robustness (expanding-window)
│   ├── s03_error_analysis.py       ← Stage 3: Error case analysis (TP/FP/FN/TN profiling)
│   ├── s04_external_eda.py         ← Stage 4: External dataset EDA & visualization
│   ├── s04_scaling_ablation.py     ← Stage 4: Scaling method ablation (standard/minmax/robust)
│   ├── s04_library_benchmark.py    ← Stage 4: Mature library reference benchmarks
│   └── s05_oof_validation.py       ← Stage 5: OOF threshold review (MinMax verification)
├── outputs/
│   ├── stage1/                     ← Prepared data + EDA outputs
│   ├── stage2/                     ← Training results + threshold scans + ablation + kNN tuning
│   ├── stage3/                     ← Cost analysis + temporal robustness + error cases
│   ├── stage4/                     ← External EDA + scaling ablation + library benchmarks
│   └── stage5/                     ← OOF threshold validation
├── docs/
│   ├── planning/                   ← Project planning documents
│   ├── process/                    ← Stage-level documentation
│   ├── Project_Status_and_Fix_Log.md
│   ├── Final_Submission_Integrated.md / _EN.md
│   └── Credit_Card_Fraud_Detection_Defense.pptx
├── latex_bundle/                   ← LaTeX final report
└── requirements.txt
```

## Current Status

- Stage 1 data preparation has been repaired and regenerated.
- Stage 2 baseline has been rerun in full 5-fold mode.
- Stage 2 threshold scans, confusion matrix, ablation, LR final analysis, and kNN tuning have been refreshed.
- Stage 3 cost-sensitive thresholding, temporal robustness, and error-case analysis have been refreshed.
- Notebook-inspired Stage 4 extensions have been added: external EDA figures, scaling ablation, and mature library reference benchmarks.
- Stage 5 OOF-threshold review has been added to verify MinMax scaling under a stricter threshold-selection protocol.
- Formal LaTeX figures in `latex_bundle/figures/` have been synchronized with the refreshed outputs.
- A 15-slide defense deck has been added at `docs/Credit_Card_Fraud_Detection_Defense.pptx`.

See `docs/Project_Status_and_Fix_Log.md` for the detailed fix log.

## Data

- Main dataset: `Credit Card Fraud Detection/creditcard.csv`
  - Raw rows: 284,807
  - Class distribution: 0:284,315 / 1:492
- External 2023 dataset: `Credit Card Fraud Detection_2023/creditcard_2023.csv`
  - Raw rows: 568,630
  - Class distribution: 0:284,315 / 1:284,315

The raw CSV files and regenerated prepared CSV files are intentionally not committed because they exceed normal GitHub repository size limits. Place the two source CSV files in the paths above before running the reproduction commands.

Generated prepared data:

- `outputs/stage1/main_prepared.csv`
- `outputs/stage1/ext2023_prepared.csv`
- `outputs/stage1/prepare_metadata.json`

## Quick Start — One-Click Pipeline

The simplest way to reproduce the entire project is via `main.py`:

```powershell
python main.py                 # Run all 5 stages
python main.py --quick         # Quick mode (2-fold CV)
python main.py --stage 1       # Run Stage 1 only
python main.py --stage 2 --quick  # Stage 2 in quick mode
python main.py --skip-stage 4 --skip-stage 5  # Skip Stage 4 and 5
```

## Manual Reproduce (Stage by Stage)

Use the Codex bundled Python runtime or another Python 3.12 environment with the packages in `requirements.txt`.

In this workspace, plotting uses the local dependency folder `.codex_pydeps`:

```powershell
$py = 'C:\Users\siruiJohn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = '.\.codex_pydeps'

# Stage 1 — Data Preparation & EDA
& $py scripts\s01_prepare_data.py
& $py scripts\s01_eda.py

# Stage 2 — Model Training & Evaluation
& $py scripts\s02_train_baseline.py
& $py scripts\s02_plot_baseline.py
& $py scripts\s02_ablation.py
& $py scripts\s02_lr_analysis.py
& $py scripts\s02_tune_knn.py
& $py scripts\s02_tune_knn_advanced.py

# Stage 3 — Deep Evaluation
& $py scripts\s03_cost_threshold.py
& $py scripts\s03_temporal_robustness.py
& $py scripts\s03_error_analysis.py

# Stage 4 — Extended Analysis
& $py scripts\s04_external_eda.py
& $py scripts\s04_scaling_ablation.py
& $py scripts\s04_library_benchmark.py

# Stage 5 — OOF Validation
& $py scripts\s05_oof_validation.py
```

## Pipeline Stages

| Stage | Scripts | Description |
|-------|---------|-------------|
| **1** | `s01_prepare_data`, `s01_eda` | Data cleaning (dedup, Amount Z-score), alignment of external dataset, and exploratory data analysis |
| **2** | `s02_train_baseline`, `s02_plot_baseline`, `s02_ablation`, `s02_lr_analysis`, `s02_tune_knn`, `s02_tune_knn_advanced` | 5-fold CV for LR/GNB/kNN, external evaluation, threshold scans, ablation study, LR feature importance, and kNN hyperparameter tuning |
| **3** | `s03_cost_threshold`, `s03_temporal_robustness`, `s03_error_analysis` | Cost-sensitive threshold optimization (FP:FN ratios), temporal robustness via expanding-window evaluation, and error case profiling |
| **4** | `s04_external_eda`, `s04_scaling_ablation`, `s04_library_benchmark` | External dataset visualizations, scaling method comparison (standard/minmax/robust), and mature sklearn library benchmarks |
| **5** | `s05_oof_validation` | Strict OOF (out-of-fold) threshold selection protocol to verify MinMax scaling promotion |

## Shared Modules

| Module | Contents |
|--------|----------|
| `scripts/utils.py` | `binary_metrics`, `choose_threshold`, `roc_auc_score_manual`, `pr_auc_score_manual`, `stratified_kfold_indices`, `stratified_subsample`, `standardize_by_train` |
| `scripts/models_from_scratch.py` | `LogisticRegressionScratch`, `GaussianNBScratch`, `KNNScratch` |

## Main Results

Current refreshed baseline:

- Logistic Regression: CV Recall 0.8605, CV F1 0.4428, CV PR-AUC 0.7116, external F1 0.6949.
- Gaussian Naive Bayes: CV Recall 0.8435, CV F1 0.1092, CV PR-AUC 0.4090, external F1 0.6411.
- kNN: CV Recall 0.8068, CV F1 0.5811, CV PR-AUC 0.6695, external F1 0.0518 on stratified external subset.

The primary model remains Logistic Regression because it has the strongest cross-dataset precision-recall balance among the from-scratch models.

Notebook-inspired extensions are documented in `docs/Notebook_Extensions.md`, with outputs saved under `outputs/stage4/`.

Stage 5 OOF-threshold review is saved under `outputs/stage5/`. It shows that MinMax scaling improves external F1 and sharply reduces false positives, but lowers external recall, so it is best treated as a low-false-alarm candidate rather than a replacement for the high-recall standard-scaling baseline.

## Deliverables

- Integrated report: `docs/Final_Submission_Integrated.md`
- English integrated report: `docs/Final_Submission_Integrated_EN.md`
- LaTeX report bundle: `latex_bundle/main.tex`
- Defense deck: `docs/Credit_Card_Fraud_Detection_Defense.pptx`
