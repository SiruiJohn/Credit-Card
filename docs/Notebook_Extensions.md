# Notebook-Inspired Extensions

This document records how the two reference notebooks were integrated into the project without changing the core course requirement of from-scratch model implementation.

## What Was Added

### 1. External EDA figures

Source inspiration:

- Class distribution plots
- Feature histograms by class
- Boxplots
- Correlation heatmaps
- Small pairplot-style feature views

Project implementation:

- Script: `scripts/external_eda_notebook_inspired.py`
- Output directory: `outputs/stage4/`

Generated files:

- `external_class_distribution.png`
- `external_feature_histograms_by_class.png`
- `external_feature_boxplots_by_class.png`
- `external_correlation_heatmap_top.png`
- `external_pairplot_v1_v4_sample.png`
- `external_eda_notebook_inspired_summary.md`

### 2. Scaling ablation

Source inspiration:

- The notebooks try different preprocessing/scaling ideas such as standard scaling, min-max scaling, and quantile-style transformations.

Project implementation:

- Script: `scripts/scaling_ablation.py`
- Models: from-scratch Logistic Regression and from-scratch distance-weighted kNN
- Scalers: standard, min-max, robust

Generated files:

- `scaling_ablation_results.csv`
- `scaling_ablation_results.json`
- `scaling_ablation_summary.md`
- `scaling_ablation_compare.png`

### 3. Mature library benchmark

Source inspiration:

- The notebooks use mature libraries such as CatBoost, LightGBM, and XGBoost and emphasize high accuracy.

Project adaptation:

- The project adds a non-core benchmark using mature scikit-learn models.
- This benchmark is explicitly treated as a reference upper-bound, not as the course-required from-scratch result.

Project implementation:

- Script: `scripts/library_benchmark.py`
- Models:
  - scikit-learn Logistic Regression
  - balanced Random Forest
  - weighted HistGradientBoosting

Generated files:

- `library_benchmark_results.csv`
- `library_benchmark_results.json`
- `library_benchmark_summary.md`
- `library_benchmark_compare.png`

## What Was Not Directly Adopted

- The notebooks' “99% accuracy” framing is not used as a project conclusion.
- The project continues to prioritize Recall, F1, and PR-AUC because the main dataset is severely imbalanced.
- The notebook pattern of random splitting only within the 2023 dataset is not used as the main validation protocol.
- Identifier-like leakage is avoided by dropping `id` from the external dataset.

