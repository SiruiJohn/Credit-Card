# Project Status and Fix Log

Updated on 2026-05-22.

## Fixed Breakpoints

### 1. External prepared dataset mismatch

Previous issue:

- `outputs/stage1/prepare_metadata.json` said the external dataset had 568,630 rows.
- `outputs/stage1/ext2023_prepared.csv` actually had 505,685 rows.
- The missing rows were positive-class samples, so Stage2/Stage3 external metrics were computed on an incomplete external set.

Fix:

- Updated `scripts/prepare_stage1_data.py` with write-after-read row validation and richer metadata.
- Regenerated `outputs/stage1/ext2023_prepared.csv`.

Current verified state:

- `outputs/stage1/ext2023_prepared.csv`: 568,630 rows.
- Class distribution: 0:284,315 / 1:284,315.
- `id` is removed.
- One duplicate row exists after dropping `id`, but it is retained to preserve the original external distribution.

### 2. Baseline was quick mode instead of full 5-fold

Previous issue:

- `outputs/stage2/phase2_baseline_summary.md` had `quick_mode: True`.
- The report described the result as 5-fold.

Fix:

- Reran `scripts/train_phase2.py` without `--quick`.

Current verified state:

- `quick_mode: False`.
- Full 5-fold baseline results are saved in `outputs/stage2/phase2_baseline_results.json`.

### 3. Stale Stage2 and Stage3 external metrics

Fix:

- Reran:
  - `scripts/plot_phase2_results.py`
  - `scripts/phase2_advanced_analysis.py`
  - `scripts/lr_final_analysis.py`
  - `scripts/tune_knn.py`
  - `scripts/tune_knn_advanced.py`
  - `scripts/cost_sensitive_threshold.py`
  - `scripts/temporal_robustness.py`
  - `scripts/error_case_analysis.py`

Current key outputs:

- `outputs/stage2/phase2_baseline_summary.md`
- `outputs/stage2/phase2_advanced_summary.md`
- `outputs/stage2/lr_final_analysis_summary.md`
- `outputs/stage3/cost_sensitive_summary.md`
- `outputs/stage3/temporal_robustness_summary.md`
- `outputs/stage3/error_case_summary.md`

## Notebook-Inspired Additions

The two external notebooks were integrated as project-native extensions without replacing the core from-scratch pipeline:

- `scripts/external_eda_notebook_inspired.py` adds external 2023 class distribution, feature histogram, boxplot, correlation heatmap, and pairplot-style figures.
- `scripts/scaling_ablation.py` compares standard, min-max, and robust scaling for from-scratch LR and kNN.
- `scripts/library_benchmark.py` adds mature scikit-learn models as non-core upper-bound references.

Outputs are stored under `outputs/stage4/`, and details are documented in `docs/Notebook_Extensions.md`.

## Stage 5 OOF Threshold Review

To verify the strong MinMax result from Stage 4 under a stricter protocol, `scripts/minmax_oof_threshold_analysis.py` was added.

- Thresholds are selected from 5-fold out-of-fold predictions on the main dataset.
- The selected threshold is then transferred to the full external 2023 dataset.
- Standard scaling remains the safer high-recall baseline.
- MinMax scaling becomes a strong low-false-alarm candidate: it improves external F1 and almost eliminates false positives, but recall drops below the high-recall operating target.

Key output: `outputs/stage5/oof_minmax_summary.md`.

## Stage 6 Cascade Architecture and Advanced Metrics

Stage 6 was added after the OOF review to turn the two validated LR operating modes into a business-facing cascade:

- Stage 1: MinMax LR at threshold 0.41 for high-confidence auto-blocking.
- Stage 2: Standard LR at threshold 0.85 as a recall safety net for remaining transactions.
- Advanced metrics added in `scripts/utils.py`: F-beta, Lift, Cumulative Gain, Amount-Weighted metrics, Expected Calibration Error, and PSI.
- Stage 6 script: `scripts/s06_cascade_architecture.py`.
- Key output: `outputs/stage6/cascade_summary.md`.

Current Stage 6 external result:

| Metric | Single Standard LR | Two-Stage Cascade |
|---|---:|---:|
| Precision | 0.5894 | 0.6025 |
| Recall | 0.8466 | 0.8942 |
| F1 | 0.6949 | 0.7200 |
| F-beta=2 | 0.7786 | 0.8153 |

Follow-up fix on 2026-05-22:

- Corrected cascade amount-weighted metrics to use the final cascade prediction mask instead of reusing the single standard-LR threshold.

### 4. Defense deck added

Fix:

- Added a 15-slide defense deck covering problem framing, data landscape, evaluation protocol, project pipeline, repaired breakpoints, model design, baseline results, threshold/cost analysis, notebook-inspired EDA, Stage4 benchmarks, Stage5 OOF MinMax review, final operating recommendation, and next steps.
- Final deck: `docs/Credit_Card_Fraud_Detection_Defense.pptx`.
- Rebuildable source and preview assets are stored under `outputs/manual-20260515-creditcard-defense/presentations/credit-card-defense/`.

### 5. Missing refreshed confusion matrix figure

Previous issue:

- `latex_bundle/main.tex` referenced `figures/lr_confusion_matrix_ext2023_thr087.png`.
- The script refreshed the JSON confusion matrix but did not regenerate this image.

Fix:

- Added confusion-matrix plotting to `scripts/lr_final_analysis.py`.
- Regenerated `outputs/stage2/lr_confusion_matrix_ext2023_thr087.png`.
- Copied the refreshed image into `latex_bundle/figures/`.

### 6. Report-result mismatches

Fix:

- Updated:
  - `docs/Final_Submission_Integrated.md`
  - `docs/Final_Submission_Integrated_EN.md`
  - `docs/final_report_main.tex`
  - `latex_bundle/main.tex`
- Synchronized key figures into `latex_bundle/figures/`.

## Current Result Snapshot

### Baseline

| Model | Main Recall | Main F1 | Main PR-AUC | External Recall | External F1 | External PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8605 | 0.4428 | 0.7116 | 0.8466 | 0.6949 | 0.7469 |
| Gaussian Naive Bayes | 0.8435 | 0.1092 | 0.4090 | 0.7974 | 0.6411 | 0.4939 |
| kNN | 0.8068 | 0.5811 | 0.6695 | 0.0274 | 0.0518 | 0.5051 |

Note: kNN external evaluation is on a stratified subset for runtime feasibility.

### LR threshold 0.85 external confusion matrix

- TP: 240,376
- TN: 118,134
- FP: 166,181
- FN: 43,939
- Precision: 0.5912
- Recall: 0.8455
- F1: 0.6959

### LR fine threshold 0.87

- Precision: 0.5913
- Recall: 0.8339
- F1: 0.6920
- Accuracy: 0.6288

### Best ablation

- `dedup_low_pos_weight`
- Threshold: 0.70
- Precision: 0.6502
- Recall: 0.8110
- F1: 0.7218

## Remaining Work

- If submission requires a fully clean package, remove generated caches and temporary marker files after final verification.
- If more model depth is needed, add CatBoost/LightGBM/XGBoost only as non-core upper-bound references.
- For a stronger final report, summarize Stage 6 cascade results in the integrated report and defense deck.
