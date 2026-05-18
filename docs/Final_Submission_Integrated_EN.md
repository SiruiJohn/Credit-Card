# AI3013 Course Project Document (Final Integrated Version)

## 0. Project Information
- Project Title: Credit Card Fraud Detection and Cross-Dataset Generalization via From-Scratch Machine Learning Models
- Task Type: Supervised Binary Classification (`0=normal`, `1=fraud`)
- Application Scenario: Fraud transaction identification and early warning in financial risk control
- Core Objective: Improve precision and F1 under a high-recall constraint, while enhancing cross-distribution generalization

## 1. Abstract
This project builds a complete pipeline for credit card fraud detection, covering data analysis, feature processing, evaluation protocol design, from-scratch model implementation, and external generalization validation. We perform stratified cross-validation on the main dataset and distribution-shift testing on the 2023 external dataset. Two from-scratch models, Logistic Regression and Gaussian Naive Bayes, are compared. We further conduct threshold scanning, confusion-matrix analysis, and ablation studies. Results show that Logistic Regression consistently outperforms Gaussian Naive Bayes on both the main and external datasets, achieving a better F1 and PR-AUC while maintaining high recall, and is therefore selected as the primary model for the next stage.

## 2. Problem Background and Data Description (Introduction + Dataset)
### 2.1 Problem Background
Credit card fraud detection is a typical highly imbalanced classification problem. If only Accuracy is considered, minority fraud samples can be largely ignored. Therefore, this project uses Recall, F1, and PR-AUC as primary metrics, emphasizing the balance between risk coverage and false-alarm control.

### 2.2 Data Sources and Scale
| Dataset | Path | #Samples | #Features | Label Distribution |
|---|---|---:|---:|---|
| Main dataset | `Credit Card Fraud Detection/creditcard.csv` | 284,807 | 31 | 0:284,315 / 1:492 |
| External validation dataset | `Credit Card Fraud Detection_2023/creditcard_2023.csv` | 568,630 | 31 | 0:284,315 / 1:284,315 |

### 2.3 Data Processing Findings
- The main dataset has 0 missing values and 1,081 duplicate rows; deduplication is included in ablation experiments.
- The `id` column is removed from the external dataset to avoid pseudo-contributions from identifier-like features.
- The amount feature shows a heavy-tail distribution; a standardized feature-processing protocol is applied and evaluated.

## 3. Method and Implementation (Methodology)
### 3.1 Model Implementation
- Logistic Regression (from scratch)
- Gaussian Naive Bayes (from scratch)
- Implementation follows course constraints: no prebuilt ML model libraries; core training logic is implemented with `NumPy/Pandas`.

### 3.2 Evaluation Protocol
- Main dataset: `5-fold` stratified cross-validation
- External validation: threshold selected on the main dataset is transferred to the 2023 dataset
- Metrics:
- Core: Recall, F1, PR-AUC
- Supporting: Precision, ROC-AUC, Accuracy
- Threshold strategy: scan `0.05~0.95`, prioritize `Recall >= 0.85`

### 3.3 Data Leakage Control
- Standardization statistics are computed using training data only.
- All model comparisons use a unified split and unified metric pipeline.

## 4. Experimental Study and Result Analysis
### 4.1 Baseline Comparison (5-fold + External Validation)
| Model | Main Recall | Main F1 | Main PR-AUC | External Recall | External F1 | External PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8605 | 0.4428 | 0.7116 | 0.8466 | 0.6949 | 0.7469 |
| Gaussian Naive Bayes | 0.8435 | 0.1092 | 0.4090 | 0.7974 | 0.6411 | 0.4939 |
| kNN | 0.8068 | 0.5811 | 0.6695 | 0.0274 | 0.0518 | 0.5051 |

Conclusion: Logistic Regression outperforms Gaussian Naive Bayes on both internal and external evaluations, especially on F1 and PR-AUC, indicating a stronger precision-recall trade-off and better robustness under distribution shift.

### 4.2 Threshold Analysis and External Confusion Matrix
- Selected threshold on main data: `0.85`
- External performance at transferred threshold:
- Precision = `0.5912`
- Recall = `0.8455`
- F1 = `0.6959`
- Confusion matrix (external dataset):
- TP = `240376`
- TN = `118134`
- FP = `166181`
- FN = `43939`

Interpretation: The current operating point is recall-oriented and meets risk-coverage goals, while still leaving room for false-positive reduction.

### 4.3 Ablation Study
| Setting | F1_ext | Recall_ext | Precision_ext | Threshold |
|---|---:|---:|---:|---:|
| dedup_low_pos_weight | 0.7218 | 0.8110 | 0.6502 | 0.70 |
| no_dedup_auto_weight | 0.6977 | 0.8343 | 0.5996 | 0.85 |
| baseline_dedup_auto_weight | 0.6974 | 0.8382 | 0.5971 | 0.85 |
| dedup_high_pos_weight | 0.6960 | 0.8618 | 0.5838 | 0.90 |
| dedup_with_amount_raw | 0.6873 | 0.8840 | 0.5622 | 0.85 |

Key findings:
- Moderately reducing positive-class weight improves precision and yields the best F1.
- Deduplication vs. non-deduplication has limited impact under the current setup.
- Adding raw amount features can increase recall but may reduce precision.

### 4.4 Notebook-Inspired Extensions
After reviewing the two external notebooks, three extensions were added without changing the core from-scratch course requirement:

- External 2023 EDA: class distribution, class-wise histograms/boxplots, correlation heatmap, and sample pairplot-style figures.
- Scaling ablation: standard, min-max, and robust scaling for from-scratch LR and kNN.
- Mature library benchmark: scikit-learn LR, Random Forest, and HistGradientBoosting as non-core references.

In the scaling ablation, LR with min-max scaling reaches Precision=`0.9995`, Recall=`0.7693`, F1=`0.8694`, and PR-AUC=`0.9634` on the external dataset, showing that preprocessing can strongly affect cross-dataset transfer. This result is kept as a promising follow-up configuration, while the main report continues to use the standard-scaling baseline as the unified comparison setting.

The mature-library benchmark shows that tree models can obtain high PR-AUC, but their main-data threshold transfers poorly to the external dataset. This reinforces the need for cross-dataset threshold validation instead of relying on same-distribution accuracy alone.

### 4.5 OOF Threshold and MinMax Mainline Review
To reduce optimistic bias from selecting thresholds on full in-sample main-data scores, we add a 5-fold out-of-fold (OOF) threshold-selection experiment. The threshold is selected from OOF scores on the main dataset, then transferred to the full external 2023 dataset.

| Scaler | OOF Threshold | External Precision | External Recall | External F1 | External PR-AUC |
|---|---:|---:|---:|---:|---:|
| standard | 0.85 | 0.5894 | 0.8466 | 0.6949 | 0.7469 |
| minmax | 0.41 | 0.9996 | 0.5695 | 0.7256 | 0.9640 |

Conclusion: MinMax scaling still improves external F1 under the stricter OOF-threshold protocol and nearly eliminates false positives (FP=59), but external recall drops to 0.5695 and no longer satisfies the high-recall target. It is therefore best treated as a low-false-alarm candidate, while standard-scaling LR remains the high-recall baseline.

## 5. Current Progress and Next Steps (Progress + Next Steps)
### 5.1 Completed
- Stage 1 completed: EDA, data preprocessing, and evaluation protocol finalization
- Stage 2 completed: from-scratch model implementation, 5-fold baseline, external validation
- Advanced analysis completed: threshold scanning, confusion matrix export, ablation study, cost-sensitive optimization, temporal robustness analysis, and error-case analysis
- Notebook-inspired extensions completed: external EDA figures, scaling ablation, and non-core mature library benchmark
- Stage 5 completed: OOF threshold selection and MinMax mainline review

### 5.2 Next Steps
- Convert useful EDA ideas from the external notebooks into additional project figures
- Add mature library models as non-core upper-bound benchmark experiments
- Consolidate presentation materials while keeping report metrics aligned with `outputs/`

## 6. Reproducibility and Submission File Mapping
### 6.1 Key Scripts
- `scripts/eda_stage1.py`
- `scripts/prepare_stage1_data.py`
- `scripts/train_phase2.py`
- `scripts/plot_phase2_results.py`
- `scripts/phase2_advanced_analysis.py`
- `scripts/lr_final_analysis.py`
- `scripts/cost_sensitive_threshold.py`
- `scripts/temporal_robustness.py`
- `scripts/error_case_analysis.py`
- `scripts/external_eda_notebook_inspired.py`
- `scripts/scaling_ablation.py`
- `scripts/library_benchmark.py`
- `scripts/minmax_oof_threshold_analysis.py`

### 6.2 Key Result Files
- `outputs/stage1/eda_summary.md`
- `outputs/stage2/phase2_baseline_summary.md`
- `outputs/stage2/phase2_advanced_summary.md`
- `outputs/stage2/ablation_results.csv`
- `outputs/stage2/confusion_matrix_ext2023.json`
- `outputs/stage2/lr_final_analysis_summary.md`
- `outputs/stage3/cost_sensitive_summary.md`
- `outputs/stage3/temporal_robustness_summary.md`
- `outputs/stage3/error_case_summary.md`
- `outputs/stage4/external_eda_notebook_inspired_summary.md`
- `outputs/stage4/scaling_ablation_summary.md`
- `outputs/stage4/library_benchmark_summary.md`
- `outputs/stage5/oof_minmax_summary.md`

## 7. Submission Checklist (Course Requirement Alignment)
- Real-world problem modeling completed: credit card fraud detection
- At least two models compared: LR and GNB
- From-scratch implementation and unified evaluation protocol completed
- Cross-validation, external generalization, visualization, and ablation all covered
- Reproducible scripts and result-file mapping completed

---

This integrated version is prepared in final-submission style and unifies the core content and metric definitions from the project plan, stage notes, evaluation protocol, and experimental draft.
