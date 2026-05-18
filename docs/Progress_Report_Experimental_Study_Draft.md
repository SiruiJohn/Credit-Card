# Experimental Study and Result Analysis（草稿）

## 1. Experimental Setup
This study evaluates two from-scratch machine learning models for credit card fraud detection: Logistic Regression (LR) and Gaussian Naive Bayes (GNB).  
The main dataset is highly imbalanced and is used for model development and cross-validation, while the 2023 dataset is used as an external validation set to test generalization under distribution shift.

- Main data evaluation: 5-fold stratified cross-validation
- External validation: fixed threshold transferred from main-data calibration
- Core metrics: Recall, F1, PR-AUC
- Supporting metrics: Precision, ROC-AUC, Accuracy

## 2. Baseline Model Comparison
### 2.1 Cross-validation results on main dataset
- LR: Recall = 0.8605, F1 = 0.4428, PR-AUC = 0.7116
- GNB: Recall = 0.8435, F1 = 0.1092, PR-AUC = 0.4090

### 2.2 External validation results on 2023 dataset
- LR: Recall = 0.8466, F1 = 0.6949, PR-AUC = 0.7469
- GNB: Recall = 0.7974, F1 = 0.6411, PR-AUC = 0.4939

### 2.3 Discussion
LR consistently outperforms GNB in both internal validation and external validation, especially on F1 and PR-AUC, indicating a better precision-recall trade-off and stronger robustness under data distribution shift. Therefore, LR is selected as the primary model for advanced analysis and ablation.

## 3. Threshold Analysis and Confusion Matrix
Threshold scanning is performed on the main dataset, and the selected operating threshold is 0.85 (high-recall-oriented setting).  
At this threshold, external validation performance is:

- Precision = 0.5912
- Recall = 0.8455
- F1 = 0.6959

External confusion matrix:
- TP = 240376
- TN = 118134
- FP = 166181
- FN = 43939

These results show that the current operating point achieves strong fraud coverage (high recall), while still leaving room to reduce false positives in later optimization.

## 4. Ablation Study
To better understand the effects of data processing and class weighting, we conducted ablation experiments with different configurations:

- `dedup_low_pos_weight` (best F1):  
  F1 = 0.7218, Recall = 0.8110, Precision = 0.6502, threshold = 0.70
- `no_dedup_auto_weight`:  
  F1 = 0.6977, Recall = 0.8343, Precision = 0.5996, threshold = 0.85
- `baseline_dedup_auto_weight`:  
  F1 = 0.6974, Recall = 0.8382, Precision = 0.5971, threshold = 0.85
- `dedup_high_pos_weight`:  
  F1 = 0.6960, Recall = 0.8618, Precision = 0.5838, threshold = 0.90
- `dedup_with_amount_raw`:  
  F1 = 0.6873, Recall = 0.8840, Precision = 0.5622, threshold = 0.85

### 4.1 Key findings from ablation
- Moderate reduction of positive-class weight improves precision while maintaining high recall, yielding the best F1.
- Deduplication has limited impact on final external performance in the current setup.
- Adding raw amount feature increases recall but may reduce precision, indicating a stronger alarm tendency.

## 5. Current Conclusion
At the current progress stage, the project has established a reproducible pipeline with from-scratch implementation, standardized evaluation protocol, cross-validation evidence, external generalization checks, threshold analysis, and ablation results.  
The evidence supports using Logistic Regression as the main model in the next stage.

## 6. Next-step Experimental Plan
- Convert useful EDA ideas from the two external notebooks into project-native figures
- Add mature library models as non-core upper-bound benchmark experiments
- Consolidate results into final report tables and presentation figures

---

## Figures and Files for Reference
- Baseline summary: `outputs/stage2/phase2_baseline_summary.md`
- Advanced summary: `outputs/stage2/phase2_advanced_summary.md`
- Ablation results: `outputs/stage2/ablation_results.csv`
- Threshold scan curves:
  - `outputs/stage2/threshold_scan_main.png`
  - `outputs/stage2/threshold_scan_ext2023.png`
- Confusion matrix export: `outputs/stage2/confusion_matrix_ext2023.json`
