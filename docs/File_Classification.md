# 项目文件分类清单

## 1. 原始数据

- `Credit Card Fraud Detection/creditcard.csv`
  - 主数据集，284,807 行，类别分布 0:284,315 / 1:492。
- `Credit Card Fraud Detection_2023/creditcard_2023.csv`
  - 外部 2023 验证集，568,630 行，类别分布 0:284,315 / 1:284,315。

## 2. 中间处理数据

- `outputs/stage1/main_prepared.csv`
  - 主数据去重后版本，283,726 行，新增 `Amount_z`。
- `outputs/stage1/ext2023_prepared.csv`
  - 外部数据去除 `id` 后版本，568,630 行，新增 `Amount_z`。
- `outputs/stage1/prepare_metadata.json`
  - 数据准备元信息、行数校验和类别分布校验。

## 3. 核心代码脚本

### Stage 1: EDA 与数据准备

- `scripts/s01_prepare_data.py`
- `scripts/s01_eda.py`

### Stage 2: 从零实现模型、基线、阈值与消融

- `scripts/models_from_scratch.py`
- `scripts/s02_train_baseline.py`
- `scripts/s02_plot_baseline.py`
- `scripts/s02_ablation.py`
- `scripts/s02_lr_analysis.py`
- `scripts/s02_tune_knn.py`
- `scripts/s02_tune_knn_advanced.py`

### Stage 3: 增强分析

- `scripts/s03_cost_threshold.py`
- `scripts/s03_temporal_robustness.py`
- `scripts/s03_error_analysis.py`

### Stage 4: Notebook 启发扩展

- `scripts/s04_external_eda.py`
- `scripts/s04_scaling_ablation.py`
- `scripts/s04_library_benchmark.py`

### Stage 5: OOF 阈值与 MinMax 主线复核

- `scripts/s05_oof_validation.py`

### Stage 6: 两阶段 Cascade 与高级业务指标

- `scripts/s06_cascade_architecture.py`
- `scripts/utils.py`

## 4. 结果产物

### Stage 1

- `outputs/stage1/eda_summary.md`
- `outputs/stage1/eda_main.json`
- `outputs/stage1/eda_2023.json`
- `outputs/stage1/class_distribution_main.png`
- `outputs/stage1/amount_distribution_main.png`

### Stage 2

- `outputs/stage2/phase2_baseline_results.json`
- `outputs/stage2/phase2_baseline_summary.md`
- `outputs/stage2/phase2_advanced_summary.md`
- `outputs/stage2/ablation_results.csv`
- `outputs/stage2/confusion_matrix_ext2023.json`
- `outputs/stage2/lr_final_analysis_summary.md`
- `outputs/stage2/lr_final_confusion_matrix_ext2023.json`
- `outputs/stage2/lr_feature_importance.csv`
- `outputs/stage2/knn_tuning_summary.md`
- `outputs/stage2/knn_advanced_summary.md`
- `outputs/stage2/*.png`

### Stage 3

- `outputs/stage3/cost_sensitive_summary.md`
- `outputs/stage3/temporal_robustness_summary.md`
- `outputs/stage3/error_case_summary.md`
- `outputs/stage3/*.csv`
- `outputs/stage3/*.png`

### Stage 4

- `outputs/stage4/external_eda_notebook_inspired_summary.md`
- `outputs/stage4/scaling_ablation_summary.md`
- `outputs/stage4/library_benchmark_summary.md`
- `outputs/stage4/*.csv`
- `outputs/stage4/*.png`

### Stage 5

- `outputs/stage5/oof_minmax_summary.md`
- `outputs/stage5/oof_minmax_comparison.csv`
- `outputs/stage5/oof_cost_sensitive_selection.csv`
- `outputs/stage5/*.png`

### Stage 6

- `outputs/stage6/cascade_summary.md`
- `outputs/stage6/cascade_comparison.json`
- `outputs/stage6/lift_curve.json`
- `outputs/stage6/cumulative_gain.json`
- `outputs/stage6/ece_main.json`
- `outputs/stage6/ece_ext.json`
- `outputs/stage6/*.png`

## 5. 文档、报告与答辩材料

- `README.md`
- `requirements.txt`
- `docs/Project_Status_and_Fix_Log.md`
- `docs/Notebook_Extensions.md`
- `docs/evaluation_protocol.md`
- `docs/Final_Submission_Integrated.md`
- `docs/Final_Submission_Integrated_EN.md`
- `docs/Progress_Report_Experimental_Study_Draft.md`
- `docs/final_report_main.tex`
- `docs/Credit_Card_Fraud_Detection_Defense.pptx`
- `latex_bundle/main.tex`
- `latex_bundle/figures/`
- `latex_bundle/refs/references.bib`

## 6. PPT 构建源文件

- `outputs/manual-20260515-creditcard-defense/presentations/credit-card-defense/slides/`

用途：保留可复建的 PPT 源文件，便于后续修改答辩材料。预览图、构建运行时和临时导出文件不纳入 GitHub；最终 PPT 已放在 `docs/Credit_Card_Fraud_Detection_Defense.pptx`。

## 7. 外部 notebook 参考

- `unmasking-deception-innovations-in-credit-card.ipynb`
- `credit-card-fraud-detection-achieving-99-acc.ipynb`

用途：作为 EDA 可视化、成熟库模型 benchmark、Pipeline 思路参考；不作为课程核心从零实现结果。

## 8. 本地运行依赖

- `.codex_pydeps/`
  - 当前工作区本地安装的扩展依赖，主要用于 `matplotlib`、`scipy` 和 `scikit-learn`。
  - 复现命令见根目录 `README.md`。

## 9. 可清理文件

以下文件不影响结果复现，可在最终打包前清理：

- `scripts/__pycache__/`
- `outputs/stage2/_knn_marker.txt`
- `outputs/stage2/_train_debug.log`
- `outputs/stage2/_tune_probe.txt`
- `outputs/stage2/knn_test_log.txt`
