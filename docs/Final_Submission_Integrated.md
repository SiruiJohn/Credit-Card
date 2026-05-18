# AI3013 课程项目文档（最终提交整合版）

## 0. 项目信息
- 项目题目：基于从零实现机器学习模型的信用卡欺诈检测与跨数据集泛化分析
- 任务类型：监督学习二分类（`0=normal`, `1=fraud`）
- 应用场景：金融风控中的欺诈交易识别与预警
- 核心目标：在高召回约束下优化精确率与 F1，提升跨分布数据泛化能力

## 1. 摘要（Abstract）
本项目围绕信用卡欺诈检测任务，完成了从数据分析、特征处理、评估协议设计到从零实现模型训练与外部泛化验证的完整流程。我们在主数据集上进行分层交叉验证，并在 2023 外部数据集上进行分布迁移测试。实验比较了 Logistic Regression 与 Gaussian Naive Bayes 两个从零实现模型，并进一步进行了阈值扫描、混淆矩阵分析和消融实验。结果表明，Logistic Regression 在主数据与外部数据上均表现更优，能够在保持较高召回率的同时获得更好的 F1 与 PR-AUC，适合作为后续阶段主模型。

## 2. 问题背景与数据说明（Introduction + Dataset）
### 2.1 问题背景
信用卡欺诈检测属于典型高不平衡分类问题。若仅关注 Accuracy，模型容易忽视少数类欺诈样本，因此本项目以 Recall、F1、PR-AUC 作为主要评估指标，强调“高风险覆盖能力”与“误报控制”的平衡。

### 2.2 数据来源与规模
| 数据集 | 路径 | 样本量 | 特征数 | 标签分布 |
|---|---|---:|---:|---|
| 主数据集 | `Credit Card Fraud Detection/creditcard.csv` | 284,807 | 31 | 0:284,315 / 1:492 |
| 外部验证集 | `Credit Card Fraud Detection_2023/creditcard_2023.csv` | 568,630 | 31 | 0:284,315 / 1:284,315 |

### 2.3 数据处理结论
- 主数据缺失值为 0，重复样本 1,081 条，已纳入去重相关消融。
- 外部数据去除 `id` 字段，避免标识类信息对建模产生伪贡献。
- 金额特征存在明显长尾，采用标准化口径并在实验中评估其影响。

## 3. 方法与实现（Methodology）
### 3.1 模型实现
- Logistic Regression（从零实现）
- Gaussian Naive Bayes（从零实现）
- 实现要求满足课程限制：不依赖现成机器学习模型库，核心训练逻辑由 `NumPy/Pandas` 实现

### 3.2 评估协议
- 主数据：`5-fold` 分层交叉验证
- 外部验证：主数据选阈值后迁移到 2023 数据
- 指标体系：
- 核心：Recall、F1、PR-AUC
- 辅助：Precision、ROC-AUC、Accuracy
- 阈值策略：`0.05~0.95` 扫描，优先满足 `Recall >= 0.85`

### 3.3 数据泄漏控制
- 标准化参数仅由训练数据计算
- 模型比较使用统一切分与统一指标口径

## 4. 实验结果（Experimental Study and Result Analysis）
### 4.1 基线比较（5-fold + 外部验证）
| 模型 | 主数据 Recall | 主数据 F1 | 主数据 PR-AUC | 外部 Recall | 外部 F1 | 外部 PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8605 | 0.4428 | 0.7116 | 0.8466 | 0.6949 | 0.7469 |
| Gaussian Naive Bayes | 0.8435 | 0.1092 | 0.4090 | 0.7974 | 0.6411 | 0.4939 |
| kNN | 0.8068 | 0.5811 | 0.6695 | 0.0274 | 0.0518 | 0.5051 |

结论：Logistic Regression 在主数据与外部数据上均显著优于 Gaussian Naive Bayes，尤其在 F1 与 PR-AUC 上优势明显，体现了更好的精确率-召回率平衡与分布迁移鲁棒性。

### 4.2 阈值分析与外部混淆矩阵
- 主数据选定阈值：`0.85`
- 迁移到外部数据后：
- Precision = `0.5912`
- Recall = `0.8455`
- F1 = `0.6959`
- 混淆矩阵（外部集）：
- TP = `240376`
- TN = `118134`
- FP = `166181`
- FN = `43939`

解读：当前工作点偏向高召回，已满足风险覆盖目标，但仍存在误报优化空间。

### 4.3 消融实验
| 配置 | F1_ext | Recall_ext | Precision_ext | Threshold |
|---|---:|---:|---:|---:|
| dedup_low_pos_weight | 0.7218 | 0.8110 | 0.6502 | 0.70 |
| no_dedup_auto_weight | 0.6977 | 0.8343 | 0.5996 | 0.85 |
| baseline_dedup_auto_weight | 0.6974 | 0.8382 | 0.5971 | 0.85 |
| dedup_high_pos_weight | 0.6960 | 0.8618 | 0.5838 | 0.90 |
| dedup_with_amount_raw | 0.6873 | 0.8840 | 0.5622 | 0.85 |

关键发现：
- 适度降低正类权重有助于提升精确率，从而取得最佳 F1。
- 去重与不去重在当前方案下差异有限。
- 加入原始金额特征可进一步提升召回，但会带来精确率下降。

### 4.4 Notebook 启发扩展
参考两份外部 notebook 后，项目新增了三类扩展，但不改变“从零实现模型”为核心的课程口径：

- 外部 2023 数据集 EDA：新增 class distribution、按类别分组的直方图/箱线图、相关性热力图、pairplot 风格采样图。
- 缩放方法消融：比较 standard、min-max、robust scaling 对从零 LR 与 kNN 的影响。
- 成熟库模型参考：使用 scikit-learn LR、Random Forest、HistGradientBoosting 作为非核心 benchmark。

缩放消融中，LR + min-max scaling 在外部集上达到 Precision=`0.9995`、Recall=`0.7693`、F1=`0.8694`、PR-AUC=`0.9634`，说明预处理会显著影响跨数据集迁移表现。该结果作为后续优化方向保留，主报告仍以 standard scaling baseline 作为统一比较口径。

成熟库 benchmark 显示，树模型虽然 PR-AUC 较高，但主数据阈值迁移到外部集后 Recall 很低，进一步说明不能只看同分布 accuracy 或排序能力，必须进行跨数据集阈值验证。

### 4.5 OOF 阈值与 MinMax 主线复核
为避免直接用全量主数据分数选阈值带来的乐观偏差，项目新增 5-fold out-of-fold（OOF）阈值选择实验：先在主数据上生成 OOF 分数并选择阈值，再训练全量模型迁移到外部集。

| Scaler | OOF Threshold | External Precision | External Recall | External F1 | External PR-AUC |
|---|---:|---:|---:|---:|---:|
| standard | 0.85 | 0.5894 | 0.8466 | 0.6949 | 0.7469 |
| minmax | 0.41 | 0.9996 | 0.5695 | 0.7256 | 0.9640 |

结论：MinMax scaling 在更严格的 OOF 阈值口径下仍提升外部 F1，并几乎消除误报（FP=59），但外部 Recall 降至 0.5695，不再满足高召回目标。因此它适合作为“低误报策略”候选；standard scaling LR 仍是高召回主线 baseline。

## 5. 当前进展与下一步计划（Progress + Next Steps）
### 5.1 已完成
- 完成第 1 阶段：EDA、数据处理与评估协议定稿
- 完成第 2 阶段：从零模型实现、5-fold 基线、外部验证
- 完成增强分析：阈值扫描、混淆矩阵导出、消融实验、成本敏感优化、时间鲁棒性分析、错误案例分析
- 完成 Notebook 启发扩展：外部 EDA 图、缩放消融、成熟库非核心 benchmark
- 完成 Stage5：OOF 阈值选择与 MinMax 主线复核

### 5.2 下一步
- 将外部 notebook 中的 EDA 图形思路整理为附加图表
- 追加成熟库模型作为非核心上限参考实验
- 进入答辩材料整合，并保持报告数字与 `outputs/` 结果文件一致

## 6. 复现与提交文件映射（Reproducibility）
### 6.1 关键脚本
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

### 6.2 关键结果文件
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

## 7. 课程要求对齐清单（Submission Checklist）
- 已完成真实问题建模：信用卡欺诈检测
- 已完成至少两个模型比较：LR 与 GNB
- 已完成从零实现与统一评估协议
- 已覆盖交叉验证、外部泛化验证、可视化与消融分析
- 已形成可复现实验脚本与结果文件清单

---

本整合版用于“最终提交风格”归档，统一了原有 `项目规划`、`阶段说明`、`评估协议` 与 `实验结果草稿` 的核心内容和口径。
