# 第1阶段交付说明（已执行）

本阶段目标：完成数据探索、特征分析、评估方案定稿、实验脚手架搭建。

## 目录结构
- `scripts/eda_stage1.py`：输出数据质量、分布、相关性与关键统计，生成图表与报告
- `scripts/prepare_stage1_data.py`：统一字段、删除无效列、可选去重、标准化配置输出
- `docs/evaluation_protocol.md`：指标定义、交叉验证策略、阈值搜索方案
- `outputs/stage1/`：脚本执行后的结果（统计 JSON、图、处理后数据等）

## 运行方式
在项目根目录执行：

```bash
python scripts/eda_stage1.py
python scripts/prepare_stage1_data.py
```

## 第1阶段验收标准
- 完成主数据与 2023 数据的基础质量检查
- 明确不平衡学习的主评估指标（Recall/F1/PR-AUC）
- 确定第2阶段可直接使用的数据输入文件与配置
