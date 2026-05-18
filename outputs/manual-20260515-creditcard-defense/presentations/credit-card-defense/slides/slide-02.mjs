import { C, addBg, addCallout, addFooter, addMetric, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "一句话结论", "Executive Summary");
  addMetric(slide, ctx, "主线模型", "Logistic Regression", "从零实现，外部集表现最稳", 78, 150, 310, C.red);
  addMetric(slide, ctx, "高召回运行点", "Recall 0.8466", "Standard scaling, OOF threshold 0.85", 430, 150, 310, C.blue);
  addMetric(slide, ctx, "低误报运行点", "FP = 59", "MinMax scaling, OOF threshold 0.41", 782, 150, 310, C.teal);
  addCallout(slide, ctx, "为什么有两个推荐模式", "欺诈检测不是单一指标最大化。若漏报成本更高，优先使用标准缩放 LR；若人工审核容量很紧，MinMax LR 能显著减少误报。", 78, 330, 318, 160, C.red, C.paper);
  addCallout(slide, ctx, "最重要的工程修复", "外部 prepared 数据已恢复到 568,630 行，训练从 quick mode 切换到完整 5-fold，所有关键结果重新生成。", 430, 330, 318, 160, C.blue, C.paper);
  addCallout(slide, ctx, "Notebook 被吸收的价值", "参考 notebook 中的 EDA、缩放实验和成熟库 benchmark 已转成可复跑脚本，并写入报告和 PPT。", 782, 330, 318, 160, C.teal, C.paper);
  ctx.addText(slide, { text: "答辩主线：问题定义 → 数据修复 → 模型与验证 → 阈值决策 → 运行建议。", left: 124, top: 545, width: 900, height: 34, fontSize: 19, bold: true, color: C.ink, align: "center" });
  addFooter(slide, ctx, 2);
  return slide;
}
