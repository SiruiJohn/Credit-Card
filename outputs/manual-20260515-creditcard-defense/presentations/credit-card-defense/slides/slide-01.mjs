import { C, addBg, addFooter, addMiniStat, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  ctx.addShape(slide, { left: 704, top: 0, width: 576, height: 720, fill: "#FFFFFFB8", line: ctx.line() });
  await ctx.addImage(slide, {
    path: `${repoRoot}/outputs/stage4/external_correlation_heatmap_top.png`,
    left: 760,
    top: 106,
    width: 404,
    height: 376,
    fit: "contain",
    alt: "External correlation heatmap",
  });
  ctx.addText(slide, { text: "信用卡欺诈检测项目", left: 72, top: 102, width: 580, height: 62, fontSize: 39, bold: true, color: C.ink, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Credit Card Fraud Detection", left: 74, top: 174, width: 420, height: 30, fontSize: 20, bold: true, color: C.red });
  ctx.addText(slide, { text: "从零实现模型 · 跨数据集验证 · 阈值与成本敏感分析", left: 74, top: 238, width: 610, height: 34, fontSize: 22, color: C.charcoal });
  ctx.addShape(slide, { left: 74, top: 308, width: 526, height: 2, fill: C.line, line: ctx.line() });
  addMiniStat(slide, ctx, "Best high-recall baseline", "LR Recall 0.8466", 74, 352, 242, C.red, C.softRed);
  addMiniStat(slide, ctx, "Low-alert candidate", "MinMax F1 0.7256", 342, 352, 242, C.teal, C.softTeal);
  ctx.addText(slide, {
    text: "本项目从数据修复、模型实现、外部验证到答辩材料形成闭环。核心结论是：标准缩放 LR 适合作为高召回主线，MinMax LR 适合作为低误报候选模式。",
    left: 74,
    top: 470,
    width: 560,
    height: 88,
    fontSize: 18,
    color: C.ink,
  });
  addFooter(slide, ctx, 1);
  return slide;
}
