import { C, addBg, addCallout, addFooter, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "模型设计：保留从零实现的可解释性", "Model Design");
  addCallout(slide, ctx, "Logistic Regression", "核心主线模型。适合解释线性特征贡献，训练稳定，可通过阈值调整实现不同召回/误报取舍。", 82, 150, 310, 170, C.red, C.softRed);
  addCallout(slide, ctx, "Gaussian Naive Bayes", "概率型解释基线。假设强、训练快，可作为 LR 之外的低复杂度对照。", 430, 150, 310, 170, C.blue, C.softBlue);
  addCallout(slide, ctx, "k-Nearest Neighbors", "非参数对照模型。主数据 CV 尚可，但外部迁移明显失败，说明距离度量对分布偏移敏感。", 778, 150, 310, 170, C.teal, C.softTeal);
  ctx.addShape(slide, { left: 98, top: 390, width: 960, height: 108, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "为什么还需要成熟库 benchmark？", left: 130, top: 416, width: 390, height: 26, fontSize: 20, bold: true, color: C.ink });
  ctx.addText(slide, {
    text: "从零实现是课程核心；成熟库结果用于 sanity check，帮助判断自实现模型是否在合理范围内，而不是替代主线实验。",
    left: 130,
    top: 456,
    width: 870,
    height: 30,
    fontSize: 17,
    color: C.charcoal,
  });
  addFooter(slide, ctx, 7);
  return slide;
}
