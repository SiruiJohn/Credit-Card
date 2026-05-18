import { C, addBg, addCallout, addFooter, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "评估协议：避免漂亮但无效的准确率", "Evaluation Protocol");
  addCallout(slide, ctx, "1. 主数据交叉验证", "使用 5-fold CV 估计模型在原始不平衡数据上的稳定性。输出 Recall、F1、PR-AUC，不用 Accuracy 做核心结论。", 90, 150, 300, 150, C.red, C.softRed);
  addCallout(slide, ctx, "2. 外部 2023 验证", "把模型迁移到完整 external 2023 数据，检查跨数据集泛化能力，并暴露分布偏移造成的召回/误报变化。", 430, 150, 300, 150, C.blue, C.softBlue);
  addCallout(slide, ctx, "3. 阈值与成本扫描", "不默认使用 0.5 阈值。通过阈值曲线、混淆矩阵和成本函数选择运行点。", 770, 150, 300, 150, C.teal, C.softTeal);
  ctx.addShape(slide, { left: 116, top: 380, width: 920, height: 104, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "本项目的判断标准", left: 148, top: 404, width: 260, height: 28, fontSize: 19, bold: true, color: C.ink });
  ctx.addText(slide, {
    text: "模型必须同时满足：主数据交叉验证可解释、外部数据不崩溃、阈值选择有依据、结果能转化成业务运行建议。",
    left: 148,
    top: 444,
    width: 840,
    height: 32,
    fontSize: 18,
    color: C.charcoal,
  });
  addFooter(slide, ctx, 4);
  return slide;
}
