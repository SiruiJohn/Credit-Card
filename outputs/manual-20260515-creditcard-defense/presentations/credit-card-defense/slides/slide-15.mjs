import { C, addBg, addFooter, addTextBlock, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "交付物与下一步", "Deliverables and Roadmap");
  addTextBlock(slide, ctx, "已完成交付", "README、项目状态日志、Notebook 扩展说明、综合报告、英文报告、LaTeX bundle、答辩 PPT 和 stage1-5 输出。", 80, 148, 470, 150, C.blue);
  addTextBlock(slide, ctx, "模型层下一步", "加入概率校准曲线、Platt/Isotonic calibration；尝试 LightGBM/XGBoost 作为强基线，但保留从零实现为课程核心。", 610, 148, 470, 150, C.red);
  addTextBlock(slide, ctx, "验证层下一步", "补充更严格的时间切分验证、漂移检测、不同成本假设下的阈值报告。", 80, 342, 470, 150, C.teal);
  addTextBlock(slide, ctx, "解释层下一步", "增加 FP/FN 样本剖析、特征贡献解释、金额和时间分组表现，让误差分析更贴近业务。", 610, 342, 470, 150, C.gold);
  ctx.addText(slide, { text: "项目当前已经具备完整提交形态；继续优化的重点应放在校准、漂移和解释性。", left: 140, top: 548, width: 880, height: 28, fontSize: 18, bold: true, color: C.ink, align: "center" });
  addFooter(slide, ctx, 15);
  return slide;
}
