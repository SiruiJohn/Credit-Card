import { C, addBg, addFooter, addProcessStep, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "项目流水线：从数据到可交付结论", "Project Pipeline");
  const w = 300;
  addProcessStep(slide, ctx, 1, "数据准备", "统一主数据与 external 2023 字段，生成 prepared 数据与元信息校验。", 82, 152, w, C.red);
  addProcessStep(slide, ctx, 2, "从零实现模型", "实现 LR、GaussianNB、kNN，保留课程要求下可解释的核心算法。", 420, 152, w, C.blue);
  addProcessStep(slide, ctx, 3, "完整 5-fold CV", "替换 quick mode，输出稳定的主数据集交叉验证结果。", 758, 152, w, C.teal);
  addProcessStep(slide, ctx, 4, "外部验证", "在完整 external 2023 上复核 LR/GNB；kNN 采用分层外部子集。", 82, 310, w, C.gold);
  addProcessStep(slide, ctx, 5, "阈值与成本", "扫描阈值、混淆矩阵、成本曲线，形成可解释运行点。", 420, 310, w, C.green);
  addProcessStep(slide, ctx, 6, "报告与答辩", "同步 README、报告、LaTeX、图表与 PPT，降低最终提交风险。", 758, 310, w, C.charcoal);
  ctx.addText(slide, { text: "输出目录：outputs/stage1-5 · docs · latex_bundle · defense deck", left: 120, top: 526, width: 900, height: 30, fontSize: 18, color: C.ink, bold: true, align: "center" });
  addFooter(slide, ctx, 5);
  return slide;
}
