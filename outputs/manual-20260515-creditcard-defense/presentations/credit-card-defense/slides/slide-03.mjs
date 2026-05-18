import { C, addBg, addFooter, addMetric, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "数据视角：极端不平衡与外部压力测试", "Data Landscape");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage1/class_distribution_main.png`, left: 78, top: 145, width: 420, height: 300, fit: "contain", alt: "Main class distribution" });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage4/external_class_distribution.png`, left: 590, top: 145, width: 400, height: 300, fit: "contain", alt: "External class distribution" });
  addMetric(slide, ctx, "Main dataset", "284,807 rows", "Fraud ratio: about 0.17%", 86, 466, 270, C.red);
  addMetric(slide, ctx, "External 2023", "568,630 rows", "Balanced: 284,315 / 284,315", 436, 466, 270, C.teal);
  addMetric(slide, ctx, "Evaluation risk", "Accuracy fails", "Use Recall, F1, PR-AUC, confusion matrix", 786, 466, 270, C.blue);
  addFooter(slide, ctx, 3);
  return slide;
}
