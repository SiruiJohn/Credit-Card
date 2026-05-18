import { C, addBg, addFooter, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "Notebook 增强：EDA 变成可复现证据", "Stage4 EDA");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage4/external_feature_histograms_by_class.png`, left: 70, top: 148, width: 500, height: 345, fit: "contain", alt: "External feature histograms by class" });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage4/external_correlation_heatmap_top.png`, left: 640, top: 145, width: 415, height: 350, fit: "contain", alt: "External correlation heatmap" });
  ctx.addText(slide, {
    text: "外部集与 Class 相关性最高的特征包括 V14、V12、V4、V11、V3、V10、V9、V16。图表用于解释模型为何能区分部分欺诈模式，也为误差分析提供方向。",
    left: 106,
    top: 526,
    width: 940,
    height: 44,
    fontSize: 16.5,
    color: C.ink,
    bold: true,
    align: "center",
  });
  addFooter(slide, ctx, 11);
  return slide;
}
