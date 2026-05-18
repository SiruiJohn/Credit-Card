import { C, addBg, addFooter, addMetric, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "Stage5 严格复核：MinMax 是低误报候选", "OOF Threshold Review");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage5/oof_minmax_external_compare.png`, left: 66, top: 142, width: 485, height: 340, fit: "contain", alt: "OOF MinMax external comparison" });
  addMetric(slide, ctx, "Standard LR", "Recall 0.8466", "thr=0.85, F1=0.6949, PR-AUC=0.7469", 630, 145, 410, C.red);
  addMetric(slide, ctx, "MinMax LR", "Precision 0.9996", "thr=0.41, F1=0.7256, Recall=0.5695", 630, 288, 410, C.teal);
  ctx.addShape(slide, { left: 630, top: 444, width: 410, height: 90, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "解释", left: 654, top: 464, width: 80, height: 24, fontSize: 17, bold: true, color: C.red });
  ctx.addText(slide, { text: "MinMax 外部误报仅 59 个，但漏报上升。它不能替代高召回主线，却适合低审核量场景。", left: 734, top: 464, width: 270, height: 48, fontSize: 14.5, color: C.ink });
  addFooter(slide, ctx, 13);
  return slide;
}
