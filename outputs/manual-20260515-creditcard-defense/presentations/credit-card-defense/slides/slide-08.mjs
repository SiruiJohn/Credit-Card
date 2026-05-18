import { C, addBg, addFooter, addTable, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "基线结果：LR 是最稳的跨数据集主线", "Baseline Results");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage2/ext_f1.png`, left: 702, top: 145, width: 380, height: 250, fit: "contain", alt: "External F1 comparison" });
  addTable(
    slide,
    ctx,
    ["Model", "CV Recall", "CV F1", "Ext Recall", "Ext F1", "Ext PR-AUC"],
    [
      ["LR", "0.8605", "0.4428", "0.8466", "0.6949", "0.7469"],
      ["GNB", "0.8435", "0.1092", "0.7974", "0.6411", "0.4939"],
      ["kNN", "0.8068", "0.5811", "0.0274", "0.0518", "0.5051"],
    ],
    82,
    166,
    [90, 112, 92, 112, 92, 112],
    40,
  );
  ctx.addText(slide, {
    text: "解释：LR 在外部集上保持最高召回与更好的综合 F1；GNB 可解释但 PR-AUC 弱；kNN 对外部分布偏移非常敏感。",
    left: 92,
    top: 360,
    width: 540,
    height: 68,
    fontSize: 17,
    color: C.ink,
    bold: true,
  });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage2/cv_f1_mean.png`, left: 702, top: 410, width: 380, height: 160, fit: "contain", alt: "CV F1 mean" });
  addFooter(slide, ctx, 8);
  return slide;
}
