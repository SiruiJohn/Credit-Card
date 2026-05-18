import { C, addBg, addFooter, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "成本与误差：不仅看分数，也看代价", "Cost and Error Analysis");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage3/cost_curve_ext2023.png`, left: 78, top: 145, width: 485, height: 310, fit: "contain", alt: "External cost curve" });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage3/error_profile_fn_vs_tp.png`, left: 638, top: 145, width: 430, height: 310, fit: "contain", alt: "FN vs TP error profile" });
  ctx.addText(slide, {
    text: "答辩表达：同一模型在不同阈值下会变成不同业务系统。成本函数越偏向漏报惩罚，越应该选择高召回阈值。",
    left: 130,
    top: 506,
    width: 880,
    height: 42,
    fontSize: 18,
    color: C.ink,
    bold: true,
    align: "center",
  });
  addFooter(slide, ctx, 10);
  return slide;
}
