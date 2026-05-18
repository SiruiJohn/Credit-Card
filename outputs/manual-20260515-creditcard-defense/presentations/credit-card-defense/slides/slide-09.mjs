import { C, addBg, addFooter, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "阈值选择：把概率输出变成业务动作", "Threshold Decision");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage2/threshold_scan_ext2023.png`, left: 72, top: 142, width: 510, height: 330, fit: "contain", alt: "External threshold scan" });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage2/lr_confusion_matrix_ext2023_thr087.png`, left: 650, top: 150, width: 410, height: 310, fit: "contain", alt: "LR confusion matrix" });
  ctx.addShape(slide, { left: 112, top: 500, width: 920, height: 70, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "关键点", left: 140, top: 522, width: 90, height: 24, fontSize: 17, bold: true, color: C.red });
  ctx.addText(slide, { text: "阈值不是固定 0.5。项目通过 Recall、Precision、F1 和成本曲线，选择符合业务目标的运行点。", left: 236, top: 522, width: 740, height: 26, fontSize: 16, color: C.ink });
  addFooter(slide, ctx, 9);
  return slide;
}
