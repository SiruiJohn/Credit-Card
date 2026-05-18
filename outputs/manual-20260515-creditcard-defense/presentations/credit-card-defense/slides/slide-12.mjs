import { C, addBg, addFooter, addTitle, repoRoot } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "缩放与库模型：把参考 notebook 的启发转成实验", "Stage4 Benchmarks");
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage4/scaling_ablation_compare.png`, left: 70, top: 152, width: 470, height: 330, fit: "contain", alt: "Scaling ablation comparison" });
  await ctx.addImage(slide, { path: `${repoRoot}/outputs/stage4/library_benchmark_compare.png`, left: 620, top: 152, width: 470, height: 330, fit: "contain", alt: "Library benchmark comparison" });
  ctx.addText(slide, {
    text: "价值：缩放实验发现 MinMax 可能提升外部 F1；库模型 benchmark 提供 sanity check。但最终结论必须经过 Stage5 的 OOF 阈值复核。",
    left: 126,
    top: 524,
    width: 900,
    height: 42,
    fontSize: 17,
    color: C.ink,
    bold: true,
    align: "center",
  });
  addFooter(slide, ctx, 12);
  return slide;
}
