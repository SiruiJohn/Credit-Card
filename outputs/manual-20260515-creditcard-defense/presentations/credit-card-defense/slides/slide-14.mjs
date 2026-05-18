import { C, addBg, addFooter, addPill, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "最终推荐：双运行模式", "Recommendation");
  ctx.addShape(slide, { left: 88, top: 152, width: 458, height: 306, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  addPill(slide, ctx, "High Coverage Mode", 120, 188, 200, C.red);
  ctx.addText(slide, { text: "标准缩放 LR", left: 120, top: 248, width: 320, height: 36, fontSize: 25, bold: true, color: C.ink });
  ctx.addText(slide, { text: "阈值 0.85\nRecall 0.8466\nF1 0.6949\n适合欺诈漏报成本高、需要覆盖更多风险交易的场景。", left: 120, top: 308, width: 350, height: 108, fontSize: 17, color: C.charcoal });
  ctx.addShape(slide, { left: 620, top: 152, width: 458, height: 306, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  addPill(slide, ctx, "Low Alert Load Mode", 652, 188, 220, C.teal);
  ctx.addText(slide, { text: "MinMax LR", left: 652, top: 248, width: 320, height: 36, fontSize: 25, bold: true, color: C.ink });
  ctx.addText(slide, { text: "阈值 0.41\nPrecision 0.9996\nF1 0.7256\n适合人工审核容量有限、误报成本更敏感的场景。", left: 652, top: 308, width: 350, height: 108, fontSize: 17, color: C.charcoal });
  ctx.addText(slide, { text: "答辩建议：不要说“某个模型绝对最好”，而是说明不同阈值和缩放方式对应不同业务成本。", left: 146, top: 524, width: 860, height: 36, fontSize: 18, bold: true, color: C.ink, align: "center" });
  addFooter(slide, ctx, 14);
  return slide;
}
