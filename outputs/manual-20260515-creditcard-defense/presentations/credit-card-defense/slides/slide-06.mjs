import { C, addBg, addFooter, addTextBlock, addTitle } from "./common.mjs";

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  addBg(slide, ctx);
  addTitle(slide, ctx, "关键修复：项目从“能跑”到“可信”", "Fix Log");
  addTextBlock(slide, ctx, "外部 prepared 文件修复", "恢复 external prepared 数据到 568,630 行，并加入写后读取校验，确保类别分布为 284,315 / 284,315。", 82, 150, 470, 135, C.red);
  addTextBlock(slide, ctx, "训练结果重新生成", "Phase2 baseline 已切换为完整 5-fold，quick_mode=False，避免把调试结果写进最终报告。", 610, 150, 470, 135, C.blue);
  addTextBlock(slide, ctx, "Notebook 内容工程化", "将参考 notebook 的 EDA、缩放比较、库模型对照改写为 Stage4 脚本，产物可复现、可追踪。", 82, 326, 470, 135, C.teal);
  addTextBlock(slide, ctx, "MinMax 结论严格复核", "Stage5 使用 OOF 预测选阈值，再迁移到外部集验证，减少直接看外部集挑阈值的偏差。", 610, 326, 470, 135, C.gold);
  ctx.addText(slide, { text: "修复后的项目主线更清楚：数据可信 → 实验可信 → 结论可信 → 交付可信。", left: 150, top: 528, width: 840, height: 32, fontSize: 18, bold: true, color: C.ink, align: "center" });
  addFooter(slide, ctx, 6);
  return slide;
}
