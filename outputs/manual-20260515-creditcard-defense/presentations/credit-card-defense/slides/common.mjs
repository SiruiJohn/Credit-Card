export const C = {
  bg: "#F7F5F0",
  paper: "#FFFFFF",
  ink: "#172026",
  muted: "#667085",
  line: "#D8D3C7",
  wash: "#ECE7DA",
  softRed: "#FBE7E4",
  softTeal: "#E3F3F1",
  softBlue: "#E7EEF9",
  softGold: "#F8EEDC",
  red: "#B42318",
  teal: "#0E6F68",
  blue: "#2457A6",
  gold: "#B7791F",
  green: "#2E7D32",
  charcoal: "#2E3440",
};

export const repoRoot = "F:/project/Credit Card";

export function addBg(slide, ctx) {
  ctx.addShape(slide, { left: 0, top: 0, width: ctx.W, height: ctx.H, fill: C.bg, line: ctx.line() });
  ctx.addShape(slide, { left: 0, top: 0, width: 12, height: ctx.H, fill: C.red, line: ctx.line() });
  ctx.addShape(slide, { left: 1160, top: 0, width: 120, height: 720, fill: "#FFFFFF66", line: ctx.line() });
}

export function addTitle(slide, ctx, title, kicker = "") {
  if (kicker) {
    ctx.addText(slide, {
      text: kicker,
      left: 70,
      top: 36,
      width: 680,
      height: 28,
      fontSize: 14,
      bold: true,
      color: C.red,
      typeface: ctx.fonts.body,
    });
  }
  ctx.addText(slide, {
    text: title,
    left: 68,
    top: kicker ? 66 : 42,
    width: 980,
    height: 58,
    fontSize: 31,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
  });
}

export function addFooter(slide, ctx, page) {
  ctx.addShape(slide, { left: 68, top: 674, width: 1080, height: 1.2, fill: C.line, line: ctx.line() });
  ctx.addText(slide, {
    text: "Credit Card Fraud Detection Project",
    left: 68,
    top: 686,
    width: 360,
    height: 20,
    fontSize: 11,
    color: C.muted,
  });
  ctx.addText(slide, {
    text: String(page).padStart(2, "0"),
    left: 1138,
    top: 684,
    width: 48,
    height: 22,
    fontSize: 12,
    color: C.muted,
    align: "right",
  });
}

export function addSectionLabel(slide, ctx, label, left = 70, top = 34, color = C.red) {
  ctx.addShape(slide, { left, top, width: 7, height: 18, fill: color, line: ctx.line() });
  ctx.addText(slide, {
    text: label,
    left: left + 14,
    top: top - 2,
    width: 360,
    height: 24,
    fontSize: 12,
    bold: true,
    color,
  });
}

export function addPill(slide, ctx, text, left, top, width, color = C.blue) {
  ctx.addShape(slide, { left, top, width, height: 30, fill: color, line: ctx.line() });
  ctx.addText(slide, {
    text,
    left: left + 12,
    top: top + 6,
    width: width - 24,
    height: 18,
    fontSize: 11,
    bold: true,
    color: "#FFFFFF",
    align: "center",
  });
}

export function addMiniStat(slide, ctx, label, value, left, top, width, color = C.blue, fill = C.paper) {
  ctx.addShape(slide, { left, top, width, height: 74, fill, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: label, left: left + 14, top: top + 12, width: width - 28, height: 20, fontSize: 11.5, color: C.muted, bold: true });
  ctx.addText(slide, { text: value, left: left + 14, top: top + 34, width: width - 28, height: 30, fontSize: 22, color, bold: true, typeface: ctx.fonts.title });
}

export function addMetric(slide, ctx, label, value, note, left, top, width, color = C.teal) {
  ctx.addShape(slide, { left, top, width, height: 112, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: label, left: left + 16, top: top + 14, width: width - 32, height: 24, fontSize: 13, color: C.muted, bold: true });
  ctx.addText(slide, { text: value, left: left + 16, top: top + 40, width: width - 32, height: 42, fontSize: 29, color, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: note, left: left + 16, top: top + 84, width: width - 32, height: 20, fontSize: 10.5, color: C.muted });
}

export function addCallout(slide, ctx, title, body, left, top, width, height, color = C.blue, fill = C.paper) {
  ctx.addShape(slide, { left, top, width, height, fill, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addShape(slide, { left, top, width, height: 6, fill: color, line: ctx.line() });
  ctx.addText(slide, { text: title, left: left + 18, top: top + 18, width: width - 36, height: 26, fontSize: 17, bold: true, color: C.ink });
  ctx.addText(slide, { text: body, left: left + 18, top: top + 54, width: width - 36, height: height - 64, fontSize: 13.5, color: C.charcoal });
}

export function addTextBlock(slide, ctx, title, body, left, top, width, height, accent = C.blue) {
  ctx.addShape(slide, { left, top, width, height, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addShape(slide, { left, top, width: 5, height, fill: accent, line: ctx.line() });
  ctx.addText(slide, { text: title, left: left + 20, top: top + 16, width: width - 34, height: 28, fontSize: 17, bold: true, color: C.ink });
  ctx.addText(slide, { text: body, left: left + 20, top: top + 52, width: width - 36, height: height - 62, fontSize: 14, color: C.charcoal, line: ctx.line(), insets: { left: 0, right: 0, top: 0, bottom: 0 } });
}

export function addTable(slide, ctx, columns, rows, left, top, widths, rowHeight = 38) {
  const totalWidth = widths.reduce((sum, value) => sum + value, 0);
  ctx.addShape(slide, { left, top, width: totalWidth, height: rowHeight, fill: C.ink, line: ctx.line() });
  let x = left;
  columns.forEach((column, index) => {
    ctx.addText(slide, { text: column, left: x + 10, top: top + 10, width: widths[index] - 20, height: 18, fontSize: 11, bold: true, color: "#FFFFFF" });
    x += widths[index];
  });
  rows.forEach((row, rowIndex) => {
    const y = top + rowHeight * (rowIndex + 1);
    ctx.addShape(slide, { left, top: y, width: totalWidth, height: rowHeight, fill: rowIndex % 2 === 0 ? C.paper : "#FAF8F3", line: { style: "solid", fill: C.line, width: 1 } });
    x = left;
    row.forEach((cell, index) => {
      ctx.addText(slide, { text: String(cell), left: x + 10, top: y + 10, width: widths[index] - 20, height: 18, fontSize: 11.5, color: C.charcoal, bold: index === 0 });
      x += widths[index];
    });
  });
}

export function addProcessStep(slide, ctx, index, title, note, left, top, width, color) {
  ctx.addShape(slide, { left, top, width, height: 105, fill: C.paper, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: String(index), left: left + 14, top: top + 15, width: 30, height: 30, fontSize: 16, bold: true, color: "#FFFFFF", fill: color, align: "center", valign: "middle" });
  ctx.addText(slide, { text: title, left: left + 58, top: top + 16, width: width - 72, height: 26, fontSize: 16, bold: true, color: C.ink });
  ctx.addText(slide, { text: note, left: left + 58, top: top + 48, width: width - 72, height: 42, fontSize: 12.5, color: C.muted });
}

export function addBar(slide, ctx, label, value, maxValue, left, top, width, color) {
  const barW = Math.max(3, width * value / maxValue);
  ctx.addText(slide, { text: label, left, top: top - 2, width: 150, height: 24, fontSize: 13, color: C.charcoal, bold: true });
  ctx.addShape(slide, { left: left + 158, top, width, height: 20, fill: "#ECE7DA", line: ctx.line() });
  ctx.addShape(slide, { left: left + 158, top, width: barW, height: 20, fill: color, line: ctx.line() });
  ctx.addText(slide, { text: value.toFixed(4), left: left + 168 + width, top: top - 1, width: 70, height: 22, fontSize: 12, color: C.ink });
}
