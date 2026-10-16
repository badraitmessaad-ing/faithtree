const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa"), md = require("react-icons/md"), bs = require("react-icons/bs");

async function icon(Comp, color = "FFFFFF", px = 512) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: px }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; pres.title = "FaithTree pipeline (Fig. 1)";
  const s = pres.addSlide(); s.background = { color: "FFFFFF" };
  const FONT = "Arial", INK = "1A1A1A", MUTED = "4A4A4A";
  const SLATE = "3D4A5C", SLATE_FILL = "F3F4F6", SLATE_LINE = "8A93A0";
  const BLUE = "2A78D6", BLUE_FILL = "EAF2FC", BLUE_DARK = "1E5FAE", GREEN = "2E9E5B";

  const pill = (x, y, w, text, color, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.21, fill: { color }, line: { color, width: 0 }, objectName: name + " pill" });
    s.addText(text, { x, y, w, h: 0.42, fontFace: FONT, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: name + " label" });
  };

  async function card({ x, y, w, h, n, Icon, title, sub, fill, line, accent, name, check }) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: line, width: 1.75 },
      shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: name + " card" });
    // number badge
    s.addShape(pres.shapes.OVAL, { x: x + 0.14, y: y + 0.14, w: 0.36, h: 0.36, fill: { color: "FFFFFF" }, line: { color: accent, width: 1.5 }, objectName: name + " number badge" });
    s.addText(String(n), { x: x + 0.14, y: y + 0.14, w: 0.36, h: 0.36, fontFace: FONT, fontSize: 12, bold: true, color: accent, align: "center", valign: "middle", margin: 0, objectName: name + " number" });
    // icon circle
    const D = 0.86, cx = x + w / 2;
    s.addShape(pres.shapes.OVAL, { x: cx - D / 2, y: y + 0.2, w: D, h: D, fill: { color: accent }, line: { color: accent, width: 0 }, objectName: name + " icon circle" });
    const I = 0.46;
    s.addImage({ data: await icon(Icon), x: cx - I / 2, y: y + 0.2 + (D - I) / 2, w: I, h: I, altText: title + " icon" });
    if (check) {
      s.addShape(pres.shapes.OVAL, { x: cx + D / 2 - 0.26, y: y + 0.2 + D - 0.3, w: 0.32, h: 0.32, fill: { color: GREEN }, line: { color: "FFFFFF", width: 1.5 }, objectName: name + " check badge" });
      s.addImage({ data: await icon(fa.FaCheck), x: cx + D / 2 - 0.19, y: y + 0.2 + D - 0.23, w: 0.18, h: 0.18, altText: "check" });
    }
    s.addText(title, { x: x + 0.1, y: y + 1.17, w: w - 0.2, h: 0.42, fontFace: FONT, fontSize: 15, bold: true, color: INK, align: "center", valign: "middle", margin: 0, objectName: name + " title" });
    s.addText(sub, { x: x + 0.12, y: y + 1.6, w: w - 0.24, h: h - 1.7, fontFace: FONT, fontSize: 12, color: MUTED, align: "center", valign: "top", margin: 0, objectName: name + " text" });
  }
  const arrow = (x1, x2, y, color, name) => s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color, width: 2.5, endArrowType: "triangle" }, objectName: name });

  // ---------- top row: model building
  pill(0.26, 0.22, 5.4, "Model building  ·  training and validation rows only", SLATE, "Model building");
  const TW = 2.3, TH = 2.55, TY = 0.86, TG = 0.33, TX0 = 0.26;
  const top = [
    [fa.FaDatabase, "Dataset profile", "target column,\ndropped identifiers\n(IPs, ports, timestamps)"],
    [fa.FaSlidersH, "Preprocessing", "leakage-aware:\nmedian imputing,\nclipping, robust scaling"],
    [fa.FaFilter, "SelectKBest", "ANOVA F-test,\nK chosen on\nvalidation"],
    [fa.FaSitemap, "Tree tuning", "restricted space,\nvalidation MCC,\nsize limit"],
    [fa.FaFlagCheckered, "Refit & final test", "refit on train + val,\nthen one test evaluation\nof the frozen model"],
  ];
  const tx = top.map((_, i) => TX0 + i * (TW + TG));
  for (let i = 0; i < top.length; i++) {
    const [Icon, title, sub] = top[i];
    await card({ x: tx[i], y: TY, w: TW, h: TH, n: i + 1, Icon, title, sub, fill: SLATE_FILL, line: SLATE_LINE, accent: SLATE, name: `Step ${i + 1}` });
    if (i < 4) arrow(tx[i] + TW + 0.04, tx[i + 1] - 0.04, TY + 0.63, SLATE, `Arrow ${i + 1}-${i + 2}`);
  }

  // ---------- connector to deployment row
  const BW = 2.95, BH = 2.4, BY = 4.42, BG = 0.33, BX0 = 0.26;
  const bx = [0, 1, 2, 3].map(i => BX0 + i * (BW + BG));
  const ex = tx[4] + TW / 2, ey1 = TY + TH, ey2 = 3.85, dx = bx[0] + BW / 2;
  s.addShape(pres.shapes.LINE, { x: ex, y: ey1, w: 0, h: ey2 - ey1, line: { color: SLATE, width: 2.5 }, objectName: "Connector down" });
  s.addShape(pres.shapes.LINE, { x: dx, y: ey2, w: ex - dx, h: 0, line: { color: SLATE, width: 2.5 }, objectName: "Connector across" });
  s.addShape(pres.shapes.LINE, { x: dx, y: ey2, w: 0, h: BY - ey2 - 0.04, line: { color: SLATE, width: 2.5, endArrowType: "triangle" }, objectName: "Connector into export" });
  s.addText("frozen model", { x: 6.1, y: ey2 - 0.36, w: 1.6, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: MUTED, align: "center", margin: 0, objectName: "Connector label" });

  // ---------- bottom row: deployment and verification
  const bottom = [
    [fa.FaFileCode, "Exact export", "raw-space thresholds\nby float32 bisection,\n10 B per node", false],
    [md.MdCallSplit, "Verification 1", "every node: R goes left,\nnext float32 goes right;\nNaN, ±∞, ±0 checked", true],
    [fa.FaLaptopCode, "Verification 2", "compiled C on the\nfull test split and\nadversarial inputs", true],
    [bs.BsCpuFill, "Verification 3", "ESP32 board:\nagreement, latency,\nflash, heap", true],
  ];
  for (let i = 0; i < bottom.length; i++) {
    const [Icon, title, sub, check] = bottom[i];
    await card({ x: bx[i], y: BY, w: BW, h: BH, n: i + 1, Icon, title, sub, fill: BLUE_FILL, line: BLUE, accent: BLUE, name: `Deploy ${i + 1}`, check });
    if (i < 3) arrow(bx[i] + BW + 0.04, bx[i + 1] - 0.04, BY + 0.63, BLUE, `Deploy arrow ${i + 1}`);
  }
  pill(0.26, 6.96, 5.4, "Deployment and verification  ·  FaithTree export", BLUE_DARK, "Deployment");

  await pres.writeFile({ fileName: process.argv[2] || "fig1_pipeline.pptx" });
  console.log("ok");
})();
