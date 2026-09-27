#!/usr/bin/env node
/* Render every Mermaid source under docs/site/source/diagrams/ to a light and a dark SVG under
   docs/site/img/diagrams/, and write manifest.json (source hash, Mermaid and Chromium versions).
   The SVGs are inputs to scripts/build_site.py, which verifies the manifest but never renders.

     npm pack mermaid@11 && tar xzf mermaid-11*.tgz            # once, anywhere
     node docs/site/source/render-diagrams.js --mermaid package/dist/mermaid.min.js [--only name]
     PLAYWRIGHT_MODULE=/opt/node22/lib/node_modules/playwright node ...   (when not on NODE_PATH)

   Rendering: a fresh browser context per diagram and theme; htmlLabels off (no foreignObject);
   render id from the source hash so ids are stable; width/height set from the viewBox. */
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
}
const MERMAID = path.resolve(arg("--mermaid", "mermaid.min.js"));
const ONLY = arg("--only", null);
const SRC = path.join(__dirname, "diagrams");
const OUT = path.resolve(__dirname, "..", "img", "diagrams");
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif';

const THEMES = {
  light: {
    theme: "base",
    themeVariables: {
      fontFamily: FONT, fontSize: "15px",
      primaryColor: "#e9eff6", primaryBorderColor: "#005493", primaryTextColor: "#101a2b",
      secondaryColor: "#fff3d1", secondaryBorderColor: "#c37530", tertiaryColor: "#ffffff",
      lineColor: "#3b4a60", textColor: "#101a2b", background: "#ffffff",
      clusterBkg: "#f3f6fa", clusterBorder: "#d6deea", edgeLabelBackground: "#ffffff",
    },
  },
  dark: {
    theme: "base",
    themeVariables: {
      fontFamily: FONT, fontSize: "15px",
      primaryColor: "#182539", primaryBorderColor: "#7fc0f5", primaryTextColor: "#e7edf6",
      secondaryColor: "#3a2f10", secondaryBorderColor: "#e6a35a", tertiaryColor: "#0f1727",
      lineColor: "#b9c5d8", textColor: "#e7edf6", background: "#121c2e",
      clusterBkg: "#0f1727", clusterBorder: "#26354d", edgeLabelBackground: "#121c2e",
    },
  },
};
// Explicit classDef colours in the sources are light-mode colours; remap them for the dark SVG.
const DARK_REMAP = [
  ["#00254b", "#1f4f86"], ["#fff3d1", "#3a2f10"], ["#e9eff6", "#182539"], ["#ffffff", "#0f1727"],
  ["#101a2b", "#e7edf6"], ["#005493", "#7fc0f5"], ["#c37530", "#e6a35a"],
  ["#e6f4ea", "#123322"], ["#1e8e3e", "#5fcf8a"], ["#fff4e5", "#3a2f10"], ["#e37400", "#e6a35a"],
  ["#f1f3f4", "#182539"], ["#5f6368", "#8d9bb1"], ["#e8eaf6", "#1c2540"], ["#3949ab", "#8ea2ff"],
  ["#111", "#e7edf6"],
];

function sources() {
  const out = [];
  (function walk(dir) {
    for (const name of fs.readdirSync(dir).sort()) {
      const p = path.join(dir, name);
      if (fs.statSync(p).isDirectory()) walk(p);
      else if (name.endsWith(".mmd")) out.push({ name: name.replace(/\.mmd$/, ""), path: p });
    }
  })(SRC);
  return out;
}

function postProcess(svg) {
  const vb = svg.match(/viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"/);
  if (!vb) throw new Error("no viewBox in rendered SVG");
  const w = Math.ceil(parseFloat(vb[1])), h = Math.ceil(parseFloat(vb[2]));
  svg = svg.replace(/<svg([^>]*)>/, (m, attrs) => {
    attrs = attrs.replace(/\swidth="[^"]*"/, "").replace(/\sheight="[^"]*"/, "").replace(/\sstyle="[^"]*"/, "");
    return `<svg${attrs} width="${w}" height="${h}">`;
  });
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n";
}

(async () => {
  if (!fs.existsSync(MERMAID)) { console.error("mermaid.min.js not found: " + MERMAID); process.exit(2); }
  let mermaidVersion = "unknown";
  const pkg = path.join(path.dirname(MERMAID), "..", "package.json");
  if (fs.existsSync(pkg)) mermaidVersion = JSON.parse(fs.readFileSync(pkg, "utf8")).version;
  fs.mkdirSync(OUT, { recursive: true });
  const manifestPath = path.join(OUT, "manifest.json");
  const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : {};
  const browser = await chromium.launch();
  const chromeVersion = browser.version();
  let n = 0;
  for (const src of sources()) {
    if (ONLY && src.name !== ONLY) continue;
    const text = fs.readFileSync(src.path, "utf8");
    const digest = crypto.createHash("sha256").update(text, "utf8").digest("hex");
    for (const [themeName, cfg] of Object.entries(THEMES)) {
      let source = text;
      if (themeName === "dark") for (const [a, b] of DARK_REMAP) source = source.split(a).join(b);
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
      await page.setContent("<!doctype html><html><body></body></html>");
      await page.addScriptTag({ path: MERMAID });
      const id = "mmd-" + digest.slice(0, 12) + "-" + themeName;
      const svg = await page.evaluate(async ({ cfg, id, source }) => {
        mermaid.initialize(Object.assign({ startOnLoad: false, securityLevel: "strict", htmlLabels: false,
          flowchart: { htmlLabels: false, useMaxWidth: false }, timeline: { useMaxWidth: false } }, cfg));
        await document.fonts.ready;
        const r = await mermaid.render(id, source);
        return r.svg;
      }, { cfg, id, source });
      await ctx.close();
      if (errors.length) throw new Error(`${src.name} (${themeName}): ${errors.join(" | ")}`);
      if (/<text[^>]*class="error-text"|Syntax error in/.test(svg)) throw new Error(`${src.name} (${themeName}): Mermaid reported a syntax error`);
      if (/<foreignObject/.test(svg)) throw new Error(`${src.name} (${themeName}): foreignObject in output`);
      fs.writeFileSync(path.join(OUT, `${src.name}-${themeName}.svg`), postProcess(svg));
    }
    manifest[src.name] = { sha256: digest, mermaid: mermaidVersion, chromium: chromeVersion };
    n += 1;
    console.log("rendered", src.name);
  }
  await browser.close();
  // drop manifest entries whose source is gone
  const names = new Set(sources().map((s) => s.name));
  for (const k of Object.keys(manifest)) if (!names.has(k)) delete manifest[k];
  const sorted = {};
  for (const k of Object.keys(manifest).sort()) sorted[k] = manifest[k];
  fs.writeFileSync(manifestPath, JSON.stringify(sorted, null, 2) + "\n");
  console.log(`rendered ${n} diagram(s) with mermaid ${mermaidVersion}, ${chromeVersion}`);
})().catch((e) => { console.error(e.message || e); process.exit(1); });
