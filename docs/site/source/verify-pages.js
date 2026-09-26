#!/usr/bin/env node
/* Open every page of docs/site/ from file:// at two widths in light and dark, assert no
   horizontal overflow, and screenshot each. Not part of the pytest suite.

     node docs/site/source/verify-pages.js --out /tmp/site-shots [--only index.html,metric.html]
     PLAYWRIGHT_MODULE=/opt/node22/lib/node_modules/playwright node ... (when playwright is not on NODE_PATH)

   Exit code 1 if any page overflows or throws a page error. */
"use strict";
const fs = require("fs");
const path = require("path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
}
const SITE = path.resolve(__dirname, "..");
const OUT = path.resolve(arg("--out", path.join(require("os").tmpdir(), "hearsay-site-shots")));
const ONLY = arg("--only", "") ? arg("--only", "").split(",") : null;
const WIDTHS = [390, 1280];
const THEMES = ["light", "dark"];

function pages(dir, prefix) {
  let out = [];
  for (const name of fs.readdirSync(dir).sort()) {
    const p = path.join(dir, name);
    const rel = prefix ? prefix + "/" + name : name;
    if (fs.statSync(p).isDirectory()) {
      if (name === "source" || name === "img") continue;
      out = out.concat(pages(p, rel));
    } else if (name.endsWith(".html")) {
      out.push(rel);
    }
  }
  return out;
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const list = ONLY || pages(SITE, "");
  let failures = 0, shots = 0;
  for (const theme of THEMES) {
    for (const width of WIDTHS) {
      const ctx = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme });
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
      for (const rel of list) {
        errors.length = 0;
        await page.goto("file://" + path.join(SITE, rel), { waitUntil: "load" });
        const m = await page.evaluate(() => ({
          sw: document.documentElement.scrollWidth, iw: window.innerWidth,
          title: document.title, h1: document.querySelectorAll("h1").length,
        }));
        const name = rel.replace(/[\/]/g, "__").replace(/\.html$/, "") + `--${width}-${theme}.png`;
        await page.screenshot({ path: path.join(OUT, name), fullPage: width === 390 ? false : false });
        shots += 1;
        const bad = [];
        if (m.sw > m.iw) bad.push(`overflow ${m.sw} > ${m.iw}`);
        if (m.h1 !== 1) bad.push(`h1 count ${m.h1}`);
        if (errors.length) bad.push("errors: " + errors.join(" | ").slice(0, 200));
        if (bad.length) { failures += 1; console.log(`FAIL ${rel} @${width} ${theme}: ${bad.join("; ")}`); }
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(`pages ${list.length}, screenshots ${shots}, failures ${failures}, out ${OUT}`);
  process.exit(failures ? 1 : 0);
})();
