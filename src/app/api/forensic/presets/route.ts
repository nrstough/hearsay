import { NextResponse } from "next/server";
import { readFile, readdir, stat } from "fs/promises";
import { join } from "path";
import { toUiAnalysis, type AnalyzeResponse } from "@/lib/hearsay";

export const dynamic = "force-dynamic";

/**
 * GET /api/forensic/presets: the demo clips under demo/, each with the explanation JSON the
 * shipped pipeline wrote for it (demo/results/<file>.json). Audio is served from public/demo/
 * when a copy exists there. Nothing is fabricated: a clip without a result JSON is skipped.
 */
export async function GET() {
  const root = process.cwd();
  const resultsDir = join(root, "demo", "results");
  let files: string[] = [];
  try {
    files = (await readdir(resultsDir)).filter((f) => f.endsWith(".json")).sort();
  } catch {
    return NextResponse.json({ success: true, presets: [] });
  }
  const presets = [];
  for (const f of files) {
    try {
      const doc = JSON.parse(await readFile(join(resultsDir, f), "utf8")) as AnalyzeResponse;
      let audioUrl: string | undefined;
      let sizeBytes: number | undefined;
      try {
        const s = await stat(join(root, "public", "demo", doc.filename));
        audioUrl = `/demo/${doc.filename}`;
        sizeBytes = s.size;
      } catch {
        audioUrl = undefined;
      }
      const ui = toUiAnalysis(doc, { timestamp: "Demo clip, scored by the shipped pipeline", isPreset: true, audioUrl, sizeBytes, title: doc.filename });
      ui.audio.id = `preset-${doc.filename}`;
      presets.push(ui);
    } catch (e) {
      console.error("preset skipped:", f, e);
    }
  }
  return NextResponse.json({ success: true, presets });
}
