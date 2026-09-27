import { NextResponse } from "next/server";
import { readFile, readdir } from "fs/promises";
import { join } from "path";
import { toUiAnalysis, type AnalyzeResponse } from "@/lib/hearsay";

export const dynamic = "force-dynamic";

const SUBMITTED_TSV = process.env.HEARSAY_SUBMITTED_TSV ?? join("submissions", "CrossExam_predictions.tsv");
/** A runner output directory holding results/<filename>.json for the test set (gitignored). */
const RESULTS_DIR = process.env.HEARSAY_RESULTS ?? join("outputs", "runner", "v2_full");

/**
 * NSA's one-time draft review of the submitted file (Sat Sep 26, 2026, ~15:30):
 * docs/reports/2026-09-26_sponsor-questions.md. The final official score comes after judging.
 */
const DRAFT_REVIEW = { minDcf: 0.0733, eer: 0.03534, source: "NSA draft review of the submitted file, Sat Sep 26, 2026" };

async function readSubmission() {
  const text = await readFile(join(process.cwd(), SUBMITTED_TSV), "utf8");
  const rows = text
    .split("\n")
    .slice(1)
    .filter((l) => l.trim().length > 0)
    .map((l) => {
      const [filename, score] = l.split("\t");
      return { filename, score: Number(score) };
    });
  return rows;
}

async function sampleItems(rows: { filename: string; score: number }[], n: number) {
  const items = [];
  for (const r of rows.slice(0, n)) {
    let doc: AnalyzeResponse | null = null;
    try {
      doc = JSON.parse(await readFile(join(process.cwd(), RESULTS_DIR, "results", `${r.filename}.json`), "utf8"));
    } catch {
      doc = null;
    }
    if (doc) {
      const ui = toUiAnalysis(doc, { timestamp: "From the submitted run", isPreset: true });
      items.push({ ...ui.audio, id: `batch-${r.filename}`, overallScore: r.score, decision: r.score >= 0.5 ? "SYNTHETIC" : "BONA_FIDE" });
    } else {
      items.push({
        id: `batch-${r.filename}`,
        title: r.filename,
        filename: r.filename,
        duration: 0,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
        sha256: "",
        size: "n/a",
        overallScore: r.score,
        decision: r.score >= 0.5 ? "SYNTHETIC" : "BONA_FIDE",
        fusedRank: 0,
        detectedVector: "score from the submitted TSV; per-file explanation not on this machine",
        timestamp: "From the submitted TSV",
        isPreset: true,
      });
    }
  }
  return items;
}

async function payload() {
  const rows = await readSubmission();
  const syntheticCount = rows.filter((r) => r.score >= 0.5).length;
  let resultsAvailable = false;
  try {
    resultsAvailable = (await readdir(join(process.cwd(), RESULTS_DIR, "results"))).length > 0;
  } catch {
    resultsAvailable = false;
  }
  return {
    total: rows.length,
    processed: rows.length,
    syntheticCount,
    bonafideCount: rows.length - syntheticCount,
    shareAboveHalf: rows.length ? syntheticCount / rows.length : 0,
    minDcf: DRAFT_REVIEW.minDcf,
    eer: DRAFT_REVIEW.eer,
    metricSource: DRAFT_REVIEW.source,
    resultsAvailable,
    samples: await sampleItems(rows, 25),
  };
}

export async function POST() {
  try {
    return NextResponse.json({ success: true, data: await payload() });
  } catch (error: unknown) {
    console.error("Batch Route Error:", error);
    return NextResponse.json({ error: "Submitted TSV not found on this machine" }, { status: 404 });
  }
}

export async function GET() {
  try {
    const p = await payload();
    return NextResponse.json({ total: p.total, processed: p.processed, status: "submitted", minDcf: p.minDcf, eer: p.eer, metricSource: p.metricSource });
  } catch {
    return NextResponse.json({ error: "Submitted TSV not found on this machine" }, { status: 404 });
  }
}
