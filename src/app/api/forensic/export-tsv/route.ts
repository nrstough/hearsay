import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join } from "path";

export const dynamic = "force-dynamic";

/** The submitted prediction file, byte for byte. Never generated here. */
const SUBMITTED_TSV = process.env.HEARSAY_SUBMITTED_TSV ?? join("submissions", "CrossExam_predictions.tsv");

export async function GET() {
  try {
    const tsv = await readFile(join(process.cwd(), SUBMITTED_TSV));
    return new NextResponse(tsv, {
      status: 200,
      headers: {
        "Content-Type": "text/tab-separated-values; charset=utf-8",
        "Content-Disposition": 'attachment; filename="CrossExam_predictions.tsv"',
      },
    });
  } catch {
    return NextResponse.json(
      { error: `Submitted TSV not found at ${SUBMITTED_TSV}; it is gitignored and lives on the scoring machine.` },
      { status: 404 }
    );
  }
}
