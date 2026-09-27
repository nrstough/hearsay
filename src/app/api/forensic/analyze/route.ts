import { NextRequest, NextResponse } from "next/server";
import { createHash } from "crypto";
import { apiBase, toUiAnalysis, type AnalyzeResponse } from "@/lib/hearsay";

export const dynamic = "force-dynamic";

/**
 * POST /api/forensic/analyze: forward the uploaded file to the HEARSAY pipeline's API
 * (`uv run uvicorn hearsay.api:app --port 8000`, endpoint POST /analyze) and adapt its
 * AnalyzeResponse for the UI. The score and every evidence sentence come from the pipeline;
 * this route never computes anything about the audio.
 */
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    if (!file) {
      return NextResponse.json({ error: "No audio file provided in request" }, { status: 400 });
    }

    const bytes = Buffer.from(await file.arrayBuffer());
    const sha256 = createHash("sha256").update(bytes).digest("hex");

    const upstream = new FormData();
    upstream.append("file", new Blob([bytes], { type: file.type || "application/octet-stream" }), file.name);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 180_000); // first request loads the models (~16 s)
    let res: Response;
    try {
      res = await fetch(`${apiBase()}/analyze`, { method: "POST", body: upstream, signal: controller.signal });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return NextResponse.json(
        {
          error:
            `HEARSAY API not reachable at ${apiBase()} (${msg}). Start it from the repo root: ` +
            "uv run uvicorn hearsay.api:app --port 8000",
        },
        { status: 503 }
      );
    } finally {
      clearTimeout(timer);
    }

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json({ error: `Pipeline returned ${res.status}: ${text.slice(0, 300)}` }, { status: 502 });
    }

    const resp = (await res.json()) as AnalyzeResponse;
    const data = toUiAnalysis(resp, { sha256, sizeBytes: bytes.length });
    data.audio.filename = file.name;
    data.audio.title = `Analyte: ${file.name}`;
    return NextResponse.json({ success: true, data });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : "Failed to analyze audio clip";
    console.error("Audio Analysis Route Error:", error);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
