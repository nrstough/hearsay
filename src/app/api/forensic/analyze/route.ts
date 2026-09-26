import { NextRequest, NextResponse } from "next/server";
import { writeFile, unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { execFile } from "child_process";
import { promisify } from "util";
import { randomBytes } from "crypto";

const execFileAsync = promisify(execFile);

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let tempFilePath: string | null = null;
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No audio file provided in request" }, { status: 400 });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Generate safe unique temporary path
    const randomId = randomBytes(8).toString("hex");
    const originalExt = file.name.includes(".") ? file.name.split(".").pop() : "wav";
    const tempFileName = `hearsay_${Date.now()}_${randomId}.${originalExt}`;
    tempFilePath = join(tmpdir(), tempFileName);

    // Write file to temp disk
    await writeFile(tempFilePath, buffer);

    // Call the Python backend analyzer via uv run
    const projectRoot = process.cwd();
    const { stdout, stderr } = await execFileAsync(
      "uv",
      ["run", "python", "-m", "hearsay.analyzer", tempFilePath],
      {
        cwd: projectRoot,
        timeout: 30000,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
        },
      }
    );

    if (stderr && !stdout) {
      console.error("Python analyzer stderr:", stderr);
      return NextResponse.json(
        { error: `Analyzer error: ${stderr.slice(0, 300)}` },
        { status: 500 }
      );
    }

    const result = JSON.parse(stdout);

    // Ensure the original user-facing filename is preserved
    if (result.audio) {
      result.audio.filename = file.name;
      result.audio.title = `Analyte: ${file.name}`;
    }

    return NextResponse.json({
      success: true,
      data: result,
    });
  } catch (error: any) {
    console.error("Audio Analysis Route Error:", error);
    return NextResponse.json(
      {
        error: error.message || "Failed to analyze audio clip",
      },
      { status: 500 }
    );
  } finally {
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
      } catch {
        // Ignore temp cleanup error
      }
    }
  }
}
