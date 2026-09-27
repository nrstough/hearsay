import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * A rule-based guide, not an analyst. It only repeats what the pipeline reported for the
 * current file (verdict, probability, evidence sentences, routing log) and states the fixed
 * facts of the shipped rule as documented in README.md. It never invents a file-specific
 * finding. No language model is involved.
 */

const RULE_FACTS =
  "The shipped rule ranks three detectors against their own training distributions and blends the ranks " +
  "0.6 (frozen XLS-R probe) + 0.2 (handcrafted spectral/prosody model) + 0.2 (trained XLS-R head). " +
  "Spectra-AASIST can only lower a score: if its margin is below -3 and the blended rank is above 0.5, the rank is halved. " +
  "Container facts route; compression, hum (ENF), splice and speaker-drift findings are shown as evidence and never change the score. " +
  "A file with no speech is pinned below every scored file.";

const METRIC_FACTS =
  "NSA scores normalized minDCF with a false alarm (real called synthetic) costing 4x a miss and about 70% of files real: " +
  "9.33 x P_FA + P_miss, minimized over thresholds, so only the ranking of scores matters. " +
  "Our submitted file scored 0.0733 (EER 3.53%) in NSA's one-time draft review; on our own holdout it scored 0.0065 and on In-the-Wild 0.228.";

type Modality = { name?: string; detail?: string; badge?: string; metric?: string; status?: string };

export async function POST(req: NextRequest) {
  try {
    const { message, currentAudio, modalities, routingLog } = await req.json();
    if (!message || typeof message !== "string") {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }
    const q = message.toLowerCase();
    const mods: Record<string, Modality> = modalities ?? {};
    const filename = currentAudio?.filename ?? "the current file";
    const p = typeof currentAudio?.overallScore === "number" ? currentAudio.overallScore.toFixed(3) : "n/a";
    const decision = currentAudio?.decision ?? "n/a";
    const log: string[] = Array.isArray(routingLog) ? routingLog : [];

    const quote = (key: string) => {
      const m = mods[key];
      return m ? `${m.name ?? key}: ${m.detail ?? "no detail"} (${m.badge ?? ""}; ${m.metric ?? ""})` : `${key}: not available for this file`;
    };

    let reply: string;
    if (q.includes("why") || q.includes("verdict") || q.includes("classified") || q.includes("fake") || q.includes("real")) {
      reply =
        `Verdict for '${filename}': ${decision}, probability of synthetic ${p}. ` +
        `${currentAudio?.detectedVector ? `Fusion: ${currentAudio.detectedVector}. ` : ""}` +
        `What the fused detectors said: ${quote("deepSpoof")} ${quote("spectral")} ` +
        (log.length ? `Routing log: ${log.join(" | ")}` : "");
    } else if (q.includes("spectral") || q.includes("prosody") || q.includes("pitch") || q.includes("handcrafted")) {
      reply = `${quote("spectral")} ${quote("prosody")}`;
    } else if (q.includes("enf") || q.includes("mains") || q.includes("hum") || q.includes("grid")) {
      reply = `${quote("enf")} Hum is recording-environment evidence and never changes the score: on our training data it marks the corpus, not the class.`;
    } else if (q.includes("speaker") || q.includes("drift") || q.includes("ecapa") || q.includes("timbre")) {
      reply = `${quote("speaker")} Drift is evidence only: on labeled data the fakes were the most self-consistent voices, so a fused drift score would push hard real recordings toward synthetic.`;
    } else if (q.includes("splice") || q.includes("seam") || q.includes("cut") || q.includes("edit")) {
      reply = `${quote("splice")} Seams are evidence only; on training data they appear in some real recordings and as vocoder artifacts in some generators.`;
    } else if (q.includes("compression") || q.includes("codec") || q.includes("transcod") || q.includes("mp3")) {
      reply = `${quote("compression")} Compression traces describe the file's pipeline, not the voice, and never enter the score.`;
    } else if (q.includes("container") || q.includes("metadata") || q.includes("header")) {
      reply = `${quote("container")} Container facts only route: on our training data the container was the label, so it is never learned.`;
    } else if (q.includes("deep") || q.includes("xls") || q.includes("spectra") || q.includes("aasist") || q.includes("probe")) {
      reply = `${quote("deepSpoof")} ${RULE_FACTS}`;
    } else if (q.includes("dcf") || q.includes("score") || q.includes("nsa") || q.includes("metric") || q.includes("eer")) {
      reply = METRIC_FACTS;
    } else if (q.includes("rule") || q.includes("fusion") || q.includes("weight") || q.includes("how")) {
      reply = RULE_FACTS;
    } else if (q.includes("routing") || q.includes("log")) {
      reply = log.length ? `Routing log for '${filename}': ${log.join(" | ")}` : "No routing log is loaded for this file.";
    } else {
      reply =
        `Guide (rule-based; it quotes the pipeline's own output and nothing else). '${filename}': ${decision}, P(synthetic) ${p}. ` +
        "Ask about the verdict, the deep detectors, spectral or prosody features, hum, speaker drift, splices, compression, the container, the routing log, the fusion rule, or the metric.";
    }

    return NextResponse.json({ success: true, content: reply, role: "assistant", createdAt: new Date().toISOString() });
  } catch (error: unknown) {
    console.error("Forensic Chat API Error:", error);
    return NextResponse.json({ error: "Failed to answer" }, { status: 500 });
  }
}
