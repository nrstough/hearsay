/**
 * Adapter between the HEARSAY pipeline's AnalyzeResponse (contract v0, see
 * docs/handoffs/2026-09-26_frontend-contract.md) and the shapes the UI components read.
 *
 * Every number the UI shows comes from the pipeline's JSON. Nothing here computes a score.
 */

export interface DetectorOut {
  name: string;
  role: "fused" | "evidence" | "routing" | "gate" | string;
  status: "ok" | "skipped" | "error" | string;
  score: number;
  evidence: string;
  features: Record<string, number>;
  seconds: number;
  error: string | null;
}

export interface AnalyzeResponse {
  filename: string;
  duration_s: number;
  probability_synthetic: number;
  verdict: "real" | "synthetic" | "undetermined" | string;
  is_speech: boolean;
  default_answer_applied: boolean;
  fusion: {
    rule: string;
    inputs: Record<string, number | null>;
    weights: Record<string, number>;
    terms: Record<string, number>;
    fused: number;
    p_fused: number;
    imputed?: string[];
    detail?: { final?: string; base?: number; e_applied?: boolean; e_rule?: Record<string, unknown> };
  };
  detectors: DetectorOut[];
  routing_log: string[];
  flag?: string;
  seconds?: number;
  version: Record<string, unknown>;
}

export interface UiModality {
  id: string;
  name: string;
  category: string;
  score: number;
  anomaly: boolean;
  status: string;
  badge: string;
  detail: string;
  metric: string;
}

export interface UiAudio {
  id: string;
  title: string;
  filename: string;
  duration: number;
  sampleRate: number;
  channels: number;
  bitDepth: number;
  sha256: string;
  size: string;
  overallScore: number;
  decision: "SYNTHETIC" | "BONA_FIDE" | "UNDETERMINED";
  fusedRank: number;
  detectedVector: string;
  timestamp: string;
  isPreset?: boolean;
  audioUrl?: string;
}

export interface UiAnalysis {
  audio: UiAudio;
  modalities: Record<string, UiModality>;
  routingLog: string[];
  fusion: AnalyzeResponse["fusion"];
  version: Record<string, unknown>;
  chartData: never[];
}

const ROLE_BADGE: Record<string, string> = {
  fused: "Enters the score",
  evidence: "Evidence only, never fused",
  routing: "Routing only, never fused",
  gate: "Speech gate",
};

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function byName(resp: AnalyzeResponse): Record<string, DetectorOut> {
  const out: Record<string, DetectorOut> = {};
  for (const d of resp.detectors) out[d.name] = d;
  return out;
}

function pct(x: number | undefined): string {
  return x === undefined || Number.isNaN(x) ? "n/a" : `${Math.round(x * 100)}%`;
}

function fmt(x: number | null | undefined, digits = 2): string {
  return x === null || x === undefined || Number.isNaN(x) ? "n/a" : x.toFixed(digits);
}

function modality(
  id: string,
  name: string,
  category: string,
  det: DetectorOut | undefined,
  opts: { weight?: number; metric?: string; badge?: string; detail?: string; score?: number } = {}
): UiModality {
  if (!det) {
    return { id, name, category, score: 0.5, anomaly: false, status: "Not run", badge: "Not run", detail: "This detector did not run for this file.", metric: "n/a" };
  }
  const score = opts.score ?? det.score;
  const errored = det.status !== "ok";
  const anomaly = !errored && det.role === "fused" ? score >= 0.5 : !errored && score > 0.5;
  const badge = opts.badge ?? (det.role === "fused" && opts.weight !== undefined
    ? `${ROLE_BADGE.fused} (weight ${opts.weight})`
    : ROLE_BADGE[det.role] ?? det.role);
  return {
    id,
    name,
    category,
    score: Number(score.toFixed(4)),
    anomaly,
    status: errored ? `Error: ${det.error ?? det.status}` : anomaly ? "Flagged" : "Clean",
    badge,
    detail: opts.detail ?? det.evidence,
    metric: opts.metric ?? `score ${fmt(score, 3)}`,
  };
}

/** Map one pipeline response to the UI's shapes. `extra` carries what only the upload knows. */
export function toUiAnalysis(
  resp: AnalyzeResponse,
  extra: { sha256?: string; sizeBytes?: number; timestamp?: string; isPreset?: boolean; audioUrl?: string; title?: string } = {}
): UiAnalysis {
  const d = byName(resp);
  const w = resp.fusion.weights ?? {};
  const t = resp.fusion.terms ?? {};
  const inputs = resp.fusion.inputs ?? {};
  const detail = resp.fusion.detail ?? {};
  const container = d.container?.features ?? {};
  const gate = d.speech_gate?.features ?? {};

  const deepParts: string[] = [];
  if (d.m1b_v3) deepParts.push(`XLS-R probe (weight ${w.m1b_v3 ?? "?"}): ${d.m1b_v3.evidence}`);
  if (d.m5_xlsr_ft) deepParts.push(`Trained XLS-R head (weight ${w.m5_xlsr_ft ?? "?"}): ${d.m5_xlsr_ft.evidence}`);
  if (d.spectra_aasist) deepParts.push(`Spectra-AASIST (suppression only): ${d.spectra_aasist.evidence}`);

  const modalities: Record<string, UiModality> = {
    container: modality("container", "Container and file metadata", "Metadata", d.container, {
      metric: `${container.sample_rate ?? "?"} Hz, ${container.channels ?? "?"} ch, ${container.bits ?? "?"}-bit`,
    }),
    spectral: modality("spectral", "Spectral features (handcrafted model)", "Frequency domain", d.handcrafted, {
      weight: w.handcrafted_v5,
      metric: `P(synthetic) ${fmt(d.handcrafted?.score, 2)}, rank ${fmt(t.handcrafted_v5, 2)}`,
    }),
    prosody: modality("prosody", "Prosody and speech gate", "Phonetics", d.speech_gate, {
      badge: gate.is_speech === 0 ? "Gate: non-speech, pinned below every scored file" : "Gate: speech present",
      metric: `voiced ${pct(gate.voiced_frac)}, pitch spread ${fmt(gate.f0_std_log, 2)}, loudness std ${fmt(gate.energy_db_std, 1)} dB`,
      score: 0.5,
    }),
    enf: modality("enf", "Mains hum (ENF, 50/60 Hz)", "Recording environment", d.enf),
    compression: modality("compression", "Compression and transcoding traces", "Codec history", d.compression, {
      metric: `effective bandwidth ${fmt(d.compression?.features?.bw_hz, 0)} Hz`,
    }),
    speaker: modality("speaker", "Speaker-embedding consistency", "Voice identity", d.speaker_drift, {
      metric: `min window similarity ${fmt(d.speaker_drift?.features?.cos_min, 2)} over ${d.speaker_drift?.features?.n_windows ?? "?"} windows`,
    }),
    deepSpoof: modality("deepSpoof", "Deep anti-spoofing (XLS-R probe, trained head, Spectra-AASIST)", "Learned representations", d.m1b_v3, {
      badge: `Enters the score (weights ${w.m1b_v3 ?? "?"} + ${w.m5_xlsr_ft ?? "?"}; Spectra suppression only)`,
      detail: deepParts.join(" ") || "No deep detector output.",
      metric: `ranks: probe ${fmt(t.m1b_v3, 2)}, head ${fmt(t.m5_xlsr_ft, 2)}; Spectra margin ${fmt(inputs.spectra_aasist, 1)}`,
    }),
    splice: modality("splice", "Splice and discontinuity detection", "Waveform", d.splice),
  };

  const decision: UiAudio["decision"] =
    resp.verdict === "synthetic" ? "SYNTHETIC" : resp.verdict === "undetermined" ? "UNDETERMINED" : "BONA_FIDE";

  const summary: string[] = [];
  if (resp.default_answer_applied) summary.push("Non-speech or decode failure: pinned below every scored file");
  else {
    summary.push(`${detail.final ?? resp.fusion.rule}: blended rank ${fmt(detail.base ?? resp.fusion.fused, 2)}`);
    summary.push(detail.e_applied ? "Spectra-AASIST pulled this file toward real" : "no suppression applied");
  }

  const sha = extra.sha256 ?? "";
  const audio: UiAudio = {
    id: sha ? `upload-${sha.slice(0, 12)}` : `result-${resp.filename}`,
    title: extra.title ?? `Analyte: ${resp.filename}`,
    filename: resp.filename,
    duration: Number((resp.duration_s ?? 0).toFixed(2)),
    sampleRate: Number(container.sample_rate ?? 16000),
    channels: Number(container.channels ?? 1),
    bitDepth: Number(container.bits ?? 16),
    sha256: sha,
    size: extra.sizeBytes !== undefined ? formatFileSize(extra.sizeBytes) : "n/a",
    overallScore: Number(resp.probability_synthetic.toFixed(4)),
    decision,
    fusedRank: Number((resp.fusion.fused ?? 0).toFixed(4)),
    detectedVector: summary.join("; "),
    timestamp: extra.timestamp ?? "Analyzed just now",
    isPreset: extra.isPreset,
    audioUrl: extra.audioUrl,
  };

  return { audio, modalities, routingLog: resp.routing_log ?? [], fusion: resp.fusion, version: resp.version, chartData: [] };
}

/** The HEARSAY API base URL (uv run uvicorn hearsay.api:app --port 8000). */
export function apiBase(): string {
  return (process.env.HEARSAY_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}
