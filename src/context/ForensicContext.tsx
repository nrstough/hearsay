"use client";

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from "react";
import type { UiAnalysis, UiAudio, UiModality } from "@/lib/hearsay";

/**
 * App state. Every score, verdict, evidence sentence and routing line comes from the HEARSAY
 * pipeline through /api/forensic/* (docs/handoffs/2026-09-26_frontend-contract.md). There is
 * no local fallback that invents a verdict: if the pipeline cannot be reached, the UI says so.
 */

export type ForensicModality = UiModality;
export type AudioItem = UiAudio;

export interface DualAxisPoint {
  time: string;
  desktop: number;
  mobile: number;
  frame: number;
  phaseDiscontinuity: number;
}

export interface BatchInfo {
  total: number;
  processed: number;
  syntheticCount: number;
  bonafideCount: number;
  shareAboveHalf: number;
  minDcf: number;
  eer: number;
  metricSource: string;
  resultsAvailable: boolean;
}

export interface ForensicContextValue {
  currentAudio: AudioItem;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  togglePlay: () => void;
  seek: (seconds: number) => void;
  frequencyBars: number[];
  isAnalyzing: boolean;
  analysisProgress: number;
  activeDetectorStage: string;
  modalities: Record<string, ForensicModality>;
  chartData: DualAxisPoint[];
  routingLog: string[];
  fusion: UiAnalysis["fusion"] | null;
  version: Record<string, unknown> | null;
  lastError: string | null;
  recentQueue: AudioItem[];
  batchTotal: number;
  batchProcessed: number;
  isBatchRunning: boolean;
  batchInfo: BatchInfo | null;
  runBatchEvaluation: () => Promise<void>;
  exportPredictionTsv: () => void;
  loadAudioFile: (file: File) => Promise<void>;
  loadPreset: (id: string) => void;
  recordVoiceSample: (blob: Blob, durationSec: number) => Promise<void>;
  runFullForensicAudit: () => Promise<void>;
  presets: AudioItem[];
}

const EMPTY_AUDIO: AudioItem = {
  id: "none",
  title: "No file analyzed yet",
  filename: "No file analyzed yet",
  duration: 0,
  sampleRate: 16000,
  channels: 1,
  bitDepth: 16,
  sha256: "",
  size: "n/a",
  overallScore: 0,
  decision: "UNDETERMINED",
  fusedRank: 0,
  detectedVector: "Upload a clip or load a demo preset; every number shown comes from the pipeline.",
  timestamp: "",
};

const MODALITY_IDS: [string, string, string][] = [
  ["container", "Container and file metadata", "Metadata"],
  ["spectral", "Spectral features (handcrafted model)", "Frequency domain"],
  ["prosody", "Prosody and speech gate", "Phonetics"],
  ["enf", "Mains hum (ENF, 50/60 Hz)", "Recording environment"],
  ["compression", "Compression and transcoding traces", "Codec history"],
  ["speaker", "Speaker-embedding consistency", "Voice identity"],
  ["deepSpoof", "Deep anti-spoofing (XLS-R probe, trained head, Spectra-AASIST)", "Learned representations"],
  ["splice", "Splice and discontinuity detection", "Waveform"],
];

/** Shown before any file has been analyzed; every field says so. */
const EMPTY_MODALITIES: Record<string, ForensicModality> = Object.fromEntries(
  MODALITY_IDS.map(([id, name, category]) => [
    id,
    { id, name, category, score: 0.5, anomaly: false, status: "Not run", badge: "No file analyzed yet", detail: "Upload a clip or load a demo preset.", metric: "n/a" },
  ])
);

/** Honest progress labels: the pipeline runs every detector on every file, in this order. */
const STAGES = [
  "Decoding once through FFmpeg to 16 kHz mono; reading header facts...",
  "Running the engineered detectors (container, compression, hum, splice, speaker drift, speech gate)...",
  "Running the handcrafted spectral/prosody model...",
  "Running the frozen XLS-R probe and the trained XLS-R head...",
  "Running Spectra-AASIST (false-alarm suppression only)...",
  "Fusing ranks (0.6 / 0.2 / 0.2), applying the suppression step and the probability map...",
  "Writing the routing log and the per-detector evidence...",
];

const ForensicContext = createContext<ForensicContextValue | null>(null);

export function useForensic() {
  const ctx = useContext(ForensicContext);
  if (!ctx) {
    throw new Error("useForensic must be used within a ForensicProvider");
  }
  return ctx;
}

export function ForensicProvider({ children }: { children: React.ReactNode }) {
  const [currentAudio, setCurrentAudio] = useState<AudioItem>(EMPTY_AUDIO);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);

  // Decorative waveform for the player only; it is not an analysis output.
  const generateSmoothWaveform = (isPlayingActive: boolean, timeNow: number) => {
    return Array.from({ length: 96 }, (_, i) => {
      const x = i / 96;
      const envelope = Math.sin(x * Math.PI) * 0.75 + 0.25;
      const f1 = Math.sin(x * 14 + (isPlayingActive ? timeNow * 0.007 : 0)) * 0.22;
      const f2 = Math.sin(x * 32 + (isPlayingActive ? timeNow * 0.013 : 1.4)) * 0.14;
      const f3 = Math.cos(x * 60 + (isPlayingActive ? timeNow * 0.021 : 2.8)) * 0.09;
      const micro = ((Math.sin(i * 883 + (isPlayingActive ? timeNow * 0.003 : 0)) + 1) / 2) * 0.12;
      const combined = (0.24 + f1 + f2 + f3 + micro) * envelope;
      return Math.max(0.06, Math.min(0.95, combined));
    });
  };

  const [frequencyBars, setFrequencyBars] = useState<number[]>(() => generateSmoothWaveform(false, 0));
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [activeDetectorStage, setActiveDetectorStage] = useState<string>("Idle");
  const [modalities, setModalities] = useState<Record<string, ForensicModality>>(EMPTY_MODALITIES);
  const [chartData] = useState<DualAxisPoint[]>([]);
  const [routingLog, setRoutingLog] = useState<string[]>([]);
  const [fusion, setFusion] = useState<UiAnalysis["fusion"] | null>(null);
  const [version, setVersion] = useState<Record<string, unknown> | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [recentQueue, setRecentQueue] = useState<AudioItem[]>([]);
  const [presets, setPresets] = useState<AudioItem[]>([]);
  const presetAnalyses = useRef<Record<string, UiAnalysis>>({});
  const lastFile = useRef<File | null>(null);

  const [batchTotal, setBatchTotal] = useState<number>(0);
  const [batchProcessed, setBatchProcessed] = useState<number>(0);
  const [isBatchRunning, setIsBatchRunning] = useState<boolean>(false);
  const [batchInfo, setBatchInfo] = useState<BatchInfo | null>(null);

  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (isPlaying) {
      if (audioElementRef.current) {
        audioElementRef.current.play().catch(() => {});
      }
      timerRef.current = setInterval(() => {
        if (audioElementRef.current) {
          setCurrentTime(audioElementRef.current.currentTime);
          if (audioElementRef.current.ended) {
            setIsPlaying(false);
            setCurrentTime(0);
          }
        } else {
          setCurrentTime((prev) => {
            const next = prev + 0.1;
            if (next >= duration) {
              setIsPlaying(false);
              return 0;
            }
            return next;
          });
        }
        setFrequencyBars(() => generateSmoothWaveform(true, Date.now()));
      }, 80);
    } else {
      if (audioElementRef.current) audioElementRef.current.pause();
      if (timerRef.current) clearInterval(timerRef.current);
      setFrequencyBars(() => generateSmoothWaveform(false, 0));
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, duration]);

  const togglePlay = () => setIsPlaying((prev) => !prev);

  const seek = (seconds: number) => {
    const clamped = Math.min(duration, Math.max(0, seconds));
    setCurrentTime(clamped);
    if (audioElementRef.current) audioElementRef.current.currentTime = clamped;
  };

  const setPlayer = (url: string | null) => {
    if (audioElementRef.current) audioElementRef.current.pause();
    audioElementRef.current = url ? new Audio(url) : null;
  };

  const applyAnalysis = useCallback((a: UiAnalysis) => {
    setCurrentAudio(a.audio);
    setDuration(a.audio.duration || 0);
    setCurrentTime(0);
    setIsPlaying(false);
    setModalities(a.modalities);
    setRoutingLog(a.routingLog ?? []);
    setFusion(a.fusion ?? null);
    setVersion(a.version ?? null);
    setLastError(null);
  }, []);

  const runStages = () => {
    let idx = 0;
    setAnalysisProgress(5);
    setActiveDetectorStage(STAGES[0]);
    const t = setInterval(() => {
      idx = Math.min(idx + 1, STAGES.length - 1);
      setActiveDetectorStage(STAGES[idx]);
      setAnalysisProgress(Math.round(((idx + 1) / (STAGES.length + 1)) * 100));
    }, 700);
    return () => clearInterval(t);
  };

  const analyzeFile = async (file: File, label: string) => {
    setIsAnalyzing(true);
    setLastError(null);
    const stop = runStages();
    try {
      const url = URL.createObjectURL(file);
      setPlayer(url);
    } catch {
      setPlayer(null);
    }
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/forensic/analyze", { method: "POST", body: formData });
      const json = await res.json();
      if (!res.ok || !json.success || !json.data?.audio) {
        throw new Error(json.error || `Server returned ${res.status}`);
      }
      const data = json.data as UiAnalysis;
      data.audio.timestamp = label;
      applyAnalysis(data);
      setRecentQueue((prev) => [data.audio, ...prev]);
      setActiveDetectorStage("Done: verdict and evidence from the pipeline");
      setAnalysisProgress(100);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLastError(msg);
      setActiveDetectorStage(`Analysis failed: ${msg}`);
      setAnalysisProgress(0);
    } finally {
      stop();
      setIsAnalyzing(false);
    }
  };

  const loadAudioFile = async (file: File) => {
    lastFile.current = file;
    await analyzeFile(file, "Uploaded just now");
  };

  const recordVoiceSample = async (blob: Blob, durationSec: number) => {
    const voiceFilename = `mic_capture_${Date.now().toString().slice(-4)}.wav`;
    const voiceFile = new File([blob], voiceFilename, { type: blob.type || "audio/wav" });
    lastFile.current = voiceFile;
    await analyzeFile(voiceFile, `Recorded just now (${Math.max(0, durationSec).toFixed(1)} s)`);
  };

  const loadPreset = (id: string) => {
    const a = presetAnalyses.current[id];
    if (!a) return;
    lastFile.current = null;
    setPlayer(a.audio.audioUrl ?? null);
    applyAnalysis(a);
    setAnalysisProgress(100);
    setActiveDetectorStage("Demo clip: scored by the shipped pipeline");
  };

  const runFullForensicAudit = async () => {
    if (lastFile.current) {
      await analyzeFile(lastFile.current, "Re-analyzed just now");
    } else if (presetAnalyses.current[currentAudio.id]) {
      loadPreset(currentAudio.id);
    } else {
      setActiveDetectorStage("Nothing to re-scan: upload a clip or load a demo preset");
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/forensic/presets");
        const json = await res.json();
        if (cancelled || !json.success) return;
        const list = (json.presets as UiAnalysis[]) ?? [];
        presetAnalyses.current = Object.fromEntries(list.map((a) => [a.audio.id, a]));
        setPresets(list.map((a) => a.audio));
        setRecentQueue(list.map((a) => a.audio));
        if (list.length > 0) {
          setPlayer(list[0].audio.audioUrl ?? null);
          applyAnalysis(list[0]);
          setAnalysisProgress(100);
          setActiveDetectorStage("Demo clip: scored by the shipped pipeline");
        }
      } catch (err) {
        if (!cancelled) setLastError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [applyAnalysis]);

  const runBatchEvaluation = async () => {
    setIsBatchRunning(true);
    try {
      const res = await fetch("/api/forensic/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "load" }) });
      const json = await res.json();
      if (!res.ok || !json.success) throw new Error(json.error || `Server returned ${res.status}`);
      const d = json.data as BatchInfo & { samples: AudioItem[] };
      setBatchInfo(d);
      setBatchTotal(d.total);
      setBatchProcessed(d.processed);
      setRecentQueue(d.samples ?? []);
      setLastError(null);
    } catch (err) {
      setLastError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsBatchRunning(false);
    }
  };

  const exportPredictionTsv = () => {
    const link = document.createElement("a");
    link.href = "/api/forensic/export-tsv";
    link.download = "CrossExam_predictions.tsv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <ForensicContext.Provider
      value={{
        currentAudio,
        isPlaying,
        currentTime,
        duration,
        togglePlay,
        seek,
        frequencyBars,
        isAnalyzing,
        analysisProgress,
        activeDetectorStage,
        modalities,
        chartData,
        routingLog,
        fusion,
        version,
        lastError,
        recentQueue,
        batchTotal,
        batchProcessed,
        isBatchRunning,
        batchInfo,
        runBatchEvaluation,
        exportPredictionTsv,
        loadAudioFile,
        loadPreset,
        recordVoiceSample,
        runFullForensicAudit,
        presets,
      }}
    >
      {children}
    </ForensicContext.Provider>
  );
}
