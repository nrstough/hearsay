"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  ChevronRight,
  ChevronLeft,
  X,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  Volume2,
  Layers,
  Activity,
  FileSpreadsheet,
  FileText,
  Bot,
  Zap,
} from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export interface TourStep {
  id: string;
  tab: "dashboard" | "spectrogram" | "modalities" | "batch" | "dossier";
  targetId: string;
  badge: string;
  title: string;
  description: string;
  rubricNote: string;
  metrics?: { label: string; value: string }[];
  actionLabel?: string;
  actionIcon?: React.ElementType;
}

interface GuidedTourProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isOpen: boolean;
  onClose: () => void;
  onOpen: () => void;
  onOpenCopilot?: () => void;
}

export function GuidedTour({
  activeTab,
  setActiveTab,
  isOpen,
  onClose,
  onOpen,
  onOpenCopilot,
}: GuidedTourProps) {
  const {
    presets,
    loadPreset,
    togglePlay,
    isPlaying,
    runBatchEvaluation,
    exportPredictionTsv,
    currentAudio,
  } = useForensic();

  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [autoProgress, setAutoProgress] = useState(0);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const [hasPromptedWelcome, setHasPromptedWelcome] = useState(false);
  const [showWelcomeModal, setShowWelcomeModal] = useState(false);

  const autoPlayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const TOUR_STEPS: TourStep[] = [
    {
      id: "step-overview",
      tab: "dashboard",
      targetId: "tour-hero",
      badge: "Architecture & Benchmark",
      title: "Autonomous 10-Detector Forensic Cockpit",
      description:
        "Engineered for the NSA Audio Authentication Challenge at HackGT 13. HEARSAY operates a 10-detector forensic pipeline running offline inference with 4-way rank fusion to detect synthetic voices, generative models, and acoustic splices.",
      rubricNote: "Achieved NSA draft-review minDCF of 0.0733 and 3.53% Equal Error Rate on the submitted 1,671-file evaluation.",
      metrics: [
        { label: "minDCF", value: "0.0733" },
        { label: "EER", value: "3.53%" },
        { label: "Throughput", value: "1.34s / clip" },
      ],
    },
    {
      id: "step-waveform",
      tab: "dashboard",
      targetId: "tour-audio-analyzer",
      badge: "Real-Time Telemetry",
      title: "Live Waveform & Audio Ingestion",
      description:
        "Inspect the 96-bar dynamic energy soundwave. Upload external audio files, capture microphone recordings in real-time, or test pre-scored pipeline presets (Apple TTS Samantha at 0.998 vs In-the-Wild Real Clip at 0.001) to view immediate score attribution.",
      rubricNote: "Decodes to 16 kHz mono float32 band-matched to 7.25 kHz with dynamic timecode scrubbing and instant probability breakdown.",
      metrics: [
        { label: "Waveform Resolution", value: "96 Bars" },
        { label: "Sample Rate", value: "16.0 kHz" },
        { label: "Band-Match", value: "7.25 kHz" },
      ],
      actionLabel: isPlaying ? "Pause Audio" : "Play & Test Apple TTS (0.998)",
      actionIcon: isPlaying ? Pause : Play,
    },
    {
      id: "step-chart",
      tab: "dashboard",
      targetId: "tour-forensic-chart",
      badge: "Telemetry & Drift",
      title: "Dual-Axis Spectral & Drift Tracking",
      description:
        "Monitors high-frequency spectral roll-off against ECAPA-TDNN speaker embedding drift across temporal windows to detect instantaneous deepfake voice conversion.",
      rubricNote: "Tracks the 7.25 kHz band-matched roll-off alongside frame-by-frame speaker consistency.",
      metrics: [
        { label: "Band-Match Cutoff", value: "7.25 kHz" },
        { label: "Cosine Drift", value: "0.44 Jump" },
        { label: "Window", value: "Sliding Frames" },
      ],
    },
    {
      id: "step-modalities",
      tab: "dashboard",
      targetId: "tour-modality-grid",
      badge: "Multi-Vector Defense",
      title: "Ten-Detector Forensic Matrix (4 Fused)",
      description:
        "HEARSAY deploys ten detectors with four fused in a rank blend: XLS-R Layer-7 Probe (M1b), Fine-Tuned XLS-R Head (M5), Handcrafted Spectral/Prosody Model, and Spectra-AASIST (suppression only), backed by six routing, gate, and evidence detectors.",
      rubricNote: "Fulfills forensic breadth across foundation models, handcrafted acoustic features, container parsing, and physical forensic traces.",
      metrics: [
        { label: "Detectors", value: "10 (4 Fused)" },
        { label: "Primary Model", value: "XLS-R M1b (60%)" },
        { label: "Mains Hum", value: "50/60 Hz Trace" },
      ],
    },
    {
      id: "step-spectrogram",
      tab: "spectrogram",
      targetId: "tour-spectrogram-view",
      badge: "Fourier Analysis",
      title: "Interactive STFT Spectrogram & ENF Trace",
      description:
        "Inspect acoustic signals in high-resolution time and frequency. Switch between 512, 1024, and 2048 FFT windows, toggle specialized colormaps (CyberNavy, Sapphire, Viridis), and inspect 7.25 kHz band-matched roll-offs and vocoder cutoff boundaries.",
      rubricNote: "Pinpoints unnatural frequency drop-offs and harmonic overtone flatlines in real-time.",
      metrics: [
        { label: "FFT Window", value: "1024 Samples" },
        { label: "Hop Size", value: "256 Samples" },
        { label: "ENF Trace", value: "50/60 Hz Monitor" },
      ],
    },
    {
      id: "step-modalities-view",
      tab: "modalities",
      targetId: "tour-modalities-view",
      badge: "Ensemble Calibration",
      title: "Modality Calibration & ROC Analytics",
      description:
        "Deep dive into the ten individual forensic detectors. Review calibrated rank weights, false-alarm suppression, and empirical evidence distributions for each analytical detector.",
      rubricNote: "Includes container metadata headers, compression quantization steps, and acoustic prosody.",
      metrics: [
        { label: "Calibration", value: "Rank Fusion" },
        { label: "Ensemble Rule", value: "A3 w0.2 + E" },
        { label: "Robustness", value: "7.25 kHz Match" },
      ],
    },
    {
      id: "step-batch",
      tab: "batch",
      targetId: "tour-batch-view",
      badge: "Enterprise Pipeline",
      title: "1,671 Audio NSA Batch Evaluation",
      description:
        "High-throughput batch pipeline capable of evaluating entire wiretap archives offline without internet connectivity. Serves the submitted 'CrossExam_predictions.tsv' byte-for-byte with the exact hash scored by the NSA.",
      rubricNote: "Includes status filters, worker thread telemetry, and calibrated submission export.",
      metrics: [
        { label: "Batch Size", value: "1,671 Clips" },
        { label: "Submission", value: "CrossExam_predictions.tsv" },
        { label: "Mode", value: "100% Offline" },
      ],
      actionLabel: "Export Calibrated TSV",
      actionIcon: FileSpreadsheet,
    },
    {
      id: "step-dossier",
      tab: "dossier",
      targetId: "tour-dossier-view",
      badge: "Chain of Custody",
      title: "Cryptographic Forensic Evidence Dossier",
      description:
        "Generates intelligence-community and court-ready dossiers featuring cryptographic SHA-256 hashes, full modality breakdowns, confidence intervals, and official forensic integrity badges.",
      rubricNote: "Includes one-click printable briefing generation and clipboard export for briefings.",
      metrics: [
        { label: "Integrity", value: "SHA-256 Validated" },
        { label: "Court Ready", value: "Standardized Briefing" },
        { label: "Export", value: "Print & TSV" },
      ],
    },
    {
      id: "step-copilot",
      tab: "dashboard",
      targetId: "tour-copilot",
      badge: "Agentic Intelligence",
      title: "Autonomous Acoustic Forensic Copilot",
      description:
        "An interactive forensic guide that quotes calibrated pipeline fields. Cross-examine scored clips, inspect per-detector evidence sentences, check routing decisions, and review rank contributions directly from the pipeline JSON.",
      rubricNote: "Strictly bounded to quoting pipeline fields, detector evidence, and calibrated routing logs.",
      metrics: [
        { label: "Agent Scope", value: "Pipeline Evidence" },
        { label: "Grounding", value: "Pipeline Fields" },
        { label: "Context", value: "Live Scored JSON" },
      ],
      actionLabel: "Open Acoustic Copilot",
      actionIcon: Bot,
    },
  ];

  // Welcome prompt check on first visitor
  useEffect(() => {
    if (typeof window === "undefined") return;
    const tourDone = localStorage.getItem("hearsay_demo_tour_completed");
    const tourPrompted = sessionStorage.getItem("hearsay_demo_tour_prompted");
    const urlParams = new URLSearchParams(window.location.search);
    const forceTour = urlParams.get("tour") === "true";

    if (forceTour) {
      onOpen();
      setCurrentStep(0);
      return;
    }

    if (!tourDone && !tourPrompted) {
      sessionStorage.setItem("hearsay_demo_tour_prompted", "true");
      setShowWelcomeModal(true);
    }
  }, [onOpen]);

  // Synchronize targetRect when step or tab changes
  useEffect(() => {
    if (!isOpen) {
      setTargetRect(null);
      return;
    }

    const step = TOUR_STEPS[currentStep];
    if (activeTab !== step.tab) {
      setActiveTab(step.tab);
    }

    const updateRect = () => {
      const el = document.getElementById(step.targetId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        const rect = el.getBoundingClientRect();
        setTargetRect(rect);
      } else {
        setTargetRect(null);
      }
    };

    const timer = setTimeout(updateRect, 350);
    window.addEventListener("resize", updateRect);
    window.addEventListener("scroll", updateRect);

    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", updateRect);
      window.removeEventListener("scroll", updateRect);
    };
  }, [isOpen, currentStep, activeTab, setActiveTab]);

  // Auto-Play timer handler
  useEffect(() => {
    if (!isOpen || !isAutoPlaying) {
      if (autoPlayTimerRef.current) clearInterval(autoPlayTimerRef.current);
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
      setAutoProgress(0);
      return;
    }

    const STEP_DURATION_MS = 8000;
    const TICK_MS = 100;
    let elapsed = 0;
    setAutoProgress(0);

    progressIntervalRef.current = setInterval(() => {
      elapsed += TICK_MS;
      setAutoProgress(Math.min(100, (elapsed / STEP_DURATION_MS) * 100));
    }, TICK_MS);

    autoPlayTimerRef.current = setTimeout(() => {
      if (currentStep < TOUR_STEPS.length - 1) {
        setCurrentStep((prev) => prev + 1);
      } else {
        setIsAutoPlaying(false);
      }
    }, STEP_DURATION_MS);

    return () => {
      if (autoPlayTimerRef.current) clearTimeout(autoPlayTimerRef.current);
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    };
  }, [isOpen, isAutoPlaying, currentStep, TOUR_STEPS.length]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        handleNext();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handlePrev();
      } else if (e.key === "Escape") {
        e.preventDefault();
        handleCloseTour();
      } else if (e.key === " ") {
        e.preventDefault();
        setIsAutoPlaying((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentStep]);

  const handleNext = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      handleCloseTour();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleCloseTour = () => {
    setIsAutoPlaying(false);
    onClose();
    if (typeof window !== "undefined") {
      localStorage.setItem("hearsay_demo_tour_completed", "true");
    }
  };

  const handleStartTour = () => {
    setShowWelcomeModal(false);
    setCurrentStep(0);
    setIsAutoPlaying(false);
    onOpen();
  };

  const handleStepAction = () => {
    const step = TOUR_STEPS[currentStep];
    if (step.id === "step-waveform") {
      const applePreset = presets.find((p) => p.filename.includes("apple") || p.id.includes("apple")) || presets[0];
      if (applePreset) {
        loadPreset(applePreset.id);
      }
      togglePlay();
    } else if (step.id === "step-batch") {
      exportPredictionTsv();
    } else if (step.id === "step-copilot") {
      if (onOpenCopilot) onOpenCopilot();
      else {
        const copilotBtn = document.querySelector("#tour-copilot button") as HTMLButtonElement | null;
        if (copilotBtn) copilotBtn.click();
      }
    }
  };

  const activeStepData = TOUR_STEPS[currentStep];

  return (
    <>
      {/* 1. Welcome Modal on First Visit */}
      {showWelcomeModal && !isOpen && (
        <div className="fixed inset-0 z-[9980] flex items-center justify-center p-4 bg-[#001730]/75 backdrop-blur-md animate-in fade-in duration-300">
          <div className="bg-white text-slate-900 border border-slate-200/90 rounded-[32px] p-7 sm:p-8 max-w-lg w-full shadow-2xl relative overflow-hidden animate-in zoom-in-95 duration-200">
            {/* Top Glow Layer */}
            <div className="absolute top-0 inset-x-0 h-2 bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530]" />

            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#005493] to-[#c37530] flex items-center justify-center text-white shadow-md">
                <Sparkles className="w-6 h-6 text-amber-200 animate-pulse" />
              </div>
              <div>
                <span className="text-[11px] font-bold text-[#c37530] uppercase tracking-wider font-mono">
                  Slide-Deck Reviewer Mode
                </span>
                <h3 className="text-xl font-bold text-slate-900 tracking-tight font-sans">
                  HEARSAY Product Tour
                </h3>
              </div>
            </div>

            <p className="text-sm text-slate-600 leading-relaxed font-sans mb-5">
              Welcome to the HEARSAY Forensic Intelligence Cockpit. Since we're presenting via slide deck, take this <strong>9-step interactive guided tour</strong> to review our ten forensic detectors, dynamic waveform analysis, batch pipeline, and forensic copilot without needing a live presenter.
            </p>

            <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 mb-6 text-xs text-slate-700 space-y-2 font-sans">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>10-Detector Forensic Matrix &amp; Calibrated minDCF (0.0733)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>Live 96-bar audio waveform &amp; test clip evaluation</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>Interactive STFT Spectrogram, Batch Queue, and Copilot</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleStartTour}
                className="btn-glow-inward-nsa flex-1 bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] hover:opacity-95 text-white font-semibold py-3 px-5 rounded-full text-sm flex items-center justify-center gap-2 shadow-md cursor-pointer transition-all"
              >
                <Play className="w-4 h-4 fill-white relative z-10" />
                <span className="relative z-10">Start Guided Tour</span>
              </button>
              <button
                onClick={() => {
                  setShowWelcomeModal(false);
                  localStorage.setItem("hearsay_demo_tour_completed", "true");
                }}
                className="btn-glow-inward-light px-5 py-3 rounded-full text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer border border-slate-200/80"
              >
                Explore on My Own
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. Interactive Spotlight & Tour Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-[9990] pointer-events-none select-none">
          {/* Subtle Dimming Backdrop */}
          <div
            className="absolute inset-0 bg-[#001730]/40 backdrop-blur-[2px] pointer-events-auto transition-opacity duration-300"
            onClick={handleCloseTour}
          />

          {/* Focused Highlight Ring around active element */}
          {targetRect && (
            <div
              className="absolute border-2 border-amber-400 rounded-3xl shadow-[0_0_40px_rgba(195,117,48,0.45)] pointer-events-none transition-all duration-300 ease-out z-[9992]"
              style={{
                top: `${Math.max(8, targetRect.top - 8)}px`,
                left: `${Math.max(8, targetRect.left - 8)}px`,
                width: `${targetRect.width + 16}px`,
                height: `${targetRect.height + 16}px`,
              }}
            >
              {/* Corner Accents */}
              <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-amber-300" />
              <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-amber-300" />
              <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-amber-300" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-amber-300" />
            </div>
          )}

          {/* 3. The Master Floating Tour Control Card */}
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[9999] w-[94vw] max-w-xl bg-white/95 text-slate-900 rounded-[28px] p-6 sm:p-7 shadow-[0_20px_60px_rgba(0,0,0,0.25)] border border-slate-200/90 backdrop-blur-xl pointer-events-auto animate-in fade-in slide-in-from-bottom-5 duration-300">
            {/* Auto-Play Progress Bar (Top Edge) */}
            {isAutoPlaying && (
              <div className="absolute top-0 inset-x-0 h-1.5 bg-slate-100 overflow-hidden rounded-t-[28px]">
                <div
                  className="h-full bg-gradient-to-r from-[#005493] via-[#c37530] to-amber-400 transition-all duration-100 ease-linear"
                  style={{ width: `${autoProgress}%` }}
                />
              </div>
            )}

            {/* Step Header */}
            <div className="flex items-center justify-between gap-3 pb-3 mb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full bg-[#005493]/10 text-[#005493] border border-[#005493]/20 font-mono">
                  Step {currentStep + 1} of {TOUR_STEPS.length}
                </span>
                <span className="text-xs font-semibold text-[#c37530] font-sans">
                  {activeStepData.badge}
                </span>
              </div>

              {/* Controls: Auto-Play toggle & Exit button */}
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                  className={`btn-glow-inward-light text-[11px] px-3 py-1 rounded-full font-medium flex items-center gap-1.5 transition-colors cursor-pointer border ${
                    isAutoPlaying
                      ? "bg-amber-50 text-amber-800 border-amber-300 font-semibold"
                      : "bg-slate-100 hover:bg-slate-200/70 text-slate-700 border-slate-200/80"
                  }`}
                  title="Toggle Auto-Play Demo (8s per feature)"
                >
                  {isAutoPlaying ? (
                    <>
                      <Pause className="w-3 h-3 text-amber-700" />
                      <span>Pause Auto</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3 h-3 text-slate-600 fill-slate-600" />
                      <span>Auto-Play</span>
                    </>
                  )}
                </button>

                <button
                  onClick={handleCloseTour}
                  className="w-7 h-7 rounded-full bg-slate-100 hover:bg-slate-200/80 text-slate-500 hover:text-slate-800 flex items-center justify-center transition-colors cursor-pointer"
                  title="Exit Tour (Esc)"
                  aria-label="Exit Tour"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Step Content */}
            <div className="space-y-3">
              <h3 className="text-lg font-bold text-slate-900 tracking-tight font-sans leading-tight">
                {activeStepData.title}
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed font-sans">
                {activeStepData.description}
              </p>

              {/* Rubric / Key Technical Note */}
              <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-3 flex items-start gap-2.5">
                <Sparkles className="w-4 h-4 text-[#c37530] shrink-0 mt-0.5" />
                <p className="text-[11px] text-slate-700 font-sans leading-snug">
                  <strong>Rubric Spotlight:</strong> {activeStepData.rubricNote}
                </p>
              </div>

              {/* Key Metrics Strip */}
              {activeStepData.metrics && (
                <div className="grid grid-cols-3 gap-2 pt-1">
                  {activeStepData.metrics.map((m, idx) => (
                    <div
                      key={idx}
                      className="bg-white border border-slate-200/80 rounded-xl px-2.5 py-1.5 text-center shadow-2xs"
                    >
                      <span className="text-[10px] text-slate-400 block font-sans">
                        {m.label}
                      </span>
                      <span className="text-xs font-bold text-slate-900 font-mono">
                        {m.value}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Optional Interactive Action in Step */}
              {activeStepData.actionLabel && (
                <div className="pt-1">
                  <button
                    onClick={handleStepAction}
                    className="btn-glow-inward-nsa w-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] hover:opacity-95 text-white font-semibold py-2 px-4 rounded-xl text-xs flex items-center justify-center gap-2 shadow-xs cursor-pointer transition-all"
                  >
                    {activeStepData.actionIcon && (
                      <activeStepData.actionIcon className="w-3.5 h-3.5 relative z-10" />
                    )}
                    <span className="relative z-10">{activeStepData.actionLabel}</span>
                  </button>
                </div>
              )}
            </div>

            {/* Tour Footer Navigation */}
            <div className="flex items-center justify-between pt-4 mt-4 border-t border-slate-100">
              <button
                onClick={handlePrev}
                disabled={currentStep === 0}
                className="btn-glow-inward-light flex items-center gap-1 text-xs font-semibold px-3 py-1.5 rounded-full border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none cursor-pointer"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Back</span>
              </button>

              {/* Progress Dots */}
              <div className="flex items-center gap-1.5">
                {TOUR_STEPS.map((s, idx) => (
                  <button
                    key={s.id}
                    onClick={() => setCurrentStep(idx)}
                    className={`h-2 rounded-full transition-all cursor-pointer ${
                      idx === currentStep
                        ? "w-6 bg-[#005493]"
                        : "w-2 bg-slate-200 hover:bg-slate-300"
                    }`}
                    aria-label={`Jump to step ${idx + 1}`}
                  />
                ))}
              </div>

              <button
                onClick={handleNext}
                className="btn-glow-inward-nsa flex items-center gap-1 text-xs font-semibold px-4 py-1.5 rounded-full bg-gradient-to-r from-[#005493] to-[#003d73] text-white hover:opacity-95 shadow-sm cursor-pointer"
              >
                <span>{currentStep === TOUR_STEPS.length - 1 ? "Finish Tour" : "Next"}</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
