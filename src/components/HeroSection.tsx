"use client";

import React, { useRef } from "react";
import { ShieldCheck, AlertOctagon, Sparkles, Upload, Mic, PlayCircle } from "lucide-react";
import { DottedNumber } from "./DottedNumber";
import { useForensic } from "@/context/ForensicContext";

interface HeroSectionProps {
  onOpenBatch?: () => void;
  onOpenAudit?: () => void;
  onStartTour?: () => void;
}

export function HeroSection({ onOpenBatch, onOpenAudit, onStartTour }: HeroSectionProps) {
  const {
    currentAudio,
    runBatchEvaluation,
    isBatchRunning,
    batchProcessed,
    batchTotal,
    runFullForensicAudit,
    isAnalyzing,
  } = useForensic();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const isSynthetic = currentAudio.decision === "SYNTHETIC";

  return (
    <div id="tour-hero" className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 pt-0 sm:pt-0.5 pb-6 sm:pb-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-end">
        {/* Left Column: Welcome & Action Buttons */}
        <div className="lg:col-span-7 flex flex-col justify-end -translate-y-2 sm:-translate-y-4 lg:-translate-y-5">
          {/* Subtitle */}
          <p className="text-white/90 text-sm sm:text-base font-normal tracking-normal mb-2 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#ffb800] animate-pulse" />
            Autonomous 8-modality audio authentication &amp; forensic intelligence
          </p>

          {/* Main Title */}
          <h1 className="text-white text-3xl sm:text-4xl lg:text-[38px] font-medium tracking-tight mb-3.5 sm:mb-4 font-sans leading-[1.18]">
            HEARSAY Forensics Cockpit
          </h1>

          {/* Action Buttons Row */}
          <div className="flex flex-wrap items-center gap-3 sm:gap-4">
            {/* Interactive Guided Demo Tour */}
            {onStartTour && (
              <button
                onClick={onStartTour}
                className="btn-glow-inward-light bg-gradient-to-r from-amber-500/25 to-[#c37530]/30 hover:from-amber-500/35 hover:to-[#c37530]/40 text-amber-200 border border-amber-400/40 hover:border-amber-300 font-semibold px-5 py-3 rounded-full text-sm transition-all flex items-center justify-center gap-2 select-none cursor-pointer shadow-md hover:scale-[1.02] active:scale-95"
              >
                <Sparkles className="w-4 h-4 text-amber-300 relative z-10 animate-pulse" />
                <span className="relative z-10">Start Guided Tour</span>
              </button>
            )}

            {/* Run Full Forensic Audit */}
            <button
              onClick={runFullForensicAudit}
              disabled={isAnalyzing}
              className="btn-glow-inward-light bg-white/95 text-slate-900 border border-white/70 font-semibold px-6 py-3 rounded-full text-sm transition-all hover:bg-white flex items-center justify-center gap-2 select-none cursor-pointer shadow-sm hover:scale-[1.02] active:scale-95 min-w-[190px]"
            >
              <Sparkles className="w-4 h-4 text-[#c37530] relative z-10" />
              <span className="relative z-10">
                {isAnalyzing ? "Running ten detectors..." : "Re-Scan Current File"}
              </span>
            </button>

            {/* Run NSA 1,671 Test Set Batch */}
            <button
              onClick={runBatchEvaluation}
              disabled={isBatchRunning}
              className="btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white font-semibold px-6 py-3 rounded-full text-sm flex items-center justify-center gap-2 select-none cursor-pointer shadow-lg shadow-[#005493]/30 hover:brightness-110 active:scale-95 transition-all min-w-[240px]"
            >
              <PlayCircle className="w-4 h-4 text-white relative z-10" />
              <span className="relative z-10">
                {isBatchRunning
                  ? `Evaluating Batch (${batchProcessed}/${batchTotal})...`
                  : "Load the submitted 1,671-file run"}
              </span>
            </button>
          </div>
        </div>

        {/* Right Column: Clean Frosted Audio Vault Glass Card */}
        <div className="lg:col-span-5 flex justify-center lg:justify-end">
          <div className="w-full max-w-[440px] relative select-none">
            {/* Active Frosted Card */}
            <div className="glass-card rounded-[28px] p-7 sm:p-8 text-white relative overflow-hidden transition-all duration-300 hover:shadow-2xl">
              {/* Subtle top glare reflection */}
              <div className="absolute inset-0 bg-gradient-to-b from-white/20 via-transparent to-transparent pointer-events-none rounded-[28px]" />

              {/* Top Row: System Status & MinDCF Metric */}
              <div className="flex items-center justify-between text-white text-xs sm:text-sm font-normal relative z-10 mb-6 sm:mb-7">
                <div className="flex items-center gap-2">
                  <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
                  </span>
                  <span className="text-xs font-semibold text-white/95 tracking-wide whitespace-nowrap">
                    Shipped rule: rank blend, Spectra suppression only
                  </span>
                </div>
                <div className="glass-pill rounded-full px-3 py-1 text-xs font-medium text-white/95 shadow-sm whitespace-nowrap border border-white/20 font-sans">
                  fused rank: {currentAudio.fusedRank.toFixed(3)}
                </div>
              </div>

              {/* Synthetic Probability Score Display */}
              <div className="relative z-10 my-6 sm:my-7">
                <span className="text-xs font-medium text-white/75 block mb-1.5 tracking-wide uppercase font-sans">
                  Synthetic Probability
                </span>
                <div className="text-4xl sm:text-[46px] tracking-tight text-white leading-none font-sans">
                  <DottedNumber
                    value={currentAudio.overallScore.toFixed(3)}
                    subduedDecimals
                    subduedClassName="opacity-80 font-normal font-sans"
                    currencyClassName="mr-1 opacity-90 font-sans font-normal"
                  />
                </div>
              </div>

              {/* Bottom Row: Verdict Pill & Decision Vector */}
              <div className="relative z-10 flex items-center justify-between gap-3">
                <div
                  className={`rounded-full px-3.5 py-1 text-xs font-semibold inline-flex items-center gap-1.5 shadow-sm whitespace-nowrap ${
                    isSynthetic
                      ? "bg-rose-500/80 text-white border border-rose-400/40"
                      : "bg-emerald-500/80 text-white border border-emerald-400/40"
                  }`}
                >
                  {isSynthetic ? (
                    <AlertOctagon className="w-3.5 h-3.5" />
                  ) : (
                    <ShieldCheck className="w-3.5 h-3.5" />
                  )}
                  <span>{isSynthetic ? "Synthetic" : "Authentic"}</span>
                </div>

                <div className="glass-pill rounded-full px-3.5 py-1 text-xs text-sky-100 font-medium inline-flex items-center gap-1 shadow-sm whitespace-nowrap">
                  <span className="truncate max-w-[160px]">{currentAudio.detectedVector}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
