"use client";

import React, { useState } from "react";
import {
  FileCode2,
  Waves,
  Mic,
  Zap,
  Cpu,
  Fingerprint,
  BrainCircuit,
  Scissors,
  ChevronRight,
  X,
} from "lucide-react";
import { DottedNumber } from "./DottedNumber";
import { useForensic, ForensicModality } from "@/context/ForensicContext";

export function ModalityGrid() {
  const { modalities, currentAudio } = useForensic();
  const [selectedModality, setSelectedModality] = useState<ForensicModality | null>(null);
  const [showAllModalities, setShowAllModalities] = useState(false);

  const modalityConfig: Record<string, { icon: React.ElementType; iconColor: string; label: string }> = {
    spectral: {
      icon: Waves,
      iconColor: "text-[#005493]",
      label: "Spectral & Vocoder",
    },
    speaker: {
      icon: Fingerprint,
      iconColor: "text-[#c37530]",
      label: "Speaker Consistency",
    },
    deepSpoof: {
      icon: BrainCircuit,
      iconColor: "text-[#005493]",
      label: "Deep SSL Anti-Spoof",
    },
    enf: {
      icon: Zap,
      iconColor: "text-[#c37530]",
      label: "60Hz ENF Mains Hum",
    },
    container: {
      icon: FileCode2,
      iconColor: "text-[#00254b]",
      label: "Container & Headers",
    },
    prosody: {
      icon: Mic,
      iconColor: "text-[#c37530]",
      label: "Prosody & Phonetics",
    },
    compression: {
      icon: Cpu,
      iconColor: "text-[#005493]",
      label: "Compression & Transcode",
    },
    splice: {
      icon: Scissors,
      iconColor: "text-rose-600",
      label: "Splice & Discontinuity",
    },
  };

  const primaryKeys = ["spectral", "speaker", "deepSpoof", "enf"];
  const displayKeys = showAllModalities ? Object.keys(modalities) : primaryKeys;

  return (
    <div id="tour-modality-grid" className="flex flex-col h-full justify-between select-none">
      {/* 2x2 Grid of Modality Cards */}
      <div className="grid grid-cols-2 gap-4 h-full">
        {displayKeys.slice(0, 4).map((key) => {
          const mod = modalities[key];
          if (!mod) return null;
          const config = modalityConfig[key] || {
            icon: Waves,
            iconColor: "text-[#005493]",
            label: mod.name,
          };
          const Icon = config.icon;
          const isFlagged = mod.anomaly;

          return (
            <div
              key={key}
              onClick={() => setSelectedModality(mod)}
              className="bg-white/95 backdrop-blur-md rounded-[24px] p-5 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-200/90 flex flex-col justify-between hover:shadow-md transition-all cursor-pointer group relative overflow-hidden"
            >
              {/* Top Accent Icon & Status Indicator (Transparent background, colored icon) */}
              <div className="flex items-center justify-between">
                <div className="w-7 h-7 flex items-center justify-center transition-transform group-hover:scale-110">
                  <Icon className={`w-5 h-5 ${config.iconColor}`} />
                </div>
                <span
                  className={`text-[11px] font-sans font-medium flex items-center gap-1.5 ${
                    isFlagged ? "text-rose-600" : "text-emerald-600"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isFlagged ? "bg-rose-500 animate-pulse" : "bg-emerald-500"
                    }`}
                  />
                  {isFlagged ? "Anomaly" : "Clean"}
                </span>
              </div>

              {/* Title & Badge */}
              <div className="my-2">
                <h4 className="text-slate-900 font-semibold text-xs sm:text-sm tracking-tight leading-snug font-sans">
                  {config.label}
                </h4>
                <p className="text-[11px] text-slate-500 font-mono mt-0.5 truncate">
                  {mod.badge}
                </p>
              </div>

              {/* Probability Meter & Metric */}
              <div className="pt-2 border-t border-slate-100 flex items-end justify-between">
                <div>
                  <span className="text-[10px] uppercase font-mono text-slate-400 block font-semibold">
                    Spoof Prob
                  </span>
                  <div className="text-lg font-mono font-bold text-slate-900 leading-tight">
                    <DottedNumber
                      value={`${(mod.score * 100).toFixed(0)}%`}
                      subduedDecimals
                      subduedClassName="text-slate-400 font-normal"
                    />
                  </div>
                </div>

                <span className="text-[11px] font-semibold text-[#005493] flex items-center gap-0.5 group-hover:translate-x-0.5 transition-transform">
                  Inspect &rarr;
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer Drawer Trigger */}
      <div className="mt-3 flex items-center justify-between bg-white px-4 py-2.5 rounded-2xl border border-slate-200/90 shadow-sm text-xs">
        <span className="text-slate-600 font-medium">
          Coverage: <strong className="text-slate-900 font-semibold">8 of 8 NSA Modalities Active</strong>
        </span>
        <button
          onClick={() => setShowAllModalities(true)}
          className="text-[#005493] hover:text-[#003d73] font-semibold cursor-pointer flex items-center gap-1 font-sans"
        >
          <span>View All 8 Modalities</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Flyout for Single Modality Details */}
      {selectedModality && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/40 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-3xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-150 text-slate-900 relative">
            <button
              onClick={() => setSelectedModality(null)}
              className="btn-glow-inward-light absolute top-5 right-5 w-8 h-8 rounded-full bg-white hover:bg-slate-50 border border-slate-200/90 flex items-center justify-center text-slate-500 hover:text-slate-800 cursor-pointer shadow-2xs"
            >
              <X className="w-4 h-4 relative z-10" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 flex items-center justify-center">
                {React.createElement(
                  modalityConfig[selectedModality.id]?.icon || Waves,
                  { className: `w-6 h-6 ${modalityConfig[selectedModality.id]?.iconColor || "text-[#005493]"}` }
                )}
              </div>
              <div>
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold font-sans">
                  {selectedModality.category}
                </span>
                <h3 className="text-lg font-semibold tracking-tight font-sans">{selectedModality.name}</h3>
              </div>
            </div>

            <div className="space-y-3 my-4">
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-500 block font-sans">Classifier Confidence</span>
                  <span className="text-xl font-bold text-slate-900 font-sans">
                    {(selectedModality.score * 100).toFixed(1)}% Spoof Likelihood
                  </span>
                </div>
                <span
                  className={`text-xs font-semibold font-sans flex items-center gap-1.5 ${
                    selectedModality.anomaly ? "text-rose-600" : "text-emerald-600"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      selectedModality.anomaly ? "bg-rose-500 animate-pulse" : "bg-emerald-500"
                    }`}
                  />
                  {selectedModality.anomaly ? "Anomaly Flagged" : "Clean"}
                </span>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 text-xs">
                <span className="text-slate-500 block mb-1 font-semibold font-sans">Diagnostic Evidence:</span>
                <p className="text-slate-700 font-medium leading-relaxed font-sans">
                  {selectedModality.detail}
                </p>
              </div>

              <div className="p-3.5 rounded-2xl bg-[#005493]/5 border border-[#005493]/20 text-xs flex items-center justify-between">
                <span className="text-[#005493] font-medium font-sans">Metric Reading:</span>
                <span className="font-bold text-[#00254b] font-sans">{selectedModality.metric}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedModality(null)}
                className="btn-glow-inward-nsa px-5 py-2 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white text-xs font-semibold cursor-pointer shadow-sm"
              >
                <span className="relative z-10">Close Telemetry</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal for All 8 Modalities List */}
      {showAllModalities && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/40 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-3xl max-w-2xl w-full p-6 shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-150 text-slate-900 relative max-h-[85vh] flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-[#005493] font-semibold">
                  Full NSA Forensic Scope
                </span>
                <h3 className="text-lg font-semibold tracking-tight font-sans">8 Forensic Analysis Modalities</h3>
              </div>
              <button
                onClick={() => setShowAllModalities(false)}
                className="btn-glow-inward-light w-8 h-8 rounded-full bg-white hover:bg-slate-50 border border-slate-200/90 flex items-center justify-center text-slate-500 hover:text-slate-800 cursor-pointer shadow-2xs"
              >
                <X className="w-4 h-4 relative z-10" />
              </button>
            </div>

            <div className="divide-y divide-slate-100 overflow-y-auto pr-1 my-3 space-y-2">
              {Object.keys(modalities).map((k) => {
                const mod = modalities[k];
                const config = modalityConfig[k] || {
                  icon: Waves,
                  iconColor: "text-[#005493]",
                  label: mod.name,
                };
                const Icon = config.icon;
                return (
                  <div key={k} className="py-3 flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3">
                      <div className="w-7 h-7 flex items-center justify-center shrink-0 mt-0.5">
                        <Icon className={`w-5 h-5 ${config.iconColor}`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-semibold text-slate-900 font-sans">{mod.name}</h4>
                          <span className="text-[10px] text-slate-400 font-medium font-sans">({mod.category})</span>
                        </div>
                        <p className="text-xs text-slate-600 mt-0.5 leading-snug font-sans">{mod.detail}</p>
                        <span className="inline-block mt-1 font-sans text-[11px] text-[#005493] bg-[#005493]/10 px-2 py-0.5 rounded font-semibold border border-[#005493]/20">
                          {mod.metric}
                        </span>
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <span className="text-xs font-bold text-slate-900 block font-sans">
                        {(mod.score * 100).toFixed(0)}%
                      </span>
                      <span
                        className={`text-[11px] font-medium font-sans flex items-center justify-end gap-1 mt-0.5 ${
                          mod.anomaly ? "text-rose-600" : "text-emerald-600"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            mod.anomaly ? "bg-rose-500 animate-pulse" : "bg-emerald-500"
                          }`}
                        />
                        {mod.anomaly ? "Anomaly" : "Clean"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-end">
              <button
                onClick={() => setShowAllModalities(false)}
                className="btn-glow-inward-nsa px-5 py-2 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white text-xs font-semibold cursor-pointer shadow-sm"
              >
                <span className="relative z-10">Done</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Re-export as CategoryGrid
export { ModalityGrid as CategoryGrid };
