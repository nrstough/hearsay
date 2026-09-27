"use client";

import React from "react";
import {
  FileCode2,
  Waves,
  Mic,
  Zap,
  Cpu,
  Fingerprint,
  BrainCircuit,
  Scissors,
  Layers,
} from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function ModalitiesView() {
  const { modalities, currentAudio } = useForensic();

  const modalityIcons: Record<string, React.ElementType> = {
    container: FileCode2,
    spectral: Waves,
    prosody: Mic,
    enf: Zap,
    compression: Cpu,
    speaker: Fingerprint,
    deepSpoof: BrainCircuit,
    splice: Scissors,
  };

  const modalityColors: Record<string, string> = {
    container: "text-[#00254b]",
    spectral: "text-[#005493]",
    prosody: "text-[#c37530]",
    enf: "text-[#c37530]",
    compression: "text-[#005493]",
    speaker: "text-[#c37530]",
    deepSpoof: "text-[#005493]",
    splice: "text-rose-600",
  };

  return (
    <div id="tour-modalities-view" className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 space-y-6 select-none relative z-10">
      {/* Banner */}
      <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-white bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] uppercase tracking-wider px-3 py-1 rounded-full shadow-sm shadow-[#005493]/20">
            Forensic Diversity &amp; Breadth (20% Scoring Rubric)
          </span>
          <h2 className="text-xl font-semibold text-slate-900 tracking-tight mt-2 font-sans">
            8-Modality Multi-Technique Forensic Orchestrator
          </h2>
          <p className="text-xs text-slate-600 mt-1 font-sans">
            Analyzing target '{currentAudio.filename}' across container, spectral, prosodic, electromagnetic, codec, biometric, neural probe, and phase domains.
          </p>
        </div>

        <div className="bg-[#005493]/10 border border-[#005493]/20 px-4 py-2.5 rounded-2xl flex items-center gap-3">
          <Layers className="w-5 h-5 text-[#005493]" />
          <div>
            <span className="text-[10px] uppercase text-[#005493] font-bold block font-sans">
              Agentic Orchestration
            </span>
            <span className="text-xs font-semibold text-slate-900 font-sans">
              8 of 8 Detectors Synced
            </span>
          </div>
        </div>
      </div>

      {/* Grid of 8 Detailed Modality Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {Object.keys(modalities).map((key) => {
          const mod = modalities[key];
          const Icon = modalityIcons[key] || Waves;
          const iconColor = modalityColors[key] || "text-[#005493]";
          const isFlagged = mod.anomaly;

          return (
            <div
              key={key}
              className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col justify-between hover:shadow-md transition-all group"
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="w-7 h-7 flex items-center justify-center transition-transform group-hover:scale-110">
                    <Icon className={`w-5 h-5 ${iconColor}`} />
                  </div>
                  <span
                    className={`text-xs font-sans font-medium flex items-center gap-1.5 ${
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

                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  {mod.category}
                </span>
                <h3 className="text-sm font-semibold text-slate-900 mt-0.5 tracking-tight font-sans">
                  {mod.name}
                </h3>
                <p className="text-xs text-slate-600 mt-2 leading-relaxed font-sans">
                  {mod.detail}
                </p>
              </div>

              <div className="mt-5 pt-3 border-t border-slate-100">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-slate-500 font-mono text-[11px] font-medium">Spoof Confidence</span>
                  <span className="font-mono font-bold text-slate-900">
                    {(mod.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full transition-all duration-300 ${
                      isFlagged
                        ? "bg-gradient-to-r from-rose-500 to-red-600"
                        : "bg-gradient-to-r from-[#005493] to-[#c37530]"
                    }`}
                    style={{ width: `${mod.score * 100}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono font-semibold text-[#005493] bg-[#005493]/10 px-2 py-0.5 rounded block truncate border border-[#005493]/20">
                  {mod.metric}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
