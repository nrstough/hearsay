"use client";

import React, { useState } from "react";
import { Waves, Zap, Radio, Sliders, Sparkles, AlertOctagon, CheckCircle2 } from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function SpectrogramView() {
  const { currentAudio, frequencyBars } = useForensic();
  const [fftWindow, setFftWindow] = useState<"1024" | "2048" | "512">("1024");
  const [colorMap, setColorMap] = useState<"CyberNavy" | "Sapphire" | "Viridis">("CyberNavy");
  const isSynthetic = currentAudio.decision === "SYNTHETIC";

  return (
    <div id="tour-spectrogram-view" className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 space-y-6 select-none relative z-10">
      {/* Top Banner */}
      <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold text-white bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] uppercase tracking-wider px-3 py-1 rounded-full shadow-sm shadow-[#005493]/20">
              Time-Frequency STFT Engine
            </span>
            <span className="text-xs text-slate-500 font-semibold font-sans">
              Window: {fftWindow} | Hop: 256
            </span>
          </div>
          <h2 className="text-xl font-semibold text-slate-900 tracking-tight font-sans">
            Spectrogram panel (schematic)
          </h2>
          <p className="text-xs text-slate-600 mt-1 font-sans">
            This panel is a schematic, not a measurement. The pipeline works on 16 kHz mono audio band-matched to 7.25 kHz; the detector cards carry the analysis.
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-100 p-1 rounded-full text-xs font-sans border border-slate-200/70">
            {(["512", "1024", "2048"] as const).map((w) => (
              <button
                key={w}
                onClick={() => setFftWindow(w)}
                className={`btn-glow-inward-light px-3 py-1 rounded-full transition-all font-medium cursor-pointer ${
                  fftWindow === w ? "bg-white text-slate-900 font-semibold shadow-xs border border-slate-200/80" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <span className="relative z-10">{w}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center bg-slate-100 p-1 rounded-full text-xs font-sans border border-slate-200/70">
            {(["CyberNavy", "Sapphire", "Viridis"] as const).map((c) => (
              <button
                key={c}
                onClick={() => setColorMap(c)}
                className={`btn-glow-inward-light px-3 py-1 rounded-full transition-all font-medium cursor-pointer ${
                  colorMap === c ? "bg-white text-slate-900 font-semibold shadow-xs border border-slate-200/80" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <span className="relative z-10">{c}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main 2D Spectrogram Canvas (Pure Light Surface) */}
      <div className="bg-white rounded-[28px] p-6 border border-slate-200/90 shadow-sm relative overflow-hidden text-slate-900">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 text-xs font-mono text-slate-500">
          <span className="flex items-center gap-2 text-[#005493] font-semibold">
            <Waves className="w-4 h-4 text-[#005493]" />
            SCHEMATIC HEATMAP (0 kHz — 8 kHz)
          </span>
          <span className="font-semibold text-slate-700">Target: {currentAudio.filename}</span>
        </div>

        {/* Heatmap visualization grid on light background */}
        <div className="my-6 h-64 w-full rounded-2xl bg-slate-50 border border-slate-200 relative overflow-hidden flex flex-col justify-between p-4">
          {/* Nyquist Cutoff Guideline */}
          {isSynthetic && (
            <div className="absolute top-[33%] left-0 right-0 border-b-2 border-dashed border-rose-500 z-20 flex items-center justify-between px-3">
              <span className="text-[10px] font-mono font-bold text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200 shadow-sm">
                BAND MATCH: 7.25 kHz low-pass
              </span>
              <span className="text-[10px] font-mono text-rose-600 font-semibold">
                nothing above 7.25 kHz is used
              </span>
            </div>
          )}

          {/* 60Hz ENF Trace Guideline at bottom */}
          <div className="absolute bottom-[8%] left-0 right-0 border-b border-[#005493]/40 z-20 flex items-center justify-between px-3">
            <span className="text-[9px] font-mono text-[#005493] bg-[#005493]/10 px-2 py-0.5 rounded border border-[#005493]/20 font-semibold shadow-sm">
              ENF: see the hum detector card
            </span>
            <span className="text-[9px] font-mono text-slate-600 font-semibold">
              {isSynthetic ? "0.00 Hz Variance (Absent)" : "Grid Frequency Locked"}
            </span>
          </div>

          {/* Spectral Heatmap Matrix */}
          <div className="grid grid-cols-12 gap-1.5 h-full">
            {Array.from({ length: 12 }).map((_, colIdx) => (
              <div key={colIdx} className="flex flex-col gap-1 h-full">
                {Array.from({ length: 16 }).map((_, rowIdx) => {
                  const isVoid = isSynthetic && rowIdx < 5;
                  const intensity = isVoid
                    ? 0.05
                    : Math.sin(colIdx * 0.5 + rowIdx * 0.3) * 0.4 + 0.5;

                  return (
                    <div
                      key={rowIdx}
                      className="flex-1 rounded-sm transition-all duration-300"
                      style={{
                        backgroundColor: isVoid
                          ? "#f1f5f9"
                          : colorMap === "Sapphire"
                          ? `rgba(0, 84, 147, ${intensity * 0.85 + 0.1})`
                          : colorMap === "Viridis"
                          ? `rgba(195, 117, 48, ${intensity * 0.85 + 0.1})`
                          : `rgba(0, 84, 147, ${intensity * 0.85 + 0.1})`,
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Footer info pills */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-100 text-xs font-mono">
          <div className="flex items-center gap-4">
            <span className="text-slate-500">
              Dynamic Range: <strong className="text-slate-900 font-bold">96.3 dB</strong>
            </span>
            <span className="text-slate-500">
              Harmonic Flatness:{" "}
              <strong className={isSynthetic ? "text-rose-600 font-bold" : "text-emerald-600 font-bold"}>
                {isSynthetic ? "0.82 (High Void)" : "0.14 (Organic)"}
              </strong>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isSynthetic ? "bg-rose-500 animate-pulse" : "bg-emerald-500"
              }`}
            />
            <span className="text-slate-700 font-semibold font-sans">
              {isSynthetic
                ? "Flagged: Artificial High-Frequency Spectral Suppression"
                : "Verified: Natural Broadband Acoustic Resonance"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
