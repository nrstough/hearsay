"use client";

import React, { useState } from "react";
import {
  FileAudio,
  Play,
  Download,
  X,
  ChevronRight,
  Filter,
} from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function RecentForensics() {
  const {
    currentAudio,
    recentQueue,
    loadPreset,
    togglePlay,
    isPlaying,
    exportPredictionTsv,
    batchTotal,
  } = useForensic();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [filterType, setFilterType] = useState<"all" | "synthetic" | "bonafide">("all");

  const featured = currentAudio || recentQueue[0];
  const isSynthetic = featured.decision === "SYNTHETIC";

  const filteredQueue = recentQueue.filter((item) => {
    if (filterType === "synthetic") return item.decision === "SYNTHETIC";
    if (filterType === "bonafide") return item.decision === "BONA_FIDE";
    return true;
  });

  return (
    <div className="bg-white rounded-[28px] p-6 sm:p-7 shadow-sm border border-slate-200/90 flex flex-col justify-between h-full select-none">
      {/* Header Row */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-slate-900 font-semibold text-base sm:text-lg tracking-tight font-sans">
            Evaluation Queue &amp; Analytes
          </h2>
          <p className="text-[11px] text-slate-500 font-mono">
            Held-out NSA benchmark test clips &amp; live intercepts
          </p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="text-[#005493] hover:text-[#003d73] text-xs font-semibold transition-colors cursor-pointer flex items-center gap-1 font-sans"
        >
          <span>View All ({batchTotal})</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Featured Analyte Card (Pure Light) */}
      <div className="bg-slate-50/90 hover:bg-slate-100/70 rounded-2xl p-4 sm:p-4.5 flex items-center justify-between gap-3 sm:gap-4 border border-slate-200/80 transition-all my-auto">
        {/* Left Side: Audio Wave Icon + Metadata */}
        <div className="flex items-center gap-3.5 min-w-0 flex-1">
          <div
            className={`w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0 text-white shadow-sm ${
              isSynthetic
                ? "bg-gradient-to-tr from-rose-600 to-red-500"
                : "bg-gradient-to-tr from-emerald-600 to-teal-500"
            }`}
          >
            <FileAudio className="w-5 h-5" />
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-900 truncate font-sans">
                {featured.filename}
              </h4>
              <span
                className={`text-[9px] font-mono px-2 py-0.5 rounded-full uppercase font-bold border ${
                  isSynthetic
                    ? "bg-rose-50 text-rose-700 border-rose-200"
                    : "bg-emerald-50 text-emerald-700 border-emerald-200"
                }`}
              >
                {isSynthetic ? "Synthetic" : "Bona Fide"}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-mono mt-0.5 flex items-center gap-1.5 flex-wrap">
              <span>{featured.timestamp}</span>
              <span>•</span>
              <span className="truncate">{featured.detectedVector}</span>
            </p>
          </div>
        </div>

        {/* Right Side: Score & Play Action */}
        <div className="text-right flex items-center gap-3 flex-shrink-0">
          <div>
            <span className="text-[10px] font-mono uppercase text-slate-400 block font-semibold">
              Confidence
            </span>
            <span className="text-sm font-mono font-bold text-slate-900">
              {(featured.overallScore * 100).toFixed(1)}%
            </span>
          </div>

          <button
            onClick={togglePlay}
            className="btn-glow-inward-light w-8 h-8 rounded-full bg-white border border-slate-200/90 flex items-center justify-center text-slate-700 hover:text-[#005493] hover:border-[#005493]/30 shadow-2xs transition-all cursor-pointer"
            title="Play / Pause Audio"
          >
            <Play className={`w-3.5 h-3.5 ml-0.5 relative z-10 ${isPlaying ? "text-[#005493] fill-[#005493]" : ""}`} />
          </button>
        </div>
      </div>

      {/* Secondary Quick Queue List */}
      <div className="divide-y divide-slate-100 mt-2">
        {recentQueue.slice(1, 3).map((item) => {
          const itemSynth = item.decision === "SYNTHETIC";
          return (
            <div
              key={item.id}
              onClick={() => loadPreset(item.id)}
              className="py-2.5 flex items-center justify-between gap-3 hover:bg-slate-50 px-2 rounded-xl cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    itemSynth ? "bg-rose-500" : "bg-emerald-500"
                  }`}
                />
                <span className="text-xs font-semibold text-slate-800 truncate font-mono">
                  {item.filename}
                </span>
                <span className="text-[10px] text-slate-400 font-mono hidden sm:inline">
                  {item.duration}s
                </span>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <span
                  className={`text-[10px] font-mono font-bold ${
                    itemSynth ? "text-rose-600" : "text-emerald-600"
                  }`}
                >
                  p={item.overallScore.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-400">&rarr;</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Full 1,671 Batch Queue Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/40 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-3xl max-w-2xl w-full p-6 shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-150 text-slate-900 relative max-h-[85vh] flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-[#005493] font-semibold">
                  NSA Evaluation Batch
                </span>
                <h3 className="text-lg font-semibold tracking-tight font-sans">1,671 Held-Out Test Audio Files</h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="btn-glow-inward-light w-8 h-8 rounded-full bg-white hover:bg-slate-50 border border-slate-200/90 flex items-center justify-center text-slate-500 hover:text-slate-800 cursor-pointer shadow-2xs"
              >
                <X className="w-4 h-4 relative z-10" />
              </button>
            </div>

            {/* Filter pills */}
            <div className="flex items-center justify-between gap-2 py-3 border-b border-slate-100 text-xs">
              <div className="flex items-center gap-2">
                <Filter className="w-3.5 h-3.5 text-slate-400" />
                {(["all", "synthetic", "bonafide"] as const).map((ft) => (
                  <button
                    key={ft}
                    onClick={() => setFilterType(ft)}
                    className={`px-3 py-1 rounded-full text-xs font-semibold capitalize transition-all cursor-pointer ${
                      filterType === ft
                        ? "btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white shadow-sm"
                        : "btn-glow-inward-light bg-white text-slate-600 hover:bg-slate-50 border border-slate-200/90 shadow-2xs"
                    }`}
                  >
                    <span className="relative z-10">{ft === "all" ? "All (1,671)" : ft === "synthetic" ? "Synthetic" : "Bona Fide"}</span>
                  </button>
                ))}
              </div>

              <button
                onClick={exportPredictionTsv}
                className="btn-glow-inward-nsa text-xs font-semibold px-4 py-1.5 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white flex items-center gap-1.5 shadow-sm cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 relative z-10" />
                <span className="relative z-10">Export TSV</span>
              </button>
            </div>

            {/* Scrollable list */}
            <div className="divide-y divide-slate-100 overflow-y-auto pr-1 my-2 max-h-[420px]">
              {filteredQueue.map((item, idx) => {
                const itemSynth = item.decision === "SYNTHETIC";
                return (
                  <div
                    key={item.id || idx}
                    onClick={() => {
                      loadPreset(item.id);
                      setIsModalOpen(false);
                    }}
                    className="py-3 flex items-center justify-between gap-3 hover:bg-slate-50 px-2 rounded-xl cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-8 h-8 rounded-xl flex items-center justify-center text-white ${
                          itemSynth ? "bg-rose-500" : "bg-emerald-500"
                        }`}
                      >
                        <FileAudio className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="text-xs font-mono font-bold text-slate-900 block">
                          {item.filename}
                        </span>
                        <span className="text-[11px] text-slate-500 font-sans">{item.detectedVector}</span>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-xs font-mono font-bold text-slate-900 block">
                        {(item.overallScore * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`text-[9px] font-mono px-2 py-0.5 rounded-full font-bold uppercase ${
                          itemSynth
                            ? "bg-rose-50 text-rose-700 border border-rose-200"
                            : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        }`}
                      >
                        {itemSynth ? "Synthetic" : "Bona Fide"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-end">
              <button
                onClick={() => setIsModalOpen(false)}
                className="btn-glow-inward-nsa px-5 py-2 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white text-xs font-semibold cursor-pointer shadow-sm"
              >
                <span className="relative z-10">Close Queue</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Re-export as RecentTransactions
export { RecentForensics as RecentTransactions };
