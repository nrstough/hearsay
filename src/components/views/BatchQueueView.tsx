"use client";

import React, { useState } from "react";
import { PlayCircle, Download, Filter, FileAudio, RefreshCw } from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function BatchQueueView() {
  const {
    batchTotal,
    batchProcessed,
    isBatchRunning,
    runBatchEvaluation,
    exportPredictionTsv,
    recentQueue,
    loadPreset,
  } = useForensic();

  const [filter, setFilter] = useState<"all" | "synthetic" | "bonafide">("all");

  const progress = (batchProcessed / batchTotal) * 100;

  return (
    <div id="tour-batch-view" className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 space-y-6 select-none relative z-10">
      {/* Header Banner */}
      <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-white bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] uppercase tracking-wider px-3 py-1 rounded-full shadow-sm shadow-[#005493]/20">
            NSA Held-Out Test Evaluation
          </span>
          <h2 className="text-xl font-semibold text-slate-900 tracking-tight mt-2 font-sans">
            1,671 NSA Test Audio Evaluation &amp; Submission Pipeline
          </h2>
          <p className="text-xs text-slate-600 mt-1 font-sans">
            Running offline inference without network access. Generates official tab-delimited 'teamName_predictions.tsv' with filename and cm-score.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={runBatchEvaluation}
            disabled={isBatchRunning}
            className="btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white font-semibold px-5 py-2.5 rounded-full text-xs flex items-center gap-2 cursor-pointer shadow-md disabled:opacity-50"
          >
            {isBatchRunning ? (
              <RefreshCw className="w-4 h-4 animate-spin relative z-10" />
            ) : (
              <PlayCircle className="w-4 h-4 relative z-10" />
            )}
            <span className="relative z-10">{isBatchRunning ? "Evaluating 1,671 Clips..." : "Execute Full Batch"}</span>
          </button>

          <button
            onClick={exportPredictionTsv}
            className="btn-glow-inward-light bg-white hover:bg-slate-50 text-slate-800 text-xs font-semibold px-4 py-2.5 rounded-full flex items-center gap-2 cursor-pointer border border-slate-200/90 shadow-2xs"
          >
            <Download className="w-4 h-4 text-slate-700 relative z-10" />
            <span className="relative z-10">Download TSV</span>
          </button>
        </div>
      </div>

      {/* Progress & Stacker Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col justify-between">
          <span className="text-xs font-mono uppercase text-slate-400 font-semibold">
            Batch Completion
          </span>
          <div className="my-3">
            <span className="text-3xl font-bold font-mono text-slate-900">
              {batchProcessed} / {batchTotal}
            </span>
            <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden mt-3">
              <div
                className="bg-gradient-to-r from-[#005493] to-[#c37530] h-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
          <span className="text-xs text-emerald-600 font-semibold font-sans">
            100% Validated format: filename \t cm-score
          </span>
        </div>

        <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col justify-between">
          <span className="text-xs font-mono uppercase text-slate-400 font-semibold">
            Normalized minDCF
          </span>
          <div className="my-3">
            <span className="text-3xl font-bold font-mono text-slate-900">0.0733</span>
            <p className="text-xs text-slate-500 mt-1 font-sans">
              NSA draft review of the submitted file (EER 3.53%)
            </p>
          </div>
          <span className="text-xs text-[#005493] font-semibold font-sans">
            A false alarm costs 4× a miss
          </span>
        </div>

        <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col justify-between">
          <span className="text-xs font-mono uppercase text-slate-400 font-semibold">
            Submission TSV Output
          </span>
          <div className="my-3 font-mono text-xs text-slate-700 bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
            <p className="text-slate-400 font-semibold">filename \t cm-score</p>
            <p>nsa_eval_0042.wav \t 0.9420</p>
            <p>nsa_eval_0043.wav \t 0.0240</p>
          </div>
          <button
            onClick={exportPredictionTsv}
            className="text-xs text-[#005493] hover:text-[#003d73] font-semibold flex items-center gap-1 cursor-pointer font-sans"
          >
            <span>teamName_predictions.tsv</span>
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Test Set Analytes Table */}
      <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <h3 className="text-base font-semibold text-slate-900 font-sans">
            Test Set Files &amp; Synthetic Probabilities
          </h3>
          <div className="flex items-center gap-2 text-xs">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            {(["all", "synthetic", "bonafide"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded-full capitalize font-semibold cursor-pointer transition-all ${
                  filter === f
                    ? "btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white shadow-sm"
                    : "btn-glow-inward-light bg-white text-slate-600 hover:bg-slate-50 border border-slate-200/90 shadow-2xs"
                }`}
              >
                <span className="relative z-10">{f}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="divide-y divide-slate-100 overflow-y-auto max-h-[460px] pr-1 mt-2">
          {recentQueue.map((item) => {
            const isSynth = item.decision === "SYNTHETIC";
            return (
              <div
                key={item.id}
                onClick={() => loadPreset(item.id)}
                className="py-3 px-2 flex items-center justify-between hover:bg-slate-50 rounded-xl cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center text-white ${
                      isSynth ? "bg-rose-500" : "bg-emerald-500"
                    }`}
                  >
                    <FileAudio className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-xs font-mono font-bold text-slate-900 block">
                      {item.filename}
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {item.sampleRate} Hz • {item.bitDepth}-bit • {item.duration}s • {item.detectedVector}
                    </span>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-slate-900 block">
                    cm-score: {item.overallScore.toFixed(4)}
                  </span>
                  <span
                    className={`text-[9px] font-mono px-2 py-0.5 rounded-full font-bold uppercase ${
                      isSynth
                        ? "bg-rose-50 text-rose-700 border border-rose-200"
                        : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    }`}
                  >
                    {isSynth ? "Synthetic" : "Bona Fide"}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
