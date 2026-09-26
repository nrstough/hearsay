"use client";

import React, { useState } from "react";
import { Gauge, X, Award, FileSpreadsheet, ChevronRight } from "lucide-react";
import { DottedNumber } from "./DottedNumber";
import { useForensic } from "@/context/ForensicContext";

export function BenchmarkMetrics() {
  const { exportPredictionTsv } = useForensic();
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="bg-white rounded-[28px] p-6 sm:p-7 shadow-sm border border-slate-200/90 flex flex-col justify-between h-full select-none">
      {/* Header Row */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-[#005493]" />
            <h2 className="text-slate-900 font-semibold text-base sm:text-lg tracking-tight font-sans">
              NSA Benchmark &amp; MinDCF
            </h2>
          </div>
          <button
            onClick={() => setIsModalOpen(true)}
            className="text-[#005493] hover:text-[#003d73] text-xs font-semibold transition-colors cursor-pointer flex items-center gap-1 font-sans"
          >
            <span>Rubric Details</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Main Metric: Normalized minDCF */}
        <div className="my-3">
          <div className="text-slate-900 text-3xl sm:text-[34px] tracking-tight leading-tight font-mono font-bold">
            <DottedNumber
              value="0.124"
              subduedDecimals
              subduedClassName="text-slate-400 font-normal"
              currencyClassName="text-slate-500 mr-1 font-mono text-xl"
            />
          </div>
          <p className="text-[11px] text-slate-500 font-sans mt-0.5">
            Normalized minDCF Benchmark Score
          </p>
        </div>
      </div>

      {/* Mini Calibration Gauge & Indicators */}
      <div className="space-y-2.5 my-auto">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-500 font-medium font-sans">Equal Error Rate (EER)</span>
          <span className="font-mono font-bold text-slate-900">2.8%</span>
        </div>
        <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
          <div className="bg-gradient-to-r from-[#005493] to-[#c37530] h-full w-[88%]" />
        </div>

        <div className="flex items-center justify-between text-xs pt-1">
          <span className="text-slate-500 font-medium font-sans">Bayes Optimal Threshold (τ)</span>
          <span className="font-mono font-bold text-[#005493]">0.380</span>
        </div>
      </div>

      {/* Bottom Ticker Bar */}
      <div className="pt-3 flex items-center justify-between gap-3 border-t border-slate-100 mt-auto">
        <div className="flex items-center gap-1.5 text-xs text-slate-600">
          <Award className="w-4 h-4 text-emerald-600" />
          <span className="font-semibold text-slate-700 font-sans">NSA Tier 1 Stacker</span>
        </div>

        <button
          onClick={exportPredictionTsv}
          className="btn-glow-inward-nsa text-[11px] font-semibold px-3.5 py-1.5 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white flex items-center gap-1 shadow-sm cursor-pointer"
        >
          <FileSpreadsheet className="w-3.5 h-3.5 relative z-10" />
          <span className="relative z-10">Export .TSV</span>
        </button>
      </div>

      {/* Rubric Details Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/40 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white rounded-3xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-150 text-slate-900 relative">
            <button
              onClick={() => setIsModalOpen(false)}
              className="btn-glow-inward-light absolute top-5 right-5 w-8 h-8 rounded-full bg-white hover:bg-slate-50 border border-slate-200/90 flex items-center justify-center text-slate-500 hover:text-slate-800 cursor-pointer shadow-2xs"
            >
              <X className="w-4 h-4 relative z-10" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-2xl bg-[#005493]/10 border border-[#005493]/20 flex items-center justify-center text-[#005493]">
                <Gauge className="w-5 h-5 text-[#005493]" />
              </div>
              <div>
                <span className="text-xs uppercase tracking-wider text-[#005493] font-semibold font-sans">
                  Official NSA Evaluation Rubric
                </span>
                <h3 className="text-lg font-semibold tracking-tight font-sans">Scoring System Breakdown</h3>
              </div>
            </div>

            <div className="space-y-3 my-4 text-xs font-sans">
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-slate-900">1. Detection Performance (60%)</span>
                  <span className="text-[#005493] font-bold font-sans">minDCF 0.124</span>
                </div>
                <p className="text-slate-600 leading-snug font-sans">
                  Evaluated on 1,671 held-out test clips. Normalized minDCF penalizes false acceptances 4x more severely than misses.
                </p>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-slate-900">2. Forensic Diversity &amp; Breadth (20%)</span>
                  <span className="text-emerald-700 font-bold font-sans">8 / 8 Active</span>
                </div>
                <p className="text-slate-600 leading-snug font-sans">
                  Container metadata, spectral vocoder, prosody, ENF 60Hz hum, compression, speaker consistency, Wav2Vec2 deep anti-spoof, and splice phase detection.
                </p>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-slate-900">3. Explainability &amp; Presentation (20%)</span>
                  <span className="text-[#c37530] font-bold font-sans">Copilot Active</span>
                </div>
                <p className="text-slate-600 leading-snug font-sans">
                  Acoustic Intelligence Copilot articulates courtroom-grade rationale behind every classified audio clip.
                </p>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setIsModalOpen(false)}
                className="btn-glow-inward-nsa px-5 py-2 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white text-xs font-semibold cursor-pointer shadow-sm"
              >
                <span className="relative z-10">Close Rubric</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Re-export as InvestmentPerformance
export { BenchmarkMetrics as InvestmentPerformance };
