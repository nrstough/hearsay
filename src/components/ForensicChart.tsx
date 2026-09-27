"use client";

import React, { useState } from "react";
import { ChevronDown, Download, Check, Activity } from "lucide-react";
import { DottedNumber } from "./DottedNumber";
import {
  LineChart,
  Grid,
  Line,
  YAxis,
  XAxis,
  ChartTooltip,
} from "./ui/LineChart";
import { useForensic } from "@/context/ForensicContext";

export function ForensicChart() {
  const { chartData, currentAudio, exportPredictionTsv } = useForensic();
  const [resolution, setResolution] = useState<"Full Clip" | "F0 Zoom" | "Spectral FFT">("Full Clip");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleExport = () => {
    exportPredictionTsv();
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const isSynthetic = currentAudio.decision === "SYNTHETIC";

  return (
    <div id="tour-forensic-chart" className="bg-white/95 backdrop-blur-md rounded-[28px] p-6 sm:p-7 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-200/90 flex flex-col justify-between h-full relative select-none overflow-visible">
      {/* Top Header Row */}
      <div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-[#005493]/10 border border-[#005493]/20 flex items-center justify-center text-[#005493]">
              <Activity className="w-4 h-4 text-[#005493]" />
            </div>
            <div>
              <h2 className="text-slate-900 font-semibold text-base sm:text-lg tracking-tight font-sans">
                Acoustic Telemetry &amp; Anomaly Trajectory
              </h2>
              <p className="text-[11px] text-slate-500 font-mono">
                Left Axis: Spectral Cutoff (kHz) | Right Axis: Speaker drift is evidence only Score
              </p>
            </div>
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-2 relative">
            {/* Resolution Dropdown */}
            <div className="relative">
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="btn-glow-inward-light bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold px-3.5 py-1.5 rounded-full flex items-center gap-1.5 cursor-pointer select-none border border-slate-200/90 shadow-2xs"
              >
                <span className="relative z-10">{resolution}</span>
                <ChevronDown
                  className={`w-3.5 h-3.5 transition-transform duration-200 relative z-10 ${
                    isDropdownOpen ? "rotate-180" : ""
                  }`}
                />
              </button>

              {isDropdownOpen && (
                <div className="absolute right-0 mt-1.5 w-32 bg-white border border-slate-200 rounded-2xl shadow-xl py-1.5 z-40 text-xs font-medium text-slate-700 animate-in fade-in zoom-in-95 duration-150">
                  {(["Full Clip", "F0 Zoom", "Spectral FFT"] as const).map((r) => (
                    <button
                      key={r}
                      onClick={() => {
                        setResolution(r);
                        setIsDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3.5 py-1.5 hover:bg-slate-50 transition-colors ${
                        resolution === r ? "text-slate-900 font-bold bg-slate-50" : ""
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Export TSV Button */}
            <button
              onClick={handleExport}
              className="btn-glow-inward-light w-8 h-8 rounded-full bg-white hover:bg-slate-50 flex items-center justify-center text-slate-600 cursor-pointer border border-slate-200/90 shadow-2xs"
              title="Export NSA TSV Predictions"
              aria-label="Export Predictions"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-600 relative z-10" />
              ) : (
                <Download className="w-3.5 h-3.5 stroke-[1.75] relative z-10" />
              )}
            </button>
          </div>
        </div>

        {/* Main Metric: Synthetic Probability Display */}
        <div className="mt-4 mb-2 flex items-baseline justify-between flex-wrap gap-2">
          <div>
            <div className="text-slate-900 text-3xl sm:text-4xl lg:text-[40px] tracking-tight leading-tight font-sans">
              <DottedNumber
                value={`p=${currentAudio.overallScore.toFixed(3)}`}
                subduedDecimals
                subduedClassName="text-slate-400 font-normal"
                currencyClassName="text-slate-500 mr-1 font-mono text-xl"
              />
            </div>
            <p
              className={`text-xs font-semibold mt-0.5 flex items-center gap-1.5 font-sans ${
                isSynthetic ? "text-rose-600" : "text-emerald-600"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isSynthetic ? "bg-rose-500 animate-pulse" : "bg-emerald-500"
                }`}
              />
              {isSynthetic
                ? "Verdict from the shipped rule: synthetic"
                : "Verdict from the shipped rule: real"}
            </p>
          </div>

          <div className="flex items-center gap-2.5 text-xs font-sans">
            <span className="flex items-center gap-1.5 text-[#005493] bg-[#005493]/10 px-3 py-1 rounded-full border border-[#005493]/20 font-medium">
              <span className="w-2 h-2 rounded-full bg-[#005493]" />
              No per-frame timeline: the pipeline scores whole clips
            </span>
            <span className="flex items-center gap-1.5 text-[#c37530] bg-[#c37530]/10 px-3 py-1 rounded-full border border-[#c37530]/20 font-medium">
              <span className="w-2 h-2 rounded-full bg-[#c37530]" />
              Speaker drift is evidence only
            </span>
          </div>
        </div>
      </div>

      {/* Chart Canvas Area: Exactly using requested LineChart compound components on pure light background */}
      <div className="relative pt-6 pb-1 w-full mt-auto">
        <LineChart data={chartData} margin={{ top: 8, right: 56, bottom: 40, left: 56 }}>
          <Grid horizontal />
          <Line dataKey="desktop" yAxisId="left" stroke="#005493" />
          <Line dataKey="mobile" yAxisId="right" stroke="#c37530" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <XAxis />
          <ChartTooltip />
        </LineChart>
      </div>
    </div>
  );
}

// Re-export as OverviewChart
export { ForensicChart as OverviewChart };
