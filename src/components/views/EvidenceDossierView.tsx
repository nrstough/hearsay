"use client";

import React, { useState } from "react";
import { FileText, CheckCircle2, AlertOctagon, Printer } from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function EvidenceDossierView() {
  const { currentAudio, modalities, routingLog } = useForensic();
  const [copied, setCopied] = useState(false);
  const isSynthetic = currentAudio.decision === "SYNTHETIC";

  const handlePrint = () => {
    window.print();
  };

  const handleCopyText = () => {
    const report = `HEARSAY FORENSIC INTELLIGENCE DOSSIER
Target File: ${currentAudio.filename}
SHA-256: ${currentAudio.sha256}
Verdict: ${currentAudio.decision} (probability of synthetic ${currentAudio.overallScore.toFixed(3)})
Fused rank before the probability map: ${currentAudio.fusedRank.toFixed(3)}
Fusion: ${currentAudio.detectedVector}
Routing log: ${routingLog.join(" | ")}
Key Forensic Findings:
- Spectral features (handcrafted model): ${modalities.spectral?.detail}
- Speaker drift (evidence only): ${modalities.speaker?.detail}
- Mains hum, ENF (evidence only): ${modalities.enf?.detail}
- Deep anti-spoofing: ${modalities.deepSpoof?.detail}
- Prosody and speech gate: ${modalities.prosody?.detail}
- Compression traces (evidence only): ${modalities.compression?.detail}
- Container (routing only): ${modalities.container?.detail}
- Splice and discontinuities (evidence only): ${modalities.splice?.detail}`;

    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 space-y-6 select-none relative z-10">
      {/* Header Banner */}
      <div className="bg-white rounded-[28px] p-6 shadow-sm border border-slate-200/90 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-white bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] uppercase tracking-wider px-3 py-1 rounded-full shadow-sm shadow-[#005493]/20">
            Courtroom &amp; Intelligence Briefing
          </span>
          <h2 className="text-xl font-semibold text-slate-900 tracking-tight mt-2 font-sans">
            Forensic Evidence Dossier
          </h2>
          <p className="text-xs text-slate-600 mt-1 font-sans">
            Authoritative, explainable acoustic breakdown answering exactly why this audio clip was classified as synthetic or authentic.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleCopyText}
            className="btn-glow-inward-light bg-white hover:bg-slate-50 text-slate-800 text-xs font-semibold px-4 py-2.5 rounded-full flex items-center gap-2 cursor-pointer border border-slate-200/90 shadow-2xs"
          >
            <FileText className="w-4 h-4 text-slate-700 relative z-10" />
            <span className="relative z-10">{copied ? "Copied to Clipboard!" : "Copy Report Text"}</span>
          </button>

          <button
            onClick={handlePrint}
            className="btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white text-xs font-semibold px-5 py-2.5 rounded-full flex items-center gap-2 cursor-pointer shadow-md"
          >
            <Printer className="w-4 h-4 relative z-10" />
            <span className="relative z-10">Print Dossier</span>
          </button>
        </div>
      </div>

      {/* Official Dossier Document Card */}
      <div className="bg-white rounded-[28px] p-8 shadow-sm border border-slate-200/90 space-y-6">
        {/* Document Header */}
        <div className="pb-6 border-b border-slate-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="font-mono text-xs text-slate-400 font-semibold">
                DOSSIER REF: NSA-HEARSAY-{currentAudio.id.toUpperCase()}
              </span>
            </div>
            <h3 className="text-2xl font-semibold text-slate-900 tracking-tight font-sans">
              Audio Authentication Report: {currentAudio.filename}
            </h3>
            <p className="text-xs font-mono text-slate-500 mt-1">
              SHA-256: {currentAudio.sha256}
            </p>
          </div>

          <div
            className={`px-4 py-2 rounded-2xl flex items-center gap-2 border ${
              isSynthetic
                ? "bg-rose-50 text-rose-800 border-rose-200"
                : "bg-emerald-50 text-emerald-800 border-emerald-200"
            }`}
          >
            {isSynthetic ? (
              <AlertOctagon className="w-5 h-5 text-rose-600" />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
            )}
            <div>
              <span className="text-[10px] font-mono uppercase font-bold block">
                Final Verdict
              </span>
              <span className="text-sm font-semibold font-sans">
                {isSynthetic ? "SYNTHETIC VOICE (FABRICATED)" : "BONA FIDE (AUTHENTIC HUMAN)"}
              </span>
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200/80">
          <h4 className="text-xs font-semibold text-slate-900 uppercase font-mono mb-1.5 tracking-wider">
            Executive Forensic Summary
          </h4>
          <p className="text-sm text-slate-700 leading-relaxed font-sans">
            {`'${currentAudio.filename}': ${currentAudio.decision}, probability of synthetic ${currentAudio.overallScore.toFixed(3)}. ${currentAudio.detectedVector}. ${routingLog.length ? "Routing log: " + routingLog.join(" | ") : ""}`}
          </p>
        </div>

        {/* Modality Findings Table */}
        <div>
          <h4 className="text-xs font-semibold text-slate-900 uppercase font-mono mb-3 tracking-wider">
            Detector findings (ten detectors; eight rubric techniques)
          </h4>
          <div className="divide-y divide-slate-100 border border-slate-200/80 rounded-2xl overflow-hidden">
            {Object.keys(modalities).map((key) => {
              const mod = modalities[key];
              return (
                <div key={key} className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
                  <div className="sm:w-1/3">
                    <span className="font-semibold text-slate-900 block font-sans">{mod.name}</span>
                    <span className="text-[11px] font-mono text-slate-400 font-medium">{mod.category}</span>
                  </div>
                  <div className="sm:w-1/2">
                    <p className="text-slate-600 font-sans">{mod.detail}</p>
                    <span className="text-[10px] font-mono text-[#005493] font-semibold mt-0.5 block">
                      {mod.metric}
                    </span>
                  </div>
                  <div className="sm:w-1/6 text-right">
                    <span
                      className={`font-mono font-bold px-2 py-0.5 rounded text-[11px] border ${
                        mod.anomaly
                          ? "bg-rose-50 text-rose-700 border-rose-200"
                          : "bg-emerald-50 text-emerald-700 border-emerald-200"
                      }`}
                    >
                      {mod.anomaly ? "FLAGGED" : "CLEAN"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
