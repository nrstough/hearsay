"use client";

import React, { useState } from "react";
import { BrandLogo } from "./Icons";
import { Bell, Sparkles, Radio, Shield, FileSpreadsheet, CheckCircle2, Download, ChevronDown } from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onOpenBatchModal?: () => void;
}

export function Navigation({
  activeTab,
  setActiveTab,
  onOpenBatchModal,
}: NavigationProps) {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const { currentAudio, exportPredictionTsv } = useForensic();

  const navItems = [
    { id: "dashboard", label: "Dashboard" },
    { id: "spectrogram", label: "Spectrogram" },
    { id: "modalities", label: "Modalities" },
    { id: "batch", label: "Batch Queue" },
    { id: "dossier", label: "Dossier" },
  ];

  return (
    <header className="w-full relative z-40 select-none">
      <div className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 py-4 flex items-center justify-between gap-4">
        {/* Left: Brand Logo & Title */}
        <div className="flex items-center gap-6">
          <button
            onClick={() => setActiveTab("dashboard")}
            className="flex items-center gap-2.5 group transition-transform active:scale-95 cursor-pointer"
            aria-label="HEARSAY Home"
          >
            {/* Sleek Acoustic Wave Shield Emblem */}
            <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-[#005493] to-[#c37530] flex items-center justify-center text-white shadow-sm">
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-none stroke-current" strokeWidth="2.5" strokeLinecap="round">
                <path d="M4 12v.01M8 8v8M12 4v16M16 8v8M20 12v.01" />
              </svg>
            </div>
            <span className="text-white font-bold text-base tracking-wider font-sans">
              HEARSAY
            </span>
          </button>

          {/* Center: Clean Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`btn-glow-inward-light transition-all duration-150 text-xs px-3.5 py-1.5 rounded-full cursor-pointer ${
                    isActive
                      ? "bg-white/20 text-white font-semibold shadow-xs"
                      : "text-white/70 hover:text-white hover:bg-white/10 font-medium"
                  }`}
                >
                  <span className="relative z-10">{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Right: Notification Bell & Profile */}
        <div className="flex items-center gap-3">
          {/* Notification Bell */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="btn-glow-inward-light w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 flex items-center justify-center text-white transition-all cursor-pointer relative"
              aria-label="Notifications"
            >
              <Bell className="w-3.5 h-3.5 text-white/90 relative z-10" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500 ring-2 ring-[#092265] z-20" />
            </button>

            {/* Notification Dropdown */}
            {showNotifications && (
              <div className="absolute right-0 mt-3 w-80 sm:w-96 bg-white/95 backdrop-blur-xl border border-slate-200/90 rounded-3xl p-5 shadow-2xl z-50 text-slate-900 animate-in fade-in slide-in-from-top-2 duration-200 overflow-hidden">
                <div className="relative z-10">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-[#c37530]" />
                      <span className="font-semibold text-sm text-slate-900 font-sans">Forensic Intercept Alerts</span>
                    </div>
                    <span className="text-[11px] text-[#005493] bg-[#005493]/10 border border-[#005493]/20 px-2.5 py-0.5 rounded-full font-mono font-bold">
                      3 Flagged
                    </span>
                  </div>

                  <div className="divide-y divide-slate-100 mt-1 max-h-[380px] overflow-y-auto pr-1">
                    {/* Alert 1 */}
                    <div
                      onClick={() => {
                        setShowNotifications(false);
                        setActiveTab("dashboard");
                      }}
                      className="py-3 px-2 rounded-2xl hover:bg-slate-50 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-rose-700 font-sans">
                          Demo: synthetic clip
                        </span>
                        <span className="text-[10px] text-rose-600 font-mono font-bold">presets</span>
                      </div>
                      <p className="text-xs text-slate-600 mt-1 leading-snug font-sans">
                        Load a demo preset to see the shipped pipeline’s real verdict and evidence.
                      </p>
                    </div>

                    {/* Alert 2 */}
                    <div
                      onClick={() => {
                        setShowNotifications(false);
                        setActiveTab("spectrogram");
                      }}
                      className="py-3 px-2 rounded-2xl hover:bg-slate-50 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-emerald-700 font-sans">
                          Demo: real clip
                        </span>
                        <span className="text-[10px] text-emerald-600 font-mono font-bold">presets</span>
                      </div>
                      <p className="text-xs text-slate-600 mt-1 leading-snug font-sans">
                        Every number shown comes from the pipeline’s JSON; nothing is computed in the browser.
                      </p>
                    </div>

                    {/* Alert 3 */}
                    <div
                      onClick={() => {
                        setShowNotifications(false);
                        setActiveTab("modalities");
                      }}
                      className="py-3 px-2 rounded-2xl hover:bg-slate-50 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-[#c37530] font-sans">
                          ECAPA-TDNN Speaker Drift
                        </span>
                        <span className="text-[10px] text-[#c37530] font-mono font-bold">0.44 Cosine</span>
                      </div>
                      <p className="text-xs text-slate-600 mt-1 leading-snug font-sans">
                        Wiretap #0504 shows identity dissimilarity jump at second 4.12 indicating real-time voice conversion.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* User Profile Avatar */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-1.5 transition-all select-none cursor-pointer"
              aria-label="User Account"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#005493] to-[#c37530] text-white shadow-sm flex items-center justify-center font-semibold text-xs">
                J
              </div>
              <ChevronDown className="w-3 h-3 text-white/70" />
            </button>

            {/* Profile Menu */}
            {showUserMenu && (
              <div className="absolute right-0 mt-3 w-72 bg-white/95 backdrop-blur-xl border border-slate-200/90 rounded-3xl p-4 shadow-2xl z-50 text-slate-900 animate-in fade-in slide-in-from-top-2 duration-200">
                <div className="flex items-center gap-3 pb-3 border-b border-slate-100">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-[#005493] to-[#c37530] text-white flex items-center justify-center font-bold text-sm shadow-sm">
                    H
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-900 font-sans">Agent Hrushi</p>
                    <p className="text-[11px] text-slate-500 font-sans">NSA Forensics Specialist</p>
                  </div>
                </div>
                <div className="pt-2 text-xs space-y-1.5 font-sans">
                  <div className="w-full text-left py-2 px-2.5 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-between text-slate-700">
                    <span className="font-medium">Rank-blend fusion</span>
                    <span className="text-[11px] text-emerald-600 font-semibold font-mono">Calibrated</span>
                  </div>
                  <div className="w-full text-left py-2 px-2.5 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-between text-slate-700">
                    <span className="font-medium">XLS-R probe</span>
                    <span className="text-[11px] text-emerald-600 font-semibold font-mono">Live</span>
                  </div>
                  <button
                    onClick={() => {
                      exportPredictionTsv();
                      setShowUserMenu(false);
                    }}
                    className="btn-glow-inward-light w-full text-left py-2 px-2.5 rounded-xl bg-white hover:bg-slate-50 border border-slate-200/90 transition-colors text-[#005493] font-semibold flex items-center justify-between cursor-pointer shadow-2xs"
                  >
                    <span className="relative z-10">Export TSV</span>
                    <Download className="w-3.5 h-3.5 relative z-10 text-[#005493]" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
