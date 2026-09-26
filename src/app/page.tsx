"use client";

import React, { useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Navigation } from "@/components/Navigation";
import { HeroSection } from "@/components/HeroSection";
import { AudioAnalyzer } from "@/components/AudioAnalyzer";
import { ForensicChart } from "@/components/ForensicChart";
import { RecentForensics } from "@/components/RecentForensics";
import { ModalityGrid } from "@/components/ModalityGrid";
import { BenchmarkMetrics } from "@/components/BenchmarkMetrics";
import { HearsayCopilot } from "@/components/HearsayCopilot";
import { SpectrogramView } from "@/components/views/SpectrogramView";
import { ModalitiesView } from "@/components/views/ModalitiesView";
import { BatchQueueView } from "@/components/views/BatchQueueView";
import { EvidenceDossierView } from "@/components/views/EvidenceDossierView";

function DashboardContent() {
  const searchParams = useSearchParams();
  const initialTab = searchParams.get("tab") || "dashboard";
  const [activeTab, setActiveTab] = useState<string>(initialTab);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("tab", tab);
      window.history.replaceState(null, "", url.toString());
    }
  };

  return (
    <main className="min-h-screen bg-[#f0f4f9] text-slate-900 flex flex-col antialiased relative overflow-x-hidden">
      {/* Upper Canvas: Master NSA Navy & Eagle Gold Glow Layer with curvy gradient mask & diffused blur */}
      <div className="absolute -top-20 left-0 right-0 h-[720px] sm:h-[780px] lg:h-[840px] pointer-events-none z-0">
        <div className="hero-curved-banner absolute inset-0 w-full h-full bg-[#001730]">
          <div className="hero-gradient absolute -top-12 inset-x-0 bottom-0 h-[calc(100%+48px)] w-full" />
        </div>
        <div className="hero-curved-blur-halo" />
      </div>

      {/* Top Navigation */}
      <div className="relative z-20 w-full">
        <Navigation
          activeTab={activeTab}
          setActiveTab={handleTabChange}
        />
      </div>

      {/* Main Content Area */}
      <div className="w-full flex-1 pt-1 sm:pt-2 pb-14 relative z-10">

        {/* Tab: Forensic Cockpit View */}
        {activeTab === "dashboard" && (
          <>
            {/* Hero Section */}
            <div className="relative z-10 w-full mb-3">
              <HeroSection />
            </div>

            {/* Voice Bar & Live Audio Waveform Analyzer (Pure Light) */}
            <div className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 mb-6 relative z-10">
              <AudioAnalyzer />
            </div>

            {/* The Master Bento Grid */}
            <div className="max-w-[1440px] mx-auto px-6 sm:px-10 lg:px-14 space-y-5 lg:space-y-6 relative z-10">
              {/* Row 1: Acoustic Telemetry Dual-Axis Chart (Left 58%) & 2x2 Modality Grid (Right 42%) */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-6 items-stretch">
                <div className="lg:col-span-7 flex flex-col">
                  <ForensicChart />
                </div>
                <div className="lg:col-span-5 flex flex-col">
                  <ModalityGrid />
                </div>
              </div>

              {/* Row 2: Evaluation Queue (Left 58%) & Benchmark minDCF Metrics (Right 42%) */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-6 items-stretch">
                <div className="lg:col-span-7 flex flex-col">
                  <RecentForensics />
                </div>
                <div className="lg:col-span-5 flex flex-col">
                  <BenchmarkMetrics />
                </div>
              </div>
            </div>
          </>
        )}

        {/* Tab: Spectrogram & Playground View */}
        {(activeTab === "spectrogram" || activeTab === "playground") && <SpectrogramView />}

        {/* Tab: 8 Modalities & Voices View */}
        {(activeTab === "modalities" || activeTab === "voices") && <ModalitiesView />}

        {/* Tab: NSA Batch Queue & Projects View */}
        {(activeTab === "batch" || activeTab === "projects") && <BatchQueueView />}

        {/* Tab: Evidence Dossier View */}
        {activeTab === "dossier" && <EvidenceDossierView />}
      </div>

      {/* Floating Acoustic Intelligence Copilot (Pure Light) */}
      <HearsayCopilot />
    </main>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#f0f4f9]" />}>
      <DashboardContent />
    </Suspense>
  );
}
