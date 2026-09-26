"use client";

import React, { useState, useRef } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Upload,
  Mic,
  MicOff,
  Radio,
  FileAudio,
  Activity,
  Layers,
  Square,
} from "lucide-react";
import { useForensic } from "@/context/ForensicContext";

export function AudioAnalyzer() {
  const {
    currentAudio,
    isPlaying,
    currentTime,
    duration,
    togglePlay,
    seek,
    frequencyBars,
    isAnalyzing,
    analysisProgress,
    activeDetectorStage,
    loadAudioFile,
    loadPreset,
    recordVoiceSample,
    presets,
  } = useForensic();

  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<NodeJS.Timeout | null>(null);
  const secondsCountRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Formatting helper
  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 100);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms
      .toString()
      .padStart(2, "0")}`;
  };

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Handle Drag & Drop
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await loadAudioFile(file);
    }
  };

  const handleFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await loadAudioFile(file);
    }
  };

  // Live Microphone Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        await recordVoiceSample(audioBlob, secondsCountRef.current || 3.0);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordSeconds(0);
      secondsCountRef.current = 0;

      recordTimerRef.current = setInterval(() => {
        secondsCountRef.current += 1;
        setRecordSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.warn("Microphone access unavailable or denied, simulating capture:", err);
      setIsRecording(true);
      setRecordSeconds(0);
      secondsCountRef.current = 0;
      recordTimerRef.current = setInterval(() => {
        secondsCountRef.current += 1;
        setRecordSeconds((prev) => {
          if (prev >= 3) {
            stopRecording();
            return 3;
          }
          return prev + 1;
        });
      }, 1000);
    }
  };

  const stopRecording = () => {
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    } else {
      const fakeBlob = new Blob(["RIFF...WAVE"], { type: "audio/wav" });
      recordVoiceSample(fakeBlob, secondsCountRef.current || 3.0);
    }
    setIsRecording(false);
  };

  const isSynthetic = currentAudio.decision === "SYNTHETIC";

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`bg-white/95 backdrop-blur-md rounded-[28px] p-6 sm:p-7 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border transition-all duration-300 relative select-none overflow-hidden ${
        isDragging
          ? "border-[#005493] ring-4 ring-[#005493]/15 bg-blue-50/30"
          : "border-slate-200/90 hover:shadow-md"
      }`}
    >
      {/* Top Header: Analyte Metadata & Presets */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-5 border-b border-slate-100 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-[#005493]/10 border border-[#005493]/20 flex items-center justify-center text-[#005493] shadow-sm">
            <Radio className="w-5 h-5 text-[#005493] animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2 min-h-[26px]">
              <h3 className="text-slate-900 font-semibold text-base tracking-tight font-sans truncate max-w-[300px] sm:max-w-md">
                {currentAudio.title}
              </h3>
              <span
                className={`text-[10px] px-2.5 py-0.5 rounded-full font-semibold border font-sans shrink-0 ${
                  isSynthetic
                    ? "bg-rose-50 text-rose-700 border-rose-200"
                    : "bg-emerald-50 text-emerald-700 border-emerald-200"
                }`}
              >
                {isSynthetic ? "Synthetic" : "Authentic"}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-sans mt-0.5 flex items-center gap-2 flex-wrap min-h-[18px]">
              <span>{currentAudio.filename}</span>
              <span>•</span>
              <span>{currentAudio.sampleRate} Hz</span>
              <span>•</span>
              <span>{currentAudio.size}</span>
            </p>
          </div>
        </div>

        {/* Preset Selector Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {presets.map((p, idx) => (
            <button
              key={p.id}
              onClick={() => loadPreset(p.id)}
              className={`text-xs px-3 py-1.5 rounded-full transition-all cursor-pointer font-medium font-sans ${
                currentAudio.id === p.id
                  ? "btn-glow-inward-nsa bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white shadow-sm font-semibold"
                  : "btn-glow-inward-light bg-white text-slate-600 hover:bg-slate-50 border border-slate-200/90 shadow-2xs"
              }`}
            >
              <span className="relative z-10">Sample {idx + 1}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Analysis Status Banner (Permanently rendered at full size to eliminate layout shift) */}
      <div className="my-4 py-2.5 px-4 rounded-2xl bg-slate-50 border border-slate-200/70 flex items-center justify-between text-xs relative z-10">
        <div className="flex items-center gap-2.5">
          <span
            className={`w-2 h-2 rounded-full ${
              isAnalyzing ? "bg-[#c37530] animate-ping" : "bg-emerald-500"
            }`}
          />
          <span className="font-sans text-slate-700 font-medium">
            {isAnalyzing ? activeDetectorStage : "Forensic Telemetry: All 8 Modalities Synchronized"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-24 bg-slate-200 h-1.5 rounded-full overflow-hidden hidden sm:block">
            <div
              className={`h-full transition-all duration-300 ${
                isAnalyzing
                  ? "bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530]"
                  : "bg-emerald-500"
              }`}
              style={{ width: `${isAnalyzing ? analysisProgress : 100}%` }}
            />
          </div>
          <span className="font-sans text-slate-500 text-[11px] font-semibold min-w-[28px] text-right">
            {isAnalyzing ? `${analysisProgress}%` : "100%"}
          </span>
        </div>
      </div>

      {/* Pure Light Voice Bar & High-Resolution Acoustic Waveform Well */}
      <div className="my-5 relative z-10">
        <div className="h-36 sm:h-40 bg-slate-50/90 rounded-2xl p-4 sm:p-5 flex flex-col justify-between border border-slate-200/80 shadow-[inset_0_2px_4px_rgba(0,0,0,0.02)] relative overflow-hidden group select-none">
          {/* Subtle light grid background */}
          <div
            className="absolute inset-0 opacity-40 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px)",
              backgroundSize: "24px 24px",
            }}
          />

          {/* Time & Playhead Position Display */}
          <div className="flex items-center justify-between text-[11px] font-sans text-slate-500 relative z-10">
            <span className="flex items-center gap-1.5 text-[#005493] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#c37530] animate-pulse" />
              Acoustic Intercept Waveform
            </span>
            <span className="text-slate-800 font-semibold font-mono tracking-wider">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          {/* Centered High-Resolution Audio Soundwave (Click or drag to scrub anywhere) */}
          <div
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const clickX = e.clientX - rect.left;
              const newTime = (clickX / rect.width) * duration;
              seek(newTime);
            }}
            className="flex items-center justify-between gap-[2px] sm:gap-[2.5px] h-20 sm:h-24 relative z-10 my-auto cursor-pointer group/wave"
          >
            {/* Center zero-axis guideline */}
            <div className="absolute left-0 right-0 h-[1px] bg-slate-200 pointer-events-none" />

            {/* Playhead Vertical Needle */}
            <div
              className="absolute top-0 bottom-0 w-[1.5px] bg-[#005493] pointer-events-none z-20 transition-all duration-75 shadow-xs"
              style={{ left: `${progressPercent}%` }}
            >
              <div className="w-2.5 h-2.5 -ml-[4.5px] -mt-1 rounded-full bg-[#005493] ring-2 ring-white shadow-xs" />
            </div>

            {frequencyBars.map((val, idx) => {
              const barProgress = (idx / frequencyBars.length) * 100;
              const isPastPlayhead = barProgress <= progressPercent;

              return (
                <div
                  key={idx}
                  className="flex-1 max-w-[3px] rounded-full transition-all duration-150"
                  style={{
                    height: `${Math.max(6, val * 94)}%`,
                    background: isPastPlayhead
                      ? "linear-gradient(to top, #005493, #c37530)"
                      : "#cbd5e1",
                    opacity: isPastPlayhead ? 1.0 : 0.6,
                  }}
                />
              );
            })}
          </div>

          {/* Interactive Light Scrub Track */}
          <div
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const clickX = e.clientX - rect.left;
              const newTime = (clickX / rect.width) * duration;
              seek(newTime);
            }}
            className="relative h-1.5 bg-slate-200/90 rounded-full cursor-pointer overflow-hidden z-20"
          >
            <div
              className="h-full bg-gradient-to-r from-[#005493] to-[#c37530] transition-all duration-75"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Control Bar: Audio Playback, File Upload, Microphone Capture */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 relative z-10">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          {/* Play/Pause Button */}
          <button
            onClick={togglePlay}
            className="btn-glow-inward-nsa w-11 h-11 rounded-full bg-gradient-to-r from-[#005493] via-[#003d73] to-[#c37530] text-white flex items-center justify-center cursor-pointer shadow-md shadow-[#005493]/25 active:scale-95 transition-transform"
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 fill-white stroke-none relative z-10" />
            ) : (
              <Play className="w-4 h-4 fill-white stroke-none ml-0.5 relative z-10" />
            )}
          </button>

          {/* Reset */}
          <button
            onClick={() => seek(0)}
            className="btn-glow-inward-light w-9 h-9 rounded-full bg-white hover:bg-slate-50 text-slate-700 flex items-center justify-center cursor-pointer transition-colors active:scale-95 border border-slate-200/90 shadow-2xs"
            title="Reset to 00:00"
          >
            <RotateCcw className="w-3.5 h-3.5 relative z-10" />
          </button>

          <div className="text-xs font-semibold text-slate-600 ml-2 hidden sm:block font-mono min-w-[48px]">
            {formatTime(currentTime)}
          </div>
        </div>

        {/* Center: Cryptographic Hash */}
        <div className="hidden md:flex items-center gap-1.5 bg-slate-100 px-3.5 py-1.5 rounded-full text-[11px] font-sans text-slate-600 border border-slate-200/60">
          <span className="font-semibold text-slate-800">SHA-256:</span>
          <span>{currentAudio.sha256.slice(0, 14)}...</span>
        </div>

        {/* Right Actions: File Upload & Live Mic Capture */}
        <div className="flex items-center gap-2.5">
          {/* File Upload Input & Button */}
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
            onChange={handleFileInputChange}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn-glow-inward-light bg-white hover:bg-slate-50 text-slate-800 text-xs font-semibold px-4 py-2.5 rounded-full flex items-center gap-2 cursor-pointer transition-all active:scale-95 border border-slate-200/90 shadow-2xs"
          >
            <Upload className="w-3.5 h-3.5 relative z-10 text-slate-700" />
            <span className="relative z-10">Upload Audio</span>
          </button>

          {/* Live Mic Capture Button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`btn-glow-inward-light text-xs font-semibold px-4 py-2.5 rounded-full flex items-center justify-center gap-2 cursor-pointer transition-all active:scale-95 border select-none min-w-[136px] ${
              isRecording
                ? "bg-rose-50 border-rose-300 text-rose-700 hover:bg-rose-100 shadow-xs"
                : "bg-white hover:bg-slate-50 text-slate-800 border-slate-200/90 shadow-2xs"
            }`}
          >
            {isRecording ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-600"></span>
                </span>
                <span className="relative z-10 text-rose-700">
                  Recording ({recordSeconds}s)
                </span>
                <Square className="w-2.5 h-2.5 fill-rose-600 text-rose-600 relative z-10 ml-0.5" />
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-rose-500 relative z-10 shadow-xs" />
                <Mic className="w-3.5 h-3.5 text-slate-700 relative z-10" />
                <span className="relative z-10">Record Voice</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
