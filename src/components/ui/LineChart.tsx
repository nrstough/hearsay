"use client";

import React, { createContext, useContext, useState, useMemo, useRef } from "react";

interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

interface LineChartContextValue {
  data: any[];
  margin: Margin;
  viewBoxWidth: number;
  viewBoxHeight: number;
  plotLeft: number;
  plotRight: number;
  plotTop: number;
  plotBottom: number;
  plotWidth: number;
  plotHeight: number;
  hoveredIndex: number | null;
  setHoveredIndex: (idx: number | null) => void;
  hoverX: number | null;
  yScales: {
    left: { min: number; max: number };
    right: { min: number; max: number };
  };
}

const LineChartContext = createContext<LineChartContextValue | null>(null);

function useLineChart() {
  const ctx = useContext(LineChartContext);
  if (!ctx) {
    throw new Error("LineChart compound components must be within a <LineChart />");
  }
  return ctx;
}

interface LineChartProps {
  data: any[];
  margin?: { top?: number; right?: number; bottom?: number; left?: number };
  children: React.ReactNode;
  className?: string;
}

export function LineChart({
  data,
  margin: customMargin,
  children,
  className = "",
}: LineChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const viewBoxWidth = 1000;
  const viewBoxHeight = 250;

  const margin: Margin = useMemo(
    () => ({
      top: customMargin?.top ?? 8,
      right: customMargin?.right ?? 56,
      bottom: customMargin?.bottom ?? 40,
      left: customMargin?.left ?? 56,
    }),
    [customMargin]
  );

  const plotLeft = margin.left;
  const plotRight = viewBoxWidth - margin.right;
  const plotTop = margin.top;
  const plotBottom = viewBoxHeight - margin.bottom;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;

  // Calculate scales for left (desktop: kHz, 0-25) and right (mobile: probability, 0-1)
  const yScales = useMemo(() => {
    let maxDesktop = 25;
    let minDesktop = 0;
    data.forEach((d) => {
      if (typeof d.desktop === "number") {
        if (d.desktop > maxDesktop) maxDesktop = Math.ceil(d.desktop * 1.1);
      }
    });

    return {
      left: { min: minDesktop, max: maxDesktop },
      right: { min: 0.0, max: 1.0 },
    };
  }, [data]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || data.length === 0) return;
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const relativeX = (clientX / rect.width) * viewBoxWidth;

    if (relativeX < plotLeft || relativeX > plotRight) {
      setHoveredIndex(null);
      setHoverX(null);
      return;
    }

    const step = plotWidth / (data.length - 1 || 1);
    const index = Math.round((relativeX - plotLeft) / step);
    const clampedIndex = Math.max(0, Math.min(data.length - 1, index));

    setHoveredIndex(clampedIndex);
    setHoverX(plotLeft + clampedIndex * step);
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setHoverX(null);
  };

  return (
    <LineChartContext.Provider
      value={{
        data,
        margin,
        viewBoxWidth,
        viewBoxHeight,
        plotLeft,
        plotRight,
        plotTop,
        plotBottom,
        plotWidth,
        plotHeight,
        hoveredIndex,
        setHoveredIndex,
        hoverX,
        yScales,
      }}
    >
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className={`relative w-full select-none bg-white rounded-2xl ${className}`}
        style={{ touchAction: "none" }}
      >
        <svg
          viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
          className="w-full h-auto overflow-visible block"
          style={{ maxHeight: "250px" }}
        >
          {children}
        </svg>
      </div>
    </LineChartContext.Provider>
  );
}

export function Grid({ horizontal = true }: { horizontal?: boolean }) {
  const { plotLeft, plotRight, plotTop, plotBottom, plotHeight } = useLineChart();

  if (!horizontal) return null;

  const lines = 4;
  return (
    <g className="grid-lines" opacity="0.6">
      {Array.from({ length: lines + 1 }).map((_, i) => {
        const y = plotTop + (plotHeight / lines) * i;
        return (
          <line
            key={i}
            x1={plotLeft}
            y1={y}
            x2={plotRight}
            y2={y}
            stroke="#e2e8f0"
            strokeDasharray="4 4"
            strokeWidth="1"
          />
        );
      })}
    </g>
  );
}

interface LineProps {
  dataKey: string;
  yAxisId?: "left" | "right";
  stroke?: string;
  strokeWidth?: number;
}

export function Line({
  dataKey,
  yAxisId = "left",
  stroke = "#0284c7",
  strokeWidth = 2.5,
}: LineProps) {
  const { data, plotLeft, plotBottom, plotWidth, plotHeight, yScales, hoveredIndex } =
    useLineChart();

  if (!data || data.length === 0) return null;

  const scale = yScales[yAxisId] || { min: 0, max: 1 };
  const step = plotWidth / (data.length - 1 || 1);

  // Compute points
  const points = data.map((d, i) => {
    const rawVal = d[dataKey] ?? 0;
    const norm = (rawVal - scale.min) / (scale.max - scale.min || 1);
    const x = plotLeft + i * step;
    const y = plotBottom - norm * plotHeight;
    return { x, y, val: rawVal };
  });

  // Construct smooth cubic bezier path
  let pathD = `M ${points[0].x} ${points[0].y}`;
  for (let i = 0; i < points.length - 1; i++) {
    const curr = points[i];
    const next = points[i + 1];
    const cpX1 = curr.x + (next.x - curr.x) / 3;
    const cpX2 = curr.x + ((next.x - curr.x) * 2) / 3;
    pathD += ` C ${cpX1} ${curr.y}, ${cpX2} ${next.y}, ${next.x} ${next.y}`;
  }

  // Construct area fill under line
  const areaD = `${pathD} L ${points[points.length - 1].x} ${plotBottom} L ${points[0].x} ${plotBottom} Z`;
  const isRightAxis = yAxisId === "right";
  const strokeColor = stroke || (isRightAxis ? "#c37530" : "#005493");

  return (
    <g className={`line-group-${dataKey}`}>
      <defs>
        <linearGradient id={`gradient-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
          <stop
            offset="0%"
            stopColor={strokeColor}
            stopOpacity={isRightAxis ? "0.10" : "0.14"}
          />
          <stop
            offset="100%"
            stopColor={strokeColor}
            stopOpacity="0.0"
          />
        </linearGradient>
      </defs>

      {/* Area Fill */}
      <path d={areaD} fill={`url(#gradient-${dataKey})`} pointerEvents="none" />

      {/* Main Trajectory Line */}
      <path
        d={pathD}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Active Data Points */}
      {points.map((pt, i) => {
        const isHovered = hoveredIndex === i;
        return (
          <circle
            key={i}
            cx={pt.x}
            cy={pt.y}
            r={isHovered ? 5.5 : 3.5}
            fill="#ffffff"
            stroke={strokeColor}
            strokeWidth={isHovered ? 2.5 : 2}
            className="transition-all duration-150"
          />
        );
      })}
    </g>
  );
}

interface YAxisProps {
  yAxisId?: "left" | "right";
  orientation?: "left" | "right";
}

export function YAxis({ yAxisId = "left", orientation = "left" }: YAxisProps) {
  const { plotLeft, plotRight, plotTop, plotBottom, plotHeight, yScales } = useLineChart();

  const scale = yScales[yAxisId] || { min: 0, max: 1 };
  const ticks = yAxisId === "left" ? [24, 18, 12, 6, 0] : [1.0, 0.75, 0.5, 0.25, 0.0];
  const isRight = orientation === "right";
  const x = isRight ? plotRight + 12 : plotLeft - 12;

  return (
    <g className={`y-axis-${yAxisId}`} fontSize="11" fill="#64748b" textAnchor={isRight ? "start" : "end"}>
      {ticks.map((t, i) => {
        const norm = (t - scale.min) / (scale.max - scale.min || 1);
        const y = plotBottom - norm * plotHeight + 4;
        const label = yAxisId === "left" ? `${t} kHz` : `${(t * 100).toFixed(0)}%`;
        return (
          <text key={i} x={x} y={y} className="font-mono text-[10px] select-none fill-slate-500">
            {label}
          </text>
        );
      })}
    </g>
  );
}

export function XAxis() {
  const { data, plotLeft, plotBottom, plotWidth, hoveredIndex } = useLineChart();

  if (!data || data.length === 0) return null;
  const step = plotWidth / (data.length - 1 || 1);

  return (
    <g className="x-axis" fontSize="11" fill="#64748b" textAnchor="middle">
      {data.map((d, i) => {
        const x = plotLeft + i * step;
        const y = plotBottom + 22;
        const isHovered = hoveredIndex === i;
        return (
          <text
            key={i}
            x={x}
            y={y}
            className={`font-mono text-[11px] transition-colors ${
              isHovered ? "fill-sky-700 font-semibold" : "fill-slate-400"
            }`}
          >
            {d.time || `${i}s`}
          </text>
        );
      })}
    </g>
  );
}

export function ChartTooltip() {
  const { data, hoveredIndex, hoverX, plotTop, plotBottom, viewBoxWidth } = useLineChart();

  if (hoveredIndex === null || !data[hoveredIndex] || hoverX === null) return null;

  const current = data[hoveredIndex];
  const desktopVal = current.desktop !== undefined ? `${current.desktop.toFixed(1)} kHz` : "N/A";
  const mobileVal =
    current.mobile !== undefined ? `${(current.mobile * 100).toFixed(1)}% Anomaly` : "N/A";
  const phaseDisc =
    current.phaseDiscontinuity !== undefined
      ? `${(current.phaseDiscontinuity * 100).toFixed(1)}%`
      : "0.0%";

  // Determine horizontal alignment to prevent edge overflowing
  const isFarRight = hoverX > viewBoxWidth * 0.7;
  const tooltipX = isFarRight ? hoverX - 215 : hoverX + 16;
  const tooltipY = Math.max(10, plotTop + 10);

  return (
    <g className="chart-tooltip-group" pointerEvents="none">
      {/* Vertical crosshair tracker */}
      <line
        x1={hoverX}
        y1={plotTop}
        x2={hoverX}
        y2={plotBottom}
        stroke="#0284c7"
        strokeWidth="1.5"
        strokeDasharray="3 3"
        opacity="0.7"
      />

      {/* Floating Pure Light Tooltip Card (Zero Dark Theme) */}
      <foreignObject x={tooltipX} y={tooltipY} width="205" height="135" className="overflow-visible">
        <div className="bg-white/95 text-slate-900 backdrop-blur-xl border border-slate-200/90 rounded-2xl p-3 shadow-xl ring-1 ring-slate-900/5 animate-in fade-in zoom-in-95 duration-100">
          <div className="flex items-center justify-between pb-1.5 border-b border-slate-100 mb-2">
            <span className="text-[11px] font-mono font-semibold text-sky-800">
              TIME: {current.time || `${hoveredIndex}s`}
            </span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200 uppercase font-mono font-semibold">
              FRAME #{hoveredIndex + 1}
            </span>
          </div>

          <div className="space-y-1.5 text-xs">
            {/* Left YAxis: Desktop (Vocoder cutoff kHz) */}
            <div className="flex items-center justify-between">
              <span className="text-slate-600 flex items-center gap-1.5 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-sky-600" />
                Vocoder Nyquist
              </span>
              <span className="font-mono font-bold text-slate-900 text-[11px]">{desktopVal}</span>
            </div>

            {/* Right YAxis: Mobile (Drift Anomaly Score) */}
            <div className="flex items-center justify-between">
              <span className="text-slate-600 flex items-center gap-1.5 text-[11px]">
                <span className="w-2 h-2 rounded-full bg-[#0e3b9f]" />
                Embedding Drift
              </span>
              <span className="font-mono font-bold text-[#0e3b9f] text-[11px]">{mobileVal}</span>
            </div>

            {/* Phase Discontinuity metric */}
            <div className="flex items-center justify-between pt-1 border-t border-slate-100">
              <span className="text-slate-500 text-[10px]">Phase Jump</span>
              <span className="font-mono font-medium text-slate-700 text-[10px]">{phaseDisc}</span>
            </div>
          </div>
        </div>
      </foreignObject>
    </g>
  );
}
