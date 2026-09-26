"use client";

import React, { createContext, useContext, useState, useMemo } from "react";
import { ArrowUpRight } from "lucide-react";
import { DottedNumber } from "../DottedNumber";

interface ChartMargin {
  top?: number;
  right?: number;
  bottom?: number;
  left?: number;
}

interface ChartContextValue {
  data: any[];
  xDataKey: string;
  margin: { top: number; right: number; bottom: number; left: number };
  viewBoxWidth: number;
  viewBoxHeight: number;
  plotLeft: number;
  plotRight: number;
  plotTop: number;
  plotBottom: number;
  plotWidth: number;
  plotHeight: number;
  barWidth: number;
  barGap: number;
  hoveredIndex: number | null;
  setHoveredIndex: (idx: number | null) => void;
  selectedIndex: number;
  setSelectedIndex: (idx: number) => void;
  maxValue: number;
  primaryDataKey: string;
  setPrimaryDataKey: (key: string) => void;
}

const ChartContext = createContext<ChartContextValue | null>(null);

function useChartContext() {
  const ctx = useContext(ChartContext);
  if (!ctx) {
    throw new Error("Chart compound components must be used inside <BarChart />");
  }
  return ctx;
}

interface BarChartProps {
  margin?: ChartMargin;
  data: any[];
  xDataKey: string;
  barGap?: number;
  children: React.ReactNode;
  className?: string;
  defaultSelectedIndex?: number;
  onSelectIndex?: (index: number) => void;
}

export function BarChart({
  margin: customMargin,
  data,
  xDataKey,
  barGap = 0,
  children,
  className = "",
  defaultSelectedIndex = 6,
  onSelectIndex,
}: BarChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [selectedIndex, setSelectedIndexState] = useState<number>(defaultSelectedIndex);
  const [primaryDataKey, setPrimaryDataKey] = useState<string>("revenue");

  const setSelectedIndex = (idx: number) => {
    setSelectedIndexState(idx);
    if (onSelectIndex) onSelectIndex(idx);
  };

  const viewBoxWidth = 1000;
  const viewBoxHeight = 230;

  const margin = useMemo(() => ({
    top: customMargin?.top ?? 8,
    right: customMargin?.right ?? 8,
    bottom: customMargin?.bottom ?? 40,
    left: customMargin?.left ?? 8,
  }), [customMargin]);

  const plotLeft = margin.left;
  const plotRight = viewBoxWidth - margin.right;
  const plotTop = margin.top;
  const plotBottom = viewBoxHeight - margin.bottom;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;

  const count = data.length || 1;
  const totalGaps = (count - 1) * barGap;
  const barWidth = (plotWidth - totalGaps) / count;

  const maxValue = useMemo(() => {
    let max = 0;
    for (const item of data) {
      const val = Number(item[primaryDataKey] ?? item.revenue ?? 0);
      if (val > max) max = val;
    }
    return max > 0 ? max * 1.15 : 100;
  }, [data, primaryDataKey]);

  const contextValue: ChartContextValue = {
    data,
    xDataKey,
    margin,
    viewBoxWidth,
    viewBoxHeight,
    plotLeft,
    plotRight,
    plotTop,
    plotBottom,
    plotWidth,
    plotHeight,
    barWidth,
    barGap,
    hoveredIndex,
    setHoveredIndex,
    selectedIndex,
    setSelectedIndex,
    maxValue,
    primaryDataKey,
    setPrimaryDataKey,
  };

  return (
    <ChartContext.Provider value={contextValue}>
      <div className={`relative w-full select-none ${className}`}>
        <svg
          viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
          className="w-full h-full overflow-visible"
        >
          <defs>
            <linearGradient id="activeBarGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.65} />
              <stop offset="45%" stopColor="#0ea5e9" stopOpacity={0.3} />
              <stop offset="100%" stopColor="transparent" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          {children}
        </svg>
      </div>
    </ChartContext.Provider>
  );
}

interface LinearGradientProps {
  id: string;
  from: string;
  to?: string;
}

export function LinearGradient({ id, from, to = "transparent" }: LinearGradientProps) {
  return (
    <defs>
      <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.78} />
        <stop offset="35%" stopColor="#0ea5e9" stopOpacity={0.42} />
        <stop offset="70%" stopColor="#0284c7" stopOpacity={0.14} />
        <stop offset="100%" stopColor={to} stopOpacity={0.0} />
      </linearGradient>
    </defs>
  );
}

interface GridProps {
  horizontal?: boolean;
  vertical?: boolean;
}

export function Grid({ horizontal = true }: GridProps) {
  const { plotLeft, plotRight, plotTop, plotHeight } = useChartContext();

  if (!horizontal) return null;

  const steps = [0, 0.25, 0.5, 0.75, 1];

  return (
    <g className="chart-grid pointer-events-none">
      {steps.map((ratio, idx) => {
        const y = plotTop + plotHeight * ratio;
        return (
          <line
            key={idx}
            x1={plotLeft}
            y1={y}
            x2={plotRight}
            y2={y}
            stroke="#e2e8f0"
            strokeWidth={1}
            strokeDasharray={ratio === 1 ? undefined : "3 3"}
            opacity={ratio === 1 ? 0.8 : 0.6}
          />
        );
      })}
    </g>
  );
}

interface BarProps {
  dataKey: string;
  fill: string;
  stroke?: string;
  lineCap?: "butt" | "round" | "square";
}

export function Bar({ dataKey, fill, stroke, lineCap = "butt" }: BarProps) {
  const {
    data,
    plotLeft,
    plotBottom,
    plotHeight,
    barWidth,
    barGap,
    maxValue,
    hoveredIndex,
    setHoveredIndex,
    selectedIndex,
    setSelectedIndex,
    setPrimaryDataKey,
  } = useChartContext();

  React.useEffect(() => {
    setPrimaryDataKey(dataKey);
  }, [dataKey, setPrimaryDataKey]);

  return (
    <g className="chart-bars">
      {data.map((item, idx) => {
        const val = Number(item[dataKey] ?? 0);
        const bHeight = Math.max(4, (val / maxValue) * plotHeight);
        const x = plotLeft + idx * (barWidth + barGap);
        const y = plotBottom - bHeight;

        const isHovered = hoveredIndex === idx;
        const isSelected = selectedIndex === idx;
        const isActive = isHovered || isSelected;

        return (
          <g
            key={idx}
            className="cursor-pointer group"
            onMouseEnter={() => setHoveredIndex(idx)}
            onMouseLeave={() => setHoveredIndex(null)}
            onClick={() => setSelectedIndex(idx)}
          >
            {/* Bar Body with Gradient Fill */}
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={bHeight}
              fill={fill}
              opacity={isActive ? 1 : 0.78}
              className="transition-opacity duration-200"
            />

            {/* Top Border Stroke of Bar */}
            {stroke && (
              <line
                x1={x}
                y1={y}
                x2={x + barWidth}
                y2={y}
                stroke={stroke}
                strokeWidth={isActive ? 3 : 2}
                strokeLinecap={lineCap}
                className="transition-all duration-200"
              />
            )}

            {/* Active Vertical Glow overlay */}
            {isActive && (
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={bHeight}
                fill="url(#activeBarGlow)"
                opacity={0.35}
                pointerEvents="none"
              />
            )}
          </g>
        );
      })}
    </g>
  );
}

export function BarXAxis() {
  const { data, xDataKey, plotLeft, plotBottom, barWidth, barGap, selectedIndex, hoveredIndex, setSelectedIndex } = useChartContext();

  return (
    <g className="chart-x-axis select-none">
      {data.map((item, idx) => {
        const x = plotLeft + idx * (barWidth + barGap) + barWidth / 2;
        const isSelected = (hoveredIndex ?? selectedIndex) === idx;

        return (
          <text
            key={idx}
            x={x}
            y={plotBottom + 24}
            textAnchor="middle"
            onClick={() => setSelectedIndex(idx)}
            className={`text-[11px] sm:text-[12px] cursor-pointer transition-colors duration-200 ${
              isSelected ? "fill-[#020b2d] font-bold" : "fill-zinc-400 hover:fill-zinc-700 font-medium"
            }`}
          >
            {item[xDataKey]}
          </text>
        );
      })}
    </g>
  );
}

interface BarLineIndicatorProps {
  data: any[];
  valueKey: string;
  xKey: string;
  stroke?: string;
}

export function BarLineIndicator({ valueKey, stroke = "var(--chart-3)" }: BarLineIndicatorProps) {
  const { data, plotLeft, plotBottom, plotHeight, barWidth, barGap, maxValue, selectedIndex, hoveredIndex } = useChartContext();

  const activeIdx = hoveredIndex ?? selectedIndex;

  const points = useMemo(() => {
    return data.map((item, idx) => {
      const val = Number(item[valueKey] ?? 0);
      const bHeight = Math.max(4, (val / maxValue) * plotHeight);
      const x = plotLeft + idx * (barWidth + barGap) + barWidth / 2;
      const y = plotBottom - bHeight;
      return { x, y };
    });
  }, [data, valueKey, maxValue, plotLeft, plotBottom, plotHeight, barWidth, barGap]);

  if (points.length === 0) return null;

  const pathD = points.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x},${pt.y}` : `${acc} L ${pt.x},${pt.y}`;
  }, "");

  const activePoint = points[activeIdx] ?? points[0];

  return (
    <g className="chart-line-indicator pointer-events-none">
      {/* Connecting Trend Line across top of bars */}
      <path
        d={pathD}
        fill="none"
        stroke={stroke}
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.9}
      />

      {/* Active Glowing Indicator Point */}
      {activePoint && (
        <g>
          {/* Ambient Glow */}
          <circle
            cx={activePoint.x}
            cy={activePoint.y}
            r={10}
            fill={stroke}
            opacity={0.25}
          />
          {/* Outer Ring */}
          <circle
            cx={activePoint.x}
            cy={activePoint.y}
            r={5.5}
            fill="#ffffff"
            stroke={stroke}
            strokeWidth={2.5}
          />
        </g>
      )}
    </g>
  );
}

interface ChartTooltipProps {
  showCrosshair?: boolean;
  showDots?: boolean;
}

export function ChartTooltip({ showCrosshair = false, showDots = false }: ChartTooltipProps) {
  const { data, xDataKey, primaryDataKey, plotLeft, plotBottom, plotHeight, barWidth, barGap, maxValue, selectedIndex, hoveredIndex, viewBoxWidth } = useChartContext();

  const activeIdx = hoveredIndex ?? selectedIndex;
  const activeItem = data[activeIdx];

  if (!activeItem) return null;

  const val = Number(activeItem[primaryDataKey] ?? 0);
  const bHeight = Math.max(4, (val / maxValue) * plotHeight);
  const anchorX = plotLeft + activeIdx * (barWidth + barGap) + barWidth / 2;
  const anchorY = plotBottom - bHeight;

  // Percentage from left for HTML tooltip positioning
  const leftPercent = Math.min(84, Math.max(16, (anchorX / viewBoxWidth) * 100));

  const displayAmount = activeItem.amount ?? `$${val.toLocaleString()}`;
  const displayGrowth = activeItem.growth ?? "+2.46%";
  const displayFullMonth = activeItem.fullMonth ?? `${activeItem[xDataKey]} 2026`;

  return (
    <foreignObject
      x={0}
      y={0}
      width="100%"
      height="100%"
      className="pointer-events-none overflow-visible"
    >
      <div className="relative w-full h-full">
        <div
          className="absolute z-30 transition-all duration-300 ease-out flex flex-col items-center"
          style={{
            left: `${leftPercent}%`,
            top: "0px",
            transform: "translateX(-50%)",
          }}
        >
          {/* Midnight/Sapphire Speech Bubble Card */}
          <div className="bg-[#020b2d] text-white rounded-2xl px-3.5 py-2 shadow-2xl relative flex flex-col gap-0.5 min-w-[145px] border border-sky-400/25">
            {/* Top row: Amount and growth pill */}
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm tracking-tight text-white">
                <DottedNumber
                  value={displayAmount}
                  currencyClassName="mr-0.5 opacity-90 font-normal"
                />
              </span>
              <span className="bg-sky-500/20 text-sky-300 border border-sky-400/20 text-[10px] font-semibold px-2 py-0.5 rounded-full inline-flex items-center gap-0.5">
                <ArrowUpRight className="w-2.5 h-2.5 stroke-[2.5]" />
                <span>{displayGrowth}</span>
              </span>
            </div>
            {/* Bottom row: Month label */}
            <span className="text-sky-200/70 text-[11px] font-normal">
              for {displayFullMonth}
            </span>
          </div>

          {/* Pin Stem with White Dot touching the bar top */}
          <div className="flex flex-col items-center -mt-0.5">
            <div className="w-2.5 h-2.5 rounded-full bg-white ring-2 ring-[#020b2d] shadow-sm z-10" />
            <div className="w-[1.5px] h-3 bg-[#020b2d] -mt-0.5" />
          </div>
        </div>
      </div>
    </foreignObject>
  );
}
