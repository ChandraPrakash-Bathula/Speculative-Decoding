import { useState } from "react";
import { DEFAULT_MARGIN, linearScale, niceTicks, formatNumber } from "./chartUtils";
import { INK } from "../../theme";

export interface LineSeries {
  label: string;
  color: string;
  points: { x: number; y: number }[];
  /** dashed lines read as "model / predicted" rather than measured */
  dashed?: boolean;
}

interface LineChartProps {
  series: LineSeries[];
  width?: number;
  height?: number;
  xLabel: string;
  yLabel: string;
  yUnit?: string;
  /** draws a reference line, e.g. speedup = 1 (break-even) */
  referenceY?: { value: number; label: string };
}

export function LineChart({
  series,
  width = 680,
  height = 320,
  xLabel,
  yLabel,
  yUnit = "",
  referenceY,
}: LineChartProps) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const margin = DEFAULT_MARGIN;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) return null;

  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  if (referenceY) ys.push(referenceY.value);

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = Math.max(...ys) * 1.12;
  const yMin = Math.min(0, ...ys);

  const x = linearScale([xMin, xMax], [0, innerW]);
  const y = linearScale([yMin, yMax], [innerH, 0]);

  const xTicks = Array.from(new Set(xs)).sort((a, b) => a - b);
  const yTicks = niceTicks(yMin, yMax, 5);

  const hoveredX = hoverX !== null ? xTicks.reduce((best, t) => (Math.abs(x(t) - hoverX) < Math.abs(x(best) - hoverX) ? t : best), xTicks[0]) : null;

  return (
    <div className="chart-svg-wrap">
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${yLabel} versus ${xLabel}`}
        onMouseLeave={() => setHoverX(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const px = ((e.clientX - rect.left) / rect.width) * width - margin.left;
          setHoverX(px);
        }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* recessive gridlines */}
          {yTicks.map((t) => (
            <line key={t} x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke={INK.grid} strokeWidth={1} />
          ))}

          {referenceY && (
            <g>
              <line
                x1={0}
                x2={innerW}
                y1={y(referenceY.value)}
                y2={y(referenceY.value)}
                stroke={INK.axis}
                strokeWidth={1}
                strokeDasharray="4 4"
              />
              <text x={innerW} y={y(referenceY.value) - 6} textAnchor="end" className="chart-reflabel">
                {referenceY.label}
              </text>
            </g>
          )}

          {/* hover crosshair */}
          {hoveredX !== null && (
            <line x1={x(hoveredX)} x2={x(hoveredX)} y1={0} y2={innerH} stroke={INK.axis} strokeWidth={1} />
          )}

          {/* axes */}
          <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke={INK.axis} strokeWidth={1} />

          {yTicks.map((t) => (
            <text key={t} x={-10} y={y(t) + 4} textAnchor="end" className="chart-tick">
              {formatNumber(t)}
              {yUnit}
            </text>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={x(t)} y={innerH + 20} textAnchor="middle" className="chart-tick">
              {t}
            </text>
          ))}

          {/* series: 2px strokes, >=8px markers, 2px surface ring on overlap */}
          {series.map((s) => {
            const path = s.points
              .slice()
              .sort((a, b) => a.x - b.x)
              .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.x)} ${y(p.y)}`)
              .join(" ");
            return (
              <g key={s.label}>
                <path
                  d={path}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={s.dashed ? "6 4" : undefined}
                />
                {s.points.map((p) => (
                  <circle
                    key={`${s.label}-${p.x}`}
                    cx={x(p.x)}
                    cy={y(p.y)}
                    r={4}
                    fill={s.color}
                    stroke={INK.surface}
                    strokeWidth={2}
                  />
                ))}
                {/* direct label at the series end, so identity never rests on color alone */}
                {s.points.length > 0 &&
                  (() => {
                    const last = s.points.slice().sort((a, b) => a.x - b.x).at(-1)!;
                    return (
                      <text
                        x={x(last.x) + 8}
                        y={y(last.y) + 4}
                        className="chart-directlabel"
                        style={{ fill: s.color }}
                      >
                        {s.label}
                      </text>
                    );
                  })()}
              </g>
            );
          })}

          {/* axis titles */}
          <text x={innerW / 2} y={innerH + 36} textAnchor="middle" className="chart-axistitle">
            {xLabel}
          </text>
          <text
            transform={`translate(${-margin.left + 12},${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            className="chart-axistitle"
          >
            {yLabel}
          </text>
        </g>
      </svg>

      {hoveredX !== null && (
        <div className="chart-tooltip">
          <strong>
            {xLabel} = {hoveredX}
          </strong>
          {series.map((s) => {
            const pt = s.points.find((p) => p.x === hoveredX);
            if (!pt) return null;
            return (
              <div key={s.label}>
                <span className="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
                {s.label}: {formatNumber(pt.y)}
                {yUnit}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
