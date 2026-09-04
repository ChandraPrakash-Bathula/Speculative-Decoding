import { useState } from "react";
import { DEFAULT_MARGIN, linearScale, niceTicks } from "./chartUtils";
import { INK } from "../../theme";

export interface GroupedDatum {
  label: string;
  values: number[];
}

interface GroupedBarChartProps {
  data: GroupedDatum[];
  seriesLabels: string[];
  colors: string[];
  width?: number;
  height?: number;
  xLabel: string;
  yLabel: string;
  format: (v: number) => string;
}

/** Two-series grouped bars: measured vs. model, side by side with a surface gap. */
export function GroupedBarChart({
  data,
  seriesLabels,
  colors,
  width = 680,
  height = 300,
  xLabel,
  yLabel,
  format,
}: GroupedBarChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const margin = DEFAULT_MARGIN;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const flat = data.flatMap((d) => d.values);
  if (flat.length === 0) return null;
  const yMax = Math.max(...flat) * 1.18;
  const y = linearScale([0, yMax], [innerH, 0]);
  const yTicks = niceTicks(0, yMax, 4);

  const slot = innerW / data.length;
  const groupW = slot - 10;
  const barW = Math.max(6, groupW / seriesLabels.length - 2); // 2px gap inside the group

  return (
    <div className="chart-svg-wrap">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yTicks.map((t) => (
            <line key={t} x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke={INK.grid} strokeWidth={1} />
          ))}
          <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke={INK.axis} strokeWidth={1} />

          {yTicks.map((t) => (
            <text key={t} x={-10} y={y(t) + 4} textAnchor="end" className="chart-tick">
              {format(t)}
            </text>
          ))}

          {data.map((d, i) => {
            const groupX = i * slot + (slot - groupW) / 2;
            return (
              <g key={d.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <rect x={i * slot} y={0} width={slot} height={innerH} fill="transparent" />
                {d.values.map((v, si) => (
                  <rect
                    key={si}
                    x={groupX + si * (barW + 2)}
                    y={y(v)}
                    width={barW}
                    height={Math.max(innerH - y(v), 1)}
                    rx={4}
                    fill={colors[si]}
                    opacity={hover === null || hover === i ? 1 : 0.55}
                  />
                ))}
                <text x={i * slot + slot / 2} y={innerH + 20} textAnchor="middle" className="chart-tick">
                  {d.label}
                </text>
              </g>
            );
          })}

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

      {hover !== null && (
        <div className="chart-tooltip">
          <strong>
            {xLabel} = {data[hover].label}
          </strong>
          {data[hover].values.map((v, si) => (
            <div key={si}>
              <span className="legend-swatch" style={{ background: colors[si] }} aria-hidden="true" />
              {seriesLabels[si]}: {format(v)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
