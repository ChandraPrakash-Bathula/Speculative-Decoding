import { useState } from "react";
import { DEFAULT_MARGIN, linearScale, niceTicks } from "./chartUtils";
import { INK } from "../../theme";

export interface BarDatum {
  label: string;
  value: number | null;
  /** shown in the tooltip as supporting detail (e.g. "37 of 42 rounds") */
  detail?: string;
}

interface BarChartProps {
  data: BarDatum[];
  color: string;
  width?: number;
  height?: number;
  xLabel: string;
  yLabel: string;
  /** formats bar-top direct labels and tooltip values */
  format: (v: number) => string;
  yMaxHint?: number;
}

/** Vertical bars, 4px rounded data-ends anchored to the baseline, 2px gaps. */
export function BarChart({
  data,
  color,
  width = 680,
  height = 300,
  xLabel,
  yLabel,
  format,
  yMaxHint,
}: BarChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const margin = DEFAULT_MARGIN;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const values = data.map((d) => d.value).filter((v): v is number => v !== null);
  if (values.length === 0) return null;
  const yMax = yMaxHint ?? Math.max(...values) * 1.15;
  const y = linearScale([0, yMax], [innerH, 0]);
  const yTicks = niceTicks(0, yMax, 4);

  const slot = innerW / data.length;
  const barW = Math.max(8, slot - 8); // 2px+ surface gap between adjacent bars

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
            const cx = i * slot + slot / 2;
            if (d.value === null) {
              return (
                <text key={d.label} x={cx} y={innerH - 6} textAnchor="middle" className="chart-tick">
                  n/a
                </text>
              );
            }
            const h = innerH - y(d.value);
            return (
              <g
                key={d.label}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                {/* hit target larger than the mark */}
                <rect x={cx - slot / 2} y={0} width={slot} height={innerH} fill="transparent" />
                <rect
                  x={cx - barW / 2}
                  y={y(d.value)}
                  width={barW}
                  height={Math.max(h, 1)}
                  rx={4}
                  fill={color}
                  opacity={hover === null || hover === i ? 1 : 0.55}
                />
                <text x={cx} y={y(d.value) - 6} textAnchor="middle" className="chart-directlabel">
                  {format(d.value)}
                </text>
                <text x={cx} y={innerH + 20} textAnchor="middle" className="chart-tick">
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

      {hover !== null && data[hover].detail && (
        <div className="chart-tooltip">
          <strong>
            {xLabel} {data[hover].label}
          </strong>
          <div>{data[hover].detail}</div>
        </div>
      )}
    </div>
  );
}
