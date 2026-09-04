import { formatMs } from "./chartUtils";

export interface StackRow {
  label: string;
  segments: { label: string; value: number; color: string }[];
}

interface StackedBarChartProps {
  rows: StackRow[];
  /** shared max so rows are comparable across conditions */
  max: number;
}

/**
 * Horizontal stacked bars for the wall-clock breakdown. Horizontal because the
 * row labels ("Baseline", "Speculative") are long, and part-to-whole is the job.
 */
export function StackedBarChart({ rows, max }: StackedBarChartProps) {
  return (
    <div className="stack-chart">
      {rows.map((row) => {
        const total = row.segments.reduce((sum, s) => sum + s.value, 0);
        return (
          <div className="stack-row" key={row.label}>
            <div className="stack-row-head">
              <span className="stack-row-label">{row.label}</span>
              <span className="stack-row-total">{formatMs(total)}</span>
            </div>
            <div className="stack-track" style={{ width: `${(total / max) * 100}%` }}>
              {row.segments.map((seg) => (
                <div
                  key={seg.label}
                  className="stack-seg"
                  style={{ flexGrow: seg.value, background: seg.color }}
                  title={`${seg.label}: ${formatMs(seg.value)}`}
                >
                  {seg.value / total > 0.14 && (
                    <span className="stack-seg-label">
                      {seg.label} · {formatMs(seg.value)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
