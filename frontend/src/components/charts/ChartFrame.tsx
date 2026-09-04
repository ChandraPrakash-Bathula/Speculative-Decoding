import type { ReactNode } from "react";

interface ChartFrameProps {
  title: string;
  subtitle?: ReactNode;
  legend?: { label: string; color: string }[];
  children: ReactNode;
  footnote?: ReactNode;
}

/**
 * Shared chart chrome: title, optional legend (always present for >= 2 series so
 * identity is never carried by color alone), the plot, and a caption.
 */
export function ChartFrame({ title, subtitle, legend, children, footnote }: ChartFrameProps) {
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <h3 className="chart-title">{title}</h3>
        {subtitle && <p className="chart-subtitle">{subtitle}</p>}
        {legend && legend.length > 1 && (
          <ul className="chart-legend">
            {legend.map((item) => (
              <li key={item.label}>
                <span className="legend-swatch" style={{ background: item.color }} aria-hidden="true" />
                {item.label}
              </li>
            ))}
          </ul>
        )}
      </figcaption>
      <div className="chart-plot">{children}</div>
      {footnote && <p className="chart-footnote">{footnote}</p>}
    </figure>
  );
}
