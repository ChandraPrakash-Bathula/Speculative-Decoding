import type { ReactNode } from "react";

interface StatTileProps {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  emphasis?: boolean;
}

export function StatTile({ label, value, detail, emphasis }: StatTileProps) {
  return (
    <div className={emphasis ? "stat stat-emphasis" : "stat"}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {detail && <span className="stat-detail">{detail}</span>}
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return <div className="stat-row">{children}</div>;
}
