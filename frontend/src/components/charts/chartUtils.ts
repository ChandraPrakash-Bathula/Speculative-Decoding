export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const DEFAULT_MARGIN: Margin = { top: 16, right: 20, bottom: 40, left: 52 };

export function linearScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** "Nice" axis ticks: at most `count` steps on a 1/2/5 x 10^n ladder. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const span = max - min;
  const rawStep = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1) * magnitude;

  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let t = start; t <= max + step * 1e-6; t += step) {
    ticks.push(Math.round(t / step) * step);
  }
  return ticks;
}

export function formatNumber(value: number, digits = 2): string {
  if (!isFinite(value)) return "–";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

export function formatPercent(value: number | null): string {
  if (value === null || !isFinite(value)) return "–";
  return `${(value * 100).toFixed(0)}%`;
}

export function formatMs(ms: number): string {
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(0)} ms`;
}
