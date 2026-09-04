/**
 * Chart color roles, referenced as CSS custom properties so that light/dark
 * swap happens in one place (styles.css) rather than in JS.
 *
 * Values come from the validated reference palette and were re-checked with the
 * palette validator for both modes: all checks pass. Light-mode aqua carries a
 * sub-3:1 contrast WARN, which is why every series in this app is
 * direct-labeled rather than identified by color alone.
 */
export const SERIES = {
  /** slot 1 - measured quantities */
  measured: "var(--series-1)",
  /** slot 2 - predicted / theoretical quantities */
  predicted: "var(--series-2)",
  /** slot 3 - used sparingly, never as the only cue */
  third: "var(--series-3)",
} as const;

export const TOKEN_STATE = {
  accepted: "var(--status-good)",
  rejected: "var(--status-critical)",
  bonus: "var(--status-warning)",
} as const;

export const INK = {
  primary: "var(--text-primary)",
  secondary: "var(--text-secondary)",
  muted: "var(--text-muted)",
  grid: "var(--gridline)",
  axis: "var(--axis)",
  surface: "var(--surface-1)",
} as const;
