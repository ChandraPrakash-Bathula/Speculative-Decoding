import type { Round } from "../api";
import { TOKEN_STATE } from "../theme";

interface RoundStripProps {
  rounds: Round[];
  /** how many rounds to reveal (drives the animation) */
  revealed: number;
}

/**
 * One row per speculation round, showing the real fate of every drafted token:
 * accepted (green), rejected and discarded (red), plus the target model's own
 * bonus token (amber) that every round ends with. Shape and position carry the
 * meaning as much as color does - accepted tokens are always leftmost, and the
 * bonus token is always last.
 */
export function RoundStrip({ rounds, revealed }: RoundStripProps) {
  return (
    <div className="round-strip">
      {rounds.slice(0, revealed).map((round, i) => {
        const rejected = Math.max(round.proposed - round.accepted, 0);
        return (
          <div className="round" key={i} title={`Round ${i + 1}: ${round.accepted}/${round.proposed} drafts accepted`}>
            <span className="round-index">{i + 1}</span>
            {Array.from({ length: round.accepted }).map((_, j) => (
              <span key={`a${j}`} className="tok" style={{ background: TOKEN_STATE.accepted }} />
            ))}
            {Array.from({ length: rejected }).map((_, j) => (
              <span key={`r${j}`} className="tok tok-rejected" style={{ background: TOKEN_STATE.rejected }} />
            ))}
            <span className="tok tok-bonus" style={{ background: TOKEN_STATE.bonus }} />
          </div>
        );
      })}
    </div>
  );
}

export function RoundLegend() {
  return (
    <ul className="chart-legend">
      <li>
        <span className="legend-swatch" style={{ background: TOKEN_STATE.accepted }} aria-hidden="true" />
        Draft token accepted
      </li>
      <li>
        <span className="legend-swatch" style={{ background: TOKEN_STATE.rejected }} aria-hidden="true" />
        Rejected / discarded
      </li>
      <li>
        <span className="legend-swatch" style={{ background: TOKEN_STATE.bonus }} aria-hidden="true" />
        Target's bonus token
      </li>
    </ul>
  );
}
