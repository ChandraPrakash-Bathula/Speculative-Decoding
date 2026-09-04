import { TOKEN_STATE } from "../theme";
import type { Round } from "../api";

export interface LiveChunk {
  text: string;
  /** how many of this chunk's tokens came from accepted drafts */
  accepted: number;
  /** drafts offered but thrown away in the round that produced this chunk */
  rejected: number;
}

interface LiveViewProps {
  title: string;
  chunks: LiveChunk[];
  rounds: Round[];
  running: boolean;
  /** baseline emits one token per pass and has no speculation to show */
  speculative: boolean;
  tokenCount: number;
  elapsedMs: number;
}

/**
 * The live pane. Text lands in bursts as each round commits, and every burst is
 * tinted by how it was produced: draft tokens the target accepted, versus the
 * one token the target always contributes itself. Watching the green outpace
 * the amber is the whole mechanism made visible.
 */
export function LiveView({
  title,
  chunks,
  rounds,
  running,
  speculative,
  tokenCount,
  elapsedMs,
}: LiveViewProps) {
  const tokensPerSecond = elapsedMs > 0 ? (tokenCount / elapsedMs) * 1000 : 0;

  return (
    <div className="live-pane">
      <div className="live-head">
        <h3>
          {title}
          {running && <span className="live-dot" aria-label="generating" />}
        </h3>
        <span className="live-metrics">
          {tokenCount} tok · {tokensPerSecond.toFixed(1)} tok/s
          {speculative && rounds.length > 0 && ` · ${rounds.length} passes`}
        </span>
      </div>

      <div className="live-text">
        {chunks.length === 0 && !running && <span className="live-placeholder">waiting</span>}
        {chunks.map((chunk, i) => (
          <span
            key={i}
            className={speculative ? "live-chunk live-chunk-spec" : "live-chunk"}
            title={
              speculative
                ? `${chunk.accepted} draft token(s) accepted, ${chunk.rejected} rejected, +1 from the target`
                : undefined
            }
          >
            {chunk.text}
          </span>
        ))}
        {running && <span className="live-caret" />}
      </div>

      {speculative && (
        <div className="live-ticker">
          {rounds.slice(-24).map((round, i) => {
            const rejected = Math.max(round.proposed - round.accepted, 0);
            return (
              <span className="ticker-round" key={i}>
                {Array.from({ length: round.accepted }).map((_, j) => (
                  <i key={`a${j}`} style={{ background: TOKEN_STATE.accepted }} />
                ))}
                {Array.from({ length: rejected }).map((_, j) => (
                  <i key={`r${j}`} style={{ background: TOKEN_STATE.rejected, opacity: 0.55 }} />
                ))}
                <i style={{ background: TOKEN_STATE.bonus, borderRadius: "50%" }} />
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
