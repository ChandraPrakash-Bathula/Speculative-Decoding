export interface Round {
  proposed: number;
  accepted: number;
}

export interface BaselineResult {
  mode: "baseline";
  text: string;
  total_time_ms: number;
  target_calls: number;
  tokens_generated: number;
  rounds: [];
}

export interface SpeculativeResult {
  mode: "speculative";
  text: string;
  total_time_ms: number;
  target_calls: number;
  tokens_generated: number;
  rounds: Round[];
  round_texts: string[];
  gamma: number;
  /** empirical per-token acceptance rate, accepted/proposed over all rounds */
  alpha_hat: number | null;
  /** measured (draft cost per token) / (target cost per forward pass) */
  c_hat: number | null;
  /** the 2023 speedup formula evaluated at this run's own alpha_hat and c_hat */
  predicted_speedup: number | null;
  draft_time_ms: number;
  target_time_ms: number;
}

export interface HealthResponse {
  status: "ok" | "loading";
  device: string;
  draft_model: string;
  target_model: string;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`/health returned ${res.status}`);
  return res.json();
}

async function generate<T>(body: Record<string, unknown>): Promise<T> {
  const res = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`/generate failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export function runBaseline(prompt: string, maxNewTokens: number) {
  return generate<BaselineResult>({
    prompt,
    K: 1,
    mode: "baseline",
    max_new_tokens: maxNewTokens,
  });
}

export function runSpeculative(prompt: string, K: number, maxNewTokens: number) {
  return generate<SpeculativeResult>({
    prompt,
    K,
    mode: "speculative",
    max_new_tokens: maxNewTokens,
  });
}

/* ---------- live streaming ---------- */

export type StreamEvent =
  | { type: "start"; mode: string; K: number; max_new_tokens: number }
  | { type: "tokens"; text: string; count: number }
  | { type: "round"; proposed: number; accepted: number }
  | { type: "error"; message: string }
  | ({ type: "done" } & SpeculativeResult);

/**
 * Streams a generation over server-sent events.
 *
 * Uses fetch + ReadableStream rather than EventSource because the request is a
 * POST with a JSON body, which EventSource cannot send.
 */
export async function streamGenerate(
  body: { prompt: string; K: number; mode: "baseline" | "speculative"; max_new_tokens: number },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`/generate/stream failed: ${res.status} ${await res.text()}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      } catch {
        // a partial or malformed frame is skipped rather than killing the stream
      }
    }
  }
}

/* ---------- derived metrics computed from measured rounds ---------- */

/**
 * Per-position acceptance rate within a speculation round.
 *
 * A round logged as {proposed: 4, accepted: 3} means draft positions 1..3 were
 * accepted and position 4 was rejected; positions beyond the rejection are
 * never evaluated, so they are excluded from that position's denominator
 * rather than counted as rejections. This is what makes the curve a real
 * conditional acceptance rate P(accept position i | reached position i), and
 * it is the direct empirical test of the i.i.d. alpha assumption the 2023
 * speedup formula makes.
 */
export function perPositionAcceptance(rounds: Round[], gamma: number) {
  const reached = new Array(gamma).fill(0);
  const accepted = new Array(gamma).fill(0);

  for (const round of rounds) {
    for (let i = 0; i < Math.min(round.proposed, gamma); i++) {
      reached[i] += 1;
      if (i < round.accepted) accepted[i] += 1;
      else break; // positions after the first rejection were never evaluated
    }
  }

  return reached.map((n, i) => ({
    position: i + 1,
    reached: n,
    accepted: accepted[i],
    rate: n > 0 ? accepted[i] / n : null,
  }));
}

/**
 * Distribution of "how many drafted tokens were accepted" per round, compared
 * against the geometric distribution the theory assumes with the same alpha.
 */
export function acceptedLengthDistribution(rounds: Round[], gamma: number, alpha: number) {
  const scored = rounds.filter((r) => r.proposed > 0);
  const counts = new Array(gamma + 1).fill(0);
  for (const round of scored) counts[Math.min(round.accepted, gamma)] += 1;

  const total = scored.length || 1;
  return counts.map((count, n) => {
    // P(exactly n accepted) = alpha^n * (1-alpha) for n < gamma; alpha^gamma for n = gamma
    const theoretical = n === gamma ? Math.pow(alpha, gamma) : Math.pow(alpha, n) * (1 - alpha);
    return {
      accepted: n,
      empirical: count / total,
      theoretical,
      count,
    };
  });
}

/** The 2023 closed form, used to draw the theoretical curve across K. */
export function theoreticalSpeedup(alpha: number, gamma: number, c: number): number {
  if (alpha > 0.999999) return (gamma + 1) / (gamma * c + 1);
  return (1 - Math.pow(alpha, gamma + 1)) / ((1 - alpha) * (gamma * c + 1));
}
