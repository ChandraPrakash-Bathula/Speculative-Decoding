import { useMemo, useRef, useState } from "react";
import {
  runSpeculative,
  streamGenerate,
  perPositionAcceptance,
  acceptedLengthDistribution,
  theoreticalSpeedup,
  type BaselineResult,
  type SpeculativeResult,
  type Round,
} from "../api";
import { SERIES } from "../theme";
import { Tex } from "../components/Tex";
import { StatTile, StatRow } from "../components/StatTile";
import { RoundStrip, RoundLegend } from "../components/RoundStrip";
import { LiveView, type LiveChunk } from "../components/LiveView";
import { ChartFrame } from "../components/charts/ChartFrame";
import { LineChart } from "../components/charts/LineChart";
import { BarChart } from "../components/charts/BarChart";
import { GroupedBarChart } from "../components/charts/GroupedBarChart";
import { StackedBarChart } from "../components/charts/StackedBarChart";
import { formatMs, formatNumber, formatPercent } from "../components/charts/chartUtils";

const PRESET_PROMPTS = [
  "Explain what a hash map is and give a short example.",
  "Write a Python function that reverses a linked list.",
  "Summarise why transformers replaced RNNs for sequence modelling.",
];

const SWEEP_K = [1, 2, 3, 4, 6, 8];

export function Playground() {
  const [prompt, setPrompt] = useState(PRESET_PROMPTS[0]);
  const [k, setK] = useState(4);
  const [maxNewTokens, setMaxNewTokens] = useState(64);

  const [baseline, setBaseline] = useState<BaselineResult | null>(null);
  const [spec, setSpec] = useState<SpeculativeResult | null>(null);
  const [sweep, setSweep] = useState<Record<number, SpeculativeResult>>({});
  const [busy, setBusy] = useState<null | string>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);

  // live-stream state, updated as events arrive from the server
  const [liveStage, setLiveStage] = useState<null | "baseline" | "speculative">(null);
  const [baseChunks, setBaseChunks] = useState<LiveChunk[]>([]);
  const [specChunks, setSpecChunks] = useState<LiveChunk[]>([]);
  const [liveRounds, setLiveRounds] = useState<Round[]>([]);
  const [baseElapsed, setBaseElapsed] = useState(0);
  const [specElapsed, setSpecElapsed] = useState(0);

  const abortRef = useRef<AbortController | null>(null);

  const outputsMatch = baseline && spec ? baseline.text === spec.text : null;

  function resetLive() {
    setBaseChunks([]);
    setSpecChunks([]);
    setLiveRounds([]);
    setBaseElapsed(0);
    setSpecElapsed(0);
  }

  /**
   * Streams one generation, pushing text and rounds into state as they land.
   *
   * The server emits a `tokens` event when a chunk is committed and a `round`
   * event with that round's accept/reject counts immediately after, so the
   * pending chunk is annotated once its round arrives.
   */
  async function streamOne(
    mode: "baseline" | "speculative",
    kValue: number,
    setChunks: React.Dispatch<React.SetStateAction<LiveChunk[]>>,
    setElapsed: (ms: number) => void,
  ) {
    const started = performance.now();
    const ticker = window.setInterval(() => setElapsed(performance.now() - started), 100);
    let result: SpeculativeResult | null = null;

    try {
      await streamGenerate(
        { prompt, K: kValue, mode, max_new_tokens: maxNewTokens },
        (event) => {
          if (event.type === "tokens") {
            setChunks((prev) => [...prev, { text: event.text, accepted: 0, rejected: 0 }]);
          } else if (event.type === "round") {
            setLiveRounds((prev) => [...prev, { proposed: event.proposed, accepted: event.accepted }]);
            // annotate the chunk this round produced
            setChunks((prev) => {
              if (prev.length === 0) return prev;
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = {
                ...last,
                accepted: event.accepted,
                rejected: Math.max(event.proposed - event.accepted, 0),
              };
              return next;
            });
          } else if (event.type === "error") {
            setError(event.message);
          } else if (event.type === "done") {
            result = event;
            setElapsed(event.total_time_ms);
          }
        },
        abortRef.current?.signal,
      );
    } finally {
      window.clearInterval(ticker);
    }
    return result;
  }

  async function runComparison() {
    setError(null);
    setBaseline(null);
    setSpec(null);
    resetLive();
    abortRef.current = new AbortController();

    try {
      setBusy("streaming baseline");
      setLiveStage("baseline");
      const b = await streamOne("baseline", 1, setBaseChunks, setBaseElapsed);
      if (b) setBaseline(b as unknown as BaselineResult);

      setBusy(`streaming speculative, K=${k}`);
      setLiveStage("speculative");
      const s = await streamOne("speculative", k, setSpecChunks, setSpecElapsed);
      if (s) setSpec(s);
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(null);
      setLiveStage(null);
    }
  }

  function stopRun() {
    abortRef.current?.abort();
    setBusy(null);
    setLiveStage(null);
  }

  async function runSweep() {
    setError(null);
    setSweep({});
    try {
      let b = baseline;
      if (!b) {
        setBusy("running baseline");
        resetLive();
        abortRef.current = new AbortController();
        setLiveStage("baseline");
        const streamed = await streamOne("baseline", 1, setBaseChunks, setBaseElapsed);
        if (streamed) {
          b = streamed as unknown as BaselineResult;
          setBaseline(b);
        }
        setLiveStage(null);
      }
      // Sequential, one request per K: keeps each HTTP call short (no proxy
      // timeout on hosted Spaces) and lets points appear as they land.
      for (const kv of SWEEP_K) {
        setBusy(`sweeping K=${kv}`);
        const s = await runSpeculative(prompt, kv, maxNewTokens);
        setSweep((prev) => ({ ...prev, [kv]: s }));
        if (kv === k) setSpec(s);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      setLiveStage(null);
    }
  }

  const measuredSpeedup = baseline && spec ? baseline.total_time_ms / spec.total_time_ms : null;

  /* ---------------- derived chart data ---------------- */

  const positionData = useMemo(() => {
    if (!spec) return [];
    return perPositionAcceptance(spec.rounds, spec.gamma).map((p) => ({
      label: `${p.position}`,
      value: p.rate,
      detail: `${p.accepted} accepted of ${p.reached} rounds that reached position ${p.position}`,
    }));
  }, [spec]);

  const distributionData = useMemo(() => {
    if (!spec || spec.alpha_hat === null) return [];
    return acceptedLengthDistribution(spec.rounds, spec.gamma, spec.alpha_hat).map((d) => ({
      label: `${d.accepted}`,
      values: [d.empirical, d.theoretical],
    }));
  }, [spec]);

  const sweepSeries = useMemo(() => {
    const entries = Object.entries(sweep)
      .map(([kv, r]) => ({ k: Number(kv), r }))
      .sort((a, b) => a.k - b.k);
    if (entries.length === 0 || !baseline) return null;

    const measured = entries.map((e) => ({
      x: e.k,
      y: baseline.total_time_ms / e.r.total_time_ms,
    }));

    // Theoretical curve uses the mean measured alpha and c across the sweep, so
    // the model line is parameterised by this machine's own measurements.
    const alphas = entries.map((e) => e.r.alpha_hat).filter((a): a is number => a !== null);
    const cs = entries.map((e) => e.r.c_hat).filter((c): c is number => c !== null);
    if (alphas.length === 0 || cs.length === 0) return { measured, predicted: null, alpha: null, c: null };

    const alpha = alphas.reduce((s, a) => s + a, 0) / alphas.length;
    const c = cs.reduce((s, v) => s + v, 0) / cs.length;
    const predicted = entries.map((e) => ({ x: e.k, y: theoreticalSpeedup(alpha, e.k, c) }));
    return { measured, predicted, alpha, c };
  }, [sweep, baseline]);

  return (
    <div className="page">
      <h1>Playground</h1>
      <p className="lede">
        Runs the same prompt twice, once through the target model alone and once with the draft model
        speculating, then reports what actually happened inside the generation loop.
      </p>

      {/* ---------------- controls ---------------- */}
      <section className="panel controls-panel">
        <label className="field">
          <span className="field-label">Prompt</span>
          <textarea
            value={prompt}
            rows={3}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={busy !== null}
          />
        </label>
        <div className="preset-row">
          {PRESET_PROMPTS.map((p, i) => (
            <button key={i} className="chip" onClick={() => setPrompt(p)} disabled={busy !== null}>
              Preset {i + 1}
            </button>
          ))}
        </div>

        <div className="controls-grid">
          <label className="field">
            <span className="field-label">
              Speculation length K = <strong>{k}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={10}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              disabled={busy !== null}
            />
            <span className="field-hint">draft tokens proposed per round</span>
          </label>

          <label className="field">
            <span className="field-label">
              Max new tokens = <strong>{maxNewTokens}</strong>
            </span>
            <input
              type="range"
              min={16}
              max={192}
              step={16}
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              disabled={busy !== null}
            />
            <span className="field-hint">longer runs give steadier timing</span>
          </label>
        </div>

        <div className="button-row">
          <button className="btn btn-primary" onClick={runComparison} disabled={busy !== null}>
            Run live comparison
          </button>
          <button className="btn btn-ghost" onClick={runSweep} disabled={busy !== null}>
            Sweep K = {SWEEP_K.join(", ")}
          </button>
          {busy && (
            <>
              <span className="busy">{busy}</span>
              <button className="btn btn-ghost btn-small" onClick={stopRun}>
                Stop
              </button>
            </>
          )}
        </div>

        {error && <p className="error-banner">{error}</p>}
      </section>

      {/* ---------------- live generation ---------------- */}
      {(baseChunks.length > 0 || specChunks.length > 0 || busy) && (
        <section className="panel">
          <h2>Live generation</h2>
          <p className="panel-note">
            Streamed over server-sent events as the model runs. The baseline dribbles out one token
            per forward pass; the speculative pane lands whole bursts at once, because each target
            pass commits every draft token it accepted plus one of its own.
          </p>
          <div className="live-grid">
            <LiveView
              title="Baseline"
              chunks={baseChunks}
              rounds={[]}
              running={liveStage === "baseline"}
              speculative={false}
              tokenCount={baseChunks.length}
              elapsedMs={baseElapsed}
            />
            <LiveView
              title={`Speculative (K=${k})`}
              chunks={specChunks}
              rounds={liveRounds}
              running={liveStage === "speculative"}
              speculative
              tokenCount={liveRounds.reduce((n, r) => n + r.accepted + 1, 0)}
              elapsedMs={specElapsed}
            />
          </div>
          <RoundLegend />
        </section>
      )}

      {/* ---------------- headline numbers ---------------- */}
      {baseline && spec && (
        <section className="panel">
          <h2>Headline</h2>
          <StatRow>
            <StatTile
              label="Measured speedup"
              value={measuredSpeedup ? `${formatNumber(measuredSpeedup)}×` : "–"}
              detail="wall clock, baseline ÷ speculative"
              emphasis
            />
            <StatTile
              label="Target forward passes"
              value={`${spec.target_calls} vs ${baseline.target_calls}`}
              detail={`${formatNumber(baseline.target_calls / spec.target_calls)}× fewer expensive passes`}
            />
            <StatTile
              label="Acceptance rate α̂"
              value={formatPercent(spec.alpha_hat)}
              detail="of drafted tokens kept"
            />
            <StatTile
              label="Draft/target cost ĉ"
              value={spec.c_hat !== null ? formatNumber(spec.c_hat, 3) : "–"}
              detail="measured, not assumed"
            />
          </StatRow>

          <div className={outputsMatch ? "verdict verdict-ok" : "verdict verdict-warn"}>
            {outputsMatch ? (
              <>
                <strong>Outputs identical.</strong> The speculative run produced exactly the same
                text as the baseline, character for character. That is the exactness property, verified on
                this run rather than assumed.
              </>
            ) : (
              <>
                <strong>Outputs differ.</strong> Under greedy decoding these should match; a
                mismatch usually means the two runs hit different stopping points.
              </>
            )}
          </div>

          <div className="output-columns">
            <div>
              <h3>Baseline output</h3>
              <pre className="output">{baseline.text}</pre>
            </div>
            <div>
              <h3>Speculative output</h3>
              <pre className="output">{spec.text}</pre>
            </div>
          </div>
        </section>
      )}

      {/* ---------------- round-by-round ---------------- */}
      {spec && spec.rounds.length > 0 && (
        <section className="panel">
          <h2>Round by round</h2>
          <p className="panel-note">
            Each row is one speculation round: {spec.gamma} drafted tokens offered, however many the
            target accepted, then the target's own bonus token. Read off the live
            <code> AssistedCandidateGenerator</code>, not reconstructed.
          </p>
          <RoundLegend />
          <RoundStrip rounds={spec.rounds} revealed={spec.rounds.length} />
          <p className="panel-note">
            {spec.rounds.length} rounds produced {spec.tokens_generated} tokens, an average of{" "}
            <strong>{formatNumber(spec.tokens_generated / spec.rounds.length)}</strong> tokens per
            target forward pass, versus exactly 1.00 for the baseline.
          </p>
        </section>
      )}

      {/* ---------------- charts ---------------- */}
      {spec && positionData.length > 0 && (
        <section className="panel">
          <h2>Where the theory bends</h2>

          <ChartFrame
            title="Acceptance rate by position within a round"
            subtitle={
              <>
                The closed-form speedup assumes a single constant <Tex>{"\\alpha"}</Tex> at every
                position. Measured, it decays: each further token conditions on more of the draft's
                own guesses.
              </>
            }
            footnote={
              <>
                Denominators shrink with position (a round that dies at position 1 never tests
                position 2), so later bars rest on fewer observations. Hover for the counts.
              </>
            }
          >
            <BarChart
              data={positionData}
              color={SERIES.measured}
              xLabel="position within speculation block"
              yLabel="P(accept | reached)"
              format={(v) => `${(v * 100).toFixed(0)}%`}
              yMaxHint={1}
            />
          </ChartFrame>

          {distributionData.length > 0 && (
            <ChartFrame
              title="Accepted tokens per round: measured vs. geometric model"
              subtitle={
                <>
                  The <Tex>{"\\alpha"}</Tex>-geometric model predicts the orange bars from this
                  run's own <Tex>{"\\hat\\alpha"}</Tex>. Gaps between the pairs are exactly where
                  the i.i.d. assumption fails.
                </>
              }
              legend={[
                { label: "Measured", color: SERIES.measured },
                { label: "Geometric model", color: SERIES.predicted },
              ]}
            >
              <GroupedBarChart
                data={distributionData}
                seriesLabels={["Measured", "Geometric model"]}
                colors={[SERIES.measured, SERIES.predicted]}
                xLabel="drafted tokens accepted in the round"
                yLabel="share of rounds"
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </ChartFrame>
          )}

          <ChartFrame
            title="Where the wall clock goes"
            subtitle="Draft time is measured directly around the draft model's forward passes, with device synchronisation; the remainder is target verification plus loop overhead."
            legend={[
              { label: "Draft model", color: SERIES.predicted },
              { label: "Target model + overhead", color: SERIES.measured },
            ]}
          >
            <StackedBarChart
              max={Math.max(baseline?.total_time_ms ?? 0, spec.total_time_ms)}
              rows={[
                ...(baseline
                  ? [
                      {
                        label: "Baseline",
                        segments: [
                          {
                            label: "Target model",
                            value: baseline.total_time_ms,
                            color: SERIES.measured,
                          },
                        ],
                      },
                    ]
                  : []),
                {
                  label: `Speculative (K=${spec.gamma})`,
                  segments: [
                    { label: "Draft", value: spec.draft_time_ms, color: SERIES.predicted },
                    { label: "Target + overhead", value: spec.target_time_ms, color: SERIES.measured },
                  ],
                },
              ]}
            />
          </ChartFrame>
        </section>
      )}

      {/* ---------------- sweep ---------------- */}
      {sweepSeries && sweepSeries.measured.length > 0 && (
        <section className="panel">
          <h2>Speedup across speculation length</h2>
          <ChartFrame
            title="Measured speedup vs. the 2023 closed form"
            subtitle={
              sweepSeries.alpha !== null && sweepSeries.c !== null ? (
                <>
                  Model line is{" "}
                  <Tex>
                    {"\\frac{1-\\alpha^{\\gamma+1}}{(1-\\alpha)(\\gamma c + 1)}"}
                  </Tex>{" "}
                  evaluated at this machine's mean measured{" "}
                  <Tex>{`\\hat\\alpha = ${sweepSeries.alpha.toFixed(2)}`}</Tex> and{" "}
                  <Tex>{`\\hat c = ${sweepSeries.c.toFixed(3)}`}</Tex>, with no fitted parameters.
                </>
              ) : (
                "Measured speedup at each speculation length."
              )
            }
            legend={[
              { label: "Measured", color: SERIES.measured },
              ...(sweepSeries.predicted ? [{ label: "Model", color: SERIES.predicted }] : []),
            ]}
            footnote="Above the dashed line, speculation wins; below it, drafting costs more than it saves."
          >
            <LineChart
              xLabel="K"
              yLabel="speedup"
              yUnit="×"
              referenceY={{ value: 1, label: "break-even" }}
              series={[
                { label: "Measured", color: SERIES.measured, points: sweepSeries.measured },
                ...(sweepSeries.predicted
                  ? [
                      {
                        label: "Model",
                        color: SERIES.predicted,
                        points: sweepSeries.predicted,
                        dashed: true,
                      },
                    ]
                  : []),
              ]}
            />
          </ChartFrame>

          <button className="btn btn-ghost btn-small" onClick={() => setShowTable((s) => !s)}>
            {showTable ? "Hide" : "Show"} data table
          </button>

          {showTable && (
            <table className="data-table">
              <caption>Every measurement behind the sweep</caption>
              <thead>
                <tr>
                  <th scope="col">K</th>
                  <th scope="col">Time</th>
                  <th scope="col">Speedup</th>
                  <th scope="col">Target passes</th>
                  <th scope="col">Tokens</th>
                  <th scope="col">α̂</th>
                  <th scope="col">ĉ</th>
                  <th scope="col">Model speedup</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(sweep)
                  .map(([kv, r]) => ({ k: Number(kv), r }))
                  .sort((a, b) => a.k - b.k)
                  .map(({ k: kv, r }) => (
                    <tr key={kv}>
                      <td>{kv}</td>
                      <td>{formatMs(r.total_time_ms)}</td>
                      <td>
                        {baseline ? `${formatNumber(baseline.total_time_ms / r.total_time_ms)}×` : "–"}
                      </td>
                      <td>{r.target_calls}</td>
                      <td>{r.tokens_generated}</td>
                      <td>{formatPercent(r.alpha_hat)}</td>
                      <td>{r.c_hat !== null ? formatNumber(r.c_hat, 3) : "–"}</td>
                      <td>
                        {r.predicted_speedup !== null ? `${formatNumber(r.predicted_speedup)}×` : "–"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {!baseline && !busy && (
        <section className="panel empty-state">
          <p>
            Nothing measured yet. Hit <strong>Run comparison</strong> for a single head-to-head, or{" "}
            <strong>Sweep K</strong> to trace the speedup curve and compare it against theory.
          </p>
          <p className="panel-note">
            First run after a cold start also pays for model loading; later runs are the honest
            numbers.
          </p>
        </section>
      )}
    </div>
  );
}
