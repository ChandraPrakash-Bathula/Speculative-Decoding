import { useEffect, useMemo, useRef, useState } from "react";
import {
  runBaseline,
  runSpeculative,
  perPositionAcceptance,
  acceptedLengthDistribution,
  theoreticalSpeedup,
  type BaselineResult,
  type SpeculativeResult,
} from "../api";
import { SERIES } from "../theme";
import { Tex } from "../components/Tex";
import { StatTile, StatRow } from "../components/StatTile";
import { RoundStrip, RoundLegend } from "../components/RoundStrip";
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
  const [revealed, setRevealed] = useState(0);
  const [showTable, setShowTable] = useState(false);

  const animRef = useRef<number | undefined>(undefined);

  // Reveal rounds one at a time, pacing the animation off the real round count.
  useEffect(() => {
    if (!spec) return;
    setRevealed(0);
    const total = spec.rounds.length;
    const step = () => {
      setRevealed((r) => {
        if (r >= total) return r;
        animRef.current = window.setTimeout(step, 90);
        return r + 1;
      });
    };
    animRef.current = window.setTimeout(step, 90);
    return () => {
      if (animRef.current) window.clearTimeout(animRef.current);
    };
  }, [spec]);

  const outputsMatch = baseline && spec ? baseline.text === spec.text : null;

  async function runComparison() {
    setBusy("Running baseline…");
    setError(null);
    setBaseline(null);
    setSpec(null);
    try {
      const b = await runBaseline(prompt, maxNewTokens);
      setBaseline(b);
      setBusy(`Running speculative (K=${k})…`);
      const s = await runSpeculative(prompt, k, maxNewTokens);
      setSpec(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function runSweep() {
    setError(null);
    setSweep({});
    try {
      let b = baseline;
      if (!b) {
        setBusy("Running baseline…");
        b = await runBaseline(prompt, maxNewTokens);
        setBaseline(b);
      }
      // Sequential, one request per K: keeps each HTTP call short (no proxy
      // timeout on hosted Spaces) and lets points appear as they land.
      for (const kv of SWEEP_K) {
        setBusy(`Sweeping K=${kv}…`);
        const s = await runSpeculative(prompt, kv, maxNewTokens);
        setSweep((prev) => ({ ...prev, [kv]: s }));
        if (kv === k) setSpec(s);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
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
        Runs the same prompt twice — once through the target model alone, once with the draft model
        speculating — and reports what actually happened inside the generation loop.
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
            Run comparison
          </button>
          <button className="btn btn-ghost" onClick={runSweep} disabled={busy !== null}>
            Sweep K = {SWEEP_K.join(", ")}
          </button>
          {busy && <span className="busy">{busy}</span>}
        </div>

        {error && <p className="error-banner">{error}</p>}
      </section>

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
                text as the baseline, character for character — the exactness property, verified on
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
          <RoundStrip rounds={spec.rounds} revealed={revealed} />
          <p className="panel-note">
            {spec.rounds.length} rounds produced {spec.tokens_generated} tokens — an average of{" "}
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
                position 2), so later bars rest on fewer observations — hover for the counts.
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
                  <Tex>{`\\hat c = ${sweepSeries.c.toFixed(3)}`}</Tex> — no fitted parameters.
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
