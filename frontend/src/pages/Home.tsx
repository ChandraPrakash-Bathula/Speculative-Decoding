import { Link } from "react-router-dom";
import { Tex } from "../components/Tex";

export function Home() {
  return (
    <div className="page">
      <section className="hero">
        <p className="eyebrow">Interactive tutorial · measured, not simulated</p>
        <h1>
          Speculative decoding, <em>with the receipts</em>
        </h1>
        <p className="lede">
          A small draft model guesses several tokens ahead; a large target model verifies them all
          in a single forward pass. Accepted guesses are free tokens. This playground runs that
          algorithm for real — Qwen2.5-0.5B drafting for Qwen2.5-1.5B through Transformers'
          assisted-generation API — and every number it reports is read off the live generation
          loop, never fabricated for the visualization.
        </p>
        <div className="hero-actions">
          <Link className="btn btn-primary" to="/playground">
            Open the playground
          </Link>
          <Link className="btn btn-ghost" to="/theory">
            Read the theory
          </Link>
        </div>
      </section>

      <section className="cards">
        <article className="card">
          <h2>The mechanism</h2>
          <p>
            Autoregressive decoding is memory-bandwidth bound: one expensive forward pass buys
            exactly one token. Speculative decoding breaks that 1:1 coupling by spending cheap
            draft compute to make the expensive pass produce several tokens at once.
          </p>
        </article>
        <article className="card">
          <h2>The guarantee</h2>
          <p>
            The accept/reject rule is built so the output distribution is <em>exactly</em> the
            target model's. A weak draft costs you speed, never correctness — a claim this app
            checks empirically by diffing the speculative output against the baseline.
          </p>
        </article>
        <article className="card">
          <h2>The catch</h2>
          <p>
            Speedup depends on the acceptance rate <Tex>{"\\alpha"}</Tex> and the draft/target
            cost ratio <Tex>{"c"}</Tex>. Push <Tex>{"K"}</Tex> too far and you burn draft
            compute on tokens that get thrown away. The playground sweeps <Tex>{"K"}</Tex> and
            shows you where the curve turns over.
          </p>
        </article>
      </section>

      <section className="what-you-get">
        <h2>What this playground measures</h2>
        <ul className="feature-list">
          <li>
            <strong>Per-round accept/reject counts</strong> hooked directly out of the
            <code>AssistedCandidateGenerator</code> — the exact object Transformers uses to run
            speculation.
          </li>
          <li>
            <strong>Draft vs. target wall-clock</strong>, timed separately with device
            synchronisation, giving a real measured <Tex>{"\\hat c"}</Tex> instead of an assumed
            constant.
          </li>
          <li>
            <strong>Per-position acceptance decay</strong> — a direct empirical test of the i.i.d.
            acceptance assumption the classical speedup formula rests on.
          </li>
          <li>
            <strong>A <Tex>{"K"}</Tex>-sweep</strong> comparing measured speedup against the
            closed-form prediction from the 2023 papers, evaluated at this hardware's own
            <Tex>{"\\hat\\alpha"}</Tex> and <Tex>{"\\hat c"}</Tex>.
          </li>
        </ul>
      </section>
    </div>
  );
}
