import { Link } from "react-router-dom";
import { Tex } from "../components/Tex";

export function Theory() {
  return (
    <div className="page page-prose">
      <h1>The theory</h1>
      <p className="lede">
        Speculative decoding is an <em>exact</em> acceleration method: it changes how many
        expensive forward passes you need, not what the model outputs. This page builds that claim
        up from the sampling rule, then states what has changed since the original 2023 papers.
      </p>

      <section>
        <h2>1. Why decoding is slow in the first place</h2>
        <p>
          Generating <Tex>{"n"}</Tex> tokens autoregressively takes <Tex>{"n"}</Tex> sequential
          forward passes through the target model, and each pass must stream every parameter out of
          memory to compute a single token. At batch size 1 this is{" "}
          <strong>memory-bandwidth bound, not compute bound</strong>: the GPU's arithmetic units sit
          mostly idle waiting on weights. That idle capacity is exactly what speculative decoding
          spends: verifying <Tex>{"K"}</Tex> candidate tokens in one pass costs barely more wall
          clock than verifying one, because the weights only have to be read once either way.
        </p>
      </section>

      <section>
        <h2>2. The algorithm</h2>
        <p>
          Write <Tex>{"p(\\cdot \\mid x_{<t})"}</Tex> for the target model's next-token
          distribution and <Tex>{"q(\\cdot \\mid x_{<t})"}</Tex> for the draft's. One round of
          speculation, with speculation length <Tex>{"\\gamma = K"}</Tex>:
        </p>
        <ol className="algo">
          <li>
            <strong>Draft.</strong> Sample <Tex>{"\\gamma"}</Tex> tokens autoregressively from the
            small model: <Tex>{"\\hat{x}_i \\sim q(\\cdot \\mid x_{<t}, \\hat{x}_{<i})"}</Tex>.
            This costs <Tex>{"\\gamma"}</Tex> cheap forward passes.
          </li>
          <li>
            <strong>Verify.</strong> Run the target model <em>once</em> over the whole speculated
            block, obtaining <Tex>{"p(\\cdot \\mid x_{<t}, \\hat{x}_{<i})"}</Tex> for every{" "}
            <Tex>{"i = 1 \\ldots \\gamma+1"}</Tex> in parallel.
          </li>
          <li>
            <strong>Accept.</strong> Walk the block left to right, accepting <Tex>{"\\hat{x}_i"}</Tex>{" "}
            with probability
            <Tex display>
              {"\\min\\!\\left(1,\\ \\frac{p(\\hat{x}_i \\mid x_{<t}, \\hat{x}_{<i})}{q(\\hat{x}_i \\mid x_{<t}, \\hat{x}_{<i})}\\right)"}
            </Tex>
          </li>
          <li>
            <strong>Correct or continue.</strong> At the first rejection, discard that token and
            every draft token after it, and resample from the <em>residual</em> distribution
            <Tex display>
              {"p'(x) \\;=\\; \\frac{\\max\\!\\left(0,\\ p(x) - q(x)\\right)}{\\sum_{x'} \\max\\!\\left(0,\\ p(x') - q(x')\\right)}"}
            </Tex>
            If instead all <Tex>{"\\gamma"}</Tex> drafts are accepted, the verification pass has
            already produced <Tex>{"p(\\cdot \\mid x_{<t+\\gamma})"}</Tex>, so one extra token is
            sampled from it for free.
          </li>
        </ol>
        <p className="callout">
          Either way, <strong>a round always emits at least one token</strong>, so the loop cannot
          stall, no matter how bad the draft model is.
        </p>
      </section>

      <section>
        <h2>3. Why the output distribution is unchanged</h2>
        <p>
          The acceptance rule above is ordinary rejection sampling arranged so the marginal
          distribution of each emitted token is exactly <Tex>{"p"}</Tex>. Sketch: a token{" "}
          <Tex>{"x"}</Tex> is emitted either by being drafted and accepted, with probability{" "}
          <Tex>{"q(x)\\min(1, p(x)/q(x)) = \\min(q(x), p(x))"}</Tex>, or by being drawn from the
          residual after a rejection, which happens with total probability{" "}
          <Tex>{"\\sum_{x'}\\max(0, p(x')-q(x'))"}</Tex> and contributes{" "}
          <Tex>{"\\max(0, p(x)-q(x))"}</Tex> to token <Tex>{"x"}</Tex>. Adding the two paths:
        </p>
        <Tex display>{"\\min(q(x), p(x)) + \\max(0,\\ p(x) - q(x)) \\;=\\; p(x)"}</Tex>
        <p>
          which holds in both cases (<Tex>{"q(x) \\ge p(x)"}</Tex> and{" "}
          <Tex>{"q(x) < p(x)"}</Tex>). So speculative sampling is <strong>exact</strong>, not an
          approximation. The draft model influences throughput only. This is the result of
          Leviathan et al. (2023) and, independently, Chen et al. (2023).
        </p>
        <p>
          This playground runs both models greedily (<Tex>{"\\arg\\max"}</Tex>), the deterministic
          special case: acceptance reduces to "did the draft's argmax match the target's argmax,"
          and a rejection is repaired with the target's own argmax. That is why the playground's
          speculative and baseline outputs come out <em>character-for-character identical</em>, a
          property it verifies on every run rather than asserting.
        </p>
      </section>

      <section>
        <h2>4. How much faster, exactly</h2>
        <p>
          Model the per-token acceptance probability as an i.i.d. constant <Tex>{"\\alpha"}</Tex>.
          The number of accepted drafts in a round is then truncated-geometric, and the expected
          number of tokens a round emits (accepted drafts, plus the guaranteed one) is
        </p>
        <Tex display>
          {"\\mathbb{E}[\\text{tokens per round}] \\;=\\; \\frac{1 - \\alpha^{\\gamma+1}}{1-\\alpha}"}
        </Tex>
        <p>
          Let <Tex>{"c"}</Tex> be the cost of one draft forward pass as a fraction of one target
          forward pass. A round costs <Tex>{"\\gamma c + 1"}</Tex> target-pass-equivalents, so
          dividing tokens-per-round by cost-per-round gives the expected speedup over ordinary
          decoding:
        </p>
        <Tex display>
          {"\\text{Speedup}(\\alpha, \\gamma, c) \\;=\\; \\frac{1 - \\alpha^{\\gamma+1}}{(1-\\alpha)\\,(\\gamma c + 1)}"}
        </Tex>
        <p>
          The numerator saturates as <Tex>{"\\gamma"}</Tex> grows (you cannot accept more than the
          draft is right about) while the denominator keeps growing linearly, which is why speedup
          versus <Tex>{"K"}</Tex> rises, peaks, and then declines. The optimum{" "}
          <Tex>{"\\gamma^\\star"}</Tex> moves right as <Tex>{"\\alpha"}</Tex> rises or{" "}
          <Tex>{"c"}</Tex> falls.
        </p>
        <p className="callout">
          <strong>The assumption worth doubting:</strong> acceptance is <em>not</em> i.i.d. in
          practice. The further into a speculated block you go, the more the draft is conditioning
          on its own guesses, so acceptance tends to decay with position. The playground plots this
          decay directly from measured rounds. See{" "}
          <Link to="/playground">per-position acceptance</Link>.
        </p>
      </section>

      <section>
        <h2>5. What changed after 2023</h2>
        <p>
          The original formulation needs a <em>separate, independently trained</em> draft model that
          shares the target's tokenizer, a real deployment burden. Most subsequent work attacks
          either <Tex>{"\\alpha"}</Tex> (accept more) or <Tex>{"c"}</Tex> (draft cheaper), often
          by eliminating the second model entirely.
        </p>
        <dl className="lit">
          <dt>Medusa (Cai et al., 2024)</dt>
          <dd>
            Attaches several extra decoding heads to the target model, each predicting a token
            further ahead. A tree of their joint top-<Tex>{"k"}</Tex> candidates is verified in one
            pass with tree attention. No separate model, and the heads train cheaply on a frozen
            backbone.
          </dd>

          <dt>EAGLE / EAGLE-2 (Li et al., 2024)</dt>
          <dd>
            Drafts at the <em>feature</em> level: autoregress on the target's own second-to-top
            layer hidden states, which are far more predictable than token distributions, then map
            to tokens once. EAGLE-2 additionally reshapes the draft tree per step using the draft's
            confidence, rather than keeping a fixed tree.
          </dd>

          <dt>Lookahead decoding (Fu et al., 2024)</dt>
          <dd>
            Removes the draft model altogether: it maintains n-gram continuations generated by
            Jacobi iteration and verifies them in the same pass, trading draft cost for extra
            target-side parallel work. Attractive when no aligned small model exists.
          </dd>

          <dt>Self-speculative decoding (Zhang et al., 2023)</dt>
          <dd>
            The target model drafts for itself by skipping a subset of its own layers, so there is
            no second checkpoint to load, serve, or keep aligned, at the cost of a draft whose
            quality you cannot tune independently.
          </dd>

          <dt>SpecInfer and tree-based verification (Miao et al., 2023)</dt>
          <dd>
            Proposes a <em>tree</em> of candidate continuations rather than one chain, so a single
            verification pass can rescue a round that a linear draft would have lost at position 1.
            Raises expected accepted length per target pass at the cost of more draft compute.
          </dd>

          <dt>Online speculative decoding (Liu et al., 2024)</dt>
          <dd>
            Continuously fine-tunes the draft on the live query distribution, so{" "}
            <Tex>{"\\alpha"}</Tex> climbs as the draft specialises to the traffic actually being
            served, turning the acceptance rate into something you operate rather than inherit.
          </dd>
        </dl>
        <p>
          This playground deliberately implements the plain 2023 algorithm with an off-the-shelf
          draft model, because it is the variant Transformers exposes through{" "}
          <code>assistant_model</code>, which keeps every number on the playground a direct
          measurement of a stock library path rather than of a bespoke reimplementation.
        </p>
      </section>

      <section>
        <h2>6. When it does not help</h2>
        <ul>
          <li>
            <strong>Large batches.</strong> The free capacity speculative decoding exploits exists
            because batch-1 decoding is bandwidth bound. At high batch sizes the target pass is
            already compute-saturated, and verifying <Tex>{"K"}</Tex> extra positions stops being
            nearly free.
          </li>
          <li>
            <strong>Low acceptance.</strong> If <Tex>{"\\alpha"}</Tex> is small, nearly every round
            collapses to one token while still paying <Tex>{"\\gamma c"}</Tex> of draft cost, and
            the formula above drops below 1, a real slowdown.
          </li>
          <li>
            <strong>Mismatched vocabularies.</strong> The vanilla algorithm requires the draft and
            target to share a tokenizer; crossing tokenizers needs the universal-assisted-decoding
            machinery and costs alignment overhead.
          </li>
        </ul>
      </section>

      <section className="refs">
        <h2>References</h2>
        <ol>
          <li>
            Leviathan, Kalman &amp; Matias. <em>Fast Inference from Transformers via Speculative
            Decoding.</em> ICML 2023.
          </li>
          <li>
            Chen et al. <em>Accelerating Large Language Model Decoding with Speculative Sampling.</em>{" "}
            2023.
          </li>
          <li>
            Miao et al. <em>SpecInfer: Accelerating Generative LLM Serving with Tree-based
            Speculative Inference and Verification.</em> 2023.
          </li>
          <li>
            Zhang et al. <em>Draft &amp; Verify: Lossless Large Language Model Acceleration via
            Self-Speculative Decoding.</em> 2023.
          </li>
          <li>
            Cai et al. <em>Medusa: Simple LLM Inference Acceleration Framework with Multiple
            Decoding Heads.</em> 2024.
          </li>
          <li>
            Li et al. <em>EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty</em>{" "}
            and <em>EAGLE-2: Faster Inference with Dynamic Draft Trees.</em> 2024.
          </li>
          <li>
            Fu et al. <em>Break the Sequential Dependency of LLM Inference Using Lookahead
            Decoding.</em> 2024.
          </li>
          <li>
            Liu et al. <em>Online Speculative Decoding.</em> 2024.
          </li>
        </ol>
      </section>

      <div className="page-cta">
        <Link className="btn btn-primary" to="/playground">
          Now measure it yourself →
        </Link>
      </div>
    </div>
  );
}
