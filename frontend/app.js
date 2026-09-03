const statusBanner = document.getElementById("status-banner");
const kSlider = document.getElementById("k");
const kValue = document.getElementById("k-value");
const generateBtn = document.getElementById("generate-btn");
const promptInput = document.getElementById("prompt");
const results = document.getElementById("results");

const baselineTimeEl = document.getElementById("baseline-time");
const specTimeEl = document.getElementById("spec-time");
const speedupEl = document.getElementById("speedup");
const baselineCallsEl = document.getElementById("baseline-calls");
const baselineTokensEl = document.getElementById("baseline-tokens");
const specCallsEl = document.getElementById("spec-calls");
const specTokensEl = document.getElementById("spec-tokens");
const baselineTextEl = document.getElementById("baseline-text");
const specTextEl = document.getElementById("spec-text");
const roundsEl = document.getElementById("rounds");

kSlider.addEventListener("input", () => {
  kValue.textContent = kSlider.value;
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusBanner.textContent = `Models loaded on ${data.device}. Ready.`;
      statusBanner.classList.add("ready");
      generateBtn.disabled = false;
      return true;
    }
    statusBanner.textContent = "Loading models on the backend, this can take a minute on first start…";
    return false;
  } catch (err) {
    statusBanner.textContent = "Backend not reachable at /health. Is uvicorn running?";
    statusBanner.classList.add("error");
    return false;
  }
}

async function waitForBackend() {
  generateBtn.disabled = true;
  let ready = await checkHealth();
  while (!ready) {
    await sleep(2000);
    ready = await checkHealth();
  }
}

async function callGenerate(prompt, K, mode) {
  const res = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, K, mode, max_new_tokens: 96 }),
  });
  if (!res.ok) {
    throw new Error(`/generate failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

function fmtMs(ms) {
  return `${ms.toFixed(0)} ms`;
}

async function animateBaseline(result) {
  baselineTextEl.textContent = "";
  const words = result.text.split(/(\s+)/);
  const perWordDelay = words.length ? Math.min(40, result.total_time_ms / words.length) : 0;
  for (const word of words) {
    baselineTextEl.textContent += word;
    await sleep(perWordDelay);
  }
}

async function animateSpeculative(result) {
  roundsEl.innerHTML = "";
  specTextEl.textContent = "";

  for (let i = 0; i < result.rounds.length; i++) {
    const round = result.rounds[i];
    const roundEl = document.createElement("div");
    roundEl.className = "round";
    roundsEl.appendChild(roundEl);

    const boxes = [];
    for (let a = 0; a < round.accepted; a++) {
      boxes.push("accepted");
    }
    for (let r = 0; r < round.proposed - round.accepted; r++) {
      boxes.push("rejected");
    }
    boxes.push("bonus");

    for (const kind of boxes) {
      const box = document.createElement("div");
      box.className = `token-box ${kind}`;
      roundEl.appendChild(box);
      await sleep(60);
      box.classList.add("show");
    }

    if (result.round_texts && result.round_texts[i]) {
      specTextEl.textContent += result.round_texts[i];
    }
    await sleep(120);
  }
}

async function onGenerate() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  const K = parseInt(kSlider.value, 10);

  generateBtn.disabled = true;
  results.hidden = false;
  baselineTextEl.textContent = "";
  specTextEl.textContent = "";
  roundsEl.innerHTML = "";
  baselineTimeEl.textContent = "…";
  specTimeEl.textContent = "…";
  speedupEl.textContent = "…";
  baselineCallsEl.textContent = "–";
  baselineTokensEl.textContent = "–";
  specCallsEl.textContent = "–";
  specTokensEl.textContent = "–";

  try {
    const baseline = await callGenerate(prompt, K, "baseline");
    baselineTimeEl.textContent = fmtMs(baseline.total_time_ms);
    baselineCallsEl.textContent = baseline.target_calls;
    baselineTokensEl.textContent = baseline.tokens_generated;
    await animateBaseline(baseline);

    const spec = await callGenerate(prompt, K, "speculative");
    specTimeEl.textContent = fmtMs(spec.total_time_ms);
    specCallsEl.textContent = spec.target_calls;
    specTokensEl.textContent = spec.tokens_generated;
    await animateSpeculative(spec);

    const speedup = baseline.total_time_ms / spec.total_time_ms;
    speedupEl.textContent = `${speedup.toFixed(2)}x`;
  } catch (err) {
    statusBanner.textContent = `Error: ${err.message}`;
    statusBanner.classList.add("error");
  } finally {
    generateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", onGenerate);
waitForBackend();
