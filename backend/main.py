"""
Speculative decoding playground backend.

Loads a small "draft" model and a larger "target" model from the same
tokenizer family (Qwen2.5) once at startup, then exposes a single
/generate endpoint that runs either plain (baseline) generation on the
target model or speculative ("assisted") generation using the draft
model as HF's `assistant_model`.

Round-level accept/reject counts are not exposed by `generate()`'s return
value, so we subclass transformers' `AssistedCandidateGenerator` and hook
its `get_candidates` / `update_candidate_strategy` methods -- the exact
points where the draft proposes tokens and the target reports how many
were accepted. Nothing here is simulated: every field in the response is
read off that hook or off wall-clock timing around the real generate() call.
"""

import json
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.streamers import BaseStreamer

DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

state: dict = {}
generate_lock = threading.Lock()


class RecordingAssistedCandidateGenerator(AssistedCandidateGenerator):
    """Same as AssistedCandidateGenerator, but logs proposed/accepted counts per round
    and times how long the draft model itself spends proposing tokens -- a real,
    measured quantity used to estimate the draft/target cost ratio `c` in the
    classic speculative-decoding speedup formula (see the theory page)."""

    def __init__(self, *args, rounds_log=None, stats=None, on_round=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds_log = rounds_log if rounds_log is not None else []
        self.stats = stats if stats is not None else {"draft_time_ms": 0.0}
        # optional callback fired the instant a round is verified, used to push
        # live events to a streaming client
        self.on_round = on_round
        self._pending_proposed = 0

    def get_candidates(self, input_ids):
        start_len = input_ids.shape[1]
        device_type = input_ids.device.type
        if device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        candidate_ids, candidate_logits = super().get_candidates(input_ids)
        if device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        self.stats["draft_time_ms"] += (time.perf_counter() - t0) * 1000
        self._pending_proposed = candidate_ids.shape[1] - start_len
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        entry = {"proposed": self._pending_proposed, "accepted": int(num_matches)}
        self.rounds_log.append(entry)
        if self.on_round is not None:
            self.on_round(entry)
        super().update_candidate_strategy(input_ids, scores, num_matches)


@contextmanager
def _speculative_candidate_generator(target_model, rounds_log, stats, on_round=None):
    """Temporarily make target_model.generate() build our recording candidate generator."""

    def patched(self, generation_config, input_ids, inputs_tensor, logits_processor, model_kwargs,
                assistant_model=None, target_tokenizer=None, assistant_tokenizer=None):
        return RecordingAssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            rounds_log=rounds_log,
            stats=stats,
            on_round=on_round,
        )

    original = target_model.__class__._get_candidate_generator
    target_model._get_candidate_generator = patched.__get__(target_model, target_model.__class__)
    try:
        yield
    finally:
        target_model._get_candidate_generator = original.__get__(target_model, target_model.__class__)


def _sync_device():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def _encode_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    encoding = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return encoding.to(DEVICE)


def run_baseline(prompt: str, max_new_tokens: int) -> dict:
    tokenizer = state["tokenizer"]
    target_model = state["target_model"]
    encoding = _encode_prompt(tokenizer, prompt)
    prompt_len = encoding["input_ids"].shape[1]

    _sync_device()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = target_model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    _sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000

    new_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    tokens_generated = new_ids.shape[0]

    return {
        "mode": "baseline",
        "text": text,
        "total_time_ms": elapsed_ms,
        "target_calls": tokens_generated,
        "tokens_generated": tokens_generated,
        "rounds": [],
    }


def _theory_vs_measurement(rounds_log: list, gamma: int, elapsed_ms: float, draft_time_ms: float, target_calls: int) -> dict:
    """Evaluate the classic speculative-decoding speedup formula (Leviathan et al. 2023;
    Chen et al. 2023) against quantities measured from THIS run, instead of assumed values:

      E[tokens/round] = (1 - alpha^(gamma+1)) / (1 - alpha)
      Speedup(alpha, gamma, c) = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma*c + 1))

    where alpha = empirical per-token acceptance rate (accepted/proposed, averaged over
    rounds that actually proposed tokens) and c = (draft cost per token) / (target cost
    per forward pass), both measured directly: draft_time_ms comes from timing the draft
    model's own forward passes, and target time is the residual of total wall time.
    """
    scored_rounds = [r for r in rounds_log if r["proposed"] > 0]
    total_proposed = sum(r["proposed"] for r in scored_rounds)
    total_accepted = sum(r["accepted"] for r in scored_rounds)
    if total_proposed == 0 or target_calls == 0:
        return {"alpha_hat": None, "c_hat": None, "predicted_speedup": None,
                "draft_time_ms": draft_time_ms, "target_time_ms": max(elapsed_ms - draft_time_ms, 0.0)}

    alpha_hat = total_accepted / total_proposed
    target_time_ms = max(elapsed_ms - draft_time_ms, 1e-6)
    target_time_per_call = target_time_ms / target_calls
    draft_time_per_token = draft_time_ms / total_proposed
    c_hat = draft_time_per_token / target_time_per_call if target_time_per_call > 0 else None

    predicted_speedup = None
    if c_hat is not None:
        if alpha_hat > 0.999999:
            predicted_speedup = (gamma + 1) / (gamma * c_hat + 1)
        else:
            predicted_speedup = (1 - alpha_hat ** (gamma + 1)) / ((1 - alpha_hat) * (gamma * c_hat + 1))

    return {
        "alpha_hat": alpha_hat,
        "c_hat": c_hat,
        "predicted_speedup": predicted_speedup,
        "draft_time_ms": draft_time_ms,
        "target_time_ms": target_time_ms,
    }


def run_speculative(prompt: str, k: int, max_new_tokens: int) -> dict:
    tokenizer = state["tokenizer"]
    target_model = state["target_model"]
    draft_model = state["draft_model"]

    draft_model.generation_config.num_assistant_tokens = k
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"

    encoding = _encode_prompt(tokenizer, prompt)
    prompt_len = encoding["input_ids"].shape[1]
    rounds_log: list = []
    stats = {"draft_time_ms": 0.0}

    _sync_device()
    start = time.perf_counter()
    with _speculative_candidate_generator(target_model, rounds_log, stats):
        with torch.no_grad():
            outputs = target_model.generate(
                **encoding,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    _sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000

    new_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    tokens_generated = new_ids.shape[0]

    # Decode the text emitted in each round (accepted tokens + the target's own
    # bonus token) so the frontend can reveal real text in sync with the rounds,
    # instead of just showing counts. This is a slice of the real output, not a guess.
    round_texts = []
    cursor = 0
    for round_entry in rounds_log:
        emitted = min(round_entry["accepted"] + 1, tokens_generated - cursor)
        emitted = max(emitted, 0)
        chunk_ids = new_ids[cursor:cursor + emitted]
        round_texts.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        cursor += emitted
    if cursor < tokens_generated:
        # trailing tokens generated without an assisted round (rare edge case)
        round_texts.append(tokenizer.decode(new_ids[cursor:], skip_special_tokens=True))

    theory = _theory_vs_measurement(rounds_log, k, elapsed_ms, stats["draft_time_ms"], len(rounds_log))

    return {
        "mode": "speculative",
        "text": text,
        "total_time_ms": elapsed_ms,
        "target_calls": len(rounds_log),
        "tokens_generated": tokens_generated,
        "rounds": rounds_log,
        "round_texts": round_texts,
        "gamma": k,
        **theory,
    }


class QueueStreamer(BaseStreamer):
    """A `generate()` streamer that pushes freshly emitted tokens onto a queue.

    `generate()` calls `put()` with every batch of new token ids. In plain
    decoding that is one token at a time; in assisted decoding it is the whole
    chunk a round committed (accepted drafts plus the target's bonus token), so
    the client sees speculation land in bursts, which is exactly the behaviour
    worth watching.
    """

    def __init__(self, tokenizer, out_queue: "queue.Queue"):
        self.tokenizer = tokenizer
        self.out_queue = out_queue
        self.first_put = True

    def put(self, value):
        if self.first_put:
            self.first_put = False  # the prompt itself, not generated output
            return
        ids = value if value.dim() == 1 else value[0]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if text:
            self.out_queue.put({"type": "tokens", "text": text, "count": int(ids.numel())})

    def end(self):
        pass


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_generation(prompt: str, k: int, mode: str, max_new_tokens: int):
    """Run generation on a worker thread, yielding SSE events as they happen.

    The generation loop is synchronous and blocking, so it runs in its own
    thread and communicates through a queue; this generator drains that queue
    and forwards each event to the browser the moment it is produced.
    """
    tokenizer = state["tokenizer"]
    target_model = state["target_model"]
    draft_model = state["draft_model"]

    events: "queue.Queue" = queue.Queue()
    encoding = _encode_prompt(tokenizer, prompt)
    prompt_len = encoding["input_ids"].shape[1]
    rounds_log: list = []
    stats = {"draft_time_ms": 0.0}
    result: dict = {}

    streamer = QueueStreamer(tokenizer, events)

    def on_round(entry):
        events.put({"type": "round", **entry})

    def worker():
        try:
            _sync_device()
            start = time.perf_counter()
            if mode == "speculative":
                draft_model.generation_config.num_assistant_tokens = k
                draft_model.generation_config.num_assistant_tokens_schedule = "constant"
                with _speculative_candidate_generator(target_model, rounds_log, stats, on_round):
                    with torch.no_grad():
                        outputs = target_model.generate(
                            **encoding,
                            assistant_model=draft_model,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id,
                            streamer=streamer,
                        )
            else:
                with torch.no_grad():
                    outputs = target_model.generate(
                        **encoding,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                    )
            _sync_device()
            elapsed_ms = (time.perf_counter() - start) * 1000

            new_ids = outputs.sequences[0][prompt_len:]
            tokens_generated = int(new_ids.shape[0])
            summary = {
                "type": "done",
                "mode": mode,
                "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "total_time_ms": elapsed_ms,
                "tokens_generated": tokens_generated,
                "rounds": rounds_log,
                "gamma": k if mode == "speculative" else 1,
                "target_calls": len(rounds_log) if mode == "speculative" else tokens_generated,
            }
            if mode == "speculative":
                summary.update(
                    _theory_vs_measurement(
                        rounds_log, k, elapsed_ms, stats["draft_time_ms"], len(rounds_log)
                    )
                )
            result.update(summary)
        except Exception as exc:  # surfaced to the client rather than swallowed
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(None)

    with generate_lock:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        yield _sse({"type": "start", "mode": mode, "K": k, "max_new_tokens": max_new_tokens})
        while True:
            item = events.get()
            if item is None:
                break
            yield _sse(item)
        thread.join()

        if result:
            yield _sse(result)


class GenerateRequest(BaseModel):
    prompt: str
    K: int = 4
    mode: Literal["baseline", "speculative"]
    max_new_tokens: int = 128


app = FastAPI(title="Speculative Decoding Playground")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    draft_model.eval()
    target_model.eval()

    state["tokenizer"] = tokenizer
    state["draft_model"] = draft_model
    state["target_model"] = target_model
    print(f"Models loaded on {DEVICE} ({DTYPE}).")


@app.get("/health")
def health():
    return {
        "status": "ok" if "target_model" in state else "loading",
        "device": DEVICE,
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    with generate_lock:
        if req.mode == "baseline":
            return run_baseline(req.prompt, req.max_new_tokens)
        return run_speculative(req.prompt, req.K, req.max_new_tokens)


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """Server-sent events: tokens and speculation rounds as they are produced."""
    return StreamingResponse(
        stream_generation(req.prompt, req.K, req.mode, req.max_new_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # stop nginx-style reverse proxies (HF Spaces included) from
            # buffering the stream, which would defeat the point
            "X-Accel-Buffering": "no",
        },
    )


# The frontend is a Vite/React SPA built to frontend/dist. Hashed assets are
# served straight from /assets; every other unmatched GET falls through to
# index.html so client-side routes (/theory, /playground) survive a hard reload.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = (FRONTEND_DIST / full_path).resolve()
        # only serve real files that stay inside dist; anything else is a route
        if full_path and candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:  # pragma: no cover - dev convenience when the frontend hasn't been built
    @app.get("/")
    def missing_build():
        return {
            "detail": "frontend/dist not found - run `npm install && npm run build` in frontend/",
        }
