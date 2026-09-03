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

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator

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
    """Same as AssistedCandidateGenerator, but logs proposed/accepted counts per round."""

    def __init__(self, *args, rounds_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds_log = rounds_log if rounds_log is not None else []
        self._pending_proposed = 0

    def get_candidates(self, input_ids):
        start_len = input_ids.shape[1]
        candidate_ids, candidate_logits = super().get_candidates(input_ids)
        self._pending_proposed = candidate_ids.shape[1] - start_len
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        self.rounds_log.append({"proposed": self._pending_proposed, "accepted": int(num_matches)})
        super().update_candidate_strategy(input_ids, scores, num_matches)


@contextmanager
def _speculative_candidate_generator(target_model, rounds_log):
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
        )

    original = target_model.__class__._get_candidate_generator
    target_model._get_candidate_generator = patched.__get__(target_model, target_model.__class__)
    try:
        yield
    finally:
        target_model._get_candidate_generator = original.__get__(target_model, target_model.__class__)


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

    start = time.perf_counter()
    with torch.no_grad():
        outputs = target_model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
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


def run_speculative(prompt: str, k: int, max_new_tokens: int) -> dict:
    tokenizer = state["tokenizer"]
    target_model = state["target_model"]
    draft_model = state["draft_model"]

    draft_model.generation_config.num_assistant_tokens = k
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"

    encoding = _encode_prompt(tokenizer, prompt)
    prompt_len = encoding["input_ids"].shape[1]
    rounds_log: list = []

    start = time.perf_counter()
    with _speculative_candidate_generator(target_model, rounds_log):
        with torch.no_grad():
            outputs = target_model.generate(
                **encoding,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
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

    return {
        "mode": "speculative",
        "text": text,
        "total_time_ms": elapsed_ms,
        "target_calls": len(rounds_log),
        "tokens_generated": tokens_generated,
        "rounds": rounds_log,
        "round_texts": round_texts,
    }


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


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
