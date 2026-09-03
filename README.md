---
title: Speculative Decoding Playground
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Speculative Decoding Playground

Real (not simulated) speculative decoding demo: a FastAPI backend runs
Qwen2.5-0.5B-Instruct as the draft model and Qwen2.5-1.5B-Instruct as the
target model via Transformers' built-in `assistant_model` assisted-generation
API, and records real per-round accept/reject counts by hooking
`AssistedCandidateGenerator`. The frontend visualizes those measurements —
nothing is randomly generated.

## Run locally

```bash
cd backend
pip install -r requirements.txt   # already satisfied in this environment
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/ — the backend also serves the frontend as
static files, so there's nothing else to start.

First request after boot will be slow while both models download from the
Hugging Face Hub (~1GB + ~3GB) and load into memory.

## Run on a Hugging Face Space

This repo is a Docker Space (see the frontmatter above). Push it to a Space
with GPU hardware selected and it builds automatically — `Dockerfile` installs
dependencies and runs `uvicorn` on port 7860, which is what the Spaces
frontmatter (`app_port: 7860`) expects.

This folder isn't its own git repo yet, so initialize one here first (don't
run `git push` from a parent directory that tracks something else):

```bash
git init
git add .
git commit -m "Speculative decoding playground"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

On a CUDA Space the backend automatically picks `cuda` + fp16 (see device
selection in `backend/main.py`); no config changes needed.

## Notes

- Device/dtype: `cuda` + fp16 if a GPU is available, else `mps` + fp16 on
  Apple Silicon, else `cpu` + fp32.
- Both baseline and speculative generation use greedy decoding, so their
  outputs are mathematically guaranteed to match — speculative decoding only
  changes how many target-model forward passes it takes to get there.
- `K` (the draft-tokens-per-round slider) is applied as
  `draft_model.generation_config.num_assistant_tokens` with a constant
  schedule, so it stays fixed across rounds instead of adapting.
