# ---------- stage 1: build the React frontend ----------
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# ---------- stage 2: python backend + built assets ----------
FROM python:3.11-slim

WORKDIR /app

# torch's CUDA kernels (e.g. Qwen2's rotary embedding) are JIT-compiled by Triton
# at runtime, which needs a C compiler on the image -- python:3.11-slim has none.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY --from=frontend /build/dist ./frontend/dist

# HF Spaces runs the container as a non-root user, so the model cache has to
# live somewhere writable by everyone rather than in the default ~/.cache.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
