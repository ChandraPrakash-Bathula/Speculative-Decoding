FROM python:3.11-slim

WORKDIR /app

# torch's CUDA kernels (e.g. Qwen2's rotary embedding) are JIT-compiled by Triton
# at runtime, which needs a C compiler on the image -- python:3.11-slim has none.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
