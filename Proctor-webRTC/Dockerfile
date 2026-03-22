# ── Base: CUDA 12.1 + cuDNN 8 runtime on Ubuntu 22.04 ─────────────────────────
# Runtime (not devel) is enough — we run inference, not compile CUDA kernels.
# CUDA 12.1 matches PyTorch 2.3 cu121 wheels.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1 + libglib2.0    → OpenCV headless + MediaPipe on Linux
# libsm6 libxext6        → OpenCV runtime
# libgomp1               → OpenMP used by YOLO/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3    /usr/bin/python  \
 && ln -sf /usr/bin/pip3       /usr/bin/pip

WORKDIR /app

# ── Layer 1: PyTorch with CUDA 12.1 ───────────────────────────────────────────
# Own layer so it's cached across code-only rebuilds (~2.5 GB download once).
RUN pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# ── Layer 2: Remaining Python dependencies ─────────────────────────────────────
# Only re-runs when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 3: YOLO model weights ───────────────────────────────────────────────
# Separate from source so a code change doesn't re-copy the 18 MB model.
COPY finalBestV5.pt .

# ── Layer 4: Application source ───────────────────────────────────────────────
# This is the only layer that changes on every code edit.
COPY core/       core/
COPY detectors/  detectors/
COPY settings/   settings/
COPY utils/      utils/
COPY config.py  .
COPY server.py  .
COPY main.py    .

# Runtime directories (logs + proof/report storage)
RUN mkdir -p logs reports

EXPOSE 8000

# --half     → FP16 on CUDA (~2× throughput, ~½ VRAM)
# --warmup 3 → pre-compile CUDA kernels at startup (eliminates first-frame latency spike)
CMD ["python", "main.py", "--half", "--warmup", "3"]
