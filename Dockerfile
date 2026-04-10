# syntax=docker/dockerfile:1
# DONUT SROIE pipeline — Vast.ai / cloud GPU image
# Build: docker build -t donut-sroie .
# Run:   docker run --gpus all -v /workspace:/workspace \
#          -e GITHUB_TOKEN=... -e HF_TOKEN=... donut-sroie

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DONUT_WORKSPACE=/workspace \
    SROIE_DATA_DIR=/workspace/ICDAR-2019-SROIE/data \
    PIP_NO_CACHE_DIR=1

# System deps: Python 3.10, git, LaTeX (paper generation), libGL (OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl wget \
    texlive-latex-base texlive-latex-extra \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# STEP 1: install torch + torchvision FIRST (required by flash-attn build)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# STEP 2: install all remaining deps — flash-attn is NOT in requirements.txt
COPY requirements.txt requirements-gpu-optional.txt ./
RUN pip install -r requirements.txt

# STEP 3: install flash-attn from the separate optional file, with
# --no-build-isolation so the already-installed torch is visible to setup.py.
RUN pip install -r requirements-gpu-optional.txt --no-build-isolation

# STEP 4: copy application code
COPY . .

# Validate import chain at build time — fails fast if constants.py is broken
RUN python -c "from constants import FIELDS, BASE_MODEL, SEED; print('Import chain OK')"

RUN mkdir -p /workspace

# Secrets are injected at runtime via env vars — never baked into the image.
# Vast.ai: set GITHUB_TOKEN, HF_TOKEN, ANTHROPIC_API_KEY in instance env vars.

# TensorBoard port (expose for Vast.ai port mapping)
EXPOSE 6006

ENTRYPOINT ["bash", "bootstrap.sh"]
