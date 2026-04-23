# Research image for NNMA experiments + parallel training work.
# - Pinned versions that survived the Gemma-3 / TL 2.18 / transformers 4.57 saga.
# - Includes TRL + PEFT + flash-attn for LoRA/SFT/DPO work next to NNMA.
# - Devel-flavoured CUDA image is required so flash-attn can compile.
#
# Build (on a machine with reasonable CPU; flash-attn takes 15-30 min):
#   docker build -t nnm-research:latest .
#
# Push to your registry (DockerHub example):
#   docker tag nnm-research:latest <your-handle>/nnm-research:latest
#   docker push <your-handle>/nnm-research:latest
#
# Use on RunPod:
#   Deploy Custom Pod -> Container Image: <your-handle>/nnm-research:latest
#   Expose HTTP 8888 (Jupyter) + TCP 22 (SSH) if you want both.
#
# Use locally:
#   docker run --gpus all -it --rm -v $(pwd):/workspace nnm-research:latest

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
        openssh-server \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# The pytorch/pytorch:2.5.1-cuda12.4 base image sometimes ships torch
# compiled against a *different* CUDA than its nvcc (observed: torch built
# for cu13.0 while the nvcc is cu12.4). flash-attn's setup.py aborts on the
# mismatch. Force the wheel to match the host nvcc (cu124).
# torch 2.6 pulls in new NVIDIA runtime libs (libcusparseLt etc.) that the
# base image doesn't ship. Install the matching nvidia-* wheels alongside
# torch itself so `import torch` succeeds.
RUN pip install --force-reinstall --no-deps \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        nvidia-cusparselt-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 \
        nvidia-nvtx-cu12 \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple

# The NVIDIA pip wheels drop their .so files under
# site-packages/nvidia/<lib>/lib/. torch has logic to find some of them, but
# libcusparseLt (new dep in torch 2.6) is missed. Register every nvidia-wheel
# lib dir with ldconfig so the dynamic linker picks them up globally.
RUN find /opt/conda/lib/python3.11/site-packages/nvidia -type d -name lib \
        > /etc/ld.so.conf.d/nvidia-pip-wheels.conf \
 && ldconfig \
 && ldconfig -p | grep -E 'cusparseLt|cublas' | head -5 \
 && python -c "import torch; assert torch.version.cuda.startswith('12.4'), torch.version.cuda; print('torch cuda OK:', torch.version.cuda, '| torch:', torch.__version__)"

# Pin torch so subsequent pip installs (transformer_lens, trl, etc.) cannot
# transitively upgrade it to a wheel built against a different CUDA runtime.
# PIP_CONSTRAINT applies to every `pip install` in this container.
# TL 2.18 requires torch>=2.6, so we anchor exactly there.
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Build / type tooling first -- flash-attn needs ninja + packaging at build time
RUN pip install \
        ninja \
        packaging \
        typing_extensions

# Core ML stack (Gemma-3 requires TL 2.18+ and transformers 4.57+)
RUN pip install \
        "transformer_lens>=2.18.0,<3.0" \
        "transformers>=4.57.0,<5.0" \
        "tokenizers>=0.20" \
        "accelerate>=0.30" \
        "bitsandbytes>=0.43" \
        "sentencepiece>=0.1.99" \
        "protobuf>=3.20"

# Data tooling
RUN pip install \
        "datasets>=2.18,<4.0" \
        "huggingface_hub>=0.23"

# Numerics (numpy<2 keeps older bnb/scipy combos happy)
RUN pip install \
        "numpy>=1.24,<2.0" \
        "scipy>=1.10" \
        "scikit-learn>=1.3"

# Training stack (TRL + PEFT for LoRA/SFT/DPO).
# PEFT >= 0.14 required for transformers 4.57+ (older PEFT imports
# BloomPreTrainedModel which was moved/removed in transformers 4.57).
RUN pip install \
        "trl>=0.12,<0.20" \
        "peft>=0.14,<0.20"

# Flash-Attention LAST. Long compile, needs torch visible at build time
# (--no-build-isolation) and disables the default wheel-finder.
RUN pip install flash-attn --no-build-isolation

# Unsloth for fast LoRA/QLoRA fine-tuning. The extras-tag pins matching
# xformers + triton wheels for the torch 2.6 / cu124 combo this image runs.
# Stick to the PyPI release -- git main wants a newer torch.
# Upstream: https://github.com/unslothai/unsloth
RUN pip install "unsloth[cu124-torch260]"

# Unsloth pulls in an older peft as a transitive dep which breaks against
# transformers 4.57 (missing BloomPreTrainedModel import). Force the newer
# peft back in after unsloth is done.
RUN pip install --force-reinstall --no-deps "peft>=0.14,<0.20"

# Sanity-check on import (fails fast if any pin broke)
RUN python -c "\
import torch, transformers, trl, peft, bitsandbytes, transformer_lens, unsloth; \
from importlib.metadata import version as _v; \
print('torch:', torch.__version__, '| cuda:', torch.version.cuda, \
      '| tl:', _v('transformer_lens'), '| transformers:', transformers.__version__, \
      '| trl:', _v('trl'), '| peft:', _v('peft'), '| unsloth:', _v('unsloth'))"

# Jupyter + plotting helpers for interactive research
RUN pip install \
        jupyterlab \
        ipywidgets \
        matplotlib \
        seaborn

# Convenience launcher in $HOME (root on this image)
RUN cat > /root/run_jupyter.sh <<'EOF' \
 && chmod +x /root/run_jupyter.sh
#!/bin/bash
# Launches JupyterLab on port 8888, no token, rooted at /workspace.
# CORS/XSRF disabled so RunPod's reverse-proxy (different host) works.
# Override: PORT=9000 ROOT_DIR=/workspace/foo bash ~/run_jupyter.sh
set -euo pipefail
PORT="${PORT:-8888}"
ROOT_DIR="${ROOT_DIR:-/workspace}"
mkdir -p "$ROOT_DIR"
exec jupyter lab \
    --ip=0.0.0.0 \
    --port="$PORT" \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.root_dir="$ROOT_DIR"
EOF

EXPOSE 8888

WORKDIR /workspace

# Keep container alive so RunPod doesn't mark it "not running".
# Users SSH in and either run `bash ~/run_jupyter.sh` or start their script.
CMD ["sleep", "infinity"]
