# syntax=docker/dockerfile:1
# =============================================================================
# RunPod Serverless — Qwen2.5-VL-3B Arabic-handwriting OCR (base + LoRA adapter)
# =============================================================================
# Build target: RunPod Serverless GitHub integration. RunPod clones this repo
# and runs `docker build` on THEIR builders (the user has weak internet, so the
# ~7.5GB base model is pulled from HF on RunPod's side, not the user's machine).
#
# Hard requirements this Dockerfile satisfies:
#   * Concrete CUDA base with a WORKING CUDA torch (torch 2.10.0 CUDA-12.8 wheel).
#   * Pinned deps mirroring the proven local inference recipe.
#   * The ~7.5GB public base model sherif1313/Arabic-English-handwritten-OCR-v3
#     is BAKED into the image at BUILD via huggingface_hub.snapshot_download to
#     /models/base, so cold starts never re-download it. The build FAILS LOUDLY
#     if the expected weight/config files are not present after download.
#   * The 14MB LoRA adapter (committed in the repo at ./adapter) is COPYed to
#     /app/adapter and loaded onto the base at container import.
#   * CMD runs the RunPod handler (runpod.serverless.start) via handler.py.
#   * No secrets baked. HF token (if ever needed) is a RUNTIME env var only.
#
# GPU sizing: Qwen2.5-VL-3B fp16 = ~7.5GB weights + CUDA/cuDNN context + vision
# activations => ~9-11GB steady state. Recommended RunPod serverless tier:
# 24GB "L4 / A5000 / RTX 3090" (~$0.00019/s ≈ $0.68/hr) as primary, with the
# 16GB "A4000" tier (~$0.00016/s ≈ $0.58/hr) as an availability fallback.
# Set the endpoint GPU priority list to [24GB, then 16GB]. Image is ~12-16GB,
# well under RunPod's 80GB image cap; container disk auto-sizes to the image.
#
# NOTE: This has NOT been executed here (needs the user's paid RunPod account).
# Treat as a first-deploy hypothesis; logging/error-handling live in handler.py.
# handler.py and ./adapter/ MUST exist in the repo before the build — neither
# was present when this Dockerfile was reviewed, so create them first (see the
# "COMPANION FILES REQUIRED" note at the very bottom).
# =============================================================================

# --- Base image ---------------------------------------------------------------
# nvidia/cuda 12.8.2 runtime on Ubuntu 24.04 (glibc 2.39 -> satisfies torch's
# manylinux_2_28 wheel). We use the plain -runtime- (NOT -cudnn-) variant on
# purpose: the torch 2.10.0 PyPI wheel bundles its own CUDA 12.8 runtime AND
# cuDNN 9.10, so a base-image cuDNN would be dead weight. Verified on Docker Hub
# (2026-07-06): 12.8.2-runtime-ubuntu24.04 exists; a 12.8.2-cudnn- variant does
# NOT (only 12.8.1-cudnn-), which is why we do not depend on a cudnn base tag.
# There is NO GPU during RunPod builds — nothing here compiles against CUDA, so
# a runtime (not devel) base is sufficient and smaller.
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

# --- OS-level env -------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System packages ----------------------------------------------------------
# python3.12 is Ubuntu 24.04's default (transformers 5.13 needs >=3.10; torch
# 2.10 ships cp312 wheels). python3-venv gives us a clean, PEP-668-free venv so
# pip installs are isolated from the distro's "externally managed" system Python
# (more robust than setting PIP_BREAK_SYSTEM_PACKAGES and installing into the
# distro site-packages). libgl1/libglib2.0-0 satisfy Pillow/torchvision image
# ops. ca-certificates is required for the HF download to verify TLS at build.
# Merged into one layer + apt lists cleaned to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Isolated virtualenv ------------------------------------------------------
# Build a venv at /opt/venv and put it first on PATH. Everything below ("python",
# "pip") then resolves to the venv interpreter — no PEP-668 friction, no reliance
# on distro symlinks, and the same interpreter runs the build-time bake step and
# the runtime handler. This avoids the Ubuntu-24.04 "externally managed" trap
# entirely (no PIP_BREAK_SYSTEM_PACKAGES hack needed).
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

WORKDIR /app

# --- Python deps: torch first (its own layer for cache friendliness) ----------
# torch 2.10.0 + torchvision 0.25.0: the DEFAULT PyPI linux x86_64 wheels are
# the CUDA 12.8 build (bundling nvidia-cuda-runtime-cu12 12.8.90 + cudnn 9.10),
# so NO --index-url is needed and we get a working CUDA torch on RunPod's amd64
# builders. We assert below that the wheel is the CUDA build (not a CPU fallback)
# so a silent CPU-wheel resolution can never ship a GPU-broken image. Pinned to
# match the proven local recipe (torch 2.10.0/tv 0.25.0). Kept in a separate RUN
# so the ~1GB torch layer is cached across rebuilds when lighter deps/app code
# below change.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==2.10.0" \
        "torchvision==0.25.0" \
    && python -c "import torch, sys; \
print('[torch] version:', torch.__version__, 'cuda build:', torch.version.cuda, flush=True); \
sys.exit(0 if (torch.version.cuda is not None) else 'FATAL: CPU-only torch wheel resolved; expected a CUDA build (torch.version.cuda is None).')"

# --- Python deps: transformers stack + HF downloader --------------------------
# Pinned to the proven-working local venv versions. Deviations from the recipe,
# forced by transformers 5.13 (all recorded in the deploy notes):
#   * safetensors floor raised >=0.5 -> >=0.8.0 (transformers 5.13 requires it).
#   * qwen-vl-utils is INCLUDED for recipe parity but is NOT imported by the
#     handler (the proven recipe uses apply_chat_template + manual PIL prep, not
#     process_vision_info). It is harmless; drop it later to trim the image.
# huggingface_hub is pinned to the transformers-5.13-compatible >=1.5,<2 range
# and provides snapshot_download used in the bake step below. hf_transfer is
# added to speed the ~7.5GB pull so we stay well inside RunPod's 30-min
# docker-build cap (enabled via HF_HUB_ENABLE_HF_TRANSFER in the bake step).
RUN python -m pip install \
        "transformers==5.13.0" \
        "peft==0.19.1" \
        "accelerate==1.14.0" \
        "qwen-vl-utils==0.0.14" \
        "pillow==12.3.0" \
        "safetensors>=0.8.0" \
        "sentencepiece>=0.2" \
        "huggingface_hub>=1.5.0,<2.0" \
        "hf_transfer>=0.1.6" \
        "runpod>=1.7.6"

# --- Bake the base model into the image (BUILD time, on RunPod's builders) -----
# Pull the PUBLIC, non-gated base model (~7.5GB, 2 safetensors shards) to a fixed
# dir /models/base. This runs on RunPod's side with good bandwidth; the user's
# weak internet is never involved. No HF token needed (public repo). We exclude
# ONLY the sample images under assets/ (and .gitattributes) — we deliberately do
# NOT use broad *.png/*.jpg ignore globs at the repo root, because a legitimate
# model asset must never be skipped by an over-broad pattern. HF_HUB_OFFLINE is
# intentionally NOT set yet (the bake needs the network); it is enabled AFTER
# this layer.
#
# HARD VALIDATION: after download we assert that config.json AND at least one
# *.safetensors shard AND a tokenizer/processor config are present. If the repo
# layout ever drifts (or an over-broad ignore pattern nukes a needed file), the
# BUILD fails HERE — loudly, on RunPod's builder — instead of shipping an image
# that crashes at cold start with a cryptic from_pretrained error.
#
# IMPORTANT build-time budget: RunPod's GitHub `docker build` step must finish
# within 30 MINUTES ("Build exceeded maximum time limit of 1800 seconds"). A
# ~7.5GB HF pull with hf_transfer is comfortably within that; if HF is slow and
# the build times out, the documented fallback is to pre-build locally and push
# to a registry (harder given weak internet) — retrying the RunPod build usually
# succeeds. Kept as its own cached layer so app-code edits below don't re-pull.
ENV BASE_MODEL_REPO="sherif1313/Arabic-English-handwritten-OCR-v3" \
    BASE_MODEL_DIR="/models/base" \
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN python -c "\
import os, sys, glob; \
from huggingface_hub import snapshot_download; \
repo=os.environ['BASE_MODEL_REPO']; dst=os.environ['BASE_MODEL_DIR']; \
print(f'[bake] snapshot_download {repo} -> {dst}', flush=True); \
p=snapshot_download(repo_id=repo, local_dir=dst, \
    ignore_patterns=['assets/*', '.gitattributes'], \
    max_workers=8); \
files=sorted(os.listdir(dst)); \
print('[bake] contents:', files, flush=True); \
has_weights=bool(glob.glob(os.path.join(dst, '*.safetensors'))); \
has_config=os.path.exists(os.path.join(dst, 'config.json')); \
has_proc=any(os.path.exists(os.path.join(dst, f)) for f in ('preprocessor_config.json','processor_config.json')); \
has_tok=any(os.path.exists(os.path.join(dst, f)) for f in ('tokenizer.json','tokenizer_config.json')); \
missing=[n for n,ok in (('*.safetensors',has_weights),('config.json',has_config),('processor_config',has_proc),('tokenizer',has_tok)) if not ok]; \
sys.exit(0 if not missing else f'FATAL: baked model at {dst} is missing required files: {missing}. Contents were: {files}')"

# --- Runtime env: force fully-offline model loading ---------------------------
# The base is baked at /models/base and the adapter is COPYed below; the handler
# must NEVER hit the network at cold start. HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
# guarantee that (with local_files_only=True in the handler as belt-and-braces).
# HF_HOME points the cache into the image dir for determinism. handler.py reads
# BASE_MODEL_DIR / ADAPTER_DIR from the env below and must load the base from
# the LOCAL PATH BASE_MODEL_DIR (NOT the HF repo id) so offline loading works.
# hf_transfer stays enabled but is a no-op offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME="/models/hf-home" \
    ADAPTER_DIR="/app/adapter" \
    TOKENIZERS_PARALLELISM=false

# --- App code: adapter + handler (last, so edits don't bust heavier layers) ---
# The 14MB LoRA adapter is committed in the repo at ./adapter. COPY it verbatim;
# handler.py must load it with PeftModel.from_pretrained(model, ADAPTER_DIR)
# (i.e. read ADAPTER_DIR from the env — do NOT hardcode a relative "./adapter",
# which would resolve against the process CWD and is fragile).
COPY adapter/ /app/adapter/
# The RunPod handler. Loads processor + base (from BASE_MODEL_DIR) + adapter ONCE
# at module import (cold start) and reuses them across invocations. It must:
#   - read BASE_MODEL_DIR / ADAPTER_DIR from os.environ,
#   - load base + adapter with local_files_only=True, dtype=torch.float16,
#     re-tie weights, .to("cuda"), .eval() (per the proven recipe),
#   - loop per-image over the ordered line-crops in job["input"], returning
#     texts 1:1 in the same order,
#   - wrap each single image in try/except so ONE bad crop cannot crash the whole
#     page (return an error marker for that index, keep going),
#   - end with runpod.serverless.start({"handler": handler}).
COPY handler.py /app/handler.py

# --- Entrypoint ---------------------------------------------------------------
# `-u` = unbuffered stdout/stderr so every print()/traceback shows up in RunPod
# logs immediately (critical for diagnosing a first deploy). handler.py ends with
# runpod.serverless.start({"handler": handler}); no HTTP server of our own — this
# is a QUEUE-type endpoint, so RunPod owns the server loop. Absolute path is used
# so the CMD is independent of WORKDIR.
CMD ["python", "-u", "/app/handler.py"]

# =============================================================================
# COMPANION FILES REQUIRED (not created by this review — build WILL fail without
# them, since COPY lines above reference both):
#   * ./adapter/            — the 14MB PEFT LoRA (adapter_config.json +
#                             adapter_model.safetensors), committed into the repo.
#   * ./handler.py          — the RunPod handler implementing the proven recipe,
#                             reading BASE_MODEL_DIR / ADAPTER_DIR from env and
#                             ending in runpod.serverless.start({"handler":...}).
# Also ensure a .dockerignore does NOT exclude ./adapter or ./handler.py.
# =============================================================================
