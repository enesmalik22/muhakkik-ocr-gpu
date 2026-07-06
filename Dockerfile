# ─────────────────────────────────────────────────────────────────────────────
# HATformer — RunPod Serverless LOAD-BALANCING worker
# Serves serve_8767.py (FastAPI: POST /recognize + GET /ping + GET /health) on GPU.
#
# ⚠️ BEFORE YOU BUILD — TWO THINGS THAT SILENTLY BREAK A FIRST DEPLOY:
#
#   (A) .dockerignore IS MANDATORY. The build context (this folder) is ~2.1 GB:
#       hatformer-muharaf/ (637M) + hatformer-synthetic/ (637M) + dataset/ (230M)
#       + hatformer-10h-naskh-best/ (640M) are all here. This Dockerfile COPYs only
#       serve_8767.py + requirements-pod.txt, but `docker build` first ships the
#       ENTIRE context to the builder — so without .dockerignore you upload ~2.1 GB
#       to the daemon on every build. On weak/hotspot internet + buildx this looks
#       like a hang and wastes time. Create `.dockerignore` NEXT TO this Dockerfile
#       with exactly:
#           *
#           !serve_8767.py
#           !requirements-pod.txt
#       (Allow-list form: ignore everything, then un-ignore only the 2 files copied.)
#
#   (B) THE MODEL MUST BE UPLOADED TO THE NETWORK VOLUME *BEFORE* THE FIRST REQUEST,
#       or every worker crashes in lifespan _load() with a FileNotFoundError and the
#       endpoint never goes healthy (you burn cold-start seconds on nothing). This
#       image is code-only ON PURPOSE (see WHY below) — it does NOT contain weights.
#
# WHY CODE-ONLY (no 640MB weights baked in):
#   The ~640MB model (hatformer-10h-naskh-best/model.safetensors = 667,903,136 B)
#   is NOT copied into this image. Baking it would make every `docker push` a
#   non-resumable multi-GB upload (torch+CUDA base + weights), re-pushed on every
#   code change. Instead the model lives on a RunPod NETWORK VOLUME (mounted at
#   /runpod-volume inside serverless workers), uploaded ONCE via the resumable S3
#   API (`aws s3 sync`, skips already-uploaded objects on a flaky hotspot). This
#   image stays tens-of-MB of extras only. The endpoint's env vars point
#   serve_8767.py at the volume mount (see RUNPOD CONSOLE below).
#
#   Upload the model ONCE (from a good-enough connection), the 3 SERVE dirs only —
#   do NOT upload dataset/ or the other two model variants (you'd triple the transfer):
#     aws s3 sync ~/Downloads/HATFormer/hatformer-10h-naskh-best \
#       s3://<VOLUME_ID>/hatformer-10h-naskh-best --region <DC> --endpoint-url https://s3api-<DC>.runpod.io/
#     aws s3 sync ~/Downloads/HATFormer/trocr \
#       s3://<VOLUME_ID>/trocr --region <DC> --endpoint-url https://s3api-<DC>.runpod.io/
#     aws s3 sync ~/Downloads/HATFormer/arabic_tokenizer_clean \
#       s3://<VOLUME_ID>/arabic_tokenizer_clean --region <DC> --endpoint-url https://s3api-<DC>.runpod.io/
#   (S3 key: Console → Settings → S3 API Keys. model.safetensors is 667MB > the
#    500MB single-PutObject cap → AWS CLI v2 auto-multiparts; if it errors use
#    RunPod's upload_large_file.py helper.)
#
# BUILD (MUST be amd64 — RunPod GPUs are x86; you are on Apple Silicon):
#   cd ~/Downloads/HATFormer   # build context = this folder (needs the .dockerignore above)
#   docker buildx build --platform linux/amd64 -t <dockerhub_user>/hatformer-lb:v1 --push .
#
# RUNPOD CONSOLE (Serverless → New Endpoint):
#   • Import from Docker Registry → <dockerhub_user>/hatformer-lb:v1
#   • Endpoint Type = "Load Balancer"     ← THE toggle (not Queue)
#   • GPU: cheapest 16GB tier (A4000 / RTX 4000 Ada) — 334M / ~3GB fp16 fits easily
#   • Active/Min workers = 0 (scale-to-zero, $0 idle) · Max workers = 1 (test)
#   • Idle timeout ≥ ~60s — the lane POSTs ONCE PER LINE CROP; a low idle timeout
#     spins the worker down between lines and RELOADS the 640MB model every time
#     (~15–40s each) → slow + burns cold-start compute. Keep it warm across a page.
#   • Execution timeout ≥ 240s (lane timeout is 180–240s)
#   • Expose HTTP Port = 8767   (must equal PORT below)
#   • Attach the Network Volume you uploaded the model to (Advanced → Network Volumes).
#     NOTE this PINS the endpoint to that volume's datacenter — pick a DC that has
#     BOTH the S3 API and the cheap GPU.
#   • Env vars (REQUIRED — these must match where you uploaded on the volume):
#       PORT=8767
#       PORT_HEALTH=8767
#       HATFORMER_HOST_DEVICE=cuda
#       HATFORMER_MODEL_DIR=/runpod-volume/hatformer-10h-naskh-best
#       HATFORMER_PROCESSOR_DIR=/runpod-volume/trocr
#       HATFORMER_TOKENIZER_FILE=/runpod-volume/arabic_tokenizer_clean/tokenizer.json
#   • Set HATFORMER_SERVICE_URL on V2 prod = https://<ENDPOINT_ID>.api.runpod.ai
#     (subdomain form — NOT the queue form api.runpod.ai/v2/<id>/runsync; the lane
#      POSTs raw multipart to /recognize, which only the LB subdomain serves).
#
# ⚠️ AUTH — THE REAL BLOCKER for "V2 lane works UNCHANGED" (cannot be fixed in this
#   image or in serve_8767.py; the container never sees a rejected request):
#   The api.runpod.ai LB proxy is documented to enforce RunPod API-key auth at the
#   edge. The V2 lane sends NO Authorization header (only Content-Type), so it will
#   very likely get 401 BEFORE reaching this container. VERIFY EMPIRICALLY FIRST
#   (~5 min, ~$0) after the endpoint is healthy:
#     curl -i https://<ENDPOINT_ID>.api.runpod.ai/ping                                    # no header → 200? or 401?
#     curl -i -H "Authorization: Bearer <KEY>" https://<ENDPOINT_ID>.api.runpod.ai/ping   # expect 200
#   • No-auth 200 → lane works unchanged; just set HATFORMER_SERVICE_URL. Done.
#   • No-auth 401/403 → add ONE header to the lane's /recognize POST:
#       Authorization: Bearer <scoped RunPod key>   (scope the key to THIS endpoint;
#       store as e.g. HATFORMER_SERVICE_TOKEN in prod .env). This is a small caller
#       edit — it currently sets only Content-Type.
#   • Zero-code-change fallback: the GPU POD flow in RUNPOD_TEST.md
#     (https://<POD_ID>-8767.proxy.runpod.net needs no key) — but a Pod does NOT
#     scale to zero (pay while running; Stop it manually).
# ─────────────────────────────────────────────────────────────────────────────

# CUDA torch is preinstalled in this base → only tiny extras to pip-install → small,
# fast push on weak internet.
#
# VERSION NOTE (verified against the model + the working local env, 2026-07-06):
#   • The checkpoint was saved with transformers 5.12.1 (config.json / generation_config.json).
#   • The recipe that PROVABLY works locally uses transformers 5.13.0 + torch 2.10.0.
#   • serve_8767.py passes VisionEncoderDecoderModel.from_pretrained(..., dtype=DTYPE):
#     the `dtype=` kwarg only exists on transformers 5.x (it is the replacement for
#     the old `torch_dtype=`). requirements-pod.txt pins `transformers>=4.53`, so a
#     fresh pip WILL resolve to the newest 5.x → `dtype=` works. But `>=4.53` is a
#     loose floor: if pip's resolver ever lands on a 4.x wheel, `dtype=` raises and
#     every /recognize 500s. Recommend tightening the pin to `transformers>=5.12`
#     in requirements-pod.txt so you can't silently get an incompatible major.
#   • transformers 5.13 requires torch>=2.4, so this base (torch 2.4.0) satisfies the
#     floor and pip will NOT try to reinstall torch. The verified combo is torch 2.10;
#     torch 2.4 is very likely fine for a 334M VisionEncoderDecoder but is one minor
#     you have not benchmarked — if you see garbled output vs. local, bump the base to
#     a runpod/pytorch 2.5/2.6 tag rather than debugging blind.
#   Confirm a live runpod/pytorch tag on Docker Hub at deploy time (tags rotate).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Fail fast, unbuffered logs (startup model-load / warmup prints show up live in RunPod).
# OFFLINE flags are safe here: serve_8767.py loads everything with local_files_only=True,
# so these only PREVENT an accidental silent Hugging Face fetch (which would hang the
# cold start and could cost time) — they never block the local volume load.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Extras only — torch already present in the base image (do NOT reinstall torch here;
# a plain `pip install torch` would pull the CPU wheel from PyPI and lose CUDA).
# requirements-pod.txt already omits torch and includes python-multipart (REQUIRED —
# without it FastAPI's UploadFile/Form parsing on /recognize 500s at import/first call).
COPY requirements-pod.txt .
RUN pip install --no-cache-dir -r requirements-pod.txt

# App code only. Model/processor/tokenizer come from the mounted network volume at
# runtime via the HATFORMER_*_DIR env vars — nothing model-sized is copied in.
COPY serve_8767.py .

# Document the intended port (RunPod actually routes via the injected PORT env var).
EXPOSE 8767

# RunPod injects PORT (and PORT_HEALTH). Bind uvicorn to $PORT via SHELL-FORM CMD so
# ${PORT} expands — an exec-form CMD would pass the literal string "${PORT}". Falls
# back to 8767 for local `docker run`. serve_8767.py has no __main__/uvicorn.run, so
# this CMD is the only launcher (correct for LB).
#
# HEALTH: serve_8767.py already exposes GET /ping (LB polls this, not /health). The
# model loads in the FastAPI lifespan BEFORE uvicorn accepts traffic, so by the time
# /ping can answer, the model is loaded → it returns 200. (Its not-ready branch
# returns 503; RunPod's LB treats 200=healthy / 204=initializing / else=unhealthy, so
# 503 reads as "unhealthy" rather than "still loading". This is fine as long as model
# load stays inside lifespan — the 503 branch is unreachable in normal operation. If
# you ever move load out of lifespan, change that 503 → 204 in serve_8767.py.)
CMD ["sh", "-c", "uvicorn serve_8767:app --host 0.0.0.0 --port ${PORT:-8767}"]