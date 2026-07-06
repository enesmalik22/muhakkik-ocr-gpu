"""RunPod Serverless handler — Arabic handwritten line OCR (Qwen2.5-VL-3B + LoRA).

Mirrors the PROVEN local (MPS) inference recipe EXACTLY, but on CUDA/fp16.
Model + processor + LoRA adapter are loaded ONCE at module import (cold start)
and reused across every invocation, per RunPod best practice.

Contract (queue-based endpoint):
    request  -> job["input"]["images"]  : list[str]  base64-encoded line crops
                                          (optional "data:image/...;base64," prefix tolerated)
                job["input"]["prompt"]   : str (optional; defaults to the exact Arabic PROMPT)
    response -> {"texts": [str, ...]}    : one decoded string per input image, SAME order, 1:1.
                On per-image failure the slot is "" and an "errors" list carries {index, message}.

Batching contract: one HTTP request per manuscript PAGE carrying an ordered list of
line crops. We loop per-image internally using the single-image recipe (correctness over
throughput — no risky batched multimodal generate). Cold start is amortised across the page.
"""

import os

# --------------------------------------------------------------------------- #
# Offline switch — MUST be set BEFORE `import transformers` / `huggingface_hub` #
# so the offline flag is honoured the moment those libs read the environment.  #
# Belt-and-suspenders alongside the explicit local_files_only=True we pass to  #
# from_pretrained: the CUDA/offline research recommends setting BOTH the env    #
# vars AND local_files_only so no code path silently hits the network at        #
# runtime. If you deliberately want to pull the base from HF at container       #
# start, set HF_HUB_OFFLINE=0 in the endpoint env (adds a ~7.5 GB cold-start    #
# download).                                                                    #
# --------------------------------------------------------------------------- #
_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "1") != "0"
if _OFFLINE:
    # Set both flags; do not clobber an explicit "0" the operator may have set.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import base64
import binascii
import io
import sys
import time
import traceback

import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

import runpod

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# The 7.5 GB base is BAKED into the image at build time (Dockerfile pulls it from
# HF on RunPod's builders). At runtime we load it from the local baked dir with no
# network access, so cold starts never re-download 7.5 GB. Override BASE_MODEL_PATH
# via a runtime env var only if you relocate the bake target.
#
# NOTE (deviation from the literal recipe): the recipe wrote local_files_only=False
# "so base pulls from HF". Because the hard requirement is to BAKE the base into the
# image, we point BASE at the baked local dir and load offline (local_files_only=True).
# This is the change the CUDA/offline research explicitly recommends; behaviour of the
# loaded weights is identical. If BASE_MODEL_PATH is set to a HF repo id and
# HF_HUB_OFFLINE=0, it will fall back to pulling from HF (slow cold start).
#
# IMPORTANT (Dockerfile responsibility): the baked /models/base dir MUST contain the
# FULL snapshot — weights + config.json + generation_config.json + tokenizer files +
# preprocessor_config.json + chat_template (chat_template.json or tokenizer_config.json).
# With local_files_only=True, a missing chat_template.json makes apply_chat_template
# fail. Bake with the whole repo, e.g.:
#   hf download sherif1313/Arabic-English-handwritten-OCR-v3 --local-dir /models/base
# (exclude only sample images, never the *.json config/template files).
# Accept both the Dockerfile's env names (BASE_MODEL_DIR / ADAPTER_DIR) and the
# older *_PATH names; fall back to the baked/COPYd defaults. Defaults match the
# Dockerfile (bake dir /models/base, COPY dir /app/adapter) so a no-env deploy works.
BASE = os.environ.get("BASE_MODEL_DIR") or os.environ.get("BASE_MODEL_PATH") or "/models/base"
ADAPTER = os.environ.get("ADAPTER_DIR") or os.environ.get("ADAPTER_PATH") or "/app/adapter"

DEVICE = "cuda"

# EXACT Arabic prompt — DO NOT ALTER.
PROMPT = (
    "اقرأ النص العربي المخطوط في هذه الصورة وانسخه كما هو فقط. "
    "حافظ على ترتيب الكلمات، ولا تشرح، ولا تترجم، ولا تضف كلمات غير موجودة."
)

# Generation limits (mirror the recipe exactly).
MAX_NEW_TOKENS = 160

# Image-prep bounds (mirror the recipe exactly).
_MAX_W = 896
_MAX_H = 192


def _log(msg: str) -> None:
    """Unbuffered stdout so lines show up in RunPod logs immediately."""
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Cold-start model load (runs ONCE at module import)                          #
# --------------------------------------------------------------------------- #

_LOAD_ERROR = None  # populated if load fails; handler will surface it per-request.
processor = None
model = None


def _load_model():
    global processor, model

    t0 = time.time()
    _log(f"[startup] python={sys.version.split()[0]} torch={torch.__version__}")
    _log(f"[startup] cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            _log(f"[startup] cuda.device={torch.cuda.get_device_name(0)}")
        except Exception as exc:  # pragma: no cover - informational only
            _log(f"[startup] cuda.device name unavailable: {exc}")
    else:
        # No GPU => this endpoint cannot serve. Fail loudly so RunPod logs show it.
        raise RuntimeError(
            "CUDA is not available at startup — this handler requires a GPU worker. "
            "Check the endpoint's GPU configuration."
        )

    _log(f"[startup] BASE={BASE} ADAPTER={ADAPTER} offline={_OFFLINE}")

    # Processor.
    processor = AutoProcessor.from_pretrained(
        BASE,
        trust_remote_code=True,
        local_files_only=_OFFLINE,
    )

    # Base model.
    # dtype=torch.float16 is the correct transformers-5.x API (the recipe's value,
    # verbatim). We pass it explicitly because v5's default dtype is "auto" (= the
    # config's saved bf16); without this it would load bf16, not fp16.
    # low_cpu_mem_usage=True is a silent no-op in v5 but kept for recipe parity.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=_OFFLINE,
    )

    # Tie-weight fixups — mirror the recipe exactly. This model's config has
    # top-level tie_word_embeddings=None but text_config.tie_word_embeddings=True,
    # and v5 changed tied-weight handling, so this manual re-tie is a needed
    # safeguard (prevents an untied lm_head after PEFT wrapping).
    model.config.tie_word_embeddings = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.tie_word_embeddings = True
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    model.config.use_cache = True
    # Saved generation_config has do_sample=true, temperature=1e-06; null out
    # temperature and force greedy at call time for deterministic OCR.
    model.generation_config.temperature = None

    # LoRA adapter (14 MB, committed in-repo, COPYd into the image).
    model = PeftModel.from_pretrained(model, ADAPTER)

    model.to(DEVICE)
    model.eval()

    _log(f"[startup] model+adapter loaded on {DEVICE} in {time.time() - t0:.1f}s")


try:
    _load_model()
except Exception as exc:  # noqa: BLE001 - we want ANY load failure recorded.
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    _log("[startup][FATAL] model failed to load:")
    _log(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Per-image helpers (mirror the recipe exactly)                               #
# --------------------------------------------------------------------------- #

def _decode_base64_image(b64: str) -> Image.Image:
    """Decode a base64 string (optionally with a data: URI prefix) to a PIL image."""
    if not isinstance(b64, str):
        raise ValueError(f"image must be a base64 string, got {type(b64).__name__}")
    s = b64.strip()
    if not s:
        raise ValueError("empty image string")
    if s.startswith("data:"):
        # data:image/png;base64,XXXX  -> keep only the part after the comma.
        comma = s.find(",")
        if comma == -1:
            raise ValueError("malformed data: URI (no comma)")
        s = s[comma + 1 :]
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64: {exc}") from exc
    if not raw:
        raise ValueError("base64 decoded to zero bytes")
    img = Image.open(io.BytesIO(raw))
    img.load()  # force decode now so a corrupt image raises here, not later.
    return img


def _prep_image(img: Image.Image) -> Image.Image:
    """EXACT image prep from the proven recipe."""
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Fit within max 896x192, never upscale in this step (scale capped at 1.0).
    w, h = img.size
    scale = min(_MAX_W / w, _MAX_H / h, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # Upscale x2 if too small (helps tiny crops).
    if img.height < 96 or max(img.size) < 900:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    # Sharpen.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    return img


@torch.inference_mode()
def _infer_one(img: Image.Image, prompt: str) -> str:
    """EXACT generation from the proven recipe (single image)."""
    qmsgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        qmsgs, tokenize=False, add_generation_prompt=True
    )
    batch = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    generated = model.generate(
        **batch,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_return_sequences=1,
    )
    trimmed = generated[:, batch["input_ids"].shape[1] :]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return out


# --------------------------------------------------------------------------- #
# Handler                                                                     #
# --------------------------------------------------------------------------- #

def handler(job):
    t_start = time.time()

    # If the model never loaded, every request must fail loudly (RAISE -> RunPod
    # marks the job FAILED and logs the traceback).
    if _LOAD_ERROR is not None or model is None or processor is None:
        raise RuntimeError(f"Model unavailable at startup: {_LOAD_ERROR}")

    job_input = job.get("input") or {}
    images = job_input.get("images")
    prompt = job_input.get("prompt") or PROMPT

    # Input validation -> graceful error (client mistake, not a worker failure).
    if images is None:
        return {"error": "missing 'images' in job['input'] (expected a list of base64 strings)"}
    if not isinstance(images, list):
        return {"error": f"'images' must be a list, got {type(images).__name__}"}
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "'prompt' must be a non-empty string when provided"}

    n = len(images)
    _log(f"[handler] job={job.get('id')} device={DEVICE} images={n}")

    if n == 0:
        _log("[handler] empty image list — returning empty texts")
        return {"texts": []}

    texts = []
    errors = []
    for i, b64 in enumerate(images):
        img_t0 = time.time()
        try:
            img = _decode_base64_image(b64)
            img = _prep_image(img)
            out = _infer_one(img, prompt)
            texts.append(out)
            _log(
                f"[handler]   [{i + 1}/{n}] ok chars={len(out)} "
                f"t={time.time() - img_t0:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001 - never let one crop kill the page.
            texts.append("")
            msg = f"{type(exc).__name__}: {exc}"
            errors.append({"index": i, "message": msg})
            _log(f"[handler]   [{i + 1}/{n}] FAILED {msg}")
            _log(traceback.format_exc())
            # A CUDA OOM (or any CUDA error) can leave the allocator/context in a
            # degraded state that spuriously fails EVERY subsequent crop on the page.
            # Reclaim the cache so one oversized crop does not poison the rest of the
            # page. Guarded + best-effort; never let cleanup itself kill the loop.
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - cleanup must never raise.
                pass

    result = {"texts": texts}
    if errors:
        result["errors"] = errors

    _log(
        f"[handler] job={job.get('id')} done images={n} "
        f"ok={n - len(errors)} err={len(errors)} total={time.time() - t_start:.2f}s"
    )
    return result


runpod.serverless.start({"handler": handler})
