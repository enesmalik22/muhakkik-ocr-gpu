"""HATFormer host inference servisi — V2 dev worker'ın çağırdığı :8767 servisi.

Dev compose HATFORMER_SERVICE_URL=http://host.docker.internal:8767 ile bu servisi
çağırır (model container'da değil host'ta koşar → emülasyon yavaşlığı yok).
Lane sözleşmesi: POST /recognize  (multipart: image=PNG, max_new_tokens=int)
                 -> JSON {text, confidence, model, device, provider, model_dir}

Çalıştırma (baybars venv'inde torch+transformers+fastapi var):
  ~/baybars-local-test/venv/bin/python -m uvicorn serve_8767:app \
      --host 0.0.0.0 --port 8767 --app-dir ~/Downloads/HATFormer

Reçete = pipelines/09-ocr/.../hatformer_naskh_recognition.py ile BİREBİR.
Cihaz cpu/float32 = prod ile birebir çıktı (HATFORMER_HOST_DEVICE=mps ile hızlandırılabilir).
"""
from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image, ImageOps
from transformers import PreTrainedTokenizerFast, TrOCRProcessor, VisionEncoderDecoderModel

_H = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("HATFORMER_MODEL_DIR", str(_H / "hatformer-10h-naskh-best")))
PROC_DIR = Path(os.getenv("HATFORMER_PROCESSOR_DIR", str(_H / "trocr")))
TOK_FILE = Path(os.getenv("HATFORMER_TOKENIZER_FILE", str(_H / "arabic_tokenizer_clean" / "tokenizer.json")))
DEVICE = os.getenv("HATFORMER_HOST_DEVICE", "cpu").strip().lower()  # cpu=prod-birebir
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

_STATE: dict = {}
_lock = threading.Lock()  # generate seri olmalı


def _load() -> None:
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(str(PROC_DIR), local_files_only=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOK_FILE))
    tokenizer.add_special_tokens(
        {"pad_token": "<pad>", "eos_token": "</s>", "cls_token": "<s>", "bos_token": "<s>"}
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        str(MODEL_DIR), dtype=DTYPE, local_files_only=True, low_cpu_mem_usage=True
    )
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.to(DEVICE)
    model.eval()
    _STATE.update(processor=processor, tokenizer=tokenizer, model=model)
    print(f"[hatformer:8767] loaded in {time.time()-t0:.1f}s device={DEVICE} model={MODEL_DIR.name}", flush=True)


def _prepare(image: Image.Image, processor) -> torch.Tensor:
    img = ImageOps.exif_transpose(image).convert("RGB")
    ow, oh = img.size
    nh = 64
    nw = max(1, int(nh * (ow / max(1, oh))))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    fw, fh = 384, 384
    canvas = Image.new("RGB", (fw, fh), (0, 0, 0))
    if resized.width <= fw:
        canvas.paste(resized, (0, 0))
    else:
        seg_w = fw
        n = min(fh // nh, (resized.width + seg_w - 1) // seg_w)
        for i in range(n):
            l = i * seg_w
            r = min(l + seg_w, resized.width)
            canvas.paste(resized.crop((l, 0, r, nh)), (0, i * nh))
    return processor(canvas, return_tensors="pt").pixel_values[0]


def _decode(ids: list[int], tokenizer) -> str:
    vs = len(tokenizer)
    return tokenizer.decode([t for t in ids if 0 <= int(t) < vs], skip_special_tokens=True).strip()


def _warmup() -> None:
    """MPS/GPU ilk-çağrı kernel derlemesini (~12s) startup'a çek → ilk gerçek OCR yavaş olmasın."""
    try:
        t0 = time.time()
        blank = Image.new("RGB", (400, 64), (255, 255, 255))
        pv = _prepare(blank, _STATE["processor"]).unsqueeze(0).to(DEVICE)
        if DTYPE == torch.float16:
            pv = pv.half()
        tok = _STATE["tokenizer"]
        with torch.inference_mode():
            _STATE["model"].generate(
                pv, num_beams=1, max_new_tokens=8,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                decoder_start_token_id=tok.bos_token_id,
            )
        print(f"[hatformer:8767] warmup done in {time.time()-t0:.1f}s device={DEVICE}", flush=True)
    except Exception as exc:  # warmup ASLA startup'ı bozmaz
        print(f"[hatformer:8767] warmup skipped: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    _warmup()
    yield
    _STATE.clear()


app = FastAPI(title="HATFormer host service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _STATE.get("model") is not None else "loading", "device": DEVICE, "model": MODEL_DIR.name}


@app.get("/ping")
def ping() -> PlainTextResponse:
    """RunPod Load Balancer sağlık probu: 200=hazır, 204=başlatılıyor.
    (RunPod 200'ü sağlıklı, 204'ü initializing sayar; başka kod → worker evict.)
    Model lifespan'de (uvicorn serve etmeye başlamadan ÖNCE) yüklendiği için,
    servis istek almaya başladığında model zaten yüklü → 200 döner."""
    if _STATE.get("model") is not None:
        return PlainTextResponse("ok", status_code=200)
    return PlainTextResponse("", status_code=204)


@app.post("/recognize")
async def recognize(image: UploadFile = File(...), max_new_tokens: str = Form("128")) -> JSONResponse:
    if _STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="model loading")
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad image: {exc}")

    processor = _STATE["processor"]
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    try:
        mnt = max(1, int(max_new_tokens))
    except (TypeError, ValueError):
        mnt = 128

    pv = _prepare(img, processor).unsqueeze(0).to(DEVICE)
    if DTYPE == torch.float16:
        pv = pv.half()

    t0 = time.time()
    with _lock, torch.inference_mode():
        out = model.generate(
            pv,
            num_beams=1,
            max_new_tokens=mnt,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
    text = _decode(out[0].tolist(), tokenizer)
    return JSONResponse(
        {
            "text": text,
            "confidence": 0.75,
            "model": "hatformer-10h-naskh-best",
            "model_dir": str(MODEL_DIR),
            "device": DEVICE,
            "provider": "host-service",
            "latency_ms": round((time.time() - t0) * 1000),
        }
    )
