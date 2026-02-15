"""
SDXL / FLUX Image Generator — FastAPI Backend

4 modes:
  1. "lightning"        — SDXL Lightning 4-step (fastest, ~3-5s)
  2. "realvis_fast"     — RealVisXL V5.0 Lightning 5-step (fast + photorealistic, ~5-10s)
  3. "realvis_quality"  — RealVisXL V5.0 full 25-step (best SDXL photorealism, ~30-60s)
  4. "flux"             — FLUX.1 Schnell 4-step (best overall quality, ~60-90s)

POST /api/generate  → returns JSON with base64 image
POST /api/generate/raw → returns raw PNG bytes
GET  /api/health    → health check
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Image Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline management ──────────────────────────────────────────────
_pipelines: dict = {}
_lock = threading.Lock()
_device_info: dict = {}


def _get_device_info():
    """Detect device once."""
    global _device_info
    if _device_info:
        return _device_info

    import torch

    if torch.backends.mps.is_available():
        _device_info = {"device": "mps", "dtype": torch.float32}
        logger.info("Device: Apple Silicon (MPS) float32")
    elif torch.cuda.is_available():
        _device_info = {"device": "cuda", "dtype": torch.float16}
        logger.info("Device: CUDA float16")
    else:
        _device_info = {"device": "cpu", "dtype": torch.float32}
        logger.info("Device: CPU float32 (will be slow)")

    return _device_info


def _disable_safety(pipe):
    """Remove any safety checkers from a pipeline."""
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


# ── 1. SDXL Lightning 4-step ─────────────────────────────────────────

def get_lightning_pipeline():
    """SDXL Lightning 4-step — fastest option."""
    if "lightning" in _pipelines:
        return _pipelines["lightning"]

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    info = _get_device_info()
    device, dtype = info["device"], info["dtype"]

    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
    LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"

    logger.info("Loading SDXL Lightning (4-step)...")
    unet_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)
    unet = UNet2DConditionModel.from_pretrained(SDXL_BASE, subfolder="unet", torch_dtype=dtype)
    unet.load_state_dict(load_file(unet_path), strict=False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE, unet=unet, torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    _disable_safety(pipe)

    if device == "mps":
        pipe.vae = pipe.vae.to(torch.float32)

    pipe._device_name = device
    _pipelines["lightning"] = pipe
    logger.info(f"SDXL Lightning (4-step) loaded on {device} ✓")
    return pipe


# ── 2. RealVisXL V5 Lightning 5-step ─────────────────────────────────

def get_realvis_fast_pipeline():
    """RealVisXL V5.0 Lightning — fast photorealistic, 5 steps."""
    if "realvis_fast" in _pipelines:
        return _pipelines["realvis_fast"]

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    info = _get_device_info()
    device, dtype = info["device"], info["dtype"]

    MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"

    logger.info("Loading RealVisXL V5.0 Lightning (5-step)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    # Euler Discrete with trailing spacing — safe on MPS, works well with distilled models
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    _disable_safety(pipe)

    if device == "mps":
        pipe.vae = pipe.vae.to(torch.float32)

    pipe._device_name = device
    _pipelines["realvis_fast"] = pipe
    logger.info(f"RealVisXL V5.0 Lightning loaded on {device} ✓")
    return pipe


# ── 3. RealVisXL V5 Quality 25-step ──────────────────────────────────

def get_realvis_quality_pipeline():
    """RealVisXL V5.0 — best SDXL photorealism, 20-30 steps."""
    if "realvis_quality" in _pipelines:
        return _pipelines["realvis_quality"]

    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    info = _get_device_info()
    device, dtype = info["device"], info["dtype"]

    MODEL_ID = "SG161222/RealVisXL_V5.0"

    logger.info("Loading RealVisXL V5.0 (quality mode)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    # Euler Discrete — most stable scheduler on MPS
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    _disable_safety(pipe)

    if device == "mps":
        pipe.vae = pipe.vae.to(torch.float32)

    pipe._device_name = device
    _pipelines["realvis_quality"] = pipe
    logger.info(f"RealVisXL V5.0 (quality) loaded on {device} ✓")
    return pipe


# ── 4. FLUX.1 Schnell 4-step ─────────────────────────────────────────

def get_flux_pipeline():
    """FLUX.1 Schnell — best overall quality, 4 steps, ~60-90s on MPS."""
    if "flux" in _pipelines:
        return _pipelines["flux"]

    import torch
    from diffusers import FluxPipeline

    info = _get_device_info()
    device = info["device"]

    MODEL_ID = "black-forest-labs/FLUX.1-schnell"

    # FLUX uses bfloat16 natively; on MPS we try bfloat16 first, fallback to float32
    if device == "mps":
        try:
            flux_dtype = torch.bfloat16
            logger.info("Loading FLUX.1 Schnell in bfloat16 for MPS...")
        except Exception:
            flux_dtype = torch.float32
            logger.info("Loading FLUX.1 Schnell in float32 for MPS (bfloat16 fallback)...")
    elif device == "cuda":
        flux_dtype = torch.bfloat16
    else:
        flux_dtype = torch.float32

    logger.info(f"Loading FLUX.1 Schnell ({flux_dtype})... (this may take a while, ~34GB model)")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, torch_dtype=flux_dtype,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    pipe._device_name = device
    _pipelines["flux"] = pipe
    logger.info(f"FLUX.1 Schnell loaded on {device} ✓")
    return pipe


# ── Helpers ──────────────────────────────────────────────────────────

def trim_prompt(text: str, max_tokens: int = 70) -> str:
    """Trim prompt to stay within CLIP's 77-token limit."""
    words = text.split()
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    last_comma = trimmed.rfind(",")
    if last_comma > len(trimmed) * 0.6:
        trimmed = trimmed[:last_comma]
    return trimmed


# ── Image History (persistent disk storage) ───────────────────────────

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def save_to_history(
    img_bytes: bytes, prompt: str, negative: str, seed: int,
    width: int, height: int, model_mode: str,
    guidance_scale: float, num_inference_steps: int, elapsed: float,
    custom_filename: Optional[str] = None,
) -> str:
    """Save a generated image + metadata JSON to the generated/ folder."""
    ts = int(time.time() * 1000)
    img_name = custom_filename or f"{ts}_{seed}_{model_mode}.png"
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        img_name += ".png"

    (GENERATED_DIR / img_name).write_bytes(img_bytes)

    meta = {
        "filename": img_name, "prompt": prompt, "negative_prompt": negative,
        "seed": seed, "width": width, "height": height, "model_mode": model_mode,
        "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps,
        "time_seconds": elapsed, "timestamp": ts,
    }
    (GENERATED_DIR / f"{img_name}.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved to history: {img_name}")
    return img_name


# ── Request / Response ────────────────────────────────────────────────

class ModelMode(str, Enum):
    lightning = "lightning"
    realvis_fast = "realvis_fast"
    realvis_quality = "realvis_quality"
    flux = "flux"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    seed: Optional[int] = Field(default=None, description="Random seed. None = random.")
    guidance_scale: float = Field(default=0, ge=0, le=20, description="CFG scale.")
    num_inference_steps: int = Field(default=4, ge=1, le=50, description="Inference steps.")
    model_mode: ModelMode = Field(default=ModelMode.lightning)


class GenerateResponse(BaseModel):
    image: str  # base64-encoded PNG
    seed: int
    width: int
    height: int
    time_seconds: float
    model_mode: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    loaded = list(_pipelines.keys())
    return {
        "status": "ok",
        "models_loaded": loaded,
        "available_modes": [m.value for m in ModelMode],
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image from a text prompt."""
    t0 = time.time()

    try:
        img_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            None, _generate_sync,
            req.prompt, req.negative_prompt,
            req.width, req.height, req.seed,
            req.guidance_scale, req.num_inference_steps,
            req.model_mode.value,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    logger.info(
        f"Generated {req.width}x{req.height} in {elapsed}s "
        f"(mode={req.model_mode.value}, seed={used_seed}, "
        f"steps={req.num_inference_steps}, cfg={req.guidance_scale})"
    )

    # Save to disk history
    try:
        save_to_history(
            img_bytes=img_bytes, prompt=req.prompt, negative=req.negative_prompt,
            seed=used_seed, width=req.width, height=req.height,
            model_mode=req.model_mode.value, guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps, elapsed=elapsed,
        )
    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")

    return GenerateResponse(
        image=b64,
        seed=used_seed,
        width=req.width,
        height=req.height,
        time_seconds=elapsed,
        model_mode=req.model_mode.value,
    )


@app.post("/api/generate/raw")
async def generate_raw(req: GenerateRequest):
    """Generate an image and return raw PNG bytes."""
    try:
        img_bytes, _ = await asyncio.get_event_loop().run_in_executor(
            None, _generate_sync,
            req.prompt, req.negative_prompt,
            req.width, req.height, req.seed,
            req.guidance_scale, req.num_inference_steps,
            req.model_mode.value,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return Response(content=img_bytes, media_type="image/png")


# ── Generation logic ─────────────────────────────────────────────────

def _generate_sync(
    prompt: str, negative: str, width: int, height: int,
    seed: Optional[int], guidance_scale: float, num_inference_steps: int,
    model_mode: str,
) -> tuple:
    """Route to the correct generation function."""
    used_seed = seed if seed is not None else random.randint(1, 2**31)

    if model_mode == "flux":
        return _generate_flux(prompt, negative, width, height, used_seed, num_inference_steps)
    elif model_mode == "realvis_fast":
        return _generate_sdxl(get_realvis_fast_pipeline, prompt, negative, width, height, used_seed, guidance_scale, num_inference_steps)
    elif model_mode == "realvis_quality":
        return _generate_sdxl(get_realvis_quality_pipeline, prompt, negative, width, height, used_seed, guidance_scale, num_inference_steps)
    else:  # lightning
        return _generate_sdxl(get_lightning_pipeline, prompt, negative, width, height, used_seed, guidance_scale, num_inference_steps)


def _generate_sdxl(
    get_pipeline_fn, prompt: str, negative: str, width: int, height: int,
    used_seed: int, guidance_scale: float, num_inference_steps: int,
) -> tuple:
    """Generate with any SDXL-based pipeline (Lightning, RealVisXL variants)."""
    import torch

    pipeline = get_pipeline_fn()
    if pipeline is None:
        raise RuntimeError("Pipeline failed to load")

    device = pipeline._device_name
    gen = torch.Generator("cpu" if device == "mps" else device).manual_seed(used_seed)

    prompt = trim_prompt(prompt, max_tokens=70)
    negative = trim_prompt(negative, max_tokens=70) if negative else ""

    with _lock:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative if negative else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=gen,
        )

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), used_seed


def _generate_flux(
    prompt: str, negative: str, width: int, height: int,
    used_seed: int, num_inference_steps: int,
) -> tuple:
    """Generate with FLUX.1 Schnell. Different API than SDXL pipelines."""
    import torch

    pipeline = get_flux_pipeline()
    if pipeline is None:
        raise RuntimeError("FLUX pipeline failed to load")

    device = pipeline._device_name
    gen = torch.Generator("cpu" if device == "mps" else device).manual_seed(used_seed)

    logger.info(f"  FLUX generating {width}x{height}, {num_inference_steps} steps...")

    with _lock:
        result = pipeline(
            prompt=prompt,
            # FLUX Schnell: guidance_scale MUST be 0 (timestep-distilled)
            guidance_scale=0.0,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            max_sequence_length=256,
            generator=gen,
        )

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), used_seed


# ── History Endpoints ─────────────────────────────────────────────────

@app.get("/api/history")
async def list_history(limit: int = 200, offset: int = 0):
    """List saved images from the generated/ folder (newest first)."""
    if not GENERATED_DIR.exists():
        return {"images": [], "total": 0}

    meta_files = sorted(GENERATED_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    total = len(meta_files)
    page = meta_files[offset : offset + limit]

    images = []
    for mf in page:
        try:
            meta = json.loads(mf.read_text())
            if (GENERATED_DIR / meta["filename"]).exists():
                images.append(meta)
        except Exception:
            continue

    return {"images": images, "total": total}


@app.get("/api/history/image/{filename}")
async def get_history_image(filename: str):
    """Serve a saved image from the generated/ folder."""
    safe = Path(filename).name
    img_path = GENERATED_DIR / safe
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(404, "Image not found")

    suffix = img_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return Response(content=img_path.read_bytes(), media_type=media_types.get(suffix, "image/png"))


@app.delete("/api/history/{filename}")
async def delete_history_image(filename: str):
    """Delete a saved image and its metadata."""
    safe = Path(filename).name
    img_path = GENERATED_DIR / safe
    meta_path = GENERATED_DIR / f"{safe}.json"

    deleted = False
    if img_path.exists():
        img_path.unlink()
        deleted = True
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    if not deleted:
        raise HTTPException(404, "Image not found")
    return {"deleted": safe}


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    Image Generator                            ║
╠═══════════════════════════════════════════════════════════════╣
║  Server:  http://0.0.0.0:{port}                                  ║
║  Docs:    http://0.0.0.0:{port}/docs                              ║
║  Health:  http://0.0.0.0:{port}/api/health                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Modes:                                                       ║
║    lightning        — SDXL Lightning 4-step      (~3-5s)      ║
║    realvis_fast     — RealVisXL V5 Lightning     (~5-10s)     ║
║    realvis_quality  — RealVisXL V5 25-step       (~30-60s)    ║
║    flux             — FLUX.1 Schnell 4-step      (~60-90s)    ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
