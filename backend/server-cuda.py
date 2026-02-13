"""
SDXL / FLUX Image Generator — FastAPI Backend (CUDA optimized)

Optimized for NVIDIA GPUs (RTX 3090, A100, etc.)
- FP16 for all SDXL models (fast, no precision issues)
- BF16 for FLUX with device_map across 2 GPUs
- xformers memory-efficient attention
- No MPS workarounds

4 modes:
  1. "lightning"        — SDXL Lightning 4-step      (~0.5-1s)
  2. "realvis_fast"     — RealVisXL V5.0 Lightning   (~1-2s)
  3. "realvis_quality"  — RealVisXL V5.0 25-step     (~5-8s)
  4. "flux"             — FLUX.1 Schnell 4-step      (~3-5s)

POST /api/generate  → returns JSON with base64 image
POST /api/generate/raw → returns raw PNG bytes
GET  /api/health    → health check

Usage:
  python server-cuda.py                          # default: GPU 0, port 8100
  CUDA_VISIBLE_DEVICES=0 python server-cuda.py   # specific GPU
  FLUX_GPUS=0,1 python server-cuda.py            # FLUX across 2 GPUs
"""

import asyncio
import base64
import io
import logging
import os
import random
import threading
import time
from enum import Enum

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Verify CUDA ───────────────────────────────────────────────────────
if not torch.cuda.is_available():
    logger.error("CUDA is not available! This server requires NVIDIA GPUs.")
    logger.error("Use server.py instead for MPS/CPU.")
    raise SystemExit(1)

_num_gpus = torch.cuda.device_count()
logger.info(f"CUDA available: {_num_gpus} GPU(s) detected")
for i in range(_num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Image Generator (CUDA)")
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

# Which GPU to use for SDXL models (default: 0)
SDXL_DEVICE = f"cuda:{os.getenv('SDXL_GPU', '0')}"
# Which GPUs to use for FLUX (default: "0,1" if 2+ GPUs, else "0")
FLUX_GPUS = os.getenv("FLUX_GPUS", "0,1" if _num_gpus >= 2 else "0")
FLUX_GPU_LIST = [int(x.strip()) for x in FLUX_GPUS.split(",")]

DTYPE = torch.float16  # FP16 for all SDXL models on CUDA


def _disable_safety(pipe):
    """Remove any safety checkers from a pipeline."""
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _enable_fast_attention(pipe):
    """Enable xformers or SDPA for fastest inference."""
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("  → xformers memory-efficient attention enabled")
    except Exception:
        # PyTorch 2.0+ SDPA is used automatically as fallback
        logger.info("  → Using PyTorch SDPA attention (xformers not available)")
    return pipe


# ── 1. SDXL Lightning 4-step ─────────────────────────────────────────

def get_lightning_pipeline():
    """SDXL Lightning 4-step — fastest option."""
    if "lightning" in _pipelines:
        return _pipelines["lightning"]

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
    LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"

    logger.info(f"Loading SDXL Lightning (4-step) on {SDXL_DEVICE}...")
    unet_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)
    unet = UNet2DConditionModel.from_pretrained(SDXL_BASE, subfolder="unet", torch_dtype=DTYPE)
    unet.load_state_dict(load_file(unet_path), strict=False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE, unet=unet, torch_dtype=DTYPE, variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(SDXL_DEVICE)
    _enable_fast_attention(pipe)
    _disable_safety(pipe)

    pipe._device_name = SDXL_DEVICE
    _pipelines["lightning"] = pipe
    logger.info(f"SDXL Lightning (4-step) loaded on {SDXL_DEVICE} ✓")
    return pipe


# ── 2. RealVisXL V5 Lightning 5-step ─────────────────────────────────

def get_realvis_fast_pipeline():
    """RealVisXL V5.0 Lightning — fast photorealistic, 5 steps."""
    if "realvis_fast" in _pipelines:
        return _pipelines["realvis_fast"]

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"

    logger.info(f"Loading RealVisXL V5.0 Lightning (5-step) on {SDXL_DEVICE}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(SDXL_DEVICE)
    _enable_fast_attention(pipe)
    _disable_safety(pipe)

    pipe._device_name = SDXL_DEVICE
    _pipelines["realvis_fast"] = pipe
    logger.info(f"RealVisXL V5.0 Lightning loaded on {SDXL_DEVICE} ✓")
    return pipe


# ── 3. RealVisXL V5 Quality 25-step ──────────────────────────────────

def get_realvis_quality_pipeline():
    """RealVisXL V5.0 — best SDXL photorealism, 20-30 steps."""
    if "realvis_quality" in _pipelines:
        return _pipelines["realvis_quality"]

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    MODEL_ID = "SG161222/RealVisXL_V5.0"

    logger.info(f"Loading RealVisXL V5.0 (quality mode) on {SDXL_DEVICE}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(SDXL_DEVICE)
    _enable_fast_attention(pipe)
    _disable_safety(pipe)

    pipe._device_name = SDXL_DEVICE
    _pipelines["realvis_quality"] = pipe
    logger.info(f"RealVisXL V5.0 (quality) loaded on {SDXL_DEVICE} ✓")
    return pipe


# ── 4. FLUX.1 Schnell 4-step ─────────────────────────────────────────

def get_flux_pipeline():
    """FLUX.1 Schnell — best overall quality, 4 steps.
    Uses device_map to split across multiple GPUs if available.
    """
    if "flux" in _pipelines:
        return _pipelines["flux"]

    from diffusers import FluxPipeline

    MODEL_ID = "black-forest-labs/FLUX.1-schnell"
    flux_dtype = torch.bfloat16  # FLUX native dtype

    if len(FLUX_GPU_LIST) > 1:
        # Split across multiple GPUs using device_map
        logger.info(
            f"Loading FLUX.1 Schnell (bfloat16) across GPUs {FLUX_GPU_LIST}..."
        )
        # Create a device map that balances across specified GPUs
        max_memory = {i: "22GiB" for i in FLUX_GPU_LIST}
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=flux_dtype,
            device_map="balanced",
            max_memory=max_memory,
        )
    else:
        # Single GPU
        device = f"cuda:{FLUX_GPU_LIST[0]}"
        logger.info(f"Loading FLUX.1 Schnell (bfloat16) on {device}...")
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID, torch_dtype=flux_dtype,
        )
        pipe = pipe.to(device)

    _enable_fast_attention(pipe)

    # Store which device for generator seeding
    pipe._device_name = f"cuda:{FLUX_GPU_LIST[0]}"
    pipe._multi_gpu = len(FLUX_GPU_LIST) > 1
    _pipelines["flux"] = pipe
    logger.info(f"FLUX.1 Schnell loaded on GPU(s) {FLUX_GPU_LIST} ✓")
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
        "device": "cuda",
        "gpu_count": _num_gpus,
        "sdxl_device": SDXL_DEVICE,
        "flux_gpus": FLUX_GPU_LIST,
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
    pipeline = get_pipeline_fn()
    if pipeline is None:
        raise RuntimeError("Pipeline failed to load")

    device = pipeline._device_name
    gen = torch.Generator(device).manual_seed(used_seed)

    prompt = trim_prompt(prompt, max_tokens=70)
    negative = trim_prompt(negative, max_tokens=70) if negative else ""

    with _lock, torch.inference_mode():
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
    """Generate with FLUX.1 Schnell."""
    pipeline = get_flux_pipeline()
    if pipeline is None:
        raise RuntimeError("FLUX pipeline failed to load")

    device = pipeline._device_name
    gen = torch.Generator(device).manual_seed(used_seed)

    logger.info(f"  FLUX generating {width}x{height}, {num_inference_steps} steps...")

    with _lock, torch.inference_mode():
        result = pipeline(
            prompt=prompt,
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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              Image Generator  (CUDA)                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Server:  http://0.0.0.0:{port}                                  ║
║  Docs:    http://0.0.0.0:{port}/docs                              ║
║  Health:  http://0.0.0.0:{port}/api/health                         ║
╠═══════════════════════════════════════════════════════════════╣
║  GPUs:    {_num_gpus} detected                                        ║
║  SDXL:    {SDXL_DEVICE}  (FP16)                                   ║
║  FLUX:    GPU(s) {FLUX_GPU_LIST}  (BF16, device_map)                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Modes:                                                       ║
║    lightning        — SDXL Lightning 4-step      (~0.5-1s)    ║
║    realvis_fast     — RealVisXL V5 Lightning     (~1-2s)      ║
║    realvis_quality  — RealVisXL V5 25-step       (~5-8s)      ║
║    flux             — FLUX.1 Schnell 4-step      (~3-5s)      ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
