"""
AI Image Generator — FastAPI Backend
Optimized for NVIDIA Grace Blackwell (GB10) with 128 GB unified memory.

All models preloaded to GPU at startup — no OOM, no tricks, just load.

Models:
  1. lightning        — SDXL Lightning 4-step
  2. realvis_fast     — RealVisXL V5.0 Lightning
  3. realvis_quality  — RealVisXL V5.0 25-step
  4. flux             — FLUX.1 Schnell 4-step
  5. hunyuan_image    — HunyuanDiT v1.2 (on demand)
  6. hunyuan_video    — HunyuanVideo (on demand)

Endpoints:
  POST /api/generate          → JSON with base64 image
  POST /api/generate/raw      → raw PNG bytes
  POST /api/generate/batch    → batch generation
  POST /api/hunyuan/image     → Hunyuan image generation
  POST /api/hunyuan/video     → Hunyuan text-to-video
  POST /api/hunyuan/video/i2v → Hunyuan image-to-video
  POST /api/hunyuan/3d        → Hunyuan image-to-3D
  POST /api/hunyuan/text-to-3d→ Hunyuan text-to-3D
  GET  /api/tts/voices        → list available TTS voices
  POST /api/tts/generate      → single TTS generation
  POST /api/tts/generate/raw  → single TTS → raw WAV
  POST /api/tts/generate/batch→ multi-voice batch TTS
  GET  /api/health            → health + loaded models
  GET  /api/gpu/stats         → GPU hardware stats
  GET  /api/history           → generation history
"""

import asyncio
import base64
import gc
import io
import json
import logging
import os
import random
import subprocess
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image as PILImage

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*non-meta.*meta.*no-op.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# ── Config ────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
PORT = int(os.getenv("PORT", "8100"))
GENERATED_DIR = Path(os.getenv("GENERATED_DIR", Path(__file__).parent / "generated"))
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Models to preload at startup (set to "" to skip)
PRELOAD_MODELS = os.getenv(
    "PRELOAD_MODELS", "lightning,realvis_fast,realvis_quality,flux"
).split(",")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── HuggingFace Auth (for gated models like FLUX.1) ──────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("HuggingFace token configured ✓")
    except Exception as e:
        logger.warning(f"HuggingFace login failed: {e}")
else:
    logger.info("No HF_TOKEN set — gated models (FLUX.1) will be skipped")

# ── Device Detection ──────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = "cuda"
    # Blackwell / modern GPUs: prefer bfloat16 for FLUX, float16 for SDXL
    SDXL_DTYPE = torch.float16
    FLUX_DTYPE = torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    SDXL_DTYPE = torch.float16
    FLUX_DTYPE = torch.bfloat16
    logger.info("Device: Apple Silicon (MPS)")
else:
    DEVICE = "cpu"
    SDXL_DTYPE = torch.float32
    FLUX_DTYPE = torch.float32
    logger.info("Device: CPU (will be slow)")

logger.info(f"Device: {DEVICE} | SDXL dtype: {SDXL_DTYPE} | FLUX dtype: {FLUX_DTYPE}")

# ── Pipeline Cache & Concurrency ─────────────────────────────────────

_pipelines: dict = {}
_lock = threading.Lock()
_active_task: Optional[str] = None
_gen_count: int = 0
_executor = ThreadPoolExecutor(max_workers=4)


# ── Helpers ───────────────────────────────────────────────────────────

def _disable_safety(pipe):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _optimize_pipe(pipe):
    """Optimize pipeline for max speed on GB10 Blackwell.
    
    The GB10 has 120 GB unified memory but limited memory bandwidth (273 GB/s LPDDR5X).
    Attention slicing reduces peak memory traffic and actually HELPS performance
    on bandwidth-limited GPUs like the GB10 (vs high-bandwidth GDDR6X on desktop GPUs).
    """
    # Enable attention slicing — reduces memory bandwidth pressure on LPDDR5X
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(slice_size="auto")
    return pipe


def trim_prompt(text: str, max_tokens: int = 70) -> str:
    words = text.split()
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    last_comma = trimmed.rfind(",")
    if last_comma > len(trimmed) * 0.6:
        trimmed = trimmed[:last_comma]
    return trimmed


# ── Pipeline Loaders ──────────────────────────────────────────────────
#
# With 128 GB unified memory on the GB10, loading is trivially simple:
#   from_pretrained() → .to("cuda") — done.
# No component-by-component tricks, no device_map, no swap management.

def _load_lightning():
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    logger.info("Loading SDXL Lightning (4-step)...")
    unet_path = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=SDXL_DTYPE,
        variant="fp16" if SDXL_DTYPE == torch.float16 else None,
    )
    pipe.unet.load_state_dict(load_file(unet_path), strict=False)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)
    _disable_safety(pipe)

    logger.info("SDXL Lightning loaded ✓")
    return pipe


def _load_realvis_fast():
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    logger.info("Loading RealVisXL V5.0 Lightning...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=SDXL_DTYPE,
        variant="fp16" if SDXL_DTYPE == torch.float16 else None,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)
    _disable_safety(pipe)

    logger.info("RealVisXL V5.0 Lightning loaded ✓")
    return pipe


def _load_realvis_quality():
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    logger.info("Loading RealVisXL V5.0 (quality)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=SDXL_DTYPE,
        variant="fp16" if SDXL_DTYPE == torch.float16 else None,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)
    _disable_safety(pipe)

    logger.info("RealVisXL V5.0 (quality) loaded ✓")
    return pipe


def _load_flux():
    from diffusers import FluxPipeline

    logger.info("Loading FLUX.1 Schnell (~18 GB)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=FLUX_DTYPE,
    )
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)

    logger.info("FLUX.1 Schnell loaded ✓")
    return pipe


def _load_hunyuan_image():
    from diffusers import HunyuanDiTPipeline

    logger.info("Loading HunyuanDiT v1.2...")
    pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)

    logger.info("HunyuanDiT loaded ✓")
    return pipe


def _load_hunyuan_video():
    from diffusers import HunyuanVideoPipeline

    logger.info("Loading HunyuanVideo (~25 GB)...")
    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(DEVICE)
    _optimize_pipe(pipe)

    logger.info("HunyuanVideo loaded ✓")
    return pipe


_LOADERS = {
    "lightning": _load_lightning,
    "realvis_fast": _load_realvis_fast,
    "realvis_quality": _load_realvis_quality,
    "flux": _load_flux,
    "hunyuan_image": _load_hunyuan_image,
    "hunyuan_video": _load_hunyuan_video,
}


def get_pipeline(model: str):
    """Get or load a pipeline. Cached forever once loaded."""
    if model in _pipelines:
        return _pipelines[model]
    loader = _LOADERS.get(model)
    if not loader:
        raise ValueError(f"Unknown model: {model}")
    pipe = loader()
    _pipelines[model] = pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe


# ── Generation Functions ──────────────────────────────────────────────

def _generate_image_sync(
    model_mode: str, prompt: str, negative: str,
    width: int, height: int, seed: int,
    guidance_scale: float, num_inference_steps: int,
) -> tuple:
    """Generate one image. Thread-safe via lock."""
    global _active_task, _gen_count

    with _lock:
        _active_task = model_mode
        try:
            pipeline = get_pipeline(model_mode)

            gen = torch.Generator(DEVICE if DEVICE == "cuda" else "cpu").manual_seed(seed)

            if model_mode == "flux":
                with torch.inference_mode():
                    result = pipeline(
                        prompt=prompt,
                        guidance_scale=0.0,
                        num_inference_steps=num_inference_steps,
                        width=width, height=height,
                        max_sequence_length=256,
                        generator=gen,
                    )
            else:
                p = trim_prompt(prompt, max_tokens=70)
                n = trim_prompt(negative, max_tokens=70) if negative else ""
                with torch.inference_mode():
                    result = pipeline(
                        prompt=p,
                        negative_prompt=n if n else None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width, height=height,
                        generator=gen,
                    )
        finally:
            _active_task = None
            _gen_count += 1

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), seed


def _generate_hunyuan_image_sync(
    prompt: str, negative: str, width: int, height: int,
    seed: int, guidance_scale: float, num_inference_steps: int,
) -> tuple:
    global _active_task, _gen_count

    with _lock:
        _active_task = "hunyuan_image"
        try:
            pipe = get_pipeline("hunyuan_image")
            gen = torch.Generator(DEVICE if DEVICE == "cuda" else "cpu").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative if negative else None,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
        finally:
            _active_task = None
            _gen_count += 1

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), seed


def _generate_hunyuan_video_sync(
    prompt: str, width: int, height: int, num_frames: int,
    seed: int, num_inference_steps: int, fps: int,
) -> tuple:
    """Generate video with HunyuanVideo. Returns (mp4_bytes, seed)."""
    global _active_task, _gen_count

    with _lock:
        _active_task = "hunyuan_video"
        try:
            pipe = get_pipeline("hunyuan_video")
            gen = torch.Generator(DEVICE if DEVICE == "cuda" else "cpu").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    height=height, width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=gen,
                )
        finally:
            _active_task = None
            _gen_count += 1

    # Export frames to MP4
    from diffusers.utils import export_to_video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    export_to_video(result.frames[0], tmp_path, fps=fps)
    video_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return video_bytes, seed


# ── History ───────────────────────────────────────────────────────────

def save_to_history(data_bytes: bytes, filename: str, media_type: str, metadata: dict) -> str:
    (GENERATED_DIR / filename).write_bytes(data_bytes)
    meta = {**metadata, "filename": filename, "media_type": media_type}
    (GENERATED_DIR / f"{filename}.json").write_text(json.dumps(meta, indent=2))
    return filename


# ══════════════════════════════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(title="AI Image Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    async def _warmup():
        logger.info("=" * 60)
        logger.info("  WARMUP — preloading models")
        logger.info(f"  Device: {DEVICE}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  Memory: {mem:.0f} GB")
        logger.info(f"  Models: {PRELOAD_MODELS}")
        logger.info("=" * 60)

        loop = asyncio.get_event_loop()
        t_start = time.time()
        loaded = 0
        failed = 0

        for model in PRELOAD_MODELS:
            model = model.strip()
            if not model or model not in _LOADERS:
                continue
            logger.info(f"  Loading {model}...")
            t0 = time.time()
            try:
                await loop.run_in_executor(_executor, get_pipeline, model)
                dt = time.time() - t0
                logger.info(f"  ✓ {model} loaded in {dt:.1f}s")
                loaded += 1
            except Exception as e:
                logger.error(f"  ✗ {model} FAILED: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed += 1

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info(f"  WARMUP DONE — {loaded} loaded, {failed} failed in {total_time:.0f}s")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU memory: {used:.1f} / {total:.0f} GB")
        logger.info("  SERVER READY — accepting requests")
        logger.info("=" * 60)

    asyncio.create_task(_warmup())


# ══════════════════════════════════════════════════════════════════════
#  Request / Response Models
# ══════════════════════════════════════════════════════════════════════

class ModelMode(str, Enum):
    lightning = "lightning"
    realvis_fast = "realvis_fast"
    realvis_quality = "realvis_quality"
    flux = "flux"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    guidance_scale: float = Field(default=0, ge=0, le=20)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    model_mode: ModelMode = Field(default=ModelMode.lightning)


class GenerateResponse(BaseModel):
    image: str
    seed: int
    width: int
    height: int
    time_seconds: float
    model_mode: str


class BatchPromptItem(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    filename: Optional[str] = None
    seed: Optional[int] = None


class BatchGenerateRequest(BaseModel):
    prompts: List[BatchPromptItem] = Field(..., min_length=1, max_length=100)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    guidance_scale: float = Field(default=0, ge=0, le=20)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    model_mode: ModelMode = Field(default=ModelMode.lightning)


class BatchResultItem(BaseModel):
    index: int
    success: bool
    image: Optional[str] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    time_seconds: Optional[float] = None
    model_mode: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: List[BatchResultItem]
    total_time_seconds: float
    successful: int
    failed: int
    model_mode: str


class HunyuanImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    guidance_scale: float = Field(default=5.0, ge=0, le=20)
    num_inference_steps: int = Field(default=25, ge=1, le=100)


class HunyuanImageResponse(BaseModel):
    image: str
    seed: int
    width: int
    height: int
    time_seconds: float


class HunyuanVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=848, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=61, ge=1, le=129)
    seed: Optional[int] = Field(default=None)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    fps: int = Field(default=15, ge=1, le=60)


class HunyuanVideoResponse(BaseModel):
    video: str
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


class Hunyuan3DRequest(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="")
    image_width: int = Field(default=1024, ge=512, le=2048)
    image_height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    guidance_scale: float = Field(default=5.0, ge=0, le=20)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    do_texture: bool = Field(default=True)


class Hunyuan3DResponse(BaseModel):
    model_glb: str
    reference_image: Optional[str] = None
    time_seconds: float
    textured: bool


# ══════════════════════════════════════════════════════════════════════
#  Image Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    try:
        img_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_image_sync,
            req.model_mode.value, req.prompt, req.negative_prompt,
            req.width, req.height, seed,
            req.guidance_scale, req.num_inference_steps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    logger.info(f"Generated {req.width}x{req.height} in {elapsed}s "
                f"(mode={req.model_mode.value}, seed={used_seed})")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            img_bytes, f"{ts}_{used_seed}_{req.model_mode.value}.png", "image/png",
            {"type": "image", "prompt": req.prompt, "negative_prompt": req.negative_prompt,
             "seed": used_seed, "width": req.width, "height": req.height,
             "model_mode": req.model_mode.value, "guidance_scale": req.guidance_scale,
             "num_inference_steps": req.num_inference_steps, "time_seconds": elapsed,
             "timestamp": ts},
        )
    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")

    return GenerateResponse(
        image=b64, seed=used_seed, width=req.width, height=req.height,
        time_seconds=elapsed, model_mode=req.model_mode.value,
    )


@app.post("/api/generate/raw")
async def generate_raw(req: GenerateRequest):
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)
    try:
        img_bytes, _ = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_image_sync,
            req.model_mode.value, req.prompt, req.negative_prompt,
            req.width, req.height, seed,
            req.guidance_scale, req.num_inference_steps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=img_bytes, media_type="image/png")


@app.post("/api/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(req: BatchGenerateRequest):
    t0 = time.time()
    model_mode = req.model_mode.value
    loop = asyncio.get_event_loop()

    async def _one(i: int, item: BatchPromptItem):
        seed = item.seed if item.seed is not None else random.randint(1, 2**31)
        t1 = time.time()
        try:
            img_bytes, used_seed = await loop.run_in_executor(
                _executor, _generate_image_sync,
                model_mode, item.prompt, req.negative_prompt,
                req.width, req.height, seed,
                req.guidance_scale, req.num_inference_steps,
            )
            elapsed = round(time.time() - t1, 2)
            b64 = base64.b64encode(img_bytes).decode("ascii")
            fn = item.filename or f"{int(time.time() * 1000)}_{used_seed}_{model_mode}.png"
            try:
                save_to_history(
                    img_bytes, fn, "image/png",
                    {"type": "image", "prompt": item.prompt, "seed": used_seed,
                     "width": req.width, "height": req.height, "model_mode": model_mode,
                     "time_seconds": elapsed, "timestamp": int(time.time() * 1000)},
                )
            except Exception:
                pass
            return {"index": i, "success": True, "image": b64, "seed": used_seed,
                    "width": req.width, "height": req.height, "time_seconds": elapsed,
                    "model_mode": model_mode, "filename": item.filename}
        except Exception as e:
            return {"index": i, "success": False, "error": str(e)}

    results = await asyncio.gather(*[_one(i, item) for i, item in enumerate(req.prompts)])
    total_time = round(time.time() - t0, 2)
    successful = sum(1 for r in results if r.get("success"))

    return BatchGenerateResponse(
        results=[BatchResultItem(**r) for r in results],
        total_time_seconds=total_time, successful=successful,
        failed=len(results) - successful, model_mode=model_mode,
    )


# ══════════════════════════════════════════════════════════════════════
#  Hunyuan Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/hunyuan/image", response_model=HunyuanImageResponse)
async def hunyuan_image(req: HunyuanImageRequest):
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    try:
        img_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_hunyuan_image_sync,
            req.prompt, req.negative_prompt, req.width, req.height,
            seed, req.guidance_scale, req.num_inference_steps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            img_bytes, f"{ts}_{used_seed}_hunyuan_image.png", "image/png",
            {"type": "hunyuan_image", "prompt": req.prompt, "seed": used_seed,
             "width": req.width, "height": req.height, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return HunyuanImageResponse(
        image=b64, seed=used_seed, width=req.width, height=req.height, time_seconds=elapsed,
    )


@app.post("/api/hunyuan/video", response_model=HunyuanVideoResponse)
async def hunyuan_video(req: HunyuanVideoRequest):
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    try:
        video_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_hunyuan_video_sync,
            req.prompt, req.width, req.height, req.num_frames,
            seed, req.num_inference_steps, req.fps,
        )
    except Exception as e:
        logger.error(f"HunyuanVideo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(video_bytes).decode("ascii")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            video_bytes, f"{ts}_{used_seed}_hunyuan_video.mp4", "video/mp4",
            {"type": "hunyuan_video", "prompt": req.prompt, "seed": used_seed,
             "width": req.width, "height": req.height, "num_frames": req.num_frames,
             "fps": req.fps, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return HunyuanVideoResponse(
        video=b64, seed=used_seed, width=req.width, height=req.height,
        num_frames=req.num_frames, fps=req.fps, time_seconds=elapsed,
    )


@app.post("/api/hunyuan/video/i2v")
async def hunyuan_video_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    width: int = Form(default=848),
    height: int = Form(default=480),
    num_frames: int = Form(default=61),
    seed: Optional[int] = Form(default=None),
    num_inference_steps: int = Form(default=30),
    fps: int = Form(default=15),
):
    """Image-to-video generation. Currently uses text-to-video with the prompt."""
    # HunyuanVideo i2v requires a specialized pipeline not yet in standard diffusers.
    # For now, we use text-to-video as a fallback if a prompt is provided.
    if not prompt:
        raise HTTPException(400, "Image-to-video requires a text prompt as guidance. Please provide a prompt describing the desired animation.")

    t0 = time.time()
    actual_seed = seed if seed is not None else random.randint(1, 2**31)

    try:
        video_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_hunyuan_video_sync,
            prompt, width, height, num_frames,
            actual_seed, num_inference_steps, fps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(video_bytes).decode("ascii")

    return {
        "video": b64, "seed": used_seed, "width": width, "height": height,
        "num_frames": num_frames, "fps": fps, "time_seconds": elapsed,
    }


@app.post("/api/hunyuan/3d")
async def hunyuan_3d(
    image: UploadFile = File(...),
    do_texture: bool = Form(default=True),
):
    """Image-to-3D generation using Hunyuan3D-2.
    
    Note: Hunyuan3D-2 requires a specialized pipeline and custom dependencies 
    not yet available in standard diffusers. This endpoint generates a reference 
    image using HunyuanDiT and returns it as a placeholder.
    """
    raise HTTPException(
        501,
        "Hunyuan3D-2 image-to-3D is not yet available. "
        "The model requires custom dependencies (Hunyuan3D-2 repo) that are being integrated. "
        "Please use text-to-3D as an alternative, or check back after the next update."
    )


@app.post("/api/hunyuan/text-to-3d")
async def hunyuan_text_to_3d(req: Hunyuan3DRequest):
    """Text-to-3D generation: Text → Image (HunyuanDiT) → 3D Model.
    
    Note: Hunyuan3D-2 requires a specialized pipeline and custom dependencies
    not yet available in standard diffusers. This endpoint is a placeholder.
    """
    raise HTTPException(
        501,
        "Hunyuan3D-2 text-to-3D is not yet available. "
        "The model requires custom dependencies (Hunyuan3D-2 repo) that are being integrated. "
        "Please check back after the next update."
    )


# ══════════════════════════════════════════════════════════════════════
#  Professional Generation — Charts, Infographics, Diagrams
# ══════════════════════════════════════════════════════════════════════

# Category → sub-types mapping
PROFESSIONAL_CATEGORIES = {
    "infographic": {
        "label": "Infographic",
        "sub_types": ["data_overview", "process", "comparison", "timeline", "statistical", "list", "geographic", "hierarchical"],
        "description": "Visual data storytelling with icons, numbers, and layouts",
    },
    "flowchart": {
        "label": "Flowchart",
        "sub_types": ["process_flow", "decision_tree", "swimlane", "system_flow", "workflow", "algorithm"],
        "description": "Step-by-step process and decision diagrams",
    },
    "chart": {
        "label": "Chart / Graph",
        "sub_types": ["bar_chart", "pie_chart", "line_graph", "scatter_plot", "area_chart", "donut_chart", "radar_chart", "waterfall"],
        "description": "Data visualization with charts and graphs",
    },
    "table": {
        "label": "Table / Matrix",
        "sub_types": ["data_table", "comparison_matrix", "feature_matrix", "pricing_table", "schedule", "scorecard"],
        "description": "Structured data in rows and columns",
    },
    "diagram": {
        "label": "Diagram",
        "sub_types": ["architecture", "network", "er_diagram", "uml", "venn", "cycle", "block_diagram", "mind_map"],
        "description": "Technical and conceptual diagrams",
    },
    "presentation": {
        "label": "Presentation Slide",
        "sub_types": ["title_slide", "key_metrics", "bullet_points", "quote_slide", "team_slide", "roadmap", "swot"],
        "description": "Single presentation slides with professional layout",
    },
    "dashboard": {
        "label": "Dashboard",
        "sub_types": ["kpi_dashboard", "analytics", "sales_dashboard", "project_status", "financial", "marketing"],
        "description": "Multi-widget dashboards showing KPIs and metrics",
    },
    "org_chart": {
        "label": "Org Chart",
        "sub_types": ["corporate", "team_structure", "project_org", "flat_org", "matrix_org"],
        "description": "Organizational hierarchy and team structure",
    },
}

# Visual styles
PROFESSIONAL_STYLES = {
    "corporate": "clean corporate professional style, white background, sans-serif fonts, subtle blue and gray palette, sharp edges, business-ready",
    "minimalist": "ultra-minimalist design, lots of whitespace, thin lines, monochrome with one accent color, elegant typography, Swiss design principles",
    "colorful": "vibrant modern colorful design, bold flat colors, friendly rounded shapes, contemporary gradient accents, engaging visual hierarchy",
    "dark": "dark mode design, dark charcoal background, bright accent colors on dark, glowing highlights, modern tech aesthetic, light text on dark",
    "pastel": "soft pastel color palette, light background, gentle rounded shapes, warm friendly aesthetic, approachable professional look",
    "blueprint": "technical blueprint style, dark blue background, white and cyan lines, grid overlay, engineering aesthetic, precise geometric layout",
    "neon": "neon futuristic style, dark background, glowing neon outlines, cyberpunk-inspired, vivid pink blue and green accents, high contrast",
    "flat": "flat design 2.0, bold solid colors, subtle shadows, material design inspired, clean iconography, modern web aesthetic",
    "gradient": "modern gradient style, smooth color transitions, glassmorphism effects, frosted glass cards, contemporary UI design, depth through gradients",
}

# Sub-type prompt fragments for richer generation
_SUBTYPE_PROMPTS = {
    "data_overview": "data overview infographic with key statistics numbers icons and visual metrics arranged in sections",
    "process": "step-by-step process infographic with numbered stages arrows and icons showing workflow progression",
    "comparison": "side-by-side comparison infographic with two or more columns showing pros cons features differences",
    "timeline": "horizontal or vertical timeline infographic with dates milestones and event descriptions connected by a line",
    "statistical": "statistics-focused infographic with large bold numbers percentage bars pie sections and data callouts",
    "list": "numbered or bulleted list infographic with icons descriptions and visual hierarchy",
    "geographic": "map-based infographic with geographic data regional highlights and location-specific statistics",
    "hierarchical": "pyramid or hierarchy infographic showing levels ranks or priority from top to bottom",
    "process_flow": "flowchart diagram with rectangular process boxes diamond decision nodes and directional arrows",
    "decision_tree": "decision tree diagram with branching yes/no paths question nodes and outcome endpoints",
    "swimlane": "swimlane flowchart with horizontal or vertical lanes for different actors departments showing process responsibilities",
    "system_flow": "system architecture flowchart showing data flow between components databases APIs and services",
    "workflow": "workflow diagram showing sequential and parallel tasks with dependencies and milestones",
    "algorithm": "algorithm flowchart with start/end ovals process rectangles decision diamonds and clear logical flow",
    "bar_chart": "bar chart with labeled axes clear data bars value labels and legend",
    "pie_chart": "pie chart with labeled segments percentage values and color-coded legend",
    "line_graph": "line graph with multiple data series plotted on x-y axes with grid lines and trend indicators",
    "scatter_plot": "scatter plot with data points axes labels correlation line and data clusters",
    "area_chart": "area chart with filled regions below lines showing volume and trends over time",
    "donut_chart": "donut chart with center statistic segmented ring and percentage labels",
    "radar_chart": "radar/spider chart with multiple axes showing multivariate data comparison",
    "waterfall": "waterfall chart showing incremental positive and negative value changes with running totals",
    "data_table": "clean data table with headers rows columns alternating row colors and aligned values",
    "comparison_matrix": "comparison matrix table with checkmarks crosses and feature comparison across products or options",
    "feature_matrix": "feature matrix showing capabilities across tiers or products with status indicators",
    "pricing_table": "pricing table with tier columns feature lists price highlights and recommended badge",
    "schedule": "schedule or timetable grid with time slots days and event blocks",
    "scorecard": "scorecard or report card layout with metrics ratings grades and performance indicators",
    "architecture": "system architecture diagram with layered components connections and technology labels",
    "network": "network topology diagram with nodes connections switches routers and cloud elements",
    "er_diagram": "entity relationship diagram with tables fields primary keys and relationship lines",
    "uml": "UML class diagram or sequence diagram with standard notation and clear relationships",
    "venn": "Venn diagram with overlapping circles showing intersections and unique elements",
    "cycle": "cycle diagram with circular flow showing repeating phases or iterative process",
    "block_diagram": "block diagram with labeled functional blocks and interconnection arrows",
    "mind_map": "mind map with central topic branching subtopics and hierarchical idea organization",
    "title_slide": "presentation title slide with large title subtitle speaker name and modern layout",
    "key_metrics": "KPI metrics slide with large numbers trend arrows and brief descriptions",
    "bullet_points": "bullet point slide with title icon bullets and supporting text",
    "quote_slide": "quote slide with large quotation prominent attribution and elegant typography",
    "team_slide": "team introduction slide with photo placeholders names titles and brief bios",
    "roadmap": "product or project roadmap slide with timeline phases milestones and deliverables",
    "swot": "SWOT analysis slide with four quadrants for strengths weaknesses opportunities and threats",
    "kpi_dashboard": "KPI dashboard with multiple metric cards charts trend lines and status indicators",
    "analytics": "analytics dashboard with traffic charts user metrics conversion funnels and date filters",
    "sales_dashboard": "sales dashboard with revenue charts pipeline funnel top deals and quota progress",
    "project_status": "project status dashboard with timeline progress bars task completion and team workload",
    "financial": "financial dashboard with revenue expense profit charts and key financial ratios",
    "marketing": "marketing dashboard with campaign metrics channel performance and engagement analytics",
    "corporate": "corporate org chart with CEO at top management layers and department branches",
    "team_structure": "team structure diagram showing team leads members and reporting lines",
    "project_org": "project organization chart showing project manager workstreams and team assignments",
    "flat_org": "flat organization diagram with minimal hierarchy and cross-functional connections",
    "matrix_org": "matrix organization chart showing dual reporting lines functional and project-based",
}


class ProfessionalCategory(str, Enum):
    infographic = "infographic"
    flowchart = "flowchart"
    chart = "chart"
    table = "table"
    diagram = "diagram"
    presentation = "presentation"
    dashboard = "dashboard"
    org_chart = "org_chart"


class ProfessionalRequest(BaseModel):
    category: ProfessionalCategory
    sub_type: str = Field(default="")
    content: str = Field(..., min_length=1, max_length=3000)
    style: str = Field(default="corporate")
    color_scheme: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    detail_level: str = Field(default="high")  # low, medium, high


class ProfessionalResponse(BaseModel):
    image: str
    seed: int
    width: int
    height: int
    time_seconds: float
    prompt_used: str
    category: str
    sub_type: str


def _build_professional_prompt(
    category: str, sub_type: str, content: str,
    style: str, color_scheme: str, detail_level: str,
) -> tuple:
    """Build an optimized prompt for professional graphic generation. Returns (prompt, negative_prompt)."""
    parts = []

    # Core instruction
    parts.append("Professional high-resolution")

    # Sub-type specific description
    st_key = sub_type if sub_type in _SUBTYPE_PROMPTS else ""
    if st_key:
        parts.append(_SUBTYPE_PROMPTS[st_key])
    else:
        cat_info = PROFESSIONAL_CATEGORIES.get(category, {})
        parts.append(f"{cat_info.get('label', category)} design")

    # User content
    parts.append(f"showing: {content}")

    # Style
    style_desc = PROFESSIONAL_STYLES.get(style, PROFESSIONAL_STYLES["corporate"])
    parts.append(style_desc)

    # Color scheme override
    if color_scheme:
        parts.append(f"using {color_scheme} color palette")

    # Detail level
    if detail_level == "high":
        parts.append("extremely detailed, sharp text rendering, crisp edges, publication-quality, vector-like precision, 4K resolution detail")
    elif detail_level == "medium":
        parts.append("well-detailed, clear text, clean layout, professional quality")
    else:
        parts.append("simple clear layout with readable text")

    # Universal quality boosters
    parts.append("perfect typography, aligned layout, print-ready quality, no artifacts, photorealistic rendering of graphic design")

    prompt = ", ".join(parts)

    # Negative prompt to avoid common issues
    negative = (
        "blurry, low quality, pixelated, distorted text, misspelled words, "
        "cropped, watermark, signature, amateur, messy layout, overlapping elements, "
        "photographic, people, faces, hands, fingers, realistic photo, 3d render, "
        "cartoon, anime, sketch, hand-drawn, illegible text, warped lines"
    )

    return prompt, negative


@app.get("/api/professional/categories")
async def professional_categories():
    """List available professional generation categories and sub-types."""
    return {
        "categories": PROFESSIONAL_CATEGORIES,
        "styles": list(PROFESSIONAL_STYLES.keys()),
        "detail_levels": ["low", "medium", "high"],
    }


@app.post("/api/generate/professional", response_model=ProfessionalResponse)
async def generate_professional(req: ProfessionalRequest):
    """Generate a professional infographic/chart/diagram using FLUX with prompt engineering."""
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    prompt, negative = _build_professional_prompt(
        req.category.value, req.sub_type, req.content,
        req.style, req.color_scheme, req.detail_level,
    )

    # Use FLUX for best text/detail rendering (4 steps schnell)
    # Fall back to realvis_quality if FLUX not loaded
    model = "flux"
    steps = 4
    guidance = 0
    if "flux" not in _pipelines:
        model = "realvis_quality"
        steps = 25
        guidance = 7.5
        negative = negative  # realvis supports negative prompt

    try:
        img_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_image_sync,
            model, prompt, negative,
            req.width, req.height, seed,
            guidance, steps,
        )
    except Exception as e:
        logger.error(f"[PRO] Generation failed: {e}")
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    logger.info(f"[PRO] Generated {req.category.value}/{req.sub_type} "
                f"{req.width}x{req.height} in {elapsed}s (model={model})")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            img_bytes, f"{ts}_{used_seed}_pro_{req.category.value}.png", "image/png",
            {"type": "professional", "category": req.category.value, "sub_type": req.sub_type,
             "content": req.content[:200], "style": req.style, "prompt": prompt[:200],
             "seed": used_seed, "width": req.width, "height": req.height,
             "model_mode": model, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return ProfessionalResponse(
        image=b64, seed=used_seed, width=req.width, height=req.height,
        time_seconds=elapsed, prompt_used=prompt,
        category=req.category.value, sub_type=req.sub_type,
    )


# ══════════════════════════════════════════════════════════════════════
#  TTS — Chatterbox Text-to-Speech
# ══════════════════════════════════════════════════════════════════════

# ── Chatterbox singleton ─────────────────────────────────────────────

_chatterbox_model = None
_chatterbox_checked = False

VOICES_DIR = Path(__file__).parent / "voices"


def _get_chatterbox():
    """Lazy-load Chatterbox TTS model (500M params)."""
    global _chatterbox_model, _chatterbox_checked
    if _chatterbox_checked:
        return _chatterbox_model
    _chatterbox_checked = True
    try:
        from chatterbox.tts import ChatterboxTTS
        logger.info("[TTS] Loading Chatterbox TTS...")
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        _chatterbox_model = model
        logger.info(f"[TTS] Chatterbox loaded on {DEVICE} ✓")
        return _chatterbox_model
    except ImportError:
        logger.warning("[TTS] chatterbox-tts not installed")
        return None
    except Exception as e:
        logger.warning(f"[TTS] Failed to load Chatterbox: {e}")
        return None


def _list_available_voices() -> List[dict]:
    """List all available voice reference WAVs."""
    voices = []
    if VOICES_DIR.exists():
        for wav in sorted(VOICES_DIR.glob("*.wav")):
            voices.append({"id": wav.stem, "name": wav.stem.capitalize(), "filename": wav.name})
    return voices


def _generate_tts_sync(
    text: str, voice: str, exaggeration: float, cfg_weight: float,
    temperature: float, repetition_penalty: float, speed: float,
) -> tuple:
    """Generate speech with Chatterbox. Returns (wav_bytes, sample_rate, duration)."""
    import re as _re
    import numpy as np

    model = _get_chatterbox()
    if model is None:
        raise RuntimeError("Chatterbox TTS not available")

    # Resolve voice reference
    voice_path = None
    if voice and voice != "default":
        vp = VOICES_DIR / f"{voice}.wav"
        if vp.exists():
            voice_path = str(vp)

    # Text cleanup
    text = _re.sub(r'\s+', ' ', text).strip()
    text = text.replace('"', '').replace('"', '').replace('"', '')
    text = text.replace('...', ',').replace('…', ',')
    text = _re.sub(r' - ', ' — ', text)

    # Pre-cache voice conditionals
    if voice_path:
        model.prepare_conditionals(voice_path, exaggeration=exaggeration)

    # Split into sentences, generate in chunks of 2
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    all_samples = []
    sample_rate = getattr(model, 'sr', 24000)
    chunk_size = 2

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if not chunk.strip():
            continue
        try:
            wav = model.generate(
                chunk,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            if hasattr(wav, 'cpu'):
                audio_np = wav.cpu().numpy()
            else:
                audio_np = np.array(wav)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            all_samples.append(audio_np)

            # Natural pause between chunks
            last_char = chunk.rstrip()[-1] if chunk.rstrip() else "."
            pause_ms = 400 if last_char in "!?" else 250 if last_char == "," else 500
            all_samples.append(np.zeros(int(sample_rate * pause_ms / 1000)))
        except Exception as e:
            logger.warning(f"[TTS] Chunk failed: {e}")

    if not all_samples:
        raise RuntimeError("TTS generated no audio")

    combined = np.concatenate(all_samples)

    # Speed adjustment via resampling (if not 1.0)
    if speed != 1.0 and 0.5 <= speed <= 2.0:
        from scipy.signal import resample
        new_len = int(len(combined) / speed)
        combined = resample(combined, new_len).astype(np.float32)

    # Convert to WAV bytes
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()
    duration = len(combined) / sample_rate

    return wav_bytes, sample_rate, duration


# ── TTS Request/Response Models ──────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    exaggeration: float = Field(default=0.35, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.65, ge=0.1, le=1.5)
    repetition_penalty: float = Field(default=1.35, ge=1.0, le=2.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class TTSResponse(BaseModel):
    audio: str  # base64 WAV
    duration: float
    voice: str
    time_seconds: float


class TTSBatchItem(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    label: str = Field(default="")


class TTSBatchRequest(BaseModel):
    items: List[TTSBatchItem] = Field(..., min_length=1, max_length=20)
    exaggeration: float = Field(default=0.35, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.65, ge=0.1, le=1.5)
    repetition_penalty: float = Field(default=1.35, ge=1.0, le=2.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    combine: bool = Field(default=False)


class TTSBatchResultItem(BaseModel):
    index: int
    success: bool
    audio: Optional[str] = None
    duration: Optional[float] = None
    voice: Optional[str] = None
    label: Optional[str] = None
    time_seconds: Optional[float] = None
    error: Optional[str] = None


class TTSBatchResponse(BaseModel):
    results: List[TTSBatchResultItem]
    combined_audio: Optional[str] = None
    combined_duration: Optional[float] = None
    total_time_seconds: float
    successful: int
    failed: int


# ── TTS Endpoints ────────────────────────────────────────────────────

@app.get("/api/tts/voices")
async def tts_voices():
    """List available voice references for cloning."""
    voices = _list_available_voices()
    return {
        "voices": voices,
        "default": "default",
        "engine": "chatterbox",
        "chatterbox_available": _get_chatterbox() is not None,
    }


@app.post("/api/tts/generate", response_model=TTSResponse)
async def tts_generate(req: TTSRequest):
    """Generate speech from text using Chatterbox TTS."""
    t0 = time.time()

    try:
        wav_bytes, sr, duration = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_tts_sync,
            req.text, req.voice, req.exaggeration, req.cfg_weight,
            req.temperature, req.repetition_penalty, req.speed,
        )
    except Exception as e:
        logger.error(f"[TTS] Generation failed: {e}")
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Save to history
    try:
        ts = int(time.time() * 1000)
        save_to_history(
            wav_bytes, f"{ts}_tts_{req.voice}.wav", "audio/wav",
            {"type": "tts", "text": req.text[:200], "voice": req.voice,
             "duration": duration, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return TTSResponse(
        audio=b64, duration=round(duration, 2),
        voice=req.voice, time_seconds=elapsed,
    )


@app.post("/api/tts/generate/raw")
async def tts_generate_raw(req: TTSRequest):
    """Generate speech and return raw WAV bytes."""
    try:
        wav_bytes, sr, duration = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_tts_sync,
            req.text, req.voice, req.exaggeration, req.cfg_weight,
            req.temperature, req.repetition_penalty, req.speed,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/tts/generate/batch", response_model=TTSBatchResponse)
async def tts_generate_batch(req: TTSBatchRequest):
    """Generate multiple TTS clips with different voices/texts. Optionally combine into one."""
    t0 = time.time()
    loop = asyncio.get_event_loop()
    results = []
    all_wav_bytes = []

    for i, item in enumerate(req.items):
        t1 = time.time()
        try:
            wav_bytes, sr, duration = await loop.run_in_executor(
                _executor, _generate_tts_sync,
                item.text, item.voice, req.exaggeration, req.cfg_weight,
                req.temperature, req.repetition_penalty, req.speed,
            )
            elapsed = round(time.time() - t1, 2)
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            results.append(TTSBatchResultItem(
                index=i, success=True, audio=b64, duration=round(duration, 2),
                voice=item.voice, label=item.label, time_seconds=elapsed,
            ))
            all_wav_bytes.append(wav_bytes)
        except Exception as e:
            results.append(TTSBatchResultItem(
                index=i, success=False, error=str(e),
            ))

    # Combine all audio if requested
    combined_b64 = None
    combined_dur = None
    if req.combine and all_wav_bytes:
        try:
            import numpy as np
            import soundfile as sf

            all_samples = []
            sample_rate = 24000
            for wb in all_wav_bytes:
                data, sr = sf.read(io.BytesIO(wb))
                all_samples.append(data)
                sample_rate = sr
                # 1s pause between speakers
                all_samples.append(np.zeros(int(sample_rate * 1.0)))

            combined = np.concatenate(all_samples)
            buf = io.BytesIO()
            sf.write(buf, combined, sample_rate, format="WAV")
            combined_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            combined_dur = round(len(combined) / sample_rate, 2)
        except Exception as e:
            logger.warning(f"[TTS] Combine failed: {e}")

    total_time = round(time.time() - t0, 2)
    successful = sum(1 for r in results if r.success)

    return TTSBatchResponse(
        results=results,
        combined_audio=combined_b64,
        combined_duration=combined_dur,
        total_time_seconds=total_time,
        successful=successful,
        failed=len(results) - successful,
    )


# ══════════════════════════════════════════════════════════════════════
#  Health & Status
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "models_loaded": list(_pipelines.keys()),
        "available_modes": [m.value for m in ModelMode],
        "tts_available": _chatterbox_model is not None,
    }


def _safe_int(val):
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(val):
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


@app.get("/api/gpu/stats")
async def gpu_stats():
    """Live GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,"
             "memory.used,memory.total,memory.free,power.draw,power.limit,"
             "fan.speed,pstate,clocks.current.graphics,clocks.current.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi failed: {result.stderr.strip()}"}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 14:
                continue
            idx = int(parts[0])
            loaded = list(_pipelines.keys())

            gpus.append({
                "index": idx,
                "name": parts[1],
                "temperature_c": _safe_int(parts[2]),
                "gpu_utilization_pct": _safe_int(parts[3]),
                "memory_utilization_pct": _safe_int(parts[4]),
                "memory_used_mb": _safe_int(parts[5]),
                "memory_total_mb": _safe_int(parts[6]),
                "memory_free_mb": _safe_int(parts[7]),
                "power_draw_w": _safe_float(parts[8]),
                "power_limit_w": _safe_float(parts[9]),
                "fan_speed_pct": _safe_int(parts[10]),
                "pstate": parts[11],
                "clock_graphics_mhz": _safe_int(parts[12]),
                "clock_memory_mhz": _safe_int(parts[13]),
                "slot": {
                    "slot_id": idx,
                    "assigned_models": list(_LOADERS.keys()),
                    "loaded_models": loaded,
                    "offloaded_models": [],
                    "active_task": _active_task,
                    "generation_count": _gen_count,
                },
            })

        # Driver info
        driver = "unknown"
        cuda_ver = "unknown"
        try:
            dr = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if dr.returncode == 0:
                driver = dr.stdout.strip().split("\n")[0].strip()
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            for smi_line in smi.stdout.split("\n"):
                if "CUDA Version" in smi_line:
                    cuda_ver = smi_line.split("CUDA Version:")[1].strip().split()[0]
                    break
        except Exception:
            pass

        total_mem = sum(g["memory_total_mb"] or 0 for g in gpus) / 1024
        total_power = sum(g["power_draw_w"] or 0 for g in gpus)
        total_power_limit = sum(g["power_limit_w"] or 0 for g in gpus)
        avg_temp = sum(g["temperature_c"] or 0 for g in gpus) / max(len(gpus), 1)
        avg_util = sum(g["gpu_utilization_pct"] or 0 for g in gpus) / max(len(gpus), 1)

        return {
            "gpus": gpus,
            "summary": {
                "gpu_count": len(gpus),
                "total_memory_gb": round(total_mem, 1),
                "total_power_draw_w": round(total_power, 1),
                "total_power_limit_w": round(total_power_limit, 1),
                "avg_temperature_c": round(avg_temp, 1),
                "avg_gpu_utilization_pct": round(avg_util, 1),
                "driver_version": driver,
                "cuda_version": cuda_ver,
            },
        }
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/queue/status")
async def queue_status():
    return {
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loaded_models": list(_pipelines.keys()),
        "active_task": _active_task,
        "generation_count": _gen_count,
    }


# ══════════════════════════════════════════════════════════════════════
#  History
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/history")
async def list_history(limit: int = 200, offset: int = 0, type_filter: Optional[str] = None):
    if not GENERATED_DIR.exists():
        return {"items": [], "total": 0}
    meta_files = sorted(GENERATED_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    items = []
    for mf in meta_files:
        try:
            meta = json.loads(mf.read_text())
            if type_filter and meta.get("type") != type_filter:
                continue
            if (GENERATED_DIR / meta["filename"]).exists():
                items.append(meta)
        except Exception:
            continue
    total = len(items)
    return {"items": items[offset:offset + limit], "total": total}


@app.get("/api/history/image/{filename}")
async def get_history_image(filename: str):
    safe = Path(filename).name
    path = GENERATED_DIR / safe
    if not path.exists():
        raise HTTPException(404, "Not found")
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return Response(content=path.read_bytes(), media_type=media.get(path.suffix.lower(), "image/png"))


@app.get("/api/history/file/{filename}")
async def get_history_file(filename: str):
    safe = Path(filename).name
    path = GENERATED_DIR / safe
    if not path.exists():
        raise HTTPException(404, "Not found")
    media = {
        ".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4",
        ".glb": "model/gltf-binary", ".webp": "image/webp",
    }
    return Response(content=path.read_bytes(), media_type=media.get(path.suffix.lower(), "application/octet-stream"))


@app.delete("/api/history/{filename}")
async def delete_history(filename: str):
    safe = Path(filename).name
    deleted = False
    for p in [GENERATED_DIR / safe, GENERATED_DIR / f"{safe}.json"]:
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(404, "Not found")
    return {"deleted": safe}


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print(f"""
 ═══════════════════════════════════════════════════════════
   AI Image Generator — {DEVICE.upper()}
 ═══════════════════════════════════════════════════════════
   Server:  http://0.0.0.0:{PORT}
   Docs:    http://0.0.0.0:{PORT}/docs
 ───────────────────────────────────────────────────────────
   Models:
     lightning        SDXL Lightning 4-step
     realvis_fast     RealVisXL V5 Lightning
     realvis_quality  RealVisXL V5 25-step
     flux             FLUX.1 Schnell 4-step
     hunyuan_image    HunyuanDiT v1.2 (on demand)
     hunyuan_video    HunyuanVideo (on demand)
     hunyuan_3d       Hunyuan3D-2 (coming soon)
 ───────────────────────────────────────────────────────────
   Preload: {', '.join(PRELOAD_MODELS)}
 ═══════════════════════════════════════════════════════════
    """)

    uvicorn.run(app, host="0.0.0.0", port=PORT)
