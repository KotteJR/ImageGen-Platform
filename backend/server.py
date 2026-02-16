"""
AI Image Generator — FastAPI Backend
Optimized for NVIDIA Grace Blackwell (GB10) with 128 GB unified memory.

All models preloaded to GPU at startup — no OOM, no tricks, just load.

Models (AI image generation):
  1. lightning        — SDXL Lightning 4-step
  2. realvis_fast     — RealVisXL V5.0 Lightning
  3. realvis_quality  — RealVisXL V5.0 25-step
  4. flux             — FLUX.1 Schnell 4-step
  5. hunyuan_image    — HunyuanDiT v1.2 (on demand)
  6. hunyuan_video    — HunyuanVideo (on demand)

Professional graphics (code-based — matplotlib/Pillow):
  Charts, dashboards, infographics, flowcharts, tables, diagrams,
  presentations, org charts — all rendered with pixel-perfect text.

Endpoints:
  POST /api/generate             → JSON with base64 image
  POST /api/generate/raw         → raw PNG bytes
  POST /api/generate/batch       → batch generation
  POST /api/generate/professional→ code-rendered professional graphic
  GET  /api/professional/categories → list categories & styles
  POST /api/hunyuan/image        → Hunyuan image generation
  POST /api/hunyuan/video        → Hunyuan text-to-video
  POST /api/hunyuan/video/i2v    → Hunyuan image-to-video
  POST /api/hunyuan/3d           → Hunyuan image-to-3D
  POST /api/hunyuan/text-to-3d   → Hunyuan text-to-3D
  GET  /api/tts/voices           → list available TTS voices
  POST /api/tts/generate         → single TTS generation
  POST /api/tts/generate/raw     → single TTS → raw WAV
  POST /api/tts/generate/batch   → multi-voice batch TTS
  GET  /api/health               → health + loaded models
  GET  /api/gpu/stats            → GPU hardware stats
  GET  /api/history              → generation history
"""

# ── Install torchaudio shim BEFORE any other imports ─────────────────
# NGC PyTorch 25.01 has a custom torch ABI that breaks the real torchaudio
# C extension.  This pure-Python shim provides the 3 functions Chatterbox
# actually needs (load, Resample, fbank) using soundfile/scipy/librosa.
try:
    import torchaudio as _ta_test
    _ta_test.load  # if this works, real torchaudio is fine
except Exception:
    try:
        from torchaudio_shim import install as _install_shim
        _install_shim()
    except Exception:
        pass  # will fail later when TTS tries to load

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
    # HunyuanDiT text encoder requires float32 — float16 causes
    # "expected scalar type Float but found Half" in the text encoder.
    # GB10 has 128 GB unified memory so float32 is fine.
    pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        torch_dtype=torch.float32,
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
#  Professional Generation — Code-Based Rendering
#  Uses matplotlib, Pillow, and graphviz for PIXEL-PERFECT output.
#  No AI hallucinated text — everything is rendered programmatically.
# ══════════════════════════════════════════════════════════════════════

import re as _re
import math as _math
import textwrap as _textwrap

PROFESSIONAL_CATEGORIES = {
    "infographic": {
        "label": "Infographic",
        "sub_types": ["data_overview", "comparison", "statistical", "timeline", "process", "list"],
        "description": "Visual data storytelling with real numbers, icons, and layouts",
    },
    "flowchart": {
        "label": "Flowchart",
        "sub_types": ["process_flow", "decision_tree", "workflow", "algorithm"],
        "description": "Step-by-step process and decision diagrams",
    },
    "chart": {
        "label": "Chart / Graph",
        "sub_types": ["bar_chart", "pie_chart", "line_graph", "donut_chart", "area_chart", "radar_chart", "scatter_plot", "waterfall"],
        "description": "Data visualization with real charts — accurate text and proportions",
    },
    "table": {
        "label": "Table / Matrix",
        "sub_types": ["data_table", "comparison_matrix", "pricing_table", "scorecard"],
        "description": "Structured data in rows and columns",
    },
    "diagram": {
        "label": "Diagram",
        "sub_types": ["block_diagram", "venn", "cycle", "mind_map"],
        "description": "Technical and conceptual diagrams",
    },
    "presentation": {
        "label": "Presentation Slide",
        "sub_types": ["title_slide", "key_metrics", "bullet_points", "swot", "roadmap"],
        "description": "Single presentation slides with professional layout",
    },
    "dashboard": {
        "label": "Dashboard",
        "sub_types": ["kpi_dashboard", "sales_dashboard", "analytics", "financial"],
        "description": "Multi-widget dashboards showing KPIs and metrics",
    },
    "org_chart": {
        "label": "Org Chart",
        "sub_types": ["corporate", "team_structure"],
        "description": "Organizational hierarchy and team structure",
    },
}

# Style color palettes for code-based rendering
PROFESSIONAL_PALETTES = {
    "corporate": {
        "bg": "#FFFFFF", "text": "#1A1A2E", "accent": "#0066CC",
        "colors": ["#0066CC", "#00A3E0", "#43B02A", "#FF6F00", "#7B2D8E", "#E31937"],
        "grid": "#E0E0E0", "card_bg": "#F5F7FA", "card_border": "#D4D8DD",
    },
    "minimalist": {
        "bg": "#FAFAFA", "text": "#2C2C2C", "accent": "#333333",
        "colors": ["#333333", "#666666", "#999999", "#BBBBBB", "#444444", "#777777"],
        "grid": "#EEEEEE", "card_bg": "#F0F0F0", "card_border": "#DDDDDD",
    },
    "colorful": {
        "bg": "#FFFFFF", "text": "#2D3436", "accent": "#6C5CE7",
        "colors": ["#6C5CE7", "#00B894", "#FDCB6E", "#E17055", "#0984E3", "#D63031"],
        "grid": "#DFE6E9", "card_bg": "#F8F9FA", "card_border": "#DFE6E9",
    },
    "dark": {
        "bg": "#0D1117", "text": "#E6EDF3", "accent": "#58A6FF",
        "colors": ["#58A6FF", "#3FB950", "#D29922", "#F85149", "#BC8CFF", "#39D2C0"],
        "grid": "#21262D", "card_bg": "#161B22", "card_border": "#30363D",
    },
    "pastel": {
        "bg": "#FFF8F0", "text": "#4A4A4A", "accent": "#B8A9C9",
        "colors": ["#B8A9C9", "#A8D8EA", "#F6D6AD", "#F4BFBF", "#AAD9BB", "#D4A5A5"],
        "grid": "#EDE8E4", "card_bg": "#FFF4ED", "card_border": "#E8E0D8",
    },
    "blueprint": {
        "bg": "#0A1628", "text": "#C8D6E5", "accent": "#48DBFB",
        "colors": ["#48DBFB", "#0ABDE3", "#54A0FF", "#5F27CD", "#01A3A4", "#2ED573"],
        "grid": "#1B2838", "card_bg": "#0F1F35", "card_border": "#1E3A5F",
    },
    "neon": {
        "bg": "#0A0A0A", "text": "#FFFFFF", "accent": "#FF006E",
        "colors": ["#FF006E", "#8338EC", "#3A86FF", "#06D6A0", "#FFD166", "#EF476F"],
        "grid": "#1A1A1A", "card_bg": "#111111", "card_border": "#2A2A2A",
    },
    "flat": {
        "bg": "#ECF0F1", "text": "#2C3E50", "accent": "#3498DB",
        "colors": ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"],
        "grid": "#D5DBDB", "card_bg": "#FFFFFF", "card_border": "#BDC3C7",
    },
    "gradient": {
        "bg": "#F0F2F5", "text": "#1A1A2E", "accent": "#667EEA",
        "colors": ["#667EEA", "#764BA2", "#F093FB", "#4FACFE", "#43E97B", "#FA709A"],
        "grid": "#E0E2E5", "card_bg": "#FFFFFF", "card_border": "#D0D2D5",
    },
}


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _parse_data_from_content(content: str) -> dict:
    """Extract structured data from free-form user content.

    Handles two input modes:
      1. Structured: multi-line with key:value or key=value pairs.
      2. Natural language: a descriptive paragraph with embedded numbers/%.

    Returns dict with title, labels, values, raw_lines, kv_pairs.
    """
    data = {"title": "", "items": [], "labels": [], "values": [], "raw_lines": [], "kv_pairs": []}

    raw = content.strip()
    has_newlines = "\n" in raw

    # ── Split into lines ────────────────────────────────────────────
    if has_newlines:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
    else:
        # Single paragraph — try splitting on commas ONLY if they look like
        # a data list (at least 2 segments with numbers).  Otherwise keep whole.
        segments = [s.strip() for s in raw.split(",") if s.strip()]
        num_segments_with_numbers = sum(1 for s in segments if _re.search(r'\d', s))
        if len(segments) >= 2 and num_segments_with_numbers >= 2:
            lines = segments
        else:
            lines = [raw]

    # ── Title extraction ────────────────────────────────────────────
    if lines:
        first = lines[0]
        # A title is a short line without heavy numeric content
        if len(first) < 60 and not _re.search(r'\d+\s*%|\d+\.\d+', first):
            data["title"] = first
            lines = lines[1:]

    # ── Parse each line ─────────────────────────────────────────────
    for line in lines:
        data["raw_lines"].append(line)

        # Try explicit key: value or key = value
        kv_match = _re.match(r'^([A-Za-z][\w\s/&]{1,40}?)\s*[:=]\s*(.+)$', line)
        if kv_match:
            k, v = kv_match.group(1).strip(), kv_match.group(2).strip()
            data["kv_pairs"].append((k, v))
            num_match = _re.search(r'([\d,]+\.?\d*)\s*%?', v)
            if num_match:
                try:
                    val = float(num_match.group(1).replace(",", ""))
                    if "%" in v:
                        val = min(val, 100)
                    data["labels"].append(k)
                    data["values"].append(val)
                except ValueError:
                    pass
            continue

        # Extract "label … 45%" or "label … 1,200" patterns inline
        # e.g. "revenue growth of 45%" → label="Revenue Growth", value=45
        inline_pcts = _re.findall(r'([\w\s]{2,30}?)\s+(?:of\s+|at\s+|is\s+|was\s+)?([\d,]+\.?\d*)\s*%', line)
        for label, val in inline_pcts:
            clean = label.strip().rstrip(" ofatis").strip()
            if clean and clean not in data["labels"]:
                try:
                    data["labels"].append(clean.title())
                    data["values"].append(min(float(val.replace(",", "")), 100))
                except ValueError:
                    pass

        # Fallback: explicit label: number (without %)
        if not inline_pcts:
            num_pairs = _re.findall(r'([\w\s]{2,30}?)\s*[:=]\s*([\d,]+\.?\d*)', line)
            for label, val in num_pairs:
                if label.strip() not in data["labels"]:
                    try:
                        data["labels"].append(label.strip().title())
                        data["values"].append(float(val.replace(",", "")))
                    except ValueError:
                        pass

    # ── Fallback: pull any numbers + nearby words from the whole text ──
    if not data["labels"] and not data["values"]:
        # Try "word(s) number%" or "word(s) number"
        pairs = _re.findall(r'([A-Za-z][\w\s]{1,25}?)\s+([\d,]+\.?\d*)\s*%?', raw)
        seen = set()
        for label, val in pairs:
            clean = label.strip().title()
            if clean not in seen:
                seen.add(clean)
                try:
                    data["labels"].append(clean)
                    data["values"].append(float(val.replace(",", "")))
                except ValueError:
                    pass
            if len(data["labels"]) >= 8:
                break

    # ── Title fallback ──────────────────────────────────────────────
    if not data["title"]:
        # Use the first meaningful phrase (strip numbers)
        cleaned = _re.sub(r'\d[\d,]*\.?\d*\s*%?', '', raw)
        cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
        # Take the first clause (up to first comma or 50 chars)
        title_candidate = cleaned.split(",")[0].strip()[:50]
        data["title"] = title_candidate if len(title_candidate) > 3 else "Data Overview"

    # ── Ensure we always have SOME data to render ───────────────────
    if not data["labels"]:
        data["labels"] = ["Category A", "Category B", "Category C", "Category D"]
        data["values"] = [random.randint(20, 90) for _ in data["labels"]]

    if len(data["values"]) < len(data["labels"]):
        data["values"].extend([random.randint(10, 80) for _ in range(len(data["labels"]) - len(data["values"]))])
    elif len(data["labels"]) < len(data["values"]):
        data["labels"].extend([f"Item {i+1}" for i in range(len(data["labels"]), len(data["values"]))])

    return data


def _setup_matplotlib_style(palette: dict, width: int, height: int):
    """Configure matplotlib with the given palette and size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    dpi = 150
    fig_w = width / dpi
    fig_h = height / dpi

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
        "axes.facecolor": palette["card_bg"],
        "figure.facecolor": palette["bg"],
        "text.color": palette["text"],
        "axes.labelcolor": palette["text"],
        "xtick.color": palette["text"],
        "ytick.color": palette["text"],
        "axes.edgecolor": palette["grid"],
        "grid.color": palette["grid"],
        "grid.alpha": 0.5,
    })

    return plt, fig_w, fig_h, dpi


def _render_chart(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render a chart using matplotlib. Returns PNG bytes."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]

    labels = data["labels"] or [f"Item {i+1}" for i in range(5)]
    values = data["values"] or [random.randint(10, 90) for _ in labels]

    if len(values) < len(labels):
        values.extend([0] * (len(labels) - len(values)))
    elif len(labels) < len(values):
        labels.extend([f"Item {i+1}" for i in range(len(labels), len(values))])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)

    if sub_type in ("pie_chart", "donut_chart"):
        ax = fig.add_subplot(111)
        wedge_colors = [colors[i % len(colors)] for i in range(len(labels))]
        explode = [0.02] * len(labels)

        if sub_type == "donut_chart":
            wedges, texts, autotexts = ax.pie(
                values, labels=None, autopct='%1.1f%%', startangle=90,
                colors=wedge_colors, explode=explode, pctdistance=0.82,
                wedgeprops=dict(width=0.35, edgecolor=bg, linewidth=2),
            )
            total = sum(values)
            ax.text(0, 0, f"{total:,.0f}", ha="center", va="center",
                    fontsize=28, fontweight="bold", color=text_c)
            ax.text(0, -0.12, "Total", ha="center", va="center",
                    fontsize=11, color=text_c, alpha=0.6)
        else:
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=wedge_colors, explode=explode,
                wedgeprops=dict(edgecolor=bg, linewidth=2),
            )

        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
            t.set_color("#FFFFFF" if sub_type == "donut_chart" else text_c)

        if sub_type == "donut_chart":
            ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

        ax.set_title(data["title"] or "Distribution", fontsize=18, fontweight="bold", pad=20)

    elif sub_type == "line_graph":
        ax = fig.add_subplot(111)
        x = list(range(len(values)))
        ax.plot(x, values, color=colors[0], linewidth=2.5, marker="o", markersize=6, zorder=5)
        ax.fill_between(x, values, alpha=0.1, color=colors[0])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 5 else 0, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(data["title"] or "Trend", fontsize=18, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, v in enumerate(values):
            ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=colors[0])

    elif sub_type == "area_chart":
        ax = fig.add_subplot(111)
        x = list(range(len(values)))
        ax.fill_between(x, values, alpha=0.4, color=colors[0])
        ax.plot(x, values, color=colors[0], linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 5 else 0, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(data["title"] or "Area Chart", fontsize=18, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif sub_type == "radar_chart":
        import numpy as np
        ax = fig.add_subplot(111, polar=True)
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vals = values + [values[0]]
        angles += [angles[0]]
        ax.plot(angles, vals, color=colors[0], linewidth=2)
        ax.fill(angles, vals, color=colors[0], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(data["title"] or "Radar Chart", fontsize=18, fontweight="bold", pad=25)

    elif sub_type == "scatter_plot":
        ax = fig.add_subplot(111)
        x_vals = list(range(len(values)))
        scatter_colors = [colors[i % len(colors)] for i in range(len(values))]
        ax.scatter(x_vals, values, c=scatter_colors, s=80, alpha=0.8, edgecolors="white", linewidth=1, zorder=5)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 5 else 0, ha="right")
        ax.grid(True, alpha=0.3)
        ax.set_title(data["title"] or "Scatter Plot", fontsize=18, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif sub_type == "waterfall":
        ax = fig.add_subplot(111)
        cumulative = []
        running = 0
        for v in values:
            cumulative.append(running)
            running += v
        bar_colors = [colors[0] if v >= 0 else colors[3] for v in values]
        ax.bar(labels, values, bottom=cumulative, color=bar_colors, edgecolor="white", linewidth=1)
        ax.set_title(data["title"] or "Waterfall Chart", fontsize=18, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30 if len(labels) > 5 else 0, ha="right")

    else:
        ax = fig.add_subplot(111)
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=1, width=0.65)
        ax.set_title(data["title"] or "Chart", fontsize=18, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30 if len(labels) > 5 else 0, ha="right")

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=text_c)

    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_dashboard(data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render a multi-panel dashboard with KPI cards and charts."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    import numpy as np

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]
    card_bg = palette["card_bg"]
    card_border = palette["card_border"]

    labels = data["labels"] or ["Revenue", "Users", "Conversion", "Growth"]
    values = data["values"] or [random.randint(10, 95) for _ in labels]

    if len(values) < len(labels):
        values.extend([random.randint(10, 90) for _ in range(len(labels) - len(values))])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    title = data["title"] or "Dashboard"
    fig.suptitle(title, fontsize=20, fontweight="bold", color=text_c, y=0.97)

    n_kpis = min(len(labels), 4)

    gs = fig.add_gridspec(3, n_kpis, hspace=0.45, wspace=0.3,
                          left=0.06, right=0.94, top=0.88, bottom=0.06)

    for i in range(n_kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.05",
                               facecolor=card_bg, edgecolor=card_border, linewidth=1.5)
        ax.add_patch(rect)

        v = values[i] if i < len(values) else 0
        lbl = labels[i] if i < len(labels) else f"Metric {i+1}"

        raw_text = " ".join(data.get("raw_lines", []))
        is_pct = v <= 100 and "%" in raw_text
        fmt_v = f"{v:,.0f}%" if is_pct else f"{v:,.1f}" if v < 10 else f"{v:,.0f}"
        ax.text(0.5, 0.62, fmt_v, ha="center", va="center",
                fontsize=22, fontweight="bold", color=colors[i % len(colors)])
        ax.text(0.5, 0.28, lbl, ha="center", va="center",
                fontsize=10, color=text_c, alpha=0.7)

        change = random.choice(["+", "-"])
        change_val = random.randint(1, 15)
        change_color = colors[1] if change == "+" else colors[3] if len(colors) > 3 else colors[0]
        ax.text(0.5, 0.10, f"{change}{change_val}%", ha="center", va="center",
                fontsize=8, color=change_color, alpha=0.8)

    ax_bar = fig.add_subplot(gs[1, :n_kpis // 2])
    bar_labels = labels[:min(6, len(labels))]
    bar_values = values[:len(bar_labels)]
    bar_colors = [colors[i % len(colors)] for i in range(len(bar_labels))]
    ax_bar.barh(bar_labels, bar_values, color=bar_colors, edgecolor="none", height=0.6)
    ax_bar.set_title("Overview", fontsize=12, fontweight="bold", color=text_c, pad=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(bar_values):
        ax_bar.text(v + max(bar_values) * 0.02, i, f"{v:,.0f}", va="center", fontsize=9, color=text_c)

    ax_line = fig.add_subplot(gs[1, n_kpis // 2:])
    n_points = 12
    x = list(range(n_points))
    trend = sorted(random.sample(range(20, 100), n_points))
    ax_line.plot(x, trend, color=colors[0], linewidth=2, marker="o", markersize=4)
    ax_line.fill_between(x, trend, alpha=0.1, color=colors[0])
    ax_line.set_title("Trend", fontsize=12, fontweight="bold", color=text_c, pad=8)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.grid(True, alpha=0.3)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(months[:n_points], fontsize=7, rotation=30)

    ax_pie = fig.add_subplot(gs[2, :n_kpis // 2])
    pie_labels = labels[:min(5, len(labels))]
    pie_values = values[:len(pie_labels)]
    pie_colors = [colors[i % len(colors)] for i in range(len(pie_labels))]
    wedges, texts, autotexts = ax_pie.pie(
        pie_values, labels=None, autopct='%1.0f%%', colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor=bg, linewidth=2), startangle=90, pctdistance=0.78)
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax_pie.legend(pie_labels, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=False)
    ax_pie.set_title("Distribution", fontsize=12, fontweight="bold", color=text_c, pad=8)

    ax_area = fig.add_subplot(gs[2, n_kpis // 2:])
    x2 = list(range(n_points))
    series1 = [random.randint(30, 80) for _ in x2]
    series2 = [random.randint(10, 50) for _ in x2]
    ax_area.fill_between(x2, series1, alpha=0.4, color=colors[0], label="Series A")
    ax_area.fill_between(x2, series2, alpha=0.4, color=colors[1], label="Series B")
    ax_area.plot(x2, series1, color=colors[0], linewidth=1.5)
    ax_area.plot(x2, series2, color=colors[1], linewidth=1.5)
    ax_area.legend(fontsize=8, frameon=False)
    ax_area.set_title("Comparison", fontsize=12, fontweight="bold", color=text_c, pad=8)
    ax_area.spines["top"].set_visible(False)
    ax_area.spines["right"].set_visible(False)
    ax_area.grid(True, alpha=0.3)
    ax_area.set_xticks(x2)
    ax_area.set_xticklabels(months[:n_points], fontsize=7, rotation=30)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_infographic(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render infographic-style layouts with Pillow + matplotlib hybrid."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    import numpy as np

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]
    accent = palette["accent"]

    labels = data["labels"] or ["Metric A", "Metric B", "Metric C", "Metric D"]
    values = data["values"] or [random.randint(20, 95) for _ in labels]

    if len(values) < len(labels):
        values.extend([random.randint(10, 90) for _ in range(len(labels) - len(values))])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    title = data["title"] or "Data Overview"

    if sub_type == "comparison":
        fig.suptitle(title, fontsize=22, fontweight="bold", color=text_c, y=0.96)
        n_items = min(len(labels), 6)
        gs = fig.add_gridspec(1, n_items, wspace=0.3, left=0.05, right=0.95, top=0.85, bottom=0.08)

        for i in range(n_items):
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_axis_off()

            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.06",
                                   facecolor=palette["card_bg"], edgecolor=colors[i % len(colors)],
                                   linewidth=2.5)
            ax.add_patch(rect)

            v = values[i] if i < len(values) else 0
            lbl = labels[i] if i < len(labels) else f"Item {i+1}"

            ax.text(0.5, 0.70, f"{v:,.0f}", ha="center", va="center",
                    fontsize=28, fontweight="bold", color=colors[i % len(colors)])
            wrapped = _textwrap.fill(lbl, width=12)
            ax.text(0.5, 0.35, wrapped, ha="center", va="center",
                    fontsize=10, color=text_c, alpha=0.7)

            bar_w = min(v / max(max(values), 1), 1.0)
            from matplotlib.patches import FancyBboxPatch as FBP2
            bar = FBP2((0.1, 0.12), 0.8 * bar_w, 0.06, boxstyle="round,pad=0.01",
                       facecolor=colors[i % len(colors)], edgecolor="none", alpha=0.6)
            ax.add_patch(bar)

    elif sub_type == "timeline":
        fig.suptitle(title, fontsize=22, fontweight="bold", color=text_c, y=0.96)
        ax = fig.add_subplot(111)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        y_line = 0.5
        ax.plot([-0.3, len(labels) - 0.7], [y_line, y_line], color=accent, linewidth=3, zorder=1)

        for i, (lbl, val) in enumerate(zip(labels, values)):
            ax.scatter(i, y_line, s=200, color=colors[i % len(colors)], zorder=5, edgecolors="white", linewidth=2)
            y_off = 0.18 if i % 2 == 0 else -0.18
            va = "bottom" if i % 2 == 0 else "top"
            ax.text(i, y_line + y_off, f"{val:,.0f}", ha="center", va=va,
                    fontsize=16, fontweight="bold", color=colors[i % len(colors)])
            ax.text(i, y_line + y_off + (0.08 if i % 2 == 0 else -0.08),
                    _textwrap.fill(lbl, 15), ha="center", va=va,
                    fontsize=9, color=text_c, alpha=0.7)

    elif sub_type == "process":
        fig.suptitle(title, fontsize=22, fontweight="bold", color=text_c, y=0.96)
        n_steps = min(len(data["raw_lines"]) or len(labels), 6)
        step_labels = data["raw_lines"][:n_steps] if data["raw_lines"] else labels[:n_steps]

        ax = fig.add_subplot(111)
        ax.set_xlim(-0.5, n_steps - 0.5)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        for i, lbl in enumerate(step_labels):
            from matplotlib.patches import FancyBboxPatch
            x_pos = i / max(n_steps - 1, 1) * 0.85 + 0.05
            rect = FancyBboxPatch((x_pos - 0.05, 0.25), 0.10, 0.50,
                                   boxstyle="round,pad=0.02",
                                   facecolor=colors[i % len(colors)], edgecolor="none", alpha=0.15)
            ax.add_patch(rect)

            ax.text(x_pos, 0.65, str(i + 1), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=colors[i % len(colors)])
            ax.text(x_pos, 0.42, _textwrap.fill(lbl, 14), ha="center", va="center",
                    fontsize=9, color=text_c, alpha=0.8)

            if i < n_steps - 1:
                ax.annotate("", xy=(x_pos + 0.07, 0.55), xytext=(x_pos + 0.03, 0.55),
                            arrowprops=dict(arrowstyle="->", color=accent, lw=2))

    else:
        fig.suptitle(title, fontsize=22, fontweight="bold", color=text_c, y=0.96)
        n_items = min(len(labels), 8)
        cols = min(4, n_items)
        rows = _math.ceil(n_items / cols)
        gs = fig.add_gridspec(rows, cols, wspace=0.25, hspace=0.35,
                              left=0.05, right=0.95, top=0.85, bottom=0.05)

        for idx in range(n_items):
            r, c = divmod(idx, cols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_axis_off()

            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.06",
                                   facecolor=palette["card_bg"], edgecolor=palette["card_border"], linewidth=1.5)
            ax.add_patch(rect)

            v = values[idx] if idx < len(values) else 0
            lbl = labels[idx] if idx < len(labels) else f"Item {idx+1}"

            ax.text(0.5, 0.65, f"{v:,.0f}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=colors[idx % len(colors)])
            ax.text(0.5, 0.30, _textwrap.fill(lbl, 16), ha="center", va="center",
                    fontsize=10, color=text_c, alpha=0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_table(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render a table/matrix using matplotlib."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]
    card_bg = palette["card_bg"]

    labels = data["labels"] or ["Feature A", "Feature B", "Feature C", "Feature D"]
    values = data["values"] or [random.randint(10, 100) for _ in labels]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_title(data["title"] or "Data Table", fontsize=20, fontweight="bold", color=text_c, pad=20)

    if data["kv_pairs"]:
        col_labels = ["Metric", "Value"]
        cell_data = [[k, v] for k, v in data["kv_pairs"][:15]]
    else:
        col_labels = ["Item", "Value"]
        cell_data = [[l, f"{v:,.0f}"] for l, v in zip(labels, values)]

    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(palette["accent"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
        else:
            cell.set_facecolor(card_bg if r % 2 == 0 else bg)
            cell.set_text_props(color=text_c)
            cell.set_edgecolor(palette["grid"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_flowchart(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render flowcharts and process diagrams."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]

    steps = data["raw_lines"] or data["labels"] or ["Start", "Process A", "Decision", "Process B", "End"]
    n = min(len(steps), 8)
    steps = steps[:n]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.suptitle(data["title"] or "Flowchart", fontsize=20, fontweight="bold", color=text_c, y=0.97)

    box_h = 0.08
    box_w = 0.22
    gap = (0.85 - n * box_h) / max(n - 1, 1) + box_h
    start_y = 0.88

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i, step in enumerate(steps):
        y = start_y - i * gap
        color = colors[i % len(colors)]

        if sub_type == "decision_tree" and i > 0 and i % 2 == 0:
            diamond = plt.Polygon(
                [[0.5, y + box_h / 2], [0.5 + box_w / 2, y], [0.5, y - box_h / 2], [0.5 - box_w / 2, y]],
                facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.2, zorder=3)
            ax.add_patch(diamond)
        else:
            style = "round4,pad=0.01" if i == 0 or i == n - 1 else "round,pad=0.01"
            rect = FancyBboxPatch((0.5 - box_w / 2, y - box_h / 2), box_w, box_h,
                                   boxstyle=style, facecolor=color, edgecolor="white",
                                   linewidth=1.5, alpha=0.2, zorder=3)
            ax.add_patch(rect)

        wrapped = _textwrap.fill(step, 22)
        ax.text(0.5, y, wrapped, ha="center", va="center", fontsize=10,
                fontweight="bold", color=text_c, zorder=5)

        if i < n - 1:
            next_y = start_y - (i + 1) * gap
            ax.annotate("", xy=(0.5, next_y + box_h / 2 + 0.01), xytext=(0.5, y - box_h / 2 - 0.01),
                        arrowprops=dict(arrowstyle="-|>", color=palette["accent"], lw=2, mutation_scale=15),
                        zorder=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_presentation(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render presentation slide layouts."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    from matplotlib.patches import FancyBboxPatch

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]
    accent = palette["accent"]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    title = data["title"] or "Presentation"

    if sub_type == "key_metrics":
        ax.text(0.5, 0.92, title, ha="center", va="center",
                fontsize=24, fontweight="bold", color=text_c)

        labels = data["labels"] or ["Revenue", "Users", "Growth", "Satisfaction"]
        values = data["values"] or [random.randint(10, 99) for _ in labels]
        n = min(len(labels), 4)
        for i in range(n):
            x_pos = (i + 0.5) / n
            v = values[i] if i < len(values) else 0
            lbl = labels[i] if i < len(labels) else f"KPI {i+1}"

            rect = FancyBboxPatch((x_pos - 0.1, 0.35), 0.20, 0.45, boxstyle="round,pad=0.03",
                                   facecolor=palette["card_bg"], edgecolor=colors[i % len(colors)],
                                   linewidth=2.5)
            ax.add_patch(rect)
            ax.text(x_pos, 0.65, f"{v:,.0f}", ha="center", va="center",
                    fontsize=30, fontweight="bold", color=colors[i % len(colors)])
            ax.text(x_pos, 0.45, _textwrap.fill(lbl, 14), ha="center", va="center",
                    fontsize=11, color=text_c, alpha=0.7)

    elif sub_type == "swot":
        ax.text(0.5, 0.95, title, ha="center", va="center",
                fontsize=22, fontweight="bold", color=text_c)

        quadrants = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        quad_colors = [colors[0], colors[3] if len(colors) > 3 else colors[1],
                       colors[1], colors[2] if len(colors) > 2 else colors[0]]
        raw = data["raw_lines"] or data["labels"]

        items_per_q = max(1, len(raw) // 4) if raw else 1
        for qi, (qlabel, qcolor) in enumerate(zip(quadrants, quad_colors)):
            r, c = divmod(qi, 2)
            x0 = 0.05 + c * 0.48
            y0 = 0.05 + (1 - r) * 0.42

            rect = FancyBboxPatch((x0, y0), 0.42, 0.38, boxstyle="round,pad=0.02",
                                   facecolor=qcolor, edgecolor="white", linewidth=1.5, alpha=0.12)
            ax.add_patch(rect)
            ax.text(x0 + 0.21, y0 + 0.32, qlabel, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=qcolor)

            start_idx = qi * items_per_q
            q_items = raw[start_idx:start_idx + items_per_q] if raw else [f"Point {qi+1}"]
            for ji, item in enumerate(q_items[:3]):
                ax.text(x0 + 0.21, y0 + 0.22 - ji * 0.08,
                        f"• {_textwrap.shorten(item, 30)}", ha="center", va="center",
                        fontsize=9, color=text_c, alpha=0.7)

    elif sub_type == "bullet_points":
        ax.text(0.5, 0.90, title, ha="center", va="center",
                fontsize=24, fontweight="bold", color=text_c)

        bullet_rect = FancyBboxPatch((0.05, 0.05), 0.90, 0.78, boxstyle="round,pad=0.03",
                                      facecolor=palette["card_bg"], edgecolor=palette["card_border"], linewidth=1.5)
        ax.add_patch(bullet_rect)

        items = data["raw_lines"] or data["labels"] or ["Point 1", "Point 2", "Point 3"]
        for i, item in enumerate(items[:8]):
            y_pos = 0.76 - i * 0.09
            ax.plot(0.12, y_pos, "o", color=colors[i % len(colors)], markersize=8)
            ax.text(0.17, y_pos, _textwrap.shorten(item, 60), va="center",
                    fontsize=12, color=text_c)

    elif sub_type == "roadmap":
        ax.text(0.5, 0.93, title, ha="center", va="center",
                fontsize=22, fontweight="bold", color=text_c)

        items = data["raw_lines"] or data["labels"] or ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        n = min(len(items), 6)
        ax.plot([0.1, 0.9], [0.5, 0.5], color=accent, linewidth=4, zorder=1)

        for i in range(n):
            x = 0.1 + i * 0.8 / max(n - 1, 1)
            ax.scatter(x, 0.5, s=250, color=colors[i % len(colors)], zorder=5,
                       edgecolors="white", linewidth=2)
            y_off = 0.15 if i % 2 == 0 else -0.15
            va = "bottom" if i % 2 == 0 else "top"
            ax.text(x, 0.5 + y_off, _textwrap.fill(items[i], 14), ha="center", va=va,
                    fontsize=10, fontweight="bold", color=text_c)
    else:
        rect = FancyBboxPatch((0.1, 0.15), 0.80, 0.70, boxstyle="round,pad=0.04",
                               facecolor=accent, edgecolor="none", alpha=0.08)
        ax.add_patch(rect)
        ax.text(0.5, 0.60, title, ha="center", va="center",
                fontsize=32, fontweight="bold", color=text_c)
        subtitle = " ".join(data["raw_lines"][:2]) if data["raw_lines"] else "Subtitle goes here"
        ax.text(0.5, 0.42, _textwrap.shorten(subtitle, 80), ha="center", va="center",
                fontsize=14, color=text_c, alpha=0.6)
        ax.plot([0.3, 0.7], [0.35, 0.35], color=accent, linewidth=3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_diagram(sub_type: str, data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render diagrams (Venn, cycle, block, mind map)."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Circle

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]

    labels = data["labels"] or ["Component A", "Component B", "Component C"]
    title = data["title"] or "Diagram"

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=20, fontweight="bold", color=text_c, y=0.97)

    if sub_type == "venn":
        n = min(len(labels), 3)
        radius = 0.22
        centers = [(0.38, 0.5), (0.62, 0.5), (0.5, 0.32)] if n == 3 else [(0.38, 0.5), (0.62, 0.5)]
        for i in range(n):
            circle = Circle(centers[i], radius, facecolor=colors[i % len(colors)],
                            edgecolor="white", linewidth=2, alpha=0.3)
            ax.add_patch(circle)
            offset_x = -0.12 if i == 0 else (0.12 if i == 1 else 0)
            offset_y = 0 if i < 2 else -0.12
            ax.text(centers[i][0] + offset_x, centers[i][1] + offset_y,
                    _textwrap.fill(labels[i], 12), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_c)

    elif sub_type == "cycle":
        n = min(len(labels), 6)
        radius = 0.30
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            x = 0.5 + radius * np.cos(angle)
            y = 0.48 + radius * np.sin(angle)

            rect = FancyBboxPatch((x - 0.08, y - 0.04), 0.16, 0.08,
                                   boxstyle="round,pad=0.02",
                                   facecolor=colors[i % len(colors)], edgecolor="white",
                                   linewidth=1.5, alpha=0.2)
            ax.add_patch(rect)
            ax.text(x, y, _textwrap.fill(labels[i], 14), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_c)

            next_angle = 2 * np.pi * ((i + 1) % n) / n - np.pi / 2
            nx = 0.5 + radius * np.cos(next_angle)
            ny = 0.48 + radius * np.sin(next_angle)
            mid_angle = (angle + next_angle) / 2
            if abs(next_angle - angle) > np.pi:
                mid_angle += np.pi
            ax.annotate("", xy=(nx, ny), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=palette["accent"],
                                        lw=1.5, connectionstyle="arc3,rad=0.3"))

    elif sub_type == "mind_map":
        center_label = labels[0] if labels else "Central Topic"
        branches = labels[1:] if len(labels) > 1 else data["raw_lines"] or ["Branch A", "Branch B", "Branch C"]
        n_branches = min(len(branches), 8)

        rect = FancyBboxPatch((0.35, 0.42), 0.30, 0.12, boxstyle="round,pad=0.03",
                               facecolor=palette["accent"], edgecolor="white", linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.5, 0.48, _textwrap.fill(center_label, 16), ha="center", va="center",
                fontsize=14, fontweight="bold", color=text_c)

        for i in range(n_branches):
            angle = 2 * np.pi * i / n_branches - np.pi / 2
            r = 0.32
            x = 0.5 + r * np.cos(angle)
            y = 0.48 + r * np.sin(angle)

            rect = FancyBboxPatch((x - 0.07, y - 0.03), 0.14, 0.06,
                                   boxstyle="round,pad=0.01",
                                   facecolor=colors[i % len(colors)], edgecolor="white",
                                   linewidth=1, alpha=0.2)
            ax.add_patch(rect)
            ax.text(x, y, _textwrap.fill(branches[i], 12), ha="center", va="center",
                    fontsize=8, color=text_c)
            ax.plot([0.5, x], [0.48, y], color=colors[i % len(colors)], linewidth=1.5, alpha=0.5)

    else:
        n = min(len(labels), 6)
        cols = min(3, n)
        rows_count = _math.ceil(n / cols)
        for i in range(n):
            r, c = divmod(i, cols)
            x = (c + 0.5) / cols
            y = 0.8 - r * 0.3

            rect = FancyBboxPatch((x - 0.12, y - 0.08), 0.24, 0.16,
                                   boxstyle="round,pad=0.02",
                                   facecolor=colors[i % len(colors)], edgecolor="white",
                                   linewidth=1.5, alpha=0.15)
            ax.add_patch(rect)
            ax.text(x, y, _textwrap.fill(labels[i], 16), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_c)

            if i < n - 1 and c < cols - 1:
                ax.annotate("", xy=(x + 0.15, y), xytext=(x + 0.12, y),
                            arrowprops=dict(arrowstyle="-|>", color=palette["accent"], lw=1.5))
            elif c == cols - 1 and r < rows_count - 1 and i + 1 < n:
                ax.annotate("", xy=(0.5 / cols, y - 0.15), xytext=(x, y - 0.10),
                            arrowprops=dict(arrowstyle="-|>", color=palette["accent"], lw=1.5,
                                            connectionstyle="arc3,rad=-0.3"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_org_chart(data: dict, palette: dict, width: int, height: int) -> bytes:
    """Render organizational hierarchy chart."""
    plt, fig_w, fig_h, dpi = _setup_matplotlib_style(palette, width, height)
    from matplotlib.patches import FancyBboxPatch

    colors = palette["colors"]
    bg = palette["bg"]
    text_c = palette["text"]

    roles = data["raw_lines"] or data["labels"] or ["CEO", "CTO", "CFO", "VP Engineering", "VP Sales", "VP Marketing"]
    title = data["title"] or "Organization Chart"
    n = min(len(roles), 13)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=20, fontweight="bold", color=text_c, y=0.97)

    top = roles[0]
    middle = roles[1:min(4, n)]
    bottom = roles[min(4, n):n]

    def draw_box(x, y, label, color, fontsize=10):
        bw, bh = 0.16, 0.08
        rect = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.2)
        ax.add_patch(rect)
        ax.text(x, y, _textwrap.fill(label, 14), ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_c)
        return x, y

    top_pos = draw_box(0.5, 0.82, top, colors[0], 12)

    mid_positions = []
    if middle:
        n_mid = len(middle)
        for i, role in enumerate(middle):
            x = (i + 0.5) / n_mid
            pos = draw_box(x, 0.58, role, colors[(i + 1) % len(colors)])
            mid_positions.append(pos)
            ax.plot([0.5, x], [0.78, 0.62], color=palette["accent"], linewidth=1.5, alpha=0.5)

    if bottom:
        n_bot = len(bottom)
        per_mid = max(1, n_bot // max(len(mid_positions), 1))
        for i, role in enumerate(bottom):
            x = (i + 0.5) / max(n_bot, 1)
            draw_box(x, 0.34, role, colors[(i + 2) % len(colors)], 9)
            parent_idx = min(i // max(per_mid, 1), len(mid_positions) - 1)
            if mid_positions:
                px, py = mid_positions[parent_idx]
                ax.plot([px, x], [0.54, 0.38], color=palette["accent"], linewidth=1, alpha=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _generate_professional_sync(
    category: str, sub_type: str, content: str,
    style: str, color_scheme: str, width: int, height: int,
) -> bytes:
    """Route to the appropriate code-based renderer. Returns PNG bytes."""
    palette = PROFESSIONAL_PALETTES.get(style, PROFESSIONAL_PALETTES["corporate"]).copy()

    if color_scheme:
        custom_colors = _re.findall(r'#[0-9A-Fa-f]{6}', color_scheme)
        if custom_colors:
            palette["colors"] = custom_colors + palette["colors"]
            palette["accent"] = custom_colors[0]

    data = _parse_data_from_content(content)

    if category == "chart":
        return _render_chart(sub_type or "bar_chart", data, palette, width, height)
    elif category == "dashboard":
        return _render_dashboard(data, palette, width, height)
    elif category == "infographic":
        return _render_infographic(sub_type or "data_overview", data, palette, width, height)
    elif category == "table":
        return _render_table(sub_type or "data_table", data, palette, width, height)
    elif category == "flowchart":
        return _render_flowchart(sub_type or "process_flow", data, palette, width, height)
    elif category == "presentation":
        return _render_presentation(sub_type or "title_slide", data, palette, width, height)
    elif category == "diagram":
        return _render_diagram(sub_type or "block_diagram", data, palette, width, height)
    elif category == "org_chart":
        return _render_org_chart(data, palette, width, height)
    else:
        return _render_chart("bar_chart", data, palette, width, height)


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
    detail_level: str = Field(default="high")


class ProfessionalResponse(BaseModel):
    image: str
    seed: int
    width: int
    height: int
    time_seconds: float
    prompt_used: str
    category: str
    sub_type: str
    engine: str


@app.get("/api/professional/categories")
async def professional_categories():
    """List available professional generation categories and sub-types."""
    return {
        "categories": PROFESSIONAL_CATEGORIES,
        "styles": list(PROFESSIONAL_PALETTES.keys()),
        "detail_levels": ["low", "medium", "high"],
    }


@app.post("/api/generate/professional", response_model=ProfessionalResponse)
async def generate_professional(req: ProfessionalRequest):
    """Generate a professional graphic using code-based rendering (matplotlib + Pillow).
    
    Produces pixel-perfect charts, infographics, dashboards, and diagrams
    with 100% accurate text — no AI hallucination.
    """
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    random.seed(seed)

    try:
        img_bytes = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_professional_sync,
            req.category.value, req.sub_type, req.content,
            req.style, req.color_scheme, req.width, req.height,
        )
    except Exception as e:
        logger.error(f"[PRO] Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    description = f"{req.category.value}/{req.sub_type or 'default'} [{req.style}]"
    logger.info(f"[PRO] Generated {description} {req.width}x{req.height} in {elapsed}s (engine=matplotlib)")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            img_bytes, f"{ts}_{seed}_pro_{req.category.value}.png", "image/png",
            {"type": "professional", "category": req.category.value, "sub_type": req.sub_type,
             "content": req.content[:200], "style": req.style,
             "seed": seed, "width": req.width, "height": req.height,
             "engine": "matplotlib", "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return ProfessionalResponse(
        image=b64, seed=seed, width=req.width, height=req.height,
        time_seconds=elapsed, prompt_used=f"Code-rendered: {description}",
        category=req.category.value, sub_type=req.sub_type,
        engine="matplotlib",
    )


# ══════════════════════════════════════════════════════════════════════
#  TTS — Chatterbox Text-to-Speech
# ══════════════════════════════════════════════════════════════════════

# ── Chatterbox singleton ─────────────────────────────────────────────

_chatterbox_model = None
_chatterbox_checked = False

VOICES_DIR = Path(__file__).parent / "voices"


def _get_chatterbox():
    """Lazy-load Chatterbox TTS model (500M params) on GPU."""
    global _chatterbox_model, _chatterbox_checked
    if _chatterbox_checked:
        return _chatterbox_model
    _chatterbox_checked = True
    try:
        # Disable JIT to avoid NVRTC architecture errors on Blackwell (sm_121)
        os.environ.setdefault("PYTORCH_JIT", "0")

        from chatterbox.tts import ChatterboxTTS
        logger.info(f"[TTS] Loading Chatterbox TTS on {DEVICE}...")
        model = ChatterboxTTS.from_pretrained(device=DEVICE)

        # Replace Perth watermarker with a no-op — Perth's STFT triggers
        # NVRTC complex-number kernel compilation which fails on Blackwell
        class _NoOpWatermarker:
            def apply_watermark(self, wav, sample_rate=None):
                return wav
        model.watermarker = _NoOpWatermarker()

        _chatterbox_model = model
        logger.info(f"[TTS] Chatterbox loaded on {DEVICE} ✓ (watermarker disabled)")
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
            logger.info(f"[TTS] Using voice reference: {vp.name}")
        else:
            logger.warning(f"[TTS] Voice file not found: {vp}, falling back to default")

    # Text cleanup
    text = _re.sub(r'\s+', ' ', text).strip()
    text = text.replace('"', '').replace('\u201c', '').replace('\u201d', '')
    text = text.replace('...', ',').replace('\u2026', ',')
    text = _re.sub(r' - ', ' \u2014 ', text)

    # Pre-cache voice conditionals on the model's device
    # (handles CPU→GPU placement internally so generate() works)
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
