"""
SDXL / FLUX / Hunyuan — FastAPI Backend (CUDA multi-GPU)

Clean architecture:
  - Each GPU gets one dedicated model
  - Simple dict cache: _pipelines["model:gpu"] = pipeline
  - One lock per GPU
  - All core models preloaded at startup, cached forever in VRAM
  - Component-by-component loading: low_cpu_mem_usage=True → .to(device)
    Minimal CPU RAM footprint (~2-3 GB peak per component)

GPU Layout (8 GPUs):
  GPU 0: SDXL Lightning           (~7 GB VRAM)
  GPU 1: RealVisXL V5 Lightning   (~7 GB VRAM)
  GPU 2: RealVisXL V5 Quality     (~7 GB VRAM)
  GPU 3: FLUX.1 Schnell           (~18 GB VRAM)
  GPU 4: Hunyuan Image            (on demand)
  GPU 5: Hunyuan Video            (on demand)
  GPU 6: Hunyuan 3D               (on demand)
  GPU 7: spare

Models:
  1. "lightning"        — SDXL Lightning 4-step      (~1s)
  2. "realvis_fast"     — RealVisXL V5.0 Lightning   (~2s)
  3. "realvis_quality"  — RealVisXL V5.0 25-step     (~8s)
  4. "flux"             — FLUX.1 Schnell 4-step      (~20-30s)
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

# Suppress diffusers/accelerate meta-tensor copy warnings (thousands of lines of noise)
warnings.filterwarnings("ignore", message=".*non-meta.*meta.*no-op.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
from enum import Enum
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image as PILImage

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Verify CUDA ───────────────────────────────────────────────────────
if not torch.cuda.is_available():
    logger.error("CUDA not available! Use server.py for MPS/CPU.")
    raise SystemExit(1)

NUM_GPUS = torch.cuda.device_count()
logger.info(f"CUDA: {NUM_GPUS} GPU(s) detected")

# List GPUs using nvidia-smi to AVOID creating CUDA contexts on all GPUs.
# Each torch.cuda.get_device_name(i) creates a ~300 MB CUDA context on GPU i.
# We only want contexts on GPUs we actually load models onto (0-3).
try:
    _smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    if _smi.returncode == 0:
        for line in _smi.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                logger.info(f"  GPU {parts[0]}: {parts[1]} ({int(parts[2])/1024:.1f} GB)")
    del _smi
except Exception:
    # Fallback: only enumerate first 4 GPUs to minimise context creation
    for i in range(min(NUM_GPUS, 4)):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

# ── Optional 3D ───────────────────────────────────────────────────────
_HAS_HY3DGEN = False
try:
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    _HAS_HY3DGEN = True
    logger.info("hy3dgen available ✓")
except ImportError:
    logger.warning("hy3dgen not installed — 3D disabled")

# ── Constants ─────────────────────────────────────────────────────────
SDXL_DTYPE = torch.float16
FLUX_DTYPE = torch.bfloat16

# ── GPU Assignments ───────────────────────────────────────────────────
# NOTE: No duplicate models. Each model gets exactly one GPU.
# This keeps startup RAM usage manageable on 32 GB systems.
if NUM_GPUS >= 8:
    GPU_MAP = {
        "lightning": [0],
        "realvis_fast": [1],
        "realvis_quality": [2],
        "flux": [3],
        "hunyuan_image": [4],
        "hunyuan_video": [5],
        "hunyuan_video_i2v": [5],
        "hunyuan_3d": [6],
    }
elif NUM_GPUS >= 4:
    GPU_MAP = {
        "lightning": [0],
        "realvis_fast": [1],
        "realvis_quality": [2],
        "flux": [3],
        "hunyuan_image": [0],
        "hunyuan_video": [0],
        "hunyuan_video_i2v": [0],
        "hunyuan_3d": [0],
    }
else:
    GPU_MAP = {k: [0] for k in [
        "lightning", "realvis_fast", "realvis_quality", "flux",
        "hunyuan_image", "hunyuan_video", "hunyuan_video_i2v", "hunyuan_3d",
    ]}

# GPUs that will be initialised during startup warmup (CUDA contexts created here only)
_STARTUP_GPUS = set()
for _m in ["lightning", "realvis_fast", "realvis_quality", "flux"]:
    _STARTUP_GPUS.update(GPU_MAP.get(_m, []))

for model, gpus in GPU_MAP.items():
    logger.info(f"  {model}: GPU {gpus}")

# ── Pipeline Cache & Locks ────────────────────────────────────────────
_pipelines: dict = {}                                          # "model:gpu" → pipeline
_gpu_locks = {i: threading.Lock() for i in range(NUM_GPUS)}
_rr_counters: dict = {}                                        # round-robin per model
_active_tasks: dict = {}                                       # gpu_id → task name
_gen_counts: dict = {}                                         # gpu_id → count
_executor = ThreadPoolExecutor(max_workers=max(NUM_GPUS * 2, 4))


def _pick_gpu(model: str) -> int:
    """Pick next GPU for this model (round-robin if multiple)."""
    gpus = GPU_MAP.get(model, [0])
    if len(gpus) == 1:
        return gpus[0]
    idx = _rr_counters.get(model, 0)
    gpu = gpus[idx % len(gpus)]
    _rr_counters[model] = idx + 1
    return gpu


# ── Helpers ───────────────────────────────────────────────────────────
def _disable_safety(pipe):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _enable_fast_attention(pipe):
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
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


# ── Memory helpers ────────────────────────────────────────────────────

def _flush_ram():
    """Aggressively free CPU RAM after moving a component to GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def _ensure_swap():
    """Create a 16 GB swap file if total swap < 8 GB.  Needs sudo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    swap_kb = int(line.split()[1])
                    if swap_kb >= 8 * 1024 * 1024:
                        return  # already enough
                    break
        swapfile = "/swapfile_ai"
        if not os.path.exists(swapfile):
            logger.warning("Swap < 8 GB — creating 16 GB swap file (needs sudo)...")
            os.system(f"sudo fallocate -l 16G {swapfile} && "
                      f"sudo chmod 600 {swapfile} && "
                      f"sudo mkswap {swapfile} && "
                      f"sudo swapon {swapfile}")
            logger.info("16 GB swap file activated ✓")
        else:
            os.system(f"sudo swapon {swapfile} 2>/dev/null")
    except Exception as e:
        logger.warning(f"Could not ensure swap: {e}")


def _ram_usage_mb():
    """Get current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _vram_summary():
    """One-line VRAM usage summary across used GPUs."""
    parts = []
    for i in sorted(_STARTUP_GPUS):
        try:
            used = torch.cuda.memory_allocated(i) / 1024**3
            parts.append(f"GPU{i}:{used:.1f}G")
        except Exception:
            parts.append(f"GPU{i}:?")
    return " ".join(parts)


# ── Pipeline Loaders — Component-by-component to GPU ─────────────────
#
# Strategy:  low_cpu_mem_usage=True  →  .to(device)  →  gc.collect()
#
# low_cpu_mem_usage=True creates the model on CPU but loads weights
# one shard at a time (never the full state_dict at once).
# .to(device) then moves parameters IN-PLACE to GPU one by one.
# gc.collect() frees any remaining CPU references.
#
# Peak CPU RAM per component: ~size of the largest safetensors shard
# (~2.5 GB for UNet, ~5 GB for FLUX transformer shards).
#
# We do NOT use device_map={"": device} because accelerate's
# load_checkpoint_and_dispatch() loads the FULL state_dict to CPU
# before dispatching, which doubles peak RAM.

def _load_sdxl_to_gpu(repo_id: str, gpu_id: int, variant: str = "fp16",
                       unet_repo: str = None, unet_subfolder: str = "unet",
                       unet_weights_path: str = None,
                       scheduler_trailing: bool = False):
    """Generic SDXL loader — load each component to CPU then move to GPU.

    Args:
        repo_id:            HuggingFace repo for the SDXL pipeline
        gpu_id:             Target GPU index
        variant:            Checkpoint variant ("fp16" or None)
        unet_repo:          If different from repo_id (e.g. base SDXL for Lightning)
        unet_subfolder:     Subfolder within the repo for the UNet
        unet_weights_path:  Optional safetensors file to load_state_dict into UNet
        scheduler_trailing: Use trailing timestep_spacing (for distilled models)
    """
    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    device = f"cuda:{gpu_id}"
    unet_source = unet_repo or repo_id

    # ── 1. UNet (~2.5 GB fp16) ──
    logger.info(f"    UNet → CPU then GPU {gpu_id}...")
    unet = UNet2DConditionModel.from_pretrained(
        unet_source, subfolder=unet_subfolder,
        torch_dtype=SDXL_DTYPE, variant=variant,
        low_cpu_mem_usage=True,
    )
    if unet_weights_path:
        from safetensors.torch import load_file
        sd = load_file(unet_weights_path, device="cpu")
        unet.load_state_dict(sd, strict=False)
        del sd
        gc.collect()
    unet = unet.to(device)
    _flush_ram()
    logger.info(f"    UNet on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")

    # ── 2. VAE (~170 MB fp16) ──
    logger.info(f"    VAE → GPU {gpu_id}...")
    vae = AutoencoderKL.from_pretrained(
        repo_id, subfolder="vae",
        torch_dtype=SDXL_DTYPE, variant=variant,
        low_cpu_mem_usage=True,
    )
    vae = vae.to(device)
    _flush_ram()

    # ── 3. Text Encoder 1 — CLIP (~500 MB) ──
    logger.info(f"    CLIP text encoder → GPU {gpu_id}...")
    text_encoder = CLIPTextModel.from_pretrained(
        repo_id, subfolder="text_encoder",
        torch_dtype=SDXL_DTYPE, variant=variant,
        low_cpu_mem_usage=True,
    )
    text_encoder = text_encoder.to(device)
    _flush_ram()

    # ── 4. Text Encoder 2 — CLIP-G (~1.4 GB) ──
    logger.info(f"    CLIP-G text encoder → GPU {gpu_id}...")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        repo_id, subfolder="text_encoder_2",
        torch_dtype=SDXL_DTYPE, variant=variant,
        low_cpu_mem_usage=True,
    )
    text_encoder_2 = text_encoder_2.to(device)
    _flush_ram()

    # ── 5. Tokenizers + Scheduler (CPU, tiny) ──
    tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer_2")
    scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
    if scheduler_trailing:
        scheduler = EulerDiscreteScheduler.from_config(
            scheduler.config, timestep_spacing="trailing"
        )

    # ── 6. Assemble pipeline — everything already on GPU ──
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline(
        unet=unet, vae=vae,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )
    _enable_fast_attention(pipe)
    _disable_safety(pipe)
    return pipe


def _load_lightning(gpu_id: int):
    from huggingface_hub import hf_hub_download

    logger.info(f"Loading Lightning → GPU {gpu_id}...")
    unet_path = hf_hub_download(
        "ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors"
    )
    pipe = _load_sdxl_to_gpu(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        gpu_id=gpu_id,
        unet_weights_path=unet_path,
        scheduler_trailing=True,
    )
    logger.info(f"Lightning loaded on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")
    return pipe


def _load_realvis_fast(gpu_id: int):
    logger.info(f"Loading RealVisXL V5 Lightning → GPU {gpu_id}...")
    pipe = _load_sdxl_to_gpu(
        repo_id="SG161222/RealVisXL_V5.0_Lightning",
        gpu_id=gpu_id,
        scheduler_trailing=True,
    )
    logger.info(f"RealVisXL V5 Lightning loaded on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")
    return pipe


def _load_realvis_quality(gpu_id: int):
    logger.info(f"Loading RealVisXL V5 Quality → GPU {gpu_id}...")
    pipe = _load_sdxl_to_gpu(
        repo_id="SG161222/RealVisXL_V5.0",
        gpu_id=gpu_id,
        scheduler_trailing=False,
    )
    logger.info(f"RealVisXL V5 Quality loaded on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")
    return pipe


def _load_flux(gpu_id: int):
    """Load FLUX.1 Schnell — component by component to GPU.

    Total VRAM: ~18 GB (fits in 24 GB RTX 3090).
    CPU RAM peak: ~5 GB (one T5/transformer shard at a time, then moved).
    """
    from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast

    device = f"cuda:{gpu_id}"
    model_id = "black-forest-labs/FLUX.1-schnell"
    logger.info(f"Loading FLUX.1 Schnell → GPU {gpu_id}...")

    # 1. Transformer (~12 GB bfloat16) — largest component
    logger.info(f"    FLUX transformer → CPU then GPU {gpu_id} (~12 GB)...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=FLUX_DTYPE,
        low_cpu_mem_usage=True,
    )
    transformer = transformer.to(device)
    _flush_ram()
    logger.info(f"    FLUX transformer on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")

    # 2. T5 text encoder (~5 GB bfloat16)
    logger.info(f"    FLUX T5 encoder → GPU {gpu_id} (~5 GB)...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=FLUX_DTYPE,
        low_cpu_mem_usage=True,
    )
    text_encoder_2 = text_encoder_2.to(device)
    _flush_ram()
    logger.info(f"    FLUX T5 on GPU {gpu_id} ✓  (RAM {_ram_usage_mb()}MB)")

    # 3. CLIP text encoder (~0.5 GB)
    logger.info(f"    FLUX CLIP encoder → GPU {gpu_id}...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=FLUX_DTYPE,
        low_cpu_mem_usage=True,
    )
    text_encoder = text_encoder.to(device)
    _flush_ram()

    # 4. VAE (~0.2 GB)
    logger.info(f"    FLUX VAE → GPU {gpu_id}...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=FLUX_DTYPE,
        low_cpu_mem_usage=True,
    )
    vae = vae.to(device)
    _flush_ram()

    # 5. Tokenizers + Scheduler (CPU only, tiny)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 6. Assemble pipeline — all components already on GPU
    pipe = FluxPipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )
    _enable_fast_attention(pipe)

    logger.info(f"FLUX.1 Schnell loaded on GPU {gpu_id} ✓ (~18 GB VRAM, RAM {_ram_usage_mb()}MB)")
    return pipe


def _load_hunyuan_image(gpu_id: int):
    from diffusers import HunyuanDiTPipeline

    device = f"cuda:{gpu_id}"
    logger.info(f"Loading HunyuanDiT → GPU {gpu_id}...")

    pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe = pipe.to(device)
    _flush_ram()
    _enable_fast_attention(pipe)

    logger.info(f"HunyuanDiT loaded on GPU {gpu_id} ✓")
    return pipe


def _load_hunyuan_video(gpu_id: int):
    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from transformers import BitsAndBytesConfig

    logger.info(f"Loading HunyuanVideo T2V (INT4) → GPU {gpu_id}...")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="transformer",
        quantization_config=quant, torch_dtype=torch.bfloat16,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", transformer=transformer, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    pipe.vae.enable_tiling()
    _flush_ram()

    logger.info(f"HunyuanVideo T2V loaded on GPU {gpu_id} ✓")
    return pipe


def _load_hunyuan_video_i2v(gpu_id: int):
    from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
    from transformers import BitsAndBytesConfig

    logger.info(f"Loading HunyuanVideo I2V (INT4) → GPU {gpu_id}...")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-I2V", subfolder="transformer",
        quantization_config=quant, torch_dtype=torch.bfloat16,
    )
    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-I2V", transformer=transformer, torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    pipe.vae.enable_tiling()
    _flush_ram()

    logger.info(f"HunyuanVideo I2V loaded on GPU {gpu_id} ✓")
    return pipe


def _load_hunyuan_3d(gpu_id: int):
    if not _HAS_HY3DGEN:
        raise RuntimeError("hy3dgen not installed")

    logger.info(f"Loading Hunyuan3D-2 → GPU {gpu_id}...")
    with torch.cuda.device(gpu_id):
        shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
        texgen = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        rembg = BackgroundRemover()

    logger.info(f"Hunyuan3D-2 loaded on GPU {gpu_id} ✓")
    return {"shapegen": shapegen, "texgen": texgen, "rembg": rembg}


# Loader registry
_LOADERS = {
    "lightning": _load_lightning,
    "realvis_fast": _load_realvis_fast,
    "realvis_quality": _load_realvis_quality,
    "flux": _load_flux,
    "hunyuan_image": _load_hunyuan_image,
    "hunyuan_video": _load_hunyuan_video,
    "hunyuan_video_i2v": _load_hunyuan_video_i2v,
    "hunyuan_3d": _load_hunyuan_3d,
}


def get_pipeline(model: str, gpu_id: int):
    """Get or load a pipeline. Cached per (model, gpu_id)."""
    key = f"{model}:{gpu_id}"
    if key in _pipelines:
        return _pipelines[key]
    loader = _LOADERS.get(model)
    if not loader:
        raise ValueError(f"Unknown model: {model}")
    pipe = loader(gpu_id)
    _pipelines[key] = pipe
    return pipe


# ── History ───────────────────────────────────────────────────────────
GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def save_to_history(data_bytes: bytes, filename: str, media_type: str, metadata: dict) -> str:
    (GENERATED_DIR / filename).write_bytes(data_bytes)
    meta = {**metadata, "filename": filename, "media_type": media_type}
    (GENERATED_DIR / f"{filename}.json").write_text(json.dumps(meta, indent=2))
    return filename


# ── Generation Functions ──────────────────────────────────────────────

def _generate_image_sync(
    model_mode: str, prompt: str, negative: str,
    width: int, height: int, seed: int,
    guidance_scale: float, num_inference_steps: int,
) -> tuple:
    """Generate one image. Thread-safe via per-GPU lock."""
    gpu_id = _pick_gpu(model_mode)

    with _gpu_locks[gpu_id]:
        _active_tasks[gpu_id] = model_mode
        try:
            pipeline = get_pipeline(model_mode, gpu_id)

            if model_mode == "flux":
                device = f"cuda:{gpu_id}"
                gen = torch.Generator(device).manual_seed(seed)
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
                device = f"cuda:{gpu_id}"
                gen = torch.Generator(device).manual_seed(seed)
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
            _active_tasks[gpu_id] = None
            _gen_counts[gpu_id] = _gen_counts.get(gpu_id, 0) + 1

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), seed


def _generate_hunyuan_image_sync(
    prompt: str, negative: str, width: int, height: int,
    seed: int, guidance_scale: float, num_inference_steps: int,
) -> tuple:
    gpu_id = _pick_gpu("hunyuan_image")
    with _gpu_locks[gpu_id]:
        _active_tasks[gpu_id] = "hunyuan_image"
        try:
            pipe = get_pipeline("hunyuan_image", gpu_id)
            gen = torch.Generator(f"cuda:{gpu_id}").manual_seed(seed)
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
            _active_tasks[gpu_id] = None
            _gen_counts[gpu_id] = _gen_counts.get(gpu_id, 0) + 1

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue(), seed


def _generate_video_sync(
    prompt: str, width: int, height: int, num_frames: int,
    seed: int, num_inference_steps: int, fps: int,
) -> tuple:
    from diffusers.utils import export_to_video

    gpu_id = _pick_gpu("hunyuan_video")
    with _gpu_locks[gpu_id]:
        _active_tasks[gpu_id] = "hunyuan_video"
        try:
            pipe = get_pipeline("hunyuan_video", gpu_id)
            gen = torch.Generator("cpu").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt, height=height, width=width,
                    num_frames=num_frames, num_inference_steps=num_inference_steps,
                    generator=gen,
                )
        finally:
            _active_tasks[gpu_id] = None
            _gen_counts[gpu_id] = _gen_counts.get(gpu_id, 0) + 1

    frames = result.frames[0]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
    try:
        export_to_video(frames, temp_path, fps=fps)
        with open(temp_path, "rb") as f:
            return f.read(), seed, len(frames)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _generate_video_i2v_sync(
    image: PILImage.Image, prompt: str,
    width: int, height: int, num_frames: int,
    seed: int, num_inference_steps: int, fps: int,
) -> tuple:
    from diffusers.utils import export_to_video

    gpu_id = _pick_gpu("hunyuan_video_i2v")
    with _gpu_locks[gpu_id]:
        _active_tasks[gpu_id] = "hunyuan_video_i2v"
        try:
            pipe = get_pipeline("hunyuan_video_i2v", gpu_id)
            gen = torch.Generator("cpu").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    image=image, prompt=prompt,
                    height=height, width=width,
                    num_frames=num_frames, num_inference_steps=num_inference_steps,
                    generator=gen,
                )
        finally:
            _active_tasks[gpu_id] = None
            _gen_counts[gpu_id] = _gen_counts.get(gpu_id, 0) + 1

    frames = result.frames[0]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
    try:
        export_to_video(frames, temp_path, fps=fps)
        with open(temp_path, "rb") as f:
            return f.read(), seed, len(frames)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _generate_3d_sync(image: PILImage.Image, do_texture: bool = True) -> bytes:
    if not _HAS_HY3DGEN:
        raise RuntimeError("hy3dgen not installed")

    gpu_id = _pick_gpu("hunyuan_3d")
    with _gpu_locks[gpu_id]:
        _active_tasks[gpu_id] = "hunyuan_3d"
        try:
            pipes = get_pipeline("hunyuan_3d", gpu_id)
            if image.mode == "RGB":
                image = pipes["rembg"](image.convert("RGBA"))
            elif image.mode != "RGBA":
                image = image.convert("RGBA")

            with torch.inference_mode():
                mesh = pipes["shapegen"](image=image)[0]
            if do_texture:
                with torch.inference_mode():
                    mesh = pipes["texgen"](mesh, image=image)
        finally:
            _active_tasks[gpu_id] = None
            _gen_counts[gpu_id] = _gen_counts.get(gpu_id, 0) + 1

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        temp_path = tmp.name
    try:
        mesh.export(temp_path)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ══════════════════════════════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(title="AI Generator (CUDA)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Startup warmup ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    async def _warmup():
        # ── 0. Ensure enough swap ──
        _ensure_swap()

        logger.info("=" * 70)
        logger.info("  WARMUP — preloading 4 image models sequentially")
        logger.info(f"  System RAM: 32 GB | GPUs: {NUM_GPUS}x RTX 3090 (24 GB each)")
        logger.info(f"  Startup GPUs: {sorted(_STARTUP_GPUS)}")
        logger.info("  Loading strategy: low_cpu_mem_usage → .to(device) → gc")
        logger.info("=" * 70)
        loop = asyncio.get_event_loop()
        t_start = time.time()

        # Only preload the 4 core image models (Hunyuan loaded on-demand)
        models_to_load = ["lightning", "realvis_fast", "realvis_quality", "flux"]
        loaded = 0
        failed = 0

        for model in models_to_load:
            for gpu_id in GPU_MAP.get(model, []):
                elapsed = int(time.time() - t_start)
                ram = _ram_usage_mb()
                logger.info(f"  [{elapsed:>4d}s | RAM {ram}MB] Loading {model} → GPU {gpu_id}...")
                t0 = time.time()
                try:
                    await loop.run_in_executor(_executor, get_pipeline, model, gpu_id)
                    dt = time.time() - t0
                    ram_after = _ram_usage_mb()
                    logger.info(
                        f"  [{int(time.time()-t_start):>4d}s | RAM {ram_after}MB]"
                        f" ✓ {model} → GPU {gpu_id} in {dt:.1f}s"
                    )
                    loaded += 1
                except Exception as e:
                    logger.error(
                        f"  [{int(time.time()-t_start):>4d}s]"
                        f" ✗ {model} → GPU {gpu_id} FAILED: {e}"
                    )
                    import traceback
                    logger.error(traceback.format_exc())
                    failed += 1

                # Let kernel settle between model loads (reclaim page cache, etc.)
                _flush_ram()
                await asyncio.sleep(3)

        total_time = time.time() - t_start
        logger.info("=" * 70)
        logger.info(f"  WARMUP DONE — {loaded} loaded, {failed} failed in {total_time:.0f}s")
        logger.info(f"  VRAM: {_vram_summary()}")
        logger.info(f"  RAM: {_ram_usage_mb()} MB")
        logger.info("=" * 70)

        if loaded > 0:
            logger.info("  SERVER READY — accepting requests")

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
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
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
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
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


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=848, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=61, ge=9, le=129)
    seed: Optional[int] = Field(default=None)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    fps: int = Field(default=15, ge=8, le=30)


class VideoResponse(BaseModel):
    video: str
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


class VideoI2VResponse(BaseModel):
    video: str
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


class ThreeDResponse(BaseModel):
    model_glb: str
    time_seconds: float
    textured: bool


class TextTo3DRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    image_width: int = Field(default=1024, ge=512, le=2048)
    image_height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    guidance_scale: float = Field(default=5.0, ge=0, le=20)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    do_texture: bool = Field(default=True)


class TextTo3DResponse(BaseModel):
    model_glb: str
    reference_image: str
    seed: int
    time_seconds: float
    image_time_seconds: float
    mesh_time_seconds: float
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


@app.post("/api/hunyuan/video", response_model=VideoResponse)
async def hunyuan_video(req: VideoRequest):
    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    try:
        video_bytes, used_seed, frame_count = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_video_sync,
            req.prompt, req.width, req.height, req.num_frames,
            seed, req.num_inference_steps, req.fps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(video_bytes).decode("ascii")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            video_bytes, f"{ts}_{used_seed}_video.mp4", "video/mp4",
            {"type": "video", "prompt": req.prompt, "seed": used_seed,
             "width": req.width, "height": req.height, "num_frames": frame_count,
             "fps": req.fps, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return VideoResponse(
        video=b64, seed=used_seed, width=req.width, height=req.height,
        num_frames=frame_count, fps=req.fps, time_seconds=elapsed,
    )


@app.post("/api/hunyuan/video/i2v", response_model=VideoI2VResponse)
async def hunyuan_video_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    width: int = Form(default=848), height: int = Form(default=480),
    num_frames: int = Form(default=61), seed: Optional[int] = Form(default=None),
    num_inference_steps: int = Form(default=30), fps: int = Form(default=15),
):
    try:
        img_bytes = await image.read()
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    t0 = time.time()
    used_seed = seed if seed is not None else random.randint(1, 2**31)

    try:
        video_bytes, _, frame_count = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_video_i2v_sync,
            pil_image, prompt, width, height, num_frames,
            used_seed, num_inference_steps, fps,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)
    b64 = base64.b64encode(video_bytes).decode("ascii")

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            video_bytes, f"{ts}_{used_seed}_i2v.mp4", "video/mp4",
            {"type": "video_i2v", "prompt": prompt, "seed": used_seed,
             "width": width, "height": height, "num_frames": frame_count,
             "fps": fps, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return VideoI2VResponse(
        video=b64, seed=used_seed, width=width, height=height,
        num_frames=frame_count, fps=fps, time_seconds=elapsed,
    )


@app.post("/api/hunyuan/3d", response_model=ThreeDResponse)
async def hunyuan_3d(
    image: UploadFile = File(...),
    do_texture: bool = Form(True),
):
    if not _HAS_HY3DGEN:
        raise HTTPException(501, "3D requires hy3dgen. Install: pip install git+https://github.com/Tencent/Hunyuan3D-2.git")

    try:
        img_bytes = await image.read()
        pil_image = PILImage.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    t0 = time.time()
    try:
        glb_bytes = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_3d_sync, pil_image, do_texture,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    elapsed = round(time.time() - t0, 2)

    try:
        ts = int(time.time() * 1000)
        save_to_history(
            glb_bytes, f"{ts}_3d.glb", "model/gltf-binary",
            {"type": "3d", "textured": do_texture, "time_seconds": elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return ThreeDResponse(
        model_glb=base64.b64encode(glb_bytes).decode("ascii"),
        time_seconds=elapsed, textured=do_texture,
    )


@app.post("/api/hunyuan/text-to-3d", response_model=TextTo3DResponse)
async def hunyuan_text_to_3d(req: TextTo3DRequest):
    if not _HAS_HY3DGEN:
        raise HTTPException(501, "3D requires hy3dgen.")

    t0 = time.time()
    seed = req.seed if req.seed is not None else random.randint(1, 2**31)

    # Step 1: image
    try:
        img_bytes, used_seed = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_hunyuan_image_sync,
            req.prompt, req.negative_prompt, req.image_width, req.image_height,
            seed, req.guidance_scale, req.num_inference_steps,
        )
    except Exception as e:
        raise HTTPException(500, f"Image gen failed: {e}")

    img_elapsed = round(time.time() - t0, 2)
    ref_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Step 2: 3D
    t1 = time.time()
    try:
        glb_bytes = await asyncio.get_event_loop().run_in_executor(
            _executor, _generate_3d_sync, ref_image, req.do_texture,
        )
    except Exception as e:
        raise HTTPException(500, f"3D gen failed: {e}")

    mesh_elapsed = round(time.time() - t1, 2)

    try:
        ts = int(time.time() * 1000)
        (GENERATED_DIR / f"{ts}_{used_seed}_ref.png").write_bytes(img_bytes)
        save_to_history(
            glb_bytes, f"{ts}_{used_seed}_text3d.glb", "model/gltf-binary",
            {"type": "text-to-3d", "prompt": req.prompt, "seed": used_seed,
             "textured": req.do_texture, "time_seconds": img_elapsed + mesh_elapsed, "timestamp": ts},
        )
    except Exception:
        pass

    return TextTo3DResponse(
        model_glb=base64.b64encode(glb_bytes).decode("ascii"),
        reference_image=base64.b64encode(img_bytes).decode("ascii"),
        seed=used_seed, time_seconds=round(img_elapsed + mesh_elapsed, 2),
        image_time_seconds=img_elapsed, mesh_time_seconds=mesh_elapsed,
        textured=req.do_texture,
    )


# ══════════════════════════════════════════════════════════════════════
#  Health & Status
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda",
        "gpu_count": NUM_GPUS,
        "models_loaded": list(_pipelines.keys()),
        "available_modes": [m.value for m in ModelMode],
        "gpu_layout": {m: gpus for m, gpus in GPU_MAP.items()},
        "capabilities": {
            "hunyuan_3d": {"supported": _HAS_HY3DGEN},
        },
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
    """Live GPU hardware stats via nvidia-smi."""
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
            assigned = [m for m, gpu_ids in GPU_MAP.items() if idx in gpu_ids]
            loaded = [k.split(":")[0] for k in _pipelines if k.endswith(f":{idx}")]

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
                    "assigned_models": assigned,
                    "loaded_models": loaded,
                    "offloaded_models": [],  # FLUX now lives entirely on GPU
                    "active_task": _active_tasks.get(idx),
                    "generation_count": _gen_counts.get(idx, 0),
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
            for line in smi.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_ver = line.split("CUDA Version:")[1].strip().split()[0]
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
        "gpu_count": NUM_GPUS,
        "gpu_layout": GPU_MAP,
        "loaded_models": list(_pipelines.keys()),
        "gpu_locks": {str(i): _gpu_locks[i].locked() for i in range(NUM_GPUS)},
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

    port = int(os.getenv("PORT", "8100"))
    print(f"""
 ================================================================
   AI Generator  (CUDA — {NUM_GPUS} GPUs)
 ================================================================
   Server:  http://0.0.0.0:{port}
   Docs:    http://0.0.0.0:{port}/docs
 ----------------------------------------------------------------
   GPU Layout (startup):""")
    for model in ["lightning", "realvis_fast", "realvis_quality", "flux"]:
        gpus = GPU_MAP.get(model, [])
        print(f"     {model:20s} → GPU {gpus}")
    print(f""" ----------------------------------------------------------------
   On-demand:""")
    for model in ["hunyuan_image", "hunyuan_video", "hunyuan_3d"]:
        gpus = GPU_MAP.get(model, [])
        print(f"     {model:20s} → GPU {gpus}")
    print(f""" ----------------------------------------------------------------
   Models:
     lightning        SDXL Lightning 4-step      (~1s)
     realvis_fast     RealVisXL V5 Lightning     (~2s)
     realvis_quality  RealVisXL V5 25-step       (~8s)
     flux             FLUX.1 Schnell 4-step      (~20-30s)
 ----------------------------------------------------------------
   Hunyuan:
     Image:   HunyuanDiT v1.2               (~5-15s)
     Video:   HunyuanVideo INT4              (~5-12min)
     3D:      Hunyuan3D-2 {'✓' if _HAS_HY3DGEN else '✗ (not installed)'}
 ================================================================
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
