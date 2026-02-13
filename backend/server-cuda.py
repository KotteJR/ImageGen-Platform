"""
Unified AI Generator — FastAPI Backend (CUDA multi-GPU)

GPU Pool Architecture:
  - Auto-detects all NVIDIA GPUs at startup
  - Single shared pool: any GPU can run any model type
  - SDXL models: 1 GPU per image  → up to N images in parallel
  - FLUX:        2 GPUs per image  → up to N/2 images in parallel
  - Hunyuan:     1 GPU per task    → shares pool with SDXL slots
  - Async queue: overflow jobs wait for available GPU slots
  - Per-GPU locks: prevents resource conflicts between model types

Image Models (fast):
  1. "lightning"        — SDXL Lightning 4-step      (~0.5-1s per GPU)
  2. "realvis_fast"     — RealVisXL V5.0 Lightning   (~1-2s per GPU)
  3. "realvis_quality"  — RealVisXL V5.0 25-step     (~5-8s per GPU)
  4. "flux"             — FLUX.1 Schnell 4-step      (~3-5s per 2-GPU pair)

Hunyuan Models (heavy):
  5. HunyuanDiT v1.2 image         — text-to-image        (~5-15s per GPU)
  6. HunyuanVideo T2V (INT4)       — text-to-video         (~5-12min per GPU)
  7. HunyuanVideo I2V (INT4)       — image-to-video        (~5-12min per GPU)
  8. Hunyuan3D-2                    — image/text-to-3D      (~2-5min per GPU)

Endpoints:
  POST /api/generate              → SDXL/FLUX single image (base64 PNG)
  POST /api/generate/batch        → SDXL/FLUX parallel batch
  POST /api/generate/raw          → SDXL/FLUX raw PNG bytes
  POST /api/hunyuan/image         → HunyuanDiT text-to-image (base64 PNG)
  POST /api/hunyuan/video         → HunyuanVideo text-to-video (base64 MP4)
  POST /api/hunyuan/video/i2v     → HunyuanVideo image-to-video (base64 MP4)
  POST /api/hunyuan/3d            → Hunyuan3D-2 image-to-3D (base64 GLB)
  POST /api/hunyuan/text-to-3d    → text → image → 3D (base64 GLB + PNG)
  GET  /api/health                → health check + full pool info
  GET  /api/queue/status          → slot details

History:
  GET    /api/history                  → list outputs (newest first)
  GET    /api/history/file/{filename}  → serve a saved file (image, video, 3D)
  DELETE /api/history/{filename}       → delete a saved output

Usage:
  python server-cuda.py                                # auto-detect all GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3 python server-cuda.py   # limit to specific GPUs
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import tempfile
import threading
import time
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
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

# ── Check optional 3D support ────────────────────────────────────────
_HAS_HY3DGEN = False
try:
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    _HAS_HY3DGEN = True
    logger.info("hy3dgen (Hunyuan3D-2) available ✓")
except ImportError:
    logger.warning(
        "hy3dgen not installed — 3D generation disabled. "
        "Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
    )

# ── Constants ─────────────────────────────────────────────────────────

SDXL_DTYPE = torch.float16
FLUX_DTYPE = torch.bfloat16
HUNYUAN_IMAGE_DTYPE = torch.float16
HUNYUAN_VIDEO_DTYPE = torch.bfloat16


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
        logger.info("    xformers memory-efficient attention enabled")
    except Exception:
        logger.info("    Using PyTorch SDPA attention (xformers not available)")
    return pipe


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


# ── Output History (persistent disk storage) ──────────────────────────

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def save_to_history(
    data_bytes: bytes,
    filename: str,
    media_type: str,
    metadata: dict,
) -> str:
    """Save a generated output + metadata JSON to the generated/ folder."""
    (GENERATED_DIR / filename).write_bytes(data_bytes)
    meta = {**metadata, "filename": filename, "media_type": media_type}
    (GENERATED_DIR / f"{filename}.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved to history: {filename}")
    return filename


# ══════════════════════════════════════════════════════════════════════
#  GPU SLOT — One slot per GPU (or 2 GPUs for FLUX)
# ══════════════════════════════════════════════════════════════════════

class GPUSlot:
    """A GPU slot that can run any model type.

    - SDXL/RealVis: 1 GPU, standard .to(device) loading
    - FLUX: 1-2 GPUs, device_map balanced loading
    - Hunyuan Image: 1 GPU, standard .to(device) loading
    - Hunyuan Video (T2V/I2V): 1 GPU, INT4 quantized + CPU offloading
    - Hunyuan 3D: 1 GPU, torch.cuda.device context

    Each slot has its own pipeline cache and threading lock.
    """

    def __init__(self, slot_id: int, gpu_ids: list):
        self.slot_id = slot_id
        self.gpu_ids = gpu_ids
        self.gpu_id = gpu_ids[0]  # primary GPU
        self.device = f"cuda:{gpu_ids[0]}"
        self.multi_gpu = len(gpu_ids) > 1
        self.lock = threading.Lock()
        self._pipelines: dict = {}
        self.generation_count = 0
        self.active_task: Optional[str] = None

    def __repr__(self):
        return f"GPUSlot(id={self.slot_id}, gpus={self.gpu_ids})"

    # ══════════════════════════════════════════════════════════════
    #  SDXL / FLUX Pipeline Loaders
    # ══════════════════════════════════════════════════════════════

    def _load_lightning(self):
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
        LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
        LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"

        logger.info(f"  Slot {self.slot_id}: Loading SDXL Lightning on {self.device}...")
        unet_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)
        unet = UNet2DConditionModel.from_pretrained(
            SDXL_BASE, subfolder="unet", torch_dtype=SDXL_DTYPE,
        )
        unet.load_state_dict(load_file(unet_path), strict=False)

        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE, unet=unet, torch_dtype=SDXL_DTYPE, variant="fp16",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing",
        )
        pipe = pipe.to(self.device)
        _enable_fast_attention(pipe)
        _disable_safety(pipe)

        logger.info(f"  Slot {self.slot_id}: Lightning loaded on {self.device}")
        return pipe

    def _load_realvis_fast(self):
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

        MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"

        logger.info(f"  Slot {self.slot_id}: Loading RealVisXL V5 Lightning on {self.device}...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, torch_dtype=SDXL_DTYPE, variant="fp16",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing",
        )
        pipe = pipe.to(self.device)
        _enable_fast_attention(pipe)
        _disable_safety(pipe)

        logger.info(f"  Slot {self.slot_id}: RealVisXL V5 Lightning loaded on {self.device}")
        return pipe

    def _load_realvis_quality(self):
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

        MODEL_ID = "SG161222/RealVisXL_V5.0"

        logger.info(f"  Slot {self.slot_id}: Loading RealVisXL V5 Quality on {self.device}...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, torch_dtype=SDXL_DTYPE, variant="fp16",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        _enable_fast_attention(pipe)
        _disable_safety(pipe)

        logger.info(f"  Slot {self.slot_id}: RealVisXL V5 Quality loaded on {self.device}")
        return pipe

    def _load_flux(self):
        from diffusers import FluxPipeline

        MODEL_ID = "black-forest-labs/FLUX.1-schnell"

        if self.multi_gpu:
            logger.info(f"  Slot {self.slot_id}: Loading FLUX across GPUs {self.gpu_ids}...")
            max_memory = {}
            for gid in self.gpu_ids:
                total_gb = torch.cuda.get_device_properties(gid).total_mem / 1024**3
                max_memory[gid] = f"{int(total_gb * 0.85)}GiB"
            pipe = FluxPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=FLUX_DTYPE,
                device_map="balanced",
                max_memory=max_memory,
            )
        else:
            logger.info(f"  Slot {self.slot_id}: Loading FLUX on {self.device}...")
            pipe = FluxPipeline.from_pretrained(
                MODEL_ID, torch_dtype=FLUX_DTYPE,
            )
            pipe = pipe.to(self.device)

        _enable_fast_attention(pipe)

        logger.info(f"  Slot {self.slot_id}: FLUX loaded on GPU(s) {self.gpu_ids}")
        return pipe

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Pipeline Loaders
    # ══════════════════════════════════════════════════════════════

    def _load_hunyuan_image(self):
        """Load HunyuanDiT for text-to-image."""
        from diffusers import HunyuanDiTPipeline

        MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

        logger.info(f"  Slot {self.slot_id}: Loading HunyuanDiT on {self.device}...")
        pipe = HunyuanDiTPipeline.from_pretrained(
            MODEL_ID, torch_dtype=HUNYUAN_IMAGE_DTYPE,
        )
        pipe = pipe.to(self.device)
        _enable_fast_attention(pipe)

        logger.info(f"  Slot {self.slot_id}: HunyuanDiT loaded on {self.device} ✓")
        return pipe

    def _load_hunyuan_video(self):
        """Load HunyuanVideo T2V with INT4 quantization + CPU offloading.

        Fits on a single 24GB RTX 3090:
          - Transformer: INT4 quantized (~6-8GB)
          - VAE with tiling: ~2-4GB
          - CPU offloading moves unused components off GPU
        """
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from transformers import BitsAndBytesConfig

        MODEL_ID = "hunyuanvideo-community/HunyuanVideo"

        logger.info(
            f"  Slot {self.slot_id}: Loading HunyuanVideo T2V (INT4) "
            f"with CPU offload on GPU {self.gpu_id}..."
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=HUNYUAN_VIDEO_DTYPE,
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=HUNYUAN_VIDEO_DTYPE,
        )

        pipe = HunyuanVideoPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=HUNYUAN_VIDEO_DTYPE,
        )
        pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)
        pipe.vae.enable_tiling()

        logger.info(
            f"  Slot {self.slot_id}: HunyuanVideo T2V loaded "
            f"(INT4 + CPU offload on GPU {self.gpu_id}) ✓"
        )
        return pipe

    def _load_hunyuan_video_i2v(self):
        """Load HunyuanVideo I2V with INT4 quantization + CPU offloading.

        Same memory profile as T2V, fits on a single 24GB RTX 3090.
        """
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
        from transformers import BitsAndBytesConfig

        MODEL_ID = "hunyuanvideo-community/HunyuanVideo-I2V"

        logger.info(
            f"  Slot {self.slot_id}: Loading HunyuanVideo I2V (INT4) "
            f"with CPU offload on GPU {self.gpu_id}..."
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=HUNYUAN_VIDEO_DTYPE,
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=HUNYUAN_VIDEO_DTYPE,
        )

        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)
        pipe.vae.enable_tiling()

        logger.info(
            f"  Slot {self.slot_id}: HunyuanVideo I2V loaded "
            f"(INT4 + CPU offload on GPU {self.gpu_id}) ✓"
        )
        return pipe

    def _load_hunyuan_3d(self):
        """Load Hunyuan3D-2 shape + texture pipelines."""
        if not _HAS_HY3DGEN:
            raise RuntimeError(
                "hy3dgen not installed. Install with: "
                "pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
            )

        MODEL_PATH = "tencent/Hunyuan3D-2"

        logger.info(f"  Slot {self.slot_id}: Loading Hunyuan3D-2 on GPU {self.gpu_id}...")

        with torch.cuda.device(self.gpu_id):
            shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH)
            texgen = Hunyuan3DPaintPipeline.from_pretrained(MODEL_PATH)
            rembg = BackgroundRemover()

        logger.info(f"  Slot {self.slot_id}: Hunyuan3D-2 loaded on GPU {self.gpu_id} ✓")
        return {"shapegen": shapegen, "texgen": texgen, "rembg": rembg}

    # ══════════════════════════════════════════════════════════════
    #  Pipeline Access (unified cache)
    # ══════════════════════════════════════════════════════════════

    # Maps pipeline keys to loader functions
    _LOADERS: dict = {}  # populated in __init_subclass__ or accessed via property

    def _get_loader(self, key: str):
        """Get the loader function for a given pipeline key."""
        loaders = {
            # SDXL / FLUX
            "lightning": self._load_lightning,
            "realvis_fast": self._load_realvis_fast,
            "realvis_quality": self._load_realvis_quality,
            "flux": self._load_flux,
            # Hunyuan
            "hunyuan_image": self._load_hunyuan_image,
            "hunyuan_video": self._load_hunyuan_video,
            "hunyuan_video_i2v": self._load_hunyuan_video_i2v,
            "hunyuan_3d": self._load_hunyuan_3d,
        }
        return loaders.get(key)

    def get_pipeline(self, key: str):
        """Get or load a pipeline. Caches per slot. Handles OOM by clearing."""
        if key in self._pipelines:
            return self._pipelines[key]

        loader = self._get_loader(key)
        if not loader:
            raise ValueError(f"Unknown pipeline key: {key}")

        try:
            pipe = loader()
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  Slot {self.slot_id}: OOM loading {key}, clearing cache...")
            self._clear_cache()
            pipe = loader()

        self._pipelines[key] = pipe
        return pipe

    def _clear_cache(self):
        """Clear all cached pipelines and free GPU memory."""
        for key in list(self._pipelines.keys()):
            del self._pipelines[key]
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    #  SDXL / FLUX Generation
    # ══════════════════════════════════════════════════════════════

    def generate(
        self,
        model_mode: str,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> tuple:
        """Generate a single SDXL/FLUX image. Thread-safe via self.lock.

        Returns: (png_bytes, seed, elapsed_seconds)
        """
        t0 = time.time()

        with self.lock:
            self.active_task = model_mode
            self.generation_count += 1

            pipeline = self.get_pipeline(model_mode)
            gen = torch.Generator(self.device).manual_seed(seed)

            with torch.inference_mode():
                if model_mode == "flux":
                    result = pipeline(
                        prompt=prompt,
                        guidance_scale=0.0,
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height,
                        max_sequence_length=256,
                        generator=gen,
                    )
                else:
                    p = trim_prompt(prompt, max_tokens=70)
                    n = trim_prompt(negative, max_tokens=70) if negative else ""
                    result = pipeline(
                        prompt=p,
                        negative_prompt=n if n else None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=gen,
                    )

            self.active_task = None

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        elapsed = round(time.time() - t0, 2)

        return buf.getvalue(), seed, elapsed

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Generation
    # ══════════════════════════════════════════════════════════════

    def generate_hunyuan_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> tuple:
        """Generate an image with HunyuanDiT.

        Returns: (png_bytes, seed, elapsed_seconds)
        """
        t0 = time.time()

        with self.lock:
            self.active_task = "hunyuan_image"
            self.generation_count += 1

            pipe = self.get_pipeline("hunyuan_image")
            gen = torch.Generator(self.device).manual_seed(seed)

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )

            self.active_task = None

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        elapsed = round(time.time() - t0, 2)

        return buf.getvalue(), seed, elapsed

    def generate_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        num_inference_steps: int,
        fps: int,
    ) -> tuple:
        """Generate a video with HunyuanVideo T2V.

        Returns: (mp4_bytes, seed, elapsed_seconds, num_frames)
        """
        from diffusers.utils import export_to_video

        t0 = time.time()

        with self.lock:
            self.active_task = "hunyuan_video"
            self.generation_count += 1

            pipe = self.get_pipeline("hunyuan_video")
            gen = torch.Generator("cpu").manual_seed(seed)

            logger.info(
                f"  Slot {self.slot_id}: Generating video "
                f"{width}x{height}, {num_frames} frames, {num_inference_steps} steps..."
            )

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=gen,
                )

            self.active_task = None

        frames = result.frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
        try:
            export_to_video(frames, temp_path, fps=fps)
            with open(temp_path, "rb") as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        elapsed = round(time.time() - t0, 2)
        return video_bytes, seed, elapsed, len(frames)

    def generate_video_i2v(
        self,
        image: PILImage.Image,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        num_inference_steps: int,
        fps: int,
    ) -> tuple:
        """Generate a video from an image with HunyuanVideo I2V.

        Returns: (mp4_bytes, seed, elapsed_seconds, num_frames)
        """
        from diffusers.utils import export_to_video

        t0 = time.time()

        with self.lock:
            self.active_task = "hunyuan_video_i2v"
            self.generation_count += 1

            pipe = self.get_pipeline("hunyuan_video_i2v")
            gen = torch.Generator("cpu").manual_seed(seed)

            logger.info(
                f"  Slot {self.slot_id}: Generating I2V video "
                f"{width}x{height}, {num_frames} frames, {num_inference_steps} steps..."
            )

            with torch.inference_mode():
                result = pipe(
                    image=image,
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=gen,
                )

            self.active_task = None

        frames = result.frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
        try:
            export_to_video(frames, temp_path, fps=fps)
            with open(temp_path, "rb") as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        elapsed = round(time.time() - t0, 2)
        return video_bytes, seed, elapsed, len(frames)

    def generate_3d(
        self,
        image: PILImage.Image,
        do_texture: bool = True,
    ) -> tuple:
        """Generate a 3D model from an image with Hunyuan3D-2.

        Returns: (glb_bytes, elapsed_seconds)
        """
        t0 = time.time()

        with self.lock:
            self.active_task = "hunyuan_3d"
            self.generation_count += 1

            pipelines = self.get_pipeline("hunyuan_3d")
            shapegen = pipelines["shapegen"]
            texgen = pipelines["texgen"]
            rembg = pipelines["rembg"]

            # Remove background if needed
            if image.mode == "RGB":
                logger.info(f"  Slot {self.slot_id}: Removing background...")
                image = rembg(image.convert("RGBA"))
            elif image.mode != "RGBA":
                image = image.convert("RGBA")

            # Step 1: Generate 3D shape
            logger.info(f"  Slot {self.slot_id}: Generating 3D shape...")
            with torch.inference_mode():
                mesh = shapegen(image=image)[0]

            # Step 2: Generate texture (optional)
            if do_texture:
                logger.info(f"  Slot {self.slot_id}: Painting texture...")
                with torch.inference_mode():
                    mesh = texgen(mesh, image=image)

            self.active_task = None

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            temp_path = tmp.name
        try:
            mesh.export(temp_path)
            with open(temp_path, "rb") as f:
                glb_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        elapsed = round(time.time() - t0, 2)
        return glb_bytes, elapsed


# ══════════════════════════════════════════════════════════════════════
#  GPU POOL — Manages all GPU slots with async queuing
# ══════════════════════════════════════════════════════════════════════

class GPUPool:
    """Manages GPU slots for parallel generation with automatic queuing.

    Architecture:
      - sdxl_queue: 1 GPU per slot → SDXL, RealVis, and ALL Hunyuan models
      - flux_queue: 2 GPUs per slot → FLUX only
      - Per-GPU asyncio locks prevent conflicts when SDXL/FLUX share hardware
      - Async slot queues provide fair FIFO scheduling
    """

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus

        # SDXL + Hunyuan: 1 GPU per slot
        self.sdxl_slots = [GPUSlot(i, [i]) for i in range(num_gpus)]

        # FLUX: 2 GPUs per slot (single GPU falls back to 1-GPU slot)
        if num_gpus <= 1:
            self.flux_slots = [GPUSlot(0, [0])] if num_gpus == 1 else []
        else:
            self.flux_slots = [
                GPUSlot(i, [i * 2, i * 2 + 1])
                for i in range(num_gpus // 2)
            ]

        self._executor = ThreadPoolExecutor(max_workers=max(num_gpus * 2, 1))

        # Async resources (initialized in init_async)
        self._sdxl_queue: Optional[asyncio.Queue] = None
        self._flux_queue: Optional[asyncio.Queue] = None
        self._gpu_locks: dict = {}

        # Stats
        self._total_generated = 0
        self._active_jobs = 0
        self._stats_lock = threading.Lock()

        logger.info(
            f"GPU Pool: {len(self.sdxl_slots)} single-GPU slots (SDXL + Hunyuan), "
            f"{len(self.flux_slots)} dual-GPU slots (FLUX)"
        )

    def init_async(self):
        """Initialize async queues and per-GPU locks. Call from async context."""
        self._gpu_locks = {i: asyncio.Lock() for i in range(self.num_gpus)}

        self._sdxl_queue = asyncio.Queue()
        for slot in self.sdxl_slots:
            self._sdxl_queue.put_nowait(slot)

        self._flux_queue = asyncio.Queue()
        for slot in self.flux_slots:
            self._flux_queue.put_nowait(slot)

        logger.info("GPU Pool async resources initialized")

    def _get_queue(self, model_mode: str) -> asyncio.Queue:
        return self._flux_queue if model_mode == "flux" else self._sdxl_queue

    async def _acquire_gpus(self, gpu_ids: list):
        """Acquire per-GPU locks in sorted order (prevents deadlocks)."""
        for gid in sorted(gpu_ids):
            await self._gpu_locks[gid].acquire()

    def _release_gpus(self, gpu_ids: list):
        """Release per-GPU locks."""
        for gid in sorted(gpu_ids):
            self._gpu_locks[gid].release()

    async def _smart_acquire_slot(self, model_mode: str, preferred_key: Optional[str] = None):
        """Pick a slot that already has the model loaded if possible.
        Falls back to any available slot. Blocks if all slots are busy."""
        queue = self._get_queue(model_mode)
        cache_key = preferred_key or model_mode

        # Drain all available slots, pick the best one, put the rest back
        available = []
        try:
            while True:
                slot = queue.get_nowait()
                available.append(slot)
        except asyncio.QueueEmpty:
            pass

        if not available:
            slot = await queue.get()
            return slot

        # Prefer a slot that already has this model loaded
        best = None
        for s in available:
            if cache_key in s._pipelines:
                best = s
                break

        if best is None:
            best = available[0]

        for s in available:
            if s is not best:
                queue.put_nowait(s)

        return best

    # ══════════════════════════════════════════════════════════════
    #  SDXL / FLUX Image Generation
    # ══════════════════════════════════════════════════════════════

    async def generate_one(
        self,
        model_mode: str,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        seed: Optional[int],
        guidance_scale: float,
        num_inference_steps: int,
    ) -> dict:
        """Generate one SDXL/FLUX image, waiting for an available GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)

        slot = await self._smart_acquire_slot(model_mode)
        return_queue = self._get_queue(model_mode)

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Job → Slot {slot.slot_id} (GPU {slot.gpu_ids}): "
                f"{model_mode}, {width}x{height}, seed={used_seed}"
            )

            try:
                loop = asyncio.get_event_loop()
                img_bytes, final_seed, elapsed = await loop.run_in_executor(
                    self._executor,
                    slot.generate,
                    model_mode, prompt, negative,
                    width, height, used_seed,
                    guidance_scale, num_inference_steps,
                )

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(
                    f"Done ← Slot {slot.slot_id}: {width}x{height} in {elapsed}s "
                    f"(mode={model_mode}, seed={final_seed})"
                )

                return {
                    "success": True,
                    "image": base64.b64encode(img_bytes).decode("ascii"),
                    "seed": final_seed,
                    "width": width,
                    "height": height,
                    "time_seconds": elapsed,
                    "model_mode": model_mode,
                    "slot_id": slot.slot_id,
                    "gpu_ids": slot.gpu_ids,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"Failed on Slot {slot.slot_id}: {e}")
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    async def generate_batch(
        self,
        model_mode: str,
        requests: list,
    ) -> list:
        """Generate multiple SDXL/FLUX images in parallel across GPU slots."""
        tasks = [
            self.generate_one(
                model_mode=model_mode,
                prompt=req["prompt"],
                negative=req.get("negative", ""),
                width=req.get("width", 1024),
                height=req.get("height", 1024),
                seed=req.get("seed"),
                guidance_scale=req.get("guidance_scale", 0),
                num_inference_steps=req.get("num_inference_steps", 4),
            )
            for req in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final.append({"success": False, "error": str(r), "index": i})
            else:
                r["index"] = i
                final.append(r)

        return final

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Image
    # ══════════════════════════════════════════════════════════════

    async def run_hunyuan_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: Optional[int],
        guidance_scale: float,
        num_inference_steps: int,
    ) -> dict:
        """Generate a Hunyuan image, waiting for an available single-GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)
        slot = await self._smart_acquire_slot("hunyuan_image", preferred_key="hunyuan_image")
        return_queue = self._sdxl_queue

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Hunyuan Image → Slot {slot.slot_id}: "
                f"{width}x{height}, seed={used_seed}"
            )

            try:
                loop = asyncio.get_event_loop()
                img_bytes, final_seed, elapsed = await loop.run_in_executor(
                    self._executor,
                    slot.generate_hunyuan_image,
                    prompt, negative_prompt,
                    width, height, used_seed,
                    guidance_scale, num_inference_steps,
                )

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(f"Hunyuan Image done ← Slot {slot.slot_id}: {elapsed}s")

                return {
                    "success": True,
                    "image": base64.b64encode(img_bytes).decode("ascii"),
                    "seed": final_seed,
                    "width": width,
                    "height": height,
                    "time_seconds": elapsed,
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"Hunyuan Image failed on Slot {slot.slot_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Video (T2V)
    # ══════════════════════════════════════════════════════════════

    async def run_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: Optional[int],
        num_inference_steps: int,
        fps: int,
    ) -> dict:
        """Generate a video, waiting for an available single-GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)
        slot = await self._smart_acquire_slot("hunyuan_video", preferred_key="hunyuan_video")
        return_queue = self._sdxl_queue

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Video T2V → Slot {slot.slot_id}: "
                f"{width}x{height}, {num_frames} frames, seed={used_seed}"
            )

            try:
                loop = asyncio.get_event_loop()
                video_bytes, final_seed, elapsed, frame_count = await loop.run_in_executor(
                    self._executor,
                    slot.generate_video,
                    prompt, width, height,
                    num_frames, used_seed,
                    num_inference_steps, fps,
                )

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(
                    f"Video T2V done ← Slot {slot.slot_id}: "
                    f"{frame_count} frames in {elapsed}s"
                )

                return {
                    "success": True,
                    "video": base64.b64encode(video_bytes).decode("ascii"),
                    "seed": final_seed,
                    "width": width,
                    "height": height,
                    "num_frames": frame_count,
                    "fps": fps,
                    "time_seconds": elapsed,
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"Video T2V failed on Slot {slot.slot_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Video (I2V)
    # ══════════════════════════════════════════════════════════════

    async def run_video_i2v(
        self,
        image: PILImage.Image,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: Optional[int],
        num_inference_steps: int,
        fps: int,
    ) -> dict:
        """Generate a video from an image, waiting for an available GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)
        slot = await self._smart_acquire_slot("hunyuan_video_i2v", preferred_key="hunyuan_video_i2v")
        return_queue = self._sdxl_queue

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Video I2V → Slot {slot.slot_id}: "
                f"{width}x{height}, {num_frames} frames, seed={used_seed}"
            )

            try:
                loop = asyncio.get_event_loop()
                video_bytes, final_seed, elapsed, frame_count = await loop.run_in_executor(
                    self._executor,
                    slot.generate_video_i2v,
                    image, prompt, width, height,
                    num_frames, used_seed,
                    num_inference_steps, fps,
                )

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(
                    f"Video I2V done ← Slot {slot.slot_id}: "
                    f"{frame_count} frames in {elapsed}s"
                )

                return {
                    "success": True,
                    "video": base64.b64encode(video_bytes).decode("ascii"),
                    "seed": final_seed,
                    "width": width,
                    "height": height,
                    "num_frames": frame_count,
                    "fps": fps,
                    "time_seconds": elapsed,
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"Video I2V failed on Slot {slot.slot_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan 3D
    # ══════════════════════════════════════════════════════════════

    async def run_3d(
        self,
        image: PILImage.Image,
        do_texture: bool = True,
    ) -> dict:
        """Generate a 3D model from an image, waiting for an available GPU slot."""
        slot = await self._smart_acquire_slot("hunyuan_3d", preferred_key="hunyuan_3d")
        return_queue = self._sdxl_queue

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"3D → Slot {slot.slot_id}: "
                f"image {image.size}, texture={do_texture}"
            )

            try:
                loop = asyncio.get_event_loop()
                glb_bytes, elapsed = await loop.run_in_executor(
                    self._executor,
                    slot.generate_3d,
                    image, do_texture,
                )

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(f"3D done ← Slot {slot.slot_id}: {elapsed}s")

                return {
                    "success": True,
                    "model_glb": base64.b64encode(glb_bytes).decode("ascii"),
                    "time_seconds": elapsed,
                    "textured": do_texture,
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"3D failed on Slot {slot.slot_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    # ══════════════════════════════════════════════════════════════
    #  Hunyuan Text-to-3D (chained: image → 3D)
    # ══════════════════════════════════════════════════════════════

    async def run_text_to_3d(
        self,
        prompt: str,
        negative_prompt: str,
        image_width: int,
        image_height: int,
        image_seed: Optional[int],
        image_guidance_scale: float,
        image_steps: int,
        do_texture: bool = True,
    ) -> dict:
        """Text → Image (HunyuanDiT) → 3D (Hunyuan3D-2). Uses same slot for both."""
        used_seed = image_seed if image_seed is not None else random.randint(1, 2**31)
        slot = await self._smart_acquire_slot("hunyuan_image", preferred_key="hunyuan_image")
        return_queue = self._sdxl_queue

        try:
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Text-to-3D → Slot {slot.slot_id}: "
                f"\"{prompt[:60]}...\", seed={used_seed}"
            )

            try:
                loop = asyncio.get_event_loop()

                # Step 1: Generate image
                logger.info(f"  Step 1/2: Generating reference image...")
                img_bytes, final_seed, img_elapsed = await loop.run_in_executor(
                    self._executor,
                    slot.generate_hunyuan_image,
                    prompt, negative_prompt,
                    image_width, image_height, used_seed,
                    image_guidance_scale, image_steps,
                )

                ref_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")

                # Step 2: Generate 3D from image
                logger.info(f"  Step 2/2: Generating 3D model from image...")
                glb_bytes, mesh_elapsed = await loop.run_in_executor(
                    self._executor,
                    slot.generate_3d,
                    ref_image, do_texture,
                )

                total_elapsed = round(img_elapsed + mesh_elapsed, 2)

                with self._stats_lock:
                    self._active_jobs -= 1
                    self._total_generated += 1

                logger.info(
                    f"Text-to-3D done ← Slot {slot.slot_id}: "
                    f"{total_elapsed}s (image: {img_elapsed}s, 3D: {mesh_elapsed}s)"
                )

                return {
                    "success": True,
                    "model_glb": base64.b64encode(glb_bytes).decode("ascii"),
                    "reference_image": base64.b64encode(img_bytes).decode("ascii"),
                    "seed": final_seed,
                    "time_seconds": total_elapsed,
                    "image_time_seconds": img_elapsed,
                    "mesh_time_seconds": mesh_elapsed,
                    "textured": do_texture,
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                }

            except Exception as e:
                with self._stats_lock:
                    self._active_jobs -= 1
                logger.error(f"Text-to-3D failed on Slot {slot.slot_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

            finally:
                self._release_gpus(slot.gpu_ids)

        finally:
            return_queue.put_nowait(slot)

    @property
    def info(self) -> dict:
        return {
            "num_gpus": self.num_gpus,
            "sdxl_slots": len(self.sdxl_slots),
            "flux_slots": len(self.flux_slots),
            "sdxl_parallel_capacity": len(self.sdxl_slots),
            "flux_parallel_capacity": len(self.flux_slots),
            "active_jobs": self._active_jobs,
            "total_generated": self._total_generated,
            "hunyuan_image_supported": True,
            "hunyuan_video_supported": True,
            "hunyuan_3d_supported": _HAS_HY3DGEN,
        }


# ══════════════════════════════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════════════════════════════

pool = GPUPool(_num_gpus)

app = FastAPI(title="AI Generator (CUDA — Unified GPU Pool)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    pool.init_async()


# ══════════════════════════════════════════════════════════════════════
#  Request / Response Models — SDXL / FLUX
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
    seed: Optional[int] = Field(default=None, description="Random seed. None = random.")
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
    slot_id: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: List[BatchResultItem]
    total_time_seconds: float
    successful: int
    failed: int
    parallel_capacity: int
    model_mode: str


# ══════════════════════════════════════════════════════════════════════
#  Request / Response Models — Hunyuan
# ══════════════════════════════════════════════════════════════════════

# -- Hunyuan Image --

class HunyuanImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None)
    guidance_scale: float = Field(default=5.0, ge=0, le=20)
    num_inference_steps: int = Field(default=25, ge=1, le=100)


class HunyuanImageResponse(BaseModel):
    image: str  # base64-encoded PNG
    seed: int
    width: int
    height: int
    time_seconds: float


# -- Hunyuan Video T2V --

class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=848, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=61, ge=9, le=129)
    seed: Optional[int] = Field(default=None)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    fps: int = Field(default=15, ge=8, le=30)


class VideoResponse(BaseModel):
    video: str  # base64-encoded MP4
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


# -- Hunyuan Video I2V --

class VideoI2VResponse(BaseModel):
    video: str  # base64-encoded MP4
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


# -- Hunyuan 3D --

class ThreeDResponse(BaseModel):
    model_glb: str  # base64-encoded GLB
    time_seconds: float
    textured: bool


# -- Hunyuan Text-to-3D --

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
    model_glb: str  # base64-encoded GLB
    reference_image: str  # base64-encoded PNG
    seed: int
    time_seconds: float
    image_time_seconds: float
    mesh_time_seconds: float
    textured: bool


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — SDXL / FLUX
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate a single image using the GPU pool (SDXL/FLUX models)."""
    result = await pool.generate_one(
        model_mode=req.model_mode.value,
        prompt=req.prompt,
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Generation failed"))

    # Save to history
    ts = int(time.time() * 1000)
    filename = f"{ts}_{result['seed']}_{result['model_mode']}.png"
    try:
        save_to_history(
            data_bytes=base64.b64decode(result["image"]),
            filename=filename,
            media_type="image/png",
            metadata={
                "type": "image",
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "seed": result["seed"],
                "width": result["width"],
                "height": result["height"],
                "model_mode": result["model_mode"],
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")

    return GenerateResponse(
        image=result["image"],
        seed=result["seed"],
        width=result["width"],
        height=result["height"],
        time_seconds=result["time_seconds"],
        model_mode=result["model_mode"],
    )


@app.post("/api/generate/raw")
async def generate_raw(req: GenerateRequest):
    """Generate an image and return raw PNG bytes."""
    result = await pool.generate_one(
        model_mode=req.model_mode.value,
        prompt=req.prompt,
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Generation failed"))

    img_bytes = base64.b64decode(result["image"])
    return Response(content=img_bytes, media_type="image/png")


@app.post("/api/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(req: BatchGenerateRequest):
    """Generate multiple images in parallel across GPU slots."""
    t0 = time.time()
    model_mode = req.model_mode.value
    capacity = len(pool.flux_slots) if model_mode == "flux" else len(pool.sdxl_slots)

    requests = [
        {
            "prompt": item.prompt,
            "negative": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "seed": item.seed,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
        }
        for item in req.prompts
    ]

    logger.info(
        f"Batch: {len(requests)} images, mode={model_mode}, "
        f"parallel_capacity={capacity}"
    )

    results = await pool.generate_batch(model_mode, requests)

    # Attach filenames + save to history
    for i, result in enumerate(results):
        custom_fn = None
        if i < len(req.prompts) and req.prompts[i].filename:
            result["filename"] = req.prompts[i].filename
            custom_fn = req.prompts[i].filename

        if result.get("success") and result.get("image"):
            try:
                ts = int(time.time() * 1000)
                fn = custom_fn or f"{ts}_{result.get('seed', 0)}_{model_mode}.png"
                save_to_history(
                    data_bytes=base64.b64decode(result["image"]),
                    filename=fn,
                    media_type="image/png",
                    metadata={
                        "type": "image",
                        "prompt": req.prompts[i].prompt if i < len(req.prompts) else "",
                        "negative_prompt": req.negative_prompt,
                        "seed": result.get("seed", 0),
                        "width": result.get("width", req.width),
                        "height": result.get("height", req.height),
                        "model_mode": model_mode,
                        "guidance_scale": req.guidance_scale,
                        "num_inference_steps": req.num_inference_steps,
                        "time_seconds": result.get("time_seconds", 0),
                        "timestamp": ts,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to save batch image {i} to history: {e}")

    total_time = round(time.time() - t0, 2)
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    logger.info(
        f"Batch done: {successful}/{len(results)} ok in {total_time}s "
        f"({capacity} parallel slots)"
    )

    return BatchGenerateResponse(
        results=[BatchResultItem(**r) for r in results],
        total_time_seconds=total_time,
        successful=successful,
        failed=failed,
        parallel_capacity=capacity,
        model_mode=model_mode,
    )


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — Hunyuan Image
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/hunyuan/image", response_model=HunyuanImageResponse)
async def hunyuan_generate_image(req: HunyuanImageRequest):
    """Generate an image using HunyuanDiT v1.2.

    Bilingual (Chinese + English) text-to-image diffusion transformer.
    ~5-15s per image depending on resolution and steps.
    """
    result = await pool.run_hunyuan_image(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Hunyuan image generation failed"))

    ts = int(time.time() * 1000)
    filename = f"{ts}_{result['seed']}_hunyuan_image.png"
    try:
        save_to_history(
            data_bytes=base64.b64decode(result["image"]),
            filename=filename,
            media_type="image/png",
            metadata={
                "type": "hunyuan_image",
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "seed": result["seed"],
                "width": result["width"],
                "height": result["height"],
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save Hunyuan image to history: {e}")

    return HunyuanImageResponse(
        image=result["image"],
        seed=result["seed"],
        width=result["width"],
        height=result["height"],
        time_seconds=result["time_seconds"],
    )


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — Hunyuan Video
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/hunyuan/video", response_model=VideoResponse)
async def hunyuan_generate_video(req: VideoRequest):
    """Generate a video using HunyuanVideo T2V.

    13B parameter diffusion transformer, INT4 quantized.
    ~5-12 minutes per clip depending on resolution and frame count.
    """
    result = await pool.run_video(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
        seed=req.seed,
        num_inference_steps=req.num_inference_steps,
        fps=req.fps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Video generation failed"))

    ts = int(time.time() * 1000)
    filename = f"{ts}_{result['seed']}_hunyuan_video.mp4"
    try:
        save_to_history(
            data_bytes=base64.b64decode(result["video"]),
            filename=filename,
            media_type="video/mp4",
            metadata={
                "type": "video",
                "prompt": req.prompt,
                "seed": result["seed"],
                "width": result["width"],
                "height": result["height"],
                "num_frames": result["num_frames"],
                "fps": result["fps"],
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save video to history: {e}")

    return VideoResponse(
        video=result["video"],
        seed=result["seed"],
        width=result["width"],
        height=result["height"],
        num_frames=result["num_frames"],
        fps=result["fps"],
        time_seconds=result["time_seconds"],
    )


@app.post("/api/hunyuan/video/i2v", response_model=VideoI2VResponse)
async def hunyuan_generate_video_i2v(
    image: UploadFile = File(..., description="Input image (PNG/JPG/WEBP)"),
    prompt: str = Form(default="", description="Text prompt to guide video generation"),
    width: int = Form(default=848, ge=256, le=1280),
    height: int = Form(default=480, ge=256, le=720),
    num_frames: int = Form(default=61, ge=9, le=129),
    seed: Optional[int] = Form(default=None),
    num_inference_steps: int = Form(default=30, ge=1, le=100),
    fps: int = Form(default=15, ge=8, le=30),
):
    """Generate a video from an image using HunyuanVideo I2V.

    Takes an input image as the first frame and generates a video from it.
    ~5-12 minutes per clip.
    """
    try:
        img_bytes = await image.read()
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    result = await pool.run_video_i2v(
        image=pil_image,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        seed=seed,
        num_inference_steps=num_inference_steps,
        fps=fps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Video I2V generation failed"))

    ts = int(time.time() * 1000)
    vid_filename = f"{ts}_{result['seed']}_hunyuan_video_i2v.mp4"
    img_filename = f"{ts}_{result['seed']}_hunyuan_video_i2v_input.png"
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        (GENERATED_DIR / img_filename).write_bytes(buf.getvalue())

        save_to_history(
            data_bytes=base64.b64decode(result["video"]),
            filename=vid_filename,
            media_type="video/mp4",
            metadata={
                "type": "video_i2v",
                "prompt": prompt,
                "input_image": img_filename,
                "seed": result["seed"],
                "width": result["width"],
                "height": result["height"],
                "num_frames": result["num_frames"],
                "fps": result["fps"],
                "num_inference_steps": num_inference_steps,
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save I2V video to history: {e}")

    return VideoI2VResponse(
        video=result["video"],
        seed=result["seed"],
        width=result["width"],
        height=result["height"],
        num_frames=result["num_frames"],
        fps=result["fps"],
        time_seconds=result["time_seconds"],
    )


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — Hunyuan 3D
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/hunyuan/3d", response_model=ThreeDResponse)
async def hunyuan_generate_3d(
    image: UploadFile = File(..., description="Input image (PNG/JPG/WEBP)"),
    do_texture: bool = Form(True, description="Apply texture painting"),
):
    """Generate a 3D model from an image using Hunyuan3D-2.

    Two-stage pipeline: shape generation + texture painting.
    ~2-5 minutes per model.
    """
    if not _HAS_HY3DGEN:
        raise HTTPException(
            501,
            "3D generation requires hy3dgen. "
            "Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
        )

    try:
        img_bytes = await image.read()
        pil_image = PILImage.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    result = await pool.run_3d(image=pil_image, do_texture=do_texture)

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "3D generation failed"))

    ts = int(time.time() * 1000)
    glb_filename = f"{ts}_hunyuan_3d.glb"
    img_filename = f"{ts}_hunyuan_3d_input.png"
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        (GENERATED_DIR / img_filename).write_bytes(buf.getvalue())

        save_to_history(
            data_bytes=base64.b64decode(result["model_glb"]),
            filename=glb_filename,
            media_type="model/gltf-binary",
            metadata={
                "type": "3d",
                "input_image": img_filename,
                "textured": result["textured"],
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save 3D model to history: {e}")

    return ThreeDResponse(
        model_glb=result["model_glb"],
        time_seconds=result["time_seconds"],
        textured=result["textured"],
    )


@app.post("/api/hunyuan/text-to-3d", response_model=TextTo3DResponse)
async def hunyuan_generate_text_to_3d(req: TextTo3DRequest):
    """Generate a 3D model from a text prompt.

    Two-step: Text → Image (HunyuanDiT) → 3D (Hunyuan3D-2).
    ~3-8 minutes total.
    """
    if not _HAS_HY3DGEN:
        raise HTTPException(
            501,
            "3D generation requires hy3dgen. "
            "Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
        )

    result = await pool.run_text_to_3d(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image_width=req.image_width,
        image_height=req.image_height,
        image_seed=req.seed,
        image_guidance_scale=req.guidance_scale,
        image_steps=req.num_inference_steps,
        do_texture=req.do_texture,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Text-to-3D generation failed"))

    ts = int(time.time() * 1000)
    glb_filename = f"{ts}_{result['seed']}_hunyuan_text3d.glb"
    img_filename = f"{ts}_{result['seed']}_hunyuan_text3d_ref.png"
    try:
        (GENERATED_DIR / img_filename).write_bytes(
            base64.b64decode(result["reference_image"])
        )
        save_to_history(
            data_bytes=base64.b64decode(result["model_glb"]),
            filename=glb_filename,
            media_type="model/gltf-binary",
            metadata={
                "type": "text-to-3d",
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "seed": result["seed"],
                "reference_image": img_filename,
                "textured": result["textured"],
                "time_seconds": result["time_seconds"],
                "image_time_seconds": result["image_time_seconds"],
                "mesh_time_seconds": result["mesh_time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save text-to-3D to history: {e}")

    return TextTo3DResponse(
        model_glb=result["model_glb"],
        reference_image=result["reference_image"],
        seed=result["seed"],
        time_seconds=result["time_seconds"],
        image_time_seconds=result["image_time_seconds"],
        mesh_time_seconds=result["mesh_time_seconds"],
        textured=result["textured"],
    )


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — Health & Status
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda",
        "gpu_count": _num_gpus,
        "pool": pool.info,
        "available_modes": [m.value for m in ModelMode],
        "capabilities": {
            "sdxl_lightning": {"supported": True, "typical_time": "0.5-1s"},
            "realvis_fast": {"supported": True, "typical_time": "1-2s"},
            "realvis_quality": {"supported": True, "typical_time": "5-8s"},
            "flux": {"supported": True, "typical_time": "3-5s"},
            "hunyuan_image": {
                "model": "HunyuanDiT v1.2 Distilled",
                "supported": True,
                "description": "Text-to-image, bilingual (CN+EN)",
                "typical_time": "5-15s",
            },
            "hunyuan_video": {
                "model": "HunyuanVideo (INT4 quantized)",
                "supported": True,
                "description": "Text-to-video, 13B params",
                "typical_time": "5-12 min",
            },
            "hunyuan_video_i2v": {
                "model": "HunyuanVideo I2V (INT4 quantized)",
                "supported": True,
                "description": "Image-to-video, 13B params",
                "typical_time": "5-12 min",
            },
            "hunyuan_3d": {
                "model": "Hunyuan3D-2",
                "supported": _HAS_HY3DGEN,
                "description": "Image-to-3D with shape gen + texture painting",
                "typical_time": "2-5 min",
                "install_hint": (
                    None if _HAS_HY3DGEN
                    else "pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
                ),
            },
        },
    }


@app.get("/api/queue/status")
async def queue_status():
    """Current GPU pool and slot status."""
    return {
        "pool": pool.info,
        "sdxl_slots": [
            {
                "slot_id": s.slot_id,
                "gpu_ids": s.gpu_ids,
                "device": s.device,
                "loaded_models": list(s._pipelines.keys()),
                "active_task": s.active_task,
                "generation_count": s.generation_count,
            }
            for s in pool.sdxl_slots
        ],
        "flux_slots": [
            {
                "slot_id": s.slot_id,
                "gpu_ids": s.gpu_ids,
                "multi_gpu": s.multi_gpu,
                "loaded_models": list(s._pipelines.keys()),
                "active_task": s.active_task,
                "generation_count": s.generation_count,
            }
            for s in pool.flux_slots
        ],
    }


# ══════════════════════════════════════════════════════════════════════
#  Endpoints — History (unified: images, videos, 3D models)
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/history")
async def list_history(
    limit: int = 200,
    offset: int = 0,
    type_filter: Optional[str] = None,
):
    """List saved outputs (newest first).

    Optional type_filter: image, hunyuan_image, video, video_i2v, 3d, text-to-3d
    """
    if not GENERATED_DIR.exists():
        return {"items": [], "total": 0}

    meta_files = sorted(
        GENERATED_DIR.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

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
    page = items[offset : offset + limit]
    return {"items": page, "total": total}


@app.get("/api/history/image/{filename}")
async def get_history_image(filename: str):
    """Serve a saved image from history (backward compatible)."""
    safe = Path(filename).name
    img_path = GENERATED_DIR / safe
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(404, "Image not found")

    suffix = img_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")

    return Response(content=img_path.read_bytes(), media_type=media_type)


@app.get("/api/history/file/{filename}")
async def get_history_file(filename: str):
    """Serve any saved file from history (image, video, or 3D model)."""
    safe = Path(filename).name
    file_path = GENERATED_DIR / safe
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "application/octet-stream",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return Response(content=file_path.read_bytes(), media_type=media_type)


@app.delete("/api/history/{filename}")
async def delete_history_file(filename: str):
    """Delete a saved output and its metadata."""
    safe = Path(filename).name
    file_path = GENERATED_DIR / safe
    meta_path = GENERATED_DIR / f"{safe}.json"

    deleted = False
    if file_path.exists():
        file_path.unlink()
        deleted = True
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    if not deleted:
        raise HTTPException(404, "File not found")
    return {"deleted": safe}


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    sdxl_cap = len(pool.sdxl_slots)
    flux_cap = len(pool.flux_slots)

    print(f"""
 ================================================================
   AI Generator  (CUDA — Unified GPU Pool)
 ================================================================
   Server:      http://0.0.0.0:{port}
   Docs:        http://0.0.0.0:{port}/docs
   Health:      http://0.0.0.0:{port}/api/health
 ----------------------------------------------------------------
   GPUs:        {_num_gpus} detected
   SDXL slots:  {sdxl_cap} (1 GPU each = {sdxl_cap} parallel)
   FLUX slots:  {flux_cap} (2 GPUs each = {flux_cap} parallel)
 ----------------------------------------------------------------
   SDXL / FLUX (fast image generation):
     lightning        SDXL Lightning 4-step      (~0.5-1s)
     realvis_fast     RealVisXL V5 Lightning     (~1-2s)
     realvis_quality  RealVisXL V5 25-step       (~5-8s)
     flux             FLUX.1 Schnell 4-step      (~3-5s/pair)
 ----------------------------------------------------------------
   Hunyuan (shares GPU pool with SDXL slots):
     Image:       HunyuanDiT v1.2 Distilled      (~5-15s)
     Video T2V:   HunyuanVideo INT4 quantized     (~5-12min)
     Video I2V:   HunyuanVideo I2V INT4 quantized (~5-12min)
     3D:          Hunyuan3D-2 {'✓' if _HAS_HY3DGEN else '✗ (hy3dgen not installed)'}                 (~2-5min)
 ----------------------------------------------------------------
   Endpoints:
     POST /api/generate            SDXL/FLUX single image
     POST /api/generate/batch      SDXL/FLUX parallel batch
     POST /api/generate/raw        SDXL/FLUX raw PNG bytes
     POST /api/hunyuan/image       text → image (PNG)
     POST /api/hunyuan/video       text → video (MP4)
     POST /api/hunyuan/video/i2v   image → video (MP4)
     POST /api/hunyuan/3d          image → 3D model (GLB)
     POST /api/hunyuan/text-to-3d  text → image → 3D model (GLB)
     GET  /api/health              health + all capabilities
     GET  /api/queue/status        GPU slot details
     GET  /api/history             list all outputs
 ================================================================
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
