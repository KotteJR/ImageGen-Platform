"""
Hunyuan AI Generator — FastAPI Backend (CUDA multi-GPU)

GPU Pool Architecture:
  - Auto-detects all NVIDIA GPUs at startup
  - Shared pool: any GPU can run any Hunyuan model type
  - 1 GPU per task → up to N tasks in parallel
  - Async queue: overflow jobs wait for available GPU slots

4 generation types:
  1. Image — HunyuanDiT v1.2              (~5-15s per GPU)
  2. Video (T2V) — HunyuanVideo text-to-video (INT4 quantized) (~5-12min per GPU)
  3. Video (I2V) — HunyuanVideo image-to-video (INT4 quantized) (~5-12min per GPU)
  4. 3D    — Hunyuan3D-2 (shape + texture) (~2-5min per GPU)

Endpoints:
  POST /api/hunyuan/image       → text-to-image (returns base64 PNG)
  POST /api/hunyuan/video       → text-to-video (returns base64 MP4)
  POST /api/hunyuan/video/i2v   → image-to-video (returns base64 MP4)
  POST /api/hunyuan/3d          → image-to-3D   (returns base64 GLB)
  POST /api/hunyuan/text-to-3d  → text → image → 3D (returns base64 GLB + PNG)
  GET  /api/health              → health check + pool info
  GET  /api/queue/status        → slot details

History:
  GET    /api/history                  → list outputs (newest first)
  GET    /api/history/file/{filename}  → serve a saved file
  DELETE /api/history/{filename}       → delete a saved output

Usage:
  python server-hunyuan.py                                # auto-detect all GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3 python server-hunyuan.py   # limit to specific GPUs
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
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Verify CUDA ───────────────────────────────────────────────────────
if not torch.cuda.is_available():
    logger.error("CUDA is not available! This server requires NVIDIA GPUs.")
    raise SystemExit(1)

_num_gpus = torch.cuda.device_count()
logger.info(f"CUDA available: {_num_gpus} GPU(s) detected")
for i in range(_num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
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
IMAGE_DTYPE = torch.float16
VIDEO_DTYPE = torch.bfloat16


def _enable_fast_attention(pipe):
    """Enable xformers or SDPA for fastest inference."""
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("    xformers attention enabled")
    except Exception:
        logger.info("    Using PyTorch SDPA attention")
    return pipe


# ── Output History (persistent disk storage) ──────────────────────────

GENERATED_DIR = Path(__file__).parent / "generated_hunyuan"
GENERATED_DIR.mkdir(exist_ok=True)


def save_to_history(
    data_bytes: bytes,
    filename: str,
    media_type: str,
    metadata: dict,
) -> str:
    """Save a generated output + metadata JSON to the generated_hunyuan/ folder."""
    (GENERATED_DIR / filename).write_bytes(data_bytes)
    meta = {**metadata, "filename": filename, "media_type": media_type}
    (GENERATED_DIR / f"{filename}.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved to history: {filename}")
    return filename


# ── GPU Slot ──────────────────────────────────────────────────────────

class HunyuanGPUSlot:
    """A single GPU slot that can run any Hunyuan model.

    Pipelines are loaded on-demand and cached per slot.
    Thread-safe via per-slot lock.
    """

    def __init__(self, slot_id: int, gpu_id: int):
        self.slot_id = slot_id
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.lock = threading.Lock()
        self._pipelines: dict = {}
        self.generation_count = 0
        self.active_task: Optional[str] = None

    def __repr__(self):
        return f"HunyuanGPUSlot(id={self.slot_id}, gpu={self.gpu_id})"

    # ── Pipeline Loaders ──────────────────────────────────────────

    def _load_image_pipeline(self):
        """Load HunyuanDiT for text-to-image."""
        from diffusers import HunyuanDiTPipeline

        MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

        logger.info(f"  Slot {self.slot_id}: Loading HunyuanDiT on {self.device}...")
        pipe = HunyuanDiTPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=IMAGE_DTYPE,
        )
        pipe = pipe.to(self.device)
        _enable_fast_attention(pipe)

        logger.info(f"  Slot {self.slot_id}: HunyuanDiT loaded on {self.device} ✓")
        return pipe

    def _load_video_pipeline(self):
        """Load HunyuanVideo with INT4 quantization + CPU offloading.

        This fits on a single 24GB RTX 3090:
          - Transformer: INT4 quantized (~6-8GB)
          - VAE with tiling: ~2-4GB
          - CPU offloading moves unused components off GPU
        """
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from transformers import BitsAndBytesConfig

        MODEL_ID = "hunyuanvideo-community/HunyuanVideo"

        logger.info(
            f"  Slot {self.slot_id}: Loading HunyuanVideo (INT4 quantized) "
            f"with CPU offload on GPU {self.gpu_id}..."
        )

        # Quantize the transformer to INT4 to fit in 24GB VRAM
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=VIDEO_DTYPE,
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=VIDEO_DTYPE,
        )

        pipe = HunyuanVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            torch_dtype=VIDEO_DTYPE,
        )

        # CPU offloading: model stays in RAM, moves to GPU only during forward pass
        pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)
        pipe.vae.enable_tiling()

        logger.info(
            f"  Slot {self.slot_id}: HunyuanVideo loaded "
            f"(INT4 + CPU offload on GPU {self.gpu_id}) ✓"
        )
        return pipe

    def _load_video_i2v_pipeline(self):
        """Load HunyuanVideo Image-to-Video with INT4 quantization + CPU offloading.

        Same memory profile as text-to-video, fits on a single 24GB RTX 3090.
        """
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
        from transformers import BitsAndBytesConfig

        MODEL_ID = "hunyuanvideo-community/HunyuanVideo-I2V"

        logger.info(
            f"  Slot {self.slot_id}: Loading HunyuanVideo I2V (INT4 quantized) "
            f"with CPU offload on GPU {self.gpu_id}..."
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=VIDEO_DTYPE,
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=VIDEO_DTYPE,
        )

        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)
        pipe.vae.enable_tiling()

        logger.info(
            f"  Slot {self.slot_id}: HunyuanVideo I2V loaded "
            f"(INT4 + CPU offload on GPU {self.gpu_id}) ✓"
        )
        return pipe

    def _load_3d_pipelines(self):
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

    # ── Pipeline Access ───────────────────────────────────────────

    def _get_or_load(self, key: str, loader):
        """Get cached pipeline or load it. Handles OOM by clearing cache."""
        if key in self._pipelines:
            return self._pipelines[key]

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

    # ── Generation Methods ────────────────────────────────────────

    def generate_image(
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
            self.active_task = "image"
            self.generation_count += 1

            pipe = self._get_or_load("image", self._load_image_pipeline)
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
        """Generate a video with HunyuanVideo.

        Returns: (mp4_bytes, seed, elapsed_seconds, num_frames)
        """
        from diffusers.utils import export_to_video

        t0 = time.time()

        with self.lock:
            self.active_task = "video"
            self.generation_count += 1

            pipe = self._get_or_load("video", self._load_video_pipeline)
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

        # Export frames to MP4
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
        image: Image.Image,
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
            self.active_task = "video_i2v"
            self.generation_count += 1

            pipe = self._get_or_load("video_i2v", self._load_video_i2v_pipeline)
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
        image: Image.Image,
        do_texture: bool = True,
    ) -> tuple:
        """Generate a 3D model from an image with Hunyuan3D-2.

        Returns: (glb_bytes, elapsed_seconds)
        """
        t0 = time.time()

        with self.lock:
            self.active_task = "3d"
            self.generation_count += 1

            pipelines = self._get_or_load("3d", self._load_3d_pipelines)
            shapegen = pipelines["shapegen"]
            texgen = pipelines["texgen"]
            rembg = pipelines["rembg"]

            # Remove background if needed (RGBA with transparent bg)
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

        # Export mesh to GLB
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


# ── GPU Pool ──────────────────────────────────────────────────────────

class HunyuanGPUPool:
    """Manages GPU slots for parallel Hunyuan generation.

    All model types share a single pool of GPU slots.
    Jobs acquire a slot, run, then release it back to the pool.
    """

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.slots = [HunyuanGPUSlot(i, i) for i in range(num_gpus)]
        self._executor = ThreadPoolExecutor(max_workers=max(num_gpus, 1))
        self._queue: Optional[asyncio.Queue] = None

        # Stats
        self._total_generated = 0
        self._active_jobs = 0
        self._stats_lock = threading.Lock()

        logger.info(f"Hunyuan GPU Pool: {num_gpus} slots (1 GPU each)")

    def init_async(self):
        """Initialize async queue. Call from async context."""
        self._queue = asyncio.Queue()
        for slot in self.slots:
            self._queue.put_nowait(slot)
        logger.info("Hunyuan GPU Pool async resources initialized")

    async def _acquire_slot(self) -> HunyuanGPUSlot:
        """Wait for and acquire an available GPU slot."""
        return await self._queue.get()

    def _release_slot(self, slot: HunyuanGPUSlot):
        """Return a GPU slot to the pool."""
        self._queue.put_nowait(slot)

    async def run_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: Optional[int],
        guidance_scale: float,
        num_inference_steps: int,
    ) -> dict:
        """Generate an image, waiting for an available GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)
        slot = await self._acquire_slot()

        try:
            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Image job → Slot {slot.slot_id}: "
                f"{width}x{height}, seed={used_seed}"
            )

            loop = asyncio.get_event_loop()
            img_bytes, final_seed, elapsed = await loop.run_in_executor(
                self._executor,
                slot.generate_image,
                prompt, negative_prompt,
                width, height, used_seed,
                guidance_scale, num_inference_steps,
            )

            with self._stats_lock:
                self._active_jobs -= 1
                self._total_generated += 1

            logger.info(f"Image done ← Slot {slot.slot_id}: {elapsed}s")

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
            logger.error(f"Image failed on Slot {slot.slot_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

        finally:
            self._release_slot(slot)

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
        """Generate a video, waiting for an available GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)
        slot = await self._acquire_slot()

        try:
            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Video job → Slot {slot.slot_id}: "
                f"{width}x{height}, {num_frames} frames, seed={used_seed}"
            )

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
                f"Video done ← Slot {slot.slot_id}: "
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
            logger.error(f"Video failed on Slot {slot.slot_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

        finally:
            self._release_slot(slot)

    async def run_video_i2v(
        self,
        image: Image.Image,
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
        slot = await self._acquire_slot()

        try:
            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Video I2V job → Slot {slot.slot_id}: "
                f"{width}x{height}, {num_frames} frames, seed={used_seed}"
            )

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
            self._release_slot(slot)

    async def run_3d(
        self,
        image: Image.Image,
        do_texture: bool = True,
    ) -> dict:
        """Generate a 3D model from an image, waiting for an available GPU slot."""
        slot = await self._acquire_slot()

        try:
            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"3D job → Slot {slot.slot_id}: "
                f"image {image.size}, texture={do_texture}"
            )

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
            self._release_slot(slot)

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
        """Text → Image (HunyuanDiT) → 3D (Hunyuan3D-2).

        Uses the same GPU slot for both steps.
        """
        used_seed = image_seed if image_seed is not None else random.randint(1, 2**31)
        slot = await self._acquire_slot()

        try:
            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Text-to-3D job → Slot {slot.slot_id}: "
                f"\"{prompt[:60]}...\", seed={used_seed}"
            )

            loop = asyncio.get_event_loop()

            # Step 1: Generate image
            logger.info(f"  Step 1/2: Generating reference image...")
            img_bytes, final_seed, img_elapsed = await loop.run_in_executor(
                self._executor,
                slot.generate_image,
                prompt, negative_prompt,
                image_width, image_height, used_seed,
                image_guidance_scale, image_steps,
            )

            # Convert to PIL Image for 3D pipeline
            ref_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

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
            self._release_slot(slot)

    @property
    def info(self) -> dict:
        return {
            "num_gpus": self.num_gpus,
            "total_slots": len(self.slots),
            "active_jobs": self._active_jobs,
            "total_generated": self._total_generated,
            "image_supported": True,
            "video_supported": True,
            "3d_supported": _HAS_HY3DGEN,
        }


# ── App ───────────────────────────────────────────────────────────────

pool = HunyuanGPUPool(_num_gpus)

app = FastAPI(title="Hunyuan AI Generator (CUDA — GPU Pool)")
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


# ── Request / Response Models ─────────────────────────────────────────

# -- Image --

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    seed: Optional[int] = Field(default=None, description="Random seed. None = random.")
    guidance_scale: float = Field(default=5.0, ge=0, le=20)
    num_inference_steps: int = Field(default=25, ge=1, le=100)


class ImageResponse(BaseModel):
    image: str  # base64-encoded PNG
    seed: int
    width: int
    height: int
    time_seconds: float


# -- Video --

class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=848, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=61, ge=9, le=129, description="Number of frames (odd, 9-129).")
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


# -- Video I2V (image-to-video) --

class VideoI2VResponse(BaseModel):
    video: str  # base64-encoded MP4
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    time_seconds: float


# -- 3D --

class ThreeDResponse(BaseModel):
    model_glb: str  # base64-encoded GLB
    time_seconds: float
    textured: bool


# -- Text-to-3D --

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
    reference_image: str  # base64-encoded PNG (the intermediate image)
    seed: int
    time_seconds: float
    image_time_seconds: float
    mesh_time_seconds: float
    textured: bool


# ── Endpoints — Image ─────────────────────────────────────────────────

@app.post("/api/hunyuan/image", response_model=ImageResponse)
async def generate_image(req: ImageRequest):
    """Generate an image using HunyuanDiT v1.2.

    Bilingual (Chinese + English) text-to-image diffusion transformer.
    ~5-15s per image depending on resolution and steps.
    """
    result = await pool.run_image(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Image generation failed"))

    # Save to history
    ts = int(time.time() * 1000)
    filename = f"{ts}_{result['seed']}_hunyuan_image.png"
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
                "guidance_scale": req.guidance_scale,
                "num_inference_steps": req.num_inference_steps,
                "time_seconds": result["time_seconds"],
                "timestamp": ts,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save image to history: {e}")

    return ImageResponse(
        image=result["image"],
        seed=result["seed"],
        width=result["width"],
        height=result["height"],
        time_seconds=result["time_seconds"],
    )


# ── Endpoints — Video ─────────────────────────────────────────────────

@app.post("/api/hunyuan/video", response_model=VideoResponse)
async def generate_video(req: VideoRequest):
    """Generate a video using HunyuanVideo.

    13B parameter diffusion transformer, INT4 quantized to fit on a single 24GB GPU.
    Generates short video clips (2-4 seconds) from text prompts.
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

    # Save to history
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


# ── Endpoints — Video I2V (image-to-video) ────────────────────────────

@app.post("/api/hunyuan/video/i2v", response_model=VideoI2VResponse)
async def generate_video_i2v(
    image: UploadFile = File(..., description="Input image (PNG/JPG/WEBP)"),
    prompt: str = Form(default="", description="Text prompt to guide video generation"),
    width: int = Form(default=848, ge=256, le=1280),
    height: int = Form(default=480, ge=256, le=720),
    num_frames: int = Form(default=61, ge=9, le=129, description="Number of frames (odd, 9-129)"),
    seed: Optional[int] = Form(default=None),
    num_inference_steps: int = Form(default=30, ge=1, le=100),
    fps: int = Form(default=15, ge=8, le=30),
):
    """Generate a video from an image using HunyuanVideo I2V.

    13B parameter diffusion transformer, INT4 quantized.
    Takes an input image as the first frame and generates a video from it.
    Optionally guided by a text prompt describing the desired motion/content.
    ~5-12 minutes per clip depending on resolution and frame count.
    """
    # Read uploaded image
    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
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

    # Save to history
    ts = int(time.time() * 1000)
    vid_filename = f"{ts}_{result['seed']}_hunyuan_video_i2v.mp4"
    img_filename = f"{ts}_{result['seed']}_hunyuan_video_i2v_input.png"
    try:
        # Save the input image
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


# ── Endpoints — 3D ────────────────────────────────────────────────────

@app.post("/api/hunyuan/3d", response_model=ThreeDResponse)
async def generate_3d(
    image: UploadFile = File(..., description="Input image (PNG/JPG/WEBP)"),
    do_texture: bool = Form(True, description="Apply texture painting"),
):
    """Generate a 3D model from an image using Hunyuan3D-2.

    Two-stage pipeline:
      1. Shape generation (Hunyuan3D-DiT flow matching)
      2. Texture painting (Hunyuan3D-Paint)

    Accepts PNG/JPG/WEBP images. Background is automatically removed.
    Returns a GLB file (3D model format).
    ~2-5 minutes per model.
    """
    if not _HAS_HY3DGEN:
        raise HTTPException(
            501,
            "3D generation requires hy3dgen. "
            "Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
        )

    # Read uploaded image
    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    result = await pool.run_3d(
        image=pil_image,
        do_texture=do_texture,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "3D generation failed"))

    # Save to history
    ts = int(time.time() * 1000)
    glb_filename = f"{ts}_hunyuan_3d.glb"
    img_filename = f"{ts}_hunyuan_3d_input.png"
    try:
        # Save the input image too
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
async def generate_text_to_3d(req: TextTo3DRequest):
    """Generate a 3D model from a text prompt.

    Two-step pipeline:
      1. Text → Image (HunyuanDiT)
      2. Image → 3D (Hunyuan3D-2)

    Returns both the 3D model (GLB) and the intermediate reference image (PNG).
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

    # Save to history
    ts = int(time.time() * 1000)
    glb_filename = f"{ts}_{result['seed']}_hunyuan_text3d.glb"
    img_filename = f"{ts}_{result['seed']}_hunyuan_text3d_ref.png"
    try:
        # Save reference image
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


# ── Endpoints — Health & Status ───────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda",
        "gpu_count": _num_gpus,
        "pool": pool.info,
        "capabilities": {
            "image": {
                "model": "HunyuanDiT v1.2 Distilled",
                "supported": True,
                "description": "Text-to-image, bilingual (CN+EN)",
                "typical_time": "5-15s",
            },
            "video": {
                "model": "HunyuanVideo (INT4 quantized)",
                "supported": True,
                "description": "Text-to-video, 13B params",
                "typical_time": "5-12 min",
            },
            "video_i2v": {
                "model": "HunyuanVideo I2V (INT4 quantized)",
                "supported": True,
                "description": "Image-to-video, 13B params",
                "typical_time": "5-12 min",
            },
            "3d": {
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
        "slots": [
            {
                "slot_id": s.slot_id,
                "gpu_id": s.gpu_id,
                "device": s.device,
                "loaded_models": list(s._pipelines.keys()),
                "active_task": s.active_task,
                "generation_count": s.generation_count,
            }
            for s in pool.slots
        ],
    }


# ── Endpoints — History ───────────────────────────────────────────────

@app.get("/api/history")
async def list_history(
    limit: int = 200,
    offset: int = 0,
    type_filter: Optional[str] = None,
):
    """List saved outputs (newest first). Optional filter by type: image, video, 3d, text-to-3d."""
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


@app.get("/api/history/file/{filename}")
async def get_history_file(filename: str):
    """Serve a saved file from history (image, video, or 3D model)."""
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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8200"))

    print(f"""
 ================================================================
   Hunyuan AI Generator  (CUDA GPU Pool)
 ================================================================
   Server:      http://0.0.0.0:{port}
   Docs:        http://0.0.0.0:{port}/docs
   Health:      http://0.0.0.0:{port}/api/health
 ----------------------------------------------------------------
   GPUs:        {_num_gpus} detected
   Slots:       {_num_gpus} (1 GPU each, shared across all models)
 ----------------------------------------------------------------
   Image:       HunyuanDiT v1.2 Distilled      (~5-15s)
   Video T2V:   HunyuanVideo INT4 quantized     (~5-12min)
   Video I2V:   HunyuanVideo I2V INT4 quantized (~5-12min)
   3D:          Hunyuan3D-2 {'✓' if _HAS_HY3DGEN else '✗ (hy3dgen not installed)'}                 (~2-5min)
 ----------------------------------------------------------------
   Endpoints:
     POST /api/hunyuan/image       text → image (PNG)
     POST /api/hunyuan/video       text → video (MP4)
     POST /api/hunyuan/video/i2v   image → video (MP4)
     POST /api/hunyuan/3d          image → 3D model (GLB)
     POST /api/hunyuan/text-to-3d  text → image → 3D model (GLB)
     GET  /api/health              health + capabilities
     GET  /api/queue/status        GPU slot details
     GET  /api/history             list all outputs
 ================================================================
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
