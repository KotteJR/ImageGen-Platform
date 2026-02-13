"""
SDXL / FLUX Image Generator — FastAPI Backend (CUDA optimized)

GPU Pool Architecture:
  - Auto-detects all NVIDIA GPUs at startup
  - SDXL models: 1 GPU per image  → up to N images in parallel
  - FLUX:        2 GPUs per image  → up to N/2 images in parallel
  - Async queue: overflow jobs wait for available GPU slots
  - Per-GPU locks: prevents resource conflicts between model types

4 models:
  1. "lightning"        — SDXL Lightning 4-step      (~0.5-1s per GPU)
  2. "realvis_fast"     — RealVisXL V5.0 Lightning   (~1-2s per GPU)
  3. "realvis_quality"  — RealVisXL V5.0 25-step     (~5-8s per GPU)
  4. "flux"             — FLUX.1 Schnell 4-step      (~3-5s per 2-GPU pair)

Endpoints:
  POST /api/generate        → single image (uses GPU pool)
  POST /api/generate/batch  → batch of images (parallel across GPUs)
  GET  /api/health          → health check + pool info
  GET  /api/queue/status    → current queue status

Examples:
  2 prompts (SDXL)  + 4 GPUs → both run simultaneously on 2 GPUs
  3 prompts (FLUX)  + 8 GPUs → all 3 run simultaneously on 6 GPUs (2 each)
  5 prompts (SDXL)  + 4 GPUs → 4 run in parallel, 1 queues until a slot frees

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional, List

import torch
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

# ── Shared Helpers ────────────────────────────────────────────────────

SDXL_DTYPE = torch.float16
FLUX_DTYPE = torch.bfloat16


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


# ── Image History (persistent disk storage) ───────────────────────────

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def save_to_history(
    img_bytes: bytes,
    prompt: str,
    negative: str,
    seed: int,
    width: int,
    height: int,
    model_mode: str,
    guidance_scale: float,
    num_inference_steps: int,
    elapsed: float,
    custom_filename: Optional[str] = None,
) -> str:
    """Save a generated image + metadata JSON to the generated/ folder."""
    ts = int(time.time() * 1000)
    img_name = custom_filename or f"{ts}_{seed}_{model_mode}.png"
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        img_name += ".png"

    img_path = GENERATED_DIR / img_name
    img_path.write_bytes(img_bytes)

    meta = {
        "filename": img_name,
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "width": width,
        "height": height,
        "model_mode": model_mode,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "time_seconds": elapsed,
        "timestamp": ts,
    }
    meta_path = GENERATED_DIR / f"{img_name}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Saved to history: {img_name}")
    return img_name


# ── GPU Slot ──────────────────────────────────────────────────────────

class GPUSlot:
    """A GPU slot that generates images independently.

    SDXL slots own 1 GPU. FLUX slots own 1-2 GPUs (device_map balanced).
    Each slot has its own pipeline cache and threading lock.
    """

    def __init__(self, slot_id: int, gpu_ids: list):
        self.slot_id = slot_id
        self.gpu_ids = gpu_ids
        self.device = f"cuda:{gpu_ids[0]}"
        self.multi_gpu = len(gpu_ids) > 1
        self.lock = threading.Lock()
        self._pipelines: dict = {}
        self.generation_count = 0
        self.active_model: Optional[str] = None

    def __repr__(self):
        return f"GPUSlot(id={self.slot_id}, gpus={self.gpu_ids})"

    # ── Pipeline Loading ──────────────────────────────────────────

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

    # ── Pipeline Access ───────────────────────────────────────────

    def _load_pipeline(self, model_mode: str):
        loaders = {
            "lightning": self._load_lightning,
            "realvis_fast": self._load_realvis_fast,
            "realvis_quality": self._load_realvis_quality,
            "flux": self._load_flux,
        }
        loader = loaders.get(model_mode)
        if not loader:
            raise ValueError(f"Unknown model mode: {model_mode}")
        return loader()

    def get_pipeline(self, model_mode: str):
        """Get or load a pipeline. Caches per slot. Handles OOM by clearing."""
        if model_mode in self._pipelines:
            return self._pipelines[model_mode]

        try:
            pipe = self._load_pipeline(model_mode)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  Slot {self.slot_id}: OOM loading {model_mode}, clearing cache...")
            for key in list(self._pipelines.keys()):
                del self._pipelines[key]
            torch.cuda.empty_cache()
            pipe = self._load_pipeline(model_mode)

        self._pipelines[model_mode] = pipe
        return pipe

    # ── Generation ────────────────────────────────────────────────

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
        """Generate a single image. Thread-safe via self.lock.

        Returns: (png_bytes, seed, elapsed_seconds)
        """
        t0 = time.time()

        with self.lock:
            self.active_model = model_mode
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

            self.active_model = None

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        elapsed = round(time.time() - t0, 2)

        return buf.getvalue(), seed, elapsed


# ── GPU Pool ──────────────────────────────────────────────────────────

class GPUPool:
    """Manages GPU slots for parallel generation with automatic queuing.

    - SDXL models: 1 slot per GPU  → up to N images in parallel
    - FLUX:        2 GPUs per slot → up to N/2 images in parallel
    - Per-GPU asyncio locks prevent conflicts when SDXL/FLUX share hardware
    - Async slot queues provide fair FIFO scheduling
    """

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus

        # SDXL: 1 GPU per slot
        self.sdxl_slots = [GPUSlot(i, [i]) for i in range(num_gpus)]

        # FLUX: 2 GPUs per slot (single GPU falls back to 1-GPU slot)
        if num_gpus <= 1:
            self.flux_slots = [GPUSlot(0, [0])] if num_gpus == 1 else []
        else:
            self.flux_slots = [
                GPUSlot(i, [i * 2, i * 2 + 1])
                for i in range(num_gpus // 2)
            ]

        self._executor = ThreadPoolExecutor(max_workers=max(num_gpus, 1))

        # Async resources (initialized in init_async)
        self._sdxl_queue: Optional[asyncio.Queue] = None
        self._flux_queue: Optional[asyncio.Queue] = None
        self._gpu_locks: dict = {}

        # Stats
        self._total_generated = 0
        self._active_jobs = 0
        self._stats_lock = threading.Lock()

        logger.info(
            f"GPU Pool: {len(self.sdxl_slots)} SDXL slots (1 GPU each), "
            f"{len(self.flux_slots)} FLUX slots (2 GPUs each)"
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
        """Generate one image, waiting for an available GPU slot."""
        used_seed = seed if seed is not None else random.randint(1, 2**31)

        # 1. Acquire a slot from the appropriate queue (blocks if all busy)
        queue = self._get_queue(model_mode)
        slot = await queue.get()

        try:
            # 2. Acquire per-GPU locks (prevents SDXL/FLUX conflicts)
            await self._acquire_gpus(slot.gpu_ids)

            with self._stats_lock:
                self._active_jobs += 1

            logger.info(
                f"Job → Slot {slot.slot_id} (GPU {slot.gpu_ids}): "
                f"{model_mode}, {width}x{height}, seed={used_seed}"
            )

            try:
                # 3. Run generation in thread pool
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
                # 4. Release GPU locks
                self._release_gpus(slot.gpu_ids)

        finally:
            # 5. Return slot to pool
            queue.put_nowait(slot)

    async def generate_batch(
        self,
        model_mode: str,
        requests: list,
    ) -> list:
        """Generate multiple images in parallel across GPU slots.

        Jobs are dispatched concurrently. If more jobs than slots,
        extras queue automatically and run as slots free up.
        """
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
        }


# ── App ───────────────────────────────────────────────────────────────

pool = GPUPool(_num_gpus)

app = FastAPI(title="Image Generator (CUDA — GPU Pool)")
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


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda",
        "gpu_count": _num_gpus,
        "pool": pool.info,
        "available_modes": [m.value for m in ModelMode],
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate a single image using the GPU pool."""
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

    # Save to disk history
    try:
        save_to_history(
            img_bytes=base64.b64decode(result["image"]),
            prompt=req.prompt,
            negative=req.negative_prompt,
            seed=result["seed"],
            width=result["width"],
            height=result["height"],
            model_mode=result["model_mode"],
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            elapsed=result["time_seconds"],
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
    """Generate multiple images in parallel across GPU slots.

    GPU allocation:
      - SDXL models: 1 GPU per image → up to N in parallel
      - FLUX:        2 GPUs per image → up to N/2 in parallel
      - Overflow jobs queue and run as slots free up
    """
    t0 = time.time()
    model_mode = req.model_mode.value
    capacity = len(pool.flux_slots) if model_mode == "flux" else len(pool.sdxl_slots)

    # Build request dicts
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

    # Run batch (parallel across GPU slots, auto-queued overflow)
    results = await pool.generate_batch(model_mode, requests)

    # Attach filenames from request + save to disk history
    for i, result in enumerate(results):
        custom_fn = None
        if i < len(req.prompts) and req.prompts[i].filename:
            result["filename"] = req.prompts[i].filename
            custom_fn = req.prompts[i].filename

        if result.get("success") and result.get("image"):
            try:
                save_to_history(
                    img_bytes=base64.b64decode(result["image"]),
                    prompt=req.prompts[i].prompt if i < len(req.prompts) else "",
                    negative=req.negative_prompt,
                    seed=result.get("seed", 0),
                    width=result.get("width", req.width),
                    height=result.get("height", req.height),
                    model_mode=model_mode,
                    guidance_scale=req.guidance_scale,
                    num_inference_steps=req.num_inference_steps,
                    elapsed=result.get("time_seconds", 0),
                    custom_filename=custom_fn,
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
                "active_model": s.active_model,
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
                "active_model": s.active_model,
                "generation_count": s.generation_count,
            }
            for s in pool.flux_slots
        ],
    }


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
            # Check that the actual image file still exists
            img_file = GENERATED_DIR / meta["filename"]
            if img_file.exists():
                images.append(meta)
        except Exception:
            continue

    return {"images": images, "total": total}


@app.get("/api/history/image/{filename}")
async def get_history_image(filename: str):
    """Serve a saved image from the generated/ folder."""
    # Sanitize filename to prevent directory traversal
    safe = Path(filename).name
    img_path = GENERATED_DIR / safe
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(404, "Image not found")

    suffix = img_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")

    return Response(content=img_path.read_bytes(), media_type=media_type)


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
    sdxl_cap = len(pool.sdxl_slots)
    flux_cap = len(pool.flux_slots)

    print(f"""
 ================================================================
   Image Generator  (CUDA  GPU Pool)
 ================================================================
   Server:      http://0.0.0.0:{port}
   Docs:        http://0.0.0.0:{port}/docs
   Health:      http://0.0.0.0:{port}/api/health
 ----------------------------------------------------------------
   GPUs:        {_num_gpus} detected
   SDXL slots:  {sdxl_cap} (1 GPU each = {sdxl_cap} parallel)
   FLUX slots:  {flux_cap} (2 GPUs each = {flux_cap} parallel)
 ----------------------------------------------------------------
   Models:
     lightning        SDXL Lightning 4-step      (~0.5-1s)
     realvis_fast     RealVisXL V5 Lightning     (~1-2s)
     realvis_quality  RealVisXL V5 25-step       (~5-8s)
     flux             FLUX.1 Schnell 4-step      (~3-5s/pair)
 ----------------------------------------------------------------
   Endpoints:
     POST /api/generate        single image
     POST /api/generate/batch  parallel batch
     POST /api/generate/raw    raw PNG bytes
     GET  /api/health          health + pool info
     GET  /api/queue/status    slot details
 ================================================================
    """)

    uvicorn.run(app, host="0.0.0.0", port=port)
