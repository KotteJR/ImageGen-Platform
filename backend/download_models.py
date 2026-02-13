"""
Pre-download all required models from HuggingFace.

Usage:
    python download_models.py          # download all 4
    python download_models.py lightning # download just one
    python download_models.py realvis_fast realvis_quality  # download specific ones
"""

import sys
import time


def download_sdxl_base():
    """Download SDXL Base 1.0 (needed by Lightning mode)."""
    from diffusers import StableDiffusionXLPipeline
    print("\n📦 [1/5] Downloading SDXL Base 1.0 (needed by Lightning)...")
    t0 = time.time()
    StableDiffusionXLPipeline.download("stabilityai/stable-diffusion-xl-base-1.0")
    print(f"   ✅ SDXL Base done in {time.time()-t0:.0f}s")


def download_lightning_unet():
    """Download only the 4-step UNet from Lightning repo (single file, ~5GB)."""
    from huggingface_hub import hf_hub_download
    print("\n📦 [2/5] Downloading SDXL Lightning 4-step UNet (single file)...")
    t0 = time.time()
    path = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")
    print(f"   ✅ Lightning UNet done in {time.time()-t0:.0f}s → {path}")


def download_realvis_fast():
    """Download RealVisXL V5.0 Lightning (~7GB)."""
    from diffusers import StableDiffusionXLPipeline
    print("\n📦 [3/5] Downloading RealVisXL V5.0 Lightning...")
    t0 = time.time()
    StableDiffusionXLPipeline.download("SG161222/RealVisXL_V5.0_Lightning")
    print(f"   ✅ RealVisXL V5 Lightning done in {time.time()-t0:.0f}s")


def download_realvis_quality():
    """Download RealVisXL V5.0 (~7GB)."""
    from diffusers import StableDiffusionXLPipeline
    print("\n📦 [4/5] Downloading RealVisXL V5.0...")
    t0 = time.time()
    StableDiffusionXLPipeline.download("SG161222/RealVisXL_V5.0")
    print(f"   ✅ RealVisXL V5 done in {time.time()-t0:.0f}s")


def download_flux():
    """Download FLUX.1 Schnell (~34GB — largest model)."""
    from diffusers import FluxPipeline
    print("\n📦 [5/5] Downloading FLUX.1 Schnell (~34GB, this will take a while)...")
    t0 = time.time()
    FluxPipeline.download("black-forest-labs/FLUX.1-schnell")
    print(f"   ✅ FLUX.1 Schnell done in {time.time()-t0:.0f}s")


MODEL_MAP = {
    "lightning": [download_sdxl_base, download_lightning_unet],
    "realvis_fast": [download_realvis_fast],
    "realvis_quality": [download_realvis_quality],
    "flux": [download_flux],
}

ALL_STEPS = [
    download_sdxl_base,
    download_lightning_unet,
    download_realvis_fast,
    download_realvis_quality,
    download_flux,
]


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_MAP.keys())

    print("=" * 60)
    print("  Model Downloader")
    print("=" * 60)
    print(f"  Downloading: {', '.join(requested)}")
    print("=" * 60)

    seen = set()
    steps = []
    for name in requested:
        if name not in MODEL_MAP:
            print(f"  ❌ Unknown model: {name}")
            print(f"     Available: {', '.join(MODEL_MAP.keys())}")
            sys.exit(1)
        for fn in MODEL_MAP[name]:
            if fn not in seen:
                seen.add(fn)
                steps.append(fn)

    t_total = time.time()
    for fn in steps:
        fn()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  🎉 All done! Total time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
