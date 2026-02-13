"""
Pre-download all required models from HuggingFace.

Usage:
    python download_models.py                      # download all SDXL/FLUX models
    python download_models.py lightning             # download just one
    python download_models.py realvis_fast flux     # download specific ones
    python download_models.py hunyuan_image         # download Hunyuan image model
    python download_models.py hunyuan_video         # download Hunyuan video model
    python download_models.py hunyuan_3d            # download Hunyuan 3D model
    python download_models.py --all                 # download everything (SDXL + FLUX + Hunyuan)
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


# ── Hunyuan Models ───────────────────────────────────────────────────

def download_hunyuan_image():
    """Download HunyuanDiT v1.2 Distilled for image generation (~12GB)."""
    from diffusers import HunyuanDiTPipeline
    print("\n📦 [H1/3] Downloading HunyuanDiT v1.2 Distilled (~12GB)...")
    t0 = time.time()
    HunyuanDiTPipeline.download("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled")
    print(f"   ✅ HunyuanDiT done in {time.time()-t0:.0f}s")


def download_hunyuan_video():
    """Download HunyuanVideo (~26GB — large model, will be INT4 quantized at runtime)."""
    from diffusers import HunyuanVideoPipeline
    print("\n📦 [H2/3] Downloading HunyuanVideo (~26GB, this will take a while)...")
    t0 = time.time()
    HunyuanVideoPipeline.download("hunyuanvideo-community/HunyuanVideo")
    print(f"   ✅ HunyuanVideo done in {time.time()-t0:.0f}s")


def download_hunyuan_3d():
    """Download Hunyuan3D-2 shape + texture models (~15GB total).

    Requires hy3dgen to be installed:
      pip install git+https://github.com/Tencent/Hunyuan3D-2.git
    """
    print("\n📦 [H3/3] Downloading Hunyuan3D-2 (~15GB)...")
    t0 = time.time()
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        print("   Downloading shape generation model...")
        Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
        print("   Downloading texture generation model...")
        Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        print(f"   ✅ Hunyuan3D-2 done in {time.time()-t0:.0f}s")
    except ImportError:
        print("   ⚠️  hy3dgen not installed — skipping Hunyuan3D-2 download.")
        print("      Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git")
        print("      Then re-run: python download_models.py hunyuan_3d")


# ── Model Map ─────────────────────────────────────────────────────────

MODEL_MAP = {
    # SDXL / FLUX models
    "lightning": [download_sdxl_base, download_lightning_unet],
    "realvis_fast": [download_realvis_fast],
    "realvis_quality": [download_realvis_quality],
    "flux": [download_flux],
    # Hunyuan models
    "hunyuan_image": [download_hunyuan_image],
    "hunyuan_video": [download_hunyuan_video],
    "hunyuan_3d": [download_hunyuan_3d],
}

SDXL_FLUX_STEPS = [
    download_sdxl_base,
    download_lightning_unet,
    download_realvis_fast,
    download_realvis_quality,
    download_flux,
]

HUNYUAN_STEPS = [
    download_hunyuan_image,
    download_hunyuan_video,
    download_hunyuan_3d,
]

ALL_STEPS = SDXL_FLUX_STEPS + HUNYUAN_STEPS


def main():
    args = sys.argv[1:]

    # Handle special flags
    if "--all" in args:
        requested = list(MODEL_MAP.keys())
    elif "--hunyuan" in args:
        requested = ["hunyuan_image", "hunyuan_video", "hunyuan_3d"]
    elif args:
        requested = args
    else:
        # Default: only SDXL/FLUX models (original behavior)
        requested = ["lightning", "realvis_fast", "realvis_quality", "flux"]

    sdxl_flux = [r for r in requested if not r.startswith("hunyuan")]
    hunyuan = [r for r in requested if r.startswith("hunyuan")]

    print("=" * 60)
    print("  Model Downloader")
    print("=" * 60)
    if sdxl_flux:
        print(f"  SDXL/FLUX: {', '.join(sdxl_flux)}")
    if hunyuan:
        print(f"  Hunyuan:   {', '.join(hunyuan)}")
    print(f"  Total:     {len(requested)} model(s)")
    print("=" * 60)
    print()
    print("  Flags:")
    print("    --all      Download everything (SDXL + FLUX + Hunyuan)")
    print("    --hunyuan  Download all Hunyuan models only")
    print(f"    Available: {', '.join(MODEL_MAP.keys())}")
    print("=" * 60)

    seen = set()
    steps = []
    for name in requested:
        if name.startswith("--"):
            continue
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
