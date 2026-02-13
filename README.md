# Adams Ass — HQ Image Generator

Multi-GPU image generation platform with parallel batch processing, automatic queuing, persistent image history, and a Next.js frontend deployable to Vercel.

## Architecture

```
┌─────────────────────────────┐       ┌──────────────────────────────────────┐
│   Frontend (Next.js)        │       │   Backend (FastAPI + PyTorch)         │
│                             │       │                                      │
│   Vercel / localhost:3000   │──────▶│   GPU Server / localhost:8100        │
│                             │       │                                      │
│   /api/generate      ──proxy──▶    │   POST /api/generate                 │
│   /api/generate/batch──proxy──▶    │   POST /api/generate/batch           │
│   /api/backend/status──proxy──▶    │   GET  /api/health                   │
│                             │       │   GET  /api/queue/status             │
└─────────────────────────────┘       │                                      │
                                      │   ┌────────────────────────────────┐ │
                                      │   │  GPU Pool                      │ │
                                      │   │                                │ │
                                      │   │  SDXL: 1 GPU per image         │ │
                                      │   │  FLUX: 2 GPUs per image        │ │
                                      │   │  Auto-queue overflow jobs      │ │
                                      │   └────────────────────────────────┘ │
                                      └──────────────────────────────────────┘
```

## Models

| Mode | Model | Steps | Speed (CUDA) | GPUs |
|------|-------|-------|-------------|------|
| `lightning` | SDXL Lightning | 4 | ~0.5-1s | 1 |
| `realvis_fast` | RealVisXL V5.0 Lightning | 5 | ~1-2s | 1 |
| `realvis_quality` | RealVisXL V5.0 | 25 | ~5-8s | 1 |
| `flux` | FLUX.1 Schnell | 4 | ~3-5s | 2 |

## GPU Pool & Batch Generation

The CUDA backend (`server-cuda.py`) auto-detects all available GPUs and creates parallel processing slots:

- **SDXL models**: 1 GPU per image. With 8 GPUs, generate 8 images simultaneously.
- **FLUX**: 2 GPUs per image (device_map balanced). With 8 GPUs, generate 4 images simultaneously.
- **Queuing**: If more prompts than available slots, overflow jobs queue automatically and run as slots free up.
- **Per-GPU locks**: Prevents resource conflicts when SDXL and FLUX share hardware.

### Examples

| Prompts | Model | GPUs Available | Parallel | Queue |
|---------|-------|---------------|----------|-------|
| 2 | SDXL | 4 | 2 simultaneous | 0 |
| 3 | FLUX | 8 | 3 simultaneous (6 GPUs) | 0 |
| 5 | SDXL | 4 | 4 simultaneous | 1 queued |
| 6 | FLUX | 8 | 4 simultaneous (8 GPUs) | 2 queued |

## Quick Start (Local Development)

### 1. Backend (Python — GPU Server)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models first (one-time)
python3 download_models.py

# Start server (auto-detects all GPUs)
python3 server-cuda.py

# Or for MPS/CPU (slower, no batch parallelism):
python3 server.py
```

Backend runs on `http://localhost:8100`

### 2. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

## Deploying Frontend to Vercel

The frontend is a standard Next.js app in the `frontend/` directory.

### Step 1: Connect to Vercel

1. Push this repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new) and import the repo
3. **Set Root Directory to `frontend`** (this is required since the Next.js app is not at the repo root)
4. Vercel will auto-detect Next.js and configure the build

### Step 2: Set Environment Variables

In Vercel Project Settings → Environment Variables, add:

| Variable | Value | Description |
|----------|-------|-------------|
| `BACKEND_URL` | `https://your-gpu-server.com` | URL of your GPU backend |

### Step 3: Run the GPU Backend

The GPU backend must run on a machine with NVIDIA GPUs (e.g., RunPod, Vast.ai, Lambda, your own server).

```bash
# On your GPU server:
cd backend
python3 server-cuda.py
```

Make sure the backend is accessible from the internet (or use a tunnel like ngrok/Cloudflare Tunnel).

### How It Works on Vercel

- The Next.js API routes (`/api/generate`, `/api/generate/batch`) act as proxies to your GPU backend
- The "Start Backend" button is disabled in cloud mode (backend must be started separately)
- Health checks still work and show backend status in the header
- `maxDuration` is set to 300s for generation routes (requires Vercel Pro for >60s)

### Vercel Plan Notes

| Plan | Max Function Duration | Suitable For |
|------|----------------------|-------------|
| Hobby | 60 seconds | SDXL Lightning, RealVis Fast |
| Pro | 300 seconds | All models including FLUX batches |

## API Reference

### Single Image

```bash
POST /api/generate
{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "seed": null,
  "guidance_scale": 0,
  "num_inference_steps": 4,
  "model_mode": "lightning"
}
```

### Batch (Parallel)

```bash
POST /api/generate/batch
{
  "prompts": [
    { "prompt": "a red car", "filename": "car.png", "seed": 42 },
    { "prompt": "a blue boat", "filename": "boat.png" },
    { "prompt": "a green plane" }
  ],
  "negative_prompt": "blurry",
  "width": 1024,
  "height": 1024,
  "guidance_scale": 0,
  "num_inference_steps": 4,
  "model_mode": "lightning"
}
```

Response includes per-image results with success/failure, timing, seed, and which GPU slot was used.

### Health Check

```bash
GET /api/health
# Returns: GPU count, pool info, loaded models, available modes
```

### Queue Status

```bash
GET /api/queue/status
# Returns: per-slot details (loaded models, generation count, active status)
```

## Features

- 4 model modes (Lightning, RealVis Fast, RealVis Quality, FLUX)
- GPU pool with automatic parallel batch generation
- Per-GPU queuing with overflow handling
- 5 aspect ratios (Square, Landscape, Portrait, Photo, Tall)
- Bulk markdown import for mass image generation
- Seed control for reproducibility
- Image history with thumbnails
- Download individual images or ZIP batches
- Base prompt + negative prompt presets
- Backend auto-start (local dev) / health monitoring (production)
- Persistent image history saved to `backend/generated/` (git-ignored)
- History loads across sessions — images survive browser refresh
- Vercel-ready frontend deployment

## Environment Variables

### Frontend (`frontend/.env.local`)

```
BACKEND_URL=http://localhost:8100
```

### Backend

```bash
PORT=8100                          # Server port (default: 8100)
CUDA_VISIBLE_DEVICES=0,1,2,3      # Limit which GPUs to use
```

## Project Structure

```
.
├── backend/
│   ├── server-cuda.py        # CUDA backend with GPU pool + batch (production)
│   ├── server.py             # MPS/CPU backend (development/Mac)
│   ├── download_models.py    # Pre-download all models
│   ├── requirements.txt      # Python dependencies
│   └── generated/            # Saved images + metadata (git-ignored)
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx          # Main UI
│   │   ├── layout.tsx        # Root layout
│   │   ├── globals.css       # Styles
│   │   └── api/
│   │       ├── generate/
│   │       │   ├── route.ts       # Single image proxy
│   │       │   └── batch/
│   │       │       └── route.ts   # Batch image proxy
│   │       ├── history/
│   │       │   ├── route.ts            # History list proxy
│   │       │   └── image/[filename]/
│   │       │       └── route.ts        # Serve history images
│   │       └── backend/
│   │           ├── start/route.ts  # Start backend (local only)
│   │           └── status/route.ts # Health check proxy
│   ├── package.json
│   ├── next.config.ts
│   ├── .env.local            # Local env (git-ignored)
│   └── .env.example          # Template for env vars
└── README.md
```
