# SDXL Lightning — Image Generator

Fast image generation app powered by SDXL Lightning (4-step inference).

## Quick Start

### 1. Backend (Python)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

## Features

- Type a prompt, get an image in ~4s (MPS) or ~1s (CUDA)
- 5 aspect ratios: Square, Landscape, Portrait, Photo, Tall
- Seed control for reproducibility
- Image history with thumbnails
- Download generated images
- Negative prompt support

## Architecture

```
Next.js (localhost:3000) → API Route → Python FastAPI (localhost:8100)
                                        └── SDXL Lightning (4-step, CFG-free)
```
# ImageGen-Platform
