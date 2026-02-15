# ─────────────────────────────────────────────────────────────────────
#  AI Image Generator — Docker Image
#  Optimized for NVIDIA Grace Blackwell (GB10) / DGX Spark
#
#  Base: NVIDIA NGC PyTorch 25.06 (CUDA 12.9, PyTorch 2.8, ARM64)
#        Native sm_121 Blackwell kernel support
# ─────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/pytorch:25.01-py3

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set HuggingFace cache to a persistent volume mount point
ENV HF_HOME=/data/models
ENV TRANSFORMERS_CACHE=/data/models
ENV DIFFUSERS_CACHE=/data/models
ENV GENERATED_DIR=/data/generated

# Suppress Python bytecode files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# NGC 25.01 does not have pip constraint files, this is safe

# Install Python dependencies (torch is already in the NGC image)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install ffmpeg for video export (HunyuanVideo)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY server.py .

# Copy voice reference files for Chatterbox TTS (voice cloning)
COPY voices/ ./voices/

# Create data directories (will be overridden by volume mounts)
RUN mkdir -p /data/models /data/generated

EXPOSE 8100

# Health check — hit /api/health every 30s
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/api/health')" || exit 1

# Run the server
CMD ["python", "server.py"]
