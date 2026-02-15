# ─────────────────────────────────────────────────────────────────────
#  AI Image Generator — Docker Image
#  Optimized for NVIDIA Grace Blackwell (GB10) / DGX Spark
#
#  Base: NVIDIA NGC PyTorch 25.01 (CUDA 12.8, PyTorch 2.6, ARM64)
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

# Install Python dependencies (torch is already in the NGC image)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install torchaudio from source (NGC 25.01 doesn't ship it; PyPI wheels are ABI-incompatible)
RUN BUILD_SOX=0 USE_CUDA=1 pip install --no-cache-dir --no-deps --no-build-isolation \
        git+https://github.com/pytorch/audio.git@v2.6.0

# Install Chatterbox TTS and deps WITHOUT touching NGC's torch/torchvision/numpy.
# Strategy: --no-deps on packages that declare torch deps, then install their sub-deps.
RUN pip install --no-cache-dir --no-deps \
        chatterbox-tts \
        s3tokenizer \
        conformer==0.3.2 \
        resemble-perth==1.0.1 \
        librosa==0.11.0 && \
    pip install --no-cache-dir --no-deps \
        spacy-pkuseg \
        pykakasi==2.3.0 \
        pyloudnorm \
        omegaconf && \
    pip install --no-cache-dir \
        antlr4-python3-runtime==4.9.3 \
        soxr \
        audioread \
        lazy_loader \
        msgpack \
        platformdirs \
        pooch \
        decorator \
        joblib \
        numba \
        llvmlite \
        future && \
    python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); from chatterbox.tts import ChatterboxTTS; print('chatterbox: OK')"

# Install ffmpeg for video export (HunyuanVideo) and TTS audio processing
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
