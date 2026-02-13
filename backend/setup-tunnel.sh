#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  Cloudflare Tunnel Setup — expose server-cuda.py to the internet
#  Run this ON the GPU server (the 8x 3090 machine)
# ══════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "  Cloudflare Tunnel Setup for GPU Backend"
echo "═══════════════════════════════════════════════"

# 1. Install cloudflared if not present
if ! command -v cloudflared &>/dev/null; then
    echo ""
    echo "[1/3] Installing cloudflared..."
    
    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    elif [ "$ARCH" = "aarch64" ]; then
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        exit 1
    fi
    
    wget -q "$DEB_URL" -O /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb
    rm /tmp/cloudflared.deb
    echo "  ✅ cloudflared installed: $(cloudflared --version)"
else
    echo "[1/3] cloudflared already installed: $(cloudflared --version)"
fi

# 2. Check if server-cuda.py is running on port 8100
echo ""
echo "[2/3] Checking if backend is running on port 8100..."
if curl -s --max-time 3 http://localhost:8100/api/health >/dev/null 2>&1; then
    echo "  ✅ Backend is running on port 8100"
else
    echo "  ⚠️  Backend not running. Start it first:"
    echo "     cd ~/sdxl-lightning-app/backend"
    echo "     source venv/bin/activate"
    echo "     nohup python server-cuda.py > server.log 2>&1 &"
    echo ""
    read -p "  Press Enter after starting the backend, or Ctrl+C to abort..."
    
    # Recheck
    if ! curl -s --max-time 5 http://localhost:8100/api/health >/dev/null 2>&1; then
        echo "  ❌ Backend still not reachable. Aborting."
        exit 1
    fi
    echo "  ✅ Backend is now running"
fi

# 3. Start the tunnel
echo ""
echo "[3/3] Starting Cloudflare Tunnel..."
echo ""
echo "  ⚡ The tunnel will output a URL like:"
echo "     https://something-random.trycloudflare.com"
echo ""
echo "  📋 Copy that URL and set it as BACKEND_URL in Vercel:"
echo "     Vercel Dashboard → Settings → Environment Variables"
echo "     BACKEND_URL = https://something-random.trycloudflare.com"
echo ""
echo "  ⚠️  Keep this terminal open! The tunnel closes when you stop it."
echo "     Use Ctrl+C to stop."
echo ""
echo "═══════════════════════════════════════════════"
echo ""

cloudflared tunnel --url http://localhost:8100
