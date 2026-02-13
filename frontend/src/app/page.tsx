"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import JSZip from "jszip";

/* ══════════════════════════════════════════════════════════════════════
   Types
   ══════════════════════════════════════════════════════════════════════ */

type ActiveTab = "image" | "video" | "3d";
type AspectRatio = "1:1" | "16:9" | "9:16" | "4:3" | "3:4";
type ModelMode = "lightning" | "realvis_fast" | "realvis_quality" | "flux" | "hunyuan";
type VideoResolution = "landscape" | "portrait" | "square";
type VideoDuration = "short" | "medium" | "long";
type VideoQuality = "fast" | "standard" | "high";
type VideoInputMode = "text" | "image";
type ThreeDInputMode = "text" | "image";

interface GeneratedImage {
  id: string;
  prompt: string;
  filename?: string;
  image?: string;
  imageUrl?: string;
  seed: number;
  width: number;
  height: number;
  time: number;
  fromHistory?: boolean;
}

interface GeneratedVideo {
  id: string;
  prompt: string;
  video: string; // base64 MP4
  seed: number;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  time: number;
}

interface Generated3D {
  id: string;
  prompt?: string;
  modelGlb: string; // base64 GLB
  referenceImage?: string; // base64 PNG
  time: number;
  textured: boolean;
}

interface PromptEntry {
  id: string;
  text: string;
  filename?: string;
}

interface GpuInfo {
  index: number;
  name: string;
  temperature_c: number | null;
  gpu_utilization_pct: number | null;
  memory_utilization_pct: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  memory_free_mb: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  fan_speed_pct: number | null;
  pstate: string;
  clock_graphics_mhz: number | null;
  clock_memory_mhz: number | null;
  slot: {
    slot_id: number;
    slot_type: string;
    loaded_models: string[];
    active_task: string | null;
    generation_count: number;
  } | null;
}

interface GpuStats {
  gpus: GpuInfo[];
  summary: {
    gpu_count: number;
    total_memory_gb: number;
    total_power_draw_w: number;
    total_power_limit_w: number;
    avg_temperature_c: number;
    avg_gpu_utilization_pct: number;
    driver_version: string;
    cuda_version: string;
  };
  pool: {
    num_gpus: number;
    sdxl_slots: Array<{
      slot_id: number;
      gpu_ids: number[];
      loaded_models: string[];
      active_task: string | null;
      generation_count: number;
    }> | number;
    flux_slots: Array<{
      slot_id: number;
      gpu_ids: number[];
      loaded_models: string[];
      active_task: string | null;
      generation_count: number;
    }> | number;
    sdxl_parallel_capacity: number;
    flux_parallel_capacity: number;
    active_jobs: number;
    total_generated: number;
  };
  error?: string;
}

/* ══════════════════════════════════════════════════════════════════════
   Constants
   ══════════════════════════════════════════════════════════════════════ */

const ASPECT_RATIOS: Record<AspectRatio, { w: number; h: number; label: string }> = {
  "1:1":  { w: 1024, h: 1024, label: "Square" },
  "16:9": { w: 1344, h: 768,  label: "Landscape" },
  "9:16": { w: 768,  h: 1344, label: "Portrait" },
  "4:3":  { w: 1152, h: 896,  label: "Photo" },
  "3:4":  { w: 896,  h: 1152, label: "Tall" },
};

const VIDEO_RESOLUTIONS: Record<VideoResolution, { w: number; h: number; label: string; ratio: string }> = {
  landscape: { w: 848, h: 480, label: "Landscape", ratio: "16:9" },
  portrait:  { w: 480, h: 848, label: "Portrait",  ratio: "9:16" },
  square:    { w: 512, h: 512, label: "Square",     ratio: "1:1" },
};

const VIDEO_DURATIONS: Record<VideoDuration, { frames: number; label: string; time: string }> = {
  short:  { frames: 31,  label: "Short",  time: "~2s" },
  medium: { frames: 61,  label: "Medium", time: "~4s" },
  long:   { frames: 121, label: "Long",   time: "~8s" },
};

const VIDEO_QUALITY_PRESETS: Record<VideoQuality, { steps: number; label: string }> = {
  fast:     { steps: 15, label: "Fast" },
  standard: { steps: 30, label: "Standard" },
  high:     { steps: 50, label: "High" },
};

const PRESET_BASE_PROMPT =
  "3D clay render, bright white background, soft studio lighting, single centered object, " +
  "minimal composition, rounded smooth shapes, matte pastel colors, cute stylized, " +
  "clean negative space, product photography style, isometric view, high key lighting, " +
  "no shadows, professional, kid-friendly, 4k, octane render";

const PRESET_NEGATIVE =
  "text, words, letters, watermark, dark, moody, night, shadows, black background, " +
  "realistic, photograph, human, face, person, scary, ugly, blurry, low quality, " +
  "cluttered, busy background, landscape, multiple scenes, nsfw, violent";

/* ══════════════════════════════════════════════════════════════════════
   Helpers
   ══════════════════════════════════════════════════════════════════════ */

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function base64ToBlob(b64: string, type = "image/png"): Blob {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type });
}

function slugify(text: string, maxLen = 40): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, maxLen);
}

function getImageSrc(img: GeneratedImage): string {
  if (img.image) return `data:image/png;base64,${img.image}`;
  if (img.imageUrl) return img.imageUrl;
  return "";
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/* ── Bulk Markdown Parser ─────────────────────────────────────────────── */

function parseBulkMarkdown(text: string): { filename: string; prompt: string }[] {
  const results: { filename: string; prompt: string }[] = [];
  const lines = text.split("\n");
  let pendingFilename: string | null = null;
  let inCodeBlock = false;
  let codeLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    const headingMatch = trimmed.match(/^#{1,4}\s+(?:\d+\.\s+)?(\S+\.(?:png|jpg|jpeg|webp|svg))\s*$/i);
    if (headingMatch && !inCodeBlock) { pendingFilename = headingMatch[1]; continue; }
    if (trimmed.startsWith("```")) {
      if (!inCodeBlock) { inCodeBlock = true; codeLines = []; }
      else {
        inCodeBlock = false;
        const prompt = codeLines.join(" ").replace(/\s+/g, " ").trim();
        if (prompt && pendingFilename) results.push({ filename: pendingFilename, prompt });
        pendingFilename = null;
      }
      continue;
    }
    if (inCodeBlock) codeLines.push(trimmed);
  }
  return results;
}

/* ══════════════════════════════════════════════════════════════════════
   Collapsible Section Component
   ══════════════════════════════════════════════════════════════════════ */

function Section({
  title, defaultOpen = false, badge, children,
}: {
  title: string; defaultOpen?: boolean; badge?: string; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border border-white/[0.06] overflow-hidden bg-white/[0.015]">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-[13px] font-medium text-white/50 hover:text-white/70 hover:bg-white/[0.02] transition"
      >
        <div className="flex items-center gap-2">
          {title}
          {badge && (
            <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] font-normal text-white/30">{badge}</span>
          )}
        </div>
        <svg className={`h-3.5 w-3.5 text-white/20 transition-transform ${open ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>
      {open && <div className="border-t border-white/[0.04] px-4 py-3">{children}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   Model Viewer Component (for 3D GLB files)
   ══════════════════════════════════════════════════════════════════════ */

function ThreeDViewer({ glbUrl }: { glbUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !glbUrl) return;
    const viewer = document.createElement("model-viewer");
    viewer.setAttribute("src", glbUrl);
    viewer.setAttribute("camera-controls", "");
    viewer.setAttribute("auto-rotate", "");
    viewer.setAttribute("shadow-intensity", "1");
    viewer.setAttribute("environment-image", "neutral");
    viewer.style.width = "100%";
    viewer.style.height = "100%";
    viewer.style.backgroundColor = "transparent";
    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(viewer);
    return () => { if (containerRef.current) containerRef.current.innerHTML = ""; };
  }, [glbUrl]);

  return <div ref={containerRef} className="w-full h-full" />;
}

/* ══════════════════════════════════════════════════════════════════════
   Main Page
   ══════════════════════════════════════════════════════════════════════ */

type BackendStatus = "checking" | "online" | "offline" | "starting";

export default function Home() {
  /* ── Tab state ────────────────────────────────────────────────── */
  const [activeTab, setActiveTab] = useState<ActiveTab>("image");

  /* ── Backend status ────────────────────────────────────────────── */
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [backendMessage, setBackendMessage] = useState<string | null>(null);

  /* ═══════════════════════════════════════════════════════════════
     GPU HARDWARE MONITOR
     ═══════════════════════════════════════════════════════════════ */
  const [showGpuPanel, setShowGpuPanel] = useState(false);
  const [gpuStats, setGpuStats] = useState<GpuStats | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const gpuPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchGpuStats = useCallback(async () => {
    try {
      const res = await fetch("/api/gpu/stats");
      if (!res.ok) { setGpuError("Backend unreachable"); return; }
      const data = await res.json();
      if (data.error) { setGpuError(data.error); setGpuStats(null); }
      else { setGpuStats(data); setGpuError(null); }
    } catch { setGpuError("Connection failed"); }
  }, []);

  useEffect(() => {
    if (showGpuPanel && backendStatus === "online") {
      fetchGpuStats();
      gpuPollRef.current = setInterval(fetchGpuStats, 3000);
    }
    return () => { if (gpuPollRef.current) clearInterval(gpuPollRef.current); };
  }, [showGpuPanel, backendStatus, fetchGpuStats]);

  /* ═══════════════════════════════════════════════════════════════
     IMAGE TAB STATE
     ═══════════════════════════════════════════════════════════════ */
  const [prompts, setPrompts] = useState<PromptEntry[]>([{ id: makeId(), text: "" }]);
  const [basePrompt, setBasePrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [showBulkImport, setShowBulkImport] = useState(false);
  const [bulkText, setBulkText] = useState("");
  const [bulkParseCount, setBulkParseCount] = useState<number | null>(null);
  const [modelMode, setModelMode] = useState<ModelMode>("lightning");
  const [aspect, setAspect] = useState<AspectRatio>("1:1");
  const [seed, setSeed] = useState("");
  const [guidanceScale, setGuidanceScale] = useState(0);
  const [steps, setSteps] = useState(4);
  const [imgLoading, setImgLoading] = useState(false);
  const [imgError, setImgError] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; name?: string } | null>(null);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [lastBatchIds, setLastBatchIds] = useState<Set<string>>(new Set());

  /* ═══════════════════════════════════════════════════════════════
     VIDEO TAB STATE
     ═══════════════════════════════════════════════════════════════ */
  const [videoMode, setVideoMode] = useState<VideoInputMode>("image");
  const [videoPrompt, setVideoPrompt] = useState("");
  const [videoResolution, setVideoResolution] = useState<VideoResolution>("landscape");
  const [videoDuration, setVideoDuration] = useState<VideoDuration>("medium");
  const [videoQuality, setVideoQuality] = useState<VideoQuality>("standard");
  const [videoFps, setVideoFps] = useState(15);
  const [videoSeed, setVideoSeed] = useState("");
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoElapsed, setVideoElapsed] = useState(0);
  const [videos, setVideos] = useState<GeneratedVideo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<GeneratedVideo | null>(null);
  const videoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // I2V specific state
  const [videoImageFile, setVideoImageFile] = useState<File | null>(null);
  const [videoImagePreview, setVideoImagePreview] = useState<string | null>(null);
  const [isVideoDragging, setIsVideoDragging] = useState(false);

  /* ═══════════════════════════════════════════════════════════════
     3D TAB STATE
     ═══════════════════════════════════════════════════════════════ */
  const [threeDMode, setThreeDMode] = useState<ThreeDInputMode>("text");
  const [threeDPrompt, setThreeDPrompt] = useState("");
  const [threeDNegative, setThreeDNegative] = useState("");
  const [threeDGuidance, setThreeDGuidance] = useState(5.0);
  const [threeDSteps, setThreeDSteps] = useState(25);
  const [threeDSeed, setThreeDSeed] = useState("");
  const [threeDTexture, setThreeDTexture] = useState(true);
  const [threeDImageFile, setThreeDImageFile] = useState<File | null>(null);
  const [threeDImagePreview, setThreeDImagePreview] = useState<string | null>(null);
  const [threeDLoading, setThreeDLoading] = useState(false);
  const [threeDError, setThreeDError] = useState<string | null>(null);
  const [threeDElapsed, setThreeDElapsed] = useState(0);
  const [models3d, setModels3d] = useState<Generated3D[]>([]);
  const [selectedModel, setSelectedModel] = useState<Generated3D | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const threeDTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const firstInputRef = useRef<HTMLTextAreaElement>(null);

  /* ═══════════════════════════════════════════════════════════════
     OBJECT URLs for media playback
     ═══════════════════════════════════════════════════════════════ */
  const selectedVideoUrl = useMemo(() => {
    if (!selectedVideo?.video) return null;
    return URL.createObjectURL(base64ToBlob(selectedVideo.video, "video/mp4"));
  }, [selectedVideo]);

  const selectedModelUrl = useMemo(() => {
    if (!selectedModel?.modelGlb) return null;
    return URL.createObjectURL(base64ToBlob(selectedModel.modelGlb, "model/gltf-binary"));
  }, [selectedModel]);

  // Cleanup object URLs
  useEffect(() => {
    return () => {
      if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
    };
  }, [selectedVideoUrl]);

  useEffect(() => {
    return () => {
      if (selectedModelUrl) URL.revokeObjectURL(selectedModelUrl);
    };
  }, [selectedModelUrl]);

  /* ═══════════════════════════════════════════════════════════════
     BACKEND STATUS
     ═══════════════════════════════════════════════════════════════ */

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch("/api/backend/status");
      const data = await res.json();
      setBackendStatus(data.running ? "online" : "offline");
    } catch { setBackendStatus("offline"); }
  }, []);

  const startBackend = useCallback(async () => {
    setBackendStatus("starting");
    setBackendMessage(null);
    try {
      const res = await fetch("/api/backend/start", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        setBackendMessage(data.message);
        let attempts = 0;
        const poll = setInterval(async () => {
          attempts++;
          try {
            const check = await fetch("/api/backend/status");
            const status = await check.json();
            if (status.running) { clearInterval(poll); setBackendStatus("online"); setBackendMessage(null); }
            else if (attempts > 60) { clearInterval(poll); setBackendStatus("offline"); setBackendMessage("Backend took too long to start."); }
          } catch { if (attempts > 60) { clearInterval(poll); setBackendStatus("offline"); } }
        }, 2000);
      } else { setBackendStatus("offline"); setBackendMessage(data.error || "Failed to start backend"); }
    } catch { setBackendStatus("offline"); setBackendMessage("Failed to send start request"); }
  }, []);

  useEffect(() => { checkBackend(); const i = setInterval(checkBackend, 10000); return () => clearInterval(i); }, [checkBackend]);

  /* ── Load image history ────────────────────────────────────────── */
  useEffect(() => {
    async function loadHistory() {
      try {
        const res = await fetch("/api/history?limit=200");
        if (!res.ok) return;
        const data = await res.json();
        if (!data.images || !Array.isArray(data.images)) return;
        const historyImages: GeneratedImage[] = data.images.map(
          (item: { filename: string; prompt: string; seed: number; width: number; height: number; time_seconds: number; timestamp: number }) => ({
            id: `history-${item.timestamp}-${item.seed}`,
            prompt: item.prompt || "",
            filename: item.filename,
            imageUrl: `/api/history/image/${encodeURIComponent(item.filename)}`,
            seed: item.seed, width: item.width, height: item.height,
            time: item.time_seconds, fromHistory: true,
          }),
        );
        if (historyImages.length > 0) setImages(historyImages);
      } catch { /* History unavailable */ }
    }
    loadHistory();
  }, []);

  useEffect(() => { firstInputRef.current?.focus(); }, []);

  /* ═══════════════════════════════════════════════════════════════
     IMAGE TAB LOGIC
     ═══════════════════════════════════════════════════════════════ */

  const updatePrompt = useCallback((id: string, text: string) => {
    setPrompts((prev) => prev.map((p) => (p.id === id ? { ...p, text } : p)));
  }, []);
  const addPrompt = useCallback(() => { setPrompts((prev) => [...prev, { id: makeId(), text: "" }]); }, []);
  const removePrompt = useCallback((id: string) => { setPrompts((prev) => (prev.length <= 1 ? prev : prev.filter((p) => p.id !== id))); }, []);

  useEffect(() => {
    if (!bulkText.trim()) { setBulkParseCount(null); return; }
    setBulkParseCount(parseBulkMarkdown(bulkText).length);
  }, [bulkText]);

  const importBulk = useCallback(() => {
    const parsed = parseBulkMarkdown(bulkText);
    if (parsed.length === 0) return;
    setPrompts(parsed.map((item) => ({ id: makeId(), text: item.prompt, filename: item.filename })));
    setBulkText(""); setBulkParseCount(null); setShowBulkImport(false);
  }, [bulkText]);

  const buildFinalPrompt = useCallback(
    (userPrompt: string) => {
      const parts = [userPrompt.trim()];
      if (basePrompt.trim()) parts.push(basePrompt.trim());
      return parts.join(", ");
    },
    [basePrompt],
  );

  const validPrompts = prompts.filter((p) => p.text.trim().length > 0);
  const isMultiPrompt = prompts.length > 1;
  const namedCount = validPrompts.filter((p) => p.filename).length;

  const generateImages = useCallback(async () => {
    if (validPrompts.length === 0 || imgLoading) return;
    setImgLoading(true); setImgError(null);
    setBatchProgress({ current: 0, total: validPrompts.length });
    const { w, h } = ASPECT_RATIOS[aspect];
    const batchIds = new Set<string>();
    let lastImg: GeneratedImage | null = null;

    // Try batch endpoint for multiple prompts (non-Hunyuan only)
    if (validPrompts.length > 1 && modelMode !== "hunyuan") {
      try {
        const res = await fetch("/api/generate/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompts: validPrompts.map((p) => ({ prompt: buildFinalPrompt(p.text), filename: p.filename, seed: seed ? parseInt(seed) : null })),
            negative_prompt: negative.trim() || "", width: w, height: h,
            guidance_scale: guidanceScale, num_inference_steps: steps, model_mode: modelMode,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.results && Array.isArray(data.results)) {
            const newImages: GeneratedImage[] = [];
            for (const result of data.results) {
              if (result.success && result.image) {
                const entry = validPrompts[result.index] || validPrompts[0];
                const img: GeneratedImage = {
                  id: `${Date.now()}-${result.seed}-${result.index}`,
                  prompt: entry.text.trim(), filename: result.filename || entry.filename,
                  image: result.image, seed: result.seed, width: result.width, height: result.height, time: result.time_seconds,
                };
                newImages.push(img); batchIds.add(img.id); lastImg = img;
              }
            }
            if (newImages.length > 0) { setImages((prev) => [...newImages.reverse(), ...prev]); setSelectedImage(newImages[newImages.length - 1]); }
            setLastBatchIds(batchIds); if (lastImg) setSelectedImage(lastImg);
            setBatchProgress(null); setImgLoading(false);
            return;
          }
        }
      } catch { /* fall through to sequential */ }
    }

    // Sequential generation
    for (let i = 0; i < validPrompts.length; i++) {
      const entry = validPrompts[i];
      setBatchProgress({ current: i + 1, total: validPrompts.length, name: entry.filename || undefined });
      const finalPrompt = buildFinalPrompt(entry.text);
      const isHunyuan = modelMode === "hunyuan";
      const apiUrl = isHunyuan ? "/api/hunyuan/image" : "/api/generate";
      const body = isHunyuan
        ? { prompt: finalPrompt, negative_prompt: negative.trim() || "", width: w, height: h, seed: seed ? parseInt(seed) : null, guidance_scale: guidanceScale, num_inference_steps: steps }
        : { prompt: finalPrompt, negative_prompt: negative.trim() || undefined, width: w, height: h, seed: seed ? parseInt(seed) : null, guidance_scale: guidanceScale, num_inference_steps: steps, model_mode: modelMode };

      try {
        const res = await fetch(apiUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        const data = await res.json();
        if (!res.ok) { setImgError(`Prompt ${i + 1} failed: ${data.error || "Generation failed"}`); continue; }
        const img: GeneratedImage = {
          id: `${Date.now()}-${data.seed}`, prompt: entry.text.trim(), filename: entry.filename,
          image: data.image, seed: data.seed, width: data.width, height: data.height, time: data.time_seconds,
        };
        batchIds.add(img.id); lastImg = img;
        setImages((prev) => [img, ...prev]); setSelectedImage(img);
      } catch { setImgError(`Prompt ${i + 1}: Failed to connect to backend.`); checkBackend(); }
    }

    setLastBatchIds(batchIds); if (lastImg) setSelectedImage(lastImg);
    setBatchProgress(null); setImgLoading(false);
  }, [validPrompts, imgLoading, aspect, modelMode, buildFinalPrompt, negative, seed, guidanceScale, steps, checkBackend]);

  const downloadImage = async (img: GeneratedImage) => {
    const link = document.createElement("a");
    if (img.image) link.href = `data:image/png;base64,${img.image}`;
    else if (img.imageUrl) { const res = await fetch(img.imageUrl); link.href = URL.createObjectURL(await res.blob()); }
    link.download = img.filename || `sdxl_${img.seed}.png`;
    link.click();
  };

  const downloadZip = useCallback(async (imagesToZip: GeneratedImage[]) => {
    if (imagesToZip.length === 0) return;
    const zip = new JSZip();
    const folder = zip.folder("images")!;
    const usedNames = new Set<string>();
    for (let i = 0; i < imagesToZip.length; i++) {
      const img = imagesToZip[i];
      let name = img.filename || `${String(i + 1).padStart(2, "0")}_${slugify(img.prompt)}_${img.seed}.png`;
      if (!name.match(/\.(png|jpg|jpeg|webp)$/i)) name += ".png";
      if (usedNames.has(name)) name = `${name.replace(/\.\w+$/, "")}_${img.seed}${name.match(/\.\w+$/)?.[0] || ".png"}`;
      usedNames.add(name);
      let blob: Blob;
      if (img.image) blob = base64ToBlob(img.image);
      else if (img.imageUrl) blob = await (await fetch(img.imageUrl)).blob();
      else continue;
      folder.file(name, blob);
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `images-${Date.now()}.zip`;
    link.click();
  }, []);

  /* ═══════════════════════════════════════════════════════════════
     VIDEO TAB LOGIC
     ═══════════════════════════════════════════════════════════════ */

  const generateVideo = useCallback(async () => {
    if (!videoPrompt.trim() || videoLoading) return;
    setVideoLoading(true); setVideoError(null); setVideoElapsed(0);
    videoTimerRef.current = setInterval(() => setVideoElapsed((p) => p + 1), 1000);

    const { w, h } = VIDEO_RESOLUTIONS[videoResolution];
    const { frames } = VIDEO_DURATIONS[videoDuration];
    const { steps: vSteps } = VIDEO_QUALITY_PRESETS[videoQuality];

    try {
      const res = await fetch("/api/hunyuan/video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: videoPrompt.trim(), width: w, height: h,
          num_frames: frames, seed: videoSeed ? parseInt(videoSeed) : null,
          num_inference_steps: vSteps, fps: videoFps,
        }),
      });
      const data = await res.json();
      if (!res.ok) { setVideoError(data.error || "Video generation failed"); return; }

      const vid: GeneratedVideo = {
        id: makeId(), prompt: videoPrompt.trim(), video: data.video,
        seed: data.seed, width: data.width, height: data.height,
        numFrames: data.num_frames, fps: data.fps, time: data.time_seconds,
      };
      setVideos((prev) => [vid, ...prev]);
      setSelectedVideo(vid);
    } catch { setVideoError("Failed to connect to Hunyuan backend."); }
    finally {
      if (videoTimerRef.current) clearInterval(videoTimerRef.current);
      setVideoLoading(false);
    }
  }, [videoPrompt, videoLoading, videoResolution, videoDuration, videoQuality, videoFps, videoSeed]);

  const downloadVideo = useCallback((vid: GeneratedVideo) => {
    const blob = base64ToBlob(vid.video, "video/mp4");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `hunyuan_video_${vid.seed}.mp4`;
    link.click();
  }, []);

  // I2V — Image file handling
  const handleVideoImage = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setVideoImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setVideoImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  }, []);

  const removeVideoImage = useCallback(() => {
    setVideoImageFile(null);
    setVideoImagePreview(null);
  }, []);

  const generateVideoI2V = useCallback(async () => {
    if (!videoImageFile || videoLoading) return;
    setVideoLoading(true); setVideoError(null); setVideoElapsed(0);
    videoTimerRef.current = setInterval(() => setVideoElapsed((p) => p + 1), 1000);

    const { w, h } = VIDEO_RESOLUTIONS[videoResolution];
    const { frames } = VIDEO_DURATIONS[videoDuration];
    const { steps: vSteps } = VIDEO_QUALITY_PRESETS[videoQuality];

    try {
      const formData = new FormData();
      formData.append("image", videoImageFile);
      if (videoPrompt.trim()) formData.append("prompt", videoPrompt.trim());
      formData.append("width", String(w));
      formData.append("height", String(h));
      formData.append("num_frames", String(frames));
      if (videoSeed) formData.append("seed", videoSeed);
      formData.append("num_inference_steps", String(vSteps));
      formData.append("fps", String(videoFps));

      const res = await fetch("/api/hunyuan/video-i2v", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) { setVideoError(data.error || "Video I2V generation failed"); return; }

      const vid: GeneratedVideo = {
        id: makeId(), prompt: videoPrompt.trim() || "(from image)", video: data.video,
        seed: data.seed, width: data.width, height: data.height,
        numFrames: data.num_frames, fps: data.fps, time: data.time_seconds,
      };
      setVideos((prev) => [vid, ...prev]);
      setSelectedVideo(vid);
    } catch { setVideoError("Failed to connect to Hunyuan backend."); }
    finally {
      if (videoTimerRef.current) clearInterval(videoTimerRef.current);
      setVideoLoading(false);
    }
  }, [videoImageFile, videoPrompt, videoLoading, videoResolution, videoDuration, videoQuality, videoFps, videoSeed]);

  /* ═══════════════════════════════════════════════════════════════
     3D TAB LOGIC
     ═══════════════════════════════════════════════════════════════ */

  // Handle 3D image upload
  const handleThreeDImage = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setThreeDImageFile(file);
    setThreeDImagePreview(URL.createObjectURL(file));
  }, []);

  const clearThreeDImage = useCallback(() => {
    if (threeDImagePreview) URL.revokeObjectURL(threeDImagePreview);
    setThreeDImageFile(null); setThreeDImagePreview(null);
  }, [threeDImagePreview]);

  // Drag-and-drop
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback(() => { setIsDragging(false); }, []);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleThreeDImage(file);
  }, [handleThreeDImage]);

  const generate3D = useCallback(async () => {
    if (threeDLoading) return;

    if (threeDMode === "text" && !threeDPrompt.trim()) return;
    if (threeDMode === "image" && !threeDImageFile) return;

    setThreeDLoading(true); setThreeDError(null); setThreeDElapsed(0);
    threeDTimerRef.current = setInterval(() => setThreeDElapsed((p) => p + 1), 1000);

    try {
      let data;
      if (threeDMode === "text") {
        const res = await fetch("/api/hunyuan/text-to-3d", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: threeDPrompt.trim(), negative_prompt: threeDNegative.trim() || "",
            image_width: 1024, image_height: 1024,
            seed: threeDSeed ? parseInt(threeDSeed) : null,
            guidance_scale: threeDGuidance, num_inference_steps: threeDSteps,
            do_texture: threeDTexture,
          }),
        });
        data = await res.json();
        if (!res.ok) { setThreeDError(data.error || "3D generation failed"); return; }
      } else {
        const formData = new FormData();
        formData.append("image", threeDImageFile!);
        formData.append("do_texture", String(threeDTexture));
        const res = await fetch("/api/hunyuan/3d", { method: "POST", body: formData });
        data = await res.json();
        if (!res.ok) { setThreeDError(data.error || "3D generation failed"); return; }
      }

      const model: Generated3D = {
        id: makeId(),
        prompt: threeDMode === "text" ? threeDPrompt.trim() : undefined,
        modelGlb: data.model_glb,
        referenceImage: data.reference_image,
        time: data.time_seconds,
        textured: data.textured,
      };
      setModels3d((prev) => [model, ...prev]);
      setSelectedModel(model);
    } catch { setThreeDError("Failed to connect to Hunyuan backend."); }
    finally {
      if (threeDTimerRef.current) clearInterval(threeDTimerRef.current);
      setThreeDLoading(false);
    }
  }, [threeDMode, threeDPrompt, threeDNegative, threeDGuidance, threeDSteps, threeDSeed, threeDTexture, threeDImageFile, threeDLoading]);

  const download3D = useCallback((model: Generated3D) => {
    const blob = base64ToBlob(model.modelGlb, "model/gltf-binary");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `hunyuan_3d_${model.id}.glb`;
    link.click();
  }, []);

  /* ═══════════════════════════════════════════════════════════════
     KEYBOARD SHORTCUT
     ═══════════════════════════════════════════════════════════════ */

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (activeTab === "image") generateImages();
      else if (activeTab === "video") { videoMode === "image" ? generateVideoI2V() : generateVideo(); }
      else if (activeTab === "3d") generate3D();
    }
  };

  /* ═══════════════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════════════ */

  const tabAccent = activeTab === "image" ? "violet" : activeTab === "video" ? "blue" : "emerald";

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0f]">

      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="border-b border-white/[0.06] px-6 py-3.5">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-purple-600">
              <svg className="h-4 w-4 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.02-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.81-1.31.04-2.3-1.32-3.14-2.53C4.25 16.56 2.93 11.3 4.7 7.72 5.57 5.94 7.36 4.86 9.28 4.84c1.28-.02 2.5.88 3.28.88.77 0 2.26-1.1 3.83-.92.68.03 2.54.29 3.73 2.01-.1.06-2.36 1.41-2.33 4.2.03 3.34 2.86 4.43 2.89 4.44-.03.08-.46 1.59-1.51 3.13l-.46.92zM13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.71.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-white/90">Adams Ass</h1>
              <p className="text-[11px] text-white/30">AI Studio</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* GPU Monitor Button */}
            <button
              onClick={() => setShowGpuPanel(!showGpuPanel)}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
                showGpuPanel
                  ? "bg-cyan-500/20 text-cyan-300"
                  : "bg-white/[0.04] text-white/30 hover:bg-white/[0.08] hover:text-white/50"
              }`}
              title="GPU Hardware Monitor"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25z" />
              </svg>
              GPUs
              {gpuStats && (
                <span className="rounded bg-white/[0.06] px-1 py-0.5 text-[9px] text-white/25">
                  {gpuStats.summary.gpu_count}
                </span>
              )}
            </button>

            <div className={`h-2 w-2 rounded-full ${
              backendStatus === "online" ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]"
                : backendStatus === "starting" || backendStatus === "checking" ? "bg-amber-400 animate-pulse"
                : "bg-red-400"
            }`} />
            <span className="text-[11px] text-white/35">
              {backendStatus === "online" && "Online"}
              {backendStatus === "offline" && "Offline"}
              {backendStatus === "starting" && "Starting..."}
              {backendStatus === "checking" && "Checking..."}
            </span>
            {backendStatus === "offline" && (
              <button onClick={startBackend} className="rounded-md bg-violet-500/15 px-2.5 py-1 text-[11px] font-medium text-violet-300 hover:bg-violet-500/25 transition">
                Start
              </button>
            )}
          </div>
        </div>
      </header>

      {backendMessage && (
        <div className="border-b border-white/[0.06] bg-amber-500/5 px-6 py-2">
          <div className="mx-auto flex max-w-7xl items-center gap-2">
            <svg className="h-3.5 w-3.5 shrink-0 text-amber-400/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-[11px] text-amber-300/60">{backendMessage}</span>
            <button onClick={() => setBackendMessage(null)} className="ml-auto text-[11px] text-white/20 hover:text-white/40">Dismiss</button>
          </div>
        </div>
      )}

      {/* ── GPU Hardware Monitor Panel ─────────────────────────── */}
      {showGpuPanel && (
        <div className="border-b border-white/[0.06] bg-[#0c0c14] px-6 py-4 animate-fade-in">
          <div className="mx-auto max-w-7xl">
            {gpuError ? (
              <div className="flex items-center gap-2 text-[12px] text-red-400/70">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                {gpuError}
                {backendStatus !== "online" && <span className="text-white/20"> (backend is offline)</span>}
              </div>
            ) : !gpuStats ? (
              <div className="flex items-center gap-2 text-[12px] text-white/20">
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-white/10 border-t-cyan-400" />
                Loading GPU stats...
              </div>
            ) : (
              <>
                {/* Summary Row */}
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <h3 className="text-[13px] font-semibold text-white/70">Hardware Monitor</h3>
                    <div className="flex items-center gap-3 text-[10px] text-white/25">
                      <span>NVIDIA Driver {gpuStats.summary.driver_version}</span>
                      <span>CUDA {gpuStats.summary.cuda_version}</span>
                      <span>{gpuStats.summary.total_memory_gb} GB VRAM</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-[10px]">
                    <span className="text-white/25">
                      Pool: <span className="text-cyan-400/60">{gpuStats.pool.active_jobs}</span> active
                      {" / "}
                      <span className="text-white/40">{gpuStats.pool.total_generated}</span> total
                    </span>
                    <span className="text-white/25">
                      Power: <span className={`${gpuStats.summary.total_power_draw_w > gpuStats.summary.total_power_limit_w * 0.8 ? "text-orange-400/60" : "text-emerald-400/50"}`}>
                        {gpuStats.summary.total_power_draw_w}W
                      </span>
                      {" / "}{gpuStats.summary.total_power_limit_w}W
                    </span>
                  </div>
                </div>

                {/* GPU Cards Grid */}
                <div className="grid grid-cols-4 gap-2">
                  {gpuStats.gpus.map((gpu) => {
                    const tempColor = (gpu.temperature_c ?? 0) > 80 ? "text-red-400" : (gpu.temperature_c ?? 0) > 60 ? "text-orange-400" : (gpu.temperature_c ?? 0) > 40 ? "text-yellow-400" : "text-emerald-400";
                    const utilColor = (gpu.gpu_utilization_pct ?? 0) > 80 ? "text-cyan-400" : (gpu.gpu_utilization_pct ?? 0) > 40 ? "text-blue-400" : "text-white/30";
                    const memPct = gpu.memory_total_mb ? Math.round(((gpu.memory_used_mb ?? 0) / gpu.memory_total_mb) * 100) : 0;
                    const memColor = memPct > 80 ? "bg-red-500/60" : memPct > 50 ? "bg-orange-500/50" : memPct > 10 ? "bg-cyan-500/50" : "bg-white/10";
                    const isActive = gpu.slot?.active_task != null;

                    return (
                      <div
                        key={gpu.index}
                        className={`rounded-lg border p-2.5 transition ${
                          isActive
                            ? "border-cyan-500/30 bg-cyan-500/[0.04]"
                            : "border-white/[0.06] bg-white/[0.015]"
                        }`}
                      >
                        {/* GPU Header */}
                        <div className="mb-2 flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <span className={`inline-block h-1.5 w-1.5 rounded-full ${isActive ? "bg-cyan-400 animate-pulse" : memPct > 10 ? "bg-emerald-400/60" : "bg-white/15"}`} />
                            <span className="text-[11px] font-medium text-white/50">GPU {gpu.index}</span>
                          </div>
                          <span className={`text-[10px] font-mono ${tempColor}`}>{gpu.temperature_c ?? "--"}°C</span>
                        </div>

                        {/* Memory Bar */}
                        <div className="mb-1.5">
                          <div className="flex items-center justify-between text-[9px] text-white/20 mb-0.5">
                            <span>VRAM</span>
                            <span>{gpu.memory_used_mb ? (gpu.memory_used_mb / 1024).toFixed(1) : "0"} / {gpu.memory_total_mb ? (gpu.memory_total_mb / 1024).toFixed(0) : "?"} GB</span>
                          </div>
                          <div className="h-1 w-full rounded-full bg-white/[0.06] overflow-hidden">
                            <div className={`h-full rounded-full transition-all duration-500 ${memColor}`} style={{ width: `${memPct}%` }} />
                          </div>
                        </div>

                        {/* Utilization */}
                        <div className="mb-1.5">
                          <div className="flex items-center justify-between text-[9px] text-white/20 mb-0.5">
                            <span>Utilization</span>
                            <span className={utilColor}>{gpu.gpu_utilization_pct ?? 0}%</span>
                          </div>
                          <div className="h-1 w-full rounded-full bg-white/[0.06] overflow-hidden">
                            <div className="h-full rounded-full bg-cyan-500/40 transition-all duration-500" style={{ width: `${gpu.gpu_utilization_pct ?? 0}%` }} />
                          </div>
                        </div>

                        {/* Stats Row */}
                        <div className="flex items-center justify-between text-[9px] text-white/15">
                          <span>{gpu.power_draw_w?.toFixed(0) ?? "--"}W</span>
                          <span>{gpu.fan_speed_pct ?? "--"}% fan</span>
                          <span>{gpu.pstate}</span>
                        </div>

                        {/* Slot Info */}
                        {gpu.slot && (
                          <div className="mt-1.5 pt-1.5 border-t border-white/[0.04]">
                            <div className="flex items-center gap-1 text-[9px]">
                              {isActive ? (
                                <span className="text-cyan-400/70 animate-pulse">Working: {gpu.slot.active_task}</span>
                              ) : gpu.slot.loaded_models.length > 0 ? (
                                <span className="text-white/20">Loaded: {gpu.slot.loaded_models.join(", ")}</span>
                              ) : (
                                <span className="text-white/10">Idle</span>
                              )}
                            </div>
                            <div className="text-[8px] text-white/10 mt-0.5">
                              Slot {gpu.slot.slot_id} ({gpu.slot.slot_type}) — {gpu.slot.generation_count} generated
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Tab Bar ─────────────────────────────────────────────── */}
      <div className="border-b border-white/[0.06] px-6">
        <div className="mx-auto flex max-w-7xl">
          {([
            { key: "image" as const, label: "Images", icon: (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
              </svg>
            ), color: "violet" },
            { key: "video" as const, label: "Video", icon: (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
              </svg>
            ), color: "blue" },
            { key: "3d" as const, label: "3D Model", icon: (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
              </svg>
            ), color: "emerald" },
          ]).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-5 py-3 text-[13px] font-medium transition border-b-2 -mb-px ${
                activeTab === tab.key
                  ? `border-${tab.color}-500 text-white/90`
                  : "border-transparent text-white/35 hover:text-white/55"
              }`}
            >
              <span className={activeTab === tab.key ? `text-${tab.color}-400` : "text-white/30"}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Content ─────────────────────────────────────────────── */}
      <div className="mx-auto flex w-full max-w-7xl flex-1 gap-6 p-6">

        {/* ══ LEFT SIDEBAR ═════════════════════════════════════════ */}
        <div className="flex w-[380px] shrink-0 flex-col gap-3 overflow-y-auto max-h-[calc(100vh-140px)] pr-1">

          {/* ────────────────────────────────────────────────────────
              IMAGE TAB CONTROLS
              ──────────────────────────────────────────────────────── */}
          {activeTab === "image" && (
            <>
              {/* Prompt Area */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                {!isMultiPrompt ? (
                  <div>
                    <textarea
                      ref={firstInputRef}
                      value={prompts[0]?.text || ""}
                      onChange={(e) => prompts[0] && updatePrompt(prompts[0].id, e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Describe the image you want to generate..."
                      rows={3}
                      className="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-white/90 placeholder-white/25 outline-none transition focus:border-violet-500/40 focus:bg-white/[0.04] resize-none leading-relaxed"
                    />
                    {prompts[0]?.filename && (
                      <div className="mt-2 flex items-center gap-1.5 px-1">
                        <svg className="h-3 w-3 shrink-0 text-violet-400/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                        </svg>
                        <span className="text-[10px] font-mono text-violet-300/50">{prompts[0].filename}</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[11px] font-medium text-white/40">
                        {validPrompts.length} of {prompts.length} prompts ready
                        {namedCount > 0 && <span className="ml-1 text-violet-300/40">({namedCount} named)</span>}
                      </span>
                    </div>
                    <div className="flex flex-col gap-2 max-h-[260px] overflow-y-auto pr-1">
                      {prompts.map((p, idx) => (
                        <div key={p.id} className="group flex items-start gap-2">
                          <span className="mt-2.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/[0.04] text-[10px] font-mono text-white/25">{idx + 1}</span>
                          <div className="flex-1 min-w-0">
                            {p.filename && (
                              <div className="mb-1 flex items-center gap-1.5">
                                <svg className="h-3 w-3 shrink-0 text-violet-400/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                                </svg>
                                <span className="truncate text-[10px] font-mono text-violet-300/50">{p.filename}</span>
                              </div>
                            )}
                            <textarea
                              ref={idx === 0 ? firstInputRef : undefined}
                              value={p.text}
                              onChange={(e) => updatePrompt(p.id, e.target.value)}
                              onKeyDown={handleKeyDown}
                              placeholder="Describe this image..."
                              rows={2}
                              className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[13px] text-white/80 placeholder-white/20 outline-none transition focus:border-violet-500/30 resize-none"
                            />
                          </div>
                          <button onClick={() => removePrompt(p.id)} className="mt-2 shrink-0 rounded-md p-1 text-white/15 hover:bg-red-500/10 hover:text-red-400 transition opacity-0 group-hover:opacity-100" title="Remove">
                            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-3 flex items-center gap-3 border-t border-white/[0.04] pt-3">
                  <button onClick={addPrompt} className="flex items-center gap-1 text-[11px] text-white/30 hover:text-white/50 transition">
                    <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                    Add prompt
                  </button>
                  <span className="text-white/10">|</span>
                  <button onClick={() => setShowBulkImport(!showBulkImport)} className={`flex items-center gap-1 text-[11px] transition ${showBulkImport ? "text-violet-400/70" : "text-white/30 hover:text-white/50"}`}>
                    <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>
                    Bulk import
                  </button>
                  {isMultiPrompt && (
                    <><span className="text-white/10">|</span><button onClick={() => setPrompts([{ id: makeId(), text: "" }])} className="text-[11px] text-white/20 hover:text-red-400/60 transition">Clear all</button></>
                  )}
                </div>

                {showBulkImport && (
                  <div className="mt-3 border-t border-white/[0.04] pt-3">
                    <p className="mb-2 text-[10px] text-white/25 leading-relaxed">
                      Paste markdown: <code className="rounded bg-white/[0.06] px-1 py-0.5 text-violet-300/60">#### filename.png</code> + <code className="rounded bg-white/[0.06] px-1 py-0.5 text-violet-300/60">```code```</code>
                    </p>
                    <textarea value={bulkText} onChange={(e) => setBulkText(e.target.value)} placeholder={`#### 1. my_image.png\n\`\`\`\nA cute fluffy 3D lion cub...\n\`\`\``} rows={6}
                      className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[11px] text-white/60 placeholder-white/15 outline-none transition focus:border-violet-500/30 resize-y font-mono leading-relaxed" />
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-[10px] text-white/25">
                        {bulkParseCount !== null && (bulkParseCount > 0 ? <span className="text-emerald-400/60">Found {bulkParseCount} prompt{bulkParseCount !== 1 ? "s" : ""}</span> : <span className="text-amber-400/50">No prompts found</span>)}
                      </span>
                      <button onClick={importBulk} disabled={!bulkParseCount} className="rounded-lg bg-violet-500/15 px-3 py-1.5 text-[11px] font-medium text-violet-300 hover:bg-violet-500/25 transition disabled:opacity-30 disabled:cursor-not-allowed">
                        Import {bulkParseCount || 0} prompts
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Generate Button */}
              <button onClick={generateImages} disabled={validPrompts.length === 0 || imgLoading}
                className={`flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold transition ${
                  imgLoading ? "bg-violet-600/40 text-white/50 cursor-wait"
                    : "bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-500 hover:to-purple-500 active:scale-[0.98] shadow-lg shadow-violet-500/20"
                } disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none`}>
                {imgLoading ? (
                  <><svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" /><path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" /></svg>
                    {batchProgress ? `${batchProgress.current}/${batchProgress.total}${batchProgress.name ? ` — ${batchProgress.name}` : ""}` : "Generating..."}</>
                ) : (
                  <>Generate{validPrompts.length > 1 ? ` ${validPrompts.length} images` : ""}
                    <kbd className="ml-1.5 rounded border border-white/20 px-1.5 py-0.5 text-[10px] font-normal text-white/40">{"\u2318"}Enter</kbd></>
                )}
              </button>

              {batchProgress && batchProgress.total > 1 && (
                <div className="w-full rounded-full bg-white/[0.04] h-1 overflow-hidden -mt-1">
                  <div className="h-full bg-gradient-to-r from-violet-500 to-purple-500 transition-all duration-500 ease-out" style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }} />
                </div>
              )}

              {imgError && <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3.5 py-2.5 text-[12px] text-red-300/80">{imgError}</div>}

              {/* Settings */}
              <div className="flex flex-col gap-2 mt-1">
                <p className="px-1 text-[10px] font-medium uppercase tracking-widest text-white/15">Settings</p>

                <Section title="Base Prompt" badge="optional">
                  <textarea value={basePrompt} onChange={(e) => setBasePrompt(e.target.value)} placeholder="Style tags to append to every prompt..." rows={3}
                    className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[12px] text-white/70 placeholder-white/20 outline-none transition focus:border-violet-500/30 resize-none leading-relaxed" />
                  <div className="mt-2 flex items-center justify-between">
                    <p className="text-[10px] text-white/20">{basePrompt.trim() ? "Appended to each prompt" : "Empty — prompts as-is"}</p>
                    {!basePrompt.trim()
                      ? <button onClick={() => setBasePrompt(PRESET_BASE_PROMPT)} className="text-[10px] text-violet-400/40 hover:text-violet-400/70 transition">Load preset</button>
                      : <button onClick={() => setBasePrompt("")} className="text-[10px] text-white/25 hover:text-white/40 transition">Clear</button>}
                  </div>
                </Section>

                <Section title="Negative Prompt" badge="optional">
                  <textarea value={negative} onChange={(e) => setNegative(e.target.value)} placeholder="Things to avoid..." rows={3}
                    className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[12px] text-white/70 placeholder-white/20 outline-none transition focus:border-violet-500/30 resize-none leading-relaxed" />
                  <div className="mt-2 flex items-center justify-between">
                    <p className="text-[10px] text-white/20">{negative.trim() ? "Active" : "None"}</p>
                    {!negative.trim()
                      ? <button onClick={() => setNegative(PRESET_NEGATIVE)} className="text-[10px] text-violet-400/40 hover:text-violet-400/70 transition">Load preset</button>
                      : <button onClick={() => setNegative("")} className="text-[10px] text-white/25 hover:text-white/40 transition">Clear</button>}
                  </div>
                </Section>

                <Section title="Model & Quality" defaultOpen={true}>
                  <div className="flex flex-col gap-4">
                    {/* Model grid */}
                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Model</label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {([
                          { key: "lightning" as const, label: "Lightning", desc: "4 steps, ~3-5s", icon: "bolt", cfg: 0, st: 4 },
                          { key: "realvis_fast" as const, label: "RealVis Fast", desc: "5 steps, ~5-10s", icon: "camera", cfg: 1.5, st: 5 },
                          { key: "realvis_quality" as const, label: "RealVis HD", desc: "25 steps, ~30-60s", icon: "sparkle", cfg: 5, st: 25 },
                          { key: "flux" as const, label: "FLUX.1", desc: "4 steps, ~60-90s", icon: "bulb", cfg: 0, st: 4 },
                          { key: "hunyuan" as const, label: "Hunyuan", desc: "25 steps, ~5-15s", icon: "globe", cfg: 5, st: 25 },
                        ]).map((m) => (
                          <button key={m.key}
                            onClick={() => { setModelMode(m.key); setGuidanceScale(m.cfg); setSteps(m.st); }}
                            className={`flex flex-col gap-0.5 rounded-lg border p-2 text-left transition ${
                              modelMode === m.key ? "border-violet-500/50 bg-violet-500/[0.08]" : "border-white/[0.06] hover:border-white/[0.1]"
                            } ${m.key === "hunyuan" ? "col-span-2" : ""}`}>
                            <div className="flex items-center gap-1.5">
                              {m.icon === "bolt" && <svg className={`h-3 w-3 ${modelMode === m.key ? "text-amber-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>}
                              {m.icon === "camera" && <svg className={`h-3 w-3 ${modelMode === m.key ? "text-orange-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" /><path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" /></svg>}
                              {m.icon === "sparkle" && <svg className={`h-3 w-3 ${modelMode === m.key ? "text-emerald-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>}
                              {m.icon === "bulb" && <svg className={`h-3 w-3 ${modelMode === m.key ? "text-sky-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" /></svg>}
                              {m.icon === "globe" && <svg className={`h-3 w-3 ${modelMode === m.key ? "text-rose-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" /></svg>}
                              <span className={`text-[10px] font-semibold ${modelMode === m.key ? "text-white/80" : "text-white/40"}`}>{m.label}</span>
                              {m.key === "hunyuan" && <span className="ml-auto rounded bg-rose-500/10 px-1.5 py-0.5 text-[8px] font-medium text-rose-400/60">Hunyuan</span>}
                            </div>
                            <span className="text-[9px] text-white/20 pl-[18px]">{m.desc}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Aspect Ratio */}
                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Aspect Ratio</label>
                      <div className="grid grid-cols-5 gap-1.5">
                        {(Object.entries(ASPECT_RATIOS) as [AspectRatio, typeof ASPECT_RATIOS["1:1"]][]).map(([key, val]) => (
                          <button key={key} onClick={() => setAspect(key)}
                            className={`flex flex-col items-center gap-0.5 rounded-lg border px-1 py-1.5 text-[10px] transition ${
                              aspect === key ? "border-violet-500/50 bg-violet-500/[0.08] text-white/70" : "border-white/[0.06] text-white/35 hover:border-white/[0.1]"
                            }`}>
                            <div className={`rounded-sm border ${aspect === key ? "border-violet-400/60" : "border-white/20"}`}
                              style={{ width: key === "9:16" || key === "3:4" ? 10 : 16, height: key === "16:9" || key === "4:3" ? 10 : 16 }} />
                            <span>{val.label}</span>
                          </button>
                        ))}
                      </div>
                      <p className="mt-1 text-[10px] text-white/15">{ASPECT_RATIOS[aspect].w} x {ASPECT_RATIOS[aspect].h}</p>
                    </div>

                    {/* CFG & Steps for quality models */}
                    {(modelMode === "realvis_fast" || modelMode === "realvis_quality" || modelMode === "hunyuan") && (
                      <div>
                        <div className="mb-1.5 flex items-center justify-between">
                          <label className="text-[11px] font-medium text-white/40">CFG Scale</label>
                          <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{guidanceScale}</span>
                        </div>
                        <input type="range" min={0} max={20} step={0.5} value={guidanceScale} onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                          className="w-full accent-violet-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-violet-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                      </div>
                    )}

                    {(modelMode === "realvis_quality" || modelMode === "hunyuan") && (
                      <div>
                        <div className="mb-1.5 flex items-center justify-between">
                          <label className="text-[11px] font-medium text-white/40">Steps</label>
                          <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{steps}</span>
                        </div>
                        <input type="range" min={10} max={100} step={1} value={steps} onChange={(e) => setSteps(parseInt(e.target.value))}
                          className="w-full accent-violet-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-violet-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                      </div>
                    )}

                    {/* Seed */}
                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Seed</label>
                      <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="Random"
                        className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-violet-500/30" />
                    </div>
                  </div>
                </Section>
              </div>

              {/* Image History */}
              {images.length > 0 && (
                <div className="mt-2">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-[11px] font-medium text-white/35">History ({images.length})</span>
                    <div className="flex items-center gap-1.5">
                      {lastBatchIds.size > 1 && (
                        <button onClick={() => downloadZip(images.filter((i) => lastBatchIds.has(i.id)))} className="flex items-center gap-1 rounded-md bg-white/[0.04] px-2 py-1 text-[10px] text-white/35 hover:text-white/50 transition" title="Download last batch as ZIP">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>Batch
                        </button>
                      )}
                      {images.length > 1 && (
                        <button onClick={() => downloadZip(images)} className="flex items-center gap-1 rounded-md bg-white/[0.04] px-2 py-1 text-[10px] text-white/35 hover:text-white/50 transition">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0l-3-3m3 3l3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" /></svg>All
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-4 gap-1.5 max-h-[280px] overflow-y-auto pr-1">
                    {images.map((img) => (
                      <button key={img.id} onClick={() => setSelectedImage(img)}
                        className={`relative aspect-square overflow-hidden rounded-lg border transition ${
                          selectedImage?.id === img.id ? "border-violet-500/60 ring-1 ring-violet-500/30"
                            : lastBatchIds.has(img.id) ? "border-violet-500/20 hover:border-violet-500/40" : "border-white/[0.06] hover:border-white/[0.12]"
                        }`}>
                        <img src={getImageSrc(img)} alt={img.prompt} className="h-full w-full object-cover" />
                        {img.filename && <div className="absolute bottom-0 inset-x-0 bg-black/60 px-1 py-0.5"><p className="text-[7px] text-white/70 truncate font-mono">{img.filename}</p></div>}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ────────────────────────────────────────────────────────
              VIDEO TAB CONTROLS
              ──────────────────────────────────────────────────────── */}
          {activeTab === "video" && (
            <>
              {/* Mode switch: From Image (I2V) or From Text (T2V) */}
              <div className="flex rounded-xl border border-white/[0.08] overflow-hidden">
                <button onClick={() => setVideoMode("image")}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-[12px] font-medium transition ${
                    videoMode === "image" ? "bg-blue-500/10 text-blue-300 border-r border-white/[0.06]" : "text-white/35 hover:text-white/55 border-r border-white/[0.06]"
                  }`}>
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" /></svg>
                  From Image
                </button>
                <button onClick={() => setVideoMode("text")}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-[12px] font-medium transition ${
                    videoMode === "text" ? "bg-blue-500/10 text-blue-300" : "text-white/35 hover:text-white/55"
                  }`}>
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" /></svg>
                  From Text
                </button>
              </div>

              {/* I2V: Image upload area */}
              {videoMode === "image" && (
                <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                  {videoImagePreview ? (
                    <div className="relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={videoImagePreview} alt="Input" className="w-full max-h-[180px] object-contain rounded-lg" />
                      <button onClick={removeVideoImage}
                        className="absolute top-1.5 right-1.5 rounded-full bg-black/70 p-1 text-white/60 hover:text-white transition">
                        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                      </button>
                    </div>
                  ) : (
                    <div
                      className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed py-8 px-4 transition cursor-pointer ${
                        isVideoDragging ? "border-blue-400/60 bg-blue-500/[0.06]" : "border-white/[0.08] hover:border-blue-400/30"
                      }`}
                      onDragOver={(e) => { e.preventDefault(); setIsVideoDragging(true); }}
                      onDragLeave={() => setIsVideoDragging(false)}
                      onDrop={(e) => { e.preventDefault(); setIsVideoDragging(false); const f = e.dataTransfer.files[0]; if (f) handleVideoImage(f); }}
                      onClick={() => { const inp = document.createElement("input"); inp.type = "file"; inp.accept = "image/*"; inp.onchange = (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleVideoImage(f); }; inp.click(); }}
                    >
                      <svg className="h-8 w-8 text-white/15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                      </svg>
                      <p className="text-[11px] text-white/30">Drop an image here or click to browse</p>
                      <p className="text-[9px] text-white/15">This image will be the first frame of your video</p>
                    </div>
                  )}
                </div>
              )}

              {/* Prompt input */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                <textarea
                  value={videoPrompt}
                  onChange={(e) => setVideoPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={videoMode === "image" ? "Describe the motion you want (optional)..." : "Describe the video you want to generate..."}
                  rows={3}
                  className="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-white/90 placeholder-white/25 outline-none transition focus:border-blue-500/40 focus:bg-white/[0.04] resize-none leading-relaxed"
                />
                <p className="mt-2 text-[10px] text-white/20">
                  {videoMode === "image"
                    ? "Image-to-Video: Upload an image as the first frame, optionally describe motion. 5-12 min."
                    : "Text-to-Video: Describe a scene to generate from scratch. 5-12 min."}
                </p>
              </div>

              <button
                onClick={videoMode === "image" ? generateVideoI2V : generateVideo}
                disabled={videoMode === "image" ? (!videoImageFile || videoLoading) : (!videoPrompt.trim() || videoLoading)}
                className={`flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold transition ${
                  videoLoading ? "bg-blue-600/40 text-white/50 cursor-wait"
                    : "bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 active:scale-[0.98] shadow-lg shadow-blue-500/20"
                } disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none`}>
                {videoLoading ? (
                  <><svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" /><path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" /></svg>
                    <span>Generating... <span className="animate-timer font-mono">{formatTime(videoElapsed)}</span></span></>
                ) : (
                  <>Generate Video <kbd className="ml-1.5 rounded border border-white/20 px-1.5 py-0.5 text-[10px] font-normal text-white/40">{"\u2318"}Enter</kbd></>
                )}
              </button>

              {videoError && <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3.5 py-2.5 text-[12px] text-red-300/80">{videoError}</div>}

              <div className="flex flex-col gap-2 mt-1">
                <p className="px-1 text-[10px] font-medium uppercase tracking-widest text-white/15">Video Settings</p>

                <Section title="Resolution & Duration" defaultOpen={true}>
                  <div className="flex flex-col gap-4">
                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Resolution</label>
                      <div className="grid grid-cols-3 gap-1.5">
                        {(Object.entries(VIDEO_RESOLUTIONS) as [VideoResolution, typeof VIDEO_RESOLUTIONS["landscape"]][]).map(([key, val]) => (
                          <button key={key} onClick={() => setVideoResolution(key)}
                            className={`flex flex-col items-center gap-0.5 rounded-lg border px-2 py-2 text-[10px] transition ${
                              videoResolution === key ? "border-blue-500/50 bg-blue-500/[0.08] text-white/70" : "border-white/[0.06] text-white/35 hover:border-white/[0.1]"
                            }`}>
                            <span className="font-medium">{val.label}</span>
                            <span className="text-[9px] text-white/20">{val.w}x{val.h}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Duration</label>
                      <div className="grid grid-cols-3 gap-1.5">
                        {(Object.entries(VIDEO_DURATIONS) as [VideoDuration, typeof VIDEO_DURATIONS["medium"]][]).map(([key, val]) => (
                          <button key={key} onClick={() => setVideoDuration(key)}
                            className={`flex flex-col items-center gap-0.5 rounded-lg border px-2 py-2 text-[10px] transition ${
                              videoDuration === key ? "border-blue-500/50 bg-blue-500/[0.08] text-white/70" : "border-white/[0.06] text-white/35 hover:border-white/[0.1]"
                            }`}>
                            <span className="font-medium">{val.label}</span>
                            <span className="text-[9px] text-white/20">{val.time} / {val.frames}f</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="mb-1.5 block text-[11px] font-medium text-white/40">Quality</label>
                      <div className="grid grid-cols-3 gap-1.5">
                        {(Object.entries(VIDEO_QUALITY_PRESETS) as [VideoQuality, typeof VIDEO_QUALITY_PRESETS["standard"]][]).map(([key, val]) => (
                          <button key={key} onClick={() => setVideoQuality(key)}
                            className={`flex flex-col items-center gap-0.5 rounded-lg border px-2 py-2 text-[10px] transition ${
                              videoQuality === key ? "border-blue-500/50 bg-blue-500/[0.08] text-white/70" : "border-white/[0.06] text-white/35 hover:border-white/[0.1]"
                            }`}>
                            <span className="font-medium">{val.label}</span>
                            <span className="text-[9px] text-white/20">{val.steps} steps</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="mb-1.5 block text-[11px] font-medium text-white/40">FPS</label>
                        <select value={videoFps} onChange={(e) => setVideoFps(parseInt(e.target.value))}
                          className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 outline-none transition focus:border-blue-500/30">
                          <option value={15}>15 fps</option>
                          <option value={24}>24 fps</option>
                          <option value={30}>30 fps</option>
                        </select>
                      </div>
                      <div>
                        <label className="mb-1.5 block text-[11px] font-medium text-white/40">Seed</label>
                        <input type="number" value={videoSeed} onChange={(e) => setVideoSeed(e.target.value)} placeholder="Random"
                          className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-blue-500/30" />
                      </div>
                    </div>
                  </div>
                </Section>
              </div>

              {/* Video History */}
              {videos.length > 0 && (
                <div className="mt-2">
                  <span className="text-[11px] font-medium text-white/35">Videos ({videos.length})</span>
                  <div className="mt-2 flex flex-col gap-2 max-h-[200px] overflow-y-auto pr-1">
                    {videos.map((vid) => (
                      <button key={vid.id} onClick={() => setSelectedVideo(vid)}
                        className={`flex items-center gap-3 rounded-lg border p-2 text-left transition ${
                          selectedVideo?.id === vid.id ? "border-blue-500/50 bg-blue-500/[0.06]" : "border-white/[0.06] hover:border-white/[0.1]"
                        }`}>
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-500/10">
                          <svg className="h-4 w-4 text-blue-400/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                          </svg>
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] text-white/60 truncate">{vid.prompt}</p>
                          <p className="text-[9px] text-white/25">{vid.width}x{vid.height} · {vid.numFrames}f · {vid.time}s</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ────────────────────────────────────────────────────────
              3D TAB CONTROLS
              ──────────────────────────────────────────────────────── */}
          {activeTab === "3d" && (
            <>
              {/* Mode switch: Text or Image */}
              <div className="flex rounded-xl border border-white/[0.08] overflow-hidden">
                <button onClick={() => setThreeDMode("text")}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-[12px] font-medium transition ${
                    threeDMode === "text" ? "bg-emerald-500/10 text-emerald-300 border-r border-white/[0.06]" : "text-white/35 hover:text-white/55 border-r border-white/[0.06]"
                  }`}>
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" /></svg>
                  From Text
                </button>
                <button onClick={() => setThreeDMode("image")}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-[12px] font-medium transition ${
                    threeDMode === "image" ? "bg-emerald-500/10 text-emerald-300" : "text-white/35 hover:text-white/55"
                  }`}>
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" /></svg>
                  From Image
                </button>
              </div>

              {/* Input area */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                {threeDMode === "text" ? (
                  <>
                    <textarea value={threeDPrompt} onChange={(e) => setThreeDPrompt(e.target.value)} onKeyDown={handleKeyDown}
                      placeholder="Describe the 3D object you want to create..."
                      rows={3}
                      className="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-white/90 placeholder-white/25 outline-none transition focus:border-emerald-500/40 focus:bg-white/[0.04] resize-none leading-relaxed" />
                    <p className="mt-2 text-[10px] text-white/20">Text → Image (HunyuanDiT) → 3D Model (Hunyuan3D-2). Takes 3-8 minutes.</p>
                  </>
                ) : (
                  <>
                    {!threeDImagePreview ? (
                      <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition cursor-pointer ${
                          isDragging ? "border-emerald-500/50 bg-emerald-500/5" : "border-white/[0.08] hover:border-white/[0.15]"
                        }`}
                        onClick={() => document.getElementById("3d-file-input")?.click()}
                      >
                        <svg className="h-8 w-8 text-white/15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                        <p className="text-[12px] text-white/30">Drop an image here or click to upload</p>
                        <p className="text-[10px] text-white/15">PNG, JPG, WEBP supported</p>
                        <input id="3d-file-input" type="file" accept="image/*" className="hidden"
                          onChange={(e) => e.target.files?.[0] && handleThreeDImage(e.target.files[0])} />
                      </div>
                    ) : (
                      <div className="relative">
                        <img src={threeDImagePreview} alt="Upload preview" className="w-full rounded-xl border border-white/[0.06] object-contain max-h-[200px]" />
                        <button onClick={clearThreeDImage}
                          className="absolute top-2 right-2 rounded-full bg-black/60 p-1.5 text-white/60 hover:text-white/90 transition">
                          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      </div>
                    )}
                    <p className="mt-2 text-[10px] text-white/20">Image → 3D Model (Hunyuan3D-2). Background auto-removed. Takes 2-5 minutes.</p>
                  </>
                )}
              </div>

              {/* Generate 3D Button */}
              <button onClick={generate3D}
                disabled={(threeDMode === "text" ? !threeDPrompt.trim() : !threeDImageFile) || threeDLoading}
                className={`flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold transition ${
                  threeDLoading ? "bg-emerald-600/40 text-white/50 cursor-wait"
                    : "bg-gradient-to-r from-emerald-600 to-teal-600 text-white hover:from-emerald-500 hover:to-teal-500 active:scale-[0.98] shadow-lg shadow-emerald-500/20"
                } disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none`}>
                {threeDLoading ? (
                  <><svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" /><path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" /></svg>
                    <span>Generating 3D... <span className="animate-timer font-mono">{formatTime(threeDElapsed)}</span></span></>
                ) : (
                  <>Generate 3D Model <kbd className="ml-1.5 rounded border border-white/20 px-1.5 py-0.5 text-[10px] font-normal text-white/40">{"\u2318"}Enter</kbd></>
                )}
              </button>

              {threeDError && <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3.5 py-2.5 text-[12px] text-red-300/80">{threeDError}</div>}

              {/* 3D Settings */}
              <div className="flex flex-col gap-2 mt-1">
                <p className="px-1 text-[10px] font-medium uppercase tracking-widest text-white/15">3D Settings</p>

                <Section title="Options" defaultOpen={true}>
                  <div className="flex flex-col gap-4">
                    {/* Texture toggle */}
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-[11px] font-medium text-white/50">Texture Painting</label>
                        <p className="text-[9px] text-white/20">Apply high-res texture to the mesh</p>
                      </div>
                      <button onClick={() => setThreeDTexture(!threeDTexture)}
                        className={`relative h-5 w-9 rounded-full transition ${threeDTexture ? "bg-emerald-500/60" : "bg-white/[0.08]"}`}>
                        <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${threeDTexture ? "left-[18px]" : "left-0.5"}`} />
                      </button>
                    </div>

                    {/* Text-to-3D specific settings */}
                    {threeDMode === "text" && (
                      <>
                        <div>
                          <label className="mb-1.5 block text-[11px] font-medium text-white/40">Negative Prompt</label>
                          <input type="text" value={threeDNegative} onChange={(e) => setThreeDNegative(e.target.value)} placeholder="Things to avoid..."
                            className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-emerald-500/30" />
                        </div>
                        <div>
                          <div className="mb-1.5 flex items-center justify-between">
                            <label className="text-[11px] font-medium text-white/40">CFG Scale</label>
                            <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{threeDGuidance}</span>
                          </div>
                          <input type="range" min={1} max={15} step={0.5} value={threeDGuidance} onChange={(e) => setThreeDGuidance(parseFloat(e.target.value))}
                            className="w-full accent-emerald-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                        </div>
                        <div>
                          <label className="mb-1.5 block text-[11px] font-medium text-white/40">Seed</label>
                          <input type="number" value={threeDSeed} onChange={(e) => setThreeDSeed(e.target.value)} placeholder="Random"
                            className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-emerald-500/30" />
                        </div>
                      </>
                    )}
                  </div>
                </Section>
              </div>

              {/* 3D History */}
              {models3d.length > 0 && (
                <div className="mt-2">
                  <span className="text-[11px] font-medium text-white/35">Models ({models3d.length})</span>
                  <div className="mt-2 flex flex-col gap-2 max-h-[200px] overflow-y-auto pr-1">
                    {models3d.map((model) => (
                      <button key={model.id} onClick={() => setSelectedModel(model)}
                        className={`flex items-center gap-3 rounded-lg border p-2 text-left transition ${
                          selectedModel?.id === model.id ? "border-emerald-500/50 bg-emerald-500/[0.06]" : "border-white/[0.06] hover:border-white/[0.1]"
                        }`}>
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
                          <svg className="h-4 w-4 text-emerald-400/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
                          </svg>
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] text-white/60 truncate">{model.prompt || "Image → 3D"}</p>
                          <p className="text-[9px] text-white/25">{model.time}s · {model.textured ? "Textured" : "Shape only"}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ══ RIGHT DISPLAY AREA ═══════════════════════════════════ */}
        <div className="flex flex-1 flex-col">

          {/* ── Image Display ─────────────────────────────────────── */}
          {activeTab === "image" && (
            selectedImage ? (
              <div className="flex flex-1 flex-col animate-fade-in">
                <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01]">
                  <img src={getImageSrc(selectedImage)} alt={selectedImage.prompt} className="max-h-full max-w-full object-contain" />
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[11px] text-white/30">
                    {selectedImage.filename && <span className="font-mono text-violet-300/50">{selectedImage.filename}</span>}
                    <span>{selectedImage.width}x{selectedImage.height}</span>
                    <span>Seed {selectedImage.seed}</span>
                    <span>{selectedImage.time}s</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => { setPrompts([{ id: makeId(), text: selectedImage.prompt, filename: selectedImage.filename }]); setSeed(String(selectedImage.seed)); }}
                      className="rounded-lg border border-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/35 hover:text-white/55 hover:border-white/[0.1] transition">Reuse</button>
                    <button onClick={() => downloadImage(selectedImage)}
                      className="rounded-lg bg-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition">Download</button>
                  </div>
                </div>
                <p className="mt-1.5 text-[11px] text-white/20 line-clamp-2">{selectedImage.prompt}</p>
              </div>
            ) : (
              <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${imgLoading ? "border-violet-500/30" : "border-white/[0.06]"}`}>
                {imgLoading ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-10 w-10 rounded-full border-2 border-violet-500/60 border-t-transparent animate-spin" />
                    <p className="text-[13px] text-white/30">Generating image...</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-white/15">
                    <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                    </svg>
                    <p className="text-[13px] text-white/25">Generated images appear here</p>
                    <p className="text-[11px] text-white/15">Choose a model and enter a prompt to get started</p>
                  </div>
                )}
              </div>
            )
          )}

          {/* ── Video Display ─────────────────────────────────────── */}
          {activeTab === "video" && (
            selectedVideo && selectedVideoUrl ? (
              <div className="flex flex-1 flex-col animate-fade-in">
                <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-white/[0.06] bg-black">
                  <video key={selectedVideoUrl} controls autoPlay loop className="max-h-full max-w-full object-contain">
                    <source src={selectedVideoUrl} type="video/mp4" />
                  </video>
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[11px] text-white/30">
                    <span>{selectedVideo.width}x{selectedVideo.height}</span>
                    <span>{selectedVideo.numFrames} frames</span>
                    <span>{selectedVideo.fps} fps</span>
                    <span>Seed {selectedVideo.seed}</span>
                    <span>{selectedVideo.time}s</span>
                  </div>
                  <button onClick={() => downloadVideo(selectedVideo)}
                    className="rounded-lg bg-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition">Download MP4</button>
                </div>
                <p className="mt-1.5 text-[11px] text-white/20 line-clamp-2">{selectedVideo.prompt}</p>
              </div>
            ) : (
              <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${videoLoading ? "border-blue-500/30" : "border-white/[0.06]"}`}>
                {videoLoading ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-10 w-10 rounded-full border-2 border-blue-500/60 border-t-transparent animate-spin" />
                    <p className="text-[13px] text-white/30">Generating video...</p>
                    <p className="text-[11px] text-white/15 animate-timer font-mono">{formatTime(videoElapsed)} elapsed</p>
                    <p className="text-[10px] text-white/10 max-w-xs text-center">Video generation takes 5-12 minutes. The model is processing your prompt across billions of parameters.</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-white/15">
                    <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                    </svg>
                    <p className="text-[13px] text-white/25">Generated videos appear here</p>
                    <p className="text-[11px] text-white/15">Upload an image or enter a text prompt to create video</p>
                  </div>
                )}
              </div>
            )
          )}

          {/* ── 3D Display ────────────────────────────────────────── */}
          {activeTab === "3d" && (
            selectedModel && selectedModelUrl ? (
              <div className="flex flex-1 flex-col animate-fade-in">
                <div className="relative flex flex-1 overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.02] to-white/[0.005]">
                  <ThreeDViewer glbUrl={selectedModelUrl} />
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[11px] text-white/30">
                    <span>{selectedModel.textured ? "Textured" : "Shape only"}</span>
                    <span>{selectedModel.time}s</span>
                    {selectedModel.prompt && <span className="truncate max-w-[200px]">{selectedModel.prompt}</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    {selectedModel.referenceImage && (
                      <button onClick={() => {
                        const blob = base64ToBlob(selectedModel.referenceImage!, "image/png");
                        const link = document.createElement("a");
                        link.href = URL.createObjectURL(blob);
                        link.download = `reference_${selectedModel.id}.png`;
                        link.click();
                      }}
                        className="rounded-lg border border-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/35 hover:text-white/55 hover:border-white/[0.1] transition">Reference Image</button>
                    )}
                    <button onClick={() => download3D(selectedModel)}
                      className="rounded-lg bg-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition">Download GLB</button>
                  </div>
                </div>
              </div>
            ) : (
              <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${threeDLoading ? "border-emerald-500/30" : "border-white/[0.06]"}`}>
                {threeDLoading ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-10 w-10 rounded-full border-2 border-emerald-500/60 border-t-transparent animate-spin" />
                    <p className="text-[13px] text-white/30">Generating 3D model...</p>
                    <p className="text-[11px] text-white/15 animate-timer font-mono">{formatTime(threeDElapsed)} elapsed</p>
                    <p className="text-[10px] text-white/10 max-w-xs text-center">
                      {threeDMode === "text"
                        ? "Step 1: Generating reference image, Step 2: Creating 3D shape & texture"
                        : "Creating 3D shape from your image and painting texture"}
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-white/15">
                    <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
                    </svg>
                    <p className="text-[13px] text-white/25">Generated 3D models appear here</p>
                    <p className="text-[11px] text-white/15">Drag to rotate, scroll to zoom. Powered by Hunyuan3D-2</p>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
