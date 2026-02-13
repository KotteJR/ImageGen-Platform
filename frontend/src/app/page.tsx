"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import JSZip from "jszip";

/* ── Types ────────────────────────────────────────────────────────────── */

type AspectRatio = "1:1" | "16:9" | "9:16" | "4:3" | "3:4";
type ModelMode = "lightning" | "realvis_fast" | "realvis_quality" | "flux";

interface GeneratedImage {
  id: string;
  prompt: string;
  filename?: string;
  image?: string; // base64 (for live-generated images)
  imageUrl?: string; // URL (for history images loaded from backend)
  seed: number;
  width: number;
  height: number;
  time: number;
  fromHistory?: boolean;
}

interface PromptEntry {
  id: string;
  text: string;
  filename?: string;
}

const ASPECT_RATIOS: Record<AspectRatio, { w: number; h: number; label: string }> = {
  "1:1":  { w: 1024, h: 1024, label: "Square" },
  "16:9": { w: 1344, h: 768,  label: "Landscape" },
  "9:16": { w: 768,  h: 1344, label: "Portrait" },
  "4:3":  { w: 1152, h: 896,  label: "Photo" },
  "3:4":  { w: 896,  h: 1152, label: "Tall" },
};

/* ── Preset suggestions (user can load if they want) ─────────────────── */

const PRESET_BASE_PROMPT =
  "3D clay render, bright white background, soft studio lighting, single centered object, " +
  "minimal composition, rounded smooth shapes, matte pastel colors, cute stylized, " +
  "clean negative space, product photography style, isometric view, high key lighting, " +
  "no shadows, professional, kid-friendly, 4k, octane render";

const PRESET_NEGATIVE =
  "text, words, letters, watermark, dark, moody, night, shadows, black background, " +
  "realistic, photograph, human, face, person, scary, ugly, blurry, low quality, " +
  "cluttered, busy background, landscape, multiple scenes, nsfw, violent";

/* ── Helpers ──────────────────────────────────────────────────────────── */

function makePromptId() {
  return `p-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function base64ToBlob(b64: string): Blob {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: "image/png" });
}

function slugify(text: string, maxLen = 40): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, maxLen);
}

function getImageSrc(img: GeneratedImage): string {
  if (img.image) return `data:image/png;base64,${img.image}`;
  if (img.imageUrl) return img.imageUrl;
  return "";
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

    const headingMatch = trimmed.match(
      /^#{1,4}\s+(?:\d+\.\s+)?(\S+\.(?:png|jpg|jpeg|webp|svg))\s*$/i,
    );
    if (headingMatch && !inCodeBlock) {
      pendingFilename = headingMatch[1];
      continue;
    }

    if (trimmed.startsWith("```")) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLines = [];
      } else {
        inCodeBlock = false;
        const prompt = codeLines.join(" ").replace(/\s+/g, " ").trim();
        if (prompt && pendingFilename) {
          results.push({ filename: pendingFilename, prompt });
        }
        pendingFilename = null;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(trimmed);
    }
  }

  return results;
}

/* ── Collapsible Section ──────────────────────────────────────────────── */

function Section({
  title,
  defaultOpen = false,
  badge,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  badge?: string;
  children: React.ReactNode;
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
            <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] font-normal text-white/30">
              {badge}
            </span>
          )}
        </div>
        <svg
          className={`h-3.5 w-3.5 text-white/20 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>
      {open && <div className="border-t border-white/[0.04] px-4 py-3">{children}</div>}
    </div>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────── */

type BackendStatus = "checking" | "online" | "offline" | "starting";

export default function Home() {
  /* ── Prompt state ──────────────────────────────────────────────── */
  const [prompts, setPrompts] = useState<PromptEntry[]>([
    { id: makePromptId(), text: "" },
  ]);
  const [basePrompt, setBasePrompt] = useState("");
  const [negative, setNegative] = useState("");

  /* ── Bulk import state ─────────────────────────────────────────── */
  const [showBulkImport, setShowBulkImport] = useState(false);
  const [bulkText, setBulkText] = useState("");
  const [bulkParseCount, setBulkParseCount] = useState<number | null>(null);

  /* ── Settings state ────────────────────────────────────────────── */
  const [modelMode, setModelMode] = useState<ModelMode>("lightning");
  const [aspect, setAspect] = useState<AspectRatio>("1:1");
  const [seed, setSeed] = useState<string>("");
  const [guidanceScale, setGuidanceScale] = useState<number>(0);
  const [steps, setSteps] = useState<number>(4);

  /* ── Generation state ──────────────────────────────────────────── */
  const [loading, setLoading] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; name?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [lastBatchIds, setLastBatchIds] = useState<Set<string>>(new Set());

  /* ── Backend status ────────────────────────────────────────────── */
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [backendMessage, setBackendMessage] = useState<string | null>(null);

  const firstInputRef = useRef<HTMLTextAreaElement>(null);

  // ── Backend status checks ─────────────────────────────────────────

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch("/api/backend/status");
      const data = await res.json();
      setBackendStatus(data.running ? "online" : "offline");
    } catch {
      setBackendStatus("offline");
    }
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
            if (status.running) {
              clearInterval(poll);
              setBackendStatus("online");
              setBackendMessage(null);
            } else if (attempts > 60) {
              clearInterval(poll);
              setBackendStatus("offline");
              setBackendMessage("Backend took too long to start.");
            }
          } catch {
            if (attempts > 60) {
              clearInterval(poll);
              setBackendStatus("offline");
            }
          }
        }, 2000);
      } else {
        setBackendStatus("offline");
        setBackendMessage(data.error || "Failed to start backend");
      }
    } catch {
      setBackendStatus("offline");
      setBackendMessage("Failed to send start request");
    }
  }, []);

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, [checkBackend]);

  // ── Load persistent history from backend ───────────────────────────
  useEffect(() => {
    async function loadHistory() {
      try {
        const res = await fetch("/api/history?limit=200");
        if (!res.ok) return;
        const data = await res.json();
        if (!data.images || !Array.isArray(data.images)) return;

        const historyImages: GeneratedImage[] = data.images.map(
          (item: { filename: string; prompt: string; seed: number; width: number; height: number; time_seconds: number; model_mode: string; timestamp: number }) => ({
            id: `history-${item.timestamp}-${item.seed}`,
            prompt: item.prompt || "",
            filename: item.filename,
            imageUrl: `/api/history/image/${encodeURIComponent(item.filename)}`,
            seed: item.seed,
            width: item.width,
            height: item.height,
            time: item.time_seconds,
            fromHistory: true,
          }),
        );

        if (historyImages.length > 0) {
          setImages(historyImages);
        }
      } catch {
        // History unavailable, that's fine
      }
    }
    loadHistory();
  }, []);

  useEffect(() => { firstInputRef.current?.focus(); }, []);

  // ── Prompt management ─────────────────────────────────────────────

  const updatePrompt = useCallback((id: string, text: string) => {
    setPrompts((prev) => prev.map((p) => (p.id === id ? { ...p, text } : p)));
  }, []);

  const addPrompt = useCallback(() => {
    setPrompts((prev) => [...prev, { id: makePromptId(), text: "" }]);
  }, []);

  const removePrompt = useCallback((id: string) => {
    setPrompts((prev) => (prev.length <= 1 ? prev : prev.filter((p) => p.id !== id)));
  }, []);

  // ── Bulk import ───────────────────────────────────────────────────

  useEffect(() => {
    if (!bulkText.trim()) {
      setBulkParseCount(null);
      return;
    }
    const parsed = parseBulkMarkdown(bulkText);
    setBulkParseCount(parsed.length);
  }, [bulkText]);

  const importBulk = useCallback(() => {
    const parsed = parseBulkMarkdown(bulkText);
    if (parsed.length === 0) return;

    const newPrompts: PromptEntry[] = parsed.map((item) => ({
      id: makePromptId(),
      text: item.prompt,
      filename: item.filename,
    }));

    setPrompts(newPrompts);
    setBulkText("");
    setBulkParseCount(null);
    setShowBulkImport(false);
  }, [bulkText]);

  // ── Build final prompt ────────────────────────────────────────────

  const buildFinalPrompt = useCallback(
    (userPrompt: string) => {
      const parts = [userPrompt.trim()];
      if (basePrompt.trim()) parts.push(basePrompt.trim());
      return parts.join(", ");
    },
    [basePrompt],
  );

  // ── Generation ────────────────────────────────────────────────────

  const validPrompts = prompts.filter((p) => p.text.trim().length > 0);

  // ── Sequential generation (single prompt or fallback) ────────────
  const generateSequential = useCallback(async () => {
    const { w, h } = ASPECT_RATIOS[aspect];
    const batchIds = new Set<string>();
    let lastImg: GeneratedImage | null = null;

    for (let i = 0; i < validPrompts.length; i++) {
      const entry = validPrompts[i];
      setBatchProgress({
        current: i + 1,
        total: validPrompts.length,
        name: entry.filename || undefined,
      });

      const finalPrompt = buildFinalPrompt(entry.text);

      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: finalPrompt,
            negative_prompt: negative.trim() || undefined,
            width: w,
            height: h,
            seed: seed ? parseInt(seed) : null,
            guidance_scale: guidanceScale,
            num_inference_steps: steps,
            model_mode: modelMode,
          }),
        });

        const data = await res.json();

        if (!res.ok) {
          setError(`Prompt ${i + 1}${entry.filename ? ` (${entry.filename})` : ""} failed: ${data.error || "Generation failed"}`);
          continue;
        }

        const img: GeneratedImage = {
          id: `${Date.now()}-${data.seed}`,
          prompt: entry.text.trim(),
          filename: entry.filename,
          image: data.image,
          seed: data.seed,
          width: data.width,
          height: data.height,
          time: data.time_seconds,
        };

        batchIds.add(img.id);
        lastImg = img;
        setImages((prev) => [img, ...prev]);
        setSelectedImage(img);
      } catch {
        setError(`Prompt ${i + 1}${entry.filename ? ` (${entry.filename})` : ""}: Failed to connect to backend.`);
        checkBackend();
      }
    }

    return { batchIds, lastImg };
  }, [validPrompts, negative, aspect, seed, checkBackend, buildFinalPrompt, guidanceScale, steps, modelMode]);

  // ── Batch generation (parallel across GPUs) ────────────────────
  const generateBatch = useCallback(async () => {
    const { w, h } = ASPECT_RATIOS[aspect];
    const batchIds = new Set<string>();
    let lastImg: GeneratedImage | null = null;

    setBatchProgress({ current: 0, total: validPrompts.length, name: "parallel" });

    try {
      const res = await fetch("/api/generate/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompts: validPrompts.map((p) => ({
            prompt: buildFinalPrompt(p.text),
            filename: p.filename,
            seed: seed ? parseInt(seed) : null,
          })),
          negative_prompt: negative.trim() || "",
          width: w,
          height: h,
          guidance_scale: guidanceScale,
          num_inference_steps: steps,
          model_mode: modelMode,
        }),
      });

      if (!res.ok) {
        // Batch endpoint not available — fall back to sequential
        return null;
      }

      const data = await res.json();

      if (!data.results || !Array.isArray(data.results)) {
        return null; // unexpected response, fall back
      }

      // Process batch results
      const newImages: GeneratedImage[] = [];
      const errorMsgs: string[] = [];

      for (const result of data.results) {
        if (result.success && result.image) {
          const entry = validPrompts[result.index] || validPrompts[0];
          const img: GeneratedImage = {
            id: `${Date.now()}-${result.seed}-${result.index}`,
            prompt: entry.text.trim(),
            filename: result.filename || entry.filename,
            image: result.image,
            seed: result.seed,
            width: result.width,
            height: result.height,
            time: result.time_seconds,
          };
          newImages.push(img);
          batchIds.add(img.id);
          lastImg = img;
        } else {
          const entry = validPrompts[result.index];
          errorMsgs.push(
            `Prompt ${result.index + 1}${entry?.filename ? ` (${entry.filename})` : ""}: ${result.error || "Failed"}`
          );
        }
      }

      // Add all images at once
      if (newImages.length > 0) {
        setImages((prev) => [...newImages.reverse(), ...prev]);
        setSelectedImage(newImages[newImages.length - 1]);
      }

      if (errorMsgs.length > 0) {
        setError(errorMsgs.join("\n"));
      }

      return { batchIds, lastImg, totalTime: data.total_time_seconds, capacity: data.parallel_capacity };
    } catch {
      return null; // network error, fall back to sequential
    }
  }, [validPrompts, negative, aspect, seed, buildFinalPrompt, guidanceScale, steps, modelMode]);

  // ── Main generate function ─────────────────────────────────────
  const generate = useCallback(async () => {
    if (validPrompts.length === 0 || loading) return;

    setLoading(true);
    setError(null);
    setBatchProgress({ current: 0, total: validPrompts.length });

    let batchIds = new Set<string>();
    let lastImg: GeneratedImage | null = null;

    if (validPrompts.length > 1) {
      // Try batch endpoint first (parallel GPU processing)
      const batchResult = await generateBatch();
      if (batchResult) {
        batchIds = batchResult.batchIds;
        lastImg = batchResult.lastImg;
      } else {
        // Batch failed or unavailable — fall back to sequential
        const seqResult = await generateSequential();
        batchIds = seqResult.batchIds;
        lastImg = seqResult.lastImg;
      }
    } else {
      // Single prompt — use single endpoint
      const seqResult = await generateSequential();
      batchIds = seqResult.batchIds;
      lastImg = seqResult.lastImg;
    }

    setLastBatchIds(batchIds);
    if (lastImg) setSelectedImage(lastImg);
    setBatchProgress(null);
    setLoading(false);
  }, [validPrompts, loading, generateBatch, generateSequential]);

  // ── Keyboard shortcut ─────────────────────────────────────────────

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      generate();
    }
  };

  // ── Downloads ─────────────────────────────────────────────────────

  const downloadImage = async (img: GeneratedImage) => {
    const link = document.createElement("a");
    if (img.image) {
      link.href = `data:image/png;base64,${img.image}`;
    } else if (img.imageUrl) {
      const res = await fetch(img.imageUrl);
      const blob = await res.blob();
      link.href = URL.createObjectURL(blob);
    }
    link.download = img.filename || `sdxl_${img.seed}.png`;
    link.click();
  };

  const downloadZip = useCallback(async (imagesToZip: GeneratedImage[]) => {
    if (imagesToZip.length === 0) return;
    const zip = new JSZip();
    const folder = zip.folder("sdxl-images")!;
    const usedNames = new Set<string>();

    for (let i = 0; i < imagesToZip.length; i++) {
      const img = imagesToZip[i];
      let name: string;
      if (img.filename) {
        name = img.filename;
        if (!name.match(/\.(png|jpg|jpeg|webp)$/i)) name += ".png";
      } else {
        name = `${String(i + 1).padStart(2, "0")}_${slugify(img.prompt)}_${img.seed}.png`;
      }

      if (usedNames.has(name)) {
        const ext = name.lastIndexOf(".");
        name = `${name.slice(0, ext)}_${img.seed}${name.slice(ext)}`;
      }
      usedNames.add(name);

      let blob: Blob;
      if (img.image) {
        blob = base64ToBlob(img.image);
      } else if (img.imageUrl) {
        const res = await fetch(img.imageUrl);
        blob = await res.blob();
      } else {
        continue;
      }
      folder.file(name, blob);
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `sdxl-batch-${Date.now()}.zip`;
    link.click();
    URL.revokeObjectURL(url);
  }, []);

  const downloadLastBatch = useCallback(() => {
    downloadZip(images.filter((img) => lastBatchIds.has(img.id)));
  }, [images, lastBatchIds, downloadZip]);

  const downloadAllImages = useCallback(() => {
    downloadZip(images);
  }, [images, downloadZip]);

  // ── Derived ────────────────────────────────────────────────────────
  const namedCount = validPrompts.filter((p) => p.filename).length;
  const isMultiPrompt = prompts.length > 1;

  // ── Render ────────────────────────────────────────────────────────

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
              <p className="text-[11px] text-white/30">
                HQ Image Generator
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${
                backendStatus === "online"
                  ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]"
                  : backendStatus === "starting" || backendStatus === "checking"
                    ? "bg-amber-400 animate-pulse"
                    : "bg-red-400"
              }`} />
              <span className="text-[11px] text-white/35">
                {backendStatus === "online" && "Online"}
                {backendStatus === "offline" && "Offline"}
                {backendStatus === "starting" && "Starting..."}
                {backendStatus === "checking" && "Checking..."}
              </span>
              {backendStatus === "offline" && (
                <button
                  onClick={startBackend}
                  className="rounded-md bg-violet-500/15 px-2.5 py-1 text-[11px] font-medium text-violet-300 hover:bg-violet-500/25 transition"
                >
                  Start
                </button>
              )}
            </div>
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
            <button onClick={() => setBackendMessage(null)} className="ml-auto text-[11px] text-white/20 hover:text-white/40">
              Dismiss
            </button>
          </div>
        </div>
      )}

      <div className="mx-auto flex w-full max-w-7xl flex-1 gap-6 p-6">
        {/* ── Left: Controls ──────────────────────────────────── */}
        <div className="flex w-[380px] shrink-0 flex-col gap-3 overflow-y-auto max-h-[calc(100vh-100px)] pr-1">

          {/* ═══ Prompt Area ═══ */}
          <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
            {/* Single prompt: clean simple input */}
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
              /* Multi-prompt: numbered list */
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[11px] font-medium text-white/40">
                    {validPrompts.length} of {prompts.length} prompts ready
                    {namedCount > 0 && (
                      <span className="ml-1 text-violet-300/40">({namedCount} with filenames)</span>
                    )}
                  </span>
                </div>
                <div className="flex flex-col gap-2 max-h-[260px] overflow-y-auto pr-1">
                  {prompts.map((p, idx) => (
                    <div key={p.id} className="group flex items-start gap-2">
                      <span className="mt-2.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/[0.04] text-[10px] font-mono text-white/25">
                        {idx + 1}
                      </span>
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
                      <button
                        onClick={() => removePrompt(p.id)}
                        className="mt-2 shrink-0 rounded-md p-1 text-white/15 hover:bg-red-500/10 hover:text-red-400 transition opacity-0 group-hover:opacity-100"
                        title="Remove"
                      >
                        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions row */}
            <div className="mt-3 flex items-center gap-3 border-t border-white/[0.04] pt-3">
              <button
                onClick={addPrompt}
                className="flex items-center gap-1 text-[11px] text-white/30 hover:text-white/50 transition"
              >
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
                Add prompt
              </button>
              <span className="text-white/10">|</span>
              <button
                onClick={() => setShowBulkImport(!showBulkImport)}
                className={`flex items-center gap-1 text-[11px] transition ${showBulkImport ? "text-violet-400/70" : "text-white/30 hover:text-white/50"}`}
              >
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                Bulk import
              </button>
              {isMultiPrompt && (
                <>
                  <span className="text-white/10">|</span>
                  <button
                    onClick={() => setPrompts([{ id: makePromptId(), text: "" }])}
                    className="text-[11px] text-white/20 hover:text-red-400/60 transition"
                  >
                    Clear all
                  </button>
                </>
              )}
            </div>

            {/* Bulk import panel */}
            {showBulkImport && (
              <div className="mt-3 border-t border-white/[0.04] pt-3">
                <p className="mb-2 text-[10px] text-white/25 leading-relaxed">
                  Paste markdown: <code className="rounded bg-white/[0.06] px-1 py-0.5 text-violet-300/60">#### filename.png</code> headings
                  + prompts in <code className="rounded bg-white/[0.06] px-1 py-0.5 text-violet-300/60">```code blocks```</code>
                </p>
                <textarea
                  value={bulkText}
                  onChange={(e) => setBulkText(e.target.value)}
                  placeholder={`#### 1. my_image.png\n\`\`\`\nA cute fluffy 3D lion cub sitting alone...\n\`\`\``}
                  rows={6}
                  className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[11px] text-white/60 placeholder-white/15 outline-none transition focus:border-violet-500/30 resize-y font-mono leading-relaxed"
                />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-[10px] text-white/25">
                    {bulkParseCount !== null && (
                      bulkParseCount > 0
                        ? <span className="text-emerald-400/60">Found {bulkParseCount} prompt{bulkParseCount !== 1 ? "s" : ""}</span>
                        : <span className="text-amber-400/50">No prompts found</span>
                    )}
                  </span>
                  <button
                    onClick={importBulk}
                    disabled={!bulkParseCount}
                    className="rounded-lg bg-violet-500/15 px-3 py-1.5 text-[11px] font-medium text-violet-300 hover:bg-violet-500/25 transition disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    Import {bulkParseCount || 0} prompts
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ═══ Generate Button ═══ */}
          <button
            onClick={generate}
            disabled={validPrompts.length === 0 || loading}
            className={`flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold transition ${
              loading
                ? "bg-violet-600/40 text-white/50 cursor-wait"
                : "bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-500 hover:to-purple-500 active:scale-[0.98] shadow-lg shadow-violet-500/20"
            } disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none`}
          >
            {loading ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                  <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" />
                </svg>
                {batchProgress
                  ? batchProgress.name === "parallel"
                    ? `Generating ${batchProgress.total} images in parallel...`
                    : `${batchProgress.current}/${batchProgress.total}${batchProgress.name ? ` — ${batchProgress.name}` : ""}`
                  : "Generating..."}
              </>
            ) : (
              <>
                Generate{validPrompts.length > 1 ? ` ${validPrompts.length} images` : ""}
                <kbd className="ml-1.5 rounded border border-white/20 px-1.5 py-0.5 text-[10px] font-normal text-white/40">
                  {"\u2318"}Enter
                </kbd>
              </>
            )}
          </button>

          {/* Batch progress bar */}
          {batchProgress && batchProgress.total > 1 && (
            <div className="w-full rounded-full bg-white/[0.04] h-1 overflow-hidden -mt-1">
              <div
                className="h-full bg-gradient-to-r from-violet-500 to-purple-500 transition-all duration-500 ease-out"
                style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
              />
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3.5 py-2.5 text-[12px] text-red-300/80">
              {error}
            </div>
          )}

          {/* ═══ Advanced Options ═══ */}
          <div className="flex flex-col gap-2 mt-1">
            <p className="px-1 text-[10px] font-medium uppercase tracking-widest text-white/15">Advanced</p>

            {/* Base Prompt */}
            <Section title="Base Prompt" defaultOpen={false} badge="optional">
              <textarea
                value={basePrompt}
                onChange={(e) => setBasePrompt(e.target.value)}
                placeholder="Style tags to append to every prompt (e.g. 3D clay render, soft lighting, 4k...)"
                rows={3}
                className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[12px] text-white/70 placeholder-white/20 outline-none transition focus:border-violet-500/30 resize-none leading-relaxed"
              />
              <div className="mt-2 flex items-center justify-between">
                <p className="text-[10px] text-white/20">
                  {basePrompt.trim() ? "Will be appended to each prompt" : "Empty — prompts sent as-is"}
                </p>
                {!basePrompt.trim() ? (
                  <button
                    onClick={() => setBasePrompt(PRESET_BASE_PROMPT)}
                    className="text-[10px] text-violet-400/40 hover:text-violet-400/70 transition"
                  >
                    Load preset
                  </button>
                ) : (
                  <button
                    onClick={() => setBasePrompt("")}
                    className="text-[10px] text-white/25 hover:text-white/40 transition"
                  >
                    Clear
                  </button>
                )}
              </div>
            </Section>

            {/* Negative Prompt */}
            <Section title="Negative Prompt" defaultOpen={false} badge="optional">
              <textarea
                value={negative}
                onChange={(e) => setNegative(e.target.value)}
                placeholder="Things to avoid (e.g. blurry, low quality, text, watermark...)"
                rows={3}
                className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[12px] text-white/70 placeholder-white/20 outline-none transition focus:border-violet-500/30 resize-none leading-relaxed"
              />
              <div className="mt-2 flex items-center justify-between">
                <p className="text-[10px] text-white/20">
                  {negative.trim() ? "Will be sent as negative prompt" : "Empty — no negative prompt"}
                </p>
                {!negative.trim() ? (
                  <button
                    onClick={() => setNegative(PRESET_NEGATIVE)}
                    className="text-[10px] text-violet-400/40 hover:text-violet-400/70 transition"
                  >
                    Load preset
                  </button>
                ) : (
                  <button
                    onClick={() => setNegative("")}
                    className="text-[10px] text-white/25 hover:text-white/40 transition"
                  >
                    Clear
                  </button>
                )}
              </div>
            </Section>

            {/* Settings */}
            <Section title="Settings" defaultOpen={false}>
              <div className="flex flex-col gap-4">
                {/* Model Mode — 2x2 grid */}
                <div>
                  <label className="mb-1.5 block text-[11px] font-medium text-white/40">Model</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    {/* Lightning */}
                    <button
                      onClick={() => { setModelMode("lightning"); setGuidanceScale(0); setSteps(4); }}
                      className={`flex flex-col gap-0.5 rounded-lg border p-2 text-left transition ${
                        modelMode === "lightning" ? "border-violet-500/50 bg-violet-500/[0.08]" : "border-white/[0.06] hover:border-white/[0.1]"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <svg className={`h-3 w-3 ${modelMode === "lightning" ? "text-amber-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                        </svg>
                        <span className={`text-[10px] font-semibold ${modelMode === "lightning" ? "text-white/80" : "text-white/40"}`}>
                          Lightning
                        </span>
                      </div>
                      <span className="text-[9px] text-white/20 pl-[18px]">4 steps, ~3-5s</span>
                    </button>

                    {/* RealVis Fast */}
                    <button
                      onClick={() => { setModelMode("realvis_fast"); setGuidanceScale(1.5); setSteps(5); }}
                      className={`flex flex-col gap-0.5 rounded-lg border p-2 text-left transition ${
                        modelMode === "realvis_fast" ? "border-violet-500/50 bg-violet-500/[0.08]" : "border-white/[0.06] hover:border-white/[0.1]"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <svg className={`h-3 w-3 ${modelMode === "realvis_fast" ? "text-orange-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
                          <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" />
                        </svg>
                        <span className={`text-[10px] font-semibold ${modelMode === "realvis_fast" ? "text-white/80" : "text-white/40"}`}>
                          RealVis Fast
                        </span>
                      </div>
                      <span className="text-[9px] text-white/20 pl-[18px]">5 steps, ~5-10s</span>
                    </button>

                    {/* RealVis Quality */}
                    <button
                      onClick={() => { setModelMode("realvis_quality"); setGuidanceScale(5); setSteps(25); }}
                      className={`flex flex-col gap-0.5 rounded-lg border p-2 text-left transition ${
                        modelMode === "realvis_quality" ? "border-violet-500/50 bg-violet-500/[0.08]" : "border-white/[0.06] hover:border-white/[0.1]"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <svg className={`h-3 w-3 ${modelMode === "realvis_quality" ? "text-emerald-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                        </svg>
                        <span className={`text-[10px] font-semibold ${modelMode === "realvis_quality" ? "text-white/80" : "text-white/40"}`}>
                          RealVis HD
                        </span>
                      </div>
                      <span className="text-[9px] text-white/20 pl-[18px]">25 steps, ~30-60s</span>
                    </button>

                    {/* FLUX */}
                    <button
                      onClick={() => { setModelMode("flux"); setGuidanceScale(0); setSteps(4); }}
                      className={`flex flex-col gap-0.5 rounded-lg border p-2 text-left transition ${
                        modelMode === "flux" ? "border-violet-500/50 bg-violet-500/[0.08]" : "border-white/[0.06] hover:border-white/[0.1]"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <svg className={`h-3 w-3 ${modelMode === "flux" ? "text-sky-400" : "text-white/25"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
                        </svg>
                        <span className={`text-[10px] font-semibold ${modelMode === "flux" ? "text-white/80" : "text-white/40"}`}>
                          FLUX.1
                        </span>
                      </div>
                      <span className="text-[9px] text-white/20 pl-[18px]">4 steps, ~60-90s</span>
                    </button>
                  </div>
                </div>

                {/* Aspect Ratio */}
                <div>
                  <label className="mb-1.5 block text-[11px] font-medium text-white/40">Aspect Ratio</label>
                  <div className="grid grid-cols-5 gap-1.5">
                    {(Object.entries(ASPECT_RATIOS) as [AspectRatio, typeof ASPECT_RATIOS["1:1"]][]).map(
                      ([key, val]) => (
                        <button
                          key={key}
                          onClick={() => setAspect(key)}
                          className={`flex flex-col items-center gap-0.5 rounded-lg border px-1 py-1.5 text-[10px] transition ${
                            aspect === key
                              ? "border-violet-500/50 bg-violet-500/[0.08] text-white/70"
                              : "border-white/[0.06] text-white/35 hover:border-white/[0.1] hover:text-white/50"
                          }`}
                        >
                          <div
                            className={`rounded-sm border ${
                              aspect === key ? "border-violet-400/60" : "border-white/20"
                            }`}
                            style={{
                              width: key === "9:16" || key === "3:4" ? 10 : 16,
                              height: key === "16:9" || key === "4:3" ? 10 : 16,
                            }}
                          />
                          <span>{val.label}</span>
                        </button>
                      ),
                    )}
                  </div>
                  <p className="mt-1 text-[10px] text-white/15">
                    {ASPECT_RATIOS[aspect].w} x {ASPECT_RATIOS[aspect].h}
                  </p>
                </div>

                {/* CFG Scale — hidden for flux/lightning (always 0) */}
                {(modelMode === "realvis_fast" || modelMode === "realvis_quality") && (
                  <div>
                    <div className="mb-1.5 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">CFG Scale</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{guidanceScale}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={20}
                      step={0.5}
                      value={guidanceScale}
                      onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                      className="w-full accent-violet-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-violet-500 [&::-webkit-slider-thumb]:cursor-pointer"
                    />
                    <div className="mt-0.5 flex justify-between text-[9px] text-white/15">
                      <span>0</span>
                      <span>20</span>
                    </div>
                  </div>
                )}

                {/* Steps — hidden for distilled models (fixed steps) */}
                {modelMode === "realvis_quality" && (
                  <div>
                    <div className="mb-1.5 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Steps</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{steps}</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={50}
                      step={1}
                      value={steps}
                      onChange={(e) => setSteps(parseInt(e.target.value))}
                      className="w-full accent-violet-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-violet-500 [&::-webkit-slider-thumb]:cursor-pointer"
                    />
                    <div className="mt-0.5 flex justify-between text-[9px] text-white/15">
                      <span>10</span>
                      <span>50</span>
                    </div>
                  </div>
                )}

                {/* Info for distilled models */}
                {(modelMode === "lightning" || modelMode === "realvis_fast" || modelMode === "flux") && (
                  <p className="text-[10px] text-white/20">
                    {modelMode === "lightning" && "Lightning uses fixed 4 steps, CFG 0. Settings are locked for optimal output."}
                    {modelMode === "realvis_fast" && "RealVis Fast uses fixed 5 steps, CFG 1.5. Distilled for speed."}
                    {modelMode === "flux" && "FLUX.1 Schnell uses fixed 4 steps, CFG 0. Best quality, different architecture."}
                  </p>
                )}

                {/* Seed */}
                <div>
                  <label className="mb-1.5 block text-[11px] font-medium text-white/40">Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    placeholder="Random"
                    className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-violet-500/30"
                  />
                </div>

                {/* Reset */}
                <button
                  onClick={() => {
                    const defaults: Record<ModelMode, { cfg: number; steps: number }> = {
                      lightning: { cfg: 0, steps: 4 },
                      realvis_fast: { cfg: 1.5, steps: 5 },
                      realvis_quality: { cfg: 5, steps: 25 },
                      flux: { cfg: 0, steps: 4 },
                    };
                    const d = defaults[modelMode];
                    setGuidanceScale(d.cfg);
                    setSteps(d.steps);
                    setSeed("");
                    setAspect("1:1");
                  }}
                  className="self-end text-[10px] text-white/20 hover:text-white/40 transition"
                >
                  Reset defaults
                </button>
              </div>
            </Section>
          </div>

          {/* ═══ History ═══ */}
          {images.length > 0 && (
            <div className="mt-2">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-medium text-white/35">
                  History ({images.length})
                </span>
                <div className="flex items-center gap-1.5">
                  {lastBatchIds.size > 1 && (
                    <button
                      onClick={downloadLastBatch}
                      className="flex items-center gap-1 rounded-md bg-white/[0.04] px-2 py-1 text-[10px] text-white/35 hover:text-white/50 transition"
                      title="Download last batch as ZIP"
                    >
                      <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                      </svg>
                      Batch
                    </button>
                  )}
                  {images.length > 1 && (
                    <button
                      onClick={downloadAllImages}
                      className="flex items-center gap-1 rounded-md bg-white/[0.04] px-2 py-1 text-[10px] text-white/35 hover:text-white/50 transition"
                      title="Download all images as ZIP"
                    >
                      <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0l-3-3m3 3l3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" />
                      </svg>
                      All
                    </button>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-4 gap-1.5 max-h-[280px] overflow-y-auto pr-1">
                {images.map((img) => (
                  <button
                    key={img.id}
                    onClick={() => setSelectedImage(img)}
                    className={`relative aspect-square overflow-hidden rounded-lg border transition ${
                      selectedImage?.id === img.id
                        ? "border-violet-500/60 ring-1 ring-violet-500/30"
                        : lastBatchIds.has(img.id)
                          ? "border-violet-500/20 hover:border-violet-500/40"
                          : "border-white/[0.06] hover:border-white/[0.12]"
                    }`}
                  >
                    <img
                      src={getImageSrc(img)}
                      alt={img.prompt}
                      className="h-full w-full object-cover"
                    />
                    {lastBatchIds.has(img.id) && lastBatchIds.size > 1 && (
                      <div className="absolute top-0.5 right-0.5 h-1.5 w-1.5 rounded-full bg-violet-400" />
                    )}
                    {img.filename && (
                      <div className="absolute bottom-0 inset-x-0 bg-black/60 px-1 py-0.5">
                        <p className="text-[7px] text-white/70 truncate font-mono">{img.filename}</p>
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right: Image Display ────────────────────────────── */}
        <div className="flex flex-1 flex-col">
          {selectedImage ? (
            <div className="flex flex-1 flex-col">
              <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01]">
                <img
                  src={getImageSrc(selectedImage)}
                  alt={selectedImage.prompt}
                  className="max-h-full max-w-full object-contain animate-fade-in"
                />
              </div>

              <div className="mt-3 flex items-center justify-between">
                <div className="flex items-center gap-3 text-[11px] text-white/30">
                  {selectedImage.filename && (
                    <span className="font-mono text-violet-300/50">{selectedImage.filename}</span>
                  )}
                  <span>{selectedImage.width}x{selectedImage.height}</span>
                  <span>Seed {selectedImage.seed}</span>
                  <span>{selectedImage.time}s</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setPrompts([{ id: makePromptId(), text: selectedImage.prompt, filename: selectedImage.filename }]);
                      setSeed(String(selectedImage.seed));
                    }}
                    className="rounded-lg border border-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/35 hover:text-white/55 hover:border-white/[0.1] transition"
                  >
                    Reuse
                  </button>
                  <button
                    onClick={() => downloadImage(selectedImage)}
                    className="rounded-lg bg-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition"
                  >
                    Download
                  </button>
                </div>
              </div>

              <p className="mt-1.5 text-[11px] text-white/20 line-clamp-2">
                {selectedImage.prompt}
              </p>
            </div>
          ) : (
            <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${
              loading ? "border-violet-500/30" : "border-white/[0.06]"
            }`}>
              {loading ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="h-10 w-10 rounded-full border-2 border-violet-500/60 border-t-transparent animate-spin" />
                  <p className="text-[13px] text-white/30">
                    {batchProgress && batchProgress.total > 1
                      ? batchProgress.name === "parallel"
                        ? `Generating ${batchProgress.total} images in parallel across GPUs...`
                        : `Generating ${batchProgress.current} of ${batchProgress.total}${batchProgress.name ? ` — ${batchProgress.name}` : ""}...`
                      : "Generating..."}
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-white/15">
                  <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                  </svg>
                  <p className="text-[13px] text-white/25">Your generated images will appear here</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
