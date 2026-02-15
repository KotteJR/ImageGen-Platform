"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import JSZip from "jszip";

/* ══════════════════════════════════════════════════════════════════════
   Types
   ══════════════════════════════════════════════════════════════════════ */

type ActiveTab = "image" | "video" | "3d" | "tts" | "professional";
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

interface TTSVoice {
  id: string;
  name: string;
  filename: string;
}

interface TTSLine {
  id: string;
  text: string;
  voice: string;
  label: string;
}

interface GeneratedTTS {
  id: string;
  text: string;
  voice: string;
  audio: string; // base64 WAV
  duration: number;
  time: number;
  label?: string;
}

interface GeneratedTTSBatch {
  id: string;
  items: GeneratedTTS[];
  combinedAudio?: string; // base64 WAV
  combinedDuration?: number;
  time: number;
}

type ProCategory = "infographic" | "flowchart" | "chart" | "table" | "diagram" | "presentation" | "dashboard" | "org_chart";
type ProStyle = "corporate" | "minimalist" | "colorful" | "dark" | "pastel" | "blueprint" | "neon" | "flat" | "gradient";
type ProDetail = "low" | "medium" | "high";

interface ProCategoryInfo {
  key: ProCategory;
  label: string;
  icon: string;
  subTypes: { key: string; label: string }[];
}

interface GeneratedPro {
  id: string;
  content: string;
  category: string;
  subType: string;
  style: string;
  image: string; // base64
  seed: number;
  width: number;
  height: number;
  time: number;
  promptUsed: string;
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
    assigned_models: string[];
    loaded_models: string[];
    offloaded_models?: string[];
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

const PRO_CATEGORIES: ProCategoryInfo[] = [
  { key: "infographic", label: "Infographic", icon: "M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z",
    subTypes: [
      { key: "data_overview", label: "Data Overview" }, { key: "process", label: "Process" },
      { key: "comparison", label: "Comparison" }, { key: "timeline", label: "Timeline" },
      { key: "statistical", label: "Statistical" }, { key: "list", label: "List" },
      { key: "geographic", label: "Geographic" }, { key: "hierarchical", label: "Hierarchical" },
    ],
  },
  { key: "flowchart", label: "Flowchart", icon: "M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5",
    subTypes: [
      { key: "process_flow", label: "Process Flow" }, { key: "decision_tree", label: "Decision Tree" },
      { key: "swimlane", label: "Swimlane" }, { key: "system_flow", label: "System Flow" },
      { key: "workflow", label: "Workflow" }, { key: "algorithm", label: "Algorithm" },
    ],
  },
  { key: "chart", label: "Chart / Graph", icon: "M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z M13.5 3.75a5.25 5.25 0 015.25 5.25H13.5V3.75z",
    subTypes: [
      { key: "bar_chart", label: "Bar Chart" }, { key: "pie_chart", label: "Pie Chart" },
      { key: "line_graph", label: "Line Graph" }, { key: "scatter_plot", label: "Scatter Plot" },
      { key: "area_chart", label: "Area Chart" }, { key: "donut_chart", label: "Donut Chart" },
      { key: "radar_chart", label: "Radar Chart" }, { key: "waterfall", label: "Waterfall" },
    ],
  },
  { key: "table", label: "Table / Matrix", icon: "M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0112 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M3.375 8.25c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m17.25-3.75h-7.5c-.621 0-1.125.504-1.125 1.125m8.625-1.125c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M12 10.875v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125M13.125 12h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125M20.625 12c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5M12 14.625v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 14.625c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m0 0v1.5c0 .621-.504 1.125-1.125 1.125",
    subTypes: [
      { key: "data_table", label: "Data Table" }, { key: "comparison_matrix", label: "Comparison Matrix" },
      { key: "feature_matrix", label: "Feature Matrix" }, { key: "pricing_table", label: "Pricing Table" },
      { key: "schedule", label: "Schedule" }, { key: "scorecard", label: "Scorecard" },
    ],
  },
  { key: "diagram", label: "Diagram", icon: "M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z M15 12a3 3 0 11-6 0 3 3 0 016 0z",
    subTypes: [
      { key: "architecture", label: "Architecture" }, { key: "network", label: "Network" },
      { key: "er_diagram", label: "ER Diagram" }, { key: "uml", label: "UML" },
      { key: "venn", label: "Venn" }, { key: "cycle", label: "Cycle" },
      { key: "block_diagram", label: "Block Diagram" }, { key: "mind_map", label: "Mind Map" },
    ],
  },
  { key: "presentation", label: "Slide", icon: "M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6",
    subTypes: [
      { key: "title_slide", label: "Title Slide" }, { key: "key_metrics", label: "Key Metrics" },
      { key: "bullet_points", label: "Bullet Points" }, { key: "quote_slide", label: "Quote" },
      { key: "team_slide", label: "Team" }, { key: "roadmap", label: "Roadmap" },
      { key: "swot", label: "SWOT Analysis" },
    ],
  },
  { key: "dashboard", label: "Dashboard", icon: "M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z",
    subTypes: [
      { key: "kpi_dashboard", label: "KPI Dashboard" }, { key: "analytics", label: "Analytics" },
      { key: "sales_dashboard", label: "Sales" }, { key: "project_status", label: "Project Status" },
      { key: "financial", label: "Financial" }, { key: "marketing", label: "Marketing" },
    ],
  },
  { key: "org_chart", label: "Org Chart", icon: "M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z",
    subTypes: [
      { key: "corporate", label: "Corporate" }, { key: "team_structure", label: "Team Structure" },
      { key: "project_org", label: "Project Org" }, { key: "flat_org", label: "Flat Org" },
      { key: "matrix_org", label: "Matrix" },
    ],
  },
];

const PRO_STYLES: { key: ProStyle; label: string; preview: string }[] = [
  { key: "corporate", label: "Corporate", preview: "bg-gradient-to-br from-blue-900 to-slate-700" },
  { key: "minimalist", label: "Minimal", preview: "bg-gradient-to-br from-gray-100 to-white" },
  { key: "colorful", label: "Colorful", preview: "bg-gradient-to-br from-rose-500 to-amber-400" },
  { key: "dark", label: "Dark", preview: "bg-gradient-to-br from-gray-900 to-zinc-800" },
  { key: "pastel", label: "Pastel", preview: "bg-gradient-to-br from-pink-200 to-blue-200" },
  { key: "blueprint", label: "Blueprint", preview: "bg-gradient-to-br from-blue-900 to-cyan-700" },
  { key: "neon", label: "Neon", preview: "bg-gradient-to-br from-purple-800 to-pink-600" },
  { key: "flat", label: "Flat", preview: "bg-gradient-to-br from-teal-500 to-emerald-400" },
  { key: "gradient", label: "Gradient", preview: "bg-gradient-to-br from-violet-500 to-fuchsia-500" },
];

const PRO_ASPECT_RATIOS: { key: string; w: number; h: number; label: string }[] = [
  { key: "1:1", w: 1024, h: 1024, label: "Square" },
  { key: "16:9", w: 1344, h: 768, label: "Wide" },
  { key: "9:16", w: 768, h: 1344, label: "Tall" },
  { key: "4:3", w: 1152, h: 896, label: "Standard" },
  { key: "3:2", w: 1216, h: 832, label: "Photo" },
];

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
  const [showModelsPanel, setShowModelsPanel] = useState(false);
  const [gpuStats, setGpuStats] = useState<GpuStats | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const gpuPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const gpuStatsRef = useRef<GpuStats | null>(null);
  useEffect(() => { gpuStatsRef.current = gpuStats; }, [gpuStats]);

  const fetchGpuStats = useCallback(async () => {
    try {
      const res = await fetch("/api/gpu/stats");
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        // Only show error if we don't already have stats (transient failures are fine)
        if (!gpuStatsRef.current) {
          if (res.status === 404) {
            setGpuError("GPU stats endpoint not available — server needs the latest server-cuda.py");
          } else {
            setGpuError(data?.error || "Backend unreachable");
          }
        }
        return;
      }
      const data = await res.json();
      if (data.error) {
        if (!gpuStatsRef.current) { setGpuError(data.error); setGpuStats(null); }
      } else {
        setGpuStats(data);
        setGpuError(null);
      }
    } catch {
      if (!gpuStatsRef.current) setGpuError("Connection failed");
    }
  }, []);

  const gpuErrorCountRef = useRef(0);

  const fetchGpuStatsWrapped = useCallback(async () => {
    // If too many consecutive errors, slow down polling
    if (gpuErrorCountRef.current > 3) return;
    await fetchGpuStats();
  }, [fetchGpuStats]);

  useEffect(() => {
    if (gpuError) gpuErrorCountRef.current++;
    else gpuErrorCountRef.current = 0;
  }, [gpuError]);

  useEffect(() => {
    if ((showGpuPanel || showModelsPanel) && backendStatus === "online") {
      gpuErrorCountRef.current = 0;
      fetchGpuStats();
      gpuPollRef.current = setInterval(fetchGpuStatsWrapped, 3000);
    }
    return () => { if (gpuPollRef.current) clearInterval(gpuPollRef.current); };
  }, [showGpuPanel, showModelsPanel, backendStatus, fetchGpuStats, fetchGpuStatsWrapped]);

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

  /* ═══════════════════════════════════════════════════════════════
     TTS TAB STATE
     ═══════════════════════════════════════════════════════════════ */
  const [ttsLines, setTtsLines] = useState<TTSLine[]>([{ id: makeId(), text: "", voice: "default", label: "Speaker 1" }]);
  const [ttsVoices, setTtsVoices] = useState<TTSVoice[]>([]);
  const [ttsVoicesLoaded, setTtsVoicesLoaded] = useState(false);
  const [ttsExaggeration, setTtsExaggeration] = useState(0.35);
  const [ttsCfgWeight, setTtsCfgWeight] = useState(0.5);
  const [ttsTemperature, setTtsTemperature] = useState(0.65);
  const [ttsRepPenalty, setTtsRepPenalty] = useState(1.35);
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [ttsCombine, setTtsCombine] = useState(false);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [ttsElapsed, setTtsElapsed] = useState(0);
  const [ttsResults, setTtsResults] = useState<GeneratedTTS[]>([]);
  const [ttsBatchResults, setTtsBatchResults] = useState<GeneratedTTSBatch[]>([]);
  const [selectedTts, setSelectedTts] = useState<GeneratedTTS | null>(null);
  const [selectedBatch, setSelectedBatch] = useState<GeneratedTTSBatch | null>(null);
  const [ttsPlayingId, setTtsPlayingId] = useState<string | null>(null);
  const ttsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);

  /* ═══════════════════════════════════════════════════════════════
     PROFESSIONAL TAB STATE
     ═══════════════════════════════════════════════════════════════ */
  const [proCategory, setProCategory] = useState<ProCategory>("infographic");
  const [proSubType, setProSubType] = useState("");
  const [proContent, setProContent] = useState("");
  const [proStyle, setProStyle] = useState<ProStyle>("corporate");
  const [proColorScheme, setProColorScheme] = useState("");
  const [proAspect, setProAspect] = useState("1:1");
  const [proDetail, setProDetail] = useState<ProDetail>("high");
  const [proSeed, setProSeed] = useState("");
  const [proLoading, setProLoading] = useState(false);
  const [proError, setProError] = useState<string | null>(null);
  const [proResults, setProResults] = useState<GeneratedPro[]>([]);
  const [selectedPro, setSelectedPro] = useState<GeneratedPro | null>(null);

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
     TTS TAB LOGIC
     ═══════════════════════════════════════════════════════════════ */

  // Fetch voices on mount / when switching to TTS tab
  useEffect(() => {
    if (activeTab !== "tts" || ttsVoicesLoaded) return;
    (async () => {
      try {
        const res = await fetch("/api/tts/voices");
        if (res.ok) {
          const data = await res.json();
          setTtsVoices(data.voices || []);
        }
      } catch { /* ignore */ }
      setTtsVoicesLoaded(true);
    })();
  }, [activeTab, ttsVoicesLoaded]);

  const addTtsLine = useCallback(() => {
    setTtsLines(prev => [...prev, { id: makeId(), text: "", voice: "default", label: `Speaker ${prev.length + 1}` }]);
  }, []);

  const removeTtsLine = useCallback((id: string) => {
    setTtsLines(prev => prev.length <= 1 ? prev : prev.filter(l => l.id !== id));
  }, []);

  const updateTtsLine = useCallback((id: string, field: keyof TTSLine, value: string) => {
    setTtsLines(prev => prev.map(l => l.id === id ? { ...l, [field]: value } : l));
  }, []);

  const playTtsAudio = useCallback((audio: string, id: string) => {
    // Stop current
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current = null;
    }
    if (ttsPlayingId === id) {
      setTtsPlayingId(null);
      return;
    }
    const blob = base64ToBlob(audio, "audio/wav");
    const url = URL.createObjectURL(blob);
    const a = new Audio(url);
    a.onended = () => { setTtsPlayingId(null); URL.revokeObjectURL(url); };
    a.play();
    ttsAudioRef.current = a;
    setTtsPlayingId(id);
  }, [ttsPlayingId]);

  const downloadTtsAudio = useCallback((audio: string, filename: string) => {
    const blob = base64ToBlob(audio, "audio/wav");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  }, []);

  const generateTts = useCallback(async () => {
    const validLines = ttsLines.filter(l => l.text.trim().length > 0);
    if (validLines.length === 0 || ttsLoading) return;
    setTtsLoading(true);
    setTtsError(null);
    setTtsElapsed(0);
    ttsTimerRef.current = setInterval(() => setTtsElapsed(prev => prev + 1), 1000);

    try {
      if (validLines.length === 1) {
        // Single generation
        const line = validLines[0];
        const res = await fetch("/api/tts/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: line.text,
            voice: line.voice,
            exaggeration: ttsExaggeration,
            cfg_weight: ttsCfgWeight,
            temperature: ttsTemperature,
            repetition_penalty: ttsRepPenalty,
            speed: ttsSpeed,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: "Unknown error" }));
          throw new Error(err.error || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const result: GeneratedTTS = {
          id: makeId(),
          text: line.text,
          voice: data.voice,
          audio: data.audio,
          duration: data.duration,
          time: data.time_seconds,
          label: line.label,
        };
        setTtsResults(prev => [result, ...prev]);
        setSelectedTts(result);
        setSelectedBatch(null);
      } else {
        // Batch generation
        const res = await fetch("/api/tts/generate/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            items: validLines.map(l => ({ text: l.text, voice: l.voice, label: l.label })),
            exaggeration: ttsExaggeration,
            cfg_weight: ttsCfgWeight,
            temperature: ttsTemperature,
            repetition_penalty: ttsRepPenalty,
            speed: ttsSpeed,
            combine: ttsCombine,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: "Unknown error" }));
          throw new Error(err.error || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const items: GeneratedTTS[] = (data.results || [])
          .filter((r: { success: boolean }) => r.success)
          .map((r: { audio: string; duration: number; voice: string; time_seconds: number; label: string; index: number }) => ({
            id: makeId(),
            text: validLines[r.index]?.text || "",
            voice: r.voice,
            audio: r.audio,
            duration: r.duration,
            time: r.time_seconds,
            label: r.label || validLines[r.index]?.label,
          }));
        const batch: GeneratedTTSBatch = {
          id: makeId(),
          items,
          combinedAudio: data.combined_audio || undefined,
          combinedDuration: data.combined_duration || undefined,
          time: data.total_time_seconds,
        };
        setTtsBatchResults(prev => [batch, ...prev]);
        setTtsResults(prev => [...items, ...prev]);
        setSelectedBatch(batch);
        setSelectedTts(null);
      }
    } catch (e) {
      setTtsError(e instanceof Error ? e.message : "TTS generation failed");
    } finally {
      setTtsLoading(false);
      if (ttsTimerRef.current) clearInterval(ttsTimerRef.current);
    }
  }, [ttsLines, ttsLoading, ttsExaggeration, ttsCfgWeight, ttsTemperature, ttsRepPenalty, ttsSpeed, ttsCombine]);

  /* ═══════════════════════════════════════════════════════════════
     PROFESSIONAL TAB LOGIC
     ═══════════════════════════════════════════════════════════════ */

  // Reset sub_type when category changes
  useEffect(() => {
    const cat = PRO_CATEGORIES.find(c => c.key === proCategory);
    if (cat && cat.subTypes.length > 0) {
      setProSubType(cat.subTypes[0].key);
    }
  }, [proCategory]);

  const currentProCategory = PRO_CATEGORIES.find(c => c.key === proCategory);

  const generateProfessional = useCallback(async () => {
    if (!proContent.trim() || proLoading) return;
    setProLoading(true);
    setProError(null);

    const aspectInfo = PRO_ASPECT_RATIOS.find(a => a.key === proAspect) || PRO_ASPECT_RATIOS[0];

    try {
      const res = await fetch("/api/generate/professional", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: proCategory,
          sub_type: proSubType,
          content: proContent,
          style: proStyle,
          color_scheme: proColorScheme || undefined,
          width: aspectInfo.w,
          height: aspectInfo.h,
          seed: proSeed ? parseInt(proSeed) : null,
          detail_level: proDetail,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Unknown error" }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const result: GeneratedPro = {
        id: makeId(),
        content: proContent,
        category: data.category,
        subType: data.sub_type,
        style: proStyle,
        image: data.image,
        seed: data.seed,
        width: data.width,
        height: data.height,
        time: data.time_seconds,
        promptUsed: data.prompt_used,
      };
      setProResults(prev => [result, ...prev]);
      setSelectedPro(result);
    } catch (e) {
      setProError(e instanceof Error ? e.message : "Professional generation failed");
    } finally {
      setProLoading(false);
    }
  }, [proContent, proCategory, proSubType, proStyle, proColorScheme, proAspect, proDetail, proSeed, proLoading]);

  const downloadPro = useCallback((pro: GeneratedPro) => {
    const blob = base64ToBlob(pro.image, "image/png");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${pro.category}_${pro.subType}_${pro.id}.png`;
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
      else if (activeTab === "tts") generateTts();
      else if (activeTab === "professional") generateProfessional();
    }
  };

  /* ═══════════════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════════════ */

  const tabAccent = activeTab === "image" ? "violet" : activeTab === "video" ? "blue" : activeTab === "tts" ? "amber" : activeTab === "professional" ? "rose" : "emerald";

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
            {/* Models Status Button */}
            <button
              onClick={() => { setShowModelsPanel(!showModelsPanel); if (showGpuPanel) setShowGpuPanel(false); }}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
                showModelsPanel
                  ? "bg-violet-500/20 text-violet-300"
                  : "bg-white/[0.04] text-white/30 hover:bg-white/[0.08] hover:text-white/50"
              }`}
              title="Model Loading Status"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
              </svg>
              Models
              {gpuStats && (
                <span className="rounded bg-white/[0.06] px-1 py-0.5 text-[9px] text-white/25">
                  {gpuStats.gpus.reduce((sum, g) => sum + (g.slot?.loaded_models?.length ?? 0), 0)}
                </span>
              )}
            </button>

            {/* GPU Monitor Button */}
            <button
              onClick={() => { setShowGpuPanel(!showGpuPanel); if (showModelsPanel) setShowModelsPanel(false); }}
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

      {/* ── Models Status Panel ──────────────────────────────────── */}
      {showModelsPanel && (
        <div className="border-b border-white/[0.06] bg-[#0c0c14] px-6 py-4 animate-fade-in">
          <div className="mx-auto max-w-7xl">
            {!gpuStats ? (
              <div className="flex items-center gap-2 text-[12px] text-white/20">
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-white/10 border-t-violet-400" />
                Loading model status...
              </div>
            ) : (
              <>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-white/70">Model Status</h3>
                  <span className="text-[10px] text-white/25">
                    {gpuStats.gpus.reduce((sum, g) => sum + (g.slot?.loaded_models?.length ?? 0), 0)} models loaded across {gpuStats.summary.gpu_count} GPUs
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {gpuStats.gpus.map((gpu) => {
                    const assigned = gpu.slot?.assigned_models ?? [];
                    const loaded = gpu.slot?.loaded_models ?? [];
                    const isActive = gpu.slot?.active_task != null;
                    const allLoaded = assigned.length > 0 && assigned.every(m => loaded.includes(m));

                    return (
                      <div
                        key={gpu.index}
                        className={`rounded-lg border p-3 ${
                          isActive
                            ? "border-cyan-500/30 bg-cyan-500/[0.04]"
                            : allLoaded
                              ? "border-emerald-500/20 bg-emerald-500/[0.03]"
                              : assigned.length > 0
                                ? "border-amber-500/20 bg-amber-500/[0.03]"
                                : "border-white/[0.06] bg-white/[0.015]"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[11px] font-medium text-white/50">GPU {gpu.index}</span>
                          {isActive ? (
                            <span className="rounded-full bg-cyan-500/20 px-1.5 py-0.5 text-[8px] text-cyan-400 animate-pulse">WORKING</span>
                          ) : allLoaded ? (
                            <span className="rounded-full bg-emerald-500/20 px-1.5 py-0.5 text-[8px] text-emerald-400">READY</span>
                          ) : assigned.length > 0 ? (
                            <span className="rounded-full bg-amber-500/20 px-1.5 py-0.5 text-[8px] text-amber-400">LOADING</span>
                          ) : (
                            <span className="rounded-full bg-white/[0.06] px-1.5 py-0.5 text-[8px] text-white/20">IDLE</span>
                          )}
                        </div>

                        {assigned.length > 0 ? (
                          <div className="space-y-1">
                            {assigned.map((model) => {
                              const isLoaded = loaded.includes(model);
                              const isOffloaded = gpu.slot?.offloaded_models?.includes(model) ?? false;
                              const isRunning = gpu.slot?.active_task === model;
                              return (
                                <div key={model} className="flex items-center gap-1.5">
                                  <span className={`inline-block h-1.5 w-1.5 rounded-full ${
                                    isRunning ? "bg-cyan-400 animate-pulse" : isLoaded ? (isOffloaded ? "bg-blue-400" : "bg-emerald-400") : "bg-amber-400 animate-pulse"
                                  }`} />
                                  <span className={`text-[10px] ${
                                    isRunning ? "text-cyan-300" : isLoaded ? "text-white/40" : "text-amber-300/60"
                                  }`}>
                                    {model}
                                  </span>
                                  <span className={`text-[8px] ml-auto ${
                                    isRunning ? "text-cyan-400/60" : isLoaded ? (isOffloaded ? "text-blue-400/50" : "text-emerald-400/50") : "text-amber-400/50"
                                  }`}>
                                    {isRunning ? "running" : isLoaded ? (isOffloaded ? "cpu offload" : "on gpu") : "loading..."}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="text-[10px] text-white/15">No model assigned</div>
                        )}

                        {(gpu.slot?.generation_count ?? 0) > 0 && (
                          <div className="mt-1.5 text-[8px] text-white/15">
                            {gpu.slot!.generation_count} images generated
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
                    {(() => {
                      const activeCount = gpuStats.gpus.filter(g => g.slot?.active_task).length;
                      const totalGen = gpuStats.gpus.reduce((sum, g) => sum + (g.slot?.generation_count ?? 0), 0);
                      return (
                        <span className="text-white/25">
                          Active: <span className="text-cyan-400/60">{activeCount}</span>
                          {" / "}
                          <span className="text-white/40">{totalGen}</span> generated
                        </span>
                      );
                    })()}
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
                              GPU {gpu.slot.slot_id} — {gpu.slot.generation_count} generated
                              {gpu.slot.assigned_models?.length > 0 && (
                                <span className="text-white/15"> · Assigned: {gpu.slot.assigned_models.join(", ")}</span>
                              )}
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
            { key: "tts" as const, label: "Voice TTS", icon: (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
            ), color: "amber" },
            { key: "professional" as const, label: "Professional", icon: (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
              </svg>
            ), color: "rose" },
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

          {/* ────────────────────────────────────────────────────────
              TTS TAB CONTROLS
              ──────────────────────────────────────────────────────── */}
          {activeTab === "tts" && (
            <>
              {/* Lines / Prompts */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[12px] font-medium text-white/50">
                    {ttsLines.length > 1 ? `${ttsLines.length} speakers` : "Text to Speech"}
                  </span>
                  <button onClick={addTtsLine}
                    className="rounded-lg border border-white/[0.08] px-2 py-1 text-[10px] text-white/40 hover:text-white/60 hover:border-white/[0.12] transition">
                    + Add Speaker
                  </button>
                </div>

                <div className="flex flex-col gap-3">
                  {ttsLines.map((line, idx) => (
                    <div key={line.id} className="rounded-lg border border-white/[0.06] bg-white/[0.01] p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <input
                          type="text" value={line.label}
                          onChange={(e) => updateTtsLine(line.id, "label", e.target.value)}
                          className="flex-1 rounded-md border border-white/[0.06] bg-transparent px-2 py-1 text-[11px] text-amber-300/70 placeholder-white/20 outline-none focus:border-amber-500/30"
                          placeholder={`Speaker ${idx + 1}`}
                        />
                        <select
                          value={line.voice}
                          onChange={(e) => updateTtsLine(line.id, "voice", e.target.value)}
                          className="rounded-md border border-white/[0.06] bg-[#12121a] px-2 py-1 text-[11px] text-white/60 outline-none focus:border-amber-500/30 cursor-pointer"
                        >
                          <option value="default">Default Voice</option>
                          {ttsVoices.map(v => (
                            <option key={v.id} value={v.id}>{v.name}</option>
                          ))}
                        </select>
                        {ttsLines.length > 1 && (
                          <button onClick={() => removeTtsLine(line.id)}
                            className="rounded p-1 text-white/20 hover:text-red-400/60 hover:bg-red-500/10 transition">
                            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                          </button>
                        )}
                      </div>
                      <textarea
                        value={line.text}
                        onChange={(e) => updateTtsLine(line.id, "text", e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={2}
                        placeholder="Enter text to speak..."
                        className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-[12px] text-white/80 placeholder-white/20 outline-none transition focus:border-amber-500/30 resize-none leading-relaxed"
                      />
                      <div className="flex justify-between mt-1">
                        <span className="text-[9px] text-white/15">{line.text.length}/5000</span>
                      </div>
                    </div>
                  ))}
                </div>

                {ttsLines.length > 1 && (
                  <div className="mt-3 flex items-center gap-2">
                    <button onClick={() => setTtsCombine(!ttsCombine)}
                      className={`relative h-4 w-7 rounded-full transition ${ttsCombine ? "bg-amber-500/60" : "bg-white/[0.08]"}`}>
                      <div className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${ttsCombine ? "left-[14px]" : "left-0.5"}`} />
                    </button>
                    <span className="text-[10px] text-white/35">Combine all into one audio file</span>
                  </div>
                )}
              </div>

              {/* Generate Button */}
              <button
                onClick={generateTts}
                disabled={ttsLoading || ttsLines.every(l => !l.text.trim())}
                className={`w-full rounded-xl py-3 text-[13px] font-semibold transition ${
                  ttsLoading || ttsLines.every(l => !l.text.trim())
                    ? "bg-white/[0.04] text-white/20 cursor-not-allowed"
                    : "bg-gradient-to-r from-amber-600 to-orange-600 text-white hover:from-amber-500 hover:to-orange-500 shadow-lg shadow-amber-500/20"
                }`}
              >
                {ttsLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="h-3.5 w-3.5 rounded-full border-2 border-white/40 border-t-transparent animate-spin" />
                    Generating... {ttsElapsed}s
                  </span>
                ) : (
                  <span>Generate Speech {ttsLines.filter(l => l.text.trim()).length > 1 ? `(${ttsLines.filter(l => l.text.trim()).length} clips)` : ""}</span>
                )}
              </button>

              {ttsError && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/[0.05] px-3 py-2">
                  <p className="text-[11px] text-red-400/80">{ttsError}</p>
                </div>
              )}

              {/* Voice Settings */}
              <div className="flex flex-col gap-2 mt-1">
                <p className="px-1 text-[10px] font-medium uppercase tracking-widest text-white/15">Voice Settings</p>
                <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 flex flex-col gap-3">
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Emotion</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{ttsExaggeration.toFixed(2)}</span>
                    </div>
                    <input type="range" min={0} max={1} step={0.05} value={ttsExaggeration} onChange={(e) => setTtsExaggeration(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                    <div className="flex justify-between text-[8px] text-white/15 mt-0.5"><span>Flat</span><span>Dramatic</span></div>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Speed</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{ttsSpeed.toFixed(1)}x</span>
                    </div>
                    <input type="range" min={0.5} max={2} step={0.1} value={ttsSpeed} onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                    <div className="flex justify-between text-[8px] text-white/15 mt-0.5"><span>0.5x Slow</span><span>2x Fast</span></div>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Temperature</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{ttsTemperature.toFixed(2)}</span>
                    </div>
                    <input type="range" min={0.1} max={1.5} step={0.05} value={ttsTemperature} onChange={(e) => setTtsTemperature(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                    <div className="flex justify-between text-[8px] text-white/15 mt-0.5"><span>Focused</span><span>Creative</span></div>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Voice Guidance</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{ttsCfgWeight.toFixed(1)}</span>
                    </div>
                    <input type="range" min={0} max={1} step={0.1} value={ttsCfgWeight} onChange={(e) => setTtsCfgWeight(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <label className="text-[11px] font-medium text-white/40">Repetition Penalty</label>
                      <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-mono text-white/40">{ttsRepPenalty.toFixed(2)}</span>
                    </div>
                    <input type="range" min={1} max={2} step={0.05} value={ttsRepPenalty} onChange={(e) => setTtsRepPenalty(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1 bg-white/[0.06] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:cursor-pointer" />
                  </div>
                </div>
              </div>

              {/* TTS History */}
              {ttsResults.length > 0 && (
                <div className="mt-2">
                  <span className="text-[11px] font-medium text-white/35">Generated ({ttsResults.length})</span>
                  <div className="mt-2 flex flex-col gap-2 max-h-[250px] overflow-y-auto pr-1">
                    {ttsResults.map((tts) => (
                      <button key={tts.id} onClick={() => { setSelectedTts(tts); setSelectedBatch(null); }}
                        className={`flex items-center gap-3 rounded-lg border p-2 text-left transition ${
                          selectedTts?.id === tts.id ? "border-amber-500/50 bg-amber-500/[0.06]" : "border-white/[0.06] hover:border-white/[0.1]"
                        }`}>
                        <button
                          onClick={(e) => { e.stopPropagation(); playTtsAudio(tts.audio, tts.id); }}
                          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition ${
                            ttsPlayingId === tts.id ? "bg-amber-500/30 text-amber-300" : "bg-amber-500/10 text-amber-400/60 hover:bg-amber-500/20"
                          }`}>
                          {ttsPlayingId === tts.id ? (
                            <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                          ) : (
                            <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5.14v14l11-7-11-7z" /></svg>
                          )}
                        </button>
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] text-white/60 truncate">{tts.label || tts.text.slice(0, 50)}</p>
                          <p className="text-[9px] text-white/25">{tts.duration}s · {tts.voice} · {tts.time}s gen</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ────────────────────────────────────────────────────────
              PROFESSIONAL TAB CONTROLS
              ──────────────────────────────────────────────────────── */}
          {activeTab === "professional" && (
            <>
              {/* Category Grid */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
                <p className="text-[10px] font-medium uppercase tracking-widest text-white/20 mb-2">Category</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {PRO_CATEGORIES.map(cat => (
                    <button key={cat.key} onClick={() => setProCategory(cat.key)}
                      className={`flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition ${
                        proCategory === cat.key
                          ? "border-rose-500/50 bg-rose-500/[0.08] text-white/80"
                          : "border-white/[0.06] text-white/35 hover:border-white/[0.1] hover:text-white/50"
                      }`}>
                      <svg className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d={cat.icon} />
                      </svg>
                      <span className="text-[11px] font-medium">{cat.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Sub-type */}
              {currentProCategory && currentProCategory.subTypes.length > 0 && (
                <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
                  <p className="text-[10px] font-medium uppercase tracking-widest text-white/20 mb-2">Type</p>
                  <div className="flex flex-wrap gap-1.5">
                    {currentProCategory.subTypes.map(st => (
                      <button key={st.key} onClick={() => setProSubType(st.key)}
                        className={`rounded-lg border px-2.5 py-1.5 text-[10px] font-medium transition ${
                          proSubType === st.key
                            ? "border-rose-500/50 bg-rose-500/[0.08] text-rose-300"
                            : "border-white/[0.06] text-white/30 hover:border-white/[0.1] hover:text-white/45"
                        }`}>
                        {st.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Content Prompt */}
              <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                <label className="mb-1.5 block text-[10px] font-medium uppercase tracking-widest text-white/20">Content / Data</label>
                <textarea
                  value={proContent}
                  onChange={(e) => setProContent(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={4}
                  placeholder="Describe the data, topics, or content you want visualized...&#10;&#10;Example: Q1 2025 sales data showing revenue growth of 45%, top 3 products, customer satisfaction at 92%, with regional breakdown for NA, EU, APAC"
                  className="w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-white/90 placeholder-white/20 outline-none transition focus:border-rose-500/40 focus:bg-white/[0.04] resize-none leading-relaxed"
                />
                <p className="mt-1 text-[9px] text-white/15">{proContent.length}/3000 · The more detail you provide, the better the result</p>
              </div>

              {/* Generate Button */}
              <button
                onClick={generateProfessional}
                disabled={proLoading || !proContent.trim()}
                className={`w-full rounded-xl py-3 text-[13px] font-semibold transition ${
                  proLoading || !proContent.trim()
                    ? "bg-white/[0.04] text-white/20 cursor-not-allowed"
                    : "bg-gradient-to-r from-rose-600 to-pink-600 text-white hover:from-rose-500 hover:to-pink-500 shadow-lg shadow-rose-500/20"
                }`}
              >
                {proLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="h-3.5 w-3.5 rounded-full border-2 border-white/40 border-t-transparent animate-spin" />
                    Generating {currentProCategory?.label}...
                  </span>
                ) : (
                  <span>Generate {currentProCategory?.label || "Professional"}</span>
                )}
              </button>

              {proError && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/[0.05] px-3 py-2">
                  <p className="text-[11px] text-red-400/80">{proError}</p>
                </div>
              )}

              {/* Style Selection */}
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                <p className="text-[10px] font-medium uppercase tracking-widest text-white/20 mb-2">Visual Style</p>
                <div className="grid grid-cols-3 gap-1.5">
                  {PRO_STYLES.map(s => (
                    <button key={s.key} onClick={() => setProStyle(s.key)}
                      className={`flex flex-col items-center gap-1 rounded-lg border p-2 transition ${
                        proStyle === s.key
                          ? "border-rose-500/50 bg-rose-500/[0.06]"
                          : "border-white/[0.06] hover:border-white/[0.1]"
                      }`}>
                      <div className={`h-5 w-full rounded ${s.preview}`} />
                      <span className="text-[9px] text-white/40">{s.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Aspect Ratio */}
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                <p className="text-[10px] font-medium uppercase tracking-widest text-white/20 mb-2">Aspect Ratio</p>
                <div className="flex gap-1.5">
                  {PRO_ASPECT_RATIOS.map(a => (
                    <button key={a.key} onClick={() => setProAspect(a.key)}
                      className={`flex-1 flex flex-col items-center gap-0.5 rounded-lg border py-2 text-[10px] transition ${
                        proAspect === a.key
                          ? "border-rose-500/50 bg-rose-500/[0.08] text-white/70"
                          : "border-white/[0.06] text-white/30 hover:border-white/[0.1]"
                      }`}>
                      <span className="font-medium">{a.key}</span>
                      <span className="text-[8px] text-white/20">{a.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Advanced Settings */}
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                <p className="text-[10px] font-medium uppercase tracking-widest text-white/20 mb-2">Options</p>
                <div className="flex flex-col gap-3">
                  <div>
                    <label className="mb-1 block text-[11px] text-white/40">Detail Level</label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["low", "medium", "high"] as ProDetail[]).map(d => (
                        <button key={d} onClick={() => setProDetail(d)}
                          className={`rounded-lg border py-1.5 text-[10px] font-medium transition ${
                            proDetail === d
                              ? "border-rose-500/50 bg-rose-500/[0.08] text-rose-300"
                              : "border-white/[0.06] text-white/30 hover:border-white/[0.1]"
                          }`}>
                          {d.charAt(0).toUpperCase() + d.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="mb-1 block text-[11px] text-white/40">Color Scheme (optional)</label>
                    <input type="text" value={proColorScheme} onChange={(e) => setProColorScheme(e.target.value)}
                      placeholder="e.g. blue and gold, monochrome green..."
                      className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-rose-500/30" />
                  </div>
                  <div>
                    <label className="mb-1 block text-[11px] text-white/40">Seed</label>
                    <input type="number" value={proSeed} onChange={(e) => setProSeed(e.target.value)} placeholder="Random"
                      className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-[12px] text-white/60 placeholder-white/20 outline-none transition focus:border-rose-500/30" />
                  </div>
                </div>
              </div>

              {/* History */}
              {proResults.length > 0 && (
                <div className="mt-1">
                  <span className="text-[11px] font-medium text-white/35">Results ({proResults.length})</span>
                  <div className="mt-2 grid grid-cols-3 gap-1.5 max-h-[200px] overflow-y-auto pr-1">
                    {proResults.map(pro => (
                      <button key={pro.id} onClick={() => setSelectedPro(pro)}
                        className={`relative overflow-hidden rounded-lg border aspect-square transition ${
                          selectedPro?.id === pro.id ? "border-rose-500/50 ring-1 ring-rose-500/30" : "border-white/[0.06] hover:border-white/[0.1]"
                        }`}>
                        <img src={`data:image/png;base64,${pro.image}`} alt="" className="h-full w-full object-cover" />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 p-1">
                          <p className="text-[8px] text-white/70 truncate">{pro.category}</p>
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

          {/* ── TTS Display ──────────────────────────────────────── */}
          {activeTab === "tts" && (
            (selectedTts || selectedBatch) ? (
              <div className="flex flex-1 flex-col animate-fade-in gap-4">
                {/* Single clip view */}
                {selectedTts && !selectedBatch && (
                  <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-white/[0.06] bg-gradient-to-b from-amber-500/[0.02] to-transparent p-8">
                    <div className="flex flex-col items-center gap-5 max-w-lg w-full">
                      {/* Waveform visual */}
                      <div className="flex items-center gap-[3px] h-16">
                        {Array.from({ length: 40 }, (_, i) => (
                          <div key={i}
                            className={`w-1 rounded-full transition-all ${ttsPlayingId === selectedTts.id ? "bg-amber-400/60 animate-pulse" : "bg-amber-400/20"}`}
                            style={{ height: `${12 + Math.sin(i * 0.7) * 20 + Math.random() * 16}px`, animationDelay: `${i * 50}ms` }}
                          />
                        ))}
                      </div>

                      {/* Voice & label */}
                      <div className="text-center">
                        <p className="text-[14px] font-medium text-white/80">{selectedTts.label || "Speech"}</p>
                        <p className="text-[11px] text-amber-300/50 mt-0.5">{selectedTts.voice === "default" ? "Default Voice" : selectedTts.voice}</p>
                      </div>

                      {/* Play controls */}
                      <button
                        onClick={() => playTtsAudio(selectedTts.audio, selectedTts.id)}
                        className={`flex items-center justify-center h-14 w-14 rounded-full transition-all ${
                          ttsPlayingId === selectedTts.id
                            ? "bg-amber-500/30 text-amber-300 ring-2 ring-amber-500/30"
                            : "bg-amber-500/20 text-amber-400 hover:bg-amber-500/30"
                        }`}
                      >
                        {ttsPlayingId === selectedTts.id ? (
                          <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                        ) : (
                          <svg className="h-6 w-6 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5.14v14l11-7-11-7z" /></svg>
                        )}
                      </button>

                      {/* Text preview */}
                      <p className="text-[12px] text-white/40 text-center leading-relaxed line-clamp-4">{selectedTts.text}</p>

                      {/* Info bar */}
                      <div className="flex items-center gap-4 text-[11px] text-white/25">
                        <span>{selectedTts.duration}s duration</span>
                        <span>Generated in {selectedTts.time}s</span>
                      </div>

                      {/* Download */}
                      <button
                        onClick={() => downloadTtsAudio(selectedTts.audio, `tts_${selectedTts.voice}_${selectedTts.id}.wav`)}
                        className="rounded-lg bg-white/[0.06] px-4 py-2 text-[11px] text-white/50 hover:bg-white/[0.1] transition"
                      >
                        Download WAV
                      </button>
                    </div>
                  </div>
                )}

                {/* Batch view */}
                {selectedBatch && (
                  <div className="flex flex-1 flex-col rounded-2xl border border-white/[0.06] bg-gradient-to-b from-amber-500/[0.02] to-transparent p-6 overflow-y-auto">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="text-[14px] font-medium text-white/80">Multi-Voice Generation</p>
                        <p className="text-[11px] text-white/30 mt-0.5">{selectedBatch.items.length} clips · {selectedBatch.time}s total</p>
                      </div>
                      {selectedBatch.combinedAudio && (
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => playTtsAudio(selectedBatch.combinedAudio!, `combined-${selectedBatch.id}`)}
                            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] transition ${
                              ttsPlayingId === `combined-${selectedBatch.id}`
                                ? "bg-amber-500/30 text-amber-300"
                                : "bg-amber-500/15 text-amber-400/70 hover:bg-amber-500/25"
                            }`}
                          >
                            {ttsPlayingId === `combined-${selectedBatch.id}` ? (
                              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                            ) : (
                              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5.14v14l11-7-11-7z" /></svg>
                            )}
                            Play Combined ({selectedBatch.combinedDuration}s)
                          </button>
                          <button
                            onClick={() => downloadTtsAudio(selectedBatch.combinedAudio!, `tts_combined_${selectedBatch.id}.wav`)}
                            className="rounded-lg bg-white/[0.06] px-3 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition"
                          >
                            Download Combined
                          </button>
                        </div>
                      )}
                    </div>

                    <div className="flex flex-col gap-3">
                      {selectedBatch.items.map((item, idx) => (
                        <div key={item.id} className="flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                          <button
                            onClick={() => playTtsAudio(item.audio, item.id)}
                            className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition ${
                              ttsPlayingId === item.id
                                ? "bg-amber-500/30 text-amber-300 ring-2 ring-amber-500/20"
                                : "bg-amber-500/10 text-amber-400/60 hover:bg-amber-500/20"
                            }`}
                          >
                            {ttsPlayingId === item.id ? (
                              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                            ) : (
                              <svg className="h-4 w-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5.14v14l11-7-11-7z" /></svg>
                            )}
                          </button>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-[11px] font-medium text-amber-300/70">{item.label || `Speaker ${idx + 1}`}</span>
                              <span className="text-[9px] text-white/20 bg-white/[0.04] rounded px-1.5 py-0.5">{item.voice}</span>
                            </div>
                            <p className="text-[11px] text-white/40 truncate mt-0.5">{item.text}</p>
                            <p className="text-[9px] text-white/20 mt-0.5">{item.duration}s · {item.time}s gen</p>
                          </div>
                          <button
                            onClick={() => downloadTtsAudio(item.audio, `tts_${item.voice}_${item.id}.wav`)}
                            className="rounded-lg border border-white/[0.06] px-2 py-1 text-[10px] text-white/30 hover:text-white/50 hover:border-white/[0.1] transition shrink-0"
                          >
                            DL
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${ttsLoading ? "border-amber-500/30" : "border-white/[0.06]"}`}>
                {ttsLoading ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-10 w-10 rounded-full border-2 border-amber-500/60 border-t-transparent animate-spin" />
                    <p className="text-[13px] text-white/30">Generating speech...</p>
                    <p className="text-[11px] text-white/15 animate-timer font-mono">{formatTime(ttsElapsed)} elapsed</p>
                    <p className="text-[10px] text-white/10 max-w-xs text-center">Chatterbox TTS is generating high-quality speech with voice cloning</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-white/15">
                    <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                    </svg>
                    <p className="text-[13px] text-white/25">Generated audio appears here</p>
                    <p className="text-[11px] text-white/15">Enter text and choose a voice to generate speech</p>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-[10px] text-white/10 max-w-sm">
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-amber-400/30 font-medium">Voice Cloning</p>
                        <p className="mt-0.5">6 built-in voices</p>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-amber-400/30 font-medium">Multi-Voice</p>
                        <p className="mt-0.5">Multiple speakers</p>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-amber-400/30 font-medium">Emotion</p>
                        <p className="mt-0.5">Control expressiveness</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          )}

          {/* ── Professional Display ──────────────────────────────── */}
          {activeTab === "professional" && (
            selectedPro ? (
              <div className="flex flex-1 flex-col animate-fade-in">
                <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01]">
                  <img src={`data:image/png;base64,${selectedPro.image}`} alt={selectedPro.content} className="max-h-full max-w-full object-contain" />
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[11px] text-white/30">
                    <span className="rounded bg-rose-500/10 px-1.5 py-0.5 text-[10px] text-rose-300/60">
                      {PRO_CATEGORIES.find(c => c.key === selectedPro.category)?.label || selectedPro.category}
                    </span>
                    <span>{selectedPro.width}x{selectedPro.height}</span>
                    <span>Seed {selectedPro.seed}</span>
                    <span>{selectedPro.time}s</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => { setProContent(selectedPro.content); setProSeed(String(selectedPro.seed)); }}
                      className="rounded-lg border border-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/35 hover:text-white/55 hover:border-white/[0.1] transition">Reuse</button>
                    <button onClick={() => downloadPro(selectedPro)}
                      className="rounded-lg bg-white/[0.06] px-2.5 py-1.5 text-[11px] text-white/50 hover:bg-white/[0.1] transition">Download</button>
                  </div>
                </div>
                <p className="mt-1.5 text-[11px] text-white/20 line-clamp-2">{selectedPro.content}</p>
                <details className="mt-1">
                  <summary className="text-[9px] text-white/15 cursor-pointer hover:text-white/25 transition">Prompt used</summary>
                  <p className="mt-1 text-[9px] text-white/10 leading-relaxed">{selectedPro.promptUsed}</p>
                </details>
              </div>
            ) : (
              <div className={`flex flex-1 flex-col items-center justify-center rounded-2xl border border-dashed transition ${proLoading ? "border-rose-500/30" : "border-white/[0.06]"}`}>
                {proLoading ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-10 w-10 rounded-full border-2 border-rose-500/60 border-t-transparent animate-spin" />
                    <p className="text-[13px] text-white/30">Generating {currentProCategory?.label}...</p>
                    <p className="text-[10px] text-white/10 max-w-xs text-center">AI is crafting a professional graphic using FLUX with optimized prompts</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-white/15">
                    <svg className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
                    </svg>
                    <p className="text-[13px] text-white/25">Professional graphics appear here</p>
                    <p className="text-[11px] text-white/15">Choose a category, describe your data, and generate</p>
                    <div className="mt-3 grid grid-cols-4 gap-2 text-[9px] text-white/10 max-w-md">
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-rose-400/30 font-medium">Infographics</p>
                        <p className="mt-0.5">Data stories</p>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-rose-400/30 font-medium">Charts</p>
                        <p className="mt-0.5">Data viz</p>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-rose-400/30 font-medium">Flowcharts</p>
                        <p className="mt-0.5">Processes</p>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] p-2 text-center">
                        <p className="text-rose-400/30 font-medium">Dashboards</p>
                        <p className="mt-0.5">KPI views</p>
                      </div>
                    </div>
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
