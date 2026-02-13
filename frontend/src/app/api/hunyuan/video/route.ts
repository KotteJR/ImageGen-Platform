import { NextRequest, NextResponse } from "next/server";

const HUNYUAN_URL = process.env.BACKEND_URL || "http://localhost:8100";

export const maxDuration = 900; // 15 minutes (video generation is slow)

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { prompt, width, height, num_frames, seed, num_inference_steps, fps } = body;
  if (!prompt || typeof prompt !== "string" || prompt.trim().length === 0) {
    return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
  }

  try {
    const res = await fetch(`${HUNYUAN_URL}/api/hunyuan/video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt.trim(),
        width: width || 848,
        height: height || 480,
        num_frames: num_frames || 61,
        seed: seed ?? null,
        num_inference_steps: num_inference_steps ?? 30,
        fps: fps || 15,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      return NextResponse.json({ error: `Backend error: ${err}` }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Connection failed";
    return NextResponse.json({ error: msg }, { status: 502 });
  }
}
