import { NextRequest, NextResponse } from "next/server";

const HUNYUAN_URL = process.env.BACKEND_URL || "http://localhost:8100";

export const maxDuration = 300; // 5 minutes

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps } = body;
  if (!prompt || typeof prompt !== "string" || prompt.trim().length === 0) {
    return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
  }

  try {
    const res = await fetch(`${HUNYUAN_URL}/api/hunyuan/image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt.trim(),
        negative_prompt: negative_prompt || "",
        width: width || 1024,
        height: height || 1024,
        seed: seed ?? null,
        guidance_scale: guidance_scale ?? 5.0,
        num_inference_steps: num_inference_steps ?? 25,
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
