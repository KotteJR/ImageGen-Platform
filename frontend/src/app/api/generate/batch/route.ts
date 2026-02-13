import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8100";

export const maxDuration = 300; // 5 minutes (batch with many images can take a while)

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { prompts, negative_prompt, width, height, guidance_scale, num_inference_steps, model_mode } = body;

  if (!Array.isArray(prompts) || prompts.length === 0) {
    return NextResponse.json({ error: "prompts array is required" }, { status: 400 });
  }

  try {
    const res = await fetch(`${BACKEND_URL}/api/generate/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompts,
        negative_prompt: negative_prompt || "",
        width: width || 1024,
        height: height || 1024,
        guidance_scale: guidance_scale ?? 0,
        num_inference_steps: num_inference_steps ?? 4,
        model_mode: model_mode || "lightning",
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      return NextResponse.json(
        { error: `Backend error: ${err}` },
        { status: res.status },
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Connection failed";
    return NextResponse.json({ error: msg }, { status: 502 });
  }
}
