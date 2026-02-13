import { NextRequest, NextResponse } from "next/server";

const HUNYUAN_URL = process.env.BACKEND_URL || "http://localhost:8100";

export const maxDuration = 900; // 15 minutes (video generation is slow)

export async function POST(request: NextRequest) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "Invalid form data" }, { status: 400 });
  }

  const imageFile = formData.get("image");
  if (!imageFile || !(imageFile instanceof Blob)) {
    return NextResponse.json({ error: "Image file is required" }, { status: 400 });
  }

  // Build form data for backend
  const backendForm = new FormData();
  backendForm.append("image", imageFile);

  // Forward optional fields
  const prompt = formData.get("prompt");
  if (prompt && typeof prompt === "string") {
    backendForm.append("prompt", prompt.trim());
  }

  const fields = ["width", "height", "num_frames", "seed", "num_inference_steps", "fps"];
  for (const field of fields) {
    const val = formData.get(field);
    if (val !== null && val !== undefined && val !== "") {
      backendForm.append(field, val.toString());
    }
  }

  try {
    const res = await fetch(`${HUNYUAN_URL}/api/hunyuan/video/i2v`, {
      method: "POST",
      body: backendForm,
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
