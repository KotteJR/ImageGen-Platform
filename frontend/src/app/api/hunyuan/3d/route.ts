import { NextRequest, NextResponse } from "next/server";

const HUNYUAN_URL = process.env.BACKEND_URL || "http://localhost:8100";

export const maxDuration = 600; // 10 minutes

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get("image");

    if (!image || !(image instanceof Blob)) {
      return NextResponse.json({ error: "Image file is required" }, { status: 400 });
    }

    // Forward the FormData to the Hunyuan backend
    const backendForm = new FormData();
    backendForm.append("image", image);

    const doTexture = formData.get("do_texture");
    if (doTexture !== null) {
      backendForm.append("do_texture", doTexture.toString());
    }

    const res = await fetch(`${HUNYUAN_URL}/api/hunyuan/3d`, {
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
