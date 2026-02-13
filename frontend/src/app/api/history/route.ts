import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8100";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") || "200";
  const offset = searchParams.get("offset") || "0";

  try {
    const res = await fetch(
      `${BACKEND_URL}/api/history?limit=${limit}&offset=${offset}`,
      { cache: "no-store" },
    );

    if (!res.ok) {
      return NextResponse.json({ images: [], total: 0 });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ images: [], total: 0 });
  }
}
