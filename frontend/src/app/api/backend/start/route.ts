import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8100";
const IS_VERCEL = !!process.env.VERCEL;

/** Check if backend is already running */
async function isRunning(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${BACKEND_URL}/api/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}

export async function POST() {
  // On Vercel (serverless), we can't start local processes.
  // The backend must be running separately on a GPU server.
  if (IS_VERCEL) {
    const running = await isRunning();
    if (running) {
      return NextResponse.json({
        success: true,
        message: "Backend is running (cloud mode)",
      });
    }
    return NextResponse.json({
      success: false,
      error:
        "Cloud mode: the GPU backend must be started separately on your server. " +
        "Set BACKEND_URL in Vercel environment variables to point to your GPU server.",
    }, { status: 503 });
  }

  // ── Local development: start the Python backend process ──

  // Don't start if already running
  if (await isRunning()) {
    return NextResponse.json({ success: true, message: "Backend is already running" });
  }

  try {
    // Dynamic imports — only available in Node.js, not Vercel Edge
    const { exec } = await import("child_process");
    const path = await import("path");
    const fs = await import("fs");

    const backendDir = path.resolve(process.cwd(), "..", "backend");
    const serverScript = path.join(backendDir, "server.py");

    if (!fs.existsSync(serverScript)) {
      return NextResponse.json(
        { success: false, error: `Backend script not found at ${serverScript}` },
        { status: 404 },
      );
    }

    const venvPython = path.join(backendDir, "venv", "bin", "python3");
    const pythonCmd = fs.existsSync(venvPython) ? venvPython : "python3";
    const logFile = path.join(backendDir, "server.log");
    const backendPort = new URL(BACKEND_URL).port || "8100";

    const cmd = `nohup "${pythonCmd}" "${serverScript}" > "${logFile}" 2>&1 &`;
    const spawnEnv = { ...process.env, PORT: backendPort };

    await new Promise<void>((resolve, reject) => {
      exec(cmd, { cwd: backendDir, env: spawnEnv }, (error) => {
        if (error) reject(error);
        else resolve();
      });
    });

    await new Promise((r) => setTimeout(r, 3000));
    const running = await isRunning();

    if (running) {
      return NextResponse.json({ success: true, message: "Backend started successfully" });
    }

    let logContent = "";
    try {
      if (fs.existsSync(logFile)) {
        logContent = fs.readFileSync(logFile, "utf-8").slice(-500);
      }
    } catch {
      // ignore
    }

    if (logContent && (logContent.includes("Error") || logContent.includes("Traceback"))) {
      return NextResponse.json({
        success: false,
        error: `Backend failed to start. Log:\n${logContent}`,
      }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      message: "Backend process launched. It may take a moment to become ready (model loading).",
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ success: false, error: msg }, { status: 500 });
  }
}
