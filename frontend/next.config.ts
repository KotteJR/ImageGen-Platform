import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow external image sources if needed
  images: {
    unoptimized: true,
  },
  // Increase serverless function timeout for long-running generation
  serverExternalPackages: [],
  // Ensure API routes work correctly on Vercel
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
