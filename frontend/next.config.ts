import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow external image sources if needed
  images: {
    unoptimized: true,
  },
  // Increase serverless function timeout for long-running generation
  serverExternalPackages: [],
  // Ensure API routes work correctly
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb", // video/3D responses can be large
    },
  },
};

export default nextConfig;
