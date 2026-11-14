import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: { root: __dirname },
  async headers() {
    return [{
      source: "/(.*)",
      headers: [{ key: "Content-Security-Policy", value: "frame-ancestors 'self' https://rjbassig.dev https://www.rjbassig.dev" }],
    }];
  },
};

export default nextConfig;
