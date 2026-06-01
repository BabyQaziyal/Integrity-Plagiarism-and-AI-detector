import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the Flask backend so the SPA and API share an
// origin. If the backend is down, we answer with a clean JSON 503 (instead of a
// raw socket error) so the UI can show a friendly "backend offline" banner.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:5000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("error", (err, _req, res) => {
            if (res && !res.headersSent && res.writeHead) {
              res.writeHead(503, { "Content-Type": "application/json" });
            }
            if (res && res.end) {
              res.end(
                JSON.stringify({
                  error:
                    "Backend not reachable. Start it with `python scripts/run_api.py` (or use ./dev.ps1).",
                })
              );
            }
            // One concise line instead of a stack-trace flood.
            console.warn(`[api proxy] backend offline: ${err.code || err.message}`);
          });
        },
      },
    },
  },
});
