import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 网关自己的地址。dev 下代理到它，但网关按 Host/Origin 白名单拦请求
// （防 DNS rebinding），所以两头都要改写成网关的地址，见下面 configure。
const GATEWAY_PORT = 8765;
const GATEWAY = `127.0.0.1:${GATEWAY_PORT}`;

const RENDERER = path.resolve(__dirname, "renderer");

export default defineConfig({
  // The renderer is the Vite root; the window loads the same build from
  // the gateway, so dev and production share one document layout.
  root: RENDERER,
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(RENDERER, "src"),
    },
  },
  build: {
    outDir: path.resolve(RENDERER, "dist"),
    // outDir sits outside the Vite root: emptying it must be explicit.
    emptyOutDir: true,
  },
  server: {
    // 固定端口：被占用时直接失败，绝不静默切到 5174。
    // 切端口会让人对着旧实例（或另一个工作区）的页面调试。
    port: 5173,
    strictPort: true,
    proxy: {
      // the core process serves both the websocket and the HTTP endpoints
      "/ws": {
        target: `ws://${GATEWAY}`,
        ws: true,
        changeOrigin: true,
        configure(proxy) {
          const rewrite = (proxyReq: { setHeader(name: string, value: string): void }) => {
            proxyReq.setHeader("origin", `http://${GATEWAY}`);
          };
          proxy.on("proxyReq", rewrite);
          proxy.on("proxyReqWs", rewrite);
        },
      },
    },
  },
});
