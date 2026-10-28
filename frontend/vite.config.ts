import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 网关自己的地址。dev 下代理到它，但网关按 Host/Origin 白名单拦请求
// （防 DNS rebinding），所以两头都要改写成网关的地址，见下面 configure。
const GATEWAY_PORT = 8765;
const GATEWAY = `127.0.0.1:${GATEWAY_PORT}`;

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
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
