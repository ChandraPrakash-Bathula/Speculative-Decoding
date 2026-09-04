import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // `npm run dev` serves the UI on :5173 and forwards API calls to the
    // FastAPI backend on :8000, so the frontend code can always use relative paths.
    proxy: {
      "/generate": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
