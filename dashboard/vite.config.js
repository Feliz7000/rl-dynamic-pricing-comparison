import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Single-file build: JS, CSS and the exported data JSON are all inlined into
// dist/index.html so the dashboard opens by double-click with no server.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  base: "./",
});
