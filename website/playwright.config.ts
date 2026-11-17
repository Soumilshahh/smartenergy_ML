import { defineConfig } from "@playwright/test";
import os from "node:os";
import path from "node:path";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  outputDir: path.join(os.tmpdir(), "smartenergy-playwright-results"),
  use: {
    baseURL: process.env.LIVE_URL || "http://127.0.0.1:4173/smartenergy_ML/",
    browserName: "chromium",
    viewport: { width: 1536, height: 1024 },
    reducedMotion: "reduce",
  },
  webServer: process.env.LIVE_URL
    ? undefined
    : {
        command:
          "node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4173 --strictPort",
        url: "http://127.0.0.1:4173/smartenergy_ML/",
        reuseExistingServer: !process.env.CI,
      },
  reporter: "list",
  timeout: 45000,
});
